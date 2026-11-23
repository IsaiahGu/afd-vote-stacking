"""P06 supplementary runs: one model, one part, every prediction saved row by row.

python supp_run.py <model alias> <part> [<part> ...]
  core     S2 (mask_labels, votes_only), S3 injections, S4 D1/D4, S11 length-matched and stripped policy
  strict   S4 D2: stricter instruction x {orig, inject8, boiler8, polshort8}
  argmap   S2 L3 / S4 D3: argument map (call 1), then a fresh call on nomination + map (call 2)
  s9       S9 prompt variants x {orig, inject8}
  title    S6 title-and-year probe
CASES=all (default): the 556 candidates; CASES=pairs: titles in results/verified_pairs.json.
CACHE_TAG: cache-name prefix (default p06s, so the new runs share one engine version).
Output: results/supp_<part>_<model>.parquet (title, group, variant, p_keep, pred, raw, n_prompt_chars, ...)
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, f"{WORKSPACE_ROOT}/shared")
sys.path.insert(0, f"{WORKSPACE_ROOT}/src")
from llm import LLM  # noqa: E402
from run_exp import anon_render  # noqa: E402
import supp_variants as sv  # noqa: E402

ROOT = WORKSPACE_ROOT
REASONING = ("gpt-6", "gpt-5", "o3", "o4", "claude-opus", "claude-sonnet")


EFFORT = {}  # "model@low" -> reasoning_effort=low for API reasoning models


def kw_for(model, long=False):
    """Same output rule as run_exp.py: logprobs where the model returns them, else the text answer."""
    model, _, effort = model.partition("@")
    if effort:
        return dict(max_tokens=6000 if long else 4000, reasoning_effort=effort)
    if model.endswith("-think"):  # local model with its thinking mode on: score the text after </think>
        return dict(max_tokens=8000, thinking=True)
    if model == "gpt-6-luna" and not long:  # same parameters as run_exp.py, so its cached answers are reused
        return dict(max_tokens=800)
    if model.startswith(("claude-opus", "claude-sonnet")):  # the relay can add hidden reasoning to Claude models
        return dict(max_tokens=8000 if long else 4000)
    if model.startswith(REASONING):
        return dict(max_tokens=4000 if long else 1500)
    if model.startswith(("claude", "gemini", "grok", "deepseek")):
        return dict(max_tokens=1500 if long else 8)
    return dict(max_tokens=1500 if long else 3, logprobs=not long, **({} if long else {"top_logprobs": 10}))


def answer_text(o):
    t = o.get("text") or ""
    return t.split("</think>")[-1].strip()


def p_keep(o, reason=False):
    if not reason:
        lp = o.get("logprobs")
        if lp:
            top = lp[0]["top"]
            k = max([v for t, v in top.items() if t.strip().lower().startswith("keep")], default=-50)
            d = max([v for t, v in top.items() if t.strip().lower().startswith("del")], default=-50)
            return float(np.exp(k) / (np.exp(k) + np.exp(d)))
    t = answer_text(o).lower()
    if reason:
        m = re.findall(r"result\s*[:\-]?\s*\**\s*(keep|delete)", t)
        if m:
            return 1.0 if m[-1] == "keep" else 0.0
        t = t.strip().splitlines()[-1] if t.strip() else ""
    t = re.sub(r"^[^a-z]*", "", t)
    return 1.0 if t.startswith("keep") else 0.0 if t.startswith("del") else 0.5


def load_cases():
    if os.environ.get("CASES_FILE"):  # e.g. data/afd_drv_cases.jsonl (S12) or data/afd_cases_more.jsonl (S8)
        cs = [json.loads(l) for l in open(f"{ROOT}/{os.environ['CASES_FILE']}")]
        if os.environ.get("PAIRS_FILE"):  # keep only the verified pairs of that set
            keep = {t for p in json.load(open(f"{ROOT}/{os.environ['PAIRS_FILE']}"))["pairs"] for t in p}
            cs = [c for c in cs if c["title"] in keep]
        return cs
    cases = [json.loads(l) for l in open(f"{ROOT}/data/afd_cases.jsonl")]
    extra = f"{ROOT}/data/afd_cases_more.jsonl"
    if os.environ.get("WITH_MORE") == "1" and os.path.exists(extra):
        cases += [json.loads(l) for l in open(extra)]
    if os.environ.get("CASES", "all") == "pairs":
        keep = {t for p in json.load(open(f"{ROOT}/results/verified_pairs.json"))["pairs"] for t in p}
        cases = [c for c in cases if c["title"] in keep]
    return cases


def meta_of(r, variant, text):
    return {"title": r["title"], "period": r["period"], "group": r["group"], "variant": variant, "outcome": r["outcome"],
            "majority": r["majority"], "n_keep": r["n_keep"], "n_delete": r["n_delete"], "margin": r["margin"],
            "n_bare": r["n_bare_keep"] + r["n_bare_delete"], "chars": len(text)}


def run_jobs(llm, jobs, meta, kw, reason=False):
    if os.environ.get("RETRY_ERRORS") == "1" and CURRENT_PART:  # only the rows that failed before, with more room
        fn = f"{ROOT}/results/supp_{CURRENT_PART}{os.environ.get('OUT_TAG', '')}_{CURRENT_MODEL}.parquet"
        old = pd.read_parquet(fn)
        bad = set(zip(old[old.error.astype(bool)].title, old[old.error.astype(bool)].variant))
        keep = [i for i, m in enumerate(meta) if (m["title"], m["variant"]) in bad]
        print(f"retrying {len(keep)} failed rows with max_tokens {RETRY_TOKENS}", flush=True)
        kw = {**kw, "max_tokens": RETRY_TOKENS}
        new = run_jobs_plain(llm, [jobs[i] for i in keep], [meta[i] for i in keep], kw, reason)
        fixed = {(m["title"], m["variant"]): m for m in new}
        prev = {(r["title"], r["variant"]): r for r in old.to_dict(orient="records")}
        # one row per item of this call, in order: the retried answer where it failed before, the old one otherwise
        return [fixed.get((m["title"], m["variant"])) or prev.get((m["title"], m["variant"]), m) for m in meta]
    return run_jobs_plain(llm, jobs, meta, kw, reason)


CURRENT_PART, CURRENT_MODEL = None, None
RETRY_TOKENS = int(os.environ.get("RETRY_TOKENS", 3000))


def run_jobs_plain(llm, jobs, meta, kw, reason=False):
    outs = llm.chat_many(jobs, **kw)
    for m, o in zip(meta, outs):
        o = o[0]
        m["error"] = o.get("finish") == "error" or (not (o.get("text") or "").strip() and not o.get("logprobs"))
        m["p_keep"] = p_keep(o, reason) if not m["error"] else np.nan
        m["pred"] = None if m["error"] else ("keep" if m["p_keep"] >= 0.5 else "delete")
        m["raw"] = answer_text(o)[-300:] if reason else answer_text(o)[:40]
        m["finish"] = o.get("finish")
    return meta


def part_core(r):
    jobs, meta = [], []
    for name, ents in sv.entry_conditions(r).items():
        text, _ = anon_render(r, ents)
        jobs.append(sv.prompt_std(text)); meta.append(meta_of(r, name, text))
    return jobs, meta


def part_strict(r):
    v = sv.entry_conditions(r)
    jobs, meta = [], []
    for a in sv.ATTACKS:
        text, _ = anon_render(r, v[a])
        jobs.append(sv.prompt_std(text, policy=sv.POLICY + sv.STRICT)); meta.append(meta_of(r, f"D2_{a}", text))
    return jobs, meta


def part_s9(r, para):
    v = sv.entry_conditions(r)
    jobs, meta = [], []
    for a in ("orig", "inject8"):
        text, _ = anon_render(r, v[a])
        P = {"P1_nopolicy": sv.prompt_std(text, policy=""), "P2_nolastsentence": sv.prompt_std(text, policy=sv.POLICY_P2),
             "P4_nopersona": sv.prompt_std(text, system=False), "P5_reason": sv.prompt_reason(text)}
        for i, tag in enumerate(("P3a", "P3b", "P3c")):
            P[tag] = [{"role": "system", "content": para[i]["system"]},
                      {"role": "user", "content": para[i]["user"].replace("{discussion}", text)}]
        for pn in sv.S9_PROMPTS:
            jobs.append(P[pn]); meta.append(meta_of(r, f"{pn}_{a}", text))
    return jobs, meta


def save(model, part, rows):
    df = pd.DataFrame(rows)
    df.to_parquet(f"{ROOT}/results/supp_{part}{os.environ.get('OUT_TAG', '')}_{model}.parquet")
    o = df[~df.error]
    print(f"[{model} {part}] rows {len(df)} errors {int(df.error.sum())}", flush=True)
    print(o.assign(agree=o.pred == o.outcome).groupby(["variant", "group"]).agree.mean().unstack().round(3).head(60).to_string(), flush=True)


def main():
    model = sys.argv[1]
    parts = sys.argv[2:] or ["core"]
    tag = os.environ.get("CACHE_TAG", "p06s")
    cases = load_cases()
    print("cases", len(cases), flush=True)
    llm = LLM(model.partition("@")[0], cache_name=f"{tag}_{model}")
    global CURRENT_PART, CURRENT_MODEL
    for part in parts:
        CURRENT_PART, CURRENT_MODEL = (None if part == "argmap" else part), model
        if part in ("core", "strict"):
            jobs, meta = [], []
            for r in cases:
                j, m = (part_core if part == "core" else part_strict)(r)
                jobs += j; meta += m
            save(model, part, run_jobs(llm, jobs, meta, kw_for(model)))
        elif part in ("s10", "orig"):  # S10 / S8: natural case plus loser +2/4/8 and winner +8; S12: natural only
            jobs, meta = [], []
            for r in cases:
                v = sv.entry_conditions(r)
                for name in (("orig", "inject2", "inject4", "inject8", "win8") if part == "s10" else ("orig",)):
                    text, _ = anon_render(r, v[name])
                    jobs.append(sv.prompt_std(text)); meta.append(meta_of(r, name, text))
            save(model, part, run_jobs(llm, jobs, meta, kw_for(model)))
        elif part == "s9":
            para = json.load(open(f"{ROOT}/data/prompt_paraphrases.json"))
            jobs, meta = [], []
            for r in cases:
                j, m = part_s9(r, para)
                jobs += j; meta += m
            reason = [m["variant"].startswith("P5") for m in meta]
            a = run_jobs(llm, [j for j, x in zip(jobs, reason) if not x], [m for m, x in zip(meta, reason) if not x], kw_for(model))
            b = run_jobs(llm, [j for j, x in zip(jobs, reason) if x], [m for m, x in zip(meta, reason) if x],
                         kw_for(model, long=True), reason=True)
            save(model, part, a + b)
        elif part == "title":
            jobs = [sv.prompt_title_only(r) for r in cases]
            meta = [meta_of(r, "title_only", sv.title_of(r)) for r in cases]
            save(model, part, run_jobs(llm, jobs, meta, kw_for(model)))
        elif part == "argmap":
            jobs1, keys = [], []
            for r in cases:
                v = sv.entry_conditions(r)
                for a in sv.ATTACKS:
                    text, _ = anon_render(r, v[a])
                    jobs1.append(sv.prompt_argmap1(text)); keys.append((r, a, text))
            kw1 = kw_for(model, long=True)
            kw1.pop("logprobs", None); kw1.pop("top_logprobs", None)
            outs1 = llm.chat_many(jobs1, **kw1)
            jobs2, meta, maps = [], [], []
            for (r, a, text), o in zip(keys, outs1):
                amap = answer_text(o[0])
                maps.append({"title": r["title"], "attack": a, "argmap": amap, "finish": o[0].get("finish"),
                             "leak_people": bool(sv.LEAK_PEOPLE.search(amap)), "leak_broad": bool(sv.LEAK_BROAD.search(amap))})
                jobs2.append(sv.prompt_argmap2(r, amap)); meta.append(meta_of(r, f"D3_{a}", text))
            pd.DataFrame(maps).to_parquet(f"{ROOT}/results/supp_argmap1_{model}.parquet")
            m1 = pd.DataFrame(maps)
            print(f"[{model}] argmap leak (people) {m1.leak_people.mean():.3f}  broad {m1.leak_broad.mean():.3f}", flush=True)
            rows = run_jobs(llm, jobs2, meta, kw_for(model))
            for row, mp in zip(rows, maps):
                if not mp["argmap"].strip() or mp["finish"] == "error":
                    row["error"] = True
            save(model, part, rows)


if __name__ == "__main__":
    main()
