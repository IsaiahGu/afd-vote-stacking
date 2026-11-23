"""P06 step 2: does the LLM count heads or weigh arguments?

Cases: keep/delete closes with >= 5 keep+delete votes (nomination = delete), no tie.
  reverse = closer's result opposite to the vote majority; control = pro-majority case matched
  on period, |margin|, total votes and length (nearest neighbour, without replacement).
Prompt: closing statement removed, usernames anonymised, short policy excerpt; answer Keep/Delete;
  p(keep) from first-token log-probabilities.
A  agreement with the closer, reverse vs control.
B  inject 2/4/8 bare votes (no reasoning) for the side the closer rejected; and remove all
   existing bare votes.  Flip rate vs the unmodified prompt.
C  remove the winning side's policy-citing votes vs the same number of its non-citing votes.
python run_exp.py <model alias> [max_reverse]
Output: results/exp_<model>.parquet
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import json
import zlib
import random
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, f"{WORKSPACE_ROOT}/shared")
from llm import LLM  # noqa: E402

ROOT = WORKSPACE_ROOT
import os  # noqa: E402
MAXREV = int(os.environ.get("MAXREV", 100000))  # cap on reverse-majority cases
POLICY = ("Wikipedia deletion discussions are not votes. The closing administrator determines rough consensus by "
          "the strength of the arguments in light of Wikipedia policies and guidelines (notability, verifiability, "
          "reliable sources, what Wikipedia is not). Arguments that ignore policy, rest on personal opinion, or are "
          "bare recommendations without reasons carry little weight, however many there are.")
SYS = "You are an experienced English Wikipedia administrator who closes Articles for Deletion discussions."
USERRE = re.compile(r"\[\[\s*(?:User|User talk|Special:Contributions)\s*:\s*([^|\]/#]+)[^\]]*\]\]", re.I)


def anon_render(rec, entries):
    names = {}

    def uid(n):
        n = n.strip()
        if n not in names:
            names[n] = f"User{len(names) + 1}"
        return names[n]

    def clean(t):
        t = re.sub(r"\[\[\s*User talk\s*:[^\]]*\]\]", "", t, flags=re.I)
        t = USERRE.sub(lambda m: uid(m.group(1)), t)
        t = re.sub(r"\{\{\s*(?:u|ping|reply to|re)\s*\|([^}|]+)[^}]*\}\}", lambda m: "@" + uid(m.group(1)), t, flags=re.I)
        t = re.sub(r"\d{1,2}:\d{2},? \d{1,2} [A-Z][a-z]+ \d{4} \(UTC\)", "", t)
        t = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", t)
        t = re.sub(r"\[https?://\S+ ([^\]]*)\]", r"\1", t)
        t = re.sub(r"\{\{[^{}]*\}\}", "", t)
        t = re.sub(r"<[^>]+>", "", t)
        t = re.sub(r"'''(.+?)'''", r"**\1**", t)
        t = t.replace("''", "")
        t = re.sub(r"\(\s*\)|\(talk\)|\btalk\b\)?", "", t)
        return re.sub(r"[ \t]+", " ", t).strip()
    title = rec["title"].split("/", 1)[-1]
    out = [f"Article nominated for deletion: {title}", "", "Nomination: " + clean(rec["nomination"]), ""]
    for e in entries:
        out.append(clean(e["text"]))
        out += ["    " + clean(r) for r in e.get("replies", [])]
    return "\n".join(out)[:40000], names


def prompt(text):
    return [{"role": "system", "content": SYS},
            {"role": "user", "content": f"{POLICY}\n\nDiscussion:\n\"\"\"\n{text}\n\"\"\"\n\nBased on the discussion and Wikipedia "
                                        f"policy, what should the closing result be? Answer with exactly one word: Keep or Delete."}]


def p_keep(o):
    lp = o.get("logprobs")
    if lp:
        top = lp[0]["top"]
        k = max([v for t, v in top.items() if t.strip().lower().startswith("keep")], default=-50)
        d = max([v for t, v in top.items() if t.strip().lower().startswith("del")], default=-50)
        return float(np.exp(k) / (np.exp(k) + np.exp(d)))
    t = o["text"].strip().lower()
    return 1.0 if t.startswith("keep") else 0.0 if t.startswith("del") else 0.5


def bare(label, i):
    return {"kind": "vote", "label": label, "text": f"*'''{label.capitalize()}'''. [[User:Newvoter{i}|Newvoter{i}]]",
            "bare": True, "policy": False, "replies": []}


def main():
    rows = [json.loads(l) for l in open(f"{ROOT}/data/afd_parsed.jsonl")]
    el = [r for r in rows if r["outcome"] in ("keep", "delete") and r["n_keep"] + r["n_delete"] >= 5 and r["n_keep"] != r["n_delete"]]
    for r in el:
        r["majority"] = "keep" if r["n_keep"] > r["n_delete"] else "delete"
        r["reverse"] = r["majority"] != r["outcome"]
        r["margin"] = abs(r["n_keep"] - r["n_delete"]); r["tot"] = r["n_keep"] + r["n_delete"]
    rev = [r for r in el if r["reverse"]]
    rng = random.Random(0)
    rng.shuffle(rev)
    rev = rev[:MAXREV]
    pool = {}
    for r in el:
        if not r["reverse"]:
            pool.setdefault(r["period"], []).append(r)
    used, ctrl = set(), []
    for r in rev:
        cands = [c for c in pool.get(r["period"], []) if c["title"] not in used]
        if not cands:
            continue
        c = min(cands, key=lambda c: (abs(c["margin"] - r["margin"]) * 3 + abs(c["tot"] - r["tot"]) +
                                      abs(np.log1p(c["len"]) - np.log1p(r["len"])) * 2))
        used.add(c["title"]); ctrl.append(c)
    print("eligible", len(el), "reverse", len(rev), "controls", len(ctrl), flush=True)
    with open(f"{ROOT}/data/afd_cases.jsonl", "w") as f:
        for r, grp in [(r, "reverse") for r in rev] + [(c, "control") for c in ctrl]:
            f.write(json.dumps({**r, "group": grp}) + "\n")


def run(model):
    """LLM step; needs only data/afd_cases.jsonl, so it can run on either server."""
    cases = [(r, r["group"]) for r in (json.loads(l) for l in open(f"{ROOT}/data/afd_cases.jsonl"))]
    jobs, meta = [], []
    for r, grp in cases:
        E = r["entries"]
        loser = "delete" if r["outcome"] == "keep" else "keep"
        variants = {"orig": E}
        for k in (2, 4, 8):
            e2 = list(E)
            rr = random.Random(zlib.crc32(r["title"].encode()) + k)
            for i in range(k):
                e2.insert(rr.randint(0, len(e2)), bare(loser, i + 1))
            variants[f"inject{k}"] = e2
        variants["remove_bare"] = [e for e in E if not e.get("bare")]
        win_pol = [i for i, e in enumerate(E) if e["label"] == r["outcome"] and e["policy"]]
        win_non = [i for i, e in enumerate(E) if e["label"] == r["outcome"] and not e["policy"]]
        n = min(len(win_pol), len(win_non))
        if n > 0:
            rr = random.Random(7)
            drop_p = set(rr.sample(win_pol, n)); drop_n = set(rr.sample(win_non, n))
            variants["drop_policy"] = [e for i, e in enumerate(E) if i not in drop_p]
            variants["drop_nonpolicy"] = [e for i, e in enumerate(E) if i not in drop_n]
        for vname, ents in variants.items():
            text, _ = anon_render(r, ents)
            jobs.append(prompt(text))
            meta.append({"title": r["title"], "period": r["period"], "group": grp, "variant": vname, "outcome": r["outcome"],
                         "majority": r["majority"], "n_keep": r["n_keep"], "n_delete": r["n_delete"], "margin": r["margin"],
                         "n_bare": r["n_bare_keep"] + r["n_bare_delete"], "n_drop": n, "chars": len(text)})
    llm = LLM(model, cache_name=f"p06_{model}", concurrency=None if model in ("local", "gemma-4-12b") else 16)
    # reasoning API models (gpt-6-*) spend hidden tokens and return no logprobs: score their text answer
    kw = dict(max_tokens=800) if model.startswith("gpt-6") else dict(max_tokens=3, logprobs=True, top_logprobs=10)
    outs = llm.chat_many(jobs, **kw)
    for m, o in zip(meta, outs):
        m["p_keep"] = p_keep(o[0]); m["pred"] = "keep" if m["p_keep"] >= 0.5 else "delete"; m["raw"] = o[0]["text"][:20]
        m["error"] = o[0].get("finish") == "error"
    df = pd.DataFrame(meta)
    df.to_parquet(f"{ROOT}/results/exp_{model}.parquet")
    print("errors", int(df.error.sum()), "of", len(df))
    o = df[df.variant == "orig"].assign(agree=lambda x: x.pred == x.outcome, follow_major=lambda x: x.pred == x.majority)
    print(o.groupby(["group", "period"])[["agree", "follow_major"]].mean().round(3))
    print(o.groupby("group")[["agree", "follow_major"]].mean().round(3))


if __name__ == "__main__":
    # python run_exp.py select            -> data/afd_cases.jsonl (server with the parsed Wikipedia data)
    # python run_exp.py run <model alias> -> results/exp_<model>.parquet
    if len(sys.argv) > 1 and sys.argv[1] == "select":
        main()
    else:
        run(sys.argv[2] if len(sys.argv) > 2 else "local")
