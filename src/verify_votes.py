"""P06 S1: verify the vote counts of the candidate cases with two API judges from different families.

1. control  build 200 synthetic discussions from real top-level votes with known counts
            (half with hard cases: nested votes, struck-and-changed votes, the same user voting twice,
            "delete or merge" mixed votes, nominator withdrawal) -> data/judge_control.jsonl
2. judge    python verify_votes.py judge <model> control|cases|more
            each discussion -> JSON with every participant's final recommendation, the nominator's
            stance and whether the close was procedural -> results/judge_<set>_<model>.jsonl
3. score    python verify_votes.py score <judge1> <judge2>
            control accuracy (gate: majority side >= 0.97, both counts exact >= 0.90), agreement of
            rule recount and the two judges on the real cases, Cohen's kappa, verified set
            -> results/verify_votes.json, results/verified_pairs.json
Mixed votes: rule A counts "delete or merge" as delete (and "keep or merge" as keep); rule B ignores them.
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import json
import random
import re
import sys
from collections import Counter

sys.path.insert(0, f"{WORKSPACE_ROOT}/shared")
sys.path.insert(0, f"{WORKSPACE_ROOT}/src")
from llm import LLM, parse_json  # noqa: E402

ROOT = WORKSPACE_ROOT
USERLINK = re.compile(r"\[\[\s*(?:User|User talk|Special:Contributions)\s*:[^\]]*\]\]", re.I)
TS = re.compile(r"\d{1,2}:\d{2},? \d{1,2} [A-Z][a-z]+ \d{4} \(UTC\)")
BOLD = re.compile(r"'''+\s*(.+?)\s*'''+")
FINALS = ("keep", "delete", "delete_or_merge", "keep_or_merge", "merge_redirect", "none")

JUDGE_SYS = "You are a careful annotator of English Wikipedia deletion discussions. You reply with JSON only."
JUDGE_INSTR = """You will read an English Wikipedia Articles for Deletion (AfD) discussion in raw wikitext.
- The nomination opens the discussion and is signed by the nominator.
- Lines starting with * or # are top-level posts; lines starting with : or ** are indented replies. Recommendations made in indented replies count like any other.
- A recommendation is usually bolded ('''Keep''', '''Delete''', '''Merge''' ...). Text inside <s>...</s>, <del>...</del> or <strike>...</strike> has been struck out (withdrawn) by its author.
- Posts are signed with [[User:NAME|...]] or [[User talk:NAME|...]] or [[Special:Contributions/NAME|...]].

Determine every participant's FINAL recommendation:
- Each participant counts once. If someone posts several times, use their last recommendation that is not struck out; a later "changed to keep" or a struck vote followed by a new one replaces the earlier vote.
- Classify each final recommendation as exactly one of:
  keep (including speedy/strong/weak keep), delete (including speedy/strong/weak delete, salt),
  delete_or_merge (delete as first choice with merge/redirect/draftify as an alternative, e.g. "Delete or redirect"),
  keep_or_merge (keep as first choice with merge/redirect as an alternative),
  merge_redirect (merge, redirect, draftify, userfy, transwiki, incubate or similar),
  none (only comments, questions, neutral, or everything struck out).
- Do not list the nominator among the participants. Report the nominator separately with stance:
  delete (default: the nomination is a delete recommendation), keep (the nominator later changed to keep),
  withdrawn (the nominator withdrew the nomination), or other.
- procedural_close is true if the discussion was closed as speedy delete or speedy keep, as a withdrawn nomination, for a copyright violation, or for another procedural reason; use the closing result shown and the discussion.

Return only this JSON:
{"nominator": {"user": "NAME", "stance": "delete|keep|withdrawn|other"},
 "participants": [{"user": "NAME", "final": "keep|delete|delete_or_merge|keep_or_merge|merge_redirect|none", "changed_or_struck": true|false}],
 "procedural_close": true|false}"""


def render_raw(rec):
    title = rec["title"].split("/", 1)[-1]
    lines = [f"AfD discussion: {title}", f"Closing result as bolded by the closer: {rec.get('close_text', '')}", "",
             "NOMINATION:", rec["nomination"], "", "DISCUSSION:"]
    for e in rec["entries"]:
        lines.append(e["text"]); lines += list(e.get("replies", []))
    return "\n".join(lines)[:60000]


def judge_prompt(text):
    return [{"role": "system", "content": JUDGE_SYS},
            {"role": "user", "content": f"{JUDGE_INSTR}\n\nDiscussion:\n\"\"\"\n{text}\n\"\"\""}]


def _norm(u):
    return re.sub(r"[\s_]+", " ", str(u or "")).strip().lower()


def counts(j, rule="A"):
    """(keep, delete) from a judge's JSON; participants deduplicated by name, nominator excluded."""
    if not isinstance(j, dict):
        return None
    nom = j.get("nominator") or {}
    nu = _norm(nom.get("user"))
    last = {}
    for p in j.get("participants") or []:
        if not isinstance(p, dict):
            continue
        u = _norm(p.get("user"))
        if u and u == nu:
            continue
        last[u or f"anon{len(last)}"] = str(p.get("final", "none")).lower().strip()
    k = d = 0
    for f in last.values():
        k += f == "keep" or (rule == "A" and f == "keep_or_merge")
        d += f == "delete" or (rule == "A" and f == "delete_or_merge")
    st = str(nom.get("stance", "delete")).lower()
    d += st == "delete"; k += st == "keep"
    return k, d


def maj(kd):
    if kd is None:
        return None
    k, d = kd
    return "keep" if k > d else "delete" if d > k else "tie"


# ---------------- control set ----------------
WORDS1 = ["Blue", "Quiet", "North", "Amber", "Silver", "Rapid", "Old", "Green", "Iron", "Red", "Cedar", "Lunar", "Stone", "Maple",
          "Harbor", "Velvet", "Copper", "Misty", "Brave", "Polar"]
WORDS2 = ["Heron", "Fox", "Quill", "Lantern", "River", "Falcon", "Reader", "Scribe", "Otter", "Pine", "Comet", "Badger", "Wren",
          "Atlas", "Mason", "Sparrow", "Harper", "Willow", "Archer", "Tide"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def sig(u, rng):
    return (f"[[User:{u}|{u}]] ([[User talk:{u}|talk]]) {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}, "
            f"{rng.randint(1, 28)} {rng.choice(MONTHS)} {rng.randint(2018, 2025)} (UTC)")


def strip_sig(t):
    t = USERLINK.sub("", t)
    t = TS.sub("", t)
    t = re.sub(r"\(\s*(?:talk|contribs)?\s*[•·|]?\s*(?:talk|contribs)?\s*\)", "", t)
    return re.sub(r"[ \t—–-]+$", "", t.strip()).strip()


def build_control(n=200, seed=0):
    rng = random.Random(seed)
    ev = {json.loads(l)["title"] for l in open(f"{ROOT}/data/afd_cases.jsonl")}
    import gzip
    pool = {"keep": [], "delete": []}
    noms = []
    # v2: the first control set had ambiguous items (procedural / neutral nominations; vote texts that still
    # carried another signer, e.g. an IP editor or an "unsigned" note). Such sources are excluded so that the
    # counts are known by construction.
    BADNOM = re.compile(r"neutral|procedur|on behalf|withdraw|\bkeep\b|abstain|no opinion|\bIP\b|unsigned|relist|"
                        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|User|talk|contribs|'''|\{\{|\}\}", re.I)
    BADVOTE = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|unsigned|preceding|User|talk|contribs|\(UTC\)|"
                         r"\{\{|\}\}|withdraw|nominator|\bnom\b|change|struck|<|>", re.I)
    for l in gzip.open(f"{ROOT}/data/afd_parsed.jsonl.gz", "rt"):
        r = json.loads(l)
        if r["title"] in ev:
            continue
        nom = strip_sig(r["nomination"])
        if 80 < len(r["nomination"]) < 900 and not BADNOM.search(nom) and len(noms) < 3000:
            noms.append(nom)
        for e in r["entries"]:
            t = e["text"]
            if (e["kind"] == "vote" and e["label"] in pool and not e["bare"] and not e["replies"] and 60 < len(t) < 500
                    and not re.search(r"<(s|del|strike)>", t, re.I) and len(BOLD.findall(t)) == 1
                    and re.match(r"^[*#]\s*'''\s*(?:strong |weak |speedy )?(keep|delete)\s*'''", t, re.I)
                    and not re.search(r"merge|redirect|draft|userf|withdraw|\bor\b", BOLD.search(t).group(1), re.I)
                    and not BADVOTE.search(strip_sig(t)[strip_sig(t).find("'''", 3) + 3:])
                    and len(pool[e["label"]]) < 6000):
                pool[e["label"]].append(strip_sig(t))
    out = []
    for i in range(n):
        hard = i >= n // 2
        names = rng.sample([a + b + (str(rng.randint(1, 99)) if rng.random() < 0.5 else "") for a in WORDS1 for b in WORDS2], 14)
        nom_user, voters = names[0], names[1:]
        while True:
            nv = rng.randint(4, 10)
            labs = [rng.choice(["keep", "delete"]) for _ in range(nv)]
            feats = rng.sample(["nested", "changed", "duplicate", "mixed", "withdrawn"], rng.randint(2, 3)) if hard else []
            final = {voters[j]: labs[j] for j in range(nv)}
            mixed_user = voters[0] if "mixed" in feats else None
            if mixed_user:
                final[mixed_user] = rng.choice(["delete_or_merge", "keep_or_merge"])
            changed_user = voters[1] if "changed" in feats else None
            first_lab = None
            if changed_user:
                first_lab = "keep" if final[changed_user] == "delete" else "delete"
            nom_stance = "withdrawn" if "withdrawn" in feats else "delete"
            truth = {}
            for rule in ("A", "B"):
                k = sum(f == "keep" or (rule == "A" and f == "keep_or_merge") for f in final.values())
                d = sum(f == "delete" or (rule == "A" and f == "delete_or_merge") for f in final.values()) + (nom_stance == "delete")
                truth[rule] = (k, d)
            if truth["A"][0] != truth["A"][1]:
                break
        lines = ["NOMINATION:", rng.choice(noms) + " " + sig(nom_user, rng), "", "DISCUSSION:"]
        posts = []  # (user, line, is_reply_allowed_target)
        for j in range(nv):
            u = voters[j]
            f = final[u]
            if u == mixed_user:
                tgt = rng.choice(["the list of minor characters", "the parent company article", "the band's article"])
                lab = "Delete or merge" if f == "delete_or_merge" else "Keep or merge"
                posts.append([u, f"*'''{lab}''' to [[{tgt.title()}]]. Not enough independent coverage for a standalone page. {sig(u, rng)}"])
            elif u == changed_user:
                body = strip_sig(rng.choice(pool[first_lab]))
                posts.append([u, f"{body} {sig(u, rng)}"])
            else:
                body = strip_sig(rng.choice(pool[f]))
                posts.append([u, f"{body} {sig(u, rng)}"])
        rng.shuffle(posts)
        blocks = [[p[1]] for p in posts]
        if "nested" in feats:  # move one plain voter's post under another post as an indented reply
            cand = [b for b, p in zip(blocks, posts) if p[0] not in (mixed_user, changed_user)]
            if len(cand) >= 2:
                a, b = cand[0], cand[1]
                b[0] = ":" + b[0].lstrip("*#").lstrip()
                a.append(b[0]); blocks.remove(b)
        if "changed" in feats:
            b = blocks[[i_ for i_, p in enumerate(blocks) if changed_user in p[0]][0]]
            new = final[changed_user]
            old = first_lab
            b.append(rng.choice([
                f"::<s>'''{old.capitalize()}'''</s> '''{new.capitalize()}''' – changed my view after the sources found above. {sig(changed_user, rng)}",
                f"::Changing to '''{new}''' given the new sources. <s>{old.capitalize()}</s> {sig(changed_user, rng)}"]))
        if "duplicate" in feats:
            plain = [p for p in posts if p[0] not in (mixed_user, changed_user)]
            u = plain[-1][0]
            lab = final[u]
            blocks.append([f"*'''{lab.capitalize()}''' Just to reiterate my view above; nothing has changed. {sig(u, rng)}"])
        if "withdrawn" in feats:
            blocks[rng.randrange(len(blocks))].append(
                f"::'''Withdrawn''' by nominator – the sources found in this discussion are enough for me. {sig(nom_user, rng)}")
        for b in blocks:
            lines += b
        text = f"AfD discussion: Synthetic test {i + 1}\nClosing result as bolded by the closer: (not shown)\n\n" + "\n".join(lines)
        out.append({"id": f"ctl{i:03d}", "hard": hard, "features": feats, "text": text, "nominator": nom_user,
                    "nominator_stance": nom_stance, "final": final,
                    "truth_A": truth["A"], "truth_B": truth["B"], "maj_A": maj(truth["A"]), "maj_B": maj(truth["B"])})
    with open(f"{ROOT}/data/judge_control.jsonl", "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print("control", len(out), Counter(f for o in out for f in o["features"]), Counter(o["maj_A"] for o in out))


def judge(model, which):
    if which == "control":
        items = [json.loads(l) for l in open(f"{ROOT}/data/judge_control.jsonl")]
        ids, texts = [o["id"] for o in items], [o["text"] for o in items]
    else:
        fn = {"cases": "afd_cases.jsonl", "more": "afd_cases_more.jsonl"}[which]
        items = [json.loads(l) for l in open(f"{ROOT}/data/{fn}")]
        ids, texts = [o["title"] for o in items], [render_raw(o) for o in items]
    llm = LLM(model, cache_name=f"p06_judge_{model}", concurrency=24)
    kw = dict(max_tokens=12000) if model.startswith(("gpt-6", "gpt-5", "o3", "o4")) else dict(max_tokens=4000)
    outs = llm.chat_many([judge_prompt(t) for t in texts], **kw)
    with open(f"{ROOT}/results/judge_{which}_{model}.jsonl", "w") as f:
        for i, o in zip(ids, outs):
            j = parse_json(o[0]["text"]) if o[0].get("finish") != "error" else None
            f.write(json.dumps({"id": i, "json": j, "finish": o[0].get("finish"), "ok": isinstance(j, dict),
                                "A": counts(j, "A"), "B": counts(j, "B")}) + "\n")
    n_ok = sum(isinstance(parse_json(o[0]["text"]), dict) for o in outs if o[0].get("finish") != "error")
    print(model, which, "parsed", n_ok, "of", len(outs))


def score_control(model):
    items = {json.loads(l)["id"]: json.loads(l) for l in open(f"{ROOT}/data/judge_control.jsonl")}
    res = [json.loads(l) for l in open(f"{ROOT}/results/judge_control_{model}.jsonl")]
    out = {}
    for sub in ("all", "easy", "hard"):
        rs = [r for r in res if sub == "all" or (items[r["id"]]["hard"] == (sub == "hard"))]
        okA = [r["A"] is not None and tuple(r["A"]) == tuple(items[r["id"]]["truth_A"]) for r in rs]
        okB = [r["B"] is not None and tuple(r["B"]) == tuple(items[r["id"]]["truth_B"]) for r in rs]
        mj = [maj(r["A"]) == items[r["id"]]["maj_A"] for r in rs]
        out[sub] = {"n": len(rs), "majority_acc": sum(mj) / len(rs), "exact_A": sum(okA) / len(rs), "exact_B": sum(okB) / len(rs)}
    feats = Counter(); fok = Counter()
    for r in res:
        for f in items[r["id"]]["features"]:
            feats[f] += 1; fok[f] += r["A"] is not None and tuple(r["A"]) == tuple(items[r["id"]]["truth_A"])
    out["exact_A_by_feature"] = {f: fok[f] / feats[f] for f in feats}
    out["pass"] = out["all"]["majority_acc"] >= 0.97 and out["all"]["exact_A"] >= 0.90
    return out


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in set(a) | set(b)) / n / n
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def score(j1, j2, which="cases"):
    import importlib.util
    spec = importlib.util.spec_from_file_location("rr", f"{ROOT}/src/recount_rules.py")
    rr = importlib.util.module_from_spec(spec); spec.loader.exec_module(rr)
    fn = {"cases": "afd_cases.jsonl", "more": "afd_cases_more.jsonl"}[which]
    cases = [json.loads(l) for l in open(f"{ROOT}/data/{fn}")]
    J = {m: {json.loads(l)["id"]: json.loads(l) for l in open(f"{ROOT}/results/judge_{which}_{m}.jsonl")} for m in (j1, j2)}
    out = {"control": {m: score_control(m) for m in (j1, j2)} if which == "cases" else None}
    rows = []
    for c in cases:
        k, d, st = rr.recount(c)
        rule_maj = "keep" if k > d else "delete" if d > k else "tie"
        row = {"title": c["title"], "group": c["group"], "outcome": c["outcome"], "rule": rule_maj, "rule_status": st}
        for m, tag in ((j1, "j1"), (j2, "j2")):
            r = J[m].get(c["title"], {})
            row[tag] = maj(r.get("A")); row[tag + "_B"] = maj(r.get("B"))
            row[tag + "_proc"] = bool((r.get("json") or {}).get("procedural_close")) if r.get("ok") else None
        for rule in ("A", "B"):
            sfx = "" if rule == "A" else "_B"
            agree = row["rule"] == row["j1" + sfx] == row["j2" + sfx] and row["rule"] in ("keep", "delete")
            consistent = (row["rule"] != c["outcome"]) == (c["group"] == "reverse")
            row["verified" + sfx] = bool(agree and consistent and row["j1_proc"] is False and row["j2_proc"] is False
                                         and st != "procedural")
        rows.append(row)
    n = len(rows)
    a = [r["rule"] for r in rows]; b = [r["j1"] or "fail" for r in rows]; cc = [r["j2"] or "fail" for r in rows]
    out["agreement"] = {"rule_j1": sum(x == y for x, y in zip(a, b)) / n, "rule_j2": sum(x == y for x, y in zip(a, cc)) / n,
                        "j1_j2": sum(x == y for x, y in zip(b, cc)) / n, "all3": sum(x == y == z for x, y, z in zip(a, b, cc)) / n,
                        "kappa_rule_j1": kappa(a, b), "kappa_rule_j2": kappa(a, cc), "kappa_j1_j2": kappa(b, cc)}
    out["verified_by_group"] = {f"{g}|{v}": n for (g, v), n in Counter((r["group"], r["verified"]) for r in rows).items()}
    out["verified_B_by_group"] = {f"{g}|{v}": n for (g, v), n in Counter((r["group"], r["verified_B"]) for r in rows).items()}
    out["why_not_verified"] = {f"{g}|{k}": n for (g, k), n in Counter(
        (r["group"], "rule_status=" + r["rule_status"] if r["rule_status"] != "holds" else
         "judge_procedural" if (r["j1_proc"] or r["j2_proc"]) else
         "judges_disagree_with_rule" if not (r["rule"] == r["j1"] == r["j2"]) else "other")
        for r in rows if not r["verified"]).items()}
    rev = [c for c in cases if c["group"] == "reverse"]; ctl = [c for c in cases if c["group"] == "control"]
    V = {r["title"]: r for r in rows}
    pairs = [(x["title"], y["title"]) for x, y in zip(rev, ctl) if V[x["title"]]["verified"] and V[y["title"]]["verified"]]
    pairs_B = [(x["title"], y["title"]) for x, y in zip(rev, ctl) if V[x["title"]]["verified_B"] and V[y["title"]]["verified_B"]]
    out["n_pairs"] = len(pairs); out["n_pairs_B"] = len(pairs_B)
    old = json.load(open(f"{ROOT}/results/recount_pairs.json"))["pairs"] if which == "cases" else []
    out["overlap_with_recount_pairs"] = len({tuple(p) for p in pairs} & {tuple(p) for p in old})
    json.dump(out, open(f"{ROOT}/results/verify_votes{'' if which == 'cases' else '_' + which}.json", "w"), indent=1)
    json.dump({"pairs": pairs, "pairs_B": pairs_B, "judges": [j1, j2]},
              open(f"{ROOT}/results/verified_pairs{'' if which == 'cases' else '_' + which}.json", "w"), indent=1)
    with open(f"{ROOT}/results/verify_votes_rows{'' if which == 'cases' else '_' + which}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "control":
        build_control()
    elif cmd == "judge":
        judge(sys.argv[2], sys.argv[3])
    elif cmd == "scorecontrol":
        print(json.dumps(score_control(sys.argv[2]), indent=1))
    elif cmd == "score":
        score(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "cases")
