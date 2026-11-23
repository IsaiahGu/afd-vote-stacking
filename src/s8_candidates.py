"""P06 S8: more reverse-majority candidates from a strict recount, over the parsed data and new years.

1. Strict recount (recount_rules.py: nested and changed votes, struck votes, nominator withdrawal) of every
   parsed discussion (data/afd_parsed.jsonl.gz) and of the newly fetched years (data/afd_raw_more.jsonl,
   parsed with parse_afd.py). Eligible: keep/delete close, not procedural, strict keep + delete >= 5, no tie.
2. Report how the loose parser's selection compares: strict-reverse cases the loose rule missed, loose-reverse
   cases the strict count rejects (the loose parser's miss and false-alarm rates).
3. New candidates = strict-reverse cases not among the 278 already used; each gets a pro-majority control from
   the same year matched on strict margin, strict total and length (nearest neighbour, without replacement).
Output: data/afd_parsed_more.jsonl.gz, data/afd_cases_more.jsonl, results/s8_screen.json
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import gzip
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, f"{WORKSPACE_ROOT}/src")
from parse_afd import parse  # noqa: E402
from recount_rules import recount  # noqa: E402

ROOT = WORKSPACE_ROOT
rows = [json.loads(l) for l in gzip.open(f"{ROOT}/data/afd_parsed.jsonl.gz", "rt")]
have = {r["title"] for r in rows}
new = []
raw = f"{ROOT}/data/afd_raw_more.jsonl"
if os.path.exists(raw) and os.environ.get("ONLY_PARSED") != "1":
    seen = set()
    for l in open(raw):
        j = json.loads(l)
        if j["title"] in seen or j["title"] in have:
            continue
        seen.add(j["title"])
        p = parse(j)
        if p:
            new.append(p)
    with gzip.open(f"{ROOT}/data/afd_parsed_more.jsonl.gz", "wt") as f:
        for p in new:
            f.write(json.dumps(p) + "\n")
used = {json.loads(l)["title"] for l in open(f"{ROOT}/data/afd_cases.jsonl")}
stats = Counter()
strict = []
for src, R in (("parsed", rows), ("new", new)):
    for r in R:
        if r["outcome"] not in ("keep", "delete"):
            continue
        k, d, st = recount({**r, "group": "x"})
        loose_el = r["n_keep"] + r["n_delete"] >= 5 and r["n_keep"] != r["n_delete"]
        loose_rev = loose_el and ((r["n_keep"] > r["n_delete"]) != (r["outcome"] == "keep"))
        el = k + d >= 5 and k != d and st != "procedural"
        rev = el and ((k > d) != (r["outcome"] == "keep"))
        stats[(src, "strict_eligible")] += el; stats[(src, "strict_reverse")] += rev
        stats[(src, "loose_reverse")] += loose_rev
        stats[(src, "strict_rev_not_loose")] += rev and not loose_rev
        stats[(src, "loose_rev_not_strict")] += loose_rev and not rev
        if el:
            strict.append({**r, "src": src, "n_keep_loose": r["n_keep"], "n_delete_loose": r["n_delete"], "n_keep": k, "n_delete": d,
                           "majority": "keep" if k > d else "delete", "reverse": rev, "margin": abs(k - d), "tot": k + d})
pairs = []
for src in ("parsed", "new"):  # matched within source, so adding fetched years never changes the parsed-data pairs
    rev = [r for r in strict if r["src"] == src and r["reverse"] and r["title"] not in used]
    rng = np.random.default_rng(0)
    rng.shuffle(rev)
    pool = {}
    for r in strict:
        if r["src"] == src and not r["reverse"] and r["title"] not in used:
            pool.setdefault(r["period"], []).append(r)
    taken = set()
    for r in rev:
        cands = [c for c in pool.get(r["period"], []) if c["title"] not in taken]
        if not cands:
            continue
        c = min(cands, key=lambda c: abs(c["margin"] - r["margin"]) * 3 + abs(c["tot"] - r["tot"]) + abs(np.log1p(c["len"]) - np.log1p(r["len"])) * 2)
        taken.add(c["title"]); pairs.append((r, c))
with open(f"{ROOT}/data/afd_cases_more.jsonl", "w") as f:
    for r, _ in pairs:
        f.write(json.dumps({**r, "group": "reverse"}) + "\n")
    for _, c in pairs:
        f.write(json.dumps({**c, "group": "control"}) + "\n")
out = {"parsed_discussions": len(rows), "new_discussions": len(new), "stats": {f"{a}|{b}": v for (a, b), v in stats.items()},
       "new_candidate_pairs": len(pairs), "new_pairs_by_year": dict(Counter(r["period"] for r, _ in pairs)),
       "new_pairs_by_source": dict(Counter(r["src"] for r, _ in pairs))}
json.dump(out, open(f"{ROOT}/results/s8_screen.json", "w"), indent=1)
print(json.dumps(out, indent=1))
