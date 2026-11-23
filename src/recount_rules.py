"""Stricter rule recount of a parsed AfD discussion (same rules as recount_reanalysis.py).

Counts nested votes in replies, applies struck and changed votes, drops the nominator's delete vote
when the nomination was withdrawn, and flags procedural closes.
recount(case) -> (keep, delete, status) with status in holds / flips / tie / procedural
  holds: the recounted majority still puts the case in its group (reverse or control)
python recount_rules.py   writes results/recount_pairs.json (the pairs used in the draft)
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import json
import re

VOTE = r"(?:strong(?:ly)?|weak|speedy|leaning?|very weak|snow)?\s*(keep|delete)"
STRUCK = re.compile(r"<(s|del|strike)>.*?</\1>", re.I | re.S)


def vote_in(line):
    t = STRUCK.sub(" ", line)
    t = re.sub(r"'''\s*'''", " ", t)
    m = re.match(r"^[*#:\s]*(?:<[^>]+>\s*)*'''+\s*" + VOTE, t[:120], re.I)
    if m:
        return m.group(1).lower()
    m = re.search(r"'''+\s*(?:changed? to|now|switch(?:ing|ed)? to)\s*" + VOTE, t[:200], re.I)
    return m.group(1).lower() if m else None


def recount(r):
    k = d = 0
    withdrawn = False
    for e in r["entries"]:
        for i, l in enumerate([e["text"]] + list(e.get("replies", []))):
            if re.search(r"'''[^']{0,30}withdraw", l, re.I) or re.search(r"withdraw(?:n|ing)? (?:this |my |the )?(?:nom|afd)", l, re.I):
                withdrawn = True
            v = vote_in(l)
            if i == 0 and v is None and e["label"] in ("keep", "delete") and not STRUCK.search(l[:120]):
                v = e["label"]
            k += v == "keep"; d += v == "delete"
    d += not withdrawn
    if re.search(r"speedy|speedily|procedural|g\d+|copyvio|withdraw", r.get("close_text", ""), re.I):
        return k, d, "procedural"
    if k == d:
        return k, d, "tie"
    maj = "keep" if k > d else "delete"
    group = r.get("group", "reverse" if maj != r["outcome"] else "control")
    return k, d, ("holds" if (maj != r["outcome"]) == (group == "reverse") else "flips")


if __name__ == "__main__":
    ROOT = WORKSPACE_ROOT
    cases = [json.loads(l) for l in open(f"{ROOT}/data/afd_cases.jsonl")]
    rev = [c for c in cases if c["group"] == "reverse"]; ctl = [c for c in cases if c["group"] == "control"]
    pairs = [(a["title"], b["title"]) for a, b in zip(rev, ctl) if recount(a)[2] == "holds" and recount(b)[2] == "holds"]
    json.dump({"pairs": pairs}, open(f"{ROOT}/results/recount_pairs.json", "w"), indent=1)
    print("recount pairs", len(pairs))
