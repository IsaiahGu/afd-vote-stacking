"""P06 step 1: parse AfD wikitext into nomination, bolded votes, comments and the close.

Outcome: keep / delete only (no consensus, merge, redirect, ... excluded).
Votes: top-level bullets whose first bold word is a recommendation; the nomination counts as a
delete vote. Effective votes = keep + delete >= 5, ties excluded.
Reverse-majority case: closer's result is opposite to the vote majority.
Bare vote: fewer than 4 words of reasoning after removing the label, signature, timestamps and
"per X" references.
Output: data/afd_parsed.jsonl
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import json
import re
import sys
from collections import Counter

ROOT = WORKSPACE_ROOT
TS = re.compile(r"\d{1,2}:\d{2},? \d{1,2} [A-Z][a-z]+ \d{4} \(UTC\)")
USERLINK = re.compile(r"\[\[\s*(?:User|User talk|Special:Contributions)\s*:\s*([^|\]/#]+)[^\]]*\]\]", re.I)
POLICY = re.compile(r"\bWP:[A-Z]|\[\[\s*(?:WP|Wikipedia)\s*:|\b(?:GNG|NBIO|NCORP|NPROF|NFOOTY|NOTNEWS|BLP1E|SIGCOV|NSPORT|NOTDIR|NOTPROMO|NOT|RS|V)\b")
BOLD = re.compile(r"'''+\s*(.+?)\s*'''+")


def label_of(b):
    b = re.sub(r"<[^>]+>|\[\[|\]\]", "", b).lower().strip()
    if re.match(r"^(comment|note|question|reply|response|info|query|relist|closing)", b):
        return "comment"
    k, d = "keep" in b, "delete" in b
    if k and not d:
        return "keep"
    if d and not k:
        return "delete"
    if re.search(r"merge|redirect|draftify|userfy|transwiki|incubate|move|rename|smerge", b):
        return "other"
    if "neutral" in b:
        return "neutral"
    return "other" if (k or d) else "unknown"


def outcome_of(b):
    b = b.lower()
    if "no consensus" in b:
        return "nc"
    if re.search(r"merge|redirect|draft|userf|withdraw|transwiki|incubat|move", b):
        return "other"
    k, d = "keep" in b, "delete" in b
    if k and not d:
        return "keep"
    if d and not k:
        return "delete"
    return "other"


def reasoning_words(line):
    t = USERLINK.sub(" ", line)
    t = TS.sub(" ", t)
    t = re.sub(r"\{\{[^}]*\}\}", " ", t)
    m = BOLD.search(t)
    if m:
        t = t[:m.start()] + " " + t[m.end():]
    t = re.sub(r"\bper (?:nom(?:inator)?|above|[A-Z][\w .-]{0,30})", " ", t, flags=re.I)
    t = re.sub(r"\(talk\)|talk|contribs|\[\[[^\]]*\]\]|[*:#'\-–—(),.!?]", " ", t)
    return len([w for w in t.split() if re.search(r"[A-Za-z]", w)])


def parse(rec):
    w = rec["wikitext"]
    m = re.search(r"The result (?:of the debate )?was\s*(?::\s*)?'''+(.+?)'''+", w, re.I)
    if not m:
        return None
    outcome = outcome_of(m.group(1))
    hdr = re.search(r"^===+\s*\[\[.*?\]\]\s*===+\s*$", w, re.M)
    if not hdr:
        return None
    body = w[hdr.end():]
    endm = re.search(r"The above discussion is preserved as an archive", body)
    if endm:
        body = body[:endm.start()]
    lines = [l.rstrip() for l in body.split("\n")]
    keep_lines, entries, nom_lines, seen_bullet = [], [], [], False
    for l in lines:
        s = l.strip()
        if not s or re.match(r"^:?\s*\{\{\s*(la|lo|ln|Find sources|findsources)", s, re.I) or "delsort-notice" in s \
                or re.match(r"^:?\s*\(\{\{Find sources", s) or re.search(r"\{\{\s*Relist|xfd_relist|Relisted to generate", s, re.I) \
                or s.startswith("<div") or s.startswith("</div") or s.startswith(":''The following discussion"):
            continue
        if re.match(r"^[*#](?![*:#])", s):
            seen_bullet = True
            bm = BOLD.search(s[:80])
            lab = label_of(bm.group(1)) if bm else "comment"
            if bm and re.search(r"<(s|del|strike)>", s[:bm.start() + 1], re.I):
                lab = "comment"  # struck-through (withdrawn) vote
            entries.append({"kind": "vote" if lab in ("keep", "delete", "other", "neutral") else "comment", "label": lab,
                            "text": s, "bare": reasoning_words(s) < 4 if lab in ("keep", "delete") else False,
                            "policy": bool(POLICY.search(s)), "replies": []})
        elif re.match(r"^[*#:]+", s) and entries:
            entries[-1]["replies"].append(s)
        elif not seen_bullet:
            nom_lines.append(s)
        elif entries:
            entries[-1]["replies"].append(s)
    if not nom_lines:
        return None
    c = Counter(e["label"] for e in entries if e["kind"] == "vote")
    keep, delete = c["keep"], c["delete"] + 1  # nominator = delete
    return {"title": rec["title"], "period": rec["period"], "log_date": rec["log_date"], "outcome": outcome,
            "close_text": m.group(1), "nomination": "\n".join(nom_lines), "entries": entries,
            "n_keep": keep, "n_delete": delete, "n_other": c["other"] + c["neutral"],
            "n_bare_keep": sum(e["bare"] for e in entries if e["label"] == "keep"),
            "n_bare_delete": sum(e["bare"] for e in entries if e["label"] == "delete"),
            "n_policy": sum(e["policy"] for e in entries), "len": len(body)}


if __name__ == "__main__":
    n = ok = 0
    st = Counter()
    with open(f"{ROOT}/data/afd_parsed.jsonl", "w") as f:
        import glob
        lines = (l for p in sorted(glob.glob(f"{ROOT}/data/afd_raw*.jsonl")) for l in open(p))
        seen = set()
        for line in lines:
            if json.loads(line)["title"] in seen:
                continue
            seen.add(json.loads(line)["title"])
            n += 1
            r = parse(json.loads(line))
            if r is None:
                st["unparsed"] += 1; continue
            st[r["outcome"]] += 1
            f.write(json.dumps(r) + "\n"); ok += 1
    print("raw", n, "parsed", ok, dict(st))
