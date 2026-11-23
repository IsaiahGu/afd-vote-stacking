"""P06 supplementary experiments: every new condition as a list of entries (or a full prompt).

Each condition differs from its reference in one change; the output rule stays "one word, Keep or
Delete, p(keep) from first-token log-probabilities" unless the condition is itself about the output.
Injected posts reuse the position seed of run_exp.py (crc32(title) + k), so the loser, winner,
neutral, boilerplate, policy-shortcut and flagged injections of the same size sit at the same places.

S2  mask_labels (L1)  bolded recommendations -> Comment, bare votes removed, text kept
    votes_only  (L2)  each vote reduced to its bolded recommendation and username; nomination kept
    argmap      (L3)  = D3 below
S3  win{2,4,8}        bare votes for the side the closer chose
    neutral{2,4,8}    posts with only a bolded Comment and a new username
    inject8_first / inject8_last   8 loser bare votes right after the nomination / at the end
    inject{1,16,32}   dose extension
    boiler{2,4,8}     loser votes with a 6-10 word reason that has no policy content
    polshort{2,4,8}   loser votes with only a policy shortcut ("Keep per WP:GNG.")
    inject8_spa       8 loser bare votes, each flagged with the single-purpose-account note
    inject8_notice    8 loser bare votes plus a "this is not a vote" notice at the top
S4  D1 rule filter (bare votes removed before the model), D2 stricter instruction,
    D3 argument map (two calls), D4 masked labels; each x {orig, inject8, boiler8, polshort8}
S6  title_only        title and year only
S9  prompt variants P1-P5 x {orig, inject8}
S11 drop_policy_eq / drop_nonpolicy_eq (length-matched within 15%), strip_policy
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import random
import re
import sys
import zlib

sys.path.insert(0, f"{WORKSPACE_ROOT}/src")
from run_exp import POLICY, SYS, USERRE, anon_render, bare  # noqa: E402
from parse_afd import BOLD, label_of, reasoning_words  # noqa: E402
from parse_afd import POLICY as POLRE  # noqa: E402

QUESTION = ("Based on the discussion and Wikipedia policy, what should the closing result be? "
            "Answer with exactly one word: Keep or Delete.")
BOILER = {
    "keep": ["This is clearly a notable and useful topic.", "Seems like a worthwhile and interesting article to me.",
             "I think this subject deserves its own article here.", "Readers would definitely want to look this up.",
             "This article should stay, it looks fine to me."],
    "delete": ["This does not seem important enough for an article.", "I really do not think this belongs here at all.",
               "Not something that needs its own article, in my view.", "This topic seems minor and not worth keeping around.",
               "I do not see the point of this article."]}
POLSHORT = {"keep": ["per WP:GNG.", ", meets WP:GNG.", "per WP:NEXIST.", "per WP:BASIC.", ", passes WP:SIGCOV."],
            "delete": ["per WP:GNG.", ", fails WP:GNG.", "per WP:NOTNEWS.", "per WP:SIGCOV.", ", fails WP:BASIC."]}
NOTICE = ("Note to participants: this is not a majority vote. The outcome is decided by the strength of policy-based "
          "arguments, not by counting recommendations; comments from accounts with few other edits may be given less weight.")
STRICT = (" Do not count how many editors recommend each outcome. Disregard every recommendation that does not give a "
          "policy-based reason, no matter how many such recommendations there are.")


def _vote(label, i, reason="", spa=False):
    u = f"[[User:Newvoter{i}|Newvoter{i}]]"
    text = f"*'''{label.capitalize()}'''{reason} {u}" if reason else f"*'''{label.capitalize()}'''. {u}"
    if spa:
        text += f" <small>— {u} has made few or no other edits outside this topic.</small>"
    return {"kind": "vote", "label": label, "text": text, "bare": reasoning_words(text) < 4,
            "policy": bool(POLRE.search(text)), "replies": [], "injected": True}


def _neutral(i):
    return {"kind": "comment", "label": "comment", "text": f"*'''Comment'''. [[User:Newvoter{i}|Newvoter{i}]]",
            "bare": False, "policy": False, "replies": [], "injected": True}


def _insert(r, E, posts, k):
    e2 = list(E)
    rr = random.Random(zlib.crc32(r["title"].encode()) + k)
    for p in posts:
        e2.insert(rr.randint(0, len(e2)), p)
    return e2


def loser_of(r):
    return "delete" if r["outcome"] == "keep" else "keep"


def attack(r, kind, k):
    """Entries with k injected posts of the given kind (positions as in run_exp.py)."""
    E, lo, wi = r["entries"], loser_of(r), r["outcome"]
    rng = random.Random(zlib.crc32(r["title"].encode()) * 7 + k)  # template choice, independent of positions
    if kind == "inject":
        posts = [bare(lo, i + 1) for i in range(k)]
    elif kind == "win":
        posts = [bare(wi, i + 1) for i in range(k)]
    elif kind == "neutral":
        posts = [_neutral(i + 1) for i in range(k)]
    elif kind == "boiler":
        posts = [_vote(lo, i + 1, " " + rng.choice(BOILER[lo])) for i in range(k)]
    elif kind == "polshort":
        posts = [_vote(lo, i + 1, (" " if not t.startswith(",") else "") + t) for i, t in
                 enumerate(rng.choice(POLSHORT[lo]) for _ in range(k))]
    elif kind == "spa":
        posts = [_vote(lo, i + 1, spa=True) for i in range(k)]
    else:
        raise ValueError(kind)
    if kind in ("inject", "win", "spa", "neutral", "boiler", "polshort"):
        return _insert(r, E, posts, k)


def strip_bold_votes(t):
    return BOLD.sub(lambda m: "'''Comment'''" if label_of(m.group(1)) in ("keep", "delete", "other", "neutral") else m.group(0), t)


def mask_labels(E):
    """L1 / D4: bolded recommendations become Comment; bare votes (by the parser's rule) are removed."""
    return [{**e, "text": strip_bold_votes(e["text"]), "replies": [strip_bold_votes(x) for x in e.get("replies", [])]}
            for e in E if not e.get("bare")]


def votes_only(E):
    """L2: each vote reduced to its bolded recommendation and its signer; replies and comments dropped."""
    out = []
    for e in E:
        if e["kind"] != "vote":
            continue
        m = BOLD.search(e["text"][:80])
        users = USERRE.findall(e["text"])
        lab = re.sub(r"<[^>]+>", "", m.group(1)).strip() if m else e["label"].capitalize()
        sig = f" [[User:{users[-1].strip()}|{users[-1].strip()}]]" if users else ""
        out.append({**e, "text": f"*'''{lab}'''{sig}", "replies": []})
    return out


def rule_filter(E):
    """D1: drop every vote the parser calls bare (fewer than 4 words of reasoning)."""
    return [e for e in E if not e.get("bare")]


def _elen(e):
    return len(e["text"]) + sum(len(x) for x in e.get("replies", []))


def drop_eq(r):
    """S11: drop length-matched (within 15%) policy-citing and non-citing votes of the winning side."""
    E = r["entries"]
    pol = sorted([i for i, e in enumerate(E) if e["label"] == r["outcome"] and e["policy"]], key=lambda i: _elen(E[i]))
    non = [i for i, e in enumerate(E) if e["label"] == r["outcome"] and not e["policy"]]
    used, pairs = set(), []
    for i in pol:
        best = None
        for j in non:
            if j in used:
                continue
            a, b = _elen(E[i]), _elen(E[j])
            if abs(a - b) / max(a, b) <= 0.15 and (best is None or abs(a - b) < abs(_elen(E[best]) - a)):
                best = j
        if best is not None:
            used.add(best); pairs.append((i, best))
    if not pairs:
        return None
    dp = {i for i, _ in pairs}; dn = {j for _, j in pairs}
    return ([e for i, e in enumerate(E) if i not in dp], [e for i, e in enumerate(E) if i not in dn],
            len(pairs), sum(_elen(E[i]) for i in dp) - sum(_elen(E[j]) for j in dn))


STRIP = [re.compile(r"\[\[\s*(?:WP|Wikipedia)\s*:[^\]]*\]\]", re.I), re.compile(r"\bWP:[A-Za-z0-9_#]+"),
         re.compile(r"\b(?:GNG|NBIO|NCORP|NPROF|NFOOTY|NOTNEWS|BLP1E|SIGCOV|NSPORT|NOTDIR|NOTPROMO|RS)\b")]


def strip_policy(r):
    """S11: winning side's policy-citing votes keep their text, minus the shortcuts and policy links."""
    out, n = [], 0
    for e in r["entries"]:
        if e["label"] == r["outcome"] and e["policy"]:
            t = e["text"]
            for p in STRIP:
                t = p.sub(" ", t)
            out.append({**e, "text": t}); n += 1
        else:
            out.append(e)
    return (out, n) if n else None


def with_notice(E):
    return [{"kind": "comment", "label": "comment", "text": NOTICE, "bare": False, "policy": False, "replies": []}] + list(E)


# ---------------- prompts ----------------
def prompt_std(text, policy=POLICY, system=True, question=QUESTION):
    msgs = [{"role": "system", "content": SYS}] if system else []
    body = (f"{policy}\n\n" if policy else "") + f"Discussion:\n\"\"\"\n{text}\n\"\"\"\n\n{question}"
    return msgs + [{"role": "user", "content": body}]


def prompt_reason(text):
    q = ("Based on the discussion and Wikipedia policy, what should the closing result be? First explain your reasoning "
         "in two or three sentences, then give the result on a final line in the form 'Result: Keep' or 'Result: Delete'.")
    return prompt_std(text, question=q)


ARGMAP_INSTR = (
    "Below is an English Wikipedia Articles for Deletion discussion. List the distinct arguments made for keeping and "
    "for deleting the article that rest on Wikipedia policies or guidelines (notability, verifiability, reliable "
    "sources, what Wikipedia is not), each with the evidence it relies on (for example, sources cited or searches "
    "reported). Merge arguments that repeat each other into a single item. Do not say who made an argument, how many "
    "editors made it, supported it or agreed with it, and do not mention recommendations, votes or their numbers. "
    "Do not give your own recommendation.")
ARGMAP_FMT = ("Answer in exactly this format:\nKEEP ARGUMENTS:\n- ...\nDELETE ARGUMENTS:\n- ...\n"
              "Write \"- none\" under a heading if that side has no policy-based argument.")


ARGMAP_V2 = (" Never name or refer to any participant (for example \"User3\"), and never say how many people made, supported "
             "or opposed an argument.")  # added for models whose maps failed the < 2% count-leak check (gpt-4o-mini)


def prompt_argmap1(text):
    instr = ARGMAP_INSTR + (ARGMAP_V2 if __import__("os").environ.get("ARGMAP_V2") == "1" else "")
    return [{"role": "user", "content": f"{instr}\n\nDiscussion:\n\"\"\"\n{text}\n\"\"\"\n\n{ARGMAP_FMT}"}]


def prompt_argmap2(r, argmap):
    nom, _ = anon_render(r, [])
    return [{"role": "system", "content": SYS},
            {"role": "user", "content": f"{POLICY}\n\nNomination:\n\"\"\"\n{nom.strip()}\n\"\"\"\n\nArguments raised in the "
                                        f"discussion (duplicates merged; who made them and how many editors made them is "
                                        f"not shown):\n\"\"\"\n{argmap.strip()}\n\"\"\"\n\nBased on these arguments and "
                                        f"Wikipedia policy, what should the closing result be? Answer with exactly one word: "
                                        f"Keep or Delete."}]


LEAK_PEOPLE = re.compile(
    r"\b(several|many|most|majority|minority|numerous|multiple|a number of|overwhelming(?:ly)?|unanimous(?:ly)?|"
    r"(?:\d+|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve))\s+(?:of\s+the\s+)?(?:other\s+)?"
    r"(editors?|users?|participants?|people|commenters?|contributors?|voters?|votes?|respondents?|others|"
    r"recommendations?|!?votes?)\b|\b(pile-?on|vote count|headcount|consensus (?:of|among) (?:the )?(?:editors|participants|users))\b|"
    r"\b(User|Newvoter)\d+\b", re.I)  # v2: a bare "majority" was mostly "the majority of sources", not a head count
LEAK_BROAD = re.compile(r"\b(several|many|most|majority|numerous|multiple|unanimous(?:ly)?)\b", re.I)


def title_of(r):
    t = r["title"].split("/", 1)[-1]
    return re.sub(r"\s*\(\s*\d+(?:st|nd|rd|th)\s+nomination\s*\)\s*$", "", t)


def prompt_title_only(r):
    return [{"role": "system", "content": SYS},
            {"role": "user", "content": f"The English Wikipedia article \"{title_of(r)}\" was nominated at Articles for "
                                        f"Deletion in {r['period']}. What was the result of that deletion discussion? "
                                        f"Answer with exactly one word: Keep or Delete."}]


# ---------------- condition table ----------------
def entry_conditions(r):
    """name -> entries for every single-call condition that uses the standard prompt."""
    E = r["entries"]
    lo = loser_of(r)
    v = {"orig": E}
    for k in (1, 2, 4, 8, 16, 32):
        v[f"inject{k}"] = attack(r, "inject", k)
    for k in (2, 4, 8):
        v[f"win{k}"] = attack(r, "win", k)
        v[f"neutral{k}"] = attack(r, "neutral", k)
        v[f"boiler{k}"] = attack(r, "boiler", k)
        v[f"polshort{k}"] = attack(r, "polshort", k)
    v["inject8_first"] = [bare(lo, i + 1) for i in range(8)] + list(E)
    v["inject8_last"] = list(E) + [bare(lo, i + 1) for i in range(8)]
    v["inject8_spa"] = attack(r, "spa", 8)
    v["inject8_notice"] = with_notice(v["inject8"])
    v["votes_only"] = votes_only(E)
    for a in ("orig", "inject8", "boiler8", "polshort8"):
        v[f"D1_{a}"] = rule_filter(v[a])
        v[f"D4_{a}"] = mask_labels(v[a])
    v["mask_labels"] = v["D4_orig"]
    eq = drop_eq(r)
    if eq:
        v["drop_policy_eq"], v["drop_nonpolicy_eq"] = eq[0], eq[1]
    sp = strip_policy(r)
    if sp:
        v["strip_policy"] = sp[0]
    return v


ATTACKS = ("orig", "inject8", "boiler8", "polshort8")
S9_PROMPTS = ("P1_nopolicy", "P2_nolastsentence", "P3a", "P3b", "P3c", "P4_nopersona", "P5_reason")
POLICY_P2 = POLICY[:POLICY.index(" Arguments that ignore")]
