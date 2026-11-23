"""P06 S5 (supervised baselines) and S10 (model scale and generation), on the verified pairs.

S5: every baseline's saved predictions (results/baseline_preds/*.parquet): agreement and balanced accuracy on
    reverse and control cases, share following the majority, and the correlation of p(keep) with the strict-recount
    keep share; mean and SD over seeds; the cross-fitted runs assembled from their held-out folds.
S10: for each model of the ladder (results/supp_s10_<alias>.parquet) and the four main models (supp_core): reverse and
    control balanced accuracy, their gap, the majority coefficient of logit(pred keep ~ closer kept + majority kept),
    and the net shift after 8 bare votes for the loser and for the winner.
python s5_s10_analysis.py   Output: results/s5_baselines.json, results/s10_ladder.json
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, f"{WORKSPACE_ROOT}/src")
from recount_rules import recount  # noqa: E402

ROOT = WORKSPACE_ROOT
PAIRS = os.environ.get("PAIRS", "verified")  # verified: the 150 pairs; combined: plus the 212 S8 pairs
SUF = "" if PAIRS == "verified" else f"_{PAIRS}"
cases = {r["title"]: r for r in map(json.loads, open(f"{ROOT}/data/afd_cases.jsonl"))}
pairs = [tuple(p) for p in json.load(open(f"{ROOT}/results/verified_pairs.json"))["pairs"]]
if PAIRS == "combined":
    cases.update({r["title"]: r for r in map(json.loads, open(f"{ROOT}/data/afd_cases_more.jsonl"))})
    pairs += [tuple(p) for p in json.load(open(f"{ROOT}/results/verified_pairs_more.json"))["pairs"]]
REV = [a for a, _ in pairs]; CTL = [b for _, b in pairs]
share = {t: (lambda k, d: k / (k + d))(*recount(c)[:2]) for t, c in cases.items()}


def balacc(titles, pred):
    d = pd.DataFrame({"o": [cases[t]["outcome"] for t in titles], "a": [pred[t] == cases[t]["outcome"] for t in titles]})
    return float(d.groupby("o").a.mean().mean())


def metrics(pk):
    """pk: title -> p(keep)."""
    pred = {t: ("keep" if p >= 0.5 else "delete") for t, p in pk.items()}
    r = [t for t in REV if t in pred]; c = [t for t in CTL if t in pred]
    maj = {t: ("keep" if share[t] > 0.5 else "delete") for t in r + c}
    x = np.array([pk[t] for t in r + c]); s = np.array([share[t] for t in r + c])
    return {"n_rev": len(r), "n_ctl": len(c), "agree_rev": float(np.mean([pred[t] == cases[t]["outcome"] for t in r])),
            "agree_ctl": float(np.mean([pred[t] == cases[t]["outcome"] for t in c])), "balacc_rev": balacc(r, pred), "balacc_ctl": balacc(c, pred),
            "follow_majority_rev": float(np.mean([pred[t] == maj[t] for t in r])), "corr_pkeep_keepshare": float(np.corrcoef(x, s)[0, 1])}


out = {}
B = {}
for f in sorted(glob.glob(f"{ROOT}/results/baseline_preds/*.parquet")):
    tag = os.path.basename(f)[:-8]
    d = pd.read_parquet(f)
    B[tag] = dict(zip(d.title, d.p_keep))
groups = {}
for tag, pk in B.items():
    m = re.match(r"(.+?)(?:_s(\d))?(?:_xfit(\d))?$", tag)
    base, seed, fold = m.group(1), m.group(2), m.group(3)
    if fold is not None:
        groups.setdefault(f"{base}_xfit", {}).update(pk)  # each candidate predicted by its held-out fold
    else:
        groups.setdefault(base, []).append(metrics(pk))
for k, v in groups.items():
    if isinstance(v, dict):
        out[k] = {"seeds": 1, **metrics(v)}
    else:
        out[k] = {"seeds": len(v), **{m: float(np.mean([x[m] for x in v])) for m in v[0]},
                  **{m + "_sd": float(np.std([x[m] for x in v])) for m in ("balacc_rev", "balacc_ctl", "follow_majority_rev")}}
json.dump(out, open(f"{ROOT}/results/s5_baselines{SUF}.json", "w"), indent=1)
for k, v in out.items():
    print(f"S5 {k:28s} seeds {v['seeds']} rev {v['balacc_rev']:.3f} ctl {v['balacc_ctl']:.3f} follow_rev {v['follow_majority_rev']:.3f} "
          f"corr(p,keep share) {v['corr_pkeep_keepshare']:.3f}")

LADDER = {"qwen3.5-2b": ("Qwen3.5-2B", 2, "2026-02"), "qwen3.5-4b": ("Qwen3.5-4B", 4, "2026-02"), "local": ("Qwen3.5-9B", 9, "2026-02"),
          "qwen3.5-9b-think": ("Qwen3.5-9B thinking", 9, "2026-02"), "qwen3.8-27b": ("Qwen3.8-27B", 27, "2026-08"),
          "granite-4.2-8b": ("Granite-4.2-8B", 8, "2026-08"), "gemma-4-12b": ("Gemma-4-12B", 12, "2026-05"),
          "gpt-4o-mini": ("gpt-4o-mini", None, "2024-07"), "gpt-6-luna": ("gpt-6-luna", None, None),
          "gpt-6-luna@low": ("gpt-6-luna (low effort)", None, None), "gpt-6-luna@high": ("gpt-6-luna (high effort)", None, None)}
res = {}
for alias, (name, size, date) in LADDER.items():
    fs = [f"{ROOT}/results/supp_s10_{alias}.parquet", f"{ROOT}/results/supp_core_{alias}.parquet"]
    if PAIRS == "combined":
        fs.append(f"{ROOT}/results/supp_s10_more_{alias}.parquet")
    fs = [f for f in fs if os.path.exists(f)]
    if not fs:
        continue
    df = pd.concat([pd.read_parquet(f) for f in fs])
    df = df[~df.error.astype(bool)].drop_duplicates(["title", "variant"])
    df["p_out"] = np.where(df.outcome == "keep", df.p_keep, 1 - df.p_keep)
    o = df[df.variant == "orig"].set_index("title")
    titles = [t for t in REV + CTL if t in o.index]
    if len(titles) < 50:
        continue
    pk = o.p_keep.to_dict()
    r = {"name": name, "size_b": size, "release": date, **metrics({t: pk[t] for t in titles})}
    oo = o.loc[titles].assign(y=lambda x: (x.pred == "keep").astype(int), ck=lambda x: (x.outcome == "keep").astype(int),
                              mk=lambda x: (x.majority == "keep").astype(int))
    try:
        f = smf.logit("y ~ ck + mk", oo).fit(disp=0)
        r["logit_majority"] = [float(f.params.mk), float(f.bse.mk), float(f.pvalues.mk)]
    except Exception as e:  # noqa: BLE001
        r["logit_error"] = str(e)[:80]
    for v, sign in (("inject8", 1), ("win8", -1), ("inject2", 1), ("inject4", 1)):
        a = df[df.variant == v].set_index("title").p_out
        t = [x for x in titles if x in a.index]
        if t:
            d = sign * (o.p_out.loc[t] - a.loc[t]).to_numpy()
            bs = d[np.random.default_rng(0).integers(0, len(d), (2000, len(d)))].mean(1)
            r[f"shift_{v}"] = [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    res[alias] = r
json.dump(res, open(f"{ROOT}/results/s10_ladder{SUF}.json", "w"), indent=1, default=float)
for a, r in res.items():
    print(f"S10 {r['name']:26s} rev {r['balacc_rev']:.3f} ctl {r['balacc_ctl']:.3f} gap {r['balacc_ctl'] - r['balacc_rev']:+.3f} "
          f"maj-coef {r.get('logit_majority', [np.nan])[0]:.2f} inj8 {r.get('shift_inject8', [np.nan])[0]:+.3f} win8 {r.get('shift_win8', [np.nan])[0]:+.3f}")
