"""P06 S10 ladder and S5 cross-fitted Ettin on the combined 362 pairs: pair-bootstrap CIs for reverse and
control balanced accuracy and the gap (control minus reverse), the majority coefficient, and paired
differences between models. Same definitions as src/s5_s10_analysis.py; read-only, prints only.
python ladder_ci.py > results/s10_ladder_ci_combined.txt"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import json
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = WORKSPACE_ROOT
R = f"{ROOT}/results"
cases = {r["title"]: r for r in map(json.loads, open(f"{ROOT}/data/afd_cases.jsonl"))}
cases.update({r["title"]: r for r in map(json.loads, open(f"{ROOT}/data/afd_cases_more.jsonl"))})
pairs = [tuple(p) for p in json.load(open(f"{R}/verified_pairs.json"))["pairs"]]
pairs += [tuple(p) for p in json.load(open(f"{R}/verified_pairs_more.json"))["pairs"]]

LADDER = ["qwen3.5-2b", "qwen3.5-4b", "local", "qwen3.5-9b-think", "qwen3.8-27b", "granite-4.2-8b",
          "gemma-4-12b", "gpt-4o-mini", "gpt-6-luna"]
NAME = {"local": "qwen3.5-9b"}


def preds(alias):
    fs = [f"{R}/supp_s10_{alias}.parquet", f"{R}/supp_core_{alias}.parquet", f"{R}/supp_s10_more_{alias}.parquet"]
    df = pd.concat([pd.read_parquet(f) for f in fs if os.path.exists(f)])
    df = df[~df.error.astype(bool)].drop_duplicates(["title", "variant"])
    o = df[df.variant == "orig"].set_index("title")
    MAJ.update(o.majority.to_dict())
    return o.pred.to_dict()


MAJ = {}
PRED = {a: preds(a) for a in LADDER}
x = pd.concat([pd.read_parquet(f"{R}/baseline_preds/ettin_lr1e-05_s0_xfit{k}.parquet") for k in range(5)])
PRED["ettin_xfit"] = dict(zip(x.title, x.pred))
x = pd.concat([pd.read_parquet(f"{R}/baseline_preds/counts_xfit{k}.parquet") for k in range(5)])
PRED["counts_xfit"] = dict(zip(x.title, x.pred))


def balacc(titles, ok):
    d = pd.DataFrame({"o": [cases[t]["outcome"] for t in titles], "a": [ok[t] for t in titles]})
    return d.groupby("o").a.mean().mean()


rng = np.random.default_rng(0)
STATS = {}
for a, pr in PRED.items():
    P = [(r, c) for r, c in pairs if r in pr and c in pr]
    ok = {t: pr[t] == cases[t]["outcome"] for p in P for t in p}
    idx = {p: i for i, p in enumerate(pairs)}
    B = [rng.integers(0, len(P), len(P)) for _ in range(2000)]

    def st(sel):
        pp = [P[i] for i in sel]
        rv, ct = balacc([r for r, _ in pp], ok), balacc([c for _, c in pp], ok)
        return rv, ct, ct - rv
    full = st(range(len(P)))
    bs = np.array([st(b) for b in B])
    STATS[a] = (P, ok)
    line = f"{NAME.get(a, a):18s} n_pairs {len(P):3d}"
    for j, k in enumerate(("rev", "ctl", "gap")):
        line += f"  {k} {full[j]:+.3f} [{np.percentile(bs[:, j], 2.5):+.3f}, {np.percentile(bs[:, j], 97.5):+.3f}]"
    if a in LADDER:
        tt = [t for p in P for t in p]
        oo = pd.DataFrame({"y": [int(pr[t] == "keep") for t in tt], "ck": [int(cases[t]["outcome"] == "keep") for t in tt],
                           "mk": [int(MAJ[t] == "keep") for t in tt]})
        try:
            f = smf.logit("y ~ ck + mk", oo).fit(disp=0)
            line += f"  majority coef {f.params.mk:.2f} (se {f.bse.mk:.2f})"
        except Exception as e:  # noqa: BLE001
            line += f"  logit failed: {str(e)[:40]}"
    print(line)

print("\npaired differences on pairs complete for both (2,000 pair resamples)")
COMP = [("qwen3.8-27b", "qwen3.5-4b", "gap"), ("qwen3.8-27b", "qwen3.5-2b", "gap"), ("local", "qwen3.5-4b", "gap"),
        ("qwen3.8-27b", "local", "gap"), ("qwen3.5-9b-think", "local", "gap"),
        ("ettin_xfit", "local", "rev"), ("ettin_xfit", "gemma-4-12b", "rev"), ("ettin_xfit", "gpt-4o-mini", "rev"),
        ("ettin_xfit", "gpt-6-luna", "rev"), ("ettin_xfit", "qwen3.8-27b", "rev"),
        ("ettin_xfit", "local", "ctl"), ("ettin_xfit", "gpt-6-luna", "ctl")]
for a, b, k in COMP:
    Pa, oka = STATS[a]; Pb, okb = STATS[b]
    P = [p for p in Pa if p in set(Pb)]
    j = ("rev", "ctl", "gap").index(k)

    def st(sel, ok):
        pp = [P[i] for i in sel]
        rv, ct = balacc([r for r, _ in pp], ok), balacc([c for _, c in pp], ok)
        return (rv, ct, ct - rv)[j]
    full = st(range(len(P)), oka) - st(range(len(P)), okb)
    d = []
    for _ in range(2000):
        s = rng.integers(0, len(P), len(P))
        d.append(st(s, oka) - st(s, okb))
    print(f"{NAME.get(a, a):18s} minus {NAME.get(b, b):18s} {k}: {full:+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]  n_pairs {len(P)}")
