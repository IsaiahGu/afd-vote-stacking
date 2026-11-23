"""Pair-bootstrap CIs for the natural gap (control minus reverse balanced accuracy) of gpt-6-luna at three
reasoning-effort levels, on the 150 verified pairs, plus paired differences between levels and the
majority coefficient. Same definitions as src/s5_s10_analysis.py; read-only, prints only."""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import json
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = WORKSPACE_ROOT
cases = {r["title"]: r for r in map(json.loads, open(f"{ROOT}/data/afd_cases.jsonl"))}
pairs = [tuple(p) for p in json.load(open(f"{ROOT}/results/verified_pairs.json"))["pairs"]]


def orig(alias):
    fs = [f"{ROOT}/results/supp_s10_{alias}.parquet", f"{ROOT}/results/supp_core_{alias}.parquet"]
    df = pd.concat([pd.read_parquet(f) for f in fs if os.path.exists(f)])
    df = df[~df.error.astype(bool)].drop_duplicates(["title", "variant"])
    return df[df.variant == "orig"].set_index("title")


def balacc(titles, ok):
    d = pd.DataFrame({"o": [cases[t]["outcome"] for t in titles], "a": [ok[t] for t in titles]})
    return d.groupby("o").a.mean().mean()


O = {a: orig(a) for a in ("gpt-6-luna@low", "gpt-6-luna", "gpt-6-luna@high")}
P = [(r, c) for r, c in pairs if all(r in o.index and c in o.index for o in O.values())]
print("pairs complete at all three levels:", len(P), "of", len(pairs))
OK = {a: {t: o.loc[t, "pred"] == cases[t]["outcome"] for t in o.index} for a, o in O.items()}


def gap(a, idx):
    pr = [P[i] for i in idx]
    return balacc([c for _, c in pr], OK[a]) - balacc([r for r, _ in pr], OK[a])


rng = np.random.default_rng(0)
B = [rng.integers(0, len(P), len(P)) for _ in range(2000)]
full = np.arange(len(P))
G = {a: (gap(a, full), np.array([gap(a, b) for b in B])) for a in O}
for a, (g, bs) in G.items():
    oo = O[a].loc[[t for p in P for t in p]].assign(y=lambda x: (x.pred == "keep").astype(int),
                                                     ck=lambda x: (x.outcome == "keep").astype(int),
                                                     mk=lambda x: (x.majority == "keep").astype(int))
    f = smf.logit("y ~ ck + mk", oo).fit(disp=0)
    print(f"{a:18s} gap {g:+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]  "
          f"majority coef {f.params.mk:.2f} (se {f.bse.mk:.2f}, p {f.pvalues.mk:.3f})")
for a, b in (("gpt-6-luna@high", "gpt-6-luna"), ("gpt-6-luna@high", "gpt-6-luna@low"), ("gpt-6-luna@low", "gpt-6-luna")):
    d = G[a][1] - G[b][1]
    print(f"{a} minus {b}: {G[a][0] - G[b][0]:+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]")
