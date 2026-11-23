"""P06 supplementary analysis: S2, S3, S4, S6, S7, S9, S11 (and S10 / S8 sets when present).

Case sets: PAIRS=verified (default; results/verified_pairs.json), recount (results/recount_pairs.json, the
draft's 192) or all (every candidate pair). Readouts, per model:
- natural conditions: balanced accuracy (mean of agreement within closer-kept and closer-deleted cases)
  on reverse and control cases, the control-minus-reverse gap (paired bootstrap over pairs), and the
  share of predictions that follow the vote majority;
- injections: net shift toward the injected side, mean over discussions of
  p(injected side | attacked) - p(injected side | reference), bootstrap CI; dose slope on log2(k) with
  SEs clustered by discussion; the same against the injected share k / (k + original votes);
- defences (D1-D4) x attacks (orig, 8 bare, 8 boilerplate, 8 policy-shortcut votes);
- S11 length-matched policy ablation and shortcut stripping; S7 margin split, policy ablation with the
  removed-character difference as covariate.
Holm correction across models for each family of tests.
python supp_analyze.py [model aliases...]   Output: results/supp_tables_<PAIRS>.json
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = WORKSPACE_ROOT
PAIRS = os.environ.get("PAIRS", "verified")
B = 2000
rng = np.random.default_rng(0)
NAMES = {"local": "Qwen3.5-9B", "gemma-4-12b": "Gemma-4-12B", "gpt-4o-mini": "gpt-4o-mini", "gpt-6-luna": "gpt-6-luna"}


def pairs_of():
    if PAIRS == "verified":
        return [tuple(p) for p in json.load(open(f"{ROOT}/results/verified_pairs.json"))["pairs"]]
    if PAIRS == "verified_B":
        return [tuple(p) for p in json.load(open(f"{ROOT}/results/verified_pairs.json"))["pairs_B"]]
    if PAIRS == "recount":
        return [tuple(p) for p in json.load(open(f"{ROOT}/results/recount_pairs.json"))["pairs"]]
    if PAIRS == "combined":  # S8: the 150 verified pairs plus the verified pairs from the strict screen and new years
        return ([tuple(p) for p in json.load(open(f"{ROOT}/results/verified_pairs.json"))["pairs"]] +
                [tuple(p) for p in json.load(open(f"{ROOT}/results/verified_pairs_more.json"))["pairs"]])
    if PAIRS == "more":
        return [tuple(p) for p in json.load(open(f"{ROOT}/results/verified_pairs_more.json"))["pairs"]]
    cases = [json.loads(l) for l in open(f"{ROOT}/data/afd_cases.jsonl")]
    rev = [c["title"] for c in cases if c["group"] == "reverse"]; ctl = [c["title"] for c in cases if c["group"] == "control"]
    return list(zip(rev, ctl))


def load(alias):
    fs = [f for f in glob.glob(f"{ROOT}/results/supp_*_{alias}.parquet")
          if "argmap1" not in f and "_drv_" not in f]  # S12 cases share titles with some pair cases
    if not fs:
        return None
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df = df[~df.error.astype(bool)].drop_duplicates(["title", "variant"], keep="last")
    df["p_out"] = np.where(df.outcome == "keep", df.p_keep, 1 - df.p_keep)
    df["agree"] = df.pred == df.outcome
    df["follow"] = df.pred == df.majority
    return df


def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if not len(x):
        return [np.nan] * 3
    bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def balacc(agree, outcome):
    a = pd.Series(np.asarray(agree, float)); o = pd.Series(np.asarray(outcome))
    return float(a.groupby(o.values).mean().mean())


def natural(df, variant, pairs):
    d = df[df.variant == variant].set_index("title")
    ok = [(r, c) for r, c in pairs if r in d.index and c in d.index]
    if len(ok) < 10:
        return None
    R = d.loc[[r for r, _ in ok]]; C = d.loc[[c for _, c in ok]]
    out = {"n_pairs": len(ok), "balacc_reverse": balacc(R.agree, R.outcome), "balacc_control": balacc(C.agree, C.outcome),
           "agree_reverse": float(R.agree.mean()), "agree_control": float(C.agree.mean()),
           "follow_majority_reverse": float(R.follow.mean()), "follow_majority_control": float(C.follow.mean()),
           "pred_keep": float((pd.concat([R, C]).pred == "keep").mean())}
    ra, ro, ca, co = R.agree.to_numpy(float), R.outcome.to_numpy(), C.agree.to_numpy(float), C.outcome.to_numpy()
    idx = rng.integers(0, len(ok), (B, len(ok)))
    gaps, revs = [], []
    for i in idx:
        g = balacc(ca[i], co[i]) - balacc(ra[i], ro[i]); gaps.append(g); revs.append(balacc(ra[i], ro[i]))
    out["gap"] = [out["balacc_control"] - out["balacc_reverse"], float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))]
    out["balacc_reverse_ci"] = [out["balacc_reverse"], float(np.percentile(revs, 2.5)), float(np.percentile(revs, 97.5))]
    out["p_gap_le0"] = float(np.mean(np.array(gaps) <= 0))
    return out


def shift(df, ref, att, titles, toward="loser"):
    """Mean change in p(injected side): toward='loser' -> p_out(ref) - p_out(att); 'winner' -> p_out(att) - p_out(ref)."""
    a = df[df.variant == ref].set_index("title").p_out; b = df[df.variant == att].set_index("title").p_out
    t = [x for x in titles if x in a.index and x in b.index]
    d = (a.loc[t] - b.loc[t]) if toward == "loser" else (b.loc[t] - a.loc[t])
    pa = df[df.variant == ref].set_index("title").pred; pb = df[df.variant == att].set_index("title").pred
    return {"n": len(t), "shift": ci(d.to_numpy()), "flip": float((pa.loc[t] != pb.loc[t]).mean()) if t else np.nan}


def dose_slope(df, kind, ks, titles, toward="loser"):
    rows = []
    o = df[df.variant == "orig"].set_index("title")
    for k in ks:
        v = df[df.variant == f"{kind}{k}"].set_index("title")
        for t in titles:
            if t in o.index and t in v.index:
                d = (o.p_out[t] - v.p_out[t]) if toward == "loser" else (v.p_out[t] - o.p_out[t])
                tot = o.n_keep[t] + o.n_delete[t]
                rows.append({"title": t, "k": k, "lk": np.log2(k), "ratio": k / (k + tot), "d": d})
    D = pd.DataFrame(rows)
    if D.empty or D.k.nunique() < 2:
        return None
    cl = {"groups": pd.factorize(D.title)[0]}
    f = smf.ols("d ~ lk", D).fit(cov_type="cluster", cov_kwds=cl)
    g = smf.ols("d ~ ratio", D).fit(cov_type="cluster", cov_kwds=cl)
    h = smf.ols("d ~ lk + ratio", D).fit(cov_type="cluster", cov_kwds=cl)
    return {"slope_log2k": [float(f.params.lk), float(f.bse.lk), float(f.pvalues.lk)],
            "slope_ratio": [float(g.params.ratio), float(g.bse.ratio), float(g.pvalues.ratio)],
            "joint": {k: [float(h.params[k]), float(h.pvalues[k])] for k in ("lk", "ratio")},
            "r2_log2k": float(f.rsquared), "r2_ratio": float(g.rsquared), "n": len(D)}


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps)
    adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (m - r) * ps[i])); adj[i] = run
    return adj.tolist()


def model_tables(alias, pairs):
    df = load(alias)
    if df is None:
        return None
    titles = [t for p in pairs for t in p]
    T = {"model": NAMES.get(alias, alias), "variants": sorted(df.variant.unique())}
    # S2 levels and every natural condition
    T["natural"] = {v: natural(df, v, pairs) for v in ["orig", "mask_labels", "votes_only", "D3_orig", "D2_orig", "D1_orig",
                                                           "D4_orig", "title_only", "strip_policy",
                                                           "P1_nopolicy_orig", "P2_nolastsentence_orig", "P3a_orig", "P3b_orig",
                                                           "P3c_orig", "P4_nopersona_orig", "P5_reason_orig"] if v in T["variants"]}
    # S3 injections relative to the original discussion
    inj = {}
    for v in [f"inject{k}" for k in (1, 2, 4, 8, 16, 32)] + ["inject8_first", "inject8_last", "inject8_spa", "inject8_notice"] + \
             [f"boiler{k}" for k in (2, 4, 8)] + [f"polshort{k}" for k in (2, 4, 8)] + [f"neutral{k}" for k in (2, 4, 8)]:
        if v in T["variants"]:
            inj[v] = {g: shift(df, "orig", v, [t for p in pairs for t in (p if g == "pooled" else [p[0] if g == "reverse" else p[1]])])
                      for g in ("pooled", "reverse", "control")}
    for k in (2, 4, 8):
        v = f"win{k}"
        if v in T["variants"]:
            inj[v] = {g: shift(df, "orig", v, [t for p in pairs for t in (p if g == "pooled" else [p[0] if g == "reverse" else p[1]])],
                               toward="winner") for g in ("pooled", "reverse", "control")}
    T["inject"] = inj
    # differences between injection kinds at the same size and positions (paired over discussions)
    comp = {}
    for a, b in [("boiler8", "inject8"), ("polshort8", "inject8"), ("inject8_spa", "inject8"), ("inject8_notice", "inject8"),
                 ("inject8_first", "inject8"), ("inject8_last", "inject8"), ("boiler4", "inject4"), ("polshort4", "inject4"),
                 ("boiler2", "inject2"), ("polshort2", "inject2")]:
        if a in T["variants"] and b in T["variants"]:
            pa = df[df.variant == a].set_index("title").p_out; pb = df[df.variant == b].set_index("title").p_out
            t = [x for x in titles if x in pa.index and x in pb.index]
            comp[f"{a}_minus_{b}"] = {"n": len(t), "extra_shift": ci((pb.loc[t] - pa.loc[t]).to_numpy())}
    T["inject_compare"] = comp
    T["dose"] = {"inject": dose_slope(df, "inject", [k for k in (1, 2, 4, 8, 16, 32) if f"inject{k}" in T["variants"]], titles),
                 "win": dose_slope(df, "win", [2, 4, 8], titles, toward="winner") if "win8" in T["variants"] else None,
                 "boiler": dose_slope(df, "boiler", [2, 4, 8], titles) if "boiler8" in T["variants"] else None,
                 "polshort": dose_slope(df, "polshort", [2, 4, 8], titles) if "polshort8" in T["variants"] else None}
    # S4 defence x attack matrix
    D = {}
    for d in ("D0", "D1", "D2", "D3", "D4"):
        ref = "orig" if d == "D0" else f"{d}_orig"
        if ref not in T["variants"]:
            continue
        row = {"natural": natural(df, ref, pairs)}
        for a in ("inject8", "boiler8", "polshort8"):
            att = a if d == "D0" else f"{d}_{a}"
            if att in T["variants"]:
                row[a] = shift(df, ref, att, titles)
                row[a + "_vs_undefended_orig"] = shift(df, "orig", att, titles)
        D[d] = row
    T["defence"] = D
    # S9 prompts: inject8 shift under each prompt
    T["s9_inject8"] = {p: shift(df, f"{p}_orig", f"{p}_inject8", titles) for p in
                       ["P1_nopolicy", "P2_nolastsentence", "P3a", "P3b", "P3c", "P4_nopersona", "P5_reason"]
                       if f"{p}_inject8" in T["variants"]}
    # S11 policy: length-matched ablation, shortcut stripping
    pol = {}
    for g in ("pooled", "reverse", "control"):
        tt = [t for p in pairs for t in (p if g == "pooled" else [p[0] if g == "reverse" else p[1]])]
        for a, b, name in [("drop_policy_eq", "drop_nonpolicy_eq", "eqlen"), ("drop_policy", "drop_nonpolicy", "orig_ablation")]:
            if a in T["variants"] and b in T["variants"]:
                pa = df[df.variant == a].set_index("title").p_out; pb = df[df.variant == b].set_index("title").p_out
                t = [x for x in tt if x in pa.index and x in pb.index]
                pol[f"{name}_{g}"] = {"n": len(t), "dp": ci((pa.loc[t] - pb.loc[t]).to_numpy())}
        if "strip_policy" in T["variants"]:
            pol[f"strip_{g}"] = shift(df, "orig", "strip_policy", tt)
    T["policy"] = pol
    # S7.2 margin split on reverse cases (recounted margin)
    import importlib.util
    spec = importlib.util.spec_from_file_location("rr", f"{ROOT}/src/recount_rules.py")
    rr = importlib.util.module_from_spec(spec); spec.loader.exec_module(rr)
    cases = {json.loads(l)["title"]: json.loads(l) for l in open(f"{ROOT}/data/afd_cases.jsonl")}
    o = df[df.variant == "orig"].set_index("title")
    rv = [r for r, _ in pairs if r in o.index and r in cases]
    mar = {t: abs(rr.recount(cases[t])[0] - rr.recount(cases[t])[1]) for t in rv}
    m1 = [t for t in rv if mar[t] == 1]; m2 = [t for t in rv if mar[t] >= 2]
    if m1 and m2:
        diffs = []
        for _ in range(B):
            a = rng.choice(m1, len(m1)); b = rng.choice(m2, len(m2))
            diffs.append(balacc(o.agree[a], o.outcome[a]) - balacc(o.agree[b], o.outcome[b]))
        T["margin_split"] = {"n_margin1": len(m1), "n_margin2plus": len(m2),
                             "balacc_margin1": balacc(o.agree[m1], o.outcome[m1]), "balacc_margin2plus": balacc(o.agree[m2], o.outcome[m2]),
                             "diff_m1_minus_m2": [float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
    # S7.5 policy ablation with the removed-character difference as covariate
    if "drop_policy" in T["variants"]:
        pa = df[df.variant == "drop_policy"].set_index("title"); pb = df[df.variant == "drop_nonpolicy"].set_index("title")
        t = [x for x in titles if x in pa.index and x in pb.index]
        X = pd.DataFrame({"title": t, "dp": (pa.p_out[t] - pb.p_out[t]).values,
                          "dchars": ((pb.chars[t] - pa.chars[t]) / 1000).values})  # chars removed by dropping policy minus non-policy
        f = smf.ols("dp ~ dchars", X).fit(cov_type="HC1")
        T["policy_char_covariate"] = {"n": len(X), "intercept_equal_chars": [float(f.params.Intercept), float(f.bse.Intercept), float(f.pvalues.Intercept)],
                                      "per_1000_chars": [float(f.params.dchars), float(f.pvalues.dchars)],
                                      "mean_extra_chars_removed": float(X.dchars.mean() * 1000)}
    return T


def main():
    aliases = sys.argv[1:] or ["local", "gemma-4-12b", "gpt-4o-mini", "gpt-6-luna"]
    pairs = pairs_of()
    out = {"pairs": PAIRS, "n_pairs": len(pairs), "models": {}}
    for a in aliases:
        t = model_tables(a, pairs)
        if t:
            out["models"][a] = t
    # Holm across models: inject8 pooled shift > 0 (one-sided via CI is reported; here two-sided bootstrap p)
    fam = {}
    for a, t in out["models"].items():
        g = t["natural"].get("orig")
        if g:
            fam.setdefault("gap_orig", []).append((a, g["p_gap_le0"]))
        d = t["dose"].get("inject")
        if d:
            fam.setdefault("dose_log2k", []).append((a, d["slope_log2k"][2]))
    out["holm"] = {k: dict(zip([a for a, _ in v], holm([p for _, p in v]))) for k, v in fam.items()}
    json.dump(out, open(f"{ROOT}/results/supp_tables_{PAIRS}.json", "w"), indent=1, default=float)
    for a, t in out["models"].items():
        print(f"\n===== {t['model']}")
        for v, r in t["natural"].items():
            if r:
                print(f"  {v:24s} n={r['n_pairs']:3d} rev {r['balacc_reverse']:.3f} ctl {r['balacc_control']:.3f} gap {r['gap'][0]:+.3f} "
                      f"[{r['gap'][1]:+.3f},{r['gap'][2]:+.3f}] follow_rev {r['follow_majority_reverse']:.3f} keep {r['pred_keep']:.2f}")
        for v, r in t["inject"].items():
            s = r["pooled"]["shift"]
            print(f"  {v:24s} shift {s[0]:+.3f} [{s[1]:+.3f},{s[2]:+.3f}] flip {r['pooled']['flip']:.3f}")
        for k, r in t["inject_compare"].items():
            s = r["extra_shift"]; print(f"  {k:30s} {s[0]:+.3f} [{s[1]:+.3f},{s[2]:+.3f}]")
        print("  dose", json.dumps(t["dose"], default=float)[:400])
        for d, r in t["defence"].items():
            nat = r["natural"]
            s = "  ".join(f"{a} {r[a]['shift'][0]:+.3f}" for a in ("inject8", "boiler8", "polshort8") if a in r)
            if nat:
                print(f"  {d} rev {nat['balacc_reverse']:.3f} ctl {nat['balacc_control']:.3f} gap {nat['gap'][0]:+.3f} | {s}")
        for k, r in t["s9_inject8"].items():
            print(f"  S9 {k:20s} inject8 shift {r['shift'][0]:+.3f}")
        for k, r in t["policy"].items():
            v = r.get("dp") or r.get("shift"); print(f"  policy {k:22s} n={r['n']} {v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}]")
        if "margin_split" in t:
            print("  margin", json.dumps(t["margin_split"], default=float))
        if "policy_char_covariate" in t:
            print("  policy chars", json.dumps(t["policy_char_covariate"], default=float))


if __name__ == "__main__":
    main()
