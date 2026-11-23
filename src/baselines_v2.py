"""P06 S5: supervised baselines redone, with every prediction saved.

Train on the eligible discussions that are not candidates (data/afd_parsed.jsonl.gz minus
data/afd_cases*.jsonl), 10% held out for validation; predict every candidate.
  counts      logistic regression on vote features only (keep / delete votes incl. nominator, keep share,
              margin, bare votes per side, policy-citing votes per side): the pure head-count reference
  ettin       Ettin-encoder-400M on the full anonymised discussion, max length = 95th percentile of the
              training lengths (head + tail beyond it), 3 epochs; LR from {1e-5, 2e-5, 3e-5} chosen on the
              validation split. One seed (0) for every run (2026-11-22); ettin_seed stays for the folds.
  ettin_mask  same encoder and LR, labels masked in training and test text (S2 L1), seed 0
  roberta512  RoBERTa-base, 255 + 255 head/tail tokens, 1 epoch, lr 2e-5 (the draft's baseline)
python baselines_v2.py counts|roberta512|ettin_lr <lr>|ettin_seed <lr> <seed>|ettin_mask <lr> <seed>
Output: results/baseline_preds/<tag>.parquet (title, p_keep, pred), results/baseline_runs.jsonl
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import gzip
import json
import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, f"{WORKSPACE_ROOT}/src")
from run_exp import anon_render  # noqa: E402

ROOT = WORKSPACE_ROOT
OUT = f"{ROOT}/results/baseline_preds"
os.makedirs(OUT, exist_ok=True)


FOLD = os.environ.get("FOLD")  # cross-fitting: "f/5" -> train also on the candidate pairs of the other folds
REV_W = int(os.environ.get("REV_W", 10))  # copies of each training reverse case under cross-fitting


def load():
    """Train / validation / evaluation discussions.

    Default (as in the draft): train on eligible discussions that are not candidates. Every loose-rule
    reverse-majority case is a candidate, so this training set has no reverse case at all and a supervised
    model can only learn that the majority wins. FOLD=f/K cross-fits instead: candidate pairs are split into
    K folds by pair; fold f is evaluated, the other folds' candidates join the training set (reverse cases
    repeated REV_W times)."""
    ev = [json.loads(l) for l in open(f"{ROOT}/data/afd_cases.jsonl")]
    more = f"{ROOT}/data/afd_cases_more.jsonl"
    if os.path.exists(more):
        ev += [json.loads(l) for l in open(more)]
    evt = {r["title"] for r in ev}
    rows = [json.loads(l) for l in gzip.open(f"{ROOT}/data/afd_parsed.jsonl.gz", "rt")]
    el = [r for r in rows if r["outcome"] in ("keep", "delete") and r["n_keep"] + r["n_delete"] >= 5
          and r["n_keep"] != r["n_delete"] and r["title"] not in evt]
    random.Random(0).shuffle(el)
    nval = len(el) // 10
    tr, va = el[nval:], el[:nval]
    if FOLD:
        f, K = (int(x) for x in FOLD.split("/"))
        rev = [r for r in ev if r["group"] == "reverse"]; ctl = [r for r in ev if r["group"] == "control"]
        pr = list(zip(rev, ctl))
        fold_of = [i % K for i in range(len(pr))]
        random.Random(1).shuffle(fold_of)
        test = [x for (a, b), k in zip(pr, fold_of) if k == f for x in (a, b)]
        extra = [x for (a, b), k in zip(pr, fold_of) if k != f for x in ([a] * REV_W + [b])]
        tr = tr + extra
        random.Random(2).shuffle(tr)
        ev = test
    return tr, va, ev


def feats(r):
    E = r["entries"]
    k, d = r["n_keep"], r["n_delete"]
    pk = sum(e["policy"] for e in E if e["label"] == "keep"); pd_ = sum(e["policy"] for e in E if e["label"] == "delete")
    return [k, d, k / (k + d), k - d, r["n_bare_keep"], r["n_bare_delete"], pk, pd_]


def log_run(rec):
    rec["fold"] = FOLD
    with open(f"{ROOT}/results/baseline_runs.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def save_preds(tag, ev, p):
    if FOLD:
        tag = f"{tag}_xfit{FOLD.split('/')[0]}"
    pd.DataFrame({"title": [r["title"] for r in ev], "p_keep": p,
                  "pred": np.where(np.asarray(p) >= 0.5, "keep", "delete")}).to_parquet(f"{OUT}/{tag}.parquet")


def run_counts():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    tr, va, ev = load()
    X = np.array([feats(r) for r in tr]); y = np.array([r["outcome"] == "keep" for r in tr])
    sc = StandardScaler().fit(X)
    m = LogisticRegression(max_iter=2000).fit(sc.transform(X), y)
    pv = m.predict_proba(sc.transform(np.array([feats(r) for r in va])))[:, 1]
    acc = float(((pv >= 0.5) == np.array([r["outcome"] == "keep" for r in va])).mean())
    p = m.predict_proba(sc.transform(np.array([feats(r) for r in ev])))[:, 1]
    save_preds("counts", ev, p)
    log_run({"tag": "counts", "val_acc": acc, "coef": dict(zip(["k", "d", "keep_share", "margin", "bare_k", "bare_d", "pol_k", "pol_d"],
                                                                    m.coef_[0].round(3).tolist()))})
    print("counts val acc", acc)


def text_of(r, mask=False):
    E = r["entries"]
    if mask:
        import supp_variants as sv
        E = sv.mask_labels(E)
    return anon_render(r, E)[0]


def run_encoder(model_dir, tag, lr, seed, epochs, max_len=None, mask=False, head_tail=None):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    tr, va, ev = load()
    tok = AutoTokenizer.from_pretrained(model_dir)
    t0 = time.time()
    enc_tr = [tok(text_of(r, mask), add_special_tokens=False)["input_ids"] for r in tr]
    lens = sorted(len(x) for x in enc_tr)
    if max_len is None:
        max_len = int(min(8000, math.ceil(lens[int(0.95 * len(lens))] / 256) * 256))
    print(f"[{tag}] tokenised {len(tr)} in {time.time() - t0:.0f}s; p50 {lens[len(lens) // 2]} p95 {lens[int(.95 * len(lens))]}"
          f" max_len {max_len}", flush=True)
    budget = max_len - 2

    def clip(x):
        if len(x) <= budget:
            return x
        h = head_tail or budget // 2
        return x[:h] + x[-(budget - h):]

    def batches(ids, ys, tok_budget, shuffle):
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        bs, cur, mx = [], [], 0
        for i in order:
            L = len(ids[i]) + 2
            if cur and max(mx, L) * (len(cur) + 1) > tok_budget:
                bs.append(cur); cur, mx = [], 0
            cur.append(i); mx = max(mx, L)
        if cur:
            bs.append(cur)
        if shuffle:
            random.shuffle(bs)
        for b in bs:
            L = max(len(ids[i]) + 2 for i in b)
            inp = torch.full((len(b), L), tok.pad_token_id); att = torch.zeros((len(b), L), dtype=torch.long)
            for k, i in enumerate(b):
                x = [tok.cls_token_id] + ids[i] + [tok.sep_token_id]
                inp[k, :len(x)] = torch.tensor(x); att[k, :len(x)] = 1
            yield b, inp.cuda(), att.cuda(), (torch.tensor([ys[i] for i in b]).cuda() if ys is not None else None)

    ids_tr = [clip(x) for x in enc_tr]
    y_tr = [1 if r["outcome"] == "keep" else 0 for r in tr]
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2, torch_dtype=torch.float32).cuda()
    tok_budget = int(os.environ.get("TOK_BUDGET", 32768))
    nb = sum(1 for _ in batches(ids_tr, y_tr, tok_budget, False))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * nb * epochs), nb * epochs)
    model.train()
    step = 0
    for ep in range(epochs):
        for b, inp, att, y in batches(ids_tr, y_tr, tok_budget, True):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.cross_entropy(model(input_ids=inp, attention_mask=att).logits.float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(); step += 1
            if step % 200 == 0:
                print(f"[{tag}] ep {ep} step {step}/{nb * epochs} loss {float(loss):.3f} {time.time() - t0:.0f}s", flush=True)
    model.eval()

    def predict(rs):
        ids = [clip(tok(text_of(r, mask), add_special_tokens=False)["input_ids"]) for r in rs]
        p = np.zeros(len(rs))
        with torch.inference_mode():
            for b, inp, att, _ in batches(ids, None, tok_budget, False):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pr = torch.softmax(model(input_ids=inp, attention_mask=att).logits.float(), -1)[:, 1].cpu().numpy()
                p[b] = pr
        return p
    pv = predict(va)
    yv = np.array([r["outcome"] == "keep" for r in va])
    val_acc = float(((pv >= 0.5) == yv).mean())
    val_nll = float(-np.mean(np.log(np.clip(np.where(yv, pv, 1 - pv), 1e-6, 1))))
    p = predict(ev)
    save_preds(tag, ev, p)
    log_run({"tag": tag, "model": os.path.basename(model_dir), "lr": lr, "seed": seed, "epochs": epochs, "max_len": max_len,
             "mask": mask, "val_acc": val_acc, "val_nll": val_nll, "n_train": len(tr), "n_val": len(va),
             "minutes": (time.time() - t0) / 60})
    print(f"[{tag}] val acc {val_acc:.4f} nll {val_nll:.4f}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    E = f"{WORKSPACE_ROOT}/models/ettin-encoder-400m"
    if cmd == "counts":
        run_counts()
    elif cmd == "roberta512":
        run_encoder(f"{WORKSPACE_ROOT}/models/roberta-base", "roberta512", 2e-5, 0, 1, max_len=512, head_tail=255)
    elif cmd == "ettin_lr":
        lr = float(sys.argv[2]); run_encoder(E, f"ettin_lr{lr:g}_s0", lr, 0, 3)
    elif cmd == "ettin_seed":
        lr, s = float(sys.argv[2]), int(sys.argv[3]); run_encoder(E, f"ettin_lr{lr:g}_s{s}", lr, s, 3)
    elif cmd == "ettin_mask":
        lr, s = float(sys.argv[2]), int(sys.argv[3]); run_encoder(E, f"ettinmask_lr{lr:g}_s{s}", lr, s, 3, mask=True)
