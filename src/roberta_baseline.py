"""P06 baseline: RoBERTa-base fine-tuned to predict keep/delete from the (anonymised) discussion.

Train on eligible discussions that are not in the evaluation set (results/exp_local.parquet),
1 epoch, head+tail truncation to 512 tokens; evaluate agreement on reverse and control cases.
Output: results/roberta_baseline.json
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import json
import random
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, f"{WORKSPACE_ROOT}/shared")
sys.path.insert(0, f"{WORKSPACE_ROOT}/src")
from embed import acquire_gpu  # noqa: E402
from run_exp import anon_render  # noqa: E402

ROOT = WORKSPACE_ROOT
acquire_gpu()
torch.manual_seed(0)
rows = [json.loads(l) for l in open(f"{ROOT}/data/afd_parsed.jsonl")]
el = [r for r in rows if r["outcome"] in ("keep", "delete") and r["n_keep"] + r["n_delete"] >= 5 and r["n_keep"] != r["n_delete"]]
ev = pd.read_json(f"{ROOT}/data/afd_cases.jsonl", lines=True)[["title", "group", "outcome", "majority"]]

evt = set(ev.title)
train = [r for r in el if r["title"] not in evt]
random.Random(0).shuffle(train)
MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else f"{WORKSPACE_ROOT}/models/roberta-base"
TAG = sys.argv[2] if len(sys.argv) > 2 else "roberta"
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=2).cuda()


def enc(texts):
    ids = []
    for t in texts:
        x = tok(t, add_special_tokens=False)["input_ids"]
        if len(x) > 510:
            x = x[:255] + x[-255:]
        ids.append([tok.cls_token_id] + x + [tok.sep_token_id])
    L = max(len(i) for i in ids)
    inp = torch.full((len(ids), L), tok.pad_token_id); att = torch.zeros((len(ids), L), dtype=torch.long)
    for k, i in enumerate(ids):
        inp[k, :len(i)] = torch.tensor(i); att[k, :len(i)] = 1
    return inp.cuda(), att.cuda()


texts = [anon_render(r, r["entries"])[0] for r in train]
y = torch.tensor([1 if r["outcome"] == "keep" else 0 for r in train])
opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
model.train()
bs = 16
for s in range(0, len(texts), bs):
    inp, att = enc(texts[s:s + bs])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = torch.nn.functional.cross_entropy(model(input_ids=inp, attention_mask=att).logits, y[s:s + bs].cuda())
    loss.backward(); opt.step(); opt.zero_grad()
    if (s // bs) % 200 == 0:
        print("step", s // bs, float(loss), flush=True)
model.eval()
byt = {r["title"]: r for r in el}
preds = []
for s in range(0, len(ev), 32):
    chunk = ev.iloc[s:s + 32]
    inp, att = enc([anon_render(byt[t], byt[t]["entries"])[0] for t in chunk.title])
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        p = torch.softmax(model(input_ids=inp, attention_mask=att).logits.float(), -1)[:, 1].cpu().numpy()
    preds += list(p)
ev = ev.assign(p_keep_rb=preds, pred_rb=np.where(np.array(preds) >= 0.5, "keep", "delete"))
out = {"n_train": len(train), "train_keep_share": float(y.float().mean())}
for g, s in ev.groupby("group"):
    out[g] = {"agree": float((s.pred_rb == s.outcome).mean()), "follow_majority": float((s.pred_rb == s.majority).mean()), "n": len(s)}
json.dump(out, open(f"{ROOT}/results/{TAG}_baseline.json", "w"), indent=1)
print(out)
