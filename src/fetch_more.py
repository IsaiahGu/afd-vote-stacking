"""P06 S8 / S12: fetch more AfD discussions, and the deletion review (DRV) daily logs.

python fetch_more.py afd  -> data/afd_raw_more.jsonl  discussions listed on the AfD daily logs of 2019-2022
                             and 2024 that are not in data/afd_parsed.jsonl.gz yet
python fetch_more.py drv  -> data/drv_raw.jsonl       DRV daily log pages 2018-01-01 .. 2026-10-31
Batches of 50 titles per request, maxlag=5, retries with back-off on HTTP 429; resumable.
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import datetime as dt
import gzip
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, f"{WORKSPACE_ROOT}/src")
from fetch_afd import fetch_titles  # noqa: E402

ROOT = WORKSPACE_ROOT
THREADS = int(os.environ.get("THREADS", 3))


def days(a, b):
    d = a
    while d <= b:
        yield d
        d += dt.timedelta(days=1)


def log_name(prefix, d):
    return f"{prefix}/Log/{d.year} {d.strftime('%B')} {d.day}"


def fetch_afd():
    out = f"{ROOT}/data/afd_raw_more.jsonl"
    have = {json.loads(l)["title"] for l in gzip.open(f"{ROOT}/data/afd_parsed.jsonl.gz", "rt")}
    if os.path.exists(out):
        have |= {json.loads(l)["title"] for l in open(out)}
    logs = [(str(y), d, log_name("Wikipedia:Articles for deletion", d))
            for y in (2019, 2020, 2021, 2022, 2024) for d in days(dt.date(y, 1, 1), dt.date(y, 12, 31))]
    print("log pages", len(logs), flush=True)
    disc = {}
    batches = [logs[i:i + 50] for i in range(0, len(logs), 50)]
    with ThreadPoolExecutor(THREADS) as ex:
        for batch, res in zip(batches, ex.map(lambda b: fetch_titles([t for _, _, t in b]), batches)):
            for period, d, t in batch:
                for m in re.finditer(r"\{\{\s*(Wikipedia:Articles for deletion/[^}|]+?)\s*\}\}", res.get(t, "")):
                    name = m.group(1).strip()
                    if "/Log/" not in name:
                        disc.setdefault(name, (period, d.isoformat()))
    todo = [t for t in disc if t not in have]
    print("discussions listed", len(disc), "to fetch", len(todo), flush=True)
    batches = [todo[i:i + 50] for i in range(0, len(todo), 50)]
    n = 0
    with open(out, "a") as f, ThreadPoolExecutor(THREADS) as ex:
        for batch, res in zip(batches, ex.map(fetch_titles, batches)):
            for t in batch:
                if t in res:
                    period, d = disc[t]
                    f.write(json.dumps({"title": t, "log_date": d, "period": period, "wikitext": res[t]}) + "\n"); n += 1
            f.flush()
            if n % 2000 < 50:
                print("fetched", n, "/", len(todo), flush=True)
    print("AFD_DONE", n, flush=True)


def fetch_drv():
    out = f"{ROOT}/data/drv_raw.jsonl"
    have = {json.loads(l)["page"] for l in open(out)} if os.path.exists(out) else set()
    logs = [(d, log_name("Wikipedia:Deletion review", d)) for d in days(dt.date(2018, 1, 1), dt.date(2026, 10, 31))]
    logs = [x for x in logs if x[1] not in have]
    batches = [logs[i:i + 50] for i in range(0, len(logs), 50)]
    n = 0
    with open(out, "a") as f, ThreadPoolExecutor(THREADS) as ex:
        for batch, res in zip(batches, ex.map(lambda b: fetch_titles([t for _, t in b]), batches)):
            for d, t in batch:
                if t in res:
                    f.write(json.dumps({"page": t, "date": d.isoformat(), "wikitext": res[t]}) + "\n"); n += 1
            f.flush()
    print("DRV_DONE", n, flush=True)


if __name__ == "__main__":
    {"afd": fetch_afd, "drv": fetch_drv}[sys.argv[1]]()
