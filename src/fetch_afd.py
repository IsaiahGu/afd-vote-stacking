"""Fetch English Wikipedia AfD discussions (raw wikitext) via the MediaWiki API.

Periods: 2018 and 2023 (before model training cut-offs) and 2026-01..2026-10 (after).
Output: data/afd_raw.jsonl  {title, log_date, period, wikitext}
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import datetime as dt
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

ROOT = WORKSPACE_ROOT
OUT = f"{ROOT}/data/afd_raw.jsonl" if len(sys.argv) < 2 else f"{ROOT}/data/afd_raw_{sys.argv[1]}.jsonl"
API = "https://en.wikipedia.org/w/api.php"
S = requests.Session()
S.headers["User-Agent"] = "afd-deliberation-research/0.1 (academic research; python-requests)"

PERIODS_EXTRA = {
    "2019": (dt.date(2019, 1, 1), dt.date(2019, 12, 31)),
    "2021": (dt.date(2021, 1, 1), dt.date(2021, 12, 31)),
    "2024": (dt.date(2024, 1, 1), dt.date(2024, 12, 31)),
    "2025": (dt.date(2025, 1, 1), dt.date(2025, 12, 31)),
}
PERIODS = {
    "2018": (dt.date(2018, 1, 1), dt.date(2018, 12, 31)),
    "2023": (dt.date(2023, 1, 1), dt.date(2023, 12, 31)),
    "2026": (dt.date(2026, 1, 1), dt.date(2026, 10, 31)),
}


if len(sys.argv) > 1:
    PERIODS = PERIODS_EXTRA


def fetch_titles(titles):
    """Return {title: wikitext} for up to 50 titles (follows normalisation/redirects)."""
    for attempt in range(8):
        try:
            r = S.post(API, data={"action": "query", "prop": "revisions", "rvprop": "content",
                                  "rvslots": "main", "titles": "|".join(titles), "format": "json",
                                  "formatversion": "2", "maxlag": "5"}, timeout=60)
            if r.status_code != 200:
                time.sleep(5 * (attempt + 1)); continue
            d = r.json()
            if "error" in d:
                time.sleep(5 * (attempt + 1)); continue
            out = {}
            norm = {n["to"]: n["from"] for n in d["query"].get("normalized", [])}
            for p in d["query"]["pages"]:
                if p.get("missing") or "revisions" not in p:
                    continue
                t = norm.get(p["title"], p["title"])
                out[t] = p["revisions"][0]["slots"]["main"]["content"]
            return out
        except Exception as e:  # noqa: BLE001
            print("retry", attempt, str(e)[:100], flush=True)
            time.sleep(5 * (attempt + 1))
    return {}


def month_name(d):
    return d.strftime("%B")


def main():
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            done.add(json.loads(line)["title"])
    # 1) log pages
    log_titles = []
    for period, (a, b) in PERIODS.items():
        d = a
        while d <= b:
            log_titles.append((period, d, f"Wikipedia:Articles for deletion/Log/{d.year} {month_name(d)} {d.day}"))
            d += dt.timedelta(days=1)
    print("log pages:", len(log_titles), flush=True)
    disc = {}  # title -> (period, date)
    batches = [log_titles[i:i + 50] for i in range(0, len(log_titles), 50)]
    with ThreadPoolExecutor(6) as ex:
        for batch, res in zip(batches, ex.map(lambda b: fetch_titles([t for _, _, t in b]), batches)):
            for period, d, t in batch:
                txt = res.get(t, "")
                for m in re.finditer(r"\{\{\s*(Wikipedia:Articles for deletion/[^}|]+?)\s*\}\}", txt):
                    name = m.group(1).strip()
                    if "/Log/" in name:
                        continue
                    disc.setdefault(name, (period, d.isoformat()))
    print("discussions found:", len(disc), flush=True)
    todo = [t for t in disc if t not in done]
    todo.sort(key=lambda t: disc[t][0], reverse=True)  # newest period first (post-cutoff data matters most)
    batches = [todo[i:i + 50] for i in range(0, len(todo), 50)]
    n = 0
    with open(OUT, "a") as f, ThreadPoolExecutor(6) as ex:
        for batch, res in zip(batches, ex.map(fetch_titles, batches)):
            for t in batch:
                if t in res:
                    period, d = disc[t]
                    f.write(json.dumps({"title": t, "log_date": d, "period": period, "wikitext": res[t]}) + "\n")
                    n += 1
            f.flush()
            if n % 2000 < 50:
                print("fetched", n, "/", len(todo), flush=True)
    print("DONE", n, flush=True)


if __name__ == "__main__":
    main()
