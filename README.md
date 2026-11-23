# Counting Heads or Weighing Arguments?
**A Vote-Stacking Stress Test for LLM Discussion Closers**

Each pair contains a discussion whose closer went against the recounted majority and a matched discussion whose closer followed it. The stress test inserts votes into the discussion shown to the model and measures movement toward the injected side.

## Inspect the paper results

[Combined results](reference/supp_tables_combined.json) report balanced accuracy, paired effects and uncertainty by model and variant. The main cohort has 362 pairs. Each table entry reports its own sample size. The two title manifests in [reference](reference/) record the original 150 pairs and the additional 212 pairs, so readers can identify the evaluated discussions without downloading wikitext.

## Prepare the cases

Run from the repository root with Linux and Python 3.10 or newer. The collection scripts read public MediaWiki revisions. A new collection can differ from the paper snapshot if pages have moved or changed.

```bash
python -m pip install -r requirements.txt
mkdir -p data results logs models
python src/fetch_afd.py
python src/parse_afd.py
python src/run_exp.py select
gzip -c data/afd_parsed.jsonl > data/afd_parsed.jsonl.gz
python src/fetch_more.py afd
python src/s8_candidates.py
cp reference/verified_pairs*.json results/
```

`parse_afd.py` extracts the closer outcome and participants' recommendations. `run_exp.py select` matches discussions. `recount_rules.py` handles changed and withdrawn votes and procedural closes. `verify_votes.py` implements the independent model recounts used during cohort construction.

## Run a closer

Serve a local OpenAI-compatible model named `qwen3.5-9b` and set `LOCAL_LLM_URL` to its API base (default http://localhost:8000/v1). Configure remote runs with `OPENAI_BASE_URL` and `OPENAI_API_KEY`. The following example runs one closer and analyzes both cohorts:

```bash
CASES=pairs python src/supp_run.py local core strict
CASES_FILE=data/afd_cases_more.jsonl PAIRS_FILE=results/verified_pairs_more.json OUT_TAG=_more python src/supp_run.py local s10
PAIRS=combined python src/supp_analyze.py local
```

Per-discussion predictions are saved to `results/supp_*.parquet`. The final command writes `results/supp_tables_combined.json`. Repeat inference for each closer and pass the aliases together to the analysis. `supp_variants.py` defines bare votes, shortcut citations, neutral comments and the masking and filtering defenses. `baselines_v2.py` trains count-based and encoder baselines. Its encoder modes require local checkpoints in `models/roberta-base` or `models/ettin-encoder-400m` and a CUDA GPU.

