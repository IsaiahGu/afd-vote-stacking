"""P06 S9: three meaning-preserving rewrites of the prompt by a model outside the tested set.

The rewrites keep the persona, the policy explanation, the discussion slot and the one-word answer;
each is checked to contain {discussion} once and to ask for "Keep or Delete".
python gen_paraphrases.py [model]   Output: data/prompt_paraphrases.json
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import json
import sys

sys.path.insert(0, f"{WORKSPACE_ROOT}/shared")
sys.path.insert(0, f"{WORKSPACE_ROOT}/src")
from llm import LLM, parse_json  # noqa: E402
from run_exp import POLICY, SYS  # noqa: E402

ROOT = WORKSPACE_ROOT
ORIG = {"system": SYS,
        "user": f"{POLICY}\n\nDiscussion:\n\"\"\"\n{{discussion}}\n\"\"\"\n\nBased on the discussion and Wikipedia policy, what "
                f"should the closing result be? Answer with exactly one word: Keep or Delete."}
ASK = ("Rewrite the following two-part prompt (a system message and a user message) in your own words, three times, "
       "with three clearly different wordings. Each rewrite must keep exactly the same meaning and every instruction: "
       "the closing-administrator persona; that deletion discussions are not votes and are decided by the strength of "
       "arguments under the named policies; that arguments ignoring policy, resting on personal opinion, or bare "
       "recommendations without reasons carry little weight however many there are; and that the answer must be exactly "
       "one word, Keep or Delete. Keep the placeholder {discussion} exactly once, inside triple quotes, where the "
       "discussion goes. Do not add or remove any instruction.\n\nSystem message:\n" + ORIG["system"] +
       "\n\nUser message:\n" + ORIG["user"] +
       "\n\nReturn only JSON: [{\"system\": \"...\", \"user\": \"...\"}, {...}, {...}]")
model = sys.argv[1] if len(sys.argv) > 1 else "claude-haiku-5-5"
out = LLM(model, cache_name="p06_paraphrase").chat([{"role": "user", "content": ASK}], max_tokens=3000, temperature=0.7)
P = parse_json(out)
assert isinstance(P, list) and len(P) == 3, out
for p in P:
    assert p["user"].count("{discussion}") == 1 and "Keep" in p["user"] and "Delete" in p["user"], p
json.dump(P, open(f"{ROOT}/data/prompt_paraphrases.json", "w"), indent=1)
print(json.dumps(P, indent=1))
