"""LCB v6 codegen on the fixed subset (resumable). usage: lcb_gen.py OUT_DIR [LIMIT]. Prompt = lcb_runner OpenAIChat format."""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__)); from oai_chat import chat, done_keys; from lcb_common import load_subset
from lcb_runner.prompts.code_generation import format_prompt_generation
from lcb_runner.lm_styles import LMStyle
out = sys.argv[1]; lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
probs = load_subset()
if lim:   # smoke: easiest first
    probs = sorted(probs, key=lambda p: ["easy", "medium", "hard"].index(p.difficulty.value))[:lim]
log = f"{out}/items.jsonl"; have = done_keys(log, "id"); fails = 0
for i, p in enumerate(probs):
    if p.question_id in have: continue
    try: r = chat(format_prompt_generation(p, LMStyle.OpenAIChat))
    except Exception as e:
        fails += 1; print(f"[{i}] {p.question_id} ERROR {e!r}", flush=True)
        if fails > 5: sys.exit(2)
        continue
    with open(log, "a") as f:
        f.write(json.dumps({"id": p.question_id, "difficulty": p.difficulty.value, **r}) + "\n")
    print(f"[{i+1}/{len(probs)}] {p.question_id} {p.difficulty.value} tok={r['completion_tokens']} wall={r['wall_s']}s finish={r['finish']}", flush=True)
