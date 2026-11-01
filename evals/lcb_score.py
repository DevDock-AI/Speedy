"""Score with lcb_runner's sandboxed executor (codegen_metrics, pass@1) + summary. usage: lcb_score.py OUT_DIR [LIMIT]"""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__)); from lcb_common import load_subset, START, END, RELEASE; from summary import write_summary
from lcb_runner.utils.extraction_utils import extract_code
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
from lcb_runner.lm_styles import LMStyle
out = sys.argv[1]; lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
probs = load_subset()
if lim: probs = sorted(probs, key=lambda p: ["easy", "medium", "hard"].index(p.difficulty.value))[:lim]
items = {json.loads(l)["id"]: json.loads(l) for l in open(f"{out}/items.jsonl")}
codes = [extract_code(items[p.question_id]["content"], LMStyle.OpenAIChat) if p.question_id in items else "" for p in probs]
m = codegen_metrics([p.get_evaluation_sample() for p in probs], [[c] for c in codes], k_list=[1], num_process_evaluate=12, timeout=6)
res = m[1]   # {idx: [[test results]]}
per = []
for i, p in enumerate(probs):
    ok = all(x is True or x == 1 for x in res[i][0]) if i in res else False
    per.append({"id": p.question_id, "difficulty": p.difficulty.value, "pass": bool(ok), "has_code": bool(codes[i]),
                "finish": items.get(p.question_id, {}).get("finish")})
json.dump(per, open(f"{out}/per_problem.json", "w"), indent=1)
json.dump({"pass@1": m[0]["pass@1"], "detail": m[0].get("detail")}, open(f"{out}/metrics.json", "w"), indent=1, default=str)
pas = 100 * sum(x["pass"] for x in per) / len(per)
byd = {d: [x["pass"] for x in per if x["difficulty"] == d] for d in ("easy", "medium", "hard")}
lines = [f"pass@1: {pas:.1f}  (lcb_runner pass@1 {100*m[0]['pass@1']:.1f})",
         "by difficulty: " + ", ".join(f"{d} {sum(v)}/{len(v)}" for d, v in byd.items() if v),
         f"no code block extracted: {sum(not x['has_code'] for x in per)}",
         f"set: {RELEASE} {START}..{END}, fixed stratified subset (lcb_subset_v6_70.json)"]
ref = [("Qwen3.8-Flash-Next (card)", 91.9), ("Qwen3.8-27B", 90.3), ("Qwen3.7-Plus", 89.6),
       ("DeepSeek-V4-Flash-0731", 90.6), ("Claude Opus 4.6 (Max)", 88.8)]
write_summary(out, "LiveCodeBench v6 codegen", len(probs), lines, ("ours UD-Q4_K_XL", pas), ref,
              extra=["subset is 33/70 hard vs card's (unknown) full set: compare only roughly"])
print(open(f"{out}/SUMMARY.md").read())
