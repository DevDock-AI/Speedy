"""Score IFBench with the official checker + write summary. usage: ifbench_score.py OUT_DIR [LIMIT]
Prompts with no response (generation error) count as failed (empty response)."""
import json, sys, os, subprocess, glob
sys.path.insert(0, os.path.dirname(__file__)); from summary import write_summary
out = sys.argv[1]; lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
IFB = os.environ.get("IFBENCH_DIR", "IFBench")
data = [json.loads(l) for l in open(f"{IFB}/data/IFBench_test.jsonl")][:lim]
resp = {json.loads(l)["prompt"]: json.loads(l)["response"] for l in open(f"{out}/responses.jsonl")} if os.path.exists(f"{out}/responses.jsonl") else {}
with open(f"{out}/input.jsonl", "w") as f:
    for d in data: f.write(json.dumps(d) + "\n")
with open(f"{out}/final-responses.jsonl", "w") as f:
    for d in data: f.write(json.dumps({"prompt": d["prompt"], "response": resp.get(d["prompt"], "")}) + "\n")
subprocess.run([f"{IFB}/.venv/bin/python", "-m", "run_eval", f"--input_data={out}/input.jsonl",
                f"--input_response_data={out}/final-responses.jsonl", f"--output_dir={out}/eval"], cwd=IFB, check=True,
               stdout=open(f"{out}/checker.log", "w"), stderr=subprocess.STDOUT)
sc = {}
for mode in ("loose", "strict"):
    rows = [json.loads(l) for l in open(glob.glob(f"{out}/eval/*eval_results_{mode}.jsonl")[0])]
    p = sum(r["follow_all_instructions"] for r in rows) / len(rows)
    ins = [x for r in rows for x in r["follow_instruction_list"]]
    sc[mode] = (100 * p, 100 * sum(ins) / len(ins))
score_lines = [f"prompt-level loose  : {sc['loose'][0]:.1f}  (IFBench paper headline metric)",
               f"prompt-level strict : {sc['strict'][0]:.1f}",
               f"instr-level loose/strict: {sc['loose'][1]:.1f} / {sc['strict'][1]:.1f}",
               f"missing responses (counted as fail): {sum(d['prompt'] not in resp for d in data)}"]
ref = [("Qwen3.8-Flash-Next (card)", 81.3), ("Qwen3.8-27B", 79.5), ("Qwen3.7-Plus", 79.1),
       ("DeepSeek-V4-Flash-0731", 79.2), ("Claude Opus 4.6 (Max)", 62.5)]
write_summary(out, "IFBench (OOD, single-turn)", len(data), score_lines, ("ours UD-Q4_K_XL (loose)", sc["loose"][0]), ref)
print(open(f"{out}/SUMMARY.md").read())
