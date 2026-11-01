"""Generate IFBench responses (resumable). usage: ifbench_gen.py OUT_DIR [LIMIT]"""
import json, sys, time, os
sys.path.insert(0, os.path.dirname(__file__)); from oai_chat import chat, done_keys
out = sys.argv[1]; lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
data = [json.loads(l) for l in open(os.path.join(os.environ.get("IFBENCH_DIR", "IFBench"), "data/IFBench_test.jsonl"))][:lim]
resp, log = f"{out}/responses.jsonl", f"{out}/items.jsonl"
have = done_keys(log, "key"); fails = 0
for i, d in enumerate(data):
    if d["key"] in have: continue
    try: r = chat([{"role": "user", "content": d["prompt"]}])
    except Exception as e:
        fails += 1; print(f"[{i}] key={d['key']} ERROR {e!r}", flush=True)
        if fails > 5: sys.exit(2)
        continue
    with open(resp, "a") as f: f.write(json.dumps({"prompt": d["prompt"], "response": r["content"]}) + "\n")
    with open(log, "a") as f: f.write(json.dumps({"key": d["key"], **{k: v for k, v in r.items() if k != "content"}, "answer_chars": len(r["content"])}) + "\n")
    print(f"[{i+1}/{len(data)}] key={d['key']} tok={r['completion_tokens']} wall={r['wall_s']}s finish={r['finish']} ans={len(r['content'])}c", flush=True)
