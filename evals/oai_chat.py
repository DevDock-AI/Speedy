"""Minimal chat client for the local Strata/Speedy server (stdlib only). The key is read from the file, never printed."""
import json, os, time, urllib.request
BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8080/v1")
KEYF = os.environ.get("KEY_FILE", os.path.expanduser("~/.config/speedy/api_key"))
MODEL = os.environ.get("MODEL", "qwen3.8-flash-next-ud-q4_k_xl")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "32768"))
SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0, presence_penalty=0,
                reasoning_effort="high", reasoning_budget_tokens=0)   # 0 = no thinking budget (config default is 16384)

def chat(messages, timeout=7200):
    key = open(KEYF).read().strip() if os.path.exists(KEYF) else "none"
    body = dict(model=MODEL, messages=messages, max_tokens=MAX_TOKENS, stream=False, **SAMPLING)
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    wall = time.time() - t
    c = d["choices"][0]; m = c.get("message") or {}; u = d.get("usage") or {}
    content = m.get("content") or ""
    reasoning = m.get("reasoning_content") or ""
    if not reasoning and "</think>" in content:          # safety: strip inline thinking if the server ever inlines it
        reasoning, content = content.split("</think>", 1)
    return dict(content=content.strip(), reasoning_chars=len(reasoning), finish=c.get("finish_reason"),
                prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"), wall_s=round(wall, 2))

def done_keys(path, field):
    if not os.path.exists(path): return set()
    return {json.loads(l)[field] for l in open(path) if l.strip()}
