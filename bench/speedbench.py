#!/usr/bin/env python3
"""Speedy speedbench: prefill / decode speed from the engine's own timing lines, with exact token counts and a
provenance record (schema speedy-bench/1).

  python3 bench/speedbench.py <label> <engine_log> [--config cfg.json] [--k 16] [--reps 5] [--out 1024]
         [--tokenizer <pack>/tokenizer] [--warm] [--run-id ID] [--url http://127.0.0.1:8080]

Protocol (compatible with the maintainer's original speedbench.py, which it extends): <reps> unique prompts of about
k x 1000 tokens (no cache reuse) asking for a long answer; each request's line
  "prompt N tokens = R reused + X read in Y ms (P tok/s), G generated in T ms (D tok/s)"
is read from the engine log.  Additions:
  --tokenizer  size the filler to exactly k x 1000 tokens with tools/strata_tokenizer.py (the old chars/29 guess
               is the fallback) and record the exact count;
  --warm       suffix mode: each rep sends the SAME long prefix twice with a different short question; the second
               request's prefill is the warm-turn (suffix) number, reported as "suffix_ms";
  --config     with --run-id: write bench/results/runs/<run-id>/<label>.json with engine commit, binary sha256,
               config sha256, GPU unlock state, the engine's mechanism / fallback / chunk / pinning lines.
Writes results/speed_<label>.json (+ speed.log line) in --results (default $SPEEDY_RESULTS or bench/results-local).
"""
import argparse, datetime, hashlib, json, os, pathlib, re, statistics, subprocess, sys, time, urllib.request, uuid

LINE = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), "
                  r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\)")
# engine lines worth keeping in the provenance record (mechanism selected, fallbacks, chunking, pinning)
MECH = re.compile(r"speedy:|prompt chunk|K-quant prompt|fused|FP16 path|fallback|falls back|pinned|pin by need|"
                  r"expert cache|lendable|STRATA_[A-Z_]+|only \d+ MiB of VRAM free", re.I)
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_tokenizer(tdir):
    sys.path.insert(0, str(ROOT / "tools"))
    import strata_tokenizer as ST
    t = pathlib.Path(tdir)
    vocab = json.loads((t / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for s, i in vocab.items():
        toks[i] = s
    return ST.Tokenizer(toks, (t / "merges.txt").read_text(encoding="utf-8").split("\n"),
                        json.loads((t / "token_type.json").read_text()))


def make_filler(k, tok):
    lines = [f"def helper_{i}(x):\n    return x * {i % 97 + 2} + {i}  # step {i}" for i in range(int(k * 1000 / 29) + 200)]
    if tok is None:
        return "\n".join(lines[:int(k * 1000 / 29)]), None
    target = k * 1000
    per = [len(tok.encode(l + "\n")) for l in lines]   # lines tokenise independently (each starts after "\n")
    n, total = 0, 0
    while n < len(lines) and total + per[n] <= target:
        total += per[n]
        n += 1
    text = "\n".join(lines[:n])
    return text, len(tok.encode(text))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("label"); ap.add_argument("log")
    ap.add_argument("--reps", type=int, default=5); ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--out", type=int, default=1024)
    ap.add_argument("--tokenizer"); ap.add_argument("--warm", action="store_true")
    ap.add_argument("--config"); ap.add_argument("--run-id")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--results", default=os.environ.get("SPEEDY_RESULTS", str(pathlib.Path(__file__).resolve().parent / "results-local")))
    a = ap.parse_args()

    def lines():
        try:
            return open(a.log, encoding="utf-8", errors="replace").read().splitlines()
        except OSError:
            return []

    tok = load_tokenizer(a.tokenizer) if a.tokenizer else None
    filler, filler_tokens = make_filler(a.k, tok)
    keyf = os.path.expanduser("~/.config/strata-llm/api_key")
    hdr = {"Content-Type": "application/json"}
    if os.path.exists(keyf):
        hdr["Authorization"] = "Bearer " + open(keyf).read().strip()

    def ask(prompt, cache):
        n0 = len(lines())
        body = json.dumps({"model": "worker", "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": a.out, "cache_prompt": cache}).encode()
        t0 = time.time()
        urllib.request.urlopen(urllib.request.Request(a.url + "/v1/chat/completions", body, hdr), timeout=1800).read()
        wall = time.time() - t0
        time.sleep(1)
        new = [m for m in (LINE.search(l) for l in lines()[n0:]) if m]
        if not new:
            return None
        m = max(new, key=lambda x: int(x[1]))   # the request that carried the prompt (reasoning adds a short follow-up)
        return {"prompt": int(m[1]), "reused": int(m[2]), "read": int(m[3]), "read_ms": int(m[4]),
                "prefill": float(m[5]), "gen": int(m[6]), "decode": float(m[8]), "wall_s": round(wall, 3)}

    task = ("Refactor this module: write the complete new Python file that replaces all helper_N functions with one "
            "parametrised function plus a lookup table of the multipliers, keep backwards-compatible wrappers for "
            "helper_0..helper_60, and add a unittest class with at least 25 test methods. Output only code.")
    res = []
    n_log0 = len(lines())
    for i in range(a.reps):
        head = f"[run {uuid.uuid4().hex}]\n{filler}\n\n"
        r = ask(head + task, cache=a.warm)
        if r is None:
            print("no engine timing line found", flush=True)
            continue
        if a.warm:
            w = ask(head + "List the first five multipliers, one per line.", cache=True)
            if w:
                r["suffix_reused"], r["suffix_read"], r["suffix_ms"] = w["reused"], w["read"], w["read_ms"]
        res.append(r)
        extra = f", suffix {r.get('suffix_read')} tok in {r.get('suffix_ms')} ms" if a.warm else ""
        print(f"  rep {i+1}: prefill {r['prefill']} t/s ({r['read']} tok), decode {r['decode']} t/s ({r['gen']} tok){extra}",
              flush=True)
    if not res:
        return 1
    pf = [r["prefill"] for r in res]; dc = [r["decode"] for r in res]
    s = {"label": a.label, "reps": res, "filler_tokens": filler_tokens,
         "prefill_med": statistics.median(pf), "prefill_min": min(pf), "prefill_max": max(pf),
         "decode_med": statistics.median(dc), "decode_min": min(dc), "decode_max": max(dc)}
    if a.warm:
        s["suffix_ms_med"] = statistics.median([r["suffix_ms"] for r in res if "suffix_ms" in r] or [0])
    os.makedirs(a.results, exist_ok=True)
    json.dump(s, open(f"{a.results}/speed_{a.label}.json", "w"), indent=1)
    line = (f"speed {a.label}: prefill {s['prefill_med']:.0f} t/s [{s['prefill_min']:.0f}-{s['prefill_max']:.0f}] | "
            f"decode {s['decode_med']:.1f} t/s [{s['decode_min']:.1f}-{s['decode_max']:.1f}] (n={len(res)})")
    print(line)
    open(f"{a.results}/speed.log", "a").write(line + "\n")

    if a.run_id and a.config:
        cfg = json.load(open(a.config))
        exe = cfg.get("exe", "")
        src = pathlib.Path(exe).resolve().parent.parent if exe else ROOT
        try:
            commit = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        except OSError:
            commit = ""
        try:
            unlock = "gpu_unlock.targets" in open("/proc/cmdline").read()
        except OSError:
            unlock = None
        log_all = lines()
        rec = {
            "schema": "speedy-bench/1", "run_id": a.run_id, "label": a.label,
            "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "engine_commit": commit, "binary_sha256": sha256(exe) if exe and os.path.exists(exe) else None,
            "config": os.path.abspath(a.config), "config_sha256": sha256(a.config),
            "env": cfg.get("env", {}), "unlock_on": unlock,
            "k": a.k, "filler_tokens_exact": filler_tokens, "warm": a.warm,
            "mechanism_lines": [l for l in log_all[:n_log0] if MECH.search(l)][-40:],
            "run_lines": [l for l in log_all[n_log0:] if MECH.search(l)][-40:],
            "result": s,
        }
        d = pathlib.Path(a.results) / "runs" / a.run_id
        d.mkdir(parents=True, exist_ok=True)
        json.dump(rec, open(d / f"{a.label}.json", "w"), indent=1)
        print("record", d / f"{a.label}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
