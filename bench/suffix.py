#!/usr/bin/env python3
"""Speedy suffix / warm-turn harness (plan F1 / PDF E04-G01-J06): how fast a follow-up turn is when a long prefix
is already in the engine (the agent case: 96-99 % of a real session's prompt is reused).

  python3 bench/suffix.py <label> <engine_log> --tokenizer <pack>/tokenizer \
      [--prefix 32,128] [--suffix 128,512,2048,8192] [--reps 2] [--scenario chain|abA|boundary]

Scenarios
  chain     prefix P (cold), then P + suffix S as a continuation of the same conversation (warm turn)
  abA       conversation A, then an unrelated B of the same size, then A again: does A survive parking?
  boundary  prefixes just below / at / above the 16384 conversation-cache grid and the 7168 prompt chunk
Each request's engine line "prompt N tokens = R reused + X read in Y ms" gives reused / read / ms; the result is
results/suffix_<label>.json with one row per (scenario, prefix, suffix, rep).  Sizes are in thousands of tokens
for --prefix and in tokens for --suffix, all made exact with the tokenizer.

Agent fidelity (user requirement): every request of a conversation is built like a real agent's - one fixed system
prompt, the same tools in the same order, the same template / reasoning parameters (--reasoning-effort, default
medium), a constant session id / prompt_cache_key per conversation, no timestamps or random ids inside the prefix, and
each follow-up only APPENDS (the previous assistant turn is sent back byte-for-byte, reasoning included).  Each warm
turn prints its reuse %: reused / (prompt - newly appended tokens); below --min-reuse (default 90 %) the run FAILS
(exit 2) - a warm-turn number measured on a broken cache is not a warm-turn number.

Status: skeleton (2026-10-29).  Prefix grid 200K, spec off/on and the proxy-capture replay (bench/replay.py)
are the next steps; the PR #89 port is decided from what this harness shows.
"""
import argparse, json, os, pathlib, re, sys, time, urllib.request, uuid

LINE = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from speedbench import load_tokenizer  # noqa: E402


SYSTEM = ("You are an autonomous coding agent working in the user's repository. You can read files, search the code, "
          "edit files and run commands with the tools below. Read the code you need before changing it, make small and "
          "correct edits, and explain briefly what you did.")
TOOLS = [{"type": "function", "function": {"name": n, "description": d, "parameters": {
    "type": "object", "properties": {k: {"type": "string"} for k in ps}, "required": list(ps)}}}
    for n, d, ps in (("read", "Read a file of the repository.", ("path",)),
                     ("grep", "Search the repository with a regular expression.", ("pattern", "path")),
                     ("edit", "Replace text in a file.", ("path", "old", "new")),
                     ("bash", "Run a shell command.", ("command",)))]


class ReuseError(RuntimeError):
    pass


def text_of(tok, n_tokens, seed):
    """About n_tokens of distinct code-like text (exact count returned)."""
    out, total, i = [], 0, 0
    while True:
        line = f"def f_{seed}_{i}(x):\n    return (x ^ {i * 2654435761 % 1000003}) + {i}  # {seed}"
        n = len(tok.encode(line + "\n"))
        if total + n > n_tokens:
            break
        out.append(line); total += n; i += 1
    s = "\n".join(out)
    return s, len(tok.encode(s))


class Client:
    def __init__(self, url, log, effort="medium"):
        self.url, self.log, self.effort = url, log, effort
        keyf = os.path.expanduser("~/.config/strata-llm/api_key")
        self.hdr = {"Content-Type": "application/json"}
        if os.path.exists(keyf):
            self.hdr["Authorization"] = "Bearer " + open(keyf).read().strip()

    def _lines(self):
        try:
            return open(self.log, encoding="utf-8", errors="replace").read().splitlines()
        except OSError:
            return []

    def turn(self, messages, session, max_tokens=64):
        """One agent request: fixed system prompt + tools first, `messages` after; returns (assistant message, stats)."""
        n0 = len(self._lines())
        body = json.dumps({"model": "worker", "messages": [{"role": "system", "content": SYSTEM}] + messages,
                           "tools": TOOLS, "max_tokens": max_tokens, "temperature": 0, "reasoning_effort": self.effort,
                           "prompt_cache_key": session, "user": session, "cache_prompt": True}).encode()
        t0 = time.time()
        r = json.loads(urllib.request.urlopen(urllib.request.Request(self.url + "/v1/chat/completions", body, self.hdr),
                                              timeout=3600).read())
        wall = time.time() - t0
        time.sleep(0.5)
        ms = [m for m in (LINE.search(l) for l in self._lines()[n0:]) if m]
        # the request that READ this turn's tokens: a reasoning config adds a short follow-up request with a larger
        # prompt (prompt + reasoning) but few tokens read, so the largest prompt is the wrong line
        m = max(ms, key=lambda x: int(x[3])) if ms else None
        msg = r["choices"][0]["message"]
        reply = {"role": "assistant", "content": msg.get("content") or ""}
        if msg.get("reasoning_content"):
            reply["reasoning_content"] = msg["reasoning_content"]   # sent back unchanged, as an agent does
        return reply, {"prompt": int(m[1]) if m else None, "reused": int(m[2]) if m else None,
                       "read": int(m[3]) if m else None, "read_ms": int(m[4]) if m else None, "wall_s": round(wall, 3)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("label"); ap.add_argument("log"); ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prefix", default="32"); ap.add_argument("--suffix", default="128,512,2048,8192")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--scenario", default="chain", choices=["chain", "abA", "boundary"])
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--results", default=os.environ.get("SPEEDY_RESULTS", str(HERE / "results-local")))
    ap.add_argument("--reasoning-effort", default="medium")
    ap.add_argument("--min-reuse", type=float, default=90.0, help="fail below this reuse %% of the cacheable prefix")
    a = ap.parse_args()
    tok = load_tokenizer(a.tokenizer)
    c = Client(a.url, a.log, a.reasoning_effort)
    bad = []

    def check(tag, prev_prompt, st):
        """reuse % of what the cache could hold: the previous turn's whole prompt (its reply is re-read either way)."""
        if st["reused"] is None or not prev_prompt:
            print(f"  {tag}: no engine line", flush=True); bad.append(tag); return
        pct = 100.0 * st["reused"] / prev_prompt
        st["reuse_pct"] = round(pct, 2)
        print(f"  {tag}: reuse {st['reused']}/{prev_prompt} = {pct:.1f} % of the cached prefix "
              f"({100.0 * st['reused'] / st['prompt']:.1f} % of the prompt)", flush=True)
        if pct < a.min_reuse:
            bad.append(tag)
    rows = []
    q = "Name the function with the largest constant. One line."
    if a.scenario == "boundary":
        prefixes = [n for b in (7168, 16384, 32768) for n in (b - 64, b, b + 64)]
    else:
        prefixes = [int(float(p) * 1000) for p in a.prefix.split(",")]
    for p in prefixes:
        for s in [int(x) for x in a.suffix.split(",")]:
            for rep in range(a.reps):
                seed = uuid.uuid4().hex[:8]   # makes each rep's prefix cold; constant within the conversation
                session = "suffix-" + seed
                ptxt, pn = text_of(tok, p, seed)
                msgs = [{"role": "user", "content": ptxt + "\n\n" + q}]
                reply, cold = c.turn(msgs, session)
                stxt, sn = text_of(tok, s, seed + "s")
                if a.scenario == "abA":
                    btxt, _ = text_of(tok, p, seed + "B")
                    c.turn([{"role": "user", "content": btxt + "\n\n" + q}], session + "B")
                msgs += [reply, {"role": "user", "content": stxt + "\n\n" + q}]
                _, warm = c.turn(msgs, session)
                row = {"scenario": a.scenario, "prefix_tokens": pn, "suffix_tokens": sn, "rep": rep, "cold": cold, "warm": warm}
                rows.append(row)
                check(f"P={pn} S={sn} rep{rep}", cold["prompt"], warm)
                print(f"{a.scenario} P={pn} S={sn} rep{rep}: cold {cold['read']} read {cold['read_ms']} ms | warm "
                      f"{warm['reused']} reused + {warm['read']} read in {warm['read_ms']} ms (wall {warm['wall_s']} s)",
                      flush=True)
    os.makedirs(a.results, exist_ok=True)
    out = f"{a.results}/suffix_{a.label}.json"
    json.dump({"label": a.label, "rows": rows}, open(out, "w"), indent=1)
    print("wrote", out)
    if bad:
        print(f"FAIL: {len(bad)} warm turn(s) reused less than {a.min_reuse} % of the cached prefix: {bad}", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
