#!/usr/bin/env python3
"""Speedy Code Mode: one MCP tool that runs a short Python program against the workspace and returns only what the
program prints (bounded), instead of the agent calling read/grep/ls tools one at a time and injecting every result
into the prompt.

Why (plan F2 / PDF E24): a local model's cost per agent turn is the prompt it must read.  Each tool round trip adds its
whole output to the conversation; a real task here read 790K prompt tokens.  With Code Mode the agent writes one
program that reads several files, greps, filters and prints a summary, and only the summary enters the context.
The pattern is Cloudflare's "Code Mode" (agents SDK, MIT); nothing is copied from it.

Protocol: MCP over stdio (JSON-RPC 2.0, newline-delimited), no third-party packages.  Tool:
  run_python(code: str, timeout_s: int = 60) -> stdout + stderr of `code`, cut to CODEMODE_MAX_BYTES (default 8000)
The program runs in a fresh `python3 -I` process with the workspace as cwd and these helpers predefined:
  read(path, start=1, end=None) -> str        lines [start, end] of a text file (1-based, inclusive)
  grep(pattern, glob="**/*", max_hits=200)    -> list of "path:line: text" (regex, skips .git/node_modules/build*)
  ls(path=".", depth=1) -> list of paths
  sh(cmd, timeout=60) -> str                  a shell command's output (same permissions as the agent's bash tool)
Opencode: add to opencode.json
  "mcp": {"codemode": {"type": "local", "command": ["python3", "<speedy>/tools/codemode_mcp.py"], "enabled": true}}
Measure before adopting (tokens_in per task and task time on the deleg/real suites); see docs/CODE_MODE.md.
"""
import json, os, subprocess, sys

MAX = int(os.environ.get("CODEMODE_MAX_BYTES", "8000"))
PRELUDE = r'''
import fnmatch, os, pathlib, re, subprocess
_SKIP = {".git", "node_modules", "__pycache__", ".venv", "vendor"}
def read(path, start=1, end=None):
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    return "\n".join(lines[max(start, 1) - 1:end])
def _walk(glob):
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in _SKIP and not d.startswith("build")]
        for n in files:
            p = os.path.relpath(os.path.join(root, n))
            if fnmatch.fnmatch(p, glob) or fnmatch.fnmatch(n, glob):
                yield p
def grep(pattern, glob="**/*", max_hits=200):
    rx, hits = re.compile(pattern), []
    for p in _walk(glob):
        try:
            with open(p, encoding="utf-8", errors="strict") as f:
                for i, line in enumerate(f, 1):
                    if rx.search(line):
                        hits.append(f"{p}:{i}: {line.rstrip()[:200]}")
                        if len(hits) >= max_hits:
                            return hits
        except (UnicodeDecodeError, OSError):
            continue
    return hits
def ls(path=".", depth=1):
    base = pathlib.Path(path)
    out = []
    for p in sorted(base.rglob("*")):
        rel = p.relative_to(base)
        if len(rel.parts) <= depth and not (set(rel.parts) & _SKIP):
            out.append(str(p) + ("/" if p.is_dir() else ""))
    return out
def sh(cmd, timeout=60):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.stdout + r.stderr
'''

TOOL = {
    "name": "run_python",
    "description": ("Run a short Python program in the workspace and get back what it prints (cut to %d bytes). "
                    "Helpers: read(path,start,end), grep(regex,glob,max_hits), ls(path,depth), sh(cmd). Use it to read "
                    "several files, search and filter in ONE call and print only what you need." % MAX),
    "inputSchema": {"type": "object", "properties": {
        "code": {"type": "string", "description": "Python 3 source; print() the result"},
        "timeout_s": {"type": "integer", "description": "seconds, default 60, max 300"}},
        "required": ["code"]},
}


def run(code, timeout_s=60):
    timeout_s = max(1, min(int(timeout_s or 60), 300))
    try:
        r = subprocess.run([sys.executable, "-I", "-c", PRELUDE + "\n" + code], capture_output=True, text=True,
                           timeout=timeout_s, cwd=os.getcwd())
        out = r.stdout + (("\n[stderr]\n" + r.stderr) if r.stderr else "")
        if r.returncode:
            out += f"\n[exit {r.returncode}]"
    except subprocess.TimeoutExpired:
        out = f"[timeout after {timeout_s} s]"
    b = out.encode("utf-8", "replace")
    if len(b) > MAX:
        out = b[:MAX].decode("utf-8", "ignore") + f"\n[... cut: {len(b)} bytes, print less]"
    return out


def handle(msg):
    m, mid = msg.get("method"), msg.get("id")
    if m == "initialize":
        ver = (msg.get("params") or {}).get("protocolVersion", "2024-11-05")
        return {"jsonrpc": "2.0", "id": mid, "result": {"protocolVersion": ver, "capabilities": {"tools": {}},
                                                       "serverInfo": {"name": "speedy-codemode", "version": "0.1"}}}
    if m == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": [TOOL]}}
    if m == "tools/call":
        p = msg.get("params") or {}
        if p.get("name") != "run_python":
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": "unknown tool"}}
        a = p.get("arguments") or {}
        text = run(a.get("code", ""), a.get("timeout_s", 60))
        return {"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": text}]}}
    if m == "ping":
        return {"jsonrpc": "2.0", "id": mid, "result": {}}
    if mid is None:   # notifications (notifications/initialized, ...)
        return None
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {m}"}}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            reply = handle(json.loads(line))
        except Exception as e:  # a bad message must not end the server
            reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(e)}}
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
