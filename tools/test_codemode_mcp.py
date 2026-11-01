"""tools/codemode_mcp.py: the MCP handshake, tools/list, a run_python call with the helpers, the output cap, timeouts."""
import json, os, subprocess, sys, pathlib

HERE = pathlib.Path(__file__).resolve().parent


def rpc(lines, env=None):
    p = subprocess.run([sys.executable, str(HERE / "codemode_mcp.py")], input="\n".join(json.dumps(l) for l in lines) + "\n",
                       capture_output=True, text=True, cwd=str(HERE.parent), env={**os.environ, **(env or {})}, timeout=60)
    return [json.loads(l) for l in p.stdout.splitlines()]


def test_handshake_and_list():
    r = rpc([{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
             {"jsonrpc": "2.0", "method": "notifications/initialized"},
             {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}])
    assert r[0]["result"]["protocolVersion"] == "2025-06-18"
    assert len(r) == 2 and r[1]["result"]["tools"][0]["name"] == "run_python"


def test_call_helpers():
    code = "print(len(grep(r'codemode', 'tools/*.py')) > 0); print(read('tools/codemode_mcp.py', 1, 1)); print(any('tools/' in p for p in ls('.', 1)))"
    r = rpc([{"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "run_python", "arguments": {"code": code}}}])
    text = r[0]["result"]["content"][0]["text"].splitlines()
    assert text[0] == "True" and text[1].startswith("#!") and text[2] == "True"


def test_cap_and_errors():
    r = rpc([{"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "run_python", "arguments": {"code": "print('x' * 5000)"}}},
             {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "run_python", "arguments": {"code": "raise SystemExit(3)"}}},
             {"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "run_python", "arguments": {"code": "import time; time.sleep(5)", "timeout_s": 1}}},
             {"jsonrpc": "2.0", "id": 7, "method": "nope"}], env={"CODEMODE_MAX_BYTES": "1000"})
    assert "[... cut" in r[0]["result"]["content"][0]["text"]
    assert "[exit 3]" in r[1]["result"]["content"][0]["text"]
    assert "timeout" in r[2]["result"]["content"][0]["text"]
    assert r[3]["error"]["code"] == -32601


if __name__ == "__main__":
    for t in (test_handshake_and_list, test_call_helpers, test_cap_and_errors):
        t()
    print("codemode_mcp tests: ok")
