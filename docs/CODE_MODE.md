# Code Mode for local agents (Speedy F2)

`tools/codemode_mcp.py` is a dependency-free MCP server with one tool, `run_python`. The agent writes a short
program that reads, greps and filters (`read`, `grep`, `ls`, `sh` are predefined) and gets back only what it prints,
cut to `CODEMODE_MAX_BYTES` (8000). One call replaces several read/grep round trips, and only the summary enters
the prompt. That matters on a local model, where every turn re-reads the prompt: one real task here read 790K
prompt tokens. Pattern from Cloudflare's Code Mode (agents SDK, MIT); no code copied.

## Trust
`run_python` runs the agent's code as a normal Python process and `sh()` runs any shell command, both with the
permissions of the user who started the server and with no sandbox. Enable it only for an agent you would let run
shell commands on this machine anyway (or start it inside a container or a dedicated user account).

## Enable (opencode)
```json
"mcp": {"codemode": {"type": "local", "command": ["python3", "/opt/speedy/tools/codemode_mcp.py"], "enabled": true}}
```
Keep the tool list and its order the same on every turn (prefix-cache stability). Optionally tell the agent in
`agent/lean.md`: "To inspect several files or search, prefer one run_python call that prints only what you need."

## Adoption gate (not yet run)
A/B on the deleg (T1-T7) and real-8 suites, same model/config: tokens_in per task, wall time per task and pass
count. Adopt only with fewer tokens_in or less wall time and no lost tasks. Tests: `python3 tools/test_codemode_mcp.py`.
