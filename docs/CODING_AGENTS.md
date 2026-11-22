# Coding agents on Speedy

Notes for running a coding agent (OpenCode, Claude Code, Codex CLI, ...) against a Speedy server.

## AGENTS.md template

[agents/AGENTS.coding.md](agents/AGENTS.coding.md) is the instruction file we use for our agent benchmarks. Copy it into
your project as `AGENTS.md` (OpenCode, Codex CLI) or `CLAUDE.md` (Claude Code) and adapt it. Two of its rules matter
most for local models:

- start background processes detached (`nohup CMD >LOG 2>&1 </dev/null &`) and stop them by PID: a bare `CMD &` keeps
  the shell call open until the client's timeout, and `pkill -f PATTERN` also matches the agent's own command line;
- when the request names a limit or cap, test below it, at it and above it.

## OpenCode: allow bash for unattended runs

OpenCode asks before most shell commands by default. An agent that runs alone (no one there to answer the prompt)
loses those steps: the call is refused and the model has to work around it. For unattended runs we recommend:

```json
{
  "permission": {
    "bash": {
      "*": "allow",
      "rm -rf *": "ask",
      "git push*": "ask",
      "git reset --hard*": "ask"
    }
  }
}
```

Keep `ask` (or `deny`) entries for the commands you consider destructive. On the
Terminal-Bench 2 8-task subset (Qwen3.8-Flash-Next UD-Q4_K_XL, 1M context, the template above) the default permission
set passed 6-7 of 8 tasks, `"bash": "allow"` passed 8 of 8. When you sit in front of the session, the default
prompts are fine.
