# Serving Speedy

Request path: client (OpenCode, Claude Code local-worker, curl) -> **stats proxy** `127.0.0.1:9000` ->
**llama-swap** `127.0.0.1:8080` -> one **Speedy server** (`serve/server.py --engine strata`, port 18080+).

- `llama-swap.yaml`: one entry per quant; the OpenAI `model` field (or an alias) selects it. Only one model fits
  the 10 GB card, so llama-swap stops the old one first. `preload` loads Q2_0 at start.
- `serve.sh <config> <port>`: what llama-swap runs. Reads the API key from `~/.config/strata-llm/api_key`
  (mode 600) into `STRATA_API_KEY`; keys are never written into configs or this repo.
- Configs: `configs/generic/*.json` (reference machine, /opt paths) or generate your own with `configs/make_configs.py`.

## Stats proxy (not included)
The reference machine runs a small aiohttp proxy (private, not part of this repo) in front of llama-swap:
- forwards `/v1/*` unchanged (OpenAI-compatible chat/completions, streaming included);
- maps the Bearer key to a client name (key file `clients.json`, outside the repo) and logs every request to
  SQLite: time, client, model, reasoning effort, stream, status, total seconds, TTFT (first content/reasoning/
  tool-call byte), prompt/completion tokens, prefill and decode t/s, session id;
- samples GPU/CPU power and utilisation once a second for a dashboard.
Any OpenAI-compatible logging proxy (or none) works; Speedy itself needs nothing from it.
