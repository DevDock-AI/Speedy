# Evals

These are the scripts behind the IFBench result in the main README. They talk to any OpenAI-compatible endpoint, by default `BASE_URL=http://127.0.0.1:8080/v1`. The API key is read from `KEY_FILE` (default `~/.config/speedy/api_key`).

| Script | What it does |
|---|---|
| `ifbench_gen.py OUT [LIMIT]` | generates IFBench responses; resumable |
| `ifbench_score.py OUT` | scores them with the official IFBench checker (`IFBENCH_DIR` = a clone of allenai/IFBench) |
| `lcb_gen.py` / `lcb_score.py` | a fixed 70-problem LiveCodeBench v6 subset (`lcb_subset_v6_70.json`); needs `lcb_runner` installed |
| `summary.py` | writes SUMMARY.md with the score, tokens and time |

The sampling settings match the model card: reasoning high, T 1.0, top_p 0.95, top_k 20, a 32K cap. See `oai_chat.py`.
