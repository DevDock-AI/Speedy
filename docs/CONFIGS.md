# Presets (configs/presets) and their flags

`configs/make_configs.py` writes one serve config (`serve/server.py --config <file>`) per preset. Paths come from the
command line or from `SPEEDY_*` environment variables (`SPEEDY_ROOT`, `SPEEDY_EXE`, `SPEEDY_DATA`, `SPEEDY_PACKS`,
`SPEEDY_MODELS`, `SPEEDY_Q2_DIR`, `SPEEDY_LOGS`). `configs/presets/` holds the files generated with the default
placeholder paths (`/opt/speedy`, `/opt/models/...`, `/var/log/speedy`); generate your own instead of editing them:

```bash
python3 configs/make_configs.py --data /opt/models/strata-data --packs /opt/models/packs --models /opt/models/gguf \
    --q2-dir /opt/models/gguf/q38-ista-q20 --max-context 229376 --out configs/local
```
`configs/local/` is git-ignored. Options: `--max-context` (default 229376), `--q4-resident-gib` (default 80),
`--cuda-lib`, `--logs`, `--exe`.

## The presets
| Preset | Pack | Role | Reasoning | Pack-specific settings |
|---|---|---|---|---|
| `speedy-q2_0-med` | Q2_0 (Strata pack) | fast default | medium | - |
| `speedy-q2_0-1m` | Q2_0 | 1M overflow tier | medium | `--max-context 1048576 --rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144`; decode ~105 t/s at 600K, cold prompt reads slow (300K 449 t/s, 600K 405 t/s: the auto chunk falls to 512 tokens, see ROADMAP) |
| `speedy-q2_0-low` | Q2_0 | fast, short answers | low (smaller budget) | no `fit_max_tokens` |
| `speedy-iq3_s` | IQ3_S | middle | medium | - |
| `speedy-ud-iq4_xs` | Unsloth UD-IQ4_XS | **quality (recommended)** | medium | `--resident-budget-gib 80`, `STRATA_HC_Q8=1`, `STRATA_PREFILL_ROUTED_STREAM=1` |
| `speedy-ud-q4_k_xl` | Unsloth UD-Q4_K_XL | quality (larger, more RAM) | medium | `--resident-budget-gib 80`, `STRATA_HC_Q8=1` |

Which one: Q2_0 for speed (prefill ~2700-2850 t/s, decode ~150 t/s), UD-IQ4_XS for quality (prefill ~1750 t/s,
decode ~95-120 t/s, 51 GB RAM left free on 128 GB). The short quality gate scored all packs perfect, so pick by
your own tasks. Numbers: [BENCHMARKS.md](BENCHMARKS.md).

## Engine arguments shared by all presets
| Flag | Why ([TUNING-HISTORY.md](TUNING-HISTORY.md)) |
|---|---|
| `--expert-cache auto --prefill auto` | the engine sizes the expert cache and the prompt chunk from free VRAM; explicit chunks were slower |
| `--spec 4 --spec-min-p 0.5` + `--mtp <dir>` | MTP drafting at depth 4: best measured (3/5/6, no MTP, lookup-chain and suffix-draft rejected) |
| `--kv int8 --kv-resident 20480` | INT8 KV; 20,480 cells per QSA layer in VRAM, the rest streamed from pinned RAM (98.5-99 % VRAM hits at 128K) |
| `--vram-reserve-mib 700` | lowest value that holds 229K on every pack (400-450 was +15 % prefill at 16K but ran out of memory at 127K) |
| `--conversation-cache-mib 8192` | parked conversations for agent turns (99.2 % prefix reuse) |
| `--adapt-async 1 --adapt-swaps 80` | asynchronous adaptive expert tier |
| `--pool-affinity measure` | measured CPU worker placement |
| `--max-context N` | 229376 in every preset (all packs fit on 10 GB with reserve 700); `speedy-q2_0-1m`: 1048576 with YaRN x4 |

Environment in every preset (the same values are Speedy's built-in defaults, [SPEEDY_DEFAULTS.md](SPEEDY_DEFAULTS.md);
repeating them pins them on an upstream binary or under `STRATA_UPSTREAM_DEFAULTS=1`): `STRATA_ROUTE_TAIL_SKIP=0`,
`STRATA_PREFILL_CPU_SHARE=0`, `STRATA_PIN_BY_NEED=1`, `STRATA_EXPDEC=1`, `STRATA_EXCHANGE_ROTATE=1`,
`STRATA_ATTN_DAILY=1`, `STRATA_VERIFY_DEFER_CPU=1` + `_BELOW=0.1`, `STRATA_PROMPT_ATTN_IMMA=1`, `STRATA_KV_PREFETCH=1`,
`STRATA_PF_GROUP_SCRATCH=1`, `STRATA_PF_HC_REGION=1`.

Sampling (Qwen's recommendation): temperature 0.6, top-p 0.95, top-k 20, min-p 0, presence penalty 1.5.

## Context tiers
- **Native 229K:** `--max-context 229376` (make_configs `--max-context 229376`). Fits every pack on 10 GB at reserve 700.
- **1M overflow (Q2_0 only on 10 GB):** add `--max-context 1048576 --rope-scaling yarn --rope-scale 4
  --yarn-orig-ctx 262144` to the Q2_0 preset's `args`. Cold reads are slow (about 400-450 t/s at 300-600K; the auto
  chunk drops to 512 tokens) and the KV takes ~12 GiB of pinned RAM. The other packs do not start at 1M on 10 GB. A
  proxy can retry a request that exceeds the native window on the 1M model (the server answers 400 "request exceeds
  the context window").

## Host settings that matter
`ulimit -l unlimited` (or `LimitMEMLOCK=infinity` in a systemd unit), transparent huge pages `always`, and enough free
RAM for the pack (`--resident-budget-gib` keeps that many GiB of experts in RAM). On an LHR RTX 30 card the optional
[gpu-unlock](GPU-UNLOCK.md) module adds 10-16 % prefill.
