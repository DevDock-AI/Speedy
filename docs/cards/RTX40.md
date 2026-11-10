# RTX 40 (Ada, sm_89): notes

Status: **untested by Speedy maintainers**. Upstream also states it has not run an RTX 40 card itself
(docs/MODELS.md: "RTX 40 cards are covered by the engine packages since 0.2.7 (sm_89 code); we have not run one
ourselves"). Production reference: [RTX30.md](RTX30.md).

## Engine

- Release asset: the **rtx40** engine (`setup.py` `PREBUILT_FAMILY` 89 -> `rtx40`; release workflow `ARCHS=89`).
  From source: `-DCMAKE_CUDA_ARCHITECTURES=89`.
- sm_89 defaults (include/strata/core/sm120_series.hpp, docs/DETAILS.md): `STRATA_MMVQ_IL_89` (the decode row tables where an
  RTX 4070 and an RTX 4080 agreed, plus a one-token Q3_K row; bitwise the same output) is on for compute capability 8.9 only;
  `STRATA_MMVQ_IL_89=0` or `STRATA_DEFAULTS=026` turns it off. `STRATA_PROMPT_ATTN_IMMA` is on (sm_80+). There is no sm_89
  "PERF" set like sm_86 / sm_120.
- `STRATA_GDN_TC`: sm_89 uses the split shared-memory plan (fits its 101,376 B per-block opt-in); needs at least 48 SMs,
  otherwise the engine keeps the older kernel.
- NVFP4 packs run at 8-bit precision on RTX 40 (no FP4 tensor cores); setup says UD-IQ4_XS is faster (docs/NVFP4.md).

## Starting points by VRAM (not measured on this card family)

| VRAM | Start from | First changes |
|:---|:---|:---|
| 8 GB (4060, 4060 Ti 8 GB) | `configs/cards/_template/1x` | as shipped; route prior 1.5 applies (≤ 13 GiB) |
| 12 GB (4070, 4070 Super / Ti) | `_template/1x` or `rtx3080-10gb/1x` | raise `--kv-resident` modestly |
| 16 GB (4060 Ti 16 GB, 4070 Ti Super, 4080) | `_template/1x` | `--kv-resident 24576`, try `--prefill auto:16384` |
| 24 GB (4090) | `_template/1x` | `--kv-resident` 24576-32768, GPU picture reader becomes the default |

## Expected to carry over from RTX 30

The reserve rule (`--vram-reserve-mib` + the engine's 256 MiB post-cache reserve), `STRATA_GDN_TC=1`, `--resident-budget-gib 80`
with 128 GB RAM, presence penalty 0 on UD presets, and on two cards `--layer-split` + `--trim-stage-weights` +
`--pipeline-windows 2`. Every one of these should be re-measured.

## Untested

Everything Speedy-specific on RTX 40. Community data in this repository: an RTX 4070 12 GB report (modified local engine,
IQ2_XS, 48.23 accepted-decode tok/s, COMMUNITY_BENCHMARKS.md) and a 4060 Ti + 5080 layer split (DETAILS.md). Follow the
agent checklist in [RTX30.md](RTX30.md#8-ai-agent-checklist) and report via a GitHub issue.
