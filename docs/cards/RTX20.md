# RTX 20 (Turing, sm_75): notes

Status: **untested**. The rtx20 engine is built in the release but was not run by the Speedy maintainers (the family was
added in v0.3.1). Production reference: [RTX30.md](RTX30.md).

## Engine

- Release asset: the **rtx20** engine (`setup.py` `PREBUILT_FAMILY` 75 -> `rtx20`; release workflow `ARCHS=75`). sm_75 is the
  oldest compute capability CUDA 13 compiles for (`CUDA13_MIN_ARCH = 75`). From source: `-DCMAKE_CUDA_ARCHITECTURES=75`.
- Turing has FP16 tensor cores but no native BF16 or TF32 tensor cores (docs/OLDER_GPUS.md). Opt-in `STRATA_BF16_TC=1` runs
  the BF16 projections on the FP16 tensor cores; `STRATA_SM75_PERF=1` turns on Turing's own kernel set (see OLDER_GPUS.md).
- Per-block opt-in shared memory is 64 KB on Turing (src/kernels/cuda/fused_gr.cu), which changes some kernel tilings.

## What falls back on sm_75

| Feature | Needs | On RTX 20 |
|:---|:---|:---|
| `STRATA_GDN_TC=1` (tensor-core GDN, in the RTX 30 presets) | sm_80+, 48+ SMs, 97,808 B (split plan, sm_86/89/120) or 124,944 B (full plan, sm_80/sm_90) shared memory per block | silently keeps the older GDN kernel (src/prefill/kernels.cu); harmless to leave set |
| `STRATA_PROMPT_ATTN_IMMA` (int8 m16n8k32 MMA prompt attention) | sm_80+ | default off; older kernel |
| `STRATA_SELECT_TWO_PASS` (TF32 QSA select) | TF32 tensor cores (sm_80+) | not measured on this card family |
| `STRATA_PF_FUSED_Q2N` and other "sm_80+" fused paths | sm_80+ | not used |
| NVFP4 packs | `w4a8` int8 MMQ runs from sm_75 | slower than IQ3_S / IQ4_NL (1.35-1.6x at 32K on RTX 20-40, NVFP4.md) |

## Starting points (not measured on this card family)

| VRAM | Start from |
|:---|:---|
| 8 GB (2060 Super, 2070, 2080) | `configs/cards/_template/1x`; 8 GB cards start; route prior 1.5 applies |
| 11 GB (2080 Ti) | `rtx3080-10gb/1x` presets |
| 24 GB (TITAN RTX) | `_template/1x`, raise `--kv-resident` |

Community data in this repository: 2x TITAN RTX 24 GB (sm_75), Strata v0.1.40.3, IQ3_S at 262K: decode 70.9 / 77.3 / 71.1
tok/s, prompt 868.9 / 1548.9 / 1626.9 tok/s at 4K / 32K / 128K (COMMUNITY_BENCHMARKS.md; upstream Strata, not Speedy).

Use the agent checklist in [RTX30.md](RTX30.md#8-ai-agent-checklist), drop `STRATA_GDN_TC` from your comparison
(it has no effect), and report via a GitHub issue.
