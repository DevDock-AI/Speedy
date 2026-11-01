# Speedy on one GPU

The best configuration measured on **one RTX 3080 10 GB** (reference machine: Threadripper PRO 5975WX, 128 GB DDR4-3200, gpu-unlock on, ReBAR + Preferred IO, `iommu=pt`). The same engine runs on two GPUs when the preset has `--layer-split` ([EDITION-2GPU.md](EDITION-2GPU.md)).

## What is in it
- Engine: Speedy on Strata v0.2.7.1, plus K1 (`STRATA_KV_STAGE_REQ`, per-request KV staging for the 1M tier) and #4119 (two-pass attention select).
- Defaults (see `speedy_defaults.hpp`, `CHANGELOG.md`): #4039 prefill scratch, KV prefetch, INT8 tensor-core prompt attention, HC_Q8 on every preset.
- Per-model settings: #4283 routed stream on UD-IQ4_XS; CPU prefill share on IQ3_S; Q4 resident budget 80 GB; 229,376-token context with a 1M YaRN x4 overflow tier for Q2_0.
- `configs/production-1gpu/`: the exact production configs and the llama-swap file used on the test machine. Paths and the GPU UUID are placeholders (`<MODELS_DIR>`, `<GPU1_UUID>`, ...; docs/CONFIG-PLACEHOLDERS.md).

## Measured (medium reasoning, 60K / 120K / 200K prompt)
| Quant | Prefill t/s | Decode t/s |
|---|---|---|
| Q2_0 | ~2850 / 2710 / 2550 | 145–163 (HC_Q8 +5.5 %) |
| IQ3_S | ~2400 / 2364 / – | 114–142 (HC_Q8 +6 %) |
| UD-IQ4_XS | 1818 / 1819 / 1795 | 95–123 |
| Q2_0 1M tier, 300K | 1527 | 126–138 |

## Final numbers (2026-10-31)
See the README table (jobs 960/961, all four quants, 60/120/200K + warm turn).
