# RTX 3080 20 GB (one card, or with a second RTX 3080 10 GB)

| Field | Value |
|:---|:---|
| Status | measured (Speedy 0.4.8 engine) |
| GPU | NVIDIA GeForce RTX 3080 20 GB, sm_86 |
| Build | `-DCMAKE_CUDA_ARCHITECTURES=86` |
| GPU unlock | gpu-unlock on |
| Power limit | 240 W (20 GB); 2x: 280 W on the 10 GB card |
| PCIe | 4.0 x16 per card, no NVLink / P2P |
| CPU | AMD Threadripper PRO 5975WX, 32 cores / 64 threads, AVX2 |
| RAM | 144 GB DDR4-3200 (7x 16 GB + 1x 32 GB) |
| Storage | 2 TB NVMe PCIe 4.0 |
| OS / driver / CUDA | Debian 13 (kernel 6.12) / 615.71 / 13.4 |
| Speedy commit | 1x: 0.4.8; 2x: 0.4.6 speed screen, presets on 0.5.0 |
| Maintainer | @DevDock-AI |

## Presets

| Folder | Contents |
|:---|:---|
| [`1x/`](1x/) | one 20 GB card: Q2_0, IQ3_S, UD-IQ3_XXS, UD-IQ4_XS, UD-Q4_K_XL at 229K (`strata-<quant>.json`) and at 1M (`strata-<quant>-1m.json`) |
| [`2x/`](2x/) | 20 GB (main, `<GPU1_UUID>`) + 10 GB (`<GPU2_UUID>`), layer split: UD-Q4_K_XL at 229K (`-20g10-pw2`) and 1M (`-1m-20g10`), UD-IQ4_XS and UD-IQ3_XXS from 0 to 1M (`-20g10-pw2`), `llama-swap.yaml` |

All presets use Speedy 0.5 defaults: `reasoning_budget_tokens` 32768 (written out so it is visible; the server default is
the same), and the tuned MTP drafter `<STRATA_DATA>/mtp/rt-tuned`, which `setup.py` writes next to the stock `mtp/rt`.
To run the stock drafter set `"mtp_drafter": "stock"` in the preset (or `STRATA_MTP_DRAFTER=stock`), or point `--mtp`
at `<STRATA_DATA>/mtp/rt`. The `2x/` presets also carry the tested 0.5 quality settings (`reasoning_nudge` [0.6, 0.8],
`STRATA_LOOP_BREAKER=1` with floor 0, `STRATA_TOOL_STEP_THINK_CAP=6144`).

`--pipeline-windows 2` in these presets is ignored on one card (it needs a layer split); it is kept so the files match
the measured configs.

## Measured (tok/s, mean of 2 runs)

**short** is a short chat prompt, **code** a coding prompt, **55K** and **110K** are prompts of about 55K and 110K
tokens. Decode is with MTP and the presets' default sampling.

| Quant | Decode short | Decode code | Decode 55K | Decode 110K | Prefill short | Prefill code | Prefill 55K | Prefill 110K |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q2_0 | 147 | 168 | 146 | 146 | 1,229 | 837 | 2,703 | 2,421 |
| IQ3_S | 110 | 125 | 127 | 120 | 1,125 | 765 * | 2,481 | 2,235 |
| UD-IQ3_XXS | 139 | 151 | 142 | 134 | 968 | 603 | 2,725 | 2,489 |
| UD-IQ4_XS | 142 | 142 | 129 | 125 | 929 | 474 | 2,725 | 2,477 |
| UD-Q4_K_XL | 122 | 121 | 122 | 115 | 729 | 346 | 2,409 | 2,200 |

\* one run only: the other run's code-prompt prefill measured 28 tok/s (a measurement fault, not reproduced) and is left out.

**1M tier:** all five `-1m` presets ran to a 250K-token prompt without errors; on UD-Q4_K_XL 1M, a greedy recall test
at about 150K tokens passed 48/48.

**2x** (20 GB + 10 GB, layer split, 0.4.6 engine, short speed screen):

| Quant | Decode short | Decode code | Decode 55K | Decode 110K | Prefill short | Prefill code | Prefill 55K | Prefill 110K |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| UD-IQ3_XXS | 197 | 210 | 204 | 180 | 1,590 | 1,710 | 3,679 | 3,957 |
| UD-IQ4_XS | 184 | 211 | 180 | 176 | 1,536 | 1,629 | 3,607 | 3,728 |
| UD-Q4_K_XL | 168 | 170 | 161 | 155 | 1,226 | 1,300 | 2,950 | 3,150 |
