# RTX 3080 10 GB

| Field | Value |
|:---|:---|
| Status | reference (every number in the main README comes from here) |
| GPU | NVIDIA GeForce RTX 3080 10 GB (LHR), sm_86 |
| Build | `-DCMAKE_CUDA_ARCHITECTURES=86` |
| GPU unlock | gpu-unlock on |
| Power limit | 320 W (GPU1), 280 W (GPU2) |
| PCIe | 4.0 x16 per card, no NVLink / P2P |
| CPU | AMD Threadripper PRO 5975WX, 32 cores / 64 threads, AVX2 |
| RAM | 128 GB DDR4-3200, 8 channels |
| Storage | 2 TB NVMe PCIe 4.0 |
| OS / driver / CUDA | Debian 13 (kernel 6.12) / 615.71 / 13.4 |
| Speedy commit | 0.3.0 |
| Maintainer | @DevDock-AI |

## Presets

| Folder | Contents |
|:---|:---|
| [`1x/`](1x/) | one card: Q2_0 (med/low reasoning), IQ3_S, UD-IQ4_XS, UD-Q4_K_XL at 229K, Q2_0 at 1M, `llama-swap.yaml` |
| [`2x/`](2x/) | two cards, layer split: every quant at 229K (`-2gpu-pw2`) and 1M (`-1m-2gpu`); UD-IQ4_XS is one preset from 0 to 1M (`-2gpu-pw2`), `llama-swap.yaml` |
| [`2x/parallel2/`](2x/parallel2/) | `"parallel": 2` variants. Measured slower on 2× 10 GB, kept for reference |

## Measured (tok/s, average ± half spread)

**1x**

| Quant | Prefill 60K | Prefill 120K | Prefill 200K | Decode | Cold 200K (s) | Follow-up (s) |
|:---|---:|---:|---:|---:|---:|---:|
| Q2_0 | 2,824 | 2,689 | 2,562 | 152 ±13 | 78 | 1.9 ±0.4 |
| IQ3_S | 2,347 | 2,303 | 2,248 | 130 ±11 | 89 | 1.75 ±0.05 |
| UD-IQ4_XS | 1,820 | 1,802 | 1,806 | 107 ±15 | 111 | 2.75 ±0.25 |
| UD-Q4_K_XL | 1,134 | 1,125 | 1,090 | 93 ±17 | 183 | 2.9 |

**2x**

| Quant | Prefill 60K | Prefill 120K | Prefill 200K | Decode | Cold 200K (s) | Follow-up (s) |
|:---|---:|---:|---:|---:|---:|---:|
| Q2_0 | 4,151 | 4,270 | 4,138 | 187 ±24 | 49 | 1.8 ±0.2 |
| IQ3_S | 4,074 | 4,284 | 4,264 | 163 ±8 | 47 | 1.6 |
| UD-IQ4_XS | 3,297 | 3,400 | 3,397 | 141 ±14 | 59 | 2.05 ±0.25 |
| UD-Q4_K_XL | 2,994 | 3,118 | 3,115 | 111 ±10 | 65 | 2.75 ±0.05 |

## Tuning notes

- Every value in these presets was adopted after a same-window A/B and a quality gate; see
  [docs/TUNING-HISTORY.md](../../../docs/TUNING-HISTORY.md).
- 10 GB is the limit that shapes these presets: `--kv-resident 20480`, `--vram-reserve-mib` (444 on one card, see below) and the
  layer split are sized for it. A card with more VRAM should start from here and raise them.
- 2x UD-IQ4_XS (`2x/strata-ud-iq4_xs-2gpu-pw2.json`, 2026-11-06): `--prefill auto:16384` caps the prompt chunk at
  16384 (55K prefill +7 %, 110K +9.5 %, 185K +11 %; 12288 and 20480 were slower) and `--kv-resident 24576`
  keeps more KV cells on the cards (decode +3.6 %, 20K-110K, same VRAM peak). Both: 3 A/B pairs + quality gate 8/8 + 4/4.
- 1x presets, `--vram-reserve-mib 444` (2026-11-09): the engine now reserves another 256 MiB after the expert cache;
  at 700 a single 10 GB card lost ~110 expert slots and the automatic prompt chunk dropped (UD-IQ4_XS 5632 -> 4608,
  UD-Q4_K_XL 3840 -> 3072): prefill -10 % / -17 %. 444 + 256 = the old 700: UD-IQ4_XS 1805-1810 t/s, UD-Q4_K_XL 1136
  at 57K, the same as Speedy 1GPU 0.2.0 (ABBA, same presets).
