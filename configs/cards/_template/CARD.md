# <GPU name and VRAM, e.g. RTX 4090 24 GB>

<!-- Copy this folder to configs/cards/<vendor-model-vram>/ (lower case, e.g. rtx4090-24gb), then fill every row.
     A card folder needs: this CARD.md, at least one <N>x/ folder with presets, and measured numbers below.
     No serial numbers, GPU UUIDs, hostnames or personal paths: use the placeholders in docs/CONFIG-PLACEHOLDERS.md. -->

| Field | Value |
|:---|:---|
| Status | community (untested by maintainers) / verified |
| GPU | model, VRAM, compute capability (e.g. sm_89) |
| Build | `-DCMAKE_CUDA_ARCHITECTURES=<NN>` and any other CMake flags |
| GPU unlock | on / off / not applicable |
| Power limit | W per card, stock or OC/UV |
| PCIe | gen and width per card as negotiated, NVLink / P2P yes/no |
| CPU | model, cores/threads, AVX2 / AVX-512 |
| RAM | total, type and speed, channels |
| Storage | drive and interface the packs load from |
| OS / driver / CUDA | |
| Speedy commit | |
| Maintainer | @your-github-handle |

## Presets

| Folder | Contents |
|:---|:---|
| `1x/` | one card: which quants, which context |

## Measured (tok/s, average ± half spread)

Same method as the reference card (bench/speedbench.py, cold prompts at about 60K / 120K / 200K, see CONTRIBUTING.md).

| Quant | Prefill 60K | Prefill 120K | Prefill 200K | Decode | Cold 200K (s) | Follow-up (s) |
|:---|---:|---:|---:|---:|---:|---:|
| | | | | | | |

## Tuning notes

<!-- What you changed from the closest existing card's presets and why, with the A/B numbers. -->
