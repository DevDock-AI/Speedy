# Speedy on two GPUs

Speedy on **two GPUs**: GPU1 is the main card (RTX 3080 10 GB, 320 W), listed first. GPU2 is a second RTX 3080 10 GB at 280 W. Both are PCIe Gen4 x16 and both are gpu-unlocked. There is no P2P (GeForce cards use the host route). The model is split by layers (`--layer-split 26`).

## Extra in this edition
- J01 fix (stale pinned_share in the split planner).
- #4549 draft-head room and #4380 fused ring per device (on); #4534 slower-stage pricing (needs `--pipeline-windows 2`); #4522 stage trim (default OFF, `--trim-stage-weights` / `STRATA_STAGE_TRIM=1`); #3306 per-GPU prefill ceiling (`STRATA_PREFILL_MAX`, opt-in).
- `configs/cards/rtx3080-10gb/2x/` holds the production presets: `strata-<quant>-2gpu-pw2.json` (229K) and `strata-<quant>-1m-2gpu.json` (1M tier). Devices are pinned by UUID with `CUDA_DEVICE_ORDER=PCI_BUS_ID`, with GPU1 listed first. Replace the `<GPU1_UUID>`/`<GPU2_UUID>` placeholders ([CONFIG-PLACEHOLDERS.md](CONFIG-PLACEHOLDERS.md)).
- #3559 (stage-local HC_Q8 upload) gives UD-IQ4_XS and UD-Q4_K_XL +7-11 % expert slots per card. `STRATA_PREFILL_STREAM_MIN=512` on UD-IQ4_XS cuts short follow-up turns: 730 tokens 2.15 -> 1.86 s.
- `tools/dual_ab.sh`: the ABBA harness used for every number below.

## Measured so far (Q2_0, medium reasoning)
| Mode | Prefill 60K / 120K / 200K | Decode |
|---|---|---|
| GPU1 only | 2811–2816 / 2675 / 2552 | 146–159 |
| Layer split GPU1+GPU2 | 4292–4334 / 4519–4529 / 4405–4420 (+53 / +69 / +73 %) | 148–162 |


## UD-IQ4_XS (medium reasoning)
| Mode | Prefill 60K / 120K / 200K | Decode 60/120/200K | Warm 2K turn @120K |
|---|---|---|---|
| GPU1 only | 1812–1823 / 1814–1825 / 1794–1802 | 98–102 / 114–121 / 109–124 | 2.4–4.1 s |
| Layer split auto (no trim) | 1648–1658 / 1656–1660 / 1620–1623 (−9 %) | 98 / 95–104 / 100–103 | 2.66–2.69 s |
| **Layer split 26 + `--trim-stage-weights` + `--vram-reserve-later-mib 500`**  | 3252 / 3379–3399 / 3404 (+78 / +86 / +89 %) | 135 / 126–128 / 115 | 1.96 s |

Why the plain split lost on IQ4_XS: every card kept the full dense copy (3.6 GiB), so CUDA1 had only 752 expert slots. The prompt chunk was then capped at 2816 tokens on both cards instead of 5632. Trimming each card to its own layers doubles the slots (2084 -> 4177) and restores the 5632 chunk. An explicit K is required for CUDA0 to trim (under `auto` only the later cards can). Keep the later-card reserve at >= 500 MiB, because at 300 MiB the MTP draft head does not fit.

Plain helper (`--expert-cache-device1 auto`) and the peer tier (host route) gave no gain on Q2 (within noise).
Layer-major prefill on a split: it stays off at the default reserves (needs ~1.2 GiB free per card). With reserves 1800/1400 it engages, and then the 120K prompt dies with `illegal memory access` (open bug, super-chunk 24576 > KV-resident 20480 suspected).

## Switches tested on the split (ABBA, 1 rep per tier)
| Switch | Q2_0 | UD-IQ4_XS | Verdict |
|---|---|---|---|
| `STRATA_PREFILL_HELP=1` (not bit-identical; quality-gated) | warm 1.45–2.0 -> 1.42–1.45 s, decode 151–158 -> 153–164 | warm 2.05–2.08 -> 2.05–2.06 s, decode same | **adopted**  |
| Helper `--remote-expert-opt` + `STRATA_DISJOINT_ADAPT=1` (instead of split) | prefill = single GPU (−40 % vs split), decode 149–166 vs 146–158 | decode 102–111 vs 117–136, prefill −50 % | rejected |
| `STRATA_SPLIT_MTP_BATCH=1` | – | decode 60/120/200K 121–124/122–125/118–121 -> 128–134/126/130–131 (+6/+2/+9 %), tokens/window ~3.2 both, warm 2.0–2.4 -> 2.0 s | adopted on IQ4_XS |
| `--pipeline-windows 2` | – | decode 121–132 vs 121–132 | no gain |
| Peer tier (host route) | = GPU1 only | – | rejected |

Decode timeline (IQ4_XS split, 120K, `STRATA_DECODE_TIMING=1`): ~25 ms/window, 3.1–3.4 tokens/window. About 11 ms of that is GPU-reach wait (the stages run in series) and about 1 ms is CPU experts, so decode is bound by the stages, not the CPU.

## All quants, final presets (split vs GPU1 only, prefill 60/120/200K)
- Q2_0: +53 / +69 / +73 %
- UD-IQ4_XS (trim, K 26, MTP batch): +78-86 / +86-90 / +89-92 %; decode +2-9 %
- UD-Q4_K_XL (trim, K 26): 1121-1135 -> 2977-3116 (+163 / +175 / +184 %); decode same or better
- IQ3_S (trim, K 26): +76 / +83 / +89 %; decode +0-10 %
Rejected: SPLIT_OWN=auto, K 24/28, Q2 MTP batch (noise), GPU2 at 320 W (+1 %). Gate: IQ4_XS and Q2 final presets 8/8 + 4/4.

## Final production numbers (2026-10-31)
See the README tables (production presets `strata-*-2gpu-pw2.json`, with decode overlap and penalties, 2 reps each).
