# Speedy — dual-GPU edition (2GPU 0.1.0)

Speedy for **two GPUs**: GPU1 = main card (RTX 3080 10 GB, 320 W) always first and fastest; GPU2 = second RTX 3080 10 GB at 280 W. Both PCIe Gen4 x16, both gpu-unlocked, no P2P (GeForce, host route). A 3080 20 GB card will be added later. Built on top of the single-GPU edition (`edition/1gpu`, tag `speedy-1gpu-v0.1.0`).

## Extra in this edition
- J01 fix (stale pinned_share in the split planner).
- #4549 draft-head room and #4380 fused ring per device (on); #4534 slower-stage pricing (needs `--pipeline-windows 2`); #4522 stage trim (default OFF, `--trim-stage-weights` / `STRATA_STAGE_TRIM=1`); #3306 per-GPU prefill ceiling (`STRATA_PREFILL_MAX`, opt-in).
- `configs/production-2gpu/`: GPU1-only control, layer split, expert helper and peer-tier configs for Q2_0 and UD-IQ4_XS. Devices are pinned by UUID with `CUDA_DEVICE_ORDER=PCI_BUS_ID`, GPU1 listed first; replace the `<GPU1_UUID>`/`<GPU2_UUID>` placeholders (docs/CONFIG-PLACEHOLDERS.md).
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
| **Layer split 26 + `--trim-stage-weights` + `--vram-reserve-later-mib 500`** (`dual-iq4-split-trim.json`) | 3252 / 3379–3399 / 3404 (+78 / +86 / +89 %) | 135 / 126–128 / 115 | 1.96 s |

Why the plain split lost on IQ4_XS: every card kept the full dense copy (3.6 GiB), so CUDA1 had only 752 expert slots. The prompt chunk was then capped at 2816 tokens on both cards instead of 5632. Trimming each card to its own layers doubles the slots (2084 -> 4177) and restores the 5632 chunk. An explicit K is required for CUDA0 to trim (under `auto` only the later cards can). Keep the later-card reserve at >= 500 MiB, because at 300 MiB the MTP draft head does not fit.

Plain helper (`--expert-cache-device1 auto`) and the peer tier (host route) gave no gain on Q2 (within noise).
Layer-major prefill on a split: it stays off at the default reserves (needs ~1.2 GiB free per card). With reserves 1800/1400 it engages, and then the 120K prompt dies with `illegal memory access` (open bug, super-chunk 24576 > KV-resident 20480 suspected).

## Switches tested on the split (ABBA, 1 rep per tier)
| Switch | Q2_0 | UD-IQ4_XS | Verdict |
|---|---|---|---|
| `STRATA_PREFILL_HELP=1` (not bit-identical, owner-approved) | warm 1.45–2.0 -> 1.42–1.45 s, decode 151–158 -> 153–164 | warm 2.05–2.08 -> 2.05–2.06 s, decode same | **adopted** (`dual-*-final.json`) |
| Helper `--remote-expert-opt` + `STRATA_DISJOINT_ADAPT=1` (instead of split) | prefill = single GPU (−40 % vs split), decode 149–166 vs 146–158 | decode 102–111 vs 117–136, prefill −50 % | rejected |
| `STRATA_SPLIT_MTP_BATCH=1` | (job 33) | decode 60/120/200K 121–124/122–125/118–121 -> 128–134/126/130–131 (+6/+2/+9 %), tokens/window ~3.2 both, warm 2.0–2.4 -> 2.0 s | adopted on IQ4_XS |
| `--pipeline-windows 2` | – | decode 121–132 vs 121–132 | no gain |
| Peer tier (host route) | = GPU1 only | – | rejected |

Decode timeline (IQ4_XS split, 120K, `STRATA_DECODE_TIMING=1`): ~25 ms/window, 3.1–3.4 tokens/window. About 11 ms of that is GPU-reach wait (the stages run in series) and about 1 ms is CPU experts, so decode is bound by the stages, not the CPU.
Binary: `speedy-release/editions/2gpu` (sha256 23fc7b9a, integ/dual ce4b55c0).

## All quants, final presets (split vs GPU1 only, prefill 60/120/200K)
- Q2_0: +53 / +69 / +73 %
- UD-IQ4_XS (trim, K 26, MTP batch): +78-86 / +86-90 / +89-92 %; decode +2-9 %
- UD-Q4_K_XL (trim, K 26): 1121-1135 -> 2977-3116 (+163 / +175 / +184 %); decode same or better
- IQ3_S (trim, K 26): +76 / +83 / +89 %; decode +0-10 %
Rejected: SPLIT_OWN=auto, K 24/28, Q2 MTP batch (noise), GPU2 at 320 W (+1 %). Gate: IQ4_XS and Q2 final presets 8/8 + 4/4.

## Final production numbers (2026-10-31)
See the README table (job 800: production presets `dual-*-final-pw2.json`, overlap + penalties, 2 reps each).
