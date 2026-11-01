# Benchmarks: method and results

All numbers on this page were measured on the reference machine below, between 2026-10-29 and 2026-10-30. Raw logs
are kept by the maintainers; every number here is copied from them, and nothing is extrapolated.

## Reference machine
| | |
|---|---|
| GPU | RTX 3080 10 GB LHR (GA102, sm_86, 10de:2216), PCIe 4.0 x16, ReBAR on (BAR1 16 GiB), power limit 320 W |
| GPU unlock | FP16/BF16 tensor rate unlocked with the optional third-party [gpu-unlock](GPU-UNLOCK.md) module (locked numbers below) |
| CPU | Threadripper PRO 5975WX (32 cores, Zen 3, AVX2, no AVX-512) |
| RAM | 128 GB DDR4-3200, 8 channels |
| Storage | NVMe |
| Software | Debian 13, kernel 6.12, NVIDIA 615.71.10, CUDA 13.4, GCC 14.2 |
| Host settings | `ulimit -l unlimited`, transparent huge pages `always`, background containers in a low-weight cgroup slice |

## Method
- **Speed** (`bench/speedbench.py`): unique prompts of exact token counts (sized with the model's tokenizer), read
  cold; prefill t/s comes from the engine's own `prompt N tokens ... read in Y ms` line; decode t/s counts committed
  tokens of a fixed-length answer. The 60K / 120K / 200K tiers are 55K / 111K / 185K-token prompts ("16K" = 15,977
  tokens). Medium reasoning everywhere.
- **Warm turn** (`bench/speedbench.py --warm`, `bench/suffix.py`): after the 120K prompt, the same session (same
  session key, the byte-identical previous user message, the assistant reply as returned) plus a 924-token tool result.
  Reported: prefix reuse (the engine's `N tokens = R reused + X read` line) and the read time. The run fails below
  90 % reuse; every arm reported here reused 99.2 %.
- **A/B rule (apples to apples)**: one variable per arm. Same binary (or, for engine comparisons, the same packs and
  configs with only the executable changed), same config and environment otherwise, same prompts, medium reasoning,
  the control in the same window, **ABBA** order (ctl, B, B, ctl or ctl, A, B, C, ctl), one read per tier per run,
  at least two blocks. 30-45 minutes per candidate. Deviations are written down; a block disturbed by another GPU job
  is discarded and re-run.
- **Noise floor** (repeated controls): prefill +-0.5 %; decode +-4-8 % (the adaptive expert tier changes which experts
  sit in VRAM between runs). A decode difference inside that band is reported as "no change".
- **Quality gate** (for anything that changes arithmetic): the short gate = 8 delegation coding tasks (T1-T8) + 4 real
  repository bug fixes, medium reasoning, must match the control; plus a long-context needle test (64K-200K) for
  changes to KV or attention. Bitwise changes need their parity test instead.
- **Provenance**: each run writes a JSON with the engine commit, binary and config SHA-256, the unlock flag, and the
  engine's mechanism / chunk / pinning lines.

## Engines compared
| Label | What |
|---|---|
| Strata 0.1.45 | Strata v0.1.45 + patch 0001 (PR #2689/#2690): the production control before the rebase |
| Strata v0.2.7.1 | upstream v0.2.7.1 (46c152a1), same packs and configs with our environment ("arm a") |
| Speedy | Speedy 0.1.0 (v0.2.7.1 + #4039 defaults; + #4283 for UD-IQ4_XS) |

## Engine comparison: Q2_0 (fast), 229K configs, one block (ctl, a, b, c, ctl)
| Arm | 60K pf / dec | 120K pf / dec | 200K pf / dec | warm 924-tok read |
|---|---|---|---|---|
| Strata 0.1.45, run 1 | 2651 / 122 | 2552 / 139 | 2394 / 132 | 1.8-2.2 s |
| Strata 0.1.45, run 2 | 2623 / 129 | 2544 / 135 | 2400 / 121 | |
| v0.2.7.1 (a) our env | 2583 / 148 | 2498 / 165 | 2383 / 152 | 1.86 s |
| (b) + `STRATA_MMVQ_IL_PF=1` | 2615 / 151 | 2502 / 153 | 2384 / 146 | 1.86 s |
| (c) + `STRATA_GRAPH_PRIO=1` | 2617 / 156 | 2508 / 158 | 2387 / 150 | 2.1 s |

ABBA repeat (ctl, a, a, ctl; 60K x3, 120K x2): decode ctl 145 / 137, 142 / 136 vs (a) 156 / 158, 161: **decode +8-17 %,
prefill -1.5-2 %.** (b) and (c) add nothing beyond noise and were not adopted.

## Engine comparison: UD-IQ4_XS (quality), 229K configs
| Arm | 60K pf / dec | 120K pf / dec | 200K pf / dec |
|---|---|---|---|
| Strata 0.1.45 | 1352 / 84 | 1337 / 98 | 1291 / 94 |
| v0.2.7.1 (a) | 1261 / 104 | 1227 / 110 | 1219 / 112 |
| (b) + MMVQ_IL_PF | 1263 / 110 | 1229 / 113 | 1214 / 106 |
| (c) + GRAPH_PRIO | 1264 / 110 | 1228 / 112 | 1215 / 109 |

Decode +12-19 %, prefill -6.7 to -8 % (smaller auto prompt chunk), fixed by #4039 below. Short gate on v0.2.7.1:
Q2_0 8/8 first-try (8.0 min) + 4/4 (3.4 min); UD-IQ4_XS 8/8 (9.7 min) + 4/4 (4.4 min). PASS.

## Speedy's upstream PRs on the v0.2.7.1 base (ABBA, same binary, environment-only arms)
Control = Speedy with the PR switches off. Ranges are the two runs of each arm.

### UD-IQ4_XS, 229K (two blocks)
| Arm | Prefill 60K / 120K / 200K | Decode 60K / 120K / 200K | Warm read | Auto chunk |
|---|---|---|---|---|
| control | 1262-1267 / 1226-1228 / 1212-1220 | 91-97 / 114-124 / 111-118 | 2.5-3.1 s | 3584 |
| #4283 routed stream | 1395-1397 / 1335 / 1310-1311 (+10.5 / +8.9 / +8.0 %) | 90-99 / 115-121 / 110-115 | 2.5-2.8 s | 3584 |
| #4039 smaller scratch | 1627-1630 / 1624-1627 / 1604-1609 (+29 / +33 / +32 %) | 95-97 / 121-125 / 115-117 | 2.6-2.7 s | 5632 |
| #4039 + #4283 | 1783-1786 / 1764-1776 / 1735 (+41 / +44 / +43 %) | 93-102 / 105-116 / 115-120 | 2.6 s | 5632 |

### IQ3_S, 131K preset (no 200K tier)
| Arm | Prefill 60K / 120K | Decode 60K / 120K | Warm read |
|---|---|---|---|
| control (chunk 4864) | 1963-1968 / 1896-1903 | 115-123 / 131-136 | 2.9 s |
| #4039 (chunk 7680) | 2395-2405 / 2354-2374 (+22 / +24 %) | 114-121 / 137-142 | 2.2-2.4 s |
| #4039 + #4283 | 2377-2384 / 2334-2347 | 122-132 / 134-135 | 2.9 s |

#4283 adds nothing on IQ3_S (it streams little), so the IQ3_S preset uses #4039 only.

Run ids (speedbench labels, `-k60` / `-k120` JSON each; raw log `ab3-iq3a.out`, engine sha256 cf60b3933cb8287c):
control `v02-iq3-ctl-iq3a1`, `v02-iq3-ctl-iq3a6`; #4039 `v02-iq3-pf-iq3a2`, `v02-iq3-pf-iq3a5`; #4039 + #4283
`v02-iq3-pfrs-iq3a3`, `v02-iq3-pfrs-iq3a4`.

### Q2_0, 131K preset (no 200K tier)
| Arm | Prefill 60K / 120K | Decode 60K / 120K | Warm read |
|---|---|---|---|
| control (chunk 7168) | 2685-2691 / 2538-2564 | 144-147 / 152-153 | 1.9-2.4 s |
| #4283 | 2653-2660 / 2530-2535 (-1 / -0.5 %, noise) | 150-155 / 152-158 | 1.8-2.5 s |
| #4039 (chunk 8192) | 2827-2848 / 2668-2682 (+5.6 / +4.9 %) | 150-163 / 146-153 | 1.8-2.0 s |

Run ids (raw log `ab3-q2a.out`, same engine): control `v02-q2-ctl-q2a1`, `v02-q2-ctl-q2a6`; #4283 `v02-q2-rs-q2a2`,
`v02-q2-rs-q2a5`; #4039 `v02-q2-pf-q2a3`, `v02-q2-pf-q2a4`. Short gate log `gate-q2-pf.out`, run `lx-v02-q2-pf`.

Decode is unchanged within noise in every arm. Short gate: Speedy UD-IQ4_XS (#4039 + #4283) 8/8 first-try (9.8 min) +
4/4 (4.3 min); Speedy Q2_0 (#4039) 8/8 (8.5 min) + 4/4 (2.7 min). PASS. #4039 is bitwise per the PR's parity check; re-verified by the short quality gate.

## Quality-pack shoot-out (Strata 0.1.45, 229K, identical procedure)
| | IQ3_S | UD-IQ4_XS | UD-Q4_K_XL |
|---|---|---|---|
| VRAM expert slots / auto chunk | 1140 / 4352 | 773 / 3840 | 629 / 3328 |
| cold 57.8K pf / dec | 1775 / 107 | 1346 / 85 | 980 / 85 |
| cold 118.9K pf / dec | 1786 / 117 | 1327 / 97 | 997 / 97 |
| cold 200.4K pf / dec | 1738 / 112 | 1278 / 88 (157 s) | 976 / 91 (205 s) |
| warm turn after 119K (924 tokens) | 99.2 %, 2.4-2.9 s | 99.2 %, 2.5-3.1 s | 99.2 %, 2.9 s |
| delegation T1-T8 | 8/8, 11.6 min | 8/8, 14.5 min | 8/8, 10.3 min |
| real bugs (4) | 4/4, 4.1 min | 4/4, 3.9 min | 4/4, 4.8 min |
| RAM available, model loaded | 60 GB | 51 GB | 35 GB |

The short gate cannot separate the three (all perfect). UD-IQ4_XS decodes like UD-Q4_K_XL, reads prompts 33-36 %
faster and leaves 16 GB more RAM, so it is the recommended quality preset.

## Context tiers
**229K native** (Q2_0, 0.1.45 base, `--max-context 229376`, KV 20480 of 229376 cells resident, 2.71 GiB pinned RAM):
| `--vram-reserve-mib` | expert slots | auto chunk | 16K pf / dec | 200,374-token cold pf / dec | +535-token turn |
|---|---|---|---|---|---|
| 700 | 2854 | 6656 | 2396 / 144 | 2377 (84.3 s) / 131 | 1.58 s |
| 1000 | 2626 | 5888 | 2509 / 152 | 2318 (86.4 s) / 133 | 1.62 s |
| 1500 | 2247 | 4608 | 2202 / 148 | 2222 (90.2 s) / 132 | 1.63 s |
About 1 % of 200K prefill per 128 MiB of reserve; 700 fits 229K on every pack and stays the default.

**1M overflow** (YaRN x4, `--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144 --max-context 1048576`, v0.2.7.1
base, reserve 700): Q2_0 starts (KV 12.38 GiB pinned RAM, 1739 expert slots, prompt chunk 512): cold 302K 449 t/s
(~11 min), decode 105; warm 540-token turn after 302K 99.8 % reuse, 2.6 s; cold 600K 405 t/s (~25 min), decode 105.
IQ3_S, UD-IQ4_XS and UD-Q4_K_XL fail at start at any reserve up to 3000 MiB (the 512-token chunk's buffers do not
fit beside the larger dense weights); UD-IQ4_XS also fails at 512K.

## Platform steps (Strata 0.1.45, Q2_0 med preset, 16K prompt, medians)
| Stage | 16K prefill / decode | Notes |
|---|---|---|
| Linux first baseline (7 DIMMs) | 2013 / 112.3 | |
| 8th DIMM (8 channels) | 2013-2032 / 144-152 | decode +22-28 %; IQ3_S 1552 / 125, UD-Q4 1007 / 76 |
| ReBAR + PreferredIO | 2045 / 148 | 32K 2278 |
| driver 615.71.10 | 2047 / 143 | |
| gpu-unlock | 2260 / 140 | 32K 2567 (+12 %) |

## GPU unlock (FP16 tensor rate of an LHR card)
| | locked | unlocked | gain |
|---|---|---|---|
| cuBLAS FP16 GEMM (FP32 acc, 8192^3) | 62.4 TFLOPS | 106.2 TFLOPS | +70 % |
| BF16 / TF32 | 62.3 / 30.8 | 108.3 / 52.9 | +74 % / +72 % |
| INT8 | 194 TOPS | 199 TOPS | none |
| Q2_0 prefill 16K / 32K | 2047 / 2283 | 2260 / 2567 | +10 % / +12 % |
| decode | 143-149 | 140-150 | none (bandwidth-bound) |
Output checksums identical; no Xid errors under ~1 h of load (max 74 C).

Prefill paths after the unlock (0.1.45; prefill / decode at 16K / 32K / 128K): Q2_0 fused INT8 (default) 2257 / 150,
2552 / 146, 2503 / 136; Q2_0 dequant + cuBLAS FP16 1867 / 145, 1989 / 146, 1971 / 137; IQ3_S fused (default)
1691 / 124, 1799 / 123, 1844 / 117; IQ3_S FP16 1567 / 128, 1648 / 119, 1663 / 121; UD-Q4_K_XL FP16 (default)
1093 / 76, 1120 / 76, 1100 / 74; UD-Q4_K_XL `STRATA_MMQ_KQ=1` 915 / 76, 1007 / 72, 1043 / 74. The defaults stay.

## Earlier adopted changes (0.1.45 base)
| Change | Result |
|---|---|
| `STRATA_PROMPT_ATTN_IMMA=1` | Q2 32K prefill +2.0 %, decode +2.1 % (ABBA 2x5); suite PASS; needle 3/3 (89.6K, 127.3K) |
| `STRATA_KV_PREFETCH=1` | Q2 32K 2557 -> 2595 (+1.5 %), 128K 2490 -> 2525 (+1.4 %), two of two blocks |
| `--resident-budget-gib 80` (Q4 packs) | UD-Q4_K_XL decode 76 -> 95 t/s (+25 %), prefill +1 %, 41 GB RAM still free |

Rejected changes with their numbers: [TUNING-HISTORY.md](TUNING-HISTORY.md). Per-change ledger: [SPEEDY.md](../SPEEDY.md).
