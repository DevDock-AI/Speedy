# Tuning history (reference machine, Qwen3.8-Flash-Next)

All numbers: `speedbench` 16K prompt, 5 reps, median prefill / decode tok/s unless stated; decode run-to-run
noise is about +-8 %, prefill +-0.5 %. Full logs are kept by the maintainers outside the repository; this file keeps what a user of Speedy needs. Method and
full tables: [BENCHMARKS.md](BENCHMARKS.md).

## Production settings and why
| Setting | Reason |
|---|---|
| `--kv int8 --kv-resident 20480` | KV in int8, 20480 cells resident in VRAM, the rest streamed: fits 131072 context on 10 GB |
| `--vram-reserve-mib 700` | lowest value that passes a 127K-token needle (400-450 OOM) |
| `--conversation-cache-mib 8192` | parked conversations for agent turns (96-99 % prefix reuse) |
| `--spec 4 --spec-min-p 0.5` + MTP head | best measured speculative depth (3/5/6 and no-MTP rejected) |
| `--expert-cache auto --prefill auto --adapt-async 1 --adapt-swaps 80` | upstream auto planner, async cache adaptation |
| `--pool-affinity measure` | measured worker placement on the 32-core Threadripper |
| `STRATA_ROUTE_TAIL_SKIP=0` | the route-tail skip is lossy: off for quality |
| `STRATA_PREFILL_CPU_SHARE=0` | GPU + PCIe beats a CPU share in prefill here |
| `STRATA_PIN_BY_NEED=1`, `STRATA_EXPDEC=1`, `STRATA_EXCHANGE_ROTATE=1`, `STRATA_ATTN_DAILY=1` | upstream speed options, each measured on/neutral-or-better |
| `STRATA_VERIFY_DEFER_CPU=1`, `_BELOW=0.1` | Strata PR #2689/#2690 (patch 0001): defer CPU experts in verify |
| `STRATA_PROMPT_ATTN_IMMA=1` | A4 IMMA prompt attention: 32K prefill +2.0 %, decode +2.1 %, suite + needle 3/3 PASS (2026-10-29) |
| `STRATA_KV_PREFETCH=1` | KV prefetch: Q2 prefill +1.5 % (32K) / +1.4 % (128K) (2026-10-29) |
| UD-Q4 / UD-IQ4: `STRATA_HC_Q8=1`, `--resident-budget-gib 80` | hc projections in Q8; budget 80 = decode +25 % over 60 (J01) |

## Results by stage (Q2_0 med config)
| Stage | 16K prefill / decode | Notes |
|---|---|---|
| Windows (s3 config, pooled 35 reps) | 2128 / 132.4 | |
| Linux first baseline (7 DIMMs) | 2013 / 112.3 | gap -5 % / -6.5 % vs Windows |
| 8th DIMM (E00) | 2013-2032 / 144-152 | decode +22-28 %; IQ3_S 1552 / 125, UD-Q4 1007 / 76 |
| ReBAR + PreferredIO (W1) | 2045 / 148 | 32K 2278 |
| driver 615.71.10 (W2) | 2047 / 143 | same |
| gpu-unlock, optional third-party ([GPU-UNLOCK.md](GPU-UNLOCK.md)) | 2260 / 140 | 32K 2567 (+12 %); FP16 GEMM 62 -> 106 TFLOPS |

Prefill path comparison after the unlock (16K / 32K / ~128K): Q2 fused int8 2257 / 2552 / 2503 (default);
Q2 dequant+FP16 1867 / 1989 / 1971; IQ3_S fused 1691 / 1799 / 1844; UD-Q4 FP16 path 1093 / 1120.
The fused INT8 paths stay the default.

## Rejected (do not re-propose without a new reason)
Platform: `-march=znver3`+LTO, governor, GPU clock lock, hugetlbfs, mimalloc/jemalloc, SMT off, C2 off,
vram-reserve < 700, `--ple-io ram`, display-off arm (not tested: risk > gain).
Engine: `native_mma` INT8 tensor-core decode (+0.0 %), low-sync handoff (<= 2.5 % ceiling), arena placement,
SLRU flags, rerank/adapt-every 1, prefill head alias (-28 %), route prior / tail skip (lossy), spec 3/5/6, no MTP,
llama.cpp-based engines (no GDN+QSA+PLE at 200K on sm_86).

## Speedy engineering items (2026-10-29/30)
Adopted: A4 IMMA prompt attention, KV prefetch, J01 resident budget 80, J13 slot pricing, C4 PSI shedding (opt-in),
v0.2.7.1 rebase (decode +14-26 % on Q2, +13-24 % on IQ4_XS), Strata PR #4039 as a default (prefill IQ4_XS +29-33 %,
IQ3_S +22-24 %, Q2 +5 %) and PR #4283 in the UD-IQ4_XS preset (+41-44 % together with #4039).

Rejected on the v0.2.7.1 base: `STRATA_MMVQ_IL_PF=1` and `STRATA_GRAPH_PRIO=1` (no gain beyond noise); #4283 on
Q2_0 and IQ3_S (no gain: they stream little).

Rejected or closed, with the reason:
- J02 arm 2 (Basalt per-expert gate/up -> down readiness in the CPU pool): bitwise and TSan-clean, but decode
  +0.8 % (noise) and -0.9 % at 2K. Decode is bound by RAM bandwidth, not by pool scheduling.
- C3 pinned bounce buffer for checkpoint saves: warm TTFT unchanged, park 110 vs 98 ms. Superseded by upstream
  `STRATA_CKPT_ASYNC` in 0.2.7.
- C01 chunk / ring / lend planner: `--prefill auto` already takes the largest chunk the VRAM allows; explicit
  6144/8192 chunks are clamped to 3072/4096 and are slower (IQ3_S 1309/1714 vs 1794 auto). The only lever left is
  fewer chunk bytes per token (#4039).
- S08 F16 WMMA expert kernels: at most +4 % on IQ3_S after the unlock and ~0 on Q2; not built.
- J03 `STRATA_PREFILL_STREAM_MIN` per pack: Q4 8192 gave cold +5.8 %, IQ3 2048 gave a 1K suffix -16 % (0.1.45 base,
  not ABBA). Not adopted before a re-measurement on 0.2.7.1.
- D1 `--lookup-chain 3` (x0.90 on agent turns), `--suffix-draft 7` (x0.997): MTP-only stays.
- KQ256 (Q4 budget 80, repeat): no gain. `STRATA_QSA_ROWS=1`: no gain at 128K. `STRATA_MMQ_DIRECT`: needs a
  pointer-table build, not pursued.
- B5 AVX2 row-kernel audit vs Basalt: the same multi-row design; Basalt's only extra is an AVX-VNNI build (Zen 3 has
  none).
- E4 KV replacement policy replay: the KV stream already hits VRAM on 98.5-99 % of block reads at 128K, so a better
  policy is worth at most 1-2 % of reads.
- F4 restore regression: passes (same tokens) with `--pcie-frac 0 --adapt-async 0`. With the adaptive tier the
  output drifts between any two runs because the GPU/CPU expert split changes the rounding; that is not a restore
  bug.

The A/B logs are kept by the maintainer; each merged item is listed in SPEEDY.md with its switch, default and
measured result.
