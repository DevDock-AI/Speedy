# RTX 50 (Blackwell, sm_120): tuning guide

Status: **community**. **The Speedy maintainers have not measured RTX 50 hardware themselves.** Everything below is
either read from the code, taken from upstream Strata measurements on an RTX PRO 5000 Blackwell (also compute capability
12.0), or a community report quoted as written in this repository. The production reference is
[RTX30.md](RTX30.md).

## 1. Engine and build

- Release asset: the **rtx50** engine (`strata-linux-x64-rtx50.zip` / `strata-windows-x64-rtx50.zip`). `setup.py` maps
  compute capability 120 to `rtx50`; the release workflow builds it with `ARCHS=120`.
- From source: `-DCMAKE_CUDA_ARCHITECTURES=120`. CUDA 12.8 or newer is required for sm_120, and CUDA 13 is recommended:
  engines built with CUDA 12.8 crashed on long prompts on Linux (#220). CUDA 13.2.0 / 13.2.1's nvcc produced garbage
  prompts and answers for sm_120 (#892, #968); setup picks another 13.x toolkit when it finds one (setup.py).
- The optional FP4 x FP4 prompt kernel is the one unit CMake builds for the arch-specific target (`120a`); the rest of the
  engine keeps the architectures as given (docs/NVFP4.md).

## 2. What the engine turns on by itself on compute capability 12.0

All of this is in [include/strata/core/sm120_series.hpp](../../include/strata/core/sm120_series.hpp) and used across
`src/` (fused_gr.cu, moe_fused_iq.cu, verify.cpp, qsa_decode_attn.cu, generate.cpp, peer_experts.cpp). It applies only to
a card that reports exactly 12.0, CUDA builds only (a HIP gfx1200/gfx1201 also reports 12.0 and is excluded; sm_121 and
sm_100 keep their defaults).

| Switch | What it does | Condition |
|:---|:---|:---|
| `STRATA_SIDE_MIXER=2` | The mixer's input-only work on branches of the verify graph | any 12.0 card |
| `STRATA_ATTN_DAILY=1` | The Daily decode attention kernel | any 12.0 card |
| `STRATA_GR_UP75=1` | Hyper-connection up projection with row loads hoisted | any 12.0 card |
| `STRATA_GR_DOWN_TG=1` (`=2` under 80 SMs) | Hyper-connection down projection, one token (or token pair) a block row | any 12.0 card |
| `STRATA_GU_SWIGLU=1` | Routed experts' gate/up, SwiGLU and q8_1 blocks in **one fused launch** | any 12.0 card |
| `STRATA_GR_UP_WIDE=1` | Up projection with one warp a row (+1.9 % story / +0.7 % code decode on RTX PRO 5000) | any 12.0 card |
| `STRATA_GR_V3` | Stream-split hyper-connection read (another summation order; +1.2 % decode on RTX PRO 5000) | discrete, more than 13 GiB |
| `STRATA_PF_FUSED` | Native packs' prompt experts on fused int8 kernels (IQ2_S / IQ3_XXS / IQ3_S / IQ4_XS gate/up; rounds differently, KL-checked) | 80+ SMs only; an RTX 5070 (48 SMs) keeps MMQ because the fused kernels were 4.7 % slower on 20K prompts there |
| `native_mmvq_il` 12.0 tables | Interleaved small-batch matvec for 5-8 tokens, plus a 2-4 token table on 80+ SM cards (`STRATA_MMVQ_IL_WIDE=0`, `STRATA_MMVQ_IL_120=0` turn them off) | any 12.0 card |
| Decode peer tier (`STRATA_PEER_TIER2`) | Experts served from a peer card instead of a layer split | 2+ alike 12.0 cards (same CC, SM count and VRAM within 10 %), each at least 15 GiB, no hand-given `--layer-split` / `--peer-device`. **Ships as log-only**: the rule is decided and logged but not applied unless `STRATA_PEER_TIER2_AUTO=1`; `STRATA_PEER_TIER2=1` forces it |

Upstream measurement for the bitwise set (RTX PRO 5000 Blackwell 72 GB, IQ3_S, every expert in VRAM): a 4-token verify
round 13.70 -> 12.23 ms with the same text.

- **`STRATA_SM120_PERF=0`** turns the whole set off (the defaults every other card has). Each switch's own variable still wins
  in both directions (`=0` off, `=1` / `=2` on).
- Also on for every sm_80+ CUDA card: `STRATA_PROMPT_ATTN_IMMA`. On a discrete card of 13 GiB or less (RTX 5070 12 GB):
  `STRATA_ROUTE_PRIOR=1.5` (residency-biased routing, changes which experts run; measured on an RTX 3060 12 GB only).
- `STRATA_GDN_TC=1` (used in the RTX 30 production presets): sm_120 uses the same split shared-memory plan as sm_86
  (97,808 B within the 101,376 B per-block opt-in); it needs at least 48 SMs. Not measured on this card family.

## 3. NVFP4 (opt-in)

From [docs/NVFP4.md](../NVFP4.md): RTX 50 has FP4 tensor cores, and setup's model menu says "native FP4 on this card" for
the NVFP4 packs (on older RTX cards UD-IQ4_XS is faster). `STRATA_PREFILL_NVFP4` picks the prompt path:

| Mode | Kernel | Prefill, 4-8K prompts | First-token KL mean |
|:---|:---|---:|---:|
| `w4a8` (default) | MMQ, int8 tensor cores | 2,503 tok/s | 0.0018 |
| `w4a4` | Blackwell FP4 MMA (sm_120a only) | 2,904 tok/s | 0.0080 (top-1 7/8) |
| `w4a4x2` | Blackwell FP4 MMA | see NVFP4.md history | |
| `fp16` | dequantize + cuBLAS | 1,905 tok/s | reference |

NVFP4.md notes the hardware and version of that table were not recorded. `w4a8` stays the recommended default; a card
without the `w4a4` image falls back to `w4a8` with a note.

## 4. Numbers reported by others

Quoted from [docs/DETAILS.md](../DETAILS.md) and [docs/COMMUNITY_BENCHMARKS.md](../COMMUNITY_BENCHMARKS.md), with their
attribution. These are upstream Strata versions, not Speedy presets, and not comparable with the RTX 30 tables.

| Card | Report | Source |
|:---|:---|:---|
| RTX 5070 12 GB, Ryzen 5 7600, 64 GB DDR5-5200, Windows | Upstream's own measurement card ("Speed (measured)"). Engine 0.1.36: Q2_0 prompt 4K 1,294 -> 1,570, 32K 2,170 -> 2,653, 128K 2,123 -> 2,468 tok/s; decode 4K 89 -> 93.5, 128K 64.5 -> 76.4 tok/s | DETAILS.md |
| RTX 5080 16 GB, Linux, CUDA 13.0, i9-14900K | The author of `STRATA_PF_FUSED_Q2N=1`: with `--prefill auto` a 16K prompt 2,592 -> 3,438 tok/s (+32-33 %), mostly from the larger chunk | DETAILS.md |
| RTX 5090 / Ryzen 9 9950X | `STRATA_ADAPT_CALLBACK_WARMUP`: 142.82 -> 207.75 tok/s (2026-10-11 report) | DETAILS.md |
| RTX 5090, Core i7-13700K, 64 GB, Windows 11 | `--pool-affinity measure`: `p-cores` 188-191 tok/s vs `all` 148 tok/s (Swift 1.5 IQ3_XXS, 128K) | DETAILS.md |
| RTX 5090, Core Ultra 9 285K, 64 GB | Strata 0.1.29, IQ2_XS, 131K context, 4K / 32K / 128K runs | COMMUNITY_BENCHMARKS.md |
| RTX 5060 Ti 16 GB, WSL2, Ryzen 7 3700X, 128 GB | Strata 0.1.41, IQ3_S: decode 48.2 / 47.7 tok/s, prompt 343 / 598 tok/s at 3.8K / 19.2K (single runs) | COMMUNITY_BENCHMARKS.md |
| RTX 5060 Ti 16 GB, Xeon E5-2680 v4, Windows 10 | Strata 0.1.41, Coder IQ1_M and Q2_0 at 262K | COMMUNITY_BENCHMARKS.md |
| 4060 Ti + 5080 layer split, i9-14900KF | `--host-core last`: steady decode, 8 % faster on average | DETAILS.md |

## 5. Agent tuning guide

1. **Detect.**
   ```bash
   nvidia-smi --query-gpu=index,name,memory.total,compute_cap,pcie.link.gen.current,pcie.link.width.current --format=csv
   nvidia-smi -q | grep -i -E 'multiprocessor|driver version'
   free -g
   ```
   Confirm `compute_cap` 12.0. Note the SM count (80+ enables `STRATA_PF_FUSED` and the wide tables) and whether VRAM is
   above 13 GiB (enables `STRATA_GR_V3`, disables the route prior default).
2. **Pick a starting preset** from [`configs/cards/_template/`](../../configs/cards/_template/) (based on the RTX 3080 1x
   UD-IQ4_XS preset) and the RTX 30 presets:

   | VRAM | Start from | First changes |
   |:---|:---|:---|
   | 12 GB (5070) | `_template/1x` | as shipped; `--vram-reserve-mib` 444-700 (the engine adds 256 MiB after the cache), `--kv-resident 20480` |
   | 16 GB (5060 Ti, 5070 Ti, 5080) | `_template/1x` | `--kv-resident 24576`, try `--prefill auto:16384` |
   | 24 GB | `_template/1x` | `--kv-resident` 24576-32768; GPU picture reader becomes the default |
   | 32 GB (5090) | `_template/1x` | `--kv-resident` 32768+, `--prefill auto:16384`; try `STRATA_GDN_CHUNKED` (1.8x on the recurrence on an RTX 5090, slower on smaller cards) |
   | 2 alike cards, 16 GB+ | `rtx3080-10gb/2x` | layer split as shipped; separately test the peer tier with `STRATA_PEER_TIER2_AUTO=1` |

   All numbers in this table are starting points, not measured on this card family.
3. **Run `strata tune auto <profile-or-config.json>`** (quick about 20 minutes, `--mode thorough` up to 2 hours,
   `--use coding` for agent work).
4. **Speedy settings to try first**, one at a time, ABBA: `STRATA_GDN_TC=1`; `--prefill auto:16384`; a larger
   `--kv-resident`; `--pipeline-windows 2` on two cards; `STRATA_SM120_PERF=0` as a control (should be slower);
   `STRATA_PREFILL_NVFP4=w4a4` only with an NVFP4 pack and only if a quality check passes.
5. **Verify.** `bench/speedbench.py` cold 60K / 120K / 200K (`--k 60/120/200 --reps 2`), medium reasoning; a needle at 120K+;
   a coding task; check answers do not end early. Check the engine log for which defaults it took and any fallbacks.
6. **Report back** via a GitHub issue (or a PR adding `configs/cards/<card>/` from the template) with: card, VRAM, SM
   count, driver/CUDA, CPU/RAM, Speedy commit, preset and changes, the speedbench table, A/B pairs and quality results
   (same format as [RTX30.md section 8](RTX30.md#8-ai-agent-checklist)).
