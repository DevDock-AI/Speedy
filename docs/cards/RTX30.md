# RTX 30 (Ampere, sm_86): tuning guide

Status: **production**. This is the card family Speedy is developed and measured on. Every number in this page comes
from the reference machine unless it says otherwise:

| Part | Reference machine |
|:---|:---|
| GPUs | 1x or 2x NVIDIA GeForce RTX 3080 10 GB (LHR), sm_86, gpu-unlock on, 320 W (GPU1) / 280 W (GPU2) |
| PCIe | 4.0 x16 per card, no NVLink / P2P (GeForce cards use the host route) |
| Workstation / CPU | Lenovo P620, AMD Threadripper PRO 5975WX (32 cores / 64 threads, AVX2) |
| RAM | 128 GB DDR4-3200, 8 channels |
| Storage | 2 TB NVMe PCIe 4.0 |
| OS / driver / CUDA | Debian 13 (kernel 6.12) / 615.71 / 13.4 |
| Model | Qwen3.8-Flash-Next (125B MoE, routed experts in system RAM, a cache of them in VRAM) |

Sources: [configs/cards/rtx3080-10gb/CARD.md](../../configs/cards/rtx3080-10gb/CARD.md), [CHANGELOG.md](../../CHANGELOG.md),
[docs/TUNING-HISTORY.md](../TUNING-HISTORY.md), [docs/EDITION-2GPU.md](../EDITION-2GPU.md), [docs/QUANTS.md](../QUANTS.md).

## 1. Engine and build

- Release asset: the **rtx30** engine (`strata-linux-x64-rtx30.zip` / `strata-windows-x64-rtx30.zip`). `setup.py` maps
  compute capability 86 to `rtx30` (`PREBUILT_FAMILY`), and the release workflow builds it with `ARCHS=86`.
- From source: `-DCMAKE_CUDA_ARCHITECTURES=86`.
- Windows on RTX 30 is built but was not run by the maintainers (docs/release-notes-0.2.4.md).
- The engine turns on an sm_86 default set by itself when every visible card is 8.6 (docs/DETAILS.md):
  `STRATA_ADAPT_NOWAIT=1`, `STRATA_ADAPT_LAG=2`, `STRATA_CPU_IQ_ALL_EXACT1=1`, `STRATA_SIDE_MIXER=2`, `STRATA_ATTN_DAILY=1`,
  `STRATA_GR_UP75=1`, `STRATA_GU_SWIGLU=1`, `STRATA_QSA_ROWS=1`, `STRATA_B6_MMVQ_ROWS=1`. `STRATA_SM86_PERF=0` turns the set off.
- Other defaults that apply to RTX 30 (include/strata/core/sm120_series.hpp): `STRATA_PROMPT_ATTN_IMMA` (int8 tensor-core
  prompt attention) is on for every CUDA card of compute capability 8.0 or newer; `STRATA_ROUTE_PRIOR` defaults to 1.5 on a
  discrete CUDA card of 13 GiB or less (a 10 GB or 12 GB card), 0 on larger cards; `STRATA_PREFILL_LAYER_MAJOR` is on for a
  CUDA layer split of 2+ cards.

## 2. Which preset to use

| Setup | Folder | Use |
|:---|:---|:---|
| 1x RTX 3080 10 GB | [`configs/cards/rtx3080-10gb/1x/`](../../configs/cards/rtx3080-10gb/1x/) | `strata-final-reason-med.json` / `-low.json` (Q2_0), `strata-iq3_s.json`, `strata-ud-iq4_xs.json`, `strata-ud-q4_k_xl.json` at 229,376 tokens; `strata-q2_0-1m.json` (1M, YaRN x4); `llama-swap.yaml` |
| 2x RTX 3080 10 GB | [`configs/cards/rtx3080-10gb/2x/`](../../configs/cards/rtx3080-10gb/2x/) | `strata-<quant>-2gpu-pw2.json` (229K) and `strata-<quant>-1m-2gpu.json` (1M). UD-IQ4_XS is one preset from 0 to 1M (`strata-ud-iq4_xs-2gpu-pw2.json`; `ud-iq4_xs-1m` is an alias of it in `llama-swap.yaml`) |
| 2x, two parallel requests | [`2x/parallel2/`](../../configs/cards/rtx3080-10gb/2x/parallel2/) | Reference only: measured slower on 2x 10 GB (two clients 42-48 s against 33 s queued; single-user decode -15-19 %) |

Paths and GPU UUIDs are placeholders (docs/CONFIG-PLACEHOLDERS.md). For the 2x presets, fill `<GPU1_UUID>,<GPU2_UUID>`
(`CUDA_DEVICE_ORDER=PCI_BUS_ID`); the first card listed is the main card.

Quant choice, all on the same machine: Q2_0 is the fastest, UD-Q4_K_XL the slowest and largest; UD-IQ4_XS is the
benchmark and default quant of the maintainers. Routed-expert sizes from docs/QUANTS.md: UD-IQ4_XS 55.4 GiB,
UD-Q4_K_XL 71.7 GiB.

### Settings at a glance (as shipped)

| Preset | Presence penalty | `STRATA_GDN_TC` | `--prefill` | `--kv-resident` | `--vram-reserve-mib` | `--layer-split` |
|:---|---:|:---:|:---|---:|---:|:---|
| 1x Q2_0 med / low | 1.5 | - | auto | 20480 | 444 | - |
| 1x IQ3_S | 0 | 1 | auto | 20480 | 444 | - |
| 1x UD-IQ4_XS, UD-Q4_K_XL | 0 | 1 | auto | 20480 | 444 | - |
| 1x Q2_0 1M | 0 | 1 | auto | 20480 | 444 | - |
| 2x Q2_0 229K | 1.5 | 1 | auto | 20480 | 700 | auto |
| 2x Q2_0 1M | 1.5 | 0 | auto | 20480 | 700 | auto |
| 2x IQ3_S 229K / 1M | 1.5 | 0 | auto | 20480 | 700 | 26 |
| 2x UD-IQ4_XS 0..1M | 0 | 1 | auto:16384 | 24576 | 700 | 26 |
| 2x UD-Q4_K_XL 229K | 0 | 1 | auto:16384 | 24576 | 700 | 26 |
| 2x UD-Q4_K_XL 1M | 0 | 0 | auto:16384 | 24576 | 700 | 26 |

All presets share `--kv int8`, `--spec 4 --spec-min-p 0.5` with the MTP head, `--expert-cache auto`,
`--conversation-cache-mib 8192`, `--adapt-async 1 --adapt-swaps 80`, `--pool-affinity measure`, and the UD presets
`--resident-budget-gib 80` (76 on 2x UD-Q4_K_XL 1M). All 2x presets add `--trim-stage-weights --vram-reserve-later-mib 500
--pipeline-windows 2`.

## 3. Why each key setting is there

| Setting | Why | Source |
|:---|:---|:---|
| `--vram-reserve-mib 444` (1x) vs `700` (2x) | 700 was the lowest reserve that passed a 127K-token needle on 10 GB (400-450 ran out of memory). Since 0.4.0 the engine reserves **another 256 MiB after the expert cache**. On one card 700 + 256 lost ~110 expert slots and the automatic prompt chunk dropped (UD-IQ4_XS 5632 -> 4608, UD-Q4_K_XL 3840 -> 3072): prefill -10 % / -17 %. 444 + 256 = the old 700, and speed is back to the 1GPU 0.2.0 level (UD-IQ4_XS 1805-1810 t/s, UD-Q4_K_XL 1136 at 57K, ABBA). The 2x presets keep 700: each card holds only its own layers there. | CARD.md, CHANGELOG 0.4.0, TUNING-HISTORY |
| `--layer-split 26 --trim-stage-weights --vram-reserve-later-mib 500` (2x) | The model is split by layers; the second card starts at layer 26. Trim makes each card keep only its own layers' dense weights, and the freed VRAM goes to its expert cache. vs one card: UD-IQ4_XS prefill +78-89 %, UD-Q4_K_XL +163-185 %, IQ3_S +76-89 %. Split K 24/28 were tested and not adopted. Q2_0 uses `auto`. | CHANGELOG 2GPU 0.1.0 |
| `--pipeline-windows 2` (2x) | Decode overlap across the two cards. With `STRATA_PIPELINE_PENALTIES=1` penalised requests run pipelined too: UD-IQ4_XS decode +8-9 %, Q2_0 +14-19 %, IQ3_S +12-28 %, UD-Q4_K_XL +10-25 %. | CHANGELOG 2GPU 0.1.0 |
| `STRATA_PW_RES_PATCH` (default on with two windows) | The route prior read a residency table that the pipelined loop did not keep current, and UD-Q4_K_XL ended short answers early (up to 18 of 48). Each stage now keeps a dirty list and patches its table on its own stream before every window. Both UD-Q4_K_XL presets went back to two windows: decode +25 % (229K) / +30 % (1M), greedy cut-offs 1 of 192. Nothing to set. | CHANGELOG 0.4.0 |
| `STRATA_GDN_TC=1` | Split tensor-core GDN (DeltaNet) recurrence in the prompt path: prefill +1-4 % on UD-IQ4_XS and UD-Q4_K_XL 229K, 1x and 2x. sm_86 uses the "split" shared-memory plan (97,808 B, under the 101,376 B per-block opt-in of sm_86 / sm_89 / sm_120); the full plan needs 124,944 B. The kernel needs sm_80+, at least 48 SMs and that shared memory; otherwise the engine silently keeps the older kernel. | CHANGELOG 0.4.0, src/prefill/gdn_tc.cuh, src/prefill/kernels.cu |
| `--prefill auto:16384` + `--kv-resident 24576` (2x UD only) | Caps the prompt chunk at 16384: prefill 55K +7 %, 110K +9.5 %, 185K +11 % (12288 and 20480 were slower). 24K resident K/V: decode +3.6 % (20K-110K), same VRAM peak. 3 A/B pairs + quality gate 8/8 + 4/4. **On one card both were slower** and were rejected, so the 1x presets keep `auto` / 20480. | CARD.md, CHANGELOG 0.4.0 |
| Presence penalty 0 (UD presets) | 1.5 -> 0 in 0.4.0: needle and gate tests clean, no loops. Q2_0 and the IQ3_S 2x presets still use 1.5. | CHANGELOG 0.4.0 |
| `--resident-budget-gib 80` (UD) | Decode +25 % over 60. | CHANGELOG 0.1.0, TUNING-HISTORY |
| `STRATA_HC_Q8=1` | Hyper-connection projections in Q8: Q2_0 decode +5.5 %, IQ3_S prefill +3.7 % / decode +6.0 % (60K ABBA). | CHANGELOG Unreleased 2026-10-30 |
| `STRATA_SELECT_TWO_PASS=1` | Two-pass TF32 QSA block select, same selection: UD-IQ4_XS prefill +0.2 / +1-2.5 / +1.6 % at 60/120/200K. | CHANGELOG |
| `STRATA_PREFILL_CPU_SHARE=0` | GPU + PCIe beats a CPU share of prefill on this machine (IQ3_S keeps `auto`). | TUNING-HISTORY |
| `STRATA_ROUTE_TAIL_SKIP=0` | The route-tail skip is lossy: off for quality. | TUNING-HISTORY |
| `STRATA_PREFILL_STREAM_MIN=512` (2x UD-IQ4_XS) | Follow-up turns of 730 tokens 2.15 -> 1.86 s, 958 tokens 1.94 -> 1.77 s. | CHANGELOG 0.3.0 |
| `STRATA_IDX_ELASTIC`, `STRATA_ROPE_HOST`, YaRN x4 (2x UD-IQ4_XS) | The indexer and host K/V copy grow with the request, so one preset covers 0 to 1M. Growth to 1M: 9/9 recall. With two windows a reply ends at prompt + 16384 tokens. | CHANGELOG 0.4.0 |
| CPU vision reader | Setup and the app pick the CPU picture reader under 24 GB of VRAM and the GPU reader from 24 GB (`--vision cpu` / `--vision gpu`). The GPU encoder takes about 1.1 GiB of VRAM, which a 10 GB card needs for experts. In the maintainers' log the CPU reader read a 640x480 picture into 300 tokens in 0.60 s at 32 threads, and `--vision` adds only an M-RoPE table in VRAM (~12.6 MB per card at 1M context). The presets in this release carry no `vision` section; add one if you need pictures. | docs/NVFP4.md (Images), maintainers' measurement log |

## 4. Measured numbers

Method: `bench/speedbench.py`, unique cold prompts at about 60K / 120K / 200K tokens, medium reasoning; tok/s, average ±
half spread (CARD.md).

**2x RTX 3080 10 GB** (CARD.md, Speedy 0.4.0 presets)

| Quant | Prefill 60K | Prefill 120K | Prefill 200K | Decode | Cold 200K (s) | Follow-up (s) |
|:---|---:|---:|---:|---:|---:|---:|
| Q2_0 | 4,151 | 4,270 | 4,138 | 187 ±24 | 49 | 1.8 ±0.2 |
| IQ3_S | 4,074 | 4,284 | 4,264 | 163 ±8 | 47 | 1.6 |
| UD-IQ4_XS | 3,297 | 3,400 | 3,397 | 141 ±14 | 59 | 2.05 ±0.25 |
| UD-Q4_K_XL | 2,994 | 3,118 | 3,115 | 111 ±10 | 65 | 2.75 ±0.05 |

**1x RTX 3080 10 GB**, CARD.md table (1GPU 0.2.0 engine, 2026-10-31; 0.4.0 with `--vram-reserve-mib 444` reaches the same level)

| Quant | Prefill 60K | Prefill 120K | Prefill 200K | Decode | Cold 200K (s) | Follow-up (s) |
|:---|---:|---:|---:|---:|---:|---:|
| Q2_0 | 2,824 | 2,689 | 2,562 | 152 ±13 | 78 | 1.9 ±0.4 |
| IQ3_S | 2,347 | 2,303 | 2,248 | 130 ±11 | 89 | 1.75 ±0.05 |
| UD-IQ4_XS | 1,820 | 1,802 | 1,806 | 107 ±15 | 111 | 2.75 ±0.25 |
| UD-Q4_K_XL | 1,134 | 1,125 | 1,090 | 93 ±17 | 183 | 2.9 |

**1x, Speedy 0.4.0 release presets** (`--vram-reserve-mib 444`), prefill tok/s

| Quant | 120K | 200K |
|:---|---:|---:|
| Q2_0 | 2,686 | 2,548 |
| IQ3_S | 2,284 | 2,227 |
| UD-IQ4_XS | 1,808 | 1,792 |
| UD-Q4_K_XL | 1,123 | 1,089 |

Other 0.4.0 results: turn-2 time to first token 1.44 s -> 0.23-0.28 s median (Strata #5643); re-sending an 824K prompt
7 s instead of 329 s (one prompt-chunk size for both cards).

## 5. RAM

- The routed experts live in system RAM. Sizes (docs/QUANTS.md): UD-IQ4_XS **55.4 GiB** of routed experts (93.7 GB
  download), UD-Q4_K_XL 71.7 GiB (111.3 GB). The PLE table is another 26.8 GiB (IQ4_NL) in both; the dense side the GPU
  reads per verify window is about 4.8 GB.
- The reference machine has 128 GB, and the UD presets ask for `--resident-budget-gib 80`. With less RAM, lower the
  budget (decode drops: 60 GiB was 25 % slower than 80 on UD) or use a smaller quant. Below about 96 GB with UD-IQ4_XS:
  not measured on this card family.
- All 8 memory channels matter: adding the 8th DIMM gave decode +22-28 % (TUNING-HISTORY). Populate every channel.

## 6. Other RTX 30 cards

None of these were measured by the maintainers. Start from the 10 GB presets and change one thing at a time.

| Card | Start from | What to change |
|:---|:---|:---|
| RTX 3090 / 3090 Ti 24 GB | `1x/` presets (or `2x/` for two cards) | `--expert-cache auto` grows on its own. Raise `--kv-resident` (try 24576, then 32768) and, on two cards, try `--prefill auto:16384` on every quant. `STRATA_ROUTE_PRIOR` defaults to 0 above 13 GiB. With 24 GB the GPU picture reader becomes the setup default. |
| RTX 3080 Ti / 3080 12 GB / 3060 12 GB | `1x/` presets | Raise `--kv-resident` modestly. Keep the reserve rule: `--vram-reserve-mib` plus the engine's 256 MiB post-cache reserve should equal what passes a long needle (444 on 10 GB); verify with a 120K+ needle before lowering. These cards still count as small (≤ 13 GiB): route prior 1.5 by default. |
| RTX 3060 12 GB, 3060 Ti, 3070 | as above | `STRATA_GDN_TC` needs at least 48 SMs; check `nvidia-smi -q` / `deviceQuery` for the SM count. On a card with fewer SMs it falls back to the older kernel (no error, no gain). `STRATA_EXCHANGE_ROTATE` never reached READY on an RTX 3060 (DETAILS.md). |
| RTX 3050 / 8 GB cards | `_template` | 8 GB cards start; a card with 8 GB or less starts as "Off" in multi-GPU role selection. Not measured on this card family. |

### Mixing two different RTX 30 cards in a layer split (unmeasured)

The 2x presets assume two equal 10 GB cards. For a mixed pair (for example 3090 + 3080):

1. Put the card with more VRAM first in `CUDA_VISIBLE_DEVICES`: the main (last-stage) card holds the head, the draft layer
   and the final check of every window.
2. Start with `--layer-split auto` (it prices the slower stage, #4534), then try explicit values around it. `26` is tuned
   for 10 GB + 10 GB only.
3. Keep `--vram-reserve-mib 700` / `--vram-reserve-later-mib 500` until a long needle passes, then lower in small steps.
4. `--pipeline-windows 2` should still apply; measure with and without.
5. Both cards are sm_86, so the sm_86 default set stays on (it is used when every visible card has the same compute capability).

None of this is measured on this card family. Report results (section 9).

## 7. Tested and rejected (do not re-propose without a new reason)

| Version | Rejected |
|:---|:---|
| 0.4.0 | CPU share of short prompt chunks, arena release, HANDOFF_PIPE, GDN_SPLIT, QFUSE, EXPDEC_R=2, CPU prefetch distance, q4_0 K/V for 1M, route prior weaker/off (fixes the cut-offs at -20/-38 % decode), 16K chunks / 24K resident on one card (slower), Strata #5595 |
| 0.3.0 | `STRATA_MMQ_DIRECT` (neutral), #2427 and split QSA attention (tolerance gate), `EXP_MODE=8`, `HELP_FRAC 0.45`, `NT=256`, Talus union-KV, StrataFast-style prompt split |
| 2GPU 0.1.0 | Expert helper / peer tier instead of the split, `STRATA_SPLIT_OWN=auto`, split K 24/28, GPU2 at 320 W (+1 %), `STRATA_SPLIT_COMMIT_ASYNC`, prefill handoff pieces, admission theta other than 0.2 |
| 0.1.0 and earlier | C3 pinned bounce buffer, J02 per-expert pool readiness (+0.8 %, noise), #2964 adaptive fetch, #4567 attention chunks, #3919 KV ring |
| Platform / engine (TUNING-HISTORY) | `-march=znver3`+LTO, governor, GPU clock lock, hugetlbfs, mimalloc/jemalloc, SMT off, C2 off, vram-reserve < 700 (before the 256 MiB post-cache reserve), `--ple-io ram`; `native_mma` int8 decode, low-sync handoff, arena placement, SLRU flags, prefill head alias (-28 %), route prior / tail skip (lossy), spec 3/5/6, no MTP |

## 8. AI agent checklist

1. **Detect the hardware.**
   ```bash
   nvidia-smi --query-gpu=index,name,memory.total,compute_cap,pcie.link.gen.current,pcie.link.width.current,power.limit --format=csv
   free -g
   lscpu | grep -E 'Model name|^CPU\(s\)|Flags' | cut -c1-200
   ```
   Confirm compute capability 8.6, VRAM per card, total RAM, memory channels if known.
2. **Pick a preset.** 10 GB cards: `configs/cards/rtx3080-10gb/1x/` or `2x/` as is. Other RTX 30 cards: the same presets,
   adjusted per section 6. Copy to `configs/mine/` and fill the placeholders (docs/CONFIG-PLACEHOLDERS.md).
3. **Tune.** Where the Terminal is installed, run `strata tune auto <profile-or-config.json>` (quick: about 20 minutes;
   `--mode thorough` up to 2 hours; `--use coding` also judges long prompts). It only keeps gains above noise and saves a
   preset; nothing changes if the current settings are already fastest. Do not use `--allow-small-differences` or
   `--allow-not-repeatable` without the user's consent.
4. **Benchmark** with the same build, config and environment except the one variable:
   ```bash
   python3 bench/speedbench.py <label>-k60  <engine.log> --k 60  --reps 2 --tokenizer <pack>/tokenizer --config <cfg.json>
   python3 bench/speedbench.py <label>-k120 <engine.log> --k 120 --reps 2 --tokenizer <pack>/tokenizer --config <cfg.json>
   python3 bench/speedbench.py <label>-k200 <engine.log> --k 200 --reps 2 --tokenizer <pack>/tokenizer --config <cfg.json>
   ```
   Medium reasoning. Interleave A/B runs (ABBA) in the same window.
5. **Quality check.** A needle at 120K+, a short coding task, and a check that answers do not end early. Any setting that
   changes output must pass these before adoption.
6. **Report** in this format (also the format for `configs/cards/<card>/CARD.md`, see `configs/cards/_template/`):

   ```text
   Card(s): <model, VRAM, sm_xx, power limit, PCIe gen/width>   CPU / RAM: <model, GB, type/speed, channels>
   OS / driver / CUDA: <...>   Speedy commit: <...>   Preset: <path>, changes: <flags>
   | Quant | Prefill 60K | Prefill 120K | Prefill 200K | Decode | Cold 200K (s) | Follow-up (s) |
   A/B: <setting> before -> after, <n> pairs; quality: <needle/gate results>
   ```
