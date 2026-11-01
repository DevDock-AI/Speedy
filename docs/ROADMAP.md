# Roadmap

Gains named below are claims of the PR authors or estimates, not measurements, unless marked "measured". Every item
goes through the A/B and quality rules in [CONTRIBUTING.md](../CONTRIBUTING.md).

## Phase 1: single GPU (RTX 3080 10 GB) - open items
- **Upstream PRs on the integration branch** (opt-in, ported, compile-checked; A/B pending): #4119 two-pass QSA select
  (author: +5 % at 256K on a 3090), #4567 32-cell QSA decode chunks, #3919 KV ring from depth, #2968 arena staging ring,
  #2964 adaptive fetch, #3310 `STRATA_PREFILL_KEEP`, #3309 warm-up prompt, #4226 persistent disk conversation cache.
- **Not ported (reason)**: #3307 drafter memory lent to long prompts (its generate.cpp changes overlap 0.2.7's
  expert-nibble and prompt-plan code; a mechanical merge duplicated that code: needs a port by hand); #4224 parked-cache
  dedup, #3711 and #3980 conversation-buffer changes (written against the pre-0.2.7 `ConversationBuffer`, which 0.2.7
  rewrote; RAM is not tight on 128 GB except with UD-Q4_K_XL); #2443 redundant-holder eviction (conflicts with #4226 in
  `conversation_cache.hpp`; #4226 kept); #1937 disk-parked conversations (superseded by #4226, the fuller design; #1937
  reports a 165K conversation re-read in 8.4 s instead of 73 s).
- **1M tier prompt reads**: with `--max-context 1048576` the auto prompt chunk falls to 512 tokens (6400 at 229K), so a
  300K-600K cold read runs at 405-449 t/s. The prompt path stages one layer's whole streamed K/V (every page of the
  maximum context, ~1 GB at 1M) out of the slots it borrows; sizing that stage by the request instead is in progress.
- **Prompt-stream threshold per pack** on the 0.2.7.1 base (`STRATA_PREFILL_STREAM_MIN`; on 0.1.45, 8192 looked
  +5.8 % cold for UD-Q4_K_XL, not ABBA).
- **UD-Q4_K_XL** re-measurement on Speedy 0.1.0 (only 0.1.45 numbers so far).
- **Lossy candidates, only behind the quality gate**: #3545 verify expert cap.
- Deferred: S08 F16 WMMA expert kernels (at most +4 % on IQ3_S), J02 arm 3 barrier removal (its premise failed).

## Phase 2: two GPUs (RTX 3080 10 GB + RTX 3080 20 GB) - needs the second card
Rules: one variable per run, 30-45 minutes per candidate, ABBA against the best single-card preset on the same
binary, 60K / 120K / 200K plus a warm turn, the short quality gate for anything that changes arithmetic.

1. **Install and validate the card** (one boot): power budget (both cards at 260 W at first), slot by cooling (both
   x16 slots are CPU-attached), the 10 GB card keeps the console so all 20 GB are usable; ReBAR on both. Per-card
   GEMM, a full-VRAM memory test and a single-card Q2_0 run on the new card alone; 10 minutes of load, Xid check.
2. **Unlock the second card** (optional, [GPU-UNLOCK.md](GPU-UNLOCK.md)): a dry run first if the board is supported,
   then add its PCI address to the module's targets; compare GEMM and checksums.
3. **Peer and BAR probes**: `nvidia-smi topo -m`, P2P bandwidth tests. GeForce cards are expected to have no P2P, so
   the host route is the baseline; a P2P driver fork is a separate, later candidate and is never stacked with the
   unlock untested.
4. **Modes**, each against single-card:
   - **A, layer split** (`--layer-split auto`, then explicit boundaries and card order). Fix the stale
     `pinned_share()` reads in the split planner first. Then #4549 (room for the draft head on the last card, a bug
     fix), #4522 (trim dense weights on auto) and #4534 (price the split by its slower stage); authors claim up to
     +70 % decode combined. `--pipeline-windows 2`, #3821 `STRATA_PREFILL_PIPE_FLOOR`, #3819 split async commit.
   - **B, helper expert cache** (`--expert-cache-device1`, `--remote-expert-opt`).
   - **C, peer tier** (`--peer-device`, host route): #3648 `STRATA_PF_PEER_PIECES`, #2721 `STRATA_PF_PEER_CPU`,
     #2720 adaptive overlap, #4020 elastic KV on the peer.
   - Unequal cards: #3306 per-GPU prompt chunk (`STRATA_PREFILL_MAX`), #4380 fused-ring check per device, #3742 /
     #4002 parallel per-stage park.
   - StrataFast-style second-GPU prompt split, only if its single-card fit check passes.
   - Prompt-time layer pipeline with decode expert tiers (Basalt's multi-GPU plan): an engineering item, only if
     the modes above leave the decode GPU wait above 15 %.
   - Power asymmetry (320 / 320 vs 260 / 260 vs 320 / 220 W) and "spend the new memory deliberately": the extra VRAM
     among expert slots, KV resident cells, prompt chunk and draft, one at a time.
5. **Quality packs with 30 GB of VRAM**: UD-IQ4_XS and UD-Q4_K_XL at 229K (bigger chunks, more slots).
6. **Decision**: Mode A if it beats single-card on cold 60/120/200K, decode at 120K and the warm turn with output
   parity; otherwise B for decode, then C; if nothing wins by 5 % on the 120-200K quality workload, the default is the
   20 GB card alone.

## 1M-token tier
Today only Q2_0 starts at 1M on 10 GB (measured: 302K at 449 t/s, 600K at 405 t/s, decode 105 t/s). With 30 GB, the
1M tier is expected to fit Q2_0, IQ3_S and UD-IQ4_XS. UD-Q4_K_XL at 1M needs more RAM than 128 GB with its full resident
budget (80 GiB experts + 12 GiB pinned KV + parking), so it is recorded as a capacity limit unless the split keeps
enough experts in VRAM. UD-Q5_K_XL (147 GiB) needs 256 GB of RAM to avoid SSD streaming.
