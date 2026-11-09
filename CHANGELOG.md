# Changelog

## Speedy 0.4.0 (2026-11-09), two windows fixed, one 0..1M preset, Strata serve fixes
Measured on 2x / 1x RTX 3080 10 GB (P620, TR 5975WX, 128 GB DDR4-3200).
- **Two-window cut-off answers fixed** (`STRATA_PW_RES_PATCH`, default on with `--pipeline-windows 2`): the route
  prior read a residency table the pipelined loop did not keep current, and UD-Q4_K_XL ended short answers early (up
  to 18 of 48). Each stage now keeps a dirty list and patches its table on its own stream before every window. Both
  UD-Q4_K_XL presets are back on two windows: decode +25 % (229K) / +30 % (1M), greedy cut-offs 1 of 192.
- **One UD-IQ4_XS preset from 0 to 1M** (`STRATA_IDX_ELASTIC`, `STRATA_KV_HOST_ELASTIC`, `STRATA_ROPE_HOST`, yarn x4):
  the indexer and the host K/V copy grow with the request. Growth to 1M: 9/9 recall. With two windows a reply ends
  ("length") at prompt + 16384 tokens, the reasoning budget admission reserves.
- **UD-IQ4_XS 2x:** prompt chunk up to 16384 and 24K resident K/V: prefill +7-11 % (55K-185K), decode +3.6 %.
- **`STRATA_GDN_TC=1`** (split tensor-core GDN) on UD-IQ4_XS and UD-Q4_K_XL 229K, 2x and 1x: prefill +1-4 %.
- **Presence penalty 1.5 -> 0** on the UD presets (needle and gate tests clean, no loops).
- **One prompt-chunk size for both cards:** re-sending an 824K prompt takes 7 s instead of 329 s.
- **Route-prior NaN guard** (as Strata #5655): a non-finite router row still picks 10 real experts.
- **Strata #5643** (follow-up turns restore the reasoning a client dropped, so the cached prompt is reused): turn-2
  time to first token 1.44 s -> 0.23-0.28 s median. **#5648**: a tool call whose opener lost its `<` is recovered.
- **1x presets: `--vram-reserve-mib 444`** (was 700): the engine now reserves 256 MiB more after the expert cache;
  700 cost single-card prefill 10 % (UD-IQ4_XS) / 17 % (UD-Q4_K_XL). See configs/cards/rtx3080-10gb/CARD.md.
- **bench/speedbench.py** reads the request's own timing line, not its reasoning follow-up (longer prompt, all reused).
- 2x `strata-ud-iq4_xs-1m-2gpu.json` removed: `ud-iq4_xs-1m` is now an alias of the 0..1M preset.
- **Rejected after measurement:** CPU share of short prompt chunks, arena release, HANDOFF_PIPE, GDN_SPLIT, QFUSE,
  EXPDEC_R=2, CPU prefetch distance, q4_0 K/V for 1M, route prior weaker/off (fixes the cut-offs at -20/-38 % decode),
  16K chunks / 24K resident on one card (slower), Strata #5595.

## Speedy 0.3.0 (2026-11-01), unified single- and dual-GPU, on Strata v0.2.7.1
- **One codebase for one or two GPUs.** The single-GPU edition (1GPU 0.2.0) and the dual-GPU edition are merged. A
  preset with `--layer-split` runs two cards; without it the engine runs on one. Production presets for both are in
  `configs/production-1gpu/` and `configs/production-2gpu/`, including the 1M tier on two GPUs.
- **#3559 stage-local HC_Q8 upload** (havok2-htwo): on a trimmed split each card uploads only its own Q8_0 rows.
  This gives UD-IQ4_XS and UD-Q4_K_XL +7-11 % expert slots per card (292 / 345 MiB saved).
- **`STRATA_PREFILL_STREAM_MIN=512`** on UD-IQ4_XS (two GPUs): follow-up turns of 730 tokens take 2.15 -> 1.86 s, and 958 tokens 1.94 -> 1.77 s.
- **`"parallel": 2` presets** in `configs/production-2gpu/parallel2/`. These are measured, not the default: slower on
  2x 10 GB (two clients 42-48 s against 33 s queued; single-user decode -15-19 %).
- **Evals:** `evals/` has the IFBench and LiveCodeBench harness. UD-Q4_K_XL scores 81.0 on IFBench (card BF16: 81.3).
- **Releases and updates** now point at DevDock-AI/Speedy.
- **Rejected after measurement:** `STRATA_MMQ_DIRECT` (neutral), #2427 and split QSA attention (tolerance gate),
  `EXP_MODE=8`, `HELP_FRAC 0.45`, `NT=256`, Talus union-KV, StrataFast-style prompt split.

## Speedy 2GPU 0.1.0 (2026-10-31), dual-GPU edition, on Speedy 0.1.0 / Strata v0.2.7.1
Two RTX 3080 10 GB; numbers in README and docs/EDITION-2GPU.md.
- **Layer split with per-card weight trim** (from Hardin22's Strata-DualGPU, ported by cobra91): each card keeps only
  its own layers' dense weights; `--layer-split 26 --trim-stage-weights --vram-reserve-later-mib 500`. UD-IQ4_XS
  prefill +78-89 %, UD-Q4_K_XL +163-185 %, IQ3_S +76-89 % vs one card. Trim is opt-in (`STRATA_STAGE_TRIM=1`).
- **Decode overlap** `--pipeline-windows 2` + **`STRATA_PIPELINE_PENALTIES=1`** (Speedy, opt-in): penalised requests
  (presence penalty) run pipelined with the exact per-row history uploaded before each last-stage launch. Production
  sampling, same binary: UD-IQ4_XS decode +8-9 %, Q2_0 +14-19 %, IQ3_S +12-28 %, UD-Q4_K_XL +10-25 %. Tolerance
  gate (serial-vs-overlap logprobs, both baselines), continuation and coding gates pass.
- `STRATA_PREFILL_HELP=1` in all 2-GPU presets (Q2_0 warm turn 1.45-2.0 -> 1.42-1.45 s); `STRATA_SPLIT_MTP_BATCH=1`
  on UD-IQ4_XS (decode +2-9 %).
- J01 fix: the prefill pinned share is set before the layer-split startability probe (it read the 1.0 default).
- Upstream PRs (default off unless noted): #3821 `STRATA_PREFILL_PIPE_FLOOR` (ruibeikaa), #3819
  `STRATA_SPLIT_COMMIT_ASYNC` (ruibeikaa), #4020 peer-device VMM/context fixes (chimpera), #3648 `STRATA_PF_PEER_PIECES`
  (Klaus Friedel), #3742 per-GPU copy threads, `STRATA_STAGE_COPY_PARALLEL=1` (wannagoxuf), #4380 fused-ring check per
  device (moses-sun, on), #4549 draft-head room on the last card and #4534 slower-stage split pricing (cobra91, from
  Strata-DualGPU), #3306 `STRATA_PREFILL_MAX` (gopinath87607).
- Counters: pipeline survival, per-request pipeline eligibility, helper occupancy (`STRATA_DECODE_TIMING=1`).
- Debug: `STRATA_LM_DEBUG_SYNC=1` names the stage/layer of a layer-major prefill CUDA fault.
- Note: `serve/server.py` ignores the request field `cache_prompt` (upstream behaviour): a request with
  `cache_prompt: false` can still be restored from the conversation cache / prompt checkpoints. For repeatability
  tests disable them in the engine config (`--conversation-cache-mib 0`, `--prompt-cache-every 0`).
- Tested, not adopted: expert helper / peer tier instead of the split, `STRATA_SPLIT_OWN=auto`, split K 24/28,
  GPU2 at 320 W (+1 %), `STRATA_SPLIT_COMMIT_ASYNC`, prefill handoff pieces, admission theta other than 0.2.
- Configs use placeholders for paths and GPU UUIDs (docs/CONFIG-PLACEHOLDERS.md).

## Unreleased (2026-10-30), on Strata v0.2.7.1
- **K1 `STRATA_KV_STAGE_REQ=<cells>`** (Speedy, opt-in): the streamed-K/V staging pool is sized by the request
  (max of `<cells>` and the request end) instead of `--max-context`; re-laid out when a request needs more. At 1M
  context (Q2_0, YaRN x4) the auto prompt chunk goes 512 -> 4608 and a 300K cold prompt 466 -> 1527 t/s (3.3x).
  **On in the `speedy-q2_0-1m` preset** (131072).
- #4119 (Stefano Sinden) `STRATA_SELECT_TWO_PASS`: two-pass TF32 QSA block select, same selection as 3xTF32.
  UD-IQ4_XS ABBA prefill +0.2/+1-2.5/+1.6 % at 60/120/200K, decode unchanged. **On in the UD-IQ4_XS preset.**
- `STRATA_HC_Q8=1` now **on in every preset** (was UD-IQ4_XS/UD-Q4_K_XL only). 60K ABBA: Q2_0 prefill 2851 -> 2831 t/s
  (-0.7 %), decode 143.2 -> 151.1 (+5.5 %); IQ3_S prefill 2269 -> 2353 (+3.7 %), decode 112.5 -> 119.3 (+6.0 %).
- Tested, not ported: #2964 adaptive fetch, #4567 attention chunks, #3919 KV ring (no gain on the reference machine).

All notable changes to Speedy. Speedy versions are independent of Strata's; each entry names its Strata base.
Numbers are from the reference machine (RTX 3080 10 GB LHR unlocked, Threadripper PRO 5975WX, 128 GB); method in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Speedy 0.1.0 (2026-10-30), on Strata v0.2.7.1
First public release.

### Base
- **Rebased onto Strata v0.2.7.1** (upstream 46c152a1) on 2026-10-30. On the reference machine at 60/120/200K:
  Q2_0 decode +14/+22/+26 % and UD-IQ4_XS decode +12-19 % over the 0.1.45 base; prefill -1.5 % (Q2_0) and -7 to -8 %
  (UD-IQ4_XS) before #4039, which more than recovers it. Short quality gate passed on both. The previous base is tagged
  `speedy-on-0.1.45`.

### Upstream pull requests (cherry-picked, opt-in switches; adopted as defaults where noted)
- #2689/#2690 (lask3802) `STRATA_VERIFY_DEFER_CPU`, `_BELOW`: re-ported to 0.2.7's cold-batch path. **Default on.**
- #4039 (merbanan) `STRATA_PF_GROUP_SCRATCH`, `STRATA_PF_HC_REGION`: smaller prompt scratch, bigger auto chunks; sizes
  fixed for 0.2.7's `STRATA_MMQ_WS` tables. ABBA prefill UD-IQ4_XS +29-33 %, IQ3_S +22-24 %, Q2_0 +5 %; bitwise per the PR's parity check; re-verified by the short quality gate.
  **Default on** (2026-10-30).
- #4283 (Stefano Sinden) `STRATA_PREFILL_ROUTED_STREAM`: routed prompt stream. UD-IQ4_XS +8-10 % alone, +41-44 %
  with #4039. **On in the UD-IQ4_XS preset** (2026-10-30).

### Defaults (src/program/speedy_defaults.hpp)
- Production environment defaults built in; `STRATA_UPSTREAM_DEFAULTS=1` restores upstream (2026-10-29).
- `STRATA_PROMPT_ATTN_IMMA=1` (32K prefill +2.0 %, decode +2.1 %; suite and needle 3/3) and `STRATA_KV_PREFETCH=1`
  (+1.4-1.5 % at 32-128K) (2026-10-29).
- `STRATA_PF_GROUP_SCRATCH=1`, `STRATA_PF_HC_REGION=1` (#4039) (2026-10-30).

### Presets
- `configs/make_configs.py` and `configs/presets/` for Q2_0 (med/low), IQ3_S, UD-IQ4_XS (new) and UD-Q4_K_XL, with
  `/opt` placeholder paths (2026-10-29/30).
- `--resident-budget-gib 80` for both Q4-class packs (decode +25 % over 60) (2026-10-29).
- 229,376-token context validated on every pack; 1M (YaRN x4) overflow tier for Q2_0 (2026-10-30).

### Fixes and robustness
- J13: the pin-by-need resident estimate is priced with `ExpertCache::slot_stride` (guard included);
  `slot_budget_test` (2026-10-29).
- C4: opt-in PSI-driven shedding of parked conversations (`STRATA_PSI_SHED`); `memory_pressure_test` (2026-10-29).

### Tools and harness
- `bench/speedbench.py`: exact token sizing, warm-suffix mode, provenance JSON per run (2026-10-29).
- `bench/suffix.py`: agent-faithful warm turns with a 90 % reuse gate (2026-10-29).
- `tools/codemode_mcp.py`: Code Mode MCP server with tests (2026-10-29).

### Dropped or rejected
- C3 pinned bounce buffer (upstream `STRATA_CKPT_ASYNC` covers it; no gain measured).
- J02 per-expert pool readiness (branch `exp/J02-pool-dep`): +0.8 %, inside noise.
- Full list with numbers: [docs/TUNING-HISTORY.md](docs/TUNING-HISTORY.md).

### Project
- Public layout: README with logo, NOTICE, licenses/, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, CITATION.cff, docs
  (BUILD-LINUX, BENCHMARKS, CONFIGS, ARCHITECTURE, GPU-UNLOCK, ROADMAP, TUNING-HISTORY, SPEEDY_DEFAULTS). Strata's
  READMEs moved to docs/upstream/. Private hostnames and paths replaced with placeholders.
- Published history: the Strata v0.2.7.1 import, then one commit for each cherry-picked upstream PR under its original
  author, then one "Speedy" commit.
- `.github/`: Speedy CI (CPU build + CPU tests; CUDA build on manual dispatch) and issue templates; Strata's bots,
  templates and FUNDING.yml removed (kept for reference in docs/upstream/github/).

## Pre-release: Speedy on Strata v0.1.45 (2026-10-29, tag `speedy-on-0.1.45`)
- Strata v0.1.45 + patch 0001 (PR #2689/#2690); production env defaults; presets; Linux build docs; serving
  examples; J13; C3 (opt-in); C4 (opt-in); Code Mode tool; IMMA and KV prefetch defaults; harness.
