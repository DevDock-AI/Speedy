# Architecture: what Speedy adds to Strata, and where

Speedy's engine is Strata's: a C++/CUDA engine that keeps a hot set of experts in VRAM (`src/core/expert_cache.cpp`),
computes cold experts on the CPU (`src/kernels/cpu/pool.cpp`, AVX2 kernels) or streams them over PCIe, reads prompts in
chunks with the experts streamed through a ring (`src/prefill/prefill.cpp`), keeps an INT8 KV cache partly resident
and partly streamed from pinned RAM, and decodes in speculative verify windows with an MTP draft head
(`src/core/verify.cpp`, `src/spec/`). The Python server (`serve/server.py`) exposes OpenAI- and Anthropic-compatible
endpoints. Strata's [HOW_IT_WORKS.md](HOW_IT_WORKS.md) and [DETAILS.md](DETAILS.md) describe the engine itself; this
page lists only what Speedy adds.

## Changes in main
| Area | Speedy change | Switch / default | Files |
|---|---|---|---|
| Start-up | Tuned environment defaults are applied before anything reads the environment. A variable that is already set wins. | on; `STRATA_UPSTREAM_DEFAULTS=1` skips all | `src/program/speedy_defaults.hpp` (+ `speedy_defaults_test.cpp`), called at the top of `main` in `src/program/generate.cpp` |
| Decode | Patch 0001 (Strata PR #2689/#2690): a layer's light CPU expert rows (routing weight below the threshold) join the residual one layer late, so the GPU does not wait for them. Layers that read cold experts in 0.2.7's batches wait for all of their rows. | `STRATA_VERIFY_DEFER_CPU=1`, `STRATA_VERIFY_DEFER_CPU_BELOW=0.1` (defaults) | `src/core/verify.cpp`, `src/core/expert_source.cpp`, `src/kernels/cuda/verify_kernels.cu` |
| Prefill | PR #4039: the MMQ path's GU / H / Hq buffers hold one expert group's rows, and the hyper-connection read's scratch lives in the shared region. Less scratch per token means a bigger `--prefill auto` chunk, so each streamed expert crosses PCIe fewer times. Bitwise. | `STRATA_PF_GROUP_SCRATCH=1`, `STRATA_PF_HC_REGION=1` (defaults) | `src/prefill/prefill.cpp` |
| Prefill | PR #4283: a streamed prompt chunk skips the experts that no token in the chunk routes to before they are copied. Bitwise. | `STRATA_PREFILL_ROUTED_STREAM=1` (UD-IQ4_XS preset), `_DEPTH` (16) | `src/prefill/prefill.cpp`, `include/strata/prefill/routed_stream.hpp` |
| Prefill | INT8-KV prompt attention on IMMA tensor cores and KV prefetch: upstream code, turned on by default. | `STRATA_PROMPT_ATTN_IMMA=1`, `STRATA_KV_PREFETCH=1` (defaults) | upstream |
| Memory planning | J13: the pin-by-need resident estimate uses `ExpertCache::slot_stride` (guard included), the same stride the cache lays out. Byte-identical when the guard is off. | always | `src/program/generate.cpp`, `tests/core/slot_budget_test.cpp` |
| Conversation cache | C4: under memory pressure (`/proc/pressure/memory` "some avg10" above the threshold), up to half of the parked conversations are shed, oldest first. | `STRATA_PSI_SHED=<percent>`, off | `include/strata/core/memory_pressure.hpp`, `src/program/generate.cpp`, `tests/core/memory_pressure_test.cpp` |
| Presets | `configs/make_configs.py` writes serve configs for five presets from your paths; `configs/presets/` holds them with `/opt` placeholders. | - | `configs/` |
| Benchmarks | `bench/speedbench.py` (exact token counts, warm mode, provenance JSON), `bench/suffix.py` (agent-faithful warm turns, 90 % reuse gate). | - | `bench/` |
| Tools | Code Mode MCP server: one `run_python` tool with read/grep/ls/sh helpers and bounded output. | client opt-in | `tools/codemode_mcp.py`, [CODE_MODE.md](CODE_MODE.md) |

## Environment variables Speedy sets by default
`STRATA_ROUTE_TAIL_SKIP=0`, `STRATA_PREFILL_CPU_SHARE=0`, `STRATA_PIN_BY_NEED=1`, `STRATA_EXPDEC=1`,
`STRATA_EXCHANGE_ROTATE=1`, `STRATA_ATTN_DAILY=1`, `STRATA_VERIFY_DEFER_CPU=1`, `STRATA_VERIFY_DEFER_CPU_BELOW=0.1`,
`STRATA_PROMPT_ATTN_IMMA=1`, `STRATA_KV_PREFETCH=1`, `STRATA_PF_GROUP_SCRATCH=1`, `STRATA_PF_HC_REGION=1`.
The engine log says `speedy: N of 12 production defaults applied`. Reasons: [SPEEDY_DEFAULTS.md](SPEEDY_DEFAULTS.md).

## On integration branches (not in main yet)
These upstream PRs are cherry-picked onto `integ/v02-prs` / `integ/v02-prs2`, each behind an opt-in switch, and move
to main only after an A/B and, where the arithmetic changes, a quality gate:
| PR | Switch | Area |
|---|---|---|
| #4119 two-pass QSA prompt select | `STRATA_SELECT_TWO_PASS=1` | `src/kernels/cuda/qsa_select.cu` |
| #4567 32-cell QSA decode chunks | `STRATA_QSA_ATTN_VARIANT=2` | QSA decode attention |
| #3919 KV ring sized from depth | `STRATA_KV_RING_ADAPT=1` | `src/core/layer.cpp` |
| #2968 resident-arena staging ring | `STRATA_ARENA_PIN_GIB` | prompt path |
| #2964 adaptive fetch | `STRATA_ADAPT_FETCH=1` or `=2` | decode |
| #3307 MTP drafter memory lent to long prompts | `STRATA_MTP_ROOM_MIN` | serve / prefill |
| #3310 slots a prompt loan leaves the cache | `STRATA_PREFILL_KEEP` | prefill |
| #3309 warm-up prompt before READY | `STRATA_WARM_PROMPT` | serve |
| #4226 persistent disk conversation cache | see the PR | conversation cache |

The full ledger, with tests and A/B results for each change: [SPEEDY.md](../SPEEDY.md).
