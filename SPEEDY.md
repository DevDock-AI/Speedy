# Speedy patch ledger

Base: **Strata v0.2.7.1** (upstream 46c152a1). The previous base, Strata v0.1.45 + patch 0001, is tagged
`speedy-on-0.1.45`. Each change has its own branch (`exp/<ID>-<name>` or `exp/PR<n>`) and is merged only with an A/B
on the reference machine plus a quality gate where the numbers change. The A/B rule is in
[CONTRIBUTING.md](CONTRIBUTING.md). Reference machine: RTX 3080 10 GB LHR (unlocked), Threadripper PRO 5975WX, 128 GB,
Debian 13, CUDA 13.4.

## In main
| ID | Change | Origin / attribution | Switch / default | Tests | Measured |
|---|---|---|---|---|---|
| 0001 | Light CPU expert rows join one layer late in verify | Strata PR #2689/#2690 (lask3802, MIT); re-ported to 0.2.7's cold batches (cold layers wait for every row) | `STRATA_VERIFY_DEFER_CPU=1`, `_BELOW=0.1`; on in the defaults | upstream verify parity tests | in production since 0.1.45 (cuts the decode GPU-wait; KL measured by the PR author) |
| defaults | Production env defaults built in | Speedy | on; `STRATA_UPSTREAM_DEFAULTS=1` turns them off | speedy_defaults_test | = presets |
| A4 | INT8-KV prompt attention on IMMA | architectds/Strata (MIT), already in upstream as an opt-in | `STRATA_PROMPT_ATTN_IMMA=1` default | qsa_prompt_attn_parity; suite; needle 3/3 | 32K prefill +2.0 %, decode +2.1 % |
| E25 | KV prefetch in prefill | upstream opt-in | `STRATA_KV_PREFETCH=1` default | - | Q2 +1.5 % (32K), +1.4 % (128K) |
| J13 | Pin-by-need estimate priced with `ExpertCache::slot_stride` (guard included) | idea: Basalt PR #120 (MIT) | always; byte-identical with guard 0 | slot_budget_test | audit fix |
| C4 | PSI-driven shedding of parked conversations | idea: Basalt PR #113 (MIT) | `STRATA_PSI_SHED=<avg10 %>`, off | memory_pressure_test | robustness |
| F2 | Code Mode MCP tool | pattern: Cloudflare Code Mode | client opt-in | tools/test_codemode_mcp.py | - |
| harness | `bench/speedbench.py` and `bench/suffix.py` (90 % reuse gate) | Speedy | - | mock-server check | - |
| presets | `configs/make_configs.py`, `configs/presets/` (5 packs) | Speedy | - | - | see docs/BENCHMARKS.md |

## Upstream PRs in main (A/B on Q2_0, IQ3_S and UD-IQ4_XS at 60/120/200K; docs/BENCHMARKS.md)
| PR | Author | Switch | Port notes | A/B | Status |
|---|---|---|---|---|---|
| #4283 routed prompt stream | Stefano Sinden (bitaccesscomau) | `STRATA_PREFILL_ROUTED_STREAM=1`, `_DEPTH` | issuer conflict with 0.2.7's copy events: off when `STRATA_PREFILL_COPY_EVENT_EVERY>1` or `_COPY_WAIT_DEDUPE=1` | ABBA, IQ4_XS: prefill +10.5/+8.9/+8.0 %, decode same. Q2: -1.0/-0.5 % (noise). IQ3_S: no gain on top of #4039 | on in the UD-IQ4_XS preset |
| #4039 smaller prompt scratch | merbanan | `STRATA_PF_GROUP_SCRATCH=1 STRATA_PF_HC_REGION=1` | sizes fixed for 0.2.7's `STRATA_MMQ_WS` tables (bounds_ints in the count and in the host offset) | ABBA, IQ4_XS: auto chunk 3584 -> 5632, prefill +29/+33/+32 %; IQ3_S chunk 4864 -> 7680, +22/+24 %; Q2: chunk 7168 -> 8192, +5.6/+4.9 %; decode same; with #4283 IQ4_XS +41-44 %. Short gate PASS | **default on** |

## Upstream PRs on integration branches (`integ/v02-prs`, `integ/v02-prs2`; opt-in, not in main)
| PR | Author | Switch | Status |
|---|---|---|---|
| #4119 two-pass QSA select | Stefano Sinden | `STRATA_SELECT_TWO_PASS=1` | A/B pending |
| #4567 QSA decode 32-cell chunks | Jackwwg83 | `STRATA_QSA_ATTN_VARIANT=2` | A/B pending (near-exact: needs the quality gate) |
| #3919 KV ring from depth | Zac Chen (xyzzing) | `STRATA_KV_RING_ADAPT=1` | A/B pending |
| #2968 resident-arena staging ring | sergqwer (sergiywith) | `STRATA_ARENA_PIN_GIB` | A/B pending |
| #2964 adaptive fetch | sergqwer | `STRATA_ADAPT_FETCH=1` or `=2` | A/B pending |
| #3307 drafter memory lent to long prompts | gopinath87607 | `STRATA_MTP_ROOM_MIN` | A/B pending |
| #3310 prefill keep | gopinath87607 | `STRATA_PREFILL_KEEP` | A/B pending |
| #3309 warm-up prompt | gopinath87607 | `STRATA_WARM_PROMPT` | A/B pending |
| #4226 persistent disk conversation cache | G0-0000 | see the PR | A/B pending |

## Not carried (reason)
- C3 pinned bounce buffer (Basalt PR #66 idea): no gain (park 110 vs 98 ms); upstream `STRATA_CKPT_ASYNC` covers it.
- J02 arm 2 pool dependency scheduling (branch `exp/J02-pool-dep`): decode is bound by RAM bandwidth, +0.8 % (noise).
- Full list of what was rejected: [docs/TUNING-HISTORY.md](docs/TUNING-HISTORY.md). Dual-GPU items: [docs/ROADMAP.md](docs/ROADMAP.md).
