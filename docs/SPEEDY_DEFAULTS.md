# Speedy defaults

Speedy starts with the engine environment that the reference machine's production presets use, so a bare
`build/strata ...` behaves like production without a config. At start-up (`main()`, before any option is read)
each variable below is set **only if it is not already in the environment**; a config's `env` block or an
explicit `export` always wins. Source: `src/program/speedy_defaults.hpp` (test: `speedy_defaults_test`).

| Variable | Speedy default | Upstream Strata | Why |
|---|---|---|---|
| `STRATA_ROUTE_TAIL_SKIP` | 0 | on for some packs | the tail skip drops low-weight experts: lossy |
| `STRATA_PREFILL_CPU_SHARE` | 0 | auto | GPU + PCIe beat a CPU share on a 3080 + 5975WX |
| `STRATA_PIN_BY_NEED` | 1 | 0 | |
| `STRATA_EXPDEC` | 1 | 0 on CUDA; 1 on AMD RDNA4 gfx12 via `arch_defaults` (decode half, not under `STRATA_DEFAULTS=026`) | on for every GPU here |
| `STRATA_EXCHANGE_ROTATE` | 1 | 0 | docs/EXCHANGE_ROTATION.md |
| `STRATA_ATTN_DAILY` | 1 | 1 on sm_86 (RTX 30xx, `nv_defaults`, off with `STRATA_SM86_PERF=0`); 0 elsewhere | on for every GPU here |
| `STRATA_VERIFY_DEFER_CPU` / `_BELOW` | 1 / 0.1 | 0 (opt-in) | Strata PR #2689/#2690. **Changes the decode arithmetic**: CPU rows of routing weight below 0.1 join the residual one layer late (upstream marks it "opt-in, changes the output"). Not bitwise; quality-gated instead: short gate Q2_0 8/8 + 4/4 and UD-IQ4_XS 8/8 + 4/4. Turn off alone with `STRATA_VERIFY_DEFER_CPU=0` |
| `STRATA_PROMPT_ATTN_IMMA` | 1 | 0 | A4: INT8-KV prompt attention on IMMA tensor cores (architectds/Strata `best`, MIT); Q2 32K prefill +2.0 %, decode +2.1 %; parity + suite + needle 3/3 (89.6K, 127.3K) |
| `STRATA_KV_PREFETCH` | 1 | 0 | E25: KV prefetch in prefill; Q2 32K prefill 2557 -> 2595 (+1.5 %), 128K 2490 -> 2525 (+1.4 %), decode flat |
| `STRATA_PF_GROUP_SCRATCH`, `STRATA_PF_HC_REGION` | 1 | 0 | Strata PR #4039 (merbanan), bitwise per the PR's parity check (re-verified by Speedy's short quality gate, not by a Speedy parity test): the MMQ scratch holds one expert group and the hc read shares the region, so `--prefill auto` picks bigger chunks (IQ4_XS 3584 -> 5632, IQ3_S 4864 -> 7680, Q2 7168 -> 8192). ABBA prefill: IQ4_XS +29-33 %, IQ3_S +22-24 %, Q2 +5 %; decode unchanged; short gate passed |

**Upstream behaviour:** `STRATA_UPSTREAM_DEFAULTS=1` sets none of them (byte-identical to the Strata control).
Any single switch can be turned off on its own by exporting it, e.g. `STRATA_VERIFY_DEFER_CPU=0` (the only default
that changes decode output; the others are speed switches checked for parity or by the quality gate).
Engine log line: `speedy: N of 12 production defaults applied`.

Not defaulted (they depend on the pack, the GPU's VRAM or the context target, so they live in the presets,
`configs/make_configs.py`): `--kv int8 --kv-resident 20480`, `--vram-reserve-mib 700`,
`--conversation-cache-mib 8192`, `--spec 4 --spec-min-p 0.5`, `--expert-cache auto`, `--adapt-async 1
--adapt-swaps 80`, `--pool-affinity measure`, `--max-context` (131072 by default; 229376 tested on every pack) + `fit_max_tokens`, and per pack
`STRATA_HC_Q8=1` / `--resident-budget-gib 80` (UD-Q4 / UD-IQ4) and `STRATA_PREFILL_ROUTED_STREAM=1` (UD-IQ4_XS, PR #4283).

New wins proven by the A/B and quality gates ([BENCHMARKS.md](BENCHMARKS.md)) are folded in here and in the presets
with their measured numbers; opt-in engineering switches (e.g. `STRATA_POOL_DEP`) stay off until they pass.

See also: [CONFIGS.md](CONFIGS.md) (presets), [ARCHITECTURE.md](ARCHITECTURE.md) (where each change lives).
