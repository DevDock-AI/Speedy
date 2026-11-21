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

**Route tail skip (lossy, off in Speedy).** `STRATA_ROUTE_TAIL_SKIP` drops a missed expert that every token of a
verify window routes at rank 7 or lower: it is neither copied nor computed, so the answers change (upstream measured
KL 0.0126 with it on; earlier 0.003-0.006). Speedy sets it to 0. Upstream Strata had it on by default on CUDA builds
since 0.1.42 and **turned it off by default in v0.2.17** for the same quality reason. To enable it anyway (it is a
decode speed gain when many experts miss the GPU cache: upstream saw x1.29 decode on a 5090), export
`STRATA_ROUTE_TAIL_SKIP=1` or put `"STRATA_ROUTE_TAIL_SKIP": "1"` in the preset's `env` block;
`STRATA_TAIL_SKIP_MAX_CACHED=<percent>` (default 90) turns it off again when that share of experts is already in VRAM.
This is separate from the route prior, which stays on in the presets.

**Upstream behaviour:** `STRATA_UPSTREAM_DEFAULTS=1` sets none of them (byte-identical to the Strata control).
Any single switch can be turned off on its own by exporting it, e.g. `STRATA_VERIFY_DEFER_CPU=0` (the only default
that changes decode output; the others are speed switches checked for parity or by the quality gate).
Engine log line: `speedy: N of 12 production defaults applied`.

Not defaulted (they depend on the pack, the GPU's VRAM or the context target, so they live in the presets,
`configs/make_configs.py`): `--kv int8 --kv-resident 20480`, `--vram-reserve-mib 700`,
`--conversation-cache-mib 8192`, `--spec 4 --spec-min-p 0.5`, `--expert-cache auto`, `--adapt-async 1
--adapt-swaps 80`, `--pool-affinity measure`, `--max-context` (131072 by default; 229376 tested on every pack) + `fit_max_tokens`, and per pack
`STRATA_HC_Q8=1` / `--resident-budget-gib 80` (UD-Q4 / UD-IQ4) and `STRATA_PREFILL_ROUTED_STREAM=1` (UD-IQ4_XS, PR #4283).

**Low-RAM and small-VRAM defaults (upstream Strata v0.2.17, Speedy 0.5.0):**
- setup's RAM budget (`--resident-budget-gib`) on a PC with 64 GB or more is now the RAM less 17 GB, not 24 (64 GB: 47 GiB, was 40),
  the RAM reading rounded up to the PC's sold size first (a 64 GB Windows PC reading 63.1 gets 47 like Linux). Below 64 GB Speedy keeps
  the old rule (the RAM less 24; 48 GB: 24, 32 GB: 8): at 32 GB the new rule cut UD-Q4_K_XL prefill by about three quarters in our test.
  A unified-memory GPU keeps less 24. On our 64 GB test: UD-Q4_K_XL decode +19..47%, prefill +27..31%, SSD reads -27%. Upstream, under a real 64 GiB limit (UD-Q4_K_XL): decode 57.7 -> 84.2 tok/s, SSD reads in a 65K prompt 97.7 -> 37.0 GB.
  The engine still lowers N to the free RAM less 4 GiB at start. Your own `--resident-budget-gib N` always wins.
- `STRATA_ADAPT_FETCH=2` (the PCIe share's fetched experts are admitted into the GPU expert cache) is the default on compute capability
  12.x cards with 16 GiB or less (RTX 5050 - 5080, laptops): upstream decode +4..+43%, tool-call decode 51 -> 63 tok/s. Not bit-exact;
  one-GPU runs only, and not with `--adapt-async`, `--pipeline-windows`, `--batch` or the resident RAM-budget mode (the log says why).
  `STRATA_ADAPT_FETCH=0` turns it off; `STRATA_SM120_PERF=0` or `STRATA_DEFAULTS=026` drops the whole sm_120 table. Other cards: unchanged.

New wins proven by the A/B and quality gates ([BENCHMARKS.md](BENCHMARKS.md)) are folded in here and in the presets
with their measured numbers; opt-in engineering switches (e.g. `STRATA_POOL_DEP`) stay off until they pass.

See also: [CONFIGS.md](CONFIGS.md) (presets), [ARCHITECTURE.md](ARCHITECTURE.md) (where each change lives).

**Server defaults (Speedy 0.5.0).** These apply when the config and the request name nothing; the request always wins.

| Setting | Default | Off / change |
|---|---|---|
| Thinking level for requests that name none | `medium` | `"default_reasoning_effort"` / `STRATA_DEFAULT_REASONING_EFFORT` (`off`, `low`, `high`) |
| Thinking budget | 32768 tokens | `"reasoning_budget_tokens": 0` (no budget) or another number |
| Reasoning nudge (a short "start converging" note at 60% and 80% of the budget) | `[0.6, 0.8]` | `"reasoning_nudge": false` or `STRATA_REASONING_NUDGE=off` |
| Loop breaker (same call + same result, no write between: a note naming the next step; the repeated tool masked for one reply after the second such turn) | on, no thinking floor | `STRATA_LOOP_BREAKER=0`; `STRATA_LOOP_BREAKER_FLOOR=N`, `STRATA_LOOP_BREAKER_MASK=0`, `STRATA_LOOP_BREAKER_NOTE=...` |
| Edit cycles (0.5.3, part of the loop breaker: edits that go back and forth on one file with no other work between; a note, then the file-editing and todo tools masked until two turns of other work followed) | on | `STRATA_LOOP_BREAKER_CYCLE=0` (`STRATA_LOOP_BREAKER_MASK=0` keeps the notes, no mask) |
| Junk writes (0.5.3, part of the loop breaker: a dummy file written instead of the step the reply described, such as `/tmp/opencode/cr8` holding `cr8` or `placeholder.txt` `x` -> `y`, is not work done: it does not reset the loop / cycle counts, and four such turns in a row get a note, then the editing tools masked) | on | `STRATA_LOOP_BREAKER=0` |
| Tool call opening always names a declared tool (0.5.3: the model writes it, the server steps in only when no declared name can follow; see `STRATA_TOOLCALL_CONSTRAIN` in [DETAILS.md](DETAILS.md)) | on | `STRATA_TOOLCALL_CONSTRAIN=0`; `eager` = the mask at every call |
| "Says X, does Y" (0.5.3: the reply's last sentence names the tool it is about to call and another is opened: written again with the named tool, once; else a note after the result) | on | `STRATA_TOOLCALL_INTENT=0` |
| Thinking cap on a reply that follows a tool result | 6144 tokens | `STRATA_TOOL_STEP_THINK_CAP=0` |
| Answer-room reserve | on | `STRATA_ANSWER_ROOM=0` / `"answer_room": false` |
| MTP drafter written by setup | tuned (`<mtp dir>-tuned`) | `setup --mtp-drafter stock`, or `"mtp_drafter": "stock"` / `STRATA_MTP_DRAFTER=stock` |

Measured on UD-Q4_K_XL (1M context) with an OpenCode-style agent suite: 24/24 tasks in 13.3 min with the loop breaker,
the medium/32K budget and the 6144 cap, against 21/24 in about 30 min without them; on 11 harder multi-file tasks 9/11 vs 8/11.

**Setup's 1-GPU profile (Speedy 0.5.0, `--speedy-profile tuned`, the default).** On one NVIDIA card with an Unsloth UD-* file,
setup now writes the settings of our tuned 1-card presets: `STRATA_ROUTE_PRIOR=1.5`, `STRATA_PREFILL_ROUTED_STREAM(_PP)=1`,
`STRATA_PREFILL_STREAM_MIN=512`, `STRATA_SELECT_TWO_PASS=1`, `STRATA_GDN_TC=1`, `--prefill auto:16384`, `--adapt-async 1 --adapt-swaps 80`
(no `--adapt-async` on RTX 50 cards of 16 GB or less, where the adaptive fetch default needs synchronous swaps), `--pool-affinity measure`.
Every config also gets `STRATA_TOOLCALL_CONSTRAIN=1`, `STRATA_TOOLCALL_REPAIR=1`, `STRATA_REPEAT_WARN=1` and
`STRATA_UNDECLARED_TOOL_CALLS=call`. Before, a setup-written config on one 20 GB card decoded 33-35% slower than our tuned preset
(UD-IQ4_XS 92 vs 138 tok/s, UD-Q4_K_XL 77 vs 118). `--speedy-profile upstream` writes upstream Strata's settings instead.
