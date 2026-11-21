# Changelog

## Speedy 0.5.3 (2026-11-21)
- **Loop breaker catches edit cycles** (on with the loop breaker; `STRATA_LOOP_BREAKER_CYCLE=0` turns it off): successful edits that go back and forth on one file while nothing else gets done were invisible to the 0.5.1 rules, because every edit succeeded and differed from the one before (seen: `PLAN.md` checkboxes `[ ]` -> `[~]` -> `[x]` -> `[ ]` for hundreds of requests). Now counted, since the last other work (a call that did not fail and is not about that file): an edit that restores text an earlier edit replaced (second time), a third different change of the same lines' checkbox marks, four edits in a row that each rewrite what the one before wrote, the same sequence of calls with the same results going round (period 2 to 6, a write in it), four turns in a row whose edits all failed (any file, any error text), and the same for whole-file writes and todo-list calls that return to an earlier content.
- **The mask now holds**: the first cycle turn gets a note that says to stop updating the file and do the next real step; from the second on, every file-editing and todo-list tool (not only the one that looped: masking `edit` alone moved the same rewrite to `write`) is kept out of the reply until two turns of other work followed, four after a relapse, up to eight. A relapse on the same file is caught on its second edit. The shell is never masked by a cycle, and a round that holds other calls (edit -> test -> revert -> test with the same results) gets the note only.
- **Not a cycle**: many edits of one file in a row while building it, edit -> test cycles, one plan update per finished step, several edits sent in one turn, a single undo, a retry after a failed edit. Eight turns in a row that only edit the session plan (`PLAN.md`, `TODO.md`) or the todo list get a soft note, never a mask.
- A failed-edit result now also means Claude Code's `<tool_use_error>` / "has not been read" / "modified since" texts.
- **Tool-name constraining rewritten: it no longer steers the model to the wrong tool** (`STRATA_TOOLCALL_CONSTRAIN`, now on by default; `0` turns it off). Since 0.4.6 the server stopped at every `<tool_call>`, wrote `\n<function=` itself and let the model pick the name under a bias. Qwen's vocabulary spells `<function=bash>` as `<function` `=b` `ash` `>` (and `=g`+`lob`, `=g`+`rep`, `=read`, `=t`+`od`+`ow`+`rite`), so the forced bare `=` token was a place the model only knows from the names written `=`+NAME: `write`, `edit`, `skill`, `task`, `webfetch`. Seen on an agent benchmark with OpenCode's ten tools: 2,377 of 2,387 constrained openings named one of those five (write 2,307) while the thinking said "invoke the `bash` tool"; one task made 1,665 `write` calls and 3 `bash` calls. With the flag off the same model wrote 81 openings, every one in the tokenizer's own spelling, 59 of them bash, and the three tasks took 76 requests instead of 2,421. Now the model writes the opening itself; the server reads along and steps in only at the first token after which no declared name can follow (another letter case and the parser's aliases count as declared). What was written stays; the rest is finished under a mask in which the model's own ranking decides: tokens every remaining name shares are written directly, and where names part, one one-token request gives the same bias to every token that continues a declared name. A valid opening costs no extra request (before: one or more per call, 20 to 600 ms). `STRATA_TOOLCALL_CONSTRAIN=eager` keeps the mask at every call, without the forced `=`. The pick keeps the request's own `-100` bans and runs without the repetition penalty (the engine applies it after the bias). Not applied when the request defers tools.
- **Tool call repair** forced the same bare `=` (`<tool_call>\n<function=`); it now forces only the tokens every declared name's opening shares.
- **"Says X, does Y"** (`STRATA_TOOLCALL_INTENT`, on; `0` turns it off): when the last sentence before a reply's first call states the tool it is about to call ("invoke the `bash` tool", "call bash instead", "let me run `cmd`") and the opening names another, the opening is written again with the stated tool, once per reply, before anything of it is streamed. Narrow on purpose: declared names only, the last sentence only, no questions, quotes, negations or later steps, and no change when the call made is the shell, a tool that only looks (read, grep, glob) or, for an edit tool, another edit tool. Where the server could not rewrite it (engine-side mask, `--batch`), such a turn gets a note after its result. On the looping benchmark transcripts it finds 1,944 such turns (1,937 of them "said bash / glob / grep / read / todowrite, called write / edit / skill / task"); on 17,300 turns of ordinary OpenCode sessions 5 (all in two sessions that were looping the same way), on 19,980 Claude Code turns none.
- **Junk writes are not work done** (loop breaker): a dummy file written in place of the step the reply described (`/tmp/opencode/cr8` holding `cr8`, `placeholder.txt` `x` -> `y`; a whole file of at most 64 bytes in a scratch directory, at most 16 bytes of change to a dummy-named file, at most 8 bytes in a file named like `x` or `cr8`) no longer counts as other work or as a workspace change: three in a row are a loop (note, then `write` masked), four turns in a row an edit cycle (note, then every editing tool masked until other work followed). Real files in a scratch directory, small edits of real files and the session plan are not judged by it.
- serve/version.py -> 0.5.3.

## Speedy 0.5.2 (2026-11-21, folded into 0.5.3)
- **Updates bring the 0.5 settings** (UPDATE.bat / `./speedy-update`, `setup.py --update`): each installed Qwen3.8-Flash-Next config gets what a fresh 0.5 install writes. The tuned MTP drafter (downloaded and SHA-checked; offline or a bad file keeps the stock one), the thinking budgets of our old generic presets (16384 / 8192 / 4096) dropped so the server default 32768 applies (the old Q2_0 "low" preset's `reasoning_effort: low` with them), and the Speedy tool-call env everywhere plus the 1-GPU speed profile on one NVIDIA card with a UD-* file. Kept: `"mtp_drafter": "stock"` / `STRATA_MTP_DRAFTER=stock`, any other budget, env entries and engine options already set, configs set up with `--speedy-profile upstream`; Qwen3.6 and DeepSeek configs are not touched. Each changed config is backed up as `strata-*.json.bak-0.5.2` and the update prints what changed and how to revert it. A second update changes nothing.
- Setup records `"speedy_profile": "upstream"` and `"mtp_drafter": "stock"` in the config when they are chosen, so an update respects them.
- serve/version.py -> 0.5.2.

## Speedy 0.5.1 (2026-11-21)
- **Loop breaker catches repeated failing edits/writes**: a write or edit whose result says it did nothing (not found, error, no match, unchanged) no longer counts as a workspace change, so the same failing call repeated is a loop (note, then the tool masked for one reply). A successful write still resets it, so edit -> test -> edit -> test is not a loop.
- **Near-identical repeats**: the same tool returning the same result 3 times in a row is a loop even when the arguments differ (a failing result, or arguments that differ only slightly). Results are compared with call ids, hex ids, timestamps and durations normalized; arguments with sorted keys.
- **Repeat warning backstop**: with the loop breaker on, the plain repeat note (same name and arguments) is still added for a call made 3+ times; a turn never gets two notes.
- **Serve fixes from upstream Strata #6474 / #6471**: restart-safe slot cleanup, STOP on an abandoned batch admission.
- serve/version.py -> 0.5.1.

## Speedy 0.5.0 (2026-11-21)
- **Smarter agent defaults** (serve; every one can be turned off, see docs/SPEEDY_DEFAULTS.md): thinking level `medium` and a 32768-token thinking budget when the config names none; the reasoning nudge at 60%/80% of the budget; the loop breaker (note naming the next step, repeated tool masked for one reply, no thinking floor); a 6144-token thinking cap on replies that follow a tool result; the answer-room reserve now also for max_tokens 128..1023 on every model family.
- **Reasoning restore**: a tool-call turn's thinking is restored by call id and spliced back, so the conversation cache continues instead of re-reading (`STRATA_REASONING_RESTORE=0` turns it off).
- **Tool calls**: argument shape repair (Strata #6315), an indented bare `<function=` held while streaming (Strata #6138), lenient missing `</parameter>`, schema-typed coercion, plural/singular parameter names (#6285), opt-in constrained tool-name decoding in the engine (Strata #5969, #6022 shape).
- **Robustness**: a stalled request body gets a 408 and frees its slot (Strata #6107, `STRATA_BODY_STALL_S`, 30 s); a client that stops reading is dropped after 120 s without progress (`STRATA_STREAM_WRITE_TIMEOUT_S`, Strata #6153 idea); a GPU that stopped answering ends the engine (#5990); `/v1/messages` timings and decode speed in the monitor.
- **Low-RAM / small-VRAM (upstream Strata v0.2.17)**: setup's RAM budget is the RAM less 17 GB from 64 GB up (old rule below 64 GB), with the reading rounded to the PC's sold size; cgroup v1 clean-cache credit and hybrid hosts; `STRATA_ADAPT_FETCH=2` default on RTX 50 cards of 16 GB or less (#2964), opt-in elsewhere; the layer-major walk counts the model's experts (#5104, #6018).
- **Setup**: a 1-GPU tuned profile (`--speedy-profile tuned`, default) writes our presets' route prior, routed prompt stream, two-pass select, GDN tensor cores and tool-call env; the tuned MTP drafter (`--mtp-drafter tuned`, default; Qwen Community License 1.0, licenses/Qwen-Community-License-1.0.txt); ccache/sccache off for vendored ggml builds (#6172); Hyper-V GPU-PV detection (#4718); WDDM shared reserve on 2+ cards; port-busy check before the download; python fallback in setup.sh.
- **Nudges and caps**: the reasoning nudge is skipped when the reply has no room left for it, and no staged nudges are sent under short max_tokens; a nudge keeps a forced tool call (`tool_choice`); the 6144-token tool-step thinking cap applies only when the request names no thinking budget of its own.
- **Setup VRAM fit check** (#25): a single-GPU Flash-Next config that cannot fit the card is caught before the download, with a clear error naming the card, the need and the options; an engine that exits at start prints a hint on the likely cause.
- **Resident quant pick on big cards** (#23): setup picks the best quant a large card holds fully resident; docs/BENCHMARKS.md notes `cache_prompt`.
- **setup.sh** (#24): python fallback, free-port check, benchmark doc.
- serve/version.py -> 0.5.0.

## Speedy 0.4.12 (2026-11-19)
- **Quoted `</think>` stays reasoning**: a typed `\n` right before the tag with a quote or backtick up to four characters back (`` `\n</think>\n\n` ``) is the model writing about its own tag, so the thinking no longer ends there. A real line break after the tag still ends it.
- **TCP_NODELAY on accepted sockets** (on by default, `STRATA_TCP_NODELAY=0` keeps Nagle on): small SSE chunks go out at once instead of waiting on the client's delayed ACK.
- **Optional malloc arena limit** (`"malloc_arena_max"` config key, `STRATA_MALLOC_ARENA_MAX` env; default 0 = glibc's own limit): caps glibc malloc arenas on Linux, with mallopt in the server and `MALLOC_ARENA_MAX` for the engine, to keep host RSS down.
- serve/version.py -> 0.4.12.

## Speedy 0.4.11 (2026-11-18)
- **Reasoning effort floor** (`"reasoning_effort_floor"` config key, `STRATA_REASONING_EFFORT_FLOOR` env; `low` / `medium` / `high` / `off`, default `low`): a request that offers tools and asks for a thinking level below the floor (`none` / `off`, `enable_thinking: false`, Anthropic `thinking: disabled`, or a server default of `none`) is raised to the floor, on `/v1/chat/completions`, `/v1/messages` and `/v1/responses`. Reason: a quality drop with effort `none` on agent/tool requests at long context. Requests that name no level, set their own reasoning budget, or have no tools are unchanged. Opt-in `"reasoning_effort_floor_ctx_tokens"` (`STRATA_REASONING_EFFORT_FLOOR_CTX_TOKENS`, default 0 = off) applies the floor to requests without tools too when the prompt is estimated above that many tokens. The env values are read by the server process itself (process environment, then the config's `env`). Logged at start and once per raised request.
- **Suffix drafter 8-gram table** (`STRATA_SUFFIX_NGRAM8`, on by default, `0` turns it off): a second table indexes every 8-gram next to the trigram table, so a quoted stretch still finds an older source when a common trigram's slots are filled by newer repeats (long-context file edits). Proposals take the longest match of both. Greedy output is bit-identical; host RAM only (one more table of the trigram table's size), no VRAM.
- **Reasoning loop detector from 300 words**: the repeat check now covers reasoning from 300 words on (all of it, up to the last 2,000 words as before); before, reasoning under 2,000 words was never checked, so short filler loops were missed.
- **`context_length` in `/v1/models`** (Strata #6116): clients that read `context_length` / `max_context_length` see the engine's context instead of assuming 32768.
- **Lost device named in the release drain; ordered pinned frees** (Strata #6135, #5990): a GPU error while waiting for in-flight work is recorded and reported as a lost device instead of a verifier bug; buffers the GPU reads are freed only after the streams that use them finish.
- **Optional small-upload ring** (Strata #6124, `STRATA_SMALL_UPLOAD_RING=1`, default off): uploads of at most 64 KiB use their own ring of 60 slots, so the host can run further ahead of the device on a layer split (about 3.75 MiB more pinned host memory when on). Adds `pinned_ring_test`.
- **logit_bias upload ordered with sampling**: the bias is copied on the sampling stream, so a draw can no longer read a half-written bias.
- **strata_bench**: a streamed delta that carries only tool calls counts as the first token (tool-call turns report a TTFT).
- serve/version.py -> 0.4.11.

## Speedy 0.4.10 (2026-11-16)
- **Async checkpoint copies on a layer split** (Strata #6075): with `STRATA_CKPT_ASYNC=1`, one lock now covers the resize of the shared checkpoint staging buffer, the device copy into it and the background copy's submit, and a copy another stage queued is waited for first. Before, one stage could free the buffer while another stage's copy still read it ("invalid resource handle", then SIGSEGV). Off by default.
- **`STRATA_EMBED_VRAM=1` is ignored on a layer split** (as in edge0 2a84e52b): the token embedding table must be readable from every stage's card (the MTP draft layer runs on the last one), so it stays in mapped host memory and the log says why. Before, the draft layer read the first card's VRAM pointer from the last card (illegal memory access). The MTP fallback gather also reports a missing `token_embd` table instead of reading a null pointer.
- **A prefill CUDA error ends the engine at once** (Strata #6054): the prompt kernels' error check used `std::exit`, which could hang in static destructors while other threads held the GPU, so the server waited for its 300 s silence watchdog. It now prints the reason (out of memory is named as such) and exits immediately; the server sees the engine end and fails the request with an error.
- **Requests during shutdown get an error** (Strata #6055): on Ctrl+C / SIGINT the server refuses new inference requests with 503 `server_draining` whether or not `--shutdown-timeout` is set, and a request cut off because the engine is closing ends with an error instead of a reply of 0 tokens with `finish_reason: "length"`.
- **Periodic prompt checkpoints on a fixed grid** (Strata #6052): `--prompt-cache-every` checkpoints are due every N tokens from where the read started, not N tokens after the last checkpoint's chunk end. With equal prompt chunks slightly shorter than N (the host-aware plan, e.g. `--prefill auto:16384` with the default 16384), every other chunk end was skipped, so a follow-up turn re-read up to twice as much of the prompt.
- **RTX 50 (sm_120): NVFP4 prompt on FP4 tensor cores by default, experimental on RTX 50, community results welcome.** On sm_120 the engine now defaults `STRATA_PREFILL_NVFP4=w4a4` (the existing FP4 x FP4 prompt kernel, built only for `120a`), for NVFP4 experts only; `STRATA_NVFP4_PROMPT=0` or `STRATA_DEFAULTS=026` keeps `w4a8`, an explicit `STRATA_PREFILL_NVFP4` wins. Not measured on an RTX 50 by us (see docs/NVFP4.md for the KL trade-off). Every other architecture, RTX 30/40 included, is unchanged; `nv_defaults_test` covers the gate.

## Speedy 0.4.9 (2026-11-15)
- **Default sampling values reverted to the proven set** after an IFBench comparison (79 vs 68 prompt-level loose): `"default_sampling": "on"` now means temperature 0.6, top_p 0.95, top_k 20, min_p 0, repetition_penalty 1.0, presence_penalty 0. Presets that had presence_penalty 1.5 before 0.4.8 get their old values back as a `default_sampling` JSON object; the client can still override every field.

## Speedy 0.4.8 (2026-11-15)
- **Server-side default sampling** (`STRATA_DEFAULT_SAMPLING` env, `"default_sampling"` config key): fills only the sampling fields a request leaves out, on every route (`/v1/chat/completions`, `/v1/messages`, `/v1/responses`). `on` = temperature 0.33, top_k 20, top_p 1.0, min_p 0.08, repetition_penalty 1.0, presence_penalty 0.06 (same tool-call/coding pass rate as greedy on UD-Q4_K_XL, about +10 % decode at 55K from higher MTP acceptance); `off` = no server or model-card defaults (absent fields greedy); or a JSON object of fields. Client fields always win; an explicit `temperature: 0` stays greedy and gets no defaulted penalty. On in all `configs/cards/` presets (replacing their `sampling` block). Logged at start, reported in `GET /speedy/version` and `/health` (`default_sampling`).
- **Server default reasoning effort** (`STRATA_DEFAULT_REASONING_EFFORT` env, `"default_reasoning_effort"` config key; `low` / `medium` / `high` / `max` / `off`): the thinking level for requests that name none. The client always wins: a request's own `reasoning_effort` / `reasoning`, `chat_template_kwargs`, Anthropic `thinking` / `output_config` or `reasoning_budget_tokens` gets no default. The older `"reasoning_effort"` key still works (read after the new one). `medium` in all `configs/cards/` presets (`strata-final-reason-low` keeps `low`). Logged at start, reported in `GET /speedy/version` and `/health` (`default_reasoning_effort`).
- **Opt-in `STRATA_DEFER_HITS_FUSE=1`**: deferred verify layers (`STRATA_VERIFY_DEFER_CPU`) fold `moe_hit_add` into the deferred row copy (same rounding), one graph node fewer per deferred layer and group. Off by default.
- **`STRATA_GRAPH_PRIO=1`** (already in the engine, opt-in): the verify window graph's kernel nodes run at their capturing stream's priority; bit-exact. Used together with the defer-hits fuse in the 20 GB + 10 GB production setup (UD-IQ4_XS and UD-Q4_K_XL).
- **Tools added in the conversation** (Strata #6005): Anthropic `tool_addition` blocks in a system message inside `messages` (Claude Code sends tools of MCP servers that connect mid-session this way) are now read: the tool is shown to the model where it was added (like a `tool_reference`, so the prompt's tool list and its cache stay the same) and a call for it is parsed as a tool call. Before, the block was dropped and the tool was invisible.
- **OpenAI compatibility for `/v1/models`:** every entry now has `created` (unix time) and `owned_by` (`"speedy"`), which strict OpenAI clients require.
- **RTX 3080 20 GB (one card) presets** in `configs/cards/rtx3080-20gb/1x/` (Q2_0, IQ3_S, UD-IQ3_XXS, UD-IQ4_XS, UD-Q4_K_XL at 229K and 1M) with `CARD.md`, and a README table.
- **Licence wording:** NOTICE and docs now say Speedy's own code is under the Speedy Personal & Non-Commercial License (LICENSE); code from Strata stays MIT (`licenses/Strata-MIT.txt`). README notes the 144 GB RAM of the 20 GB + 10 GB machine.

## Speedy 0.4.7 (2026-11-14)
- **New-release notice and `speedy-update`:** the server checks GitHub's latest release at start and every `STRATA_UPDATE_CHECK_HOURS` (3) hours in the background, anonymously (User-Agent `Speedy/<version>` only), cached in the user cache dir; a newer one is logged, reported in `GET /speedy/version` and `/health` (`"speedy"`), and sent as the `X-Speedy-Update` header. Never in model output, never installs by itself; `STRATA_UPDATE_CHECK=0` turns it off. `./speedy-update` (`serve/update.py`) updates a git checkout (release tag + rebuild, `--rollback`) or a zip install (verified engine zip unpacked beside the old one); `--check`, `--yes`.
- **Crash stack traces in the server** (Strata #5921): faulthandler is armed in `serve/server.py`, so a native crash prints the Python stack of every thread; SIGUSR1 prints them on a hang. `STRATA_NO_FAULTHANDLER=1` turns it off.

## Speedy 0.4.6 (2026-11-13)
- **Constrained tool-call opening** (`STRATA_TOOLCALL_CONSTRAIN=1`, opt-in): when the model writes `<tool_call>`, the next text is `\n<function=` (forced) and then one of the request's declared tool names and `>\n`, picked by the model one token at a time under a logit bias that only lets through tokens that continue a declared name (the rest is forced once one name is left). Decoding is free again after it. The bias is applied where the engine samples (the verifier), so an MTP/speculative draft that breaks it is rejected like any other draft. Stops the `<4>` / `<=edit>` / undeclared-name slips at the token after `<tool_call>`. Needs an engine with `logit_bias=1` and no `--batch`; otherwise off. Logs `[strata] constrained tool call: NAME (picks, ms)`, metric `strata:tool_calls_constrained_total`.
- **Old thinking dropped at long context** (`STRATA_DROP_THINKING_ABOVE=N`, opt-in, e.g. 300000): when the conversation is above about N tokens, the thinking of older assistant turns is left out of the prompt (the last turn keeps its own). The turns that lose it are fixed in blocks of `STRATA_DROP_THINKING_STEP` (16) assistant turns, so the prompt prefix changes once per block, not on every request. Logs the tokens saved, metric `strata:thinking_tokens_dropped_total`.
- **Repeat warning** (`STRATA_REPEAT_WARN=1`, opt-in): an assistant tool call identical (name and arguments) to one of the previous `STRATA_REPEAT_WARN_WINDOW` (3) calls gets a short note after its result ("You already made this exact call; its result is above. Do something different."), and the latest one is logged to `STRATA_TOOLCALL_FAIL_LOG` with outcome `repeat`. Metric `strata:tool_call_repeats_total`.

## Speedy 0.4.5 (2026-11-13)
- **Tool call repair** (`STRATA_TOOLCALL_REPAIR=1`, on in the presets): when a reply holds a `<tool_call>` that does not parse into a call of a declared tool (seen on UD-Q4_K_XL at 80-220K context: `<4>`, `<=edit>`, `<` where `<function=NAME>` belongs), the broken text is held back and the call is regenerated from the cached prefix with `<function=` forced (up to 2 tries); if every try fails the original reply is returned as before. Logs `tool call repaired`, metric `strata:tool_calls_repaired_total`.
- **Malformed tool calls are logged** one JSON line each to `STRATA_TOOLCALL_FAIL_LOG` (outcome repaired / failed / off / unrepaired, declared tools, snippet).
- Strata #5901: finish tool arguments that quote an unclosed call. Strata #5883: keep a value whose text has `</parameter>` then `</function>`. Strata #3371: a CPU image reader starts on demand without elastic VRAM.

## Speedy 0.4.4 (2026-11-12)
- **Opt-in `--conversation-cache-rewind-keep N`** (Strata #5862): when a follow-up resumes the live conversation and at most N of its live tokens are past the resume point (shared prefix proven equal), the park is skipped instead of copying the whole K/V image. On 2x RTX 3080 (20 GB + 10 GB) at ~271K tokens, a follow-up whose answer was edited (splice miss): first token 0.90/0.80 s vs 1.50/0.92 s, parks 13 -> 1; normal follow-ups unchanged (0.23-0.25 s, splice hit). Default 0 parks as before.

## Speedy 0.4.3 (2026-11-12)
- **Follow-up turns reuse the cache after a reply ends early or hits max_tokens** (Strata #4677 + 13eedf55, now on every backend): a decode window commits only the rows the request can still emit (up to max_new and the first EOS), so the kept conversation is exactly what the client got. Fresh-session test: turn-2 cache hits 10/10 vs 4/10 (median time to first token 0.15 s vs 1.36 s). No emitted token changes; `STRATA_SERVE_STOP_BOUND=0` turns it off.
- **Route prior: no crash on a degenerate router row** (Strata #5655): a token whose router logits are all NaN/-inf keeps the router's own top-k instead of publishing a sentinel id (which was an out-of-bounds write and an "expert id is out of range" abort). Out-of-range ids are now reported with layer, entry and id.

## Speedy 0.4.2 (2026-11-12)
- **Opt-in `STRATA_PREFILL_ROUTED_STREAM_PP=1`** (with `STRATA_PREFILL_ROUTED_STREAM=1`): the routed expert stream also runs on the primary of a two-GPU prompt (chunks up to `STRATA_PREFILL_ROUTED_STREAM_PP_MAX`, default 4096 tokens). Same bits (in-process off/on gate); UD-IQ4_XS warm turns 4-16 % faster. On in the 2x RTX 3080 presets.
- **Strata #5805:** a bare `<function=...>` tool call inside the reasoning is read as a tool call; a lone `<parameter=arguments>` is unwrapped.
- **Strata #3517:** the conversation cache pin follows the system-prompt root, not chain item 0.

## Speedy 0.4.1 (2026-11-11)
- **Two windows + penalties: no deadlock when the penalty-history staging grows** (PR #9 by William Stone): the pinned staging is sized once at capture, instead of being freed and reallocated while the other window is in flight.
- UD-Q4_K_XL presets: STRATA_RP_SNAPSHOT=1 - the route prior reads the experts resident at load, so answers no longer depend on expert-swap timing (some UD-Q4_K_XL answers could end early on two windows; greedy test 48/48 x2 with it). Costs ~15-20 % decode on UD-Q4_K_XL; IQ4_XS/IQ3_S/Q2_0 unchanged.
- **Follow-up turns stay in the conversation cache:** an assistant turn sent back without its thinking is re-encoded from the exact ids the model generated (whitespace at the start of the thinking or at a cut-off end no longer breaks the prefix). Follow-up turn 2 cache hits 5/6 vs 3/6 in a fresh-session test; every turn that cannot be re-encoded logs `[strata] restored turn not spliced: <reason>`.
- **Tool steps a client merged into one assistant message are split back** to the steps the server generated, so the cached prompt is reused (`STRATA_TOOL_STEP_SPLIT=0` turns it off).
- **Opt-in `"canonical_tools": true`** renders tools in a stable order (sorted names and keys).

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
