<div align="center">

<img src="docs/assets/speedy-logo.webp" alt="Speedy LLM inference" width="520">

### Speed you can feel. Answers you can trust. Agents that don't get stuck.

Run big open models (Qwen3.8-Flash-Next, DeepSeek V4.1-Flash, Qwen3.6) locally. Built for coding agents: no infinite loops, no broken tool calls, no answers cut off.

[![License: Personal & Non-Commercial](https://img.shields.io/badge/License-Personal%20%26%20Non--Commercial-orange.svg)](LICENSE)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows-lightgrey)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20RTX%2020%E2%80%9350-76B900)
![Context](https://img.shields.io/badge/context-229K%20%7C%20grows%20to%201M-blue)
![Terminal-Bench 2.1 subset](https://img.shields.io/badge/Terminal--Bench%202.1%20subset-22%2F24-8A63D2)
[![Support](https://img.shields.io/badge/support-ko--fi-ff5e5b)](https://ko-fi.com/devdock)

</div>

Speedy runs large open-weight models on consumer NVIDIA cards and serves them over the OpenAI and Anthropic APIs.
Its main, most tuned model is **Qwen3.8-Flash-Next** (125B parameters, mixture-of-experts); DeepSeek V4.1-Flash and
Qwen3.6-35B-A3B run too ([docs/MODELS.md](docs/MODELS.md)). It is a measured, tuned fork of
[Strata](https://github.com/Niko1221/Strata).

Speed is half of it. The other half is what a local model usually gets wrong in real work: agents that loop, tool
calls that don't parse, thinking that runs out before the answer. Speedy fixes those on the server, so the coding
agent you already use works better without changing it.

## At a glance

Everything below was measured on one desktop PC with two used gaming cards (**RTX 3080 20 GB + RTX 3080 10 GB**),
running Qwen3.8-Flash-Next in its best-quality 4-bit file (UD-Q4_K_XL). Details and the comparison runs are in
[Benchmarks](#benchmarks).

| | Result | What it means |
|:---|:---|:---|
| **Terminal-Bench 2.1**, 24-task subset | **22 of 24** solved | a public benchmark of real terminal jobs (build, debug, configure servers), driven by the OpenCode agent |
| **Our coding suite**, 24 tasks | **24 of 24** solved in 12.9 min | bug fixes, features, refactors and browser UI work, checked by tests and screenshots |
| **Generation speed** | **155-170 tokens/s** | best-quality file on both cards; 115-122 tokens/s on the 20 GB card alone |
| **Reading speed** | **about 3,000 tokens/s** | a 55K-110K-token prompt on both cards (about 2,300 on one) |
| **Context** | **229K tokens, grows to 1M** | a session grown to 1M tokens still answered 9 of 9 recall questions |

## Quick start

**Windows**

1. Clone the repository, or download and extract a [release](https://github.com/DevDock-AI/Speedy/releases/latest).
2. Run `START-HERE.bat` (in PowerShell: `.\START-HERE.bat`).
3. Pick the model and the file size. On an RTX 20, 30, 40 or 50 card setup downloads a ready-made engine; on other
   cards it compiles from source (Visual Studio 2022 Build Tools and CUDA 13.x).

Expect somewhat lower reading (prefill) speed than on Linux. Building it yourself:
[docs/SOURCE_INSTALL.md](docs/SOURCE_INSTALL.md#from-source-windows), [docs/INSTALL.md](docs/INSTALL.md).

**Linux**

```bash
git clone https://github.com/DevDock-AI/Speedy.git speedy && cd speedy
export PATH=/usr/local/cuda/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_MMQ_TQ_OVERLAY=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_Q6K_EXPERTS=ON
cmake --build build --target strata -j
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

./setup.sh        # downloads the model, prepares it, writes a config tuned for your card and starts the server
```

Set `-DCMAKE_CUDA_ARCHITECTURES` to your card: 75 for RTX 20, 86 for RTX 30, 89 for RTX 40, 120 for RTX 50.
Full steps: [docs/BUILD-LINUX.md](docs/BUILD-LINUX.md). Choosing a model and file size: [docs/MODELS.md](docs/MODELS.md).

**Then point your client at it.** The server speaks the OpenAI and Anthropic APIs, so OpenCode, Claude Code, Codex CLI
and other clients work by setting the base URL ([docs/CODING_AGENTS.md](docs/CODING_AGENTS.md),
[docs/DETAILS.md](docs/DETAILS.md)). Set an API key before you listen on anything other than `127.0.0.1`
([SECURITY.md](SECURITY.md)).

**What you need:** an NVIDIA RTX 20-50 card and a lot of system RAM, because the part of the model that does not fit
on the card stays in RAM. For the 4-bit files of Qwen3.8-Flash-Next plan for about 128 GB; smaller files and smaller
models need less ([docs/MODELS.md](docs/MODELS.md)).

## Why Speedy

- 🎯 **Finishes real coding tasks.** A loop breaker notices an agent repeating a call or editing in circles, tells it the next step, and pauses the editing tools if it keeps going.
- 🔧 **Reliable tool calls.** Tool names are checked against what the client declared, malformed calls are repaired, and a call that contradicts what the model just said it would do is picked again.
- 🧠 **Thinks, then answers.** Medium reasoning with a 32K-token thinking budget by default, reminders at 60 % and 80 % of it so the model wraps up, and room reserved so a reply is never cut off before the answer.
- ⚡ **Fast.** 110-168 tokens/s on one RTX 3080 20 GB and 155-216 with a second card, depending on the model file size ([Speed](#speed)). A follow-up turn in a long conversation is read from cache instead of from the start.
- 📚 **Long context.** 229K tokens as the model ships, and the same config grows to 1M.
- 🔌 **Drop-in API.** OpenAI- and Anthropic-compatible. Images in chat are supported.

Every default can be turned off, and a request's own settings always win ([docs/SPEEDY_DEFAULTS.md](docs/SPEEDY_DEFAULTS.md)).

## Benchmarks

### Coding agents

Setup for every row: Speedy 0.5.4, Qwen3.8-Flash-Next UD-Q4_K_XL with the 1M-context config, RTX 3080 20 GB +
RTX 3080 10 GB, medium reasoning. The agent is [OpenCode](https://opencode.ai) talking to the real server, with
shell commands allowed without asking and our [AGENTS.md template](docs/agents/AGENTS.coding.md). Times are the
agent's working time for the whole set. One run per row unless it says otherwise.

| Test | Speedy 0.5.4 | Compared with |
|:---|:---|:---|
| **Terminal-Bench 2.1, 24 tasks.** A fixed subset of the public benchmark: coding, debugging, web/server setup and tool use, each with a time limit of 30 min or less. | **22 of 24** in 133 min | Speedy 0.5.3: 22 of 24 in 142 min (that run used OpenCode's default permissions and the earlier template) |
| **Terminal-Bench 2.1, 8 tasks.** Eight of the 24 that we rerun after every change, including the two that fail most often. | **8 of 8** in 37 min | Speedy 0.5.3, same client settings, two runs: 8 of 8 in 52 min and 5 of 8 in 59 min |
| **Our coding suite, 24 tasks.** Small bug fixes, features and refactors checked by tests, plus browser UI tasks checked by screenshots. | **24 of 24** in 12.9 min | Speedy 0.5.3, two runs: 23 of 24 both times (13.1 min). Same model with the loop breaker and the tool-step thinking limit off, two runs: 21 and 24 of 24 in about 20 min |

How to read it: these are single runs, and the same setup can move by a task or two between runs (0.5.3 scored
between 5 and 8 on the 8-task set across its runs). The 24-task subset is ours, so the score is not comparable with
the public Terminal-Bench leaderboard. What the numbers show is that 0.5.4 solves the same tasks as before and gets
there with less wasted time.

### Model quality

| Test | Result | Compared with |
|:---|:---|:---|
| **IFBench** (does the model follow instructions; 300 prompts, loose scoring per prompt) | **81.0** | 81.3 on the model's official card for the full-precision model |
| **Recall in a long prompt** (find a planted fact; 48 questions, prompts of more than 200K tokens) | **48 of 48** | |
| **Recall while a session grows to 1M tokens** | **9 of 9** | |
| **Tool calls well-formed** (24 requests that must call a tool) | **24 of 24**, none malformed | |

IFBench was run once with high reasoning on an earlier Speedy version (2026-11-01); the 0.3-point gap is inside the
roughly ±2.3-point noise of a 300-prompt run ([`evals/`](evals/)). The recall and tool-call checks are release checks
on the two-card setup above, last run for 0.5.3; 0.5.4 uses the same engine. Engine changes are adopted only after a
back-to-back speed comparison and a quality check: the output is compared token by token with the previous engine,
then the recall and tool-call checks run.

### Speed

All values are tokens per second for Qwen3.8-Flash-Next. **Decode** is how fast the answer is written (with the
model's built-in draft head guessing a few tokens ahead, which does not change the output). **Prefill** is how fast a
new prompt is read. **short** is a short chat prompt, **code** a coding prompt, **55K** and **110K** are prompts of
about that many tokens. Rows are model file sizes, from smallest to largest; larger files answer better and run slower.

**One RTX 3080 20 GB**

| Model file | Decode short | Decode code | Decode 55K | Decode 110K | Prefill short | Prefill code | Prefill 55K | Prefill 110K |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Q2_0** <sub>fastest</sub> | **147** | **168** | **146** | **146** | 1,229 | 837 | 2,703 | 2,421 |
| **IQ3_S** | **110** | **125** | **127** | **120** | 1,125 | 765 \* | 2,481 | 2,235 |
| **UD-IQ3_XXS** | **139** | **151** | **142** | **134** | 968 | 603 | 2,725 | 2,489 |
| **UD-IQ4_XS** <sub>good balance</sub> | **142** | **142** | **129** | **125** | 929 | 474 | 2,725 | 2,477 |
| **UD-Q4_K_XL** <sub>best quality</sub> | **122** | **121** | **122** | **115** | 729 | 346 | 2,409 | 2,200 |
| **Signal AP-Q4_K_XL** <sub>faster drop-in, see below</sub> | **136** | **148** | **138** | **127** | 860 | 440 | 2,414 | 2,216 |

<sub>Speedy 0.4.8 engine, mean of 2 runs. \* one run; the other measured 28 tokens/s (a measurement fault) and is left out. Configs: [`configs/cards/rtx3080-20gb/1x/`](configs/cards/rtx3080-20gb/1x/). Signal row: Speedy 0.5.4 with the Speedy MTP drafter, measured back to back with UD-Q4_K_XL on the stock drafter, which gave 114 / 112 / 113 / 114 decode in that run.</sub>

**RTX 3080 20 GB + RTX 3080 10 GB** (the model's layers are split across the two cards)

| Model file | Decode short | Decode code | Decode 55K | Decode 110K | Prefill short | Prefill code | Prefill 55K | Prefill 110K |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Q2_0** <sub>fastest</sub> | **200** | **216** | **214** | **186** | 1,929 | 2,041 | 3,908 | 3,941 |
| **IQ3_S** | **205** | **196** | **184** | **195** | 1,644 | 1,726 | 3,990 | 4,037 |
| **UD-IQ3_XXS** | **197** | **210** | **204** | **180** | 1,590 | 1,710 | 3,679 | 3,957 |
| **UD-IQ4_XS** <sub>good balance</sub> | **184** | **211** | **180** | **176** | 1,536 | 1,629 | 3,607 | 3,728 |
| **UD-Q4_K_XL** <sub>best quality</sub> | **168** | **170** | **161** | **155** | 1,226 | 1,300 | 2,950 | 3,150 |
| **Signal AP-Q4_K_XL** <sub>faster drop-in, see below</sub> | **187** | **218** | **176** | **145** | 804 | 452 | 3,134 | 3,006 |

<sub>Speedy 0.4.6 engine, a short screen of 1-2 runs per cell, so single cells can be off by a few percent. Configs: [`configs/cards/rtx3080-20gb/2x/`](configs/cards/rtx3080-20gb/2x/). Signal row: Speedy 0.5.4 with the Speedy MTP drafter, measured back to back with UD-Q4_K_XL on the stock drafter, which gave 154 / 174 / 162 / 138 decode in that run.</sub>

**Signal (faster drop-in).** [Signal-3.8-Flash-Next](https://huggingface.co/agentionai/Signal-3.8-Flash-Next-GGUF) by
agentionai is a light fine-tune of Qwen3.8-Flash-Next: same 125B size, same architecture, vision encoder and draft head,
and the same quality tier as UD-Q4_K_XL. It is trained to write more directly, with shorter reasoning, so the draft
head's guesses are accepted more often. Speedy pairs it with its own coding-tuned draft head,
[Qwen3.8-Flash-Next-Speedy-MTP](https://huggingface.co/DevDock-AI/Qwen3.8-Flash-Next-Speedy-MTP), which `setup.sh` downloads with every Qwen3.8-Flash-Next install
(about 10% more decode speed than the stock draft head on Signal). With the stock draft head on both, on the 20 GB + 10 GB pair (Speedy 0.5.4,
1M config, back-to-back runs) it decoded about 15% faster than UD-Q4_K_XL (prefill +1-2%) and scored the same on our
tests: coding suite 24 of 24, recall 48 of 48, tool calls 24 of 24. Licence: Qwen Community License (the model's own).
`setup.sh` does not download it yet; to try it by hand:

```sh
.venv/bin/hf download agentionai/Signal-3.8-Flash-Next-GGUF AP-Q4_K_XL/Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf --local-dir models/signal
.venv/bin/python tools/iq_pack.py --gguf models/signal/AP-Q4_K_XL/Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf --out packs/signal-xl --compat-bf16
```

Then, in a copy of your UD-Q4_K_XL config, point `--native` at the Signal GGUF and `--pack` at `packs/signal-xl`; keep
everything else, including `--mtp`, which already points at the Speedy drafter (`mtp/rt-tuned`; an update also switches a Signal config's `--mtp` to it). The GGUF is about 101 GB and the experts are read from it, so keep it on a fast SSD.

Test machine: Linux, Threadripper PRO 5975WX, 144 GB RAM, cards limited to 240 W (20 GB) and 280 W (10 GB), with the
optional [gpu-unlock](docs/GPU-UNLOCK.md) module; without it expect lower prefill and the same decode. Other GPUs, the
1M-token runs and the method: [docs/BENCHMARKS.md](docs/BENCHMARKS.md); results from other people's machines:
[docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md).

## Under the hood ([changelog](CHANGELOG.md))

- **Reasoning defaults**: effort `medium`, 32K thinking budget, reminders to wrap up, room reserved for the answer.
- **Loop breaker** for agents: repeated calls, edits that go back and forth and dummy writes get a note naming the next step; if the agent keeps at it, the editing tools pause until it does real work.
- **Right tool**: tool names are checked against what the client declared without steering the model's own choice, and a call that contradicts what the model just said it would do is picked again.
- **Tool-call repair**: broken calls are regenerated, argument shapes and types are fixed.
- **Thinking restored across turns**, so a tool-call conversation continues from cache instead of being read again.
- **Stall and timeout guards**: a stuck request or a client that stopped reading frees its slot.
- **Tuned draft head**: faster decode with identical output.
- **Setup**: a tuned single-GPU profile by default, a check that the context you pick fits your VRAM, the best model file that fits fully on big cards, and updates that bring existing installs to the same settings.

## Ready-made configs

`./setup.sh` writes a config for your card. If you have one of the cards below you can also start from the exact
configs we measured: one folder per GPU model under [`configs/cards/`](configs/cards/), each with a `CARD.md`
(hardware, build flags, numbers). Paths and GPU IDs are placeholders you fill in
([docs/CONFIG-PLACEHOLDERS.md](docs/CONFIG-PLACEHOLDERS.md)). Then start the server with it:

```bash
ulimit -l unlimited
.venv/bin/python serve/server.py --engine strata --config configs/mine/strata-ud-iq4_xs.json --port 8080
```

| Folder | Contents |
|---|---|
| [`rtx3080-20gb/1x/`](configs/cards/rtx3080-20gb/1x/) | one RTX 3080 20 GB: every model file size at 229K and at 1M context |
| [`rtx3080-20gb/2x/`](configs/cards/rtx3080-20gb/2x/) | RTX 3080 20 GB + RTX 3080 10 GB: UD-Q4_K_XL, UD-IQ4_XS, UD-IQ3_XXS with the current defaults, plus a `llama-swap.yaml` to switch between them |
| [`rtx3080-10gb/1x/`](configs/cards/rtx3080-10gb/1x/), [`2x/`](configs/cards/rtx3080-10gb/2x/) | one and two RTX 3080 10 GB, plus `llama-swap.yaml` |

Got a different card? Copy [`configs/cards/_template/`](configs/cards/_template/) ([how](configs/README.md#add-a-card)),
or point an AI coding agent at this repo and let it tune a config for your hardware ([docs/cards/](docs/cards/README.md)).
AMD builds through Strata's HIP backend ([docs/AMD_HIP.md](docs/AMD_HIP.md)); it compiles in CI but is untested.

## Updates

The server checks GitHub for a newer release in the background (anonymous, never installs anything by itself;
`STRATA_UPDATE_CHECK=0` turns it off). `./speedy-update` shows the release notes, asks, and updates
(Windows: `python serve\update.py`); `--check` only reports and `--rollback` goes back to what ran before.

## Documentation

| Document | Contents |
|---|---|
| [docs/BUILD-LINUX.md](docs/BUILD-LINUX.md), [docs/SOURCE_INSTALL.md](docs/SOURCE_INSTALL.md) | build and install, Linux and Windows |
| [docs/SPEEDY_DEFAULTS.md](docs/SPEEDY_DEFAULTS.md), [docs/CONFIGS.md](docs/CONFIGS.md) | the built-in defaults and every flag |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | method, hardware, full tables, noise floor |
| [docs/DETAILS.md](docs/DETAILS.md) | API routes, clients, images, server options |
| [docs/MULTI_GPU.md](docs/MULTI_GPU.md), [docs/TUNING-HISTORY.md](docs/TUNING-HISTORY.md) | using two or more cards; everything tried, adopted and rejected |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md), [docs/upstream/](docs/upstream) | when something breaks; Strata's original guides |

## Contributing

Results from other GPUs are very welcome, on Windows too: open an issue with the **Performance report** template.
Speed claims need a back-to-back comparison on the same machine, your hardware and a quality check ([CONTRIBUTING.md](CONTRIBUTING.md)); security issues go privately ([SECURITY.md](SECURITY.md)).

## License

Speedy Personal & Non-Commercial License v1.0 ([LICENSE](LICENSE)) for original Speedy contributions: free to run, study, modify and use for personal, non-commercial purposes (AI-assisted changes included); commercial use or public redistribution of modified versions needs written permission. Speedy is built on [Strata](https://github.com/Niko1221/Strata) (MIT, [licenses/Strata-MIT.txt](licenses/Strata-MIT.txt)); third-party components keep their own licenses (see [licenses/](licenses/)). Model weights are not included and keep their own licenses.

---

<sub><b>Credits.</b> Built on <a href="https://github.com/Niko1221/Strata">Strata</a> by Niko1221 and the Strata contributors.
Dual-GPU groundwork from Hardin22's Strata-DualGPU. Upstream work carried: lask3802 (#2689/#2690), Stefano Sinden / bitaccesscomau (#4283, #4119),
merbanan (#4039), havok2-htwo (#3559), cobra91, chimpera, moses-sun, wannagoxuf, ruibeikaa, Klaus Friedel, Gopi, Jackwwg83 (#4567),
Zac Chen / xyzzing (#3919), sergqwer (#2968, #2964), gopinath87607 (#3307, #3309, #3310), G0-0000 (#4226).
Ideas: StrataFast, architectds (IMMA attention), Cloudflare Code Mode.
Optional: <a href="https://github.com/Zagitta/gpu-unlock">gpu-unlock</a> (Zagitta). Kernels derived from <a href="https://github.com/ggml-org/llama.cpp">llama.cpp / ggml</a>.
Models: <a href="https://huggingface.co/Qwen">Qwen</a>, <a href="https://huggingface.co/unsloth">Unsloth</a> UD quants,
<a href="https://huggingface.co/ISTA-DASLab">ISTA-DASLab</a> GSQ-RCO quants. Benchmarks: <a href="https://github.com/allenai/IFBench">IFBench</a>, LiveCodeBench.
Full attribution: <a href="NOTICE">NOTICE</a>.</sub>

<div align="center"><br>If Speedy saved you some GPU money, <a href="https://ko-fi.com/devdock">buy me a coffee</a> ☕</div>
