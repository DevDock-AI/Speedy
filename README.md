<div align="center">

<img src="docs/assets/speedy-logo.webp" alt="Speedy LLM inference" width="560">

### A 125B mixture-of-experts model at 200K+ context, on consumer RTX 3080 cards.

Speedy runs **Qwen3.8-Flash-Next** (125B MoE) with a **229K-token native context** and a **1M-token tier** on one or two
**RTX 3080 10 GB** cards. It is a measured, tuned fork of [Strata](https://github.com/Niko1221/Strata). Every change in
it was adopted only after a same-window A/B and a quality gate.

**Developed and tested on RTX 3080s, likely fine on many other NVIDIA cards.** Every number below comes from RTX 3080
10 GB cards, because that is the hardware we own. The engine itself needs any NVIDIA GPU with compute capability 8.0+
(Ampere or newer: RTX 30xx, 40xx, 50xx, A-series / workstation cards), so other cards should work. Set
`-DCMAKE_CUDA_ARCHITECTURES` to your card (80 A100, 86 RTX 30xx, 89 RTX 40xx, 120 RTX 50xx). Cards with more VRAM will fit
more experts and likely run faster, but the presets are tuned for 10 GB and haven't been measured elsewhere.
**AMD** (RX 7900 / RDNA3, RDNA4) builds from source through Strata's HIP backend ([docs/AMD_HIP.md](docs/AMD_HIP.md)); it compiles in our CI but we have not run it. Results from other GPUs are very welcome: add a card folder under [`configs/cards/`](configs/README.md#add-a-card).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![GPU](https://img.shields.io/badge/GPU-RTX%203080%2010%20GB%20(1%20or%202)-76B900)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows-lightgrey)
![CUDA](https://img.shields.io/badge/CUDA-13.x-76B900)
![Context](https://img.shields.io/badge/context-229K%20native%20%7C%201M%20tier-blue)
![gpu-unlock](https://img.shields.io/badge/measured%20with-gpu--unlock-orange)
![IFBench](https://img.shields.io/badge/IFBench-81.0%20(Q4__K__XL)-8A63D2)
[![Support](https://img.shields.io/badge/support-ko--fi-ff5e5b)](https://ko-fi.com/devdock)

</div>

---

## Contents

| | |
|---|---|
| **[Results at a glance](#results-at-a-glance)** | Speed per quant, one GPU and two GPUs |
| **[Quality](#quality)** | IFBench, long-context recall, quality gates |
| **[Quick start: Linux](#quick-start-linux)** | Build, presets, serve |
| **[Quick start: Windows](#quick-start-windows)** | Build and run on Windows 10/11 |
| **[Presets: replicate our numbers](#presets-replicate-our-numbers)** | The exact production configs |
| **[Documentation](#documentation)** | Where everything lives |
| **[Contributing](#contributing)** | Evidence-first pull requests |

## Results at a glance

Each value is the **average ± half the spread** across the measured runs. All runs use medium reasoning and real prompts.

- **Prefill** is a cold prompt read at about 60K / 120K / 200K tokens.
- **Decode** is generation speed with MTP speculative decoding.
- **Cold 200K** is the time to read a fresh 200K-token prompt.
- **Follow-up** is the time to read a short follow-up (1-2K new tokens) in a 120K-token conversation where 99 % of the prompt is cached.

> [!IMPORTANT]
> **All numbers were measured with both RTX 3080 cards unlocked by [gpu-unlock](https://github.com/Zagitta/gpu-unlock)**
> (Zagitta). This is a third-party Linux kernel module that removes the LHR limit on the FP16/BF16 tensor rate. Speedy
> doesn't bundle or require it.
>
> | Unlock effect | Locked | Unlocked |
> |:---|---:|---:|
> | FP16 GEMM | 62 TFLOPS | **106 TFLOPS** (+70 %) |
> | Prefill | baseline | **+10-12 %** (Q2_0), more on FP16 prompt paths such as UD-Q4_K_XL |
> | Decode | unchanged | unchanged (bandwidth-bound) |
>
> Output is bit-identical locked and unlocked. Without the module, expect correspondingly lower prefill. Setup and
> caveats are in [docs/GPU-UNLOCK.md](docs/GPU-UNLOCK.md).
>
> **Measured on:** Lenovo ThinkStation P620
>
> | Part | Spec |
> |:---|:---|
> | GPUs | 2× RTX 3080 10 GB, PCIe 4.0 x16 each, no NVLink, caps 320 W / 280 W, both unlocked |
> | CPU | AMD Threadripper PRO 5975WX, 32 cores / 64 threads |
> | RAM | 128 GB DDR4-3200, 8 channels (8× 16 GB) |
> | Storage | 2 TB Samsung PM9A1 NVMe (PCIe 4.0) |
> | OS | Debian 13, kernel 6.12, NVIDIA 615.71, CUDA 13.4 |
> | <ins>One</ins>-GPU numbers | same machine, first card only |
>
> You need about **128 GB of RAM** for the Q4-class quants (Q2_0 needs less) and 60-120 GB of disk per quant.

### <ins>One</ins> RTX 3080 10 GB

| Quant | Prefill 60K<br>(tok/s) | Prefill 120K<br>(tok/s) | Prefill 200K<br>(tok/s) | Decode<br>(tok/s) | Cold 200K<br>(s) | Follow-up<br>(s) |
|:---|---:|---:|---:|---:|---:|---:|
| **Q2_0** <sub>fastest</sub> | 2,824 | 2,689 | 2,562 | **152** ±13 | 78 | 1.9 ±0.4 |
| **IQ3_S** | 2,347 | 2,303 | 2,248 | **130** ±11 | 89 | 1.75 ±0.05 |
| **UD-IQ4_XS** <sub>quality</sub> | 1,820 | 1,802 | 1,806 | **107** ±15 | 111 | 2.75 ±0.25 |
| **UD-Q4_K_XL** <sub>best</sub> | 1,134 | 1,125 | 1,090 | **93** ±17 | 183 | 2.9 |

### <ins>Two</ins> RTX 3080 10 GB (layer split)

| Quant | Prefill 60K<br>(tok/s) | Prefill 120K<br>(tok/s) | Prefill 200K<br>(tok/s) | Decode<br>(tok/s) | Cold 200K<br>(s) | Follow-up<br>(s) |
|:---|---:|---:|---:|---:|---:|---:|
| **Q2_0** <sub>fastest</sub> | 4,151 | 4,270 | 4,138 | **187** ±24 | 49 | 1.8 ±0.2 |
| **IQ3_S** | 4,074 | 4,284 | 4,264 | **163** ±8 | 47 | 1.6 |
| **UD-IQ4_XS** <sub>quality</sub> | 3,297 | 3,400 | 3,397 | **141** ±14 | 59 | 2.05 ±0.25 |
| **UD-Q4_K_XL** <sub>best</sub> | 2,994 | 3,118 | 3,115 | **111** ±10 | 65 | 2.75 ±0.05 |

### Against the starting point

These compare Strata 0.1.45 on the same machine with Speedy on two GPUs.

| Quant | Prefill gain | Decode gain |
|:---|---:|---:|
| **Q2_0** | +65 % ±8 | +43 % ±8 |
| **IQ3_S** | +138 % ±8 | +45 % |
| **UD-IQ4_XS** | +156 % ±11 | +55 % ±5 |
| **UD-Q4_K_XL** | +212 % ±7 | +20 % |

**1M-token context (two GPUs).** A growing coding session reached **1,017,288 tokens in 26 turns (18 min)** with no
errors. It passed all 9 recall checks, with facts planted early and asked about at 163K to 976K tokens. 93-96 % of each turn was served from cache.
Needle-in-a-haystack recall was **47/48** single facts across the depth range up to about 1M tokens.

<sub>Method: same packs and configs except the one variable under test, the control in the same window, ABBA order.
Noise floor: prefill ±0.5 %, decode ±4-8 %. Full tables and every A/B: [docs/BENCHMARKS.md](docs/BENCHMARKS.md),
[docs/EDITION-1GPU.md](docs/EDITION-1GPU.md), [docs/EDITION-2GPU.md](docs/EDITION-2GPU.md).</sub>

## Quality

| Benchmark | Speedy, UD-Q4_K_XL (about 4.5 bits per weight) | Qwen3.8-Flash-Next, official card (BF16) |
|---|---|---|
| **IFBench** (300 prompts, prompt-level loose) | **81.0** (strict 78.7) | 81.3 |

The two scores are within the noise of a 300-prompt run, which is about ±2.3 points. Our run had a 32K reasoning cap and a single sample. It
generated 2.39M tokens over 8 h at 83 t/s, and no answer hit the cap. The harness is in [`evals/`](evals/).

Every adopted engine change also had to pass:
- **Tolerance gate v2:** token-by-token comparison against the reference across 60K and 120K prompts.
- **Coding gate:** 8 delegation tasks and 4 real-repository bug fixes.
- **Needle recall** at 90K to 1M tokens, for any change that touches attention.

## Quick start: Linux

```bash
# 1. Build (details: docs/BUILD-LINUX.md)
git clone https://github.com/DevDock-AI/Speedy.git speedy && cd speedy
export PATH=/usr/local/cuda/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_MMQ_TQ_OVERLAY=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_Q6K_EXPERTS=ON
cmake --build build --target strata -j
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 2. Models: setup downloads the GGUF, builds the packs and the MTP draft head (docs/MODELS.md, docs/UNSLOTH_Q4.md)
./setup.sh

# 3. Take a production preset and fill in your paths (docs/CONFIG-PLACEHOLDERS.md)
mkdir -p configs/mine && cp configs/cards/rtx3080-10gb/2x/strata-ud-iq4_xs-2gpu-pw2.json configs/mine/
sed -i -e "s#<SPEEDY_DIR>#$PWD#g" -e "s#<MODELS_DIR>#/models/qwen38#g" -e "s#<STRATA_DATA>#/models/strata-data#g" \
       -e "s#<LOG_DIR>#$HOME/speedy-logs#g" \
       -e "s#<GPU1_UUID>#$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)#g" \
       -e "s#<GPU2_UUID>#$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader)#g" configs/mine/*.json

# 4. OpenAI- and Anthropic-compatible server
ulimit -l unlimited
.venv/bin/python serve/server.py --engine strata --config configs/mine/strata-ud-iq4_xs-2gpu-pw2.json --port 8080
```

With **one GPU**, use the presets in `configs/cards/rtx3080-10gb/1x/` instead. Several quants can share one port through
llama-swap; see `configs/production-*/llama-swap.yaml` and [docs/LLAMA_SWAP.md](docs/LLAMA_SWAP.md).
Set an API key before you listen on anything other than `127.0.0.1` ([SECURITY.md](SECURITY.md)).

## Quick start: Windows

Speedy keeps Strata's Windows support: the same engine source, the same `serve/` server and the same setup scripts.

**Option A, setup does everything:**
1. Clone the repository, or download and extract a release.
2. Run `START-HERE.bat`. In PowerShell, run `.\START-HERE.bat`.
3. Pick the model and quant. On an **RTX 30** card setup downloads the ready-made engine (`strata-windows-x64-rtx30.zip` from the [latest release](https://github.com/DevDock-AI/Speedy/releases/latest)); on other cards it compiles from source, so install the build tools in option B.

**Option B, build the engine yourself:**
1. Install the **Visual Studio 2022 Build Tools** (C++ workload) and **CUDA 13.x** (13.3 or newer also works with VS 2026).
2. Open the **x64 Native Tools Command Prompt** and run:
```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 ^
      -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_MMQ_TQ_OVERLAY=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_Q6K_EXPERTS=ON
cmake --build build --target strata
```
With a CUDA **12.x** toolkit, add `-DCMAKE_CUDA_RUNTIME_LIBRARY=Shared`: MSVC refuses the default static runtime there.

3. Copy a preset from `configs/cards/rtx3080-10gb/1x/` or `configs/cards/rtx3080-10gb/2x/`. Replace the placeholders with Windows paths (use forward slashes or escaped backslashes in JSON), set `"exe"` to `build/strata.exe`, and start it with `python serve\server.py --engine strata --config <preset> --port 8080`.

**Windows notes:**
- The reference numbers above are from Linux. Under Windows (WDDM), expect somewhat lower prefill.
- The optional gpu-unlock module is Linux-only. Without it, an LHR card runs at its stock FP16 tensor rate.
- Pinned host memory is limited by Windows itself, not by `ulimit`. If the engine warns about pinned memory, lower `--kv-resident` or `--conversation-cache-mib`.
- Windows results are welcome: open an issue with the **Performance report** template.

Strata's desktop app and its auto-update channel are not published for Speedy; use the source install.
Full Windows steps: [docs/SOURCE_INSTALL.md](docs/SOURCE_INSTALL.md#from-source-windows) and
[docs/INSTALL.md](docs/INSTALL.md).

## Presets: replicate our numbers

Presets live in **one folder per GPU model** under [`configs/cards/`](configs/cards/), with a `CARD.md` holding that
card's hardware, build flags and measured numbers. The RTX 3080 presets are the production configs exactly as they ran;
only machine paths and GPU UUIDs are replaced by placeholders.

**Tuned Speedy on another card?** Copy [`configs/cards/_template/`](configs/cards/_template/), add your presets and
numbers, and open a PR. See [configs/README.md](configs/README.md#add-a-card).

| Folder | Contents |
|---|---|
| [`configs/cards/rtx3080-10gb/2x/`](configs/cards/rtx3080-10gb/2x/) | `strata-<quant>-2gpu-pw2.json` (229K native) and `strata-<quant>-1m-2gpu.json` (1M tier) for Q2_0, IQ3_S, UD-IQ4_XS, UD-Q4_K_XL, plus `llama-swap.yaml` |
| [`configs/cards/rtx3080-10gb/1x/`](configs/cards/rtx3080-10gb/1x/) | the single-GPU production presets, plus `llama-swap.yaml` |
| [`configs/cards/rtx3080-10gb/2x/parallel2/`](configs/cards/rtx3080-10gb/2x/parallel2/) | `"parallel": 2` variants (two requests decoded together). **Slower on 2× 10 GB**, see below |

The key two-GPU settings are:
- `--layer-split 26`
- `--trim-stage-weights --vram-reserve-later-mib 500`
- `--pipeline-windows 2` with `STRATA_PIPELINE_PENALTIES=1` (decode overlap)
- `STRATA_PREFILL_HELP=1`
- `STRATA_PREFILL_STREAM_MIN=512` (IQ4_XS)

What every flag does: [docs/CONFIGS.md](docs/CONFIGS.md). Built-in defaults: [docs/SPEEDY_DEFAULTS.md](docs/SPEEDY_DEFAULTS.md).

**About `parallel: 2`:** on 2× 10 GB, two clients sending at once finished in **42-48 s** with batch slots, against **33 s**
when the requests simply queue (Q2_0). With slots on, a single user also loses 15-19 % decode. The two conversations route to
different experts, so batching saves little while the slots take expert-cache VRAM. It may pay off on cards with more VRAM.

## Documentation

| Document | Contents |
|---|---|
| [docs/BUILD-LINUX.md](docs/BUILD-LINUX.md), [docs/SOURCE_INSTALL.md](docs/SOURCE_INSTALL.md) | build and install, Linux and Windows |
| [docs/CONFIGS.md](docs/CONFIGS.md), [docs/CONFIG-PLACEHOLDERS.md](docs/CONFIG-PLACEHOLDERS.md) | every flag, and how to fill in the presets |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | method, hardware, full tables, noise floor |
| [docs/EDITION-1GPU.md](docs/EDITION-1GPU.md), [docs/EDITION-2GPU.md](docs/EDITION-2GPU.md) | per-edition details and A/Bs |
| [docs/TUNING-HISTORY.md](docs/TUNING-HISTORY.md) | everything tried, adopted and rejected |
| [docs/MULTI_GPU.md](docs/MULTI_GPU.md) | layer split and second-GPU options |
| [docs/upstream/](docs/upstream) | Strata's original README and guides |

## Contributing

Pull requests are welcome, and speed claims need evidence: numbers from an A/B in the same window, your hardware, and a
quality-gate result. See [CONTRIBUTING.md](CONTRIBUTING.md). Report security issues privately ([SECURITY.md](SECURITY.md)).

## License

MIT ([LICENSE](LICENSE)). Model weights are not included and keep their own licenses.

---

<sub><b>Credits.</b> Built on <a href="https://github.com/Niko1221/Strata">Strata</a> by Niko1221 and the Strata contributors.
Dual-GPU groundwork from Hardin22's Strata-DualGPU. Upstream work carried: lask3802 (#2689/#2690), Stefano Sinden / bitaccesscomau (#4283, #4119),
merbanan (#4039), havok2-htwo (#3559), cobra91, chimpera, moses-sun, wannagoxuf, ruibeikaa, Klaus Friedel, Gopi, Jackwwg83 (#4567),
Zac Chen / xyzzing (#3919), sergqwer (#2968, #2964), gopinath87607 (#3307, #3309, #3310), G0-0000 (#4226).
Ideas: <a href="https://github.com/jesdga95/basalt">Basalt</a> (Jesús Gulfo), StrataFast, architectds (IMMA attention), Cloudflare Code Mode.
Optional: <a href="https://github.com/Zagitta/gpu-unlock">gpu-unlock</a> (Zagitta). Kernels derived from <a href="https://github.com/ggml-org/llama.cpp">llama.cpp / ggml</a>.
Models: <a href="https://huggingface.co/Qwen">Qwen</a>, <a href="https://huggingface.co/unsloth">Unsloth</a> UD quants,
<a href="https://huggingface.co/ISTA-DASLab">ISTA-DASLab</a> GSQ-RCO quants. Benchmarks: <a href="https://github.com/allenai/IFBench">IFBench</a>, LiveCodeBench.
Full attribution: <a href="NOTICE">NOTICE</a>.</sub>

<div align="center">
<br>
If Speedy saved you some GPU money, <a href="https://ko-fi.com/devdock">buy me a coffee</a> ☕
</div>
