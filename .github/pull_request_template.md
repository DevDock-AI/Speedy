<!-- Read CONTRIBUTING.md first. A PR without numbers is closed until they are added. -->
## What changes
<!-- One or two sentences, from the user's side. -->

## Why
<!-- The problem or hypothesis; link an issue if there is one. -->

## Type
- [ ] perf (measured speed change)   - [ ] fix   - [ ] feat   - [ ] docs / tooling only (skip Evidence and Quality)

## Hardware and setup (required)
<!-- Mandatory for every perf, fix and feat PR. A PR with any empty row is closed until it is filled.
     Helpers: Linux `nvidia-smi -q | grep -iE "product name|power limit|link (gen|width)"`, `lscpu`, `free -g`, `sudo dmidecode -t memory`;
     Windows `nvidia-smi -q`, `systeminfo`, `wmic memorychip get capacity,speed`. No serial numbers, UUIDs or hostnames. -->

| Part | Spec |
|:---|:---|
| GPU(s) | model, VRAM, count |
| GPU power limit / clocks | e.g. 320 W, stock or OC/UV |
| GPU unlock | gpu-unlock on / off |
| PCIe | gen and width per GPU as negotiated (e.g. 4.0 x16 / 4.0 x8), NVLink/P2P yes/no |
| CPU | model, cores/threads, AVX2 / AVX-512 |
| RAM | total, type and speed, channels populated (e.g. 128 GB DDR4-3200, 8 ch) |
| Storage | drive model and interface the packs load from (e.g. NVMe PCIe 4.0) |
| OS / driver / CUDA | e.g. Debian 13 kernel 6.12 / 615.71 / 13.4, or Windows 11 / driver / CUDA |
| Build | commit, CMake flags |
| Run | pack/quant, preset, any STRATA_* env vars |

## Evidence (same-window ABBA)
<!-- Arms (control vs candidate), run order, number of blocks, prompt sizes. Paste bench/speedbench.py output. -->

| Metric | Control | Candidate | Change |
|---|---|---|---|
| Prefill 60K / 120K / 200K (t/s) | | | |
| Decode (t/s) | | | |
| Warm turn after 120K (s) | | | |
| VRAM / expert slots per GPU | | | |

## Quality
<!-- One of: bitwise (how checked, same engine launch) / tolerance gate v2 + coding gate results / needle test results. -->

## Tests
<!-- ctest / pytest commands run and their results; what you could not run and why. -->

## Checklist
- [ ] Hardware and setup table fully filled in
- [ ] Off by default, or the A/B and quality gate above justify a new default
- [ ] One hypothesis in this PR
- [ ] Builds on Linux; Windows impact considered (say if built/run there)
- [ ] SPEEDY.md / CHANGELOG.md / docs/TUNING-HISTORY.md updated (including if rejected)
- [ ] Upstream authors credited (cherry-pick -x, NOTICE)
- [ ] No secrets, personal data, machine paths, GPU UUIDs or assistant/tool attribution in code, configs, commits or screenshots
