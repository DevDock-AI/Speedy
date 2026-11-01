<!-- Read CONTRIBUTING.md first. A PR without numbers is closed until they are added. -->
## What changes
<!-- One or two sentences, from the user's side. -->

## Why
<!-- The problem or hypothesis; link an issue if there is one. -->

## Type
- [ ] perf (measured speed change)   - [ ] fix   - [ ] feat   - [ ] docs / tooling only (skip Evidence and Quality)

## Hardware and setup
<!-- GPU(s) + VRAM + power limit, CPU, RAM, PCIe, OS (Linux / Windows), driver, CUDA, commit, pack/quant, preset, STRATA_* vars. -->

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
- [ ] Off by default, or the A/B and quality gate above justify a new default
- [ ] One hypothesis in this PR
- [ ] Builds on Linux; Windows impact considered (say if built/run there)
- [ ] SPEEDY.md / CHANGELOG.md / docs/TUNING-HISTORY.md updated (including if rejected)
- [ ] Upstream authors credited (cherry-pick -x, NOTICE)
- [ ] No secrets, personal data, machine paths, GPU UUIDs or assistant/tool attribution in code, configs, commits or screenshots
