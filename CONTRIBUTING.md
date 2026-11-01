# Contributing to Speedy

Speedy exists to be fast **with no quality loss**, and every claim it makes is measured. Contributions follow the same
rule: **a change lands with evidence, or it doesn't land.** These rules apply to everyone, including the maintainers.

Speedy is a fork of [Strata](https://github.com/Niko1221/Strata). Engine fixes that are not specific to Speedy belong
upstream first; Speedy carries them until they are merged there. Please follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Ground rules

- **Off by default until proven.** A new behaviour ships behind an environment variable or flag that defaults to off.
  It becomes a default only after it passes the bar below.
- **One hypothesis per PR.** A PR changes one thing you can measure. A bug fix doesn't carry a drive-by refactor.
- **Quality is not traded for speed.** A change that is faster but fails a quality gate is rejected, whatever the gain.
- **Small gains count.** A measured gain with no downside is kept, even below 5 %. A decode gain inside the noise band is not a gain.
- **Linux and Windows both stay buildable.** Don't add POSIX-only or Windows-only code outside the existing platform
  layers, and say which OS you built and ran on.
- **Credit people.** Cherry-pick upstream PRs with `git cherry-pick -x` so the author is kept, and add them to [NOTICE](NOTICE).
  Code taken from another project keeps its license (`licenses/`, plus a NOTICE entry).

## The evidence bar (every PR that claims speed or touches the engine)

1. **Tests.** Add or extend a ctest target. If you claim bitwise parity, a parity test must prove it.
2. **A same-window A/B on real hardware.**
   - The same build flags, config and environment except the one variable. Use the same prompts and token counts (`bench/speedbench.py`) and medium reasoning.
   - Run the control in the same window, in **ABBA** order (control, candidate, candidate, control), with at least two blocks.
   - Measure cold prefill and decode at about 60K / 120K / 200K, plus one warm follow-up turn after 120K (`bench/speedbench.py --warmlist 256,1024,2048,8192`).
   - Nothing else may use the GPU meanwhile. Write down every deviation.
   - Noise floor on the reference machine: prefill ±0.5 %, decode ±4-8 %.
3. **A quality gate for anything that can change output:**
   - **Bitwise:** an exact token and logprob comparison within the same engine launch.
   - **Otherwise:** tolerance gate v2 (token-level against the control at 60K and 120K) and the coding gate (8 delegation tasks plus 4 real-repository bug fixes).
   - **Changes to KV, attention or long context:** also the needle test (90K to 1M).
4. **Numbers in the PR**, filled into the template: pack, preset, arms, runs, and every table before and after.
5. **Full hardware listed.** The template's hardware table is mandatory: GPU model, VRAM, power limit and unlock state, PCIe gen/width per GPU, CPU, RAM size/speed/channels, storage, OS, driver and CUDA. Speed numbers mean nothing without it.
6. **Docs in the same PR.** Record the result, including rejections, in [SPEEDY.md](SPEEDY.md),
   [CHANGELOG.md](CHANGELOG.md) and [docs/TUNING-HISTORY.md](docs/TUNING-HISTORY.md). A new default also updates
   `src/program/speedy_defaults.hpp` (with `speedy_defaults_test`), the presets in `configs/`, and
   [docs/SPEEDY_DEFAULTS.md](docs/SPEEDY_DEFAULTS.md).

Docs-only and tooling PRs skip steps 2-3. Say so in the template.

## Build and test

```bash
# Linux
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_MMQ_TQ_OVERLAY=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build -j 1 --output-on-failure      # serial: GPU parity tests each need most of a 10 GB card
```

On Windows, use the same CMake lines from an **x64 Native Tools Command Prompt** (VS 2022 Build Tools + CUDA 13.x).
See [docs/SOURCE_INSTALL.md](docs/SOURCE_INSTALL.md#from-source-windows). Stop any running server before the GPU tests.

## Adding a GPU card

Tuned presets for a card you own are welcome even if they change nothing in the engine. Follow
[configs/README.md](configs/README.md#add-a-card): one folder per card under `configs/cards/`, a filled `CARD.md`
with measured numbers, and `python3 tools/check_card_configs.py` passing.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org): `<type>(<scope>): <subject>`. The subject is the changelog entry.

| Type | For |
|---|---|
| `perf` | a measured speed change (put the numbers in the body) |
| `feat` | a new capability |
| `fix` | a bug fix |
| `docs` | documentation only |
| `refactor` | structure changed, behaviour unchanged (bitwise) |
| `test` | tests only |
| `chore` | tooling, CI, release |

Scopes: `prefill`, `decode`, `split`, `kv`, `qsa`, `mtp`, `serve`, `presets`, `bench`, `evals`, `windows`, `docs`.

- **Imperative mood, lowercase, no trailing period.** Example: `perf(split): overlap decode windows across stages`.
- **The body says why, plus the measured effect.** The diff already shows what changed.
- **Trailers credit people only.** Use `Co-authored-by:` for the human authors whose work is in the commit. Don't add tool or assistant trailers.

## Pull requests

`main` is protected: everything lands through a reviewed pull request. Squash merges use the PR title as the commit
subject, so the title follows the commit convention. The [PR template](.github/pull_request_template.md) is required;
**a PR without its numbers is closed until they are added.**

## Hygiene and security

- **Never commit personal or machine data.** That means secrets, API keys, tokens, hostnames, LAN or VPN addresses, GPU UUIDs, home-directory or machine paths, personal emails, editor or assistant settings, or chat logs. Presets use the placeholders in [docs/CONFIG-PLACEHOLDERS.md](docs/CONFIG-PLACEHOLDERS.md).
- Benchmark output goes to `bench/results-local/` (git-ignored), and build trees to `build*/`.
- **In workflows, never interpolate `${{ ... }}` into a `run:` block.** Pass values through `env:`. Give each job the narrowest `permissions:` it needs.
- In Python, pass an argv list to `subprocess` (no `shell=True`). In shell, quote every expansion.

Security issues go through private reporting, not issues: see [SECURITY.md](SECURITY.md).
