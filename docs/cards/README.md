# Per-GPU-family tuning guides

Guides for people and AI agents setting up Speedy (Qwen3.8-Flash-Next) on NVIDIA cards. Each guide says which preset to
start from, what the engine turns on by itself for that architecture, and what is measured versus untested.

| Card family | Compute capability | Engine build (release) | Status | Guide |
|:---|:---|:---|:---|:---|
| RTX 30 (Ampere) | 8.6 (sm_86) | `rtx30` | **production** (2x and 1x RTX 3080 10 GB) | [RTX30.md](RTX30.md) |
| RTX 50 (Blackwell) | 12.0 (sm_120) | `rtx50` | community (not measured by maintainers) | [RTX50.md](RTX50.md) |
| RTX 40 (Ada) | 8.9 (sm_89) | `rtx40` | untested | [RTX40.md](RTX40.md) |
| RTX 20 (Turing) | 7.5 (sm_75) | `rtx20` | untested (built, not run) | [RTX20.md](RTX20.md) |

Presets live in [`configs/cards/`](../../configs/cards/); new cards start from
[`configs/cards/_template/`](../../configs/cards/_template/).
