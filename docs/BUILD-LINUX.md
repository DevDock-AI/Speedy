# Building Speedy on Linux (Debian 13, CUDA 13.4, RTX 3080 / sm_86)

Reference machine: Threadripper PRO 5975WX (Zen 3, AVX2, **no AVX-512**), 128 GB DDR4 (8 channels),
RTX 3080 10 GB LHR (10de:2216) on PCIe 4.0 x16, ReBAR on (BAR1 16 GiB), NVIDIA 615.71.10, kernel 6.12, GCC 14.2.

## Engine
```bash
# CUDA from NVIDIA's apt repo (cuda-toolkit-13-4); nvcc is not on PATH by default
export PATH=/usr/local/cuda/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_NATIVE_EXPERTS=ON
cmake --build build --target strata -j 24
```
The CMake cache options of the production build (everything else default): `STRATA_MMQ_KQUANTS=ON` (K-quant prompt
kernels, `libstrata_kq.so`, used by `STRATA_MMQ_KQ=1`), `STRATA_Q6K_EXPERTS=ON`, `STRATA_NATIVE_EXPERTS=ON`,
`STRATA_PREFILL_MMQ=OFF`, `STRATA_MMQ_TAB=OFF`, `STRATA_PORTABLE=OFF`.

## Tests
```bash
# CPU + CUDA parity targets (ctest; GPU parity tests need the card, not the model)
cmake -S . -B build-tests -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DSTRATA_BUILD_TESTS=ON
cmake --build build-tests -j 24
ctest --test-dir build-tests -j 1   # serial: the GPU parity tests each need most of a 10 GB card (stop any running server)
# ThreadSanitizer build of the CPU expert pool (no CUDA)
cmake -S . -B build-tsan -DCMAKE_BUILD_TYPE=RelWithDebInfo -DSTRATA_BUILD_TESTS=ON \
  -DCMAKE_CXX_FLAGS=-fsanitize=thread -DCMAKE_C_FLAGS=-fsanitize=thread -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=thread
cmake --build build-tsan --target pool_test pool_tasks_test pool_stress pool_dep_test
```

## Python side
`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`; `data/expert-profile.bin` ships in the repo.
Packs (`Strata-data/packs/<quant>`, `mtp/rt`) are built by Strata's setup and are accepted unchanged across
Windows/Linux.

## Runtime environment
- `ulimit -l unlimited` (or `LimitMEMLOCK=infinity` in the systemd unit): the expert arena is mlocked and ~28 GB
  is `cudaHostRegister`ed.
- Transparent huge pages `always` (the arena gets ~34 GB AnonHugePages).
- Background containers isolated in a cgroup slice (`AllowedCPUs` on SMT siblings of one CCD, low CPU/IO weight);
  the engine unit gets `CPUWeight=1000`, `OOMScoreAdjust=-900`. Measured: a 16-thread load in the slice did not
  change speed.

## Tried on this machine and rejected (no gain or a cost)
| Change | Result |
|---|---|
| `-O3 -march=znver3` + LTO rebuild | no gain (the ggml CPU parts are already `-march=native`) |
| CPU governor `performance` (amd-pstate EPP already performance) | no gain |
| GPU clock lock `-lgc 2100 -lmc 9501` | no gain (prefill is power-capped at 320 W) |
| 2 MB hugetlbfs pages for the arena | no gain |
| mimalloc / jemalloc via LD_PRELOAD | noise |
| SMT off, C2 idle state off | no gain |
| `--vram-reserve-mib` 400-450 | prefill +15 % on a 16K benchmark only (chunk 7680 -> 7936), but OOM at 127K context -> rejected; 256 fails at start |
| `--ple-io ram --no-ple-prefetch` | no gain, -26 GB RAM |
| `nvidia-drm.modeset=0` (console off the GPU) | not tested (owner decision: risk > gain) |
| `pcie_aspm=off` | moot (ASPM already disabled in BIOS) |

Adopted at the platform level (2026-10-29): 8th DIMM (decode +22-55 %), BIOS ReBAR + PreferredIO (prefill +1.5 %),
driver 615.71.10, and the optional third-party gpu-unlock module ([GPU-UNLOCK.md](GPU-UNLOCK.md); FP16/BF16 tensor throughput 62 -> 106 TFLOPS, Q2 prefill +10-12 %).
See docs/TUNING-HISTORY.md.

## Known test results (v0.2.7.1 base, RTX 3080 10 GB, CUDA 13.4)
`ctest -j 1` with the GPU free: all tests pass except three that do not depend on Speedy's changes:
`ple_parity` (needs the Q2_0 GGUF at `../../Q2_0/`, a developer-machine path), `isa_guard_check` (two AVX-512
reference functions in `libstrata_kernels_cpu.a` flagged by the ISA lint under GCC 14) and `cuda_spill_guard` (two
fused native prompt kernels use 48 bytes of stack on sm_86 with nvcc 13.4). Running ctest with `-j 6` on a 10 GB card
fails ~80 GPU tests with `out of memory`; that is the parallelism, not a bug.

## See also
[CONFIGS.md](CONFIGS.md) (presets), [SPEEDY_DEFAULTS.md](SPEEDY_DEFAULTS.md) (built-in defaults),
[TUNING-HISTORY.md](TUNING-HISTORY.md) (what was tried), [GPU-UNLOCK.md](GPU-UNLOCK.md) (optional FP16 unlock),
[../CONTRIBUTING.md](../CONTRIBUTING.md) (tests and the A/B rule).
