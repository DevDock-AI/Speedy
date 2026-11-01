# Optional: unlocking the FP16 tensor rate of an LHR card (gpu-unlock)

Speedy does **not** include, require or install any GPU unlock. This page describes an optional, third-party Linux
kernel module, **[gpu-unlock](https://github.com/Zagitta/gpu-unlock)** (by Zagitta, under its own license), that was
installed on the reference machine. All "unlocked" numbers in this repository were measured with it; the locked
numbers are listed beside them, so you can see what to expect without it.

## What it changed on the reference machine
RTX 3080 10 GB LHR (PCI ID 10de:2216), driver 615.71.10, kernel 6.12, 320 W power limit
([BENCHMARKS.md](BENCHMARKS.md)):

| | locked | unlocked |
|---|---|---|
| cuBLAS FP16 GEMM, FP32 accumulate (8192^3) | 62.4 TFLOPS | 106.2 TFLOPS (+70 %) |
| BF16 / TF32 | 62.3 / 30.8 TFLOPS | 108.3 / 52.9 TFLOPS |
| INT8 GEMM | 194 TOPS | 199 TOPS (unchanged) |
| Q2_0 prefill 16K / 32K | 2047 / 2283 t/s | 2260 / 2567 t/s (+10 / +12 %) |
| Decode | 143-149 t/s | 140-150 t/s (unchanged: bandwidth-bound) |

GEMM output checksums were bit-identical locked vs unlocked. No Xid errors in about an hour of mixed load (GPU max
74 C, power-capped at 320 W, no thermal throttling). The gain is largest on the FP16 prompt paths (UD-Q4_K_XL); the
fused INT8 paths of Q2_0 / IQ3_S gain less.

## How it was used (outline; follow the project's own documentation)
1. Check that your card's PCI device ID is supported by the module.
2. Build and install the module and its firmware as the project describes (on the reference machine: a DKMS package;
   the firmware step downloads pinned NVIDIA sources and checks their hashes). Secure Boot was off; with Secure Boot on
   you must sign the module.
3. Enable it per device with a kernel parameter naming the GPU's PCI address, for example
   `gpu_unlock.targets=0000:XX:00.0`, first in a separate, one-shot boot entry; keep a boot entry without the
   parameter as a fallback.
4. After boot, check the kernel log for the module's completion line and that the NVIDIA driver bound normally
   (`nvidia-smi`), then compare a GEMM benchmark with your locked numbers.

## Caveats
- It is third-party kernel code that changes how the GPU runs. Use it at your own risk; it may affect your warranty,
  and a failure can hang the boot. Have a fallback boot entry and console access.
- After a firmware timeout, power the machine off fully rather than rebooting.
- It depends on specific GPU IDs, driver versions and kernels; re-check after any driver or kernel update.
- It does not help decode, which is limited by RAM and PCIe bandwidth.
- Speedy's defaults and presets do not assume it. Without it, expect roughly 10-16 % lower prefill on the reference
  machine and the same decode.
