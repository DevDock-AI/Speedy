# AMD Radeon RX 7900 XTX 24 GB

| Field | Value |
|:---|:---|
| Status | community (untested by maintainers); settings and numbers from nicejunjie/Strata-rdna3 |
| GPU | RX 7900 XTX, 24 GB, gfx1100 (RDNA3) |
| Build | `-DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON` (+ `-DSTRATA_ROCWMMA_INCLUDE_DIR=<rocWMMA>/library/include`, `-DCMAKE_PREFIX_PATH=<ROCm root>` so libhsa-runtime64 is found) |
| GPU unlock | not applicable |
| Power limit | stock |
| PCIe | 4.0 (reference box: x8) |
| CPU | reference box: Ryzen 9 9950X3D2 (AVX-512); `--pool-workers` 12 there, set it to your physical cores |
| RAM | reference box: 128 GB |
| Storage | NVMe |
| OS / driver / CUDA | Linux, ROCm 7.x (HIP, no CUDA) |
| Speedy commit | dev/rdna3-port |
| Maintainer | community |

## Presets

| Folder | Contents |
|:---|:---|
| `1x/` | one card: IQ3_S, UD-IQ4_XS, UD-Q4_K_XL (the fork's "1 GPU" settings, docs/RDNA3_SETTINGS.md there) |

## Measured upstream of this port (tok/s, one 7900 XTX, fork release rg23)

| Quant | Prefill 32K | Prefill 128K | Decode prose 32K | Decode code 32K |
|:---|---:|---:|---:|---:|
| IQ3_S | 3,541 | 3,553 | 114.1 | 139.3 |
| UD-IQ4_XS | 3,455 | 3,480 | 94.2 | 122.3 |
| UD-Q4_K_XL | 3,613 | 3,676 | 78.3 | 95.6 |

Stock Strata on the same card, UD-IQ4_XS 32K prefill: 1,151 tok/s. Not yet re-measured on this port.

## Tuning notes

- `GPU_MAX_HW_QUEUES=6` with one card (8 with two).
- `STRATA_HIPBLASLT_TUNING` points at `tools/hip/gfx1100-hipblaslt-100200.txt`.
- `--mtp-draft-vocab data/draft_vocab_zh60k.bin` is the fork's draft vocabulary.
- `STRATA_UPSTREAM_DEFAULTS=1`: the fork measured these settings on upstream defaults; Speedy's defaults (docs/SPEEDY_DEFAULTS.md) were tuned on RTX 3080.
- `--vision` dropped (the release engine has no vision).
