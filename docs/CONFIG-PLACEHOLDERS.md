# Placeholders in the edition configs

The configs in `configs/production-*gpu/` (and the harness in `tools/`) are the exact files used on the reference
machine, with that machine's paths and GPU UUIDs replaced by placeholders. Fill them in before use.

| Placeholder | Meaning | Example |
|---|---|---|
| `<SPEEDY_DIR>` | this repository (the server runs `serve/server.py` from here; `cwd` and `serve_dir`) | `/opt/speedy` |
| `<SPEEDY_DIR>/build/strata` | the engine binary you built (see README, "Build") | |
| `<SPEEDY_DIR>/data/expert-profile.bin` | the expert profile (`--expert-profile`); any path that holds Strata's `expert-profile.bin` | |
| `<MODELS_DIR>` | where the GGUF files (`models/...`) and Strata packs (`packs/...`, with `tokenizer/`) live | `/models/qwen38` |
| `<STRATA_DATA>` | Strata's data dir: `mtp/rt` (MTP draft head) and the Q2_0 pack (`packs/q2_0`) | `/models/strata-data` |
| `<LOG_DIR>` | where the server writes its log (`log`) | `/var/log/speedy` |
| `<CONFIG_DIR>` | where you keep these JSON files (llama-swap `cmd` lines, harness) | `/etc/speedy` |
| `<LLAMA_SWAP_DIR>` | the directory with the llama-swap `serve.sh` wrapper (docs/LLAMA_SWAP.md) | |
| `<SCRIPTS_DIR>`, `<BENCH_DIR>` | harness only: start/clean scripts and the benchmark dir (`bench/`) | |
| `<GPU1_UUID>` | the main (fastest) card, `nvidia-smi -L` gives `GPU-xxxxxxxx-...` | `GPU-0a1b...` |
| `<GPU2_UUID>` | 2-GPU edition only: the second card | |

Devices are pinned by UUID with `CUDA_DEVICE_ORDER=PCI_BUS_ID` and the main card listed first. On a one-GPU machine
you can delete `CUDA_VISIBLE_DEVICES` instead.

Fill in everything at once, e.g.:

```sh
sed -i -e "s#<SPEEDY_DIR>#$PWD#g" -e "s#<MODELS_DIR>#/models/qwen38#g" -e "s#<STRATA_DATA>#/models/strata-data#g" \
       -e "s#<LOG_DIR>#$HOME/speedy-logs#g" -e "s#<CONFIG_DIR>#$PWD/configs/mine#g" \
       -e "s#<GPU1_UUID>#$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)#g" configs/mine/*.json
```

File names inside `<MODELS_DIR>` (e.g. `models/q38-ud-iq4xs/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf`) are
the reference machine's layout; point them at your copies. `configs/make_configs.py` writes the generic presets
for your paths instead.
