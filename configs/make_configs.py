#!/usr/bin/env python3
"""Write Speedy's production presets (serve/server.py configs) for this machine.

The presets are the settings measured on the reference machine (RTX 3080 10 GB LHR, Threadripper PRO 5975WX, 128 GB,
Debian 13, CUDA 13.4); see docs/TUNING-HISTORY.md for why each flag is there and what was rejected.
Every path comes from the command line (or the environment), so the same presets work on another box:

  python3 configs/make_configs.py --data /opt/models/strata-data --packs /opt/models/packs \
      --models /opt/models/gguf --q2-dir /opt/models/gguf/q38-ista-q20 --out configs/local

Writes speedy-q2_0-med.json, speedy-q2_0-1m.json, speedy-q2_0-low.json, speedy-iq3_s.json, speedy-ud-iq4_xs.json, speedy-ud-q4_k_xl.json.
configs/generic/ holds them generated with the default (/opt/...) paths.
Adapted from the reference machine's own config generator.
"""
import argparse, copy, json, os

# Engine environment of every preset.  The same values are Speedy's built-in defaults (docs/SPEEDY_DEFAULTS.md);
# they are repeated here so a preset also pins them under STRATA_UPSTREAM_DEFAULTS=1 or on the Strata control binary.
PROD_ENV = {
    "STRATA_TOOLCALL_REPAIR": "1",        # serve: a broken <tool_call> is written again from "<function=" (not the engine)
    "STRATA_ROUTE_TAIL_SKIP": "0",        # lossy route tail skip off (quality)
    "STRATA_PREFILL_CPU_SHARE": "0",      # no CPU share in prefill (GPU + PCIe faster here)
    "STRATA_PIN_BY_NEED": "1",
    "STRATA_EXPDEC": "1",
    "STRATA_EXCHANGE_ROTATE": "1",
    "STRATA_ATTN_DAILY": "1",
    "STRATA_VERIFY_DEFER_CPU": "1",       # Strata PR #2689/#2690 (patch 0001)
    "STRATA_VERIFY_DEFER_CPU_BELOW": "0.1",
    "STRATA_PROMPT_ATTN_IMMA": "1",       # A4: INT8-KV prompt attention on IMMA (32K prefill +2.0 %, decode +2.1 %; suite + needle 3/3 PASS)
    "STRATA_KV_PREFETCH": "1",            # E25: KV prefetch (Q2 32K prefill +1.5 %, 128K +1.4 %)
    "STRATA_PF_GROUP_SCRATCH": "1",       # PR #4039: smaller prompt scratch -> bigger chunks (IQ4_XS +29-33 %, IQ3_S +22-24 %, Q2 +5 %)
    "STRATA_PF_HC_REGION": "1",
    "STRATA_HC_Q8": "1",   # 2026-10-30: on for every quant (Q2_0/IQ3_S decode +5-6 %)
}

SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0, "presence_penalty": 1.5}


def base_args(a, pack, native, ple, max_context=None):
    args = ["--pack", pack, "--native", native]
    if ple:
        args += ["--ple-gguf", ple]
    args += [
        "--expert-profile", os.path.join(a.speedy, "data", "expert-profile.bin"),
        "--expert-cache", "auto", "--prefill", "auto",
        "--spec", "4", "--spec-min-p", "0.5",
        "--mtp", os.path.join(a.data, "mtp", "rt"),
        "--max-context", str(max_context or a.max_context),
        "--kv", "int8", "--kv-resident", "20480",
        "--vram-reserve-mib", "700",           # 400 = +15 % prefill but OOM at 127K (rejected)
        "--conversation-cache-mib", "8192",
        "--adapt-async", "1", "--adapt-swaps", "80",
        "--pool-affinity", "measure",
    ]
    return args


def make(a, tag, name, pack, native, ple, effort, budget_tokens, extra_env=None, extra_args=None, fit=True, max_context=None):
    c = {
        "exe": a.exe or os.path.join(a.speedy, "build", "strata"),
        "args": base_args(a, pack, native, ple, max_context) + (extra_args or []),
        "cwd": a.speedy,
        "tokenizer": os.path.join(pack, "tokenizer"),
        "model_name": name,
        "log": os.path.join(a.logs, f"speedy-{tag}.log"),
        "lib_dirs": [a.cuda_lib],
        "port": 8080, "gpu": 0, "gpus_asked": True, "draft_vocab": "en",
        "sampling": copy.deepcopy(SAMPLING),
        "reasoning_effort": effort,
        "env": {**PROD_ENV, **(extra_env or {})},
        "reasoning_budget_tokens": budget_tokens,
        "reasoning_close_retry": True,
        "repeat_stop_tokens": 256,
        "host": "127.0.0.1",
    }
    if fit:
        c["fit_max_tokens"] = True
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"speedy-{tag}.json")
    with open(path, "w") as f:
        json.dump(c, f, indent=2)
        f.write("\n")
    print("wrote", path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    env = os.environ.get
    p.add_argument("--speedy", default=env("SPEEDY_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    p.add_argument("--exe", default=env("SPEEDY_EXE"), help="engine binary (default <speedy>/build/strata)")
    p.add_argument("--data", default=env("SPEEDY_DATA", "/opt/models/strata-data"), help="Strata-data (packs/, mtp/)")
    p.add_argument("--packs", default=env("SPEEDY_PACKS", "/opt/models/packs"), help="IQ3_S / UD-Q4 packs")
    p.add_argument("--models", default=env("SPEEDY_MODELS", "/opt/models/gguf"))
    p.add_argument("--q2-dir", default=env("SPEEDY_Q2_DIR", "/opt/models/gguf/q38-ista-q20"))
    p.add_argument("--logs", default=env("SPEEDY_LOGS", "/var/log/speedy"))
    p.add_argument("--cuda-lib", default="/usr/local/cuda/lib64")
    p.add_argument("--max-context", type=int, default=229376, help="every quant fits 229376 on 10 GB with reserve 700")
    p.add_argument("--q4-resident-gib", type=int, default=80, help="UD-Q4 / UD-IQ4 --resident-budget-gib (J01: 80 = decode +25 %% on 128 GB RAM)")
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "local"))
    a = p.parse_args()

    q2 = os.path.join(a.q2_dir, "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-0000{}-of-00002.gguf")
    q2pack = os.path.join(a.data, "packs", "q2_0")
    make(a, "q2_0-med", "qwen3.8-flash-next-q2_0", q2pack, q2.format(1), q2.format(2), "medium", 8192)
    # 1M overflow tier: YaRN x4 over the native 262144; K1 STRATA_KV_STAGE_REQ sizes the K/V staging pool per request
    # (300K cold prompt 466 -> 1527 t/s, docs/CONFIGS.md)
    make(a, "q2_0-1m", "qwen3.8-flash-next-q2_0-1m", q2pack, q2.format(1), q2.format(2), "medium", 8192,
         extra_env={"STRATA_KV_STAGE_REQ": "131072"},
         max_context=1048576, extra_args=["--rope-scaling", "yarn", "--rope-scale", "4", "--yarn-orig-ctx", "262144"])
    make(a, "q2_0-low", "qwen3.8-flash-next-q2_0", q2pack, q2.format(1), q2.format(2), "low", 4096, fit=False)
    iq = os.path.join(a.models, "q38-ista-iq3s", "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000{}-of-00002.gguf")
    make(a, "iq3_s", "qwen3.8-flash-next-iq3_s", os.path.join(a.packs, "iq3_s"), iq.format(1), iq.format(2),
         "medium", 16384)
    make(a, "ud-iq4_xs", "qwen3.8-flash-next-ud-iq4_xs", os.path.join(a.packs, "unsloth-ud-iq4_xs"),
         os.path.join(a.models, "q38-ud-iq4xs", "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"), None,
         "medium", 16384, extra_env={"STRATA_HC_Q8": "1", "STRATA_PREFILL_ROUTED_STREAM": "1",   # PR #4283: +8-10 % alone
                    "STRATA_SELECT_TWO_PASS": "1"},   # PR #4119: +1-2.5 % prefill at 120/200K
         extra_args=["--resident-budget-gib", str(a.q4_resident_gib)])
    make(a, "ud-q4_k_xl", "qwen3.8-flash-next-ud-q4_k_xl", os.path.join(a.packs, "unsloth-ud-q4_k_xl"),
         os.path.join(a.models, "q38-ud-q4kxl", "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"), None,
         "medium", 16384, extra_env={"STRATA_HC_Q8": "1"},
         extra_args=["--resident-budget-gib", str(a.q4_resident_gib)])


if __name__ == "__main__":
    main()
