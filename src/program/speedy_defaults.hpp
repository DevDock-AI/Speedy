// src/program/speedy_defaults.hpp - Speedy's engine-environment defaults (docs/SPEEDY_DEFAULTS.md).
//
// Speedy runs with the settings proven in production on the reference machine (RTX 3080 10 GB, configs/make_configs.py)
// even without a config: each STRATA_* variable below is set at start-up unless the environment already has it.
// Only behaviour switches measured as speed-neutral-or-better and quality-safe are here; pack- and VRAM-dependent
// command-line options (--kv, --vram-reserve-mib, ...) stay in the presets.
// STRATA_UPSTREAM_DEFAULTS=1 skips all of it: the engine then behaves exactly like upstream Strata.
#pragma once

#include <cstdio>
#include <cstdlib>

namespace speedy {

struct EnvDefault { const char* name; const char* value; };

inline constexpr EnvDefault kEnvDefaults[] = {
    {"STRATA_ROUTE_TAIL_SKIP", "0"},           // lossy route tail skip off
    {"STRATA_PREFILL_CPU_SHARE", "0"},
    {"STRATA_PIN_BY_NEED", "1"},
    {"STRATA_EXPDEC", "1"},
    {"STRATA_EXCHANGE_ROTATE", "1"},
    {"STRATA_ATTN_DAILY", "1"},
    {"STRATA_VERIFY_DEFER_CPU", "1"},          // Strata PR #2689/#2690 (patch 0001)
    {"STRATA_VERIFY_DEFER_CPU_BELOW", "0.1"},
    {"STRATA_PROMPT_ATTN_IMMA", "1"},          // A4 IMMA prompt attention (from architectds/Strata): 32K prefill +2.0 %, decode +2.1 %; suite + needle 3/3
    {"STRATA_KV_PREFETCH", "1"},               // E25 KV prefetch: Q2 32K prefill +1.5 %, 128K +1.4 %
    {"STRATA_PF_GROUP_SCRATCH", "1"},          // Strata PR #4039 (merbanan), bitwise per its parity check: bigger auto prompt chunks.
    {"STRATA_PF_HC_REGION", "1"},              // ABBA prefill: IQ4_XS +29-33 %, IQ3_S +22-24 %, Q2 +5 %; decode same
};

/// Sets every unset default; returns how many it set (0 under STRATA_UPSTREAM_DEFAULTS=1).
inline int apply_env_defaults(std::FILE* log = stderr) {
    const char* up = std::getenv("STRATA_UPSTREAM_DEFAULTS");
    if (up != nullptr && *up != '\0' && *up != '0') {
        if (log) std::fprintf(log, "speedy: STRATA_UPSTREAM_DEFAULTS set - upstream Strata defaults\n");
        return 0;
    }
    int set = 0;
#if defined(_WIN32)
    for (const auto& d : kEnvDefaults)
        if (std::getenv(d.name) == nullptr && _putenv_s(d.name, d.value) == 0) ++set;
#else
    for (const auto& d : kEnvDefaults)
        if (std::getenv(d.name) == nullptr && setenv(d.name, d.value, 0) == 0) ++set;
#endif
    if (log && set > 0)
        std::fprintf(log, "speedy: %d of %d production defaults applied (docs/SPEEDY_DEFAULTS.md; "
                          "STRATA_UPSTREAM_DEFAULTS=1 for upstream behaviour)\n", set, (int) (sizeof kEnvDefaults / sizeof kEnvDefaults[0]));
    return set;
}

}  // namespace speedy
