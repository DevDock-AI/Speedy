// src/core/adapt_fetch_default_test.cpp - STRATA_ADAPT_FETCH=2 as the default of a compute capability 12.x card with at most 16 GiB
// (ported from upstream Strata v0.2.17 ca99e886f; Speedy's sm_120 table is the NVFP4 prompt row only, so the counts differ).  No GPU needed.
// Own program (not nv_defaults_test): apply_nv_defaults reads qwen_defaults_026(), which caches STRATA_DEFAULTS at its first call.
#include "strata/core/nv_defaults.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static void unset(const char* k) {
#if defined(_WIN32)
    _putenv_s(k, "");
#else
    unsetenv(k);
#endif
}
static void put(const char* k, const char* v) {
#if defined(_WIN32)
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}
static const char* val(const char* k) { return std::getenv(k); }
using Tbl = std::vector<std::pair<std::string, std::string>>;
static bool has(const Tbl& t, const char* k, const char* v) {
    for (const auto& kv : t) if (kv.first == k) return kv.second == v;
    return false;
}
static bool lacks(const Tbl& t, const char* k) {
    for (const auto& kv : t) if (kv.first == k) return false;
    return true;
}
static bool in(const std::vector<std::string>& v, const char* k) {
    for (const auto& s : v) if (s == k) return true;
    return false;
}
static void clear_table() { unset("STRATA_ADAPT_FETCH"); unset("STRATA_PREFILL_NVFP4"); }

int main() {
    using namespace strata::core;
    const uint64_t GiB = 1ull << 30;
    for (const char* k : {"STRATA_SM120_PERF", "STRATA_SM121_PERF", "STRATA_SM86_PERF", "STRATA_SM60_PERF", "STRATA_NVFP4_PROMPT",
                          "STRATA_DEFAULTS", "STRATA_EMULATE_CC", "STRATA_ADAPT_FETCH", "STRATA_PREFILL_NVFP4"})
        unset(k);

    // the card rule: compute capability 12.x, 0 < memory <= 16 GiB
    CHECK(kAdaptFetchMaxVramBytes == 16 * GiB);
    CHECK(adapt_fetch_default_card(12, 0, 12 * GiB));                 // RTX 5070
    CHECK(adapt_fetch_default_card(12, 0, 16 * GiB - (1ull << 20)));  // RTX 5060 Ti / 5070 Ti / 5080 report ~15.9 GiB
    CHECK(adapt_fetch_default_card(12, 0, 16 * GiB));
    CHECK(adapt_fetch_default_card(12, 0, 8 * GiB));                  // RTX 5060 8 GB, laptops
    CHECK(adapt_fetch_default_card(12, 1, 16 * GiB));
    CHECK(!adapt_fetch_default_card(12, 0, 16 * GiB + 1));
    CHECK(!adapt_fetch_default_card(12, 0, 24 * GiB));                // RTX PRO 4000
    CHECK(!adapt_fetch_default_card(12, 0, 32 * GiB));                // RTX 5090
    CHECK(!adapt_fetch_default_card(12, 1, 121 * GiB));               // GB10 unified memory
    CHECK(!adapt_fetch_default_card(12, 0, 0));                       // unknown
    CHECK(!adapt_fetch_default_card(8, 6, 10 * GiB));                 // RTX 3080 10 GB (our 10GB card): other architecture
    CHECK(!adapt_fetch_default_card(8, 6, 20 * GiB));                 // our 20GB card
    CHECK(!adapt_fetch_default_card(8, 9, 16 * GiB));                 // RTX 4080
    CHECK(!adapt_fetch_default_card(6, 0, 16 * GiB));                 // P100

    // the tables
    {
        const Tbl t = nv_default_env(12, 0, false, 12 * GiB);
        CHECK(t.size() == 2 && has(t, "STRATA_ADAPT_FETCH", "2") && has(t, "STRATA_PREFILL_NVFP4", "w4a4"));
        CHECK(nv_default_env(12, 0, false, 32 * GiB).size() == 1);
        CHECK(nv_default_env(12, 0, false).size() == 1);              // no memory given: as before
        CHECK(nv_default_env(12, 0, true, 12 * GiB).empty());         // STRATA_DEFAULTS=026
        put("STRATA_NVFP4_PROMPT", "0");                              // the NVFP4 row's own switch keeps the fetch default
        const Tbl n = nv_default_env(12, 0, false, 12 * GiB);
        CHECK(n.size() == 1 && has(n, "STRATA_ADAPT_FETCH", "2"));
        CHECK(nv_default_env(12, 0, false).empty());                  // and the old behaviour without memory
        unset("STRATA_NVFP4_PROMPT");
        put("STRATA_SM120_PERF", "0");
        CHECK(nv_default_env(12, 0, false, 12 * GiB).empty());
        unset("STRATA_SM120_PERF");
        CHECK(lacks(nv_default_env(12, 1, false, 121 * GiB), "STRATA_ADAPT_FETCH"));
        CHECK(has(nv_default_env(12, 1, false, 16 * GiB), "STRATA_ADAPT_FETCH", "2"));
    }
    CHECK(lacks(nv_default_env(8, 6, false, 10 * GiB), "STRATA_ADAPT_FETCH"));
    CHECK(lacks(nv_default_env(8, 6, false, 20 * GiB), "STRATA_ADAPT_FETCH"));
    CHECK(lacks(nv_default_env(6, 0, false, 16 * GiB), "STRATA_ADAPT_FETCH"));
    CHECK(nv_default_env(8, 9, false, 16 * GiB).empty());

    {   // apply: sets it where the user has not
        clear_table();
        const auto set = apply_nv_defaults(12, 0, 12 * GiB);
        CHECK(in(set, "STRATA_ADAPT_FETCH") && set.size() == 2);
        CHECK(val("STRATA_ADAPT_FETCH") != nullptr && std::string(val("STRATA_ADAPT_FETCH")) == "2");
        CHECK(adapt_fetch_is_default());
    }
    {   // over 16 GiB
        clear_table();
        const auto set = apply_nv_defaults(12, 0, 32 * GiB);
        CHECK(!in(set, "STRATA_ADAPT_FETCH") && set.size() == 1 && val("STRATA_ADAPT_FETCH") == nullptr && !adapt_fetch_is_default());
    }
    for (const char* mine : {"0", "1", "2"}) {   // the user's value wins, 0 included
        clear_table();
        put("STRATA_ADAPT_FETCH", mine);
        const auto set = apply_nv_defaults(12, 0, 12 * GiB);
        CHECK(!in(set, "STRATA_ADAPT_FETCH") && set.size() == 1);
        CHECK(std::string(val("STRATA_ADAPT_FETCH")) == mine && !adapt_fetch_is_default());
    }
    {   // our box: sm_86, 20 GB main card -> nothing about the fetch
        clear_table();
        const auto set = apply_nv_defaults(8, 6, 20 * GiB);
        CHECK(!in(set, "STRATA_ADAPT_FETCH") && val("STRATA_ADAPT_FETCH") == nullptr && !adapt_fetch_is_default());
        for (const char* k : {"STRATA_ADAPT_NOWAIT", "STRATA_ADAPT_LAG", "STRATA_CPU_IQ_ALL_EXACT1", "STRATA_SIDE_MIXER", "STRATA_ATTN_DAILY",
                              "STRATA_GR_UP75", "STRATA_GU_SWIGLU", "STRATA_QSA_ROWS", "STRATA_B6_MMVQ_ROWS"}) unset(k);
    }
    {   // no memory given
        clear_table();
        const auto set = apply_nv_defaults(12, 0);
        CHECK(set.size() == 1 && val("STRATA_ADAPT_FETCH") == nullptr && !adapt_fetch_is_default());
    }
    clear_table();
    if (fails == 0) std::printf("adapt_fetch_default_test: ok\n");
    return fails == 0 ? 0 : 1;
}
