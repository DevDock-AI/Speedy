// J13 (Speedy; audit after Basalt PR #120): the expert cache's slot stride is what every VRAM budget must price.
// Exact fit, one byte below, 256-byte boundaries, the STRATA_MMQ_DIRECT guard, mixed slot sizes, and a greedy fit
// over a profile that must never exceed the byte budget open_sized() will allocate.
#include "strata/core/expert_cache.hpp"
#include <cstdint>
#include <cstdio>
#include <vector>

using strata::core::ExpertCache;

namespace {
int fails = 0;
void check(bool ok, const char* what) { if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++fails; } }

// the greedy fit generate.cpp uses (pin-by-need estimate and the sized auto cache): slots taken while they fit
size_t greedy(const std::vector<int64_t>& blobs, uint64_t cap, int64_t guard, uint64_t& used) {
    used = 0;
    size_t n = 0;
    for (int64_t b : blobs) {
        const uint64_t s = (uint64_t) ExpertCache::slot_stride(b, guard);
        if (used + s > cap) break;
        used += s;
        ++n;
    }
    return n;
}
}  // namespace

int main() {
    check(ExpertCache::slot_stride(256) == 256, "256 stays 256");
    check(ExpertCache::slot_stride(255) == 256, "255 rounds to 256");
    check(ExpertCache::slot_stride(257) == 512, "257 rounds to 512");
    check(ExpertCache::slot_stride(1) == 256, "1 byte takes a slot");
    check(ExpertCache::slot_stride(1382400) == 1382400, "Q2 blob is 256-aligned");
    check(ExpertCache::slot_stride(1382400 + 16) == 1382400 + 256, "NVFP4 scale tail rounds up");
    check(ExpertCache::slot_stride(1382400, 64) == 1382400 + 256, "guard adds a stride step");
    check(ExpertCache::slot_stride(1382400 - 64, 64) == 1382400, "guard that fits the padding");
    for (int64_t b = 1; b < 4096; b += 7)
        for (int64_t g : {0, 32, 256}) {
            const int64_t s = ExpertCache::slot_stride(b, g);
            if (s % 256 != 0 || s < b + g || s - (b + g) >= 256) { check(false, "stride invariant"); break; }
        }
    // mixed sizes: the greedy fit never exceeds the cap, and the exact sum fits exactly
    const std::vector<int64_t> blobs = {1382400, 2654208, 1382400 + 16, 917504, 2654208 - 1};
    for (int64_t g : {0, 64}) {
        uint64_t total = 0;
        for (int64_t b : blobs) total += (uint64_t) ExpertCache::slot_stride(b, g);
        uint64_t used = 0;
        check(greedy(blobs, total, g, used) == blobs.size() && used == total, "exact fit takes every slot");
        check(greedy(blobs, total - 1, g, used) == blobs.size() - 1, "one byte below drops the last slot");
        for (uint64_t cap = 0; cap < total; cap += 99991) {
            greedy(blobs, cap, g, used);
            if (used > cap) { check(false, "greedy fit over the cap"); break; }
        }
    }
    // pricing slots with the unpadded blob (the pre-J13 estimate with a guard) over-counts: show the case exists
    {
        const int64_t g = 64;
        uint64_t used = 0;
        const std::vector<int64_t> same(10, 1382400);
        const uint64_t cap = 10ull * 1382400;
        check(greedy(same, cap, g, used) == 9, "guarded slots: 9 of 10 fit where unpadded pricing says 10");
    }
    if (fails == 0) std::printf("slot_budget_test: ok\n");
    return fails == 0 ? 0 : 1;
}
