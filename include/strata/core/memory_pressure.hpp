// include/strata/core/memory_pressure.hpp - Speedy C4: Linux pressure-stall information (PSI) for the parked
// conversation cache.  Idea after Basalt PR #113 (jesdga95/basalt, MIT); thresholds and code are our own.
//
// /proc/pressure/memory reports the share of wall time in which some (or all) tasks stalled on memory:
//   some avg10=1.23 avg60=0.50 avg300=0.10 total=123456
//   full avg10=0.80 avg60=0.30 avg300=0.05 total=65432
// MemAvailable alone misses the case this guards: the box has free pages, but reclaim (page cache churn from Docker,
// Plex, a build) already stalls tasks, and a large conversation snapshot makes it worse.  Under STRATA_PSI_SHED=<x>
// (opt-in; x = the "some avg10" percentage, e.g. 10) the engine sheds the oldest parked conversations before it parks
// a new one while the pressure is at or above x.  Unset, or a kernel without PSI: nothing changes.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>

namespace strata::core {

struct MemoryPressure {
    double some_avg10 = 0, some_avg60 = 0, full_avg10 = 0, full_avg60 = 0;
};

/// Parses the text of /proc/pressure/memory; nullopt when the "some" line is missing or malformed.
inline std::optional<MemoryPressure> parse_memory_pressure(const std::string& text) {
    MemoryPressure p;
    bool some = false;
    size_t at = 0;
    while (at < text.size()) {
        size_t end = text.find('\n', at);
        if (end == std::string::npos) end = text.size();
        const std::string line = text.substr(at, end - at);
        at = end + 1;
        double a10 = 0, a60 = 0, a300 = 0;
        unsigned long long total = 0;
        char kind[8] = {};
        if (std::sscanf(line.c_str(), "%7s avg10=%lf avg60=%lf avg300=%lf total=%llu", kind, &a10, &a60, &a300, &total) != 5)
            continue;
        if (a10 < 0 || a60 < 0 || a10 > 100 || a60 > 100) return std::nullopt;
        if (std::strcmp(kind, "some") == 0) { p.some_avg10 = a10; p.some_avg60 = a60; some = true; }
        else if (std::strcmp(kind, "full") == 0) { p.full_avg10 = a10; p.full_avg60 = a60; }
    }
    if (!some) return std::nullopt;
    return p;
}

/// Reads /proc/pressure/memory (or `path`); nullopt without PSI (non-Linux, CONFIG_PSI off, psi=0).
inline std::optional<MemoryPressure> read_memory_pressure(const char* path = "/proc/pressure/memory") {
    std::FILE* f = std::fopen(path, "r");
    if (!f) return std::nullopt;
    char buf[512];
    const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
    std::fclose(f);
    return parse_memory_pressure(std::string(buf, n));
}

/// STRATA_PSI_SHED=<some avg10 %> in (0, 100] (opt-in); a negative value when unset, 0 or invalid.
inline double psi_shed_threshold() {
    static const double v = [] {
        const char* e = std::getenv("STRATA_PSI_SHED");
        if (e == nullptr || *e == '\0') return -1.0;
        char* end = nullptr;
        const double x = std::strtod(e, &end);
        return end != e && x > 0 && x <= 100 ? x : -1.0;   // 0 would shed on any pressure: refused
    }();
    return v;
}

/// Whether the cache should shed parked conversations before parking another one.
inline bool psi_should_shed(const std::optional<MemoryPressure>& p, double threshold) {
    return threshold >= 0 && p && p->some_avg10 >= threshold;
}

}  // namespace strata::core
