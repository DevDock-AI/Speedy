// Speedy C4: the PSI parser and the shed decision (CPU only, no /proc dependency except the optional live read).
#include "strata/core/memory_pressure.hpp"
#include <cstdio>

using namespace strata::core;

int main() {
    int fails = 0;
    auto check = [&](bool ok, const char* what) { if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++fails; } };
    auto p = parse_memory_pressure("some avg10=12.50 avg60=3.00 avg300=0.35 total=73200285\n"
                                   "full avg10=8.00 avg60=1.00 avg300=0.33 total=64654114\n");
    check(p && p->some_avg10 == 12.5 && p->some_avg60 == 3.0 && p->full_avg10 == 8.0, "both lines");
    check(parse_memory_pressure("some avg10=0.00 avg60=0.00 avg300=0.00 total=0").has_value(), "no newline, no full line");
    check(!parse_memory_pressure("").has_value(), "empty");
    check(!parse_memory_pressure("full avg10=1.00 avg60=1.00 avg300=1.00 total=5\n").has_value(), "full only");
    check(!parse_memory_pressure("some avg10=abc\n").has_value(), "malformed");
    check(!parse_memory_pressure("some avg10=150.00 avg60=0.00 avg300=0.00 total=1\n").has_value(), "out of range");
    check(psi_should_shed(p, 10.0), "12.5 >= 10 sheds");
    check(!psi_should_shed(p, 20.0), "12.5 < 20 keeps");
    check(!psi_should_shed(p, -1.0), "unset never sheds");
    check(!psi_should_shed(std::nullopt, 0.0), "no PSI never sheds");
    check(!read_memory_pressure("/nonexistent/pressure").has_value(), "missing file");
    if (auto live = read_memory_pressure()) std::printf("live: some avg10 %.2f full avg10 %.2f\n", live->some_avg10, live->full_avg10);
    if (fails == 0) std::printf("memory_pressure_test: ok\n");
    return fails ? 1 : 0;
}
