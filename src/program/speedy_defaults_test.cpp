// Speedy env defaults: unset -> set, explicit values kept, STRATA_UPSTREAM_DEFAULTS=1 -> nothing set.
#include "speedy_defaults.hpp"
#include <cstring>

#if defined(_WIN32)   // _putenv_s with an empty value removes the variable
static void set_env(const char* n, const char* v) { _putenv_s(n, v); }
static void unset_env(const char* n) { _putenv_s(n, ""); }
#else
static void set_env(const char* n, const char* v) { setenv(n, v, 1); }
static void unset_env(const char* n) { unsetenv(n); }
#endif

int main() {
    for (const auto& d : speedy::kEnvDefaults) unset_env(d.name);
    set_env("STRATA_UPSTREAM_DEFAULTS", "1");
    if (speedy::apply_env_defaults(nullptr) != 0 || std::getenv("STRATA_EXPDEC") != nullptr) return 1;
    unset_env("STRATA_UPSTREAM_DEFAULTS");
    set_env("STRATA_EXPDEC", "0");
    const int n = (int) (sizeof speedy::kEnvDefaults / sizeof speedy::kEnvDefaults[0]);
    if (speedy::apply_env_defaults(nullptr) != n - 1) return 2;
    if (std::strcmp(std::getenv("STRATA_EXPDEC"), "0") != 0) return 3;            // explicit value kept
    if (std::strcmp(std::getenv("STRATA_ROUTE_TAIL_SKIP"), "0") != 0) return 4;
    if (speedy::apply_env_defaults(nullptr) != 0) return 5;                        // idempotent
    std::printf("speedy_defaults_test: ok\n");
    return 0;
}
