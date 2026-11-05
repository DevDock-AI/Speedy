// strata/core/sm120_series.hpp - bitwise kernel switches that are ON by default on a compute capability 12.0 card
// (RTX 50 series / RTX PRO Blackwell workstation cards).
//
// 147-verify measured these exact switches on an RTX PRO 5000 Blackwell 72 GB (IQ3_S, every expert in VRAM): together
// a verify round of 4 tokens 13.70 -> 12.23 ms with the same text (notes-147-l620.md).  Each one is bitwise equal to
// the kernels it replaces (their own parity tests and the identity gate), so only the speed changes:
//   STRATA_SIDE_MIXER=2  the mixer's input-only work on branches of the verify graph
//   STRATA_ATTN_DAILY=1  the Daily's decode attention (attn_chunk_kernel_pre75 + merge v2)
//   STRATA_GR_UP75=1     the hyper-connection up projection with its rows' loads hoisted
//   STRATA_GR_DOWN_TG=1  the hyper-connection down projection with one token a block row (T times the blocks); =2 (token
//                        pairs) on a 12.0 card with fewer than 80 SMs
//   STRATA_GU_SWIGLU=1   gate/up, SwiGLU and the q8_1 blocks of the routed experts in one launch
//   STRATA_GR_UP_WIDE=1  the hyper-connection up projection with one warp a row (1,280 blocks instead of 160; +1.9%
//                        story / +0.7% code decode on the RTX PRO 5000, 5 pairs, token ids identical; bitwise)
// and one that rounds differently (KL-checked, the AMD gfx12 rule): STRATA_PF_FUSED (the native packs' prompt experts on the
// fused int8 kernels, IQ2_S / IQ3_XXS / IQ3_S / IQ4_XS gate/up only, 12.0 cards with 80+ SMs only; =0 keeps MMQ, =1 takes every
// covered format on any card).
//
//   STRATA_SM120_PERF=0  turns the whole set off on a 12.0 card (the defaults every other card has).
// Each switch's own variable still wins in both directions (=0 off, =1 / =2 on), on any card, as before.
// Nothing here changes a card that is not a 12.0 (an sm_121 / sm_100 card keeps its defaults).
#pragma once

#include "strata/core/emulate.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace strata {

/// STRATA_SM120_PERF (read once): unset or anything but "0" = on
inline bool sm120_series() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_SM120_PERF");
        return e == nullptr || std::atoi(e) != 0;
    }();
    return on;
}

/// the device (raw cudaDevAttrComputeCapabilityMajor / Minor) is a 12.0 card, or emulates one (STRATA_EMULATE_CC=120)
inline bool is_sm120(int major, int minor) { return cc_major_of(major) == 12 && cc_minor_of(minor) == 0; }

/// the default of a bitwise switch of the set: on for a 12.0 card unless STRATA_SM120_PERF=0
inline bool sm120_default(int major, int minor) { return is_sm120(major, minor) && sm120_series(); }

// ---- 0.2.7 defaults of the CUDA speed work (pure rules; the engine reads the device and calls these; tests in arch_defaults_test)
// `hip`: a HIP build, where a gfx1200 / gfx1201 reports compute capability 12.0 as well and none of these is measured or wanted.
// `v026`: STRATA_DEFAULTS=026 (defaults_026.hpp qwen_defaults_026()): every one of these is then off, as in v0.2.6.  An explicit variable
// (STRATA_ROUTE_PRIOR=2, STRATA_GR_V3=1, ...) is decided by the caller before it asks the rule, so it still wins.
// `total_bytes`: the card's memory (cudaDeviceProp::totalGlobalMem); `integrated`: a GPU that shares the PC's memory.
constexpr double kSmallCardGiB = 13.0;   // "a 12 GB card": 12 GB cards report 11.6 - 12.0 GiB, 16 GB ones 15.6 - 16.0

/// The residency-bias bonus a discrete CUDA card of 13 GiB or less gets when the user set nothing: THE one place the value lives.  1.5 since the
/// owner decision of 2026-10-26 (it was 2.0 in the first 0.2.7 candidates): on the RTX 3060 12 GB with IQ3_XXS the copy and code check found real
/// defects at 2.0 and none in 19 tasks at 1.5 or 1.0, and 1.5 keeps the speed (story +15%, code +37% against off).  A longer 61-task check of
/// 0 / 1.5 / 2.0 is still running; if it shows defects at 1.5 this falls back to 1.0.
constexpr float kRoutePriorSmallCard = 1.5f;

/// STRATA_ROUTE_PRIOR when the user set nothing: kRoutePriorSmallCard (residency-biased routing, changes which experts run; owner-approved for 12 GB
/// cards) on a discrete CUDA card of 13 GiB or less, 0 everywhere else.  Measured on an RTX 3060 12 GB only.
inline float route_prior_default(bool hip, bool integrated, double total_bytes, bool v026 = false) {
    return !v026 && !hip && !integrated && total_bytes > 0.0 && total_bytes / 1073741824.0 <= kSmallCardGiB ? kRoutePriorSmallCard : 0.0f;
}

/// STRATA_GR_V3 (the stream-split hyper-connection read, another summation order) when the user set nothing: ON for a discrete compute capability
/// 12.0 card (STRATA_SM120_PERF=0 turns the set off) with more than 13 GiB, CUDA builds only.  Measured: RTX PRO 5000 72 GB +1.2% decode clean,
/// -3.0% / -5.9% on 2x RTX A4000 (sm_86), +1.0% / -3.4% on an RTX 3060 (sm_86): so not those; a 12 GB sm_120 card is unmeasured.
inline bool gr_v3_default(bool hip, int major, int minor, bool integrated, double total_bytes, bool v026 = false) {
    return !v026 && !hip && !integrated && sm120_default(major, minor) && total_bytes / 1073741824.0 > kSmallCardGiB;
}

/// STRATA_PROMPT_ATTN_IMMA (the prompt attention on int8 tensor cores, another summation order) when the user set nothing: ON for every CUDA card of
/// compute capability 8.0 or newer (the int8 m16n8k32 MMA), not Turing / Volta / Pascal, not HIP.  `cc` = 10 * major + minor as the engine
/// resolves it (an emulated or forced Turing counts as 75).  Measured on sm_120 (RTX PRO 5000) and sm_86 (2x RTX A4000): +1.1 .. +1.8% prompt.
inline bool prompt_attn_imma_default(bool hip, int cc, bool v026 = false) { return !v026 && !hip && cc >= 80; }

/// STRATA_MMVQ_IL_89 (#3453: the decode kernel row tables of an Ada card, the cells where an RTX 4070 and an RTX 4080 agreed; bitwise the same
/// output, only the rows-a-warp split differs) when the user set nothing: ON for compute capability 8.9 only (owner decision, 26 October: not yet
/// measured on an Ada card by us), off under STRATA_DEFAULTS=026 and in HIP builds.  `cc` = 10 * major + minor as the engine resolves it.
inline bool mmvq_il_89_default(bool hip, int cc, bool v026 = false) { return !v026 && !hip && cc == 89; }

/// The two sm_80-only defaults of 0.2.7 (#3779: the wide dense dequant and the MMVQ-IL sm_80 tables, both bitwise the old output, measured on a
/// CMP 170HX only) given the text of their variable (nullptr = unset): "1" on and "0" off win, unset is on unless STRATA_DEFAULTS=026.
inline bool sm80_default_on(const char* env, bool v026 = false) {
    if (env != nullptr && env[0] == '1') return true;
    if (env != nullptr && env[0] == '0') return false;
    return !v026;
}

/// The answer for STRATA_MMVQ_IL_89 given the variable's text (nullptr = unset): "1" on and "0" off win over everything, anything else (unset,
/// empty, other text) is the rule above.
inline bool mmvq_il_89_resolve(const char* env, bool hip, int cc, bool v026 = false) {
    if (env != nullptr && env[0] == '1') return true;
    if (env != nullptr && env[0] == '0') return false;
    return mmvq_il_89_default(hip, cc, v026);
}

/// STRATA_GR_UP_WIDE (the hyper-connection up projection, one warp a row; bitwise the old kernel) when the user set nothing: ON for a compute
/// capability 12.0 card (STRATA_SM120_PERF=0 turns the set off), CUDA builds only.
inline bool gr_up_wide_default(bool hip, int major, int minor, bool v026 = false) { return !v026 && !hip && sm120_default(major, minor); }

/// STRATA_PREFILL_LAYER_MAJOR (#3976: the layer-major prompt walk, which reads each layer's streamed experts once per super-chunk instead of once
/// per chunk; another summation order of the routed experts, so KL-checked, not bitwise) when the user set nothing: ON for a CUDA layer split
/// (`stages` = the number of GPUs the prompt path runs on, 2 or more), OFF on one GPU, in a HIP build and under STRATA_DEFAULTS=026.  Measured on
/// 2x RTX A4000 (sm_86) with Qwen3.8-Flash-Next IQ3_XXS and streamed experts only.  The engine adds its own per-prompt declines (image prompts,
/// DFlash2 taps, KVA, a batch read that yields, a peer GPU, no VRAM for a super-chunk): those read the chunk-major path as before.
inline bool prefill_layer_major_default(bool hip, int stages, bool v026 = false) { return !v026 && !hip && stages >= 2; }

/// The answer for STRATA_PREFILL_LAYER_MAJOR given the variable's text (nullptr = unset): any text that starts a number other than 0 is on, "0" (or any
/// text that reads as 0) is off, both win over everything; unset or empty is the rule above.  An explicit "1" is honoured on one GPU too (the
/// opt-in of the first ports); a HIP build still declines at run time.
inline bool prefill_layer_major_resolve(const char* env, bool hip, int stages, bool v026 = false) {
    if (env != nullptr && env[0] != '\0') return std::atoi(env) != 0;
    return prefill_layer_major_default(hip, stages, v026);
}


// ---- R17 arch gate (PLAN-PEER-EXPERT-TIER §17): the decode peer expert tier (STRATA_PEER_TIER2)
// A pure rule (no device calls; the engine reads the cards and calls it; table in arch_defaults_test).  Default ON only for:
//   CUDA build, not STRATA_DEFAULTS=026 / STRATA_UPSTREAM_DEFAULTS=1, STRATA_SM120_PERF not 0, n >= 2 cards, every card compute
//   capability 12.0, the cards alike (same CC, multiProcessorCount and VRAM within 10 % of the largest), every card >= 15 GiB.
// STRATA_PEER_TIER2=1 / =0 wins both ways on any card.  A layout given by hand (--layer-split, --peer-device) wins over the rule:
// no auto switch (an explicit STRATA_PEER_TIER2=1 still turns the tier on with --peer-device).  STRATA_EMULATE_CC=120 makes the
// rule see 12.0 cards for a DRY RUN only: it decides and logs, the engine never acts on it (sm_120 kernels are not dispatched).
// R17 review B1: the rule ships as a LOG ONLY until an outside 50xx A/B lands.  A rule_on decision is a dry run (decided and
// logged, never applied) unless STRATA_PEER_TIER2_AUTO=1 (PeerTierInput::rule_acts) lets it act; STRATA_PEER_TIER2=1 still forces it.
// The transport (P2P or mapped host) never gates the tier: it is picked by peer_tier_transport below.
constexpr double kPeerTierMinGiB = 15.0;   // 16 GB cards report 15.6 - 16.0 GiB
constexpr double kPeerTierAlikeTol = 0.10;

enum class PeerTierWhy {
    env_on, env_off, user_layout, hip, v026, upstream, sm120_off, one_card, not_sm120, unlike, small, rule_on,
};

struct PeerTierCard {
    int major = 0, minor = 0;   ///< raw compute capability
    int sms = 0;                ///< cudaDevAttrMultiProcessorCount
    double total_bytes = 0;     ///< cudaDeviceProp::totalGlobalMem
};

struct PeerTierInput {
    const PeerTierCard* cards = nullptr;
    int n = 0;
    bool hip = false, v026 = false, upstream = false;
    bool sm120_perf = true;            ///< STRATA_SM120_PERF not 0
    const char* env = nullptr;         ///< STRATA_PEER_TIER2 text (nullptr / "" = unset)
    bool user_layer_split = false;     ///< --layer-split given
    bool user_peer_device = false;     ///< --peer-device given
    int emulate_cc = 0;                ///< STRATA_EMULATE_CC as given (120 = dry run of the rule as if every card were 12.0)
    bool rule_acts = false;            ///< STRATA_PEER_TIER2_AUTO=1: a rule_on decision is applied (default: logged only, A/B pending)
};

struct PeerTierDecision {
    bool on = false;          ///< the tier is on (forced or by the rule)
    bool dry_run = false;     ///< decided under STRATA_EMULATE_CC=120: log only, the engine must not act
    bool auto_layout = false; ///< the engine may pick the peer layout itself (rule on, no user layout, not a dry run)
    bool pending_ab = false;  ///< rule_on but not applied: no STRATA_PEER_TIER2_AUTO=1 (dry_run is set too)
    PeerTierWhy why = PeerTierWhy::not_sm120;
};

inline bool peer_tier_alike(const PeerTierCard* c, int n) {
    if (n < 2) return false;
    int smax = 0;
    double vmax = 0;
    for (int i = 0; i < n; ++i) { smax = c[i].sms > smax ? c[i].sms : smax; vmax = c[i].total_bytes > vmax ? c[i].total_bytes : vmax; }
    if (smax <= 0 || vmax <= 0) return false;
    for (int i = 0; i < n; ++i) {
        if (c[i].major != c[0].major || c[i].minor != c[0].minor) return false;
        if (c[i].sms < (1.0 - kPeerTierAlikeTol) * smax || c[i].total_bytes < (1.0 - kPeerTierAlikeTol) * vmax) return false;
    }
    return true;
}

inline PeerTierDecision peer_tier_decide(const PeerTierInput& in) {
    PeerTierDecision d;
    d.dry_run = in.emulate_cc == 120;
    auto out = [&](bool on, PeerTierWhy w) { d.on = on; d.why = w; d.auto_layout = on && w == PeerTierWhy::rule_on && !d.dry_run; return d; };
    if (in.env != nullptr && in.env[0] != '\0') {
        if (std::atoi(in.env) != 0) {
            if (in.user_layer_split) return out(false, PeerTierWhy::user_layout);   // a layer split by hand: the tier has no card
            d.dry_run = false;   // an explicit opt-in acts on any card (the tier's kernels are generic CUDA)
            d.auto_layout = !in.user_peer_device;
            d.on = true; d.why = PeerTierWhy::env_on;
            return d;
        }
        d.dry_run = false;
        return out(false, PeerTierWhy::env_off);
    }
    if (in.user_layer_split || in.user_peer_device) return out(false, PeerTierWhy::user_layout);
    if (in.hip) return out(false, PeerTierWhy::hip);
    if (in.v026) return out(false, PeerTierWhy::v026);
    if (in.upstream) return out(false, PeerTierWhy::upstream);
    if (in.n < 2 || in.cards == nullptr) return out(false, PeerTierWhy::one_card);
    for (int i = 0; i < in.n; ++i) {
        const bool s120 = d.dry_run || (in.cards[i].major == 12 && in.cards[i].minor == 0);
        if (!s120) return out(false, PeerTierWhy::not_sm120);
    }
    if (!in.sm120_perf) return out(false, PeerTierWhy::sm120_off);
    if (!peer_tier_alike(in.cards, in.n)) return out(false, PeerTierWhy::unlike);
    for (int i = 0; i < in.n; ++i)
        if (in.cards[i].total_bytes / 1073741824.0 < kPeerTierMinGiB) return out(false, PeerTierWhy::small);
    if (!d.dry_run && !in.rule_acts) { d.dry_run = true; d.pending_ab = true; }   // B1: log only until the 50xx A/B
    return out(true, PeerTierWhy::rule_on);
}

/// The transport of the peer rows: 0 mapped host, 1 P2P.  P2P only when the pair has peer access both ways AND the readback test moved
/// real data (p2p_copies_data, #2666) AND the user asked for P2P rows (STRATA_PEER_P2P_ROWS=1); otherwise mapped host (always works).
inline int peer_tier_transport(bool can_access, bool readback_ok, bool rows_env) { return can_access && readback_ok && rows_env ? 1 : 0; }

inline const char* peer_tier_why_text(PeerTierWhy w) {
    switch (w) {
        case PeerTierWhy::env_on: return "STRATA_PEER_TIER2=1";
        case PeerTierWhy::env_off: return "STRATA_PEER_TIER2=0";
        case PeerTierWhy::user_layout: return "a layout given by hand (--layer-split / --peer-device) wins";
        case PeerTierWhy::hip: return "HIP build";
        case PeerTierWhy::v026: return "STRATA_DEFAULTS=026";
        case PeerTierWhy::upstream: return "STRATA_UPSTREAM_DEFAULTS=1";
        case PeerTierWhy::sm120_off: return "STRATA_SM120_PERF=0";
        case PeerTierWhy::one_card: return "one GPU";
        case PeerTierWhy::not_sm120: return "not every card is sm_120 (measured no gain on 2x RTX 3080 sm_86)";
        case PeerTierWhy::unlike: return "the cards are not alike (CC, SM count, VRAM within 10%)";
        case PeerTierWhy::small: return "a card has less than 15 GiB";
        case PeerTierWhy::rule_on: return "sm_120 alike cards >= 15 GiB";
    }
    return "?";
}

/// The one log line.  `transport`: "mapped-host" / "P2P"; `p2p`: whether the pair has working P2P (readback); `cc`: 10 * major + minor of card 0.
inline std::string peer_tier_log_line(const PeerTierDecision& d, int n, int cc, double gib0, const char* transport, bool p2p) {
    char b[512];
    if (d.on)
        std::snprintf(b, sizeof b, "speedy: peer tier %s%s (%s; sm_%d x%d, %.1f GiB each, transport %s, P2P %s)%s; STRATA_PEER_TIER2=0 turns it off",
                      d.pending_ab ? "WOULD be ON" : "ON", d.why == PeerTierWhy::rule_on ? " by default" : "",
                      peer_tier_why_text(d.why), d.dry_run && !d.pending_ab ? 120 : cc, n, gib0, transport, p2p ? "yes" : "no",
                      d.pending_ab ? " [A/B pending: logged only, not applied; STRATA_PEER_TIER2_AUTO=1 or STRATA_PEER_TIER2=1 applies it]"
                      : d.dry_run ? " [STRATA_EMULATE_CC=120 dry run: decided and logged only, not applied]" : "");
    else
        std::snprintf(b, sizeof b, "speedy: peer tier off: %s (sm_%d x%d)%s; STRATA_PEER_TIER2=1 forces it", peer_tier_why_text(d.why),
                      d.dry_run ? 120 : cc, n, d.dry_run ? " [STRATA_EMULATE_CC=120 dry run]" : "");
    return b;
}

}  // namespace strata
