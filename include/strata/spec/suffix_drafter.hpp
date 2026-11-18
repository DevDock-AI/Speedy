// include/strata/spec/suffix_drafter.hpp - plan v0.3 P6: the suffix-lookup drafter (no weights, no GPU).
//
// Proposes the tokens that followed the longest earlier occurrence of the sequence's current suffix: when the
// output quotes its input (edits, refactors, repeated code) the continuation is usually exact, and a verify pass
// accepts many tokens at once. It needs no model and costs microseconds.
//
// Index: every trigram of the history (prompt + accepted output) maps to its WAYS most recent end positions in a
// fixed-size open-addressing table, so memory is bounded (~20 bytes per history token) and appends are O(1).
// Optional (long_table, STRATA_SUFFIX_NGRAM8): a second table indexes every LONG_ORDER-gram the same way. A common
// trigram's few ways fill with its newest repeats; the longer key still points at an older source of a quoted stretch
// (long-context file edits). Costs one more table of the same size (host RAM only).
// A proposal checks the candidates of both, extends each match backwards up to `max_match`, and takes the longest
// (most recent on ties, long-table candidates first). Matches shorter than `min_match` propose nothing.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::spec {

class SuffixDrafter {
public:
    static constexpr int WAYS = 4;
    static constexpr int LONG_ORDER = 8;

    explicit SuffixDrafter(int min_match = 3, int max_match = 32, size_t capacity_tokens = 1u << 19,
                           bool long_table = false);

    /// STRATA_SUFFIX_NGRAM8: "1" enables the LONG_ORDER-gram table, "0" disables it, unset = on.
    static bool long_table_from_env();

    void reset();
    /// Add tokens to the history (the prompt, then every accepted token).
    void append(const int32_t* tokens, size_t n);
    void append(int32_t token) { append(&token, 1); }

    /// Write up to `max_k` proposed next tokens to `out`; returns how many (0 = no match of at least min_match).
    int propose(int max_k, int32_t* out);

    /// --lookup-chain: as propose(), for the history followed by `extra` (the draft head's proposal for the next
    /// tokens), so the lookup continues where those drafts end. The history itself is unchanged; the extra tokens are
    /// only part of the suffix being matched and may be read back as part of a periodic continuation.
    int propose_after(const int32_t* extra, int n_extra, int max_k, int32_t* out);

    /// Length of the match behind the last proposal (0 if none).
    int last_match() const { return last_match_; }
    size_t size() const { return hist_.size(); }
    const std::vector<int32_t>& history() const { return hist_; }

private:
    struct Slot {
        uint64_t key = 0;             // trigram hash + 1 (0 = empty)
        uint32_t pos[WAYS] = {};      // end positions, most recent first
        uint8_t n = 0;
    };
    Slot* find_slot(std::vector<Slot>& table, uint64_t key, bool insert);
    static void push(Slot* s, size_t end);
    uint64_t key_at(size_t end) const;
    uint64_t long_key_at(size_t end) const;

    static constexpr uint32_t LONG_SLOT_FLAG = 1u << 31;
    int min_match_, max_match_;
    std::vector<int32_t> hist_;
    std::vector<Slot> table_, long_;    // long_ empty unless long_table
    std::vector<uint32_t> used_;        // filled slot indices (LONG_SLOT_FLAG = long_), so reset() clears only those
    size_t mask_ = 0;
    int last_match_ = 0;
};

}  // namespace strata::spec
