/*
 * This file is part of Memento Filter <https://github.com/n3slami/Memento_Filter>.
 * Copyright (C) 2024 Navid Eslami.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <cstring>
#include <string>
#include <tuple>
#include <numeric>
#include <sstream>
#include <unordered_set>
#include <vector>

#include <argparse/argparse.hpp>

#include "memento.h"
#include "memento_int.h"

namespace {

using clock_type = std::chrono::high_resolution_clock;

struct OffsetScanStats {
    uint64_t saturated_blocks = 0;
    uint64_t first_saturated_block = std::numeric_limits<uint64_t>::max();
    uint8_t max_offset = 0;
};

struct Projection {
    uint64_t bucket = 0;
    uint64_t fingerprint = 0;
};

struct BlockTraceState {
    uint8_t offset = 0;
    uint64_t occupieds = 0;
    uint64_t runends = 0;
};

struct RunResult {
    uint64_t requested_inserts = 0;
    double attack_insert_ratio = 1.0;
    uint64_t attack_inserts_actual = 0;
    uint64_t random_inserts_actual = 0;
    uint64_t requested_hot_slots = 0;
    uint64_t effective_hot_slots = 0;
    uint64_t actual_inserts = 0;
    uint64_t insert_failures = 0;
    uint64_t final_saturated_blocks = 0;
    uint8_t final_max_offset = 0;
    uint64_t point_pos_checks = 0;
    uint64_t point_pos_fn = 0;
    uint64_t point_neg_checks = 0;
    uint64_t point_neg_fp = 0;
    uint64_t short_range_pos_checks = 0;
    uint64_t short_range_pos_hits = 0;
    uint64_t short_range_neg_checks = 0;
    uint64_t short_range_neg_fp = 0;
    uint64_t long_range_pos_checks = 0;
    uint64_t long_range_pos_hits = 0;
    uint64_t long_range_neg_checks = 0;
    uint64_t long_range_neg_fp = 0;
    double insert_ms = 0.0;
    double point_pos_ms = 0.0;
    double point_neg_ms = 0.0;
    double short_range_pos_ms = 0.0;
    double short_range_neg_ms = 0.0;
    double long_range_pos_ms = 0.0;
    double long_range_neg_ms = 0.0;
    double point_pos_qps = 0.0;
    double point_neg_qps = 0.0;
    double short_range_pos_qps = 0.0;
    double short_range_neg_qps = 0.0;
    double long_range_pos_qps = 0.0;
    double long_range_neg_qps = 0.0;
    uint64_t key_a = 0;
    uint64_t key_b = 0;
    uint64_t proj_bucket_a = 0;
    uint64_t proj_bucket_b = 0;
    uint64_t proj_fp_a = 0;
    uint64_t proj_fp_b = 0;
};

struct LatencySample {
    uint64_t op_index = 0;
    uint64_t latency_ns = 0;
};

struct SegmentLatencyStats {
    uint64_t count = 0;
    uint64_t min_ns = std::numeric_limits<uint64_t>::max();
    uint64_t max_ns = 0;
    long double sum_ns = 0.0L;
    uint64_t sample_cap = 0;
    std::vector<LatencySample> samples;
    std::mt19937_64 rng;

    SegmentLatencyStats() : rng(0xC001D00DULL) {}

    explicit SegmentLatencyStats(uint64_t cap, uint64_t seed)
        : sample_cap(cap), rng(seed) {
        samples.reserve(static_cast<size_t>(cap));
    }

    void observe(uint64_t op_index, uint64_t ns) {
        ++count;
        sum_ns += static_cast<long double>(ns);
        if (ns < min_ns) {
            min_ns = ns;
        }
        if (ns > max_ns) {
            max_ns = ns;
        }

        if (sample_cap == 0) {
            return;
        }
        if (samples.size() < sample_cap) {
            samples.push_back({op_index, ns});
            return;
        }
        std::uniform_int_distribution<uint64_t> pick(0, count - 1);
        const uint64_t idx = pick(rng);
        if (idx < sample_cap) {
            samples[static_cast<size_t>(idx)] = {op_index, ns};
        }
    }

    double mean_ns() const {
        return count ? static_cast<double>(sum_ns / static_cast<long double>(count)) : 0.0;
    }
};

inline uint64_t bitmask(uint64_t nbits) {
    return nbits == 64 ? 0xffffffffffffffffULL : ((1ULL << nbits) - 1ULL);
}

__attribute__((always_inline))
static inline uint32_t fast_reduce(uint32_t hash, uint32_t n) {
    return static_cast<uint32_t>((static_cast<uint64_t>(hash) * n) >> 32);
}

static inline const qfblock* get_block_ptr(const QF* qf, uint64_t block_index) {
#if QF_BITS_PER_SLOT > 0
    return &qf->blocks[block_index];
#else
    return reinterpret_cast<const qfblock*>(
        reinterpret_cast<const char*>(qf->blocks) +
        block_index * (sizeof(qfblock) + QF_SLOTS_PER_BLOCK * qf->metadata->bits_per_slot / 8)
    );
#endif
}

inline uint32_t popcnt64(uint64_t x) {
    return static_cast<uint32_t>(__builtin_popcountll(x));
}

inline uint64_t mix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

inline uint64_t pack_point(uint64_t key, uint64_t memento) {
    return mix64(key) ^ (mix64(memento + 0x9e3779b97f4a7c15ULL) << 1);
}

inline uint64_t get_slot_value(const QF* qf, uint64_t index) {
    const qfblock* b = get_block_ptr(qf, index / QF_SLOTS_PER_BLOCK);
    const uint64_t slot_in_block = index % QF_SLOTS_PER_BLOCK;

#if QF_BITS_PER_SLOT == 8 || QF_BITS_PER_SLOT == 16 || QF_BITS_PER_SLOT == 32 || QF_BITS_PER_SLOT == 64
    return b->slots[slot_in_block] & bitmask(qf->metadata->bits_per_slot);
#elif QF_BITS_PER_SLOT > 0
    uint64_t pvalue = 0;
    const uint8_t* p = &b->slots[slot_in_block * QF_BITS_PER_SLOT / 8];
    std::memcpy(&pvalue, p, sizeof(pvalue));
    return (pvalue >> ((slot_in_block * QF_BITS_PER_SLOT) % 8)) & bitmask(QF_BITS_PER_SLOT);
#else
    uint64_t pvalue = 0;
    const uint8_t* p = &b->slots[slot_in_block * qf->metadata->bits_per_slot / 8];
    std::memcpy(&pvalue, p, sizeof(pvalue));
    return (pvalue >> ((slot_in_block * qf->metadata->bits_per_slot) % 8)) & bitmask(qf->metadata->bits_per_slot);
#endif
}

inline bool is_occupied_bit(const QF* qf, uint64_t index) {
    const qfblock* b = get_block_ptr(qf, index / QF_SLOTS_PER_BLOCK);
    const uint64_t word = (index % QF_SLOTS_PER_BLOCK) / 64;
    const uint64_t bit = (index % QF_SLOTS_PER_BLOCK) % 64;
    return ((b->occupieds[word] >> bit) & 1ULL) != 0;
}

inline bool is_runend_bit(const QF* qf, uint64_t index) {
    const qfblock* b = get_block_ptr(qf, index / QF_SLOTS_PER_BLOCK);
    const uint64_t word = (index % QF_SLOTS_PER_BLOCK) / 64;
    const uint64_t bit = (index % QF_SLOTS_PER_BLOCK) % 64;
    return ((b->runends[word] >> bit) & 1ULL) != 0;
}

void dump_slots_for_blocks(const QF* qf,
                           uint64_t op_index,
                           uint64_t block_start,
                           uint64_t block_count) {
    const uint64_t end_block = std::min<uint64_t>(qf->metadata->nblocks, block_start + block_count);
    std::cout << "[slotdump] op=" << op_index
              << " block_start=" << block_start
              << " block_count=" << block_count
              << " end_block=" << end_block
              << " bits_per_slot=" << qf->metadata->bits_per_slot
              << " fp_bits=" << qf->metadata->fingerprint_bits
              << " memento_bits=" << qf->metadata->memento_bits
              << "\n";

    for (uint64_t bi = block_start; bi < end_block; ++bi) {
        const qfblock* b = get_block_ptr(qf, bi);
        std::cout << "[slotdump] block=" << bi
                  << " offset=" << static_cast<uint64_t>(b->offset)
                  << " occupieds=0x" << std::hex << b->occupieds[0]
                  << " runends=0x" << b->runends[0]
                  << std::dec << "\n";

        const uint64_t base = bi * QF_SLOTS_PER_BLOCK;
        for (uint64_t si = 0; si < QF_SLOTS_PER_BLOCK; ++si) {
            const uint64_t global_slot = base + si;
            const uint64_t raw = get_slot_value(qf, global_slot);
            const uint64_t fp = raw >> qf->metadata->memento_bits;
            const uint64_t mem = raw & bitmask(qf->metadata->memento_bits);
            const bool occ = is_occupied_bit(qf, global_slot);
            const bool run = is_runend_bit(qf, global_slot);
            std::cout << "[slotdump]   slot=" << global_slot
                      << " inblock=" << si
                      << " raw=" << raw
                      << " fp=" << fp
                      << " memento=" << mem
                      << " occupied=" << (occ ? 1 : 0)
                      << " runend=" << (run ? 1 : 0)
                      << "\n";
        }
    }
}

Projection project_key_hash(const QF* qf, uint64_t key_hash) {
    const uint64_t bucket_index_hash_size = qf->metadata->key_bits - qf->metadata->fingerprint_bits;
    const uint64_t orig_q_bits = qf->metadata->original_quotient_bits;
    const uint64_t orig_nslots = qf->metadata->nslots >> (bucket_index_hash_size - orig_q_bits);

    const uint64_t reduced = fast_reduce(
        static_cast<uint32_t>((key_hash & bitmask(orig_q_bits)) << (32 - orig_q_bits)),
        static_cast<uint32_t>(orig_nslots)
    );

    uint64_t key = key_hash;
    key &= ~bitmask(orig_q_bits);
    key |= reduced;

    Projection out;
    out.fingerprint = (key >> bucket_index_hash_size) & bitmask(qf->metadata->fingerprint_bits);
    out.bucket = (reduced << (bucket_index_hash_size - orig_q_bits)) |
                 ((key >> orig_q_bits) & bitmask(bucket_index_hash_size - orig_q_bits));
    return out;
}

OffsetScanStats scan_offsets(const QF* qf) {
    OffsetScanStats s;
    for (uint64_t i = 0; i < qf->metadata->nblocks; ++i) {
        const uint8_t off = get_block_ptr(qf, i)->offset;
        if (off == std::numeric_limits<uint8_t>::max()) {
            ++s.saturated_blocks;
            if (s.first_saturated_block == std::numeric_limits<uint64_t>::max()) {
                s.first_saturated_block = i;
            }
        }
        if (off > s.max_offset) {
            s.max_offset = off;
        }
    }
    return s;
}

uint64_t find_key_for_projection(const QF* qf,
                                 uint64_t start,
                                 uint64_t target_bucket,
                                 int64_t target_fp,
                                 uint64_t max_tries) {
    for (uint64_t x = start, t = 0; t < max_tries; ++x, ++t) {
        Projection p = project_key_hash(qf, x);
        if (p.bucket != target_bucket) {
            continue;
        }
        if (target_fp >= 0 && p.fingerprint != static_cast<uint64_t>(target_fp)) {
            continue;
        }
        if (p.fingerprint == 0) {
            continue;
        }
        return x;
    }
    return std::numeric_limits<uint64_t>::max();
}

bool find_random_key_for_bucket(const QF* qf,
                                uint64_t target_bucket,
                                uint64_t max_tries,
                                std::mt19937_64& rng,
                                std::uniform_int_distribution<uint64_t>& dist,
                                uint64_t& out_key) {
    for (uint64_t t = 0; t < max_tries; ++t) {
        const uint64_t key = dist(rng);
        const Projection p = project_key_hash(qf, key);
        if (p.bucket != target_bucket) {
            continue;
        }
        if (p.fingerprint == 0) {
            continue;
        }
        out_key = key;
        return true;
    }
    return false;
}

struct BucketKeyPlan {
    uint64_t bucket_index_hash_size = 0;
    uint64_t orig_q_bits = 0;
    uint64_t upper_bits = 0;
    uint64_t orig_nslots = 0;
    uint64_t fp_bits = 0;
};

bool init_bucket_key_plan(const QF* qf, BucketKeyPlan& plan) {
    if (!qf || !qf->metadata) {
        return false;
    }
    plan.fp_bits = qf->metadata->fingerprint_bits;
    if (plan.fp_bits >= 63) {
        return false;
    }
    plan.bucket_index_hash_size = qf->metadata->key_bits - plan.fp_bits;
    plan.orig_q_bits = qf->metadata->original_quotient_bits;
    if (plan.orig_q_bits == 0 || plan.orig_q_bits > 32) {
        return false;
    }
    plan.upper_bits = plan.bucket_index_hash_size - plan.orig_q_bits;
    plan.orig_nslots = qf->metadata->nslots >> plan.upper_bits;
    if (plan.orig_nslots == 0) {
        return false;
    }
    return true;
}

std::vector<uint64_t> build_inverse_reduce(const BucketKeyPlan& plan) {
    std::vector<uint64_t> inv(plan.orig_nslots, std::numeric_limits<uint64_t>::max());
    const uint64_t q_mask = bitmask(plan.orig_q_bits);
    const uint64_t total = 1ULL << plan.orig_q_bits;
    for (uint64_t qbits = 0; qbits < total; ++qbits) {
        const uint32_t hash = static_cast<uint32_t>((qbits & q_mask) << (32 - plan.orig_q_bits));
        const uint32_t reduced = fast_reduce(hash, static_cast<uint32_t>(plan.orig_nslots));
        if (inv[reduced] == std::numeric_limits<uint64_t>::max()) {
            inv[reduced] = qbits;
        }
    }
    return inv;
}

bool build_key_for_bucket(const QF* qf,
                          const BucketKeyPlan& plan,
                          const std::vector<uint64_t>& inv_reduce,
                          uint64_t target_bucket,
                          uint64_t fp_value,
                          uint64_t& out_key) {
    if (plan.orig_nslots == 0 || target_bucket >= qf->metadata->nslots) {
        return false;
    }
    const uint64_t reduced = target_bucket >> plan.upper_bits;
    const uint64_t upper = target_bucket & bitmask(plan.upper_bits);
    if (reduced >= plan.orig_nslots) {
        return false;
    }
    const uint64_t qbits = inv_reduce[reduced];
    if (qbits == std::numeric_limits<uint64_t>::max()) {
        return false;
    }
    uint64_t key = 0;
    key |= qbits;
    key |= (upper << plan.orig_q_bits);
    if (plan.fp_bits > 0) {
        const uint64_t fp_mask = bitmask(plan.fp_bits);
        const uint64_t fp = (fp_value & fp_mask);
        if (fp == 0) {
            return false;
        }
        key |= (fp << plan.bucket_index_hash_size);
    }

    const Projection p = project_key_hash(qf, key);
    if (p.bucket != target_bucket || p.fingerprint == 0) {
        return false;
    }
    out_key = key;
    return true;
}

void print_usage_header() {
    std::cout << "[+] Running memento offset-overflow experiment" << std::endl;
}

bool probe_variable_length_counter_encoding(uint64_t key_bits,
                                            uint64_t memento_bits,
                                            uint32_t seed,
                                            uint64_t key_search_tries) {
    if (memento_bits == 0 || memento_bits > 15) {
        return false;
    }

    const uint64_t per_prefix_cap = 1ULL << memento_bits;
    const uint64_t probe_inserts = per_prefix_cap + 1; // force counter overflow path
    const uint64_t probe_nslots = std::max<uint64_t>(8192, probe_inserts * 4);

    QF* test_qf = static_cast<QF*>(malloc(sizeof(QF)));
    if (!test_qf) {
        return false;
    }

    bool ok = false;
    if (!qf_malloc(test_qf, probe_nslots, key_bits, memento_bits, QF_HASH_DEFAULT, seed ^ 0xA5A5A5A5U)) {
        free(test_qf);
        return false;
    }
    qf_set_auto_resize(test_qf, false);

    const uint64_t key = find_key_for_projection(test_qf, 1, 0, -1, key_search_tries);
    if (key != std::numeric_limits<uint64_t>::max()) {
        ok = true;
        for (uint64_t i = 0; i < probe_inserts; ++i) {
            if (qf_insert_single(test_qf, key, 0, QF_NO_LOCK | QF_KEY_IS_HASH) < 0) {
                ok = false;
                break;
            }
        }
    }

    qf_free(test_qf);
    free(test_qf);
    return ok;
}

void append_csv(const std::string& file,
                const std::string& mode,
                const QF* qf,
                const RunResult& r,
                const std::string& positive_range_protocol) {
    const bool needs_header = !std::ifstream(file).good() || std::ifstream(file).peek() == std::ifstream::traits_type::eof();
    std::ofstream out(file, std::ios::app);
    if (!out.good()) {
        std::cerr << "[!] failed to open csv file: " << file << std::endl;
        return;
    }

    if (needs_header) {
        out << "mode,positive_range_protocol,attack_insert_ratio,attack_inserts_actual,random_inserts_actual,requested_hot_slots,effective_hot_slots,nslots,nblocks,key_bits,fingerprint_bits,memento_bits,bits_per_slot,"
            << "requested_inserts,actual_inserts,insert_failures,"
            << "final_saturated_blocks,final_saturated_ratio,final_max_offset,"
            << "point_pos_checks,point_pos_fn,point_neg_checks,point_neg_fp,"
            << "short_range_pos_checks,short_range_pos_hits,short_range_neg_checks,short_range_neg_fp,"
            << "long_range_pos_checks,long_range_pos_hits,long_range_neg_checks,long_range_neg_fp,"
            << "insert_ms,point_pos_ms,point_neg_ms,short_range_pos_ms,short_range_neg_ms,long_range_pos_ms,long_range_neg_ms,"
            << "point_pos_qps,point_neg_qps,short_range_pos_qps,short_range_neg_qps,long_range_pos_qps,long_range_neg_qps,"
            << "key_a,key_b,proj_bucket_a,proj_bucket_b,proj_fp_a,proj_fp_b\n";
    }

    const double saturated_ratio = qf->metadata->nblocks == 0
        ? 0.0
        : static_cast<double>(r.final_saturated_blocks) / static_cast<double>(qf->metadata->nblocks);

    out << mode << ","
        << positive_range_protocol << ","
        << r.attack_insert_ratio << ","
        << r.attack_inserts_actual << ","
        << r.random_inserts_actual << ","
        << r.requested_hot_slots << ","
        << r.effective_hot_slots << ","
        << qf->metadata->nslots << ","
        << qf->metadata->nblocks << ","
        << qf->metadata->key_bits << ","
        << qf->metadata->fingerprint_bits << ","
        << qf->metadata->memento_bits << ","
        << qf->metadata->bits_per_slot << ","
        << r.requested_inserts << ","
        << r.actual_inserts << ","
        << r.insert_failures << ","
        << r.final_saturated_blocks << ","
        << saturated_ratio << ","
        << static_cast<uint64_t>(r.final_max_offset) << ","
        << r.point_pos_checks << ","
        << r.point_pos_fn << ","
        << r.point_neg_checks << ","
        << r.point_neg_fp << ","
        << r.short_range_pos_checks << ","
        << r.short_range_pos_hits << ","
        << r.short_range_neg_checks << ","
        << r.short_range_neg_fp << ","
        << r.long_range_pos_checks << ","
        << r.long_range_pos_hits << ","
        << r.long_range_neg_checks << ","
        << r.long_range_neg_fp << ","
        << r.insert_ms << ","
        << r.point_pos_ms << ","
        << r.point_neg_ms << ","
        << r.short_range_pos_ms << ","
        << r.short_range_neg_ms << ","
        << r.long_range_pos_ms << ","
        << r.long_range_neg_ms << ","
        << r.point_pos_qps << ","
        << r.point_neg_qps << ","
        << r.short_range_pos_qps << ","
        << r.short_range_neg_qps << ","
        << r.long_range_pos_qps << ","
        << r.long_range_neg_qps << ","
        << r.key_a << ","
        << r.key_b << ","
        << r.proj_bucket_a << ","
        << r.proj_bucket_b << ","
        << r.proj_fp_a << ","
        << r.proj_fp_b << "\n";
}

void print_blocks_snapshot(const QF* qf,
                           uint64_t op_index,
                           uint64_t key_hash,
                           uint64_t memento,
                           const Projection& proj) {
    std::cout << "[trace] op=" << op_index
              << " key_hash=" << key_hash
              << " bucket=" << proj.bucket
              << " fp=" << proj.fingerprint
              << " memento=" << memento
              << " nblocks=" << qf->metadata->nblocks
              << "\n";

    for (uint64_t bi = 0; bi < qf->metadata->nblocks; ++bi) {
        const qfblock* b = get_block_ptr(qf, bi);
        const uint64_t occ_bits = b->occupieds[0];
        const uint64_t run_bits = b->runends[0];
        std::cout << "[trace]   block=" << bi
                  << " offset=" << static_cast<uint64_t>(b->offset)
                  << " occ_popcnt=" << popcnt64(occ_bits)
                  << " run_popcnt=" << popcnt64(run_bits)
                  << " occupieds=0x" << std::hex << occ_bits
                  << " runends=0x" << run_bits
                  << std::dec << "\n";
    }
}

void print_blocks_snapshot_delta(const QF* qf,
                                 uint64_t op_index,
                                 uint64_t key_hash,
                                 uint64_t memento,
                                 const Projection& proj,
                                 std::vector<BlockTraceState>& prev_state,
                                 bool force_full) {
    if (prev_state.size() != qf->metadata->nblocks) {
        prev_state.assign(qf->metadata->nblocks, {});
        force_full = true;
    }

    std::cout << "[trace] op=" << op_index
              << " key_hash=" << key_hash
              << " bucket=" << proj.bucket
              << " fp=" << proj.fingerprint
              << " memento=" << memento
              << " nblocks=" << qf->metadata->nblocks
              << (force_full ? " mode=full" : " mode=delta")
              << "\n";

    uint64_t changed = 0;
    for (uint64_t bi = 0; bi < qf->metadata->nblocks; ++bi) {
        const qfblock* b = get_block_ptr(qf, bi);
        const uint64_t occ_bits = b->occupieds[0];
        const uint64_t run_bits = b->runends[0];

        const bool has_changed = force_full
            || prev_state[bi].offset != b->offset
            || prev_state[bi].occupieds != occ_bits
            || prev_state[bi].runends != run_bits;

        if (has_changed) {
            ++changed;
            std::cout << "[trace]   block=" << bi
                      << " offset=" << static_cast<uint64_t>(b->offset)
                      << " occ_popcnt=" << popcnt64(occ_bits)
                      << " run_popcnt=" << popcnt64(run_bits)
                      << " occupieds=0x" << std::hex << occ_bits
                      << " runends=0x" << run_bits
                      << std::dec << "\n";
        }

        prev_state[bi].offset = b->offset;
        prev_state[bi].occupieds = occ_bits;
        prev_state[bi].runends = run_bits;
    }

    if (!force_full) {
        std::cout << "[trace]   changed_blocks=" << changed << "\n";
    }
}

} // namespace

int main(int argc, char const *argv[]) {
    argparse::ArgumentParser parser("bench-memento-overflow");
    parser.add_argument("arg").help("target bits per key (bpk)").scan<'g', double>();
    parser.add_argument("--mode")
        .help("attack mode: single-prefix | dual-hot | x-hot | sq | random")
        .default_value(std::string("single-prefix"));
    parser.add_argument("--hot-slots")
        .help("number of adjacent hot buckets (used when mode=x-hot)")
        .default_value(uint64_t(2))
        .scan<'u', uint64_t>();
    parser.add_argument("--xhot-schedule")
        .help("x-hot insertion schedule: round-robin | burst | zipf")
        .default_value(std::string("round-robin"));
    parser.add_argument("--burst-len")
        .help("burst length for xhot-schedule=burst")
        .default_value(uint64_t(64))
        .scan<'u', uint64_t>();
    parser.add_argument("--zipf-s")
        .help("zipf parameter s for xhot-schedule=zipf")
        .default_value(1.2)
        .scan<'g', double>();
    parser.add_argument("--requested-inserts")
        .help("number of insert operations to issue")
        .default_value(uint64_t(1024))
        .scan<'u', uint64_t>();
    parser.add_argument("--attack-insert-ratio")
        .help("fraction of insertions using attack mode in non-random modes; remaining insertions are random")
        .default_value(1.0)
        .scan<'g', double>();
    parser.add_argument("--query-count")
        .help("number of queries per query mode (point/short-range/long-range)")
        .default_value(uint64_t(200000))
        .scan<'u', uint64_t>();
    parser.add_argument("--strict-load-factor")
        .help("size nslots as ceil(requested_inserts / target_load) without extra slack")
        .default_value(false)
        .implicit_value(true);
    parser.add_argument("--run-negative-queries")
        .help("also execute negative query workloads (point/short-range/long-range)")
        .default_value(false)
        .implicit_value(true);
    parser.add_argument("--allow-duplicate-mementos")
        .help("for non-random modes, allow reusing memento values when variable-length counter encoding is supported")
        .default_value(true)
        .implicit_value(true);
    parser.add_argument("--positive-range-protocol")
        .help("positive range query protocol: start-at-hit | contains-hit")
        .default_value(std::string("start-at-hit"));
    parser.add_argument("--target-load")
        .help("nslots sizing load target")
        .default_value(0.95)
        .scan<'g', double>();
    parser.add_argument("--fixed-nslots")
        .help("force nslots to this value (0 means auto-size from requested-inserts/target-load)")
        .default_value(uint64_t(0))
        .scan<'u', uint64_t>();
    parser.add_argument("--memento-size")
        .help("memento bits; -1 means auto")
        .default_value(-1)
        .scan<'i', int>();
    parser.add_argument("--seed")
        .help("hash seed for qf")
        .default_value(uint32_t(1380))
        .scan<'u', uint32_t>();
    parser.add_argument("--key-search-tries")
        .help("max tries to find crafted hash-keys")
        .default_value(uint64_t(5000000))
        .scan<'u', uint64_t>();
    parser.add_argument("--csv")
        .help("append result to csv file")
        .default_value(std::string(""));
    parser.add_argument("--latency-profile")
        .help("collect segmented latency profile for insert/point/range loops")
        .default_value(false)
        .implicit_value(true);
    parser.add_argument("--latency-sample-size")
        .help("per-segment reservoir sample size for latency distribution output")
        .default_value(uint64_t(5000))
        .scan<'u', uint64_t>();
    parser.add_argument("--latency-summary-csv")
        .help("append latency summary rows to csv file")
        .default_value(std::string(""));
    parser.add_argument("--latency-samples-csv")
        .help("append sampled latency rows to csv file")
        .default_value(std::string(""));
    parser.add_argument("--latency-tag")
        .help("tag written into latency csv rows")
        .default_value(std::string(""));
    parser.add_argument("--trace-blocks")
        .help("print per-operation per-block storage snapshot (intended for dual-hot debugging)")
        .default_value(false)
        .implicit_value(true);
    parser.add_argument("--trace-max-ops")
        .help("limit number of traced operations; 0 means trace all operations")
        .default_value(uint64_t(0))
        .scan<'u', uint64_t>();
    parser.add_argument("--trace-delta-only")
        .help("print only blocks changed since previous operation (dual-hot trace)")
        .default_value(false)
        .implicit_value(true);
    parser.add_argument("--dump-slots-at-op")
        .help("dump per-slot state at this successful insert index; 0 disables")
        .default_value(uint64_t(0))
        .scan<'u', uint64_t>();
    parser.add_argument("--dump-block-start")
        .help("start block index for per-slot dump")
        .default_value(uint64_t(0))
        .scan<'u', uint64_t>();
    parser.add_argument("--dump-block-count")
        .help("number of blocks for per-slot dump")
        .default_value(uint64_t(6))
        .scan<'u', uint64_t>();

    parser.add_argument("--def-reconstruct")
        .help("enable lightweight reconstruct (resize) when offsets exceed threshold")
        .default_value(false)
        .implicit_value(true);
    parser.add_argument("--reconstruct-threshold")
        .help("offset threshold to trigger reconstruct (per-block offset)")
        .default_value(uint64_t(32))
        .scan<'u', uint64_t>();

    parser.add_argument("--def-adaptive-verify")
        .help("enable adaptive verification when block offset >= threshold")
        .default_value(false)
        .implicit_value(true);
    parser.add_argument("--verify-threshold")
        .help("offset threshold to trigger adaptive verification")
        .default_value(uint64_t(16))
        .scan<'u', uint64_t>();
    parser.add_argument("--def-keepsake-rle")
        .help("enable keepsake-box run-length / counter compression for repeated mementos")
        .default_value(false)
        .implicit_value(true);
    parser.add_argument("--def-reconstruct")
        .help("enable one-time reconstruction/re-encoding when offset reaches threshold")
        .default_value(false)
        .implicit_value(true);
    parser.add_argument("--reconstruct-threshold")
        .help("offset threshold that triggers reconstruction")
        .default_value(uint64_t(32))
        .scan<'u', uint64_t>();

    try {
        parser.parse_args(argc, argv);
    } catch (const std::runtime_error &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    print_usage_header();

    const double bpk = parser.get<double>("arg");
    const std::string mode = parser.get<std::string>("--mode");
    const uint64_t requested_inserts = parser.get<uint64_t>("--requested-inserts");
    const double attack_insert_ratio = parser.get<double>("--attack-insert-ratio");
    const uint64_t query_count = parser.get<uint64_t>("--query-count");
    std::string xhot_schedule = parser.get<std::string>("--xhot-schedule");
    const uint64_t burst_len = parser.get<uint64_t>("--burst-len");
    const double zipf_s = parser.get<double>("--zipf-s");
    const bool strict_load_factor = parser.get<bool>("--strict-load-factor");
    const bool run_negative_queries = parser.get<bool>("--run-negative-queries");
    const bool allow_duplicate_mementos_requested = parser.get<bool>("--allow-duplicate-mementos");
    const std::string positive_range_protocol = parser.get<std::string>("--positive-range-protocol");
    const double target_load = parser.get<double>("--target-load");
    const uint64_t fixed_nslots = parser.get<uint64_t>("--fixed-nslots");
    const int memento_size_arg = parser.get<int>("--memento-size");
    const uint32_t seed = parser.get<uint32_t>("--seed");
    const uint64_t key_search_tries = parser.get<uint64_t>("--key-search-tries");
    const std::string csv_file = parser.get<std::string>("--csv");
    const bool latency_profile = parser.get<bool>("--latency-profile");
    const uint64_t latency_sample_size = parser.get<uint64_t>("--latency-sample-size");
    const std::string latency_summary_csv = parser.get<std::string>("--latency-summary-csv");
    const std::string latency_samples_csv = parser.get<std::string>("--latency-samples-csv");
    const std::string latency_tag = parser.get<std::string>("--latency-tag");
    const bool trace_blocks = parser.get<bool>("--trace-blocks");
    const uint64_t trace_max_ops = parser.get<uint64_t>("--trace-max-ops");
    const bool trace_delta_only = parser.get<bool>("--trace-delta-only");
    const uint64_t dump_slots_at_op = parser.get<uint64_t>("--dump-slots-at-op");
    const uint64_t dump_block_start = parser.get<uint64_t>("--dump-block-start");
    const uint64_t dump_block_count = parser.get<uint64_t>("--dump-block-count");

    const bool def_adaptive_verify = parser.get<bool>("--def-adaptive-verify");
    const uint64_t verify_threshold = parser.get<uint64_t>("--verify-threshold");
    const bool def_keepsake_rle = parser.get<bool>("--def-keepsake-rle");
    const bool def_reconstruct = parser.get<bool>("--def-reconstruct");
    const uint64_t reconstruct_threshold = parser.get<uint64_t>("--reconstruct-threshold");

    if (mode != "single-prefix" && mode != "dual-hot" && mode != "x-hot" && mode != "random" && mode != "sq") {
        std::cerr << "[!] unsupported mode: " << mode << std::endl;
        return 2;
    }
    if (mode == "x-hot") {
        if (xhot_schedule != "round-robin" && xhot_schedule != "burst" && xhot_schedule != "zipf") {
            std::cerr << "[!] unsupported --xhot-schedule: " << xhot_schedule << std::endl;
            return 2;
        }
    }
    if (burst_len == 0) {
        std::cerr << "[!] --burst-len must be >= 1" << std::endl;
        return 2;
    }
    if (zipf_s <= 0.0) {
        std::cerr << "[!] --zipf-s must be > 0" << std::endl;
        return 2;
    }
    if (attack_insert_ratio < 0.0 || attack_insert_ratio > 1.0) {
        std::cerr << "[!] --attack-insert-ratio must be in [0, 1]" << std::endl;
        return 2;
    }
    if (positive_range_protocol != "start-at-hit" && positive_range_protocol != "contains-hit") {
        std::cerr << "[!] unsupported --positive-range-protocol: " << positive_range_protocol << std::endl;
        return 2;
    }
    if (fixed_nslots > 0 && fixed_nslots < 256) {
        std::cerr << "[!] --fixed-nslots must be 0 or >= 256" << std::endl;
        return 2;
    }

    uint64_t requested_hot_slots = 1;
    if (mode == "dual-hot") {
        requested_hot_slots = 2;
    } else if (mode == "x-hot") {
        requested_hot_slots = parser.get<uint64_t>("--hot-slots");
    } else if (mode == "random") {
        requested_hot_slots = 0;
    } else if (mode == "sq") {
        requested_hot_slots = 0;
    }
    if (mode != "random" && mode != "sq" && requested_hot_slots == 0) {
        std::cerr << "[!] --hot-slots must be >= 1" << std::endl;
        return 2;
    }

    uint32_t memento_bits = 8;
    if (memento_size_arg >= 0) {
        memento_bits = static_cast<uint32_t>(memento_size_arg);
    }

    const uint64_t per_prefix_cap = (memento_bits >= 63)
        ? std::numeric_limits<uint64_t>::max()
        : (1ULL << memento_bits);

    uint64_t n_slots = fixed_nslots;
    if (n_slots == 0) {
        const uint64_t approx_required_prefixes = std::max<uint64_t>(1, requested_hot_slots);
        const uint64_t logical_items = std::max<uint64_t>(requested_inserts, approx_required_prefixes);
        n_slots = strict_load_factor
            ? static_cast<uint64_t>(std::ceil(static_cast<double>(logical_items) / target_load))
            : static_cast<uint64_t>(std::ceil(static_cast<double>(logical_items) / target_load + std::sqrt(static_cast<double>(logical_items))));
        if (n_slots < 256) {
            n_slots = 256;
        }
    }

    int64_t fingerprint_size_signed = static_cast<int64_t>(std::llround(bpk * target_load - static_cast<double>(memento_bits) - 2.125));
    if (fingerprint_size_signed < 1) {
        fingerprint_size_signed = 1;
    }
    const uint64_t fingerprint_size = static_cast<uint64_t>(fingerprint_size_signed);

    uint64_t quotient_bits = 0;
    while ((1ULL << quotient_bits) <= n_slots && quotient_bits < 63) {
        ++quotient_bits;
    }
    const uint64_t key_bits = quotient_bits + fingerprint_size;

    const bool variable_length_counter_encoding = probe_variable_length_counter_encoding(
        key_bits, memento_bits, seed, key_search_tries);
    const bool allow_duplicate_mementos = allow_duplicate_mementos_requested && variable_length_counter_encoding;

    if (mode == "sq") {
        xhot_schedule = "sq";
    }

    std::cout << "[+] mode=" << mode
              << " hot_slots=" << requested_hot_slots
              << " requested_inserts=" << requested_inserts
              << " attack_insert_ratio=" << attack_insert_ratio
              << " query_count=" << query_count
              << " xhot_schedule=" << xhot_schedule
              << " burst_len=" << burst_len
              << " zipf_s=" << zipf_s
              << " strict_load_factor=" << (strict_load_factor ? 1 : 0)
              << " run_negative_queries=" << (run_negative_queries ? 1 : 0)
              << " variable_length_counter_encoding=" << (variable_length_counter_encoding ? 1 : 0)
              << " allow_duplicate_mementos=" << (allow_duplicate_mementos ? 1 : 0)
              << " positive_range_protocol=" << positive_range_protocol
              << " n_slots=" << n_slots
              << " key_bits=" << key_bits
              << " fingerprint_bits=" << fingerprint_size
              << " memento_bits=" << memento_bits
              << std::endl;

    QF *qf = static_cast<QF *>(malloc(sizeof(QF)));
    if (!qf) {
        std::cerr << "[!] malloc QF failed" << std::endl;
        return 3;
    }

    if (!qf_malloc(qf, n_slots, key_bits, memento_bits, QF_HASH_DEFAULT, seed)) {
        std::cerr << "[!] qf_malloc failed" << std::endl;
        free(qf);
        return 4;
    }
    qf_set_auto_resize(qf, false);

    std::vector<std::pair<uint64_t, uint64_t>> inserted_pairs;
    inserted_pairs.reserve(requested_inserts);

    RunResult result;
    result.requested_inserts = requested_inserts;
    result.attack_insert_ratio = attack_insert_ratio;
    result.requested_hot_slots = requested_hot_slots;

    std::vector<uint64_t> hot_keys;
    std::vector<Projection> hot_projections;
    hot_keys.reserve(requested_hot_slots);
    hot_projections.reserve(requested_hot_slots);

    if (mode != "random" && mode != "sq") {
        // Find a non-zero-fingerprint crafted hash-key for one bucket.
        uint64_t key_a = find_key_for_projection(qf, 1, 0, -1, key_search_tries);
        if (key_a == std::numeric_limits<uint64_t>::max()) {
            std::cerr << "[!] failed to find first crafted key within search budget" << std::endl;
            qf_free(qf);
            free(qf);
            return 5;
        }

        Projection p_a = project_key_hash(qf, key_a);
        hot_keys.push_back(key_a);
        hot_projections.push_back(p_a);

        result.key_a = key_a;
        result.proj_bucket_a = p_a.bucket;
        result.proj_fp_a = p_a.fingerprint;

        uint64_t next_search_start = key_a + 1;
        for (uint64_t i = 1; i < requested_hot_slots; ++i) {
            const uint64_t target_bucket = (p_a.bucket + i) % qf->metadata->nslots;
            const uint64_t key_i = find_key_for_projection(qf, next_search_start, target_bucket, -1, key_search_tries);
            if (key_i == std::numeric_limits<uint64_t>::max()) {
                std::cerr << "[!] failed to find crafted key for hot bucket index=" << i
                          << " target_bucket=" << target_bucket << std::endl;
                qf_free(qf);
                free(qf);
                return 6;
            }
            Projection p_i = project_key_hash(qf, key_i);
            hot_keys.push_back(key_i);
            hot_projections.push_back(p_i);
            next_search_start = key_i + 1;
        }
    }

    result.effective_hot_slots = hot_keys.size();
    if (hot_keys.size() >= 2) {
        result.key_b = hot_keys[1];
        result.proj_bucket_b = hot_projections[1].bucket;
        result.proj_fp_b = hot_projections[1].fingerprint;
    }

    std::vector<uint64_t> used_per_hot(hot_keys.size(), 0);
    std::vector<BlockTraceState> trace_prev_state;
    std::unordered_map<uint64_t, uint32_t> keepsake_counters;
    std::mt19937_64 rng(static_cast<uint64_t>(seed) ^ 0x9e3779b97f4a7c15ULL);
    std::discrete_distribution<uint64_t> zipf_dist;
    if (mode == "x-hot" && !hot_keys.empty() && xhot_schedule == "zipf") {
        std::vector<double> weights(hot_keys.size(), 1.0);
        for (uint64_t i = 0; i < hot_keys.size(); ++i) {
            weights[i] = 1.0 / std::pow(static_cast<double>(i + 1), zipf_s);
        }
        zipf_dist = std::discrete_distribution<uint64_t>(weights.begin(), weights.end());
    }
    std::uniform_int_distribution<uint64_t> random_key_dist(0, bitmask(std::min<uint64_t>(key_bits, 64)));
    std::uniform_int_distribution<uint64_t> random_memento_dist(0, per_prefix_cap - 1);
    std::bernoulli_distribution attack_insert_dist(attack_insert_ratio);
    uint64_t attack_insert_seq = 0;

    uint64_t sq_target_inserts = 0;
    uint64_t sq_inserted = 0;
    uint64_t sq_next_l = 0;
    bool reconstruct_done = false;
    if (mode == "sq") {
        const double target_f = std::floor(static_cast<double>(requested_inserts) * attack_insert_ratio);
        sq_target_inserts = static_cast<uint64_t>(target_f);
        if (attack_insert_ratio > 0.0 && sq_target_inserts == 0) {
            sq_target_inserts = 1;
        }
        if (sq_target_inserts > qf->metadata->nslots) {
            sq_target_inserts = qf->metadata->nslots;
        }
    }

    SegmentLatencyStats insert_prepare(latency_sample_size, seed ^ 0x1001ULL);
    SegmentLatencyStats insert_core(latency_sample_size, seed ^ 0x1002ULL);
    SegmentLatencyStats insert_post(latency_sample_size, seed ^ 0x1003ULL);
    SegmentLatencyStats point_prepare(latency_sample_size, seed ^ 0x2001ULL);
    SegmentLatencyStats point_core(latency_sample_size, seed ^ 0x2002ULL);
    SegmentLatencyStats short_prepare(latency_sample_size, seed ^ 0x3001ULL);
    SegmentLatencyStats short_core(latency_sample_size, seed ^ 0x3002ULL);
    SegmentLatencyStats long_prepare(latency_sample_size, seed ^ 0x4001ULL);
    SegmentLatencyStats long_core(latency_sample_size, seed ^ 0x4002ULL);

    uint64_t insert_core_ns_sum = 0;
    for (uint64_t i = 0; i < requested_inserts; ++i) {
        const auto seg0 = clock_type::now();
        uint64_t key = 0;
        uint64_t memento = 0;

        bool use_attack_insert = (mode != "random") && attack_insert_dist(rng);
        if (mode == "sq") {
            use_attack_insert = (sq_inserted < sq_target_inserts);
        }

        if (mode == "random" || !use_attack_insert) {
            key = random_key_dist(rng);
            memento = random_memento_dist(rng);
            ++result.random_inserts_actual;
        } else if (mode == "sq") {
            if (sq_inserted >= sq_target_inserts) {
                key = random_key_dist(rng);
                memento = random_memento_dist(rng);
                ++result.random_inserts_actual;
            } else {
                const uint64_t target_bucket = sq_next_l;
                if (target_bucket >= qf->metadata->nslots) {
                    break;
                }
                uint64_t candidate = 0;
                bool found = find_random_key_for_bucket(
                    qf, target_bucket, key_search_tries, rng, random_key_dist, candidate);
                if (!found) {
                    break;
                }
                key = candidate;
                memento = random_memento_dist(rng);
                ++sq_next_l;
                ++sq_inserted;
                ++result.attack_inserts_actual;
                ++attack_insert_seq;
            }
        } else {
            uint64_t chosen_hot_idx = std::numeric_limits<uint64_t>::max();
            const uint64_t rr_idx = hot_keys.empty() ? 0 : (attack_insert_seq % hot_keys.size());
            const uint64_t burst_idx = hot_keys.empty() ? 0 : ((attack_insert_seq / burst_len) % hot_keys.size());
            const uint64_t zipf_idx = (mode == "x-hot" && !hot_keys.empty() && xhot_schedule == "zipf")
                ? zipf_dist(rng)
                : rr_idx;
            uint64_t preferred_idx = rr_idx;
            if (mode == "x-hot") {
                if (xhot_schedule == "burst") {
                    preferred_idx = burst_idx;
                } else if (xhot_schedule == "zipf") {
                    preferred_idx = zipf_idx;
                }
            }

            if (allow_duplicate_mementos) {
                if (!hot_keys.empty()) {
                    chosen_hot_idx = preferred_idx;
                }
            } else {
                for (uint64_t probe = 0; probe < hot_keys.size(); ++probe) {
                    const uint64_t idx = (preferred_idx + probe) % hot_keys.size();
                    if (used_per_hot[idx] < per_prefix_cap) {
                        chosen_hot_idx = idx;
                        break;
                    }
                }
            }

            if (chosen_hot_idx == std::numeric_limits<uint64_t>::max()) {
                break;
            }

            key = hot_keys[chosen_hot_idx];
            if (allow_duplicate_mementos) {
                memento = used_per_hot[chosen_hot_idx] % per_prefix_cap;
                ++used_per_hot[chosen_hot_idx];
            } else {
                memento = used_per_hot[chosen_hot_idx]++;
            }
            ++result.attack_inserts_actual;
            ++attack_insert_seq;
        }
        Projection proj = project_key_hash(qf, key);

        bool do_insert = true;
        int64_t rc = 0;

        if (def_adaptive_verify) {
            // Use adaptive query with memento list size threshold
            const int pq = qf_point_query_adaptive(qf, key, memento, QF_NO_LOCK | QF_KEY_IS_HASH, verify_threshold);
            if (pq > 0) {
                do_insert = false; // already present per adaptive verification
            }
        }

        if (def_reconstruct && !reconstruct_done) {
            const uint64_t block_idx = proj.bucket / QF_SLOTS_PER_BLOCK;
            if (block_idx < qf->metadata->nblocks) {
                const uint8_t off = get_block_ptr(qf, block_idx)->offset;
                if (off >= static_cast<uint8_t>(reconstruct_threshold)) {
                    const uint64_t new_nslots = (qf->metadata->nslots <= (std::numeric_limits<uint64_t>::max() / 2))
                        ? (qf->metadata->nslots * 2)
                        : qf->metadata->nslots;
                    const int64_t rc_resize = qf_resize_malloc(qf, new_nslots);
                    if (rc_resize < 0) {
                        ++result.insert_failures;
                        continue;
                    }
                    reconstruct_done = true;
                }
            }
        }

        if (def_keepsake_rle) {
            const uint64_t ks_key = (proj.bucket << 32) | memento;
            const uint64_t max_keeps = (qf->metadata->memento_bits >= 63)
                ? std::numeric_limits<uint64_t>::max()
                : ((1ULL << qf->metadata->memento_bits) - 1ULL);
            auto it = keepsake_counters.find(ks_key);
            if (it == keepsake_counters.end()) {
                // first occurrence: materialize into filter
                keepsake_counters.emplace(ks_key, 1u);
                do_insert = true;
            } else if (static_cast<uint64_t>(it->second) < max_keeps) {
                // compress duplicate: increment counter, skip physical insert
                ++(it->second);
                do_insert = false;
            } else {
                // counter at max: materialize one more and reset counter
                it->second = 1u;
                do_insert = true;
            }
        }

        if (def_reconstruct) {
            const uint64_t block_idx = proj.bucket / QF_SLOTS_PER_BLOCK;
            if (block_idx < qf->metadata->nblocks) {
                const uint8_t off = get_block_ptr(qf, block_idx)->offset;
                if (off >= static_cast<uint8_t>(reconstruct_threshold)) {
                    std::cerr << "[info] reconstruct triggered at op=" << result.actual_inserts + 1
                              << " block=" << block_idx << " offset=" << static_cast<uint64_t>(off) << "\n";
                    const uint64_t new_nslots = qf->metadata->nslots * 2;
                    const int64_t copied = qf_resize_malloc(qf, new_nslots);
                    if (copied < 0) {
                        std::cerr << "[warn] reconstruct failed (qf_resize_malloc)\n";
                    } else {
                        std::cerr << "[info] reconstruct completed: new nslots=" << qf->metadata->nslots << " copied=" << copied << "\n";
                    }
                }
            }
        }

        const auto seg1 = clock_type::now();
        if (do_insert) {
            rc = qf_insert_single(qf, key, memento, QF_NO_LOCK | QF_KEY_IS_HASH);
        } else {
            rc = 0; // treated as successful compressed insert
        }
        const auto seg2 = clock_type::now();
        insert_core_ns_sum += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(seg2 - seg1).count());
        if (rc < 0) {
            ++result.insert_failures;
            continue;
        }
        if (do_insert) {
            inserted_pairs.emplace_back(key, memento);
            ++result.actual_inserts;
        }

        if (dump_slots_at_op > 0 && result.actual_inserts == dump_slots_at_op) {
            dump_slots_for_blocks(qf, result.actual_inserts, dump_block_start, dump_block_count);
        }

        if (trace_blocks) {
            if (trace_max_ops == 0 || result.actual_inserts <= trace_max_ops) {
                const Projection p_cur = project_key_hash(qf, key);
                if (trace_delta_only) {
                    const bool force_full = (result.actual_inserts == 1);
                    print_blocks_snapshot_delta(qf, result.actual_inserts, key, memento,
                                                p_cur, trace_prev_state, force_full);
                } else {
                    print_blocks_snapshot(qf, result.actual_inserts, key, memento, p_cur);
                }
            }
        }
        const auto seg3 = clock_type::now();

        if (latency_profile) {
            insert_prepare.observe(i, static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(seg1 - seg0).count()));
            insert_core.observe(i, static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(seg2 - seg1).count()));
            insert_post.observe(i, static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(seg3 - seg2).count()));
        }
    }
    result.insert_ms = static_cast<double>(insert_core_ns_sum) / 1e6;

    OffsetScanStats final_scan = scan_offsets(qf);
    result.final_saturated_blocks = final_scan.saturated_blocks;
    result.final_max_offset = final_scan.max_offset;

    const uint64_t max_memento_query = (memento_bits >= 63)
        ? std::numeric_limits<uint64_t>::max()
        : ((1ULL << memento_bits) - 1ULL);
    const uint64_t max_key_query = bitmask(std::min<uint64_t>(qf->metadata->key_bits, 64));
    std::vector<uint64_t> inserted_keys_sorted;
    inserted_keys_sorted.reserve(inserted_pairs.size());
    std::unordered_set<uint64_t> inserted_point_set;
    inserted_point_set.reserve(inserted_pairs.size() * 2 + 1);
    for (const auto& km : inserted_pairs) {
        inserted_keys_sorted.push_back(km.first);
        inserted_point_set.insert(pack_point(km.first, km.second));
    }
    std::sort(inserted_keys_sorted.begin(), inserted_keys_sorted.end());
    inserted_keys_sorted.erase(std::unique(inserted_keys_sorted.begin(), inserted_keys_sorted.end()), inserted_keys_sorted.end());

    auto range_has_inserted_key = [&](uint64_t l_key, uint64_t r_key) {
        auto it = std::lower_bound(inserted_keys_sorted.begin(), inserted_keys_sorted.end(), l_key);
        return it != inserted_keys_sorted.end() && *it <= r_key;
    };

    auto positive_range_query = [&](uint64_t R,
                                    SegmentLatencyStats* prep_stats,
                                    SegmentLatencyStats* core_stats,
                                    uint64_t& checks,
                                    uint64_t& hits,
                                    double& elapsed_ms,
                                    double& qps) {
        if (query_count == 0) {
            checks = hits = 0;
            elapsed_ms = qps = 0.0;
            return;
        }
        std::uniform_int_distribution<uint64_t> shift_dist(0, R - 1);
        const uint64_t max_l = (max_key_query >= R - 1) ? (max_key_query - (R - 1)) : 0;
        uint64_t core_ns_sum = 0;
        for (uint64_t i = 0; i < query_count; ++i) {
            const auto s0 = clock_type::now();
            const uint64_t anchor_key = inserted_pairs.empty()
                ? random_key_dist(rng)
                : inserted_pairs[i % inserted_pairs.size()].first;
            uint64_t l_key = anchor_key;
            if (positive_range_protocol == "contains-hit") {
                const uint64_t shift = shift_dist(rng);
                l_key = (anchor_key >= shift) ? (anchor_key - shift) : 0;
            }
            if (l_key > max_l) {
                l_key = max_l;
            }
            const uint64_t r_key = (std::numeric_limits<uint64_t>::max() - l_key < R - 1)
                ? std::numeric_limits<uint64_t>::max()
                : (l_key + R - 1);
            const auto s1 = clock_type::now();
            const int qr = (def_adaptive_verify ? qf_range_query_adaptive(qf, l_key, 0, r_key, max_memento_query, QF_NO_LOCK | QF_KEY_IS_HASH, verify_threshold) : qf_range_query(qf, l_key, 0, r_key, max_memento_query, QF_NO_LOCK | QF_KEY_IS_HASH));
            const auto s2 = clock_type::now();
            core_ns_sum += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(s2 - s1).count());
            ++checks;
            if (qr > 0) {
                ++hits;
            }
            if (latency_profile && prep_stats && core_stats) {
                prep_stats->observe(i, static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(s1 - s0).count()));
                core_stats->observe(i, static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(s2 - s1).count()));
            }
        }
        elapsed_ms = static_cast<double>(core_ns_sum) / 1e6;
        qps = elapsed_ms > 0.0 ? (1000.0 * static_cast<double>(checks) / elapsed_ms) : 0.0;
    };

    auto negative_range_query = [&](uint64_t R,
                                    uint64_t& checks,
                                    uint64_t& fps,
                                    double& elapsed_ms,
                                    double& qps) {
        if (query_count == 0) {
            checks = fps = 0;
            elapsed_ms = qps = 0.0;
            return;
        }
        const uint64_t max_l = (max_key_query >= R - 1) ? (max_key_query - (R - 1)) : 0;
        std::uniform_int_distribution<uint64_t> ldist(0, max_l);
        uint64_t core_ns_sum = 0;
        for (uint64_t i = 0; i < query_count; ++i) {
            uint64_t l_key = 0;
            uint64_t r_key = R - 1;
            bool found = false;
            for (uint64_t tries = 0; tries < 2000; ++tries) {
                l_key = ldist(rng);
                r_key = l_key + R - 1;
                if (!range_has_inserted_key(l_key, r_key)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                // fallback deterministic scan
                uint64_t cursor = i % (max_l + 1);
                for (uint64_t scan = 0; scan <= max_l; ++scan) {
                    l_key = cursor;
                    r_key = l_key + R - 1;
                    if (!range_has_inserted_key(l_key, r_key)) {
                        found = true;
                        break;
                    }
                    cursor = (cursor == max_l) ? 0 : (cursor + 1);
                }
            }
            if (!found) {
                continue;
            }
            const auto s1 = clock_type::now();
            const int qr = (def_adaptive_verify ? qf_range_query_adaptive(qf, l_key, 0, r_key, max_memento_query, QF_NO_LOCK | QF_KEY_IS_HASH, verify_threshold) : qf_range_query(qf, l_key, 0, r_key, max_memento_query, QF_NO_LOCK | QF_KEY_IS_HASH));
            const auto s2 = clock_type::now();
            core_ns_sum += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(s2 - s1).count());
            ++checks;
            if (qr > 0) {
                ++fps;
            }
        }
        elapsed_ms = static_cast<double>(core_ns_sum) / 1e6;
        qps = elapsed_ms > 0.0 ? (1000.0 * static_cast<double>(checks) / elapsed_ms) : 0.0;
    };

    uint64_t point_pos_core_ns_sum = 0;
    for (uint64_t i = 0; i < query_count; ++i) {
        const auto s0 = clock_type::now();
        const auto& km = inserted_pairs.empty()
            ? std::pair<uint64_t, uint64_t>(random_key_dist(rng), random_memento_dist(rng))
            : inserted_pairs[i % inserted_pairs.size()];
        const auto s1 = clock_type::now();

        const int qr = (def_adaptive_verify ? qf_point_query_adaptive(qf, km.first, km.second, QF_NO_LOCK | QF_KEY_IS_HASH, verify_threshold) : qf_point_query(qf, km.first, km.second, QF_NO_LOCK | QF_KEY_IS_HASH));
        const auto s2 = clock_type::now();
        point_pos_core_ns_sum += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(s2 - s1).count());
        ++result.point_pos_checks;
        if (qr <= 0) {
            ++result.point_pos_fn;
        }
        if (latency_profile) {
            point_prepare.observe(i, static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(s1 - s0).count()));
            point_core.observe(i, static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(s2 - s1).count()));
        }
    }
    result.point_pos_ms = static_cast<double>(point_pos_core_ns_sum) / 1e6;
    result.point_pos_qps = result.point_pos_ms > 0.0
        ? (1000.0 * static_cast<double>(result.point_pos_checks) / result.point_pos_ms)
        : 0.0;

    if (run_negative_queries) {
        uint64_t point_neg_core_ns_sum = 0;
        for (uint64_t i = 0; i < query_count; ++i) {
            uint64_t key = 0;
            uint64_t memento = 0;
            bool found = false;
            for (uint64_t tries = 0; tries < 2000; ++tries) {
                key = random_key_dist(rng);
                memento = random_memento_dist(rng);
                if (inserted_point_set.find(pack_point(key, memento)) == inserted_point_set.end()) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                continue;
            }
            const auto s1 = clock_type::now();
            const int qr = (def_adaptive_verify ? qf_point_query_adaptive(qf, key, memento, QF_NO_LOCK | QF_KEY_IS_HASH, verify_threshold) : qf_point_query(qf, key, memento, QF_NO_LOCK | QF_KEY_IS_HASH));
            const auto s2 = clock_type::now();
            point_neg_core_ns_sum += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(s2 - s1).count());
            ++result.point_neg_checks;
            if (qr > 0) {
                ++result.point_neg_fp;
            }
        }
        result.point_neg_ms = static_cast<double>(point_neg_core_ns_sum) / 1e6;
        result.point_neg_qps = result.point_neg_ms > 0.0
            ? (1000.0 * static_cast<double>(result.point_neg_checks) / result.point_neg_ms)
            : 0.0;
    }

    const uint64_t short_R = 1ULL << 5;
    positive_range_query(short_R,
                         &short_prepare,
                         &short_core,
                         result.short_range_pos_checks,
                         result.short_range_pos_hits,
                         result.short_range_pos_ms,
                         result.short_range_pos_qps);
    if (run_negative_queries) {
        negative_range_query(short_R,
                             result.short_range_neg_checks,
                             result.short_range_neg_fp,
                             result.short_range_neg_ms,
                             result.short_range_neg_qps);
    }

    const uint64_t long_R = 1ULL << 10;
    positive_range_query(long_R,
                         &long_prepare,
                         &long_core,
                         result.long_range_pos_checks,
                         result.long_range_pos_hits,
                         result.long_range_pos_ms,
                         result.long_range_pos_qps);
    if (run_negative_queries) {
        negative_range_query(long_R,
                             result.long_range_neg_checks,
                             result.long_range_neg_fp,
                             result.long_range_neg_ms,
                             result.long_range_neg_qps);
    }

    const double sat_ratio = qf->metadata->nblocks == 0
        ? 0.0
        : static_cast<double>(result.final_saturated_blocks) / static_cast<double>(qf->metadata->nblocks);

    std::cout << "build_nslots=" << qf->metadata->nslots << "\n"
              << "nblocks=" << qf->metadata->nblocks << "\n"
              << "key_bits=" << qf->metadata->key_bits << "\n"
              << "fingerprint_bits=" << qf->metadata->fingerprint_bits << "\n"
              << "memento_bits=" << qf->metadata->memento_bits << "\n"
              << "requested_hot_slots=" << result.requested_hot_slots << "\n"
              << "effective_hot_slots=" << result.effective_hot_slots << "\n"
              << "requested_inserts=" << result.requested_inserts << "\n"
              << "attack_insert_ratio=" << result.attack_insert_ratio << "\n"
              << "attack_inserts_actual=" << result.attack_inserts_actual << "\n"
              << "random_inserts_actual=" << result.random_inserts_actual << "\n"
              << "actual_inserts=" << result.actual_inserts << "\n"
              << "insert_failures=" << result.insert_failures << "\n"
              << "final_saturated_blocks=" << result.final_saturated_blocks << "\n"
              << "final_saturated_ratio=" << sat_ratio << "\n"
              << "final_max_offset=" << static_cast<uint64_t>(result.final_max_offset) << "\n"
              << "point_pos_checks=" << result.point_pos_checks << "\n"
              << "point_pos_fn=" << result.point_pos_fn << "\n"
              << "point_neg_checks=" << result.point_neg_checks << "\n"
              << "point_neg_fp=" << result.point_neg_fp << "\n"
              << "short_range_pos_checks=" << result.short_range_pos_checks << "\n"
              << "short_range_pos_hits=" << result.short_range_pos_hits << "\n"
              << "short_range_neg_checks=" << result.short_range_neg_checks << "\n"
              << "short_range_neg_fp=" << result.short_range_neg_fp << "\n"
              << "long_range_pos_checks=" << result.long_range_pos_checks << "\n"
              << "long_range_pos_hits=" << result.long_range_pos_hits << "\n"
              << "long_range_neg_checks=" << result.long_range_neg_checks << "\n"
              << "long_range_neg_fp=" << result.long_range_neg_fp << "\n"
              << "insert_ms=" << result.insert_ms << "\n"
              << "point_pos_ms=" << result.point_pos_ms << "\n"
              << "point_neg_ms=" << result.point_neg_ms << "\n"
              << "short_range_pos_ms=" << result.short_range_pos_ms << "\n"
              << "short_range_neg_ms=" << result.short_range_neg_ms << "\n"
              << "long_range_pos_ms=" << result.long_range_pos_ms << "\n"
              << "long_range_neg_ms=" << result.long_range_neg_ms << "\n"
              << "point_pos_qps=" << result.point_pos_qps << "\n"
              << "point_neg_qps=" << result.point_neg_qps << "\n"
              << "short_range_pos_qps=" << result.short_range_pos_qps << "\n"
              << "short_range_neg_qps=" << result.short_range_neg_qps << "\n"
              << "long_range_pos_qps=" << result.long_range_pos_qps << "\n"
              << "long_range_neg_qps=" << result.long_range_neg_qps << "\n"
              << "key_a=" << result.key_a << " bucket_a=" << result.proj_bucket_a << " fp_a=" << result.proj_fp_a << "\n"
              << "key_b=" << result.key_b << " bucket_b=" << result.proj_bucket_b << " fp_b=" << result.proj_fp_b << "\n";

    if (!csv_file.empty()) {
        append_csv(csv_file, mode, qf, result, positive_range_protocol);
        std::cout << "[+] appended csv: " << csv_file << std::endl;
    }

    auto append_latency_summary = [&](const std::string& file,
                                      const std::string& segment,
                                      const SegmentLatencyStats& st) {
        if (file.empty()) {
            return;
        }
        const bool needs_header = !std::ifstream(file).good() || std::ifstream(file).peek() == std::ifstream::traits_type::eof();
        std::ofstream out(file, std::ios::app);
        if (!out.good()) {
            return;
        }
        if (needs_header) {
            out << "tag,mode,xhot_schedule,requested_hot_slots,attack_insert_ratio,segment,count,mean_ns,min_ns,max_ns\n";
        }
        out << latency_tag << ","
            << mode << ","
            << xhot_schedule << ","
            << requested_hot_slots << ","
            << attack_insert_ratio << ","
            << segment << ","
            << st.count << ","
            << st.mean_ns() << ","
            << (st.count ? st.min_ns : 0) << ","
            << st.max_ns << "\n";
    };

    auto append_latency_samples = [&](const std::string& file,
                                      const std::string& segment,
                                      const SegmentLatencyStats& st) {
        if (file.empty()) {
            return;
        }
        const bool needs_header = !std::ifstream(file).good() || std::ifstream(file).peek() == std::ifstream::traits_type::eof();
        std::ofstream out(file, std::ios::app);
        if (!out.good()) {
            return;
        }
        if (needs_header) {
            out << "tag,mode,xhot_schedule,requested_hot_slots,attack_insert_ratio,segment,op_index,latency_ns\n";
        }
        for (const auto& s : st.samples) {
            out << latency_tag << ","
                << mode << ","
                << xhot_schedule << ","
                << requested_hot_slots << ","
                << attack_insert_ratio << ","
                << segment << ","
                << s.op_index << ","
                << s.latency_ns << "\n";
        }
    };

    if (latency_profile) {
        append_latency_summary(latency_summary_csv, "insert_prepare", insert_prepare);
        append_latency_summary(latency_summary_csv, "insert_core", insert_core);
        append_latency_summary(latency_summary_csv, "insert_post", insert_post);
        append_latency_summary(latency_summary_csv, "point_prepare", point_prepare);
        append_latency_summary(latency_summary_csv, "point_core", point_core);
        append_latency_summary(latency_summary_csv, "short_prepare", short_prepare);
        append_latency_summary(latency_summary_csv, "short_core", short_core);
        append_latency_summary(latency_summary_csv, "long_prepare", long_prepare);
        append_latency_summary(latency_summary_csv, "long_core", long_core);

        append_latency_samples(latency_samples_csv, "insert_prepare", insert_prepare);
        append_latency_samples(latency_samples_csv, "insert_core", insert_core);
        append_latency_samples(latency_samples_csv, "insert_post", insert_post);
        append_latency_samples(latency_samples_csv, "point_prepare", point_prepare);
        append_latency_samples(latency_samples_csv, "point_core", point_core);
        append_latency_samples(latency_samples_csv, "short_prepare", short_prepare);
        append_latency_samples(latency_samples_csv, "short_core", short_core);
        append_latency_samples(latency_samples_csv, "long_prepare", long_prepare);
        append_latency_samples(latency_samples_csv, "long_core", long_core);
    }

    qf_free(qf);
    free(qf);
    return 0;
}
