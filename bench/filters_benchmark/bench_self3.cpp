/*
 * Enhanced Memento Filter with False Positive Cache
 * Allocates alpha=0.1 of space for frequent false positives
 */


#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <boost/sort/sort.hpp>

#include "../bench_template.hpp"
#include "memento.h"
#include "memento_int.h"

#include <unordered_set>
#include <queue>

// 混合 LRU 和频率的优先级 Cache
struct FPCacheLRUFreq {
    struct CacheEntry {
        uint64_t key;
        uint32_t frequency;
        uint32_t last_access_time;
        double priority_score;

        CacheEntry(uint64_t key, uint32_t f, uint32_t t) : key(key), frequency(f), last_access_time(t) {
            // 优先级得分：频率权重 0.7，时间权重 0.3
            priority_score = 0.3 * frequency + 0.7 * last_access_time;
        }

        // void update(uint32_t f, uint32_t t) {
        //     frequency = f;
        //     last_access_time = t;
        //     priority_score = 1.0 * frequency + 0 * last_access_time;
        // }

    
        
        bool operator<(const CacheEntry& other) const {
            return priority_score < other.priority_score;  // 得分低的先被淘汰
        }
    };
    
    std::multiset<CacheEntry> pq;
    // std::map<uint64_t, std::pair<uint32_t, uint32_t>> key_info; // key -> (frequency, last_time)
    std::map<uint64_t, std::multiset<CacheEntry>::iterator> pq_iterators; // key -> iterator in pq
    uint64_t max_size;
    uint32_t current_time;
    uint64_t cur_range_fp_size;

    FPCacheLRUFreq(uint64_t size) : max_size(size), current_time(0) {}
    // 优化后的范围查询 - O(log n + k)，k 是结果数量
    uint64_t* get_keys_in_range(uint64_t left, uint64_t right) {
        auto it_start = pq_iterators.lower_bound(left);   // 找到第一个 >= left 的元素
        auto it_end = pq_iterators.upper_bound(right);    // 找到第一个 > right 的元素
        uint64_t* keys = new uint64_t[std::distance(it_start, it_end)];
        for (auto it = it_start; it != it_end; ++it) {
            keys[std::distance(it_start, it)] = it->first;
        }
        cur_range_fp_size = std::distance(it_start, it_end);
        return keys;
    }
    // 检查 FP Cache 是否包含指定的 FP
    // 后续根据potential_fp_keys的有序性考虑优化
    bool contain_all_keys(uint64_t *fp_keys, uint64_t *fp_keys_size, 
         uint64_t *potential_fp_keys, uint64_t *potential_fp_keys_size) {
        bool all_found = true;
        for (uint64_t i = 0; i < *fp_keys_size; ++i) {
            if (!pq_iterators.count(fp_keys[i])) {
                all_found = false;
                potential_fp_keys[*potential_fp_keys_size] = fp_keys[i];
                (*potential_fp_keys_size)++;
            }
        }
        return all_found;
    }


    bool contains(uint64_t key) {
        current_time++;
        auto it = pq_iterators.find(key);
        if (it != pq_iterators.end()) {
            // 获取当前频率并增加1
            uint32_t new_frequency = it->second->frequency + 1;
            // 先删除旧元素
            pq.erase(it->second);
            // 插入更新后的元素
            auto new_it = pq.insert(CacheEntry(key, new_frequency, current_time));
            pq_iterators[key] = new_it;
            
            return true;
        }
        return false;
    }
    
    int insert(uint64_t key) {
        // 检查是否需要淘汰
        if (pq_iterators.size() >= max_size) {
            evict_lowest_priority();
            auto new_it = pq.insert(CacheEntry(key, 1, current_time));
            pq_iterators[key] = new_it;
            return 0; // 表示发生了淘汰
        }
        
        // 插入新元素
        auto new_it = pq.insert(CacheEntry(key, 1, current_time));
        pq_iterators[key] = new_it;
        return 1; // 表示成功插入
    }
    
private:
    void evict_lowest_priority() {
        if (!pq.empty()) {
            auto entry = *pq.begin();  // 最小优先级的条目
            auto key = entry.key;
            
            // 从数据结构中移除
            pq.erase(pq.begin());

            pq_iterators.erase(key);
        }
    }
    
public:
    size_t size() const { return pq_iterators.size();} 
    void clear() {
        pq.clear();
        // key_info.clear();
        pq_iterators.clear();
        current_time = 0;
    }
    ~FPCacheLRUFreq() {
        clear();
    }
};

// Enhanced QF structure with FP cache
struct QF_Enhanced {
    QF *qf;
    FPCacheLRUFreq *fp_cache;
    double alpha;  // fraction of space for FP cache
    
    QF_Enhanced(QF *filter, double a) : qf(filter), alpha(a) {
        // Calculate cache size based on alpha
        uint64_t total_space = qf_get_total_size_in_bytes(filter);
        uint64_t cache_entries = total_space * alpha / 2; // each entey has 2 bytes
        fp_cache = new FPCacheLRUFreq(cache_entries);
        std::cerr << "FP Cache size: " << cache_entries << " entries" << std::endl;
    }
    
    ~QF_Enhanced() {
        delete fp_cache;
    }
};



inline uint64_t MurmurHash64A(const void * key, int len, unsigned int seed)
{
	const uint64_t m = 0xc6a4a7935bd1e995;
	const int r = 47;

	uint64_t h = seed ^ (len * m);

	const uint64_t * data = (const uint64_t *)key;
	const uint64_t * end = data + (len/8);

	while(data != end) {
		uint64_t k = *data++;

		k *= m;
		k ^= k >> r;
		k *= m;

		h ^= k;
		h *= m;
	}

	const unsigned char * data2 = (const unsigned char*)data;

	switch(len & 7) {
		case 7: h ^= (uint64_t)data2[6] << 48; do {} while (0);  /* fallthrough */
		case 6: h ^= (uint64_t)data2[5] << 40; do {} while (0);  /* fallthrough */
		case 5: h ^= (uint64_t)data2[4] << 32; do {} while (0);  /* fallthrough */
		case 4: h ^= (uint64_t)data2[3] << 24; do {} while (0);  /* fallthrough */
		case 3: h ^= (uint64_t)data2[2] << 16; do {} while (0);  /* fallthrough */
		case 2: h ^= (uint64_t)data2[1] << 8; do {} while (0); /* fallthrough */
		case 1: h ^= (uint64_t)data2[0];
						h *= m;
	};

	h ^= h >> r;
	h *= m;
	h ^= h >> r;

	return h;
}

__attribute__((always_inline))
static inline uint32_t fast_reduce(uint32_t hash, uint32_t n) {
    // http://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
    return (uint32_t) (((uint64_t) hash * n) >> 32);
}

inline void check_iteration_validity(QF *qf, bool mode)
{
    QFi iter;
    qf_iterator_from_position(qf, &iter, 0);
    uint64_t hash_result, memento_result[256];
    uint64_t last_run = iter.run, last_fingerprint = 0, current_fingerprint;
    uint64_t cnt = 0;
    do {
        if (mode) {
            std::cerr << "run=" << iter.run << " current=" << iter.current << " vs. nslots=" << qf->metadata->nslots << std::endl;
            //qf_dump_block(qf, 2654 / QF_SLOTS_PER_BLOCK);
        }
        int result_length = qfi_get_hash(&iter, &hash_result, memento_result);
        current_fingerprint = hash_result >> (qf->metadata->key_bits - qf->metadata->fingerprint_bits);
        assert(current_fingerprint > 0);
        if (iter.run != last_run)
            last_run = iter.run;
        else {
            if (last_fingerprint > current_fingerprint) {
                std::cerr << "HMMM iter.run=" << iter.run << " iter.current=" << iter.current << std::endl;
            }
            assert(last_fingerprint <= current_fingerprint);
        }
        last_fingerprint = current_fingerprint;
        for (int i = 1; i < result_length; i++) {
            if (memento_result[i] < memento_result[i - 1]) {
                std::cerr << "run=" << iter.run << " current=" << iter.current << std::endl;
                for (int j = 0; j < result_length; j++)
                    std::cerr << memento_result[j] << ' ';
                std::cerr << std::endl;
            }
            assert(memento_result[i] >= memento_result[i - 1]);
        }

    } while (qfi_next(&iter) >= 0);
}

template <typename t_itr, typename... Args>
inline QF_Enhanced *init_self3(const t_itr begin, const t_itr end, const double bpk, Args... args)
{
    auto&& t = std::forward_as_tuple(args...);
    auto queries_temp = std::get<0>(t);
    auto query_lengths = std::vector<uint64_t>(queries_temp.size());
    std::transform(queries_temp.begin(), queries_temp.end(), query_lengths.begin(), [](auto x) {
        auto [left, right, result] = x;
        return right - left + 1;
    });
    const uint64_t n_items = std::distance(begin, end);
    //const uint64_t seed = std::chrono::steady_clock::now().time_since_epoch().count();
    const uint64_t seed = 1380;
    const uint64_t max_range_size = *std::max_element(query_lengths.begin(), query_lengths.end());
    const double load_factor = 0.95;

    const double alpha = 0.01;  // 10% for FP cache
    const double effective_bpk = bpk * (1.0 - alpha);

    const uint64_t n_slots = n_items / load_factor + std::sqrt(n_items);
    int predef_memento_size = std::get<1>(t);
    uint32_t memento_bits = 1;
    if (predef_memento_size == -1) {
        while ((1ULL << memento_bits) < max_range_size)
            memento_bits++;
        memento_bits = memento_bits < 2 ? 2 : memento_bits;
    }
    else 
        memento_bits = predef_memento_size;
    const uint32_t fingerprint_size = round(effective_bpk * load_factor - memento_bits - 2.125);
    uint32_t key_size = 0;
    while ((1ULL << key_size) <= n_slots)
        key_size++;
    key_size += fingerprint_size;
    std::cerr << "fingerprint_size=" << fingerprint_size << " memento_bits=" << memento_bits << std::endl;

    QF *qf = (QF *) malloc(sizeof(QF));
    qf_malloc(qf, n_slots, key_size, memento_bits, QF_HASH_DEFAULT, seed);
    qf_set_auto_resize(qf, true);

    start_timer(build_time);

    auto key_hashes = std::vector<uint64_t>(n_items);
    const uint64_t address_size = key_size - fingerprint_size;
    const uint64_t address_mask = (1ULL << address_size) - 1;
    const uint64_t memento_mask = (1ULL << memento_bits) - 1;
    const uint64_t hash_mask = (1ULL << key_size) - 1;
    std::transform(begin, end, key_hashes.begin(), [&](auto x) {
            auto y = x >> memento_bits;
            uint64_t hash = MurmurHash64A(((void *)&y), sizeof(y), seed) & hash_mask;
            const uint64_t address = fast_reduce((hash & address_mask) << (32 - address_size),
                                                    n_slots);
            hash = (hash >> address_size) | (address << fingerprint_size);
            return (hash << memento_bits) | (x & memento_mask);
            });
    /*
     * The following code uses the Boost library to sort the elements in a single thread, via spreadsort function.
     * This function is faster than std::sort and exploits the fact that the size of the maximum hash is bounded
     * via hybrid radix sort.
     */
    boost::sort::spreadsort::spreadsort(key_hashes.begin(), key_hashes.end());

    qf_bulk_load(qf, &key_hashes[0], key_hashes.size(), QF_NO_LOCK | QF_KEY_IS_HASH);

    stop_timer(build_time);

    check_iteration_validity(qf, false);

        // Create enhanced structure with FP cache
    QF_Enhanced *enhanced = new QF_Enhanced(qf, alpha);

    return enhanced;
}

template <typename value_type>
inline bool query_self3(QF_Enhanced *f, const value_type left, const value_type right)
{
    value_type l_key = left >> f->qf->metadata->memento_bits;
    value_type l_memento = left & ((1ULL << f->qf->metadata->memento_bits) - 1);
    if (left == right) {
        // Check FP cache first
        if (f->fp_cache->contains(left)) {
            return false;
        }
        return qf_point_query(f->qf, l_key, l_memento, QF_NO_LOCK);
    }
    value_type r_key = right >> f->qf->metadata->memento_bits;
    value_type r_memento = right & ((1ULL << f->qf->metadata->memento_bits) - 1);
    return qf_range_query(f->qf, l_key, l_memento, r_key, r_memento, QF_NO_LOCK);
}

inline size_t size_self3(QF_Enhanced *f)
{
    return qf_get_total_size_in_bytes(f->qf) + f->fp_cache->max_size * 2; // bytes, main size + FP cache size
}

template <typename InitFun, typename RangeFun, typename SizeFun, typename key_type, typename... Args>
void experiment_with_fp_learning(InitFun init_f, RangeFun range_f, SizeFun size_f, const double param, InputKeys<key_type> &keys, Workload<key_type> &queries, Args... args)
{
    auto f = init_f(keys.begin(), keys.end(), param, args...);

    std::cout << "[+] data structure constructed in " << test_out["build_time"] << "ms, starting queries" << std::endl;
    auto fp = 0, fn = 0;
    auto evictCnt = 0;

    start_timer(query_time);
    for (auto q : queries)
    {
        const auto [left, right, original_result] = q;
        uint64_t l_key = left >> f->qf->metadata->memento_bits;
        uint64_t l_memento = left & ((1ULL << f->qf->metadata->memento_bits) - 1);
        uint64_t r_key = right >> f->qf->metadata->memento_bits;
        uint64_t r_memento = right & ((1ULL << f->qf->metadata->memento_bits) - 1);
        // bool query_result = range_f(f, left, right);
        bool query_result;
        if (left == right) {
            if (qf_point_query(f->qf, l_key, l_memento, QF_NO_LOCK) && ! f->fp_cache->contains(left)) 
                query_result = true;
             else 
                query_result = false;
            if (query_result && !original_result) 
            {
                fp++;
                f->fp_cache->insert(left);
            }
            else if (!query_result && original_result)
            {
                std::cerr << "[!] alert, found false negative!" << std::endl;
                fn++;
            }
        } else {
            // uint64_t fp_key = 0;
            // uint64_t* fp_keys;
            uint64_t* fp_keys = new uint64_t[right - left + 1];
            uint64_t fp_keys_size = 0;
            // query_result = qf_range_query_fp_learning2(f->qf, l_key, l_memento, r_key, r_memento, QF_NO_LOCK, &fp_key, fps, fps_size);
            query_result = qf_range_query_fp_learning3(f->qf, l_key, l_memento, r_key, r_memento, QF_NO_LOCK, fp_keys, &fp_keys_size);
            if (query_result) {
                uint64_t *potential_fp_keys = new uint64_t[right - left + 1];
                uint64_t potential_fp_keys_size = 0;
                if (f->fp_cache->contain_all_keys(fp_keys, &fp_keys_size, potential_fp_keys, &potential_fp_keys_size)) {
                    query_result = false; // 如果 FP Cache 包含所有 FP，则认为查询结果为 false]
                } 
                if (query_result && !original_result) {
                    fp++;
                    // assert(potential_fp_keys[i] >= left && potential_fp_keys[i] <= right);
                    for (uint64_t i = 0; i < potential_fp_keys_size; ++i) {
                        int result = f->fp_cache->insert(potential_fp_keys[i]);
                        if (result == 0) {
                            // 发生了淘汰
                            evictCnt++;
                        }
                    }
                }
                delete[] potential_fp_keys;
            }
            else if (!query_result && original_result)
            {
                std::cerr << "[!] alert, found false negative!" << std::endl;
                fn++;
            }
            delete[] fp_keys;

        }
    }
    stop_timer(query_time);

    auto size = size_f(f);
    test_out.add_measure("size", size);
    test_out.add_measure("bpk", TO_BPK(size, keys.size()));
    test_out.add_measure("fpr", ((double)fp / queries.size()));
    test_out.add_measure("false_neg", fn);
    test_out.add_measure("n_keys", keys.size());
    test_out.add_measure("n_queries", queries.size());
    test_out.add_measure("false_positives", fp);
    test_out.add_measure("fpCacheSize", f->fp_cache->size());
    test_out.add_measure("fpCacheMaxSize", f->fp_cache->max_size);
    test_out.add_measure("evictCount", evictCnt);
    std::cout << "[+] test executed successfully, printing stats and closing." << std::endl;
}

int main(int argc, char const *argv[])
{
    auto parser = init_parser("bench-memento");

    try
    {
        parser.parse_args(argc, argv);
    }
    catch (const std::runtime_error &err)
    {
        std::cerr << err.what() << std::endl;
        std::cerr << parser;
        std::exit(1);
    }

    auto [ keys, queries, arg, memento_size ] = read_parser_arguments_memento(parser);

    experiment_with_fp_learning(pass_fun(init_self3), pass_ref(query_self3), 
                pass_ref(size_self3), arg, keys, queries, queries, memento_size);

    print_test();

    return 0;
}




