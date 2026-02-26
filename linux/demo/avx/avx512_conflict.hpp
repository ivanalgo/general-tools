template<typename T>
struct AVX512_CONFLICT {
    static constexpr const char* CATEGORY = "avx512";
    static constexpr const char* CLASS_TYPE = "conflict";

    static constexpr size_t LANES = 512 / (8 * sizeof(T));
    static constexpr size_t INPUT_SIZE = LANES;

    static constexpr int INPUT_ARGS = 1;
    using ARG1_TYPE = T;
    static constexpr size_t ARG1_SIZE = INPUT_SIZE;
    
    using OUTPUT_TYPE = T;
    static constexpr size_t OUTPUT_SIZE = INPUT_SIZE;

    // VPCONFLICT: Detects conflicts (duplicate elements) in a vector
    // For each element, returns a mask of previous elements that are equal to it.
    
    /* SISD Conflict */
    static void sisd_conflict(const T* a, T* out) {
        for(size_t i=0; i<INPUT_SIZE; ++i) {
            T mask = 0;
            for(size_t j=0; j<i; ++j) {
                if(a[i] == a[j]) {
                    mask |= (T(1) << j);
                }
            }
            out[i] = mask;
        }
    }

    /* AVX512 Conflict */
    static void avx512_conflict(const T* a, T* out) {
        if constexpr (std::is_same_v<T, int32_t>) {
            __m512i va = _mm512_loadu_si512(a);
            __m512i res = _mm512_conflict_epi32(va);
            _mm512_storeu_si512(out, res);
        } else if constexpr (std::is_same_v<T, int64_t>) {
            __m512i va = _mm512_loadu_si512(a);
            __m512i res = _mm512_conflict_epi64(va);
            _mm512_storeu_si512(out, res);
        } else {
             // Not supported for other types directly
        }
    }
    
    // LZCNT: Count Leading Zeros
    /* SISD LZCNT */
    static void sisd_lzcnt(const T* a, T* out) {
        for(size_t i=0; i<INPUT_SIZE; ++i) {
            if constexpr (std::is_same_v<T, int32_t>) {
                uint32_t val = (uint32_t)a[i];
                if (val == 0) out[i] = 32;
                else out[i] = __builtin_clz(val);
            } else if constexpr (std::is_same_v<T, int64_t>) {
                uint64_t val = (uint64_t)a[i];
                if (val == 0) out[i] = 64;
                else out[i] = __builtin_clzll(val);
            }
        }
    }

    /* AVX512 LZCNT */
    static void avx512_lzcnt(const T* a, T* out) {
        if constexpr (std::is_same_v<T, int32_t>) {
            __m512i va = _mm512_loadu_si512(a);
            __m512i res = _mm512_lzcnt_epi32(va);
            _mm512_storeu_si512(out, res);
        } else if constexpr (std::is_same_v<T, int64_t>) {
            __m512i va = _mm512_loadu_si512(a);
            __m512i res = _mm512_lzcnt_epi64(va);
            _mm512_storeu_si512(out, res);
        }
    }

    static constexpr auto make_ops() {
        return std::array{
            OpEntry1<ARG1_TYPE, OUTPUT_TYPE>{ "conflict", avx512_conflict, sisd_conflict },
            OpEntry1<ARG1_TYPE, OUTPUT_TYPE>{ "lzcnt", avx512_lzcnt, sisd_lzcnt }
        };
    }
};

#ifdef REGISTER_TEST
using AVX512_CONFLICT_INT = AVX512_CONFLICT<int32_t>;
REGISTER_TEST(AVX512_CONFLICT_INT)
using AVX512_CONFLICT_I64 = AVX512_CONFLICT<int64_t>;
REGISTER_TEST(AVX512_CONFLICT_I64)
#endif
