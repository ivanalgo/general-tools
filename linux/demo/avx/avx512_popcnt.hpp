template<typename T>
struct AVX512_POPCNT {
    static constexpr const char* CATEGORY = "avx512";
    static constexpr const char* CLASS_TYPE = "popcnt";

    static constexpr size_t LANES = 512 / (8 * sizeof(T));
    static constexpr size_t INPUT_SIZE = LANES;

    static constexpr int INPUT_ARGS = 1;
    using ARG1_TYPE = T;
    static constexpr size_t ARG1_SIZE = INPUT_SIZE;

    using OUTPUT_TYPE = T;
    static constexpr size_t OUTPUT_SIZE = INPUT_SIZE;

    /* SISD Popcnt */
    static void sisd_popcnt(const T* a, T* out) {
        for(size_t i=0; i<INPUT_SIZE; ++i) {
            if constexpr (std::is_same_v<T, int32_t>) {
                out[i] = __builtin_popcount((uint32_t)a[i]);
            } else if constexpr (std::is_same_v<T, int64_t>) {
                out[i] = __builtin_popcountll((uint64_t)a[i]);
            }
        }
    }

    /* AVX512 Popcnt */
    #pragma GCC target("avx512vpopcntdq")
    static void avx512_popcnt(const T* a, T* out) {
        if constexpr (std::is_same_v<T, int32_t>) {
            __m512i va = _mm512_loadu_si512(a);
            __m512i res = _mm512_popcnt_epi32(va);
            _mm512_storeu_si512(out, res);
        } else if constexpr (std::is_same_v<T, int64_t>) {
            __m512i va = _mm512_loadu_si512(a);
            __m512i res = _mm512_popcnt_epi64(va);
            _mm512_storeu_si512(out, res);
        }
    }
    #pragma GCC reset_options

    static constexpr auto make_ops() {
        return std::array{
            OpEntry1<ARG1_TYPE, OUTPUT_TYPE>{ "popcnt", avx512_popcnt, sisd_popcnt }
        };
    }
};

// Register tests
#ifdef REGISTER_TEST
// Popcnt requires AVX512_VPOPCNTDQ extension
using AVX512_POPCNT_INT = AVX512_POPCNT<int32_t>;
REGISTER_TEST(AVX512_POPCNT_INT)
using AVX512_POPCNT_I64 = AVX512_POPCNT<int64_t>;
REGISTER_TEST(AVX512_POPCNT_I64)
#endif
