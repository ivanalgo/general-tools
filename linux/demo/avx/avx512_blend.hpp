template<typename T>
struct AVX512_BLEND {
    static constexpr const char* CATEGORY = "avx512";
    static constexpr const char* CLASS_TYPE = "blend";

    static constexpr size_t INPUT_SIZE = 512 / (8 * sizeof(T));

    static constexpr int INPUT_ARGS = 3;
    using ARG1_TYPE = T;
    static constexpr size_t ARG1_SIZE = INPUT_SIZE;
    using ARG2_TYPE = T;
    static constexpr size_t ARG2_SIZE = INPUT_SIZE;
    // mask 类型
    // 64 lanes  -> uint64_t  (对应 __mmask64)
    // 32 lanes  -> uint32_t  (对应 __mmask32)
    // 16 lanes  -> uint16_t  (对应 __mmask16)
    // 8 lanes   -> uint8_t   (对应 __mmask8)
    using ARG3_TYPE =
        std::conditional_t<INPUT_SIZE == 64, uint64_t,   // __mmask64
        std::conditional_t<INPUT_SIZE == 32, uint32_t,   // __mmask32
        std::conditional_t<INPUT_SIZE == 16, uint16_t,   // __mmask16
        std::conditional_t<INPUT_SIZE == 8,  uint8_t,
        void>>>>;
    static constexpr size_t ARG3_SIZE = INPUT_SIZE;

    using OUTPUT_TYPE = T;
    static constexpr size_t OUTPUT_SIZE = INPUT_SIZE;

    /* SISD 实现 */
    static void sisd_blend(const T* a, const T* b, const ARG3_TYPE *mask, T* out) {
        for (size_t i = 0; i < INPUT_SIZE; ++i) {
            out[i] = (*mask & (1 << i)) ? b[i] : a[i];
        }
    }

    /* AVX512 实现 */
    static void avx512_blend(const T* a, const T* b, const ARG3_TYPE *mask, T* out) {
        if constexpr (std::is_same_v<T, float>) {
            __m512 va = _mm512_loadu_ps(a);
            __m512 vb = _mm512_loadu_ps(b);
            __m512 vc = _mm512_mask_mov_ps(va, *mask, vb);
            _mm512_storeu_ps(out, vc);
        } else if constexpr (std::is_same_v<T, double>) {
            __m512d va = _mm512_loadu_pd(a);
            __m512d vb = _mm512_loadu_pd(b);
            __m512d vc = _mm512_mask_mov_pd(va, *mask, vb);
            _mm512_storeu_pd(out, vc);
        } else if constexpr (std::is_same_v<T, int>) {
            __m512i va = _mm512_loadu_si512(a);
            __m512i vb = _mm512_loadu_si512(b);
            __m512i vc = _mm512_mask_mov_epi32(va, *mask, vb);
            _mm512_storeu_si512(out, vc);
        }
    }

	static constexpr auto make_ops() {
    	return std::array{
        	OpEntry3<ARG1_TYPE, ARG2_TYPE, ARG3_TYPE, OUTPUT_TYPE>{ "blend", avx512_blend, sisd_blend }
    	};
	}

};

// Register tests
#ifdef REGISTER_TEST
using AVX512_BLEND_INT = AVX512_BLEND<int>;
REGISTER_TEST(AVX512_BLEND_INT)
using AVX512_BLEND_FLOAT = AVX512_BLEND<float>;
REGISTER_TEST(AVX512_BLEND_FLOAT)
using AVX512_BLEND_DOUBLE = AVX512_BLEND<double>;
REGISTER_TEST(AVX512_BLEND_DOUBLE)
#endif
