template <typename T>
struct AVX512_BLEND {
    using ARG1_TYPE = T;
    using ARG2_TYPE = T;
	using ARG3_TYPE = std::conditional_t<
        std::is_same_v<T, double>,
        uint8_t,        // __mmask8
        uint16_t        // __mmask16 (float / int32)
    >;
    using OUTPUT_TYPE = T;
    static constexpr size_t INPUT_SIZE = 512 / (8 * sizeof(T));
    static constexpr size_t OUTPUT_SIZE = INPUT_SIZE;
    static constexpr const char* CLASS_NAME = "AVX512_BLEND";
    static constexpr int INPUT_ARGS = 3; // a, b, mask

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
        	OpEntry3<ARG1_TYPE, ARG2_TYPE, ARG3_TYPE, OUTPUT_TYPE>{ "avx512 blend", avx512_blend, sisd_blend }
    	};
	}

};
