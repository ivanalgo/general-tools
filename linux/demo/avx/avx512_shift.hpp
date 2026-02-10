template <typename T>
struct AVX512_SHIFT {
    using ARG1_TYPE = T;
    using ARG2_TYPE = T;
    using OUTPUT_TYPE = T;
    static constexpr int INPUT_SIZE = 512 / (8 * sizeof(T));
	static constexpr int OUTPUT_SIZE = INPUT_SIZE;
    static constexpr const char* CLASS_NAME = "AVX512_SHIFT";
    static constexpr int INPUT_ARGS = 2;

    /* SISD 实现 */
	static void sisd_sll(const T* a, const T* b, T* out)
	{
		for (int i = 0; i < INPUT_SIZE; ++i) {
			uint32_t k = static_cast<uint32_t>(b[i]);
			if (k >= 32)
				out[i] = 0;
			else
				out[i] = a[i] << k;
		}
	}


	static void sisd_srl(const T* a, const T* b, T* out)
	{
		for (int i = 0; i < INPUT_SIZE; ++i) {
			int32_t k = b[i];
			if ((k & ~31) != 0)
				out[i] = 0;
			else
				out[i] = static_cast<uint32_t>(a[i]) >> k;
		}
	}


	static void sisd_sra(const T* a, const T* b, T* out)
	{
		for (int i = 0; i < INPUT_SIZE; ++i) {
			int32_t k = b[i];
			if ((k & ~31) != 0)
				out[i] = (a[i] < 0) ? -1 : 0;
			else
				out[i] = a[i] >> k;
		}
	}

    /* AVX512 实现 */
    static void avx512_sll(const T* a, const T* b, T* out) {
        if constexpr (std::is_same_v<T, int>) {
            __m512i va = _mm512_loadu_si512(a);
            __m512i vb = _mm512_loadu_si512(b);
            __m512i vc = _mm512_sllv_epi32(va, vb);
            _mm512_storeu_si512(out, vc);
        }
    }
    static void avx512_srl(const T* a, const T* b, T* out) {
        if constexpr (std::is_same_v<T, int>) {
            __m512i va = _mm512_loadu_si512(a);
            __m512i vb = _mm512_loadu_si512(b);
            __m512i vc = _mm512_srlv_epi32(va, vb);
            _mm512_storeu_si512(out, vc);
        }
    }
    static void avx512_sra(const T* a, const T* b, T* out) {
        if constexpr (std::is_same_v<T, int>) {
            __m512i va = _mm512_loadu_si512(a);
            __m512i vb = _mm512_loadu_si512(b);
            __m512i vc = _mm512_srav_epi32(va, vb);
            _mm512_storeu_si512(out, vc);
        }
    }

	static constexpr auto make_ops() {
    	return std::array{
        	OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "avx512 sll", avx512_sll, sisd_sll },
        	OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "avx512 srl", avx512_srl, sisd_srl },
        	OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "avx512 sra", avx512_sra, sisd_sra },
    	};
	}
};
