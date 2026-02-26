template <typename T>
struct AVX512_COMPRESS {
    static constexpr const char* CATEGORY = "avx512";
    static constexpr const char* CLASS_TYPE = "compress";

    static constexpr size_t LANES = 512 / (8 * sizeof(T));
    static constexpr size_t INPUT_SIZE = LANES;

    using MASK_TYPE =
        std::conditional_t<LANES == 8, uint8_t,
        std::conditional_t<LANES == 16, uint16_t, void>>;

    static constexpr int INPUT_ARGS = 2;
    using ARG1_TYPE = T;
    static constexpr size_t ARG1_SIZE = INPUT_SIZE;
    using ARG2_TYPE = MASK_TYPE;
    static constexpr size_t ARG2_SIZE = INPUT_SIZE;

    using OUTPUT_TYPE = T;
    static constexpr size_t OUTPUT_SIZE = INPUT_SIZE;

	static void sisd_compress(const T* a,
							  const MASK_TYPE* mask_ptr,
							  T* out)
	{
		MASK_TYPE mask = mask_ptr[0];
		size_t idx = 0;

		for (size_t i = 0; i < LANES; ++i) {
			if (mask & (1 << i))
				out[idx++] = a[i];
		}

		for (; idx < LANES; ++idx) {
			out[idx] = T(0);
		}
	}

	static void avx_compress(const T* a,
						 const MASK_TYPE* mask_ptr,
						 T* out)
	{
		MASK_TYPE mask = mask_ptr[0];

		for (size_t i = 0; i < LANES; ++i)
			out[i] = T(0);

		if constexpr (std::is_same_v<T, float>) {

			__mmask16 k = static_cast<__mmask16>(mask);
			__m512 va = _mm512_loadu_ps(a);
			_mm512_mask_compressstoreu_ps(out, k, va);

		} else if constexpr (std::is_same_v<T, double>) {

			__mmask8 k = static_cast<__mmask8>(mask);
			__m512d va = _mm512_loadu_pd(a);
			_mm512_mask_compressstoreu_pd(out, k, va);

		} else if constexpr (std::is_same_v<T, int32_t>) {

			__mmask16 k = static_cast<__mmask16>(mask);
			__m512i va = _mm512_loadu_si512(a);
			_mm512_mask_compressstoreu_epi32(out, k, va);

		} else {
			static_assert(!sizeof(T*), "Unsupported type in AVX512_COMPRESS");
		}
	}

	static constexpr auto make_ops()
	{
		return std::array{
			OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{
				"compress",
				avx_compress,
				sisd_compress
			}
		};
	}
};
