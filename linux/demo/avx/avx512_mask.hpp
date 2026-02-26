template<typename T>
struct AVX512_MASK {
    static constexpr const char* CATEGORY = "avx512";
    static constexpr const char* CLASS_TYPE = "mask";

    static constexpr size_t INPUT_SIZE = 512 / (8 * sizeof(T));

    static constexpr int INPUT_ARGS = 3;
    using ARG1_TYPE = T;
    static constexpr size_t ARG1_SIZE = INPUT_SIZE;
    using ARG2_TYPE = T;
    static constexpr size_t ARG2_SIZE = INPUT_SIZE;

    using MASK_TYPE = std::conditional_t<
        std::is_same_v<T, double>,
        uint8_t,
        uint16_t
    >;
    using ARG3_TYPE = MASK_TYPE;
    static constexpr size_t ARG3_SIZE = 1;

    using OUTPUT_TYPE = T;
    static constexpr size_t OUTPUT_SIZE = INPUT_SIZE;

	/* ================= SISD ================= */

	static void sisd_mask_add(const T* a,
							  const T* b,
							  const MASK_TYPE* mask_ptr,
							  T* c)
	{
		MASK_TYPE mask = mask_ptr[0];

		for (size_t i = 0; i < INPUT_SIZE; ++i) {
			if (mask & (1 << i))
				c[i] = a[i] + b[i];	  // active lane
			else
				c[i] = a[i];			 // merge 语义
		}
	}

	static void sisd_maskz_add(const T* a,
							   const T* b,
							   const MASK_TYPE* mask_ptr,
							   T* c)
	{
		MASK_TYPE mask = mask_ptr[0];

		for (size_t i = 0; i < INPUT_SIZE; ++i) {
			if (mask & (1 << i))
				c[i] = a[i] + b[i];
			else
				c[i] = 0;				// zero 语义
		}
	}

	/* ================= AVX512 ================= */

	static void avx_mask_add(const T* a,
							 const T* b,
							 const MASK_TYPE* mask_ptr,
							 T* c)
	{
		MASK_TYPE mask = mask_ptr[0];

		if constexpr (std::is_same_v<T, float>) {

			__m512 va = _mm512_loadu_ps(a);
			__m512 vb = _mm512_loadu_ps(b);
			__m512 vc = _mm512_mask_add_ps(va, mask, va, vb);
			_mm512_storeu_ps(c, vc);

		} else if constexpr (std::is_same_v<T, double>) {

			__m512d va = _mm512_loadu_pd(a);
			__m512d vb = _mm512_loadu_pd(b);
			__m512d vc = _mm512_mask_add_pd(va, mask, va, vb);
			_mm512_storeu_pd(c, vc);

		} else if constexpr (std::is_same_v<T, int32_t>) {

			__m512i va = _mm512_loadu_si512(a);
			__m512i vb = _mm512_loadu_si512(b);
			__m512i vc = _mm512_mask_add_epi32(va, mask, va, vb);
			_mm512_storeu_si512(c, vc);

		} else {
			static_assert(std::is_same_v<T, void>,
						  "mask add unsupported type");
		}
	}

	static void avx_maskz_add(const T* a,
							  const T* b,
							  const MASK_TYPE* mask_ptr,
							  T* c)
	{
		MASK_TYPE mask = mask_ptr[0];

		if constexpr (std::is_same_v<T, float>) {

			__m512 va = _mm512_loadu_ps(a);
			__m512 vb = _mm512_loadu_ps(b);
			__m512 vc = _mm512_maskz_add_ps(mask, va, vb);
			_mm512_storeu_ps(c, vc);

		} else if constexpr (std::is_same_v<T, double>) {

			__m512d va = _mm512_loadu_pd(a);
			__m512d vb = _mm512_loadu_pd(b);
			__m512d vc = _mm512_maskz_add_pd(mask, va, vb);
			_mm512_storeu_pd(c, vc);

		} else if constexpr (std::is_same_v<T, int32_t>) {

			__m512i va = _mm512_loadu_si512(a);
			__m512i vb = _mm512_loadu_si512(b);
			__m512i vc = _mm512_maskz_add_epi32(mask, va, vb);
			_mm512_storeu_si512(c, vc);

		} else {
			static_assert(std::is_same_v<T, void>,
						  "maskz add unsupported type");
		}
	}

	/* ================= OPS ================= */

	static constexpr auto make_ops()
	{
		return std::array{
			OpEntry3<ARG1_TYPE, ARG2_TYPE, ARG3_TYPE, OUTPUT_TYPE>{
				"mask add (merge)",
				avx_mask_add,
				sisd_mask_add
			},
			OpEntry3<ARG1_TYPE, ARG2_TYPE, ARG3_TYPE, OUTPUT_TYPE>{
				"maskz add (zero)",
				avx_maskz_add,
				sisd_maskz_add
			},
		};
	}
};

// Register tests
#ifdef REGISTER_TEST
using AVX512_MASK_INT = AVX512_MASK<int>;
REGISTER_TEST(AVX512_MASK_INT)
using AVX512_MASK_FLOAT = AVX512_MASK<float>;
REGISTER_TEST(AVX512_MASK_FLOAT)
using AVX512_MASK_DOUBLE = AVX512_MASK<double>;
REGISTER_TEST(AVX512_MASK_DOUBLE)
#endif
