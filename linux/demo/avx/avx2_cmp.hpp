template <typename T>
struct AVX2_CMP_BOOL_TYPE;

template<>
struct AVX2_CMP_BOOL_TYPE<float> {
	using type = int32_t;
};

template<>
struct AVX2_CMP_BOOL_TYPE<int> {
	using type = int32_t;
};

template<>
struct AVX2_CMP_BOOL_TYPE<double> {
	using type = int64_t;
};

template<typename T>
struct AVX2_CMP {
	using ARG1_TYPE = T;
	using ARG2_TYPE = T;
	using OUTPUT_TYPE = typename AVX2_CMP_BOOL_TYPE<T>::type;

	static constexpr int INPUT_SIZE = 256 / (8 * sizeof(T));
	static constexpr const char* CLASS_NAME = "AVX2_CMP";
	static constexpr int INPUT_ARGS = 2;

	/* ================= AVX ================= */

	static void avx_gt(const T* a, const T* b, OUTPUT_TYPE* out)
	{
		if constexpr (std::is_same_v<T, float>) {
			__m256 va = _mm256_loadu_ps(a);
			__m256 vb = _mm256_loadu_ps(b);
			__m256 vc = _mm256_cmp_ps(va, vb, _CMP_GT_OQ);
			_mm256_storeu_si256((__m256i*)out,
								_mm256_castps_si256(vc));
		} else if constexpr (std::is_same_v<T, double>) {
			__m256d va = _mm256_loadu_pd(a);
			__m256d vb = _mm256_loadu_pd(b);
			__m256d vc = _mm256_cmp_pd(va, vb, _CMP_GT_OQ);
			_mm256_storeu_si256((__m256i*)out,
								_mm256_castpd_si256(vc));
		} else if constexpr (std::is_same_v<T, int>) {
			__m256i va = _mm256_loadu_si256((const __m256i*)a);
			__m256i vb = _mm256_loadu_si256((const __m256i*)b);
			__m256i vc = _mm256_cmpgt_epi32(va, vb);
			_mm256_storeu_si256((__m256i*)out, vc);
		} else {
			static_assert(std::is_same_v<T, void>,
						  "AVX2_CMP: unsupported type");
		}
	}

	static void avx_eq(const T* a, const T* b, OUTPUT_TYPE* out)
	{
		if constexpr (std::is_same_v<T, float>) {
			__m256 va = _mm256_loadu_ps(a);
			__m256 vb = _mm256_loadu_ps(b);
			__m256 vc = _mm256_cmp_ps(va, vb, _CMP_EQ_OQ);
			_mm256_storeu_si256((__m256i*)out,
								_mm256_castps_si256(vc));
		} else if constexpr (std::is_same_v<T, double>) {
			__m256d va = _mm256_loadu_pd(a);
			__m256d vb = _mm256_loadu_pd(b);
			__m256d vc = _mm256_cmp_pd(va, vb, _CMP_EQ_OQ);
			_mm256_storeu_si256((__m256i*)out,
								_mm256_castpd_si256(vc));
		} else if constexpr (std::is_same_v<T, int>) {
			__m256i va = _mm256_loadu_si256((const __m256i*)a);
			__m256i vb = _mm256_loadu_si256((const __m256i*)b);
			__m256i vc = _mm256_cmpeq_epi32(va, vb);
			_mm256_storeu_si256((__m256i*)out, vc);
		} else {
			static_assert(std::is_same_v<T, void>,
						  "AVX2_CMP: unsupported type");
		}
	}

	/* ================= SISD ================= */

	static void sisd_gt(const T* a, const T* b, OUTPUT_TYPE* out)
	{
		for (int i = 0; i < INPUT_SIZE; ++i)
			out[i] = (a[i] > b[i]) ? -1 : 0;
	}

	static void sisd_eq(const T* a, const T* b, OUTPUT_TYPE* out)
	{
		for (int i = 0; i < INPUT_SIZE; ++i)
			out[i] = (a[i] == b[i]) ? -1 : 0;
	}

	/* ================= OPS ================= */

	static constexpr auto make_ops()
	{
		return std::array{
			OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "cmp_gt", avx_gt, sisd_gt },
			OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "cmp_eq", avx_eq, sisd_eq },
		};
	}
};
