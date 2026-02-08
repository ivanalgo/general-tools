template<typename T>
struct AVX2_BLEND {
	using ARG1_TYPE = T;
	using ARG2_TYPE = T;
	using ARG3_TYPE = typename AVX2_CMP_BOOL_TYPE<T>::type;
	using MASK_TYPE = ARG3_TYPE;
	using OUTPUT_TYPE = T;

	static constexpr int INPUT_SIZE = 256 / (8 * sizeof(T));
	static constexpr const char* CLASS_NAME = "AVX2_BLEND";
	static constexpr int INPUT_ARGS = 3;

	static void avx_blend(const T* a,
						  const T* b,
						  const MASK_TYPE* mask,
						  T* out)
	{
		if constexpr (std::is_same_v<T, int32_t>) {
			__m256i va = _mm256_loadu_si256((const __m256i*)a);
			__m256i vb = _mm256_loadu_si256((const __m256i*)b);
			__m256i vm = _mm256_loadu_si256((const __m256i*)mask);

			__m256i res =
				_mm256_or_si256(
					_mm256_andnot_si256(vm, va),
					_mm256_and_si256(vm, vb));

			_mm256_storeu_si256((__m256i*)out, res);

		} else if constexpr (std::is_same_v<T, float>) {
			__m256 va = _mm256_loadu_ps(a);
			__m256 vb = _mm256_loadu_ps(b);
			__m256 vm = _mm256_castsi256_ps(
							_mm256_loadu_si256(
								(const __m256i*)mask));

			__m256 res =
				_mm256_or_ps(
					_mm256_andnot_ps(vm, va),
					_mm256_and_ps(vm, vb));

			_mm256_storeu_ps(out, res);

		} else if constexpr (std::is_same_v<T, double>) {
			__m256d va = _mm256_loadu_pd(a);
			__m256d vb = _mm256_loadu_pd(b);
			__m256d vm = _mm256_castsi256_pd(
							_mm256_loadu_si256(
								(const __m256i*)mask));

			__m256d res =
				_mm256_or_pd(
					_mm256_andnot_pd(vm, va),
					_mm256_and_pd(vm, vb));

			_mm256_storeu_pd(out, res);

		} else {
			static_assert(std::is_same_v<T, void>,
						  "AVX2_BLEND: unsupported type");
		}
	}

	static void sisd_blend(const T* a,
						   const T* b,
						   const MASK_TYPE* mask,
						   T* out)
	{
		for (int i = 0; i < INPUT_SIZE; ++i)
			out[i] = mask[i] ? b[i] : a[i];
	}

	static constexpr auto make_ops()
	{
		return std::array{
			OpEntry3<ARG1_TYPE, ARG2_TYPE, ARG3_TYPE, OUTPUT_TYPE>{
				"blend", avx_blend, sisd_blend
			},
		};
	}
};

