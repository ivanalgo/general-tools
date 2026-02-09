template<typename T>
struct AVX2_FMA {
	using ARG1_TYPE = T;
	using ARG2_TYPE = T;
	using ARG3_TYPE = T;
	using OUTPUT_TYPE = T;
	static constexpr int INPUT_SIZE = 256 / (8 * sizeof(T));
	static constexpr const char* CLASS_NAME = "AVX2_FMA";
	static constexpr int INPUT_ARGS = 3;

	/* ================= AVX ================= */

	static void avx_fmadd(const T* a,
						  const T* b,
						  const T* c,
						  T* out)
	{
		if constexpr (std::is_same_v<T, float>) {
			__m256 va = _mm256_loadu_ps(a);
			__m256 vb = _mm256_loadu_ps(b);
			__m256 vc = _mm256_loadu_ps(c);
			__m256 vd = _mm256_fmadd_ps(va, vb, vc);
			_mm256_storeu_ps(out, vd);
		} else if constexpr (std::is_same_v<T, double>) {
			__m256d va = _mm256_loadu_pd(a);
			__m256d vb = _mm256_loadu_pd(b);
			__m256d vc = _mm256_loadu_pd(c);
			__m256d vd = _mm256_fmadd_pd(va, vb, vc);
			_mm256_storeu_pd(out, vd);
		} else {
			static_assert(std::is_same_v<T, void>,
						  "AVX2_FMA: unsupported type");
		}
	}

	static void avx_fmsub(const T* a,
						  const T* b,
						  const T* c,
						  T* out)
	{
		if constexpr (std::is_same_v<T, float>) {
			__m256 va = _mm256_loadu_ps(a);
			__m256 vb = _mm256_loadu_ps(b);
			__m256 vc = _mm256_loadu_ps(c);
			__m256 vd = _mm256_fmsub_ps(va, vb, vc);
			_mm256_storeu_ps(out, vd);
		} else if constexpr (std::is_same_v<T, double>) {
			__m256d va = _mm256_loadu_pd(a);
			__m256d vb = _mm256_loadu_pd(b);
			__m256d vc = _mm256_loadu_pd(c);
			__m256d vd = _mm256_fmsub_pd(va, vb, vc);
			_mm256_storeu_pd(out, vd);
		} else {
			static_assert(std::is_same_v<T, void>,
						  "AVX2_FMA: unsupported type");
		}
	}

	static void avx_fnmadd(const T* a,
						   const T* b,
						   const T* c,
						   T* out)
	{
		if constexpr (std::is_same_v<T, float>) {
			__m256 va = _mm256_loadu_ps(a);
			__m256 vb = _mm256_loadu_ps(b);
			__m256 vc = _mm256_loadu_ps(c);
			__m256 vd = _mm256_fnmadd_ps(va, vb, vc);
			_mm256_storeu_ps(out, vd);
		} else if constexpr (std::is_same_v<T, double>) {
			__m256d va = _mm256_loadu_pd(a);
			__m256d vb = _mm256_loadu_pd(b);
			__m256d vc = _mm256_loadu_pd(c);
			__m256d vd = _mm256_fnmadd_pd(va, vb, vc);
			_mm256_storeu_pd(out, vd);
		} else {
			static_assert(std::is_same_v<T, void>,
						  "AVX2_FMA: unsupported type");
		}
	}

	static void avx_fnmsub(const T* a,
						   const T* b,
						   const T* c,
						   T* out)
	{
		if constexpr (std::is_same_v<T, float>) {
			__m256 va = _mm256_loadu_ps(a);
			__m256 vb = _mm256_loadu_ps(b);
			__m256 vc = _mm256_loadu_ps(c);
			__m256 vd = _mm256_fnmsub_ps(va, vb, vc);
			_mm256_storeu_ps(out, vd);
		} else if constexpr (std::is_same_v<T, double>) {
			__m256d va = _mm256_loadu_pd(a);
			__m256d vb = _mm256_loadu_pd(b);
			__m256d vc = _mm256_loadu_pd(c);
			__m256d vd = _mm256_fnmsub_pd(va, vb, vc);
			_mm256_storeu_pd(out, vd);
		} else {
			static_assert(std::is_same_v<T, void>,
						  "AVX2_FMA: unsupported type");
		}
	}

	/* ================= SISD ================= */

	static void sisd_fmadd(const T* a,
						   const T* b,
						   const T* c,
						   T* out)
	{
		for (int i = 0; i < INPUT_SIZE; ++i)
			out[i] = a[i] * b[i] + c[i];
	}

	static void sisd_fmsub(const T* a,
						   const T* b,
						   const T* c,
						   T* out)
	{
		for (int i = 0; i < INPUT_SIZE; ++i)
			out[i] = a[i] * b[i] - c[i];
	}

	static void sisd_fnmadd(const T* a,
							const T* b,
							const T* c,
							T* out)
	{
		for (int i = 0; i < INPUT_SIZE; ++i)
			out[i] = -a[i] * b[i] + c[i];
	}

	static void sisd_fnmsub(const T* a,
							const T* b,
							const T* c,
							T* out)
	{
		for (int i = 0; i < INPUT_SIZE; ++i)
			out[i] = -a[i] * b[i] - c[i];
	}

	/* ================= OPS TABLE ================= */

	static constexpr auto make_ops()
	{
		return std::array{
			OpEntry3<ARG1_TYPE, ARG2_TYPE, ARG3_TYPE, OUTPUT_TYPE>{ "fmadd",  avx_fmadd,  sisd_fmadd  },
			OpEntry3<ARG1_TYPE, ARG2_TYPE, ARG3_TYPE, OUTPUT_TYPE>{ "fmsub",  avx_fmsub,  sisd_fmsub  },
			OpEntry3<ARG1_TYPE, ARG2_TYPE, ARG3_TYPE, OUTPUT_TYPE>{ "fnmadd", avx_fnmadd, sisd_fnmadd },
			OpEntry3<ARG1_TYPE, ARG2_TYPE, ARG3_TYPE, OUTPUT_TYPE>{ "fnmsub", avx_fnmsub, sisd_fnmsub },
		};
	}
};
