template<typename T>
struct AVX2_BITWISE {
	using ARG1_TYPE  = T;
	using ARG2_TYPE  = T;
	using OUTPUT_TYPE = T;

	static constexpr int INPUT_SIZE = 256 / (8 * sizeof(T));
	static constexpr const char* CLASS_NAME = "AVX2_BITWISE";
	static constexpr int INPUT_ARGS = 2;

	/* ================= AVX ================= */

	static void avx_and(const T* a, const T* b, OUTPUT_TYPE* out)
	{
		if constexpr (std::is_same_v<T, float>) {
			__m256 va = _mm256_loadu_ps(a);
			__m256 vb = _mm256_loadu_ps(b);
			__m256 vc = _mm256_and_ps(va, vb);
			_mm256_storeu_ps(out, vc);

		} else if constexpr (std::is_same_v<T, double>) {
			__m256d va = _mm256_loadu_pd(a);
			__m256d vb = _mm256_loadu_pd(b);
			__m256d vc = _mm256_and_pd(va, vb);
			_mm256_storeu_pd(out, vc);

		} else if constexpr (std::is_same_v<T, int>) {
			__m256i va = _mm256_loadu_si256((const __m256i*)a);
			__m256i vb = _mm256_loadu_si256((const __m256i*)b);
			__m256i vc = _mm256_and_si256(va, vb);
			_mm256_storeu_si256((__m256i*)out, vc);

		} else {
			static_assert(std::is_same_v<T, void>,
						  "AVX2_BITWISE: unsupported type");
		}
	}

	static void avx_or(const T* a, const T* b, OUTPUT_TYPE* out)
	{
		if constexpr (std::is_same_v<T, float>) {
			__m256 va = _mm256_loadu_ps(a);
			__m256 vb = _mm256_loadu_ps(b);
			__m256 vc = _mm256_or_ps(va, vb);
			_mm256_storeu_ps(out, vc);

		} else if constexpr (std::is_same_v<T, double>) {
			__m256d va = _mm256_loadu_pd(a);
			__m256d vb = _mm256_loadu_pd(b);
			__m256d vc = _mm256_or_pd(va, vb);
			_mm256_storeu_pd(out, vc);

		} else if constexpr (std::is_same_v<T, int>) {
			__m256i va = _mm256_loadu_si256((const __m256i*)a);
			__m256i vb = _mm256_loadu_si256((const __m256i*)b);
			__m256i vc = _mm256_or_si256(va, vb);
			_mm256_storeu_si256((__m256i*)out, vc);

		} else {
			static_assert(std::is_same_v<T, void>,
						  "AVX2_BITWISE: unsupported type");
		}
	}

	static void avx_xor(const T* a, const T* b, OUTPUT_TYPE* out)
	{
		if constexpr (std::is_same_v<T, float>) {
			__m256 va = _mm256_loadu_ps(a);
			__m256 vb = _mm256_loadu_ps(b);
			__m256 vc = _mm256_xor_ps(va, vb);
			_mm256_storeu_ps(out, vc);

		} else if constexpr (std::is_same_v<T, double>) {
			__m256d va = _mm256_loadu_pd(a);
			__m256d vb = _mm256_loadu_pd(b);
			__m256d vc = _mm256_xor_pd(va, vb);
			_mm256_storeu_pd(out, vc);

		} else if constexpr (std::is_same_v<T, int>) {
			__m256i va = _mm256_loadu_si256((const __m256i*)a);
			__m256i vb = _mm256_loadu_si256((const __m256i*)b);
			__m256i vc = _mm256_xor_si256(va, vb);
			_mm256_storeu_si256((__m256i*)out, vc);

		} else {
			static_assert(std::is_same_v<T, void>,
						  "AVX2_BITWISE: unsupported type");
		}
	}

	static void avx_andnot(const T* a, const T* b, OUTPUT_TYPE* out)
	{
		if constexpr (std::is_same_v<T, float>) {
			__m256 va = _mm256_loadu_ps(a);
			__m256 vb = _mm256_loadu_ps(b);
			__m256 vc = _mm256_andnot_ps(va, vb);
			_mm256_storeu_ps(out, vc);

		} else if constexpr (std::is_same_v<T, double>) {
			__m256d va = _mm256_loadu_pd(a);
			__m256d vb = _mm256_loadu_pd(b);
			__m256d vc = _mm256_andnot_pd(va, vb);
			_mm256_storeu_pd(out, vc);

		} else if constexpr (std::is_same_v<T, int>) {
			__m256i va = _mm256_loadu_si256((const __m256i*)a);
			__m256i vb = _mm256_loadu_si256((const __m256i*)b);
			__m256i vc = _mm256_andnot_si256(va, vb);
			_mm256_storeu_si256((__m256i*)out, vc);

		} else {
			static_assert(std::is_same_v<T, void>,
						  "AVX2_BITWISE: unsupported type");
		}
	}

	/* ================= SISD ================= */

	static void sisd_and(const T* a, const T* b, OUTPUT_TYPE* out)
	{
		if constexpr (std::is_same_v<T, int>) {
			for (int i = 0; i < INPUT_SIZE; ++i)
				out[i] = a[i] & b[i];
		} else if constexpr (std::is_same_v<T, float>) {
			for (int i = 0; i < INPUT_SIZE; ++i) {
				uint32_t ua, ub, uc;
				std::memcpy(&ua, &a[i], sizeof(ua));
				std::memcpy(&ub, &b[i], sizeof(ub));
				uc = ua & ub;
				std::memcpy(&out[i], &uc, sizeof(out[i]));
			}
		} else if constexpr (std::is_same_v<T, double>) {
			for (int i = 0; i < INPUT_SIZE; ++i) {
				uint64_t ua, ub, uc;
				std::memcpy(&ua, &a[i], sizeof(ua));
				std::memcpy(&ub, &b[i], sizeof(ub));
				uc = ua & ub;
				std::memcpy(&out[i], &uc, sizeof(out[i]));
			}
		}
	}

	static void sisd_or(const T* a, const T* b, OUTPUT_TYPE* out)
	{
		if constexpr (std::is_same_v<T, int>) {
			for (int i = 0; i < INPUT_SIZE; ++i)
				out[i] = a[i] | b[i];
		} else if constexpr (std::is_same_v<T, float>) {
			for (int i = 0; i < INPUT_SIZE; ++i) {
				uint32_t ua, ub, uc;
				std::memcpy(&ua, &a[i], sizeof(ua));
				std::memcpy(&ub, &b[i], sizeof(ub));
				uc = ua | ub;
				std::memcpy(&out[i], &uc, sizeof(out[i]));
			}
		} else if constexpr (std::is_same_v<T, double>) {
			for (int i = 0; i < INPUT_SIZE; ++i) {
				uint64_t ua, ub, uc;
				std::memcpy(&ua, &a[i], sizeof(ua));
				std::memcpy(&ub, &b[i], sizeof(ub));
				uc = ua | ub;
				std::memcpy(&out[i], &uc, sizeof(out[i]));
			}
		}
	}

	static void sisd_xor(const T* a, const T* b, OUTPUT_TYPE* out)
	{
		if constexpr (std::is_same_v<T, int>) {
			for (int i = 0; i < INPUT_SIZE; ++i)
				out[i] = a[i] ^ b[i];
		} else if constexpr (std::is_same_v<T, float>) {
			for (int i = 0; i < INPUT_SIZE; ++i) {
				uint32_t ua, ub, uc;
				std::memcpy(&ua, &a[i], sizeof(ua));
				std::memcpy(&ub, &b[i], sizeof(ub));
				uc = ua ^ ub;
				std::memcpy(&out[i], &uc, sizeof(out[i]));
			}
		} else if constexpr (std::is_same_v<T, double>) {
			for (int i = 0; i < INPUT_SIZE; ++i) {
				uint64_t ua, ub, uc;
				std::memcpy(&ua, &a[i], sizeof(ua));
				std::memcpy(&ub, &b[i], sizeof(ub));
				uc = ua ^ ub;
				std::memcpy(&out[i], &uc, sizeof(out[i]));
			}
		}
	}

	static void sisd_andnot(const T* a, const T* b, OUTPUT_TYPE* out)
	{
		if constexpr (std::is_same_v<T, int>) {
			for (int i = 0; i < INPUT_SIZE; ++i)
				out[i] = (~a[i]) & b[i];
		} else if constexpr (std::is_same_v<T, float>) {
			for (int i = 0; i < INPUT_SIZE; ++i) {
				uint32_t ua, ub, uc;
				std::memcpy(&ua, &a[i], sizeof(ua));
				std::memcpy(&ub, &b[i], sizeof(ub));
				uc = (~ua) & ub;
				std::memcpy(&out[i], &uc, sizeof(out[i]));
			}
		} else if constexpr (std::is_same_v<T, double>) {
			for (int i = 0; i < INPUT_SIZE; ++i) {
				uint64_t ua, ub, uc;
				std::memcpy(&ua, &a[i], sizeof(ua));
				std::memcpy(&ub, &b[i], sizeof(ub));
				uc =(~ ua) & ub;
				std::memcpy(&out[i], &uc, sizeof(out[i]));
			}
		}
	}

	/* ================= OPS ================= */

	static constexpr auto make_ops()
	{
		return std::array{
			OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "and",	   avx_and,	   sisd_and	   },
			OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "or",	   avx_or,	   sisd_or	   },
			OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "xor",	   avx_xor,	   sisd_xor	   },
			OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "andnot", avx_andnot, sisd_andnot },
		};
	}
};
