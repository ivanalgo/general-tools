template <typename T>
struct AVX {
    using INPUT_TYPE = T;
	static constexpr const char *CLASS_NAME = "avx";
	static constexpr const int INPUT_ARGS = 2;

    static constexpr int BIT_WIDTH =
        std::is_integral_v<T> ? 128 : 256;

    static constexpr int INPUT_SIZE =
        BIT_WIDTH / (8 * sizeof(T));

    /* ================= add ================= */
    static void avx_add(const T* a, const T* b, T* c)
    {
        if constexpr (std::is_same_v<T, float>) {
            __m256 va = _mm256_loadu_ps(a);
            __m256 vb = _mm256_loadu_ps(b);
            __m256 vc = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(c, vc);

		} else if constexpr (std::is_same_v<T, double>) {
			__m256d va = _mm256_loadu_pd(a);
			__m256d vb = _mm256_loadu_pd(b);
			__m256d vc = _mm256_add_pd(va, vb);
			 _mm256_storeu_pd(c, vc);
        } else if constexpr (std::is_same_v<T, int>) {
            __m128i va = _mm_loadu_si128((const __m128i*)a);
            __m128i vb = _mm_loadu_si128((const __m128i*)b);
            __m128i vc = _mm_add_epi32(va, vb);
            _mm_storeu_si128((__m128i*)c, vc);

        } else {
            static_assert(std::is_same_v<T, void>,
                          "AVX::avx_add unsupported type");
        }
    }

    static void sisd_add(const T* a, const T* b, T* c)
    {
        for (int i = 0; i < INPUT_SIZE; ++i)
            c[i] = a[i] + b[i];
    }

    /* ================= sub ================= */
    static void avx_sub(const T* a, const T* b, T* c)
    {
        if constexpr (std::is_same_v<T, float>) {
            __m256 va = _mm256_loadu_ps(a);
            __m256 vb = _mm256_loadu_ps(b);
            __m256 vc = _mm256_sub_ps(va, vb);
            _mm256_storeu_ps(c, vc);

		} else if constexpr (std::is_same_v<T, double>) {
			__m256d va = _mm256_loadu_pd(a);
            __m256d vb = _mm256_loadu_pd(b);
            __m256d vc = _mm256_sub_pd(va, vb);
            _mm256_storeu_pd(c, vc);
        } else if constexpr (std::is_same_v<T, int>) {
            __m128i va = _mm_loadu_si128((const __m128i*)a);
            __m128i vb = _mm_loadu_si128((const __m128i*)b);
            __m128i vc = _mm_sub_epi32(va, vb);
            _mm_storeu_si128((__m128i*)c, vc);

        } else {
            static_assert(std::is_same_v<T, void>,
                          "AVX::avx_sub unsupported type");
        }
    }

    static void sisd_sub(const T* a, const T* b, T* c)
    {
        for (int i = 0; i < INPUT_SIZE; ++i)
            c[i] = a[i] - b[i];
    }

    /* ================= mul ================= */
    static void avx_mul(const T* a, const T* b, T* c)
    {
        if constexpr (std::is_same_v<T, float>) {
            __m256 va = _mm256_loadu_ps(a);
            __m256 vb = _mm256_loadu_ps(b);
            __m256 vc = _mm256_mul_ps(va, vb);
            _mm256_storeu_ps(c, vc);
		} else if constexpr (std::is_same_v<T, double>) {
            __m256d va = _mm256_loadu_pd(a);
            __m256d vb = _mm256_loadu_pd(b);
            __m256d vc = _mm256_mul_pd(va, vb);
            _mm256_storeu_pd(c, vc);
        } else if constexpr (std::is_same_v<T, int>) {
            __m128i va = _mm_loadu_si128((const __m128i*)a);
            __m128i vb = _mm_loadu_si128((const __m128i*)b);
            __m128i vc = _mm_mullo_epi32(va, vb); // SSE4.1
            _mm_storeu_si128((__m128i*)c, vc);

        } else {
            static_assert(std::is_same_v<T, void>,
                          "AVX::avx_mul unsupported type");
        }
    }

    static void sisd_mul(const T* a, const T* b, T* c)
    {
        for (int i = 0; i < INPUT_SIZE; ++i)
            c[i] = a[i] * b[i];
    }

    /* ================= div ================= */
    static void avx_div(const T* a, const T* b, T* c)
    {
		if constexpr (std::is_same_v<T, float>) {
			__m256 va = _mm256_loadu_ps(a);
			__m256 vb = _mm256_loadu_ps(b);
			__m256 vc = _mm256_div_ps(va, vb);
			_mm256_storeu_ps(c, vc);
		} else if constexpr (std::is_same_v<T, double>) {
			__m256d va = _mm256_loadu_pd(a);
			__m256d vb = _mm256_loadu_pd(b);
			__m256d vc = _mm256_div_pd(va, vb);
			_mm256_storeu_pd(c, vc);
		} else {
			static_assert(std::is_same_v<T, void>,
						"AVX::avx_div unsupported type");
		}
    }

    static void sisd_div(const T* a, const T* b, T* c)
    {
        for (int i = 0; i < INPUT_SIZE; ++i)
            c[i] = a[i] / b[i];
    }

    /* ========= OPS TABLE ========= */
    static constexpr auto make_ops()
    {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T,double>) {
            return std::array{
                OpEntry<T>{ "avx add", avx_add, sisd_add },
                OpEntry<T>{ "avx sub", avx_sub, sisd_sub },
                OpEntry<T>{ "avx mul", avx_mul, sisd_mul },
                OpEntry<T>{ "avx div", avx_div, sisd_div },
            };
        } else if constexpr (std::is_integral_v<T>) {
            return std::array{
                OpEntry<T>{ "sse add", avx_add, sisd_add },
                OpEntry<T>{ "sse sub", avx_sub, sisd_sub },
                OpEntry<T>{ "sse mul", avx_mul, sisd_mul },
            };
        }
    }
};
