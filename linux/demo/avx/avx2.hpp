template <typename T>
struct AVX2 {
    using INPUT_TYPE = T;
	static constexpr const char *CLASS_NAME = "avx2";
	static constexpr const int INPUT_ARGS = 2;
    static constexpr size_t INPUT_SIZE = 256 / (8 * sizeof(T));

    static void sisd_add(const T *a, const T *b, T *c) {
        for (size_t i = 0; i < INPUT_SIZE; i++) {
            c[i] = a[i] + b[i];
        }
    }

    static void sisd_sub(const T* a, const T* b, T* c)
    {
        for (size_t i = 0; i < INPUT_SIZE; ++i)
            c[i] = a[i] - b[i];
    }

    static void sisd_mul(const T* a, const T* b, T* c)
    {
        for (size_t i = 0; i < INPUT_SIZE; ++i)
            c[i] = a[i] * b[i];
    }

	static void sisd_div(const T *a, const T *b, T *c)
	{
		for (size_t i = 0; i < INPUT_SIZE; ++i)
			c[i] = a[i] / b[i];
	}

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
            __m256i va = _mm256_loadu_si256((const __m256i*)a);
            __m256i vb = _mm256_loadu_si256((const __m256i*)b);
            __m256i vc = _mm256_add_epi32(va, vb);
            _mm256_storeu_si256((__m256i*)c, vc);
        } else {
            static_assert(std::is_same_v<T, void>,
                          "avx_add: unsupported type");
        }
    }

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
            __m256i va = _mm256_loadu_si256((const __m256i*)a);
            __m256i vb = _mm256_loadu_si256((const __m256i*)b);
            __m256i vc = _mm256_sub_epi32(va, vb);
            _mm256_storeu_si256((__m256i*)c, vc);
        } else {
            static_assert(std::is_same_v<T, void>,
                          "avx_sub: unsupported type");
        }
    }

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
            __m256i va = _mm256_loadu_si256((const __m256i*)a);
            __m256i vb = _mm256_loadu_si256((const __m256i*)b);
            __m256i vc = _mm256_mullo_epi32(va, vb);
            _mm256_storeu_si256((__m256i*)c, vc);
        } else {
            static_assert(std::is_same_v<T, void>,
                          "avx_mul: unsupported type");
        }
    }

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
        } else if constexpr (std::is_same_v<T, int>) {
            // AVX2 没有整数向量除法
            static_assert(std::is_same_v<T, void>,
                          "avx_div: int division is not supported in AVX2");
        } else {
            static_assert(std::is_same_v<T, void>,
                          "avx_div: unsupported type");
        }
    }

	static constexpr auto make_ops()
    {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            return std::array{
                OpEntry<T>{ "avx2 add", avx_add, sisd_add },
                OpEntry<T>{ "avx2 sub", avx_sub, sisd_sub },
                OpEntry<T>{ "avx2 mul", avx_mul, sisd_mul },
                OpEntry<T>{ "avx2 div", avx_div, sisd_div },
            };
        } else {
            return std::array{
                OpEntry<T>{ "avx2 add", avx_add, sisd_add },
                OpEntry<T>{ "avx2 sub", avx_sub, sisd_sub },
                OpEntry<T>{ "avx2 mul", avx_mul, sisd_mul },
            };
        }
    }
};

template<typename T>
struct AVX2_FMA {
    using INPUT_TYPE = T;
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
            OpEntry3<T>{ "fmadd",  avx_fmadd,  sisd_fmadd  },
            OpEntry3<T>{ "fmsub",  avx_fmsub,  sisd_fmsub  },
            OpEntry3<T>{ "fnmadd", avx_fnmadd, sisd_fnmadd },
            OpEntry3<T>{ "fnmsub", avx_fnmsub, sisd_fnmsub },
        };
    }
};
