template <typename T>
struct AVX512 {
    using ARG1_TYPE   = T;
    using ARG2_TYPE   = T;
    using OUTPUT_TYPE = T;

    static constexpr const char* CLASS_NAME = "avx512";
    static constexpr int INPUT_ARGS = 2;
    static constexpr size_t INPUT_SIZE = 512 / (8 * sizeof(T));
	static constexpr size_t OUTPUT_SIZE = INPUT_SIZE;

    /* ================= SISD ================= */

    static void sisd_add(const T* a, const T* b, T* c)
    {
        for (size_t i = 0; i < INPUT_SIZE; ++i)
            c[i] = a[i] + b[i];
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

    static void sisd_div(const T* a, const T* b, T* c)
    {
        for (size_t i = 0; i < INPUT_SIZE; ++i)
            c[i] = a[i] / b[i];
    }

    /* ================= AVX512 ================= */

    static void avx_add(const T* a, const T* b, T* c)
    {
        if constexpr (std::is_same_v<T, float>) {
            __m512 va = _mm512_loadu_ps(a);
            __m512 vb = _mm512_loadu_ps(b);
            __m512 vc = _mm512_add_ps(va, vb);
            _mm512_storeu_ps(c, vc);

        } else if constexpr (std::is_same_v<T, double>) {
            __m512d va = _mm512_loadu_pd(a);
            __m512d vb = _mm512_loadu_pd(b);
            __m512d vc = _mm512_add_pd(va, vb);
            _mm512_storeu_pd(c, vc);

        } else if constexpr (std::is_same_v<T, int32_t>) {
            __m512i va = _mm512_loadu_si512(a);
            __m512i vb = _mm512_loadu_si512(b);
            __m512i vc = _mm512_add_epi32(va, vb);
            _mm512_storeu_si512(c, vc);

        } else {
            static_assert(std::is_same_v<T, void>,
                          "avx_add: unsupported type");
        }
    }

    static void avx_sub(const T* a, const T* b, T* c)
    {
        if constexpr (std::is_same_v<T, float>) {
            __m512 va = _mm512_loadu_ps(a);
            __m512 vb = _mm512_loadu_ps(b);
            __m512 vc = _mm512_sub_ps(va, vb);
            _mm512_storeu_ps(c, vc);

        } else if constexpr (std::is_same_v<T, double>) {
            __m512d va = _mm512_loadu_pd(a);
            __m512d vb = _mm512_loadu_pd(b);
            __m512d vc = _mm512_sub_pd(va, vb);
            _mm512_storeu_pd(c, vc);

        } else if constexpr (std::is_same_v<T, int32_t>) {
            __m512i va = _mm512_loadu_si512(a);
            __m512i vb = _mm512_loadu_si512(b);
            __m512i vc = _mm512_sub_epi32(va, vb);
            _mm512_storeu_si512(c, vc);

        } else {
            static_assert(std::is_same_v<T, void>,
                          "avx_sub: unsupported type");
        }
    }

    static void avx_mul(const T* a, const T* b, T* c)
    {
        if constexpr (std::is_same_v<T, float>) {
            __m512 va = _mm512_loadu_ps(a);
            __m512 vb = _mm512_loadu_ps(b);
            __m512 vc = _mm512_mul_ps(va, vb);
            _mm512_storeu_ps(c, vc);

        } else if constexpr (std::is_same_v<T, double>) {
            __m512d va = _mm512_loadu_pd(a);
            __m512d vb = _mm512_loadu_pd(b);
            __m512d vc = _mm512_mul_pd(va, vb);
            _mm512_storeu_pd(c, vc);

        } else if constexpr (std::is_same_v<T, int32_t>) {
            __m512i va = _mm512_loadu_si512(a);
            __m512i vb = _mm512_loadu_si512(b);
            __m512i vc = _mm512_mullo_epi32(va, vb);
            _mm512_storeu_si512(c, vc);

        } else {
            static_assert(std::is_same_v<T, void>,
                          "avx_mul: unsupported type");
        }
    }

    static void avx_div(const T* a, const T* b, T* c)
    {
        if constexpr (std::is_same_v<T, float>) {
            __m512 va = _mm512_loadu_ps(a);
            __m512 vb = _mm512_loadu_ps(b);
            __m512 vc = _mm512_div_ps(va, vb);
            _mm512_storeu_ps(c, vc);

        } else if constexpr (std::is_same_v<T, double>) {
            __m512d va = _mm512_loadu_pd(a);
            __m512d vb = _mm512_loadu_pd(b);
            __m512d vc = _mm512_div_pd(va, vb);
            _mm512_storeu_pd(c, vc);

        } else if constexpr (std::is_same_v<T, int32_t>) {
            static_assert(std::is_same_v<T, void>,
                          "avx_div: int division is not supported in AVX512");

        } else {
            static_assert(std::is_same_v<T, void>,
                          "avx_div: unsupported type");
        }
    }

    /* ================= OPS ================= */

    static constexpr auto make_ops()
    {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            return std::array{
                OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "avx512 add", avx_add, sisd_add },
                OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "avx512 sub", avx_sub, sisd_sub },
                OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "avx512 mul", avx_mul, sisd_mul },
                OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "avx512 div", avx_div, sisd_div },
            };
        } else {
            return std::array{
                OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "avx512 add", avx_add, sisd_add },
                OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "avx512 sub", avx_sub, sisd_sub },
                OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "avx512 mul", avx_mul, sisd_mul },
            };
        }
    }
};
