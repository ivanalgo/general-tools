template<typename T>
struct AVX512_FMA {

    using ARG1_TYPE   = T;
    using ARG2_TYPE   = T;
    using ARG3_TYPE   = T;
    using OUTPUT_TYPE = T;

    static constexpr size_t LANES =
        512 / (8 * sizeof(T));

    static constexpr size_t INPUT_SIZE  = LANES;
    static constexpr size_t OUTPUT_SIZE = LANES;

    static constexpr const char* CLASS_NAME = "avx512_fma";
    static constexpr int INPUT_ARGS = 3;

    /* ================= SISD ================= */

    static void sisd_fma(const T* a,
                         const T* b,
                         const T* c,
                         T* out)
    {
        for (size_t i = 0; i < LANES; ++i)
            out[i] = a[i] * b[i] + c[i];
    }

    /* ================= AVX ================= */

    static void avx_fma(const T* a,
                        const T* b,
                        const T* c,
                        T* out)
    {
        if constexpr (std::is_same_v<T, float>) {

            __m512 va = _mm512_loadu_ps(a);
            __m512 vb = _mm512_loadu_ps(b);
            __m512 vc = _mm512_loadu_ps(c);

            __m512 vr = _mm512_fmadd_ps(va, vb, vc);

            _mm512_storeu_ps(out, vr);

        } else if constexpr (std::is_same_v<T, double>) {

            __m512d va = _mm512_loadu_pd(a);
            __m512d vb = _mm512_loadu_pd(b);
            __m512d vc = _mm512_loadu_pd(c);

            __m512d vr = _mm512_fmadd_pd(va, vb, vc);

            _mm512_storeu_pd(out, vr);

        } else {
            static_assert(!sizeof(T*),
                          "FMA only supports float/double");
        }
    }

    static constexpr auto make_ops()
    {
        return std::array{
            OpEntry3<ARG1_TYPE, ARG2_TYPE, ARG3_TYPE, OUTPUT_TYPE>{
                "avx512 fma",
                avx_fma,
                sisd_fma
            }
        };
    }
};
