template <typename T>
struct AVX512_CMP
{
    using ARG1_TYPE   = T;
    using ARG2_TYPE   = T;

    // Store raw mask bits
    using OUTPUT_TYPE = std::conditional_t<
        std::is_same_v<T, double>,
        uint8_t,        // __mmask8
        uint16_t        // __mmask16 (float / int32)
    >;

    static constexpr const char* CLASS_NAME = "AVX512_CMP";
    static constexpr int  INPUT_ARGS  = 2;
    static constexpr size_t OUTPUT_SIZE = 1;

    static constexpr size_t INPUT_SIZE =
        std::is_same_v<T, double> ? 8 : 16;

    /* ========================= AVX ========================= */

    static void avx_gt(const T* a, const T* b, OUTPUT_TYPE* out)
    {
        if constexpr (std::is_same_v<T, float>) {
            __m512 va = _mm512_loadu_ps(a);
            __m512 vb = _mm512_loadu_ps(b);
            __mmask16 k = _mm512_cmp_ps_mask(va, vb, _CMP_GT_OQ);
            out[0] = static_cast<OUTPUT_TYPE>(k);

        } else if constexpr (std::is_same_v<T, double>) {
            __m512d va = _mm512_loadu_pd(a);
            __m512d vb = _mm512_loadu_pd(b);
            __mmask8 k = _mm512_cmp_pd_mask(va, vb, _CMP_GT_OQ);
            out[0] = static_cast<OUTPUT_TYPE>(k);

        } else if constexpr (std::is_same_v<T, int32_t>) {
            __m512i va = _mm512_loadu_si512(a);
            __m512i vb = _mm512_loadu_si512(b);
            __mmask16 k = _mm512_cmp_epi32_mask(va, vb, _MM_CMPINT_GT);
            out[0] = static_cast<OUTPUT_TYPE>(k);

        } else {
            static_assert(std::is_same_v<T, void>,
                          "AVX512_CMP::avx_gt unsupported type");
        }
    }

    static void avx_eq(const T* a, const T* b, OUTPUT_TYPE* out)
    {
        if constexpr (std::is_same_v<T, float>) {
            __m512 va = _mm512_loadu_ps(a);
            __m512 vb = _mm512_loadu_ps(b);
            __mmask16 k = _mm512_cmp_ps_mask(va, vb, _CMP_EQ_OQ);
            out[0] = static_cast<OUTPUT_TYPE>(k);

        } else if constexpr (std::is_same_v<T, double>) {
            __m512d va = _mm512_loadu_pd(a);
            __m512d vb = _mm512_loadu_pd(b);
            __mmask8 k = _mm512_cmp_pd_mask(va, vb, _CMP_EQ_OQ);
            out[0] = static_cast<OUTPUT_TYPE>(k);

        } else if constexpr (std::is_same_v<T, int32_t>) {
            __m512i va = _mm512_loadu_si512(a);
            __m512i vb = _mm512_loadu_si512(b);
            __mmask16 k = _mm512_cmp_epi32_mask(va, vb, _MM_CMPINT_EQ);
            out[0] = static_cast<OUTPUT_TYPE>(k);

        } else {
            static_assert(std::is_same_v<T, void>,
                          "AVX512_CMP::avx_eq unsupported type");
        }
    }

    /* ========================= SISD ========================= */

    static void sisd_gt(const T* a, const T* b, OUTPUT_TYPE* out)
    {
        OUTPUT_TYPE mask = 0;

        for (size_t i = 0; i < INPUT_SIZE; ++i) {
            if (a[i] > b[i])
                mask |= (OUTPUT_TYPE(1) << i);
        }

        out[0] = mask;
    }

    static void sisd_eq(const T* a, const T* b, OUTPUT_TYPE* out)
    {
        OUTPUT_TYPE mask = 0;

        for (size_t i = 0; i < INPUT_SIZE; ++i) {
            if (a[i] == b[i])
                mask |= (OUTPUT_TYPE(1) << i);
        }

        out[0] = mask;
    }

    /* ========================= OPS ========================= */

    static constexpr auto make_ops()
    {
        return std::array{
            OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{
                "avx512 cmp gt", avx_gt, sisd_gt
            },
            OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{
                "avx512 cmp eq", avx_eq, sisd_eq
            },
        };
    }
};
