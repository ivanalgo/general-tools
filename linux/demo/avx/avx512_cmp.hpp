template <typename T>
struct AVX512_CMP {
    static constexpr const char* CLASS_NAME = "AVX512_CMP";

    static constexpr size_t INPUT_SIZE = 512 / (8 * sizeof(T));

    static constexpr int INPUT_ARGS = 2;
    using ARG1_TYPE = T;
    static constexpr size_t ARG1_SIZE = INPUT_SIZE;
    using ARG2_TYPE = T;
    static constexpr size_t ARG2_SIZE = INPUT_SIZE;

    // ---- OUTPUT ----
    // AVX-512 cmp 输出为 mask 寄存器
    // 这里使用标准整型来表达 bit mask：
    //
    // 64 lanes  -> uint64_t  (对应 __mmask64)
    // 32 lanes  -> uint32_t  (对应 __mmask32)
    // 16 lanes  -> uint16_t  (对应 __mmask16)
    // 8 lanes   -> uint8_t   (对应 __mmask8)

    using OUTPUT_TYPE =
        std::conditional_t<INPUT_SIZE == 64, uint64_t,   // __mmask64
        std::conditional_t<INPUT_SIZE == 32, uint32_t,   // __mmask32
        std::conditional_t<INPUT_SIZE == 16, uint16_t,   // __mmask16
        std::conditional_t<INPUT_SIZE == 8,  uint8_t,    // __mmask8
        void>>>>;
    static constexpr size_t OUTPUT_SIZE = 1;

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
