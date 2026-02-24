template<typename A, typename B, bool SATURATE = false>
struct AVX512_VNNI {

    using ARG1_TYPE   = A;
    using ARG2_TYPE   = B;
    using ARG3_TYPE   = int32_t;
    using OUTPUT_TYPE = int32_t;

    static constexpr const char* CLASS_NAME = "avx512_vnni";
    static constexpr int INPUT_ARGS = 3;

    /* ========== lane 计算 ========== */

    static constexpr size_t A_LANES = 512 / (8 * sizeof(A));
    static constexpr size_t GROUP_SIZE   = (sizeof(A) == 1 ? 4 : 2);
    static constexpr size_t OUT_LANES = A_LANES / GROUP_SIZE;

    static constexpr size_t INPUT_SIZE  = A_LANES;
    static constexpr size_t OUTPUT_SIZE = OUT_LANES;

    /* ================= SISD ================= */

    static void sisd_vnni(const A* a,
                          const B* b,
                          const int32_t* c,
                          int32_t* out)
    {
        for (size_t i = 0; i < OUT_LANES; ++i) {

            int32_t acc = c[i];

            for (size_t j = 0; j < GROUP_SIZE; ++j) {
                size_t idx = i * GROUP_SIZE + j;

                acc += static_cast<int32_t>(a[idx]) *
                       static_cast<int32_t>(b[idx]);
            }

            out[i] = acc;
        }
    }

    /* ================= AVX ================= */

    static void avx_vnni(const A* a,
                         const B* b,
                         const int32_t* c,
                         int32_t* out)
    {
        __m512i va = _mm512_loadu_si512(a);
        __m512i vb = _mm512_loadu_si512(b);
        __m512i vc = _mm512_loadu_si512(c);

        __m512i vr;

        if constexpr (sizeof(A) == 1 && !SATURATE) {
            vr = _mm512_dpbusd_epi32(vc, va, vb);
        }
        else if constexpr (sizeof(A) == 1 && SATURATE) {
            vr = _mm512_dpbusds_epi32(vc, va, vb);
        }
        else if constexpr (sizeof(A) == 2 && !SATURATE) {
            vr = _mm512_dpwssd_epi32(vc, va, vb);
        }
        else if constexpr (sizeof(A) == 2 && SATURATE) {
            vr = _mm512_dpwssds_epi32(vc, va, vb);
        }
        else {
            static_assert(!sizeof(A*),
                "Unsupported VNNI type combination");
        }

        _mm512_storeu_si512(out, vr);
    }

    static constexpr auto make_ops()
    {
        return std::array{
            OpEntry3<ARG1_TYPE, ARG2_TYPE, ARG3_TYPE, OUTPUT_TYPE>{
                "avx512 vnni",
                avx_vnni,
                sisd_vnni
            }
        };
    }
};
