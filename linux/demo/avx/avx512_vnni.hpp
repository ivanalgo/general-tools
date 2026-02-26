template<typename A, typename B, bool SATURATE = false>
struct AVX512_VNNI {
    static constexpr const char* CATEGORY = "avx512";
    static constexpr const char* CLASS_TYPE = "vnni";

    static constexpr size_t A_LANES = 512 / (8 * sizeof(A));
    static constexpr size_t GROUP_SIZE   = (sizeof(A) == 1 ? 4 : 2);
    static constexpr size_t OUT_LANES = A_LANES / GROUP_SIZE;

    static constexpr size_t INPUT_SIZE = A_LANES;

    static constexpr int INPUT_ARGS = 3;
    using ARG1_TYPE = A;
    static constexpr size_t ARG1_SIZE = INPUT_SIZE;
    using ARG2_TYPE = B;
    static constexpr size_t ARG2_SIZE = INPUT_SIZE;
    using ARG3_TYPE = int32_t;
    static constexpr size_t ARG3_SIZE = OUT_LANES;

    using OUTPUT_TYPE = int32_t;
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
                "vnni",
                avx_vnni,
                sisd_vnni
            }
        };
    }
};

// Register tests
#ifdef REGISTER_TEST
// INT8 for vnni in reference
using AVX512_VNNI_U8 = AVX512_VNNI<uint8_t, int8_t>;
REGISTER_TEST(AVX512_VNNI_U8)
// REGISTER_TEST(AVX512_VNNI<uint8_t, int8_t, true>) // Template macro limitation, see below
// INT16 for vnni in reference
using AVX512_VNNI_I16 = AVX512_VNNI<int16_t, int16_t>;
REGISTER_TEST(AVX512_VNNI_I16)
// REGISTER_TEST(AVX512_VNNI<int16_t, int16_t, true>)

// Macro can't handle commas in template args directly without varargs or alias
// Manual registration for multi-arg templates
using AVX512_VNNI_U8_SAT = AVX512_VNNI<uint8_t, int8_t, true>;
REGISTER_TEST(AVX512_VNNI_U8_SAT)
using AVX512_VNNI_I16_SAT = AVX512_VNNI<int16_t, int16_t, true>;
REGISTER_TEST(AVX512_VNNI_I16_SAT)
#endif
