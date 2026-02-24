template<typename SRC, typename DST>
struct AVX512_CONVERT {

    using ARG1_TYPE   = SRC;
    using OUTPUT_TYPE = DST;

    static constexpr size_t SRC_LANES =
        512 / (8 * sizeof(SRC));

    static constexpr size_t DST_LANES =
        512 / (8 * sizeof(DST));

    static constexpr size_t INPUT_SIZE  = SRC_LANES;
    static constexpr size_t OUTPUT_SIZE = DST_LANES;

    static constexpr const char* CLASS_NAME = "avx512_convert";
    static constexpr int INPUT_ARGS = 1;

    /* ================= SISD ================= */

    static void sisd_convert(const SRC* a, DST* out)
    {
        constexpr size_t N =
            (SRC_LANES < DST_LANES ? SRC_LANES : DST_LANES);

        for (size_t i = 0; i < N; ++i)
            out[i] = static_cast<DST>(a[i]);
    }

    /* ================= AVX ================= */

    static void avx_convert(const SRC* a, DST* out)
    {
        if constexpr (std::is_same_v<SRC,int32_t> &&
                      std::is_same_v<DST,float>) {

            __m512i va = _mm512_loadu_si512(a);
            __m512  vc = _mm512_cvtepi32_ps(va);
            _mm512_storeu_ps(out, vc);

        } else if constexpr (std::is_same_v<SRC,int32_t> &&
                             std::is_same_v<DST,double>) {

            __m256i lo = _mm256_loadu_si256((__m256i*)a);
            __m512d vc = _mm512_cvtepi32_pd(lo);
            _mm512_storeu_pd(out, vc);

        } else if constexpr (std::is_same_v<SRC,float> &&
                             std::is_same_v<DST,int32_t>) {

            __m512 va = _mm512_loadu_ps(a);
            __m512i vc = _mm512_cvtps_epi32(va);
            _mm512_storeu_si512(out, vc);

        } else if constexpr (std::is_same_v<SRC,double> &&
                             std::is_same_v<DST,int32_t>) {

            __m512d va = _mm512_loadu_pd(a);
            __m256i vc = _mm512_cvtpd_epi32(va);
            _mm256_storeu_si256((__m256i*)out, vc);

        } else if constexpr (std::is_same_v<SRC,float> &&
                             std::is_same_v<DST,double>) {

            __m256 lo = _mm256_loadu_ps(a);
            __m512d vc = _mm512_cvtps_pd(lo);
            _mm512_storeu_pd(out, vc);

        } else if constexpr (std::is_same_v<SRC,double> &&
                             std::is_same_v<DST,float>) {

            __m512d va = _mm512_loadu_pd(a);
            __m256 vc = _mm512_cvtpd_ps(va);
            _mm256_storeu_ps(out, vc);

        } else {

            static_assert(!sizeof(SRC*),
                          "Unsupported convert type pair");
        }
    }

    static constexpr auto make_ops()
    {
        return std::array{
            OpEntry1<ARG1_TYPE, OUTPUT_TYPE>{
                "avx512 convert",
                avx_convert,
                sisd_convert
            }
        };
    }
};
