template <typename T>
struct AVX512_REDUCE {
    static constexpr const char* CLASS_NAME = "avx512_reduce";

    static constexpr size_t INPUT_SIZE = 512 / (8 * sizeof(T));

    static constexpr int INPUT_ARGS = 1;
    using ARG1_TYPE = T;
    static constexpr size_t ARG1_SIZE = INPUT_SIZE;

    using OUTPUT_TYPE = T;
    static constexpr size_t OUTPUT_SIZE = 1;

    /* ================= SISD ================= */

    static void sisd_sum(const T* a, T* out)
    {
        T sum = 0;
        for (size_t i = 0; i < INPUT_SIZE; ++i)
            sum += a[i];
        out[0] = sum;
    }

    static void sisd_max(const T* a, T* out)
    {
        T m = a[0];
        for (size_t i = 1; i < INPUT_SIZE; ++i)
            m = std::max(m, a[i]);
        out[0] = m;
    }

    static void sisd_min(const T* a, T* out)
    {
        T m = a[0];
        for (size_t i = 1; i < INPUT_SIZE; ++i)
            m = std::min(m, a[i]);
        out[0] = m;
    }

    /* ================= AVX512 ================= */

    static void avx_sum(const T* a, T* out)
    {
        if constexpr (std::is_same_v<T, float>) {

            __m512 v = _mm512_loadu_ps(a);
            out[0] = _mm512_reduce_add_ps(v);

        } else if constexpr (std::is_same_v<T, double>) {

            __m512d v = _mm512_loadu_pd(a);
            out[0] = _mm512_reduce_add_pd(v);

        } else if constexpr (std::is_same_v<T, int32_t>) {

            __m512i v = _mm512_loadu_si512(a);
            out[0] = _mm512_reduce_add_epi32(v);

        } else {
            static_assert(std::is_same_v<T, void>,
                          "avx_sum: unsupported type");
        }
    }

    static void avx_max(const T* a, T* out)
    {
        if constexpr (std::is_same_v<T, float>) {

            __m512 v = _mm512_loadu_ps(a);
            out[0] = _mm512_reduce_max_ps(v);

        } else if constexpr (std::is_same_v<T, double>) {

            __m512d v = _mm512_loadu_pd(a);
            out[0] = _mm512_reduce_max_pd(v);

        } else if constexpr (std::is_same_v<T, int32_t>) {

            __m512i v = _mm512_loadu_si512(a);
            out[0] = _mm512_reduce_max_epi32(v);

        } else {
            static_assert(std::is_same_v<T, void>,
                          "avx_max: unsupported type");
        }
    }

    static void avx_min(const T* a, T* out)
    {
        if constexpr (std::is_same_v<T, float>) {

            __m512 v = _mm512_loadu_ps(a);
            out[0] = _mm512_reduce_min_ps(v);

        } else if constexpr (std::is_same_v<T, double>) {

            __m512d v = _mm512_loadu_pd(a);
            out[0] = _mm512_reduce_min_pd(v);

        } else if constexpr (std::is_same_v<T, int32_t>) {

            __m512i v = _mm512_loadu_si512(a);
            out[0] = _mm512_reduce_min_epi32(v);

        } else {
            static_assert(std::is_same_v<T, void>,
                          "avx_min: unsupported type");
        }
    }

    /* ================= OPS ================= */

    static constexpr auto make_ops()
    {
        return std::array{
            OpEntry1<ARG1_TYPE, OUTPUT_TYPE>{ "avx512 reduce sum", avx_sum, sisd_sum },
            OpEntry1<ARG1_TYPE, OUTPUT_TYPE>{ "avx512 reduce max", avx_max, sisd_max },
            OpEntry1<ARG1_TYPE, OUTPUT_TYPE>{ "avx512 reduce min", avx_min, sisd_min },
        };
    }
};
