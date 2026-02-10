template<typename T>
struct AVX2_MINMAX {
    using ARG1_TYPE   = T;
    using ARG2_TYPE   = T;
    using OUTPUT_TYPE = T;

    static constexpr int INPUT_SIZE =
        256 / (8 * sizeof(T));

    static constexpr const char* CLASS_NAME = "AVX2_MINMAX";
    static constexpr int INPUT_ARGS = 2;

    /* ================= AVX ================= */

    static void avx_max(const T* a, const T* b, T* out)
    {
        if constexpr (std::is_same_v<T, int>) {
            __m256i va = _mm256_loadu_si256((const __m256i*)a);
            __m256i vb = _mm256_loadu_si256((const __m256i*)b);
            __m256i vc = _mm256_max_epi32(va, vb);
            _mm256_storeu_si256((__m256i*)out, vc);

        } else if constexpr (std::is_same_v<T, float>) {
            __m256 va = _mm256_loadu_ps(a);
            __m256 vb = _mm256_loadu_ps(b);
            __m256 vc = _mm256_max_ps(va, vb);
            _mm256_storeu_ps(out, vc);

        } else if constexpr (std::is_same_v<T, double>) {
            __m256d va = _mm256_loadu_pd(a);
            __m256d vb = _mm256_loadu_pd(b);
            __m256d vc = _mm256_max_pd(va, vb);
            _mm256_storeu_pd(out, vc);

        } else {
            static_assert(std::is_same_v<T, void>,
                "AVX2_MINMAX: unsupported type");
        }
    }

    static void avx_min(const T* a, const T* b, T* out)
    {
        if constexpr (std::is_same_v<T, int32_t>) {
            __m256i va = _mm256_loadu_si256((const __m256i*)a);
            __m256i vb = _mm256_loadu_si256((const __m256i*)b);
            __m256i vc = _mm256_min_epi32(va, vb);
            _mm256_storeu_si256((__m256i*)out, vc);

        } else if constexpr (std::is_same_v<T, float>) {
            __m256 va = _mm256_loadu_ps(a);
            __m256 vb = _mm256_loadu_ps(b);
            __m256 vc = _mm256_min_ps(va, vb);
            _mm256_storeu_ps(out, vc);

        } else if constexpr (std::is_same_v<T, double>) {
            __m256d va = _mm256_loadu_pd(a);
            __m256d vb = _mm256_loadu_pd(b);
            __m256d vc = _mm256_min_pd(va, vb);
            _mm256_storeu_pd(out, vc);

        } else {
            static_assert(std::is_same_v<T, void>,
                "AVX2_MINMAX: unsupported type");
        }
    }

    /* ================= SISD ================= */
    /* 模拟 x86 min/max 指令语义 */

    static void sisd_max(const T* a, const T* b, T* out)
    {
        for (int i = 0; i < INPUT_SIZE; ++i) {
            if constexpr (std::is_floating_point_v<T>) {
                if (std::isnan(a[i]) || std::isnan(b[i]))
                    out[i] = b[i];
                else
                    out[i] = (a[i] > b[i]) ? a[i] : b[i];
            } else {
                out[i] = (a[i] > b[i]) ? a[i] : b[i];
            }
        }
    }

    static void sisd_min(const T* a, const T* b, T* out)
    {
        for (int i = 0; i < INPUT_SIZE; ++i) {
            if constexpr (std::is_floating_point_v<T>) {
                if (std::isnan(a[i]) || std::isnan(b[i]))
                    out[i] = b[i];
                else
                    out[i] = (a[i] < b[i]) ? a[i] : b[i];
            } else {
                out[i] = (a[i] < b[i]) ? a[i] : b[i];
            }
        }
    }

    /* ================= OPS ================= */

    static constexpr auto make_ops()
    {
        return std::array{
            OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{
                "max", avx_max, sisd_max
            },
            OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{
                "min", avx_min, sisd_min
            },
        };
    }
};

