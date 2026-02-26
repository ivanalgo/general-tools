struct AVX2_PERMUTE
{
    static constexpr const char* CATEGORY = "avx2";
    static constexpr const char* CLASS_TYPE = "permute";

    /* ============================================================
     * shuffle_epi32 (imm8 固化版本)
     * 每个 128bit lane 内独立 shuffle
     * ============================================================ */
    template <typename T>
    struct SHUFFLE_0123 {
        static constexpr const char* CATEGORY = "avx2";
        static constexpr const char* CLASS_TYPE = "permute";

        static constexpr size_t INPUT_SIZE = 256 / (8 * sizeof(T));

        static constexpr int INPUT_ARGS = 1;
        using ARG1_TYPE = T;
        static constexpr size_t ARG1_SIZE = INPUT_SIZE;

        using OUTPUT_TYPE = T;
        static constexpr size_t OUTPUT_SIZE = INPUT_SIZE;

        /* ---------- AVX ---------- */
        static void avx(const int* a, int* out)
        {
            __m256i v = _mm256_loadu_si256((const __m256i*)a);
            __m256i r = _mm256_shuffle_epi32(v, _MM_SHUFFLE(0, 1, 2, 3));
            _mm256_storeu_si256((__m256i*)out, r);
        }

        /* ---------- SISD ---------- */
        static void sisd(const int* a, int* out)
        {
            for (size_t lane = 0; lane < 2; ++lane) {
                size_t base = lane * 4;
                out[base+0] = a[base+3];
                out[base+1] = a[base+2];
                out[base+2] = a[base+1];
                out[base+3] = a[base+0];
            }
        }

        static constexpr auto make_ops()
        {
            return std::array{
                OpEntry1<ARG1_TYPE, OUTPUT_TYPE>{
                    "shuffle_0123", avx, sisd
                }
            };
        }
    };

    /* ============================================================
     * unpacklo_epi32
     * lane 内 interleave low
     * ============================================================ */
    template <typename T>
    struct UNPACKLO {
        static constexpr const char* CATEGORY = "avx2";
        static constexpr const char* CLASS_TYPE = "permute";

        static constexpr size_t INPUT_SIZE = 256 / (8 * sizeof(T));

        static constexpr int INPUT_ARGS = 2;
        using ARG1_TYPE = T;
        static constexpr size_t ARG1_SIZE = INPUT_SIZE;
        using ARG2_TYPE = T;
        static constexpr size_t ARG2_SIZE = INPUT_SIZE;

        using OUTPUT_TYPE = T;
        static constexpr size_t OUTPUT_SIZE = INPUT_SIZE;

        static void avx(const int* a, const int* b, int* out)
        {
            __m256i va = _mm256_loadu_si256((const __m256i*)a);
            __m256i vb = _mm256_loadu_si256((const __m256i*)b);
            __m256i r  = _mm256_unpacklo_epi32(va, vb);
            _mm256_storeu_si256((__m256i*)out, r);
        }

        static void sisd(const int* a, const int* b, int* out)
        {
            for (size_t lane = 0; lane < 2; ++lane) {
                size_t base = lane * 4;
                out[base+0] = a[base+0];
                out[base+1] = b[base+0];
                out[base+2] = a[base+1];
                out[base+3] = b[base+1];
            }
        }

        static constexpr auto make_ops()
        {
            return std::array{
                OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{
                    "unpacklo", avx, sisd
                }
            };
        }
    };

    /* ============================================================
     * unpackhi_epi32
     * ============================================================ */
    template <typename T>
    struct UNPACKHI {
        static constexpr const char* CATEGORY = "avx2";
        static constexpr const char* CLASS_TYPE = "permute";

        static constexpr size_t INPUT_SIZE = 256 / (8 * sizeof(T));

        static constexpr int INPUT_ARGS = 2;
        using ARG1_TYPE = T;
        static constexpr size_t ARG1_SIZE = INPUT_SIZE;
        using ARG2_TYPE = T;
        static constexpr size_t ARG2_SIZE = INPUT_SIZE;

        using OUTPUT_TYPE = T;
        static constexpr size_t OUTPUT_SIZE = INPUT_SIZE;

        static void avx(const int* a, const int* b, int* out)
        {
            __m256i va = _mm256_loadu_si256((const __m256i*)a);
            __m256i vb = _mm256_loadu_si256((const __m256i*)b);
            __m256i r  = _mm256_unpackhi_epi32(va, vb);
            _mm256_storeu_si256((__m256i*)out, r);
        }

        static void sisd(const int* a, const int* b, int* out)
        {
            for (size_t lane = 0; lane < 2; ++lane) {
                size_t base = lane * 4;
                out[base+0] = a[base+2];
                out[base+1] = b[base+2];
                out[base+2] = a[base+3];
                out[base+3] = b[base+3];
            }
        }

        static constexpr auto make_ops()
        {
            return std::array{
                OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{
                    "unpackhi", avx, sisd
                }
            };
        }
    };

    /* ============================================================
     * permute2x128 (cross-lane)
     * ============================================================ */
    template <typename T>
    struct SWAP_LANES {
        static constexpr const char* CATEGORY = "avx2";
        static constexpr const char* CLASS_TYPE = "permute";

        static constexpr size_t INPUT_SIZE = 256 / (8 * sizeof(T));

        static constexpr int INPUT_ARGS = 1;
        using ARG1_TYPE = T;
        static constexpr size_t ARG1_SIZE = INPUT_SIZE;

        using OUTPUT_TYPE = T;
        static constexpr size_t OUTPUT_SIZE = INPUT_SIZE;

        static void avx(const int* a, int* out)
        {
            __m256i v = _mm256_loadu_si256((const __m256i*)a);
            __m256i r = _mm256_permute2x128_si256(v, v, 0x01);
            _mm256_storeu_si256((__m256i*)out, r);
        }

        static void sisd(const int* a, int* out)
        {
            for (size_t i = 0; i < 4; ++i) {
                out[i]   = a[i+4];
                out[i+4] = a[i];
            }
        }

        static constexpr auto make_ops()
        {
            return std::array{
                OpEntry1<ARG1_TYPE, OUTPUT_TYPE>{
                    "swap_lanes", avx, sisd
                }
            };
        }
    };

    /* ============================================================
     * permutevar8x32
     * ============================================================ */
    template <typename T>
    struct PERMUTEVAR {
        static constexpr const char* CATEGORY = "avx2";
        static constexpr const char* CLASS_TYPE = "permute";

        static constexpr size_t INPUT_SIZE = 256 / (8 * sizeof(T));

        static constexpr int INPUT_ARGS = 2;
        using ARG1_TYPE = T;
        static constexpr size_t ARG1_SIZE = INPUT_SIZE;
        using ARG2_TYPE = T;
        static constexpr size_t ARG2_SIZE = INPUT_SIZE;

        using OUTPUT_TYPE = T;
        static constexpr size_t OUTPUT_SIZE = INPUT_SIZE;

        static void avx(const int* a, const int* idx, int* out)
        {
            __m256i v   = _mm256_loadu_si256((const __m256i*)a);
            __m256i vid = _mm256_loadu_si256((const __m256i*)idx);
            __m256i r   = _mm256_permutevar8x32_epi32(v, vid);
            _mm256_storeu_si256((__m256i*)out, r);
        }

        static void sisd(const int* a, const int* idx, int* out)
        {
            for (size_t i = 0; i < 8; ++i)
                out[i] = a[idx[i] & 7];
        }

        static constexpr auto make_ops()
        {
            return std::array{
                OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{
                    "permutevar", avx, sisd
                }
            };
        }
    };
};
