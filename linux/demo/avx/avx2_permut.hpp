namespace AVX2_PERMUTE
{
    /* ============================================================
     * shuffle_epi32 (imm8 固化版本)
     * 每个 128bit lane 内独立 shuffle
     * ============================================================ */
    template<typename T>
    struct SHUFFLE_0123;

    template<>
    struct SHUFFLE_0123<int>
    {
        using ARG1_TYPE   = int;
        using OUTPUT_TYPE = int;

        static constexpr int INPUT_SIZE = 8;
        static constexpr int INPUT_ARGS = 1;
		static constexpr size_t OUTPUT_SIZE = INPUT_SIZE;
        static constexpr const char* CLASS_NAME = "AVX2_SHUFFLE_0123";

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
            for (int lane = 0; lane < 2; ++lane) {
                int base = lane * 4;
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
    template<typename T>
    struct UNPACKLO;

    template<>
    struct UNPACKLO<int>
    {
        using ARG1_TYPE   = int;
        using ARG2_TYPE   = int;
        using OUTPUT_TYPE = int;

        static constexpr int INPUT_SIZE = 8;
		static constexpr size_t OUTPUT_SIZE = INPUT_SIZE;
        static constexpr int INPUT_ARGS = 2;
        static constexpr const char* CLASS_NAME = "AVX2_UNPACKLO";

        static void avx(const int* a, const int* b, int* out)
        {
            __m256i va = _mm256_loadu_si256((const __m256i*)a);
            __m256i vb = _mm256_loadu_si256((const __m256i*)b);
            __m256i r  = _mm256_unpacklo_epi32(va, vb);
            _mm256_storeu_si256((__m256i*)out, r);
        }

        static void sisd(const int* a, const int* b, int* out)
        {
            for (int lane = 0; lane < 2; ++lane) {
                int base = lane * 4;
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
    template<typename T>
    struct UNPACKHI;

    template<>
    struct UNPACKHI<int>
    {
        using ARG1_TYPE   = int;
        using ARG2_TYPE   = int;
        using OUTPUT_TYPE = int;

        static constexpr int INPUT_SIZE = 8;
		static constexpr size_t OUTPUT_SIZE = INPUT_SIZE;
        static constexpr int INPUT_ARGS = 2;
        static constexpr const char* CLASS_NAME = "AVX2_UNPACKHI";

        static void avx(const int* a, const int* b, int* out)
        {
            __m256i va = _mm256_loadu_si256((const __m256i*)a);
            __m256i vb = _mm256_loadu_si256((const __m256i*)b);
            __m256i r  = _mm256_unpackhi_epi32(va, vb);
            _mm256_storeu_si256((__m256i*)out, r);
        }

        static void sisd(const int* a, const int* b, int* out)
        {
            for (int lane = 0; lane < 2; ++lane) {
                int base = lane * 4;
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
    template<typename T>
    struct SWAP_LANES;

    template<>
    struct SWAP_LANES<int>
    {
        using ARG1_TYPE   = int;
        using OUTPUT_TYPE = int;

        static constexpr int INPUT_SIZE = 8;
		static constexpr size_t OUTPUT_SIZE = INPUT_SIZE;
        static constexpr int INPUT_ARGS = 1;
        static constexpr const char* CLASS_NAME = "AVX2_SWAP_LANES";

        static void avx(const int* a, int* out)
        {
            __m256i v = _mm256_loadu_si256((const __m256i*)a);
            __m256i r = _mm256_permute2x128_si256(v, v, 0x01);
            _mm256_storeu_si256((__m256i*)out, r);
        }

        static void sisd(const int* a, int* out)
        {
            for (int i = 0; i < 4; ++i) {
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
    template<typename T>
    struct PERMUTEVAR;

    template<>
    struct PERMUTEVAR<int>
    {
        using ARG1_TYPE   = int;
        using ARG2_TYPE   = int;
        using OUTPUT_TYPE = int;

        static constexpr int INPUT_SIZE = 8;
		static constexpr size_t OUTPUT_SIZE = INPUT_SIZE;
        static constexpr int INPUT_ARGS = 2;
        static constexpr const char* CLASS_NAME = "AVX2_PERMUTEVAR";

        static void avx(const int* a, const int* idx, int* out)
        {
            __m256i v   = _mm256_loadu_si256((const __m256i*)a);
            __m256i vid = _mm256_loadu_si256((const __m256i*)idx);
            __m256i r   = _mm256_permutevar8x32_epi32(v, vid);
            _mm256_storeu_si256((__m256i*)out, r);
        }

        static void sisd(const int* a, const int* idx, int* out)
        {
            for (int i = 0; i < 8; ++i)
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
