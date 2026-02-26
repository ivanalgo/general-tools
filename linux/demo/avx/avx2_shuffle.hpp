template <typename T>
struct AVX2_SHUFFLE {
    static constexpr const char* CATEGORY = "avx2";
    static constexpr const char* CLASS_TYPE = "shuffle";

    static constexpr size_t LANES = 256 / (8 * sizeof(T));
    static constexpr size_t INPUT_SIZE = LANES;

    static constexpr int INPUT_ARGS = 2;
    using ARG1_TYPE = T;
    static constexpr size_t ARG1_SIZE = INPUT_SIZE;
    using ARG2_TYPE = int8_t; // Shuffle control mask (byte granularity for shuffle_epi8)
    static constexpr size_t ARG2_SIZE = 32; // Always 32 bytes for AVX2 shuffle_epi8

    using OUTPUT_TYPE = T;
    static constexpr size_t OUTPUT_SIZE = INPUT_SIZE;

    /* SISD Shuffle */
    static void sisd_shuffle(const T* a, const int8_t* mask, T* out) {
        // We simulate pshufb behavior:
        // For each byte in output, select byte from input based on mask.
        // If mask bit 7 is set, result is 0.
        // Else, use lower 4 bits as index (within 128-bit lane).
        
        const uint8_t* input_bytes = reinterpret_cast<const uint8_t*>(a);
        uint8_t* output_bytes = reinterpret_cast<uint8_t*>(out);
        const uint8_t* mask_bytes = reinterpret_cast<const uint8_t*>(mask);
        
        for (size_t i = 0; i < 32; ++i) {
            // Lane logic: 0-15 use 0-15 input, 16-31 use 16-31 input
            size_t lane_offset = (i < 16) ? 0 : 16;
            
            uint8_t idx = mask_bytes[i];
            if (idx & 0x80) {
                output_bytes[i] = 0;
            } else {
                uint8_t lookup_idx = (idx & 0x0F) + lane_offset;
                output_bytes[i] = input_bytes[lookup_idx];
            }
        }
    }

    /* AVX2 Shuffle */
    static void avx2_shuffle(const T* a, const int8_t* mask, T* out) {
        __m256i va = _mm256_loadu_si256((const __m256i*)a);
        __m256i vm = _mm256_loadu_si256((const __m256i*)mask);
        
        __m256i res = _mm256_shuffle_epi8(va, vm);
        
        _mm256_storeu_si256((__m256i*)out, res);
    }

    static constexpr auto make_ops() {
        if constexpr (std::is_integral_v<T>) {
             return std::array{
                OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "shuffle_epi8", avx2_shuffle, sisd_shuffle }
            };
        } else {
             // Not defining for float/double as shuffle_epi8 is integer domain,
             // though we can cast. For simplicity, only test int8/int16/int32.
             return std::array<OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>, 0>{};
        }
    }
};

// Register tests
#ifdef REGISTER_TEST
using AVX2_SHUFFLE_I8 = AVX2_SHUFFLE<int8_t>;
REGISTER_TEST(AVX2_SHUFFLE_I8)
using AVX2_SHUFFLE_I32 = AVX2_SHUFFLE<int32_t>;
REGISTER_TEST(AVX2_SHUFFLE_I32)
#endif
