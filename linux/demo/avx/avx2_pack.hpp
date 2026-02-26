template <typename T>
struct AVX2_PACK {
    static constexpr const char* CATEGORY = "avx2";
    static constexpr const char* CLASS_TYPE = "pack";

    static constexpr size_t LANES = 256 / (8 * sizeof(T));
    static constexpr size_t INPUT_SIZE = LANES; // Input element count

    // For Pack, we typically go from larger type to smaller type.
    // E.g. packssdw: int32 -> int16
    // Inputs are two vectors of source type.
    
    // We define this struct for the SOURCE type T.
    // The output type will be half the size of T.
    
    static constexpr int INPUT_ARGS = 2;
    using ARG1_TYPE = T;
    static constexpr size_t ARG1_SIZE = INPUT_SIZE;
    using ARG2_TYPE = T;
    static constexpr size_t ARG2_SIZE = INPUT_SIZE;

    // Output type deduction
    using OUTPUT_TYPE = 
        std::conditional_t<std::is_same_v<T, int32_t>, int16_t,
        std::conditional_t<std::is_same_v<T, int16_t>, int8_t,
        void>>;
    
    // Output has double the number of elements per vector, but same total bits (256)
    // Actually, pack instructions take 2 input vectors and produce 1 output vector of same width (256).
    // So output element count = 2 * Input element count.
    static constexpr size_t OUTPUT_SIZE = LANES * 2;

    /* SISD Pack (Signed Saturate) */
    static void sisd_packss(const T* a, const T* b, OUTPUT_TYPE* out) {
        // Lane behavior:
        // AVX2 Pack instructions usually work within 128-bit lanes.
        // packssdw:
        //   Output[0..3] = saturate(a[0..3])
        //   Output[4..7] = saturate(b[0..3])  <-- Note: B comes after A in low lane
        //   Output[8..11] = saturate(a[4..7])
        //   Output[12..15] = saturate(b[4..7])
        
        constexpr size_t half_in = INPUT_SIZE / 2; // Elements per 128-bit lane in input
        
        auto saturate = [](T val) -> OUTPUT_TYPE {
            constexpr T max_val = std::numeric_limits<OUTPUT_TYPE>::max();
            constexpr T min_val = std::numeric_limits<OUTPUT_TYPE>::min();
            if (val > max_val) return max_val;
            if (val < min_val) return min_val;
            return static_cast<OUTPUT_TYPE>(val);
        };

        // Low lane (first 128 bits of output)
        int out_idx = 0;
        for(size_t i=0; i<half_in; ++i) out[out_idx++] = saturate(a[i]);
        for(size_t i=0; i<half_in; ++i) out[out_idx++] = saturate(b[i]);
        
        // High lane (second 128 bits of output)
        for(size_t i=half_in; i<INPUT_SIZE; ++i) out[out_idx++] = saturate(a[i]);
        for(size_t i=half_in; i<INPUT_SIZE; ++i) out[out_idx++] = saturate(b[i]);
    }
    
    /* SISD Pack (Unsigned Saturate) - only for int16 -> uint8 */
    static void sisd_packus(const T* a, const T* b, typename std::make_unsigned<OUTPUT_TYPE>::type* out) {
         constexpr int half_in = INPUT_SIZE / 2; 
         using UOUT = typename std::make_unsigned<OUTPUT_TYPE>::type;
         
         auto saturate = [](T val) -> UOUT {
            constexpr T max_val = std::numeric_limits<UOUT>::max(); // 255
            constexpr T min_val = 0;
            if (val > max_val) return max_val;
            if (val < min_val) return min_val;
            return static_cast<UOUT>(val);
        };

        int out_idx = 0;
        for(size_t i=0; i<half_in; ++i) out[out_idx++] = saturate(a[i]);
        for(size_t i=0; i<half_in; ++i) out[out_idx++] = saturate(b[i]);
        for(size_t i=half_in; i<INPUT_SIZE; ++i) out[out_idx++] = saturate(a[i]);
        for(size_t i=half_in; i<INPUT_SIZE; ++i) out[out_idx++] = saturate(b[i]);
    }

    /* AVX2 Pack */
    static void avx2_packss(const T* a, const T* b, OUTPUT_TYPE* out) {
        if constexpr (std::is_same_v<T, int32_t>) {
            __m256i va = _mm256_loadu_si256((const __m256i*)a);
            __m256i vb = _mm256_loadu_si256((const __m256i*)b);
            __m256i res = _mm256_packs_epi32(va, vb);
            _mm256_storeu_si256((__m256i*)out, res);
        } else if constexpr (std::is_same_v<T, int16_t>) {
            __m256i va = _mm256_loadu_si256((const __m256i*)a);
            __m256i vb = _mm256_loadu_si256((const __m256i*)b);
            __m256i res = _mm256_packs_epi16(va, vb);
            _mm256_storeu_si256((__m256i*)out, res);
        }
    }
    
    // Special case handling for unsigned output in packus
    // Since our template structure expects OUTPUT_TYPE to be defined by T,
    // and packus returns unsigned, we might need a separate struct or just force cast.
    // For simplicity, we only test packss here.
    
    static constexpr auto make_ops() {
        return std::array{
            OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "packss", avx2_packss, sisd_packss }
        };
    }
};

// Register tests
#ifdef REGISTER_TEST
using AVX2_PACK_I32 = AVX2_PACK<int32_t>;
REGISTER_TEST(AVX2_PACK_I32)
using AVX2_PACK_I16 = AVX2_PACK<int16_t>;
REGISTER_TEST(AVX2_PACK_I16)
#endif
