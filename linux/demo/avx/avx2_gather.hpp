template<typename T>
struct AVX2_GATHER {
    static constexpr const char* CATEGORY = "avx2";
    static constexpr const char* CLASS_TYPE = "gather";

    static constexpr size_t LANES = 256 / (8 * sizeof(T));
    static constexpr size_t INPUT_SIZE = LANES;
    
    // Using a fixed "Global" or "Static" buffer for the base memory.
    static constexpr size_t MEM_SIZE = 1024;
    static T memory_buffer[MEM_SIZE];

    static constexpr int INPUT_ARGS = 2;
    using ARG1_TYPE = T; // Placeholder for base address logic (we use static buffer)
    static constexpr size_t ARG1_SIZE = INPUT_SIZE;
    
    using ARG2_TYPE = int32_t; // Index
    static constexpr size_t ARG2_SIZE = INPUT_SIZE;

    using OUTPUT_TYPE = T;
    static constexpr size_t OUTPUT_SIZE = INPUT_SIZE;

    static void arg2_init(ARG2_TYPE (&idx)[INPUT_SIZE], const TestConfig& config) {
        std::mt19937 rng(12345);
        std::uniform_int_distribution<int32_t> dist(0, MEM_SIZE - 1);
        for(size_t i=0; i<INPUT_SIZE; ++i) {
            idx[i] = dist(rng);
        }
    }
    
    /* SISD Gather */
    static void sisd_gather(const T* /*ignored*/, const int32_t* idx, T* out) {
        // Init memory
        for(size_t k=0; k<MEM_SIZE; ++k) memory_buffer[k] = static_cast<T>(k);

        for (size_t i = 0; i < INPUT_SIZE; ++i) {
            int32_t index = idx[i];
            if (index >= 0 && index < (int32_t)MEM_SIZE)
                out[i] = memory_buffer[index];
            else
                out[i] = 0;
        }
    }

    /* AVX2 Gather */
    static void avx2_gather(const T* /*ignored*/, const int32_t* idx, T* out) {
        // Re-init memory to match SISD state
        for(size_t k=0; k<MEM_SIZE; ++k) memory_buffer[k] = static_cast<T>(k);

        if constexpr (std::is_same_v<T, float>) {
            __m256i vidx = _mm256_loadu_si256((const __m256i*)idx);
            // scale=4 for float
            __m256 val = _mm256_i32gather_ps(memory_buffer, vidx, 4);
            _mm256_storeu_ps(out, val);
        } else if constexpr (std::is_same_v<T, int32_t>) {
            __m256i vidx = _mm256_loadu_si256((const __m256i*)idx);
            // scale=4 for int32
            __m256i val = _mm256_i32gather_epi32(reinterpret_cast<const int*>(memory_buffer), vidx, 4);
            _mm256_storeu_si256((__m256i*)out, val);
        } else if constexpr (std::is_same_v<T, double>) {
            // AVX2 gather for double with 32-bit index (i32gather_pd)
            // It gathers 4 doubles (256-bit) using 4 indices (from __m128i, half of ARG2)
            // But our ARG2 is 256-bit (8 ints). 
            // Since INPUT_SIZE for double is 4, we only need 4 indices.
            // We load the first 4 indices (128-bit).
            
            __m128i vidx = _mm_loadu_si128((const __m128i*)idx);
            __m256d val = _mm256_i32gather_pd(memory_buffer, vidx, 8);
            _mm256_storeu_pd(out, val);
        } else if constexpr (std::is_same_v<T, int64_t>) {
             // i32gather_epi64: gathers 4 64-bit integers using 4 32-bit indices
             __m128i vidx = _mm_loadu_si128((const __m128i*)idx);
             __m256i val = _mm256_i32gather_epi64(reinterpret_cast<const long long*>(memory_buffer), vidx, 8);
             _mm256_storeu_si256((__m256i*)out, val);
        }
    }

    static constexpr auto make_ops() {
        return std::array{
            OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "gather", avx2_gather, sisd_gather }
        };
    }
};

template<typename T>
T AVX2_GATHER<T>::memory_buffer[AVX2_GATHER<T>::MEM_SIZE];

// Register tests
#ifdef REGISTER_TEST
using AVX2_GATHER_INT = AVX2_GATHER<int>;
REGISTER_TEST(AVX2_GATHER_INT)
using AVX2_GATHER_FLOAT = AVX2_GATHER<float>;
REGISTER_TEST(AVX2_GATHER_FLOAT)
using AVX2_GATHER_DOUBLE = AVX2_GATHER<double>;
REGISTER_TEST(AVX2_GATHER_DOUBLE)
using AVX2_GATHER_I64 = AVX2_GATHER<int64_t>;
REGISTER_TEST(AVX2_GATHER_I64)
#endif
