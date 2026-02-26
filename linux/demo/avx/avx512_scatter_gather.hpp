template<typename T>
struct AVX512_SCATTER_GATHER {
    static constexpr const char* CATEGORY = "avx512";
    static constexpr const char* CLASS_TYPE = "scatter_gather";

    static constexpr size_t LANES = 512 / (8 * sizeof(T));
    static constexpr size_t INPUT_SIZE = LANES;
    
    // For gather/scatter, we need:
    // 1. Base array (large enough to scatter into/gather from)
    // 2. Index array
    // 3. Source/Destination values
    
    // We'll simulate a larger memory block by using a static buffer or just checking correctness logic
    // But to fit into the framework's fixed size array input, we can do this:
    // ARG1: Data (to scatter) or Index (to gather)
    // ARG2: Index (to scatter) or Data (result of gather - not input)
    
    // Let's define:
    // Gather: Out = Base[Index]
    // Scatter: Base[Index] = In
    
    // But the framework expects: avx(arg1, arg2, out)
    
    // Simplified Test:
    // Gather: arg1 = index, arg2 = ignored (or base content?), out = gathered values
    // Scatter: arg1 = data, arg2 = index, out = scattered values (into a temp buffer, then read back?)
    
    // To make it robust and fit the framework:
    // We will use a fixed "Global" or "Static" buffer for the base memory.
    static constexpr size_t MEM_SIZE = 1024;
    static T memory_buffer[MEM_SIZE];

    static constexpr int INPUT_ARGS = 2;
    using ARG1_TYPE = T; // Data or Index base
    static constexpr size_t ARG1_SIZE = INPUT_SIZE;
    
    using ARG2_TYPE = int32_t; // Index
    static constexpr size_t ARG2_SIZE = INPUT_SIZE;

    using OUTPUT_TYPE = T;
    static constexpr size_t OUTPUT_SIZE = INPUT_SIZE;

    static void arg2_init(ARG2_TYPE (&idx)[INPUT_SIZE], const TestConfig& config) {
        // Init indices to be within [0, MEM_SIZE)
        // Ensure no conflict for scatter to verify easily, though scatter handles conflict (last write wins usually)
        std::mt19937 rng(12345);
        std::uniform_int_distribution<int32_t> dist(0, MEM_SIZE - 1);
        
        // For scatter, we want unique indices to verify all writes clearly? 
        // Or just random is fine.
        for(size_t i=0; i<INPUT_SIZE; ++i) {
            idx[i] = dist(rng);
        }
    }
    
    // Initialize memory buffer once
    static void InitMemory() {
        for(size_t i=0; i<MEM_SIZE; ++i) {
            memory_buffer[i] = static_cast<T>(i); // Pattern
        }
    }

    /* SISD Gather */
    static void sisd_gather(const T* /*ignored*/, const int32_t* idx, T* out) {
        // InitMemory(); // Ensure known state? Or assume init?
        // Let's assume InitMemory is called before or we just use what's there.
        // Better: Initialize memory here to be deterministic
        for(size_t k=0; k<MEM_SIZE; ++k) memory_buffer[k] = static_cast<T>(k);

        for (size_t i = 0; i < INPUT_SIZE; ++i) {
            int32_t index = idx[i];
            if (index >= 0 && index < (int32_t)MEM_SIZE)
                out[i] = memory_buffer[index];
            else
                out[i] = 0; // Boundary check safe
        }
    }

    /* AVX512 Gather */
    static void avx512_gather(const T* /*ignored*/, const int32_t* idx, T* out) {
        // Re-init memory to match SISD state
        for(size_t k=0; k<MEM_SIZE; ++k) memory_buffer[k] = static_cast<T>(k);

        if constexpr (std::is_same_v<T, float>) {
            __m512i vidx = _mm512_loadu_si512(idx);
            // scale=4 for float (4 bytes)
            __m512 val = _mm512_i32gather_ps(vidx, memory_buffer, 4);
            _mm512_storeu_ps(out, val);
        } else if constexpr (std::is_same_v<T, int32_t>) {
            __m512i vidx = _mm512_loadu_si512(idx);
            __m512i val = _mm512_i32gather_epi32(vidx, memory_buffer, 4);
            _mm512_storeu_si512(out, val);
        } else if constexpr (std::is_same_v<T, double>) {
            // AVX512 gather for double usually uses __m256i for index if we want 8 doubles (512 bit)
            // But our ARG2 is int32_t array of size INPUT_SIZE (8).
            // _mm512_i32gather_pd expects __m256i index for 8 elements.
            __m256i vidx = _mm256_loadu_si256((__m256i*)idx);
            __m512d val = _mm512_i32gather_pd(vidx, memory_buffer, 8);
            _mm512_storeu_pd(out, val);
        }
    }

    /* SISD Scatter */
    static void sisd_scatter(const T* val, const int32_t* idx, T* out) {
        // Reset memory
        std::memset(memory_buffer, 0, sizeof(memory_buffer));

        for (size_t i = 0; i < INPUT_SIZE; ++i) {
            int32_t index = idx[i];
            if (index >= 0 && index < (int32_t)MEM_SIZE)
                memory_buffer[index] = val[i];
        }
        
        // Copy modified memory back to out to verify?
        // Or just copy the values at indices to verify they were written?
        for (size_t i = 0; i < INPUT_SIZE; ++i) {
             int32_t index = idx[i];
             if (index >= 0 && index < (int32_t)MEM_SIZE)
                out[i] = memory_buffer[index];
             else 
                out[i] = 0;
        }
    }

    /* AVX512 Scatter */
    static void avx512_scatter(const T* val, const int32_t* idx, T* out) {
         // Reset memory
        std::memset(memory_buffer, 0, sizeof(memory_buffer));

        if constexpr (std::is_same_v<T, float>) {
            __m512i vidx = _mm512_loadu_si512(idx);
            __m512 vval = _mm512_loadu_ps(val);
            _mm512_i32scatter_ps(memory_buffer, vidx, vval, 4);
        } else if constexpr (std::is_same_v<T, int32_t>) {
            __m512i vidx = _mm512_loadu_si512(idx);
            __m512i vval = _mm512_loadu_si512(val);
            _mm512_i32scatter_epi32(memory_buffer, vidx, vval, 4);
        } else if constexpr (std::is_same_v<T, double>) {
            __m256i vidx = _mm256_loadu_si256((__m256i*)idx);
            __m512d vval = _mm512_loadu_pd(val);
            _mm512_i32scatter_pd(memory_buffer, vidx, vval, 8);
        }
        
        // Read back to verify
        for (size_t i = 0; i < INPUT_SIZE; ++i) {
             int32_t index = idx[i];
             if (index >= 0 && index < (int32_t)MEM_SIZE)
                out[i] = memory_buffer[index];
             else 
                out[i] = 0;
        }
    }

    static constexpr auto make_ops() {
        return std::array{
            OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "gather", avx512_gather, sisd_gather },
            OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "scatter", avx512_scatter, sisd_scatter }
        };
    }
};

template<typename T>
T AVX512_SCATTER_GATHER<T>::memory_buffer[AVX512_SCATTER_GATHER<T>::MEM_SIZE];

// Register tests
#ifdef REGISTER_TEST
using AVX512_SG_INT = AVX512_SCATTER_GATHER<int>;
REGISTER_TEST(AVX512_SG_INT)
using AVX512_SG_FLOAT = AVX512_SCATTER_GATHER<float>;
REGISTER_TEST(AVX512_SG_FLOAT)
using AVX512_SG_DOUBLE = AVX512_SCATTER_GATHER<double>;
REGISTER_TEST(AVX512_SG_DOUBLE)
#endif
