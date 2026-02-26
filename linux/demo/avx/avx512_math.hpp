template<typename T>
struct AVX512_MATH {
    static constexpr const char* CATEGORY = "avx512";
    static constexpr const char* CLASS_TYPE = "math";

    static constexpr size_t LANES = 512 / (8 * sizeof(T));
    static constexpr size_t INPUT_SIZE = LANES;

    static constexpr int INPUT_ARGS = 1;
    using ARG1_TYPE = T;
    static constexpr size_t ARG1_SIZE = INPUT_SIZE;

    using OUTPUT_TYPE = T;
    static constexpr size_t OUTPUT_SIZE = INPUT_SIZE;

    /* SISD Math */
    static void sisd_rcp14(const T* a, T* out) {
        for(size_t i=0; i<INPUT_SIZE; ++i) {
            out[i] = 1.0f / a[i]; 
            // Approximation check needs high tolerance
        }
    }

    static void sisd_rsqrt14(const T* a, T* out) {
        for(size_t i=0; i<INPUT_SIZE; ++i) {
            out[i] = 1.0f / std::sqrt(a[i]);
        }
    }
    
    // Note: EXP2 is not easily available in standard C++ without cmath exp2f
    static void sisd_exp2(const T* a, T* out) {
        for(size_t i=0; i<INPUT_SIZE; ++i) {
            out[i] = std::exp2(a[i]);
        }
    }

    /* AVX512 Math */
    static void avx512_rcp14(const T* a, T* out) {
        if constexpr (std::is_same_v<T, float>) {
            __m512 va = _mm512_loadu_ps(a);
            __m512 res = _mm512_rcp14_ps(va);
            _mm512_storeu_ps(out, res);
        } else if constexpr (std::is_same_v<T, double>) {
            __m512d va = _mm512_loadu_pd(a);
            __m512d res = _mm512_rcp14_pd(va);
            _mm512_storeu_pd(out, res);
        }
    }

    static void avx512_rsqrt14(const T* a, T* out) {
        if constexpr (std::is_same_v<T, float>) {
            __m512 va = _mm512_loadu_ps(a);
            __m512 res = _mm512_rsqrt14_ps(va);
            _mm512_storeu_ps(out, res);
        } else if constexpr (std::is_same_v<T, double>) {
            __m512d va = _mm512_loadu_pd(a);
            __m512d res = _mm512_rsqrt14_pd(va);
            _mm512_storeu_pd(out, res);
        }
    }
    
    static void avx512_exp2(const T* a, T* out) {
         if constexpr (std::is_same_v<T, float>) {
            __m512 va = _mm512_loadu_ps(a);
            // _mm512_exp2a23_ps available in AVX512ER usually, but exp2 is common enough?
            // Wait, exp2 is AVX512ER (Exponential & Reciprocal).
            // Many CPUs (SKX/ICX) do NOT support ER. ER is for Xeon Phi (KNL).
            // Let's check if we should include it.
            // If we include it and CPU doesn't support it, it will crash (SIGILL).
            // Let's comment it out or include it but warn.
            // For now, let's skip exp2 to avoid SIGILL on standard Xeons.
            
            // Actually, rcp14 and rsqrt14 ARE in AVX512F. 
            // rcp28/rsqrt28/exp2 are AVX512ER.
            // So we only test rcp14/rsqrt14.
            (void)va; (void)out;
        }
    }

    static constexpr auto make_ops() {
        return std::array{
            OpEntry1<ARG1_TYPE, OUTPUT_TYPE>{ "rcp14", avx512_rcp14, sisd_rcp14 },
            OpEntry1<ARG1_TYPE, OUTPUT_TYPE>{ "rsqrt14", avx512_rsqrt14, sisd_rsqrt14 }
        };
    }
};

// Register tests
#ifdef REGISTER_TEST
using AVX512_MATH_FLOAT = AVX512_MATH<float>;
REGISTER_TEST(AVX512_MATH_FLOAT)
using AVX512_MATH_DOUBLE = AVX512_MATH<double>;
REGISTER_TEST(AVX512_MATH_DOUBLE)
#endif
