template <typename T>
struct AVX512_PERMUTE {
    using ARG1_TYPE = T;
    using ARG2_TYPE = std::conditional_t<std::is_same_v<T, double>, int64_t, int>;
    using OUTPUT_TYPE = T;
    static constexpr size_t INPUT_SIZE = 512 / (8 * sizeof(T));
    static constexpr size_t OUTPUT_SIZE = 512 / (8 * sizeof(T));
    static constexpr const char* CLASS_NAME = "AVX512_PERMUTE";
    static constexpr int INPUT_ARGS = 2; // a + idx

	static void arg2_init(ARG2_TYPE (&idx)[INPUT_SIZE]) {

    	// 初始化顺序 0~INPUT_SIZE-1
    	for (size_t i = 0; i < INPUT_SIZE; ++i)
        	idx[i] = static_cast<ARG2_TYPE>(i);

    	// 用时间和 random_device 混合生成种子
		auto seed = static_cast<uint32_t>(
        	std::chrono::high_resolution_clock::now().time_since_epoch().count()
    	) ^ std::random_device{}();
    	std::mt19937 rng(seed);

    	// Fisher-Yates 洗牌
    	for (size_t i = INPUT_SIZE - 1; i > 0; --i) {
        	std::uniform_int_distribution<size_t> dist(0, i);
        	size_t j = dist(rng);
        	std::swap(idx[i], idx[j]);
    	}
	}

    /* SISD 实现 */
    static void sisd_permutevar(const T* a, const ARG2_TYPE* idx, T* out) {
        for (size_t i = 0; i < INPUT_SIZE; ++i) {
            out[i] = a[idx[i] & (INPUT_SIZE-1)];
        }
    }

    /* AVX512 实现 */
    static void avx512_permutevar(const T* a, const ARG2_TYPE* idx, T* out) {
        if constexpr (std::is_same_v<T, int>) {
            __m512i va = _mm512_loadu_si512(a);
            __m512i vidx = _mm512_loadu_si512(idx);
            __m512i vc = _mm512_permutexvar_epi32(vidx, va);
            _mm512_storeu_si512(out, vc);
        } else if constexpr (std::is_same_v<T, float>) {
            __m512 va = _mm512_loadu_ps(a);
            __m512i vidx = _mm512_loadu_si512(idx);
            __m512 vc = _mm512_permutexvar_ps(vidx, va);
            _mm512_storeu_ps(out, vc);
        } else if constexpr (std::is_same_v<T, double>) {
            __m512d va = _mm512_loadu_pd(a);
            __m512i vidx = _mm512_loadu_si512(idx);
            __m512d vc = _mm512_permutexvar_pd(vidx, va);
            _mm512_storeu_pd(out, vc);
        }
    }

	static constexpr auto make_ops() {
	    return std::array{
    	    OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "avx512 permutevar", avx512_permutevar, sisd_permutevar }
    	};
	}

};
