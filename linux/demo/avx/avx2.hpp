#include <cstdint>
#include <cstring>

template <typename T>
struct AVX2 {
	using ARG1_TYPE = T;
	using ARG2_TYPE = T;
	using OUTPUT_TYPE = T;
	static constexpr const char *CLASS_NAME = "avx2";
	static constexpr const int INPUT_ARGS = 2;
	static constexpr size_t INPUT_SIZE = 256 / (8 * sizeof(T));
	static constexpr size_t OUTPUT_SIZE = INPUT_SIZE;

	static void sisd_add(const T *a, const T *b, T *c) {
		for (size_t i = 0; i < INPUT_SIZE; i++) {
			c[i] = a[i] + b[i];
		}
	}

	static void sisd_sub(const T* a, const T* b, T* c)
	{
		for (size_t i = 0; i < INPUT_SIZE; ++i)
			c[i] = a[i] - b[i];
	}

	static void sisd_mul(const T* a, const T* b, T* c)
	{
		for (size_t i = 0; i < INPUT_SIZE; ++i)
			c[i] = a[i] * b[i];
	}

	static void sisd_div(const T *a, const T *b, T *c)
	{
		for (size_t i = 0; i < INPUT_SIZE; ++i)
			c[i] = a[i] / b[i];
	}

	static void avx_add(const T* a, const T* b, T* c)
	{
		if constexpr (std::is_same_v<T, float>) {
			__m256 va = _mm256_loadu_ps(a);
			__m256 vb = _mm256_loadu_ps(b);
			__m256 vc = _mm256_add_ps(va, vb);
			_mm256_storeu_ps(c, vc);
		} else if constexpr (std::is_same_v<T, double>) {
			__m256d va = _mm256_loadu_pd(a);
			__m256d vb = _mm256_loadu_pd(b);
			__m256d vc = _mm256_add_pd(va, vb);
			_mm256_storeu_pd(c, vc);
		} else if constexpr (std::is_same_v<T, int>) {
			__m256i va = _mm256_loadu_si256((const __m256i*)a);
			__m256i vb = _mm256_loadu_si256((const __m256i*)b);
			__m256i vc = _mm256_add_epi32(va, vb);
			_mm256_storeu_si256((__m256i*)c, vc);
		} else {
			static_assert(std::is_same_v<T, void>,
						  "avx_add: unsupported type");
		}
	}

	static void avx_sub(const T* a, const T* b, T* c)
	{
		if constexpr (std::is_same_v<T, float>) {
			__m256 va = _mm256_loadu_ps(a);
			__m256 vb = _mm256_loadu_ps(b);
			__m256 vc = _mm256_sub_ps(va, vb);
			_mm256_storeu_ps(c, vc);
		} else if constexpr (std::is_same_v<T, double>) {
			__m256d va = _mm256_loadu_pd(a);
			__m256d vb = _mm256_loadu_pd(b);
			__m256d vc = _mm256_sub_pd(va, vb);
			_mm256_storeu_pd(c, vc);
		} else if constexpr (std::is_same_v<T, int>) {
			__m256i va = _mm256_loadu_si256((const __m256i*)a);
			__m256i vb = _mm256_loadu_si256((const __m256i*)b);
			__m256i vc = _mm256_sub_epi32(va, vb);
			_mm256_storeu_si256((__m256i*)c, vc);
		} else {
			static_assert(std::is_same_v<T, void>,
						  "avx_sub: unsupported type");
		}
	}

	static void avx_mul(const T* a, const T* b, T* c)
	{
		if constexpr (std::is_same_v<T, float>) {
			__m256 va = _mm256_loadu_ps(a);
			__m256 vb = _mm256_loadu_ps(b);
			__m256 vc = _mm256_mul_ps(va, vb);
			_mm256_storeu_ps(c, vc);
		} else if constexpr (std::is_same_v<T, double>) {
			__m256d va = _mm256_loadu_pd(a);
			__m256d vb = _mm256_loadu_pd(b);
			__m256d vc = _mm256_mul_pd(va, vb);
			_mm256_storeu_pd(c, vc);
		} else if constexpr (std::is_same_v<T, int>) {
			__m256i va = _mm256_loadu_si256((const __m256i*)a);
			__m256i vb = _mm256_loadu_si256((const __m256i*)b);
			__m256i vc = _mm256_mullo_epi32(va, vb);
			_mm256_storeu_si256((__m256i*)c, vc);
		} else {
			static_assert(std::is_same_v<T, void>,
						  "avx_mul: unsupported type");
		}
	}

	static void avx_div(const T* a, const T* b, T* c)
	{
		if constexpr (std::is_same_v<T, float>) {
			__m256 va = _mm256_loadu_ps(a);
			__m256 vb = _mm256_loadu_ps(b);
			__m256 vc = _mm256_div_ps(va, vb);
			_mm256_storeu_ps(c, vc);
		} else if constexpr (std::is_same_v<T, double>) {
			__m256d va = _mm256_loadu_pd(a);
			__m256d vb = _mm256_loadu_pd(b);
			__m256d vc = _mm256_div_pd(va, vb);
			_mm256_storeu_pd(c, vc);
		} else if constexpr (std::is_same_v<T, int>) {
			// AVX2 没有整数向量除法
			static_assert(std::is_same_v<T, void>,
						  "avx_div: int division is not supported in AVX2");
		} else {
			static_assert(std::is_same_v<T, void>,
						  "avx_div: unsupported type");
		}
	}

	static constexpr auto make_ops()
	{
		if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
			return std::array{
				OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "avx2 add", avx_add, sisd_add },
				OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "avx2 sub", avx_sub, sisd_sub },
				OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "avx2 mul", avx_mul, sisd_mul },
				OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "avx2 div", avx_div, sisd_div },
			};
		} else {
			return std::array{
				OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "avx2 add", avx_add, sisd_add },
				OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "avx2 sub", avx_sub, sisd_sub },
				OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "avx2 mul", avx_mul, sisd_mul },
			};
		}
	}
};
