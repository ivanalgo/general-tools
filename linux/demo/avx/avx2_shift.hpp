#include <immintrin.h>
#include <array>
#include <type_traits>
#include <cstdint>

template<typename T>
struct AVX2_SHIFT {
	static_assert(std::is_same_v<T, int32_t>);

	using ARG1_TYPE  = T;
	using ARG2_TYPE  = T;
	using OUTPUT_TYPE = T;

	static constexpr int INPUT_SIZE = 8;
	static constexpr const char* CLASS_NAME = "AVX2_SHIFT";
	static constexpr int INPUT_ARGS = 2;

	/* ================= AVX ================= */

	static void avx_sll(const T* a, const T* b, T* out)
	{
		__m256i va = _mm256_loadu_si256((const __m256i*)a);
		__m256i vb = _mm256_loadu_si256((const __m256i*)b);
		__m256i vc = _mm256_sllv_epi32(va, vb);
		_mm256_storeu_si256((__m256i*)out, vc);
	}

	static void avx_srl(const T* a, const T* b, T* out)
	{
		__m256i va = _mm256_loadu_si256((const __m256i*)a);
		__m256i vb = _mm256_loadu_si256((const __m256i*)b);
		__m256i vc = _mm256_srlv_epi32(va, vb);
		_mm256_storeu_si256((__m256i*)out, vc);
	}

	static void avx_sra(const T* a, const T* b, T* out)
	{
		__m256i va = _mm256_loadu_si256((const __m256i*)a);
		__m256i vb = _mm256_loadu_si256((const __m256i*)b);
		__m256i vc = _mm256_srav_epi32(va, vb);
		_mm256_storeu_si256((__m256i*)out, vc);
	}

	/* ================= SISD ================= */

	static void sisd_sll(const T* a, const T* b, T* out)
	{
		for (int i = 0; i < INPUT_SIZE; ++i) {
			uint32_t k = static_cast<uint32_t>(b[i]);
			if (k >= 32)
				out[i] = 0;
			else
				out[i] = a[i] << k;
		}
	}


	static void sisd_srl(const T* a, const T* b, T* out)
	{
		for (int i = 0; i < INPUT_SIZE; ++i) {
			int32_t k = b[i];
			if ((k & ~31) != 0)
				out[i] = 0;
			else
				out[i] = static_cast<uint32_t>(a[i]) >> k;
		}
	}


	static void sisd_sra(const T* a, const T* b, T* out)
	{
		for (int i = 0; i < INPUT_SIZE; ++i) {
			int32_t k = b[i];
			if ((k & ~31) != 0)
				out[i] = (a[i] < 0) ? -1 : 0;
			else
				out[i] = a[i] >> k;
		}
	}

	/* ================= OPS ================= */

	static constexpr auto make_ops()
	{
		return std::array{
			OpEntry<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "sll", avx_sll, sisd_sll },
			OpEntry<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "srl", avx_srl, sisd_srl },
			OpEntry<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "sra", avx_sra, sisd_sra },
		};
	}
};
