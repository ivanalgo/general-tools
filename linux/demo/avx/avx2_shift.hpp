#include <immintrin.h>
#include <array>
#include <type_traits>
#include <cstdint>

template <typename T>
struct AVX2_SHIFT {
    static constexpr const char* CATEGORY = "avx2";
    static constexpr const char* CLASS_TYPE = "shift";

    static constexpr size_t INPUT_SIZE = 256 / (8 * sizeof(T));

    static constexpr int INPUT_ARGS = 2;
    using ARG1_TYPE = T;
    static constexpr size_t ARG1_SIZE = INPUT_SIZE;
    using ARG2_TYPE = T;
    static constexpr size_t ARG2_SIZE = INPUT_SIZE;

    using OUTPUT_TYPE = T;
    static constexpr size_t OUTPUT_SIZE = INPUT_SIZE;

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
		for (size_t i = 0; i < INPUT_SIZE; ++i) {
			uint32_t k = static_cast<uint32_t>(b[i]);
			if (k >= 32)
				out[i] = 0;
			else
				out[i] = a[i] << k;
		}
	}


	static void sisd_srl(const T* a, const T* b, T* out)
	{
		for (size_t i = 0; i < INPUT_SIZE; ++i) {
			int32_t k = b[i];
			// 逻辑右移：若移位量 >= 32，通常结果为 0
			// x86 scalar shift 若移位量 >= 32 行为未定义或只取低5位
			// 但 AVX2 srlv 指令对于 count >= 32 会置零
			// 我们要模拟 AVX2 行为
			if ((k & ~31) != 0) // k < 0 || k >= 32
				out[i] = 0;
			else
				out[i] = (uint32_t)a[i] >> k;
		}
	}


	static void sisd_sra(const T* a, const T* b, T* out)
	{
		for (size_t i = 0; i < INPUT_SIZE; ++i) {
			int32_t k = b[i];
			// 算术右移：若移位量 >= 32，结果取决于符号位 (全0或全1)
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
			OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "sll", avx_sll, sisd_sll },
			OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "srl", avx_srl, sisd_srl },
			OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "sra", avx_sra, sisd_sra },
		};
	}
};

// Register tests
#ifdef REGISTER_TEST
using AVX2_SHIFT_INT = AVX2_SHIFT<int>;
REGISTER_TEST(AVX2_SHIFT_INT)
#endif
