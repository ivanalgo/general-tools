#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <chrono>
#include <iomanip>

/*
 * memcopy.cpp file aims to demonstrate the perofrmance of different
 * implement of memcopy behaviors, such in assignment of C language,
 * rep movb instruction, AVX/AVX2/AVX512.
 */

/* =========================
 * aligned alloc
 * ========================= */
static void *aligned_malloc(std::size_t size, std::size_t align)
{
	void *p = nullptr;
	if (posix_memalign(&p, align, size) != 0)
		return nullptr;
	return p;
}

/* =========================
 * scalar copy
 * ========================= */
struct ScalarCopy {
	static void copy(void *dst, const void *src, std::size_t bytes)
	{
		std::size_t qwords = bytes / 8;
		std::size_t tail   = bytes % 8;

		auto *d = static_cast<uint64_t *>(dst);
		auto *s = static_cast<const uint64_t *>(src);

		for (std::size_t i = 0; i < qwords; ++i)
			d[i] = s[i];

		if (tail) {
			auto *dc = reinterpret_cast<char *>(d + qwords);
			auto *sc = reinterpret_cast<const char *>(s + qwords);
			for (std::size_t i = 0; i < tail; ++i)
				dc[i] = sc[i];
		}
	}

	static const char *name() { return "scalar_u64"; }
};

/* =========================
 * rep movsq
 * ========================= */
struct RepMovsqCopy {
	static void copy(void *dst, const void *src, std::size_t bytes)
	{
		std::size_t qwords = bytes / 8;
		std::size_t tail   = bytes % 8;

		asm volatile(
			"rep movsq\n"
			: "+D"(dst), "+S"(src), "+c"(qwords)
			:
			: "memory");

		if (tail) {
			auto *d = static_cast<unsigned char *>(dst);
			auto *s = static_cast<const unsigned char *>(src);
			for (std::size_t i = 0; i < tail; ++i)
				d[i] = s[i];
		}
	}

	static const char *name() { return "rep_movsq"; }
};

/* =========================
 * AVX1 (256-bit float)
 * ========================= */
struct AVX1Copy {
	static constexpr std::size_t kBlock = 32; // 256 bit

	static void copy(void *dst, const void *src, std::size_t bytes)
	{
		std::size_t i = 0;
		auto *d = static_cast<char *>(dst);
		auto *s = static_cast<const char *>(src);

		for (; i + 2 * kBlock <= bytes; i += 2 * kBlock) {
			__m256 v0 = _mm256_loadu_ps(
				reinterpret_cast<const float *>(s + i));
			__m256 v1 = _mm256_loadu_ps(
				reinterpret_cast<const float *>(s + i + kBlock));

			_mm256_storeu_ps(
				reinterpret_cast<float *>(d + i), v0);
			_mm256_storeu_ps(
				reinterpret_cast<float *>(d + i + kBlock), v1);
		}

		if (i < bytes)
			std::memcpy(d + i, s + i, bytes - i);
	}

	static const char *name() { return "avx1_ps"; }
};

/* =========================
 * AVX2 (256-bit integer)
 * ========================= */
struct AVX2Copy {
	static constexpr std::size_t kBlock = 32;

	static void copy(void *dst, const void *src, std::size_t bytes)
	{
		std::size_t i = 0;
		auto *d = static_cast<char *>(dst);
		auto *s = static_cast<const char *>(src);

		for (; i + 2 * kBlock <= bytes; i += 2 * kBlock) {
			__m256i v0 = _mm256_loadu_si256(
				reinterpret_cast<const __m256i *>(s + i));
			__m256i v1 = _mm256_loadu_si256(
				reinterpret_cast<const __m256i *>(s + i + kBlock));

			_mm256_storeu_si256(
				reinterpret_cast<__m256i *>(d + i), v0);
			_mm256_storeu_si256(
				reinterpret_cast<__m256i *>(d + i + kBlock), v1);
		}

		if (i < bytes)
			std::memcpy(d + i, s + i, bytes - i);
	}

	static const char *name() { return "avx2_si256"; }
};

/* =========================
 * AVX-512
 * ========================= */
struct AVX512Copy {
	static constexpr std::size_t kBlock = 64;

	static void copy(void *dst, const void *src, std::size_t bytes)
	{
		std::size_t i = 0;
		auto *d = static_cast<char *>(dst);
		auto *s = static_cast<const char *>(src);

		for (; i + kBlock <= bytes; i += kBlock) {
			__m512i v = _mm512_loadu_si512(s + i);
			_mm512_storeu_si512(d + i, v);
		}

		if (i < bytes)
			std::memcpy(d + i, s + i, bytes - i);
	}

	static const char *name() { return "avx512_si512"; }
};

/* =========================
 * benchmark
 * ========================= */
template <typename Impl>
static void benchmark(std::size_t bytes)
{
	void *src = aligned_malloc(bytes, 64);
	void *dst = aligned_malloc(bytes, 64);

	std::memset(src, 0xaa, bytes);
	std::memset(dst, 0x00, bytes);

	auto t0 = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 10000; i++) {
		Impl::copy(dst, src, bytes);
	}
	auto t1 = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> sec = t1 - t0;
	double gbps = bytes / sec.count() / (1024.0 * 1024.0 * 1024.0);

	std::cout << std::left  << std::setw(14) << Impl::name()
		  << " | size: " << std::right << std::setw(6)
		  << (bytes / (1024 * 1024)) << " MB"
		  << " | time: " << std::fixed << std::setw(10)
		  << std::setprecision(6) << sec.count() << " s"
		  << " | BW: "   << std::setw(8)
		  << std::setprecision(2) << gbps << " GB/s\n";

	std::free(src);
	std::free(dst);
}

int main()
{
	constexpr std::size_t size = (1UL << 20) * 25 / 2; // less than L3

	benchmark<ScalarCopy>(size);
	benchmark<RepMovsqCopy>(size);

#ifdef __AVX__
	benchmark<AVX1Copy>(size);
#endif

#ifdef __AVX2__
	benchmark<AVX2Copy>(size);
#endif

#ifdef __AVX512F__
	benchmark<AVX512Copy>(size);
#endif

	return 0;
}

