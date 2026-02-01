#include <immintrin.h>
#include <type_traits>
#include <stdint.h>
#include <random>
#include <assert.h>
#include <iostream>
#include <array>
#include <chrono>

template <typename T>
struct OpEntry {
    const char* name;

    void (*avx)(const T*, const T*, T*);
    void (*sisd)(const T*, const T*, T*);
};

template <typename T>
struct AVX {
	using INPUT_TYPE = T;
	static constexpr int INPUT_SIZE = 256 / (8 * sizeof(T));

	static void avx_add(const T* a, const T* b, T* c) {
        if constexpr (std::is_same_v<T, float>) {
            avx_add_float(a, b, c);
        } else if constexpr (std::is_same_v<T, int>) {
            avx_add_int(a, b, c);
        } else {
            static_assert(std::is_same_v<T, void>,
                          "AVX::avx_add only supports float or int");
        }
    }

    static void avx_add_float(const float* a,
                              const float* b,
                              float* c) {
        __m256 va = _mm256_loadu_ps(a);
        __m256 vb = _mm256_loadu_ps(b);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(c, vc);
    }

    static void avx_add_int(const int* a,
                            const int* b,
                            int* c) {
        __m256i va = _mm256_loadu_si256((__m256i*)a);
        __m256i vb = _mm256_loadu_si256((__m256i*)b);
        __m256i vc = _mm256_add_epi32(va, vb);
        _mm256_storeu_si256((__m256i*)c, vc);
    }

	static void sisd_add(const T *a, const T *b, T *c) {
		for (int i = 0; i < INPUT_SIZE; i++) {
			c[i] = a[i] + b[i];
		}
	}

	/* ================= sub ================= */
    static void avx_sub(const T* a, const T* b, T* c)
    {
        if constexpr (std::is_same_v<T, float>) {
            __m256 va = _mm256_loadu_ps(a);
            __m256 vb = _mm256_loadu_ps(b);
            __m256 vc = _mm256_sub_ps(va, vb);
            _mm256_storeu_ps(c, vc);
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

    static void sisd_sub(const T* a, const T* b, T* c)
    {
        for (size_t i = 0; i < INPUT_SIZE; ++i)
            c[i] = a[i] - b[i];
    }


    /* ================= mul ================= */
    static void avx_mul(const T* a, const T* b, T* c)
    {
        if constexpr (std::is_same_v<T, float>) {
            __m256 va = _mm256_loadu_ps(a);
            __m256 vb = _mm256_loadu_ps(b);
            __m256 vc = _mm256_mul_ps(va, vb);
            _mm256_storeu_ps(c, vc);
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

    static void sisd_mul(const T* a, const T* b, T* c)
    {
        for (size_t i = 0; i < INPUT_SIZE; ++i)
            c[i] = a[i] * b[i];
    }

    /* ================= div ================= */
    static void avx_div(const T* a, const T* b, T* c)
    {
        if constexpr (std::is_same_v<T, float>) {
            __m256 va = _mm256_loadu_ps(a);
            __m256 vb = _mm256_loadu_ps(b);
            __m256 vc = _mm256_div_ps(va, vb);
            _mm256_storeu_ps(c, vc);
        } else {
            static_assert(std::is_same_v<T, void>,
                          "avx_div: integer division not supported");
        }
    }

    static void sisd_div(const T* a, const T* b, T* c)
    {
        if constexpr (std::is_same_v<T, float>) {
            for (size_t i = 0; i < INPUT_SIZE; ++i)
                c[i] = a[i] / b[i];
        } else {
            static_assert(std::is_same_v<T, void>,
                          "sisd_div: integer division not supported");
        }
    }

	/* ========= OPS TABLE ========= */
    static constexpr auto make_ops()
    {
        if constexpr (std::is_same_v<T, float>) {
            return std::array{
                OpEntry<T>{ "avx add", avx_add, sisd_add },
                OpEntry<T>{ "avx sub", avx_sub, sisd_sub },
                OpEntry<T>{ "avx mul", avx_mul, sisd_mul },
                OpEntry<T>{ "avx div", avx_div, sisd_div },
            };
        } else if constexpr (std::is_integral_v<T>) {
            return std::array{
                OpEntry<T>{ "avx add", avx_add, sisd_add },
                OpEntry<T>{ "avx sub", avx_sub, sisd_sub },
                OpEntry<T>{ "avx mul", avx_mul, sisd_mul },
            };
        }
    }
};

template <typename T>
struct AVX2 {
    using INPUT_TYPE = T;
    static constexpr size_t INPUT_SIZE = 256 / (8 * sizeof(T));

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
        if constexpr (std::is_same_v<T, float>) {
            return std::array{
                OpEntry<T>{ "avx2 add", avx_add, sisd_add },
                OpEntry<T>{ "avx2 sub", avx_sub, sisd_sub },
                OpEntry<T>{ "avx2 mul", avx_mul, sisd_mul },
                OpEntry<T>{ "avx2 div", avx_div, sisd_div },
            };
        } else {
            return std::array{
                OpEntry<T>{ "avx2 add", avx_add, sisd_add },
                OpEntry<T>{ "avx2 sub", avx_sub, sisd_sub },
                OpEntry<T>{ "avx2 mul", avx_mul, sisd_mul },
            };
        }
    }
};

template <typename T, size_t N>
void RandomInit(T (&a)[N])
{
    // 用时间和 random_device 混合，避免退化
    auto seed = static_cast<uint32_t>(
        std::chrono::high_resolution_clock::now()
            .time_since_epoch()
            .count()
    ) ^ std::random_device{}();

    std::mt19937 rng(seed);

    if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> dist(
            static_cast<T>(-1.0),
            static_cast<T>(1.0)
        );
        for (size_t i = 0; i < N; ++i) {
            a[i] = dist(rng);
        }
    } else if constexpr (std::is_integral_v<T>) {
        std::uniform_int_distribution<T> dist(
            static_cast<T>(-100),
            static_cast<T>(100)
        );
        for (size_t i = 0; i < N; ++i) {
            a[i] = dist(rng);
        }
    } else {
        static_assert(std::is_same_v<T, void>,
                      "RandomInit: unsupported type");
    }
}

template <typename T, size_t N>
bool CmpResult(T (&a)[N], T (&b)[N])
{
	for (size_t i = 0; i < N; ++i) {
		if (a[i] != b[i])
			return false;
	}

	return true;
}

template <typename T, size_t N>
void Debug(const char *token, T (&a)[N])
{
	std::cout << token;
	for (size_t i = 0; i < N; ++i) {
		std::cout << a[i] << " ";
	}

	std::cout << "\n";
}

template <typename Class>
void RandomTest()
{
	constexpr auto ops = Class::make_ops();
	for (const auto& op : ops) {
		typename Class::INPUT_TYPE a[Class::INPUT_SIZE];
		typename Class::INPUT_TYPE b[Class::INPUT_SIZE];
		typename Class::INPUT_TYPE avx_c[Class::INPUT_SIZE];
		typename Class::INPUT_TYPE sisd_c[Class::INPUT_SIZE];

		RandomInit(a);
		RandomInit(b);

		op.avx(a, b, avx_c);
		op.sisd(a, b, sisd_c);

		std::cout << op.name << "\n";
		Debug("a = ", a);
		Debug("b = ", b);
		Debug("c = ", avx_c);

		assert(CmpResult(avx_c, sisd_c));
	}
}

int main()
{
	RandomTest<AVX<int>>();
	RandomTest<AVX<float>>();
	RandomTest<AVX2<int>>();
	RandomTest<AVX2<float>>();
}
