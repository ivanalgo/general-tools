#include <immintrin.h>
#include <type_traits>
#include <stdint.h>
#include <random>
#include <assert.h>
#include <iostream>
#include <array>
#include <chrono>

template <typename ARG1_TYPE, typename ARG2_TYPE, typename OUT_TYPE>
struct OpEntry {
    const char* name;

    void (*avx)(const ARG1_TYPE*, const ARG2_TYPE*, OUT_TYPE*);
    void (*sisd)(const ARG1_TYPE*, const ARG2_TYPE*, OUT_TYPE*);
};

template <typename ARG1_TYPE, typename ARG2_TYPE, typename ARG3_TYPE, typename OUT_TYPE>
struct OpEntry3 {
	const char* name;
	void (*avx)(const ARG1_TYPE*, const ARG2_TYPE*, const ARG3_TYPE*, OUT_TYPE*);
	void (*sisd)(const ARG1_TYPE*, const ARG2_TYPE*, const ARG3_TYPE*, OUT_TYPE*);
};

#include "avx1.hpp"
#include "avx2.hpp"
#include "avx2_fma.hpp"
#include "avx2_cmp.hpp"
#include "avx2_bitwise.hpp"
#include "avx2_shift.hpp"
#include "avx2_blend.hpp"
#include "avx2_minmax.hpp"

template <typename Class, typename = void>
struct has_arg3_init : std::false_type {};

template <typename Class>
struct has_arg3_init<Class, std::void_t<decltype(Class::arg3_init(std::declval<typename Class::ARG3_TYPE(&)[Class::INPUT_SIZE]>()))>> : std::true_type {};

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
		if constexpr (Class::INPUT_ARGS == 2) {
			typename Class::ARG1_TYPE a[Class::INPUT_SIZE];
			typename Class::ARG2_TYPE b[Class::INPUT_SIZE];
			typename Class::OUTPUT_TYPE avx_c[Class::INPUT_SIZE];
			typename Class::OUTPUT_TYPE sisd_c[Class::INPUT_SIZE];

			RandomInit(a);
			RandomInit(b);

			op.avx(a, b, avx_c);
			op.sisd(a, b, sisd_c);

			std::cout << Class::CLASS_NAME << ":" << op.name << "\n";
			Debug("a = ", a);
			Debug("b = ", b);
			Debug("avx_c = ", avx_c);
			Debug("sisd_c = ", sisd_c);

			assert(CmpResult(avx_c, sisd_c));
		} else if constexpr (Class::INPUT_ARGS == 3) {
			typename Class::ARG1_TYPE a[Class::INPUT_SIZE];
			typename Class::ARG2_TYPE b[Class::INPUT_SIZE];
			typename Class::ARG3_TYPE c[Class::INPUT_SIZE];
			typename Class::OUTPUT_TYPE avx_d[Class::INPUT_SIZE];
			typename Class::OUTPUT_TYPE sisd_d[Class::INPUT_SIZE];

			RandomInit(a);
			RandomInit(b);
			if constexpr (has_arg3_init<Class>::value) {
				Class::arg3_init(c);
			} else {
				RandomInit(c);
			}

			op.avx(a, b, c, avx_d);
			op.sisd(a, b, c, sisd_d);

			std::cout << Class::CLASS_NAME << ":" << op.name << "\n";
			Debug("a = ", a);
			Debug("b = ", b);
			Debug("c = ", c);
			Debug("avx_d = ", avx_d);
			Debug("sisd_d = ", sisd_d);

			assert(CmpResult(avx_d, sisd_d));

		}
	}
}

int main()
{
	RandomTest<AVX<int>>();
	RandomTest<AVX<float>>();
	RandomTest<AVX<double>>();
	RandomTest<AVX2<int>>();
	RandomTest<AVX2<float>>();
	RandomTest<AVX2_FMA<float>>();
	RandomTest<AVX2_FMA<double>>();
	RandomTest<AVX2_CMP<int>>();
	RandomTest<AVX2_CMP<float>>();
	RandomTest<AVX2_CMP<double>>();
	RandomTest<AVX2_BITWISE<int>>();
	RandomTest<AVX2_BITWISE<float>>();
	RandomTest<AVX2_BITWISE<double>>();
	RandomTest<AVX2_SHIFT<int>>();
	RandomTest<AVX2_BLEND<int>>();
	RandomTest<AVX2_BLEND<float>>();
	RandomTest<AVX2_BLEND<double>>();
	RandomTest<AVX2_MINMAX<int>>();
	RandomTest<AVX2_MINMAX<float>>();
	RandomTest<AVX2_MINMAX<double>>();
}
