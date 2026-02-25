#include <immintrin.h>
#include <type_traits>
#include <stdint.h>
#include <random>
#include <assert.h>
#include <iostream>
#include <array>
#include <chrono>
#include <tuple>
#include <utility>
#include <string>
#include <typeinfo>

template <typename T>
std::string GetTypeName() {
    if constexpr (std::is_same_v<T, int>) return "int";
    else if constexpr (std::is_same_v<T, float>) return "float";
    else if constexpr (std::is_same_v<T, double>) return "double";
    else if constexpr (std::is_same_v<T, int8_t>) return "int8_t";
    else if constexpr (std::is_same_v<T, uint8_t>) return "uint8_t";
    else if constexpr (std::is_same_v<T, int16_t>) return "int16_t";
    else if constexpr (std::is_same_v<T, uint16_t>) return "uint16_t";
    else if constexpr (std::is_same_v<T, int64_t>) return "int64_t";
    else if constexpr (std::is_same_v<T, uint64_t>) return "uint64_t";
    else return typeid(T).name();
}


template <typename ARG1_TYPE, typename OUT_TYPE>
struct OpEntry1 {
    const char* name;

    void (*avx)(const ARG1_TYPE*, OUT_TYPE*);
    void (*sisd)(const ARG1_TYPE*, OUT_TYPE*);
};

template <typename ARG1_TYPE, typename ARG2_TYPE, typename OUT_TYPE>
struct OpEntry2 {
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
#include "avx2_permut.hpp"
#include "avx512.hpp"
#include "avx512_cmp.hpp"
#include "avx512_bitwise.hpp"
#include "avx512_shift.hpp"
#include "avx512_blend.hpp"
#include "avx512_permute.hpp"
#include "avx512_reduce.hpp"
#include "avx512_mask.hpp"
#include "avx512_compress.hpp"
#include "avx512_convert.hpp"
#include "avx512_fma.hpp"
#include "avx512_vnni.hpp"

template <typename Class, typename = void>
struct has_arg3_init : std::false_type {};

template <typename Class>
struct has_arg3_init<Class, std::void_t<decltype(Class::arg3_init(std::declval<typename Class::ARG3_TYPE(&)[Class::INPUT_SIZE]>()))>> : std::true_type {};

template <typename Class, typename = void>
struct has_arg2_init : std::false_type {};

template <typename Class>
struct has_arg2_init<Class, std::void_t<decltype(Class::arg2_init(std::declval<typename Class::ARG2_TYPE(&)[Class::INPUT_SIZE]>()))>> : std::true_type {};

template <typename Class, size_t I>
constexpr size_t GetArgSize() {
    if constexpr (I == 0) {
        return Class::ARG1_SIZE;
    } else if constexpr (I == 1) {
        return Class::ARG2_SIZE;
    } else if constexpr (I == 2) {
        return Class::ARG3_SIZE;
    }
}

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
		if constexpr (std::is_floating_point_v<T>) {
			if (std::fabs(a[i] - b[i]) > 1e5)
				return false;
		} else if (a[i] != b[i]) {
			return false;
		}
	}

	return true;
}

template <typename T, size_t N>
void Debug(const char *token, T (&a)[N])
{
	std::cout << token;
	for (size_t i = 0; i < N; ++i) {
		// use +a[i] to make a[i] become int type if it is int8_t
		std::cout << +a[i] << " ";
	}

	std::cout << "\n";
}

template <typename Class, size_t I> struct ArgType;
template <typename Class> struct ArgType<Class, 0> { using type = typename Class::ARG1_TYPE; };
template <typename Class> struct ArgType<Class, 1> { using type = typename Class::ARG2_TYPE; };
template <typename Class> struct ArgType<Class, 2> { using type = typename Class::ARG3_TYPE; };

template <typename Class, size_t... Is>
void PrintOpSignatureHelper(const char* op_name, std::index_sequence<Is...>) {
    std::cout << "<";
    ((std::cout << (Is == 0 ? "" : ", ") 
                << GetTypeName<typename ArgType<Class, Is>::type>() 
                << "[" << GetArgSize<Class, Is>() << "]"), ...);
    std::cout << " -> " << GetTypeName<typename Class::OUTPUT_TYPE>() 
              << "[" << Class::OUTPUT_SIZE << "]>";
}

template <typename T, size_t N>
auto& AsCArray(std::array<T, N>& arr) {
    return *reinterpret_cast<T(*)[N]>(arr.data());
}

template <typename Class, size_t I, typename T, size_t N>
void InitArg(std::array<T, N>& arr) {
    auto& c_arr = AsCArray(arr);
    if constexpr (I == 0) {
        RandomInit(c_arr);
    } else if constexpr (I == 1) {
        if constexpr (has_arg2_init<Class>::value) {
            Class::arg2_init(c_arr);
        } else {
            RandomInit(c_arr);
        }
    } else if constexpr (I == 2) {
        if constexpr (has_arg3_init<Class>::value) {
            Class::arg3_init(c_arr);
        } else {
            RandomInit(c_arr);
        }
    }
}

template <typename Class, typename Op, size_t... Is>
void RunTestImpl(const Op& op, std::index_sequence<Is...>) {
    std::tuple<std::array<typename ArgType<Class, Is>::type, GetArgSize<Class, Is>()>...> inputs;

    (InitArg<Class, Is>(std::get<Is>(inputs)), ...);

    typename Class::OUTPUT_TYPE avx_out[Class::OUTPUT_SIZE];
    typename Class::OUTPUT_TYPE sisd_out[Class::OUTPUT_SIZE];

    std::apply([&](auto&... args) {
        op.avx(args.data()..., avx_out);
        op.sisd(args.data()..., sisd_out);
    }, inputs);

    std::cout << Class::CLASS_NAME << ":" << op.name;
    PrintOpSignatureHelper<Class>(op.name, std::make_index_sequence<Class::INPUT_ARGS>{});
    std::cout << "\n";
    
    const char* arg_names[] = {"a = ", "b = ", "c = "};
    std::apply([&](auto&... args) {
        size_t idx = 0;
        ((Debug(arg_names[idx++], AsCArray(args))), ...);
    }, inputs);

    char out_name[] = "avx_? = ";
    out_name[4] = 'a' + sizeof...(Is) + 1; // 'b', 'c', 'd'
    Debug(out_name, avx_out);
    
    out_name[0] = 's'; out_name[1] = 'i'; out_name[2] = 's'; out_name[3] = 'd';
    Debug(out_name, sisd_out);

    assert(CmpResult(avx_out, sisd_out));
}

template <typename Class>
void RandomTest()
{
	constexpr auto ops = Class::make_ops();
	for (const auto& op : ops) {
		RunTestImpl<Class>(op, std::make_index_sequence<Class::INPUT_ARGS>{});
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
	RandomTest<AVX2_PERMUTE::SHUFFLE_0123<int>>();
	RandomTest<AVX2_PERMUTE::UNPACKLO<int>>();
	RandomTest<AVX2_PERMUTE::UNPACKHI<int>>();
	RandomTest<AVX2_PERMUTE::SWAP_LANES<int>>();
	RandomTest<AVX2_PERMUTE::PERMUTEVAR<int>>();
	RandomTest<AVX512<int>>();
	RandomTest<AVX512<float>>();
	RandomTest<AVX512<double>>();
	RandomTest<AVX512_CMP<int>>();
	RandomTest<AVX512_CMP<float>>();
	RandomTest<AVX512_CMP<double>>();
    RandomTest<AVX512_BITWISE<int>>();
    RandomTest<AVX512_BITWISE<float>>();
    RandomTest<AVX512_BITWISE<double>>();
    RandomTest<AVX512_SHIFT<int>>();
    RandomTest<AVX512_BLEND<int>>();
    RandomTest<AVX512_BLEND<float>>();
    RandomTest<AVX512_BLEND<double>>();
    RandomTest<AVX512_PERMUTE<int>>();
    RandomTest<AVX512_PERMUTE<float>>();
    RandomTest<AVX512_PERMUTE<double>>();
    RandomTest<AVX512_CMP<int>>();
    RandomTest<AVX512_CMP<float>>();
    RandomTest<AVX512_CMP<double>>();
	RandomTest<AVX512_REDUCE<int>>();
	RandomTest<AVX512_REDUCE<float>>();
	RandomTest<AVX512_REDUCE<double>>();
	RandomTest<AVX512_MASK<int>>();
	RandomTest<AVX512_MASK<float>>();
	RandomTest<AVX512_MASK<double>>();
	RandomTest<AVX512_COMPRESS<int>>();
	RandomTest<AVX512_COMPRESS<float>>();
	RandomTest<AVX512_COMPRESS<double>>();
#if 0
	RandomTest<AVX512_CONVERT<int, float>>();
	RandomTest<AVX512_CONVERT<int, double>>();
	RandomTest<AVX512_CONVERT<float, int>>();
	RandomTest<AVX512_CONVERT<double, int>>();
#endif

	RandomTest<AVX512_FMA<float>>();
	RandomTest<AVX512_FMA<double>>();

	// INT8 for vnni in reference
	RandomTest<AVX512_VNNI<uint8_t, int8_t>>();
	RandomTest<AVX512_VNNI<uint8_t, int8_t, true>>();
	// INT16 for vnni in reference
	RandomTest<AVX512_VNNI<int16_t, int16_t>>();
	RandomTest<AVX512_VNNI<int16_t, int16_t, true>>();


}
