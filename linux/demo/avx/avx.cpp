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
#include <vector>
#include <string_view>
#include <algorithm>
#include <map>
#include <set>

struct TestConfig {
    std::string category_filter; // "avx", "avx2", "avx512" or "" (all)
    std::string class_filter;    // match for CLASS_TYPE (e.g. "cmp", "bitwise")
    std::string op_filter;       // match for op name
    std::string type_filter;     // partial match for type signature
    bool help_mode = false;      // If true, collect info and print help
};

class HelpCollector {
public:
    static HelpCollector& Instance() {
        static HelpCollector instance;
        return instance;
    }

    void Add(const std::string& category, const std::string& class_type, const std::string& op_name) {
        categories.insert(category);
        category_classes[category].insert(class_type);
        class_ops[{category, class_type}].insert(op_name);
    }

    void PrintHelp(const TestConfig& config, const char* prog_name) {
        std::cout << "Usage: " << prog_name << " [options]\n\n";
        
        if (config.category_filter.empty()) {
            std::cout << "Available Categories (use --avx<version> --help to see details):\n";
            for (const auto& cat : categories) {
                std::cout << "  --avx" << (cat.substr(0, 3) == "avx" ? cat.substr(3) : cat) << "\n";
            }
        } else {
            std::string cat = config.category_filter;
            if (categories.find(cat) == categories.end()) {
                std::cout << "Error: Category '" << cat << "' not found.\n";
                return;
            }

            if (config.class_filter.empty()) {
                std::cout << "Available Classes in " << cat << " (use --class=<name> --help to see operations):\n";
                for (const auto& cls : category_classes[cat]) {
                    std::cout << "  --class=" << cls << "\n";
                }
            } else {
                std::string cls = config.class_filter;
                if (category_classes[cat].find(cls) == category_classes[cat].end()) {
                    std::cout << "Error: Class '" << cls << "' not found in category '" << cat << "'.\n";
                    return;
                }

                std::cout << "Available Operations in " << cat << "::" << cls << ":\n";
                for (const auto& op : class_ops[{cat, cls}]) {
                    std::cout << "  --function=" << op << "\n";
                }
            }
        }
        
        std::cout << "\nGeneral Options:\n";
        std::cout << "  --function=<name>      Filter by operation name (partial match)\n";
        std::cout << "  --type=<sig>     Filter by type signature (partial match)\n";
        std::cout << "  --help           Show this help message\n";
    }

private:
    std::set<std::string> categories;
    std::map<std::string, std::set<std::string>> category_classes;
    std::map<std::pair<std::string, std::string>, std::set<std::string>> class_ops;
};

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

template <typename Class, size_t... Is>
std::string GetOpSignature(const char* op_name, std::index_sequence<Is...>) {
    std::string sig = "<";
    auto append_arg = [&](auto i) {
        if (i != 0) sig += ", ";
        sig += GetTypeName<typename ArgType<Class, decltype(i)::value>::type>();
        sig += "[";
        sig += std::to_string(GetArgSize<Class, decltype(i)::value>());
        sig += "]";
    };
    (append_arg(std::integral_constant<size_t, Is>{}), ...);
    sig += " -> ";
    sig += GetTypeName<typename Class::OUTPUT_TYPE>();
    sig += "[";
    sig += std::to_string(Class::OUTPUT_SIZE);
    sig += "]>";
    return sig;
}

template <typename Class, typename Op, size_t... Is>
void RunTestImpl(const Op& op, const TestConfig& config, std::index_sequence<Is...>) {
    std::string category = Class::CATEGORY;
    std::string class_type = Class::CLASS_TYPE;
    std::string op_name = op.name;
    std::string type_sig = GetOpSignature<Class>(op.name, std::index_sequence<Is...>{});

    // Check filters
    if (!config.category_filter.empty()) {
        if (category != config.category_filter) return;
    }
    if (!config.class_filter.empty()) {
         if (class_type != config.class_filter) return;
    }
    if (!config.op_filter.empty()) {
        if (op_name.find(config.op_filter) == std::string::npos) return;
    }
    if (!config.type_filter.empty()) {
        if (type_sig.find(config.type_filter) == std::string::npos) return;
    }

    std::tuple<std::array<typename ArgType<Class, Is>::type, GetArgSize<Class, Is>()>...> inputs;

    (InitArg<Class, Is>(std::get<Is>(inputs)), ...);

    typename Class::OUTPUT_TYPE avx_out[Class::OUTPUT_SIZE];
    typename Class::OUTPUT_TYPE sisd_out[Class::OUTPUT_SIZE];

    std::apply([&](auto&... args) {
        op.avx(args.data()..., avx_out);
        op.sisd(args.data()..., sisd_out);
    }, inputs);

    std::cout << category << ":" << class_type << ":" << op_name << type_sig << "\n";
    
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
void RandomTest(const TestConfig& config)
{
	constexpr auto ops = Class::make_ops();
    
    if (config.help_mode) {
        for (const auto& op : ops) {
            HelpCollector::Instance().Add(Class::CATEGORY, Class::CLASS_TYPE, op.name);
        }
        return;
    }

	for (const auto& op : ops) {
		RunTestImpl<Class>(op, config, std::make_index_sequence<Class::INPUT_ARGS>{});
	}
}

void RunAllTests(const TestConfig& config)
{
	RandomTest<AVX<int>>(config);
	RandomTest<AVX<float>>(config);
	RandomTest<AVX<double>>(config);
	RandomTest<AVX2<int>>(config);
	RandomTest<AVX2<float>>(config);
	RandomTest<AVX2_FMA<float>>(config);
	RandomTest<AVX2_FMA<double>>(config);
	RandomTest<AVX2_CMP<int>>(config);
	RandomTest<AVX2_CMP<float>>(config);
	RandomTest<AVX2_CMP<double>>(config);
	RandomTest<AVX2_BITWISE<int>>(config);
	RandomTest<AVX2_BITWISE<float>>(config);
	RandomTest<AVX2_BITWISE<double>>(config);
	RandomTest<AVX2_SHIFT<int>>(config);
	RandomTest<AVX2_BLEND<int>>(config);
	RandomTest<AVX2_BLEND<float>>(config);
	RandomTest<AVX2_BLEND<double>>(config);
	RandomTest<AVX2_MINMAX<int>>(config);
	RandomTest<AVX2_MINMAX<float>>(config);
	RandomTest<AVX2_MINMAX<double>>(config);
	RandomTest<AVX2_PERMUTE::SHUFFLE_0123<int>>(config);
	RandomTest<AVX2_PERMUTE::UNPACKLO<int>>(config);
	RandomTest<AVX2_PERMUTE::UNPACKHI<int>>(config);
	RandomTest<AVX2_PERMUTE::SWAP_LANES<int>>(config);
	RandomTest<AVX2_PERMUTE::PERMUTEVAR<int>>(config);
	RandomTest<AVX512<int>>(config);
	RandomTest<AVX512<float>>(config);
	RandomTest<AVX512<double>>(config);
	RandomTest<AVX512_CMP<int>>(config);
	RandomTest<AVX512_CMP<float>>(config);
	RandomTest<AVX512_CMP<double>>(config);
    RandomTest<AVX512_BITWISE<int>>(config);
    RandomTest<AVX512_BITWISE<float>>(config);
    RandomTest<AVX512_BITWISE<double>>(config);
    RandomTest<AVX512_SHIFT<int>>(config);
    RandomTest<AVX512_BLEND<int>>(config);
    RandomTest<AVX512_BLEND<float>>(config);
    RandomTest<AVX512_BLEND<double>>(config);
    RandomTest<AVX512_PERMUTE<int>>(config);
    RandomTest<AVX512_PERMUTE<float>>(config);
    RandomTest<AVX512_PERMUTE<double>>(config);
    RandomTest<AVX512_CMP<int>>(config);
    RandomTest<AVX512_CMP<float>>(config);
    RandomTest<AVX512_CMP<double>>(config);
	RandomTest<AVX512_REDUCE<int>>(config);
	RandomTest<AVX512_REDUCE<float>>(config);
	RandomTest<AVX512_REDUCE<double>>(config);
	RandomTest<AVX512_MASK<int>>(config);
	RandomTest<AVX512_MASK<float>>(config);
	RandomTest<AVX512_MASK<double>>(config);
	RandomTest<AVX512_COMPRESS<int>>(config);
	RandomTest<AVX512_COMPRESS<float>>(config);
	RandomTest<AVX512_COMPRESS<double>>(config);
#if 0
	RandomTest<AVX512_CONVERT<int, float>>(config);
	RandomTest<AVX512_CONVERT<int, double>>(config);
	RandomTest<AVX512_CONVERT<float, int>>(config);
	RandomTest<AVX512_CONVERT<double, int>>(config);
#endif

	RandomTest<AVX512_FMA<float>>(config);
	RandomTest<AVX512_FMA<double>>(config);

	// INT8 for vnni in reference
	RandomTest<AVX512_VNNI<uint8_t, int8_t>>(config);
	RandomTest<AVX512_VNNI<uint8_t, int8_t, true>>(config);
	// INT16 for vnni in reference
	RandomTest<AVX512_VNNI<int16_t, int16_t>>(config);
	RandomTest<AVX512_VNNI<int16_t, int16_t, true>>(config);
}

int main(int argc, char** argv)
{
    TestConfig config;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            config.help_mode = true;
        } else if (arg.find("--avx") == 0) {
            config.category_filter = arg.substr(2); // "avx", "avx2", "avx512"
        } else if (arg.find("--class=") == 0) {
            config.class_filter = arg.substr(8);
        } else if (arg.find("--function=") == 0) {
            config.op_filter = arg.substr(11);
        } else if (arg.find("--type=") == 0) {
            config.type_filter = arg.substr(7);
        }
    }

    if (config.help_mode) {
        RunAllTests(config); // Collect info
        HelpCollector::Instance().PrintHelp(config, argv[0]);
        return 0;
    }

    RunAllTests(config);
    return 0;
}
