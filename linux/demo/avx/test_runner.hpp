#pragma once

#include "test_framework.hpp"
#include <random>
#include <chrono>
#include <array>
#include <tuple>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>
#include <cstring>

// Forward declarations of helper functions used in RunTestImpl
template <typename Class, size_t I> struct ArgType;
template <typename Class> struct ArgType<Class, 0> { using type = typename Class::ARG1_TYPE; };
template <typename Class> struct ArgType<Class, 1> { using type = typename Class::ARG2_TYPE; };
template <typename Class> struct ArgType<Class, 2> { using type = typename Class::ARG3_TYPE; };

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

// ----------------------------------------------------------------------
// Data Initialization with Boundary Values
// ----------------------------------------------------------------------

template <typename T, size_t N>
void RandomInit(T (&a)[N]) {
    // Seed with time and random_device
    auto seed = static_cast<uint32_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()
    ) ^ std::random_device{}();

    std::mt19937 rng(seed);

    if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> dist(static_cast<T>(-10.0), static_cast<T>(10.0));
        
        // Fill most with random values
        for (size_t i = 0; i < N; ++i) {
            a[i] = dist(rng);
        }

        // Inject special values if array is large enough
        if (N >= 4) {
            a[0] = static_cast<T>(0.0);
            a[1] = static_cast<T>(-0.0);
            a[2] = std::numeric_limits<T>::infinity();
            a[3] = -std::numeric_limits<T>::infinity();
        }
        if (N >= 6) {
            a[4] = std::numeric_limits<T>::quiet_NaN();
            a[5] = std::numeric_limits<T>::min(); // smallest positive normal
        }
    } else if constexpr (std::is_integral_v<T>) {
        std::uniform_int_distribution<T> dist(static_cast<T>(-100), static_cast<T>(100));
        
        for (size_t i = 0; i < N; ++i) {
            a[i] = dist(rng);
        }

        // Inject special values
        if (N >= 4) {
            a[0] = 0;
            a[1] = 1;
            a[2] = -1;
            a[3] = std::numeric_limits<T>::max();
        }
        if (N >= 5) {
            a[4] = std::numeric_limits<T>::min();
        }
    } else {
        // Fallback or error
        std::memset(a, 0, sizeof(a));
    }
}

// ----------------------------------------------------------------------
// Comparison Logic
// ----------------------------------------------------------------------

template <typename T, size_t N>
bool CmpResult(T (&a)[N], T (&b)[N], size_t& first_failure_idx) {
    for (size_t i = 0; i < N; ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            // Check for NaN
            if (std::isnan(a[i]) && std::isnan(b[i])) continue;
            // Check for Infinity equality
            if (std::isinf(a[i]) && std::isinf(b[i]) && (a[i] > 0) == (b[i] > 0)) continue;
            
            // Allow small error for float/double
            if (std::fabs(a[i] - b[i]) > 1e-4) {
                first_failure_idx = i;
                return false;
            }
        } else {
            if (a[i] != b[i]) {
                first_failure_idx = i;
                return false;
            }
        }
    }
    return true;
}

// ----------------------------------------------------------------------
// Debug Output
// ----------------------------------------------------------------------

template <typename T, size_t N>
void Debug(const char *token, T (&a)[N]) {
    std::cout << token;
    for (size_t i = 0; i < N; ++i) {
        // Promote int8_t/uint8_t to int for printing
        if constexpr (sizeof(T) == 1) {
            std::cout << +a[i] << " ";
        } else {
            std::cout << a[i] << " ";
        }
    }
    std::cout << "\n";
}

template <typename T, size_t N>
void AsCArrayHelper(std::array<T, N>& arr, T*& out_ptr) {
    out_ptr = arr.data();
}

template <typename T, size_t N>
auto& AsCArray(std::array<T, N>& arr) {
    return *reinterpret_cast<T(*)[N]>(arr.data());
}

// ----------------------------------------------------------------------
// Helper Traits for Arguments
// ----------------------------------------------------------------------

template <typename Class, typename = void>
struct has_arg3_init : std::false_type {};
template <typename Class>
struct has_arg3_init<Class, std::void_t<decltype(Class::arg3_init(std::declval<typename Class::ARG3_TYPE(&)[Class::INPUT_SIZE]>()))>> : std::true_type {};

template <typename Class, typename = void>
struct has_arg2_init : std::false_type {};
template <typename Class>
struct has_arg2_init<Class, std::void_t<decltype(Class::arg2_init(std::declval<typename Class::ARG2_TYPE(&)[Class::INPUT_SIZE]>()))>> : std::true_type {};

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

// ----------------------------------------------------------------------
// Test Runner Implementation
// ----------------------------------------------------------------------

template <typename Class, typename Op, size_t... Is>
void RunTestImpl(const Op& op, const TestConfig& config, std::index_sequence<Is...>) {
    std::string category = Class::CATEGORY;
    std::string class_type = Class::CLASS_TYPE;
    std::string op_name = op.name;
    std::string type_sig = GetOpSignature<Class>(op.name, std::index_sequence<Is...>{});

    // 1. Filter Check
    if (!config.category_filter.empty() && category != config.category_filter) return;
    if (!config.class_filter.empty() && class_type != config.class_filter) return;
    if (!config.op_filter.empty() && op_name.find(config.op_filter) == std::string::npos) return;
    if (!config.type_filter.empty() && type_sig.find(config.type_filter) == std::string::npos) return;

    // 2. Prepare Inputs
    std::tuple<std::array<typename ArgType<Class, Is>::type, GetArgSize<Class, Is>()>...> inputs;
    (InitArg<Class, Is>(std::get<Is>(inputs)), ...);

    // 3. Prepare Outputs
    typename Class::OUTPUT_TYPE avx_out[Class::OUTPUT_SIZE];
    typename Class::OUTPUT_TYPE sisd_out[Class::OUTPUT_SIZE];
    // Initialize output buffers to a known pattern (e.g. 0)
    std::memset(avx_out, 0, sizeof(avx_out));
    std::memset(sisd_out, 0, sizeof(sisd_out));

    // 4. Run Tests
    std::apply([&](auto&... args) {
        op.avx(args.data()..., avx_out);
        op.sisd(args.data()..., sisd_out);
    }, inputs);

    // 5. Verify Results
    size_t failure_idx = 0;
    bool pass = CmpResult(avx_out, sisd_out, failure_idx);

    if (!pass) {
        std::cout << Color::Red << "[FAIL] " << Color::Reset 
                  << category << ":" << class_type << ":" << op_name << type_sig << "\n";
        
        const char* arg_names[] = {"  Arg0 = ", "  Arg1 = ", "  Arg2 = "};
        std::apply([&](auto&... args) {
            size_t idx = 0;
            ((Debug(arg_names[idx++], AsCArray(args))), ...);
        }, inputs);

        Debug("  AVX  = ", avx_out);
        Debug("  SISD = ", sisd_out);
        std::cout << "  Diff at index " << failure_idx << "\n";
        std::cout << "--------------------------------------------------\n";
    } else {
        // Print success only if verbose or explicitly requested (optional)
        // For now, mimic original behavior: print inputs/outputs always?
        // Or maybe just print a summary line? 
        // The original code printed everything. Let's keep printing but make it look nicer.
        
        std::cout << Color::Green << "[PASS] " << Color::Reset 
                  << category << ":" << class_type << ":" << op_name << type_sig << "\n";

        // Print details only if config.verbose (TODO: add verbose flag) or keep original behavior
        // Original behavior was always print.
        const char* arg_names[] = {"  a = ", "  b = ", "  c = "};
        std::apply([&](auto&... args) {
            size_t idx = 0;
            ((Debug(arg_names[idx++], AsCArray(args))), ...);
        }, inputs);

        char out_name[] = "  avx_? = ";
        out_name[6] = 'a' + sizeof...(Is) + 1; // 'b', 'c', 'd'
        Debug(out_name, avx_out);
        
        // Debug("  sisd = ", sisd_out); // SISD output is redundant if they match
    }
}

// Wrapper class to auto-register tests
template <typename Class>
class TestWrapper : public ITest {
public:
    void Run(const TestConfig& config) override {
        constexpr auto ops = Class::make_ops();
        for (const auto& op : ops) {
            RunTestImpl<Class>(op, config, std::make_index_sequence<Class::INPUT_ARGS>{});
        }
    }

    std::string GetCategory() const override { return Class::CATEGORY; }
    std::string GetClassType() const override { return Class::CLASS_TYPE; }
    
    std::vector<std::string> GetOpNames() const override {
        std::vector<std::string> names;
        constexpr auto ops = Class::make_ops();
        for (const auto& op : ops) {
            names.push_back(op.name);
        }
        return names;
    }
};

// Macro to register a test class
#define REGISTER_TEST(Class) \
    static AutoRegister<TestWrapper<Class>> register_##Class;

