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

#include "test_framework.hpp"
#include "test_runner.hpp"

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
#include "avx2_gather.hpp"
#include "avx2_shuffle.hpp"
#include "avx2_pack.hpp"
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
#include "avx512_scatter_gather.hpp"
#include "avx512_ternary.hpp"
#include "avx512_conflict.hpp"
#include "avx512_math.hpp"
#include "avx512_popcnt.hpp"

// Auto-Registration of Tests is now handled in individual header files

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
        } else if (arg.find("--init=") == 0) {
            config.init_mode = arg.substr(7);
        } else if (arg == "--verbose") {
            config.verbose = true;
        }
    }

    if (config.help_mode) {
        TestRegistry::Instance().PrintHelp(config, argv[0]);
        return 0;
    }

    // Run all registered tests (they will internally filter based on config)
    for (const auto& test : TestRegistry::Instance().GetTests()) {
        test->Run(config);
    }

    return 0;
}
