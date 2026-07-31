# Intel AMX C/C++ examples

This directory demonstrates the complete data path of one AMX GEMM block:

    row-major A/B memory
            |
            +-- pack B into BF16 pairs or INT8 groups
            |
            +-- LDTILECFG / TILELOADD
            |
            +-- TDPBF16PS or INT8 TMUL
            |
            +-- TILESTORED -> row-major C

Both examples use non-owning pointer/stride views. A C array, std::array,
vector.data(), aligned_alloc, mmap, a tensor buffer, or a submatrix can
therefore be loaded without copying. B packing is a real layout conversion and
should be cached when the same weights are reused.

## Build and run

Requirements:

- Linux on x86-64.
- A compiler with AMX intrinsic support (GCC 12+ is recommended).
- A processor/VM exposing AMX-TILE, AMX-BF16 and AMX-INT8.
- A Linux kernel supporting ARCH_REQ_XCOMP_PERM for XTILEDATA.

    make
    make run

Optional AMX-FP16 C++ build:

    make fp16
    ./amx_cpp_fp16

The inputs are all ones. Correct output is:

- BF16/FP16: C[0][0] = 32.
- Every INT8 signedness combination: C[0][0] = 64.

These are educational one-block kernels. A production GEMM also needs
M/N/K blocking, remainder handling, CPU feature dispatch, threading, cache
blocking and NUMA placement.
