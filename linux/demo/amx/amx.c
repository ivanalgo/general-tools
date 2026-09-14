#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

// Intel AMX C demo: zero-copy tile views, B packing and TMUL GEMM.
// Linux/x86-64 only. The examples compute one maximum-size tile block.
#if !defined(__x86_64__)
#error "Intel AMX requires an x86-64 target."
#endif

#include <errno.h>
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#if defined(__linux__)
#include <asm/prctl.h>
#endif

#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif

#ifndef ARCH_XCOMP_TILEDATA
#define ARCH_XCOMP_TILEDATA 18
#endif

typedef uint16_t bf16;
typedef uint16_t fp16;

typedef struct {
    const void *base;
    ptrdiff_t stride_bytes;
    uint8_t rows;
    uint16_t colsb;
} amx_const_tile_view;

typedef struct {
    void *base;
    ptrdiff_t stride_bytes;
    uint8_t rows;
    uint16_t colsb;
} amx_tile_view;

#define AMX_CONST_TILE_VIEW(array)                                          \
    ((amx_const_tile_view){                                                 \
        &(array)[0][0], (ptrdiff_t)sizeof((array)[0]),                      \
        (uint8_t)(sizeof(array) / sizeof((array)[0])),                      \
        (uint16_t)sizeof((array)[0])})

#define AMX_TILE_VIEW(array)                                                \
    ((amx_tile_view){                                                       \
        &(array)[0][0], (ptrdiff_t)sizeof((array)[0]),                      \
        (uint8_t)(sizeof(array) / sizeof((array)[0])),                      \
        (uint16_t)sizeof((array)[0])})

// Pointer adapter for vector-like/external/tensor memory. This performs no
// allocation or copy; the caller owns the pointed-to storage and its lifetime.
static amx_const_tile_view amx_const_view(
    const void *base, ptrdiff_t stride_bytes, uint8_t rows, uint16_t colsb) {
    amx_const_tile_view v = {base, stride_bytes, rows, colsb};
    return v;
}

static amx_tile_view amx_view(
    void *base, ptrdiff_t stride_bytes, uint8_t rows, uint16_t colsb) {
    amx_tile_view v = {base, stride_bytes, rows, colsb};
    return v;
}

typedef struct __attribute__((packed, aligned(64))) {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved0[14];
    uint16_t colsb[8];
    uint16_t reserved1[8];
    uint8_t rows[8];
    uint8_t reserved2[8];
} amx_tile_config;

_Static_assert(sizeof(amx_tile_config) == 64, "TILECFG must be 64 bytes");

static int request_amx_permission(void) {
#if !defined(__linux__)
    fputs("This demo requests AMX state through Linux.\n", stderr);
    return -1;
#else
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM,
                ARCH_XCOMP_TILEDATA) != 0) {
        fprintf(stderr, "ARCH_REQ_XCOMP_PERM: %s\n", strerror(errno));
        return -1;
    }
    return 0;
#endif
}

static int valid_tile(amx_const_tile_view v) {
    return v.base != NULL && v.rows >= 1 && v.rows <= 16 &&
           v.colsb >= 1 && v.colsb <= 64 &&
           v.stride_bytes >= (ptrdiff_t)v.colsb;
}

static int valid_output(amx_tile_view v) {
    return valid_tile(amx_const_view(
        v.base, v.stride_bytes, v.rows, v.colsb));
}

static void bind_tile(amx_tile_config *cfg, int reg,
                      uint8_t rows, uint16_t colsb) {
    cfg->rows[reg] = rows;
    cfg->colsb[reg] = colsb;
}

// ------------------------------- BF16 -------------------------------------

enum { BF16_M = 16, BF16_K = 32, BF16_N = 16 };

// B is row-major B[32][16]. Bpack is [16][32], where every adjacent pair is
// {B[2p][n], B[2p+1][n]}. b_stride allows padded input matrices.
static void pack_b_bf16(
    bf16 bpack[BF16_K / 2][BF16_N * 2],
    const bf16 *b, ptrdiff_t b_stride) {
    for (int p = 0; p < BF16_K / 2; ++p)
        for (int n = 0; n < BF16_N; ++n) {
            const bf16 *row0 =
                (const bf16 *)((const char *)b + (2 * p) * b_stride);
            const bf16 *row1 =
                (const bf16 *)((const char *)b + (2 * p + 1) * b_stride);
            bpack[p][2 * n] = row0[n];
            bpack[p][2 * n + 1] = row1[n];
        }
}

static int amx_gemm_bf16_packed(
    amx_const_tile_view a,
    amx_const_tile_view bpack,
    amx_tile_view c) {
    if (!valid_tile(a) || !valid_tile(bpack) || !valid_output(c) ||
        a.rows != BF16_M || a.colsb != BF16_K * sizeof(bf16) ||
        bpack.rows != BF16_K / 2 ||
        bpack.colsb != BF16_N * 2 * sizeof(bf16) ||
        c.rows != BF16_M || c.colsb != BF16_N * sizeof(float))
        return -1;

    amx_tile_config cfg = {0};
    cfg.palette_id = 1;
    bind_tile(&cfg, 0, c.rows, c.colsb);
    bind_tile(&cfg, 1, a.rows, a.colsb);
    bind_tile(&cfg, 2, bpack.rows, bpack.colsb);

    _tile_loadconfig(&cfg);
    _tile_zero(0);
    _tile_loadd(1, a.base, a.stride_bytes);
    _tile_loadd(2, bpack.base, bpack.stride_bytes);
    _tile_dpbf16ps(0, 1, 2);
    _tile_stored(0, c.base, c.stride_bytes);
    _tile_release();
    return 0;
}

// ------------------------------- INT8 -------------------------------------

enum { INT8_M = 16, INT8_K = 64, INT8_N = 16 };

typedef enum {
    AMX_S8_S8,
    AMX_S8_U8,
    AMX_U8_S8,
    AMX_U8_U8
} amx_int8_mode;

// Signedness does not affect packing: all four INT8 instructions use the
// same byte layout, with four K bytes grouped for every output column.
static void pack_b_int8(
    uint8_t bpack[INT8_K / 4][INT8_N * 4],
    const void *b, ptrdiff_t b_stride) {
    for (int p = 0; p < INT8_K / 4; ++p)
        for (int n = 0; n < INT8_N; ++n)
            for (int g = 0; g < 4; ++g) {
                const uint8_t *row =
                    (const uint8_t *)b + (p * 4 + g) * b_stride;
                bpack[p][n * 4 + g] = row[n];
            }
}

static int amx_gemm_int8_packed(
    amx_int8_mode mode,
    amx_const_tile_view a,
    amx_const_tile_view bpack,
    amx_tile_view c) {
    if (!valid_tile(a) || !valid_tile(bpack) || !valid_output(c) ||
        a.rows != INT8_M || a.colsb != INT8_K ||
        bpack.rows != INT8_K / 4 || bpack.colsb != INT8_N * 4 ||
        c.rows != INT8_M || c.colsb != INT8_N * sizeof(int32_t))
        return -1;

    amx_tile_config cfg = {0};
    cfg.palette_id = 1;
    bind_tile(&cfg, 0, c.rows, c.colsb);
    bind_tile(&cfg, 1, a.rows, a.colsb);
    bind_tile(&cfg, 2, bpack.rows, bpack.colsb);

    _tile_loadconfig(&cfg);
    _tile_zero(0);
    _tile_loadd(1, a.base, a.stride_bytes);
    _tile_loadd(2, bpack.base, bpack.stride_bytes);
    switch (mode) {
    case AMX_S8_S8: _tile_dpbssd(0, 1, 2); break;
    case AMX_S8_U8: _tile_dpbsud(0, 1, 2); break;
    case AMX_U8_S8: _tile_dpbusd(0, 1, 2); break;
    case AMX_U8_U8: _tile_dpbuud(0, 1, 2); break;
    default:
        _tile_release();
        return -1;
    }
    _tile_stored(0, c.base, c.stride_bytes);
    _tile_release();
    return 0;
}

int main(void) {
    if (request_amx_permission() != 0)
        return EXIT_FAILURE;

    // Native C arrays are used directly as zero-copy A/C views.
    bf16 a_bf16[BF16_M][BF16_K] __attribute__((aligned(64)));
    bf16 b_bf16[BF16_K][BF16_N] __attribute__((aligned(64)));
    bf16 bp_bf16[BF16_K / 2][BF16_N * 2] __attribute__((aligned(64)));
    float c_bf16[BF16_M][BF16_N] __attribute__((aligned(64))) = {{0}};
    for (int m = 0; m < BF16_M; ++m)
        for (int k = 0; k < BF16_K; ++k)
            a_bf16[m][k] = 0x3f80;  // BF16 1.0
    for (int k = 0; k < BF16_K; ++k)
        for (int n = 0; n < BF16_N; ++n)
            b_bf16[k][n] = 0x3f80;
    pack_b_bf16(bp_bf16, &b_bf16[0][0], sizeof(b_bf16[0]));
    if (amx_gemm_bf16_packed(
            AMX_CONST_TILE_VIEW(a_bf16),
            AMX_CONST_TILE_VIEW(bp_bf16),
            amx_view(&c_bf16[0][0], sizeof(c_bf16[0]),
                     BF16_M, sizeof(c_bf16[0]))) != 0)
        return EXIT_FAILURE;
    printf("%-20s C[0][0]=%.1f, expected=%d\n",
           "BF16 x BF16", c_bf16[0][0], BF16_K);

    uint8_t a_i8[INT8_M][INT8_K] __attribute__((aligned(64)));
    uint8_t b_i8[INT8_K][INT8_N] __attribute__((aligned(64)));
    uint8_t bp_i8[INT8_K / 4][INT8_N * 4] __attribute__((aligned(64)));
    int32_t c_i8[INT8_M][INT8_N] __attribute__((aligned(64)));
    memset(a_i8, 1, sizeof(a_i8));
    memset(b_i8, 1, sizeof(b_i8));
    pack_b_int8(bp_i8, &b_i8[0][0], sizeof(b_i8[0]));

    static const char *names[] = {
        "S8 x S8", "S8 x U8", "U8 x S8", "U8 x U8"};
    for (int mode = AMX_S8_S8; mode <= AMX_U8_U8; ++mode) {
        memset(c_i8, 0, sizeof(c_i8));
        if (amx_gemm_int8_packed(
                (amx_int8_mode)mode,
                AMX_CONST_TILE_VIEW(a_i8),
                AMX_CONST_TILE_VIEW(bp_i8),
                AMX_TILE_VIEW(c_i8)) != 0)
            return EXIT_FAILURE;
        printf("%-20s C[0][0]=%d, expected=%d\n",
               names[mode], c_i8[0][0], INT8_K);
    }
    return EXIT_SUCCESS;
}
