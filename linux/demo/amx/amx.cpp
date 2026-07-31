#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

// Educational Intel AMX example: memory-side tiles, B packing and TMUL GEMM.
// Linux/x86-64 only. This is intentionally a one-block teaching kernel, not
// a replacement for oneDNN/oneMKL.

#if !defined(__x86_64__)
#error "Intel AMX requires an x86-64 target."
#endif

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <immintrin.h>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <sys/syscall.h>
#include <type_traits>
#include <unistd.h>

#if defined(__linux__)
#include <asm/prctl.h>
#endif

#ifndef ARCH_REQ_XCOMP_PERM
// Old Linux UAPI headers may predate ARCH_REQ_XCOMP_PERM.
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif

#ifndef ARCH_XCOMP_TILEDATA
#define ARCH_XCOMP_TILEDATA 18
#endif

// --------------------------- Memory-side types ----------------------------

// BF16 and FP16 are stored as raw 16-bit payloads. TMUL interprets their
// meaning; TILELOADD itself only moves bytes.
struct bf16 {
    std::uint16_t bits{};
};

struct fp16 {
    std::uint16_t bits{};
};

static_assert(sizeof(bf16) == 2);
static_assert(sizeof(fp16) == 2);

inline bf16 bf16_one() { return {0x3f80}; }  // 1.0 in BF16
inline fp16 fp16_one() { return {0x3c00}; }  // 1.0 in IEEE FP16

// A zero-copy, non-owning 2-D memory view. RowStrideBytes may be larger than
// Cols*sizeof(T), so submatrices and padded/leading-dimension matrices work.
template <class T, std::size_t Rows, std::size_t Cols>
class MatrixView {
public:
    using value_type = T;
    static constexpr std::size_t rows = Rows;
    static constexpr std::size_t cols = Cols;

    constexpr MatrixView(T* base, std::ptrdiff_t row_stride_bytes)
        : base_(base), row_stride_bytes_(row_stride_bytes) {
        if (base == nullptr || row_stride_bytes <
                                   static_cast<std::ptrdiff_t>(Cols * sizeof(T)))
            throw std::invalid_argument("invalid matrix view");
    }

    constexpr T& operator()(std::size_t r, std::size_t c) const {
        auto* row = reinterpret_cast<std::byte*>(const_cast<std::remove_const_t<T>*>(base_))
                    + r * row_stride_bytes_;
        return reinterpret_cast<T*>(row)[c];
    }
    constexpr T* data() const { return base_; }
    constexpr std::ptrdiff_t stride_bytes() const { return row_stride_bytes_; }

    constexpr MatrixView<const std::remove_const_t<T>, Rows, Cols>
    as_const() const {
        return {base_, row_stride_bytes_};
    }

private:
    T* base_;
    std::ptrdiff_t row_stride_bytes_;
};

// Deduction helpers: no copy is performed.
template <class T, std::size_t Rows, std::size_t Cols>
constexpr auto matrix_view(T (&a)[Rows][Cols]) {
    return MatrixView<T, Rows, Cols>{&a[0][0], sizeof(a[0])};
}
template <class T, std::size_t Rows, std::size_t Cols>
constexpr auto matrix_view(std::array<T, Rows * Cols>& a) {
    return MatrixView<T, Rows, Cols>{a.data(), Cols * sizeof(T)};
}

template <class T, std::size_t Rows, std::size_t Cols>
constexpr auto matrix_view(const std::array<T, Rows * Cols>& a) {
    return MatrixView<const T, Rows, Cols>{a.data(), Cols * sizeof(T)};
}

// Generic pointer adapter for vector.data(), aligned_alloc(), mmap(), a
// framework tensor, or a submatrix. Lifetime stays with the caller.
template <class T, std::size_t Rows, std::size_t Cols>
constexpr auto matrix_view(T* base, std::ptrdiff_t row_stride_bytes) {
    return MatrixView<T, Rows, Cols>{base, row_stride_bytes};
}

// A MatrixView whose shape is legal for one AMX tile. This view is the object
// directly consumed by TILELOADD/TILESTORED.
template <class T, std::size_t Rows, std::size_t Cols>
class AMX_TileView : public MatrixView<T, Rows, Cols> {
public:
    using Base = MatrixView<T, Rows, Cols>;
    using value_type = T;
    static constexpr std::size_t rows = Rows;
    static constexpr std::size_t cols = Cols;
    static constexpr std::size_t col_bytes = Cols * sizeof(T);

    static_assert(std::is_trivially_copyable_v<std::remove_const_t<T>>);
    static_assert(Rows >= 1 && Rows <= 16);
    static_assert(col_bytes >= 1 && col_bytes <= 64);

    using Base::Base;

    template <int Reg>
    void load() const {
        static_assert(Reg >= 0 && Reg < 8);
        _tile_loadd(Reg, this->data(), this->stride_bytes());
    }

    template <int Reg>
    void load_t1() const {
        static_assert(Reg >= 0 && Reg < 8);
        _tile_stream_loadd(Reg, this->data(), this->stride_bytes());
    }

    template <int Reg>
    void store() const requires (!std::is_const_v<T>) {
        static_assert(Reg >= 0 && Reg < 8);
        _tile_stored(Reg, this->data(), this->stride_bytes());
    }

    constexpr auto as_const() const {
        return AMX_TileView<const std::remove_const_t<T>, Rows, Cols>{
            this->data(), this->stride_bytes()};
    }
};

template <class T, std::size_t Rows, std::size_t Cols>
constexpr auto tile_view(T (&a)[Rows][Cols]) {
    return AMX_TileView<T, Rows, Cols>{&a[0][0], sizeof(a[0])};
}

template <class T, std::size_t Rows, std::size_t Cols>
constexpr auto tile_view(std::array<T, Rows * Cols>& a) {
    return AMX_TileView<T, Rows, Cols>{a.data(), Cols * sizeof(T)};
}

template <class T, std::size_t Rows, std::size_t Cols>
constexpr auto tile_view(T* base, std::ptrdiff_t row_stride_bytes) {
    return AMX_TileView<T, Rows, Cols>{base, row_stride_bytes};
}

// Optional owning storage. Algorithms consume view(), not this concrete type,
// so ownership never leaks into the compute API.
template <class T, std::size_t Rows, std::size_t Cols>
class alignas(64) AMX_Tile {
public:
    using value_type = T;
    static constexpr std::size_t rows = Rows;
    static constexpr std::size_t cols = Cols;
    static constexpr std::size_t col_bytes = Cols * sizeof(T);
    static constexpr std::size_t stride_bytes = col_bytes;

    static_assert(std::is_trivially_copyable_v<T>);
    static_assert(Rows >= 1 && Rows <= 16, "an AMX tile has at most 16 rows");
    static_assert(col_bytes >= 1 && col_bytes <= 64,
                  "an AMX tile row has at most 64 bytes");

    T& operator()(std::size_t r, std::size_t c) {
        return storage_[r * Cols + c];
    }
    const T& operator()(std::size_t r, std::size_t c) const {
        return storage_[r * Cols + c];
    }
    T* data() { return storage_.data(); }
    const T* data() const { return storage_.data(); }
    void fill(T value) { storage_.fill(value); }

    auto view() {
        return AMX_TileView<T, Rows, Cols>{data(), stride_bytes};
    }
    auto view() const {
        return AMX_TileView<const T, Rows, Cols>{data(), stride_bytes};
    }

private:
    std::array<T, Rows * Cols> storage_{};
};

// Exact 64-byte LDTILECFG memory image for palette 1.
struct alignas(64) TileConfig {
    std::uint8_t palette_id = 1;
    std::uint8_t start_row = 0;
    std::uint8_t reserved0[14]{};
    std::uint16_t colsb[8]{};
    std::uint16_t reserved1[8]{};
    std::uint8_t rows[8]{};
    std::uint8_t reserved2[8]{};

    template <int Reg, class Tile>
    void bind() {
        static_assert(Reg >= 0 && Reg < 8);
        using BareTile = std::remove_cvref_t<Tile>;
        rows[Reg] = static_cast<std::uint8_t>(BareTile::rows);
        colsb[Reg] = static_cast<std::uint16_t>(BareTile::col_bytes);
    }
};
static_assert(sizeof(TileConfig) == 64);

inline void request_amx_permission() {
#if !defined(__linux__)
    throw std::runtime_error("this example requests AMX state through Linux");
#else
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM,
                ARCH_XCOMP_TILEDATA) != 0) {
        throw std::runtime_error(
            "ARCH_REQ_XCOMP_PERM failed: kernel/VM may not expose AMX");
    }
#endif
}

class AMX_Session {
public:
    explicit AMX_Session(const TileConfig& cfg) { _tile_loadconfig(&cfg); }
    AMX_Session(const AMX_Session&) = delete;
    AMX_Session& operator=(const AMX_Session&) = delete;
    ~AMX_Session() { _tile_release(); }
};

// ------------------------------- B packing --------------------------------

// Mathematical B is row-major B[K][N].
// Physical Bpack has [K/Group] rows; in each row, the Group K-values for the
// same output column n are adjacent:
//   Bpack[p][n*Group+g] = B[p*Group+g][n].
template <class T, std::size_t K, std::size_t N, std::size_t Group>
AMX_Tile<T, K / Group, N * Group>
pack_b(MatrixView<const T, K, N> b) {
    static_assert(K % Group == 0);
    AMX_Tile<T, K / Group, N * Group> packed;
    for (std::size_t p = 0; p < K / Group; ++p)
        for (std::size_t n = 0; n < N; ++n)
            for (std::size_t g = 0; g < Group; ++g)
                packed(p, n * Group + g) = b(p * Group + g, n);
    return packed;
}

// --------------------------- TMUL mode traits ------------------------------

struct BF16_F32 {
    using input = bf16;
    using accum = float;
    static constexpr std::size_t group = 2;
    static constexpr std::string_view name = "BF16 x BF16 -> FP32";
    static void tmul() { _tile_dpbf16ps(0, 1, 2); }
};

#if defined(AMX_ENABLE_FP16)
struct FP16_F32 {
    using input = fp16;
    using accum = float;
    static constexpr std::size_t group = 2;
    static constexpr std::string_view name = "FP16 x FP16 -> FP32";
    static void tmul() { _tile_dpfp16ps(0, 1, 2); }
};
#endif

struct S8_S8_S32 {
    using input = std::int8_t;
    using accum = std::int32_t;
    static constexpr std::size_t group = 4;
    static constexpr std::string_view name = "S8 x S8 -> S32";
    static void tmul() { _tile_dpbssd(0, 1, 2); }
};

struct S8_U8_S32 {
    using input = std::int8_t;
    using accum = std::int32_t;
    static constexpr std::size_t group = 4;
    static constexpr std::string_view name = "S8 x U8 -> S32";
    static void tmul() { _tile_dpbsud(0, 1, 2); }
};

struct U8_S8_S32 {
    using input = std::uint8_t;
    using b_input = std::int8_t;
    using accum = std::int32_t;
    static constexpr std::size_t group = 4;
    static constexpr std::string_view name = "U8 x S8 -> S32";
    static void tmul() { _tile_dpbusd(0, 1, 2); }
};

struct U8_U8_S32 {
    using input = std::uint8_t;
    using accum = std::int32_t;
    static constexpr std::size_t group = 4;
    static constexpr std::string_view name = "U8 x U8 -> S32";
    static void tmul() { _tile_dpbuud(0, 1, 2); }
};

template <class Mode, class = void>
struct mode_b {
    using type = typename Mode::input;
};

template <class Mode>
struct mode_b<Mode, std::void_t<typename Mode::b_input>> {
    using type = typename Mode::b_input;
};

template <class Mode>
using mode_b_t = typename mode_b<Mode>::type;

// ------------------------- One-block matrix multiply ----------------------

// Maximum-width teaching block:
//   BF16/FP16: A[16][32] * B[32][16] -> C[16][16]
//   INT8:      A[16][64] * B[64][16] -> C[16][16]
//
// Hot path. It owns no matrix memory and performs no allocation or packing.
// Register contract: tmm0=C accumulator, tmm1=A, tmm2=prepacked B.
template <class Mode>
void amx_gemm_block_packed(
    AMX_TileView<const typename Mode::input, 16,
                 64 / sizeof(typename Mode::input)> a,
    AMX_TileView<const mode_b_t<Mode>,
                 (64 / sizeof(typename Mode::input)) / Mode::group,
                 16 * Mode::group> bpack,
    AMX_TileView<typename Mode::accum, 16, 16> c) {
    using AType = typename Mode::input;
    using BType = mode_b_t<Mode>;

    static_assert(sizeof(AType) == sizeof(BType));

    TileConfig cfg;
    cfg.template bind<0, decltype(c)>();
    cfg.template bind<1, decltype(a)>();
    cfg.template bind<2, decltype(bpack)>();

    AMX_Session session(cfg);
    _tile_zero(0);
    a.template load<1>();
    bpack.template load<2>();
    Mode::tmul();
    c.template store<0>();
}

// Convenience layer: accepts ordinary row-major/strided B, packs it once for
// this call, and returns owning C storage. Latency-sensitive callers should
// call pack_b() outside their loop and use amx_gemm_block_packed().
template <class Mode>
auto amx_gemm_block(
    AMX_TileView<const typename Mode::input, 16,
                 64 / sizeof(typename Mode::input)> a,
    MatrixView<const mode_b_t<Mode>,
               64 / sizeof(typename Mode::input), 16> b) {
    using AType = typename Mode::input;
    using BType = mode_b_t<Mode>;
    using CType = typename Mode::accum;
    constexpr std::size_t K = 64 / sizeof(AType);
    constexpr std::size_t G = Mode::group;

    auto bpack = pack_b<BType, K, 16, G>(b);
    AMX_Tile<CType, 16, 16> c;
    amx_gemm_block_packed<Mode>(a, bpack.view().as_const(), c.view());
    return c;
}

template <class T>
T one();
template <> inline std::int8_t one<std::int8_t>() { return 1; }
template <> inline std::uint8_t one<std::uint8_t>() { return 1; }
template <> inline bf16 one<bf16>() { return bf16_one(); }
template <> inline fp16 one<fp16>() { return fp16_one(); }

template <class Mode>
void run_demo() {
    using AType = typename Mode::input;
    using BType = mode_b_t<Mode>;
    constexpr std::size_t K = 64 / sizeof(AType);

    // Deliberately use native C++ arrays to demonstrate zero-copy adapters.
    AType a[16][K];
    BType b[K][16];
    typename Mode::accum c[16][16]{};
    for (auto& row : a)
        for (auto& x : row) x = one<AType>();
    for (auto& row : b)
        for (auto& x : row) x = one<BType>();

    auto av = tile_view(a).as_const();
    auto bv = matrix_view(b).as_const();

    // Pack B outside the compute call. The packed object can be cached and
    // reused for every A block that multiplies the same weights.
    auto bpack = pack_b<BType, K, 16, Mode::group>(bv);
    amx_gemm_block_packed<Mode>(av, bpack.view().as_const(), tile_view(c));

    std::cout << std::left << std::setw(24) << Mode::name
              << " C[0][0]=" << c[0][0]
              << ", expected=" << K << '\n';
}

int main() {
    try {
        request_amx_permission();

        // AMX-TILE management exercised by every mode:
        // LDTILECFG, TILEZERO, TILELOADD, TILESTORED and TILERELEASE.
        run_demo<BF16_F32>();
        run_demo<S8_S8_S32>();
        run_demo<S8_U8_S32>();
        run_demo<U8_S8_S32>();
        run_demo<U8_U8_S32>();

#if defined(AMX_ENABLE_FP16)
        run_demo<FP16_F32>();
#else
        std::cout << "FP16 demo not built; add -DAMX_ENABLE_FP16 "
                     "-mamx-fp16 on supported compiler/hardware.\n";
#endif
    } catch (const std::exception& e) {
        std::cerr << "AMX demo failed: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
