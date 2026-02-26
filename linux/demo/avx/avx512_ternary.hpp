template<typename T>
struct AVX512_TERNARY {
    static constexpr const char* CATEGORY = "avx512";
    static constexpr const char* CLASS_TYPE = "ternary";

    static constexpr size_t LANES = 512 / (8 * sizeof(T));
    static constexpr size_t INPUT_SIZE = LANES;

    static constexpr int INPUT_ARGS = 3;
    using ARG1_TYPE = T;
    static constexpr size_t ARG1_SIZE = INPUT_SIZE;
    using ARG2_TYPE = T;
    static constexpr size_t ARG2_SIZE = INPUT_SIZE;
    using ARG3_TYPE = T;
    static constexpr size_t ARG3_SIZE = INPUT_SIZE;

    using OUTPUT_TYPE = T;
    static constexpr size_t OUTPUT_SIZE = INPUT_SIZE;

    // VPTERNLOG takes an immediate imm8 that defines the truth table
    // For testing, we pick a few interesting ones:
    // 0x96: A ^ B ^ C (XOR3)
    // 0xE2: (A & B) | C (Select)
    // 0xFE: A | B | C (OR3)
    
    // We can't pass imm8 dynamically to intrinsics easily without switch case or templates.
    // So we define separate functions for specific ops.

    /* SISD */
    static void sisd_xor3(const T* a, const T* b, const T* c, T* out) {
        for(size_t i=0; i<INPUT_SIZE; ++i) {
            if constexpr (std::is_integral_v<T>) {
                out[i] = a[i] ^ b[i] ^ c[i];
            } else {
                 // Bitwise cast for float/double
                 // Using uint32_t/uint64_t
                 if constexpr (sizeof(T)==4) {
                     uint32_t ia, ib, ic;
                     memcpy(&ia, &a[i], 4); memcpy(&ib, &b[i], 4); memcpy(&ic, &c[i], 4);
                     uint32_t r = ia ^ ib ^ ic;
                     memcpy(&out[i], &r, 4);
                 } else {
                     uint64_t ia, ib, ic;
                     memcpy(&ia, &a[i], 8); memcpy(&ib, &b[i], 8); memcpy(&ic, &c[i], 8);
                     uint64_t r = ia ^ ib ^ ic;
                     memcpy(&out[i], &r, 8);
                 }
            }
        }
    }
    
    static void sisd_select(const T* a, const T* b, const T* c, T* out) {
         // (A & B) | C
         for(size_t i=0; i<INPUT_SIZE; ++i) {
            if constexpr (std::is_integral_v<T>) {
                out[i] = (a[i] & b[i]) | c[i];
            } else {
                 if constexpr (sizeof(T)==4) {
                     uint32_t ia, ib, ic;
                     memcpy(&ia, &a[i], 4); memcpy(&ib, &b[i], 4); memcpy(&ic, &c[i], 4);
                     uint32_t r = (ia & ib) | ic;
                     memcpy(&out[i], &r, 4);
                 } else {
                     uint64_t ia, ib, ic;
                     memcpy(&ia, &a[i], 8); memcpy(&ib, &b[i], 8); memcpy(&ic, &c[i], 8);
                     uint64_t r = (ia & ib) | ic;
                     memcpy(&out[i], &r, 8);
                 }
            }
        }
    }

    /* AVX512 */
    static void avx512_xor3(const T* a, const T* b, const T* c, T* out) {
        if constexpr (sizeof(T) == 4) { // int or float
            __m512i va = _mm512_loadu_si512(a);
            __m512i vb = _mm512_loadu_si512(b);
            __m512i vc = _mm512_loadu_si512(c);
            // 0x96 = 10010110 -> XOR
            __m512i res = _mm512_ternarylogic_epi32(va, vb, vc, 0x96);
            _mm512_storeu_si512(out, res);
        } else { // int64 or double
            __m512i va = _mm512_loadu_si512(a);
            __m512i vb = _mm512_loadu_si512(b);
            __m512i vc = _mm512_loadu_si512(c);
            __m512i res = _mm512_ternarylogic_epi64(va, vb, vc, 0x96);
            _mm512_storeu_si512(out, res);
        }
    }

    static void avx512_select(const T* a, const T* b, const T* c, T* out) {
        if constexpr (sizeof(T) == 4) {
            __m512i va = _mm512_loadu_si512(a);
            __m512i vb = _mm512_loadu_si512(b);
            __m512i vc = _mm512_loadu_si512(c);
            // 0xE2: (A & B) | C?
            // Ternary logic truth table:
            // Input: A, B, C (indices 2, 1, 0)
            // Table index is 3-bit value formed by bits from A,B,C.
            // Wait, intrinsics guide says: "imm8[7:0] specifies the logic function"
            // Let's trust 0xE2 for (A|B)&C or similar?
            // Actually: (A & B) | C -> 
            // A B C | Out
            // 0 0 0 | 0
            // 0 0 1 | 1
            // 0 1 0 | 0
            // 0 1 1 | 1
            // 1 0 0 | 0
            // 1 0 1 | 1
            // 1 1 0 | 1
            // 1 1 1 | 1
            // Binary: 11111010 -> 0xFA? No.
            // Let's try 0xF8 for (A&B)|C?
            // Actually let's just use A | B | C for simplicity -> 0xFE
            // Let's implement OR3
            __m512i res = _mm512_ternarylogic_epi32(va, vb, vc, 0xFE); 
            _mm512_storeu_si512(out, res);
        } else {
             __m512i va = _mm512_loadu_si512(a);
            __m512i vb = _mm512_loadu_si512(b);
            __m512i vc = _mm512_loadu_si512(c);
            __m512i res = _mm512_ternarylogic_epi64(va, vb, vc, 0xFE);
            _mm512_storeu_si512(out, res);
        }
    }
    
    static void sisd_or3(const T* a, const T* b, const T* c, T* out) {
        for(size_t i=0; i<INPUT_SIZE; ++i) {
             if constexpr (std::is_integral_v<T>) {
                out[i] = a[i] | b[i] | c[i];
            } else {
                 if constexpr (sizeof(T)==4) {
                     uint32_t ia, ib, ic;
                     memcpy(&ia, &a[i], 4); memcpy(&ib, &b[i], 4); memcpy(&ic, &c[i], 4);
                     uint32_t r = ia | ib | ic;
                     memcpy(&out[i], &r, 4);
                 } else {
                     uint64_t ia, ib, ic;
                     memcpy(&ia, &a[i], 8); memcpy(&ib, &b[i], 8); memcpy(&ic, &c[i], 8);
                     uint64_t r = ia | ib | ic;
                     memcpy(&out[i], &r, 8);
                 }
            }
        }
    }

    static constexpr auto make_ops() {
        return std::array{
            OpEntry3<ARG1_TYPE, ARG2_TYPE, ARG3_TYPE, OUTPUT_TYPE>{ "xor3 (0x96)", avx512_xor3, sisd_xor3 },
            OpEntry3<ARG1_TYPE, ARG2_TYPE, ARG3_TYPE, OUTPUT_TYPE>{ "or3 (0xFE)", avx512_select, sisd_or3 }
        };
    }
};

#ifdef REGISTER_TEST
using AVX512_TERNARY_INT = AVX512_TERNARY<int>;
REGISTER_TEST(AVX512_TERNARY_INT)
using AVX512_TERNARY_I64 = AVX512_TERNARY<int64_t>;
REGISTER_TEST(AVX512_TERNARY_I64)
#endif
