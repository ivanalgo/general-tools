template <typename T>
struct AVX512_BITWISE {
    using ARG1_TYPE = T;
    using ARG2_TYPE = T;
    using OUTPUT_TYPE = T;
    static constexpr size_t INPUT_SIZE = 512 / (8 * sizeof(T));
	static constexpr size_t OUTPUT_SIZE = INPUT_SIZE;
    static constexpr const char* CLASS_NAME = "AVX512_BITWISE";
    static constexpr int INPUT_ARGS = 2;

    /* SISD 实现 */
	static void sisd_and(const T* a, const T* b, T* out) {
        if constexpr (std::is_integral_v<T>) {
            for (size_t i = 0; i < INPUT_SIZE; ++i)
                out[i] = a[i] & b[i];
        } else if constexpr (std::is_same_v<T, float>) {
            for (size_t i = 0; i < INPUT_SIZE; ++i) {
                uint32_t ua, ub, uc;
                std::memcpy(&ua, &a[i], sizeof(ua));
                std::memcpy(&ub, &b[i], sizeof(ub));
                uc = ua & ub;
                std::memcpy(&out[i], &uc, sizeof(uc));
            }
        } else if constexpr (std::is_same_v<T, double>) {
            for (size_t i = 0; i < INPUT_SIZE; ++i) {
                uint64_t ua, ub, uc;
                std::memcpy(&ua, &a[i], sizeof(ua));
                std::memcpy(&ub, &b[i], sizeof(ub));
                uc = ua & ub;
                std::memcpy(&out[i], &uc, sizeof(uc));
            }
        }
    }

    static void sisd_or(const T* a, const T* b, T* out) {
        if constexpr (std::is_integral_v<T>) {
            for (size_t i = 0; i < INPUT_SIZE; ++i)
                out[i] = a[i] | b[i];
        } else if constexpr (std::is_same_v<T, float>) {
            for (size_t i = 0; i < INPUT_SIZE; ++i) {
                uint32_t ua, ub, uc;
                std::memcpy(&ua, &a[i], sizeof(ua));
                std::memcpy(&ub, &b[i], sizeof(ub));
                uc = ua | ub;
                std::memcpy(&out[i], &uc, sizeof(uc));
            }
        } else if constexpr (std::is_same_v<T, double>) {
            for (size_t i = 0; i < INPUT_SIZE; ++i) {
                uint64_t ua, ub, uc;
                std::memcpy(&ua, &a[i], sizeof(ua));
                std::memcpy(&ub, &b[i], sizeof(ub));
                uc = ua | ub;
                std::memcpy(&out[i], &uc, sizeof(uc));
            }
        }
    }

    static void sisd_xor(const T* a, const T* b, T* out) {
        if constexpr (std::is_integral_v<T>) {
            for (size_t i = 0; i < INPUT_SIZE; ++i)
                out[i] = a[i] ^ b[i];
        } else if constexpr (std::is_same_v<T, float>) {
            for (size_t i = 0; i < INPUT_SIZE; ++i) {
                uint32_t ua, ub, uc;
                std::memcpy(&ua, &a[i], sizeof(ua));
                std::memcpy(&ub, &b[i], sizeof(ub));
                uc = ua ^ ub;
                std::memcpy(&out[i], &uc, sizeof(uc));
            }
        } else if constexpr (std::is_same_v<T, double>) {
            for (size_t i = 0; i < INPUT_SIZE; ++i) {
                uint64_t ua, ub, uc;
                std::memcpy(&ua, &a[i], sizeof(ua));
                std::memcpy(&ub, &b[i], sizeof(ub));
                uc = ua ^ ub;
                std::memcpy(&out[i], &uc, sizeof(uc));
            }
        }
    }

    static void sisd_andnot(const T* a, const T* b, T* out) {
        if constexpr (std::is_integral_v<T>) {
            for (size_t i = 0; i < INPUT_SIZE; ++i)
                out[i] = ~a[i] & b[i];
        } else if constexpr (std::is_same_v<T, float>) {
            for (size_t i = 0; i < INPUT_SIZE; ++i) {
                uint32_t ua, ub, uc;
                std::memcpy(&ua, &a[i], sizeof(ua));
                std::memcpy(&ub, &b[i], sizeof(ub));
                uc = ~ua & ub;
                std::memcpy(&out[i], &uc, sizeof(uc));
            }
        } else if constexpr (std::is_same_v<T, double>) {
            for (size_t i = 0; i < INPUT_SIZE; ++i) {
                uint64_t ua, ub, uc;
                std::memcpy(&ua, &a[i], sizeof(ua));
                std::memcpy(&ub, &b[i], sizeof(ub));
                uc = ~ua & ub;
                std::memcpy(&out[i], &uc, sizeof(uc));
            }
        }
    }

    /* AVX512 实现 */
	   static void avx512_and(const T* a, const T* b, T* out) {
        if constexpr (std::is_same_v<T, int>) {
            __m512i va = _mm512_loadu_si512(a);
            __m512i vb = _mm512_loadu_si512(b);
            __m512i vc = _mm512_and_si512(va, vb);
            _mm512_storeu_si512(out, vc);
        } else if constexpr (std::is_same_v<T, float>) {
            __m512 va = _mm512_loadu_ps(a);
            __m512 vb = _mm512_loadu_ps(b);
            __m512i vai = _mm512_castps_si512(va);
            __m512i vbi = _mm512_castps_si512(vb);
            __m512i vr  = _mm512_and_si512(vai, vbi);
            __m512 vc  = _mm512_castsi512_ps(vr);
            _mm512_storeu_ps(out, vc);
        } else if constexpr (std::is_same_v<T, double>) {
            __m512d va = _mm512_loadu_pd(a);
            __m512d vb = _mm512_loadu_pd(b);
            __m512i vai = _mm512_castpd_si512(va);
            __m512i vbi = _mm512_castpd_si512(vb);
            __m512i vr  = _mm512_and_si512(vai, vbi);
            __m512d vc  = _mm512_castsi512_pd(vr);
            _mm512_storeu_pd(out, vc);
        }
    }

    static void avx512_or(const T* a, const T* b, T* out) {
        if constexpr (std::is_same_v<T, int>) {
            __m512i va = _mm512_loadu_si512(a);
            __m512i vb = _mm512_loadu_si512(b);
            __m512i vc = _mm512_or_si512(va, vb);
            _mm512_storeu_si512(out, vc);
        } else if constexpr (std::is_same_v<T, float>) {
            __m512 va = _mm512_loadu_ps(a);
            __m512 vb = _mm512_loadu_ps(b);
            __m512i vai = _mm512_castps_si512(va);
            __m512i vbi = _mm512_castps_si512(vb);
            __m512i vr  = _mm512_or_si512(vai, vbi);
            __m512 vc  = _mm512_castsi512_ps(vr);
            _mm512_storeu_ps(out, vc);
        } else if constexpr (std::is_same_v<T, double>) {
            __m512d va = _mm512_loadu_pd(a);
            __m512d vb = _mm512_loadu_pd(b);
            __m512i vai = _mm512_castpd_si512(va);
            __m512i vbi = _mm512_castpd_si512(vb);
            __m512i vr  = _mm512_or_si512(vai, vbi);
            __m512d vc  = _mm512_castsi512_pd(vr);
            _mm512_storeu_pd(out, vc);
        }
    }

    static void avx512_xor(const T* a, const T* b, T* out) {
        if constexpr (std::is_same_v<T, int>) {
            __m512i va = _mm512_loadu_si512(a);
            __m512i vb = _mm512_loadu_si512(b);
            __m512i vc = _mm512_xor_si512(va, vb);
            _mm512_storeu_si512(out, vc);
        } else if constexpr (std::is_same_v<T, float>) {
            __m512 va = _mm512_loadu_ps(a);
            __m512 vb = _mm512_loadu_ps(b);
            __m512i vai = _mm512_castps_si512(va);
            __m512i vbi = _mm512_castps_si512(vb);
            __m512i vr  = _mm512_xor_si512(vai, vbi);
            __m512 vc  = _mm512_castsi512_ps(vr);
            _mm512_storeu_ps(out, vc);
        } else if constexpr (std::is_same_v<T, double>) {
            __m512d va = _mm512_loadu_pd(a);
            __m512d vb = _mm512_loadu_pd(b);
            __m512i vai = _mm512_castpd_si512(va);
            __m512i vbi = _mm512_castpd_si512(vb);
            __m512i vr  = _mm512_xor_si512(vai, vbi);
            __m512d vc  = _mm512_castsi512_pd(vr);
            _mm512_storeu_pd(out, vc);
        }
    }

    static void avx512_andnot(const T* a, const T* b, T* out) {
        if constexpr (std::is_same_v<T, int>) {
            __m512i va = _mm512_loadu_si512(a);
            __m512i vb = _mm512_loadu_si512(b);
            __m512i vc = _mm512_andnot_si512(va, vb);
            _mm512_storeu_si512(out, vc);
        } else if constexpr (std::is_same_v<T, float>) {
            __m512 va = _mm512_loadu_ps(a);
            __m512 vb = _mm512_loadu_ps(b);
            __m512i vai = _mm512_castps_si512(va);
            __m512i vbi = _mm512_castps_si512(vb);
            __m512i vr  = _mm512_andnot_si512(vai, vbi);
            __m512 vc  = _mm512_castsi512_ps(vr);
            _mm512_storeu_ps(out, vc);
        } else if constexpr (std::is_same_v<T, double>) {
            __m512d va = _mm512_loadu_pd(a);
            __m512d vb = _mm512_loadu_pd(b);
            __m512i vai = _mm512_castpd_si512(va);
            __m512i vbi = _mm512_castpd_si512(vb);
            __m512i vr  = _mm512_andnot_si512(vai, vbi);
            __m512d vc  = _mm512_castsi512_pd(vr);
            _mm512_storeu_pd(out, vc);
        }
    }

	static constexpr auto make_ops() {
	    return std::array{
    	    OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "avx512 and", avx512_and, sisd_and },
        	OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "avx512 or",  avx512_or,  sisd_or },
        	OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "avx512 xor", avx512_xor, sisd_xor },
        	OpEntry2<ARG1_TYPE, ARG2_TYPE, OUTPUT_TYPE>{ "avx512 andnot", avx512_andnot, sisd_andnot },
    	};
	}
};
