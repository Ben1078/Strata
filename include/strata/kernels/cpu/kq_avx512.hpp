// include/strata/kernels/cpu/kq_avx512.hpp - Q5_K / Q8_0 expert rows for 1..8 tokens on AVX-512 (VNNI, BW, VL),
// bit-exact against ggml-cpu's vec_dot (see kq_avx512.cpp).  Call only where cpu_avx512_ok().
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// Q5_K (13) and Q8_0 (8).
bool kq512_supported(int type) noexcept;
/// ff[t][r] = silu(gate_r . act[t]) * (up_r . act[t]) for rows [r0, r1), any nt (in groups of 8).
void kq512_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                   float* const* ff, int r0, int r1);
/// out[t][r] = w_r . act[t] for rows [r0, r1).
void kq512_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                int r0, int r1);

}  // namespace strata::kernels::cpu
