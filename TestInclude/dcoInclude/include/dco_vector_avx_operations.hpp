/**
dco/c++/base v4.4.1
    -- Algorithmic Differentiation by Operator Overloading in C++

COPYRIGHT 2024
The Numerical Algorithms Group Limited and
Software and Tools for Computational Engineering @ RWTH Aachen University

This file is part of dco/c++.
**/

#pragma once

#ifdef DCO_USE_INTEL_SVML

#include <immintrin.h>
extern "C" {

__m256d __svml_sin4(__m256d);
__m256d __svml_cos4(__m256d);
__m256d __svml_exp4(__m256d);
__m256 __svml_sinf8(__m256);
__m256 __svml_cosf8(__m256);
__m256 __svml_expf8(__m256);
}

namespace dco {
namespace vector_ns {

#define DCO_SPEC_INTEL_VEC_DOUBLE(op)                                                                                  \
  template <std::size_t VECTOR_SIZE, typename = std::enable_if_t<(VECTOR_SIZE % 4 == 0)>>                              \
  DCO_VECTOR_INLINE vector_t<double, VECTOR_SIZE> op(vector_t<double, VECTOR_SIZE> const& x) noexcept {                \
    vector_t<double, VECTOR_SIZE> ret;                                                                                 \
    for (std::size_t i = 0; i < VECTOR_SIZE / 4; ++i) {                                                                \
      __m256d data = _mm256_loadu_pd(x._values + i * 4);                                                               \
      __m256d values = __svml_##op##4(data);                                                                           \
      _mm256_storeu_pd(ret._values + i * 4, values);                                                                   \
    }                                                                                                                  \
    return ret;                                                                                                        \
  }
DCO_SPEC_INTEL_VEC_DOUBLE(sin)
DCO_SPEC_INTEL_VEC_DOUBLE(cos)
DCO_SPEC_INTEL_VEC_DOUBLE(exp)

#define DCO_SPEC_INTEL_VEC_FLOAT(op)                                                                                   \
  template <std::size_t VECTOR_SIZE, typename = std::enable_if_t<(VECTOR_SIZE % 8 == 0)>>                              \
  DCO_VECTOR_INLINE vector_t<float, VECTOR_SIZE> op(vector_t<float, VECTOR_SIZE> const& x) noexcept {                  \
    vector_t<float, VECTOR_SIZE> ret;                                                                                  \
    for (std::size_t i = 0; i < VECTOR_SIZE / 8; ++i) {                                                                \
      __m256 data = _mm256_loadu_ps(x._values + i * 8);                                                                \
      __m256 values = __svml_##op##f8(data);                                                                           \
      _mm256_storeu_ps(ret._values + i * 8, values);                                                                   \
    }                                                                                                                  \
    return ret;                                                                                                        \
  }
DCO_SPEC_INTEL_VEC_FLOAT(sin)
DCO_SPEC_INTEL_VEC_FLOAT(cos)
DCO_SPEC_INTEL_VEC_FLOAT(exp)

#undef DCO_SPEC_INTEL_VEC_FLOAT
#undef DCO_SPEC_INTEL_VEC_DOUBLE
} // namespace vector_ns
} // namespace dco
#endif

#ifdef DCO_USE_INTEL_MKL
#include "mkl.h"
namespace dco {
namespace vector_ns {
template <std::size_t VECTOR_SIZE, typename = std::enable_if_t<(VECTOR_SIZE >= 4)>>
DCO_VECTOR_INLINE vector_t<double, VECTOR_SIZE> erfc(vector_t<double, VECTOR_SIZE> const& x) noexcept {
  vector_t<double, VECTOR_SIZE> ret;
  vdErfc(VECTOR_SIZE, x._values, ret._values);
  return ret;
}

template <std::size_t VECTOR_SIZE, typename = std::enable_if_t<(VECTOR_SIZE >= 4)>>
DCO_VECTOR_INLINE vector_t<double, VECTOR_SIZE> log(vector_t<double, VECTOR_SIZE> const& x) noexcept {
  vector_t<double, VECTOR_SIZE> ret;
  vdLn(VECTOR_SIZE, x._values, ret._values);
  return ret;
}
} // namespace vector_ns
} // namespace dco
#endif
