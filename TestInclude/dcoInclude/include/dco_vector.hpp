/**
dco/c++/base v4.4.1
    -- Algorithmic Differentiation by Operator Overloading in C++

COPYRIGHT 2024
The Numerical Algorithms Group Limited and
Software and Tools for Computational Engineering @ RWTH Aachen University

This file is part of dco/c++.
**/

#pragma once

#include "dco_vector_math.hpp"
#include "dco_vector_traits.hpp"

namespace dco {
namespace vector_ns {

template <typename FLOAT_T, std::size_t VECTOR_SIZE> struct alignas(alignof(FLOAT_T[VECTOR_SIZE])) vector_t {
  static constexpr std::size_t _vector_size = VECTOR_SIZE;
  static constexpr std::size_t _vector_level = traits::type_info<FLOAT_T>::vector_level + 1;

  FLOAT_T _values[_vector_size];

  DCO_VECTOR_INLINE FLOAT_T const& operator[](std::size_t const i) const noexcept { return _values[i]; }
  DCO_VECTOR_INLINE FLOAT_T& operator[](std::size_t const i) noexcept { return _values[i]; }

  DCO_VECTOR_INLINE FLOAT_T const& _scalarvalue() const { return _values[0]; }
  DCO_VECTOR_INLINE FLOAT_T& _scalarvalue() { return _values[0]; }

  DCO_VECTOR_INLINE vector_t() noexcept {
    auto const& initval = FLOAT_T();
    for (std::size_t i = 0; i < _vector_size; ++i) {
      _values[i] = initval;
    }
  }

  DCO_VECTOR_INLINE vector_t(vector_t const& other) noexcept {
    for (std::size_t i = 0; i < _vector_size; ++i) {
      _values[i] = other._values[i];
    }
  }

  template <class OTHERFLOAT_T> DCO_VECTOR_INLINE vector_t(vector_t<OTHERFLOAT_T, _vector_size> const& other) noexcept {
    for (std::size_t i = 0; i < _vector_size; ++i) {
      _values[i] = static_cast<FLOAT_T>(other._values[i]);
    }
  }

  template <class OTHERFLOAT_T, typename TI = traits::type_info<vector_t>,
            typename check_vs = std::enable_if_t<(TI::vector_size > 1)>>
  DCO_VECTOR_INLINE vector_t(vector_t<OTHERFLOAT_T, 1> const& other) noexcept {

    const FLOAT_T float_scalar = static_cast<FLOAT_T>(other._values[0]);
    for (std::size_t i = 0; i < _vector_size; ++i) {
      _values[i] = float_scalar;
    }
  }

  DCO_VECTOR_INLINE vector_t& operator=(vector_t const& other) noexcept {
    for (std::size_t i = 0; i < _vector_size; ++i) {
      _values[i] = other._values[i];
    }
    return *this;
  }

  template <class OTHERFLOAT_T>
  DCO_VECTOR_INLINE vector_t& operator=(vector_t<OTHERFLOAT_T, _vector_size> const& other) noexcept {
    for (std::size_t i = 0; i < _vector_size; ++i) {
      _values[i] = static_cast<FLOAT_T>(other._values[i]);
    }
    return *this;
  }

  DCO_VECTOR_INLINE vector_t(FLOAT_T const& scalar) noexcept {
    for (std::size_t i = 0; i < _vector_size; ++i) {
      _values[i] = scalar;
    }
  }

  template <typename OTHERFLOAT_T, typename TI = traits::type_info<OTHERFLOAT_T>,
            typename check_vl = std::enable_if_t<!(TI::vector_level >= _vector_level)>>
  DCO_VECTOR_INLINE vector_t(OTHERFLOAT_T const& scalar) noexcept {

    const FLOAT_T float_scalar = static_cast<FLOAT_T>(scalar);
    for (std::size_t i = 0; i < _vector_size; ++i) {
      _values[i] = float_scalar;
    }
  }

#ifndef DCO_VECTOR_EXCLUDE_ASSIGNMENT_DEFINITION
  DCO_VECTOR_INLINE vector_t& operator=(FLOAT_T const& scalar) noexcept {
    for (std::size_t i = 0; i < _vector_size; ++i) {
      _values[i] = scalar;
    }
    return *this;
  }

  template <class OTHERFLOAT_T, typename TI = traits::type_info<OTHERFLOAT_T>,
            typename check_arithmetic = std::enable_if_t<TI::is_arithmetic>>
  DCO_VECTOR_INLINE vector_t& operator=(OTHERFLOAT_T const& scalar) noexcept {

    const FLOAT_T float_scalar = static_cast<FLOAT_T>(scalar);
    for (std::size_t i = 0; i < _vector_size; ++i) {
      _values[i] = float_scalar;
    }
    return *this;
  }
#endif

  template <class LAMBDA_T>
  DCO_VECTOR_INLINE vector_t(::dco::vector_ns::ctor_lambda_t const&, LAMBDA_T const& lambda) noexcept {
    for (std::size_t i = 0; i < _vector_size; ++i) {
      _values[i] = lambda(i);
    }
  }

  template <class OTHERFLOAT_T, class TI = traits::type_info<OTHERFLOAT_T>,
            typename check_arithmetic = std::enable_if_t<TI::is_arithmetic>>
  DCO_VECTOR_INLINE vector_t& operator+=(OTHERFLOAT_T const& rhs) {
    *this = *this + rhs;
    return *this;
  }

  template <class OTHERFLOAT_T, class TI = traits::type_info<OTHERFLOAT_T>,
            typename check_arithmetic = std::enable_if_t<TI::is_arithmetic>>
  DCO_VECTOR_INLINE vector_t& operator*=(OTHERFLOAT_T const& rhs) {
    *this = *this * rhs;
    return *this;
  }
};
} // namespace vector_ns
} // namespace dco

#include "dco_vector_operations.hpp"

#if defined __AVX2__ || defined __AVX__

#include "dco_vector_avx_operations.hpp"

#else
#ifndef DCO_DISABLE_AVX2_WARNING
#ifdef _MSC_VER
#pragma message(                                                                                                       \
    "WARNING: Vector type is enabled; but no AVX2 seems to be available; if AVX2 is supported by your CPU, you might have to use '/arch:AVX2' as compilation flag. Otherwise, if you cannot or don't want to use AVX2, you can disable this warning by defining 'DCO_DISABLE_AVX2_WARNING'.")
#else
#warning                                                                                                               \
    "Vector type is enabled; but no AVX2 seems to be available; if AVX2 is supported by your CPU, you might have to use '-march=native' as compilation flag. Otherwise, if you cannot or don't want to use AVX2, you can disable this warning by defining 'DCO_DISABLE_AVX2_WARNING'."
#endif
#endif
#endif

namespace dco {
template <typename FLOAT_T, std::size_t VECTOR_SIZE> struct gv {

  static_assert(std::is_fundamental_v<FLOAT_T>, "dco/c++: gv<...> can only be used with fundamental types");

  typedef vector_ns::vector_t<FLOAT_T, VECTOR_SIZE> type;
  typedef type active_t;

  typedef type active_value_t;
  typedef type value_t;
  typedef FLOAT_T scalar_value_t;
  typedef type passive_t;
  typedef scalar_value_t scalar_passive_t;
  typedef internal::data_g<gv> data_t;
  typedef void derivative_t;
  typedef void tape_t;
  typedef void local_gradient_t;
  typedef void local_gradient_with_activity_t;
  typedef void external_adjoint_object_t;
  typedef internal::jacobian_preaccumulator_t<> jacobian_preaccumulator_t;
  static constexpr bool is_dco_type = true;
  static constexpr bool is_adjoint_type = false;
  static constexpr bool is_tangent_type = false;
  static constexpr bool is_vector_type = true;
  static constexpr int order = dco::mode<scalar_value_t>::order;
  static constexpr int vector_size = VECTOR_SIZE;
  static constexpr std::size_t derivative_vector_length = 0;

  static constexpr std::size_t p1f_size = 0;
};

template <typename FLOAT_T, std::size_t VECTOR_SIZE>
struct mode<vector_ns::vector_t<FLOAT_T, VECTOR_SIZE>> : public gv<FLOAT_T, VECTOR_SIZE> {};

template <typename FLOAT_T, std::size_t VECTOR_SIZE> FLOAT_T vsum(vector_ns::vector_t<FLOAT_T, VECTOR_SIZE> const& x) {
  FLOAT_T ret = 0;
  for (std::size_t i = 0; i < VECTOR_SIZE; ++i) {
    ret += x[i];
  }
  return ret;
}

namespace folding {

template <typename T> struct is_zero_trait<T, std::enable_if_t<(vector_ns::traits::type_info<T>::vector_level > 0)>> {
  static bool get(const T& x) {
    for (std::size_t i = 0; i < T::_vector_size; ++i) {
      if (!is_zero_trait<typename vector_ns::traits::type_info<T>::vector_scalar_t>::get(x[i])) {
        return false;
      }
    }
    return true;
  }
};

} // namespace folding

namespace internal {
template <class FLOAT_T, std::size_t VECTOR_SIZE>
struct passive_value_type_of<vector_ns::vector_t<FLOAT_T, VECTOR_SIZE>> {
  typedef typename passive_value_type_of<FLOAT_T>::TYPE TYPE;
};
} // namespace internal

} // namespace dco
