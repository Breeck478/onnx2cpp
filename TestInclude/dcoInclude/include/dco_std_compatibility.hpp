/**
dco/c++/base v4.4.1
    -- Algorithmic Differentiation by Operator Overloading in C++

COPYRIGHT 2024
The Numerical Algorithms Group Limited and
Software and Tools for Computational Engineering @ RWTH Aachen University

This file is part of dco/c++.
**/

#pragma once

#ifdef DCO_STD_COMPATIBILITY
namespace std {
using dco::internal::ceil;
using dco::internal::floor;
using dco::internal::trunc;

using dco::internal::max;
using dco::internal::min;

using dco::internal::abs;
using dco::internal::acos;
using dco::internal::acosh;
using dco::internal::asin;
using dco::internal::asinh;
using dco::internal::atan;
using dco::internal::atan2;
using dco::internal::atanh;
using dco::internal::cbrt;
using dco::internal::cos;
using dco::internal::cosh;
using dco::internal::erf;
using dco::internal::erfc;
using dco::internal::exp;
using dco::internal::exp2;
using dco::internal::expm1;
using dco::internal::fabs;
using dco::internal::fpclassify;
using dco::internal::frexp;
using dco::internal::hypot;
using dco::internal::isfinite;
using dco::internal::isgreater;
using dco::internal::isgreaterequal;
using dco::internal::isinf;
using dco::internal::isless;
using dco::internal::islessequal;
using dco::internal::islessgreater;
using dco::internal::isnan;
using dco::internal::isnormal;
using dco::internal::isunordered;
using dco::internal::ldexp;
using dco::internal::llrint;
using dco::internal::llround;
using dco::internal::log;
using dco::internal::log10;
using dco::internal::log1p;
using dco::internal::log2;
using dco::internal::lrint;
using dco::internal::lround;
using dco::internal::nearbyint;
using dco::internal::pow;
using dco::internal::rint;
using dco::internal::round;
using dco::internal::scalbln;
using dco::internal::scalbn;
using dco::internal::signbit;
using dco::internal::sin;
using dco::internal::sinh;
using dco::internal::sqrt;
using dco::internal::tan;
using dco::internal::tanh;

template <class DCO_TAPE_REAL, class DATA_HANDLER>
struct numeric_limits<dco::internal::active_type<DCO_TAPE_REAL, DATA_HANDLER>> : public numeric_limits<DCO_TAPE_REAL> {
  typedef dco::internal::active_type<DCO_TAPE_REAL, DATA_HANDLER> DCO_T;
  static constexpr DCO_T min() { return numeric_limits<typename DCO_T::VALUE_TYPE>::min(); }
  static constexpr DCO_T max() { return numeric_limits<typename DCO_T::VALUE_TYPE>::max(); }
  static constexpr DCO_T epsilon() { return numeric_limits<typename DCO_T::VALUE_TYPE>::epsilon(); }
  static constexpr DCO_T round_error() { return numeric_limits<typename DCO_T::VALUE_TYPE>::round_error(); }
  static constexpr DCO_T infinity() { return numeric_limits<typename DCO_T::VALUE_TYPE>::infinity(); }
  static constexpr DCO_T quiet_NaN() { return numeric_limits<typename DCO_T::VALUE_TYPE>::quiet_NaN(); }
  static constexpr DCO_T signaling_NaN() { return numeric_limits<typename DCO_T::VALUE_TYPE>::signaling_NaN(); }
  static constexpr DCO_T denorm_min() { return numeric_limits<typename DCO_T::VALUE_TYPE>::denorm_min(); }
  static constexpr DCO_T lowest() { return numeric_limits<typename DCO_T::VALUE_TYPE>::lowest(); }
};
} // namespace std
#endif

#ifdef DCO_STD_IS_ARITHMETIC_SPEC

namespace std {

template <typename T, typename D> class is_arithmetic<dco::internal::active_type<T, D>> {
public:
  static constexpr bool value = true;
};
template <typename T, typename D> class is_arithmetic<dco::internal::active_type<T, D> const> {
public:
  static constexpr bool value = true;
};

} // namespace std
#endif
