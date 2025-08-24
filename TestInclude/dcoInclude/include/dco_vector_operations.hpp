/**
dco/c++/base v4.4.1
    -- Algorithmic Differentiation by Operator Overloading in C++

COPYRIGHT 2024
The Numerical Algorithms Group Limited and
Software and Tools for Computational Engineering @ RWTH Aachen University

This file is part of dco/c++.
**/

#pragma once

#include "dco_vector_common.hpp"
#include "dco_vector_math.hpp"
#include "dco_vector.hpp"

namespace dco {
namespace vector_ns {

template <typename T> using passive_type_t = typename dco::internal::passive_value_type_of<T>::TYPE;

template <typename FLOAT1_T, typename FLOAT2_T>
using common_passive_type_t = std::common_type_t<passive_type_t<FLOAT1_T>, passive_type_t<FLOAT2_T>>;

template <std::size_t VECTOR_SIZE, typename FLOAT_T>
DCO_VECTOR_INLINE std::ostream& operator<<(std::ostream& out, const vector_t<FLOAT_T, VECTOR_SIZE>& v) noexcept {

  if constexpr (VECTOR_SIZE == 1) {
    out << v[0];
  } else {
    out << "[";
    for (std::size_t i = 0; i < VECTOR_SIZE - 1; ++i) {
      out << v[i] << ",";
    }
    out << v[VECTOR_SIZE - 1] << "]";
  }
  return out;
}

template <std::size_t VECTOR_SIZE, typename FLOAT_T>
DCO_VECTOR_INLINE std::istream& operator>>(std::istream& in, vector_t<FLOAT_T, VECTOR_SIZE>& v) noexcept {
  if constexpr (VECTOR_SIZE == 1) {
    in >> v[0];
  } else {
    char dummy;
    in >> dummy;
    for (std::size_t i = 0; i < VECTOR_SIZE - 1; ++i) {
      in >> v[0];
      in >> dummy;
    }
    in >> v[VECTOR_SIZE - 1];
    in >> dummy;
  }
  return in;
}

#define DCO_COMPARE_VECTOR(OP)                                                                                         \
  template <typename FLOAT1_T, typename FLOAT2_T>                                                                      \
  DCO_VECTOR_INLINE bool operator OP(const vector_t<FLOAT1_T, 1>& x1, const vector_t<FLOAT2_T, 1>& x2) noexcept {      \
    using common_passive_type_t = common_passive_type_t<FLOAT1_T, FLOAT2_T>;                                           \
    return static_cast<common_passive_type_t>(x1._values[0]) OP static_cast<common_passive_type_t>(x2._values[0]);     \
  }                                                                                                                    \
  template <std::size_t VECTOR_SIZE, typename FLOAT1_T, typename FLOAT2_T>                                             \
  DCO_VECTOR_INLINE std::bitset<VECTOR_SIZE> operator OP(const vector_t<FLOAT1_T, VECTOR_SIZE>& x1,                    \
                                                         const vector_t<FLOAT2_T, VECTOR_SIZE>& x2) noexcept {         \
    using common_passive_type_t = common_passive_type_t<FLOAT1_T, FLOAT2_T>;                                           \
    std::bitset<VECTOR_SIZE> ret;                                                                                      \
    for (std::size_t i = 0; i < VECTOR_SIZE; ++i) {                                                                    \
      ret[i] = static_cast<common_passive_type_t>(passive_value(x1._values[i]))                                        \
          OP static_cast<common_passive_type_t>(passive_value(x2._values[i]));                                         \
    }                                                                                                                  \
    return ret;                                                                                                        \
  }                                                                                                                    \
  template <std::size_t VECTOR_SIZE, typename FLOAT1_T, typename FLOAT2_T,                                             \
            typename check_vs = std::enable_if_t<(VECTOR_SIZE > 1)>>                                                   \
  DCO_VECTOR_INLINE std::bitset<VECTOR_SIZE> operator OP(const vector_t<FLOAT1_T, VECTOR_SIZE>& x1,                    \
                                                         const vector_t<FLOAT2_T, 1>& x2) noexcept {                   \
    using common_passive_type_t = common_passive_type_t<FLOAT1_T, FLOAT2_T>;                                           \
    std::bitset<VECTOR_SIZE> ret;                                                                                      \
    for (std::size_t i = 0; i < VECTOR_SIZE; ++i) {                                                                    \
      ret[i] = static_cast<common_passive_type_t>(passive_value(x1._values[i]))                                        \
          OP static_cast<common_passive_type_t>(passive_value(x2._values[0]));                                         \
    }                                                                                                                  \
    return ret;                                                                                                        \
  }                                                                                                                    \
  template <std::size_t VECTOR_SIZE, typename FLOAT1_T, typename FLOAT2_T,                                             \
            typename check_vs = std::enable_if_t<(VECTOR_SIZE > 1)>>                                                   \
  DCO_VECTOR_INLINE std::bitset<VECTOR_SIZE> operator OP(const vector_t<FLOAT1_T, 1>& x1,                              \
                                                         const vector_t<FLOAT2_T, VECTOR_SIZE>& x2) noexcept {         \
    using common_passive_type_t = common_passive_type_t<FLOAT1_T, FLOAT2_T>;                                           \
    std::bitset<VECTOR_SIZE> ret;                                                                                      \
    for (std::size_t i = 0; i < VECTOR_SIZE; ++i) {                                                                    \
      ret[i] = static_cast<common_passive_type_t>(passive_value(x1._values[0]))                                        \
          OP static_cast<common_passive_type_t>(passive_value(x2._values[i]));                                         \
    }                                                                                                                  \
    return ret;                                                                                                        \
  }                                                                                                                    \
  template <std::size_t VECTOR_SIZE, typename FLOAT1_T, typename FLOAT2_T,                                             \
            typename check_vs = std::enable_if_t<(VECTOR_SIZE > 1)>,                                                   \
            typename =                                                                                                 \
                std::enable_if_t<(traits::type_info<typename dco::mode<FLOAT2_T>::passive_t>::vector_level == 0)>>     \
  DCO_VECTOR_INLINE std::bitset<VECTOR_SIZE> operator OP(const vector_t<FLOAT1_T, VECTOR_SIZE>& x1,                    \
                                                         const FLOAT2_T & x2) noexcept {                               \
    using common_passive_type_t = common_passive_type_t<FLOAT1_T, FLOAT2_T>;                                           \
    std::bitset<VECTOR_SIZE> ret;                                                                                      \
    for (std::size_t i = 0; i < VECTOR_SIZE; ++i) {                                                                    \
      ret[i] = static_cast<common_passive_type_t>(passive_value(x1._values[i]))                                        \
          OP static_cast<common_passive_type_t>(passive_value(x2));                                                    \
    }                                                                                                                  \
    return ret;                                                                                                        \
  }                                                                                                                    \
  template <std::size_t VECTOR_SIZE, typename FLOAT1_T, typename FLOAT2_T,                                             \
            typename check_vs = std::enable_if_t<(VECTOR_SIZE > 1)>,                                                   \
            typename =                                                                                                 \
                std::enable_if_t<(traits::type_info<typename dco::mode<FLOAT1_T>::passive_t>::vector_level == 0)>>     \
  DCO_VECTOR_INLINE std::bitset<VECTOR_SIZE> operator OP(const FLOAT1_T & x1,                                          \
                                                         const vector_t<FLOAT2_T, VECTOR_SIZE>& x2) noexcept {         \
    using common_passive_type_t = common_passive_type_t<FLOAT1_T, FLOAT2_T>;                                           \
    std::bitset<VECTOR_SIZE> ret;                                                                                      \
    for (std::size_t i = 0; i < VECTOR_SIZE; ++i) {                                                                    \
      ret[i] = static_cast<common_passive_type_t>(passive_value(x1))                                                   \
          OP static_cast<common_passive_type_t>(passive_value(x2._values[i]));                                         \
    }                                                                                                                  \
    return ret;                                                                                                        \
  }                                                                                                                    \
  template <typename FLOAT1_T, typename FLOAT2_T>                                                                      \
  DCO_VECTOR_INLINE bool operator OP(const FLOAT1_T& x1, const vector_t<FLOAT2_T, 1>& x2) noexcept {                   \
    using common_passive_type_t = common_passive_type_t<FLOAT1_T, FLOAT2_T>;                                           \
    return static_cast<common_passive_type_t>(passive_value(x1))                                                       \
        OP static_cast<common_passive_type_t>(passive_value(x2._values[0]));                                           \
  }                                                                                                                    \
  template <typename FLOAT1_T, typename FLOAT2_T>                                                                      \
  DCO_VECTOR_INLINE bool operator OP(const vector_t<FLOAT1_T, 1>& x1, const FLOAT2_T& x2) noexcept {                   \
    using common_passive_type_t = common_passive_type_t<FLOAT1_T, FLOAT2_T>;                                           \
    return static_cast<common_passive_type_t>(passive_value(x1._values[0]))                                            \
        OP static_cast<common_passive_type_t>(passive_value(x2));                                                      \
  }

DCO_COMPARE_VECTOR(==)
DCO_COMPARE_VECTOR(!=)
DCO_COMPARE_VECTOR(>)
DCO_COMPARE_VECTOR(>=)
DCO_COMPARE_VECTOR(<)
DCO_COMPARE_VECTOR(<=)
#undef DCO_COMPARE_VECTOR

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<TI1_T::vector_level == TI2_T::vector_level>,
          typename check_vs = std::enable_if_t<TI1_T::vector_size == TI2_T::vector_size>,
          typename OP_T = vector_ns::math::BOP_ADD<typename TI1_T::vector_scalar_t, typename TI2_T::vector_scalar_t>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI1_T::vector_size>,
          vector_ns::ctor_vector_vv_e = vector_ns::ctor_vector_vv_default>
RET_T DCO_VECTOR_INLINE operator+(const ARG1_T& v1, const ARG2_T& v2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(v1[i], v2[i]); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<(TI1_T::vector_level > TI2_T::vector_level)>,
          typename ignore_ad = std::enable_if_t<(TI2_T::is_active_type == false)>,
          typename OP_T = vector_ns::math::BOP_ADD<typename TI1_T::vector_scalar_t, ARG2_T>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI1_T::vector_size>,
          vector_ns::ctor_vector_vs_e = vector_ns::ctor_vector_vs_default>
RET_T DCO_VECTOR_INLINE operator+(const ARG1_T& v1, const ARG2_T& x2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(v1[i], x2); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<!(TI1_T::vector_level >= TI2_T::vector_level)>,
          typename ignore_ad = std::enable_if_t<(TI1_T::is_active_type == false)>,
          typename OP_T = vector_ns::math::BOP_ADD<ARG1_T, typename TI2_T::vector_scalar_t>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI2_T::vector_size>,
          vector_ns::ctor_vector_sv_e = vector_ns::ctor_vector_sv_default>
RET_T DCO_VECTOR_INLINE operator+(const ARG1_T& x1, const ARG2_T& v2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(x1, v2[i]); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<TI1_T::vector_level == TI2_T::vector_level>,
          typename check_vs = std::enable_if_t<((TI2_T::vector_size > 1) && (TI1_T::vector_size == 1))>,
          typename OP_T = vector_ns::math::BOP_ADD<typename TI1_T::vector_scalar_t, typename TI2_T::vector_scalar_t>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI2_T::vector_size>,
          vector_ns::ctor_vector_1v_e = vector_ns::ctor_vector_1v_default>
RET_T DCO_VECTOR_INLINE operator+(const ARG1_T& v1, const ARG2_T& v2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(v1[0], v2[i]); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<(TI1_T::vector_level == TI2_T::vector_level)>,
          typename check_vs = std::enable_if_t<((TI2_T::vector_size == 1) && (TI1_T::vector_size > 1))>,
          typename OP_T = vector_ns::math::BOP_ADD<typename TI1_T::vector_scalar_t, typename TI2_T::vector_scalar_t>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI1_T::vector_size>,
          vector_ns::ctor_vector_v1_e = vector_ns::ctor_vector_v1_default>
RET_T DCO_VECTOR_INLINE operator+(const ARG1_T& v1, const ARG2_T& v2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(v1[i], v2[0]); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<TI1_T::vector_level == TI2_T::vector_level>,
          typename check_vs = std::enable_if_t<TI1_T::vector_size == TI2_T::vector_size>,
          typename OP_T = vector_ns::math::BOP_ATAN2<typename TI1_T::vector_scalar_t, typename TI2_T::vector_scalar_t>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI1_T::vector_size>,
          vector_ns::ctor_vector_vv_e = vector_ns::ctor_vector_vv_default>
RET_T DCO_VECTOR_INLINE atan2(const ARG1_T& v1, const ARG2_T& v2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(v1[i], v2[i]); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<(TI1_T::vector_level > TI2_T::vector_level)>,
          typename ignore_ad = std::enable_if_t<(TI2_T::is_active_type == false)>,
          typename OP_T = vector_ns::math::BOP_ATAN2<typename TI1_T::vector_scalar_t, ARG2_T>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI1_T::vector_size>,
          vector_ns::ctor_vector_vs_e = vector_ns::ctor_vector_vs_default>
RET_T DCO_VECTOR_INLINE atan2(const ARG1_T& v1, const ARG2_T& x2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(v1[i], x2); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<!(TI1_T::vector_level >= TI2_T::vector_level)>,
          typename ignore_ad = std::enable_if_t<(TI1_T::is_active_type == false)>,
          typename OP_T = vector_ns::math::BOP_ATAN2<ARG1_T, typename TI2_T::vector_scalar_t>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI2_T::vector_size>,
          vector_ns::ctor_vector_sv_e = vector_ns::ctor_vector_sv_default>
RET_T DCO_VECTOR_INLINE atan2(const ARG1_T& x1, const ARG2_T& v2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(x1, v2[i]); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<TI1_T::vector_level == TI2_T::vector_level>,
          typename check_vs = std::enable_if_t<((TI2_T::vector_size > 1) && (TI1_T::vector_size == 1))>,
          typename OP_T = vector_ns::math::BOP_ATAN2<typename TI1_T::vector_scalar_t, typename TI2_T::vector_scalar_t>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI2_T::vector_size>,
          vector_ns::ctor_vector_1v_e = vector_ns::ctor_vector_1v_default>
RET_T DCO_VECTOR_INLINE atan2(const ARG1_T& v1, const ARG2_T& v2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(v1[0], v2[i]); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<(TI1_T::vector_level == TI2_T::vector_level)>,
          typename check_vs = std::enable_if_t<((TI2_T::vector_size == 1) && (TI1_T::vector_size > 1))>,
          typename OP_T = vector_ns::math::BOP_ATAN2<typename TI1_T::vector_scalar_t, typename TI2_T::vector_scalar_t>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI1_T::vector_size>,
          vector_ns::ctor_vector_v1_e = vector_ns::ctor_vector_v1_default>
RET_T DCO_VECTOR_INLINE atan2(const ARG1_T& v1, const ARG2_T& v2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(v1[i], v2[0]); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<TI1_T::vector_level == TI2_T::vector_level>,
          typename check_vs = std::enable_if_t<TI1_T::vector_size == TI2_T::vector_size>,
          typename OP_T = vector_ns::math::BOP_DIV<typename TI1_T::vector_scalar_t, typename TI2_T::vector_scalar_t>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI1_T::vector_size>,
          vector_ns::ctor_vector_vv_e = vector_ns::ctor_vector_vv_default>
RET_T DCO_VECTOR_INLINE operator/(const ARG1_T& v1, const ARG2_T& v2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(v1[i], v2[i]); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<(TI1_T::vector_level > TI2_T::vector_level)>,
          typename ignore_ad = std::enable_if_t<(TI2_T::is_active_type == false)>,
          typename OP_T = vector_ns::math::BOP_DIV<typename TI1_T::vector_scalar_t, ARG2_T>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI1_T::vector_size>,
          vector_ns::ctor_vector_vs_e = vector_ns::ctor_vector_vs_default>
RET_T DCO_VECTOR_INLINE operator/(const ARG1_T& v1, const ARG2_T& x2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(v1[i], x2); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<!(TI1_T::vector_level >= TI2_T::vector_level)>,
          typename ignore_ad = std::enable_if_t<(TI1_T::is_active_type == false)>,
          typename OP_T = vector_ns::math::BOP_DIV<ARG1_T, typename TI2_T::vector_scalar_t>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI2_T::vector_size>,
          vector_ns::ctor_vector_sv_e = vector_ns::ctor_vector_sv_default>
RET_T DCO_VECTOR_INLINE operator/(const ARG1_T& x1, const ARG2_T& v2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(x1, v2[i]); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<TI1_T::vector_level == TI2_T::vector_level>,
          typename check_vs = std::enable_if_t<((TI2_T::vector_size > 1) && (TI1_T::vector_size == 1))>,
          typename OP_T = vector_ns::math::BOP_DIV<typename TI1_T::vector_scalar_t, typename TI2_T::vector_scalar_t>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI2_T::vector_size>,
          vector_ns::ctor_vector_1v_e = vector_ns::ctor_vector_1v_default>
RET_T DCO_VECTOR_INLINE operator/(const ARG1_T& v1, const ARG2_T& v2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(v1[0], v2[i]); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<(TI1_T::vector_level == TI2_T::vector_level)>,
          typename check_vs = std::enable_if_t<((TI2_T::vector_size == 1) && (TI1_T::vector_size > 1))>,
          typename OP_T = vector_ns::math::BOP_DIV<typename TI1_T::vector_scalar_t, typename TI2_T::vector_scalar_t>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI1_T::vector_size>,
          vector_ns::ctor_vector_v1_e = vector_ns::ctor_vector_v1_default>
RET_T DCO_VECTOR_INLINE operator/(const ARG1_T& v1, const ARG2_T& v2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(v1[i], v2[0]); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<TI1_T::vector_level == TI2_T::vector_level>,
          typename check_vs = std::enable_if_t<TI1_T::vector_size == TI2_T::vector_size>,
          typename OP_T = vector_ns::math::BOP_MUL<typename TI1_T::vector_scalar_t, typename TI2_T::vector_scalar_t>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI1_T::vector_size>,
          vector_ns::ctor_vector_vv_e = vector_ns::ctor_vector_vv_default>
RET_T DCO_VECTOR_INLINE operator*(const ARG1_T& v1, const ARG2_T& v2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(v1[i], v2[i]); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<(TI1_T::vector_level > TI2_T::vector_level)>,
          typename ignore_ad = std::enable_if_t<(TI2_T::is_active_type == false)>,
          typename OP_T = vector_ns::math::BOP_MUL<typename TI1_T::vector_scalar_t, ARG2_T>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI1_T::vector_size>,
          vector_ns::ctor_vector_vs_e = vector_ns::ctor_vector_vs_default>
RET_T DCO_VECTOR_INLINE operator*(const ARG1_T& v1, const ARG2_T& x2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(v1[i], x2); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<!(TI1_T::vector_level >= TI2_T::vector_level)>,
          typename ignore_ad = std::enable_if_t<(TI1_T::is_active_type == false)>,
          typename OP_T = vector_ns::math::BOP_MUL<ARG1_T, typename TI2_T::vector_scalar_t>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI2_T::vector_size>,
          vector_ns::ctor_vector_sv_e = vector_ns::ctor_vector_sv_default>
RET_T DCO_VECTOR_INLINE operator*(const ARG1_T& x1, const ARG2_T& v2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(x1, v2[i]); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<TI1_T::vector_level == TI2_T::vector_level>,
          typename check_vs = std::enable_if_t<((TI2_T::vector_size > 1) && (TI1_T::vector_size == 1))>,
          typename OP_T = vector_ns::math::BOP_MUL<typename TI1_T::vector_scalar_t, typename TI2_T::vector_scalar_t>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI2_T::vector_size>,
          vector_ns::ctor_vector_1v_e = vector_ns::ctor_vector_1v_default>
RET_T DCO_VECTOR_INLINE operator*(const ARG1_T& v1, const ARG2_T& v2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(v1[0], v2[i]); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<(TI1_T::vector_level == TI2_T::vector_level)>,
          typename check_vs = std::enable_if_t<((TI2_T::vector_size == 1) && (TI1_T::vector_size > 1))>,
          typename OP_T = vector_ns::math::BOP_MUL<typename TI1_T::vector_scalar_t, typename TI2_T::vector_scalar_t>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI1_T::vector_size>,
          vector_ns::ctor_vector_v1_e = vector_ns::ctor_vector_v1_default>
RET_T DCO_VECTOR_INLINE operator*(const ARG1_T& v1, const ARG2_T& v2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(v1[i], v2[0]); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<TI1_T::vector_level == TI2_T::vector_level>,
          typename check_vs = std::enable_if_t<TI1_T::vector_size == TI2_T::vector_size>,
          typename OP_T = vector_ns::math::BOP_POW<typename TI1_T::vector_scalar_t, typename TI2_T::vector_scalar_t>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI1_T::vector_size>,
          vector_ns::ctor_vector_vv_e = vector_ns::ctor_vector_vv_default>
RET_T DCO_VECTOR_INLINE pow(const ARG1_T& v1, const ARG2_T& v2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(v1[i], v2[i]); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<(TI1_T::vector_level > TI2_T::vector_level)>,
          typename ignore_ad = std::enable_if_t<(TI2_T::is_active_type == false)>,
          typename OP_T = vector_ns::math::BOP_POW<typename TI1_T::vector_scalar_t, ARG2_T>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI1_T::vector_size>,
          vector_ns::ctor_vector_vs_e = vector_ns::ctor_vector_vs_default>
RET_T DCO_VECTOR_INLINE pow(const ARG1_T& v1, const ARG2_T& x2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(v1[i], x2); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<!(TI1_T::vector_level >= TI2_T::vector_level)>,
          typename ignore_ad = std::enable_if_t<(TI1_T::is_active_type == false)>,
          typename OP_T = vector_ns::math::BOP_POW<ARG1_T, typename TI2_T::vector_scalar_t>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI2_T::vector_size>,
          vector_ns::ctor_vector_sv_e = vector_ns::ctor_vector_sv_default>
RET_T DCO_VECTOR_INLINE pow(const ARG1_T& x1, const ARG2_T& v2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(x1, v2[i]); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<TI1_T::vector_level == TI2_T::vector_level>,
          typename check_vs = std::enable_if_t<((TI2_T::vector_size > 1) && (TI1_T::vector_size == 1))>,
          typename OP_T = vector_ns::math::BOP_POW<typename TI1_T::vector_scalar_t, typename TI2_T::vector_scalar_t>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI2_T::vector_size>,
          vector_ns::ctor_vector_1v_e = vector_ns::ctor_vector_1v_default>
RET_T DCO_VECTOR_INLINE pow(const ARG1_T& v1, const ARG2_T& v2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(v1[0], v2[i]); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<(TI1_T::vector_level == TI2_T::vector_level)>,
          typename check_vs = std::enable_if_t<((TI2_T::vector_size == 1) && (TI1_T::vector_size > 1))>,
          typename OP_T = vector_ns::math::BOP_POW<typename TI1_T::vector_scalar_t, typename TI2_T::vector_scalar_t>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI1_T::vector_size>,
          vector_ns::ctor_vector_v1_e = vector_ns::ctor_vector_v1_default>
RET_T DCO_VECTOR_INLINE pow(const ARG1_T& v1, const ARG2_T& v2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(v1[i], v2[0]); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<TI1_T::vector_level == TI2_T::vector_level>,
          typename check_vs = std::enable_if_t<TI1_T::vector_size == TI2_T::vector_size>,
          typename OP_T = vector_ns::math::BOP_HYPOT<typename TI1_T::vector_scalar_t, typename TI2_T::vector_scalar_t>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI1_T::vector_size>,
          vector_ns::ctor_vector_vv_e = vector_ns::ctor_vector_vv_default>
RET_T DCO_VECTOR_INLINE hypot(const ARG1_T& v1, const ARG2_T& v2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(v1[i], v2[i]); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<(TI1_T::vector_level > TI2_T::vector_level)>,
          typename ignore_ad = std::enable_if_t<(TI2_T::is_active_type == false)>,
          typename OP_T = vector_ns::math::BOP_HYPOT<typename TI1_T::vector_scalar_t, ARG2_T>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI1_T::vector_size>,
          vector_ns::ctor_vector_vs_e = vector_ns::ctor_vector_vs_default>
RET_T DCO_VECTOR_INLINE hypot(const ARG1_T& v1, const ARG2_T& x2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(v1[i], x2); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<!(TI1_T::vector_level >= TI2_T::vector_level)>,
          typename ignore_ad = std::enable_if_t<(TI1_T::is_active_type == false)>,
          typename OP_T = vector_ns::math::BOP_HYPOT<ARG1_T, typename TI2_T::vector_scalar_t>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI2_T::vector_size>,
          vector_ns::ctor_vector_sv_e = vector_ns::ctor_vector_sv_default>
RET_T DCO_VECTOR_INLINE hypot(const ARG1_T& x1, const ARG2_T& v2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(x1, v2[i]); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<TI1_T::vector_level == TI2_T::vector_level>,
          typename check_vs = std::enable_if_t<((TI2_T::vector_size > 1) && (TI1_T::vector_size == 1))>,
          typename OP_T = vector_ns::math::BOP_HYPOT<typename TI1_T::vector_scalar_t, typename TI2_T::vector_scalar_t>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI2_T::vector_size>,
          vector_ns::ctor_vector_1v_e = vector_ns::ctor_vector_1v_default>
RET_T DCO_VECTOR_INLINE hypot(const ARG1_T& v1, const ARG2_T& v2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(v1[0], v2[i]); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<(TI1_T::vector_level == TI2_T::vector_level)>,
          typename check_vs = std::enable_if_t<((TI2_T::vector_size == 1) && (TI1_T::vector_size > 1))>,
          typename OP_T = vector_ns::math::BOP_HYPOT<typename TI1_T::vector_scalar_t, typename TI2_T::vector_scalar_t>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI1_T::vector_size>,
          vector_ns::ctor_vector_v1_e = vector_ns::ctor_vector_v1_default>
RET_T DCO_VECTOR_INLINE hypot(const ARG1_T& v1, const ARG2_T& v2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(v1[i], v2[0]); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<TI1_T::vector_level == TI2_T::vector_level>,
          typename check_vs = std::enable_if_t<TI1_T::vector_size == TI2_T::vector_size>,
          typename OP_T = vector_ns::math::BOP_SUB<typename TI1_T::vector_scalar_t, typename TI2_T::vector_scalar_t>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI1_T::vector_size>,
          vector_ns::ctor_vector_vv_e = vector_ns::ctor_vector_vv_default>
RET_T DCO_VECTOR_INLINE operator-(const ARG1_T& v1, const ARG2_T& v2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(v1[i], v2[i]); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<(TI1_T::vector_level > TI2_T::vector_level)>,
          typename ignore_ad = std::enable_if_t<(TI2_T::is_active_type == false)>,
          typename OP_T = vector_ns::math::BOP_SUB<typename TI1_T::vector_scalar_t, ARG2_T>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI1_T::vector_size>,
          vector_ns::ctor_vector_vs_e = vector_ns::ctor_vector_vs_default>
RET_T DCO_VECTOR_INLINE operator-(const ARG1_T& v1, const ARG2_T& x2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(v1[i], x2); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<!(TI1_T::vector_level >= TI2_T::vector_level)>,
          typename ignore_ad = std::enable_if_t<(TI1_T::is_active_type == false)>,
          typename OP_T = vector_ns::math::BOP_SUB<ARG1_T, typename TI2_T::vector_scalar_t>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI2_T::vector_size>,
          vector_ns::ctor_vector_sv_e = vector_ns::ctor_vector_sv_default>
RET_T DCO_VECTOR_INLINE operator-(const ARG1_T& x1, const ARG2_T& v2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(x1, v2[i]); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<TI1_T::vector_level == TI2_T::vector_level>,
          typename check_vs = std::enable_if_t<((TI2_T::vector_size > 1) && (TI1_T::vector_size == 1))>,
          typename OP_T = vector_ns::math::BOP_SUB<typename TI1_T::vector_scalar_t, typename TI2_T::vector_scalar_t>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI2_T::vector_size>,
          vector_ns::ctor_vector_1v_e = vector_ns::ctor_vector_1v_default>
RET_T DCO_VECTOR_INLINE operator-(const ARG1_T& v1, const ARG2_T& v2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(v1[0], v2[i]); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<(TI1_T::vector_level == TI2_T::vector_level)>,
          typename check_vs = std::enable_if_t<((TI2_T::vector_size == 1) && (TI1_T::vector_size > 1))>,
          typename OP_T = vector_ns::math::BOP_SUB<typename TI1_T::vector_scalar_t, typename TI2_T::vector_scalar_t>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI1_T::vector_size>,
          vector_ns::ctor_vector_v1_e = vector_ns::ctor_vector_v1_default>
RET_T DCO_VECTOR_INLINE operator-(const ARG1_T& v1, const ARG2_T& v2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(v1[i], v2[0]); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<TI1_T::vector_level == TI2_T::vector_level>,
          typename check_vs = std::enable_if_t<TI1_T::vector_size == TI2_T::vector_size>,
          typename OP_T = vector_ns::math::BOP_MAX<typename TI1_T::vector_scalar_t, typename TI2_T::vector_scalar_t>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI1_T::vector_size>,
          vector_ns::ctor_vector_vv_e = vector_ns::ctor_vector_vv_default>
RET_T DCO_VECTOR_INLINE max(const ARG1_T& v1, const ARG2_T& v2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(v1[i], v2[i]); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<(TI1_T::vector_level > TI2_T::vector_level)>,
          typename ignore_ad = std::enable_if_t<(TI2_T::is_active_type == false)>,
          typename OP_T = vector_ns::math::BOP_MAX<typename TI1_T::vector_scalar_t, ARG2_T>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI1_T::vector_size>,
          vector_ns::ctor_vector_vs_e = vector_ns::ctor_vector_vs_default>
RET_T DCO_VECTOR_INLINE max(const ARG1_T& v1, const ARG2_T& x2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(v1[i], x2); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<!(TI1_T::vector_level >= TI2_T::vector_level)>,
          typename ignore_ad = std::enable_if_t<(TI1_T::is_active_type == false)>,
          typename OP_T = vector_ns::math::BOP_MAX<ARG1_T, typename TI2_T::vector_scalar_t>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI2_T::vector_size>,
          vector_ns::ctor_vector_sv_e = vector_ns::ctor_vector_sv_default>
RET_T DCO_VECTOR_INLINE max(const ARG1_T& x1, const ARG2_T& v2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(x1, v2[i]); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<TI1_T::vector_level == TI2_T::vector_level>,
          typename check_vs = std::enable_if_t<((TI2_T::vector_size > 1) && (TI1_T::vector_size == 1))>,
          typename OP_T = vector_ns::math::BOP_MAX<typename TI1_T::vector_scalar_t, typename TI2_T::vector_scalar_t>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI2_T::vector_size>,
          vector_ns::ctor_vector_1v_e = vector_ns::ctor_vector_1v_default>
RET_T DCO_VECTOR_INLINE max(const ARG1_T& v1, const ARG2_T& v2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(v1[0], v2[i]); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<(TI1_T::vector_level == TI2_T::vector_level)>,
          typename check_vs = std::enable_if_t<((TI2_T::vector_size == 1) && (TI1_T::vector_size > 1))>,
          typename OP_T = vector_ns::math::BOP_MAX<typename TI1_T::vector_scalar_t, typename TI2_T::vector_scalar_t>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI1_T::vector_size>,
          vector_ns::ctor_vector_v1_e = vector_ns::ctor_vector_v1_default>
RET_T DCO_VECTOR_INLINE max(const ARG1_T& v1, const ARG2_T& v2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(v1[i], v2[0]); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<TI1_T::vector_level == TI2_T::vector_level>,
          typename check_vs = std::enable_if_t<TI1_T::vector_size == TI2_T::vector_size>,
          typename OP_T = vector_ns::math::BOP_MIN<typename TI1_T::vector_scalar_t, typename TI2_T::vector_scalar_t>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI1_T::vector_size>,
          vector_ns::ctor_vector_vv_e = vector_ns::ctor_vector_vv_default>
RET_T DCO_VECTOR_INLINE min(const ARG1_T& v1, const ARG2_T& v2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(v1[i], v2[i]); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<(TI1_T::vector_level > TI2_T::vector_level)>,
          typename ignore_ad = std::enable_if_t<(TI2_T::is_active_type == false)>,
          typename OP_T = vector_ns::math::BOP_MIN<typename TI1_T::vector_scalar_t, ARG2_T>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI1_T::vector_size>,
          vector_ns::ctor_vector_vs_e = vector_ns::ctor_vector_vs_default>
RET_T DCO_VECTOR_INLINE min(const ARG1_T& v1, const ARG2_T& x2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(v1[i], x2); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<!(TI1_T::vector_level >= TI2_T::vector_level)>,
          typename ignore_ad = std::enable_if_t<(TI1_T::is_active_type == false)>,
          typename OP_T = vector_ns::math::BOP_MIN<ARG1_T, typename TI2_T::vector_scalar_t>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI2_T::vector_size>,
          vector_ns::ctor_vector_sv_e = vector_ns::ctor_vector_sv_default>
RET_T DCO_VECTOR_INLINE min(const ARG1_T& x1, const ARG2_T& v2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(x1, v2[i]); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<TI1_T::vector_level == TI2_T::vector_level>,
          typename check_vs = std::enable_if_t<((TI2_T::vector_size > 1) && (TI1_T::vector_size == 1))>,
          typename OP_T = vector_ns::math::BOP_MIN<typename TI1_T::vector_scalar_t, typename TI2_T::vector_scalar_t>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI2_T::vector_size>,
          vector_ns::ctor_vector_1v_e = vector_ns::ctor_vector_1v_default>
RET_T DCO_VECTOR_INLINE min(const ARG1_T& v1, const ARG2_T& v2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(v1[0], v2[i]); });
}

template <typename ARG1_T, typename ARG2_T, typename TI1_T = vector_ns::traits::type_info<ARG1_T>,
          typename TI2_T = vector_ns::traits::type_info<ARG2_T>,
          typename check_vl = std::enable_if_t<(TI1_T::vector_level == TI2_T::vector_level)>,
          typename check_vs = std::enable_if_t<((TI2_T::vector_size == 1) && (TI1_T::vector_size > 1))>,
          typename OP_T = vector_ns::math::BOP_MIN<typename TI1_T::vector_scalar_t, typename TI2_T::vector_scalar_t>,
          typename RET_T = vector_ns::vector_t<typename OP_T::primal_t, TI1_T::vector_size>,
          vector_ns::ctor_vector_v1_e = vector_ns::ctor_vector_v1_default>
RET_T DCO_VECTOR_INLINE min(const ARG1_T& v1, const ARG2_T& v2) noexcept {

  return RET_T(vector_ns::ctor_lambda_t(),
               [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return OP_T::eval(v1[i], v2[0]); });
}

template <std::size_t VECTOR_SIZE, class FLOAT_T, class TOP = vector_ns::math::UOP_ACOS<FLOAT_T>,
          class TRET = vector_ns::vector_t<typename TOP::primal_t, VECTOR_SIZE>>
DCO_VECTOR_INLINE TRET acos(const vector_ns::vector_t<FLOAT_T, VECTOR_SIZE>& v) noexcept {
  return TRET(vector_ns::ctor_lambda_t(),
              [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return TOP::eval(v[i]); });
}

template <std::size_t VECTOR_SIZE, class FLOAT_T, class TOP = vector_ns::math::UOP_ASIN<FLOAT_T>,
          class TRET = vector_ns::vector_t<typename TOP::primal_t, VECTOR_SIZE>>
DCO_VECTOR_INLINE TRET asin(const vector_ns::vector_t<FLOAT_T, VECTOR_SIZE>& v) noexcept {
  return TRET(vector_ns::ctor_lambda_t(),
              [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return TOP::eval(v[i]); });
}

template <std::size_t VECTOR_SIZE, class FLOAT_T, class TOP = vector_ns::math::UOP_ATAN<FLOAT_T>,
          class TRET = vector_ns::vector_t<typename TOP::primal_t, VECTOR_SIZE>>
DCO_VECTOR_INLINE TRET atan(const vector_ns::vector_t<FLOAT_T, VECTOR_SIZE>& v) noexcept {
  return TRET(vector_ns::ctor_lambda_t(),
              [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return TOP::eval(v[i]); });
}

template <std::size_t VECTOR_SIZE, class FLOAT_T, class TOP = vector_ns::math::UOP_COS<FLOAT_T>,
          class TRET = vector_ns::vector_t<typename TOP::primal_t, VECTOR_SIZE>>
DCO_VECTOR_INLINE TRET cos(const vector_ns::vector_t<FLOAT_T, VECTOR_SIZE>& v) noexcept {
  return TRET(vector_ns::ctor_lambda_t(),
              [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return TOP::eval(v[i]); });
}

template <std::size_t VECTOR_SIZE, class FLOAT_T, class TOP = vector_ns::math::UOP_COSH<FLOAT_T>,
          class TRET = vector_ns::vector_t<typename TOP::primal_t, VECTOR_SIZE>>
DCO_VECTOR_INLINE TRET cosh(const vector_ns::vector_t<FLOAT_T, VECTOR_SIZE>& v) noexcept {
  return TRET(vector_ns::ctor_lambda_t(),
              [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return TOP::eval(v[i]); });
}

template <std::size_t VECTOR_SIZE, class FLOAT_T, class TOP = vector_ns::math::UOP_EXP<FLOAT_T>,
          class TRET = vector_ns::vector_t<typename TOP::primal_t, VECTOR_SIZE>>
DCO_VECTOR_INLINE TRET exp(const vector_ns::vector_t<FLOAT_T, VECTOR_SIZE>& v) noexcept {
  return TRET(vector_ns::ctor_lambda_t(),
              [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return TOP::eval(v[i]); });
}

template <std::size_t VECTOR_SIZE, class FLOAT_T, class TOP = vector_ns::math::UOP_LOG<FLOAT_T>,
          class TRET = vector_ns::vector_t<typename TOP::primal_t, VECTOR_SIZE>>
DCO_VECTOR_INLINE TRET log(const vector_ns::vector_t<FLOAT_T, VECTOR_SIZE>& v) noexcept {

  return TRET(vector_ns::ctor_lambda_t(),
              [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return TOP::eval(v[i]); });
}

template <std::size_t VECTOR_SIZE, class FLOAT_T, class TOP = vector_ns::math::UOP_ACOSH<FLOAT_T>,
          class TRET = vector_ns::vector_t<typename TOP::primal_t, VECTOR_SIZE>>
DCO_VECTOR_INLINE TRET acosh(const vector_ns::vector_t<FLOAT_T, VECTOR_SIZE>& v) noexcept {

  return TRET(vector_ns::ctor_lambda_t(),
              [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return TOP::eval(v[i]); });
}

template <std::size_t VECTOR_SIZE, class FLOAT_T, class TOP = vector_ns::math::UOP_ERF<FLOAT_T>,
          class TRET = vector_ns::vector_t<typename TOP::primal_t, VECTOR_SIZE>>
DCO_VECTOR_INLINE TRET erf(const vector_ns::vector_t<FLOAT_T, VECTOR_SIZE>& v) noexcept {

  return TRET(vector_ns::ctor_lambda_t(),
              [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return TOP::eval(v[i]); });
}

template <std::size_t VECTOR_SIZE, class FLOAT_T, class TOP = vector_ns::math::UOP_ERFC<FLOAT_T>,
          class TRET = vector_ns::vector_t<typename TOP::primal_t, VECTOR_SIZE>>
DCO_VECTOR_INLINE TRET erfc(const vector_ns::vector_t<FLOAT_T, VECTOR_SIZE>& v) noexcept {

  return TRET(vector_ns::ctor_lambda_t(),
              [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return TOP::eval(v[i]); });
}

template <std::size_t VECTOR_SIZE, class FLOAT_T, class TOP = vector_ns::math::UOP_ASINH<FLOAT_T>,
          class TRET = vector_ns::vector_t<typename TOP::primal_t, VECTOR_SIZE>>
DCO_VECTOR_INLINE TRET asinh(const vector_ns::vector_t<FLOAT_T, VECTOR_SIZE>& v) noexcept {

  return TRET(vector_ns::ctor_lambda_t(),
              [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return TOP::eval(v[i]); });
}

template <std::size_t VECTOR_SIZE, class FLOAT_T, class TOP = vector_ns::math::UOP_ATANH<FLOAT_T>,
          class TRET = vector_ns::vector_t<typename TOP::primal_t, VECTOR_SIZE>>
DCO_VECTOR_INLINE TRET atanh(const vector_ns::vector_t<FLOAT_T, VECTOR_SIZE>& v) noexcept {

  return TRET(vector_ns::ctor_lambda_t(),
              [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return TOP::eval(v[i]); });
}

template <std::size_t VECTOR_SIZE, class FLOAT_T, class TOP = vector_ns::math::UOP_EXPM1<FLOAT_T>,
          class TRET = vector_ns::vector_t<typename TOP::primal_t, VECTOR_SIZE>>
DCO_VECTOR_INLINE TRET expm1(const vector_ns::vector_t<FLOAT_T, VECTOR_SIZE>& v) noexcept {

  return TRET(vector_ns::ctor_lambda_t(),
              [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return TOP::eval(v[i]); });
}

template <std::size_t VECTOR_SIZE, class FLOAT_T, class TOP = vector_ns::math::UOP_LOG1P<FLOAT_T>,
          class TRET = vector_ns::vector_t<typename TOP::primal_t, VECTOR_SIZE>>
DCO_VECTOR_INLINE TRET log1p(const vector_ns::vector_t<FLOAT_T, VECTOR_SIZE>& v) noexcept {

  return TRET(vector_ns::ctor_lambda_t(),
              [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return TOP::eval(v[i]); });
}

template <std::size_t VECTOR_SIZE, class FLOAT_T, class TOP = vector_ns::math::UOP_LOG10<FLOAT_T>,
          class TRET = vector_ns::vector_t<typename TOP::primal_t, VECTOR_SIZE>>
DCO_VECTOR_INLINE TRET log10(const vector_ns::vector_t<FLOAT_T, VECTOR_SIZE>& v) noexcept {

  return TRET(vector_ns::ctor_lambda_t(),
              [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return TOP::eval(v[i]); });
}

template <std::size_t VECTOR_SIZE, class FLOAT_T, class TOP = vector_ns::math::UOP_ABS<FLOAT_T>,
          class TRET = vector_ns::vector_t<typename TOP::primal_t, VECTOR_SIZE>>
DCO_VECTOR_INLINE TRET abs(const vector_ns::vector_t<FLOAT_T, VECTOR_SIZE>& v) noexcept {
  return TRET(vector_ns::ctor_lambda_t(),
              [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return TOP::eval(v[i]); });
}

template <std::size_t VECTOR_SIZE, class FLOAT_T, class TOP = vector_ns::math::UOP_FABS<FLOAT_T>,
          class TRET = vector_ns::vector_t<typename TOP::primal_t, VECTOR_SIZE>>
DCO_VECTOR_INLINE TRET fabs(const vector_ns::vector_t<FLOAT_T, VECTOR_SIZE>& v) noexcept {
  return TRET(vector_ns::ctor_lambda_t(),
              [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return TOP::eval(v[i]); });
}

template <std::size_t VECTOR_SIZE, class FLOAT_T, class TOP = vector_ns::math::UOP_MINUS<FLOAT_T>,
          class TRET = vector_ns::vector_t<typename TOP::primal_t, VECTOR_SIZE>>
DCO_VECTOR_INLINE TRET operator-(const vector_ns::vector_t<FLOAT_T, VECTOR_SIZE>& v) noexcept {

  return TRET(vector_ns::ctor_lambda_t(),
              [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return TOP::eval(v[i]); });
}

template <std::size_t VECTOR_SIZE, class FLOAT_T, class TOP = vector_ns::math::UOP_PLUS<FLOAT_T>,
          class TRET = vector_ns::vector_t<typename TOP::primal_t, VECTOR_SIZE>>
DCO_VECTOR_INLINE TRET operator+(const vector_ns::vector_t<FLOAT_T, VECTOR_SIZE>& v) noexcept {

  return TRET(vector_ns::ctor_lambda_t(),
              [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return TOP::eval(v[i]); });
}

template <std::size_t VECTOR_SIZE, class FLOAT_T, class TOP = vector_ns::math::UOP_SIN<FLOAT_T>,
          class TRET = vector_ns::vector_t<typename TOP::primal_t, VECTOR_SIZE>>
DCO_VECTOR_INLINE TRET sin(const vector_ns::vector_t<FLOAT_T, VECTOR_SIZE>& v) noexcept {

  return TRET(vector_ns::ctor_lambda_t(),
              [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return TOP::eval(v[i]); });
}

template <std::size_t VECTOR_SIZE, class FLOAT_T, class TOP = vector_ns::math::UOP_SINH<FLOAT_T>,
          class TRET = vector_ns::vector_t<typename TOP::primal_t, VECTOR_SIZE>>
DCO_VECTOR_INLINE TRET sinh(const vector_ns::vector_t<FLOAT_T, VECTOR_SIZE>& v) noexcept {

  return TRET(vector_ns::ctor_lambda_t(),
              [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return TOP::eval(v[i]); });
}

template <std::size_t VECTOR_SIZE, class FLOAT_T, class TOP = vector_ns::math::UOP_SQRT<FLOAT_T>,
          class TRET = vector_ns::vector_t<typename TOP::primal_t, VECTOR_SIZE>>
DCO_VECTOR_INLINE TRET sqrt(const vector_ns::vector_t<FLOAT_T, VECTOR_SIZE>& v) noexcept {

  return TRET(vector_ns::ctor_lambda_t(),
              [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return TOP::eval(v[i]); });
}

template <std::size_t VECTOR_SIZE, class FLOAT_T, class TOP = vector_ns::math::UOP_TAN<FLOAT_T>,
          class TRET = vector_ns::vector_t<typename TOP::primal_t, VECTOR_SIZE>>
DCO_VECTOR_INLINE TRET tan(const vector_ns::vector_t<FLOAT_T, VECTOR_SIZE>& v) noexcept {

  return TRET(vector_ns::ctor_lambda_t(),
              [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return TOP::eval(v[i]); });
}

template <std::size_t VECTOR_SIZE, class FLOAT_T, class TOP = vector_ns::math::UOP_TANH<FLOAT_T>,
          class TRET = vector_ns::vector_t<typename TOP::primal_t, VECTOR_SIZE>>
DCO_VECTOR_INLINE TRET tanh(const vector_ns::vector_t<FLOAT_T, VECTOR_SIZE>& v) noexcept {

  return TRET(vector_ns::ctor_lambda_t(),
              [&](const std::size_t i) DCO_VECTOR_INLINE_LAMBDA { return TOP::eval(v[i]); });
}

} // namespace vector_ns
} // namespace dco
