/**
 * @file algebra.hpp
 * @brief State algebra adapter and concept for generic container support.
 */
#pragma once

#include <cmath>
#include <concepts>
#include <cstddef>

namespace ode {

/** @brief Default algebra implementation for indexable containers of doubles. */
template <class State>
struct DefaultAlgebra {
  static std::size_t size(const State& x) { return x.size(); }

  static void resize_like(State& x, const State& ref) {
    if constexpr (requires { x.resize(ref.size()); }) {
      x.resize(ref.size());
    }
  }

  static void assign(State& dst, const State& src) { dst = src; }

  static void set_zero(State& x) {
    for (auto& v : x) {
      v = 0.0;
    }
  }

  static void axpy(double a, const State& x, State& y) {
    const auto n = size(y);
    for (std::size_t i = 0; i < n; ++i) {
      y[i] += a * x[i];
    }
  }

  static bool finite(const State& x) {
    for (const auto v : x) {
      if (!std::isfinite(v)) {
        return false;
      }
    }
    return true;
  }
};

/** @brief Concept describing the algebra operations required by the RK engine. */
template <class Algebra, class State>
concept AlgebraFor = requires(State a, const State b, double s) {
  { Algebra::size(a) } -> std::convertible_to<std::size_t>;
  { Algebra::resize_like(a, b) };
  { Algebra::assign(a, b) };
  { Algebra::set_zero(a) };
  { Algebra::axpy(s, b, a) };
  { Algebra::finite(b) } -> std::convertible_to<bool>;
};

}  // namespace ode
