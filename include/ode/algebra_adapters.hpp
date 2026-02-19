/**
 * @file algebra_adapters.hpp
 * @brief Optional algebra adapters for fixed-size arrays, Eigen vectors, and custom accessors.
 */
#pragma once

#include <array>
#include <cmath>
#include <cstddef>

namespace ode {

template <std::size_t N>
struct StdArrayAlgebra {
  using State = std::array<double, N>;

  static constexpr std::size_t size(const State&) { return N; }
  static void resize_like(State&, const State&) {}
  static void assign(State& dst, const State& src) { dst = src; }

  static void set_zero(State& x) {
    x.fill(0.0);
  }

  static void axpy(double a, const State& x, State& y) {
    for (std::size_t i = 0; i < N; ++i) {
      y[i] += a * x[i];
    }
  }

  static bool finite(const State& x) {
    for (double v : x) {
      if (!std::isfinite(v)) {
        return false;
      }
    }
    return true;
  }
};

template <class Accessor, class State>
struct AccessorAlgebra {
  static std::size_t size(const State& x) { return Accessor::size(x); }
  static void resize_like(State& x, const State& ref) { Accessor::resize_like(x, ref); }
  static void assign(State& dst, const State& src) { Accessor::assign(dst, src); }
  static void set_zero(State& x) { Accessor::set_zero(x); }
  static void axpy(double a, const State& x, State& y) { Accessor::axpy(a, x, y); }
  static bool finite(const State& x) { return Accessor::finite(x); }
};

#if __has_include(<Eigen/Core>)
#include <Eigen/Core>

template <class T>
struct EigenVectorAlgebra;

template <class Scalar, int Rows, int Options, int MaxRows, int MaxCols>
struct EigenVectorAlgebra<Eigen::Matrix<Scalar, Rows, 1, Options, MaxRows, MaxCols>> {
  using State = Eigen::Matrix<Scalar, Rows, 1, Options, MaxRows, MaxCols>;

  static std::size_t size(const State& x) { return static_cast<std::size_t>(x.size()); }
  static void resize_like(State& x, const State& ref) {
    if constexpr (Rows == Eigen::Dynamic) {
      x.resize(ref.size());
    }
  }
  static void assign(State& dst, const State& src) { dst = src; }
  static void set_zero(State& x) { x.setZero(); }

  static void axpy(double a, const State& x, State& y) {
    y += static_cast<Scalar>(a) * x;
  }

  static bool finite(const State& x) {
    for (int i = 0; i < x.size(); ++i) {
      if (!std::isfinite(static_cast<double>(x(i)))) {
        return false;
      }
    }
    return true;
  }
};
#endif

}  // namespace ode
