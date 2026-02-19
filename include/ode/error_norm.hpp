#pragma once

#include <cmath>
#include <cstddef>

#include "ode/algebra.hpp"

namespace ode {

template <class State, class Algebra = DefaultAlgebra<State>>
requires AlgebraFor<Algebra, State>
[[nodiscard]] double weighted_rms_error(const State& err,
                                        const State& y,
                                        const State& y_high,
                                        double atol,
                                        double rtol) {
  const std::size_t n = Algebra::size(y);
  if (n == 0) {
    return 0.0;
  }
  double acc = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double scale = atol + rtol * std::max(std::abs(y[i]), std::abs(y_high[i]));
    const double ratio = err[i] / scale;
    acc += ratio * ratio;
  }
  return std::sqrt(acc / static_cast<double>(n));
}

}  // namespace ode
