/**
 * @file diagnostics.hpp
 * @brief Lightweight stiffness diagnostics based on user-supplied Jacobian.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace ode::stiff {

struct StiffnessReport {
  double jacobian_inf_norm = 0.0;
  double step_stiffness_ratio = 0.0;  // h * ||J||_inf
  bool likely_stiff = false;
};

namespace detail {

[[nodiscard]] inline double InfNormRowMajor(const std::vector<double>& a, std::size_t n) {
  double max_row = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    double sum = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      sum += std::abs(a[i * n + j]);
    }
    max_row = std::max(max_row, sum);
  }
  return max_row;
}

}  // namespace detail

/**
 * @brief Assess stiffness using a Jacobian and current step size.
 *
 * Heuristic criterion: likely stiff if `h * ||J||_inf > threshold`.
 */
template <class JacobianFn>
[[nodiscard]] inline StiffnessReport assess_stiffness(JacobianFn&& jacobian_fn,
                                                       double t,
                                                       const std::vector<double>& x,
                                                       double h,
                                                       double threshold = 2.0) {
  StiffnessReport out{};
  std::vector<double> j;
  if (!jacobian_fn(t, x, j)) {
    return out;
  }
  const std::size_t n = x.size();
  if (j.size() != n * n || n == 0 || !std::isfinite(h)) {
    return out;
  }
  out.jacobian_inf_norm = detail::InfNormRowMajor(j, n);
  out.step_stiffness_ratio = std::abs(h) * out.jacobian_inf_norm;
  out.likely_stiff = std::isfinite(out.step_stiffness_ratio) && (out.step_stiffness_ratio > threshold);
  return out;
}

}  // namespace ode::stiff

