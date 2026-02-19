/**
 * @file controller.hpp
 * @brief Adaptive step-size controller for embedded RK methods.
 */
#pragma once

#include <algorithm>
#include <cmath>

namespace ode {

/** @brief Safety-clamped power-law step-size update controller. */
struct StepSizeController {
  double safety = 0.9;
  double fac_min = 0.2;
  double fac_max = 5.0;

  /** @brief Propose next step size from current step and normalized error. */
  [[nodiscard]] double propose(double h, double err_norm, int order_high) const {
    if (err_norm <= 0.0) {
      return h * fac_max;
    }
    const double p = static_cast<double>(order_high);
    double fac = safety * std::pow(err_norm, -1.0 / p);
    if (!std::isfinite(fac)) {
      fac = fac_min;
    }
    fac = std::clamp(fac, fac_min, fac_max);
    return h * fac;
  }
};

/** @brief Adaptive update controller that combines LTE and invariant-drift constraints. */
struct InvariantStepSizeController {
  double safety = 0.9;
  double fac_min = 0.2;
  double fac_max = 5.0;
  double invariant_safety = 0.9;

  /** @brief Propose next step from LTE and invariant normalized error signals. */
  [[nodiscard]] double propose(double h, double err_norm, double invariant_norm, int order_high) const {
    const StepSizeController lte_controller{safety, fac_min, fac_max};
    const double h_lte = lte_controller.propose(h, err_norm, order_high);
    double fac_inv = fac_max;
    if (invariant_norm > 0.0) {
      fac_inv = invariant_safety * std::pow(invariant_norm, -0.5);
      if (!std::isfinite(fac_inv)) {
        fac_inv = fac_min;
      }
      fac_inv = std::clamp(fac_inv, fac_min, fac_max);
    }
    return h * std::min(std::abs(h_lte / h), fac_inv);
  }
};

}  // namespace ode
