#pragma once

#include <algorithm>
#include <cmath>

namespace ode {

struct StepSizeController {
  double safety = 0.9;
  double fac_min = 0.2;
  double fac_max = 5.0;

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

}  // namespace ode
