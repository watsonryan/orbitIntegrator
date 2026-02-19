#include <cmath>
#include <vector>

#include "ode/logging.hpp"
#include "ode/stiff/diagnostics.hpp"

int main() {
  {
    auto jac = [](double, const std::vector<double>& x, std::vector<double>& j) {
      (void)x;
      j.assign(1, -1000.0);
      return true;
    };
    const auto rep = ode::stiff::assess_stiffness(jac, 0.0, std::vector<double>{1.0}, 0.01, 2.0);
    if (!rep.likely_stiff || rep.step_stiffness_ratio < 9.9) {
      ode::log::Error("stiff diagnostics failed for stiff sample");
      return 1;
    }
  }

  {
    auto jac = [](double, const std::vector<double>& x, std::vector<double>& j) {
      (void)x;
      j.assign(1, -0.1);
      return true;
    };
    const auto rep = ode::stiff::assess_stiffness(jac, 0.0, std::vector<double>{1.0}, 0.1, 2.0);
    if (rep.likely_stiff) {
      ode::log::Error("stiff diagnostics false positive for nonstiff sample");
      return 1;
    }
  }

  return 0;
}

