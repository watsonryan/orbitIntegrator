#include <cmath>
#include <vector>

#include "ode/logging.hpp"
#include "ode/poincare.hpp"

int main() {
  using State = std::vector<double>;
  constexpr double kPi = 3.14159265358979323846;

  auto rhs = [](double, const State& y, State& dydt) {
    dydt.resize(2);
    dydt[0] = y[1];
    dydt[1] = -y[0];
  };
  auto section = [](double, const State& y) {
    return y[0];
  };

  {
    ode::IntegratorOptions opt;
    opt.adaptive = true;
    opt.rtol = 1e-10;
    opt.atol = 1e-12;
    opt.h_init = 0.05;

    const State y0{1.0, 0.0};
    const auto res = ode::poincare::integrate_poincare<State>(
        ode::RKMethod::RKF78, rhs, section, 0.0, y0, 20.0 * kPi, opt, 3, ode::poincare::CrossingDirection::Positive);
    if (res.status != ode::IntegratorStatus::Success || res.crossings.size() != 3) {
      ode::log::Error("poincare crossing extraction failed");
      return 1;
    }

    for (std::size_t k = 0; k < res.crossings.size(); ++k) {
      const double t_expected = 1.5 * kPi + 2.0 * kPi * static_cast<double>(k);
      if (std::abs(res.crossings[k].t - t_expected) > 5e-3) {
        ode::log::Error("poincare crossing time mismatch");
        return 1;
      }
      if (std::abs(res.crossings[k].y[0]) > 5e-4) {
        ode::log::Error("poincare crossing not on section");
        return 1;
      }
    }
  }

  {
    auto map = [](const State& x) {
      return State{
          x[0] + 0.2 * (x[0] - 1.0),
          x[1] + 0.3 * (x[1] + 2.0),
      };
    };

    const State guess{2.0, -1.0};
    const auto corr = ode::poincare::differential_correction<State>(map, guess);
    if (!corr.success) {
      ode::log::Error("differential correction failed");
      return 1;
    }
    if (corr.residual_inf > 1e-9) {
      ode::log::Error("differential correction residual too large");
      return 1;
    }
    if (std::abs(corr.corrected[0] - 1.0) > 1e-6 || std::abs(corr.corrected[1] + 2.0) > 1e-6) {
      ode::log::Error("differential correction fixed point mismatch");
      return 1;
    }
  }

  return 0;
}
