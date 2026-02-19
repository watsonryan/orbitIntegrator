#include <cmath>
#include <limits>
#include <vector>

#include "ode/logging.hpp"
#include "ode/ode.hpp"

namespace {

double SolveFixed(ode::RKMethod method, double h) {
  using State = std::vector<double>;
  auto rhs = [](double, const State& y, State& dydt) {
    dydt.resize(1);
    dydt[0] = y[0];
  };
  State y0{1.0};
  ode::IntegratorOptions opt;
  opt.adaptive = false;
  opt.fixed_h = h;
  const auto res = ode::integrate(method, rhs, 0.0, y0, 1.0, opt);
  if (res.status != ode::IntegratorStatus::Success) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return res.y[0];
}

double EstimateOrder(double y_h, double y_h2, double y_h4, double y_ref) {
  const double e1 = std::abs(y_h - y_ref);
  const double e2 = std::abs(y_h2 - y_ref);
  const double e4 = std::abs(y_h4 - y_ref);
  if (!(e1 > 0.0 && e2 > 0.0 && e4 > 0.0)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return std::log2(e1 / e2);
}

}  // namespace

int main() {
  const double y_ref = std::exp(1.0);

  {
    const double y_h = SolveFixed(ode::RKMethod::RK4, 0.2);
    const double y_h2 = SolveFixed(ode::RKMethod::RK4, 0.1);
    const double y_h4 = SolveFixed(ode::RKMethod::RK4, 0.05);
    const double p = EstimateOrder(y_h, y_h2, y_h4, y_ref);
    if (!std::isfinite(p) || p < 3.7 || p > 4.3) {
      ode::log::Error("RK4 Richardson order check failed, p=", p);
      return 1;
    }
  }

  {
    const double y_h = SolveFixed(ode::RKMethod::RK8, 0.2);
    const double y_h2 = SolveFixed(ode::RKMethod::RK8, 0.1);
    const double y_h4 = SolveFixed(ode::RKMethod::RK8, 0.05);
    const double p = EstimateOrder(y_h, y_h2, y_h4, y_ref);
    if (!std::isfinite(p) || p < 7.0 || p > 9.0) {
      ode::log::Error("RK8 Richardson order check failed, p=", p);
      return 1;
    }
  }

  return 0;
}
