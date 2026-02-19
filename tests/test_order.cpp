#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "ode/ode.hpp"

namespace {

using State = std::vector<double>;

[[nodiscard]] double run_fixed(const ode::RKMethod method, const double h) {
  State y0{1.0};
  auto rhs = [](double, const State& y, State& dydt) {
    dydt.resize(y.size());
    dydt[0] = y[0];
  };

  ode::IntegratorOptions opt;
  opt.adaptive = false;
  opt.fixed_h = h;
  opt.max_steps = 2000000;

  const auto res = ode::integrate(method, rhs, 0.0, y0, 1.0, opt);
  if (res.status != ode::IntegratorStatus::Success) {
    std::cerr << "fixed integration failed\n";
    std::exit(1);
  }
  return std::abs(res.y[0] - std::exp(1.0));
}

[[nodiscard]] double estimate_order(const double e_h, const double e_h2) {
  return std::log2(e_h / e_h2);
}

void expect_range(const char* label, const double value, const double lo, const double hi) {
  if (!(value > lo && value < hi)) {
    std::cerr << label << " out of range: " << value << " expected in (" << lo << ", " << hi << ")\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  {
    const double e1 = run_fixed(ode::RKMethod::RK4, 0.2);
    const double e2 = run_fixed(ode::RKMethod::RK4, 0.1);
    const double p = estimate_order(e1, e2);
    expect_range("RK4 order", p, 3.7, 4.3);
  }

  {
    const double e1 = run_fixed(ode::RKMethod::RKF45, 0.2);
    const double e2 = run_fixed(ode::RKMethod::RKF45, 0.1);
    const double p = estimate_order(e1, e2);
    expect_range("RKF45(high) order", p, 4.6, 5.4);
  }

  {
    const double e1 = run_fixed(ode::RKMethod::RKF78, 0.2);
    const double e2 = run_fixed(ode::RKMethod::RKF78, 0.1);
    const double p = estimate_order(e1, e2);
    expect_range("RKF78(high) order", p, 7.2, 8.8);
  }

  {
    const double e1 = run_fixed(ode::RKMethod::RK8, 0.2);
    const double e2 = run_fixed(ode::RKMethod::RK8, 0.1);
    const double p = estimate_order(e1, e2);
    expect_range("RK8 order", p, 7.2, 8.8);
  }

  return 0;
}
