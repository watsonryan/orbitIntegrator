#include <array>
#include <cmath>
#include <iostream>

#include "ode/ode.hpp"

int main() {
  using State = std::array<double, 2>;  // x, v

  const State y0{1.0, 0.0};
  auto rhs = [](double, const State& y, State& dydt) {
    dydt[0] = y[1];
    dydt[1] = -y[0];
  };

  ode::IntegratorOptions opt;
  opt.adaptive = true;
  opt.rtol = 1e-11;
  opt.atol = 1e-13;
  opt.h_init = 0.05;

  const double t1 = 2.0 * 3.14159265358979323846;
  const auto res = ode::integrate(ode::RKMethod::RKF78, rhs, 0.0, y0, t1, opt);
  if (res.status != ode::IntegratorStatus::Success) {
    std::cerr << "array state integration failed\n";
    return 1;
  }

  if (std::abs(res.y[0] - 1.0) > 1e-9 || std::abs(res.y[1]) > 1e-9) {
    std::cerr << "array state final mismatch: x=" << res.y[0] << " v=" << res.y[1] << "\n";
    return 1;
  }

  return 0;
}
