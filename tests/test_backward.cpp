#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "ode/ode.hpp"
#include "ode/logging.hpp"

int main() {
  using State = std::vector<double>;

  State y1{std::exp(1.0)};
  auto rhs = [](double, const State& y, State& dydt) {
    dydt.resize(y.size());
    dydt[0] = y[0];
  };

  ode::IntegratorOptions opt;
  opt.adaptive = true;
  opt.rtol = 1e-11;
  opt.atol = 1e-14;
  opt.h_init = 0.2;

  const auto res = ode::integrate(ode::RKMethod::RKF78, rhs, 1.0, y1, 0.0, opt);
  if (res.status != ode::IntegratorStatus::Success) {
    ode::log::Error("backward integration failed");
    return 1;
  }

  if (std::abs(res.y[0] - 1.0) > 1e-10) {
    ode::log::Error("backward result mismatch: ", res.y[0]);    return 1;
  }

  return 0;
}
