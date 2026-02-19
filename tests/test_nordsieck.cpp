#include <cmath>
#include <iostream>
#include <vector>

#include "ode/multistep/nordsieck_abm4.hpp"
#include "ode/logging.hpp"

int main() {
  using State = std::vector<double>;

  const State y0{1.0};
  auto rhs = [](double, const State& y, State& dydt) {
    dydt.resize(1);
    dydt[0] = y[0];
  };

  ode::multistep::NordsieckAbmOptions opt;
  opt.rtol = 1e-8;
  opt.atol = 1e-12;
  opt.h_init = 0.01;
  opt.h_min = 1e-8;
  opt.h_max = 0.1;

  const auto res = ode::multistep::integrate_nordsieck_abm4(rhs, 0.0, y0, 1.0, opt);
  if (res.status != ode::IntegratorStatus::Success) {
    ode::log::Error("nordsieck integration failed");
    return 1;
  }

  const double err = std::abs(res.y[0] - std::exp(1.0));
  if (err > 5e-5) {
    ode::log::Error("nordsieck error too large: ", err);    return 1;
  }

  if (res.stats.accepted_steps <= 0 || res.stats.rhs_evals <= 0) {
    ode::log::Error("nordsieck stats invalid");
    return 1;
  }

  return 0;
}
