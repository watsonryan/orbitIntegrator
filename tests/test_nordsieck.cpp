#include <cmath>
#include <iostream>
#include <vector>

#include "ode/multistep/nordsieck_abm4.hpp"

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
    std::cerr << "nordsieck integration failed\n";
    return 1;
  }

  const double err = std::abs(res.y[0] - std::exp(1.0));
  if (err > 5e-5) {
    std::cerr << "nordsieck error too large: " << err << "\n";
    return 1;
  }

  if (res.stats.accepted_steps <= 0 || res.stats.rhs_evals <= 0) {
    std::cerr << "nordsieck stats invalid\n";
    return 1;
  }

  return 0;
}
