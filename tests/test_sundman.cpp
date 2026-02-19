#include <cmath>
#include <iostream>
#include <vector>

#include "ode/ode.hpp"
#include "ode/logging.hpp"

int main() {
  using State = std::vector<double>;

  const State y0{1.0};
  auto rhs = [](double, const State& y, State& dydt) {
    dydt.resize(1);
    dydt[0] = y[0];
  };

  // Baseline standard integration.
  ode::IntegratorOptions opt;
  opt.adaptive = true;
  opt.rtol = 1e-10;
  opt.atol = 1e-12;
  opt.h_init = 0.1;

  const auto ref = ode::integrate(ode::RKMethod::RKF78, rhs, 0.0, y0, 1.0, opt);
  if (ref.status != ode::IntegratorStatus::Success) {
    ode::log::Error("reference integration failed");
    return 1;
  }

  // Sundman with constant dt/ds should match standard stepping behavior numerically.
  auto dt_ds_const = [](double, const State&) { return 1.0; };
  const auto sund_const = ode::integrate_sundman(ode::RKMethod::RKF78, rhs, dt_ds_const, 0.0, y0, 1.0, opt);
  if (sund_const.status != ode::IntegratorStatus::Success) {
    ode::log::Error("sundman const integration failed");
    return 1;
  }

  if (std::abs(ref.y[0] - sund_const.y[0]) > 1e-9) {
    ode::log::Error("sundman const mismatch: ref=", ref.y[0], " sund=", sund_const.y[0]);    return 1;
  }

  // Non-constant positive dt/ds still reaches the same endpoint solution.
  auto dt_ds_var = [](double t, const State&) { return 0.5 + 0.5 * t; };
  const auto sund_var = ode::integrate_sundman(ode::RKMethod::RKF45, rhs, dt_ds_var, 0.0, y0, 1.0, opt);
  if (sund_var.status != ode::IntegratorStatus::Success) {
    ode::log::Error("sundman variable integration failed");
    return 1;
  }
  if (std::abs(sund_var.y[0] - std::exp(1.0)) > 2e-6) {
    ode::log::Error("sundman variable mismatch: ", sund_var.y[0]);    return 1;
  }

  return 0;
}
