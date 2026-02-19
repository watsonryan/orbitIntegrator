#include <cmath>
#include <iostream>
#include <vector>

#include "ode/dense_events.hpp"

int main() {
  using State = std::vector<double>;

  const State y0{0.0};
  auto rhs = [](double, const State&, State& dydt) {
    dydt.resize(1);
    dydt[0] = 1.0;
  };

  ode::IntegratorOptions opt;
  opt.adaptive = true;
  opt.rtol = 1e-10;
  opt.atol = 1e-12;
  opt.h_init = 0.1;

  ode::DenseOutputOptions<State> dense_opt;
  dense_opt.record_accepted_steps = true;
  dense_opt.uniform_sample_dt = 0.05;

  ode::EventOptions<State> event_opt;
  event_opt.function = [](double, const State& y) { return y[0] - 0.5; };
  event_opt.direction = ode::EventDirection::Rising;
  event_opt.terminal = true;

  const auto out = ode::integrate_with_dense_events(ode::RKMethod::RKF45, rhs, 0.0, y0, 1.0, opt, dense_opt, event_opt);
  if (out.integration.status != ode::IntegratorStatus::Success) {
    std::cerr << "dense/event integration failed\n";
    return 1;
  }
  if (out.events.empty()) {
    std::cerr << "expected at least one event\n";
    return 1;
  }
  if (std::abs(out.events.front().t - 0.5) > 1e-2) {
    std::cerr << "event time mismatch: " << out.events.front().t << "\n";
    return 1;
  }

  State ys{};
  if (!out.dense.sample_linear(0.25, ys)) {
    std::cerr << "sample_linear failed\n";
    return 1;
  }
  if (std::abs(ys[0] - 0.25) > 2e-2) {
    std::cerr << "dense sample mismatch: " << ys[0] << "\n";
    return 1;
  }

  return 0;
}
