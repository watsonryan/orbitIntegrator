#include <cmath>
#include <vector>

#include "ode/logging.hpp"
#include "ode/ode.hpp"

int main() {
  using State = std::vector<double>;

  auto rhs = [](double, const State& y, State& dydt) {
    dydt.resize(1);
    dydt[0] = y[0];
  };

  std::vector<ode::BatchTask<State>> tasks{
      {.t0 = 0.0, .t1 = 1.0, .y0 = {1.0}},
      {.t0 = 0.0, .t1 = 2.0, .y0 = {2.0}},
      {.t0 = 1.0, .t1 = 0.0, .y0 = {std::exp(1.0)}}};

  ode::IntegratorOptions opt;
  opt.adaptive = true;
  opt.rtol = 1e-11;
  opt.atol = 1e-13;
  opt.h_init = 0.01;

  ode::BatchWorkspace<State> ws;
  ws.Reserve(tasks.size());
  const auto results = ode::integrate_batch(ode::RKMethod::RKF78, rhs, tasks, opt, &ws);
  if (results.size() != tasks.size()) {
    ode::log::Error("batch results size mismatch");
    return 1;
  }
  if (results[0].status != ode::IntegratorStatus::Success ||
      std::abs(results[0].y[0] - std::exp(1.0)) > 1e-9) {
    ode::log::Error("batch case 0 mismatch");
    return 1;
  }
  if (results[1].status != ode::IntegratorStatus::Success ||
      std::abs(results[1].y[0] - 2.0 * std::exp(2.0)) > 1e-8) {
    ode::log::Error("batch case 1 mismatch");
    return 1;
  }
  if (results[2].status != ode::IntegratorStatus::Success ||
      std::abs(results[2].y[0] - 1.0) > 1e-9) {
    ode::log::Error("batch case 2 mismatch");
    return 1;
  }

  std::vector<ode::IntegratorResult<State>> out;
  ode::integrate_batch_inplace(ode::RKMethod::RKF78, rhs, tasks, opt, out);
  if (out.size() != tasks.size()) {
    ode::log::Error("batch inplace size mismatch");
    return 1;
  }

  return 0;
}

