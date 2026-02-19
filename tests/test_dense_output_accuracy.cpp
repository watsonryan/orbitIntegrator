#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "ode/dense_events.hpp"
#include "ode/logging.hpp"

double MaxDenseError(double h_max) {
  using State = std::vector<double>;

  auto rhs = [](double t, const State&, State& dydt) {
    dydt.resize(1);
    dydt[0] = std::cos(t);
  };

  ode::IntegratorOptions opt;
  opt.adaptive = true;
  opt.rtol = 1e-12;
  opt.atol = 1e-14;
  opt.h_init = 0.2;
  opt.h_max = h_max;

  ode::DenseOutputOptions<State> dense;
  dense.record_accepted_steps = false;
  dense.uniform_sample_dt = 0.05;

  const auto out = ode::integrate_with_dense_events(ode::RKMethod::RKF78, rhs, 0.0, State{0.0}, 2.0, opt, dense, {});
  if (out.integration.status != ode::IntegratorStatus::Success) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  double max_err = 0.0;
  State ys;
  for (double t = 0.05; t <= 1.85; t += 0.1) {
    if (!out.dense.sample_linear(t, ys)) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    max_err = std::max(max_err, std::abs(ys[0] - std::sin(t)));
  }
  return max_err;
}

int main() {
  const double e1 = MaxDenseError(0.4);
  const double e2 = MaxDenseError(0.2);
  if (!std::isfinite(e1) || !std::isfinite(e2)) {
    ode::log::Error("dense output accuracy run failed");
    return 1;
  }
  if (!(e2 < e1 && e2 < 2e-2)) {
    ode::log::Error("dense output accuracy/regression check failed, e1=", e1, " e2=", e2);
    return 1;
  }
  return 0;
}
