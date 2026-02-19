#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "ode/ode.hpp"

namespace {

using State = std::vector<double>;

struct RunSummary {
  double abs_err;
  ode::IntegratorResult<State> result;
};

[[nodiscard]] RunSummary run(const ode::RKMethod method, const double tol) {
  State y0{1.0};
  auto rhs = [](double, const State& y, State& dydt) {
    dydt.resize(y.size());
    dydt[0] = y[0];
  };

  ode::IntegratorOptions opt;
  opt.adaptive = true;
  opt.rtol = tol;
  opt.atol = tol * 1e-3;
  opt.h_init = 0.5;
  opt.h_min = 1e-14;
  opt.h_max = 1.0;
  opt.max_steps = 1000000;

  auto res = ode::integrate(method, rhs, 0.0, y0, 1.0, opt);
  if (res.status != ode::IntegratorStatus::Success) {
    std::cerr << "adaptive integration failed\n";
    std::exit(1);
  }

  return RunSummary{std::abs(res.y[0] - std::exp(1.0)), std::move(res)};
}

void check_monotonic(const char* label, const double e1, const double e2, const double e3) {
  if (!(e2 < e1 && e3 < e2)) {
    std::cerr << label << " error did not shrink monotonically: " << e1 << " " << e2 << " " << e3 << "\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  {
    const auto a = run(ode::RKMethod::RKF45, 1e-6);
    const auto b = run(ode::RKMethod::RKF45, 1e-9);
    const auto c = run(ode::RKMethod::RKF45, 1e-12);
    check_monotonic("RKF45", a.abs_err, b.abs_err, c.abs_err);
    if (c.result.stats.rejected_steps == 0) {
      std::cerr << "RKF45 expected at least one rejected step for strict tolerance\n";
      return 1;
    }
  }

  {
    const auto a = run(ode::RKMethod::RKF78, 1e-6);
    const auto b = run(ode::RKMethod::RKF78, 1e-9);
    const auto c = run(ode::RKMethod::RKF78, 1e-12);
    check_monotonic("RKF78", a.abs_err, b.abs_err, c.abs_err);
  }

  return 0;
}
