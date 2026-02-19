#include <cmath>
#include <vector>

#include "ode/logging.hpp"
#include "ode/ode.hpp"

namespace {

double Energy(const std::vector<double>& y) {
  return 0.5 * (y[0] * y[0] + y[1] * y[1]);
}

}  // namespace

int main() {
  using State = std::vector<double>;

  auto rhs = [](double, const State& y, State& dydt) {
    dydt.resize(2);
    dydt[0] = y[1];
    dydt[1] = -y[0];
  };

  const State y0{1.0, 0.0};
  const double t0 = 0.0;
  const double t1 = 1000.0;
  const double e0 = Energy(y0);

  ode::IntegratorOptions base_opt;
  base_opt.adaptive = true;
  base_opt.rtol = 1e-4;
  base_opt.atol = 1e-7;
  base_opt.h_init = 0.5;

  const auto base = ode::integrate(ode::RKMethod::RKF78, rhs, t0, y0, t1, base_opt);
  if (base.status != ode::IntegratorStatus::Success) {
    ode::log::Error("baseline adaptive integration failed");
    return 1;
  }
  const double drift_base = std::abs(Energy(base.y) - e0) / e0;

  ode::IntegratorOptions inv_opt = base_opt;
  inv_opt.invariant_rtol = 1e-6;
  const auto inv = ode::integrate_invariant(ode::RKMethod::RKF78, rhs, Energy, t0, y0, t1, inv_opt);
  if (inv.status != ode::IntegratorStatus::Success) {
    ode::log::Error("invariant-aware integration failed, status=", ode::ToString(inv.status));
    return 1;
  }
  const double drift_inv = std::abs(Energy(inv.y) - e0) / e0;

  if (!(drift_inv < drift_base)) {
    ode::log::Error("expected invariant-aware drift to improve, base=", drift_base, " inv=", drift_inv);
    return 1;
  }
  if (inv.stats.rejected_steps <= base.stats.rejected_steps) {
    ode::log::Error("expected invariant controller to reject additional steps");
    return 1;
  }

  return 0;
}
