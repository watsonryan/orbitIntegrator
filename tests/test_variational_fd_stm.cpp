#include <cmath>
#include <vector>

#include "ode/logging.hpp"
#include "ode/ode.hpp"
#include "ode/variational.hpp"

int main() {
  using State = ode::variational::State;
  using Matrix = ode::variational::Matrix;

  auto rhs = [](double, const State& x, State& dxdt) {
    dxdt.resize(2);
    dxdt[0] = x[1];
    dxdt[1] = -std::sin(x[0]);
  };
  auto jac = [](double, const State& x, Matrix& a) {
    a.assign(4, 0.0);
    a[1] = 1.0;               // d xdot / d v
    a[2] = -std::cos(x[0]);   // d vdot / d x
    return true;
  };

  ode::IntegratorOptions opt;
  opt.rtol = 1e-11;
  opt.atol = 1e-13;
  opt.h_init = 1e-3;

  const State x0{0.3, -0.2};
  const double t1 = 5.0;
  const auto stm = ode::variational::integrate_state_stm(ode::RKMethod::RKF78, rhs, jac, 0.0, x0, t1, opt);
  if (stm.status != ode::IntegratorStatus::Success) {
    ode::log::Error("variational fd/stm reference run failed");
    return 1;
  }

  const double eps = 1e-7;
  const State d0{eps, -2.0 * eps};
  State x0p = x0;
  x0p[0] += d0[0];
  x0p[1] += d0[1];

  const auto ref = ode::integrate(ode::RKMethod::RKF78, rhs, 0.0, x0, t1, opt);
  const auto per = ode::integrate(ode::RKMethod::RKF78, rhs, 0.0, x0p, t1, opt);
  if (ref.status != ode::IntegratorStatus::Success || per.status != ode::IntegratorStatus::Success) {
    ode::log::Error("variational fd/stm trajectory runs failed");
    return 1;
  }

  const State delta_true{per.y[0] - ref.y[0], per.y[1] - ref.y[1]};
  const State delta_lin{
      stm.phi[0] * d0[0] + stm.phi[1] * d0[1],
      stm.phi[2] * d0[0] + stm.phi[3] * d0[1]};

  const double num = std::hypot(delta_true[0] - delta_lin[0], delta_true[1] - delta_lin[1]);
  const double den = std::max(1e-16, std::hypot(delta_true[0], delta_true[1]));
  const double rel = num / den;
  if (rel > 3e-4) {
    ode::log::Error("variational FD-vs-STM mismatch, rel=", rel);
    return 1;
  }

  return 0;
}
