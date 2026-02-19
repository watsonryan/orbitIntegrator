#include <cmath>
#include <vector>

#include "ode/logging.hpp"
#include "ode/ode.hpp"

int main() {
  using State = ode::variational::State;
  using Matrix = ode::variational::Matrix;

  auto dynamics = [](double, const auto& x, auto& dxdt) {
    dxdt.resize(1);
    dxdt[0] = -0.1 * x[0];
  };
  auto jac = [](double, const State& x, Matrix& a) {
    (void)x;
    a.assign(1, -0.1);
    return true;
  };
  auto q = [](double, const State& x, Matrix& q_out) {
    (void)x;
    q_out.assign(1, 0.01);
    return true;
  };

  ode::IntegratorOptions opt;
  opt.rtol = 1e-12;
  opt.atol = 1e-14;
  opt.h_init = 1e-3;

  State x0{1.0};
  Matrix p0{0.5};

  const auto stm = ode::variational::integrate_state_stm(ode::RKMethod::RKF78, dynamics, jac, 0.0, x0, 10.0, opt);
  if (stm.status != ode::IntegratorStatus::Success) {
    ode::log::Error("variational STM wrapper failed");
    return 1;
  }
  if (std::abs(stm.x[0] - std::exp(-1.0)) > 1e-10) {
    ode::log::Error("variational STM state mismatch");
    return 1;
  }

  const auto cov = ode::variational::integrate_state_stm_cov(
      ode::RKMethod::RKF78, dynamics, jac, q, 0.0, x0, p0, 10.0, opt);
  if (cov.status != ode::IntegratorStatus::Success) {
    ode::log::Error("variational cov wrapper failed");
    return 1;
  }
  const double p_ref = p0[0] * std::exp(-2.0) + 0.05 * (1.0 - std::exp(-2.0));
  if (std::abs(cov.p[0] - p_ref) > 1e-9) {
    ode::log::Error("variational covariance mismatch p=", cov.p[0], " ref=", p_ref);
    return 1;
  }

  return 0;
}
