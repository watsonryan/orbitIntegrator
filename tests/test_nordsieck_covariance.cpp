#include <cmath>
#include <vector>

#include "ode/logging.hpp"
#include "ode/ode.hpp"

int main() {
  using State = ode::variational::State;
  using Matrix = ode::variational::Matrix;

  auto dynamics = [](double, const State& x, State& dxdt) {
    dxdt.assign(1, -0.1 * x[0]);
  };
  auto jac = [](double, const State&, Matrix& a) {
    a.assign(1, -0.1);
    return true;
  };
  auto q = [](double, const State&, Matrix& q_out) {
    q_out.assign(1, 0.01);
    return true;
  };

  const State x0{1.0};
  const Matrix p0{0.5};
  const double tf = 10.0;

  ode::IntegratorOptions rk_opt;
  rk_opt.adaptive = true;
  rk_opt.rtol = 1e-11;
  rk_opt.atol = 1e-14;
  rk_opt.h_init = 1e-3;
  rk_opt.h_max = 0.2;

  ode::multistep::NordsieckAbmOptions nopt;
  nopt.rtol = 1e-10;
  nopt.atol = 1e-13;
  nopt.h_init = 0.01;
  nopt.h_min = 1e-8;
  nopt.h_max = 0.2;
  nopt.segment_steps = 8;

  const auto rk = ode::variational::integrate_state_stm_cov(
      ode::RKMethod::RKF78, dynamics, jac, q, 0.0, x0, p0, tf, rk_opt);
  if (rk.status != ode::IntegratorStatus::Success) {
    ode::log::Error("RKF78 variational covariance failed");
    return 1;
  }

  const auto nord = ode::variational::integrate_state_stm_cov_nordsieck_abm6(
      dynamics, jac, q, 0.0, x0, p0, tf, nopt);
  if (nord.status != ode::IntegratorStatus::Success) {
    ode::log::Error("Nordsieck ABM6 variational covariance failed");
    return 1;
  }

  const double x_ref = std::exp(-0.1 * tf);
  const double phi_ref = std::exp(-0.1 * tf);
  const double p_ref = p0[0] * std::exp(-0.2 * tf) + 0.05 * (1.0 - std::exp(-0.2 * tf));

  const double err_n_x = std::abs(nord.x[0] - x_ref);
  const double err_n_phi = std::abs(nord.phi[0] - phi_ref);
  const double err_n_p = std::abs(nord.p[0] - p_ref);

  if (err_n_x > 5e-8 || err_n_phi > 5e-8 || err_n_p > 1e-7) {
    ode::log::Error("Nordsieck ABM6 absolute covariance errors too large: ", err_n_x, ", ", err_n_phi, ", ", err_n_p);
    return 1;
  }

  if (std::abs(nord.x[0] - rk.x[0]) > 5e-8 ||
      std::abs(nord.phi[0] - rk.phi[0]) > 5e-8 ||
      std::abs(nord.p[0] - rk.p[0]) > 1e-7) {
    ode::log::Error("Nordsieck ABM6 does not match RKF78 covariance solution closely enough");
    return 1;
  }

  return 0;
}
