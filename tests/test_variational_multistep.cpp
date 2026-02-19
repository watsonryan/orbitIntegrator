#include <cmath>
#include <vector>

#include "ode/logging.hpp"
#include "ode/multistep/adams_bashforth_moulton.hpp"
#include "ode/multistep/gauss_jackson8.hpp"
#include "ode/variational.hpp"

int main() {
  using State = std::vector<double>;
  constexpr double kPi = 3.14159265358979323846;

  {
    auto rhs = [](double, const State& x, State& dxdt) {
      dxdt.assign(1, -0.3 * x[0]);
    };
    auto jac = [](double, const State&, ode::variational::Matrix& a) {
      a.assign(1, -0.3);
      return true;
    };

    ode::multistep::AdamsBashforthMoultonOptions opt;
    opt.h = 0.01;
    opt.mode = ode::multistep::PredictorCorrectorMode::Iterated;
    opt.corrector_iterations = 2;

    const State x0{2.0};
    const double tf = 5.0;
    const auto r4 = ode::variational::integrate_state_stm_abm4(rhs, jac, 0.0, x0, tf, opt);
    const auto r6 = ode::variational::integrate_state_stm_abm6(rhs, jac, 0.0, x0, tf, opt);
    if (r4.status != ode::IntegratorStatus::Success || r6.status != ode::IntegratorStatus::Success) {
      ode::log::Error("ABM variational run failed");
      return 1;
    }

    const double exact_x = x0[0] * std::exp(-0.3 * tf);
    const double exact_phi = std::exp(-0.3 * tf);
    if (std::abs(r4.x[0] - exact_x) > 2e-5 || std::abs(r4.phi[0] - exact_phi) > 2e-5) {
      ode::log::Error("ABM4 variational mismatch");
      return 1;
    }
    if (std::abs(r6.x[0] - exact_x) > 1e-5 || std::abs(r6.phi[0] - exact_phi) > 1e-5) {
      ode::log::Error("ABM6 variational mismatch");
      return 1;
    }
  }

  {
    auto accel = [](double, const State& r, const State&, State& a) {
      a.assign(1, -r[0]);
    };
    auto accel_jac = [](double, const State&, const State&, ode::variational::Matrix& ar, ode::variational::Matrix& av) {
      ar.assign(1, -1.0);
      av.assign(1, 0.0);
      return true;
    };

    ode::multistep::GaussJackson8Options gj_opt;
    gj_opt.h = 0.01;
    gj_opt.corrector_iterations = 2;

    const State x0{1.0, 0.0};
    const double tf = 2.0 * kPi;
    const auto res = ode::variational::integrate_state_stm_gauss_jackson8(accel, accel_jac, 0.0, x0, tf, gj_opt);
    if (res.status != ode::IntegratorStatus::Success) {
      ode::log::Error("Gauss-Jackson variational run failed, status=", ode::ToString(res.status));
      return 1;
    }

    if (std::abs(res.x[0] - 1.0) > 4e-5 || std::abs(res.x[1]) > 4e-5) {
      ode::log::Error("Gauss-Jackson variational state mismatch");
      return 1;
    }

    const double c = std::cos(tf);
    const double s = std::sin(tf);
    if (std::abs(res.phi[0] - c) > 8e-4 ||
        std::abs(res.phi[1] - s) > 8e-4 ||
        std::abs(res.phi[2] + s) > 8e-4 ||
        std::abs(res.phi[3] - c) > 8e-4) {
      ode::log::Error("Gauss-Jackson variational STM mismatch");
      return 1;
    }
  }

  return 0;
}
