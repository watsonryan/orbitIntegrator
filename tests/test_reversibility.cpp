#include <cmath>
#include <vector>

#include "ode/logging.hpp"
#include "ode/ode.hpp"

int main() {
  using State = std::vector<double>;

  // First-order reversible check on harmonic oscillator with RK8.
  {
    auto rhs = [](double, const State& y, State& dydt) {
      dydt.resize(2);
      dydt[0] = y[1];
      dydt[1] = -y[0];
    };
    const State y0{1.0, 0.0};
    ode::IntegratorOptions opt;
    opt.adaptive = false;
    opt.fixed_h = 0.01;

    const auto fwd = ode::integrate(ode::RKMethod::RK8, rhs, 0.0, y0, 20.0, opt);
    if (fwd.status != ode::IntegratorStatus::Success) {
      ode::log::Error("reversibility forward run failed");
      return 1;
    }
    const auto bwd = ode::integrate(ode::RKMethod::RK8, rhs, 20.0, fwd.y, 0.0, opt);
    if (bwd.status != ode::IntegratorStatus::Success) {
      ode::log::Error("reversibility backward run failed");
      return 1;
    }
    if (std::abs(bwd.y[0] - y0[0]) > 5e-10 || std::abs(bwd.y[1] - y0[1]) > 5e-10) {
      ode::log::Error("RK8 reversibility mismatch");
      return 1;
    }
  }

  // Second-order reversible check with Gauss-Jackson.
  {
    auto accel = [](double, const std::vector<double>& r, const std::vector<double>&, std::vector<double>& a) {
      a.resize(1);
      a[0] = -r[0];
    };
    std::vector<double> r0{1.0};
    std::vector<double> v0{0.0};
    ode::multistep::GaussJackson8Options gj_opt;
    gj_opt.h = 0.01;
    gj_opt.corrector_iterations = 2;
    const auto fwd = ode::multistep::integrate_gauss_jackson8(accel, 0.0, r0, v0, 20.0, gj_opt);
    if (fwd.status != ode::IntegratorStatus::Success) {
      ode::log::Error("GJ reversibility forward run failed");
      return 1;
    }
    const auto bwd = ode::multistep::integrate_gauss_jackson8(accel, 20.0, fwd.r, fwd.v, 0.0, gj_opt);
    if (bwd.status != ode::IntegratorStatus::Success) {
      ode::log::Error("GJ reversibility backward run failed");
      return 1;
    }
    if (std::abs(bwd.r[0] - r0[0]) > 3e-7 || std::abs(bwd.v[0] - v0[0]) > 3e-7) {
      ode::log::Error("GJ reversibility mismatch");
      return 1;
    }
  }

  return 0;
}

