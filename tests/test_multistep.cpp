#include <cmath>
#include <iostream>
#include <vector>

#include "ode/multistep/adams_bashforth_moulton.hpp"
#include "ode/ode.hpp"

int main() {
  using State = std::vector<double>;

  auto rhs = [](double, const State& y, State& dydt) {
    dydt.resize(1);
    dydt[0] = y[0];
  };

  {
    const State y0{1.0};
    ode::multistep::AdamsBashforthMoultonOptions opt;
    opt.h = 0.01;
    opt.corrector_iterations = 2;

    const auto res = ode::multistep::integrate_abm4(rhs, 0.0, y0, 1.0, opt);
    if (res.status != ode::IntegratorStatus::Success) {
      std::cerr << "ABM4 forward failed\n";
      return 1;
    }

    const double err = std::abs(res.y[0] - std::exp(1.0));
    if (err > 1e-7) {
      std::cerr << "ABM4 forward error too large: " << err << "\n";
      return 1;
    }
  }

  {
    const State y1{std::exp(1.0)};
    ode::multistep::AdamsBashforthMoultonOptions opt;
    opt.h = 0.01;
    opt.corrector_iterations = 2;

    const auto res = ode::multistep::integrate_abm4(rhs, 1.0, y1, 0.0, opt);
    if (res.status != ode::IntegratorStatus::Success) {
      std::cerr << "ABM4 backward failed\n";
      return 1;
    }

    const double err = std::abs(res.y[0] - 1.0);
    if (err > 1e-7) {
      std::cerr << "ABM4 backward error too large: " << err << "\n";
      return 1;
    }
  }

  {
    const State y0{1.0};
    ode::multistep::AdamsBashforthMoultonOptions ms_opt;
    ms_opt.h = 0.01;
    ms_opt.corrector_iterations = 2;

    ode::IntegratorOptions rk_opt;
    rk_opt.adaptive = true;
    rk_opt.rtol = 1e-12;
    rk_opt.atol = 1e-14;
    rk_opt.h_init = 0.01;

    const auto abm = ode::multistep::integrate_abm4(rhs, 0.0, y0, 1.0, ms_opt);
    const auto rk = ode::integrate(ode::RKMethod::RKF78, rhs, 0.0, y0, 1.0, rk_opt);
    if (abm.status != ode::IntegratorStatus::Success || rk.status != ode::IntegratorStatus::Success) {
      std::cerr << "ABM/RK compare failed\n";
      return 1;
    }

    if (std::abs(abm.y[0] - rk.y[0]) > 5e-8) {
      std::cerr << "ABM vs RK mismatch: " << abm.y[0] << " vs " << rk.y[0] << "\n";
      return 1;
    }
  }

  return 0;
}
