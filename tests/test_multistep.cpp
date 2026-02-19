#include <cmath>
#include <iostream>
#include <vector>

#include "ode/multistep/adams_bashforth_moulton.hpp"
#include "ode/multistep/adams_high_order.hpp"
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
    opt.mode = ode::multistep::PredictorCorrectorMode::Iterated;
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
    const State y0{1.0};
    ode::multistep::AdamsBashforthMoultonOptions abm4_opt;
    abm4_opt.h = 0.02;
    abm4_opt.mode = ode::multistep::PredictorCorrectorMode::Iterated;
    abm4_opt.corrector_iterations = 2;

    ode::multistep::AdamsBashforthMoultonOptions abm6_opt = abm4_opt;

    const auto abm4 = ode::multistep::integrate_abm4(rhs, 0.0, y0, 1.0, abm4_opt);
    const auto abm6 = ode::multistep::integrate_abm6(rhs, 0.0, y0, 1.0, abm6_opt);
    if (abm4.status != ode::IntegratorStatus::Success || abm6.status != ode::IntegratorStatus::Success) {
      std::cerr << "ABM4/ABM6 run failed\n";
      return 1;
    }
    const double exact = std::exp(1.0);
    const double e4 = std::abs(abm4.y[0] - exact);
    const double e6 = std::abs(abm6.y[0] - exact);
    if (!(e6 < e4)) {
      std::cerr << "Expected ABM6 to be more accurate than ABM4: e4=" << e4 << " e6=" << e6 << "\n";
      return 1;
    }
  }

  {
    const State y1{std::exp(1.0)};
    ode::multistep::AdamsBashforthMoultonOptions opt;
    opt.h = 0.01;
    opt.mode = ode::multistep::PredictorCorrectorMode::Iterated;
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
    ms_opt.mode = ode::multistep::PredictorCorrectorMode::Iterated;
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

  {
    const State y0{1.0};
    ode::multistep::AdamsBashforthMoultonOptions pec_opt;
    pec_opt.h = 0.01;
    pec_opt.mode = ode::multistep::PredictorCorrectorMode::PEC;

    ode::multistep::AdamsBashforthMoultonOptions pece_opt = pec_opt;
    pece_opt.mode = ode::multistep::PredictorCorrectorMode::PECE;

    ode::multistep::AdamsBashforthMoultonOptions iter_opt = pec_opt;
    iter_opt.mode = ode::multistep::PredictorCorrectorMode::Iterated;
    iter_opt.corrector_iterations = 2;

    const auto pec = ode::multistep::integrate_abm4(rhs, 0.0, y0, 1.0, pec_opt);
    const auto pece = ode::multistep::integrate_abm4(rhs, 0.0, y0, 1.0, pece_opt);
    const auto iter = ode::multistep::integrate_abm4(rhs, 0.0, y0, 1.0, iter_opt);
    if (pec.status != ode::IntegratorStatus::Success ||
        pece.status != ode::IntegratorStatus::Success ||
        iter.status != ode::IntegratorStatus::Success) {
      std::cerr << "ABM mode run failed\n";
      return 1;
    }

    const double exact = std::exp(1.0);
    const double e_pec = std::abs(pec.y[0] - exact);
    const double e_pece = std::abs(pece.y[0] - exact);
    const double e_iter = std::abs(iter.y[0] - exact);
    if (!(e_pece <= e_pec * 1.2)) {
      std::cerr << "PECE should be at least as accurate as PEC in this case\n";
      return 1;
    }
    if (!(e_iter <= e_pece * 1.2)) {
      std::cerr << "Iterated should be at least as accurate as PECE in this case\n";
      return 1;
    }
  }

  return 0;
}
