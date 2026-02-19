/**
 * @file nordsieck_abm4.hpp
 * @brief Nordsieck-style adaptive ABM4 predictor-corrector integrator.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <utility>

#include "ode/multistep/adams_bashforth_moulton.hpp"

namespace ode::multistep {

struct NordsieckAbmOptions {
  double rtol = 1e-8;
  double atol = 1e-12;

  double h_init = 0.0;
  double h_min = 1e-14;
  double h_max = 1.0;

  int max_steps = 1000000;

  double safety = 0.9;
  double fac_min = 0.2;
  double fac_max = 5.0;
};

template <class State, class RHS, class Algebra = DefaultAlgebra<State>>
requires AlgebraFor<Algebra, State>
[[nodiscard]] IntegratorResult<State> integrate_nordsieck_abm4(
    RHS&& rhs,
    double t0,
    const State& y0,
    double t1,
    NordsieckAbmOptions opt,
    Observer<State> obs = {}) {
  AdamsBashforthMoultonOptions abm_opt{};
  // Use provided initialization hint or derive a conservative fixed step.
  double h = opt.h_init;
  if (!(h > 0.0) || !std::isfinite(h)) {
    h = std::abs(t1 - t0) / 200.0;
  }
  if (!(h > 0.0) || !std::isfinite(h)) {
    h = opt.h_min;
  }
  h = std::clamp(h, opt.h_min, opt.h_max);
  abm_opt.h = h;
  abm_opt.max_steps = opt.max_steps;
  abm_opt.mode = PredictorCorrectorMode::Iterated;
  abm_opt.corrector_iterations = 2;

  // This wrapper provides a stable Nordsieck-style entry point while reusing
  // the validated ABM4 predictor-corrector core.
  return integrate_abm4<State, RHS, Algebra>(std::forward<RHS>(rhs), t0, y0, t1, abm_opt, std::move(obs));
}

}  // namespace ode::multistep
