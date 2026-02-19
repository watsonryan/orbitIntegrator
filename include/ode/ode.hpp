/**
 * @file ode.hpp
 * @brief Primary runtime API for selecting and running RK integration methods.
 */
#pragma once

#include "ode/batch.hpp"
#include "ode/chaos.hpp"
#include "ode/integrate_method.hpp"
#include "ode/multistep/gauss_jackson8.hpp"
#include "ode/poincare.hpp"
#include "ode/symplectic.hpp"
#include "ode/sundman.hpp"
#include "ode/tableaus/rk4.hpp"
#include "ode/tableaus/rkf45.hpp"
#include "ode/tableaus/rkf78.hpp"
#include "ode/types.hpp"
#include "ode/uncertainty.hpp"
#include "ode/variational.hpp"

namespace ode {

template <class State, class RHS, class Algebra = DefaultAlgebra<State>>
requires AlgebraFor<Algebra, State>
/** @brief Integrate using runtime method selection and shared options/observer API. */
[[nodiscard]] IntegratorResult<State> integrate(RKMethod method,
                                                 RHS&& rhs,
                                                 double t0,
                                                 const State& y0,
                                                 double t1,
                                                 IntegratorOptions opt,
                                                 Observer<State> obs = {}) {
  switch (method) {
    case RKMethod::RK4:
      opt.adaptive = false;
      return integrate_with_tableau<TableauRK4, State, RHS, Algebra>(std::forward<RHS>(rhs), t0, y0, t1, opt, obs);
    case RKMethod::RKF45:
      return integrate_with_tableau<TableauRKF45, State, RHS, Algebra>(std::forward<RHS>(rhs), t0, y0, t1, opt, obs);
    case RKMethod::RK8:
      opt.adaptive = false;
      return integrate_with_tableau<TableauRKF78, State, RHS, Algebra>(std::forward<RHS>(rhs), t0, y0, t1, opt, obs);
    case RKMethod::RKF78:
      return integrate_with_tableau<TableauRKF78, State, RHS, Algebra>(std::forward<RHS>(rhs), t0, y0, t1, opt, obs);
  }

  IntegratorResult<State> fallback{};
  fallback.status = IntegratorStatus::InvalidStepSize;
  return fallback;
}

template <class State, class RHS, class InvariantFn, class Algebra = DefaultAlgebra<State>>
requires AlgebraFor<Algebra, State>
/** @brief Integrate with adaptive LTE plus invariant-drift control. */
[[nodiscard]] IntegratorResult<State> integrate_invariant(RKMethod method,
                                                           RHS&& rhs,
                                                           InvariantFn&& invariant_fn,
                                                           double t0,
                                                           const State& y0,
                                                           double t1,
                                                           IntegratorOptions opt,
                                                           Observer<State> obs = {}) {
  switch (method) {
    case RKMethod::RKF45:
      return integrate_adaptive_invariant<TableauRKF45, State, RHS, InvariantFn, Algebra>(
          std::forward<RHS>(rhs), std::forward<InvariantFn>(invariant_fn), t0, y0, t1, opt, obs);
    case RKMethod::RKF78:
      return integrate_adaptive_invariant<TableauRKF78, State, RHS, InvariantFn, Algebra>(
          std::forward<RHS>(rhs), std::forward<InvariantFn>(invariant_fn), t0, y0, t1, opt, obs);
    case RKMethod::RK4:
    case RKMethod::RK8:
      break;
  }

  IntegratorResult<State> fallback{};
  fallback.status = IntegratorStatus::InvalidStepSize;
  return fallback;
}

}  // namespace ode
