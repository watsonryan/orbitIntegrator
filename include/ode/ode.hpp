#pragma once

#include "ode/drivers.hpp"
#include "ode/tableaus/rk4.hpp"
#include "ode/tableaus/rkf45.hpp"
#include "ode/tableaus/rkf78.hpp"
#include "ode/types.hpp"

namespace ode {

template <class State, class RHS, class Algebra = DefaultAlgebra<State>>
requires AlgebraFor<Algebra, State>
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
      return integrate_fixed<TableauRK4, State, RHS, Algebra>(std::forward<RHS>(rhs), t0, y0, t1, opt, obs);
    case RKMethod::RKF45:
      if (opt.adaptive) {
        return integrate_adaptive<TableauRKF45, State, RHS, Algebra>(std::forward<RHS>(rhs), t0, y0, t1, opt, obs);
      }
      return integrate_fixed<TableauRKF45, State, RHS, Algebra>(std::forward<RHS>(rhs), t0, y0, t1, opt, obs);
    case RKMethod::RK8:
      opt.adaptive = false;
      return integrate_fixed<TableauRKF78, State, RHS, Algebra>(std::forward<RHS>(rhs), t0, y0, t1, opt, obs);
    case RKMethod::RKF78:
      if (opt.adaptive) {
        return integrate_adaptive<TableauRKF78, State, RHS, Algebra>(std::forward<RHS>(rhs), t0, y0, t1, opt, obs);
      }
      return integrate_fixed<TableauRKF78, State, RHS, Algebra>(std::forward<RHS>(rhs), t0, y0, t1, opt, obs);
  }

  IntegratorResult<State> fallback{};
  fallback.status = IntegratorStatus::InvalidStepSize;
  return fallback;
}

}  // namespace ode
