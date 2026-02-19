/**
 * @file integrate_method.hpp
 * @brief Compile-time tableau integration entry points.
 */
#pragma once

#include "ode/drivers.hpp"

namespace ode {

template <class Tableau, class State, class RHS, class Algebra = DefaultAlgebra<State>>
requires AlgebraFor<Algebra, State>
/** @brief Integrate using a compile-time-selected tableau type. */
[[nodiscard]] IntegratorResult<State> integrate_with_tableau(RHS&& rhs,
                                                              double t0,
                                                              const State& y0,
                                                              double t1,
                                                              IntegratorOptions opt,
                                                              Observer<State> obs = {}) {
  if constexpr (Tableau::has_embedded) {
    if (opt.adaptive) {
      return integrate_adaptive<Tableau, State, RHS, Algebra>(std::forward<RHS>(rhs), t0, y0, t1, opt, obs);
    }
    return integrate_fixed<Tableau, State, RHS, Algebra>(std::forward<RHS>(rhs), t0, y0, t1, opt, obs);
  }

  opt.adaptive = false;
  return integrate_fixed<Tableau, State, RHS, Algebra>(std::forward<RHS>(rhs), t0, y0, t1, opt, obs);
}

}  // namespace ode
