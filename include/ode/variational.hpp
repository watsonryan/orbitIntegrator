/**
 * @file variational.hpp
 * @brief Model-agnostic variational propagation APIs (STM and covariance).
 */
#pragma once

#include <cstddef>
#include <utility>

#include "ode/uncertainty.hpp"

namespace ode::variational {

using State = ode::DynamicState;
using Matrix = ode::DynamicMatrix;
using StateStmResult = ode::uncertainty::StateStmResult;
using StateStmCovResult = ode::uncertainty::StateStmCovResult;
using Dual = ode::uncertainty::Dual;

template <class Dynamics>
[[nodiscard]] inline bool jacobian_forward_ad(Dynamics&& dynamics, double t, const State& x, Matrix& a_out) {
  return ode::uncertainty::jacobian_forward_ad(std::forward<Dynamics>(dynamics), t, x, a_out);
}

template <class RHS, class JacobianFn>
[[nodiscard]] inline StateStmResult integrate_state_stm(RKMethod method,
                                                        RHS&& rhs,
                                                        JacobianFn&& jacobian_fn,
                                                        double t0,
                                                        const State& x0,
                                                        double t1,
                                                        IntegratorOptions opt,
                                                        Observer<State> obs = {}) {
  return ode::uncertainty::integrate_state_stm(
      method, std::forward<RHS>(rhs), std::forward<JacobianFn>(jacobian_fn), t0, x0, t1, opt, obs);
}

template <class RHS, class JacobianFn, class ProcessNoiseFn>
[[nodiscard]] inline StateStmCovResult integrate_state_stm_cov(RKMethod method,
                                                               RHS&& rhs,
                                                               JacobianFn&& jacobian_fn,
                                                               ProcessNoiseFn&& q_fn,
                                                               double t0,
                                                               const State& x0,
                                                               const Matrix& p0,
                                                               double t1,
                                                               IntegratorOptions opt,
                                                               Observer<State> obs = {}) {
  return ode::uncertainty::integrate_state_stm_cov(method,
                                                   std::forward<RHS>(rhs),
                                                   std::forward<JacobianFn>(jacobian_fn),
                                                   std::forward<ProcessNoiseFn>(q_fn),
                                                   t0,
                                                   x0,
                                                   p0,
                                                   t1,
                                                   opt,
                                                   obs);
}

[[nodiscard]] inline Matrix propagate_covariance_discrete(const Matrix& phi,
                                                          const Matrix& p0,
                                                          const Matrix& qd,
                                                          std::size_t n) {
  return ode::uncertainty::propagate_covariance_discrete(phi, p0, qd, n);
}

}  // namespace ode::variational
