/**
 * @file batch.hpp
 * @brief Batch propagation helpers for running many independent trajectories with shared settings.
 */
#pragma once

#include <cstddef>
#include <vector>

#include "ode/integrate_method.hpp"
#include "ode/tableaus/rk4.hpp"
#include "ode/tableaus/rkf45.hpp"
#include "ode/tableaus/rkf78.hpp"
#include "ode/types.hpp"

namespace ode {

template <class State>
struct BatchTask {
  double t0 = 0.0;
  double t1 = 0.0;
  State y0{};
};

template <class State>
struct BatchWorkspace {
  std::vector<IntegratorResult<State>> results{};

  void Reserve(std::size_t count) {
    results.reserve(count);
  }
};

template <class State, class RHS, class Algebra = DefaultAlgebra<State>>
requires AlgebraFor<Algebra, State>
void integrate_batch_inplace(RKMethod method,
                             RHS&& rhs,
                             const std::vector<BatchTask<State>>& tasks,
                             const IntegratorOptions& opt,
                             std::vector<IntegratorResult<State>>& out_results) {
  auto&& rhs_ref = rhs;
  out_results.clear();
  out_results.reserve(tasks.size());

  auto run_one = [&](double t0, const State& y0, double t1) {
    IntegratorOptions local_opt = opt;
    switch (method) {
      case RKMethod::RK4:
        local_opt.adaptive = false;
        return integrate_with_tableau<TableauRK4, State, decltype(rhs_ref), Algebra>(
            rhs_ref, t0, y0, t1, local_opt);
      case RKMethod::RKF45:
        return integrate_with_tableau<TableauRKF45, State, decltype(rhs_ref), Algebra>(
            rhs_ref, t0, y0, t1, local_opt);
      case RKMethod::RK8:
        local_opt.adaptive = false;
        return integrate_with_tableau<TableauRKF78, State, decltype(rhs_ref), Algebra>(
            rhs_ref, t0, y0, t1, local_opt);
      case RKMethod::RKF78:
        return integrate_with_tableau<TableauRKF78, State, decltype(rhs_ref), Algebra>(
            rhs_ref, t0, y0, t1, local_opt);
    }
    IntegratorResult<State> fallback{};
    fallback.status = IntegratorStatus::InvalidStepSize;
    return fallback;
  };

  for (const auto& task : tasks) {
    out_results.push_back(run_one(task.t0, task.y0, task.t1));
  }
}

template <class State, class RHS, class Algebra = DefaultAlgebra<State>>
requires AlgebraFor<Algebra, State>
[[nodiscard]] std::vector<IntegratorResult<State>> integrate_batch(RKMethod method,
                                                                   RHS&& rhs,
                                                                   const std::vector<BatchTask<State>>& tasks,
                                                                   const IntegratorOptions& opt,
                                                                   BatchWorkspace<State>* workspace = nullptr) {
  if (workspace != nullptr) {
    integrate_batch_inplace<State, RHS, Algebra>(method, std::forward<RHS>(rhs), tasks, opt, workspace->results);
    return workspace->results;
  }

  std::vector<IntegratorResult<State>> results;
  integrate_batch_inplace<State, RHS, Algebra>(method, std::forward<RHS>(rhs), tasks, opt, results);
  return results;
}

}  // namespace ode
