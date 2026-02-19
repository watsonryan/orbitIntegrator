/**
 * @file sundman.hpp
 * @brief Optional Sundman-transformed stepping wrappers for explicit RK methods.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <utility>

#include "ode/algebra.hpp"
#include "ode/error_norm.hpp"
#include "ode/rk_stepper.hpp"
#include "ode/tableaus/rk4.hpp"
#include "ode/tableaus/rkf45.hpp"
#include "ode/tableaus/rkf78.hpp"
#include "ode/types.hpp"

namespace ode {

template <class State>
using SundmanScale = std::function<double(double, const State&)>;  // dt/ds

template <class Tableau, class State, class RHS, class ScaleFn, class Algebra = DefaultAlgebra<State>>
requires AlgebraFor<Algebra, State>
[[nodiscard]] IntegratorResult<State> integrate_with_tableau_sundman(
    RHS&& rhs,
    ScaleFn&& dt_ds,
    double t0,
    const State& y0,
    double t1,
    IntegratorOptions opt,
    Observer<State> obs = {}) {
  IntegratorResult<State> out{};
  out.t = t0;
  out.y = y0;

  const int dir_t = (t1 > t0) - (t1 < t0);
  if (dir_t == 0) {
    return out;
  }

  if (opt.max_steps <= 0) {
    out.status = IntegratorStatus::MaxStepsExceeded;
    return out;
  }

  if (opt.adaptive) {
    if constexpr (!Tableau::has_embedded) {
      out.status = IntegratorStatus::InvalidStepSize;
      return out;
    }
    if (!(opt.rtol > 0.0) || !(opt.atol > 0.0) || !std::isfinite(opt.rtol) || !std::isfinite(opt.atol)) {
      out.status = IntegratorStatus::InvalidTolerance;
      return out;
    }
  }

  if (!(opt.h_min > 0.0) || !(opt.h_max >= opt.h_min) || !std::isfinite(opt.h_min) || !std::isfinite(opt.h_max)) {
    out.status = IntegratorStatus::InvalidStepSize;
    return out;
  }

  // h_* are interpreted as pseudo-time step ds bounds.
  double ds = 0.0;
  const double span = std::abs(t1 - t0);
  if (opt.adaptive) {
    ds = (opt.h_init > 0.0 && std::isfinite(opt.h_init)) ? opt.h_init : (span / 100.0);
  } else {
    if (!(opt.fixed_h > 0.0) || !std::isfinite(opt.fixed_h)) {
      out.status = IntegratorStatus::InvalidStepSize;
      return out;
    }
    ds = opt.fixed_h;
  }
  if (!(ds > 0.0) || !std::isfinite(ds)) {
    ds = opt.h_min;
  }

  ExplicitRKStepper<Tableau, State, Algebra> stepper(y0);
  State y_high{};
  State err{};
  Algebra::resize_like(y_high, y0);
  Algebra::resize_like(err, y0);

  double s = 0.0;
  for (int step = 0; step < opt.max_steps; ++step) {
    const double rem_t = t1 - out.t;
    if ((dir_t > 0 && rem_t <= 0.0) || (dir_t < 0 && rem_t >= 0.0)) {
      out.status = IntegratorStatus::Success;
      return out;
    }

    const double scale0 = dt_ds(out.t, out.y);
    if (!std::isfinite(scale0) || scale0 == 0.0 || ((scale0 > 0.0 ? 1 : -1) != dir_t)) {
      out.status = IntegratorStatus::InvalidStepSize;
      return out;
    }

    const double ds_try = std::max(opt.h_min, std::min(opt.h_max, ds));
    double dt_try = ds_try * scale0;
    if (std::abs(dt_try) > std::abs(rem_t)) {
      dt_try = rem_t;
    }
    const double ds_step = dt_try / scale0;

    auto rhs_s = [&](double s_stage, const State& y_stage, State& dyds) {
      const double t_stage = out.t + (s_stage - s) * scale0;
      rhs(t_stage, y_stage, dyds);
      for (std::size_t i = 0; i < dyds.size(); ++i) {
        dyds[i] *= scale0;
      }
    };

    out.stats.attempted_steps += 1;
    const bool ok = stepper.step(rhs_s, s, out.y, ds_step, y_high, opt.adaptive ? &err : nullptr);
    out.stats.rhs_evals += Tableau::stages;
    out.stats.last_h = dt_try;

    if (!ok || !Algebra::finite(y_high)) {
      out.status = IntegratorStatus::NaNDetected;
      return out;
    }

    if (!opt.adaptive) {
      out.t += dt_try;
      s += ds_step;
      Algebra::assign(out.y, y_high);
      out.stats.accepted_steps += 1;
      if (obs && !obs(out.t, out.y)) {
        out.status = IntegratorStatus::UserStopped;
        return out;
      }
      continue;
    }

    const double err_norm = weighted_rms_error<State, Algebra>(err, out.y, y_high, opt.atol, opt.rtol);
    out.stats.last_error_norm = err_norm;
    if (!std::isfinite(err_norm)) {
      out.status = IntegratorStatus::NaNDetected;
      return out;
    }

    if (err_norm <= 1.0) {
      out.t += dt_try;
      s += ds_step;
      Algebra::assign(out.y, y_high);
      out.stats.accepted_steps += 1;
      if (obs && !obs(out.t, out.y)) {
        out.status = IntegratorStatus::UserStopped;
        return out;
      }
    } else {
      out.stats.rejected_steps += 1;
      if (!opt.allow_step_reject) {
        out.status = IntegratorStatus::InvalidStepSize;
        return out;
      }
    }

    const double p = static_cast<double>(Tableau::order_high);
    const double fac = std::clamp(opt.safety * std::pow(std::max(err_norm, std::numeric_limits<double>::min()), -1.0 / p),
                                  opt.fac_min, opt.fac_max);
    ds *= fac;
    if (ds < opt.h_min) {
      out.status = IntegratorStatus::StepSizeUnderflow;
      return out;
    }
    ds = std::max(opt.h_min, std::min(opt.h_max, ds));
  }

  out.status = IntegratorStatus::MaxStepsExceeded;
  return out;
}

template <class State, class RHS, class ScaleFn, class Algebra = DefaultAlgebra<State>>
requires AlgebraFor<Algebra, State>
[[nodiscard]] IntegratorResult<State> integrate_sundman(
    RKMethod method,
    RHS&& rhs,
    ScaleFn&& dt_ds,
    double t0,
    const State& y0,
    double t1,
    IntegratorOptions opt,
    Observer<State> obs = {}) {
  switch (method) {
    case RKMethod::RK4:
      opt.adaptive = false;
      return integrate_with_tableau_sundman<TableauRK4, State, RHS, ScaleFn, Algebra>(
          std::forward<RHS>(rhs), std::forward<ScaleFn>(dt_ds), t0, y0, t1, opt, std::move(obs));
    case RKMethod::RKF45:
      return integrate_with_tableau_sundman<TableauRKF45, State, RHS, ScaleFn, Algebra>(
          std::forward<RHS>(rhs), std::forward<ScaleFn>(dt_ds), t0, y0, t1, opt, std::move(obs));
    case RKMethod::RK8:
      opt.adaptive = false;
      return integrate_with_tableau_sundman<TableauRKF78, State, RHS, ScaleFn, Algebra>(
          std::forward<RHS>(rhs), std::forward<ScaleFn>(dt_ds), t0, y0, t1, opt, std::move(obs));
    case RKMethod::RKF78:
      return integrate_with_tableau_sundman<TableauRKF78, State, RHS, ScaleFn, Algebra>(
          std::forward<RHS>(rhs), std::forward<ScaleFn>(dt_ds), t0, y0, t1, opt, std::move(obs));
  }

  IntegratorResult<State> fallback{};
  fallback.status = IntegratorStatus::InvalidStepSize;
  return fallback;
}

}  // namespace ode
