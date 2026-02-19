/**
 * @file drivers.hpp
 * @brief Fixed-step and adaptive integration drivers built on the generic stepper.
 */
#pragma once

#include <cmath>
#include <limits>

#include "ode/controller.hpp"
#include "ode/error_norm.hpp"
#include "ode/rk_stepper.hpp"
#include "ode/types.hpp"

namespace ode {

namespace detail {

[[nodiscard]] inline int sign(double x) {
  return (x > 0.0) - (x < 0.0);
}

[[nodiscard]] inline double clamp_abs_signed(double h, double h_min, double h_max, int dir) {
  const double ah = std::abs(h);
  const double bounded = std::max(h_min, std::min(h_max, ah));
  return static_cast<double>(dir) * bounded;
}

template <class State, class RHS, class Algebra>
requires AlgebraFor<Algebra, State>
[[nodiscard]] inline double estimate_initial_step(RHS&& rhs,
                                                  double t0,
                                                  const State& y0,
                                                  double t1,
                                                  double atol,
                                                  double rtol,
                                                  double h_min,
                                                  double h_max,
                                                  int dir) {
  State dydt0{};
  Algebra::resize_like(dydt0, y0);
  rhs(t0, y0, dydt0);
  if (!Algebra::finite(dydt0)) {
    return static_cast<double>(dir) * h_min;
  }

  const std::size_t n = Algebra::size(y0);
  if (n == 0) {
    return static_cast<double>(dir) * h_min;
  }

  double y_norm2 = 0.0;
  double f_norm2 = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double scale = atol + rtol * std::abs(y0[i]);
    const double yi = y0[i] / scale;
    const double fi = dydt0[i] / scale;
    y_norm2 += yi * yi;
    f_norm2 += fi * fi;
  }
  const double y_norm = std::sqrt(y_norm2 / static_cast<double>(n));
  const double f_norm = std::sqrt(f_norm2 / static_cast<double>(n));

  double h0 = 0.01;
  if (f_norm > 1e-16) {
    h0 = 0.01 * (y_norm / f_norm);
  }
  if (!std::isfinite(h0) || h0 <= 0.0) {
    h0 = std::abs(t1 - t0) / 100.0;
  }
  if (!std::isfinite(h0) || h0 <= 0.0) {
    h0 = h_min;
  }
  return clamp_abs_signed(static_cast<double>(dir) * h0, h_min, h_max, dir);
}

template <class State>
[[nodiscard]] inline bool call_observer(const Observer<State>& obs, double t, const State& y) {
  if (!obs) {
    return true;
  }
  return obs(t, y);
}

}  // namespace detail

template <class Tableau, class State, class RHS, class Algebra = DefaultAlgebra<State>>
requires AlgebraFor<Algebra, State>
/** @brief Integrate with fixed step size until endpoint, stop, or error condition. */
[[nodiscard]] IntegratorResult<State> integrate_fixed(RHS&& rhs,
                                                      double t0,
                                                      const State& y0,
                                                      double t1,
                                                      const IntegratorOptions& opt,
                                                      const Observer<State>& obs = {}) {
  IntegratorResult<State> out{};
  out.t = t0;
  out.y = y0;

  if (!(opt.fixed_h > 0.0) || !std::isfinite(opt.fixed_h)) {
    out.status = IntegratorStatus::InvalidStepSize;
    return out;
  }
  if (opt.max_steps <= 0) {
    out.status = IntegratorStatus::MaxStepsExceeded;
    return out;
  }

  const int dir = detail::sign(t1 - t0);
  if (dir == 0) {
    return out;
  }

  double h = static_cast<double>(dir) * opt.fixed_h;
  ExplicitRKStepper<Tableau, State, Algebra> stepper(y0);
  State y_next{};
  Algebra::resize_like(y_next, y0);

  for (int step = 0; step < opt.max_steps; ++step) {
    if (out.t == t1) {
      out.status = IntegratorStatus::Success;
      return out;
    }

    const double remaining = t1 - out.t;
    if (detail::sign(remaining) != dir) {
      out.status = IntegratorStatus::Success;
      return out;
    }
    if (std::abs(h) > std::abs(remaining)) {
      h = remaining;
    }

    out.stats.attempted_steps += 1;
    const bool ok = stepper.step(rhs, out.t, out.y, h, y_next, nullptr);
    out.stats.rhs_evals += Tableau::stages;
    out.stats.last_h = h;

    if (!ok || !Algebra::finite(y_next)) {
      out.status = IntegratorStatus::NaNDetected;
      return out;
    }

    out.t += h;
    Algebra::assign(out.y, y_next);
    out.stats.accepted_steps += 1;

    if (!detail::call_observer(obs, out.t, out.y)) {
      out.status = IntegratorStatus::UserStopped;
      return out;
    }
  }

  out.status = IntegratorStatus::MaxStepsExceeded;
  return out;
}

template <class Tableau, class State, class RHS, class Algebra = DefaultAlgebra<State>>
requires AlgebraFor<Algebra, State>
/** @brief Integrate with adaptive embedded error control until endpoint or failure. */
[[nodiscard]] IntegratorResult<State> integrate_adaptive(RHS&& rhs,
                                                         double t0,
                                                         const State& y0,
                                                         double t1,
                                                         const IntegratorOptions& opt,
                                                         const Observer<State>& obs = {}) {
  IntegratorResult<State> out{};
  out.t = t0;
  out.y = y0;

  if constexpr (!Tableau::has_embedded) {
    out.status = IntegratorStatus::InvalidStepSize;
    return out;
  }

  if (!(opt.rtol > 0.0) || !(opt.atol > 0.0) || !std::isfinite(opt.rtol) || !std::isfinite(opt.atol)) {
    out.status = IntegratorStatus::InvalidTolerance;
    return out;
  }
  if (!(opt.h_min > 0.0) || !(opt.h_max >= opt.h_min) || !std::isfinite(opt.h_min) || !std::isfinite(opt.h_max)) {
    out.status = IntegratorStatus::InvalidStepSize;
    return out;
  }
  if (opt.max_steps <= 0) {
    out.status = IntegratorStatus::MaxStepsExceeded;
    return out;
  }

  const int dir = detail::sign(t1 - t0);
  if (dir == 0) {
    return out;
  }

  const double span = std::abs(t1 - t0);
  double h = 0.0;
  if (opt.h_init > 0.0 && std::isfinite(opt.h_init)) {
    h = detail::clamp_abs_signed(static_cast<double>(dir) * opt.h_init, opt.h_min, opt.h_max, dir);
  } else {
    h = detail::estimate_initial_step<State, RHS, Algebra>(std::forward<RHS>(rhs),
                                                           t0, y0, t1, opt.atol, opt.rtol,
                                                           opt.h_min, opt.h_max, dir);
  }
  if (!(std::abs(h) > 0.0) || !std::isfinite(h) || span == 0.0) {
    h = static_cast<double>(dir) * opt.h_min;
  }

  StepSizeController controller{opt.safety, opt.fac_min, opt.fac_max};
  ExplicitRKStepper<Tableau, State, Algebra> stepper(y0);
  State y_high{};
  State err{};
  Algebra::resize_like(y_high, y0);
  Algebra::resize_like(err, y0);

  for (int step = 0; step < opt.max_steps; ++step) {
    if (out.t == t1) {
      out.status = IntegratorStatus::Success;
      return out;
    }

    const double remaining = t1 - out.t;
    if (detail::sign(remaining) != dir) {
      out.status = IntegratorStatus::Success;
      return out;
    }
    if (std::abs(h) > std::abs(remaining)) {
      h = remaining;
    }

    out.stats.attempted_steps += 1;
    const bool ok = stepper.step(rhs, out.t, out.y, h, y_high, &err);
    out.stats.rhs_evals += Tableau::stages;

    if (!ok || !Algebra::finite(y_high) || !Algebra::finite(err)) {
      out.status = IntegratorStatus::NaNDetected;
      return out;
    }

    const double err_norm = weighted_rms_error<State, Algebra>(err, out.y, y_high, opt.atol, opt.rtol);
    out.stats.last_h = h;
    out.stats.last_error_norm = err_norm;
    if (!std::isfinite(err_norm)) {
      out.status = IntegratorStatus::NaNDetected;
      return out;
    }

    if (err_norm <= 1.0) {
      out.t += h;
      Algebra::assign(out.y, y_high);
      out.stats.accepted_steps += 1;

      if (!detail::call_observer(obs, out.t, out.y)) {
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

    const double h_new = controller.propose(
        h,
        (err_norm > 0.0 ? err_norm : std::numeric_limits<double>::min()),
        Tableau::order_high);

    if (!std::isfinite(h_new)) {
      out.status = IntegratorStatus::StepSizeUnderflow;
      return out;
    }
    if (std::abs(h_new) < opt.h_min) {
      out.status = IntegratorStatus::StepSizeUnderflow;
      return out;
    }

    h = detail::clamp_abs_signed(h_new, opt.h_min, opt.h_max, dir);
    if (std::abs(h) < opt.h_min) {
      out.status = IntegratorStatus::StepSizeUnderflow;
      return out;
    }
  }

  out.status = IntegratorStatus::MaxStepsExceeded;
  return out;
}

template <class Tableau, class State, class RHS, class InvariantFn, class Algebra = DefaultAlgebra<State>>
requires AlgebraFor<Algebra, State>
/** @brief Integrate with adaptive LTE control plus invariant-drift acceptance constraints. */
[[nodiscard]] IntegratorResult<State> integrate_adaptive_invariant(RHS&& rhs,
                                                                   InvariantFn&& invariant_fn,
                                                                   double t0,
                                                                   const State& y0,
                                                                   double t1,
                                                                   const IntegratorOptions& opt,
                                                                   const Observer<State>& obs = {}) {
  IntegratorResult<State> out{};
  out.t = t0;
  out.y = y0;

  if constexpr (!Tableau::has_embedded) {
    out.status = IntegratorStatus::InvalidStepSize;
    return out;
  }

  if (!(opt.rtol > 0.0) || !(opt.atol > 0.0) || !std::isfinite(opt.rtol) || !std::isfinite(opt.atol) ||
      !(opt.invariant_rtol > 0.0) || !std::isfinite(opt.invariant_rtol)) {
    out.status = IntegratorStatus::InvalidTolerance;
    return out;
  }
  if (!(opt.h_min > 0.0) || !(opt.h_max >= opt.h_min) || !std::isfinite(opt.h_min) || !std::isfinite(opt.h_max)) {
    out.status = IntegratorStatus::InvalidStepSize;
    return out;
  }
  if (opt.max_steps <= 0) {
    out.status = IntegratorStatus::MaxStepsExceeded;
    return out;
  }

  const int dir = detail::sign(t1 - t0);
  if (dir == 0) {
    return out;
  }

  const double span = std::abs(t1 - t0);
  double h = 0.0;
  if (opt.h_init > 0.0 && std::isfinite(opt.h_init)) {
    h = detail::clamp_abs_signed(static_cast<double>(dir) * opt.h_init, opt.h_min, opt.h_max, dir);
  } else {
    h = detail::estimate_initial_step<State, RHS, Algebra>(std::forward<RHS>(rhs),
                                                           t0, y0, t1, opt.atol, opt.rtol,
                                                           opt.h_min, opt.h_max, dir);
  }
  if (!(std::abs(h) > 0.0) || !std::isfinite(h) || span == 0.0) {
    h = static_cast<double>(dir) * opt.h_min;
  }

  const double invariant_init = invariant_fn(y0);
  if (!std::isfinite(invariant_init)) {
    out.status = IntegratorStatus::InvalidTolerance;
    return out;
  }
  const double invariant_global_scale = std::max(1.0, std::abs(invariant_init));

  InvariantStepSizeController controller{opt.safety, opt.fac_min, opt.fac_max, opt.safety};
  ExplicitRKStepper<Tableau, State, Algebra> stepper(y0);
  State y_high{};
  State err{};
  Algebra::resize_like(y_high, y0);
  Algebra::resize_like(err, y0);

  for (int step = 0; step < opt.max_steps; ++step) {
    if (out.t == t1) {
      out.status = IntegratorStatus::Success;
      return out;
    }

    const double remaining = t1 - out.t;
    if (detail::sign(remaining) != dir) {
      out.status = IntegratorStatus::Success;
      return out;
    }
    if (std::abs(h) > std::abs(remaining)) {
      h = remaining;
    }

    out.stats.attempted_steps += 1;
    const bool ok = stepper.step(rhs, out.t, out.y, h, y_high, &err);
    out.stats.rhs_evals += Tableau::stages;
    if (!ok || !Algebra::finite(y_high) || !Algebra::finite(err)) {
      out.status = IntegratorStatus::NaNDetected;
      return out;
    }

    const double err_norm = weighted_rms_error<State, Algebra>(err, out.y, y_high, opt.atol, opt.rtol);
    out.stats.last_h = h;
    out.stats.last_error_norm = err_norm;
    if (!std::isfinite(err_norm)) {
      out.status = IntegratorStatus::NaNDetected;
      return out;
    }

    const double invariant_current = invariant_fn(out.y);
    const double invariant_next = invariant_fn(y_high);
    if (!std::isfinite(invariant_current) || !std::isfinite(invariant_next)) {
      out.status = IntegratorStatus::NaNDetected;
      return out;
    }
    const double invariant_step_scale = std::max(1.0, std::abs(invariant_current));
    const double invariant_step_rel = std::abs(invariant_next - invariant_current) / invariant_step_scale;
    out.stats.last_invariant_error = std::abs(invariant_next - invariant_init) / invariant_global_scale;
    const double invariant_norm = invariant_step_rel / opt.invariant_rtol;

    const bool accept = (err_norm <= 1.0) && (invariant_norm <= 1.0);
    if (accept) {
      out.t += h;
      Algebra::assign(out.y, y_high);
      out.stats.accepted_steps += 1;
      if (!detail::call_observer(obs, out.t, out.y)) {
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

    const double h_new = controller.propose(
        h,
        (err_norm > 0.0 ? err_norm : std::numeric_limits<double>::min()),
        (invariant_norm > 0.0 ? invariant_norm : std::numeric_limits<double>::min()),
        Tableau::order_high);
    if (!std::isfinite(h_new) || std::abs(h_new) < opt.h_min) {
      out.status = IntegratorStatus::StepSizeUnderflow;
      return out;
    }

    h = detail::clamp_abs_signed(h_new, opt.h_min, opt.h_max, dir);
    if (std::abs(h) < opt.h_min) {
      out.status = IntegratorStatus::StepSizeUnderflow;
      return out;
    }
  }

  out.status = IntegratorStatus::MaxStepsExceeded;
  return out;
}

}  // namespace ode
