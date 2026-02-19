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
  double h = (opt.h_init > 0.0) ? opt.h_init : span / 100.0;
  if (!(h > 0.0) || !std::isfinite(h)) {
    h = opt.h_min;
  }
  h = detail::clamp_abs_signed(static_cast<double>(dir) * h, opt.h_min, opt.h_max, dir);

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

    const double h_new = controller.propose(h, (err_norm > 0.0 ? err_norm : std::numeric_limits<double>::min()),
                                            Tableau::order_high);

    if (!std::isfinite(h_new)) {
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
