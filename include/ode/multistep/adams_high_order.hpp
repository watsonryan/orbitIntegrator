/**
 * @file adams_high_order.hpp
 * @brief Higher-order Adams-Bashforth-Moulton predictor-corrector integrators.
 */
#pragma once

#include <array>
#include <cmath>

#include "ode/algebra.hpp"
#include "ode/rk_stepper.hpp"
#include "ode/tableaus/rkf78.hpp"
#include "ode/types.hpp"

namespace ode::multistep {

template <class State, class RHS, class Algebra = DefaultAlgebra<State>>
requires AlgebraFor<Algebra, State>
[[nodiscard]] IntegratorResult<State> integrate_abm6(
    RHS&& rhs,
    double t0,
    const State& y0,
    double t1,
    AdamsBashforthMoultonOptions opt,
    Observer<State> obs = {}) {
  IntegratorResult<State> out{};
  out.t = t0;
  out.y = y0;

  if (!(opt.h > 0.0) || !std::isfinite(opt.h) || opt.max_steps <= 0) {
    out.status = IntegratorStatus::InvalidStepSize;
    return out;
  }
  if (opt.mode == PredictorCorrectorMode::Iterated && opt.corrector_iterations <= 0) {
    out.status = IntegratorStatus::InvalidStepSize;
    return out;
  }

  const int dir = (t1 > t0) - (t1 < t0);
  if (dir == 0) {
    return out;
  }
  const double h_fixed = static_cast<double>(dir) * opt.h;

  std::array<State, 6> f_hist{};  // f_{n-5},...,f_n
  for (auto& f : f_hist) {
    Algebra::resize_like(f, y0);
    Algebra::set_zero(f);
  }

  State y_next{}, y_pred{}, y_corr{}, f_pred{}, f_next{};
  Algebra::resize_like(y_next, y0);
  Algebra::resize_like(y_pred, y0);
  Algebra::resize_like(y_corr, y0);
  Algebra::resize_like(f_pred, y0);
  Algebra::resize_like(f_next, y0);

  auto eval_rhs = [&](double t, const State& y, State& dydt) -> bool {
    rhs(t, y, dydt);
    out.stats.rhs_evals += 1;
    return Algebra::finite(dydt);
  };

  if (!eval_rhs(out.t, out.y, f_hist[5])) {
    out.status = IntegratorStatus::NaNDetected;
    return out;
  }

  ExplicitRKStepper<TableauRKF78, State, Algebra> startup(y0);
  int accepted = 0;
  while (accepted < 5) {
    const double rem = t1 - out.t;
    if ((dir > 0 && rem <= 0.0) || (dir < 0 && rem >= 0.0)) {
      out.status = IntegratorStatus::Success;
      return out;
    }

    double h = h_fixed;
    if (std::abs(h) > std::abs(rem)) {
      h = rem;
    }

    out.stats.attempted_steps += 1;
    const bool ok = startup.step(rhs, out.t, out.y, h, y_next, nullptr);
    out.stats.rhs_evals += TableauRKF78::stages;
    out.stats.last_h = h;
    if (!ok || !Algebra::finite(y_next)) {
      out.status = IntegratorStatus::NaNDetected;
      return out;
    }

    out.t += h;
    Algebra::assign(out.y, y_next);
    out.stats.accepted_steps += 1;
    if (obs && !obs(out.t, out.y)) {
      out.status = IntegratorStatus::UserStopped;
      return out;
    }

    for (int i = 0; i < 5; ++i) {
      f_hist[static_cast<std::size_t>(i)] = f_hist[static_cast<std::size_t>(i + 1)];
    }
    if (!eval_rhs(out.t, out.y, f_hist[5])) {
      out.status = IntegratorStatus::NaNDetected;
      return out;
    }
    accepted += 1;
  }

  for (int step = 0; step < opt.max_steps; ++step) {
    const double rem = t1 - out.t;
    if ((dir > 0 && rem <= 0.0) || (dir < 0 && rem >= 0.0)) {
      out.status = IntegratorStatus::Success;
      return out;
    }

    if (std::abs(rem) < std::abs(h_fixed) * (1.0 - 1e-14)) {
      out.stats.attempted_steps += 1;
      const bool ok = startup.step(rhs, out.t, out.y, rem, y_next, nullptr);
      out.stats.rhs_evals += TableauRKF78::stages;
      out.stats.last_h = rem;
      if (!ok || !Algebra::finite(y_next)) {
        out.status = IntegratorStatus::NaNDetected;
        return out;
      }
      out.t += rem;
      Algebra::assign(out.y, y_next);
      out.stats.accepted_steps += 1;
      if (obs && !obs(out.t, out.y)) {
        out.status = IntegratorStatus::UserStopped;
        return out;
      }
      out.status = IntegratorStatus::Success;
      return out;
    }

    const double h = h_fixed;
    out.stats.attempted_steps += 1;
    out.stats.last_h = h;

    // AB6 predictor.
    Algebra::assign(y_pred, out.y);
    Algebra::axpy((h / 1440.0) * 4277.0, f_hist[5], y_pred);
    Algebra::axpy((h / 1440.0) * -7923.0, f_hist[4], y_pred);
    Algebra::axpy((h / 1440.0) * 9982.0, f_hist[3], y_pred);
    Algebra::axpy((h / 1440.0) * -7298.0, f_hist[2], y_pred);
    Algebra::axpy((h / 1440.0) * 2877.0, f_hist[1], y_pred);
    Algebra::axpy((h / 1440.0) * -475.0, f_hist[0], y_pred);
    if (!Algebra::finite(y_pred)) {
      out.status = IntegratorStatus::NaNDetected;
      return out;
    }

    const double t_next = out.t + h;
    if (!eval_rhs(t_next, y_pred, f_pred)) {
      out.status = IntegratorStatus::NaNDetected;
      return out;
    }

    auto apply_am6 = [&](const State& f_np1, State& y_out) {
      Algebra::assign(y_out, out.y);
      Algebra::axpy((h / 1440.0) * 475.0, f_np1, y_out);
      Algebra::axpy((h / 1440.0) * 1427.0, f_hist[5], y_out);
      Algebra::axpy((h / 1440.0) * -798.0, f_hist[4], y_out);
      Algebra::axpy((h / 1440.0) * 482.0, f_hist[3], y_out);
      Algebra::axpy((h / 1440.0) * -173.0, f_hist[2], y_out);
      Algebra::axpy((h / 1440.0) * 27.0, f_hist[1], y_out);
    };

    if (opt.mode == PredictorCorrectorMode::PEC) {
      apply_am6(f_pred, y_corr);
      if (!Algebra::finite(y_corr)) {
        out.status = IntegratorStatus::NaNDetected;
        return out;
      }
      Algebra::assign(f_next, f_pred);
    } else if (opt.mode == PredictorCorrectorMode::PECE) {
      apply_am6(f_pred, y_corr);
      if (!Algebra::finite(y_corr) || !eval_rhs(t_next, y_corr, f_next)) {
        out.status = IntegratorStatus::NaNDetected;
        return out;
      }
    } else {
      const int iters = std::max(1, opt.corrector_iterations);
      Algebra::assign(f_next, f_pred);
      for (int it = 0; it < iters; ++it) {
        apply_am6(f_next, y_corr);
        if (!Algebra::finite(y_corr) || !eval_rhs(t_next, y_corr, f_next)) {
          out.status = IntegratorStatus::NaNDetected;
          return out;
        }
      }
    }

    out.t = t_next;
    Algebra::assign(out.y, y_corr);
    out.stats.accepted_steps += 1;

    for (int i = 0; i < 5; ++i) {
      f_hist[static_cast<std::size_t>(i)] = f_hist[static_cast<std::size_t>(i + 1)];
    }
    f_hist[5] = f_next;

    if (obs && !obs(out.t, out.y)) {
      out.status = IntegratorStatus::UserStopped;
      return out;
    }
  }

  out.status = IntegratorStatus::MaxStepsExceeded;
  return out;
}

}  // namespace ode::multistep
