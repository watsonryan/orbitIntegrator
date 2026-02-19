/**
 * @file adams_bashforth_moulton.hpp
 * @brief Fixed-step Adams-Bashforth-Moulton predictor-corrector integrator (AB4/AM4).
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

#include "ode/algebra.hpp"
#include "ode/rk_stepper.hpp"
#include "ode/tableaus/rkf78.hpp"
#include "ode/types.hpp"

namespace ode::multistep {

enum class PredictorCorrectorMode {
  PEC,       // Predictor -> Evaluate(predicted) -> Correct
  PECE,      // Predictor -> Evaluate(predicted) -> Correct -> Evaluate(corrected)
  Iterated   // Repeated correct/evaluate cycles
};

struct AdamsBashforthMoultonOptions {
  double h = 0.0;
  int max_steps = 1000000;
  PredictorCorrectorMode mode = PredictorCorrectorMode::PECE;
  // Used only when mode == Iterated.
  int corrector_iterations = 1;
};

template <class State, class RHS, class Algebra = DefaultAlgebra<State>>
requires AlgebraFor<Algebra, State>
[[nodiscard]] IntegratorResult<State> integrate_abm4(
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

  std::array<State, 4> f_hist{};   // f_{n-3}, f_{n-2}, f_{n-1}, f_n
  for (auto& f : f_hist) {
    Algebra::resize_like(f, y0);
    Algebra::set_zero(f);
  }

  State y_next{};
  State y_pred{};
  State y_corr{};
  State f_pred{};
  State f_next{};
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

  // Bootstrap derivatives with RKF78 one-step propagation.
  if (!eval_rhs(out.t, out.y, f_hist[3])) {
    out.status = IntegratorStatus::NaNDetected;
    return out;
  }

  ExplicitRKStepper<TableauRKF78, State, Algebra> startup_stepper(y0);
  int accepted = 0;
  while (accepted < 3) {
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
    const bool ok = startup_stepper.step(rhs, out.t, out.y, h, y_next, nullptr);
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

    f_hist[0] = f_hist[1];
    f_hist[1] = f_hist[2];
    f_hist[2] = f_hist[3];
    if (!eval_rhs(out.t, out.y, f_hist[3])) {
      out.status = IntegratorStatus::NaNDetected;
      return out;
    }

    accepted += 1;
  }

  // Main AB4/AM4 loop.
  for (int step = 0; step < opt.max_steps; ++step) {
    const double rem = t1 - out.t;
    if ((dir > 0 && rem <= 0.0) || (dir < 0 && rem >= 0.0)) {
      out.status = IntegratorStatus::Success;
      return out;
    }

    // For non-uniform final step, fallback to RK startup stepper for accuracy/stability.
    if (std::abs(rem) < std::abs(h_fixed) * (1.0 - 1e-14)) {
      out.stats.attempted_steps += 1;
      const bool ok = startup_stepper.step(rhs, out.t, out.y, rem, y_next, nullptr);
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

    // AB4 predictor: y_{n+1}^{(p)} = y_n + h/24 * (55f_n - 59f_{n-1} + 37f_{n-2} - 9f_{n-3})
    Algebra::assign(y_pred, out.y);
    Algebra::axpy((h / 24.0) * 55.0, f_hist[3], y_pred);
    Algebra::axpy((h / 24.0) * -59.0, f_hist[2], y_pred);
    Algebra::axpy((h / 24.0) * 37.0, f_hist[1], y_pred);
    Algebra::axpy((h / 24.0) * -9.0, f_hist[0], y_pred);
    if (!Algebra::finite(y_pred)) {
      out.status = IntegratorStatus::NaNDetected;
      return out;
    }

    double t_next = out.t + h;
    if (!eval_rhs(t_next, y_pred, f_pred)) {
      out.status = IntegratorStatus::NaNDetected;
      return out;
    }

    auto apply_am4_corrector = [&](const State& f_np1_estimate, State& y_out) {
      Algebra::assign(y_out, out.y);
      // y_{n+1} = y_n + h/24 * (9f_{n+1} + 19f_n - 5f_{n-1} + f_{n-2})
      Algebra::axpy((h / 24.0) * 9.0, f_np1_estimate, y_out);
      Algebra::axpy((h / 24.0) * 19.0, f_hist[3], y_out);
      Algebra::axpy((h / 24.0) * -5.0, f_hist[2], y_out);
      Algebra::axpy((h / 24.0) * 1.0, f_hist[1], y_out);
    };

    if (opt.mode == PredictorCorrectorMode::PEC) {
      apply_am4_corrector(f_pred, y_corr);
      if (!Algebra::finite(y_corr)) {
        out.status = IntegratorStatus::NaNDetected;
        return out;
      }
      Algebra::assign(f_next, f_pred);
    } else if (opt.mode == PredictorCorrectorMode::PECE) {
      apply_am4_corrector(f_pred, y_corr);
      if (!Algebra::finite(y_corr)) {
        out.status = IntegratorStatus::NaNDetected;
        return out;
      }
      if (!eval_rhs(t_next, y_corr, f_next)) {
        out.status = IntegratorStatus::NaNDetected;
        return out;
      }
    } else {
      const int n_iter = std::max(1, opt.corrector_iterations);
      Algebra::assign(f_next, f_pred);
      for (int it = 0; it < n_iter; ++it) {
        apply_am4_corrector(f_next, y_corr);
        if (!Algebra::finite(y_corr)) {
          out.status = IntegratorStatus::NaNDetected;
          return out;
        }
        if (!eval_rhs(t_next, y_corr, f_next)) {
          out.status = IntegratorStatus::NaNDetected;
          return out;
        }
      }
    }

    if (!Algebra::finite(y_corr) || !Algebra::finite(f_next)) {
      out.status = IntegratorStatus::NaNDetected;
      return out;
    }

    // Accept and shift history.
    out.t = t_next;
    Algebra::assign(out.y, y_corr);
    out.stats.accepted_steps += 1;

    f_hist[0] = f_hist[1];
    f_hist[1] = f_hist[2];
    f_hist[2] = f_hist[3];
    f_hist[3] = f_next;

    if (obs && !obs(out.t, out.y)) {
      out.status = IntegratorStatus::UserStopped;
      return out;
    }
  }

  out.status = IntegratorStatus::MaxStepsExceeded;
  return out;
}

}  // namespace ode::multistep
