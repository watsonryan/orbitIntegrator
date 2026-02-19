/**
 * @file nordsieck_abm4.hpp
 * @brief Nordsieck-style adaptive multistep predictor-corrector integrators.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "ode/multistep/adams_bashforth_moulton.hpp"
#include "ode/multistep/adams_high_order.hpp"

namespace ode::multistep {

struct NordsieckAbmOptions {
  double rtol = 1e-8;
  double atol = 1e-12;

  double h_init = 0.0;
  double h_min = 1e-14;
  double h_max = 1.0;

  int max_steps = 1000000;

  double safety = 0.9;
  double fac_min = 0.2;
  double fac_max = 5.0;
  int segment_steps = 8;
  int max_restarts = 16;
};

namespace detail {

[[nodiscard]] inline double component_scale(double yi, double yref, double atol, double rtol) {
  return atol + rtol * std::max(std::abs(yi), std::abs(yref));
}

template <class State, class Algebra = DefaultAlgebra<State>>
requires AlgebraFor<Algebra, State>
[[nodiscard]] inline double normalized_error(const State& coarse,
                                             const State& fine,
                                             const State& ref,
                                             double atol,
                                             double rtol) {
  const std::size_t n = Algebra::size(ref);
  if (Algebra::size(coarse) != n || Algebra::size(fine) != n || n == 0) {
    return std::numeric_limits<double>::infinity();
  }
  double err_max = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double sc = component_scale(ref[i], fine[i], atol, rtol);
    const double e = std::abs(coarse[i] - fine[i]) / sc;
    err_max = std::max(err_max, e);
  }
  return err_max;
}

inline void accumulate_stats(IntegratorStats& dst, const IntegratorStats& src) {
  dst.attempted_steps += src.attempted_steps;
  dst.accepted_steps += src.accepted_steps;
  dst.rejected_steps += src.rejected_steps;
  dst.rhs_evals += src.rhs_evals;
  dst.last_h = src.last_h;
}

template <class State,
          class RHS,
          class Algebra = DefaultAlgebra<State>,
          class HighSegmentIntegrator,
          class LowSegmentIntegrator>
requires AlgebraFor<Algebra, State>
[[nodiscard]] IntegratorResult<State> integrate_nordsieck_adaptive_driver(
    RHS&& rhs,
    HighSegmentIntegrator&& high_segment_integrator,
    LowSegmentIntegrator&& low_segment_integrator,
    int error_order,
    double t0,
    const State& y0,
    double t1,
    NordsieckAbmOptions opt,
    Observer<State> obs = {}) {
  IntegratorResult<State> out{};
  out.t = t0;
  out.y = y0;
  auto& rhs_ref = rhs;

  if (opt.max_steps <= 0 || opt.max_restarts <= 0 || opt.segment_steps <= 0 || !(opt.h_min > 0.0) ||
      !(opt.h_max >= opt.h_min) || !(opt.rtol > 0.0) || !(opt.atol > 0.0) || error_order <= 0) {
    out.status = IntegratorStatus::InvalidStepSize;
    return out;
  }

  const int dir = (t1 > t0) - (t1 < t0);
  if (dir == 0) {
    return out;
  }

  double h = opt.h_init;
  if (!(h > 0.0) || !std::isfinite(h)) {
    h = std::abs(t1 - t0) / 200.0;
  }
  h = std::clamp(h, opt.h_min, opt.h_max);

  for (int seg = 0; seg < opt.max_steps; ++seg) {
    const double rem = t1 - out.t;
    if ((dir > 0 && rem <= 0.0) || (dir < 0 && rem >= 0.0)) {
      out.status = IntegratorStatus::Success;
      return out;
    }

    const double step_mag = std::min(h, std::abs(rem) / static_cast<double>(opt.segment_steps));
    if (!(step_mag > 0.0) || !std::isfinite(step_mag)) {
      out.status = IntegratorStatus::InvalidStepSize;
      return out;
    }
    const double h_seg = static_cast<double>(dir) * step_mag;
    const double t_seg_end = out.t + static_cast<double>(opt.segment_steps) * h_seg;

    bool accepted = false;
    State accepted_state{};
    IntegratorStats accepted_stats{};
    bool have_fallback = false;
    State fallback_state{};
    IntegratorStats fallback_stats{};

    for (int restart = 0; restart < opt.max_restarts; ++restart) {
      const auto high = high_segment_integrator(rhs_ref, out.t, out.y, t_seg_end, h_seg, opt, {});
      out.stats.attempted_steps += 1;
      if (high.status != IntegratorStatus::Success || !Algebra::finite(high.y)) {
        h = std::max(opt.h_min, 0.5 * std::abs(h_seg));
        out.stats.rejected_steps += 1;
        continue;
      }

      const auto low = low_segment_integrator(rhs_ref, out.t, out.y, t_seg_end, h_seg, opt, {});
      if (low.status != IntegratorStatus::Success || !Algebra::finite(low.y)) {
        h = std::max(opt.h_min, 0.5 * std::abs(h_seg));
        out.stats.rejected_steps += 1;
        continue;
      }
      const double err = normalized_error<State, Algebra>(low.y, high.y, high.y, opt.atol, opt.rtol);
      const bool ok = std::isfinite(err) && err <= 1.0;
      const double fac_raw = opt.safety * std::pow(std::max(err, 1e-14), -1.0 / static_cast<double>(error_order + 1));
      const double fac = std::clamp(fac_raw, opt.fac_min, opt.fac_max);

      if (std::isfinite(err)) {
        have_fallback = true;
        fallback_state = high.y;
        fallback_stats = {};
        accumulate_stats(fallback_stats, high.stats);
        accumulate_stats(fallback_stats, low.stats);
      }

      if (ok) {
        accepted = true;
        accepted_state = high.y;
        accumulate_stats(accepted_stats, high.stats);
        accumulate_stats(accepted_stats, low.stats);
        h = std::clamp(std::abs(h_seg) * fac, opt.h_min, opt.h_max);
        break;
      }

      out.stats.rejected_steps += 1;
      h = std::clamp(std::abs(h_seg) * std::max(opt.fac_min, 0.5 * fac), opt.h_min, opt.h_max);
    }

    if (!accepted) {
      if (!have_fallback) {
        out.status = IntegratorStatus::MaxStepsExceeded;
        return out;
      }
      accepted = true;
      accepted_state = fallback_state;
      accepted_stats = fallback_stats;
      h = std::max(opt.h_min, 0.5 * std::abs(h_seg));
    }

    out.t = t_seg_end;
    Algebra::assign(out.y, accepted_state);
    out.stats.accepted_steps += 1;
    out.stats.last_h = static_cast<double>(dir) * h;
    out.stats.rhs_evals += accepted_stats.rhs_evals;

    if (obs && !obs(out.t, out.y)) {
      out.status = IntegratorStatus::UserStopped;
      return out;
    }
  }

  out.status = IntegratorStatus::MaxStepsExceeded;
  return out;
}

}  // namespace detail

template <class State, class RHS, class Algebra = DefaultAlgebra<State>>
requires AlgebraFor<Algebra, State>
[[nodiscard]] IntegratorResult<State> integrate_nordsieck_abm4(
    RHS&& rhs,
    double t0,
    const State& y0,
    double t1,
    NordsieckAbmOptions opt,
    Observer<State> obs = {}) {
  const auto high_segment_integrator = [](RHS& rhs_fn,
                                          double ta,
                                          const State& ya,
                                          double tb,
                                          double h_step,
                                          const NordsieckAbmOptions& nopt,
                                          Observer<State> o) {
    AdamsBashforthMoultonOptions abm_opt{};
    abm_opt.h = std::abs(h_step);
    abm_opt.max_steps = std::max(1, nopt.segment_steps + 8);
    abm_opt.mode = PredictorCorrectorMode::Iterated;
    abm_opt.corrector_iterations = 2;
    return integrate_abm4<State, RHS, Algebra>(rhs_fn, ta, ya, tb, abm_opt, std::move(o));
  };
  const auto low_segment_integrator = [](RHS& rhs_fn,
                                         double ta,
                                         const State& ya,
                                         double tb,
                                         double h_step,
                                         const NordsieckAbmOptions& nopt,
                                         Observer<State> o) {
    AdamsBashforthMoultonOptions abm_opt{};
    abm_opt.h = std::abs(h_step);
    abm_opt.max_steps = std::max(1, nopt.segment_steps + 8);
    abm_opt.mode = PredictorCorrectorMode::PECE;
    abm_opt.corrector_iterations = 1;
    return integrate_abm4<State, RHS, Algebra>(rhs_fn, ta, ya, tb, abm_opt, std::move(o));
  };
  return detail::integrate_nordsieck_adaptive_driver<State, RHS, Algebra>(
      std::forward<RHS>(rhs), high_segment_integrator, low_segment_integrator, 4, t0, y0, t1, opt, std::move(obs));
}

template <class State, class RHS, class Algebra = DefaultAlgebra<State>>
requires AlgebraFor<Algebra, State>
[[nodiscard]] IntegratorResult<State> integrate_nordsieck_abm6(
    RHS&& rhs,
    double t0,
    const State& y0,
    double t1,
    NordsieckAbmOptions opt,
    Observer<State> obs = {}) {
  const auto high_segment_integrator = [](RHS& rhs_fn,
                                          double ta,
                                          const State& ya,
                                          double tb,
                                          double h_step,
                                          const NordsieckAbmOptions& nopt,
                                          Observer<State> o) {
    AdamsBashforthMoultonOptions abm_opt{};
    abm_opt.h = std::abs(h_step);
    abm_opt.max_steps = std::max(1, nopt.segment_steps + 12);
    abm_opt.mode = PredictorCorrectorMode::Iterated;
    abm_opt.corrector_iterations = 2;
    return integrate_abm6<State, RHS, Algebra>(rhs_fn, ta, ya, tb, abm_opt, std::move(o));
  };
  const auto low_segment_integrator = [](RHS& rhs_fn,
                                         double ta,
                                         const State& ya,
                                         double tb,
                                         double h_step,
                                         const NordsieckAbmOptions& nopt,
                                         Observer<State> o) {
    AdamsBashforthMoultonOptions abm_opt{};
    abm_opt.h = std::abs(h_step);
    abm_opt.max_steps = std::max(1, nopt.segment_steps + 12);
    abm_opt.mode = PredictorCorrectorMode::PECE;
    abm_opt.corrector_iterations = 1;
    return integrate_abm6<State, RHS, Algebra>(rhs_fn, ta, ya, tb, abm_opt, std::move(o));
  };
  return detail::integrate_nordsieck_adaptive_driver<State, RHS, Algebra>(
      std::forward<RHS>(rhs), high_segment_integrator, low_segment_integrator, 6, t0, y0, t1, opt, std::move(obs));
}

}  // namespace ode::multistep
