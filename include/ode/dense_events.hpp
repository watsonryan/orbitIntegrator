/**
 * @file dense_events.hpp
 * @brief Dense-output recording and event detection wrappers around the RK drivers.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <utility>
#include <vector>

#include "ode/algebra.hpp"
#include "ode/ode.hpp"

namespace ode {

enum class EventDirection {
  Any,
  Rising,
  Falling
};

template <class State>
struct DenseOutput {
  std::vector<double> times{};
  std::vector<State> states{};

  [[nodiscard]] bool sample_linear(double t_query, State& out) const {
    if (times.empty() || states.empty() || times.size() != states.size()) {
      return false;
    }
    if (t_query < times.front() || t_query > times.back()) {
      return false;
    }

    const auto it = std::lower_bound(times.begin(), times.end(), t_query);
    if (it == times.begin()) {
      out = states.front();
      return true;
    }
    if (it == times.end()) {
      out = states.back();
      return true;
    }

    const std::size_t i1 = static_cast<std::size_t>(std::distance(times.begin(), it));
    const std::size_t i0 = i1 - 1;
    const double t0 = times[i0];
    const double t1 = times[i1];
    const double alpha = (t_query - t0) / (t1 - t0);

    out = states[i0];
    for (std::size_t j = 0; j < out.size(); ++j) {
      out[j] = states[i0][j] + alpha * (states[i1][j] - states[i0][j]);
    }
    return true;
  }
};

template <class State>
struct DenseOutputOptions {
  bool record_accepted_steps{true};
  double uniform_sample_dt{0.0};
};

template <class State>
using EventFunction = std::function<double(double, const State&)>;

template <class State>
struct EventOptions {
  EventFunction<State> function{};
  EventDirection direction{EventDirection::Any};
  bool terminal{true};
  int max_events{32};
};

template <class State>
struct EventRecord {
  double t{};
  State y{};
  double g_before{};
  double g_after{};
};

template <class State>
struct DenseEventResult {
  IntegratorResult<State> integration{};
  DenseOutput<State> dense{};
  std::vector<EventRecord<State>> events{};
};

namespace detail {

inline bool EventCrossed(double g0, double g1, EventDirection direction) {
  if (!std::isfinite(g0) || !std::isfinite(g1)) {
    return false;
  }
  if (g0 == 0.0 || g1 == 0.0) {
    return true;
  }
  if ((g0 < 0.0 && g1 > 0.0) || (g0 > 0.0 && g1 < 0.0)) {
    if (direction == EventDirection::Any) {
      return true;
    }
    if (direction == EventDirection::Rising) {
      return g1 > g0;
    }
    return g1 < g0;
  }
  return false;
}

}  // namespace detail

template <class State, class RHS, class Algebra = DefaultAlgebra<State>>
requires AlgebraFor<Algebra, State>
[[nodiscard]] DenseEventResult<State> integrate_with_dense_events(
    RKMethod method,
    RHS&& rhs,
    double t0,
    const State& y0,
    double t1,
    IntegratorOptions opt,
    DenseOutputOptions<State> dense_opt = {},
    EventOptions<State> event_opt = {}) {
  DenseEventResult<State> out{};

  State y_prev = y0;
  double t_prev = t0;
  bool have_prev = true;
  double g_prev = 0.0;
  if (event_opt.function) {
    g_prev = event_opt.function(t0, y0);
  }

  double next_uniform_t = t0;
  if (dense_opt.uniform_sample_dt > 0.0) {
    const double dir = (t1 >= t0) ? 1.0 : -1.0;
    next_uniform_t = t0 + dir * dense_opt.uniform_sample_dt;
  }

  out.dense.times.push_back(t0);
  out.dense.states.push_back(y0);

  Observer<State> obs = [&](double t_curr, const State& y_curr) {
    if (dense_opt.record_accepted_steps) {
      out.dense.times.push_back(t_curr);
      out.dense.states.push_back(y_curr);
    }

    if (dense_opt.uniform_sample_dt > 0.0 && have_prev) {
      const double dt = dense_opt.uniform_sample_dt;
      const double dir = (t1 >= t0) ? 1.0 : -1.0;
      while ((dir > 0.0 && next_uniform_t < t_curr) || (dir < 0.0 && next_uniform_t > t_curr)) {
        const double alpha = (next_uniform_t - t_prev) / (t_curr - t_prev);
        State yi{};
        Algebra::resize_like(yi, y_prev);
        Algebra::assign(yi, y_prev);
        State dy{};
        Algebra::resize_like(dy, y_prev);
        Algebra::assign(dy, y_curr);
        Algebra::axpy(-1.0, y_prev, dy);
        Algebra::axpy(alpha, dy, yi);
        out.dense.times.push_back(next_uniform_t);
        out.dense.states.push_back(std::move(yi));
        next_uniform_t += dir * dt;
      }
    }

    if (event_opt.function && have_prev && static_cast<int>(out.events.size()) < event_opt.max_events) {
      const double g_curr = event_opt.function(t_curr, y_curr);
      if (detail::EventCrossed(g_prev, g_curr, event_opt.direction)) {
        double alpha = 0.5;
        const double denom = (g_prev - g_curr);
        if (std::abs(denom) > 0.0) {
          alpha = std::clamp(g_prev / denom, 0.0, 1.0);
        }

        EventRecord<State> ev{};
        ev.t = t_prev + alpha * (t_curr - t_prev);
        ev.y = y_prev;
        State dy{};
        Algebra::resize_like(dy, y_prev);
        Algebra::assign(dy, y_curr);
        Algebra::axpy(-1.0, y_prev, dy);
        Algebra::axpy(alpha, dy, ev.y);
        ev.g_before = g_prev;
        ev.g_after = g_curr;
        out.events.push_back(std::move(ev));

        if (event_opt.terminal) {
          t_prev = t_curr;
          y_prev = y_curr;
          g_prev = g_curr;
          return false;
        }
      }
      g_prev = g_curr;
    }

    t_prev = t_curr;
    y_prev = y_curr;
    have_prev = true;
    return true;
  };

  out.integration = integrate<State, RHS, Algebra>(method, std::forward<RHS>(rhs), t0, y0, t1, opt, obs);
  if (out.integration.status == IntegratorStatus::UserStopped && !out.events.empty()) {
    out.integration.status = IntegratorStatus::Success;
  }
  return out;
}

}  // namespace ode
