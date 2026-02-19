/**
 * @file poincare.hpp
 * @brief Poincare section utilities and single-shooting periodic-orbit helpers.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "ode/integrate_method.hpp"
#include "ode/tableaus/rk4.hpp"
#include "ode/tableaus/rkf45.hpp"
#include "ode/tableaus/rkf78.hpp"
#include "ode/types.hpp"

namespace ode::poincare {

enum class CrossingDirection {
  Any,
  Positive,
  Negative
};

template <class State>
struct SectionCrossing {
  double t = 0.0;
  State y{};
  double section_value = 0.0;
};

template <class State>
struct PoincareResult {
  IntegratorStatus status = IntegratorStatus::Success;
  std::vector<SectionCrossing<State>> crossings{};
  IntegratorStats stats{};
};

namespace detail {

template <class State>
[[nodiscard]] inline State interpolate_state(const State& a, const State& b, double alpha) {
  State out = a;
  const std::size_t n = a.size();
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = a[i] + alpha * (b[i] - a[i]);
  }
  return out;
}

[[nodiscard]] inline bool direction_ok(double s0, double s1, CrossingDirection direction) {
  if (direction == CrossingDirection::Any) {
    return (s0 <= 0.0 && s1 >= 0.0) || (s0 >= 0.0 && s1 <= 0.0);
  }
  if (direction == CrossingDirection::Positive) {
    return s0 < 0.0 && s1 >= 0.0;
  }
  return s0 > 0.0 && s1 <= 0.0;
}

template <class State, class RHS>
[[nodiscard]] inline IntegratorResult<State> integrate_runtime(RKMethod method,
                                                               RHS&& rhs,
                                                               double t0,
                                                               const State& y0,
                                                               double t1,
                                                               IntegratorOptions opt,
                                                               Observer<State> obs) {
  switch (method) {
    case RKMethod::RK4:
      opt.adaptive = false;
      return integrate_with_tableau<TableauRK4, State, RHS>(std::forward<RHS>(rhs), t0, y0, t1, opt, obs);
    case RKMethod::RKF45:
      return integrate_with_tableau<TableauRKF45, State, RHS>(std::forward<RHS>(rhs), t0, y0, t1, opt, obs);
    case RKMethod::RK8:
      opt.adaptive = false;
      return integrate_with_tableau<TableauRKF78, State, RHS>(std::forward<RHS>(rhs), t0, y0, t1, opt, obs);
    case RKMethod::RKF78:
      return integrate_with_tableau<TableauRKF78, State, RHS>(std::forward<RHS>(rhs), t0, y0, t1, opt, obs);
  }
  IntegratorResult<State> out{};
  out.status = IntegratorStatus::InvalidStepSize;
  return out;
}

}  // namespace detail

template <class State, class RHS, class SectionFn>
[[nodiscard]] PoincareResult<State> integrate_poincare(RKMethod method,
                                                       RHS&& rhs,
                                                       SectionFn&& section_fn,
                                                       double t0,
                                                       const State& y0,
                                                       double t1,
                                                       IntegratorOptions opt,
                                                       std::size_t max_crossings,
                                                       CrossingDirection direction = CrossingDirection::Any) {
  PoincareResult<State> out{};
  if (max_crossings == 0) {
    return out;
  }

  bool have_prev = false;
  double prev_t = t0;
  State prev_y = y0;
  double prev_s = section_fn(t0, y0);

  const Observer<State> obs = [&](double t, const State& y) {
    const double s = section_fn(t, y);
    if (!std::isfinite(s)) {
      return false;
    }
    if (have_prev) {
      if (detail::direction_ok(prev_s, s, direction)) {
        const double denom = (s - prev_s);
        const double alpha = (std::abs(denom) > std::numeric_limits<double>::min())
                                 ? std::clamp((-prev_s) / denom, 0.0, 1.0)
                                 : 0.0;
        SectionCrossing<State> cross{};
        cross.t = prev_t + alpha * (t - prev_t);
        cross.y = detail::interpolate_state(prev_y, y, alpha);
        cross.section_value = section_fn(cross.t, cross.y);
        out.crossings.push_back(std::move(cross));
        if (out.crossings.size() >= max_crossings) {
          return false;
        }
      }
    }
    have_prev = true;
    prev_t = t;
    prev_y = y;
    prev_s = s;
    return true;
  };

  const auto res = detail::integrate_runtime<State>(method, std::forward<RHS>(rhs), t0, y0, t1, opt, obs);
  out.status = (res.status == IntegratorStatus::UserStopped && out.crossings.size() >= max_crossings)
                   ? IntegratorStatus::Success
                   : res.status;
  out.stats = res.stats;
  return out;
}

template <class State, class MapFn>
inline void periodic_residual(MapFn&& poincare_map, const State& x, State& r) {
  const State px = poincare_map(x);
  r = px;
  for (std::size_t i = 0; i < x.size(); ++i) {
    r[i] -= x[i];
  }
}

template <class State, class MapFn>
[[nodiscard]] inline bool finite_difference_jacobian(MapFn&& poincare_map,
                                                      const State& x,
                                                      double eps,
                                                      std::vector<double>& j) {
  const std::size_t n = x.size();
  j.assign(n * n, 0.0);
  if (!(eps > 0.0) || !std::isfinite(eps)) {
    return false;
  }
  State f0{};
  periodic_residual(poincare_map, x, f0);
  if (f0.size() != n) {
    return false;
  }
  for (std::size_t k = 0; k < n; ++k) {
    State xp = x;
    xp[k] += eps;
    State fp{};
    periodic_residual(poincare_map, xp, fp);
    if (fp.size() != n) {
      return false;
    }
    for (std::size_t i = 0; i < n; ++i) {
      j[i * n + k] = (fp[i] - f0[i]) / eps;
    }
  }
  return true;
}

[[nodiscard]] inline bool solve_linear(std::vector<double> a,
                                       std::vector<double> b,
                                       std::size_t n,
                                       std::vector<double>& x) {
  for (std::size_t i = 0; i < n; ++i) {
    std::size_t piv = i;
    double piv_abs = std::abs(a[i * n + i]);
    for (std::size_t r = i + 1; r < n; ++r) {
      const double v = std::abs(a[r * n + i]);
      if (v > piv_abs) {
        piv_abs = v;
        piv = r;
      }
    }
    if (piv_abs < 1e-14) {
      return false;
    }
    if (piv != i) {
      for (std::size_t c = i; c < n; ++c) {
        std::swap(a[i * n + c], a[piv * n + c]);
      }
      std::swap(b[i], b[piv]);
    }
    const double d = a[i * n + i];
    for (std::size_t c = i; c < n; ++c) {
      a[i * n + c] /= d;
    }
    b[i] /= d;
    for (std::size_t r = i + 1; r < n; ++r) {
      const double f = a[r * n + i];
      if (f == 0.0) {
        continue;
      }
      for (std::size_t c = i; c < n; ++c) {
        a[r * n + c] -= f * a[i * n + c];
      }
      b[r] -= f * b[i];
    }
  }

  x.assign(n, 0.0);
  for (std::size_t ii = 0; ii < n; ++ii) {
    const std::size_t i = n - 1 - ii;
    double s = b[i];
    for (std::size_t c = i + 1; c < n; ++c) {
      s -= a[i * n + c] * x[c];
    }
    x[i] = s;
  }
  return true;
}

template <class State>
[[nodiscard]] inline double norm_inf(const State& x) {
  double m = 0.0;
  for (double v : x) {
    m = std::max(m, std::abs(v));
  }
  return m;
}

template <class State>
struct DifferentialCorrectionResult {
  bool success = false;
  State corrected{};
  State residual{};
  double residual_inf = 0.0;
};

template <class State, class MapFn>
[[nodiscard]] DifferentialCorrectionResult<State> differential_correction(MapFn&& poincare_map,
                                                                          const State& x0,
                                                                          double fd_eps = 1e-6) {
  DifferentialCorrectionResult<State> out{};
  const std::size_t n = x0.size();
  if (n == 0) {
    return out;
  }

  State r{};
  periodic_residual(poincare_map, x0, r);
  out.residual = r;
  out.residual_inf = norm_inf(r);

  std::vector<double> j{};
  if (!finite_difference_jacobian(poincare_map, x0, fd_eps, j)) {
    return out;
  }
  std::vector<double> rhs(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    rhs[i] = -r[i];
  }

  std::vector<double> dx{};
  if (!solve_linear(j, rhs, n, dx)) {
    return out;
  }

  out.corrected = x0;
  for (std::size_t i = 0; i < n; ++i) {
    out.corrected[i] += dx[i];
  }
  State r1{};
  periodic_residual(poincare_map, out.corrected, r1);
  out.residual = r1;
  out.residual_inf = norm_inf(r1);
  out.success = std::isfinite(out.residual_inf);
  return out;
}

}  // namespace ode::poincare
