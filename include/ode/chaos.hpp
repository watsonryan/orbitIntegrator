/**
 * @file chaos.hpp
 * @brief Finite-time chaos indicators (FLI and MEGNO) from variational dynamics.
 */
#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "ode/integrate_method.hpp"
#include "ode/tableaus/rk4.hpp"
#include "ode/tableaus/rkf45.hpp"
#include "ode/tableaus/rkf78.hpp"
#include "ode/types.hpp"

namespace ode::chaos {

struct FliResult {
  IntegratorStatus status = IntegratorStatus::Success;
  double t = 0.0;
  double fli = 0.0;
  IntegratorStats stats{};
};

struct MegnoResult {
  IntegratorStatus status = IntegratorStatus::Success;
  double t = 0.0;
  double megno = 0.0;
  double mean_megno = 0.0;
  IntegratorStats stats{};
};

namespace detail {

[[nodiscard]] inline double dot(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    s += a[i] * b[i];
  }
  return s;
}

[[nodiscard]] inline double norm(const std::vector<double>& x) {
  return std::sqrt(dot(x, x));
}

template <class State, class RHS>
[[nodiscard]] inline IntegratorResult<State> integrate_runtime(RKMethod method,
                                                               RHS&& rhs,
                                                               double t0,
                                                               const State& y0,
                                                               double t1,
                                                               IntegratorOptions opt) {
  switch (method) {
    case RKMethod::RK4:
      opt.adaptive = false;
      return integrate_with_tableau<TableauRK4, State, RHS>(std::forward<RHS>(rhs), t0, y0, t1, opt, {});
    case RKMethod::RKF45:
      return integrate_with_tableau<TableauRKF45, State, RHS>(std::forward<RHS>(rhs), t0, y0, t1, opt, {});
    case RKMethod::RK8:
      opt.adaptive = false;
      return integrate_with_tableau<TableauRKF78, State, RHS>(std::forward<RHS>(rhs), t0, y0, t1, opt, {});
    case RKMethod::RKF78:
      return integrate_with_tableau<TableauRKF78, State, RHS>(std::forward<RHS>(rhs), t0, y0, t1, opt, {});
  }
  IntegratorResult<State> out{};
  out.status = IntegratorStatus::InvalidStepSize;
  return out;
}

}  // namespace detail

template <class RHS, class JacobianFn>
[[nodiscard]] FliResult compute_fli(RKMethod method,
                                    RHS&& rhs,
                                    JacobianFn&& jacobian_fn,
                                    double t0,
                                    const std::vector<double>& x0,
                                    const std::vector<double>& delta0,
                                    double t1,
                                    IntegratorOptions opt) {
  FliResult out{};
  const std::size_t n = x0.size();
  if (n == 0 || delta0.size() != n) {
    out.status = IntegratorStatus::InvalidStepSize;
    return out;
  }

  std::vector<double> y0(2 * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    y0[i] = x0[i];
    y0[n + i] = delta0[i];
  }
  const double d0 = detail::norm(delta0);
  if (!(d0 > 0.0) || !std::isfinite(d0)) {
    out.status = IntegratorStatus::InvalidStepSize;
    return out;
  }

  auto rhs_aug = [&](double t, const std::vector<double>& y, std::vector<double>& dydt) {
    dydt.assign(2 * n, 0.0);
    std::vector<double> x(n, 0.0), d(n, 0.0), dxdt, ddt(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      x[i] = y[i];
      d[i] = y[n + i];
    }

    rhs(t, x, dxdt);
    std::vector<double> a;
    const bool ok = jacobian_fn(t, x, a);
    if (!ok || dxdt.size() != n || a.size() != n * n) {
      std::fill(dydt.begin(), dydt.end(), std::numeric_limits<double>::quiet_NaN());
      return;
    }
    for (std::size_t i = 0; i < n; ++i) {
      dydt[i] = dxdt[i];
    }
    for (std::size_t i = 0; i < n; ++i) {
      double s = 0.0;
      for (std::size_t j = 0; j < n; ++j) {
        s += a[i * n + j] * d[j];
      }
      dydt[n + i] = s;
    }
  };

  const auto res = detail::integrate_runtime(method, rhs_aug, t0, y0, t1, opt);
  out.status = res.status;
  out.t = res.t;
  out.stats = res.stats;
  if (res.status != IntegratorStatus::Success || res.y.size() != 2 * n) {
    return out;
  }

  std::vector<double> d(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    d[i] = res.y[n + i];
  }
  const double dn = detail::norm(d);
  out.fli = std::log(std::max(dn / d0, std::numeric_limits<double>::min()));
  return out;
}

template <class RHS, class JacobianFn>
[[nodiscard]] MegnoResult compute_megno(RKMethod method,
                                        RHS&& rhs,
                                        JacobianFn&& jacobian_fn,
                                        double t0,
                                        const std::vector<double>& x0,
                                        const std::vector<double>& delta0,
                                        double t1,
                                        IntegratorOptions opt) {
  MegnoResult out{};
  const std::size_t n = x0.size();
  if (n == 0 || delta0.size() != n) {
    out.status = IntegratorStatus::InvalidStepSize;
    return out;
  }

  std::vector<double> y0(2 * n + 2, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    y0[i] = x0[i];
    y0[n + i] = delta0[i];
  }
  // y[2n] = Y accumulator, y[2n+1] = Z accumulator where mean MEGNO = Z / (t-t0).
  y0[2 * n] = 0.0;
  y0[2 * n + 1] = 0.0;

  auto rhs_aug = [&](double t, const std::vector<double>& y, std::vector<double>& dydt) {
    dydt.assign(2 * n + 2, 0.0);
    std::vector<double> x(n, 0.0), d(n, 0.0), dxdt, ddt(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      x[i] = y[i];
      d[i] = y[n + i];
    }

    rhs(t, x, dxdt);
    std::vector<double> a;
    const bool ok = jacobian_fn(t, x, a);
    if (!ok || dxdt.size() != n || a.size() != n * n) {
      std::fill(dydt.begin(), dydt.end(), std::numeric_limits<double>::quiet_NaN());
      return;
    }
    for (std::size_t i = 0; i < n; ++i) {
      dydt[i] = dxdt[i];
    }
    for (std::size_t i = 0; i < n; ++i) {
      double s = 0.0;
      for (std::size_t j = 0; j < n; ++j) {
        s += a[i * n + j] * d[j];
      }
      ddt[i] = s;
      dydt[n + i] = s;
    }

    const double dn2 = detail::dot(d, d);
    const double numer = detail::dot(d, ddt);
    const double tau = std::max(std::abs(t - t0), 1e-12);
    const double ydot = (dn2 > 0.0) ? 2.0 * (numer / dn2) * tau : 0.0;
    dydt[2 * n] = ydot;
    dydt[2 * n + 1] = y[2 * n];
  };

  const auto res = detail::integrate_runtime(method, rhs_aug, t0, y0, t1, opt);
  out.status = res.status;
  out.t = res.t;
  out.stats = res.stats;
  if (res.status != IntegratorStatus::Success || res.y.size() != 2 * n + 2) {
    return out;
  }

  out.megno = res.y[2 * n];
  const double dt = std::max(std::abs(res.t - t0), 1e-12);
  out.mean_megno = res.y[2 * n + 1] / dt;
  return out;
}

}  // namespace ode::chaos
