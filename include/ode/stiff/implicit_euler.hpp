/**
 * @file implicit_euler.hpp
 * @brief Basic stiff ODE module: implicit Euler with Newton iterations.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace ode::stiff {

enum class Status {
  Success,
  InvalidStepSize,
  MaxStepsExceeded,
  NewtonFailed,
  LinearSolveFailed,
  NaNDetected
};

struct Options {
  double h = 1e-3;
  int max_steps = 100000;
  double newton_tol = 1e-10;
  int newton_max_iter = 12;
  double fd_eps = 1e-8;
};

struct Stats {
  int steps = 0;
  int newton_iters = 0;
  int rhs_evals = 0;
};

struct Result {
  Status status{Status::Success};
  double t = 0.0;
  std::vector<double> y{};
  Stats stats{};
};

namespace detail {

inline double InfNorm(const std::vector<double>& v) {
  double m = 0.0;
  for (double x : v) {
    m = std::max(m, std::abs(x));
  }
  return m;
}

inline bool Finite(const std::vector<double>& v) {
  for (double x : v) {
    if (!std::isfinite(x)) {
      return false;
    }
  }
  return true;
}

inline bool SolveDense(std::vector<double> a, std::vector<double> b, std::size_t n, std::vector<double>& x) {
  for (std::size_t k = 0; k < n; ++k) {
    std::size_t piv = k;
    double max_val = std::abs(a[k * n + k]);
    for (std::size_t i = k + 1; i < n; ++i) {
      const double v = std::abs(a[i * n + k]);
      if (v > max_val) {
        max_val = v;
        piv = i;
      }
    }
    if (max_val < 1e-18) {
      return false;
    }
    if (piv != k) {
      for (std::size_t j = k; j < n; ++j) {
        std::swap(a[k * n + j], a[piv * n + j]);
      }
      std::swap(b[k], b[piv]);
    }

    const double diag = a[k * n + k];
    for (std::size_t i = k + 1; i < n; ++i) {
      const double f = a[i * n + k] / diag;
      if (f == 0.0) {
        continue;
      }
      for (std::size_t j = k; j < n; ++j) {
        a[i * n + j] -= f * a[k * n + j];
      }
      b[i] -= f * b[k];
    }
  }

  x.assign(n, 0.0);
  for (std::size_t ii = 0; ii < n; ++ii) {
    const std::size_t i = n - 1 - ii;
    double s = b[i];
    for (std::size_t j = i + 1; j < n; ++j) {
      s -= a[i * n + j] * x[j];
    }
    x[i] = s / a[i * n + i];
  }
  return Finite(x);
}

}  // namespace detail

template <class RHS>
[[nodiscard]] Result integrate_implicit_euler(RHS&& rhs,
                                              double t0,
                                              const std::vector<double>& y0,
                                              double t1,
                                              const Options& opt = {}) {
  Result out{};
  out.t = t0;
  out.y = y0;

  if (!(opt.h > 0.0) || !std::isfinite(opt.h) || opt.max_steps <= 0) {
    out.status = Status::InvalidStepSize;
    return out;
  }

  const int dir = (t1 > t0) - (t1 < t0);
  if (dir == 0) {
    return out;
  }

  const std::size_t n = y0.size();
  std::vector<double> y_prev(n), y_guess(n), f(n), f_pert(n), resid(n), delta(n), rhs_newton(n);
  std::vector<double> jac(n * n, 0.0);

  double h = static_cast<double>(dir) * opt.h;

  for (int step = 0; step < opt.max_steps; ++step) {
    const double remaining = t1 - out.t;
    if ((dir > 0 && remaining <= 0.0) || (dir < 0 && remaining >= 0.0)) {
      out.status = Status::Success;
      return out;
    }
    if (std::abs(h) > std::abs(remaining)) {
      h = remaining;
    }

    y_prev = out.y;
    y_guess = y_prev;
    const double t_next = out.t + h;

    bool newton_ok = false;
    for (int it = 0; it < opt.newton_max_iter; ++it) {
      rhs(t_next, y_guess, f);
      out.stats.rhs_evals += 1;
      if (!detail::Finite(f)) {
        out.status = Status::NaNDetected;
        return out;
      }

      for (std::size_t i = 0; i < n; ++i) {
        resid[i] = y_guess[i] - y_prev[i] - h * f[i];
      }

      if (detail::InfNorm(resid) < opt.newton_tol) {
        newton_ok = true;
        out.stats.newton_iters += (it + 1);
        break;
      }

      std::fill(jac.begin(), jac.end(), 0.0);
      for (std::size_t j = 0; j < n; ++j) {
        const double yj = y_guess[j];
        const double dy = opt.fd_eps * (1.0 + std::abs(yj));
        y_guess[j] = yj + dy;
        rhs(t_next, y_guess, f_pert);
        out.stats.rhs_evals += 1;
        y_guess[j] = yj;

        if (!detail::Finite(f_pert)) {
          out.status = Status::NaNDetected;
          return out;
        }

        for (std::size_t i = 0; i < n; ++i) {
          const double dfi_dyj = (f_pert[i] - f[i]) / dy;
          jac[i * n + j] = -h * dfi_dyj;
        }
      }
      for (std::size_t i = 0; i < n; ++i) {
        jac[i * n + i] += 1.0;
        rhs_newton[i] = -resid[i];
      }

      if (!detail::SolveDense(jac, rhs_newton, n, delta)) {
        out.status = Status::LinearSolveFailed;
        return out;
      }

      for (std::size_t i = 0; i < n; ++i) {
        y_guess[i] += delta[i];
      }
      if (!detail::Finite(y_guess)) {
        out.status = Status::NaNDetected;
        return out;
      }
      if (detail::InfNorm(delta) < opt.newton_tol) {
        newton_ok = true;
        out.stats.newton_iters += (it + 1);
        break;
      }
    }

    if (!newton_ok) {
      out.status = Status::NewtonFailed;
      return out;
    }

    out.y = y_guess;
    out.t = t_next;
    out.stats.steps += 1;
  }

  out.status = Status::MaxStepsExceeded;
  return out;
}

}  // namespace ode::stiff
