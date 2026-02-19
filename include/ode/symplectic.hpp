/**
 * @file symplectic.hpp
 * @brief Fixed-step symplectic integrators for separable second-order systems.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "ode/types.hpp"

namespace ode::symplectic {

struct SymplecticOptions {
  double h = 0.0;
  int max_steps = 1000000;
};

struct SecondOrderResult {
  IntegratorStatus status = IntegratorStatus::Success;
  double t = 0.0;
  std::vector<double> r{};
  std::vector<double> v{};
  IntegratorStats stats{};
};

struct SymplecticVariationalResult {
  IntegratorStatus status = IntegratorStatus::Success;
  double t = 0.0;
  std::vector<double> r{};
  std::vector<double> v{};
  std::vector<double> phi{};  // Row-major (2n)x(2n).
  IntegratorStats stats{};
};

struct SundmanSymplecticOptions {
  double ds = 0.0;
  int max_steps = 1000000;
  double g_min = 1e-8;
  double g_max = 1e8;
};

[[nodiscard]] inline bool finite_vec(const std::vector<double>& x) {
  for (double v : x) {
    if (!std::isfinite(v)) {
      return false;
    }
  }
  return true;
}

template <class AccelerationFn>
[[nodiscard]] inline bool verlet_single_step(AccelerationFn&& acceleration,
                                             double t,
                                             double h,
                                             const std::vector<double>& r,
                                             const std::vector<double>& v,
                                             std::vector<double>& a0,
                                             std::vector<double>& r_next,
                                             std::vector<double>& v_next,
                                             std::vector<double>& a1,
                                             IntegratorStats& stats) {
  const std::size_t n = r.size();
  if (a0.size() != n || r_next.size() != n || v_next.size() != n || a1.size() != n) {
    return false;
  }

  acceleration(t, r, a0);
  stats.rhs_evals += 1;
  if (!finite_vec(a0)) {
    return false;
  }

  for (std::size_t i = 0; i < n; ++i) {
    const double v_half = v[i] + 0.5 * h * a0[i];
    r_next[i] = r[i] + h * v_half;
    v_next[i] = v_half;
  }

  acceleration(t + h, r_next, a1);
  stats.rhs_evals += 1;
  if (!finite_vec(a1) || !finite_vec(r_next)) {
    return false;
  }

  for (std::size_t i = 0; i < n; ++i) {
    v_next[i] += 0.5 * h * a1[i];
  }

  return finite_vec(v_next);
}

inline void matvec(const std::vector<double>& a,
                   std::size_t n,
                   const std::vector<double>& x,
                   std::vector<double>& y) {
  y.assign(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    double s = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      s += a[i * n + j] * x[j];
    }
    y[i] = s;
  }
}

template <class AccelerationFn>
[[nodiscard]] SecondOrderResult integrate_stormer_verlet(AccelerationFn&& acceleration,
                                                         double t0,
                                                         const std::vector<double>& r0,
                                                         const std::vector<double>& v0,
                                                         double t1,
                                                         SymplecticOptions opt,
                                                         Observer<std::vector<double>> obs = {}) {
  SecondOrderResult out{};
  out.t = t0;
  out.r = r0;
  out.v = v0;

  const std::size_t n = r0.size();
  if (n == 0 || v0.size() != n || !(opt.h > 0.0) || !std::isfinite(opt.h) || opt.max_steps <= 0) {
    out.status = IntegratorStatus::InvalidStepSize;
    return out;
  }

  const int dir = (t1 > t0) - (t1 < t0);
  if (dir == 0) {
    return out;
  }

  const double h_fixed = static_cast<double>(dir) * opt.h;
  std::vector<double> a0(n, 0.0), a1(n, 0.0), r_next(n, 0.0), v_next(n, 0.0);

  auto&& accel = acceleration;
  for (int step = 0; step < opt.max_steps; ++step) {
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
    out.stats.last_h = h;
    if (!verlet_single_step(accel,
                            out.t, h, out.r, out.v, a0, r_next, v_next, a1, out.stats)) {
      out.status = IntegratorStatus::NaNDetected;
      return out;
    }

    out.r = r_next;
    out.v = v_next;
    out.t += h;
    out.stats.accepted_steps += 1;

    if (obs && !obs(out.t, out.r)) {
      out.status = IntegratorStatus::UserStopped;
      return out;
    }
  }

  out.status = IntegratorStatus::MaxStepsExceeded;
  return out;
}

template <class AccelerationFn>
[[nodiscard]] SecondOrderResult integrate_yoshida4(AccelerationFn&& acceleration,
                                                   double t0,
                                                   const std::vector<double>& r0,
                                                   const std::vector<double>& v0,
                                                   double t1,
                                                   SymplecticOptions opt,
                                                   Observer<std::vector<double>> obs = {}) {
  SecondOrderResult out{};
  out.t = t0;
  out.r = r0;
  out.v = v0;

  const std::size_t n = r0.size();
  if (n == 0 || v0.size() != n || !(opt.h > 0.0) || !std::isfinite(opt.h) || opt.max_steps <= 0) {
    out.status = IntegratorStatus::InvalidStepSize;
    return out;
  }

  const int dir = (t1 > t0) - (t1 < t0);
  if (dir == 0) {
    return out;
  }

  const double h_fixed = static_cast<double>(dir) * opt.h;
  constexpr double cbrt2 = 1.2599210498948731648;
  constexpr double w1 = 1.0 / (2.0 - cbrt2);
  constexpr double w0 = -cbrt2 / (2.0 - cbrt2);

  std::vector<double> a0(n, 0.0), a1(n, 0.0), r_tmp(n, 0.0), v_tmp(n, 0.0);
  std::vector<double> r_work = out.r;
  std::vector<double> v_work = out.v;

  auto&& accel = acceleration;
  for (int step = 0; step < opt.max_steps; ++step) {
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
    out.stats.last_h = h;

    const double h1 = w1 * h;
    const double h0 = w0 * h;

    if (!verlet_single_step(accel,
                            out.t, h1, out.r, out.v, a0, r_tmp, v_tmp, a1, out.stats)) {
      out.status = IntegratorStatus::NaNDetected;
      return out;
    }
    r_work = r_tmp;
    v_work = v_tmp;

    if (!verlet_single_step(accel,
                            out.t + h1, h0, r_work, v_work, a0, r_tmp, v_tmp, a1, out.stats)) {
      out.status = IntegratorStatus::NaNDetected;
      return out;
    }
    r_work = r_tmp;
    v_work = v_tmp;

    if (!verlet_single_step(accel,
                            out.t + h1 + h0, h1, r_work, v_work, a0, r_tmp, v_tmp, a1, out.stats)) {
      out.status = IntegratorStatus::NaNDetected;
      return out;
    }

    out.r = r_tmp;
    out.v = v_tmp;
    out.t += h;
    out.stats.accepted_steps += 1;

    if (obs && !obs(out.t, out.r)) {
      out.status = IntegratorStatus::UserStopped;
      return out;
    }
  }

  out.status = IntegratorStatus::MaxStepsExceeded;
  return out;
}

template <class AccelerationFn, class SundmanFn>
[[nodiscard]] SecondOrderResult integrate_stormer_verlet_sundman(AccelerationFn&& acceleration,
                                                                 SundmanFn&& sundman_g,
                                                                 double t0,
                                                                 const std::vector<double>& r0,
                                                                 const std::vector<double>& v0,
                                                                 double t1,
                                                                 SundmanSymplecticOptions opt,
                                                                 Observer<std::vector<double>> obs = {}) {
  SecondOrderResult out{};
  out.t = t0;
  out.r = r0;
  out.v = v0;

  const std::size_t n = r0.size();
  if (n == 0 || v0.size() != n || !(opt.ds > 0.0) || !std::isfinite(opt.ds) || opt.max_steps <= 0 ||
      !(opt.g_min > 0.0) || !(opt.g_max >= opt.g_min)) {
    out.status = IntegratorStatus::InvalidStepSize;
    return out;
  }

  const int dir = (t1 > t0) - (t1 < t0);
  if (dir == 0) {
    return out;
  }
  const double ds_signed = static_cast<double>(dir) * opt.ds;
  std::vector<double> a0(n, 0.0), a1(n, 0.0), r_next(n, 0.0), v_next(n, 0.0);
  auto&& accel = acceleration;
  auto&& g_fn = sundman_g;

  for (int step = 0; step < opt.max_steps; ++step) {
    const double rem_t = t1 - out.t;
    if ((dir > 0 && rem_t <= 0.0) || (dir < 0 && rem_t >= 0.0)) {
      out.status = IntegratorStatus::Success;
      return out;
    }

    double g = g_fn(out.t, out.r, out.v);
    if (!std::isfinite(g)) {
      out.status = IntegratorStatus::NaNDetected;
      return out;
    }
    g = std::clamp(g, opt.g_min, opt.g_max);

    double dt = ds_signed * g;
    if (std::abs(dt) > std::abs(rem_t)) {
      dt = rem_t;
    }

    out.stats.attempted_steps += 1;
    out.stats.last_h = dt;
    if (!verlet_single_step(accel, out.t, dt, out.r, out.v, a0, r_next, v_next, a1, out.stats)) {
      out.status = IntegratorStatus::NaNDetected;
      return out;
    }

    out.r = r_next;
    out.v = v_next;
    out.t += dt;
    out.stats.accepted_steps += 1;

    if (obs && !obs(out.t, out.r)) {
      out.status = IntegratorStatus::UserStopped;
      return out;
    }
  }

  out.status = IntegratorStatus::MaxStepsExceeded;
  return out;
}

template <class AccelerationFn, class JacobianFn>
[[nodiscard]] SymplecticVariationalResult integrate_stormer_verlet_variational(
    AccelerationFn&& acceleration,
    JacobianFn&& jacobian_r,
    double t0,
    const std::vector<double>& r0,
    const std::vector<double>& v0,
    double t1,
    SymplecticOptions opt,
    Observer<std::vector<double>> obs = {}) {
  SymplecticVariationalResult out{};
  out.t = t0;
  out.r = r0;
  out.v = v0;

  const std::size_t n = r0.size();
  if (n == 0 || v0.size() != n || !(opt.h > 0.0) || !std::isfinite(opt.h) || opt.max_steps <= 0) {
    out.status = IntegratorStatus::InvalidStepSize;
    return out;
  }
  const int dir = (t1 > t0) - (t1 < t0);
  if (dir == 0) {
    out.phi.assign((2 * n) * (2 * n), 0.0);
    for (std::size_t i = 0; i < 2 * n; ++i) {
      out.phi[i * (2 * n) + i] = 1.0;
    }
    return out;
  }

  const std::size_t dim = 2 * n;
  out.phi.assign(dim * dim, 0.0);
  for (std::size_t i = 0; i < dim; ++i) {
    out.phi[i * dim + i] = 1.0;
  }

  const double h_fixed = static_cast<double>(dir) * opt.h;
  std::vector<double> a0(n, 0.0), a1(n, 0.0), r_next(n, 0.0), v_next(n, 0.0);
  std::vector<double> a_mat0(n * n, 0.0), a_mat1(n * n, 0.0);
  std::vector<double> dr0(n, 0.0), dv0(n, 0.0), dvh(n, 0.0), dr1(n, 0.0), dv1(n, 0.0);
  std::vector<double> tmp(n, 0.0);
  auto&& accel = acceleration;
  auto&& jac = jacobian_r;

  for (int step = 0; step < opt.max_steps; ++step) {
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
    out.stats.last_h = h;
    if (!verlet_single_step(accel, out.t, h, out.r, out.v, a0, r_next, v_next, a1, out.stats)) {
      out.status = IntegratorStatus::NaNDetected;
      return out;
    }

    if (!jac(out.t, out.r, a_mat0) || !jac(out.t + h, r_next, a_mat1) ||
        a_mat0.size() != n * n || a_mat1.size() != n * n || !finite_vec(a_mat0) || !finite_vec(a_mat1)) {
      out.status = IntegratorStatus::NaNDetected;
      return out;
    }

    std::vector<double> phi_next(dim * dim, 0.0);
    for (std::size_t col = 0; col < dim; ++col) {
      for (std::size_t i = 0; i < n; ++i) {
        dr0[i] = out.phi[i * dim + col];
        dv0[i] = out.phi[(n + i) * dim + col];
      }

      matvec(a_mat0, n, dr0, tmp);
      for (std::size_t i = 0; i < n; ++i) {
        dvh[i] = dv0[i] + 0.5 * h * tmp[i];
        dr1[i] = dr0[i] + h * dvh[i];
      }

      matvec(a_mat1, n, dr1, tmp);
      for (std::size_t i = 0; i < n; ++i) {
        dv1[i] = dvh[i] + 0.5 * h * tmp[i];
        phi_next[i * dim + col] = dr1[i];
        phi_next[(n + i) * dim + col] = dv1[i];
      }
    }

    if (!finite_vec(phi_next)) {
      out.status = IntegratorStatus::NaNDetected;
      return out;
    }

    out.r = r_next;
    out.v = v_next;
    out.phi = phi_next;
    out.t += h;
    out.stats.accepted_steps += 1;

    if (obs && !obs(out.t, out.r)) {
      out.status = IntegratorStatus::UserStopped;
      return out;
    }
  }

  out.status = IntegratorStatus::MaxStepsExceeded;
  return out;
}

}  // namespace ode::symplectic
