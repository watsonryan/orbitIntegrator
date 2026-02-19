/**
 * @file gauss_jackson8.hpp
 * @brief Fixed-step Gauss-Jackson-style 8th-order predictor-corrector for second-order systems.
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "ode/types.hpp"

namespace ode::multistep {

struct GaussJackson8Options {
  double h = 0.0;
  int max_steps = 1000000;
  int corrector_iterations = 2;
};

struct SecondOrderResult {
  IntegratorStatus status = IntegratorStatus::Success;
  double t = 0.0;
  std::vector<double> r{};
  std::vector<double> v{};
  IntegratorStats stats{};
};

template <class AccelerationFn>
[[nodiscard]] SecondOrderResult integrate_gauss_jackson8(AccelerationFn&& acceleration,
                                                         double t0,
                                                         const std::vector<double>& r0,
                                                         const std::vector<double>& v0,
                                                         double t1,
                                                         GaussJackson8Options opt,
                                                         Observer<std::vector<double>> obs = {}) {
  SecondOrderResult out{};
  out.t = t0;
  out.r = r0;
  out.v = v0;

  const std::size_t n = r0.size();
  if (n == 0 || v0.size() != n || !(opt.h > 0.0) || !std::isfinite(opt.h) || opt.max_steps <= 0 ||
      opt.corrector_iterations <= 0) {
    out.status = IntegratorStatus::InvalidStepSize;
    return out;
  }

  const int dir = (t1 > t0) - (t1 < t0);
  if (dir == 0) {
    return out;
  }
  const double h_fixed = static_cast<double>(dir) * opt.h;

  // AB8 predictor coefficients (newest to oldest): n, n-1, ..., n-7.
  constexpr std::array<double, 8> kAb8{
      434241.0 / 120960.0,  -1152169.0 / 120960.0, 2183877.0 / 120960.0,  -2664477.0 / 120960.0,
      2102243.0 / 120960.0, -1041723.0 / 120960.0, 295767.0 / 120960.0,   -36799.0 / 120960.0};
  // AM8 corrector coefficients: n+1, n, ..., n-6.
  constexpr std::array<double, 8> kAm8{
      36799.0 / 120960.0,  139849.0 / 120960.0,  -121797.0 / 120960.0, 123133.0 / 120960.0,
      -88547.0 / 120960.0, 41499.0 / 120960.0,   -11351.0 / 120960.0,  1375.0 / 120960.0};

  std::array<std::vector<double>, 8> r_hist{};  // r_{n-7}..r_n
  std::array<std::vector<double>, 8> v_hist{};  // v_{n-7}..v_n
  std::array<std::vector<double>, 8> a_hist{};  // a_{n-7}..a_n
  for (int i = 0; i < 8; ++i) {
    r_hist[static_cast<std::size_t>(i)].assign(n, 0.0);
    v_hist[static_cast<std::size_t>(i)].assign(n, 0.0);
    a_hist[static_cast<std::size_t>(i)].assign(n, 0.0);
  }
  r_hist[7] = r0;
  v_hist[7] = v0;
  acceleration(out.t, out.r, out.v, a_hist[7]);
  out.stats.rhs_evals += 1;

  auto finite_vec = [](const std::vector<double>& x) {
    for (double v : x) {
      if (!std::isfinite(v)) {
        return false;
      }
    }
    return true;
  };
  if (!finite_vec(a_hist[7])) {
    out.status = IntegratorStatus::NaNDetected;
    return out;
  }

  std::vector<double> r_next(n, 0.0);
  std::vector<double> v_next(n, 0.0);
  std::vector<double> a_next(n, 0.0);

  // Startup with RK4 on the first-order form, building 8 history points.
  for (int s = 0; s < 7; ++s) {
    const double rem = t1 - out.t;
    if ((dir > 0 && rem <= 0.0) || (dir < 0 && rem >= 0.0)) {
      out.status = IntegratorStatus::Success;
      return out;
    }
    double h = h_fixed;
    if (std::abs(h) > std::abs(rem)) {
      h = rem;
    }

    std::vector<double> k1r = out.v;
    std::vector<double> k1v = a_hist[7];
    std::vector<double> k2r(n, 0.0), k2v(n, 0.0);
    std::vector<double> k3r(n, 0.0), k3v(n, 0.0);
    std::vector<double> k4r(n, 0.0), k4v(n, 0.0);
    std::vector<double> rt(n, 0.0), vt(n, 0.0), at(n, 0.0);

    for (std::size_t i = 0; i < n; ++i) {
      rt[i] = out.r[i] + 0.5 * h * k1r[i];
      vt[i] = out.v[i] + 0.5 * h * k1v[i];
    }
    acceleration(out.t + 0.5 * h, rt, vt, at);
    out.stats.rhs_evals += 1;
    if (!finite_vec(at)) {
      out.status = IntegratorStatus::NaNDetected;
      return out;
    }
    k2r = vt;
    k2v = at;

    for (std::size_t i = 0; i < n; ++i) {
      rt[i] = out.r[i] + 0.5 * h * k2r[i];
      vt[i] = out.v[i] + 0.5 * h * k2v[i];
    }
    acceleration(out.t + 0.5 * h, rt, vt, at);
    out.stats.rhs_evals += 1;
    if (!finite_vec(at)) {
      out.status = IntegratorStatus::NaNDetected;
      return out;
    }
    k3r = vt;
    k3v = at;

    for (std::size_t i = 0; i < n; ++i) {
      rt[i] = out.r[i] + h * k3r[i];
      vt[i] = out.v[i] + h * k3v[i];
    }
    acceleration(out.t + h, rt, vt, at);
    out.stats.rhs_evals += 1;
    if (!finite_vec(at)) {
      out.status = IntegratorStatus::NaNDetected;
      return out;
    }
    k4r = vt;
    k4v = at;

    for (std::size_t i = 0; i < n; ++i) {
      r_next[i] = out.r[i] + (h / 6.0) * (k1r[i] + 2.0 * k2r[i] + 2.0 * k3r[i] + k4r[i]);
      v_next[i] = out.v[i] + (h / 6.0) * (k1v[i] + 2.0 * k2v[i] + 2.0 * k3v[i] + k4v[i]);
    }
    acceleration(out.t + h, r_next, v_next, a_next);
    out.stats.rhs_evals += 1;
    if (!finite_vec(r_next) || !finite_vec(v_next) || !finite_vec(a_next)) {
      out.status = IntegratorStatus::NaNDetected;
      return out;
    }

    for (int i = 0; i < 7; ++i) {
      r_hist[static_cast<std::size_t>(i)] = r_hist[static_cast<std::size_t>(i + 1)];
      v_hist[static_cast<std::size_t>(i)] = v_hist[static_cast<std::size_t>(i + 1)];
      a_hist[static_cast<std::size_t>(i)] = a_hist[static_cast<std::size_t>(i + 1)];
    }
    r_hist[7] = r_next;
    v_hist[7] = v_next;
    a_hist[7] = a_next;

    out.t += h;
    out.r = r_next;
    out.v = v_next;
    out.stats.attempted_steps += 1;
    out.stats.accepted_steps += 1;
    out.stats.last_h = h;
    if (obs && !obs(out.t, out.r)) {
      out.status = IntegratorStatus::UserStopped;
      return out;
    }
  }

  std::vector<double> r_pred(n, 0.0), v_pred(n, 0.0), a_pred(n, 0.0);
  std::vector<double> r_corr(n, 0.0), v_corr(n, 0.0), a_corr(n, 0.0);
  for (int step = 0; step < opt.max_steps; ++step) {
    const double rem = t1 - out.t;
    if ((dir > 0 && rem <= 0.0) || (dir < 0 && rem >= 0.0)) {
      out.status = IntegratorStatus::Success;
      return out;
    }
    // Fall back to short RK4 finishing step.
    if (std::abs(rem) < std::abs(h_fixed) * (1.0 - 1e-14)) {
      const double h = rem;
      std::vector<double> k1r = out.v;
      std::vector<double> k1v = a_hist[7];
      std::vector<double> k2r(n, 0.0), k2v(n, 0.0);
      std::vector<double> k3r(n, 0.0), k3v(n, 0.0);
      std::vector<double> k4r(n, 0.0), k4v(n, 0.0);
      std::vector<double> rt(n, 0.0), vt(n, 0.0), at(n, 0.0);
      for (std::size_t i = 0; i < n; ++i) {
        rt[i] = out.r[i] + 0.5 * h * k1r[i];
        vt[i] = out.v[i] + 0.5 * h * k1v[i];
      }
      acceleration(out.t + 0.5 * h, rt, vt, at);
      out.stats.rhs_evals += 1;
      k2r = vt;
      k2v = at;
      for (std::size_t i = 0; i < n; ++i) {
        rt[i] = out.r[i] + 0.5 * h * k2r[i];
        vt[i] = out.v[i] + 0.5 * h * k2v[i];
      }
      acceleration(out.t + 0.5 * h, rt, vt, at);
      out.stats.rhs_evals += 1;
      k3r = vt;
      k3v = at;
      for (std::size_t i = 0; i < n; ++i) {
        rt[i] = out.r[i] + h * k3r[i];
        vt[i] = out.v[i] + h * k3v[i];
      }
      acceleration(out.t + h, rt, vt, at);
      out.stats.rhs_evals += 1;
      k4r = vt;
      k4v = at;
      for (std::size_t i = 0; i < n; ++i) {
        out.r[i] += (h / 6.0) * (k1r[i] + 2.0 * k2r[i] + 2.0 * k3r[i] + k4r[i]);
        out.v[i] += (h / 6.0) * (k1v[i] + 2.0 * k2v[i] + 2.0 * k3v[i] + k4v[i]);
      }
      out.t += h;
      out.stats.attempted_steps += 1;
      out.stats.accepted_steps += 1;
      out.stats.last_h = h;
      out.status = IntegratorStatus::Success;
      return out;
    }

    const double h = h_fixed;
    out.stats.attempted_steps += 1;
    out.stats.last_h = h;
    // Predictor (AB8) for both r' = v and v' = a.
    for (std::size_t i = 0; i < n; ++i) {
      double dv = 0.0;
      double dr = 0.0;
      for (int j = 0; j < 8; ++j) {
        dv += kAb8[static_cast<std::size_t>(j)] * a_hist[static_cast<std::size_t>(7 - j)][i];
        dr += kAb8[static_cast<std::size_t>(j)] * v_hist[static_cast<std::size_t>(7 - j)][i];
      }
      v_pred[i] = out.v[i] + h * dv;
      r_pred[i] = out.r[i] + h * dr;
    }
    if (!finite_vec(r_pred) || !finite_vec(v_pred)) {
      out.status = IntegratorStatus::NaNDetected;
      return out;
    }

    const double t_next = out.t + h;
    acceleration(t_next, r_pred, v_pred, a_pred);
    out.stats.rhs_evals += 1;
    if (!finite_vec(a_pred)) {
      out.status = IntegratorStatus::NaNDetected;
      return out;
    }

    r_corr = r_pred;
    v_corr = v_pred;
    a_corr = a_pred;
    const int iter_count = std::max(1, opt.corrector_iterations);
    for (int it = 0; it < iter_count; ++it) {
      for (std::size_t i = 0; i < n; ++i) {
        double dv = kAm8[0] * a_corr[i];
        double dr = kAm8[0] * v_corr[i];
        for (int j = 1; j < 8; ++j) {
          dv += kAm8[static_cast<std::size_t>(j)] * a_hist[static_cast<std::size_t>(8 - j)][i];
          dr += kAm8[static_cast<std::size_t>(j)] * v_hist[static_cast<std::size_t>(8 - j)][i];
        }
        v_corr[i] = out.v[i] + h * dv;
        r_corr[i] = out.r[i] + h * dr;
      }
      acceleration(t_next, r_corr, v_corr, a_corr);
      out.stats.rhs_evals += 1;
      if (!finite_vec(r_corr) || !finite_vec(v_corr) || !finite_vec(a_corr)) {
        out.status = IntegratorStatus::NaNDetected;
        return out;
      }
    }

    for (int i = 0; i < 7; ++i) {
      r_hist[static_cast<std::size_t>(i)] = r_hist[static_cast<std::size_t>(i + 1)];
      v_hist[static_cast<std::size_t>(i)] = v_hist[static_cast<std::size_t>(i + 1)];
      a_hist[static_cast<std::size_t>(i)] = a_hist[static_cast<std::size_t>(i + 1)];
    }
    r_hist[7] = r_corr;
    v_hist[7] = v_corr;
    a_hist[7] = a_corr;

    out.t = t_next;
    out.r = r_corr;
    out.v = v_corr;
    out.stats.accepted_steps += 1;
    if (obs && !obs(out.t, out.r)) {
      out.status = IntegratorStatus::UserStopped;
      return out;
    }
  }

  out.status = IntegratorStatus::MaxStepsExceeded;
  return out;
}

}  // namespace ode::multistep

