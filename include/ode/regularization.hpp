/**
 * @file regularization.hpp
 * @brief Close-approach regularization helpers for planar two-body propagation.
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "ode/sundman.hpp"
#include "ode/types.hpp"

namespace ode::regularization {

struct TwoBody2DState {
  double x = 0.0;
  double y = 0.0;
  double vx = 0.0;
  double vy = 0.0;
};

struct TwoBody3DState {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double vx = 0.0;
  double vy = 0.0;
  double vz = 0.0;
};

struct RegularizationOptions {
  double ds = 1e-3;
  int max_steps = 1000000;
  double min_radius_km = 1e-12;
};

struct RegularizationResult {
  IntegratorStatus status = IntegratorStatus::Success;
  double t = 0.0;
  TwoBody2DState state{};
  IntegratorStats stats{};
};

struct RegularizationResult3D {
  IntegratorStatus status = IntegratorStatus::Success;
  double t = 0.0;
  TwoBody3DState state{};
  IntegratorStats stats{};
};

[[nodiscard]] inline double radius(const TwoBody2DState& s) {
  return std::sqrt(s.x * s.x + s.y * s.y);
}

[[nodiscard]] inline double specific_energy(double mu, const TwoBody2DState& s) {
  const double r = radius(s);
  const double v2 = s.vx * s.vx + s.vy * s.vy;
  return 0.5 * v2 - mu / r;
}

[[nodiscard]] inline double radius(const TwoBody3DState& s) {
  return std::sqrt(s.x * s.x + s.y * s.y + s.z * s.z);
}

[[nodiscard]] inline double specific_energy(double mu, const TwoBody3DState& s) {
  const double r = radius(s);
  const double v2 = s.vx * s.vx + s.vy * s.vy + s.vz * s.vz;
  return 0.5 * v2 - mu / r;
}

[[nodiscard]] inline bool to_levi_civita(const TwoBody2DState& s,
                                         std::array<double, 2>& u,
                                         std::array<double, 2>& up) {
  const double r = radius(s);
  if (!(r > 0.0) || !std::isfinite(r)) {
    return false;
  }
  const double theta = std::atan2(s.y, s.x);
  const double sr = std::sqrt(r);
  u[0] = sr * std::cos(0.5 * theta);
  u[1] = sr * std::sin(0.5 * theta);
  if (!std::isfinite(u[0]) || !std::isfinite(u[1])) {
    return false;
  }

  // u' where ' is d/ds with Sundman dt/ds=r.
  up[0] = 0.5 * (u[0] * s.vx + u[1] * s.vy);
  up[1] = 0.5 * (-u[1] * s.vx + u[0] * s.vy);
  return std::isfinite(up[0]) && std::isfinite(up[1]);
}

[[nodiscard]] inline bool from_levi_civita(const std::array<double, 2>& u,
                                           const std::array<double, 2>& up,
                                           TwoBody2DState& s_out) {
  const double u1 = u[0];
  const double u2 = u[1];
  const double q = u1 * u1 + u2 * u2;
  if (!(q > 0.0) || !std::isfinite(q)) {
    return false;
  }

  s_out.x = u1 * u1 - u2 * u2;
  s_out.y = 2.0 * u1 * u2;
  s_out.vx = 2.0 * (u1 * up[0] - u2 * up[1]) / q;
  s_out.vy = 2.0 * (u2 * up[0] + u1 * up[1]) / q;
  return std::isfinite(s_out.x) && std::isfinite(s_out.y) &&
         std::isfinite(s_out.vx) && std::isfinite(s_out.vy);
}

[[nodiscard]] inline RegularizationResult integrate_two_body_levi_civita(double mu,
                                                                          double t0,
                                                                          const TwoBody2DState& s0,
                                                                          double t1,
                                                                          RegularizationOptions opt) {
  RegularizationResult out{};
  out.t = t0;
  out.state = s0;

  if (!(opt.ds > 0.0) || !std::isfinite(opt.ds) || opt.max_steps <= 0 || !(mu > 0.0) ||
      !(opt.min_radius_km > 0.0)) {
    out.status = IntegratorStatus::InvalidStepSize;
    return out;
  }

  const int dir = (t1 > t0) - (t1 < t0);
  if (dir == 0) {
    return out;
  }
  const double ds_nominal = static_cast<double>(dir) * opt.ds;

  std::array<double, 2> u{}, up{};
  if (!to_levi_civita(s0, u, up)) {
    out.status = IntegratorStatus::InvalidStepSize;
    return out;
  }
  const double energy = specific_energy(mu, s0);

  // z = [u1, u2, up1, up2, t]
  std::array<double, 5> z{u[0], u[1], up[0], up[1], t0};
  std::array<double, 5> k1{}, k2{}, k3{}, k4{}, zt{};

  auto rhs = [&](const std::array<double, 5>& in, std::array<double, 5>& dzds) {
    const double uu1 = in[0];
    const double uu2 = in[1];
    const double q = std::max(opt.min_radius_km, uu1 * uu1 + uu2 * uu2);
    dzds[0] = in[2];
    dzds[1] = in[3];
    dzds[2] = 0.5 * energy * uu1;
    dzds[3] = 0.5 * energy * uu2;
    dzds[4] = q;
  };

  for (int step = 0; step < opt.max_steps; ++step) {
    const double rem = t1 - z[4];
    if ((dir > 0 && rem <= 0.0) || (dir < 0 && rem >= 0.0)) {
      TwoBody2DState s{};
      if (!from_levi_civita({z[0], z[1]}, {z[2], z[3]}, s)) {
        out.status = IntegratorStatus::NaNDetected;
        return out;
      }
      out.t = z[4];
      out.state = s;
      out.status = IntegratorStatus::Success;
      return out;
    }

    const double q = std::max(opt.min_radius_km, z[0] * z[0] + z[1] * z[1]);
    double ds = ds_nominal;
    const double ds_from_time = rem / q;
    if (std::abs(ds) > std::abs(ds_from_time)) {
      ds = ds_from_time;
    }
    if (!(std::abs(ds) > 0.0) || !std::isfinite(ds)) {
      out.status = IntegratorStatus::StepSizeUnderflow;
      return out;
    }

    out.stats.attempted_steps += 1;
    out.stats.last_h = ds * q;

    rhs(z, k1);
    for (int i = 0; i < 5; ++i) {
      zt[i] = z[i] + 0.5 * ds * k1[i];
    }
    rhs(zt, k2);
    for (int i = 0; i < 5; ++i) {
      zt[i] = z[i] + 0.5 * ds * k2[i];
    }
    rhs(zt, k3);
    for (int i = 0; i < 5; ++i) {
      zt[i] = z[i] + ds * k3[i];
    }
    rhs(zt, k4);

    for (int i = 0; i < 5; ++i) {
      z[i] += (ds / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
      if (!std::isfinite(z[i])) {
        out.status = IntegratorStatus::NaNDetected;
        return out;
      }
    }
    out.stats.accepted_steps += 1;
    out.stats.rhs_evals += 4;
  }

  out.status = IntegratorStatus::MaxStepsExceeded;
  return out;
}

[[nodiscard]] inline RegularizationResult integrate_two_body_sundman(double mu,
                                                                     RKMethod method,
                                                                     double t0,
                                                                     const TwoBody2DState& s0,
                                                                     double t1,
                                                                     IntegratorOptions opt,
                                                                     double min_radius_km = 1e-12) {
  RegularizationResult out{};
  out.t = t0;
  out.state = s0;
  if (!(mu > 0.0) || !(min_radius_km > 0.0)) {
    out.status = IntegratorStatus::InvalidStepSize;
    return out;
  }

  std::vector<double> y0{s0.x, s0.y, s0.vx, s0.vy};
  auto rhs = [mu](double, const std::vector<double>& y, std::vector<double>& dydt) {
    dydt.assign(4, 0.0);
    const double x = y[0];
    const double yy = y[1];
    const double r2 = x * x + yy * yy;
    const double r = std::sqrt(r2);
    const double inv_r3 = 1.0 / (r2 * r);
    dydt[0] = y[2];
    dydt[1] = y[3];
    dydt[2] = -mu * x * inv_r3;
    dydt[3] = -mu * yy * inv_r3;
  };

  auto dt_ds = [min_radius_km](double, const std::vector<double>& y) {
    const double r = std::sqrt(y[0] * y[0] + y[1] * y[1]);
    return std::max(min_radius_km, r);
  };

  const auto res = ode::integrate_sundman(method, rhs, dt_ds, t0, y0, t1, opt);
  out.status = res.status;
  out.t = res.t;
  out.stats = res.stats;
  if (res.y.size() == 4) {
    out.state = {res.y[0], res.y[1], res.y[2], res.y[3]};
  }
  return out;
}

template <class Accel2DFn>
[[nodiscard]] inline RegularizationResult integrate_cowell_sundman_2d(
    Accel2DFn&& acceleration,
    RKMethod method,
    double t0,
    const TwoBody2DState& s0,
    double t1,
    IntegratorOptions opt,
    double min_radius_km = 1e-12) {
  RegularizationResult out{};
  out.t = t0;
  out.state = s0;
  if (!(min_radius_km > 0.0)) {
    out.status = IntegratorStatus::InvalidStepSize;
    return out;
  }

  std::vector<double> y0{s0.x, s0.y, s0.vx, s0.vy};
  auto rhs = [&](double t, const std::vector<double>& y, std::vector<double>& dydt) {
    dydt.assign(4, 0.0);
    std::array<double, 2> a{};
    const TwoBody2DState s{y[0], y[1], y[2], y[3]};
    acceleration(t, s, a);
    dydt[0] = y[2];
    dydt[1] = y[3];
    dydt[2] = a[0];
    dydt[3] = a[1];
  };

  auto dt_ds = [min_radius_km](double, const std::vector<double>& y) {
    const double r = std::sqrt(y[0] * y[0] + y[1] * y[1]);
    return std::max(min_radius_km, r);
  };

  const auto res = ode::integrate_sundman(method, rhs, dt_ds, t0, y0, t1, opt);
  out.status = res.status;
  out.t = res.t;
  out.stats = res.stats;
  if (res.y.size() == 4) {
    out.state = {res.y[0], res.y[1], res.y[2], res.y[3]};
  }
  return out;
}

[[nodiscard]] inline bool to_ks(const TwoBody3DState& s,
                                std::array<double, 4>& u,
                                std::array<double, 4>& up) {
  const double r = radius(s);
  if (!(r > 0.0) || !std::isfinite(r)) {
    return false;
  }

  // Gauge choice u4 = 0.
  const double denom = 2.0 * std::max(1e-18, (r + s.x));
  const double u1 = std::sqrt(std::max(0.0, 0.5 * (r + s.x)));
  if (!(u1 > 0.0) || !std::isfinite(u1)) {
    return false;
  }
  const double u2 = s.y / (2.0 * u1);
  const double u3 = s.z / (2.0 * u1);
  const double u4 = 0.0;
  (void)denom;

  u = {u1, u2, u3, u4};
  if (!std::isfinite(u2) || !std::isfinite(u3)) {
    return false;
  }

  // u' = 0.5 * L(u)^T * v for dt/ds = r.
  const double vx = s.vx;
  const double vy = s.vy;
  const double vz = s.vz;
  up[0] = 0.5 * (u1 * vx + u2 * vy + u3 * vz);
  up[1] = 0.5 * (-u2 * vx + u1 * vy + u4 * vz);
  up[2] = 0.5 * (-u3 * vx - u4 * vy + u1 * vz);
  up[3] = 0.5 * (u4 * vx - u3 * vy + u2 * vz);
  return std::isfinite(up[0]) && std::isfinite(up[1]) && std::isfinite(up[2]) && std::isfinite(up[3]);
}

[[nodiscard]] inline bool from_ks(const std::array<double, 4>& u,
                                  const std::array<double, 4>& up,
                                  TwoBody3DState& s_out) {
  const double u1 = u[0];
  const double u2 = u[1];
  const double u3 = u[2];
  const double u4 = u[3];

  const double x = u1 * u1 - u2 * u2 - u3 * u3 + u4 * u4;
  const double y = 2.0 * (u1 * u2 - u3 * u4);
  const double z = 2.0 * (u1 * u3 + u2 * u4);
  const double r = std::max(1e-18, u1 * u1 + u2 * u2 + u3 * u3 + u4 * u4);

  const double rp_x = 2.0 * (u1 * up[0] - u2 * up[1] - u3 * up[2] + u4 * up[3]);
  const double rp_y = 2.0 * (u2 * up[0] + u1 * up[1] - u4 * up[2] - u3 * up[3]);
  const double rp_z = 2.0 * (u3 * up[0] + u4 * up[1] + u1 * up[2] + u2 * up[3]);

  s_out.x = x;
  s_out.y = y;
  s_out.z = z;
  s_out.vx = rp_x / r;
  s_out.vy = rp_y / r;
  s_out.vz = rp_z / r;
  return std::isfinite(s_out.x) && std::isfinite(s_out.y) && std::isfinite(s_out.z) &&
         std::isfinite(s_out.vx) && std::isfinite(s_out.vy) && std::isfinite(s_out.vz);
}

[[nodiscard]] inline RegularizationResult3D integrate_two_body_ks(double mu,
                                                                  double t0,
                                                                  const TwoBody3DState& s0,
                                                                  double t1,
                                                                  RegularizationOptions opt) {
  RegularizationResult3D out{};
  out.t = t0;
  out.state = s0;
  if (!(opt.ds > 0.0) || !std::isfinite(opt.ds) || opt.max_steps <= 0 || !(mu > 0.0) ||
      !(opt.min_radius_km > 0.0)) {
    out.status = IntegratorStatus::InvalidStepSize;
    return out;
  }

  const int dir = (t1 > t0) - (t1 < t0);
  if (dir == 0) {
    return out;
  }
  const double ds_nominal = static_cast<double>(dir) * opt.ds;

  std::array<double, 4> u{}, up{};
  if (!to_ks(s0, u, up)) {
    out.status = IntegratorStatus::InvalidStepSize;
    return out;
  }
  const double energy = specific_energy(mu, s0);

  // z = [u1,u2,u3,u4,up1,up2,up3,up4,t]
  std::array<double, 9> z{u[0], u[1], u[2], u[3], up[0], up[1], up[2], up[3], t0};
  std::array<double, 9> k1{}, k2{}, k3{}, k4{}, zt{};

  auto rhs = [&](const std::array<double, 9>& in, std::array<double, 9>& dzds) {
    const double q = std::max(opt.min_radius_km, in[0] * in[0] + in[1] * in[1] + in[2] * in[2] + in[3] * in[3]);
    dzds[0] = in[4];
    dzds[1] = in[5];
    dzds[2] = in[6];
    dzds[3] = in[7];
    dzds[4] = 0.5 * energy * in[0];
    dzds[5] = 0.5 * energy * in[1];
    dzds[6] = 0.5 * energy * in[2];
    dzds[7] = 0.5 * energy * in[3];
    dzds[8] = q;
  };

  for (int step = 0; step < opt.max_steps; ++step) {
    const double rem = t1 - z[8];
    if ((dir > 0 && rem <= 0.0) || (dir < 0 && rem >= 0.0)) {
      TwoBody3DState s{};
      if (!from_ks({z[0], z[1], z[2], z[3]}, {z[4], z[5], z[6], z[7]}, s)) {
        out.status = IntegratorStatus::NaNDetected;
        return out;
      }
      out.t = z[8];
      out.state = s;
      out.status = IntegratorStatus::Success;
      return out;
    }

    const double q = std::max(opt.min_radius_km, z[0] * z[0] + z[1] * z[1] + z[2] * z[2] + z[3] * z[3]);
    double ds = ds_nominal;
    const double ds_from_time = rem / q;
    if (std::abs(ds) > std::abs(ds_from_time)) {
      ds = ds_from_time;
    }
    if (!(std::abs(ds) > 0.0) || !std::isfinite(ds)) {
      out.status = IntegratorStatus::StepSizeUnderflow;
      return out;
    }

    out.stats.attempted_steps += 1;
    out.stats.last_h = ds * q;

    rhs(z, k1);
    for (int i = 0; i < 9; ++i) {
      zt[i] = z[i] + 0.5 * ds * k1[i];
    }
    rhs(zt, k2);
    for (int i = 0; i < 9; ++i) {
      zt[i] = z[i] + 0.5 * ds * k2[i];
    }
    rhs(zt, k3);
    for (int i = 0; i < 9; ++i) {
      zt[i] = z[i] + ds * k3[i];
    }
    rhs(zt, k4);

    for (int i = 0; i < 9; ++i) {
      z[i] += (ds / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
      if (!std::isfinite(z[i])) {
        out.status = IntegratorStatus::NaNDetected;
        return out;
      }
    }
    out.stats.accepted_steps += 1;
    out.stats.rhs_evals += 4;
  }

  out.status = IntegratorStatus::MaxStepsExceeded;
  return out;
}

template <class Accel3DFn>
[[nodiscard]] inline RegularizationResult3D integrate_cowell_sundman_3d(
    Accel3DFn&& acceleration,
    RKMethod method,
    double t0,
    const TwoBody3DState& s0,
    double t1,
    IntegratorOptions opt,
    double min_radius_km = 1e-12) {
  RegularizationResult3D out{};
  out.t = t0;
  out.state = s0;
  if (!(min_radius_km > 0.0)) {
    out.status = IntegratorStatus::InvalidStepSize;
    return out;
  }

  std::vector<double> y0{s0.x, s0.y, s0.z, s0.vx, s0.vy, s0.vz};
  auto rhs = [&](double t, const std::vector<double>& y, std::vector<double>& dydt) {
    dydt.assign(6, 0.0);
    std::array<double, 3> a{};
    const TwoBody3DState s{y[0], y[1], y[2], y[3], y[4], y[5]};
    acceleration(t, s, a);
    dydt[0] = y[3];
    dydt[1] = y[4];
    dydt[2] = y[5];
    dydt[3] = a[0];
    dydt[4] = a[1];
    dydt[5] = a[2];
  };

  auto dt_ds = [min_radius_km](double, const std::vector<double>& y) {
    const double r = std::sqrt(y[0] * y[0] + y[1] * y[1] + y[2] * y[2]);
    return std::max(min_radius_km, r);
  };

  const auto res = ode::integrate_sundman(method, rhs, dt_ds, t0, y0, t1, opt);
  out.status = res.status;
  out.t = res.t;
  out.stats = res.stats;
  if (res.y.size() == 6) {
    out.state = {res.y[0], res.y[1], res.y[2], res.y[3], res.y[4], res.y[5]};
  }
  return out;
}

}  // namespace ode::regularization
