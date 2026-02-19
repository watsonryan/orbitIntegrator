#include <cmath>
#include <vector>

#include "ode/logging.hpp"
#include "ode/ode.hpp"

namespace {

double L2Pos2D(const ode::regularization::TwoBody2DState& a, const ode::regularization::TwoBody2DState& b) {
  const double dx = a.x - b.x;
  const double dy = a.y - b.y;
  return std::sqrt(dx * dx + dy * dy);
}

double L2Vel2D(const ode::regularization::TwoBody2DState& a, const ode::regularization::TwoBody2DState& b) {
  const double dvx = a.vx - b.vx;
  const double dvy = a.vy - b.vy;
  return std::sqrt(dvx * dvx + dvy * dvy);
}

double L2Pos3D(const ode::regularization::TwoBody3DState& a, const ode::regularization::TwoBody3DState& b) {
  const double dx = a.x - b.x;
  const double dy = a.y - b.y;
  const double dz = a.z - b.z;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double L2Vel3D(const ode::regularization::TwoBody3DState& a, const ode::regularization::TwoBody3DState& b) {
  const double dvx = a.vx - b.vx;
  const double dvy = a.vy - b.vy;
  const double dvz = a.vz - b.vz;
  return std::sqrt(dvx * dvx + dvy * dvy + dvz * dvz);
}

}  // namespace

int main() {
  constexpr double mu = 398600.4418;  // km^3/s^2

  {
    // High-eccentricity planar case.
    const double rp = 7000.0;
    const double ra = 42000.0;
    const double a = 0.5 * (rp + ra);
    const double vp = std::sqrt(mu * (2.0 / rp - 1.0 / a));
    const ode::regularization::TwoBody2DState s0{rp, 0.0, 0.0, vp};
    const double tf = 6.0 * 3600.0;

    std::vector<double> y0{s0.x, s0.y, s0.vx, s0.vy};
    auto rhs2d = [](double, const std::vector<double>& y, std::vector<double>& dydt) {
      dydt.assign(4, 0.0);
      const double r2 = y[0] * y[0] + y[1] * y[1];
      const double r = std::sqrt(r2);
      const double inv_r3 = 1.0 / (r2 * r);
      constexpr double mu2 = 398600.4418;
      dydt[0] = y[2];
      dydt[1] = y[3];
      dydt[2] = -mu2 * y[0] * inv_r3;
      dydt[3] = -mu2 * y[1] * inv_r3;
    };

    ode::IntegratorOptions opt;
    opt.adaptive = true;
    opt.rtol = 1e-12;
    opt.atol = 1e-15;
    opt.h_init = 5.0;
    opt.h_min = 1e-8;
    opt.h_max = 60.0;

    const auto cowell = ode::integrate(ode::RKMethod::RKF78, rhs2d, 0.0, y0, tf, opt);
    if (cowell.status != ode::IntegratorStatus::Success) {
      ode::log::Error("cowell 2d reference failed");
      return 1;
    }
    const ode::regularization::TwoBody2DState ref2d{cowell.y[0], cowell.y[1], cowell.y[2], cowell.y[3]};

    ode::regularization::RegularizationOptions ropt;
    ropt.ds = 1e-4;
    ropt.max_steps = 4000000;
    ropt.min_radius_km = 1e-9;
    const auto lc = ode::regularization::integrate_two_body_levi_civita(mu, 0.0, s0, tf, ropt);
    if (lc.status != ode::IntegratorStatus::Success) {
      ode::log::Error("levi-civita propagation failed");
      return 1;
    }

    const auto sund = ode::regularization::integrate_two_body_sundman(mu, ode::RKMethod::RKF78, 0.0, s0, tf, opt, 1e-9);
    if (sund.status != ode::IntegratorStatus::Success) {
      ode::log::Error("sundman propagation failed");
      return 1;
    }

    const double lc_pos = L2Pos2D(lc.state, ref2d);
    const double lc_vel = L2Vel2D(lc.state, ref2d);
    const double su_pos = L2Pos2D(sund.state, ref2d);
    const double su_vel = L2Vel2D(sund.state, ref2d);

    if (lc_pos > 1e-6 || lc_vel > 1e-10 || su_pos > 1e-6 || su_vel > 1e-9) {
      ode::log::Error("2d regularization parity mismatch");
      return 1;
    }

    // Full-Cowell (perturbed) parity via Sundman transform path.
    auto accel2d = [](double, const ode::regularization::TwoBody2DState& s, std::array<double, 2>& a_out) {
      const double r2 = s.x * s.x + s.y * s.y;
      const double r = std::sqrt(r2);
      const double inv_r3 = 1.0 / (r2 * r);
      constexpr double mu2 = 398600.4418;
      constexpr double alpha_drag = 1e-9;
      a_out[0] = -mu2 * s.x * inv_r3 - alpha_drag * s.vx;
      a_out[1] = -mu2 * s.y * inv_r3 - alpha_drag * s.vy;
    };
    auto rhs2d_pert = [](double, const std::vector<double>& y, std::vector<double>& dydt) {
      dydt.assign(4, 0.0);
      const double r2 = y[0] * y[0] + y[1] * y[1];
      const double r = std::sqrt(r2);
      const double inv_r3 = 1.0 / (r2 * r);
      constexpr double mu2 = 398600.4418;
      constexpr double alpha_drag = 1e-9;
      dydt[0] = y[2];
      dydt[1] = y[3];
      dydt[2] = -mu2 * y[0] * inv_r3 - alpha_drag * y[2];
      dydt[3] = -mu2 * y[1] * inv_r3 - alpha_drag * y[3];
    };
    const auto cowell_pert = ode::integrate(ode::RKMethod::RKF78, rhs2d_pert, 0.0, y0, tf, opt);
    if (cowell_pert.status != ode::IntegratorStatus::Success) {
      ode::log::Error("cowell 2d perturbed reference failed");
      return 1;
    }
    const auto sund_pert = ode::regularization::integrate_cowell_sundman_2d(
        accel2d, ode::RKMethod::RKF78, 0.0, s0, tf, opt, 1e-9);
    if (sund_pert.status != ode::IntegratorStatus::Success) {
      ode::log::Error("cowell-sundman 2d perturbed run failed");
      return 1;
    }
    const ode::regularization::TwoBody2DState refp{
        cowell_pert.y[0], cowell_pert.y[1], cowell_pert.y[2], cowell_pert.y[3]};
    if (L2Pos2D(sund_pert.state, refp) > 2e-6 || L2Vel2D(sund_pert.state, refp) > 2e-9) {
      ode::log::Error("cowell-sundman 2d perturbed parity mismatch");
      return 1;
    }
  }

  {
    // 3D case for KS.
    const ode::regularization::TwoBody3DState s0{9000.0, 500.0, 1000.0, -1.0, 6.6, 0.8};
    const double tf = 2.0 * 3600.0;

    std::vector<double> y0{s0.x, s0.y, s0.z, s0.vx, s0.vy, s0.vz};
    auto rhs3d = [](double, const std::vector<double>& y, std::vector<double>& dydt) {
      dydt.assign(6, 0.0);
      const double r2 = y[0] * y[0] + y[1] * y[1] + y[2] * y[2];
      const double r = std::sqrt(r2);
      const double inv_r3 = 1.0 / (r2 * r);
      constexpr double mu3 = 398600.4418;
      dydt[0] = y[3];
      dydt[1] = y[4];
      dydt[2] = y[5];
      dydt[3] = -mu3 * y[0] * inv_r3;
      dydt[4] = -mu3 * y[1] * inv_r3;
      dydt[5] = -mu3 * y[2] * inv_r3;
    };

    ode::IntegratorOptions opt;
    opt.adaptive = true;
    opt.rtol = 1e-12;
    opt.atol = 1e-15;
    opt.h_init = 5.0;
    opt.h_min = 1e-8;
    opt.h_max = 60.0;

    const auto cowell = ode::integrate(ode::RKMethod::RKF78, rhs3d, 0.0, y0, tf, opt);
    if (cowell.status != ode::IntegratorStatus::Success) {
      ode::log::Error("cowell 3d reference failed");
      return 1;
    }
    const ode::regularization::TwoBody3DState ref3d{cowell.y[0], cowell.y[1], cowell.y[2], cowell.y[3], cowell.y[4], cowell.y[5]};

    ode::regularization::RegularizationOptions ropt;
    ropt.ds = 5e-4;
    ropt.max_steps = 4000000;
    ropt.min_radius_km = 1e-9;
    const auto ks = ode::regularization::integrate_two_body_ks(mu, 0.0, s0, tf, ropt);
    if (ks.status != ode::IntegratorStatus::Success) {
      ode::log::Error("ks propagation failed");
      return 1;
    }

    const double ks_pos = L2Pos3D(ks.state, ref3d);
    const double ks_vel = L2Vel3D(ks.state, ref3d);
    if (ks_pos > 2e-2 || ks_vel > 2e-5) {
      ode::log::Error("ks parity mismatch");
      return 1;
    }

    // Full-Cowell (perturbed) parity via 3D Sundman transform path.
    auto accel3d = [](double, const ode::regularization::TwoBody3DState& s, std::array<double, 3>& a_out) {
      const double r2 = s.x * s.x + s.y * s.y + s.z * s.z;
      const double r = std::sqrt(r2);
      const double inv_r3 = 1.0 / (r2 * r);
      constexpr double mu3 = 398600.4418;
      constexpr double alpha_drag = 1e-9;
      a_out[0] = -mu3 * s.x * inv_r3 - alpha_drag * s.vx;
      a_out[1] = -mu3 * s.y * inv_r3 - alpha_drag * s.vy;
      a_out[2] = -mu3 * s.z * inv_r3 - alpha_drag * s.vz;
    };
    auto rhs3d_pert = [](double, const std::vector<double>& y, std::vector<double>& dydt) {
      dydt.assign(6, 0.0);
      const double r2 = y[0] * y[0] + y[1] * y[1] + y[2] * y[2];
      const double r = std::sqrt(r2);
      const double inv_r3 = 1.0 / (r2 * r);
      constexpr double mu3 = 398600.4418;
      constexpr double alpha_drag = 1e-9;
      dydt[0] = y[3];
      dydt[1] = y[4];
      dydt[2] = y[5];
      dydt[3] = -mu3 * y[0] * inv_r3 - alpha_drag * y[3];
      dydt[4] = -mu3 * y[1] * inv_r3 - alpha_drag * y[4];
      dydt[5] = -mu3 * y[2] * inv_r3 - alpha_drag * y[5];
    };
    const auto cowell_pert = ode::integrate(ode::RKMethod::RKF78, rhs3d_pert, 0.0, y0, tf, opt);
    if (cowell_pert.status != ode::IntegratorStatus::Success) {
      ode::log::Error("cowell 3d perturbed reference failed");
      return 1;
    }
    const auto sund_pert = ode::regularization::integrate_cowell_sundman_3d(
        accel3d, ode::RKMethod::RKF78, 0.0, s0, tf, opt, 1e-9);
    if (sund_pert.status != ode::IntegratorStatus::Success) {
      ode::log::Error("cowell-sundman 3d perturbed run failed");
      return 1;
    }
    const ode::regularization::TwoBody3DState refp{
        cowell_pert.y[0], cowell_pert.y[1], cowell_pert.y[2],
        cowell_pert.y[3], cowell_pert.y[4], cowell_pert.y[5]};
    if (L2Pos3D(sund_pert.state, refp) > 2e-6 || L2Vel3D(sund_pert.state, refp) > 2e-9) {
      ode::log::Error("cowell-sundman 3d perturbed parity mismatch");
      return 1;
    }
  }

  return 0;
}
