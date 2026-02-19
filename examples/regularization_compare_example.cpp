#include <cmath>
#include <vector>

#include "ode/ode.hpp"
#include "ode/logging.hpp"

int main() {
  constexpr double mu = 398600.4418;  // km^3/s^2
  const double rp = 7000.0;
  const double ra = 42000.0;
  const double a = 0.5 * (rp + ra);
  const double vp = std::sqrt(mu * (2.0 / rp - 1.0 / a));
  const double tf = 6.0 * 3600.0;

  const ode::regularization::TwoBody2DState s0{rp, 0.0, 0.0, vp};
  std::vector<double> y0{s0.x, s0.y, s0.vx, s0.vy};

  auto rhs = [](double, const std::vector<double>& y, std::vector<double>& dydt) {
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

  ode::regularization::RegularizationOptions ropt;
  ropt.ds = 1e-4;
  ropt.max_steps = 4000000;
  ropt.min_radius_km = 1e-9;

  const auto cowell = ode::integrate(ode::RKMethod::RKF78, rhs, 0.0, y0, tf, opt);
  const auto lc = ode::regularization::integrate_two_body_levi_civita(mu, 0.0, s0, tf, ropt);
  const auto sund = ode::regularization::integrate_two_body_sundman(mu, ode::RKMethod::RKF78, 0.0, s0, tf, opt, 1e-9);

  ode::log::Info("Cowell: x y vx vy = ", cowell.y[0], " ", cowell.y[1], " ", cowell.y[2], " ", cowell.y[3]);
  ode::log::Info("Levi-Civita: x y vx vy = ", lc.state.x, " ", lc.state.y, " ", lc.state.vx, " ", lc.state.vy);
  ode::log::Info("Sundman: x y vx vy = ", sund.state.x, " ", sund.state.y, " ", sund.state.vx, " ", sund.state.vy);
  ode::log::Info("Delta LC-Cowell: ",
                 (lc.state.x - cowell.y[0]), " ",
                 (lc.state.y - cowell.y[1]), " ",
                 (lc.state.vx - cowell.y[2]), " ",
                 (lc.state.vy - cowell.y[3]));
  ode::log::Info("Delta Sundman-Cowell: ",
                 (sund.state.x - cowell.y[0]), " ",
                 (sund.state.y - cowell.y[1]), " ",
                 (sund.state.vx - cowell.y[2]), " ",
                 (sund.state.vy - cowell.y[3]));

  const ode::regularization::TwoBody3DState s03{9000.0, 500.0, 1000.0, -1.0, 6.6, 0.8};
  const double tf3 = 2.0 * 3600.0;
  std::vector<double> y03{s03.x, s03.y, s03.z, s03.vx, s03.vy, s03.vz};
  auto rhs3 = [](double, const std::vector<double>& y, std::vector<double>& dydt) {
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
  const auto cowell3 = ode::integrate(ode::RKMethod::RKF78, rhs3, 0.0, y03, tf3, opt);
  const auto ks = ode::regularization::integrate_two_body_ks(mu, 0.0, s03, tf3, ropt);
  ode::log::Info("Cowell3D: x y z vx vy vz = ",
                 cowell3.y[0], " ", cowell3.y[1], " ", cowell3.y[2], " ",
                 cowell3.y[3], " ", cowell3.y[4], " ", cowell3.y[5]);
  ode::log::Info("KS3D: x y z vx vy vz = ",
                 ks.state.x, " ", ks.state.y, " ", ks.state.z, " ",
                 ks.state.vx, " ", ks.state.vy, " ", ks.state.vz);
  ode::log::Info("Delta KS3D-Cowell3D: ",
                 (ks.state.x - cowell3.y[0]), " ",
                 (ks.state.y - cowell3.y[1]), " ",
                 (ks.state.z - cowell3.y[2]), " ",
                 (ks.state.vx - cowell3.y[3]), " ",
                 (ks.state.vy - cowell3.y[4]), " ",
                 (ks.state.vz - cowell3.y[5]));
  return 0;
}
