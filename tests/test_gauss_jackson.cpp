#include <cmath>
#include <vector>

#include "ode/logging.hpp"
#include "ode/multistep/gauss_jackson8.hpp"
#include "ode/ode.hpp"

int main() {
  using Vec = std::vector<double>;

  auto accel = [](double, const Vec& r, const Vec& /*v*/, Vec& a) {
    a.resize(1);
    a[0] = -r[0];
  };

  const double t0 = 0.0;
  const double t1 = 2.0 * 3.14159265358979323846;
  const Vec r0{1.0};
  const Vec v0{0.0};

  ode::multistep::GaussJackson8Options gj_opt;
  gj_opt.h = 0.01;
  gj_opt.corrector_iterations = 2;
  const auto gj = ode::multistep::integrate_gauss_jackson8(accel, t0, r0, v0, t1, gj_opt);
  if (gj.status != ode::IntegratorStatus::Success) {
    ode::log::Error("Gauss-Jackson run failed, status=", ode::ToString(gj.status));
    return 1;
  }
  if (std::abs(gj.r[0] - 1.0) > 3e-5 || std::abs(gj.v[0]) > 3e-5) {
    ode::log::Error("Gauss-Jackson oscillator mismatch r=", gj.r[0], " v=", gj.v[0]);
    return 1;
  }

  // Cross-check against first-order RKF78 using y=[r,v].
  auto rhs = [](double, const Vec& y, Vec& dydt) {
    dydt.resize(2);
    dydt[0] = y[1];
    dydt[1] = -y[0];
  };
  ode::IntegratorOptions rk_opt;
  rk_opt.adaptive = true;
  rk_opt.rtol = 1e-12;
  rk_opt.atol = 1e-14;
  rk_opt.h_init = 0.01;
  const Vec y0{1.0, 0.0};
  const auto rk = ode::integrate(ode::RKMethod::RKF78, rhs, t0, y0, t1, rk_opt);
  if (rk.status != ode::IntegratorStatus::Success) {
    ode::log::Error("RKF78 reference failed");
    return 1;
  }
  if (std::abs(gj.r[0] - rk.y[0]) > 5e-5 || std::abs(gj.v[0] - rk.y[1]) > 5e-5) {
    ode::log::Error("Gauss-Jackson vs RKF78 mismatch r=", gj.r[0], " rk_r=", rk.y[0], " v=", gj.v[0], " rk_v=", rk.y[1]);
    return 1;
  }

  return 0;
}

