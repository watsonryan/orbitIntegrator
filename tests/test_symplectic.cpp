#include <cmath>
#include <vector>

#include "ode/logging.hpp"
#include "ode/ode.hpp"

namespace {

double HarmonicEnergy(double r, double v) {
  return 0.5 * (r * r + v * v);
}

}  // namespace

int main() {
  using Vec = std::vector<double>;

  auto accel = [](double, const Vec& r, Vec& a) {
    a.resize(1);
    a[0] = -r[0];
  };

  auto rhs = [](double, const Vec& y, Vec& dydt) {
    dydt.resize(2);
    dydt[0] = y[1];
    dydt[1] = -y[0];
  };

  const double t0 = 0.0;
  const double tf = 5000.0;
  const double h = 0.2;
  const Vec r0{1.0};
  const Vec v0{0.0};
  const Vec y0{1.0, 0.0};
  const double e0 = HarmonicEnergy(1.0, 0.0);

  ode::symplectic::SymplecticOptions sopt;
  sopt.h = h;

  const auto sv = ode::symplectic::integrate_stormer_verlet(accel, t0, r0, v0, tf, sopt);
  if (sv.status != ode::IntegratorStatus::Success) {
    ode::log::Error("Stormer-Verlet failed, status=", ode::ToString(sv.status));
    return 1;
  }

  const auto y4 = ode::symplectic::integrate_yoshida4(accel, t0, r0, v0, tf, sopt);
  if (y4.status != ode::IntegratorStatus::Success) {
    ode::log::Error("Yoshida-4 failed, status=", ode::ToString(y4.status));
    return 1;
  }

  ode::IntegratorOptions rk_opt;
  rk_opt.adaptive = false;
  rk_opt.fixed_h = h;
  const auto rk = ode::integrate(ode::RKMethod::RK4, rhs, t0, y0, tf, rk_opt);
  if (rk.status != ode::IntegratorStatus::Success) {
    ode::log::Error("RK4 reference failed, status=", ode::ToString(rk.status));
    return 1;
  }

  const double drift_sv = std::abs(HarmonicEnergy(sv.r[0], sv.v[0]) - e0) / e0;
  const double drift_y4 = std::abs(HarmonicEnergy(y4.r[0], y4.v[0]) - e0) / e0;
  const double drift_rk = std::abs(HarmonicEnergy(rk.y[0], rk.y[1]) - e0) / e0;

  if (!(drift_sv < drift_rk)) {
    ode::log::Error("Expected Stormer-Verlet energy drift < RK4 drift, got sv=", drift_sv, " rk4=", drift_rk);
    return 1;
  }
  if (!(drift_y4 < drift_sv * 1.2)) {
    ode::log::Error("Expected Yoshida-4 drift <= Stormer-Verlet drift, got y4=", drift_y4, " sv=", drift_sv);
    return 1;
  }

  return 0;
}
