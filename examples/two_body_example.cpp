#include <array>
#include <cmath>
#include <numbers>

#include "ode/logging.hpp"
#include "ode/ode.hpp"

int main() {
  using State = std::array<double, 6>;  // x y z vx vy vz

  constexpr double kMu = 398600.4418;  // km^3/s^2 (Earth)
  constexpr double kOrbitalRadiusKm = 7000.0;
  const double kOrbitalSpeedKms = std::sqrt(kMu / kOrbitalRadiusKm);

  const State y0 = {kOrbitalRadiusKm, 0.0, 0.0, 0.0, kOrbitalSpeedKms, 0.0};

  auto rhs = [](double, const State& y, State& dydt) {
    const double x = y[0];
    const double yv = y[1];
    const double z = y[2];
    const double vx = y[3];
    const double vy = y[4];
    const double vz = y[5];

    const double r2 = x * x + yv * yv + z * z;
    const double r = std::sqrt(r2);
    const double inv_r3 = 1.0 / (r2 * r);
    constexpr double mu = 398600.4418;

    dydt[0] = vx;
    dydt[1] = vy;
    dydt[2] = vz;
    dydt[3] = -mu * x * inv_r3;
    dydt[4] = -mu * yv * inv_r3;
    dydt[5] = -mu * z * inv_r3;
  };

  const double period_s = 2.0 * std::numbers::pi * std::sqrt((kOrbitalRadiusKm * kOrbitalRadiusKm * kOrbitalRadiusKm) / kMu);

  ode::IntegratorOptions opt;
  opt.adaptive = true;
  opt.rtol = 1e-12;
  opt.atol = 1e-15;
  opt.h_init = 10.0;
  opt.h_min = 1e-6;
  opt.h_max = 60.0;

  const auto res = ode::integrate(ode::RKMethod::RKF78, rhs, 0.0, y0, period_s, opt);
  if (res.status != ode::IntegratorStatus::Success) {
    ode::log::Error("integration failed, status=", ode::ToString(res.status));
    return 1;
  }

  ode::log::Info("Final state after ~1 orbit:");
  ode::log::Info("r = [", res.y[0], ", ", res.y[1], ", ", res.y[2], "] km");
  ode::log::Info("v = [", res.y[3], ", ", res.y[4], ", ", res.y[5], "] km/s");
  ode::log::Info("accepted steps = ", res.stats.accepted_steps, " rejected = ", res.stats.rejected_steps);

  return 0;
}
