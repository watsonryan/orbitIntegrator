#include <cmath>
#include <vector>

#include "ode/logging.hpp"
#include "ode/ode.hpp"

int main() {
  using State = std::vector<double>;

  // Lorenz attractor consistency: high-accuracy RKF78 vs RKF45.
  {
    auto lorenz = [](double, const State& y, State& dydt) {
      constexpr double sigma = 10.0;
      constexpr double rho = 28.0;
      constexpr double beta = 8.0 / 3.0;
      dydt.resize(3);
      dydt[0] = sigma * (y[1] - y[0]);
      dydt[1] = y[0] * (rho - y[2]) - y[1];
      dydt[2] = y[0] * y[1] - beta * y[2];
    };

    const State y0{1.0, 1.0, 1.0};
    ode::IntegratorOptions opt78;
    opt78.rtol = 1e-12;
    opt78.atol = 1e-14;
    opt78.h_init = 1e-3;
    const auto r78 = ode::integrate(ode::RKMethod::RKF78, lorenz, 0.0, y0, 1.0, opt78);
    if (r78.status != ode::IntegratorStatus::Success) {
      ode::log::Error("lorenz RKF78 failed");
      return 1;
    }

    ode::IntegratorOptions opt45 = opt78;
    opt45.rtol = 1e-10;
    opt45.atol = 1e-12;
    const auto r45 = ode::integrate(ode::RKMethod::RKF45, lorenz, 0.0, y0, 1.0, opt45);
    if (r45.status != ode::IntegratorStatus::Success) {
      ode::log::Error("lorenz RKF45 failed");
      return 1;
    }

    const double diff = std::abs(r78.y[0] - r45.y[0]) + std::abs(r78.y[1] - r45.y[1]) + std::abs(r78.y[2] - r45.y[2]);
    if (diff > 3e-6) {
      ode::log::Error("lorenz solver inconsistency too large, diff=", diff);
      return 1;
    }
  }

  // Van der Pol (mu=1) consistency.
  {
    auto vdp = [](double, const State& y, State& dydt) {
      constexpr double mu = 1.0;
      dydt.resize(2);
      dydt[0] = y[1];
      dydt[1] = mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
    };
    const State y0{2.0, 0.0};
    ode::IntegratorOptions opt_ref;
    opt_ref.rtol = 1e-12;
    opt_ref.atol = 1e-14;
    opt_ref.h_init = 1e-3;
    const auto ref = ode::integrate(ode::RKMethod::RKF78, vdp, 0.0, y0, 10.0, opt_ref);
    if (ref.status != ode::IntegratorStatus::Success) {
      ode::log::Error("vdp reference failed");
      return 1;
    }

    ode::IntegratorOptions opt;
    opt.rtol = 1e-8;
    opt.atol = 1e-10;
    opt.h_init = 1e-2;
    const auto test = ode::integrate(ode::RKMethod::RKF45, vdp, 0.0, y0, 10.0, opt);
    if (test.status != ode::IntegratorStatus::Success) {
      ode::log::Error("vdp test failed");
      return 1;
    }
    const double diff = std::abs(ref.y[0] - test.y[0]) + std::abs(ref.y[1] - test.y[1]);
    if (diff > 5e-4) {
      ode::log::Error("vdp solver inconsistency too large, diff=", diff);
      return 1;
    }
  }

  // Kepler energy drift sanity with fixed-step RK8.
  {
    constexpr double mu = 398600.4418;
    constexpr double r0 = 7000.0;
    const double v0 = std::sqrt(mu / r0);
    const double period = 2.0 * 3.14159265358979323846 * std::sqrt((r0 * r0 * r0) / mu);

    auto two_body = [](double, const State& y, State& dydt) {
      constexpr double mu_local = 398600.4418;
      dydt.resize(6);
      const double x = y[0];
      const double yv = y[1];
      const double z = y[2];
      const double r2 = x * x + yv * yv + z * z;
      const double r = std::sqrt(r2);
      const double inv_r3 = 1.0 / (r2 * r);
      dydt[0] = y[3];
      dydt[1] = y[4];
      dydt[2] = y[5];
      dydt[3] = -mu_local * x * inv_r3;
      dydt[4] = -mu_local * yv * inv_r3;
      dydt[5] = -mu_local * z * inv_r3;
    };

    auto specific_energy = [](const State& y) {
      const double r = std::sqrt(y[0] * y[0] + y[1] * y[1] + y[2] * y[2]);
      const double v2 = y[3] * y[3] + y[4] * y[4] + y[5] * y[5];
      return 0.5 * v2 - mu / r;
    };

    State y0{r0, 0.0, 0.0, 0.0, v0, 0.0};
    ode::IntegratorOptions opt;
    opt.adaptive = false;
    opt.fixed_h = 10.0;
    const auto res = ode::integrate(ode::RKMethod::RK8, two_body, 0.0, y0, 10.0 * period, opt);
    if (res.status != ode::IntegratorStatus::Success) {
      ode::log::Error("kepler RK8 failed");
      return 1;
    }
    const double e0 = specific_energy(y0);
    const double e1 = specific_energy(res.y);
    const double rel = std::abs((e1 - e0) / e0);
    if (rel > 2e-8) {
      ode::log::Error("kepler energy drift too large, rel=", rel);
      return 1;
    }
  }

  return 0;
}

