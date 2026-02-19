#include <cmath>
#include <vector>

#include "ode/logging.hpp"
#include "ode/symplectic.hpp"

int main() {
  using Vec = std::vector<double>;
  constexpr double kPi = 3.14159265358979323846;

  auto accel = [](double, const Vec& r, Vec& a) {
    a.assign(1, -r[0]);
  };
  auto jac_r = [](double, const Vec&, Vec& a) {
    a.assign(1, -1.0);
    return true;
  };
  auto g = [](double, const Vec& r, const Vec&) {
    return 0.5 + std::abs(r[0]);
  };

  const Vec r0{1.0};
  const Vec v0{0.0};
  const double t1 = 2.0 * kPi;

  {
    ode::symplectic::SundmanSymplecticOptions opt;
    opt.ds = 0.01;
    opt.g_min = 0.1;
    opt.g_max = 2.0;
    const auto res = ode::symplectic::integrate_stormer_verlet_sundman(accel, g, 0.0, r0, v0, t1, opt);
    if (res.status != ode::IntegratorStatus::Success) {
      ode::log::Error("sundman symplectic integration failed");
      return 1;
    }
    if (std::abs(res.r[0] - 1.0) > 3e-2 || std::abs(res.v[0]) > 3e-2) {
      ode::log::Error("sundman symplectic final-state mismatch");
      return 1;
    }
  }

  {
    ode::symplectic::SymplecticOptions opt;
    opt.h = 0.01;
    const auto res = ode::symplectic::integrate_stormer_verlet_variational(accel, jac_r, 0.0, r0, v0, t1, opt);
    if (res.status != ode::IntegratorStatus::Success) {
      ode::log::Error("symplectic variational integration failed");
      return 1;
    }
    if (res.phi.size() != 4) {
      ode::log::Error("symplectic variational STM shape mismatch");
      return 1;
    }
    const double c = std::cos(t1);
    const double s = std::sin(t1);
    if (std::abs(res.phi[0] - c) > 2e-3 ||
        std::abs(res.phi[1] - s) > 2e-3 ||
        std::abs(res.phi[2] + s) > 2e-3 ||
        std::abs(res.phi[3] - c) > 2e-3) {
      ode::log::Error("symplectic variational STM mismatch");
      return 1;
    }
  }

  return 0;
}
