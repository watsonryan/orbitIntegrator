#include <cmath>
#include <vector>

#include "ode/logging.hpp"
#include "ode/ode.hpp"

namespace {

constexpr double kMu = 398600.4418;

double SpecificEnergy(const std::vector<double>& y) {
  const double r = std::sqrt(y[0] * y[0] + y[1] * y[1] + y[2] * y[2]);
  const double v2 = y[3] * y[3] + y[4] * y[4] + y[5] * y[5];
  return 0.5 * v2 - kMu / r;
}

double AngularMomentumNorm(const std::vector<double>& y) {
  const double hx = y[1] * y[5] - y[2] * y[4];
  const double hy = y[2] * y[3] - y[0] * y[5];
  const double hz = y[0] * y[4] - y[1] * y[3];
  return std::sqrt(hx * hx + hy * hy + hz * hz);
}

}  // namespace

int main() {
  using State = std::vector<double>;

  auto rhs = [](double, const State& y, State& dydt) {
    dydt.resize(6);
    const double x = y[0];
    const double yy = y[1];
    const double z = y[2];
    const double r2 = x * x + yy * yy + z * z;
    const double r = std::sqrt(r2);
    const double inv_r3 = 1.0 / (r2 * r);
    dydt[0] = y[3];
    dydt[1] = y[4];
    dydt[2] = y[5];
    dydt[3] = -kMu * x * inv_r3;
    dydt[4] = -kMu * yy * inv_r3;
    dydt[5] = -kMu * z * inv_r3;
  };

  const double r0 = 7000.0;
  const double v0 = std::sqrt(kMu / r0);
  State y0{r0, 0.0, 0.0, 0.0, v0, 0.0};
  const double period = 2.0 * 3.14159265358979323846 * std::sqrt((r0 * r0 * r0) / kMu);
  const double tf = 20.0 * period;

  ode::IntegratorOptions opt;
  opt.adaptive = false;
  opt.fixed_h = 20.0;
  const auto res = ode::integrate(ode::RKMethod::RK8, rhs, 0.0, y0, tf, opt);
  if (res.status != ode::IntegratorStatus::Success) {
    ode::log::Error("long-horizon propagation failed");
    return 1;
  }

  const double e0 = SpecificEnergy(y0);
  const double e1 = SpecificEnergy(res.y);
  const double h0 = AngularMomentumNorm(y0);
  const double h1 = AngularMomentumNorm(res.y);
  const double e_rel = std::abs((e1 - e0) / e0);
  const double h_rel = std::abs((h1 - h0) / h0);

  if (e_rel > 2e-7) {
    ode::log::Error("energy drift too large, rel=", e_rel);
    return 1;
  }
  if (h_rel > 2e-7) {
    ode::log::Error("angular momentum drift too large, rel=", h_rel);
    return 1;
  }

  return 0;
}

