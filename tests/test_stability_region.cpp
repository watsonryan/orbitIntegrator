#include <cmath>
#include <limits>
#include <vector>

#include "ode/logging.hpp"
#include "ode/ode.hpp"

namespace {

double AmplificationAfter(double lambda, double h, double t_end) {
  using State = std::vector<double>;
  auto rhs = [lambda](double, const State& y, State& dydt) {
    dydt.resize(1);
    dydt[0] = lambda * y[0];
  };
  ode::IntegratorOptions opt;
  opt.adaptive = false;
  opt.fixed_h = h;
  const auto res = ode::integrate(ode::RKMethod::RK4, rhs, 0.0, State{1.0}, t_end, opt);
  if (res.status != ode::IntegratorStatus::Success) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return std::abs(res.y[0]);
}

}  // namespace

int main() {
  // Stable and unstable samples around RK4 real-axis stability cutoff (~ -2.785).
  const double h = 0.1;
  const double amp_stable = AmplificationAfter(-20.0, h, 10.0);   // z = -2.0
  const double amp_unstable = AmplificationAfter(-30.0, h, 10.0); // z = -3.0

  if (!std::isfinite(amp_stable) || !std::isfinite(amp_unstable)) {
    ode::log::Error("stability region run failed");
    return 1;
  }
  if (!(amp_stable < 1.0)) {
    ode::log::Error("expected stable sample to decay, amp=", amp_stable);
    return 1;
  }
  if (!(amp_unstable > 1.0)) {
    ode::log::Error("expected unstable sample to grow, amp=", amp_unstable);
    return 1;
  }

  // Coarse boundary detection for regression: find first unstable z on negative real axis.
  double crossing_z = 0.0;
  for (double z = -1.0; z >= -4.0; z -= 0.05) {
    const double lambda = z / h;
    const double amp = AmplificationAfter(lambda, h, 5.0);
    if (!std::isfinite(amp)) {
      ode::log::Error("stability sweep failure at z=", z);
      return 1;
    }
    if (amp > 1.0) {
      crossing_z = z;
      break;
    }
  }
  if (!(crossing_z < -2.6 && crossing_z > -3.1)) {
    ode::log::Error("unexpected RK4 stability boundary estimate, z=", crossing_z);
    return 1;
  }

  return 0;
}
