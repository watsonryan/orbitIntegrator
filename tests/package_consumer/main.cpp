#include <vector>

#include "ode/ode.hpp"

int main() {
  using State = std::vector<double>;
  const State y0{1.0};

  auto rhs = [](double, const State& y, State& dydt) {
    dydt.resize(y.size());
    dydt[0] = y[0];
  };

  ode::IntegratorOptions opt;
  opt.adaptive = true;
  opt.rtol = 1e-10;
  opt.atol = 1e-12;

  const auto res = ode::integrate(ode::RKMethod::RKF45, rhs, 0.0, y0, 1.0, opt);
  return (res.status == ode::IntegratorStatus::Success) ? 0 : 1;
}
