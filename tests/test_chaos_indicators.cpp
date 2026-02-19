#include <vector>

#include "ode/chaos.hpp"
#include "ode/logging.hpp"

int main() {
  using State = std::vector<double>;

  ode::IntegratorOptions opt;
  opt.adaptive = true;
  opt.rtol = 1e-9;
  opt.atol = 1e-12;
  opt.h_init = 0.01;

  const State delta0{1e-8, 0.0};

  auto rhs_osc = [](double, const State& x, State& dxdt) {
    dxdt.resize(2);
    dxdt[0] = x[1];
    dxdt[1] = -x[0];
  };
  auto jac_osc = [](double, const State&, State& a) {
    a.assign(4, 0.0);
    a[1] = 1.0;
    a[2] = -1.0;
    return true;
  };

  const auto fli_reg = ode::chaos::compute_fli(ode::RKMethod::RKF78, rhs_osc, jac_osc, 0.0, State{1.0, 0.0}, delta0, 50.0, opt);
  if (fli_reg.status != ode::IntegratorStatus::Success) {
    ode::log::Error("regular FLI run failed");
    return 1;
  }

  const auto megno_reg =
      ode::chaos::compute_megno(ode::RKMethod::RKF78, rhs_osc, jac_osc, 0.0, State{1.0, 0.0}, delta0, 50.0, opt);
  if (megno_reg.status != ode::IntegratorStatus::Success) {
    ode::log::Error("regular MEGNO run failed");
    return 1;
  }

  auto rhs_lorenz = [](double, const State& x, State& dxdt) {
    constexpr double sigma = 10.0;
    constexpr double rho = 28.0;
    constexpr double beta = 8.0 / 3.0;
    dxdt.resize(3);
    dxdt[0] = sigma * (x[1] - x[0]);
    dxdt[1] = x[0] * (rho - x[2]) - x[1];
    dxdt[2] = x[0] * x[1] - beta * x[2];
  };
  auto jac_lorenz = [](double, const State& x, State& a) {
    constexpr double sigma = 10.0;
    constexpr double rho = 28.0;
    constexpr double beta = 8.0 / 3.0;
    a.assign(9, 0.0);
    a[0] = -sigma;
    a[1] = sigma;
    a[3] = rho - x[2];
    a[4] = -1.0;
    a[5] = -x[0];
    a[6] = x[1];
    a[7] = x[0];
    a[8] = -beta;
    return true;
  };

  const State delta0_l{1e-8, 0.0, 0.0};
  const auto fli_chaos =
      ode::chaos::compute_fli(ode::RKMethod::RKF78, rhs_lorenz, jac_lorenz, 0.0, State{1.0, 1.0, 1.0}, delta0_l, 20.0, opt);
  if (fli_chaos.status != ode::IntegratorStatus::Success) {
    ode::log::Error("chaotic FLI run failed");
    return 1;
  }
  const auto megno_chaos =
      ode::chaos::compute_megno(ode::RKMethod::RKF78, rhs_lorenz, jac_lorenz, 0.0, State{1.0, 1.0, 1.0}, delta0_l, 20.0, opt);
  if (megno_chaos.status != ode::IntegratorStatus::Success) {
    ode::log::Error("chaotic MEGNO run failed");
    return 1;
  }

  if (!(fli_chaos.fli > fli_reg.fli + 2.0)) {
    ode::log::Error("FLI failed to separate chaotic and regular dynamics");
    return 1;
  }
  if (!(megno_chaos.mean_megno > megno_reg.mean_megno + 0.5)) {
    ode::log::Error("MEGNO failed to separate chaotic and regular dynamics");
    return 1;
  }

  return 0;
}
