#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "ode/ode.hpp"

int main() {
  using State = std::vector<double>;
  const State y0{1.0};

  auto rhs_good = [](double, const State& y, State& dydt) {
    dydt.resize(y.size());
    dydt[0] = y[0];
  };

  {
    ode::IntegratorOptions opt;
    opt.adaptive = true;
    opt.rtol = -1.0;
    const auto res = ode::integrate(ode::RKMethod::RKF45, rhs_good, 0.0, y0, 1.0, opt);
    if (res.status != ode::IntegratorStatus::InvalidTolerance) {
      std::cerr << "expected InvalidTolerance\n";
      return 1;
    }
  }

  {
    ode::IntegratorOptions opt;
    opt.adaptive = false;
    opt.fixed_h = 0.0;
    const auto res = ode::integrate(ode::RKMethod::RK4, rhs_good, 0.0, y0, 1.0, opt);
    if (res.status != ode::IntegratorStatus::InvalidStepSize) {
      std::cerr << "expected InvalidStepSize\n";
      return 1;
    }
  }

  {
    ode::IntegratorOptions opt;
    opt.adaptive = false;
    opt.fixed_h = 0.1;
    opt.max_steps = 3;
    const auto res = ode::integrate(ode::RKMethod::RK4, rhs_good, 0.0, y0, 1.0, opt);
    if (res.status != ode::IntegratorStatus::MaxStepsExceeded) {
      std::cerr << "expected MaxStepsExceeded\n";
      return 1;
    }
  }

  {
    ode::IntegratorOptions opt;
    opt.adaptive = false;
    opt.fixed_h = 0.1;
    int accepted = 0;
    ode::Observer<State> obs = [&accepted](double, const State&) {
      ++accepted;
      return accepted < 3;
    };
    const auto res = ode::integrate(ode::RKMethod::RK4, rhs_good, 0.0, y0, 1.0, opt, obs);
    if (res.status != ode::IntegratorStatus::UserStopped) {
      std::cerr << "expected UserStopped\n";
      return 1;
    }
  }

  {
    ode::IntegratorOptions opt;
    opt.adaptive = false;
    opt.fixed_h = 0.3;
    const auto res = ode::integrate(ode::RKMethod::RK4, rhs_good, 0.0, y0, 1.0, opt);
    if (res.status != ode::IntegratorStatus::Success) {
      std::cerr << "expected Success for endpoint clamp\n";
      return 1;
    }
    if (std::abs(res.t - 1.0) > 1e-15) {
      std::cerr << "expected exact endpoint hit\n";
      return 1;
    }
  }

  {
    ode::IntegratorOptions opt;
    opt.adaptive = true;
    opt.rtol = 1e-10;
    opt.atol = 1e-13;
    opt.h_init = 10.0;
    opt.h_max = 0.05;
    const auto res = ode::integrate(ode::RKMethod::RKF45, rhs_good, 0.0, y0, 0.2, opt);
    if (res.status != ode::IntegratorStatus::Success) {
      std::cerr << "expected Success for h_max cap case\n";
      return 1;
    }
    if (std::abs(res.stats.last_h) > opt.h_max + 1e-15) {
      std::cerr << "expected last_h <= h_max\n";
      return 1;
    }
  }

  {
    auto rhs_stiffish = [](double, const State& y, State& dydt) {
      dydt.resize(y.size());
      dydt[0] = -1000.0 * y[0];
    };
    ode::IntegratorOptions opt;
    opt.adaptive = true;
    opt.rtol = 1e-12;
    opt.atol = 1e-15;
    opt.h_init = 1e-3;
    opt.h_min = 1e-4;
    opt.h_max = 1.0;
    const auto res = ode::integrate(ode::RKMethod::RKF45, rhs_stiffish, 0.0, y0, 1.0, opt);
    if (res.status != ode::IntegratorStatus::StepSizeUnderflow) {
      std::cerr << "expected StepSizeUnderflow\n";
      return 1;
    }
  }

  {
    auto rhs_nan = [](double t, const State& y, State& dydt) {
      dydt.resize(y.size());
      if (t > 0.2) {
        dydt[0] = std::numeric_limits<double>::quiet_NaN();
      } else {
        dydt[0] = y[0];
      }
    };
    ode::IntegratorOptions opt;
    opt.adaptive = true;
    opt.rtol = 1e-9;
    opt.atol = 1e-12;
    opt.h_init = 0.1;
    const auto res = ode::integrate(ode::RKMethod::RKF45, rhs_nan, 0.0, y0, 1.0, opt);
    if (res.status != ode::IntegratorStatus::NaNDetected) {
      std::cerr << "expected NaNDetected\n";
      return 1;
    }
  }

  return 0;
}
