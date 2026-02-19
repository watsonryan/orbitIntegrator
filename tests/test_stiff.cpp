#include <cmath>
#include <iostream>
#include <vector>

#include "ode/stiff/implicit_euler.hpp"

int main() {
  {
    std::vector<double> y0{1.0};
    auto rhs = [](double, const std::vector<double>& y, std::vector<double>& dydt) {
      dydt.resize(1);
      dydt[0] = -15.0 * y[0];
    };

    ode::stiff::Options opt;
    opt.h = 0.01;
    opt.newton_tol = 1e-12;

    const auto res = ode::stiff::integrate_implicit_euler(rhs, 0.0, y0, 1.0, opt);
    if (res.status != ode::stiff::Status::Success) {
      std::cerr << "stiff solver failed on linear test\n";
      return 1;
    }

    const double exact = std::exp(-15.0);
    if (std::abs(res.y[0] - exact) > 2e-4) {
      std::cerr << "linear stiff mismatch: " << res.y[0] << " vs " << exact << "\n";
      return 1;
    }
  }

  {
    std::vector<double> y0{1.0};
    auto rhs = [](double t, const std::vector<double>& y, std::vector<double>& dydt) {
      dydt.resize(1);
      dydt[0] = -1000.0 * (y[0] - std::cos(t)) - std::sin(t);
    };

    ode::stiff::Options opt;
    opt.h = 0.01;

    const auto res = ode::stiff::integrate_implicit_euler(rhs, 0.0, y0, 1.0, opt);
    if (res.status != ode::stiff::Status::Success) {
      std::cerr << "stiff solver failed on forced test\n";
      return 1;
    }

    const double exact = std::cos(1.0);
    if (std::abs(res.y[0] - exact) > 5e-2) {
      std::cerr << "forced stiff mismatch: " << res.y[0] << " vs " << exact << "\n";
      return 1;
    }
  }

  return 0;
}
