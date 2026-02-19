#include <array>
#include <cmath>
#include <iostream>
#include <utility>

#include "ode/algebra_adapters.hpp"
#include "ode/logging.hpp"
#include "ode/ode.hpp"

namespace {

struct CustomState {
  std::array<double, 2> d{};

  [[nodiscard]] std::size_t size() const { return d.size(); }
  double& operator[](std::size_t i) { return d[i]; }
  const double& operator[](std::size_t i) const { return d[i]; }
};

struct CustomAccessor {
  static std::size_t size(const CustomState&) { return 2; }
  static void resize_like(CustomState&, const CustomState&) {}
  static void assign(CustomState& dst, const CustomState& src) { dst = src; }
  static void set_zero(CustomState& x) { x.d = {0.0, 0.0}; }
  static void axpy(double a, const CustomState& x, CustomState& y) {
    y.d[0] += a * x.d[0];
    y.d[1] += a * x.d[1];
  }
  static bool finite(const CustomState& x) {
    return std::isfinite(x.d[0]) && std::isfinite(x.d[1]);
  }
};

}  // namespace

int main() {
  {
    using State = std::array<double, 2>;
    State y0{1.0, 0.0};
    auto rhs = [](double, const State& y, State& dydt) {
      dydt[0] = y[1];
      dydt[1] = -y[0];
    };

    ode::IntegratorOptions opt;
    opt.adaptive = true;
    opt.rtol = 1e-11;
    opt.atol = 1e-13;
    opt.h_init = 0.05;

    const double t1 = 2.0 * 3.14159265358979323846;
    const auto res = ode::integrate<State, decltype(rhs), ode::StdArrayAlgebra<2>>(
        ode::RKMethod::RKF78, std::move(rhs), 0.0, y0, t1, opt);
    if (res.status != ode::IntegratorStatus::Success) {
      ode::log::Error("StdArrayAlgebra integration failed");
      return 1;
    }
    if (std::abs(res.y[0] - 1.0) > 1e-9 || std::abs(res.y[1]) > 1e-9) {
      ode::log::Error("StdArrayAlgebra mismatch");
      return 1;
    }
  }

  {
    using State = CustomState;
    using Algebra = ode::AccessorAlgebra<CustomAccessor, State>;

    State y0{};
    y0[0] = 1.0;
    y0[1] = 0.0;

    auto rhs = [](double, const State& y, State& dydt) {
      dydt[0] = y[1];
      dydt[1] = -y[0];
    };

    ode::IntegratorOptions opt;
    opt.adaptive = true;
    opt.rtol = 1e-11;
    opt.atol = 1e-13;
    opt.h_init = 0.05;

    const double t1 = 2.0 * 3.14159265358979323846;
    const auto res = ode::integrate<State, decltype(rhs), Algebra>(
        ode::RKMethod::RKF78, std::move(rhs), 0.0, y0, t1, opt);
    if (res.status != ode::IntegratorStatus::Success) {
      ode::log::Error("AccessorAlgebra integration failed");
      return 1;
    }
    if (std::abs(res.y[0] - 1.0) > 1e-9 || std::abs(res.y[1]) > 1e-9) {
      ode::log::Error("AccessorAlgebra mismatch");
      return 1;
    }
  }

  return 0;
}
