#include <cmath>
#include <iostream>
#include <vector>

#include "ode/ode.hpp"

namespace {

constexpr double kPi = 3.14159265358979323846;

double Phi11(double t) {
  return 2.0 * std::exp(-t) - std::exp(-2.0 * t);
}
double Phi12(double t) {
  return std::exp(-t) - std::exp(-2.0 * t);
}
double Phi21(double t) {
  return -2.0 * std::exp(-t) + 2.0 * std::exp(-2.0 * t);
}
double Phi22(double t) {
  return -std::exp(-t) + 2.0 * std::exp(-2.0 * t);
}

int TestLinearStateStmWithAutoDiff() {
  using State = ode::DynamicState;
  using Matrix = ode::DynamicMatrix;

  auto dynamics = [](double /*t*/, const auto& x, auto& dxdt) {
    dxdt.resize(2);
    dxdt[0] = x[1];
    dxdt[1] = -2.0 * x[0] - 3.0 * x[1];
  };

  auto jac_ad = [&dynamics](double t, const State& x, Matrix& a) {
    return ode::uncertainty::jacobian_forward_ad(dynamics, t, x, a);
  };

  State x0{1.0, -0.5};
  ode::IntegratorOptions opt;
  opt.rtol = 1e-11;
  opt.atol = 1e-13;
  opt.h_init = 1e-3;

  const double t1 = 1.0;
  const auto res =
      ode::uncertainty::integrate_state_stm(ode::RKMethod::RKF78, dynamics, jac_ad, 0.0, x0, t1, opt);
  if (res.status != ode::IntegratorStatus::Success) {
    std::cerr << "state/stm integration failed\n";
    return 1;
  }

  const double p11 = Phi11(t1);
  const double p12 = Phi12(t1);
  const double p21 = Phi21(t1);
  const double p22 = Phi22(t1);
  const double x_ref0 = p11 * x0[0] + p12 * x0[1];
  const double x_ref1 = p21 * x0[0] + p22 * x0[1];

  auto absdiff = [](double a, double b) { return std::abs(a - b); };
  if (absdiff(res.phi[0], p11) > 1e-10 || absdiff(res.phi[1], p12) > 1e-10 ||
      absdiff(res.phi[2], p21) > 1e-10 || absdiff(res.phi[3], p22) > 1e-10) {
    std::cerr << "stm mismatch\n";
    return 1;
  }
  if (absdiff(res.x[0], x_ref0) > 1e-10 || absdiff(res.x[1], x_ref1) > 1e-10) {
    std::cerr << "state mismatch with analytic Phi*x0\n";
    return 1;
  }
  return 0;
}

int TestContinuousCovariancePropagation() {
  using State = ode::DynamicState;
  using Matrix = ode::DynamicMatrix;

  auto dynamics = [](double /*t*/, const State& x, State& dxdt) {
    dxdt.assign(x.size(), 0.0);
  };
  auto jac_zero = [](double /*t*/, const State& x, Matrix& a) {
    const std::size_t n = x.size();
    a.assign(n * n, 0.0);
    return true;
  };
  auto q_const = [](double /*t*/, const State& x, Matrix& q) {
    const std::size_t n = x.size();
    q.assign(n * n, 0.0);
    q[0] = 2.0;
    q[3] = 0.5;
    return true;
  };

  State x0{7000.0, 7.5};
  Matrix p0{10.0, 1.0, 1.0, 8.0};

  ode::IntegratorOptions opt;
  opt.rtol = 1e-11;
  opt.atol = 1e-13;
  opt.h_init = 1e-3;
  const double t1 = 2.0 * kPi;

  const auto res = ode::uncertainty::integrate_state_stm_cov(
      ode::RKMethod::RKF78, dynamics, jac_zero, q_const, 0.0, x0, p0, t1, opt);
  if (res.status != ode::IntegratorStatus::Success) {
    std::cerr << "state/stm/cov integration failed\n";
    return 1;
  }

  const double dt = t1;
  Matrix p_ref = p0;
  p_ref[0] += 2.0 * dt;
  p_ref[3] += 0.5 * dt;

  if (std::abs(res.phi[0] - 1.0) > 1e-11 || std::abs(res.phi[1]) > 1e-11 ||
      std::abs(res.phi[2]) > 1e-11 || std::abs(res.phi[3] - 1.0) > 1e-11) {
    std::cerr << "phi should remain identity for A=0\n";
    return 1;
  }

  for (std::size_t i = 0; i < p_ref.size(); ++i) {
    if (std::abs(res.p[i] - p_ref[i]) > 1e-8) {
      std::cerr << "covariance mismatch at i=" << i << "\n";
      return 1;
    }
  }
  return 0;
}

int TestDiscreteCovarianceStep() {
  using Matrix = ode::DynamicMatrix;
  const std::size_t n = 2;

  Matrix phi{
      1.0, 2.0,
      0.0, 1.0};
  Matrix p0{
      3.0, 1.0,
      1.0, 4.0};
  Matrix qd{
      0.5, 0.0,
      0.0, 0.25};

  const Matrix p1 = ode::uncertainty::propagate_covariance_discrete(phi, p0, qd, n);
  const Matrix ref{
      23.5, 9.0,
      9.0, 4.25};
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (std::abs(p1[i] - ref[i]) > 1e-12) {
      std::cerr << "discrete covariance update mismatch at i=" << i << "\n";
      return 1;
    }
  }
  return 0;
}

}  // namespace

int main() {
  if (TestLinearStateStmWithAutoDiff() != 0) {
    return 1;
  }
  if (TestContinuousCovariancePropagation() != 0) {
    return 1;
  }
  if (TestDiscreteCovarianceStep() != 0) {
    return 1;
  }
  return 0;
}
