#include <cmath>
#include <vector>

#include "ode/logging.hpp"
#include "ode/variational.hpp"

namespace {

double MinEigenvalue2x2(const std::vector<double>& p) {
  const double a = p[0];
  const double b = 0.5 * (p[1] + p[2]);
  const double d = p[3];
  const double tr = a + d;
  const double det_term = std::sqrt(std::max(0.0, 0.25 * tr * tr - (a * d - b * b)));
  return 0.5 * tr - det_term;
}

}  // namespace

int main() {
  using State = ode::variational::State;
  using Matrix = ode::variational::Matrix;

  // Discrete covariance PSD/symmetry check.
  {
    Matrix phi{
        1.0, 0.1,
        0.0, 1.0};
    Matrix p0{
        1.0, 0.2,
        0.2, 0.5};
    Matrix qd{
        0.01, 0.0,
        0.0, 0.02};
    const Matrix p1 = ode::variational::propagate_covariance_discrete(phi, p0, qd, 2);
    if (std::abs(p1[1] - p1[2]) > 1e-12) {
      ode::log::Error("covariance symmetry check failed");
      return 1;
    }
    if (MinEigenvalue2x2(p1) < -1e-12) {
      ode::log::Error("covariance PSD check failed");
      return 1;
    }
  }

  // Continuous vs one-step discrete approximation consistency for small dt.
  {
    auto rhs = [](double, const State& x, State& dxdt) {
      dxdt.resize(2);
      dxdt[0] = x[1];
      dxdt[1] = -0.2 * x[1];
    };
    auto jac = [](double, const State&, Matrix& a) {
      a.assign(4, 0.0);
      a[1] = 1.0;
      a[3] = -0.2;
      return true;
    };
    auto q = [](double, const State&, Matrix& q_out) {
      q_out.assign(4, 0.0);
      q_out[0] = 0.01;
      q_out[3] = 0.02;
      return true;
    };

    const double dt = 1e-3;
    ode::IntegratorOptions opt;
    opt.rtol = 1e-12;
    opt.atol = 1e-14;
    opt.h_init = dt;
    opt.h_max = dt;
    State x0{1.0, 0.0};
    Matrix p0{
        0.5, 0.0,
        0.0, 0.3};

    const auto cont = ode::variational::integrate_state_stm_cov(
        ode::RKMethod::RKF78, rhs, jac, q, 0.0, x0, p0, dt, opt);
    if (cont.status != ode::IntegratorStatus::Success) {
      ode::log::Error("continuous covariance propagation failed");
      return 1;
    }
    Matrix qd{
        0.01 * dt, 0.0,
        0.0, 0.02 * dt};
    const Matrix disc = ode::variational::propagate_covariance_discrete(cont.phi, p0, qd, 2);
    const double err = std::abs(cont.p[0] - disc[0]) + std::abs(cont.p[1] - disc[1]) +
                       std::abs(cont.p[2] - disc[2]) + std::abs(cont.p[3] - disc[3]);
    if (err > 5e-7) {
      ode::log::Error("continuous/discrete covariance consistency failed, err=", err);
      return 1;
    }
  }

  return 0;
}

