#include <cmath>

#include <Eigen/Core>

#include "ode/eigen_api.hpp"
#include "ode/logging.hpp"

namespace {

constexpr double kPi = 3.14159265358979323846;

struct GenericDynamics {
  template <class Scalar>
  void operator()(double, const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x,
                  Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& dxdt) const {
    dxdt.resize(2);
    dxdt(0) = x(1);
    dxdt(1) = -2.0 * x(0) - 3.0 * x(1);
  }
};

int TestEigenIntegrateHarmonic() {
  using Vec = ode::eigen::Vector;

  Vec y0(2);
  y0 << 1.0, 0.0;

  auto rhs = [](double, const Vec& y, Vec& dydt) {
    dydt.resize(2);
    dydt(0) = y(1);
    dydt(1) = -y(0);
  };

  ode::IntegratorOptions opt;
  opt.adaptive = true;
  opt.rtol = 1e-11;
  opt.atol = 1e-13;
  opt.h_init = 0.05;

  const auto res = ode::eigen::integrate(ode::RKMethod::RKF78, rhs, 0.0, y0, 2.0 * kPi, opt);
  if (res.status != ode::IntegratorStatus::Success) {
    ode::log::Error("eigen integrate failed");
    return 1;
  }
  if (std::abs(res.y(0) - 1.0) > 1e-9 || std::abs(res.y(1)) > 1e-9) {
    ode::log::Error("eigen integrate mismatch");
    return 1;
  }
  return 0;
}

int TestEigenStateStmCov() {
  using Vec = ode::eigen::Vector;
  using Mat = ode::eigen::Matrix;

  auto dynamics = [](double, const Vec& x, Vec& dxdt) {
    dxdt.resize(2);
    dxdt(0) = x(1);
    dxdt(1) = -2.0 * x(0) - 3.0 * x(1);
  };
  auto jac = [](double, const Vec&, Mat& a) {
    a.resize(2, 2);
    a << 0.0, 1.0, -2.0, -3.0;
    return true;
  };
  auto q = [](double, const Vec&, Mat& q_out) {
    q_out.setZero(2, 2);
    q_out(0, 0) = 0.1;
    q_out(1, 1) = 0.2;
    return true;
  };
  auto q_zero = [](double, const Vec&, Mat& q_out) {
    q_out.setZero(2, 2);
    return true;
  };

  Vec x0(2);
  x0 << 1.0, -0.5;
  Mat p0 = Mat::Identity(2, 2);

  ode::IntegratorOptions opt;
  opt.rtol = 1e-11;
  opt.atol = 1e-13;
  opt.h_init = 1e-3;

  const auto out = ode::eigen::uncertainty::integrate_state_stm_cov(
      ode::RKMethod::RKF78, dynamics, jac, q, 0.0, x0, p0, 1.0, opt);
  const auto out_zero_q = ode::eigen::uncertainty::integrate_state_stm_cov(
      ode::RKMethod::RKF78, dynamics, jac, q_zero, 0.0, x0, p0, 1.0, opt);
  if (out.status != ode::IntegratorStatus::Success) {
    ode::log::Error("eigen uncertainty integration failed");
    return 1;
  }
  if (out_zero_q.status != ode::IntegratorStatus::Success) {
    ode::log::Error("eigen uncertainty integration failed for zero-Q reference");
    return 1;
  }
  if (out.x.size() != 2 || out.phi.rows() != 2 || out.phi.cols() != 2 || out.p.rows() != 2 || out.p.cols() != 2) {
    ode::log::Error("eigen uncertainty shape mismatch");
    return 1;
  }
  if (!(out.p(0, 0) > out_zero_q.p(0, 0)) || !(out.p(1, 1) > out_zero_q.p(1, 1))) {
    ode::log::Error("expected larger covariance than zero-Q case");
    return 1;
  }
  return 0;
}

int TestEigenForwardAdJacobian() {
  using Vec = ode::eigen::Vector;
  using Mat = ode::eigen::Matrix;

  Vec x(2);
  x << 0.2, -0.4;
  Mat a;
  const bool ok = ode::eigen::uncertainty::jacobian_forward_ad(GenericDynamics{}, 0.0, x, a);
  if (!ok) {
    ode::log::Error("eigen jacobian_forward_ad failed");
    return 1;
  }
  if (a.rows() != 2 || a.cols() != 2) {
    ode::log::Error("eigen jacobian shape mismatch");
    return 1;
  }
  if (std::abs(a(0, 0) - 0.0) > 1e-12 || std::abs(a(0, 1) - 1.0) > 1e-12 ||
      std::abs(a(1, 0) + 2.0) > 1e-12 || std::abs(a(1, 1) + 3.0) > 1e-12) {
    ode::log::Error("eigen jacobian values mismatch");
    return 1;
  }
  return 0;
}

}  // namespace

int main() {
  if (TestEigenIntegrateHarmonic() != 0) {
    return 1;
  }
  if (TestEigenStateStmCov() != 0) {
    return 1;
  }
  if (TestEigenForwardAdJacobian() != 0) {
    return 1;
  }
  return 0;
}
