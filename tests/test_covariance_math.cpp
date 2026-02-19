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

  // Joseph-form update should preserve symmetry/PSD.
  {
    const std::size_t n = 2;
    const std::size_t m = 1;
    Matrix p_prior{
        0.25, 0.05,
        0.05, 0.40};
    Matrix h{
        1.0, 0.0};
    Matrix r{
        0.01};
    const double s_innov = h[0] * (p_prior[0] * h[0] + p_prior[1] * h[1]) +
                           h[1] * (p_prior[2] * h[0] + p_prior[3] * h[1]) + r[0];
    Matrix k{
        (p_prior[0] * h[0] + p_prior[1] * h[1]) / s_innov,
        (p_prior[2] * h[0] + p_prior[3] * h[1]) / s_innov};

    const Matrix p_post = ode::variational::covariance_joseph_update(p_prior, k, h, r, n, m);
    if (p_post.size() != 4) {
      ode::log::Error("joseph update returned invalid size");
      return 1;
    }
    if (std::abs(p_post[1] - p_post[2]) > 1e-12) {
      ode::log::Error("joseph update symmetry check failed");
      return 1;
    }
    if (MinEigenvalue2x2(p_post) < -1e-12) {
      ode::log::Error("joseph update PSD check failed");
      return 1;
    }

    Matrix s_prior;
    if (!ode::variational::cholesky_lower(p_prior, n, s_prior)) {
      ode::log::Error("joseph test: prior cholesky failed");
      return 1;
    }
    Matrix s_r;
    if (!ode::variational::cholesky_lower(r, m, s_r)) {
      ode::log::Error("joseph test: measurement cholesky failed");
      return 1;
    }
    const Matrix s_post = ode::variational::covariance_measurement_update_sqrt_information(s_prior, h, s_r, n, m);
    if (s_post.size() != n * n) {
      ode::log::Error("sqrt information update returned invalid size");
      return 1;
    }
    Matrix p_post_s(4, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        double s = 0.0;
        for (std::size_t kk = 0; kk < n; ++kk) {
          s += s_post[i * n + kk] * s_post[j * n + kk];
        }
        p_post_s[i * n + j] = s;
      }
    }
    const double err = std::abs(p_post_s[0] - p_post[0]) +
                       std::abs(p_post_s[1] - p_post[1]) +
                       std::abs(p_post_s[2] - p_post[2]) +
                       std::abs(p_post_s[3] - p_post[3]);
    if (err > 1e-8) {
      ode::log::Error("sqrt information update mismatch with Joseph form, err=", err);
      return 1;
    }
  }

  // Square-root discrete propagation consistency with direct covariance propagation.
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
    Matrix s0;
    if (!ode::variational::cholesky_lower(p0, 2, s0)) {
      ode::log::Error("cholesky_lower failed for valid SPD matrix");
      return 1;
    }
    const Matrix s1 = ode::variational::propagate_covariance_discrete_sqrt(phi, s0, qd, 2);
    if (s1.size() != 4) {
      ode::log::Error("sqrt covariance propagation returned invalid size");
      return 1;
    }
    Matrix p1_from_s(4, 0.0);
    for (std::size_t i = 0; i < 2; ++i) {
      for (std::size_t j = 0; j < 2; ++j) {
        double s = 0.0;
        for (std::size_t k = 0; k < 2; ++k) {
          s += s1[i * 2 + k] * s1[j * 2 + k];
        }
        p1_from_s[i * 2 + j] = s;
      }
    }

    const Matrix p1_direct = ode::variational::propagate_covariance_discrete(phi, p0, qd, 2);
    const double err = std::abs(p1_from_s[0] - p1_direct[0]) +
                       std::abs(p1_from_s[1] - p1_direct[1]) +
                       std::abs(p1_from_s[2] - p1_direct[2]) +
                       std::abs(p1_from_s[3] - p1_direct[3]);
    if (err > 1e-8) {
      ode::log::Error("sqrt covariance propagation mismatch, err=", err);
      return 1;
    }
  }

  return 0;
}
