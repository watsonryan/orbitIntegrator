/**
 * @file eigen_api.hpp
 * @brief Eigen-first convenience wrappers for integration and uncertainty propagation.
 */
#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

#if __has_include(<Eigen/Core>)
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <unsupported/Eigen/AutoDiff>
#else
#error "Eigen headers not found. Install Eigen >= 5 and/or enable ODE_FETCH_DEPS."
#endif

#include "ode/algebra_adapters.hpp"
#include "ode/ode.hpp"

namespace ode::eigen {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
using MatrixRM = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <class RHS>
[[nodiscard]] inline IntegratorResult<Vector> integrate(RKMethod method,
                                                        RHS&& rhs,
                                                        double t0,
                                                        const Vector& y0,
                                                        double t1,
                                                        IntegratorOptions opt,
                                                        Observer<Vector> obs = {}) {
  return ode::integrate<Vector, RHS, ode::EigenVectorAlgebra<Vector>>(
      method, std::forward<RHS>(rhs), t0, y0, t1, opt, obs);
}

template <class RHS, class DtDs>
[[nodiscard]] inline IntegratorResult<Vector> integrate_sundman(RKMethod method,
                                                                RHS&& rhs,
                                                                DtDs&& dt_ds,
                                                                double t0,
                                                                const Vector& y0,
                                                                double t1,
                                                                IntegratorOptions opt,
                                                                Observer<Vector> obs = {}) {
  return ode::integrate_sundman<Vector, RHS, DtDs, ode::EigenVectorAlgebra<Vector>>(
      method, std::forward<RHS>(rhs), std::forward<DtDs>(dt_ds), t0, y0, t1, opt, obs);
}

namespace detail {

[[nodiscard]] inline bool finite(const Vector& x) {
  return x.allFinite();
}

[[nodiscard]] inline bool finite(const Matrix& x) {
  return x.allFinite();
}

[[nodiscard]] inline MatrixRM map_row_major(const Vector& z, Eigen::Index offset, Eigen::Index n) {
  MatrixRM out(n, n);
  for (Eigen::Index i = 0; i < n; ++i) {
    for (Eigen::Index j = 0; j < n; ++j) {
      out(i, j) = z(offset + i * n + j);
    }
  }
  return out;
}

inline void write_row_major(const MatrixRM& m, Vector& z, Eigen::Index offset) {
  const Eigen::Index n = m.rows();
  for (Eigen::Index i = 0; i < n; ++i) {
    for (Eigen::Index j = 0; j < n; ++j) {
      z(offset + i * n + j) = m(i, j);
    }
  }
}

}  // namespace detail

namespace uncertainty {

struct StateStmResult {
  IntegratorStatus status = IntegratorStatus::Success;
  double t = 0.0;
  Vector x{};
  Matrix phi{};
  IntegratorStats stats{};
};

struct StateStmCovResult {
  IntegratorStatus status = IntegratorStatus::Success;
  double t = 0.0;
  Vector x{};
  Matrix phi{};
  Matrix p{};
  IntegratorStats stats{};
};

template <class Dynamics>
[[nodiscard]] inline bool jacobian_forward_ad(Dynamics&& dynamics, double t, const Vector& x, Matrix& a_out) {
  using AD = Eigen::AutoDiffScalar<Eigen::VectorXd>;
  const Eigen::Index n = x.size();

  Eigen::Matrix<AD, Eigen::Dynamic, 1> x_ad(n);
  for (Eigen::Index i = 0; i < n; ++i) {
    x_ad(i).value() = x(i);
    x_ad(i).derivatives() = Eigen::VectorXd::Zero(n);
    x_ad(i).derivatives()(i) = 1.0;
  }

  Eigen::Matrix<AD, Eigen::Dynamic, 1> f_ad;
  dynamics(t, x_ad, f_ad);
  if (f_ad.size() != n) {
    return false;
  }

  a_out.resize(n, n);
  for (Eigen::Index i = 0; i < n; ++i) {
    if (!std::isfinite(f_ad(i).value()) || f_ad(i).derivatives().size() != n) {
      return false;
    }
    for (Eigen::Index j = 0; j < n; ++j) {
      const double dij = f_ad(i).derivatives()(j);
      if (!std::isfinite(dij)) {
        return false;
      }
      a_out(i, j) = dij;
    }
  }
  return true;
}

template <class RHS, class JacobianFn>
[[nodiscard]] inline StateStmResult integrate_state_stm(RKMethod method,
                                                        RHS&& rhs,
                                                        JacobianFn&& jacobian_fn,
                                                        double t0,
                                                        const Vector& x0,
                                                        double t1,
                                                        IntegratorOptions opt,
                                                        Observer<Vector> obs = {}) {
  const Eigen::Index n = x0.size();
  StateStmResult out{};
  out.t = t0;
  out.x = x0;
  out.phi = Matrix::Identity(n, n);

  Vector z0(n + n * n);
  z0.head(n) = x0;
  detail::write_row_major(MatrixRM::Identity(n, n), z0, n);

  auto rhs_aug = [n, &rhs, &jacobian_fn](double t, const Vector& z, Vector& dzdt) {
    const Vector x = z.head(n);

    Vector xdot;
    rhs(t, x, xdot);

    dzdt.resize(n + n * n);
    if (xdot.size() != n || !detail::finite(xdot)) {
      dzdt.setConstant(std::numeric_limits<double>::quiet_NaN());
      return;
    }
    dzdt.head(n) = xdot;

    Matrix a;
    if (!jacobian_fn(t, x, a) || a.rows() != n || a.cols() != n || !detail::finite(a)) {
      dzdt.setConstant(std::numeric_limits<double>::quiet_NaN());
      return;
    }

    const MatrixRM phi = detail::map_row_major(z, n, n);
    const MatrixRM phi_dot = (a * phi).eval();
    detail::write_row_major(phi_dot, dzdt, n);
  };

  Observer<Vector> obs_aug = {};
  if (obs) {
    obs_aug = [n, obs = std::move(obs)](double t, const Vector& z) {
      return obs(t, z.head(n));
    };
  }

  const auto res = ode::eigen::integrate(method, rhs_aug, t0, z0, t1, opt, obs_aug);

  out.status = res.status;
  out.t = res.t;
  out.stats = res.stats;
  if (res.y.size() == n + n * n) {
    out.x = res.y.head(n);
    out.phi = detail::map_row_major(res.y, n, n);
  }
  return out;
}

template <class RHS, class JacobianFn, class ProcessNoiseFn>
[[nodiscard]] inline StateStmCovResult integrate_state_stm_cov(RKMethod method,
                                                               RHS&& rhs,
                                                               JacobianFn&& jacobian_fn,
                                                               ProcessNoiseFn&& q_fn,
                                                               double t0,
                                                               const Vector& x0,
                                                               const Matrix& p0,
                                                               double t1,
                                                               IntegratorOptions opt,
                                                               Observer<Vector> obs = {}) {
  const Eigen::Index n = x0.size();
  StateStmCovResult out{};
  out.t = t0;
  out.x = x0;
  out.phi = Matrix::Identity(n, n);
  out.p = p0;

  if (p0.rows() != n || p0.cols() != n) {
    out.status = IntegratorStatus::InvalidStepSize;
    return out;
  }

  Vector z0(n + 2 * n * n);
  z0.head(n) = x0;
  detail::write_row_major(MatrixRM::Identity(n, n), z0, n);
  detail::write_row_major(p0, z0, n + n * n);

  auto rhs_aug = [n, &rhs, &jacobian_fn, &q_fn](double t, const Vector& z, Vector& dzdt) {
    const Vector x = z.head(n);

    Vector xdot;
    rhs(t, x, xdot);

    dzdt.resize(n + 2 * n * n);
    if (xdot.size() != n || !detail::finite(xdot)) {
      dzdt.setConstant(std::numeric_limits<double>::quiet_NaN());
      return;
    }
    dzdt.head(n) = xdot;

    Matrix a;
    if (!jacobian_fn(t, x, a) || a.rows() != n || a.cols() != n || !detail::finite(a)) {
      dzdt.setConstant(std::numeric_limits<double>::quiet_NaN());
      return;
    }

    Matrix q;
    if (!q_fn(t, x, q) || q.rows() != n || q.cols() != n || !detail::finite(q)) {
      dzdt.setConstant(std::numeric_limits<double>::quiet_NaN());
      return;
    }

    const MatrixRM phi = detail::map_row_major(z, n, n);
    const MatrixRM p = detail::map_row_major(z, n + n * n, n);

    const MatrixRM phi_dot = (a * phi).eval();
    const MatrixRM p_dot = (a * p + p * a.transpose() + q).eval();

    detail::write_row_major(phi_dot, dzdt, n);
    detail::write_row_major(p_dot, dzdt, n + n * n);
  };

  Observer<Vector> obs_aug = {};
  if (obs) {
    obs_aug = [n, obs = std::move(obs)](double t, const Vector& z) {
      return obs(t, z.head(n));
    };
  }

  const auto res = ode::eigen::integrate(method, rhs_aug, t0, z0, t1, opt, obs_aug);

  out.status = res.status;
  out.t = res.t;
  out.stats = res.stats;
  if (res.y.size() == n + 2 * n * n) {
    out.x = res.y.head(n);
    out.phi = detail::map_row_major(res.y, n, n);
    out.p = detail::map_row_major(res.y, n + n * n, n);
  }
  return out;
}

[[nodiscard]] inline Matrix propagate_covariance_discrete(const Matrix& phi,
                                                          const Matrix& p0,
                                                          const Matrix& qd) {
  if (phi.rows() != phi.cols() || p0.rows() != phi.rows() || p0.cols() != phi.cols() ||
      qd.rows() != phi.rows() || qd.cols() != phi.cols()) {
    return Matrix{};
  }
  return (phi * p0 * phi.transpose() + qd).eval();
}

[[nodiscard]] inline Matrix covariance_joseph_update(const Matrix& p_prior,
                                                     const Matrix& k_gain,
                                                     const Matrix& h_mat,
                                                     const Matrix& r_meas) {
  if (p_prior.rows() != p_prior.cols() || k_gain.rows() != p_prior.rows() || h_mat.cols() != p_prior.cols() ||
      k_gain.cols() != h_mat.rows() || r_meas.rows() != r_meas.cols() || r_meas.rows() != h_mat.rows()) {
    return Matrix{};
  }
  const Matrix i = Matrix::Identity(p_prior.rows(), p_prior.cols());
  const Matrix a = (i - k_gain * h_mat).eval();
  return (a * p_prior * a.transpose() + k_gain * r_meas * k_gain.transpose()).eval();
}

[[nodiscard]] inline Matrix propagate_covariance_discrete_sqrt(const Matrix& phi,
                                                               const Matrix& s0,
                                                               const Matrix& qd) {
  if (phi.rows() != phi.cols() || s0.rows() != phi.rows() || s0.cols() != phi.cols() ||
      qd.rows() != phi.rows() || qd.cols() != phi.cols()) {
    return Matrix{};
  }
  const Matrix p0 = (s0 * s0.transpose()).eval();
  const Matrix p1 = (phi * p0 * phi.transpose() + qd).eval();
  Eigen::LLT<Matrix> llt(p1);
  if (llt.info() != Eigen::Success) {
    return Matrix{};
  }
  return llt.matrixL();
}

}  // namespace uncertainty

namespace variational {
using StateStmResult = uncertainty::StateStmResult;
using StateStmCovResult = uncertainty::StateStmCovResult;

template <class Dynamics>
[[nodiscard]] inline bool jacobian_forward_ad(Dynamics&& dynamics, double t, const Vector& x, Matrix& a_out) {
  return uncertainty::jacobian_forward_ad(std::forward<Dynamics>(dynamics), t, x, a_out);
}

template <class RHS, class JacobianFn>
[[nodiscard]] inline StateStmResult integrate_state_stm(RKMethod method,
                                                        RHS&& rhs,
                                                        JacobianFn&& jacobian_fn,
                                                        double t0,
                                                        const Vector& x0,
                                                        double t1,
                                                        IntegratorOptions opt,
                                                        Observer<Vector> obs = {}) {
  return uncertainty::integrate_state_stm(
      method, std::forward<RHS>(rhs), std::forward<JacobianFn>(jacobian_fn), t0, x0, t1, opt, obs);
}

template <class RHS, class JacobianFn, class ProcessNoiseFn>
[[nodiscard]] inline StateStmCovResult integrate_state_stm_cov(RKMethod method,
                                                               RHS&& rhs,
                                                               JacobianFn&& jacobian_fn,
                                                               ProcessNoiseFn&& q_fn,
                                                               double t0,
                                                               const Vector& x0,
                                                               const Matrix& p0,
                                                               double t1,
                                                               IntegratorOptions opt,
                                                               Observer<Vector> obs = {}) {
  return uncertainty::integrate_state_stm_cov(method,
                                              std::forward<RHS>(rhs),
                                              std::forward<JacobianFn>(jacobian_fn),
                                              std::forward<ProcessNoiseFn>(q_fn),
                                              t0,
                                              x0,
                                              p0,
                                              t1,
                                              opt,
                                              obs);
}

[[nodiscard]] inline Matrix propagate_covariance_discrete(const Matrix& phi, const Matrix& p0, const Matrix& qd) {
  return uncertainty::propagate_covariance_discrete(phi, p0, qd);
}

[[nodiscard]] inline Matrix covariance_joseph_update(const Matrix& p_prior,
                                                     const Matrix& k_gain,
                                                     const Matrix& h_mat,
                                                     const Matrix& r_meas) {
  return uncertainty::covariance_joseph_update(p_prior, k_gain, h_mat, r_meas);
}

[[nodiscard]] inline Matrix propagate_covariance_discrete_sqrt(const Matrix& phi,
                                                               const Matrix& s0,
                                                               const Matrix& qd) {
  return uncertainty::propagate_covariance_discrete_sqrt(phi, s0, qd);
}
}  // namespace variational
}  // namespace ode::eigen
