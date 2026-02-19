/**
 * @file eigen_api.hpp
 * @brief Eigen-first convenience wrappers for integration and uncertainty propagation.
 */
#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#if __has_include(<Eigen/Core>)
#include <Eigen/Core>
#else
#error "Eigen headers not found. Install Eigen >= 5 and/or enable ODE_FETCH_DEPS."
#endif

#include "ode/algebra_adapters.hpp"
#include "ode/ode.hpp"
#include "ode/uncertainty.hpp"

namespace ode::eigen {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

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

[[nodiscard]] inline ode::DynamicState ToStd(const Vector& x) {
  return ode::DynamicState(x.data(), x.data() + x.size());
}

[[nodiscard]] inline Vector ToEigen(const ode::DynamicState& x) {
  Vector out(static_cast<Eigen::Index>(x.size()));
  for (std::size_t i = 0; i < x.size(); ++i) {
    out(static_cast<Eigen::Index>(i)) = x[i];
  }
  return out;
}

[[nodiscard]] inline ode::DynamicMatrix ToStd(const Matrix& m) {
  const std::size_t r = static_cast<std::size_t>(m.rows());
  const std::size_t c = static_cast<std::size_t>(m.cols());
  ode::DynamicMatrix out(r * c, 0.0);
  for (std::size_t i = 0; i < r; ++i) {
    for (std::size_t j = 0; j < c; ++j) {
      out[i * c + j] = m(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
    }
  }
  return out;
}

[[nodiscard]] inline Matrix ToEigen(const ode::DynamicMatrix& m, std::size_t n) {
  Matrix out(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      out(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = m[i * n + j];
    }
  }
  return out;
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
  const std::size_t n = static_cast<std::size_t>(x.size());
  auto dyn_std = [&dynamics, n](double ti, const ode::DynamicState& xs, ode::DynamicState& dxs) {
    Vector xe(static_cast<Eigen::Index>(n));
    for (std::size_t i = 0; i < n; ++i) {
      xe(static_cast<Eigen::Index>(i)) = xs[i];
    }
    Vector dxe;
    dynamics(ti, xe, dxe);
    dxs.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      dxs[i] = dxe(static_cast<Eigen::Index>(i));
    }
  };

  ode::DynamicMatrix a_std;
  if (!ode::uncertainty::jacobian_forward_ad(dyn_std, t, detail::ToStd(x), a_std)) {
    return false;
  }
  a_out = detail::ToEigen(a_std, n);
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
  const std::size_t n = static_cast<std::size_t>(x0.size());

  auto rhs_std = [&rhs, n](double t, const ode::DynamicState& xs, ode::DynamicState& dxs) {
    Vector xe(static_cast<Eigen::Index>(n));
    for (std::size_t i = 0; i < n; ++i) {
      xe(static_cast<Eigen::Index>(i)) = xs[i];
    }
    Vector dxe;
    rhs(t, xe, dxe);
    dxs.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      dxs[i] = dxe(static_cast<Eigen::Index>(i));
    }
  };

  auto jac_std = [&jacobian_fn, n](double t, const ode::DynamicState& xs, ode::DynamicMatrix& a) {
    Vector xe(static_cast<Eigen::Index>(n));
    for (std::size_t i = 0; i < n; ++i) {
      xe(static_cast<Eigen::Index>(i)) = xs[i];
    }
    Matrix ae;
    if (!jacobian_fn(t, xe, ae) || ae.rows() != static_cast<Eigen::Index>(n) ||
        ae.cols() != static_cast<Eigen::Index>(n)) {
      return false;
    }
    a = detail::ToStd(ae);
    return true;
  };

  Observer<ode::DynamicState> obs_std = {};
  if (obs) {
    obs_std = [obs = std::move(obs), n](double t, const ode::DynamicState& xs) {
      Vector xe(static_cast<Eigen::Index>(n));
      for (std::size_t i = 0; i < n; ++i) {
        xe(static_cast<Eigen::Index>(i)) = xs[i];
      }
      return obs(t, xe);
    };
  }

  const auto out_std = ode::uncertainty::integrate_state_stm(
      method, rhs_std, jac_std, t0, detail::ToStd(x0), t1, opt, obs_std);

  StateStmResult out{};
  out.status = out_std.status;
  out.t = out_std.t;
  out.stats = out_std.stats;
  out.x = detail::ToEigen(out_std.x);
  out.phi = detail::ToEigen(out_std.phi, n);
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
  const std::size_t n = static_cast<std::size_t>(x0.size());

  auto rhs_std = [&rhs, n](double t, const ode::DynamicState& xs, ode::DynamicState& dxs) {
    Vector xe(static_cast<Eigen::Index>(n));
    for (std::size_t i = 0; i < n; ++i) {
      xe(static_cast<Eigen::Index>(i)) = xs[i];
    }
    Vector dxe;
    rhs(t, xe, dxe);
    dxs.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      dxs[i] = dxe(static_cast<Eigen::Index>(i));
    }
  };

  auto jac_std = [&jacobian_fn, n](double t, const ode::DynamicState& xs, ode::DynamicMatrix& a) {
    Vector xe(static_cast<Eigen::Index>(n));
    for (std::size_t i = 0; i < n; ++i) {
      xe(static_cast<Eigen::Index>(i)) = xs[i];
    }
    Matrix ae;
    if (!jacobian_fn(t, xe, ae) || ae.rows() != static_cast<Eigen::Index>(n) ||
        ae.cols() != static_cast<Eigen::Index>(n)) {
      return false;
    }
    a = detail::ToStd(ae);
    return true;
  };

  auto q_std = [&q_fn, n](double t, const ode::DynamicState& xs, ode::DynamicMatrix& q) {
    Vector xe(static_cast<Eigen::Index>(n));
    for (std::size_t i = 0; i < n; ++i) {
      xe(static_cast<Eigen::Index>(i)) = xs[i];
    }
    Matrix qe;
    if (!q_fn(t, xe, qe) || qe.rows() != static_cast<Eigen::Index>(n) ||
        qe.cols() != static_cast<Eigen::Index>(n)) {
      return false;
    }
    q = detail::ToStd(qe);
    return true;
  };

  Observer<ode::DynamicState> obs_std = {};
  if (obs) {
    obs_std = [obs = std::move(obs), n](double t, const ode::DynamicState& xs) {
      Vector xe(static_cast<Eigen::Index>(n));
      for (std::size_t i = 0; i < n; ++i) {
        xe(static_cast<Eigen::Index>(i)) = xs[i];
      }
      return obs(t, xe);
    };
  }

  const auto out_std = ode::uncertainty::integrate_state_stm_cov(method,
                                                                  rhs_std,
                                                                  jac_std,
                                                                  q_std,
                                                                  t0,
                                                                  detail::ToStd(x0),
                                                                  detail::ToStd(p0),
                                                                  t1,
                                                                  opt,
                                                                  obs_std);

  StateStmCovResult out{};
  out.status = out_std.status;
  out.t = out_std.t;
  out.stats = out_std.stats;
  out.x = detail::ToEigen(out_std.x);
  out.phi = detail::ToEigen(out_std.phi, n);
  out.p = detail::ToEigen(out_std.p, n);
  return out;
}

[[nodiscard]] inline Matrix propagate_covariance_discrete(const Matrix& phi,
                                                          const Matrix& p0,
                                                          const Matrix& qd) {
  const auto n = static_cast<std::size_t>(phi.rows());
  const auto p = ode::uncertainty::propagate_covariance_discrete(
      detail::ToStd(phi), detail::ToStd(p0), detail::ToStd(qd), n);
  return detail::ToEigen(p, n);
}

}  // namespace uncertainty
}  // namespace ode::eigen
