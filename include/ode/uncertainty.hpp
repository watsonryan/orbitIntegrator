/**
 * @file uncertainty.hpp
 * @brief State transition matrix and covariance propagation utilities.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "ode/integrate_method.hpp"
#include "ode/tableaus/rk4.hpp"
#include "ode/tableaus/rkf45.hpp"
#include "ode/tableaus/rkf78.hpp"
#include "ode/types.hpp"

namespace ode {

using DynamicState = std::vector<double>;
using DynamicMatrix = std::vector<double>;  // Row-major n x n.

namespace uncertainty {

namespace detail {

[[nodiscard]] inline std::size_t idx(std::size_t n, std::size_t row, std::size_t col) {
  return row * n + col;
}

[[nodiscard]] inline bool finite_matrix(const DynamicMatrix& m) {
  for (double v : m) {
    if (!std::isfinite(v)) {
      return false;
    }
  }
  return true;
}

inline void set_identity(DynamicMatrix& phi, std::size_t n) {
  phi.assign(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    phi[idx(n, i, i)] = 1.0;
  }
}

inline void matmul_nn(const DynamicMatrix& a, const DynamicMatrix& b, std::size_t n, DynamicMatrix& out) {
  out.assign(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t k = 0; k < n; ++k) {
      const double aik = a[idx(n, i, k)];
      if (aik == 0.0) {
        continue;
      }
      for (std::size_t j = 0; j < n; ++j) {
        out[idx(n, i, j)] += aik * b[idx(n, k, j)];
      }
    }
  }
}

inline void matmul_nt(const DynamicMatrix& a, const DynamicMatrix& b, std::size_t n, DynamicMatrix& out) {
  out.assign(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t k = 0; k < n; ++k) {
      const double aik = a[idx(n, i, k)];
      if (aik == 0.0) {
        continue;
      }
      for (std::size_t j = 0; j < n; ++j) {
        out[idx(n, i, j)] += aik * b[idx(n, j, k)];
      }
    }
  }
}

inline void transpose(const DynamicMatrix& in, std::size_t n, DynamicMatrix& out) {
  out.resize(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      out[idx(n, j, i)] = in[idx(n, i, j)];
    }
  }
}

template <class RHS>
[[nodiscard]] inline IntegratorResult<DynamicState> integrate_runtime(RKMethod method,
                                                                      RHS&& rhs,
                                                                      double t0,
                                                                      const DynamicState& y0,
                                                                      double t1,
                                                                      IntegratorOptions opt,
                                                                      Observer<DynamicState> obs = {}) {
  switch (method) {
    case RKMethod::RK4:
      opt.adaptive = false;
      return integrate_with_tableau<TableauRK4, DynamicState, RHS>(std::forward<RHS>(rhs), t0, y0, t1, opt, obs);
    case RKMethod::RKF45:
      return integrate_with_tableau<TableauRKF45, DynamicState, RHS>(std::forward<RHS>(rhs), t0, y0, t1, opt, obs);
    case RKMethod::RK8:
      opt.adaptive = false;
      return integrate_with_tableau<TableauRKF78, DynamicState, RHS>(std::forward<RHS>(rhs), t0, y0, t1, opt, obs);
    case RKMethod::RKF78:
      return integrate_with_tableau<TableauRKF78, DynamicState, RHS>(std::forward<RHS>(rhs), t0, y0, t1, opt, obs);
  }
  IntegratorResult<DynamicState> fallback{};
  fallback.status = IntegratorStatus::InvalidStepSize;
  return fallback;
}

}  // namespace detail

/**
 * @brief Result payload for state and STM propagation.
 */
struct StateStmResult {
  IntegratorStatus status = IntegratorStatus::Success;
  double t = 0.0;
  DynamicState x{};
  DynamicMatrix phi{};
  IntegratorStats stats{};
};

/**
 * @brief Result payload for state, STM, and covariance propagation.
 */
struct StateStmCovResult {
  IntegratorStatus status = IntegratorStatus::Success;
  double t = 0.0;
  DynamicState x{};
  DynamicMatrix phi{};
  DynamicMatrix p{};
  IntegratorStats stats{};
};

/**
 * @brief Minimal dynamic forward-mode dual number for Jacobian extraction.
 */
struct Dual {
  double val = 0.0;
  DynamicState d{};

  Dual() = default;
  explicit Dual(double v, std::size_t n = 0) : val(v), d(n, 0.0) {}
};

[[nodiscard]] inline Dual operator+(const Dual& a, const Dual& b) {
  Dual out(a.val + b.val, a.d.size());
  for (std::size_t i = 0; i < out.d.size(); ++i) {
    out.d[i] = a.d[i] + b.d[i];
  }
  return out;
}

[[nodiscard]] inline Dual operator-(const Dual& a, const Dual& b) {
  Dual out(a.val - b.val, a.d.size());
  for (std::size_t i = 0; i < out.d.size(); ++i) {
    out.d[i] = a.d[i] - b.d[i];
  }
  return out;
}

[[nodiscard]] inline Dual operator*(const Dual& a, const Dual& b) {
  Dual out(a.val * b.val, a.d.size());
  for (std::size_t i = 0; i < out.d.size(); ++i) {
    out.d[i] = a.d[i] * b.val + a.val * b.d[i];
  }
  return out;
}

[[nodiscard]] inline Dual operator/(const Dual& a, const Dual& b) {
  const double inv = 1.0 / b.val;
  Dual out(a.val * inv, a.d.size());
  for (std::size_t i = 0; i < out.d.size(); ++i) {
    out.d[i] = (a.d[i] - out.val * b.d[i]) * inv;
  }
  return out;
}

[[nodiscard]] inline Dual operator+(const Dual& a, double b) {
  Dual out = a;
  out.val += b;
  return out;
}

[[nodiscard]] inline Dual operator+(double a, const Dual& b) {
  return b + a;
}

[[nodiscard]] inline Dual operator-(const Dual& a, double b) {
  Dual out = a;
  out.val -= b;
  return out;
}

[[nodiscard]] inline Dual operator-(double a, const Dual& b) {
  Dual out(a - b.val, b.d.size());
  for (std::size_t i = 0; i < out.d.size(); ++i) {
    out.d[i] = -b.d[i];
  }
  return out;
}

[[nodiscard]] inline Dual operator*(const Dual& a, double b) {
  Dual out(a.val * b, a.d.size());
  for (std::size_t i = 0; i < out.d.size(); ++i) {
    out.d[i] = a.d[i] * b;
  }
  return out;
}

[[nodiscard]] inline Dual operator*(double a, const Dual& b) {
  return b * a;
}

[[nodiscard]] inline Dual operator/(const Dual& a, double b) {
  const double inv = 1.0 / b;
  Dual out(a.val * inv, a.d.size());
  for (std::size_t i = 0; i < out.d.size(); ++i) {
    out.d[i] = a.d[i] * inv;
  }
  return out;
}

[[nodiscard]] inline Dual operator/(double a, const Dual& b) {
  const double inv = 1.0 / b.val;
  Dual out(a * inv, b.d.size());
  for (std::size_t i = 0; i < out.d.size(); ++i) {
    out.d[i] = -a * b.d[i] * inv * inv;
  }
  return out;
}

[[nodiscard]] inline Dual operator-(const Dual& a) {
  Dual out(-a.val, a.d.size());
  for (std::size_t i = 0; i < out.d.size(); ++i) {
    out.d[i] = -a.d[i];
  }
  return out;
}

[[nodiscard]] inline Dual exp(const Dual& x) {
  const double ex = std::exp(x.val);
  Dual out(ex, x.d.size());
  for (std::size_t i = 0; i < out.d.size(); ++i) {
    out.d[i] = ex * x.d[i];
  }
  return out;
}

[[nodiscard]] inline Dual log(const Dual& x) {
  Dual out(std::log(x.val), x.d.size());
  const double inv = 1.0 / x.val;
  for (std::size_t i = 0; i < out.d.size(); ++i) {
    out.d[i] = x.d[i] * inv;
  }
  return out;
}

[[nodiscard]] inline Dual sin(const Dual& x) {
  Dual out(std::sin(x.val), x.d.size());
  const double c = std::cos(x.val);
  for (std::size_t i = 0; i < out.d.size(); ++i) {
    out.d[i] = c * x.d[i];
  }
  return out;
}

[[nodiscard]] inline Dual cos(const Dual& x) {
  Dual out(std::cos(x.val), x.d.size());
  const double s = -std::sin(x.val);
  for (std::size_t i = 0; i < out.d.size(); ++i) {
    out.d[i] = s * x.d[i];
  }
  return out;
}

[[nodiscard]] inline Dual sqrt(const Dual& x) {
  const double s = std::sqrt(x.val);
  Dual out(s, x.d.size());
  const double scale = 0.5 / s;
  for (std::size_t i = 0; i < out.d.size(); ++i) {
    out.d[i] = x.d[i] * scale;
  }
  return out;
}

[[nodiscard]] inline Dual pow(const Dual& x, double p) {
  const double vp = std::pow(x.val, p);
  Dual out(vp, x.d.size());
  const double scale = p * std::pow(x.val, p - 1.0);
  for (std::size_t i = 0; i < out.d.size(); ++i) {
    out.d[i] = scale * x.d[i];
  }
  return out;
}

/**
 * @brief Compute dense Jacobian A = df/dx with forward-mode automatic differentiation.
 *
 * @tparam Dynamics Callable with signature:
 *         `void(double t, const std::vector<Scalar>& x, std::vector<Scalar>& dxdt)`
 */
template <class Dynamics>
[[nodiscard]] bool jacobian_forward_ad(Dynamics&& dynamics,
                                       double t,
                                       const DynamicState& x,
                                       DynamicMatrix& a_out) {
  const std::size_t n = x.size();
  a_out.assign(n * n, 0.0);
  if (n == 0) {
    return true;
  }

  std::vector<Dual> x_ad(n);
  for (std::size_t i = 0; i < n; ++i) {
    x_ad[i] = Dual(x[i], n);
    x_ad[i].d[i] = 1.0;
  }

  std::vector<Dual> f_ad;
  dynamics(t, x_ad, f_ad);
  if (f_ad.size() != n) {
    return false;
  }

  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(f_ad[i].val) || f_ad[i].d.size() != n) {
      return false;
    }
    for (double d : f_ad[i].d) {
      if (!std::isfinite(d)) {
        return false;
      }
    }
    for (std::size_t j = 0; j < n; ++j) {
      a_out[detail::idx(n, i, j)] = f_ad[i].d[j];
    }
  }
  return true;
}

/**
 * @brief Propagate state and STM using `dPhi/dt = A(x,t) * Phi`.
 *
 * @tparam RHS Callable: `void(double, const DynamicState&, DynamicState&)`
 * @tparam JacobianFn Callable: `bool(double, const DynamicState&, DynamicMatrix&)`
 */
template <class RHS, class JacobianFn>
[[nodiscard]] StateStmResult integrate_state_stm(RKMethod method,
                                                 RHS&& rhs,
                                                 JacobianFn&& jacobian_fn,
                                                 double t0,
                                                 const DynamicState& x0,
                                                 double t1,
                                                 IntegratorOptions opt,
                                                 Observer<DynamicState> obs = {}) {
  const std::size_t n = x0.size();
  StateStmResult out{};
  out.t = t0;
  out.x = x0;
  detail::set_identity(out.phi, n);

  DynamicState z0(n + n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    z0[i] = x0[i];
  }
  for (std::size_t i = 0; i < n * n; ++i) {
    z0[n + i] = out.phi[i];
  }

  auto rhs_aug = [n, &rhs, &jacobian_fn](double t, const DynamicState& z, DynamicState& dzdt) {
    DynamicState x(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      x[i] = z[i];
    }

    DynamicState xdot;
    rhs(t, x, xdot);
    dzdt.assign(n + n * n, 0.0);
    if (xdot.size() != n) {
      std::fill(dzdt.begin(), dzdt.end(), std::numeric_limits<double>::quiet_NaN());
      return;
    }
    for (std::size_t i = 0; i < n; ++i) {
      dzdt[i] = xdot[i];
    }

    DynamicMatrix a;
    if (!jacobian_fn(t, x, a) || a.size() != n * n || !detail::finite_matrix(a)) {
      std::fill(dzdt.begin(), dzdt.end(), std::numeric_limits<double>::quiet_NaN());
      return;
    }

    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        double sum = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
          const double aik = a[detail::idx(n, i, k)];
          const double phi_kj = z[n + detail::idx(n, k, j)];
          sum += aik * phi_kj;
        }
        dzdt[n + detail::idx(n, i, j)] = sum;
      }
    }
  };

  auto obs_aug = [n, &obs](double t, const DynamicState& z) -> bool {
    if (!obs) {
      return true;
    }
    DynamicState x(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      x[i] = z[i];
    }
    return obs(t, x);
  };

  const auto res = detail::integrate_runtime(method, rhs_aug, t0, z0, t1, opt, obs_aug);
  out.status = res.status;
  out.t = res.t;
  out.stats = res.stats;
  if (res.y.size() == n + n * n) {
    const auto n_off = static_cast<std::ptrdiff_t>(n);
    out.x.assign(res.y.begin(), res.y.begin() + n_off);
    out.phi.assign(res.y.begin() + n_off, res.y.end());
  }
  return out;
}

/**
 * @brief Propagate covariance one step with the discrete-time update.
 *
 * Computes `P1 = Phi * P0 * Phi^T + Qd`.
 */
[[nodiscard]] inline DynamicMatrix propagate_covariance_discrete(const DynamicMatrix& phi,
                                                                 const DynamicMatrix& p0,
                                                                 const DynamicMatrix& qd,
                                                                 std::size_t n) {
  DynamicMatrix tmp;
  DynamicMatrix out;
  detail::matmul_nn(phi, p0, n, tmp);
  detail::matmul_nt(tmp, phi, n, out);
  if (qd.size() == n * n) {
    for (std::size_t i = 0; i < n * n; ++i) {
      out[i] += qd[i];
    }
  }
  return out;
}

/**
 * @brief Joseph-form covariance measurement update.
 *
 * Computes:
 * `P = (I - K H) P_prior (I - K H)^T + K R K^T`
 *
 * Shapes:
 * - `P_prior`: n x n
 * - `K`: n x m
 * - `H`: m x n
 * - `R`: m x m
 */
[[nodiscard]] inline DynamicMatrix covariance_joseph_update(const DynamicMatrix& p_prior,
                                                            const DynamicMatrix& k_gain,
                                                            const DynamicMatrix& h_mat,
                                                            const DynamicMatrix& r_meas,
                                                            std::size_t n,
                                                            std::size_t m) {
  if (p_prior.size() != n * n || k_gain.size() != n * m || h_mat.size() != m * n || r_meas.size() != m * m) {
    return {};
  }

  auto idx_nm = [](std::size_t rows, std::size_t row, std::size_t col) {
    return row * rows + col;
  };
  auto idx_rect = [](std::size_t cols, std::size_t row, std::size_t col) {
    return row * cols + col;
  };

  DynamicMatrix kh(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      double s = 0.0;
      for (std::size_t k = 0; k < m; ++k) {
        s += k_gain[idx_rect(m, i, k)] * h_mat[idx_rect(n, k, j)];
      }
      kh[idx_nm(n, i, j)] = s;
    }
  }

  DynamicMatrix i_m_kh(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      i_m_kh[idx_nm(n, i, j)] = (i == j ? 1.0 : 0.0) - kh[idx_nm(n, i, j)];
    }
  }

  DynamicMatrix tmp;
  DynamicMatrix a_p_at;
  detail::matmul_nn(i_m_kh, p_prior, n, tmp);
  detail::matmul_nt(tmp, i_m_kh, n, a_p_at);

  DynamicMatrix kr(n * m, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < m; ++j) {
      double s = 0.0;
      for (std::size_t k = 0; k < m; ++k) {
        s += k_gain[idx_rect(m, i, k)] * r_meas[idx_rect(m, k, j)];
      }
      kr[idx_rect(m, i, j)] = s;
    }
  }

  DynamicMatrix k_r_kt(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      double s = 0.0;
      for (std::size_t k = 0; k < m; ++k) {
        s += kr[idx_rect(m, i, k)] * k_gain[idx_rect(m, j, k)];
      }
      k_r_kt[idx_nm(n, i, j)] = s;
    }
  }

  DynamicMatrix out(n * n, 0.0);
  for (std::size_t i = 0; i < n * n; ++i) {
    out[i] = a_p_at[i] + k_r_kt[i];
  }
  return out;
}

/**
 * @brief Compute lower-triangular Cholesky factor `L` where `A = L L^T`.
 */
[[nodiscard]] inline bool cholesky_lower(const DynamicMatrix& a, std::size_t n, DynamicMatrix& l_out) {
  if (a.size() != n * n) {
    return false;
  }
  for (double v : a) {
    if (!std::isfinite(v)) {
      return false;
    }
  }

  const auto id = [n](std::size_t r, std::size_t c) { return r * n + c; };
  DynamicMatrix l(n * n, 0.0);

  double jitter = 0.0;
  for (int attempt = 0; attempt < 6; ++attempt) {
    std::fill(l.begin(), l.end(), 0.0);
    bool ok = true;
    for (std::size_t i = 0; i < n && ok; ++i) {
      for (std::size_t j = 0; j <= i; ++j) {
        double s = a[id(i, j)];
        if (i == j) {
          s += jitter;
        }
        for (std::size_t k = 0; k < j; ++k) {
          s -= l[id(i, k)] * l[id(j, k)];
        }
        if (i == j) {
          if (!(s > 0.0) || !std::isfinite(s)) {
            ok = false;
            break;
          }
          l[id(i, j)] = std::sqrt(s);
        } else {
          const double ljj = l[id(j, j)];
          if (!(ljj > 0.0) || !std::isfinite(ljj)) {
            ok = false;
            break;
          }
          l[id(i, j)] = s / ljj;
          if (!std::isfinite(l[id(i, j)])) {
            ok = false;
            break;
          }
        }
      }
    }
    if (ok) {
      l_out = std::move(l);
      return true;
    }
    jitter = (jitter == 0.0) ? 1e-15 : jitter * 10.0;
  }
  return false;
}

/**
 * @brief Discrete covariance propagation in square-root form.
 *
 * Returns lower-triangular `S1` such that:
 * `S1 * S1^T = Phi * (S0*S0^T) * Phi^T + Qd`
 */
[[nodiscard]] inline DynamicMatrix propagate_covariance_discrete_sqrt(const DynamicMatrix& phi,
                                                                      const DynamicMatrix& s0,
                                                                      const DynamicMatrix& qd,
                                                                      std::size_t n) {
  if (phi.size() != n * n || s0.size() != n * n || qd.size() != n * n) {
    return {};
  }

  DynamicMatrix p0;
  detail::matmul_nt(s0, s0, n, p0);
  DynamicMatrix p1 = propagate_covariance_discrete(phi, p0, qd, n);

  DynamicMatrix s1;
  if (!cholesky_lower(p1, n, s1)) {
    return {};
  }
  return s1;
}

/**
 * @brief Propagate state, STM, and covariance with continuous-time Riccati form.
 *
 * Integrates:
 * - `dx/dt = f(t, x)`
 * - `dPhi/dt = A(x,t) * Phi`
 * - `dP/dt = A*P + P*A^T + Q`
 *
 * @tparam RHS Callable: `void(double, const DynamicState&, DynamicState&)`
 * @tparam JacobianFn Callable: `bool(double, const DynamicState&, DynamicMatrix&)`
 * @tparam ProcessNoiseFn Callable: `bool(double, const DynamicState&, DynamicMatrix&)`
 */
template <class RHS, class JacobianFn, class ProcessNoiseFn>
[[nodiscard]] StateStmCovResult integrate_state_stm_cov(RKMethod method,
                                                        RHS&& rhs,
                                                        JacobianFn&& jacobian_fn,
                                                        ProcessNoiseFn&& q_fn,
                                                        double t0,
                                                        const DynamicState& x0,
                                                        const DynamicMatrix& p0,
                                                        double t1,
                                                        IntegratorOptions opt,
                                                        Observer<DynamicState> obs = {}) {
  const std::size_t n = x0.size();
  StateStmCovResult out{};
  out.t = t0;
  out.x = x0;
  detail::set_identity(out.phi, n);
  out.p = p0;

  if (p0.size() != n * n) {
    out.status = IntegratorStatus::InvalidStepSize;
    return out;
  }

  DynamicState z0(n + n * n + n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    z0[i] = x0[i];
  }
  for (std::size_t i = 0; i < n * n; ++i) {
    z0[n + i] = out.phi[i];
    z0[n + n * n + i] = p0[i];
  }

  auto rhs_aug = [n, &rhs, &jacobian_fn, &q_fn](double t, const DynamicState& z, DynamicState& dzdt) {
    DynamicState x(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      x[i] = z[i];
    }

    DynamicState xdot;
    rhs(t, x, xdot);
    dzdt.assign(n + n * n + n * n, 0.0);
    if (xdot.size() != n) {
      std::fill(dzdt.begin(), dzdt.end(), std::numeric_limits<double>::quiet_NaN());
      return;
    }
    for (std::size_t i = 0; i < n; ++i) {
      dzdt[i] = xdot[i];
    }

    DynamicMatrix a;
    if (!jacobian_fn(t, x, a) || a.size() != n * n || !detail::finite_matrix(a)) {
      std::fill(dzdt.begin(), dzdt.end(), std::numeric_limits<double>::quiet_NaN());
      return;
    }

    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        double sum = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
          const double aik = a[detail::idx(n, i, k)];
          const double phi_kj = z[n + detail::idx(n, k, j)];
          sum += aik * phi_kj;
        }
        dzdt[n + detail::idx(n, i, j)] = sum;
      }
    }

    DynamicMatrix q;
    if (!q_fn(t, x, q) || q.size() != n * n || !detail::finite_matrix(q)) {
      std::fill(dzdt.begin(), dzdt.end(), std::numeric_limits<double>::quiet_NaN());
      return;
    }

    const std::size_t p_offset = n + n * n;
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        double ap = 0.0;
        double pat = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
          const double pik = z[p_offset + detail::idx(n, i, k)];
          const double pkj = z[p_offset + detail::idx(n, k, j)];
          ap += a[detail::idx(n, i, k)] * pkj;
          pat += pik * a[detail::idx(n, j, k)];
        }
        dzdt[p_offset + detail::idx(n, i, j)] = ap + pat + q[detail::idx(n, i, j)];
      }
    }
  };

  auto obs_aug = [n, &obs](double t, const DynamicState& z) -> bool {
    if (!obs) {
      return true;
    }
    DynamicState x(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      x[i] = z[i];
    }
    return obs(t, x);
  };

  const auto res = detail::integrate_runtime(method, rhs_aug, t0, z0, t1, opt, obs_aug);
  out.status = res.status;
  out.t = res.t;
  out.stats = res.stats;
  if (res.y.size() == n + n * n + n * n) {
    const auto n_off = static_cast<std::ptrdiff_t>(n);
    const auto n2_off = static_cast<std::ptrdiff_t>(n + n * n);
    out.x.assign(res.y.begin(), res.y.begin() + n_off);
    out.phi.assign(res.y.begin() + n_off, res.y.begin() + n2_off);
    out.p.assign(res.y.begin() + n2_off, res.y.end());
  }
  return out;
}

}  // namespace uncertainty
}  // namespace ode
