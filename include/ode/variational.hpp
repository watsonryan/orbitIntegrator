/**
 * @file variational.hpp
 * @brief Model-agnostic variational propagation APIs (STM and covariance).
 */
#pragma once

#include <cstddef>
#include <limits>
#include <utility>

#include "ode/multistep/adams_bashforth_moulton.hpp"
#include "ode/multistep/adams_high_order.hpp"
#include "ode/multistep/gauss_jackson8.hpp"
#include "ode/multistep/nordsieck_abm4.hpp"
#include "ode/uncertainty.hpp"

namespace ode::variational {

using State = ode::DynamicState;
using Matrix = ode::DynamicMatrix;
using StateStmResult = ode::uncertainty::StateStmResult;
using StateStmCovResult = ode::uncertainty::StateStmCovResult;
using Dual = ode::uncertainty::Dual;

template <class Dynamics>
[[nodiscard]] inline bool jacobian_forward_ad(Dynamics&& dynamics, double t, const State& x, Matrix& a_out) {
  return ode::uncertainty::jacobian_forward_ad(std::forward<Dynamics>(dynamics), t, x, a_out);
}

template <class RHS, class JacobianFn>
[[nodiscard]] inline StateStmResult integrate_state_stm(RKMethod method,
                                                        RHS&& rhs,
                                                        JacobianFn&& jacobian_fn,
                                                        double t0,
                                                        const State& x0,
                                                        double t1,
                                                        IntegratorOptions opt,
                                                        Observer<State> obs = {}) {
  return ode::uncertainty::integrate_state_stm(
      method, std::forward<RHS>(rhs), std::forward<JacobianFn>(jacobian_fn), t0, x0, t1, opt, obs);
}

template <class RHS, class JacobianFn, class ProcessNoiseFn>
[[nodiscard]] inline StateStmCovResult integrate_state_stm_cov(RKMethod method,
                                                               RHS&& rhs,
                                                               JacobianFn&& jacobian_fn,
                                                               ProcessNoiseFn&& q_fn,
                                                               double t0,
                                                               const State& x0,
                                                               const Matrix& p0,
                                                               double t1,
                                                               IntegratorOptions opt,
                                                               Observer<State> obs = {}) {
  return ode::uncertainty::integrate_state_stm_cov(method,
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

[[nodiscard]] inline Matrix propagate_covariance_discrete(const Matrix& phi,
                                                          const Matrix& p0,
                                                          const Matrix& qd,
                                                          std::size_t n) {
  return ode::uncertainty::propagate_covariance_discrete(phi, p0, qd, n);
}

[[nodiscard]] inline Matrix covariance_joseph_update(const Matrix& p_prior,
                                                     const Matrix& k_gain,
                                                     const Matrix& h_mat,
                                                     const Matrix& r_meas,
                                                     std::size_t n,
                                                     std::size_t m) {
  return ode::uncertainty::covariance_joseph_update(p_prior, k_gain, h_mat, r_meas, n, m);
}

[[nodiscard]] inline bool cholesky_lower(const Matrix& a, std::size_t n, Matrix& l_out) {
  return ode::uncertainty::cholesky_lower(a, n, l_out);
}

[[nodiscard]] inline Matrix propagate_covariance_discrete_sqrt(const Matrix& phi,
                                                               const Matrix& s0,
                                                               const Matrix& qd,
                                                               std::size_t n) {
  return ode::uncertainty::propagate_covariance_discrete_sqrt(phi, s0, qd, n);
}

[[nodiscard]] inline Matrix covariance_measurement_update_sqrt_information(const Matrix& s_prior,
                                                                           const Matrix& h_mat,
                                                                           const Matrix& s_r,
                                                                           std::size_t n,
                                                                           std::size_t m) {
  return ode::uncertainty::covariance_measurement_update_sqrt_information(s_prior, h_mat, s_r, n, m);
}

template <class RHS, class JacobianFn>
[[nodiscard]] inline StateStmResult integrate_state_stm_abm4(RHS&& rhs,
                                                             JacobianFn&& jacobian_fn,
                                                             double t0,
                                                             const State& x0,
                                                             double t1,
                                                             ode::multistep::AdamsBashforthMoultonOptions opt,
                                                             Observer<State> obs = {}) {
  StateStmResult out{};
  out.t = t0;
  out.x = x0;

  const std::size_t n = x0.size();
  out.phi.assign(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    out.phi[i * n + i] = 1.0;
  }
  if (n == 0) {
    return out;
  }

  State y0_aug(n + n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    y0_aug[i] = x0[i];
    y0_aug[n + i * n + i] = 1.0;
  }

  auto rhs_aug = [&](double t, const State& y_aug, State& dydt_aug) {
    dydt_aug.assign(n + n * n, 0.0);
    State x(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      x[i] = y_aug[i];
    }

    State dxdt;
    rhs(t, x, dxdt);
    Matrix a;
    const bool jac_ok = jacobian_fn(t, x, a);
    if (dxdt.size() != n || !jac_ok || a.size() != n * n) {
      std::fill(dydt_aug.begin(), dydt_aug.end(), std::numeric_limits<double>::quiet_NaN());
      return;
    }

    for (std::size_t i = 0; i < n; ++i) {
      dydt_aug[i] = dxdt[i];
    }
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        double sum = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
          sum += a[i * n + k] * y_aug[n + k * n + j];
        }
        dydt_aug[n + i * n + j] = sum;
      }
    }
  };

  const Observer<State> obs_aug = [&](double t, const State& y_aug) {
    if (!obs) {
      return true;
    }
    State x(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      x[i] = y_aug[i];
    }
    return obs(t, x);
  };

  const auto res = ode::multistep::integrate_abm4(rhs_aug, t0, y0_aug, t1, opt, obs_aug);
  out.status = res.status;
  out.t = res.t;
  out.stats = res.stats;
  if (res.y.size() != n + n * n) {
    out.status = IntegratorStatus::NaNDetected;
    return out;
  }
  out.x.assign(res.y.begin(), res.y.begin() + static_cast<std::ptrdiff_t>(n));
  out.phi.assign(res.y.begin() + static_cast<std::ptrdiff_t>(n), res.y.end());
  return out;
}

template <class RHS, class JacobianFn>
[[nodiscard]] inline StateStmResult integrate_state_stm_abm6(RHS&& rhs,
                                                             JacobianFn&& jacobian_fn,
                                                             double t0,
                                                             const State& x0,
                                                             double t1,
                                                             ode::multistep::AdamsBashforthMoultonOptions opt,
                                                             Observer<State> obs = {}) {
  StateStmResult out{};
  out.t = t0;
  out.x = x0;

  const std::size_t n = x0.size();
  out.phi.assign(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    out.phi[i * n + i] = 1.0;
  }
  if (n == 0) {
    return out;
  }

  State y0_aug(n + n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    y0_aug[i] = x0[i];
    y0_aug[n + i * n + i] = 1.0;
  }

  auto rhs_aug = [&](double t, const State& y_aug, State& dydt_aug) {
    dydt_aug.assign(n + n * n, 0.0);
    State x(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      x[i] = y_aug[i];
    }

    State dxdt;
    rhs(t, x, dxdt);
    Matrix a;
    const bool jac_ok = jacobian_fn(t, x, a);
    if (dxdt.size() != n || !jac_ok || a.size() != n * n) {
      std::fill(dydt_aug.begin(), dydt_aug.end(), std::numeric_limits<double>::quiet_NaN());
      return;
    }

    for (std::size_t i = 0; i < n; ++i) {
      dydt_aug[i] = dxdt[i];
    }
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        double sum = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
          sum += a[i * n + k] * y_aug[n + k * n + j];
        }
        dydt_aug[n + i * n + j] = sum;
      }
    }
  };

  const Observer<State> obs_aug = [&](double t, const State& y_aug) {
    if (!obs) {
      return true;
    }
    State x(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      x[i] = y_aug[i];
    }
    return obs(t, x);
  };

  const auto res = ode::multistep::integrate_abm6(rhs_aug, t0, y0_aug, t1, opt, obs_aug);
  out.status = res.status;
  out.t = res.t;
  out.stats = res.stats;
  if (res.y.size() != n + n * n) {
    out.status = IntegratorStatus::NaNDetected;
    return out;
  }
  out.x.assign(res.y.begin(), res.y.begin() + static_cast<std::ptrdiff_t>(n));
  out.phi.assign(res.y.begin() + static_cast<std::ptrdiff_t>(n), res.y.end());
  return out;
}

template <class RHS, class JacobianFn, class ProcessNoiseFn>
[[nodiscard]] inline StateStmCovResult integrate_state_stm_cov_nordsieck_abm4(
    RHS&& rhs,
    JacobianFn&& jacobian_fn,
    ProcessNoiseFn&& q_fn,
    double t0,
    const State& x0,
    const Matrix& p0,
    double t1,
    ode::multistep::NordsieckAbmOptions opt,
    Observer<State> obs = {}) {
  StateStmCovResult out{};
  out.t = t0;
  out.x = x0;

  const std::size_t n = x0.size();
  if (n == 0 || p0.size() != n * n) {
    out.status = IntegratorStatus::InvalidStepSize;
    return out;
  }

  out.phi.assign(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    out.phi[i * n + i] = 1.0;
  }
  out.p = p0;

  State y0_aug(n + n * n + n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    y0_aug[i] = x0[i];
    y0_aug[n + i * n + i] = 1.0;
  }
  for (std::size_t i = 0; i < n * n; ++i) {
    y0_aug[n + n * n + i] = p0[i];
  }

  auto rhs_aug = [&](double t, const State& y_aug, State& dydt_aug) {
    dydt_aug.assign(n + n * n + n * n, 0.0);
    State x(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      x[i] = y_aug[i];
    }

    State dxdt;
    rhs(t, x, dxdt);
    Matrix a;
    Matrix q;
    const bool jac_ok = jacobian_fn(t, x, a);
    const bool q_ok = q_fn(t, x, q);
    if (dxdt.size() != n || !jac_ok || !q_ok || a.size() != n * n || q.size() != n * n) {
      std::fill(dydt_aug.begin(), dydt_aug.end(), std::numeric_limits<double>::quiet_NaN());
      return;
    }

    for (std::size_t i = 0; i < n; ++i) {
      dydt_aug[i] = dxdt[i];
    }

    const std::size_t phi_off = n;
    const std::size_t p_off = n + n * n;

    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        double phi_dot = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
          phi_dot += a[i * n + k] * y_aug[phi_off + k * n + j];
        }
        dydt_aug[phi_off + i * n + j] = phi_dot;
      }
    }

    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        double ap = 0.0;
        double pat = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
          ap += a[i * n + k] * y_aug[p_off + k * n + j];
          pat += y_aug[p_off + i * n + k] * a[j * n + k];
        }
        dydt_aug[p_off + i * n + j] = ap + pat + q[i * n + j];
      }
    }
  };

  const Observer<State> obs_aug = [&](double t, const State& y_aug) {
    if (!obs) {
      return true;
    }
    State x(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      x[i] = y_aug[i];
    }
    return obs(t, x);
  };

  const auto res = ode::multistep::integrate_nordsieck_abm4(rhs_aug, t0, y0_aug, t1, opt, obs_aug);
  out.status = res.status;
  out.t = res.t;
  out.stats = res.stats;
  if (res.y.size() != y0_aug.size()) {
    out.status = IntegratorStatus::NaNDetected;
    return out;
  }
  out.x.assign(res.y.begin(), res.y.begin() + static_cast<std::ptrdiff_t>(n));
  out.phi.assign(res.y.begin() + static_cast<std::ptrdiff_t>(n),
                 res.y.begin() + static_cast<std::ptrdiff_t>(n + n * n));
  out.p.assign(res.y.begin() + static_cast<std::ptrdiff_t>(n + n * n), res.y.end());
  return out;
}

template <class RHS, class JacobianFn, class ProcessNoiseFn>
[[nodiscard]] inline StateStmCovResult integrate_state_stm_cov_nordsieck_abm6(
    RHS&& rhs,
    JacobianFn&& jacobian_fn,
    ProcessNoiseFn&& q_fn,
    double t0,
    const State& x0,
    const Matrix& p0,
    double t1,
    ode::multistep::NordsieckAbmOptions opt,
    Observer<State> obs = {}) {
  StateStmCovResult out{};
  out.t = t0;
  out.x = x0;

  const std::size_t n = x0.size();
  if (n == 0 || p0.size() != n * n) {
    out.status = IntegratorStatus::InvalidStepSize;
    return out;
  }

  out.phi.assign(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    out.phi[i * n + i] = 1.0;
  }
  out.p = p0;

  State y0_aug(n + n * n + n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    y0_aug[i] = x0[i];
    y0_aug[n + i * n + i] = 1.0;
  }
  for (std::size_t i = 0; i < n * n; ++i) {
    y0_aug[n + n * n + i] = p0[i];
  }

  auto rhs_aug = [&](double t, const State& y_aug, State& dydt_aug) {
    dydt_aug.assign(n + n * n + n * n, 0.0);
    State x(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      x[i] = y_aug[i];
    }

    State dxdt;
    rhs(t, x, dxdt);
    Matrix a;
    Matrix q;
    const bool jac_ok = jacobian_fn(t, x, a);
    const bool q_ok = q_fn(t, x, q);
    if (dxdt.size() != n || !jac_ok || !q_ok || a.size() != n * n || q.size() != n * n) {
      std::fill(dydt_aug.begin(), dydt_aug.end(), std::numeric_limits<double>::quiet_NaN());
      return;
    }

    for (std::size_t i = 0; i < n; ++i) {
      dydt_aug[i] = dxdt[i];
    }

    const std::size_t phi_off = n;
    const std::size_t p_off = n + n * n;

    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        double phi_dot = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
          phi_dot += a[i * n + k] * y_aug[phi_off + k * n + j];
        }
        dydt_aug[phi_off + i * n + j] = phi_dot;
      }
    }

    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        double ap = 0.0;
        double pat = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
          ap += a[i * n + k] * y_aug[p_off + k * n + j];
          pat += y_aug[p_off + i * n + k] * a[j * n + k];
        }
        dydt_aug[p_off + i * n + j] = ap + pat + q[i * n + j];
      }
    }
  };

  const Observer<State> obs_aug = [&](double t, const State& y_aug) {
    if (!obs) {
      return true;
    }
    State x(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      x[i] = y_aug[i];
    }
    return obs(t, x);
  };

  const auto res = ode::multistep::integrate_nordsieck_abm6(rhs_aug, t0, y0_aug, t1, opt, obs_aug);
  out.status = res.status;
  out.t = res.t;
  out.stats = res.stats;
  if (res.y.size() != y0_aug.size()) {
    out.status = IntegratorStatus::NaNDetected;
    return out;
  }
  out.x.assign(res.y.begin(), res.y.begin() + static_cast<std::ptrdiff_t>(n));
  out.phi.assign(res.y.begin() + static_cast<std::ptrdiff_t>(n),
                 res.y.begin() + static_cast<std::ptrdiff_t>(n + n * n));
  out.p.assign(res.y.begin() + static_cast<std::ptrdiff_t>(n + n * n), res.y.end());
  return out;
}

template <class AccelerationFn, class AccelJacobianFn>
[[nodiscard]] inline StateStmResult integrate_state_stm_gauss_jackson8(
    AccelerationFn&& acceleration,
    AccelJacobianFn&& accel_jacobian_fn,
    double t0,
    const State& x0,
    double t1,
    ode::multistep::GaussJackson8Options opt) {
  StateStmResult out{};
  out.t = t0;
  out.x = x0;

  if (x0.empty() || (x0.size() % 2) != 0) {
    out.status = IntegratorStatus::InvalidStepSize;
    return out;
  }

  const std::size_t n = x0.size() / 2;
  const std::size_t stm_dim = 2 * n;
  out.phi.assign(stm_dim * stm_dim, 0.0);
  for (std::size_t i = 0; i < stm_dim; ++i) {
    out.phi[i * stm_dim + i] = 1.0;
  }

  State r0(n, 0.0), v0(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    r0[i] = x0[i];
    v0[i] = x0[n + i];
  }

  const std::size_t cols = 2 * n;
  const std::size_t ext_n = n * (1 + cols);
  State r_ext(ext_n, 0.0), v_ext(ext_n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    r_ext[i] = r0[i];
    v_ext[i] = v0[i];
  }
  for (std::size_t col = 0; col < cols; ++col) {
    const std::size_t base = n + col * n;
    if (col < n) {
      r_ext[base + col] = 1.0;
    } else {
      v_ext[base + (col - n)] = 1.0;
    }
  }

  auto accel_ext = [&](double t, const State& r, const State& v, State& a) {
    a.assign(ext_n, 0.0);
    State r_nom(n, 0.0), v_nom(n, 0.0), a_nom;
    for (std::size_t i = 0; i < n; ++i) {
      r_nom[i] = r[i];
      v_nom[i] = v[i];
    }

    acceleration(t, r_nom, v_nom, a_nom);
    Matrix ar;
    Matrix av;
    const bool jac_ok = accel_jacobian_fn(t, r_nom, v_nom, ar, av);
    if (a_nom.size() != n || !jac_ok || ar.size() != n * n || av.size() != n * n) {
      std::fill(a.begin(), a.end(), std::numeric_limits<double>::quiet_NaN());
      return;
    }

    for (std::size_t i = 0; i < n; ++i) {
      a[i] = a_nom[i];
    }

    for (std::size_t col = 0; col < cols; ++col) {
      const std::size_t base = n + col * n;
      for (std::size_t i = 0; i < n; ++i) {
        double da = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
          da += ar[i * n + j] * r[base + j] + av[i * n + j] * v[base + j];
        }
        a[base + i] = da;
      }
    }
  };

  const auto res = ode::multistep::integrate_gauss_jackson8(accel_ext, t0, r_ext, v_ext, t1, opt);
  out.status = res.status;
  out.t = res.t;
  out.stats = res.stats;
  if (res.r.size() != ext_n || res.v.size() != ext_n) {
    out.status = IntegratorStatus::NaNDetected;
    return out;
  }

  out.x.assign(2 * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    out.x[i] = res.r[i];
    out.x[n + i] = res.v[i];
  }
  out.phi.assign(stm_dim * stm_dim, 0.0);
  for (std::size_t col = 0; col < cols; ++col) {
    const std::size_t base = n + col * n;
    for (std::size_t i = 0; i < n; ++i) {
      out.phi[i * stm_dim + col] = res.r[base + i];
      out.phi[(n + i) * stm_dim + col] = res.v[base + i];
    }
  }
  return out;
}

}  // namespace ode::variational
