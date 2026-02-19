/**
 * @file rk_stepper.hpp
 * @brief Generic explicit Runge-Kutta single-step engine driven by a tableau.
 */
#pragma once

#include <array>

#include "ode/algebra.hpp"

namespace ode {

template <class Tableau, class State, class Algebra = DefaultAlgebra<State>>
requires AlgebraFor<Algebra, State>
class ExplicitRKStepper {
 public:
  /** @brief Construct a stepper and pre-size all stage buffers from a template state. */
  explicit ExplicitRKStepper(const State& y_template) {
    for (auto& stage : k_) {
      Algebra::resize_like(stage, y_template);
      Algebra::set_zero(stage);
    }
    Algebra::resize_like(y_tmp_, y_template);
    Algebra::resize_like(y_low_, y_template);
    Algebra::set_zero(y_tmp_);
    Algebra::set_zero(y_low_);
  }

  template <class RHS>
  /**
   * @brief Execute one RK trial step and optionally compute embedded error estimate.
   * @return true on success; false on RHS/state-size/finite-value failures.
   */
  [[nodiscard]] bool step(RHS&& rhs,
                          double t,
                          const State& y,
                          double h,
                          State& y_high,
                          State* err_out) {
    const auto n = Algebra::size(y);
    constexpr bool kHasContiguous =
        requires(const State& c, State& m) {
          c.data();
          m.data();
        };

    for (int i = 0; i < Tableau::stages; ++i) {
      if constexpr (kHasContiguous) {
        const auto* yp = y.data();
        auto* ytmp = y_tmp_.data();
        for (std::size_t q = 0; q < n; ++q) {
          ytmp[q] = yp[q];
        }
        for (int j = 0; j < i; ++j) {
          const double aij = Tableau::a[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
          if (aij == 0.0) {
            continue;
          }
          const double s = h * aij;
          const auto* kj = k_[static_cast<std::size_t>(j)].data();
          for (std::size_t q = 0; q < n; ++q) {
            ytmp[q] += s * kj[q];
          }
        }
      } else {
        Algebra::assign(y_tmp_, y);
        for (int j = 0; j < i; ++j) {
          const double aij = Tableau::a[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
          if (aij != 0.0) {
            Algebra::axpy(h * aij, k_[static_cast<std::size_t>(j)], y_tmp_);
          }
        }
      }

      rhs(t + Tableau::c[static_cast<std::size_t>(i)] * h, y_tmp_, k_[static_cast<std::size_t>(i)]);

      if (Algebra::size(k_[static_cast<std::size_t>(i)]) != n) {
        return false;
      }
      if (!Algebra::finite(k_[static_cast<std::size_t>(i)])) {
        return false;
      }
    }

    if constexpr (kHasContiguous) {
      const auto* yp = y.data();
      auto* yhp = y_high.data();
      for (std::size_t q = 0; q < n; ++q) {
        yhp[q] = yp[q];
      }
      for (int i = 0; i < Tableau::stages; ++i) {
        const double bi = Tableau::b_high[static_cast<std::size_t>(i)];
        if (bi == 0.0) {
          continue;
        }
        const double s = h * bi;
        const auto* ki = k_[static_cast<std::size_t>(i)].data();
        for (std::size_t q = 0; q < n; ++q) {
          yhp[q] += s * ki[q];
        }
      }
    } else {
      Algebra::assign(y_high, y);
      for (int i = 0; i < Tableau::stages; ++i) {
        const double bi = Tableau::b_high[static_cast<std::size_t>(i)];
        if (bi != 0.0) {
          Algebra::axpy(h * bi, k_[static_cast<std::size_t>(i)], y_high);
        }
      }
    }

    if (!Algebra::finite(y_high)) {
      return false;
    }

    if (err_out != nullptr) {
      if constexpr (Tableau::has_embedded) {
        if constexpr (kHasContiguous) {
          const auto* yp = y.data();
          auto* ylp = y_low_.data();
          for (std::size_t q = 0; q < n; ++q) {
            ylp[q] = yp[q];
          }
          for (int i = 0; i < Tableau::stages; ++i) {
            const double bi = Tableau::b_low[static_cast<std::size_t>(i)];
            if (bi == 0.0) {
              continue;
            }
            const double s = h * bi;
            const auto* ki = k_[static_cast<std::size_t>(i)].data();
            for (std::size_t q = 0; q < n; ++q) {
              ylp[q] += s * ki[q];
            }
          }
        } else {
          Algebra::assign(y_low_, y);
          for (int i = 0; i < Tableau::stages; ++i) {
            const double bi = Tableau::b_low[static_cast<std::size_t>(i)];
            if (bi != 0.0) {
              Algebra::axpy(h * bi, k_[static_cast<std::size_t>(i)], y_low_);
            }
          }
        }

        if constexpr (kHasContiguous) {
          const auto* yhp = y_high.data();
          const auto* ylp = y_low_.data();
          auto* ep = err_out->data();
          for (std::size_t q = 0; q < n; ++q) {
            ep[q] = yhp[q] - ylp[q];
          }
        } else {
          Algebra::assign(*err_out, y_high);
          Algebra::axpy(-1.0, y_low_, *err_out);
        }

        if (!Algebra::finite(*err_out)) {
          return false;
        }
      } else {
        Algebra::resize_like(*err_out, y_high);
        Algebra::set_zero(*err_out);
      }
    }

    return true;
  }

 private:
  std::array<State, Tableau::stages> k_{};
  State y_tmp_{};
  State y_low_{};
};

}  // namespace ode
