#pragma once

#include <array>

#include "ode/algebra.hpp"

namespace ode {

template <class Tableau, class State, class Algebra = DefaultAlgebra<State>>
requires AlgebraFor<Algebra, State>
class ExplicitRKStepper {
 public:
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
  [[nodiscard]] bool step(RHS&& rhs,
                          double t,
                          const State& y,
                          double h,
                          State& y_high,
                          State* err_out) {
    const auto n = Algebra::size(y);

    for (int i = 0; i < Tableau::stages; ++i) {
      Algebra::assign(y_tmp_, y);
      for (int j = 0; j < i; ++j) {
        const double aij = Tableau::a[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
        if (aij != 0.0) {
          Algebra::axpy(h * aij, k_[static_cast<std::size_t>(j)], y_tmp_);
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

    Algebra::assign(y_high, y);
    for (int i = 0; i < Tableau::stages; ++i) {
      const double bi = Tableau::b_high[static_cast<std::size_t>(i)];
      if (bi != 0.0) {
        Algebra::axpy(h * bi, k_[static_cast<std::size_t>(i)], y_high);
      }
    }

    if (!Algebra::finite(y_high)) {
      return false;
    }

    if (err_out != nullptr) {
      if constexpr (Tableau::has_embedded) {
        Algebra::assign(y_low_, y);
        for (int i = 0; i < Tableau::stages; ++i) {
          const double bi = Tableau::b_low[static_cast<std::size_t>(i)];
          if (bi != 0.0) {
            Algebra::axpy(h * bi, k_[static_cast<std::size_t>(i)], y_low_);
          }
        }

        Algebra::assign(*err_out, y_high);
        Algebra::axpy(-1.0, y_low_, *err_out);

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
