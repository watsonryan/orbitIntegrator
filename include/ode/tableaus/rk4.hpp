/**
 * @file rk4.hpp
 * @brief Classic 4-stage 4th-order fixed-step Runge-Kutta tableau.
 */
#pragma once

#include <array>

namespace ode {

/** @brief Classical RK4 tableau. */
struct TableauRK4 {
  static constexpr int stages = 4;
  static constexpr bool has_embedded = false;
  static constexpr int order_high = 4;
  static constexpr int order_low = 0;

  static constexpr std::array<double, stages> c = {0.0, 0.5, 0.5, 1.0};

  static constexpr std::array<std::array<double, stages>, stages> a = {{
      {{0.0, 0.0, 0.0, 0.0}},
      {{0.5, 0.0, 0.0, 0.0}},
      {{0.0, 0.5, 0.0, 0.0}},
      {{0.0, 0.0, 1.0, 0.0}},
  }};

  static constexpr std::array<double, stages> b_high = {1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0};
};

}  // namespace ode
