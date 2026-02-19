/**
 * @file rkf45.hpp
 * @brief Fehlberg 4(5) embedded explicit Runge-Kutta tableau.
 */
#pragma once

#include <array>

namespace ode {

/** @brief Standard RKF45 tableau with 5th-order accepted solution and 4th-order embedded estimate. */
struct TableauRKF45 {
  static constexpr int stages = 6;
  static constexpr bool has_embedded = true;
  static constexpr int order_high = 5;
  static constexpr int order_low = 4;

  static constexpr std::array<double, stages> c = {
      0.0,
      1.0 / 4.0,
      3.0 / 8.0,
      12.0 / 13.0,
      1.0,
      1.0 / 2.0,
  };

  static constexpr std::array<std::array<double, stages>, stages> a = {{
      {{0, 0, 0, 0, 0, 0}},
      {{1.0 / 4.0, 0, 0, 0, 0, 0}},
      {{3.0 / 32.0, 9.0 / 32.0, 0, 0, 0, 0}},
      {{1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0, 0, 0, 0}},
      {{439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0, 0, 0}},
      {{-8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0, 0}},
  }};

  static constexpr std::array<double, stages> b_low = {
      25.0 / 216.0,
      0.0,
      1408.0 / 2565.0,
      2197.0 / 4104.0,
      -1.0 / 5.0,
      0.0,
  };

  static constexpr std::array<double, stages> b_high = {
      16.0 / 135.0,
      0.0,
      6656.0 / 12825.0,
      28561.0 / 56430.0,
      -9.0 / 50.0,
      2.0 / 55.0,
  };
};

}  // namespace ode
