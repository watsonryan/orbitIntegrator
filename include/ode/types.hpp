/**
 * @file types.hpp
 * @brief Public API enums, options, status, stats, and result containers.
 */
#pragma once

#include <functional>

namespace ode {

/** @brief Supported explicit Runge-Kutta methods. */
enum class RKMethod {
  RK4,
  RKF45,
  RK8,
  RKF78
};

/** @brief Integration configuration options. */
struct IntegratorOptions {
  double rtol = 1e-8;
  double atol = 1e-12;

  double h_init = 0.0;
  double h_min = 1e-16;
  double h_max = 1e+16;

  bool adaptive = true;
  double fixed_h = 0.0;

  int max_steps = 1000000;

  double safety = 0.9;
  double fac_min = 0.2;
  double fac_max = 5.0;
  // Optional invariant drift bound for invariant-aware adaptive control APIs.
  double invariant_rtol = 0.0;

  bool allow_step_reject = true;
};

/** @brief Terminal status returned by an integration run. */
enum class IntegratorStatus {
  Success,
  MaxStepsExceeded,
  StepSizeUnderflow,
  InvalidTolerance,
  InvalidStepSize,
  NaNDetected,
  UserStopped
};

/** @brief Convert IntegratorStatus to stable string token. */
[[nodiscard]] inline const char* ToString(IntegratorStatus status) {
  switch (status) {
    case IntegratorStatus::Success:
      return "success";
    case IntegratorStatus::MaxStepsExceeded:
      return "max_steps_exceeded";
    case IntegratorStatus::StepSizeUnderflow:
      return "step_size_underflow";
    case IntegratorStatus::InvalidTolerance:
      return "invalid_tolerance";
    case IntegratorStatus::InvalidStepSize:
      return "invalid_step_size";
    case IntegratorStatus::NaNDetected:
      return "nan_detected";
    case IntegratorStatus::UserStopped:
      return "user_stopped";
  }
  return "unknown";
}

/** @brief Runtime counters and last-step telemetry. */
struct IntegratorStats {
  int attempted_steps = 0;
  int accepted_steps = 0;
  int rejected_steps = 0;
  long long rhs_evals = 0;
  double last_h = 0.0;
  double last_error_norm = 0.0;
  double last_invariant_error = 0.0;
};

/** @brief Generic integration result payload. */
template <class State>
struct IntegratorResult {
  IntegratorStatus status = IntegratorStatus::Success;
  double t = 0.0;
  State y{};
  IntegratorStats stats{};
};

/** @brief Optional callback invoked after each accepted step; return false to stop. */
template <class State>
using Observer = std::function<bool(double, const State&)>;

}  // namespace ode
