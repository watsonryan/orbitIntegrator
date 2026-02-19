#pragma once

#include <functional>

namespace ode {

enum class RKMethod {
  RK4,
  RKF45,
  RK8,
  RKF78
};

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

  bool allow_step_reject = true;
};

enum class IntegratorStatus {
  Success,
  MaxStepsExceeded,
  StepSizeUnderflow,
  InvalidTolerance,
  InvalidStepSize,
  NaNDetected,
  UserStopped
};

struct IntegratorStats {
  int attempted_steps = 0;
  int accepted_steps = 0;
  int rejected_steps = 0;
  long long rhs_evals = 0;
  double last_h = 0.0;
  double last_error_norm = 0.0;
};

template <class State>
struct IntegratorResult {
  IntegratorStatus status = IntegratorStatus::Success;
  double t = 0.0;
  State y{};
  IntegratorStats stats{};
};

template <class State>
using Observer = std::function<bool(double, const State&)>;

}  // namespace ode
