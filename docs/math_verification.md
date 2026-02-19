# Mathematical Verification Strategy

Author: Watson

This document maps implemented tests to numerical-analysis objectives.

## 1) Order and Richardson Checks

Goal: verify expected convergence order.

Method:

- Run fixed-step solutions with `h`, `h/2`, `h/4`.
- Compare errors against analytic reference.
- Estimate order from error ratios.

Coverage:

- RK4 and RK8 regression checks.

## 2) Dense-Output Accuracy

Goal: validate interpolation quality between accepted states.

Method:

- Compare dense samples against analytic solution.
- Confirm improved dense accuracy when underlying integration is refined.

Coverage:

- Dense output accuracy regression test.

## 3) Reversibility

Goal: check forward/backward consistency.

Method:

- Integrate from `t0 -> t1`, then `t1 -> t0`.
- Compare reconstructed initial state.

Coverage:

- RK8 and Gauss-Jackson reversibility checks.

## 4) Stability Region Characterization

Goal: regression check of linear stability behavior.

Method on `y' = lambda y`:

- test stable and unstable samples for RK4 real-axis values
- coarse sweep to estimate boundary crossing

Coverage:

- RK4 stability-region regression test.

## 5) Stiffness Diagnostics

Goal: validate stiffness indicator behavior.

Method:

- Evaluate `h * ||J||_inf` on stiff and non-stiff synthetic Jacobians.
- Confirm expected classification.

Coverage:

- Stiff diagnostics regression test.

## 6) Variational FD-vs-STM

Goal: verify linearized sensitivity propagation.

Method:

- propagate nominal and perturbed trajectories
- compare true perturbation with `Phi * delta_x0`

Coverage:

- nonlinear trajectory FD-vs-STM consistency test.

## 7) Covariance Math Checks

Goal: ensure covariance algebra correctness.

Method:

- symmetry and PSD checks
- continuous-time covariance propagation vs discrete update consistency for small `dt`

Coverage:

- covariance math regression test.

## 8) Long-Horizon Invariants

Goal: detect long-term drift in conservative dynamics.

Method:

- propagate Kepler problem over many periods
- track relative drift in specific energy and angular momentum norm

Coverage:

- long-horizon invariant regression test.

## 9) Canonical ODE Cross-Checks

Goal: broad nonlinear confidence beyond a single reference ODE.

Coverage:

- Lorenz cross-solver consistency
- Van der Pol cross-solver consistency
- Kepler energy drift sanity

## Running Verification

Use:

```bash
ctest --preset macos-debug --output-on-failure
```

Math-focused subset (example):

```bash
ctest --preset macos-debug -R "ode_test_richardson|ode_test_reversibility|ode_test_stability_region|ode_test_stiff_diagnostics|ode_test_variational_fd_stm|ode_test_covariance_math|ode_test_long_horizon_invariants|ode_test_dense_output_accuracy" --output-on-failure
```

