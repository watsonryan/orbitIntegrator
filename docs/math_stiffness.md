# Stiff Integration and Stiffness Diagnostics

Author: Watson

## Implicit Euler (Stiff Module)

For:

`y' = f(t, y)`

implicit Euler step solves:

`y_{n+1} = y_n + h f(t_{n+1}, y_{n+1})`

Define residual:

`R(y_{n+1}) = y_{n+1} - y_n - h f(t_{n+1}, y_{n+1})`

Solve `R = 0` by Newton iterations:

`J_R * delta = -R`

`y_{n+1}^{k+1} = y_{n+1}^k + delta`

with:

`J_R = I - h * df/dy`

This implementation uses finite-difference Jacobians and dense Gaussian elimination.

## Stiffness Diagnostic Heuristic

The helper `ode::stiff::assess_stiffness` computes:

- `||J||_inf` from a user-provided Jacobian
- ratio `rho = h * ||J||_inf`

and flags likely stiffness if:

`rho > threshold` (default threshold `2.0`).

This is a practical indicator, not a formal proof of stiffness.

## Recommended Use

- Use diagnostics to detect potentially stiff regions.
- Switch to stiff-capable integration path where needed.
- Validate switch logic using stiff benchmark problems and tolerance sweeps.

