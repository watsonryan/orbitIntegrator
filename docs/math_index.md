# Mathematical Documentation Index

Author: Watson

This document indexes the mathematical references for the integrators and verification suite in this repository.

## Scope

These docs describe numerical methods and analysis implemented in `orbitIntegrator`:

- Explicit Runge-Kutta (RK4, RKF45, RKF78/RK8)
- Multistep methods (ABM4, ABM6, Nordsieck ABM4, Gauss-Jackson-style second-order)
- Dense output and event detection
- Variational propagation (STM and covariance)
- Stiff integration helper and stiffness diagnostics
- Verification methodology (order, stability, reversibility, invariants, covariance checks)

## Documents

- `docs/math_runge_kutta.md`
- `docs/math_multistep.md`
- `docs/math_variational_covariance.md`
- `docs/math_stiffness.md`
- `docs/math_regularization.md`
- `docs/math_verification.md`

## Notation

- IVP: `y' = f(t, y)`, `y(t0) = y0`
- Step size: `h`
- State dimension: `n`
- Jacobian: `A(t) = df/dx`
- State transition matrix (STM): `Phi(t, t0)`
- Covariance: `P`
- Process noise (continuous): `Q`

All matrices are real-valued. Unless noted otherwise, vectors are column vectors.
