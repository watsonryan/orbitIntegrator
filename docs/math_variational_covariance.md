# Variational and Covariance Propagation

Author: Watson

## Variational Equations

Given nonlinear dynamics:

`x' = f(t, x)`

the Jacobian is:

`A(t) = df/dx`

STM dynamics:

`Phi' = A * Phi`, with `Phi(t0) = I`

The propagated perturbation relation:

`delta x(t) ~= Phi(t, t0) * delta x(t0)`

## Covariance Propagation

Continuous-time covariance equation:

`P' = A P + P A^T + Q`

where `Q` is continuous process-noise intensity.

Discrete-time covariance update:

`P_{k+1} = Phi P_k Phi^T + Q_d`

where `Q_d` is integrated/discretized process noise over the step.

Joseph-form measurement covariance update:

`P^+ = (I - K H) P^- (I - K H)^T + K R K^T`

Square-root propagation (Cholesky form):

`P = S S^T`, propagate with
`P_{k+1} = Phi P_k Phi^T + Q_d`, then refactor to `S_{k+1}`.

Square-root measurement update (information-factor form):

Build stacked matrix
`T = [R_prior; S_r^{-1} H]`,
where `R_prior^T R_prior = (P^-)^{-1}` and `R = S_r S_r^T`.
QR-factorization of `T` yields upper-triangular information factor `R_post`,
then recover covariance factor `S_post = (R_post^{-1})^T`.

## AD Jacobian Support

This repo provides forward-mode AD helpers to compute `A = df/dx` from a generic RHS, reducing manual Jacobian derivation burden.

## Consistency Checks

Implemented verification includes:

- Finite-difference trajectory perturbation vs STM-predicted perturbation
- Covariance symmetry and PSD checks (small-dimensional regression cases)
- Continuous-time covariance propagation vs one-step discrete approximation at small `dt`
- Joseph-form symmetry/PSD validation checks
- Square-root propagation consistency against direct covariance propagation
- Square-root information measurement update consistency against Joseph form

## API Namespaces

- `ode::uncertainty::*`: original STM/covariance API
- `ode::variational::*`: model-agnostic alias layer
- `ode::eigen::uncertainty::*` / `ode::eigen::variational::*`: Eigen-first variants
