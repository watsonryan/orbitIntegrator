# Regularization Methods

Author: Watson

This note documents the regularization models added under `ode::regularization` for the two-body problem and their numerical interpretation.

## Scope

Implemented methods:

- Planar Levi-Civita regularization (`integrate_two_body_levi_civita`)
- Sundman-transformed Cowell propagation (`integrate_two_body_sundman`)
- 3D KS regularization (`integrate_two_body_ks`)

Current scope is Kepler two-body dynamics (no extra perturbing forces in the regularized equations).

## Motivation

For close approaches, standard Cowell propagation in physical time can suffer from:

- very small stable time steps near periapsis
- stronger local truncation sensitivity to `1/r^2` acceleration variation

Regularization introduces:

- a transformed independent variable `s`
- transformed coordinates where near-singular behavior is smoothed

The intent is improved robustness and consistent accuracy through small-radius segments.

## Sundman Transform

The Sundman transform introduces

`dt/ds = g(x)`, with `g(x) > 0`

The implemented two-body wrapper uses:

`g(x) = max(r, r_min)` where `r = ||r_vec||`

and propagates the original Cowell ODE in pseudo-time `s`.

This leaves the state model unchanged while concentrating pseudo-time steps near small `r`.

## Levi-Civita (2D)

For planar motion, position is mapped as:

- `x = u1^2 - u2^2`
- `y = 2 u1 u2`

with Sundman relation `dt/ds = r = u1^2 + u2^2`.

With this transform, the Kepler problem reduces to a harmonic-like regularized form driven by specific orbital energy, and close-approach stiffness in physical coordinates is reduced.

Implementation notes:

- input/output remain physical `(x, y, vx, vy, t)`
- internals propagate `(u, u', t)` in pseudo-time using RK4
- specific energy is computed from initial physical state and treated as constant

## KS (3D)

The KS transform maps 4D regularized coordinates `u` to 3D physical position `r`.

In this implementation:

- gauge is fixed with `u4 = 0` at initialization
- time transform uses `dt/ds = ||u||^2`
- Kepler dynamics are propagated in regularized coordinates via RK4 in pseudo-time

As with Levi-Civita:

- interface is physical `(x, y, z, vx, vy, vz, t)`
- internals propagate `(u, u', t)`

## Numerical Equivalence Expectations

Regularized and non-regularized methods are mathematically equivalent for the same model, but not bitwise-identical numerically.

Reasons:

- different state coordinates
- different independent variable (`s` vs `t`)
- different local truncation/discretization pathways

So parity is evaluated using physical-state error tolerances, not exact floating-point identity.

In current regression tests/examples, observed deltas against Cowell are far below centimeter-level in position for validated scenarios.

## Practical Selection Guidance

- Use `integrate_two_body_sundman` first when you want a low-friction robustness/performance option while keeping Cowell equations.
- Use `integrate_two_body_levi_civita` for planar close-approach regularization.
- Use `integrate_two_body_ks` for 3D close-approach regularization.

## Current Limitations

- Regularized equations currently target pure two-body dynamics only.
- Perturbation-compatible regularized formulations (J2/drag/SRP/etc.) are not yet included.
- No explicit gauge-control strategy beyond fixed initialization.
