# Reference Benchmark Corpus

Author: Watson

This corpus defines multi-regime regression scenarios with strict numerical
error envelopes against a high-accuracy RKF78 reference.

## Scenarios

1. `HighEccTwoBody2D`
- Regime: high-eccentricity two-body orbital motion.
- Dynamics: central gravity (`mu = 398600.4418 km^3/s^2`).
- Horizon: 6 hours.

2. `LEODrag2D`
- Regime: drag-perturbed LEO-like planar propagation.
- Dynamics: central gravity plus linear drag term.
- Horizon: 2 hours.

3. `GEOJ2Like3D`
- Regime: GEO-like three-dimensional long-arc propagation.
- Dynamics: central gravity plus J2 perturbation.
- Horizon: 24 hours.

## Compared Methods

- RKF45 adaptive
- ABM6 fixed-step predictor-corrector
- Nordsieck adaptive ABM6
- Sundman + RKF78

All are evaluated against RKF78 high-accuracy reference trajectories with
scenario-specific infinity-norm envelopes.

## Purpose

- Catch regressions across diverse operational regimes.
- Prevent method-specific tuning from overfitting to one ODE family.
- Provide reproducible acceptance gates for future modeling changes.
