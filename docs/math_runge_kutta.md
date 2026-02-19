# Runge-Kutta Methods

Author: Watson

## Problem

Solve:

`y' = f(t, y),  y(t0) = y0`

over `t in [t0, t1]`.

## Explicit RK Step

Given tableau `(a_ij, b_i, c_i)`, stages:

`k_i = f(t_n + c_i h, y_n + h * sum_{j=1}^{i-1} a_ij k_j)`

state update:

`y_{n+1} = y_n + h * sum_i b_i k_i`

## Embedded RK Pair (Adaptive)

For embedded pairs, compute two solutions:

- high-order `y^{(p)}`
- low-order `y^{(p-1)}`

error estimate:

`e = y^{(p)} - y^{(p-1)}`

weighted RMS norm:

`||e||_wrms = sqrt((1/n) * sum_i (e_i / (atol + rtol * max(|y_i|, |y_i^{(p)}|)))^2 )`

accept step if `||e||_wrms <= 1`.

## Step-Size Controller

Typical proportional update:

`h_new = h * safety * ||e||^{-1/p}`

bounded by user factors `fac_min`, `fac_max`, and clipped to `[h_min, h_max]`.

## Methods in This Repo

- RK4: fixed-step 4th-order explicit method.
- RKF45: Fehlberg embedded 4(5).
- RKF78: embedded 7(8) pair.
- RK8 alias: fixed-step use of RKF78 high-order weights.

## Sundman Transformation

Optional transformed independent variable `s`:

`dt/ds = g(t, y) > 0`

Converted system:

`dy/ds = f(t, y) * g(t, y)`

and integrate in `s` while advancing `t`.

