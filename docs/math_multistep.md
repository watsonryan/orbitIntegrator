# Multistep Methods

Author: Watson

## Adams-Bashforth-Moulton (ABM)

ABM methods use previous derivative history for predictor/corrector updates.

For ABM4:

- Predictor (AB4):
  `y_{n+1}^p = y_n + h/24 * (55 f_n - 59 f_{n-1} + 37 f_{n-2} - 9 f_{n-3})`
- Corrector (AM4):
  `y_{n+1} = y_n + h/24 * (9 f_{n+1} + 19 f_n - 5 f_{n-1} + f_{n-2})`

Modes:

- `PEC`: predictor -> evaluate -> correct
- `PECE`: predictor -> evaluate -> correct -> evaluate
- `Iterated`: repeated correct/evaluate cycles

ABM6 extends this with 6-step AB/AM coefficients for higher order.

## Nordsieck Representation

Nordsieck methods propagate scaled derivatives:

`z = [y, h y', h^2/2! y'', ...]`

Advantages:

- compact polynomial state
- natural step-size adaptation
- robust high-order predictor/corrector updates

The repo includes an adaptive Nordsieck ABM4-style implementation.

## Gauss-Jackson-Style Second-Order Integrator

Target form:

`r'' = a(t, r, v)`, with `v = r'`

Implementation here uses:

- fixed-step startup via RK4
- AB8-style predictor for `r` and `v`
- AM8-style corrector with configurable iterations

This is suited to Cowell-type propagation where acceleration is modeled directly.

## Practical Notes

- Multistep methods require startup history from one-step methods.
- They are sensitive to discontinuities and abrupt force changes.
- For short final segments with nonuniform `h`, this repo falls back to one-step finishing updates.

