# orbitIntegrator

Author: Watson

C++20 explicit Runge-Kutta integrator for non-stiff ODEs:
- RK4 (fixed-step)
- RKF45 (embedded adaptive 4(5), fixed-step using high-order solution also supported)
- RKF78 (embedded adaptive 7(8), fixed-step using high-order solution also supported)
- RK8 alias (fixed-step using RKF78 high-order weights)

## Build and test

```bash
cmake --preset macos-debug
cmake --build --preset macos-debug -j
ctest --preset macos-debug --output-on-failure
```

## API quick start

```cpp
#include <ode/ode.hpp>

using State = std::vector<double>;
State y0{1.0};

auto rhs = [](double, const State& y, State& dydt) {
  dydt.resize(y.size());
  dydt[0] = y[0];
};

ode::IntegratorOptions opt;
opt.adaptive = true;
opt.rtol = 1e-10;
opt.atol = 1e-12;

auto res = ode::integrate(ode::RKMethod::RKF78, rhs, 0.0, y0, 1.0, opt);
```
