#include <chrono>
#include <cstdlib>
#include <vector>

#include "ode/logging.hpp"
#include "ode/ode.hpp"

namespace {

int EnvInt(const char* key, int default_val) {
  if (const char* v = std::getenv(key)) {
    const int parsed = std::atoi(v);
    if (parsed > 0) {
      return parsed;
    }
  }
  return default_val;
}

}  // namespace

int main() {
  using State = std::vector<double>;
  State y0{1.0};

  auto rhs = [](double, const State& y, State& dydt) {
    dydt.resize(y.size());
    dydt[0] = -0.1 * y[0] + std::sin(y[0]);
  };

  ode::IntegratorOptions opt;
  opt.adaptive = true;
  opt.rtol = 1e-9;
  opt.atol = 1e-12;
  opt.h_init = 0.01;
  opt.h_max = 0.1;

  const int samples = EnvInt("ODE_PERF_SAMPLES", 20);
  const int iterations = EnvInt("ODE_PERF_ITERATIONS", 1000);

  auto t0 = std::chrono::steady_clock::now();
  long long accepted = 0;
  for (int s = 0; s < samples; ++s) {
    for (int i = 0; i < iterations; ++i) {
      const auto res = ode::integrate(ode::RKMethod::RKF78, rhs, 0.0, y0, 10.0, opt);
      if (res.status != ode::IntegratorStatus::Success) {
        ode::log::Error("integration failed, status=", ode::ToString(res.status));
        return 1;
      }
      accepted += res.stats.accepted_steps;
    }
  }
  auto t1 = std::chrono::steady_clock::now();
  const double sec = std::chrono::duration<double>(t1 - t0).count();

  const double runs = static_cast<double>(samples) * static_cast<double>(iterations);
  ode::log::Info("runs=", runs,
                 " seconds=", sec,
                 " runs_per_sec=", (runs / sec),
                 " avg_accepted_steps=", (static_cast<double>(accepted) / runs));
  return 0;
}
