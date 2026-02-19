#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include "ode/ode.hpp"

namespace {

int EnvInt(const char* key, int fallback) {
  const char* raw = std::getenv(key);
  if (raw == nullptr || *raw == '\0') {
    return fallback;
  }
  char* end = nullptr;
  const long v = std::strtol(raw, &end, 10);
  if (end == raw || *end != '\0' || v <= 0 || v > 1000000) {
    return fallback;
  }
  return static_cast<int>(v);
}

template <class Fn>
double MeanMs(Fn&& fn, int samples, int iterations) {
  using Clock = std::chrono::steady_clock;
  volatile double sink = 0.0;
  double total_ms = 0.0;
  for (int s = 0; s < samples; ++s) {
    const auto t0 = Clock::now();
    for (int i = 0; i < iterations; ++i) {
      sink += fn();
    }
    const auto t1 = Clock::now();
    total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
  return total_ms / static_cast<double>(samples * iterations);
}

}  // namespace

int main() {
  constexpr double mu = 398600.4418;
  const int samples = EnvInt("ODE_REG_BENCH_SAMPLES", 10);
  const int iterations = EnvInt("ODE_REG_BENCH_ITERS", 20);

  // 2D high-eccentricity case.
  const double rp = 7000.0;
  const double ra = 42000.0;
  const double a = 0.5 * (rp + ra);
  const double vp = std::sqrt(mu * (2.0 / rp - 1.0 / a));
  const double tf2 = 6.0 * 3600.0;
  const ode::regularization::TwoBody2DState s02{rp, 0.0, 0.0, vp};
  std::vector<double> y02{s02.x, s02.y, s02.vx, s02.vy};

  auto rhs2 = [](double, const std::vector<double>& y, std::vector<double>& dydt) {
    dydt.assign(4, 0.0);
    const double r2 = y[0] * y[0] + y[1] * y[1];
    const double r = std::sqrt(r2);
    const double inv_r3 = 1.0 / (r2 * r);
    constexpr double mu2 = 398600.4418;
    dydt[0] = y[2];
    dydt[1] = y[3];
    dydt[2] = -mu2 * y[0] * inv_r3;
    dydt[3] = -mu2 * y[1] * inv_r3;
  };

  ode::IntegratorOptions opt;
  opt.adaptive = true;
  opt.rtol = 1e-12;
  opt.atol = 1e-15;
  opt.h_init = 5.0;
  opt.h_min = 1e-8;
  opt.h_max = 60.0;

  ode::regularization::RegularizationOptions ropt;
  ropt.ds = 1e-4;
  ropt.max_steps = 4000000;
  ropt.min_radius_km = 1e-9;

  const auto cow2 = ode::integrate(ode::RKMethod::RKF78, rhs2, 0.0, y02, tf2, opt);
  const auto lc2 = ode::regularization::integrate_two_body_levi_civita(mu, 0.0, s02, tf2, ropt);
  const auto su2 = ode::regularization::integrate_two_body_sundman(mu, ode::RKMethod::RKF78, 0.0, s02, tf2, opt, 1e-9);

  const double cow2_ms = MeanMs([&]() { return ode::integrate(ode::RKMethod::RKF78, rhs2, 0.0, y02, tf2, opt).stats.rhs_evals; },
                                samples, iterations);
  const double lc2_ms = MeanMs([&]() { return ode::regularization::integrate_two_body_levi_civita(mu, 0.0, s02, tf2, ropt).stats.rhs_evals; },
                               samples, iterations);
  const double su2_ms = MeanMs([&]() { return ode::regularization::integrate_two_body_sundman(mu, ode::RKMethod::RKF78, 0.0, s02, tf2, opt, 1e-9).stats.rhs_evals; },
                               samples, iterations);

  // 3D case.
  const ode::regularization::TwoBody3DState s03{9000.0, 500.0, 1000.0, -1.0, 6.6, 0.8};
  std::vector<double> y03{s03.x, s03.y, s03.z, s03.vx, s03.vy, s03.vz};
  const double tf3 = 2.0 * 3600.0;
  auto rhs3 = [](double, const std::vector<double>& y, std::vector<double>& dydt) {
    dydt.assign(6, 0.0);
    const double r2 = y[0] * y[0] + y[1] * y[1] + y[2] * y[2];
    const double r = std::sqrt(r2);
    const double inv_r3 = 1.0 / (r2 * r);
    constexpr double mu3 = 398600.4418;
    dydt[0] = y[3];
    dydt[1] = y[4];
    dydt[2] = y[5];
    dydt[3] = -mu3 * y[0] * inv_r3;
    dydt[4] = -mu3 * y[1] * inv_r3;
    dydt[5] = -mu3 * y[2] * inv_r3;
  };

  const auto cow3 = ode::integrate(ode::RKMethod::RKF78, rhs3, 0.0, y03, tf3, opt);
  const auto ks3 = ode::regularization::integrate_two_body_ks(mu, 0.0, s03, tf3, ropt);
  const double cow3_ms = MeanMs([&]() { return ode::integrate(ode::RKMethod::RKF78, rhs3, 0.0, y03, tf3, opt).stats.rhs_evals; },
                                samples, iterations);
  const double ks3_ms = MeanMs([&]() { return ode::regularization::integrate_two_body_ks(mu, 0.0, s03, tf3, ropt).stats.rhs_evals; },
                               samples, iterations);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "regularization_benchmark\n";
  std::cout << "samples=" << samples << " iterations=" << iterations << "\n";
  std::cout << "2D_Cowell_ms=" << cow2_ms << " rhs=" << cow2.stats.rhs_evals << "\n";
  std::cout << "2D_LeviCivita_ms=" << lc2_ms << " rhs=" << lc2.stats.rhs_evals << "\n";
  std::cout << "2D_Sundman_ms=" << su2_ms << " rhs=" << su2.stats.rhs_evals << "\n";
  std::cout << "3D_Cowell_ms=" << cow3_ms << " rhs=" << cow3.stats.rhs_evals << "\n";
  std::cout << "3D_KS_ms=" << ks3_ms << " rhs=" << ks3.stats.rhs_evals << "\n";
  return 0;
}
