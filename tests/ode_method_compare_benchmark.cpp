#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "ode/logging.hpp"
#include "ode/multistep/adams_bashforth_moulton.hpp"
#include "ode/multistep/adams_high_order.hpp"
#include "ode/multistep/nordsieck_abm4.hpp"
#include "ode/ode.hpp"

namespace {

using State = std::vector<double>;

struct MethodMetrics {
  std::string name;
  double elapsed_sec{0.0};
  double runs_per_sec{0.0};
  double mean_abs_err{0.0};
  double mean_steps{0.0};
  double mean_rhs{0.0};
};

int EnvInt(const char* key, int default_val) {
  if (const char* v = std::getenv(key)) {
    const int parsed = std::atoi(v);
    if (parsed > 0) {
      return parsed;
    }
  }
  return default_val;
}

constexpr double kT0 = 0.0;
constexpr double kT1 = 10.0;

double ExactSolution(double t) {
  constexpr double a = 0.009900990099009901;   // 1/101
  constexpr double b = 0.09900990099009901;    // 10/101
  constexpr double c = 0.9900990099009901;     // 100/101
  return c * std::exp(-0.1 * t) + a * std::cos(t) + b * std::sin(t);
}

auto MakeRhs() {
  return [](double t, const State& y, State& dydt) {
    dydt.resize(y.size());
    dydt[0] = -0.1 * y[0] + 0.1 * std::cos(t);
  };
}

MethodMetrics RunRkf78(int total_runs, const State& y0) {
  auto rhs = MakeRhs();

  ode::IntegratorOptions opt;
  opt.adaptive = true;
  opt.rtol = 1e-9;
  opt.atol = 1e-12;
  opt.h_init = 0.01;
  opt.h_max = 0.2;

  double err_sum = 0.0;
  double steps_sum = 0.0;
  double rhs_sum = 0.0;

  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < total_runs; ++i) {
    const auto res = ode::integrate(ode::RKMethod::RKF78, rhs, kT0, y0, kT1, opt);
    if (res.status != ode::IntegratorStatus::Success) {
      ode::log::Error("RKF78 run failed, status=", ode::ToString(res.status));
      std::exit(1);
    }
    err_sum += std::abs(res.y[0] - ExactSolution(kT1));
    steps_sum += static_cast<double>(res.stats.accepted_steps);
    rhs_sum += static_cast<double>(res.stats.rhs_evals);
  }
  const auto t1 = std::chrono::steady_clock::now();

  const double sec = std::chrono::duration<double>(t1 - t0).count();
  return MethodMetrics{"RKF78(adaptive)", sec, total_runs / sec, err_sum / total_runs, steps_sum / total_runs,
                       rhs_sum / total_runs};
}

MethodMetrics RunAbm4(int total_runs,
                      const State& y0,
                      ode::multistep::PredictorCorrectorMode mode,
                      int iter_count,
                      const char* label) {
  auto rhs = MakeRhs();

  ode::multistep::AdamsBashforthMoultonOptions opt;
  opt.h = 0.01;
  opt.mode = mode;
  opt.corrector_iterations = iter_count;

  double err_sum = 0.0;
  double steps_sum = 0.0;
  double rhs_sum = 0.0;

  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < total_runs; ++i) {
    const auto res = ode::multistep::integrate_abm4(rhs, kT0, y0, kT1, opt);
    if (res.status != ode::IntegratorStatus::Success) {
      ode::log::Error("ABM4 run failed, status=", ode::ToString(res.status));
      std::exit(1);
    }
    err_sum += std::abs(res.y[0] - ExactSolution(kT1));
    steps_sum += static_cast<double>(res.stats.accepted_steps);
    rhs_sum += static_cast<double>(res.stats.rhs_evals);
  }
  const auto t1 = std::chrono::steady_clock::now();

  const double sec = std::chrono::duration<double>(t1 - t0).count();
  return MethodMetrics{label, sec, total_runs / sec, err_sum / total_runs, steps_sum / total_runs,
                       rhs_sum / total_runs};
}

MethodMetrics RunAbm6(int total_runs, const State& y0) {
  auto rhs = MakeRhs();

  ode::multistep::AdamsBashforthMoultonOptions opt;
  opt.h = 0.01;
  opt.mode = ode::multistep::PredictorCorrectorMode::Iterated;
  opt.corrector_iterations = 2;

  double err_sum = 0.0;
  double steps_sum = 0.0;
  double rhs_sum = 0.0;

  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < total_runs; ++i) {
    const auto res = ode::multistep::integrate_abm6(rhs, kT0, y0, kT1, opt);
    if (res.status != ode::IntegratorStatus::Success) {
      ode::log::Error("ABM6 run failed, status=", ode::ToString(res.status));
      std::exit(1);
    }
    err_sum += std::abs(res.y[0] - ExactSolution(kT1));
    steps_sum += static_cast<double>(res.stats.accepted_steps);
    rhs_sum += static_cast<double>(res.stats.rhs_evals);
  }
  const auto t1 = std::chrono::steady_clock::now();

  const double sec = std::chrono::duration<double>(t1 - t0).count();
  return MethodMetrics{"ABM6-Iter2", sec, total_runs / sec, err_sum / total_runs, steps_sum / total_runs,
                       rhs_sum / total_runs};
}

MethodMetrics RunNordsieckAbm4(int total_runs, const State& y0) {
  auto rhs = MakeRhs();

  ode::multistep::NordsieckAbmOptions opt;
  opt.rtol = 1e-8;
  opt.atol = 1e-12;
  opt.h_init = 0.01;
  opt.h_min = 1e-8;
  opt.h_max = 0.2;

  double err_sum = 0.0;
  double steps_sum = 0.0;
  double rhs_sum = 0.0;

  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < total_runs; ++i) {
    const auto res = ode::multistep::integrate_nordsieck_abm4(rhs, kT0, y0, kT1, opt);
    if (res.status != ode::IntegratorStatus::Success) {
      ode::log::Error("Nordsieck ABM4 run failed, status=", ode::ToString(res.status));
      std::exit(1);
    }
    err_sum += std::abs(res.y[0] - ExactSolution(kT1));
    steps_sum += static_cast<double>(res.stats.accepted_steps);
    rhs_sum += static_cast<double>(res.stats.rhs_evals);
  }
  const auto t1 = std::chrono::steady_clock::now();

  const double sec = std::chrono::duration<double>(t1 - t0).count();
  return MethodMetrics{"Nordsieck-ABM4", sec, total_runs / sec, err_sum / total_runs, steps_sum / total_runs,
                       rhs_sum / total_runs};
}

MethodMetrics RunSundmanRkf78(int total_runs, const State& y0) {
  auto rhs = MakeRhs();

  ode::IntegratorOptions opt;
  opt.adaptive = true;
  opt.rtol = 1e-9;
  opt.atol = 1e-12;
  opt.h_init = 0.01;
  opt.h_max = 0.2;

  auto dt_ds = [](double t, const State&) {
    return 0.5 + 0.5 * std::cos(0.2 * t);
  };

  double err_sum = 0.0;
  double steps_sum = 0.0;
  double rhs_sum = 0.0;

  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < total_runs; ++i) {
    const auto res = ode::integrate_sundman(ode::RKMethod::RKF78, rhs, dt_ds, kT0, y0, kT1, opt);
    if (res.status != ode::IntegratorStatus::Success) {
      ode::log::Error("Sundman RKF78 run failed, status=", ode::ToString(res.status));
      std::exit(1);
    }
    err_sum += std::abs(res.y[0] - ExactSolution(kT1));
    steps_sum += static_cast<double>(res.stats.accepted_steps);
    rhs_sum += static_cast<double>(res.stats.rhs_evals);
  }
  const auto t1 = std::chrono::steady_clock::now();

  const double sec = std::chrono::duration<double>(t1 - t0).count();
  return MethodMetrics{"Sundman+RKF78", sec, total_runs / sec, err_sum / total_runs, steps_sum / total_runs,
                       rhs_sum / total_runs};
}

void PrintRow(const MethodMetrics& m) {
  std::cout << std::left << std::setw(16) << m.name
            << "  " << std::right << std::setw(11) << std::fixed << std::setprecision(1) << m.runs_per_sec
            << "  " << std::setw(12) << std::scientific << std::setprecision(3) << m.mean_abs_err
            << "  " << std::setw(10) << std::fixed << std::setprecision(1) << m.mean_steps
            << "  " << std::setw(9) << std::fixed << std::setprecision(1) << m.mean_rhs
            << "\n";
}

}  // namespace

int main() {
  const int samples = EnvInt("ODE_COMPARE_SAMPLES", 20);
  const int iterations = EnvInt("ODE_COMPARE_ITERATIONS", 1000);
  const int total_runs = samples * iterations;

  State y0{1.0};

  const auto rk = RunRkf78(total_runs, y0);
  const auto abm_pec = RunAbm4(total_runs, y0, ode::multistep::PredictorCorrectorMode::PEC, 1, "ABM4-PEC");
  const auto abm_pece = RunAbm4(total_runs, y0, ode::multistep::PredictorCorrectorMode::PECE, 1, "ABM4-PECE");
  const auto abm_iter = RunAbm4(total_runs, y0, ode::multistep::PredictorCorrectorMode::Iterated, 2, "ABM4-Iter2");
  const auto abm6_iter = RunAbm6(total_runs, y0);
  const auto nord = RunNordsieckAbm4(total_runs, y0);
  const auto sund = RunSundmanRkf78(total_runs, y0);

  std::cout << "method             runs/sec    mean_abs_err   mean_steps   mean_rhs\n";
  PrintRow(rk);
  PrintRow(abm_pec);
  PrintRow(abm_pece);
  PrintRow(abm_iter);
  PrintRow(abm6_iter);
  PrintRow(nord);
  PrintRow(sund);

  return 0;
}
