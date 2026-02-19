#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "ode/logging.hpp"
#include "ode/multistep/adams_bashforth_moulton.hpp"
#include "ode/multistep/adams_high_order.hpp"
#include "ode/multistep/nordsieck_abm4.hpp"
#include "ode/ode.hpp"

namespace {

using State = std::vector<double>;
using Rhs = std::function<void(double, const State&, State&)>;

struct Scenario {
  std::string name;
  double t0;
  double t1;
  State y0;
  Rhs rhs;
  double rkf45_inf_env;
  double abm6_inf_env;
  double nordsieck6_inf_env;
  double sundman_inf_env;
};

double Linf(const State& a, const State& b) {
  if (a.size() != b.size()) {
    return std::numeric_limits<double>::infinity();
  }
  double e = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    e = std::max(e, std::abs(a[i] - b[i]));
  }
  return e;
}

bool RunScenario(const Scenario& sc) {
  ode::IntegratorOptions ref_opt;
  ref_opt.adaptive = true;
  ref_opt.rtol = 1e-12;
  ref_opt.atol = 1e-15;
  ref_opt.h_init = 1e-2;
  ref_opt.h_max = 60.0;
  const auto ref = ode::integrate(ode::RKMethod::RKF78, sc.rhs, sc.t0, sc.y0, sc.t1, ref_opt);
  if (ref.status != ode::IntegratorStatus::Success) {
    ode::log::Error("reference RKF78 failed for scenario ", sc.name);
    return false;
  }

  ode::IntegratorOptions rkf45_opt = ref_opt;
  rkf45_opt.rtol = 1e-9;
  rkf45_opt.atol = 1e-12;
  const auto rkf45 = ode::integrate(ode::RKMethod::RKF45, sc.rhs, sc.t0, sc.y0, sc.t1, rkf45_opt);
  if (rkf45.status != ode::IntegratorStatus::Success) {
    ode::log::Error("RKF45 failed for scenario ", sc.name);
    return false;
  }

  ode::multistep::AdamsBashforthMoultonOptions abm6_opt;
  abm6_opt.h = 0.25;
  abm6_opt.mode = ode::multistep::PredictorCorrectorMode::Iterated;
  abm6_opt.corrector_iterations = 2;
  const auto abm6 = ode::multistep::integrate_abm6(sc.rhs, sc.t0, sc.y0, sc.t1, abm6_opt);
  if (abm6.status != ode::IntegratorStatus::Success) {
    ode::log::Error("ABM6 failed for scenario ", sc.name);
    return false;
  }

  ode::multistep::NordsieckAbmOptions nopt;
  nopt.rtol = 1e-10;
  nopt.atol = 1e-13;
  nopt.h_init = 0.1;
  nopt.h_min = 1e-6;
  nopt.h_max = 10.0;
  nopt.segment_steps = 4;
  nopt.max_restarts = 32;
  const auto nord6 = ode::multistep::integrate_nordsieck_abm6(sc.rhs, sc.t0, sc.y0, sc.t1, nopt);
  if (nord6.status != ode::IntegratorStatus::Success) {
    ode::log::Error("Nordsieck ABM6 failed for scenario ", sc.name);
    return false;
  }

  auto dt_ds = [](double, const State& y) {
    if (y.size() < 2) {
      return 1.0;
    }
    const double r2 = y[0] * y[0] + y[1] * y[1];
    const double r = std::sqrt(std::max(1e-12, r2));
    return std::max(1e-6, r);
  };
  ode::IntegratorOptions su_opt = ref_opt;
  su_opt.rtol = 1e-11;
  su_opt.atol = 1e-14;
  const auto sund = ode::integrate_sundman(ode::RKMethod::RKF78, sc.rhs, dt_ds, sc.t0, sc.y0, sc.t1, su_opt);
  if (sund.status != ode::IntegratorStatus::Success) {
    ode::log::Error("Sundman RKF78 failed for scenario ", sc.name);
    return false;
  }

  const double e_rkf45 = Linf(rkf45.y, ref.y);
  const double e_abm6 = Linf(abm6.y, ref.y);
  const double e_nord6 = Linf(nord6.y, ref.y);
  const double e_sund = Linf(sund.y, ref.y);

  if (e_rkf45 > sc.rkf45_inf_env || e_abm6 > sc.abm6_inf_env ||
      e_nord6 > sc.nordsieck6_inf_env || e_sund > sc.sundman_inf_env) {
    ode::log::Error("corpus envelope fail [", sc.name, "] rkf45=", e_rkf45,
                    " abm6=", e_abm6, " nord6=", e_nord6, " sund=", e_sund);
    return false;
  }

  return true;
}

}  // namespace

int main() {
  constexpr double mu = 398600.4418;

  const double rp = 7000.0;
  const double ra = 42000.0;
  const double a = 0.5 * (rp + ra);
  const double vp = std::sqrt(mu * (2.0 / rp - 1.0 / a));

  std::vector<Scenario> corpus;

  corpus.push_back(Scenario{
      .name = "HighEccTwoBody2D",
      .t0 = 0.0,
      .t1 = 6.0 * 3600.0,
      .y0 = {rp, 0.0, 0.0, vp},
      .rhs = [=](double, const State& y, State& dydt) {
        dydt.assign(4, 0.0);
        const double r2 = y[0] * y[0] + y[1] * y[1];
        const double r = std::sqrt(r2);
        const double inv_r3 = 1.0 / (r2 * r);
        dydt[0] = y[2];
        dydt[1] = y[3];
        dydt[2] = -mu * y[0] * inv_r3;
        dydt[3] = -mu * y[1] * inv_r3;
      },
      .rkf45_inf_env = 2e-2,
      .abm6_inf_env = 8e-2,
      .nordsieck6_inf_env = 3e-2,
      .sundman_inf_env = 2e-4,
  });

  corpus.push_back(Scenario{
      .name = "LEODrag2D",
      .t0 = 0.0,
      .t1 = 2.0 * 3600.0,
      .y0 = {7000.0, 0.0, 0.0, 7.55},
      .rhs = [=](double, const State& y, State& dydt) {
        constexpr double alpha = 1e-9;
        dydt.assign(4, 0.0);
        const double r2 = y[0] * y[0] + y[1] * y[1];
        const double r = std::sqrt(r2);
        const double inv_r3 = 1.0 / (r2 * r);
        dydt[0] = y[2];
        dydt[1] = y[3];
        dydt[2] = -mu * y[0] * inv_r3 - alpha * y[2];
        dydt[3] = -mu * y[1] * inv_r3 - alpha * y[3];
      },
      .rkf45_inf_env = 1e-3,
      .abm6_inf_env = 3e-2,
      .nordsieck6_inf_env = 5e-3,
      .sundman_inf_env = 2e-5,
  });

  corpus.push_back(Scenario{
      .name = "GEOJ2Like3D",
      .t0 = 0.0,
      .t1 = 24.0 * 3600.0,
      .y0 = {42164.0, 0.0, 10.0, 0.0, 3.07466, 0.0},
      .rhs = [=](double, const State& y, State& dydt) {
        constexpr double j2 = 1.08262668e-3;
        constexpr double re = 6378.137;
        dydt.assign(6, 0.0);
        const double x = y[0], yy = y[1], z = y[2];
        const double r2 = x * x + yy * yy + z * z;
        const double r = std::sqrt(r2);
        const double inv_r3 = 1.0 / (r2 * r);
        const double z2 = z * z;
        const double r5 = r2 * r2 * r;
        const double c = 1.5 * j2 * mu * re * re / r5;
        const double fxy = c * (5.0 * z2 / r2 - 1.0);
        const double fz = c * (5.0 * z2 / r2 - 3.0);
        dydt[0] = y[3];
        dydt[1] = y[4];
        dydt[2] = y[5];
        dydt[3] = -mu * x * inv_r3 + x * fxy;
        dydt[4] = -mu * yy * inv_r3 + yy * fxy;
        dydt[5] = -mu * z * inv_r3 + z * fz;
      },
      .rkf45_inf_env = 5e-2,
      .abm6_inf_env = 5e-1,
      .nordsieck6_inf_env = 2e-1,
      .sundman_inf_env = 1e-2,
  });

  for (const auto& sc : corpus) {
    if (!RunScenario(sc)) {
      return 1;
    }
  }

  return 0;
}
