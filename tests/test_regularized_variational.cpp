#include <cmath>
#include <vector>

#include "ode/logging.hpp"
#include "ode/ode.hpp"

namespace {

double LinfDiff(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) {
    return std::numeric_limits<double>::infinity();
  }
  double e = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    e = std::max(e, std::abs(a[i] - b[i]));
  }
  return e;
}

}  // namespace

int main() {
  constexpr double mu = 398600.4418;

  {
    const ode::regularization::TwoBody2DState s0{7000.0, 200.0, -0.2, 7.5};
    const std::vector<double> x0{s0.x, s0.y, s0.vx, s0.vy};
    std::vector<double> p0(16, 0.0);
    for (int i = 0; i < 4; ++i) {
      p0[static_cast<std::size_t>(i * 4 + i)] = 1e-4;
    }
    const double tf = 1800.0;

    auto accel = [](double, const ode::regularization::TwoBody2DState& s, std::array<double, 2>& a) {
      const double r2 = s.x * s.x + s.y * s.y;
      const double r = std::sqrt(r2);
      const double inv_r3 = 1.0 / (r2 * r);
      a[0] = -mu * s.x * inv_r3;
      a[1] = -mu * s.y * inv_r3;
    };
    auto accel_jac = [](double, const ode::regularization::TwoBody2DState& s, std::array<double, 4>& ar, std::array<double, 4>& av) {
      const double r2 = s.x * s.x + s.y * s.y;
      const double r = std::sqrt(r2);
      const double r3 = r2 * r;
      const double r5 = r3 * r2;
      ar[0] = mu * (3.0 * s.x * s.x / r5 - 1.0 / r3);
      ar[1] = mu * (3.0 * s.x * s.y / r5);
      ar[2] = ar[1];
      ar[3] = mu * (3.0 * s.y * s.y / r5 - 1.0 / r3);
      av = {0.0, 0.0, 0.0, 0.0};
      return true;
    };
    auto q_reg = [](double, const ode::regularization::TwoBody2DState&, ode::DynamicMatrix& q) {
      q.assign(16, 0.0);
      q[0] = 1e-10;
      q[5] = 1e-10;
      q[10] = 1e-10;
      q[15] = 1e-10;
      return true;
    };

    auto rhs = [](double, const std::vector<double>& x, std::vector<double>& dxdt) {
      dxdt.assign(4, 0.0);
      const double r2 = x[0] * x[0] + x[1] * x[1];
      const double r = std::sqrt(r2);
      const double inv_r3 = 1.0 / (r2 * r);
      dxdt[0] = x[2];
      dxdt[1] = x[3];
      dxdt[2] = -mu * x[0] * inv_r3;
      dxdt[3] = -mu * x[1] * inv_r3;
    };
    auto jac = [](double, const std::vector<double>& x, ode::DynamicMatrix& a) {
      a.assign(16, 0.0);
      const double r2 = x[0] * x[0] + x[1] * x[1];
      const double r = std::sqrt(r2);
      const double r3 = r2 * r;
      const double r5 = r3 * r2;
      a[2] = 1.0;
      a[7] = 1.0;
      a[8] = mu * (3.0 * x[0] * x[0] / r5 - 1.0 / r3);
      a[9] = mu * (3.0 * x[0] * x[1] / r5);
      a[12] = a[9];
      a[13] = mu * (3.0 * x[1] * x[1] / r5 - 1.0 / r3);
      return true;
    };
    auto q = [](double, const std::vector<double>&, ode::DynamicMatrix& q_out) {
      q_out.assign(16, 0.0);
      q_out[0] = 1e-10;
      q_out[5] = 1e-10;
      q_out[10] = 1e-10;
      q_out[15] = 1e-10;
      return true;
    };

    ode::IntegratorOptions opt;
    opt.adaptive = true;
    opt.rtol = 1e-10;
    opt.atol = 1e-13;
    opt.h_init = 1e-2;
    opt.h_max = 30.0;

    const auto reg = ode::regularization::integrate_cowell_sundman_variational_2d(
        accel, accel_jac, q_reg, ode::RKMethod::RKF78, 0.0, s0, p0, tf, opt, 1e-9);
    const auto ref = ode::variational::integrate_state_stm_cov(
        ode::RKMethod::RKF78, rhs, jac, q, 0.0, x0, p0, tf, opt);

    if (reg.status != ode::IntegratorStatus::Success || ref.status != ode::IntegratorStatus::Success) {
      ode::log::Error("regularized variational 2d run failed");
      return 1;
    }

    const double ex = LinfDiff(reg.x, ref.x);
    const double ephi = LinfDiff(reg.phi, ref.phi);
    const double ep = LinfDiff(reg.p, ref.p);
    if (ex > 2e-6 || ephi > 2e-6 || ep > 2e-6) {
      ode::log::Error("regularized variational 2d mismatch: ", ex, " ", ephi, " ", ep);
      return 1;
    }
  }

  return 0;
}
