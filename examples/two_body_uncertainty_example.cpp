#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <vector>

#include "ode/ode.hpp"

namespace {

using State = ode::DynamicState;
using Matrix = ode::DynamicMatrix;

std::size_t idx(std::size_t n, std::size_t r, std::size_t c) {
  return r * n + c;
}

template <class Scalar>
void TwoBodyRhs(double /*t*/, const std::vector<Scalar>& x, std::vector<Scalar>& dxdt) {
  constexpr double kMu = 398600.4418;  // km^3/s^2
  dxdt.resize(6);

  const Scalar r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
  const Scalar r = sqrt(r2);
  const Scalar inv_r3 = 1.0 / (r2 * r);

  dxdt[0] = x[3];
  dxdt[1] = x[4];
  dxdt[2] = x[5];
  dxdt[3] = -kMu * x[0] * inv_r3;
  dxdt[4] = -kMu * x[1] * inv_r3;
  dxdt[5] = -kMu * x[2] * inv_r3;
}

double MatrixInfNorm(const Matrix& m, std::size_t n) {
  double max_row_sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    double row_sum = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      row_sum += std::abs(m[idx(n, i, j)]);
    }
    max_row_sum = std::max(max_row_sum, row_sum);
  }
  return max_row_sum;
}

}  // namespace

int main() {
  constexpr double kMu = 398600.4418;  // km^3/s^2
  constexpr double kOrbitalRadiusKm = 7000.0;
  const double kOrbitalSpeedKms = std::sqrt(kMu / kOrbitalRadiusKm);
  const double period_s =
      2.0 * std::numbers::pi * std::sqrt((kOrbitalRadiusKm * kOrbitalRadiusKm * kOrbitalRadiusKm) / kMu);

  State x0{kOrbitalRadiusKm, 0.0, 0.0, 0.0, kOrbitalSpeedKms, 0.0};
  Matrix p0(36, 0.0);
  p0[idx(6, 0, 0)] = 1e-2;  // km^2
  p0[idx(6, 1, 1)] = 1e-2;
  p0[idx(6, 2, 2)] = 1e-2;
  p0[idx(6, 3, 3)] = 1e-8;  // (km/s)^2
  p0[idx(6, 4, 4)] = 1e-8;
  p0[idx(6, 5, 5)] = 1e-8;

  ode::IntegratorOptions opt;
  opt.adaptive = true;
  opt.rtol = 1e-11;
  opt.atol = 1e-14;
  opt.h_init = 10.0;
  opt.h_min = 1e-6;
  opt.h_max = 60.0;

  auto jac_ad = [](double t, const State& x, Matrix& a) {
    return ode::uncertainty::jacobian_forward_ad(TwoBodyRhs<ode::uncertainty::Dual>, t, x, a);
  };
  auto q_zero = [](double /*t*/, const State& x, Matrix& q) {
    const std::size_t n = x.size();
    q.assign(n * n, 0.0);
    return true;
  };

  const auto out = ode::uncertainty::integrate_state_stm_cov(
      ode::RKMethod::RKF78, TwoBodyRhs<double>, jac_ad, q_zero, 0.0, x0, p0, period_s, opt);

  if (out.status != ode::IntegratorStatus::Success) {
    std::cerr << "integration failed, status=" << ode::ToString(out.status) << "\n";
    return 1;
  }

  Matrix i_minus_phi(36, 0.0);
  for (std::size_t r = 0; r < 6; ++r) {
    for (std::size_t c = 0; c < 6; ++c) {
      i_minus_phi[idx(6, r, c)] = out.phi[idx(6, r, c)];
      if (r == c) {
        i_minus_phi[idx(6, r, c)] -= 1.0;
      }
    }
  }

  const Matrix p_final_discrete = ode::uncertainty::propagate_covariance_discrete(out.phi, p0, Matrix(36, 0.0), 6);
  Matrix p_diff(36, 0.0);
  for (std::size_t i = 0; i < 36; ++i) {
    p_diff[i] = out.p[i] - p_final_discrete[i];
  }

  std::cout << std::setprecision(12);
  std::cout << "Two-body uncertainty propagation complete\n";
  std::cout << "status: " << ode::ToString(out.status) << "\n";
  std::cout << "t_final [s]: " << out.t << " (period ~ " << period_s << ")\n";
  std::cout << "state_final [km, km/s]:\n";
  std::cout << "  r = [" << out.x[0] << ", " << out.x[1] << ", " << out.x[2] << "]\n";
  std::cout << "  v = [" << out.x[3] << ", " << out.x[4] << ", " << out.x[5] << "]\n";
  std::cout << "STM first row: [" << out.phi[idx(6, 0, 0)] << ", " << out.phi[idx(6, 0, 1)] << ", "
            << out.phi[idx(6, 0, 2)] << ", " << out.phi[idx(6, 0, 3)] << ", " << out.phi[idx(6, 0, 4)] << ", "
            << out.phi[idx(6, 0, 5)] << "]\n";
  std::cout << "||Phi - I||_inf: " << MatrixInfNorm(i_minus_phi, 6) << "\n";
  std::cout << "continuous-vs-discrete covariance consistency (Q=0): ||P - Phi*P0*Phi^T||_inf = "
            << MatrixInfNorm(p_diff, 6) << "\n";
  std::cout << "steps accepted/rejected: " << out.stats.accepted_steps << "/" << out.stats.rejected_steps << "\n";
  return 0;
}

