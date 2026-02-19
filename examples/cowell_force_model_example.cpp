#include <cmath>
#include <numbers>
#include <vector>

#include <Eigen/Core>

#include "ode/eigen_api.hpp"
#include "ode/logging.hpp"

namespace {

using Vec3 = Eigen::Vector3d;
using State = ode::eigen::Vector;

struct ForceBreakdown {
  Vec3 grav{Vec3::Zero()};
  Vec3 j2{Vec3::Zero()};
  Vec3 drag{Vec3::Zero()};
  Vec3 srp{Vec3::Zero()};
  Vec3 rel{Vec3::Zero()};
  Vec3 tidal{Vec3::Zero()};
};

class CowellForceModel {
 public:
  [[nodiscard]] ForceBreakdown Accelerations(double t, const Vec3& r_km, const Vec3& v_kms) const {
    ForceBreakdown out{};
    out.grav = CentralGravity(r_km);
    out.j2 = J2Acceleration(r_km);
    out.drag = DragAcceleration(r_km, v_kms);
    out.srp = SrpAcceleration();
    out.rel = RelativityAcceleration(r_km, v_kms);
    out.tidal = TidalAcceleration(t, r_km);
    return out;
  }

  [[nodiscard]] Vec3 TotalAcceleration(double t, const Vec3& r_km, const Vec3& v_kms) const {
    const ForceBreakdown a = Accelerations(t, r_km, v_kms);
    return a.grav + a.j2 + a.drag + a.srp + a.rel + a.tidal;
  }

 private:
  static constexpr double kMu = 398600.4418;       // km^3/s^2
  static constexpr double kRe = 6378.1363;         // km
  static constexpr double kJ2 = 1.08262668e-3;     // -
  static constexpr double kCdAOverM = 0.02;        // m^2/kg (small demo value)
  static constexpr double kRho0 = 4.0e-13;         // kg/m^3 (very rough LEO-ish)
  static constexpr double kH = 60.0;               // km scale height
  static constexpr double kCrAOverM = 0.01;        // m^2/kg (small demo value)
  static constexpr double kPsr = 4.56e-6;          // N/m^2
  static constexpr double kC = 299792458.0;        // m/s

  [[nodiscard]] static Vec3 CentralGravity(const Vec3& r_km) {
    const double r = r_km.norm();
    const double inv_r3 = 1.0 / (r * r * r);
    return -kMu * inv_r3 * r_km;
  }

  [[nodiscard]] static Vec3 J2Acceleration(const Vec3& r_km) {
    const double x = r_km.x();
    const double y = r_km.y();
    const double z = r_km.z();
    const double r2 = r_km.squaredNorm();
    const double r = std::sqrt(r2);
    const double z2 = z * z;
    const double factor = 1.5 * kJ2 * kMu * kRe * kRe / std::pow(r, 5);
    const double txy = 5.0 * z2 / r2 - 1.0;
    const double tz = 5.0 * z2 / r2 - 3.0;
    return Vec3{factor * x * txy, factor * y * txy, factor * z * tz};
  }

  [[nodiscard]] static Vec3 DragAcceleration(const Vec3& r_km, const Vec3& v_kms) {
    const double alt_km = r_km.norm() - kRe;
    const double rho = kRho0 * std::exp(-alt_km / kH);
    const Vec3 v_ms = 1000.0 * v_kms;
    const double v = v_ms.norm();
    if (v == 0.0) {
      return Vec3::Zero();
    }
    const Vec3 a_ms2 = -0.5 * rho * kCdAOverM * v * v_ms;
    return 1e-3 * a_ms2;  // km/s^2
  }

  [[nodiscard]] static Vec3 SrpAcceleration() {
    const Vec3 sun_dir = Vec3{1.0, 0.0, 0.0}.normalized();
    const Vec3 a_ms2 = kPsr * kCrAOverM * sun_dir;
    return 1e-3 * a_ms2;  // km/s^2
  }

  [[nodiscard]] static Vec3 RelativityAcceleration(const Vec3& r_km, const Vec3& v_kms) {
    const Vec3 r_m = 1000.0 * r_km;
    const Vec3 v_ms = 1000.0 * v_kms;
    const double r = r_m.norm();
    const double v2 = v_ms.squaredNorm();
    const double rv = r_m.dot(v_ms);
    const double mu_m = kMu * 1e9;  // m^3/s^2
    const double coeff = mu_m / (kC * kC * r * r * r);
    const Vec3 a_ms2 = coeff * ((4.0 * mu_m / r - v2) * r_m + 4.0 * rv * v_ms);
    return 1e-3 * a_ms2;  // km/s^2
  }

  [[nodiscard]] static Vec3 TidalAcceleration(double t, const Vec3& r_km) {
    const double w = 2.0 * std::numbers::pi / 43200.0;  // semidiurnal-ish
    const double amp = 1e-10;  // km/s^2 tiny placeholder
    return amp * std::sin(w * t) * r_km.normalized();
  }
};

}  // namespace

int main() {
  constexpr double kMu = 398600.4418;  // km^3/s^2
  constexpr double kR0 = 7000.0;       // km
  const double v0_kms = std::sqrt(kMu / kR0);
  const double period_s = 2.0 * std::numbers::pi * std::sqrt((kR0 * kR0 * kR0) / kMu);

  CowellForceModel forces;

  State x0(6);
  x0 << kR0, 0.0, 0.0, 0.0, v0_kms, 0.0;

  auto rhs = [&forces](double t, const State& x, State& dxdt) {
    const Vec3 r = x.segment<3>(0);
    const Vec3 v = x.segment<3>(3);
    const Vec3 a = forces.TotalAcceleration(t, r, v);
    dxdt.resize(6);
    dxdt.segment<3>(0) = v;
    dxdt.segment<3>(3) = a;
  };

  ode::IntegratorOptions opt;
  opt.adaptive = true;
  opt.rtol = 1e-11;
  opt.atol = 1e-14;
  opt.h_init = 10.0;
  opt.h_min = 1e-6;
  opt.h_max = 60.0;

  const double tf = 2.0 * period_s;
  const auto rk = ode::eigen::integrate(ode::RKMethod::RKF78, rhs, 0.0, x0, tf, opt);
  if (rk.status != ode::IntegratorStatus::Success) {
    ode::log::Error("Cowell RKF78 propagation failed, status=", ode::ToString(rk.status));
    return 1;
  }

  std::vector<double> r0{ x0(0), x0(1), x0(2) };
  std::vector<double> v0_vec{x0(3), x0(4), x0(5)};
  auto accel = [&forces](double t,
                         const std::vector<double>& r,
                         const std::vector<double>& v,
                         std::vector<double>& a) {
    const Vec3 r3{r[0], r[1], r[2]};
    const Vec3 v3{v[0], v[1], v[2]};
    const Vec3 at = forces.TotalAcceleration(t, r3, v3);
    a.resize(3);
    a[0] = at.x();
    a[1] = at.y();
    a[2] = at.z();
  };

  ode::multistep::GaussJackson8Options gj_opt;
  gj_opt.h = 10.0;
  gj_opt.corrector_iterations = 2;
  const auto gj = ode::multistep::integrate_gauss_jackson8(accel, 0.0, r0, v0_vec, tf, gj_opt);
  if (gj.status != ode::IntegratorStatus::Success) {
    ode::log::Error("Cowell Gauss-Jackson propagation failed, status=", ode::ToString(gj.status));
    return 1;
  }

  const ForceBreakdown a0 = forces.Accelerations(0.0, x0.segment<3>(0), x0.segment<3>(3));
  const double dr =
      std::sqrt(std::pow(gj.r[0] - rk.y(0), 2) + std::pow(gj.r[1] - rk.y(1), 2) + std::pow(gj.r[2] - rk.y(2), 2));
  const double dv =
      std::sqrt(std::pow(gj.v[0] - rk.y(3), 2) + std::pow(gj.v[1] - rk.y(4), 2) + std::pow(gj.v[2] - rk.y(5), 2));

  ode::log::Info("Cowell force-model example complete (RKF78 + Gauss-Jackson)");
  ode::log::Info("t_final [s] = ", rk.t, " (", tf, ")");
  ode::log::Info("RKF78 final r [km] = [", rk.y(0), ", ", rk.y(1), ", ", rk.y(2), "]");
  ode::log::Info("RKF78 final v [km/s] = [", rk.y(3), ", ", rk.y(4), ", ", rk.y(5), "]");
  ode::log::Info("RKF78 steps accepted/rejected = ", rk.stats.accepted_steps, "/", rk.stats.rejected_steps);
  ode::log::Info("GJ final r [km] = [", gj.r[0], ", ", gj.r[1], ", ", gj.r[2], "]");
  ode::log::Info("GJ final v [km/s] = [", gj.v[0], ", ", gj.v[1], ", ", gj.v[2], "]");
  ode::log::Info("GJ steps accepted/rejected = ", gj.stats.accepted_steps, "/", gj.stats.rejected_steps);
  ode::log::Info("GJ-RKF78 delta |dr| [km] = ", dr, "  |dv| [km/s] = ", dv);
  ode::log::Info("a_grav  [km/s^2] = ", a0.grav.transpose());
  ode::log::Info("a_j2    [km/s^2] = ", a0.j2.transpose());
  ode::log::Info("a_drag  [km/s^2] = ", a0.drag.transpose());
  ode::log::Info("a_srp   [km/s^2] = ", a0.srp.transpose());
  ode::log::Info("a_rel   [km/s^2] = ", a0.rel.transpose());
  ode::log::Info("a_tidal [km/s^2] = ", a0.tidal.transpose());

  return 0;
}
