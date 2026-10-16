"""G0/P1: frame conventions, ideal 1D nozzle vs. analytic references, contour geometry."""
import numpy as np
import pytest

from rocket_dt.dynamics import frames as fr
from rocket_dt.propulsion import geometry, nozzle1d


def test_quat_rotation_convention():
    q = fr.quat_from_axis_angle([0, 0, 1], np.pi / 2)
    R = fr.quat_to_rotmat(q)
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0])  # body x -> world y
    assert np.allclose(R.T @ R, np.eye(3))
    q2 = fr.quat_from_axis_angle([1, 0, 0], 0.3)
    assert np.allclose(fr.quat_to_rotmat(fr.quat_mul(q, q2)), R @ fr.quat_to_rotmat(q2))


def test_quat_kinematics_constant_rate():
    w = np.array([0.0, 0.0, 0.7])
    q, dt = np.array([1.0, 0, 0, 0]), 1e-3
    for _ in range(1000):  # RK4 on q_dot = 0.5 q (x) [0, w]
        k1 = fr.quat_derivative(q, w); k2 = fr.quat_derivative(q + dt / 2 * k1, w)
        k3 = fr.quat_derivative(q + dt / 2 * k2, w); k4 = fr.quat_derivative(q + dt * k3, w)
        q = fr.quat_normalize(q + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
    assert np.allclose(q, fr.quat_from_axis_angle([0, 0, 1], 0.7), atol=1e-10)


def test_isa_sea_level_and_range():
    assert fr.isa_pressure(0.0) == pytest.approx(101325.0)
    assert fr.isa_pressure(1000.0) == pytest.approx(89874.6, rel=1e-4)
    with pytest.raises(ValueError):
        fr.isa_pressure(20000.0)


def test_area_mach_textbook_values():
    assert nozzle1d.area_ratio_from_mach(2.0, 1.4) == pytest.approx(1.6875, rel=1e-4)
    assert nozzle1d.area_ratio_from_mach(1.0, 1.4) == pytest.approx(1.0)
    assert nozzle1d.supersonic_mach_from_area_ratio(1.6875, 1.4) == pytest.approx(2.0, rel=1e-6)
    assert nozzle1d.vandenkerckhove(1.4) == pytest.approx(0.6847, rel=1e-3)


def test_thrust_matches_independent_thrust_coefficient(engines):
    e = engines["conical_A"]
    g, eps = e.gas.gamma, e.expansion_ratio
    for thr, pa in [(1.0, 101325.0), (0.7, 80000.0), (1.0, 0.0)]:
        p = nozzle1d.evaluate(e, thr, pa)
        pr = p.exit_pressure_pa / p.chamber_pressure_pa
        cf_mom = np.sqrt(2 * g * g / (g - 1) * (2 / (g + 1)) ** ((g + 1) / (g - 1)) * (1 - pr ** ((g - 1) / g)))
        lam = 0.5 * (1 + np.cos(e.conical_half_angle_rad))
        cf = lam * cf_mom + (p.exit_pressure_pa - pa) / p.chamber_pressure_pa * eps
        assert p.thrust_n == pytest.approx(cf * p.chamber_pressure_pa * e.throat_area_m2, rel=1e-9)
        assert p.isp_s == pytest.approx(p.thrust_n / (p.mass_flow_kg_s * fr.G0), rel=1e-12)
        # F = mdot*Ve*lambda + (Pe - Pa) Ae  (thrust equation, S13)
        assert p.thrust_n == pytest.approx(lam * p.mass_flow_kg_s * p.exit_velocity_m_s
                                           + (p.exit_pressure_pa - pa) * e.exit_area_m2, rel=1e-12)


def test_mass_flow_linear_in_throttle_and_same_for_same_throat(engines):
    a, b = engines["conical_A"], engines["bell_B"]
    m1 = nozzle1d.evaluate(a, 1.0, 1e5).mass_flow_kg_s
    assert nozzle1d.evaluate(a, 0.5, 1e5).mass_flow_kg_s == pytest.approx(0.5 * m1)
    assert nozzle1d.evaluate(b, 1.0, 1e5).mass_flow_kg_s == pytest.approx(m1)


def test_bell_beats_conical_same_eps_and_flags_separation(engines):
    a, b, c = engines["conical_A"], engines["bell_B"], engines["bell_C"]
    assert nozzle1d.evaluate(b, 1.0, 101325).thrust_n > nozzle1d.evaluate(a, 1.0, 101325).thrust_n
    low = nozzle1d.evaluate(c, 0.4, 101325)
    assert not low.valid and "separation" in low.invalid_reason
    assert nozzle1d.evaluate(a, 0.4, 101325).valid
    with pytest.raises(ValueError):
        nozzle1d.evaluate(a, 0.2, 101325)  # below throttle range


def test_contour_area_ratio_and_lengths(engines):
    cs = {n: geometry.build_contour(e) for n, e in engines.items()}
    for n, e in engines.items():
        rt = np.sqrt(e.throat_area_m2 / np.pi)
        c = cs[n]
        assert c.r_m[c.throat_index] == pytest.approx(rt, rel=1e-9)
        assert (c.exit_radius_m / rt) ** 2 == pytest.approx(e.expansion_ratio, rel=1e-6)  # area ratio, not radius ratio
        assert c.r_m[c.throat_index:].min() == pytest.approx(rt, rel=1e-9)
        assert np.all(np.diff(c.r_m[c.throat_index:]) >= -1e-12)
    assert cs["bell_B"].nozzle_length_m < cs["conical_A"].nozzle_length_m  # 80% bell is shorter
    assert cs["bell_C"].nozzle_length_m > cs["bell_B"].nozzle_length_m
