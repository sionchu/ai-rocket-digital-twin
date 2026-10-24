"""ADR-0011: variable-mass compensation on a fixed-mass body reproduces the variable-mass reference integrator,
and the aero model behaves physically (signs, magnitudes, damping)."""
from pathlib import Path

import numpy as np
import pytest

from rocket_dt.dynamics.aero import AeroModel, aero_wrench
from rocket_dt.dynamics.frames import GRAVITY_WORLD, quat_derivative, quat_from_axis_angle, quat_normalize, quat_to_rotmat
from rocket_dt.dynamics.mass import MassModel
from rocket_dt.dynamics.reference_6dof import IDX_P, IDX_Q, IDX_V, IDX_W, Reference6DoF, StepInputs, make_state
from rocket_dt.dynamics.varmass import compensated_wrench
from rocket_dt.dynamics.wrench import PointForce, total_wrench_body
from rocket_dt.schemas import load_engine, load_vehicle

ROOT = Path(__file__).resolve().parents[1]
E = (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]))
MDOT = 6.0  # kg/s, constant burn (about the landing burn rate)


@pytest.fixture(scope="module")
def setup():
    veh = load_vehicle(ROOT / "configs/vehicle_v0.yaml").with_engine("conical_A")
    mm = MassModel(veh, load_engine(ROOT / "configs/engines/conical_A.yaml"))
    aero = AeroModel.load(ROOT / "configs/aero_v0.yaml", veh.mass.structure_radius_m, veh.mass.structure_length_m)
    return veh, mm, aero


def _forces(veh, t, m_prop, mass):
    """Open-loop thrust ~1.05 g with a gimbal oscillation and a roll RCS pulse."""
    gx, gy = 0.04 * np.sin(1.3 * t), 0.03 * np.cos(0.9 * t)
    d = np.array([np.sin(gy) * np.cos(gx), -np.sin(gx), np.cos(gx) * np.cos(gy)])
    pivot = np.asarray(veh.layout.main_engine.gimbal_pivot_body_m, float)
    out = [PointForce("main", 1.05 * mass * 9.80665 * d, pivot)]
    if 2.0 < t < 3.0:
        out.append(PointForce("rcs", np.array([0.0, 100.0, 0.0]), np.array([0.6, 0.0, 3.0])))
    return out


def _run_reference(veh, mm, aero, wind, x0, dt, n):
    dyn = Reference6DoF(mm, None, aero)
    x, traj = x0.copy(), []
    for k in range(n):
        ms = mm.evaluate(x[13], x[14])
        u = StepInputs(_forces(veh, k * dt, x[13], ms.mass_kg), MDOT, 0.0, np.zeros(3), np.zeros(3), wind)
        x = dyn.step(x, u, dt)
        traj.append(np.concatenate([x[IDX_P], x[IDX_V], x[IDX_Q], x[IDX_W]]))
    return np.array(traj)


def _run_fixed_body(veh, mm, aero, wind, x0, dt, n, compensate=True):
    """What Isaac does: one rigid body with the full-propellant m0, I0, c0; state is the c0 point."""
    ms0 = mm.evaluate(mm.prop0, mm.rcs0)
    m0, c0, i0 = ms0.mass_kg, ms0.com_body_m, np.diag(ms0.inertia_com_body)
    k_aero = aero.arrays()
    q, w, mp, mr = x0[IDX_Q].copy(), x0[IDX_W].copy(), x0[13], x0[14]
    rot = quat_to_rotmat(q)
    ms = mm.evaluate(mp, mr)
    p = x0[IDX_P] - rot @ (ms.com_body_m - c0)  # c0 point
    ez = np.array([0.0, 0.0, 1.0])

    def rates(mp):
        cd, cdd = mm.com_z_rates(np.array([mp]), np.array([mr]), MDOT, 0.0)
        return cd[0] * ez, cdd[0] * ez

    v = x0[IDX_V] - rot @ (np.cross(w, ms.com_body_m - c0) + rates(mp)[0])
    traj = []

    def deriv(p, v, q, w, mp, forces):
        rot = quat_to_rotmat(q)
        ms = mm.evaluate(max(mp, 0.0), mr)
        d = ms.com_body_m - c0
        d_dot, d_ddot = rates(mp)
        pc, vc = p + rot @ d, v + rot @ (np.cross(w, d) + d_dot)  # true CoM state
        f_b, tau_b = total_wrench_body(forces, ms.com_body_m)
        fa, ta = aero_wrench(k_aero, pc[2:3], rot[None], vc[None], w[None], ms.com_body_m[None], wind[None])
        f_w, tau = rot @ (f_b + fa[0]), tau_b + ta[0]
        if compensate:
            f_app, tau_app, _ = compensated_wrench(f_w[None], tau[None], w[None], rot[None], np.array([ms.mass_kg]),
                                                   np.diag(ms.inertia_com_body)[None], d[None], m0, i0, E,
                                                   d_dot[None], d_ddot[None])
            f_app, tau_app = f_app[0], tau_app[0]
        else:
            f_app, tau_app = f_w, tau
        a = f_app / m0 + GRAVITY_WORLD
        w_dot = (tau_app - np.cross(w, i0 * w)) / i0
        return v, a, quat_derivative(q, w), w_dot, -MDOT

    t = 0.0
    for _ in range(n):
        s = (p, v, q, w, mp)
        fk = _forces(veh, t, mp, mm.evaluate(mp, mr).mass_kg)  # zero-order hold per step, as the reference
        k1 = deriv(*s, fk)
        k2 = deriv(*[a + 0.5 * dt * b for a, b in zip(s, k1)], fk)
        k3 = deriv(*[a + 0.5 * dt * b for a, b in zip(s, k2)], fk)
        k4 = deriv(*[a + dt * b for a, b in zip(s, k3)], fk)
        p, v, q, w, mp = [a + dt / 6 * (b1 + 2 * b2 + 2 * b3 + b4) for a, b1, b2, b3, b4 in zip(s, k1, k2, k3, k4)]
        q = quat_normalize(q)
        t += dt
        rot = quat_to_rotmat(q)
        d = mm.evaluate(mp, mr).com_body_m - c0
        traj.append(np.concatenate([p + rot @ d, v + rot @ (np.cross(w, d) + rates(mp)[0]), q, w]))
    return np.array(traj)


def test_compensation_matches_variable_mass_reference(setup):
    veh, mm, aero = setup
    dt, n = 1 / 200, 1200  # 6 s, 36 kg burned; the open-loop vehicle tumbles to ~70 deg, a stress case
    wind = np.array([8.0, -3.0, 0.0])
    x0 = make_state([3.0, -2.0, 60.0], [1.0, 0.5, -6.0], quat_from_axis_angle([0, 1, 0], 0.05), [0, 0, 0.1],
                    mm.prop0, mm.rcs0)
    ref = _run_reference(veh, mm, aero, wind, x0, dt, n)
    comp = _run_fixed_body(veh, mm, aero, wind, x0, dt, n, compensate=True)
    raw = _run_fixed_body(veh, mm, aero, wind, x0, dt, n, compensate=False)
    err_c = np.abs(comp[:, :3] - ref[:, :3]).max()
    err_raw = np.abs(raw[:, :3] - ref[:, :3]).max()
    att_c = np.abs(comp[:, 10:13] - ref[:, 10:13]).max()
    # same equations rearranged: the residual is the zero-order-hold discretisation (it halves when dt halves:
    # 1.9 cm at 1/200 s, 0.9 cm at 1/400 s), against ~1.1 m without compensation
    assert err_c < 0.025, err_c
    assert att_c < 0.01, att_c
    assert err_raw > 0.5, err_raw


def test_aero_signs_and_magnitudes(setup):
    _, mm, aero = setup
    k = aero.arrays()
    com = np.array([[0.0, 0.0, mm.evaluate(mm.prop0, mm.rcs0).com_body_m[2]]])
    one = lambda v, w, wind: aero_wrench(k, np.array([10.0]), np.eye(3)[None], np.array([v], float),  # noqa: E731
                                         np.array([w], float), com, np.array([wind], float))
    f, t = one([0, 0, -6], [0, 0, 0], [0, 0, 0])  # tail-first descent: drag pushes up, no moment
    assert f[0, 2] > 0 and abs(f[0, :2]).max() < 1e-9 and abs(t).max() < 1e-9
    f, t = one([0, 0, 0], [0, 0, 0], [10, 0, 0])  # crosswind pushes downwind; CoP above CoM tips the nose downwind
    assert f[0, 0] > 0 and t[0, 1] > 0
    f, t = one([0, 0, 0], [0, 0.5, 0], [0, 0, 0])  # pitch rate is damped
    assert t[0, 1] < 0
    v1500 = aero.wind_speed_for_force(1500.0)  # v0 wind force maps to a plausible wind speed
    assert 12.0 < v1500 < 20.0
    f, _ = aero_wrench(k, np.array([aero.wind_ref_height_m]), np.eye(3)[None], np.zeros((1, 3)), np.zeros((1, 3)),
                       np.zeros((1, 3)), np.array([[v1500, 0, 0]]))
    assert 1200.0 < f[0, 0] < 1700.0  # shear profile and ISA density at 10 m keep it near 1500 N


def test_batched_mass_matches_evaluate(setup):
    _, mm, _ = setup
    mp, mr = np.array([450.0, 300.0, 100.0, 0.0]), np.array([10.0, 5.0, 0.0, 0.0])
    m, cz, ixx, izz = mm.batched(mp, mr)
    for i in range(4):
        s = mm.evaluate(mp[i], mr[i])
        assert m[i] == pytest.approx(s.mass_kg) and cz[i] == pytest.approx(s.com_body_m[2])
        assert ixx[i] == pytest.approx(s.inertia_com_body[0, 0]) and izz[i] == pytest.approx(s.inertia_com_body[2, 2])
