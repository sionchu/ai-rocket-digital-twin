"""G0: reference 6-DoF integrator — Newton/Euler sanity, conservation, convergence, actuators."""
import numpy as np
import pytest

from rocket_dt.dynamics.frames import G0, quat_to_rotmat
from rocket_dt.dynamics.mass import MassModel
from rocket_dt.dynamics.reference_6dof import IDX_MP, IDX_P, IDX_Q, IDX_V, IDX_W, Reference6DoF, StepInputs, make_state
from rocket_dt.dynamics.wrench import PointForce
from rocket_dt.propulsion.actuators import ActuatorModel

Z3 = np.zeros(3)


def _u(forces=(), mdot=0.0, df=Z3, dt=Z3):
    return StepInputs(list(forces), mdot, 0.0, np.asarray(df, float), np.asarray(dt, float))


@pytest.fixture
def mm(vehicle, engines):
    return MassModel(vehicle, engines["conical_A"])


def run(dyn, x, u, dt, n):
    for _ in range(n):
        x = dyn.step(x, u, dt)
    return x


def test_free_fall_exact(mm):
    dyn = Reference6DoF(mm, None)
    x = run(dyn, make_state([0, 0, 100.0], Z3, [1, 0, 0, 0], Z3, 450, 10), _u(), 0.01, 300)
    assert x[IDX_P][2] == pytest.approx(100.0 - 0.5 * G0 * 9.0, abs=1e-9)
    assert x[IDX_V][2] == pytest.approx(-G0 * 3.0, abs=1e-9)


def test_static_hover_equilibrium(mm):
    ms = mm.evaluate(450, 10)
    dyn = Reference6DoF(mm, None)
    f = PointForce("main", np.array([0, 0, ms.mass_kg * G0]), np.array([0, 0, -3.0]))  # on axis -> no torque
    x = run(dyn, make_state([0, 0, 50.0], Z3, [1, 0, 0, 0], Z3, 450, 10), _u([f]), 0.005, 400)
    assert np.allclose(x[IDX_V], 0, atol=1e-9) and np.allclose(x[IDX_W], 0, atol=1e-12)


def test_offset_thrust_angular_acceleration(mm):
    ms = mm.evaluate(450, 10)
    r, F = np.array([0.0, 0.0, -3.0]), np.array([100.0, 0, 0])
    dyn = Reference6DoF(mm, None)
    dx = dyn.deriv(make_state([0, 0, 50.0], Z3, [1, 0, 0, 0], Z3, 450, 10), _u([PointForce("s", F, r)]))
    tau = np.cross(r - ms.com_body_m, F)
    assert np.allclose(dx[IDX_W], np.linalg.solve(ms.inertia_com_body, tau))


def test_constant_roll_torque(mm):
    izz = mm.evaluate(450, 10).inertia_com_body[2, 2]
    dyn = Reference6DoF(mm, None)
    x = run(dyn, make_state([0, 0, 50.0], Z3, [1, 0, 0, 0], Z3, 450, 10), _u(dt=[0, 0, 50.0]), 0.005, 400)
    assert x[IDX_W][2] == pytest.approx(50.0 / izz * 2.0, rel=1e-9)


def test_torque_free_conserves_momentum_and_energy(mm):
    ms = mm.evaluate(450, 10)
    dyn = Reference6DoF(mm, None)
    x0 = make_state([0, 0, 50.0], [3, -2, 1], [1, 0, 0, 0], [0.4, 0.0, 2.0], 450, 10)
    I = ms.inertia_com_body

    def inv(x):
        R = quat_to_rotmat(x[IDX_Q])
        return R @ (I @ x[IDX_W]), 0.5 * x[IDX_W] @ I @ x[IDX_W]

    L0, E0 = inv(x0)
    u = _u(df=[0, 0, ms.mass_kg * G0])  # cancel gravity: free body
    x = run(dyn, x0, u, 0.005, 2000)
    L1, E1 = inv(x)
    assert np.allclose(L1, L0, rtol=1e-7, atol=1e-6) and E1 == pytest.approx(E0, rel=1e-7)
    assert np.allclose(x[IDX_V], [3, -2, 1], atol=1e-9)  # linear momentum


def test_rk4_fourth_order_convergence(mm):
    ms = mm.evaluate(450, 10)
    dyn = Reference6DoF(mm, None)
    x0 = make_state([0, 0, 50.0], Z3, [1, 0, 0, 0], [0.3, -0.2, 1.0], 450, 10)
    u = _u([PointForce("m", np.array([0, 0, 1.3 * ms.mass_kg * G0]), np.array([0.05, 0, -3.0]))])
    T = 2.0
    ref = run(dyn, x0, u, T / 3200, 3200)
    errs = [np.linalg.norm(run(dyn, x0, u, T / n, n)[:13] - ref[:13]) for n in (50, 100)]
    assert errs[0] / errs[1] == pytest.approx(16.0, rel=0.25)


def test_propellant_consumption(mm):
    dyn = Reference6DoF(mm, None)
    x = run(dyn, make_state([0, 0, 50.0], Z3, [1, 0, 0, 0], Z3, 450, 10), _u(mdot=5.0), 0.005, 400)
    assert x[IDX_MP] == pytest.approx(440.0)
    x = run(dyn, x, _u(mdot=5.0), 0.005, 40000)
    assert x[IDX_MP] == 0.0  # clamped, never negative


def test_ground_contact_rest(vehicle, mm):
    dyn = Reference6DoF(mm, vehicle.gear)
    ms = mm.evaluate(450, 10)
    feet_z = min(p[2] for p in vehicle.gear.foot_positions_body_m)
    x = make_state([0, 0, ms.com_body_m[2] - feet_z + 0.3], Z3, [1, 0, 0, 0], Z3, 450, 10)
    x = run(dyn, x, _u(), 0.005, 1200)
    assert dyn.last_contacts == 4 and np.linalg.norm(x[IDX_V]) < 1e-3
    sink = (ms.mass_kg * G0 / 4) / vehicle.gear.stiffness_n_m
    assert x[IDX_P][2] == pytest.approx(ms.com_body_m[2] - feet_z - sink, abs=1e-3)


def test_rcs_impulse_exact_and_min_on_time(vehicle, engines):
    act = ActuatorModel(vehicle, engines["conical_A"])
    dt = 0.005
    act.command(0.0025, True, 0.7, [0, 0], [0.03, 0.005, 0.015, 0.0])  # unaligned with dt
    total = np.zeros(4)
    t = 0.0
    for _ in range(40):
        total += act.step(t, dt).rcs_fraction * dt
        t += dt
    assert total[0] == pytest.approx(0.03)   # full 30 ms pulse integrated exactly
    assert total[1] == 0.0                   # 5 ms < half min-on -> dropped
    assert total[2] == pytest.approx(0.02)   # 15 ms rounded up to 20 ms min impulse bit


def test_gimbal_rate_and_angle_limits(vehicle, engines):
    act = ActuatorModel(vehicle, engines["conical_A"])
    me = vehicle.layout.main_engine
    act.command(0.0, True, 0.7, [1.0, -1.0], None)
    s = act.step(0.0, 0.01)
    assert np.allclose(np.abs(s.gimbal_rad), me.max_gimbal_rate_rad_s * 0.01)
    for k in range(200):
        s = act.step(0.01 * k, 0.01)
    assert np.allclose(np.abs(s.gimbal_rad), me.max_gimbal_rad)
