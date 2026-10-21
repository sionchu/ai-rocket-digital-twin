"""Torch mirrors (Isaac env) vs numpy references. Skipped where torch is not installed."""
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rocket_dt.dynamics.frames import isa_pressure  # noqa: E402
from rocket_dt.dynamics.wrench import PointForce, total_wrench_body  # noqa: E402
from rocket_dt.isaac_env import torch_models as tm  # noqa: E402
from rocket_dt.isaac_env.wrench_bridge import batched_wrench_torch  # noqa: E402
from rocket_dt.propulsion.actuators import ActuatorModel  # noqa: E402
from rocket_dt.propulsion.adapter import gimbal_direction  # noqa: E402
from rocket_dt.propulsion.performance_map import build_ideal_map  # noqa: E402
from rocket_dt.schemas import load_engine, load_vehicle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _setup(name="conical_A"):
    veh = load_vehicle(ROOT / "configs/vehicle_v0.yaml").with_engine(name)
    eng = load_engine(ROOT / f"configs/engines/{name}.yaml")
    return veh, eng


@pytest.mark.parametrize("name", ["conical_A", "bell_C"])
def test_map_lookup_matches_numpy(name):
    _, eng = _setup(name)
    m = build_ideal_map(eng)
    tmap = tm.TorchPerfMap(m, "cpu")
    rng = np.random.default_rng(0)
    thr = rng.uniform(eng.throttle_min, eng.throttle_max, 200)
    pa = rng.uniform(m.ambient_pressure_pa[0], m.ambient_pressure_pa[-1], 200)
    f, md, ok = tmap.lookup(torch.tensor(thr, dtype=torch.float32), torch.tensor(pa, dtype=torch.float32))
    for k in range(200):
        rf, rmd, _, rok = m.lookup(thr[k], pa[k])
        assert abs(f[k].item() - rf) / rf < 1e-5 and abs(md[k].item() - rmd) / rmd < 1e-5
        assert bool(ok[k]) == rok
    _, _, ok_out = tmap.lookup(torch.tensor([1.2]), torch.tensor([101325.0]))
    assert not bool(ok_out[0])  # outside the box -> invalid, never extrapolated silently


def test_isa_and_gimbal():
    alt = torch.tensor([0.0, 50.0, 1000.0], dtype=torch.float64)
    assert torch.allclose(tm.isa_pressure(alt), torch.tensor([isa_pressure(a) for a in alt.tolist()], dtype=torch.float64))
    g = torch.tensor([[0.05, -0.08]])
    assert np.allclose(tm.gimbal_direction(g)[0].numpy(), gimbal_direction(0.05, -0.08), atol=1e-7)


def test_actuators_match_numpy():
    veh, eng = _setup()
    dt, n_sub = 1 / 200, 4
    ref = ActuatorModel(veh, eng, throttle0=0.7)
    ta = tm.TorchActuators(veh, eng, 1, n_sub * dt, "cpu")
    ta.reset(torch.tensor([0]), 0.7)
    rng = np.random.default_rng(1)
    t = 0.0
    for k in range(400):
        if k % n_sub == 0:
            thr = rng.uniform(0.3, 1.1)
            gim = rng.uniform(-0.2, 0.2, 2)
            on = rng.uniform(0, n_sub * dt, 4) * (rng.uniform(size=4) < 0.6)
            ref.command(t, True, thr, gim, on)
            ta.command(torch.tensor([t]), torch.tensor([thr]), torch.tensor([gim], dtype=torch.float32),
                       torch.tensor([on], dtype=torch.float32))
        s = ref.step(t, dt)
        thr_t, gim_t, frac_t = ta.step(torch.tensor([t]), dt)
        assert abs(thr_t.item() - s.throttle) < 1e-5
        assert np.allclose(gim_t[0].numpy(), s.gimbal_rad, atol=1e-6)
        assert np.allclose(frac_t[0].numpy(), s.rcs_fraction, atol=2e-4), (k, frac_t, s.rcs_fraction)
        t += dt


def test_torch_wrench_matches_reference():
    rng = np.random.default_rng(2)
    f, p = rng.normal(size=(3, 5, 3)), rng.normal(size=(3, 5, 3))
    com = rng.normal(size=(3, 3))
    F, T = batched_wrench_torch(torch.tensor(f), torch.tensor(p), torch.tensor(com))
    for i in range(3):
        rf, rt = total_wrench_body([PointForce(str(k), f[i, k], p[i, k]) for k in range(5)], com[i])
        assert np.allclose(F[i, 0].numpy(), rf) and np.allclose(T[i, 0].numpy(), rt)


def test_stacked_maps_select_per_env():
    maps = []
    for name in ("conical_A", "bell_B", "bell_C"):
        maps.append(build_ideal_map(_setup(name)[1]))
    tmap = tm.TorchPerfMap(maps, "cpu")
    thr = torch.tensor([0.5, 0.5, 0.5, 0.9])
    pa = torch.tensor([101325.0] * 4)
    f, _, ok = tmap.lookup(thr, pa, torch.tensor([0, 1, 2, 2]))
    for k, (m, t) in enumerate(zip([maps[0], maps[1], maps[2], maps[2]], thr.tolist())):
        rf, _, _, rok = m.lookup(t, 101325.0)
        assert abs(f[k].item() - rf) / rf < 1e-5 and bool(ok[k]) == rok


def test_aero_varmass_mass_torch_match_numpy():
    """ADR-0011 code is backend neutral: torch results equal the numpy (reference) results."""
    from rocket_dt.dynamics.aero import AeroModel, aero_wrench
    from rocket_dt.dynamics.mass import MassModel
    from rocket_dt.dynamics.varmass import compensated_wrench
    from rocket_dt.schemas import load_engine, load_vehicle
    root = Path(__file__).resolve().parents[1]
    veh = load_vehicle(root / "configs/vehicle_v0.yaml").with_engine("conical_A")
    mm = MassModel(veh, load_engine(root / "configs/engines/conical_A.yaml"))
    am = AeroModel.load(root / "configs/aero_v0.yaml", 0.6, 7.0)
    rng = np.random.default_rng(0)
    n = 5
    a = np.linalg.qr(rng.normal(size=(n, 3, 3)))[0]
    rot = a * np.sign(np.linalg.det(a))[:, None, None]
    p_z, v, w = rng.uniform(1, 60, n), rng.normal(0, 5, (n, 3)), rng.normal(0, 0.3, (n, 3))
    com, wind = np.c_[np.zeros((n, 2)), rng.uniform(-0.4, -0.2, n)], rng.normal(0, 8, (n, 3)) * [1, 1, 0]
    t = lambda x: torch.tensor(np.asarray(x), dtype=torch.float64)  # noqa: E731
    f_np, tq_np = aero_wrench(am.arrays(), p_z, rot, v, w, com, wind)
    f_t, tq_t = aero_wrench(am.arrays(t), t(p_z), t(rot), t(v), t(w), t(com), t(wind))
    np.testing.assert_allclose(f_t.numpy(), f_np, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(tq_t.numpy(), tq_np, rtol=1e-9, atol=1e-9)
    mp, mr = rng.uniform(0, 450, n), rng.uniform(0, 10, n)
    for x_np, x_t in zip(mm.batched(mp, mr), mm.batched(t(mp), t(mr))):
        np.testing.assert_allclose(x_t.numpy(), x_np, rtol=1e-12)
    e_np, e_t = tuple(np.eye(3)), tuple(torch.eye(3, dtype=torch.float64))
    m, cz, ixx, izz = mm.batched(mp, mr)
    i3 = np.c_[ixx, ixx, izz]
    d = np.c_[np.zeros((n, 2)), cz - mm.evaluate(mm.prop0, mm.rcs0).com_body_m[2]]
    args_np = (v * 1000, w * 500, w, rot, m, i3, d, 1207.0, np.array([4016.0, 4016.0, 120.0]), e_np, d * 0.01, d * 0.001)
    out_np = compensated_wrench(*args_np)
    out_t = compensated_wrench(*[t(x) if isinstance(x, np.ndarray) else x for x in args_np[:9]], e_t,
                               t(d * 0.01), t(d * 0.001))
    for x_t, x_np in zip(out_t, out_np):
        np.testing.assert_allclose(x_t.numpy(), x_np, rtol=1e-9, atol=1e-9)
