"""G0: performance-map contract, mass/CoM/inertia model, r x F wrench, control authority."""
import json

import numpy as np
import pytest

from rocket_dt.control.allocation import authority_report
from rocket_dt.dynamics.mass import MassModel
from rocket_dt.dynamics.wrench import PointForce, total_wrench_body
from rocket_dt.propulsion import nozzle1d
from rocket_dt.propulsion.adapter import ActuatorState, PropulsionAdapter, gimbal_direction
from rocket_dt.propulsion.performance_map import EnginePerformanceMap, OutOfMapRange, build_ideal_map
from rocket_dt.schemas import EngineDesign, SchemaError, VehicleConfig


@pytest.fixture(scope="module")
def maps(engines):
    return {n: build_ideal_map(e) for n, e in engines.items()}


def test_map_nodes_exact_and_bounded(engines, maps):
    m, e = maps["conical_A"], engines["conical_A"]
    t, p = m.throttle[3], m.ambient_pressure_pa[5]
    f, mdot, isp, ok = m.lookup(t, p)
    ref = nozzle1d.evaluate(e, t, p)
    assert f == pytest.approx(ref.thrust_n) and mdot == pytest.approx(ref.mass_flow_kg_s) and ok
    with pytest.raises(OutOfMapRange):
        m.lookup(1.05, 101325)
    with pytest.raises(OutOfMapRange):
        m.lookup(0.7, 120000)
    thr = m.throttle_for_thrust(0.5 * (m.lookup(0.5, 95000)[0] + m.lookup(0.6, 95000)[0]), 95000)
    assert 0.5 < thr < 0.6


def test_map_invalid_cells_propagate(maps):
    assert not maps["bell_C"].lookup(0.4, 101325)[3]
    assert maps["conical_A"].lookup(0.4, 101325)[3]


def test_map_hash_roundtrip_and_tamper(tmp_path, maps):
    path = tmp_path / "m.json"
    h = maps["bell_B"].save(path)
    assert EnginePerformanceMap.load(path).content_hash() == h
    d = json.loads(path.read_text())
    d["values"]["thrust_n"][0][0] *= 1.01
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        EnginePerformanceMap.load(path)


def test_schema_rejects_bad_inputs(engines, vehicle):
    d = engines["conical_A"].to_dict()
    with pytest.raises(SchemaError):
        EngineDesign.from_dict({**d, "expansion_ratio": 0.8})
    with pytest.raises(SchemaError):
        EngineDesign.from_dict({**d, "exit_area_m2": 1.0})  # inconsistent with At*eps
    with pytest.raises(SchemaError):
        EngineDesign.from_dict({**d, "units": "imperial"})
    vd = vehicle.to_dict()
    with pytest.raises(SchemaError):
        VehicleConfig.from_dict({**vd, "frame_convention": "NED"})


def test_mass_properties(vehicle, engines):
    mm = MassModel(vehicle, engines["conical_A"])
    full = mm.evaluate(450.0, 10.0)
    empty = mm.evaluate(0.0, 10.0)
    assert full.mass_kg == pytest.approx(mm.dry_mass_kg + 460.0)
    assert empty.mass_kg == pytest.approx(mm.dry_mass_kg + 10.0)
    for s in (full, empty):
        assert np.allclose(s.inertia_com_body, s.inertia_com_body.T)
        assert np.all(np.linalg.eigvalsh(s.inertia_com_body) > 0)
        assert np.allclose(s.com_body_m[:2], 0)
    # CoM path is continuous as propellant drains
    zs = [mm.evaluate(m, 10.0).com_body_m[2] for m in np.linspace(450, 0, 200)]
    assert np.max(np.abs(np.diff(zs))) < 0.01
    # engine swap changes mass properties: B (short bell) lighter, C (wide bell) heavier than A
    em = {n: MassModel(vehicle.with_engine(n), e).engine_mass_kg for n, e in engines.items()}
    assert em["bell_B"] < em["conical_A"] < em["bell_C"]
    with pytest.raises(ValueError):
        MassModel(vehicle, engines["bell_B"])  # vehicle says conical_A


def test_wrench_central_offset_and_rcs_pair(vehicle):
    com = np.array([0.0, 0.0, -0.2])
    f, tau = total_wrench_body([PointForce("main", np.array([0, 0, 1e4]), np.array([0, 0, -3.0]))], com)
    assert np.allclose(tau, 0) and np.allclose(f, [0, 0, 1e4])
    r, F = np.array([0.1, 0.0, -3.0]), np.array([0, 0, 1e4])
    _, tau = total_wrench_body([PointForce("off", F, r)], com)
    assert np.allclose(tau, np.cross(r - com, F))
    assert tau[1] == pytest.approx(-0.1 * 1e4)  # +x offset of +z thrust -> negative pitch about y
    rcs = {t.id: t for t in vehicle.layout.rcs}
    pair = [PointForce(i, rcs[i].max_thrust_n * np.array(rcs[i].force_direction_body),
                       np.array(rcs[i].position_body_m)) for i in ("rcs_01", "rcs_03")]
    f, tau = total_wrench_body(pair, com)
    assert np.allclose(f, 0) and np.allclose(tau, [0, 0, 2 * 0.6 * 100.0])


def test_gimbal_torque_sign_and_magnitude(vehicle, engines, maps):
    ad = PropulsionAdapter(vehicle, maps["conical_A"])
    out = ad.thruster_forces(ActuatorState(True, 1.0, np.array([0.0, 0.05])), 101325, 400, 10)
    com = np.array([0.0, 0.0, -0.2])
    _, tau = total_wrench_body(out.forces, com)
    lever = com[2] - (-3.0)
    assert tau[1] == pytest.approx(-lever * out.main_thrust_n * np.sin(0.05), rel=1e-9)
    assert np.allclose(gimbal_direction(0, 0), [0, 0, 1])


def test_control_authority(vehicle):
    com, F = np.array([0, 0, -0.2]), 1.2e4
    main_only = authority_report(vehicle, com, F, include_rcs=False)
    assert main_only.rank == 3 and "tau_z" in main_only.missing and not main_only.full_authority
    full = authority_report(vehicle, com, F)
    assert full.rank == 4 and full.full_authority
    failed = np.array([True, False, True, False])  # both +roll thrusters out
    degraded = authority_report(vehicle, com, F, rcs_failed=failed)
    assert not degraded.full_authority and "tau_z" in degraded.missing


def test_e0_matches_hand_calculation():
    """E0 numbers must equal independent hand formulas: a_z = F cos(dy)/m - g0, alpha_y = tau_y / I_yy."""
    from pathlib import Path

    from rocket_dt.dynamics.frames import G0
    from rocket_dt.evaluation.experiments import e0_sensitivity

    rows = e0_sensitivity(Path(__file__).resolve().parents[1], throttle=1.0, gimbal_rad=(0.0, 0.035))
    for r in rows:
        m = None
        # recover mass from the reported engine/CoM data via the same vehicle file
        from rocket_dt.dynamics.mass import MassModel
        from rocket_dt.schemas import load_engine, load_vehicle
        root = Path(__file__).resolve().parents[1]
        v = load_vehicle(root / "configs/vehicle_v0.yaml").with_engine(r["engine"])
        ms = MassModel(v, load_engine(root / f"configs/engines/{r['engine']}.yaml")).evaluate(450.0, 10.0)
        m = ms.mass_kg
        lever = ms.com_body_m[2] - (-3.0)
        assert r["acc_z_m_s2"] == pytest.approx(r["thrust_n"] * np.cos(0.035) / m - G0, rel=1e-9)
        tau_y = -lever * r["thrust_n"] * np.sin(0.035)
        assert r["ang_acc_y_rad_s2"] == pytest.approx(tau_y / ms.inertia_com_body[1, 1], rel=1e-6)
    a, b, c = (next(x for x in rows if x["engine"] == n) for n in ("conical_A", "bell_B", "bell_C"))
    assert c["acc_z_m_s2"] < a["acc_z_m_s2"] < b["acc_z_m_s2"]  # design change propagates to vehicle response
