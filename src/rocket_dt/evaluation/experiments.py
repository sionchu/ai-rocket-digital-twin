"""P0 experiments on the Python reference integrator with the PD baseline.

E0  open-loop sensitivity: identical actuator command on engines A/B/C ->
    initial linear and angular acceleration (does the design change F and tau?).
E1  same vehicle mass properties (engine A hardware), only the performance map changes.
E2  full system: each engine brings its own map AND its own mass/CoM/inertia.
S3  randomized landing suite (fixed seeds, evaluation only) per engine.
All maps here are ideal-1D (not CFD-verified); results are labelled accordingly.
"""
from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from rocket_dt.dynamics.frames import quat_from_axis_angle
from rocket_dt.dynamics.mass import MassModel
from rocket_dt.dynamics.reference_6dof import IDX_V, IDX_W, Reference6DoF, StepInputs, make_state
from rocket_dt.evaluation.simulate import Scenario, run_scenario
from rocket_dt.propulsion.adapter import ActuatorState, PropulsionAdapter
from rocket_dt.propulsion.performance_map import build_ideal_map
from rocket_dt.schemas import load_engine, load_vehicle

ENGINES = ("conical_A", "bell_B", "bell_C")


def load_all(root: Path):
    veh = load_vehicle(root / "configs/vehicle_v0.yaml")
    eng = {n: load_engine(root / f"configs/engines/{n}.yaml") for n in ENGINES}
    maps = {n: build_ideal_map(e) for n, e in eng.items()}
    return veh, eng, maps


def e0_sensitivity(root: Path, throttle=1.0, gimbal_rad=(0.0, 0.035)):
    veh, eng, maps = load_all(root)
    rows = []
    for n in ENGINES:
        v = veh.with_engine(n)
        mm = MassModel(v, eng[n])
        out = PropulsionAdapter(v, maps[n]).thruster_forces(
            ActuatorState(True, throttle, np.array(gimbal_rad)), 101325.0, 450.0, 10.0)
        dyn = Reference6DoF(mm, None)
        x = make_state([0, 0, 50.0], np.zeros(3), [1, 0, 0, 0], np.zeros(3), 450.0, 10.0)
        dx = dyn.deriv(x, StepInputs(out.forces, out.main_mass_flow_kg_s, 0.0, np.zeros(3), np.zeros(3)))
        ms = mm.evaluate(450.0, 10.0)
        rows.append({"engine": n, "thrust_n": out.main_thrust_n, "mdot_kg_s": out.main_mass_flow_kg_s,
                     "engine_mass_kg": mm.engine_mass_kg, "com_z_m": float(ms.com_body_m[2]),
                     "Ixx_kg_m2": float(ms.inertia_com_body[0, 0]),
                     "acc_z_m_s2": float(dx[IDX_V][2]), "ang_acc_y_rad_s2": float(dx[IDX_W][1]),
                     "map_valid": out.map_valid})
    return rows


def _load_sc(root: Path, name: str) -> Scenario:
    return Scenario(**yaml.safe_load(open(root / f"configs/scenarios/{name}.yaml")))


def _job(args):
    root, exp, n, scen_name, sc = args
    veh, eng, maps = load_all(root)
    mass_engine = eng["conical_A"] if exp == "E1" else eng[n]
    log, met, ep = run_scenario(sc, veh, mass_engine, eng[n], maps[n])
    return {"experiment": exp, "engine": n, "scenario": scen_name, "metrics": met,
            "map_hash": maps[n].content_hash(), "mass_engine": mass_engine.engine_design_id}, log, ep


def e1_e2(root: Path, workers: int = 4):
    jobs = [(root, exp, n, s, _load_sc(root, s)) for exp in ("E1", "E2") for n in ENGINES for s in ("hover", "landing")]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(_job, jobs))


def randomized_scenarios(base: Scenario, n: int, seed0: int = 1000):
    out = []
    for k in range(n):
        rng = np.random.default_rng(seed0 + k)
        ang = rng.uniform(0, 2 * np.pi)
        wind_dir = rng.uniform(0, 2 * np.pi)
        wind = rng.uniform(0, 300.0)
        out.append(replace(
            base, name=f"{base.name}_s{seed0 + k}", seed=seed0 + k,
            com_position_w=[float(8 * np.sqrt(rng.uniform()) * np.cos(ang)),
                            float(8 * np.sqrt(rng.uniform()) * np.sin(ang)), float(rng.uniform(40, 80))],
            velocity_w=[float(rng.uniform(-1.5, 1.5)), float(rng.uniform(-1.5, 1.5)), float(rng.uniform(-8, -4))],
            tilt_axis=[float(np.cos(ang + 1)), float(np.sin(ang + 1)), 0.0], tilt_rad=float(np.radians(rng.uniform(0, 4))),
            omega_b=[0.0, 0.0, float(rng.uniform(-0.2, 0.2))],
            disturbance_force_w=[float(wind * np.cos(wind_dir)), float(wind * np.sin(wind_dir)), 0.0]))
    return out


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def landing_suite(root: Path, n: int = 30, workers: int = 4, seed0: int = 1000, rcs_failed=()):
    base = _load_sc(root, "landing")
    base = replace(base, rcs_failed=list(rcs_failed))
    scs = randomized_scenarios(base, n, seed0)
    jobs = [(root, "S3", e, sc.name, sc) for e in ENGINES for sc in scs]
    with ProcessPoolExecutor(workers) as ex:
        res = [r[0] for r in ex.map(_job, jobs)]
    summary = {}
    for e in ENGINES:
        ms = [r["metrics"] for r in res if r["engine"] == e]
        succ = [m["success"] for m in ms]
        td = [m["touchdown"] for m in ms if m["touchdown"]]
        k = int(sum(succ))
        summary[e] = {
            "n": len(ms), "n_success": k, "success_rate": k / len(ms), "wilson95": wilson(k, len(ms)),
            "n_physics_valid": int(sum(m["physics_valid"] for m in ms)),
            "n_success_and_physics_valid": int(sum(m["success"] and m["physics_valid"] for m in ms)),
            "td_vz_abs_p95": float(np.percentile([abs(t["vz"]) for t in td], 95)) if td else None,
            "td_vh_p95": float(np.percentile([t["vh"] for t in td], 95)) if td else None,
            "td_pos_err_median": float(np.median([t["pos_err"] for t in td])) if td else None,
            "td_pos_err_p95": float(np.percentile([t["pos_err"] for t in td], 95)) if td else None,
            "td_pos_err_max": float(np.max([t["pos_err"] for t in td])) if td else None,
            "td_tilt_deg_max": float(np.degrees(np.max([t["tilt"] for t in td]))) if td else None,
            "fuel_mean_kg": float(np.mean([m["fuel_used_kg"] for m in ms])),
            "fuel_p95_kg": float(np.percentile([m["fuel_used_kg"] for m in ms], 95)),
            "gimbal_saturation_mean": float(np.mean([m["gimbal_saturation_frac"] for m in ms])),
            "failure_reasons": sorted({fr for m in ms for fr in m.get("failure_reasons", [])}),
        }
    return res, summary
