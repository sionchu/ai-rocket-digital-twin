"""Closed-loop scenario runner on the Python reference integrator.

Produces a ControlEpisode-style record: seed, design ids, initial state,
commands, applied thrust, truth trajectory, events and termination reason.
Physics and control rates are fixed per scenario; this integrator is the
single authoritative integrator for these runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pathlib import Path

from rocket_dt.control.baseline import CascadedPD, PDGains
from rocket_dt.dynamics.aero import AeroModel
from rocket_dt.dynamics.frames import (isa_pressure, quat_from_axis_angle, quat_mul, quat_normalize, quat_to_rotmat,
                                       tilt_angle)
from rocket_dt.dynamics.mass import MassModel
from rocket_dt.dynamics.reference_6dof import (IDX_MP, IDX_MR, IDX_P, IDX_Q, IDX_V, IDX_W, Reference6DoF,
                                               StepInputs, make_state)
from rocket_dt.propulsion.actuators import ActuatorModel
from rocket_dt.propulsion.adapter import PropulsionAdapter
from rocket_dt.propulsion.performance_map import EnginePerformanceMap
from rocket_dt.schemas import EngineDesign, VehicleConfig

# Provisional landing thresholds (ADR-0005): to be agreed before G4 comparisons.
LANDING_LIMITS = {"vz_max_m_s": 2.0, "vh_max_m_s": 1.0, "tilt_max_rad": float(np.radians(10)),
                  "pos_err_max_m": 2.0, "final_tilt_max_rad": float(np.radians(15))}


@dataclass
class Scenario:
    name: str
    mode: str  # hover | landing
    duration_s: float
    com_position_w: list
    velocity_w: list
    tilt_axis: list = field(default_factory=lambda: [1.0, 0.0, 0.0])
    tilt_rad: float = 0.0
    omega_b: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    target_w: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    physics_hz: float = 200.0
    control_hz: float = 50.0
    settle_s: float = 3.0
    disturbance_force_w: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    disturbance_torque_b: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rcs_failed: list = field(default_factory=list)
    fault_known_to_controller: bool = True  # False: actuator fails silently, allocator still uses it
    seed: int = 0
    propellant_mass_kg: float | None = None  # override initial load
    # ADR-0011 (all off by default = v0 behaviour)
    aero: bool = False  # quasi-steady aero (configs/aero_v0.yaml); the wind becomes a wind velocity
    wind_ref_w: list | None = None  # wind at the reference height; None + aero -> from disturbance_force_w
    gust_amp: float = 0.0  # sinusoidal gust: force x (1 + a sin), or wind speed x sqrt(1 + a sin) with aero
    gust_period_s: float = 4.0
    gust_phase: float = 0.0
    noise_pos_m: float = 0.0  # controller measurement noise (1 sigma) and latency in control steps
    noise_vel_m_s: float = 0.0
    noise_att_rad: float = 0.0
    noise_rate_rad_s: float = 0.0
    obs_delay_steps: int = 0


def run_scenario(sc: Scenario, vehicle: VehicleConfig, mass_engine: EngineDesign, map_engine: EngineDesign,
                 perf_map: EnginePerformanceMap, gains: PDGains | None = None):
    if map_engine.engine_design_id != perf_map.engine_design_id:
        raise ValueError("map/engine id mismatch")
    veh = vehicle.with_engine(map_engine.engine_design_id)
    mm = MassModel(veh.with_engine(mass_engine.engine_design_id), mass_engine)
    aero = AeroModel.load(Path(__file__).resolve().parents[3] / "configs/aero_v0.yaml", veh.mass.structure_radius_m,
                          veh.mass.structure_length_m) if sc.aero else None
    dyn = Reference6DoF(mm, veh.gear, aero)
    adapter = PropulsionAdapter(veh, perf_map)
    act = ActuatorModel(veh, map_engine, throttle0=0.7)
    dt = 1.0 / sc.physics_hz
    n_sub = int(round(sc.physics_hz / sc.control_hz))
    ctrl = CascadedPD(veh, mm, perf_map, n_sub * dt, gains)
    failed = np.zeros(len(veh.layout.rcs), dtype=bool)
    ids = [t.id for t in veh.layout.rcs]
    for rid in sc.rcs_failed:
        failed[ids.index(rid)] = True
    act.failed |= failed

    m_prop0 = veh.mass.propellant_mass_kg if sc.propellant_mass_kg is None else sc.propellant_mass_kg
    q0 = quat_from_axis_angle(sc.tilt_axis, sc.tilt_rad)
    x = make_state(sc.com_position_w, sc.velocity_w, q0, sc.omega_b, m_prop0, veh.mass.rcs_gas_mass_kg)
    feet = np.array(veh.gear.foot_positions_body_m)
    d_force = np.asarray(sc.disturbance_force_w, float)
    d_torque = np.asarray(sc.disturbance_torque_b, float)
    wind_ref = None
    if aero is not None:  # the v0 wind force becomes the wind speed that gives that force (ADR-0011)
        if sc.wind_ref_w is not None:
            wind_ref = np.asarray(sc.wind_ref_w, float)
        else:
            fm = np.linalg.norm(d_force)
            wind_ref = d_force / fm * aero.wind_speed_for_force(fm) if fm > 0 else np.zeros(3)
        d_force = np.zeros(3)
    rng = np.random.default_rng(20_000 + sc.seed)  # measurement noise stream
    noisy = sc.noise_pos_m > 0 or sc.obs_delay_steps > 0
    x_hist = []

    log = {k: [] for k in ("t", "p", "v", "q", "w", "m_prop", "m_rcs", "throttle", "gimbal", "rcs_frac",
                           "thrust", "contacts", "map_valid", "mass", "com_b")}
    events, reason = [], "timeout"
    touchdown = None
    t, k = 0.0, 0
    engine_on = True
    map_invalid_steps = 0
    n_steps = int(round(sc.duration_s / dt))
    while k < n_steps:
        ms = mm.evaluate(x[IDX_MP], x[IDX_MR])
        alt = float(x[IDX_P][2])
        pa = isa_pressure(max(alt, 0.0))
        if k % n_sub == 0:
            td_com_z = ms.com_body_m[2] - feet[:, 2].min()
            if engine_on:
                x_c = x
                if noisy:
                    x_hist.append(x.copy())
                    x_c = x_hist[max(0, len(x_hist) - 1 - sc.obs_delay_steps)].copy()
                    x_c[IDX_P] += rng.normal(0, sc.noise_pos_m, 3)
                    x_c[IDX_V] += rng.normal(0, sc.noise_vel_m_s, 3)
                    x_c[IDX_W] += rng.normal(0, sc.noise_rate_rad_s, 3)
                    ax = rng.normal(size=3)
                    dq = quat_from_axis_angle(ax / np.linalg.norm(ax), abs(rng.normal()) * sc.noise_att_rad)
                    x_c[IDX_Q] = quat_normalize(quat_mul(x_c[IDX_Q], dq))
                cmd = ctrl(x_c, pa, "hover" if sc.mode == "hover" else "descent", sc.target_w, td_com_z,
                           act.state.throttle, failed if sc.fault_known_to_controller else None)
                act.command(t, True, cmd.throttle, cmd.gimbal_rad, cmd.rcs_on_time_s)
            else:
                act.command(t, False, act.tmin, np.zeros(2), None)
        a = act.step(t, dt)
        out = adapter.thruster_forces(a, pa, x[IDX_MP], x[IDX_MR], failed)
        if not out.map_valid:
            map_invalid_steps += 1
        gust = 1.0 + sc.gust_amp * np.sin(2 * np.pi * t / sc.gust_period_s + sc.gust_phase)
        wind_t = None if wind_ref is None else wind_ref * np.sqrt(gust)
        x = dyn.step(x, StepInputs(out.forces, out.main_mass_flow_kg_s, out.rcs_mass_flow_kg_s, d_force * gust,
                                   d_torque, wind_t), dt)
        t += dt
        k += 1
        if k % n_sub == 0:
            for key, val in (("t", t), ("p", x[IDX_P]), ("v", x[IDX_V]), ("q", x[IDX_Q]), ("w", x[IDX_W]),
                             ("m_prop", x[IDX_MP]), ("m_rcs", x[IDX_MR]), ("throttle", a.throttle if a.engine_on else 0.0),
                             ("gimbal", a.gimbal_rad.copy()), ("rcs_frac", a.rcs_fraction.copy()),
                             ("thrust", out.main_thrust_n), ("contacts", dyn.last_contacts),
                             ("map_valid", out.map_valid), ("mass", ms.mass_kg), ("com_b", ms.com_body_m)):
                log[key].append(np.array(val, dtype=float))
        tilt = tilt_angle(x[IDX_Q])
        if tilt > np.radians(60):
            reason = "tip_over_or_loss_of_control"
            events.append((t, reason))
            break
        if x[IDX_MP] <= 0 and engine_on and sc.mode != "landing_done":
            events.append((t, "propellant_depleted"))
            if touchdown is None:
                reason = "propellant_depleted"
        if sc.mode == "landing":
            if touchdown is None and dyn.last_contacts > 0:
                v = x[IDX_V]
                touchdown = {"t": t, "vz": float(v[2]), "vh": float(np.linalg.norm(v[:2])), "tilt": float(tilt),
                             "pos_err": float(np.linalg.norm(x[IDX_P][:2] - np.asarray(sc.target_w)[:2])),
                             "omega": float(np.linalg.norm(x[IDX_W]))}
                engine_on = False
                events.append((t, "touchdown_engine_cutoff"))
                n_steps = min(n_steps, k + int(round(sc.settle_s / dt)))
            if x[IDX_P][2] < -1.0:
                reason = "below_ground"
                break
    metrics = _metrics(sc, log, touchdown, reason, events, m_prop0, map_invalid_steps, veh, mm)
    episode = {"scenario": sc.__dict__, "vehicle_engine_id": veh.engine_design_id,
               "mass_engine_id": mass_engine.engine_design_id, "map_hash": perf_map.content_hash(),
               "events": events}
    return {k2: np.array(v2) for k2, v2 in log.items()}, metrics, episode


def _metrics(sc, log, td, reason, events, m_prop0, map_invalid_steps, veh, mm):
    p = np.array(log["p"])
    t = np.array(log["t"])
    tilt = np.array([tilt_angle(q) for q in log["q"]])
    g = np.array(log["gimbal"])
    gmax = veh.layout.main_engine.max_gimbal_rad
    m = {"scenario": sc.name, "termination_reason": reason, "duration_s": float(t[-1]) if len(t) else 0.0,
         "fuel_used_kg": float(m_prop0 - log["m_prop"][-1]),
         "rcs_gas_used_kg": float(veh.mass.rcs_gas_mass_kg - log["m_rcs"][-1]),
         "engine_mass_kg": float(mm.engine_mass_kg), "dry_mass_kg": float(mm.dry_mass_kg),
         "max_tilt_deg": float(np.degrees(tilt.max())),
         "gimbal_saturation_frac": float(np.mean(np.any(np.abs(g) >= gmax - 1e-6, axis=1))),
         "map_invalid_steps": int(map_invalid_steps),
         # an episode that used map cells flagged invalid (e.g. separation risk) is not evidence
         "physics_valid": bool(map_invalid_steps == 0)}
    thr = np.array(log["throttle"])
    on = thr > 0
    m["mean_throttle_engine_on"] = float(thr[on].mean()) if on.any() else 0.0
    m["mean_thrust_n"] = float(np.array(log["thrust"])[on].mean()) if on.any() else 0.0
    if sc.mode == "hover":
        sel = t >= min(5.0, t[-1] / 2)
        err = p[sel] - np.asarray(sc.target_w)
        m.update({"pos_rms_m": float(np.sqrt(np.mean(np.sum(err ** 2, 1)))),
                  "alt_rms_m": float(np.sqrt(np.mean(err[:, 2] ** 2))),
                  "tilt_rms_deg": float(np.degrees(np.sqrt(np.mean(tilt[sel] ** 2)))),
                  "success": reason == "timeout"})
        if reason == "timeout":
            m["termination_reason"] = "completed"
    else:
        m["touchdown"] = td
        final_tilt = float(tilt[-1])
        ok = td is not None and reason == "timeout"
        fails = []
        if td is None:
            fails.append(reason if reason != "timeout" else "no_touchdown")
        else:
            L = LANDING_LIMITS
            if abs(td["vz"]) > L["vz_max_m_s"]:
                fails.append("vertical_speed")
            if td["vh"] > L["vh_max_m_s"]:
                fails.append("horizontal_speed")
            if td["tilt"] > L["tilt_max_rad"]:
                fails.append("touchdown_tilt")
            if td["pos_err"] > L["pos_err_max_m"]:
                fails.append("position_error")
            if final_tilt > L["final_tilt_max_rad"]:
                fails.append("tip_over")
            if reason != "timeout":
                fails.append(reason)
        m["success"] = bool(ok and not fails)
        m["failure_reasons"] = fails
        m["termination_reason"] = "landed" if m["success"] else (fails[0] if fails else reason)
        m["final_tilt_deg"] = float(np.degrees(final_tilt))
    return m
