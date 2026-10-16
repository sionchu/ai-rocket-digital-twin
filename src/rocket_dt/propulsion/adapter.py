"""Propulsion Adapter: actuator state + performance map -> per-thruster point forces.

This is the single bridge from the propulsion track to any dynamics engine.
The Python reference integrator and the (future) Isaac Lab environment both
call `thruster_forces`; Isaac would pass the returned body-frame forces and
application points to its external wrench API. The adapter never integrates.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rocket_dt.dynamics.frames import G0
from rocket_dt.dynamics.wrench import PointForce
from rocket_dt.propulsion.performance_map import EnginePerformanceMap
from rocket_dt.schemas import VehicleConfig


def gimbal_direction(delta_x: float, delta_y: float) -> np.ndarray:
    """Unit thrust direction in body frame: R_y(delta_y) @ R_x(delta_x) @ +Z_body."""
    cx, sx = np.cos(delta_x), np.sin(delta_x)
    cy, sy = np.cos(delta_y), np.sin(delta_y)
    return np.array([cx * sy, -sx, cx * cy])


@dataclass
class ActuatorState:
    engine_on: bool = True
    throttle: float = 1.0  # actual (after lag), within map range while on
    gimbal_rad: np.ndarray = field(default_factory=lambda: np.zeros(2))
    rcs_fraction: np.ndarray | None = None  # per-thruster on-fraction of current step [0, 1]


@dataclass(frozen=True)
class PropulsionOutput:
    forces: list[PointForce]
    main_mass_flow_kg_s: float
    rcs_mass_flow_kg_s: float
    main_thrust_n: float
    map_valid: bool


class PropulsionAdapter:
    def __init__(self, vehicle: VehicleConfig, perf_map: EnginePerformanceMap):
        if vehicle.engine_design_id != perf_map.engine_design_id:
            raise ValueError(f"map {perf_map.engine_design_id} != vehicle engine {vehicle.engine_design_id}")
        self.vehicle = vehicle
        self.map = perf_map
        me = vehicle.layout.main_engine
        self.pivot = np.asarray(me.gimbal_pivot_body_m, float)
        if not np.allclose(me.thrust_axis_body, (0, 0, 1)):
            raise NotImplementedError("P0 supports thrust_axis_body = +Z only")
        self.rcs = vehicle.layout.rcs
        self.rcs_pos = np.array([t.position_body_m for t in self.rcs], float).reshape(-1, 3)
        self.rcs_dir = np.array([t.force_direction_body for t in self.rcs], float).reshape(-1, 3)
        self.rcs_fmax = np.array([t.max_thrust_n for t in self.rcs], float)
        self.rcs_mdot = self.rcs_fmax / (np.array([t.isp_s for t in self.rcs], float) * G0)

    def thruster_forces(self, act: ActuatorState, ambient_pressure_pa: float,
                        m_prop: float, m_rcs: float, rcs_failed: np.ndarray | None = None) -> PropulsionOutput:
        forces: list[PointForce] = []
        f_main, mdot_main, valid = 0.0, 0.0, True
        if act.engine_on and m_prop > 0:
            f_main, mdot_main, _, valid = self.map.lookup(act.throttle, ambient_pressure_pa)
            d = gimbal_direction(*act.gimbal_rad)
            forces.append(PointForce("main", f_main * d, self.pivot))
        mdot_rcs = 0.0
        if act.rcs_fraction is not None and m_rcs > 0 and len(self.rcs):
            frac = np.clip(act.rcs_fraction, 0.0, 1.0).copy()
            if rcs_failed is not None:
                frac[rcs_failed] = 0.0
            for k, t in enumerate(self.rcs):
                if frac[k] > 0:
                    forces.append(PointForce(t.id, frac[k] * self.rcs_fmax[k] * self.rcs_dir[k], self.rcs_pos[k]))
            mdot_rcs = float(frac @ self.rcs_mdot)
        return PropulsionOutput(forces, mdot_main, mdot_rcs, f_main, valid)
