"""Engine layouts for VtvlEnv (label `win-isaac`). Same vehicle body, mass properties and total main thrust.

a  gimbal_rcs  one gimballed main engine (+-6 deg) + 4 roll RCS                     (vehicle_v0, baseline)
b  tri_diff    3 fixed engines on a 0.42 m ring, independent throttles (pitch/yaw by
               differential throttle), each 1/3 of the engine map; + 4 roll RCS
c  quad_diff   4 fixed engines on a 0.42 m ring, independent throttles, each 1/4; + 4 roll RCS
d  coldgas     one fixed main engine (no gimbal) + 8 cold-gas thrusters: the 4 roll RCS and
               4 radial 400 N pitch/yaw thrusters on the nose ring (closest to a thruster-only
               attitude system, the idea behind the announced Roadster thruster package)

Assumptions (documented in ADR-0010): rigid-body mass/CoM/inertia stay those of vehicle_v0 + conical_A (fixed
mass, ADR-0009); every engine uses the conical_A map scaled by its share; cold-gas gas budget is raised for layout d
(propellant accounting only, mass fixed).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rocket_dt.schemas import RcsThruster, VehicleConfig

LAYOUTS = ("gimbal_rcs", "tri_diff", "quad_diff", "coldgas")
RING_R = 0.42  # engine ring radius [m] (inside the 0.6 m hull)


@dataclass
class LayoutSpec:
    name: str
    engine_pos_b: np.ndarray  # (E, 3) nozzle mount points (thrust application)
    engine_share: np.ndarray  # (E,) fraction of the engine map thrust per engine
    gimbal: bool  # engine 0 gimballed (layout a only)
    independent_throttles: bool
    rcs: list = field(default_factory=list)
    rcs_gas_kg: float | None = None  # None: vehicle default

    @property
    def n_engines(self) -> int:
        return len(self.engine_share)

    @property
    def n_throttle_actions(self) -> int:
        return self.n_engines if self.independent_throttles else 1

    @property
    def act_dim(self) -> int:
        return self.n_throttle_actions + (2 if self.gimbal else 0) + len(self.rcs)

    @property
    def obs_dim(self) -> int:
        core = 16  # rel pos 3, v_b 3, g_b 3, w_b 3, altitude, vz, fuel, available thrust
        act_state = (1 + 2) if self.gimbal else self.n_engines
        return core + act_state + self.act_dim


def layout_spec(name: str, veh: VehicleConfig) -> LayoutSpec:
    pivot = np.array(veh.layout.main_engine.gimbal_pivot_body_m, float)
    roll = list(veh.layout.rcs)
    if name == "gimbal_rcs":
        return LayoutSpec(name, pivot[None], np.array([1.0]), True, False, roll)
    if name in ("tri_diff", "quad_diff"):
        k = 3 if name == "tri_diff" else 4
        ang0 = math.pi / 2 if k == 3 else math.pi / 4
        pos = np.array([[RING_R * math.cos(ang0 + 2 * math.pi * i / k), RING_R * math.sin(ang0 + 2 * math.pi * i / k),
                         pivot[2]] for i in range(k)])
        return LayoutSpec(name, pos, np.full(k, 1.0 / k), False, True, roll)
    if name == "coldgas":
        r0 = roll[0]
        z = r0.position_body_m[2]
        rad = veh.mass.structure_radius_m
        pitch_yaw = [RcsThruster(f"cg_{tag}", (px * rad, py * rad, z), (-px, -py, 0.0), 400.0, r0.valve_delay_s,
                                 r0.min_on_time_s, r0.isp_s)
                     for tag, px, py in (("px", 1, 0), ("nx", -1, 0), ("py", 0, 1), ("ny", 0, -1))]
        return LayoutSpec(name, pivot[None], np.array([1.0]), False, False, roll + pitch_yaw, rcs_gas_kg=80.0)
    raise ValueError(f"unknown layout {name}; choose from {LAYOUTS}")
