"""Cascaded PD baseline (G3 reference): guidance -> thrust vector -> attitude.

Outer loop: desired CoM acceleration from position/velocity errors (hover)
or a velocity-vs-height descent profile (landing). The thrust vector
T = m (a_des - g) sets throttle (through the performance-map inverse) and
the desired body +Z. Inner loop: geometric SO(3) attitude PD (Lee et al.)
giving body torque; pitch/yaw torque -> gimbal angles, roll torque -> RCS.
Uses true state in P0 (sensor models are a later stage).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rocket_dt.control.allocation import RcsAllocator
from rocket_dt.dynamics.frames import G0, quat_to_rotmat
from rocket_dt.dynamics.mass import MassModel
from rocket_dt.dynamics.reference_6dof import IDX_MP, IDX_MR, IDX_P, IDX_Q, IDX_V, IDX_W
from rocket_dt.propulsion.performance_map import EnginePerformanceMap
from rocket_dt.schemas import VehicleConfig


@dataclass
class PDGains:
    kp_pos: float = 0.6
    kd_pos: float = 1.4
    kp_vz: float = 1.5
    kp_att: float = 4.0  # [1/s^2] scaled by inertia
    kd_att: float = 3.0  # [1/s]
    max_tilt_rad: float = np.radians(12.0)
    max_horiz_acc: float = 2.0


@dataclass
class Command:
    engine_on: bool
    throttle: float
    gimbal_rad: np.ndarray
    rcs_on_time_s: np.ndarray
    debug: dict


def _vee(m):
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


class CascadedPD:
    def __init__(self, vehicle: VehicleConfig, mass_model: MassModel, perf_map: EnginePerformanceMap,
                 control_dt: float, gains: PDGains | None = None):
        self.v = vehicle
        self.mm = mass_model
        self.map = perf_map
        self.dt = control_dt
        self.g = gains or PDGains()
        self.alloc = RcsAllocator(vehicle)
        self.pivot = np.asarray(vehicle.layout.main_engine.gimbal_pivot_body_m, float)
        self.gmax = vehicle.layout.main_engine.max_gimbal_rad

    def desired_accel(self, x, mode: str, target_w: np.ndarray, touchdown_com_z: float) -> np.ndarray:
        g = self.g
        p, v = x[IDX_P], x[IDX_V]
        a = np.zeros(3)
        a[:2] = g.kp_pos * (target_w[:2] - p[:2]) - g.kd_pos * v[:2]
        if mode == "hover":
            a[2] = g.kp_pos * (target_w[2] - p[2]) - g.kd_pos * v[2]
        elif mode == "descent":
            h = max(p[2] - touchdown_com_z, 0.0)
            vz_ref = -float(np.clip(0.3 * h, 0.8, 6.0))
            a[2] = g.kp_vz * (vz_ref - v[2])
        else:
            raise ValueError(mode)
        n = np.linalg.norm(a[:2])
        if n > g.max_horiz_acc:
            a[:2] *= g.max_horiz_acc / n
        return a

    def __call__(self, x: np.ndarray, ambient_pa: float, mode: str, target_w, touchdown_com_z: float,
                 actual_throttle: float, rcs_failed=None) -> Command:
        ms = self.mm.evaluate(x[IDX_MP], x[IDX_MR])
        rot = quat_to_rotmat(x[IDX_Q])
        w = x[IDX_W]
        a_des = self.desired_accel(x, mode, np.asarray(target_w, float), touchdown_com_z)
        t_w = ms.mass_kg * (a_des + np.array([0.0, 0.0, G0]))
        # tilt limit on the thrust vector
        tilt = np.arctan2(np.linalg.norm(t_w[:2]), t_w[2])
        if tilt > self.g.max_tilt_rad:
            horiz = t_w[:2] / np.linalg.norm(t_w[:2])
            t_w[:2] = horiz * t_w[2] * np.tan(self.g.max_tilt_rad)
        zb_des = t_w / np.linalg.norm(t_w)
        f_cmd = float(t_w @ rot[:, 2])  # project on current nose axis
        throttle = self.map.throttle_for_thrust(max(f_cmd, 0.0), ambient_pa)

        # desired attitude: nose along zb_des, heading keeps body x near world x
        x_ref = np.array([1.0, 0.0, 0.0])
        xb = x_ref - (x_ref @ zb_des) * zb_des
        xb /= np.linalg.norm(xb)
        r_des = np.column_stack([xb, np.cross(zb_des, xb), zb_des])
        e_r = 0.5 * _vee(r_des.T @ rot - rot.T @ r_des)
        i_b = ms.inertia_com_body
        tau = -self.g.kp_att * (i_b @ e_r) - self.g.kd_att * (i_b @ w) + np.cross(w, i_b @ w)

        # roll via RCS first; its pitch/yaw side torque is fed forward into the gimbal command
        duty = self.alloc.allocate(tau[2], ms.com_body_m, rcs_failed)
        tau_rcs = self.alloc.effect(ms.com_body_m)[:3] @ duty
        tau = tau.copy()
        tau[:2] -= tau_rcs[:2]
        # pitch/yaw via gimbal: tau_x = -l F sin dx ; tau_y = -l F cos dx sin dy, l = com_z - pivot_z
        f_now = self.map.lookup(actual_throttle, ambient_pa)[0]
        lever = ms.com_body_m[2] - self.pivot[2]
        dx = np.arcsin(np.clip(-tau[0] / (lever * f_now), -1, 1))
        dy = np.arcsin(np.clip(-tau[1] / (lever * f_now * np.cos(dx)), -1, 1))
        gimbal = np.clip([dx, dy], -self.gmax, self.gmax)

        return Command(True, throttle, gimbal, duty * self.dt,
                       {"a_des": a_des, "tau_cmd": tau, "f_cmd": f_cmd, "zb_des": zb_des})
