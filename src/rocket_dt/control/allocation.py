"""Control-authority check and RCS allocation.

Authority: linearized effectiveness matrix B mapping actuator perturbations
(main thrust magnitude, gimbal x, gimbal y, each RCS thruster) to the
controlled body wrench [F_z, tau_x, tau_y, tau_z] at a nominal point.
Full attitude + vertical authority requires rank 4 AND that every axis can be
driven in both signs (unilateral thrusters are checked by sign coverage).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear

from rocket_dt.schemas import VehicleConfig

AXES = ("F_z", "tau_x", "tau_y", "tau_z")


@dataclass
class AuthorityReport:
    rank: int
    full_authority: bool
    sign_coverage: dict[str, tuple[bool, bool]]
    missing: list[str]
    B: np.ndarray
    columns: list[str]


def effectiveness_matrix(vehicle: VehicleConfig, com_b: np.ndarray, thrust_n: float,
                         include_main: bool = True, include_rcs: bool = True,
                         rcs_failed: np.ndarray | None = None):
    cols, names = [], []
    me = vehicle.layout.main_engine
    r = np.asarray(me.gimbal_pivot_body_m) - com_b
    if include_main:
        cols.append(np.array([1.0, *np.cross(r, [0, 0, 1.0])]))  # d/dF
        names.append("main_thrust")
        # d/d(delta_x) at 0: direction derivative (0,-1,0); d/d(delta_y): (1,0,0)
        for nm, dd in (("gimbal_x", np.array([0, -1.0, 0])), ("gimbal_y", np.array([1.0, 0, 0]))):
            cols.append(np.array([0.0, *np.cross(r, thrust_n * dd)]))
            names.append(nm)
    if include_rcs:
        for k, t in enumerate(vehicle.layout.rcs):
            if rcs_failed is not None and rcs_failed[k]:
                continue
            f = t.max_thrust_n * np.asarray(t.force_direction_body)
            tau = np.cross(np.asarray(t.position_body_m) - com_b, f)
            cols.append(np.array([f[2], *tau]))
            names.append(t.id)
    return np.array(cols).T if cols else np.zeros((4, 0)), names


def authority_report(vehicle: VehicleConfig, com_b: np.ndarray, thrust_n: float, **kw) -> AuthorityReport:
    B, names = effectiveness_matrix(vehicle, com_b, thrust_n, **kw)
    rank = int(np.linalg.matrix_rank(B, tol=1e-6)) if B.size else 0
    bidirectional = {"main_thrust", "gimbal_x", "gimbal_y"}  # throttle around a mid point, gimbal +/-
    cov = {}
    for i, ax in enumerate(AXES):
        pos = neg = False
        for j, nm in enumerate(names):
            v = B[i, j]
            if abs(v) < 1e-9:
                continue
            if nm in bidirectional:
                pos = neg = True
            elif v > 0:
                pos = True
            else:
                neg = True
        cov[ax] = (pos, neg)
    missing = [ax for ax, (p, n) in cov.items() if not (p and n)]
    return AuthorityReport(rank, rank == 4 and not missing, cov, missing, B, names)


class RcsAllocator:
    """Bounded least squares duty allocation for the RCS: min ||W (A u - b)||, 0 <= u <= 1.

    Targets the commanded roll torque while asking for zero net lateral force
    and zero pitch/yaw torque (those are left to the gimbal).
    """

    def __init__(self, vehicle: VehicleConfig, w_force: float = 0.05):
        self.rcs = vehicle.layout.rcs
        self.w_force = w_force
        self.vehicle = vehicle

    def allocate(self, tau_z_cmd: float, com_b: np.ndarray, failed: np.ndarray | None = None) -> np.ndarray:
        n = len(self.rcs)
        if n == 0:
            return np.zeros(0)
        A = np.zeros((5, n))
        for k, t in enumerate(self.rcs):
            f = t.max_thrust_n * np.asarray(t.force_direction_body)
            tau = np.cross(np.asarray(t.position_body_m) - com_b, f)
            A[:, k] = [tau[0], tau[1], tau[2], self.w_force * f[0], self.w_force * f[1]]
        ub = np.ones(n)
        if failed is not None:
            ub = np.where(failed, 1e-12, 1.0)
        b = np.array([0.0, 0.0, tau_z_cmd, 0.0, 0.0])
        res = lsq_linear(A, b, bounds=(np.zeros(n), ub), method="bvls")
        return np.clip(res.x, 0.0, 1.0)
