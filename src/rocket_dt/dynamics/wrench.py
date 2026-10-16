"""Force/torque aggregation about the CoM (the only place r x F is computed)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PointForce:
    """A force applied at a point; both in body frame [N], [m]."""
    source_id: str
    force_body_n: np.ndarray
    position_body_m: np.ndarray


def total_wrench_body(forces: list[PointForce], com_body_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """F_total_body = sum F_i ; Tau_total_body = sum (r_i - r_com) x F_i."""
    f = np.zeros(3)
    tau = np.zeros(3)
    for pf in forces:
        f += pf.force_body_n
        tau += np.cross(pf.position_body_m - com_body_m, pf.force_body_n)
    return f, tau
