"""Bridge from Propulsion Adapter point forces to an Isaac Lab body wrench.

Isaac Lab 2.3.x interprets wrench `positions` relative to the link origin and
3.0 relative to the CoM (docs/research/03). To stay version independent the
bridge sums forces and computes torque about the CoM itself, then hands
(F_body, tau_body) to `permanent_wrench_composer.set_forces_and_torques(...,
positions=None, is_global=False)` every `_apply_action()` (ADR-0003).
This module is pure numpy so it can be unit tested without Isaac.
"""
from __future__ import annotations

import numpy as np

from rocket_dt.dynamics.wrench import PointForce, total_wrench_body


def wrench_about_com(forces: list[PointForce], com_body_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return total_wrench_body(forces, com_body_m)


def batched_wrench(force_b: np.ndarray, pos_b: np.ndarray, com_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised form for N envs x K thrusters: force_b, pos_b (N, K, 3); com_b (N, 3).

    Returns (N, 1, 3) force and torque arrays shaped like Isaac's single-body wrench buffers.
    """
    f = force_b.sum(axis=1)
    tau = np.cross(pos_b - com_b[:, None, :], force_b).sum(axis=1)
    return f[:, None, :], tau[:, None, :]


def batched_wrench_torch(force_b, pos_b, com_b):
    """Torch twin of `batched_wrench` (same shapes and math) for GPU env buffers."""
    import torch

    f = force_b.sum(dim=1)
    tau = torch.linalg.cross(pos_b - com_b[:, None, :], force_b, dim=-1).sum(dim=1)
    return f[:, None, :], tau[:, None, :]
