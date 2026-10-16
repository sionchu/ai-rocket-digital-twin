"""Single source of truth for frames, units and quaternion conventions.

Conventions (ADR-0002):
- SI units everywhere (m, kg, s, N, Pa, K, rad).
- World frame: right-handed, Z-up, origin on the landing pad surface.
- Body frame: right-handed, +Z_body = nose direction, origin at the geometric
  reference point (vehicle mid-length on the centerline). The CoM is NOT the
  body origin; it is computed by the mass model.
- Main engine nominal thrust acts along +Z_body (pushes the vehicle nose-ward)
  and is applied at the gimbal pivot below the CoM.
- Quaternions are scalar-first (w, x, y, z), Hamilton product, and map body
  vectors into the world frame: v_world = R(q) @ v_body. This matches the
  Isaac Lab root-state quaternion ordering.
"""
from __future__ import annotations

import numpy as np

G0 = 9.80665  # standard gravity [m/s^2], also used for Isp
GRAVITY_WORLD = np.array([0.0, 0.0, -G0])
FRAME_CONVENTION = "world_Z_up_body_Z_forward"


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n == 0.0:
        raise ValueError("zero quaternion")
    return q / n


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Rotation matrix R_world_body for a unit quaternion (w, x, y, z)."""
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_from_axis_angle(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    s = np.sin(angle / 2.0)
    return np.array([np.cos(angle / 2.0), *(axis * s)])


def quat_derivative(q: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    """q_dot = 0.5 * q (x) [0, omega_body] for body-frame angular velocity."""
    return 0.5 * quat_mul(q, np.array([0.0, *omega_body]))


def tilt_angle(q: np.ndarray) -> float:
    """Angle between body +Z (nose) and world +Z [rad]."""
    zb = quat_to_rotmat(q)[:, 2]
    return float(np.arccos(np.clip(zb[2], -1.0, 1.0)))


def isa_pressure(altitude_m: float) -> float:
    """ISA troposphere static pressure [Pa], valid 0..11 km."""
    if not (-500.0 <= altitude_m <= 11000.0):
        raise ValueError(f"altitude {altitude_m} m outside ISA troposphere model")
    return 101325.0 * (1.0 - 2.25577e-5 * altitude_m) ** 5.25588
