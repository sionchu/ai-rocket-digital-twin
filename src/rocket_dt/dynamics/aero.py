"""Quasi-steady aerodynamics: viscous-crossflow strip theory + axial drag, with a power-law wind profile (ADR-0011).

Backend-neutral: `aero_wrench` uses only operators, `.sum`, `.clip`, `**` and `einsum`, so the same code runs on
numpy arrays (reference integrator, tests) and on torch tensors (Isaac env). Build the constant arrays once with
`AeroModel.arrays(convert)`, where `convert` turns a numpy array into the caller's backend.

Model, per body strip k (centre s_k on the axis, width 2 r(z_k), length dz):
  u_k = R^T (v_com - w(h_k)) + omega x (s_k - c)      body-frame velocity of the strip relative to the air
  F_k = -1/2 rho Cdc (2 r dz) |u_k,perp| u_k,perp       crossflow drag (no lift from the axial component)
  F_ax = -1/2 rho Cd_ax A_base |u_ax| u_ax  along z_b   axial drag (tail-first or nose-first coefficient)
The moment about the CoM comes from the strip positions, so the centre of pressure and the aerodynamic
rate damping follow from the geometry. Not modelled: compressibility, skin friction, plume-flow interaction,
legs/fins as separate surfaces, unsteady wake.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

RHO0 = 1.225  # ISA sea-level density [kg/m^3]


def isa_density(h):
    """ISA troposphere density for altitude h [m] (array or tensor, h >= 0)."""
    t = 288.15 - 0.0065 * h
    return 101325.0 * (t / 288.15) ** 5.25588 / (287.05 * t)


@dataclass(frozen=True)
class AeroModel:
    aero_id: str
    crossflow_cd: float
    axial_cd_tail_first: float
    axial_cd_nose_first: float
    strip_z: np.ndarray  # (K,) body z of strip centres
    strip_area: np.ndarray  # (K,) projected side area of each strip, 2 r dz
    base_area: float
    wind_ref_height_m: float
    wind_shear_exponent: float
    wind_min_height_m: float

    @classmethod
    def load(cls, path: str | Path, radius_m: float, length_m: float) -> "AeroModel":
        c = yaml.safe_load(open(path))
        k, ln = int(c["n_strips"]), float(c["nose_length_m"])
        edges = np.linspace(-length_m / 2, length_m / 2, k + 1)
        z = 0.5 * (edges[1:] + edges[:-1])
        dz = edges[1] - edges[0]
        z_nose0 = length_m / 2 - ln
        rho = (radius_m ** 2 + ln ** 2) / (2 * radius_m)  # tangent ogive radius
        x = np.clip(z - z_nose0, 0.0, ln)
        r = np.where(z > z_nose0, np.sqrt(np.maximum(rho ** 2 - x ** 2, 0.0)) + radius_m - rho, radius_m)
        return cls(c["aero_id"], float(c["crossflow_cd"]), float(c["axial_cd_tail_first"]),
                   float(c["axial_cd_nose_first"]), z, 2 * r * dz, float(np.pi * radius_m ** 2),
                   float(c["wind_ref_height_m"]), float(c["wind_shear_exponent"]), float(c["wind_min_height_m"]))

    @property
    def side_area(self) -> float:
        return float(self.strip_area.sum())

    def wind_speed_for_force(self, force_n):
        """Reference-height wind speed whose steady crossflow force on the upright vehicle (sea-level density,
        uniform profile) equals `force_n`; used to turn the v0 force-type wind into a wind velocity."""
        return (np.asarray(force_n, float) / (0.5 * RHO0 * self.crossflow_cd * self.side_area)) ** 0.5

    def arrays(self, convert=np.asarray) -> dict:
        s = np.zeros((len(self.strip_z), 3))
        s[:, 2] = self.strip_z
        out = {"s": s, "area": self.strip_area, "perp": np.array([1.0, 1.0, 0.0]),
               "ex": np.array([1.0, 0.0, 0.0]), "ey": np.array([0.0, 1.0, 0.0]), "ez": np.array([0.0, 0.0, 1.0])}
        out = {k: convert(v) for k, v in out.items()}
        out.update(cdc=self.crossflow_cd, cd_tail=self.axial_cd_tail_first, cd_nose=self.axial_cd_nose_first,
                   a_base=self.base_area, h_ref=self.wind_ref_height_m, alpha=self.wind_shear_exponent,
                   h_min=self.wind_min_height_m)
        return out


def _cross(a, b, k):
    """a x b on the last axis using the unit vectors in k (backend neutral)."""
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    return ((ay * bz - az * by)[..., None] * k["ex"] + (az * bx - ax * bz)[..., None] * k["ey"]
            + (ax * by - ay * bx)[..., None] * k["ez"])


def aero_wrench(k: dict, p_z, rot, v_w, omega_b, com_b, wind_ref_w):
    """Aerodynamic force and moment in the body frame, moment about the CoM.

    Shapes (n batch): p_z (n,) CoM height, rot (n,3,3) body->world, v_w (n,3) CoM velocity, omega_b (n,3),
    com_b (n,3) CoM in the body frame, wind_ref_w (n,3) wind at the reference height. Returns (F_b, tau_b), (n,3).
    """
    r = k["s"][None] - com_b[:, None, :]  # (n,K,3) strip centre relative to the CoM, body frame
    h = p_z[:, None] + (rot[:, None, 2, :] * r).sum(-1)  # (n,K) world height of each strip
    f = (h.clip(min=k["h_min"]) / k["h_ref"]) ** k["alpha"]
    v_air_w = v_w[:, None, :] - wind_ref_w[:, None, :] * f[..., None]
    u = (rot[:, None, :, :] * v_air_w[..., :, None]).sum(-2) + _cross(omega_b[:, None, :], r, k)  # R^T v + w x r
    u_n = u * k["perp"]
    s_n = (u_n * u_n).sum(-1) ** 0.5
    rho = isa_density(p_z.clip(min=0.0))
    f_k = (-0.5 * k["cdc"]) * rho[:, None, None] * k["area"][None, :, None] * s_n[..., None] * u_n
    u_ax = u[..., 2].mean(-1)
    cd_ax = k["cd_tail"] + (k["cd_nose"] - k["cd_tail"]) * (u_ax > 0)
    f_ax = -0.5 * rho * cd_ax * k["a_base"] * abs(u_ax) * u_ax
    force = f_k.sum(1) + f_ax[:, None] * k["ez"]
    torque = _cross(r, f_k, k).sum(1)
    return force, torque
