"""Parametric axisymmetric nozzle contours (wall radius r(x)).

x = 0 at the throat, x > 0 downstream. The contour is the CFD fluid-domain
boundary; visual meshes are separate assets. Bell contours use the Rao
parabolic (quadratic Bezier) approximation with wall angles taken from the
design record (Rao-chart values are inputs, not computed here).
CadQuery solids are not generated in P0 (NOT_RUN): the contour arrays are the
canonical geometry consumed by the CFD case generator and the mass model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rocket_dt.schemas import EngineDesign

CONTRACTION_RATIO = 4.0  # chamber/throat area ratio (assumed, same for A/B/C)
CONVERGENT_HALF_ANGLE = np.radians(30.0)
R_UP = 1.5  # upstream throat arc radius / rt
R_DN = 0.382  # downstream throat arc radius / rt (Rao)


@dataclass(frozen=True)
class NozzleContour:
    engine_design_id: str
    x_m: np.ndarray
    r_m: np.ndarray
    throat_index: int
    nozzle_length_m: float  # throat -> exit
    exit_radius_m: float
    divergent_wall_area_m2: float

    def check(self) -> None:
        assert np.all(np.diff(self.x_m) > 0), "x must increase"
        assert np.all(self.r_m > 0)


def _arc(xc, rc, radius, a0, a1, n):
    a = np.linspace(a0, a1, n)
    return xc + radius * np.sin(a), rc - radius * np.cos(a)


def _convergent(rt: float, n: int = 40):
    rc = rt * np.sqrt(CONTRACTION_RATIO)
    th = CONVERGENT_HALF_ANGLE
    # upstream arc centered at (0, rt + R_UP*rt) from angle -th to 0
    xa, ra = _arc(0.0, rt + R_UP * rt, R_UP * rt, -th, 0.0, n)
    # straight convergent line back up to chamber radius
    r0 = ra[0]
    x_start = xa[0] - (rc - r0) / np.tan(th)
    xl = np.linspace(x_start, xa[0], n, endpoint=False)
    rl = rc - (xl - x_start) * np.tan(th)
    return np.concatenate([xl, xa]), np.concatenate([rl, ra])


def build_contour(d: EngineDesign, n: int = 120) -> NozzleContour:
    rt = np.sqrt(d.throat_area_m2 / np.pi)
    re = rt * np.sqrt(d.expansion_ratio)
    xc, rcv = _convergent(rt)
    if d.family == "conical":
        alpha = d.conical_half_angle_rad
        xd, rd = _arc(0.0, rt + R_DN * rt, R_DN * rt, 0.0, alpha, n // 4)
        x1 = xd[-1] + (re - rd[-1]) / np.tan(alpha)
        xs = np.linspace(xd[-1], x1, n)[1:]
        rs = rd[-1] + (xs - xd[-1]) * np.tan(alpha)
        xdiv, rdiv = np.concatenate([xd, xs]), np.concatenate([rd, rs])
    else:
        tn, te = d.bell_theta_n_rad, d.bell_theta_e_rad
        xd, rd = _arc(0.0, rt + R_DN * rt, R_DN * rt, 0.0, tn, n // 4)
        l_cone15 = (np.sqrt(d.expansion_ratio) - 1.0) * rt / np.tan(np.radians(15.0))
        ln = d.bell_length_fraction * l_cone15
        nx, ny = xd[-1], rd[-1]
        ex, ey = ln, re
        # control point: intersection of tangent lines at N and E
        m1, m2 = np.tan(tn), np.tan(te)
        qx = (ey - ny + m1 * nx - m2 * ex) / (m1 - m2)
        qy = ny + m1 * (qx - nx)
        t = np.linspace(0.0, 1.0, n)[1:]
        xs = (1 - t) ** 2 * nx + 2 * (1 - t) * t * qx + t ** 2 * ex
        rs = (1 - t) ** 2 * ny + 2 * (1 - t) * t * qy + t ** 2 * ey
        xdiv, rdiv = np.concatenate([xd, xs]), np.concatenate([rd, rs])
    x = np.concatenate([xc, xdiv[1:]])
    r = np.concatenate([rcv, rdiv[1:]])
    throat_idx = len(xc) - 1
    seg = np.hypot(np.diff(xdiv), np.diff(rdiv))
    rmid = 0.5 * (rdiv[1:] + rdiv[:-1])
    wall_area = float(np.sum(2 * np.pi * rmid * seg))
    c = NozzleContour(d.engine_design_id, x, r, throat_idx, float(xdiv[-1]), float(rdiv[-1]), wall_area)
    c.check()
    return c


def engine_mass_kg(d: EngineDesign, contour: NozzleContour | None = None) -> float:
    """Engine dry mass = base (chamber, valves; assumed) + divergent wall shell."""
    c = contour or build_contour(d)
    return d.engine_base_mass_kg + d.nozzle_wall_areal_density_kg_m2 * c.divergent_wall_area_m2


def nozzle_wall_centroid_x(contour: NozzleContour) -> float:
    """Axial centroid (from throat) of the divergent wall shell."""
    i = contour.throat_index
    x, r = contour.x_m[i:], contour.r_m[i:]
    seg = np.hypot(np.diff(x), np.diff(r))
    w = 2 * np.pi * 0.5 * (r[1:] + r[:-1]) * seg
    return float(np.sum(w * 0.5 * (x[1:] + x[:-1])) / np.sum(w))
