"""Procedural mesh builders for the VTVL v1 visual assets (numpy only).

Every builder returns a Mesh (points, normals, uvs, face counts/indices) in body-frame metres. Faces are wound
counter-clockwise seen from outside (USD rightHanded). Revolved profiles follow the rule normal = (dz, -dr) in the
(r, z) plane, so a profile walked upward has an outward normal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import vtvl_spec as S


@dataclass
class Mesh:
    P: list = field(default_factory=list)
    N: list = field(default_factory=list)
    UV: list = field(default_factory=list)
    counts: list = field(default_factory=list)
    idx: list = field(default_factory=list)
    n: int = 0

    def add(self, P, N, UV, faces):
        P = np.asarray(P, float).reshape(-1, 3)
        N = np.asarray(N, float).reshape(-1, 3)
        UV = np.asarray(UV, float).reshape(-1, 2)
        faces = np.asarray(faces, int)
        self.P.append(P)
        self.N.append(N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12))
        self.UV.append(UV)
        self.counts.append(np.full(len(faces), faces.shape[1], int))
        self.idx.append((faces + self.n).ravel())
        self.n += len(P)
        return self

    def extend(self, other: "Mesh"):
        if other.n == 0:
            return self
        P, N, UV, counts, idx = other.arrays()
        self.P.append(P)
        self.N.append(N)
        self.UV.append(UV)
        self.counts.append(counts)
        self.idx.append(idx + self.n)
        self.n += len(P)
        return self

    def arrays(self):
        return (np.concatenate(self.P), np.concatenate(self.N), np.concatenate(self.UV),
                np.concatenate(self.counts), np.concatenate(self.idx))

    def flipped(self) -> "Mesh":
        P, N, UV, counts, idx = self.arrays()
        out = Mesh()
        out.P, out.N, out.UV, out.counts, out.n = [P], [-N], [UV], [counts], len(P)
        rev, o = [], 0
        for c in counts:
            rev.append(idx[o:o + c][::-1])
            o += c
        out.idx = [np.concatenate(rev)]
        return out

    def transformed(self, R: np.ndarray, t, scale: float = 1.0) -> "Mesh":
        P, N, UV, counts, idx = self.arrays()
        out = Mesh()
        out.P = [(P * scale) @ R.T + np.asarray(t, float)]
        out.N = [N @ R.T]
        out.UV = [UV]
        out.counts = [counts]
        out.idx = [idx]
        out.n = len(P)
        return out


# ------------------------------------------------------------------------------------------- primitives
def revolve(r, z, nseg=96, v=None, u_scale=1.0, smooth=True, th0=0.0, th1=2 * math.pi, centre=(0.0, 0.0)):
    """Surface of revolution about +Z. u = u_scale * (theta - th0)/(th1 - th0); v given per profile point."""
    r = np.asarray(r, float)
    z = np.asarray(z, float)
    m = len(r)
    v = np.asarray(v if v is not None else np.linspace(0, 1, m), float)
    if not smooth:  # hard edges: one revolve per profile segment
        out = Mesh()
        for i in range(m - 1):
            out.extend(revolve(r[i:i + 2], z[i:i + 2], nseg, v[i:i + 2], u_scale, True, th0, th1, centre))
        return out
    th = th0 + (th1 - th0) * np.arange(nseg + 1) / nseg
    c, s = np.cos(th), np.sin(th)
    P = np.zeros((m, nseg + 1, 3))
    P[..., 0] = r[:, None] * c[None] + centre[0]
    P[..., 1] = r[:, None] * s[None] + centre[1]
    P[..., 2] = z[:, None]
    # profile normals (dz, -dr), averaged at interior points
    dr, dz = np.diff(r), np.diff(z)
    seg_n = np.stack([dz, -dr], 1)
    seg_n /= np.maximum(np.linalg.norm(seg_n, axis=1, keepdims=True), 1e-12)
    pn = np.zeros((m, 2))
    pn[:-1] += seg_n
    pn[1:] += seg_n
    pn /= np.maximum(np.linalg.norm(pn, axis=1, keepdims=True), 1e-12)
    N = np.zeros((m, nseg + 1, 3))
    N[..., 0] = pn[:, 0:1] * c[None]
    N[..., 1] = pn[:, 0:1] * s[None]
    N[..., 2] = pn[:, 1:2]
    UV = np.zeros((m, nseg + 1, 2))
    UV[..., 0] = u_scale * np.arange(nseg + 1)[None] / nseg
    UV[..., 1] = v[:, None]
    j, k = np.meshgrid(np.arange(m - 1), np.arange(nseg), indexing="ij")
    a = j * (nseg + 1) + k
    faces = np.stack([a, a + 1, a + nseg + 2, a + nseg + 1], -1).reshape(-1, 4)
    return Mesh().add(P.reshape(-1, 3), N.reshape(-1, 3), UV.reshape(-1, 2), faces)


def disc(r0, r1, z, nseg=64, up=True, uv_planar=None, nring=2, centre=(0.0, 0.0)):
    """Flat annulus/disc at height z, normal +z (up) or -z. uv_planar=(scale, offset) maps uv = xy*scale+offset."""
    rr = np.linspace(r0, r1, nring) if up is False else np.linspace(r1, r0, nring)
    m = revolve(rr, np.full(nring, z), nseg, centre=centre)  # walking inward -> +z normal
    if not up:
        m = revolve(np.linspace(r0, r1, nring), np.full(nring, z), nseg, centre=centre)
    if uv_planar is not None:
        P = m.P[0]
        m.UV = [P[:, :2] * uv_planar[0] + uv_planar[1]]
    return m


def frame_from_axis(axis):
    """Rotation matrix whose third column is the given axis."""
    zax = np.asarray(axis, float)
    zax = zax / np.linalg.norm(zax)
    ref = np.array([0.0, 0.0, 1.0]) if abs(zax[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    xax = np.cross(ref, zax)
    xax /= np.linalg.norm(xax)
    yax = np.cross(zax, xax)
    return np.stack([xax, yax, zax], 1)


def tube(p0, p1, r, nseg=20, caps=True, r1=None, u_len_scale=1.0):
    """Cylinder (or cone if r1 given) from p0 to p1 with flat caps."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = np.linalg.norm(p1 - p0)
    R = frame_from_axis(p1 - p0)
    r1 = r if r1 is None else r1
    m = revolve([r, r1], [0.0, L], nseg, v=[0, L * u_len_scale], u_scale=2 * math.pi * r * u_len_scale)
    if caps:
        m.extend(revolve([0.0, r], [0.0, 0.0], nseg))  # bottom, normal -z
        if r1 > 0:
            m.extend(revolve([r1, 0.0], [L, L], nseg))  # top, normal +z
    return m.transformed(R, p0)


def box(center, axes, size):
    """Oriented box. axes: 3x3 with columns = local x, y, z unit vectors; size: full extents."""
    c = np.asarray(center, float)
    A = np.asarray(axes, float)
    h = np.asarray(size, float) / 2
    m = Mesh()
    for ax in range(3):
        for sgn in (1, -1):
            n = A[:, ax] * sgn
            a1, a2 = (ax + 1) % 3, (ax + 2) % 3
            if sgn < 0:
                a1, a2 = a2, a1
            e1, e2 = A[:, a1] * h[a1], A[:, a2] * h[a2]
            fc = c + n * h[ax]
            P = [fc - e1 - e2, fc + e1 - e2, fc + e1 + e2, fc - e1 + e2]
            UV = [(0, 0), (2 * h[a1], 0), (2 * h[a1], 2 * h[a2]), (0, 2 * h[a2])]
            m.add(P, [n] * 4, UV, [[0, 1, 2, 3]])
    return m


def sphere(c, r, nseg=24, nring=12):
    t = np.linspace(-math.pi / 2, math.pi / 2, nring + 1)
    m = revolve(r * np.cos(t), r * np.sin(t), nseg)
    return m.transformed(np.eye(3), c)


def radial_axes(phi):
    """Columns: e_r, e_t, e_z for azimuth phi."""
    return np.array([[math.cos(phi), -math.sin(phi), 0.0], [math.sin(phi), math.cos(phi), 0.0], [0.0, 0.0, 1.0]])


# ------------------------------------------------------------------------------------------- hull
def ogive_profile(n=48):
    """Blunted tangent ogive (length NOSE_LEN, base radius HULL_R, tip radius NOSE_TIP_R), tip at Z_TOP.

    Returns (r, z) from the nose base up to the tip, and the base z."""
    R, L, rn = S.HULL_R, S.NOSE_LEN, S.NOSE_TIP_R
    rho = (R * R + L * L) / (2 * R)
    x0 = L - math.sqrt((rho - rn) ** 2 - (rho - R) ** 2)
    yt = rn * (rho - R) / (rho - rn)
    xt = x0 - math.sqrt(rn ** 2 - yt ** 2)
    xa = x0 - rn
    x_og = np.linspace(L, xt, n)
    r_og = np.sqrt(rho ** 2 - (L - x_og) ** 2) + R - rho
    x_sp = np.linspace(xt, xa, 10)[1:]
    r_sp = np.sqrt(np.maximum(rn ** 2 - (x_sp - x0) ** 2, 0))
    x = np.r_[x_og, x_sp]
    r = np.r_[r_og, r_sp]
    z = S.Z_TOP - (x - xa)
    return r, z, S.Z_TOP - (L - xa)


def r_hull(zq):
    r, z, zb = ogive_profile(200)
    zq = np.asarray(zq, float)
    return np.where(zq <= zb, S.HULL_R, np.interp(zq, z, r))


def hull_mesh():
    r_n, z_n, zb = ogive_profile()
    z_cyl = np.unique(np.r_[np.linspace(S.Z_BOT, zb, 29), list(S.WELD_Z), S.Z_SKIRT_TOP, S.Z_BAND_BOT, S.Z_BAND_TOP])
    z_cyl = z_cyl[z_cyl < zb]
    r = np.r_[np.full(len(z_cyl), S.HULL_R), r_n]
    z = np.r_[z_cyl, z_n]
    v = (z - S.Z_BOT) / S.HULL_LEN
    return revolve(r, z, 128, v=v)


def weld_rings():
    m = Mesh()
    zb = ogive_profile()[2]
    for zw in S.WELD_Z:
        if zw >= zb - 0.01:
            continue
        rr = [S.HULL_R - 0.001, S.HULL_R + 0.0025, S.HULL_R + 0.0025, S.HULL_R - 0.001]
        zz = [zw - 0.009, zw - 0.004, zw + 0.004, zw + 0.009]
        m.extend(revolve(rr, zz, 128, v=[(q - S.Z_BOT) / S.HULL_LEN for q in zz]))
    return m


def skirt_inner():
    """Inside of the open aft skirt + bottom lip."""
    zz = [S.Z_HEATSHIELD, S.Z_BOT]  # walk downward at the inner radius -> inward normal
    m = revolve([S.SKIRT_INNER_R] * 2, zz, 96, v=[0, 0.42], u_scale=3.7)
    m.extend(revolve([S.SKIRT_INNER_R, S.HULL_R], [S.Z_BOT, S.Z_BOT], 96, v=[0, 0.02], u_scale=3.7))
    return m


def heat_shield():
    m = disc(0.0, S.SKIRT_INNER_R + 0.002, S.Z_HEATSHIELD, 96, up=False, uv_planar=(1.0, 0.5), nring=4)
    return m


# ------------------------------------------------------------------------------------------- legs
def leg_parts(feet):
    """Returns dict of meshes: struts (carbon), rods (chrome), fittings (metal), feet (dark metal)."""
    out = {k: Mesh() for k in ("struts", "rods", "fittings", "feet")}
    for f in feet:
        f = np.asarray(f, float)
        phi = math.atan2(f[1], f[0])
        rf = math.hypot(f[0], f[1])
        A = radial_axes(phi)
        er, et = A[:, 0], A[:, 1]
        H = 0.70 * er + np.array([0, 0, S.Z_LEG_HINGE - 0.05])
        ankle = rf * er + np.array([0, 0, f[2] + S.FOOT_PAD_H + 0.11])
        # hinge clevis on the hull + pin
        out["fittings"].extend(box(0.64 * er + np.array([0, 0, S.Z_LEG_HINGE]), A, (0.12, 0.26, 0.34)))
        out["fittings"].extend(tube(H - 0.13 * et, H + 0.13 * et, 0.035, 16))
        # telescoping main strut
        d = ankle - H
        out["struts"].extend(tube(H, H + 0.62 * d, 0.078, 24))
        out["fittings"].extend(tube(H + 0.58 * d, H + 0.64 * d, 0.09, 24))
        out["rods"].extend(tube(H + 0.6 * d, ankle, 0.054, 24))
        out["fittings"].extend(sphere(H, 0.085, 20, 10))
        # V brace from the aft skirt to the knee
        K = H + 0.52 * d
        for sgn in (1, -1):
            a = phi + sgn * 0.42
            P = S.HULL_R * np.array([math.cos(a), math.sin(a), 0.0]) + np.array([0, 0, -3.22])
            Ab = radial_axes(a)
            out["fittings"].extend(box(P + 0.02 * Ab[:, 0], Ab, (0.07, 0.12, 0.12)))
            out["struts"].extend(tube(P + 0.04 * Ab[:, 0], K, 0.036, 16))
        out["fittings"].extend(sphere(K, 0.06, 16, 8))
        # foot: chamfered puck, post and ankle ball; bottom exactly at the physics contact z
        zb = f[2]
        pr = [0.0, S.FOOT_PAD_R - 0.025, S.FOOT_PAD_R, S.FOOT_PAD_R, S.FOOT_PAD_R - 0.07, 0.09, 0.0]
        pz = [zb, zb, zb + 0.02, zb + 0.05, zb + S.FOOT_PAD_H, zb + S.FOOT_PAD_H + 0.01, zb + S.FOOT_PAD_H + 0.01]
        foot = revolve(pr, pz, 48, smooth=False)
        out["feet"].extend(foot.transformed(np.eye(3), (f[0], f[1], 0.0)))
        out["feet"].extend(tube((f[0], f[1], zb + S.FOOT_PAD_H), ankle, 0.045, 16))
        out["fittings"].extend(sphere(ankle, 0.07, 20, 10))
    return out


# ------------------------------------------------------------------------------------------- grid fins
def grid_fin(phi, z=S.Z_FIN, r0=0.68, r1=1.16, half_w=0.23, depth=0.075):
    A = radial_axes(phi)
    er, et, ez = A[:, 0], A[:, 1], A[:, 2]
    m = Mesh()
    fr = 0.022
    c = lambda s, t: s * er + t * et + z * ez
    # outer frame
    m.extend(box(c(r0 + fr / 2, 0), A, (fr, 2 * half_w, depth)))
    m.extend(box(c(r1 - fr / 2, 0), A, (fr, 2 * half_w, depth)))
    m.extend(box(c(0.5 * (r0 + r1), half_w - fr / 2), A, (r1 - r0, fr, depth)))
    m.extend(box(c(0.5 * (r0 + r1), -half_w + fr / 2), A, (r1 - r0, fr, depth)))
    # diagonal lattice, clipped to the frame interior
    s0, s1, t0, t1 = r0 + fr, r1 - fr, -half_w + fr, half_w - fr
    pitch = 0.062 * math.sqrt(2)
    for sgn in (1, -1):
        for cc in np.arange(-1.0, 1.0, pitch):
            # line: (s - smid) - sgn * t = cc
            smid = 0.5 * (s0 + s1)
            pts = []
            for t in (t0, t1):
                s = smid + cc + sgn * t
                if s0 <= s <= s1:
                    pts.append((s, t))
            for s in (s0, s1):
                t = sgn * (s - smid - cc)
                if t0 <= t <= t1:
                    pts.append((s, t))
            if len(pts) < 2:
                continue
            pts = sorted(set((round(a, 9), round(b, 9)) for a, b in pts))
            (sa, ta), (sb, tb) = pts[0], pts[-1]
            L = math.hypot(sb - sa, tb - ta)
            if L < 0.02:
                continue
            d = ((sb - sa) * er + (tb - ta) * et) / L
            n = np.cross(ez, d)
            m.extend(box(c(0.5 * (sa + sb), 0.5 * (ta + tb)), np.stack([d, n, ez], 1), (L, 0.006, depth * 0.96)))
    # root shaft and actuator housing
    m.extend(tube(c(S.HULL_R - 0.01, 0), c(r0 + 0.005, 0), 0.032, 16))
    m.extend(box(c(S.HULL_R + 0.02, 0) + np.array([0, 0, 0.05]), A, (0.07, 0.20, 0.30)))
    return m


def raceway(phi=math.radians(S.RACEWAY_DEG), z0=-2.88, z1=2.36):
    A = radial_axes(phi)
    er = A[:, 0]
    tray = box(er * (S.HULL_R + 0.02) + np.array([0, 0, 0.5 * (z0 + z1)]), A, (0.055, 0.10, z1 - z0))
    brackets = Mesh()
    for zz in np.arange(z0 + 0.15, z1 - 0.1, 0.55):
        brackets.extend(box(er * (S.HULL_R + 0.024) + np.array([0, 0, zz]), A, (0.064, 0.13, 0.035)))
    return tray, brackets


# ------------------------------------------------------------------------------------------- RCS
def small_nozzle(throat, axis, length=0.07, r_t=0.012, r_e=0.03):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    p0 = np.asarray(throat, float)
    outer = tube(p0, p0 + axis * length, r_t, 20, caps=False, r1=r_e)
    inner = tube(p0 + axis * 0.004, p0 + axis * length, r_t * 0.8, 20, caps=False, r1=r_e * 0.9).flipped()
    return outer, inner


def roll_pods(rcs):
    """Pods at the roll-thruster positions (+-x, z=3.0) with a nozzle each way along +-y."""
    pods, noz, noz_in = Mesh(), Mesh(), Mesh()
    seen = set()
    for t in rcs:
        p = np.asarray(t.position_body_m, float)
        key = (round(p[0], 3), round(p[1], 3), round(p[2], 3))
        if not t.id.startswith("rcs_"):
            continue
        if key not in seen:
            seen.add(key)
            phi = math.atan2(p[1], p[0])
            A = radial_axes(phi)
            r_in = float(r_hull(p[2] + 0.13)) - 0.04
            r_out = S.HULL_R
            pods.extend(box(A[:, 0] * 0.5 * (r_in + r_out) + np.array([0, 0, p[2]]), A, (r_out - r_in, 0.22, 0.26)))
        # nozzle: exhaust opposite to the force direction, on the pod side face
        fdir = np.asarray(t.force_direction_body, float)
        ex = -fdir
        phi = math.atan2(p[1], p[0])
        er = radial_axes(phi)[:, 0]
        throat = er * (S.HULL_R - 0.06) + np.array([0, 0, p[2]]) + ex * 0.105
        o, i = small_nozzle(throat, ex, 0.06, 0.011, 0.026)
        noz.extend(o)
        noz_in.extend(i)
    return pods, noz, noz_in


def coldgas_ring(rcs):
    """Collar at z=3.0 hugging the ogive, carrying the 4 radial cold-gas thrusters (exhaust radially outward)."""
    za, zb_ = 2.93, 3.07
    ro = 0.585
    zz = np.linspace(za, zb_, 8)
    ring = Mesh()
    ring.extend(revolve([float(r_hull(za)) - 0.004, ro], [za, za], 96))  # bottom face (normal -z)
    ring.extend(revolve([ro, ro], [za, zb_], 96))  # outer face
    ring.extend(revolve([ro, float(r_hull(zb_)) - 0.004], [zb_, zb_], 96))  # top face
    ring.extend(revolve((r_hull(zz) - 0.004)[::-1], zz[::-1], 96))  # inner face (hidden)
    housings, noz, noz_in = Mesh(), Mesh(), Mesh()
    for t in rcs:
        if not t.id.startswith("cg_"):
            continue
        p = np.asarray(t.position_body_m, float)
        phi = math.atan2(p[1], p[0])
        A = radial_axes(phi)
        er = A[:, 0]
        housings.extend(box(er * 0.575 + np.array([0, 0, p[2]]), A, (0.06, 0.13, 0.16)))
        ex = -np.asarray(t.force_direction_body, float)  # = +er (exhaust outward)
        o, i = small_nozzle(er * S.HULL_R + np.array([0, 0, p[2]]), ex, 0.07, 0.016, 0.04)
        noz.extend(o)
        noz_in.extend(i)
    return ring, housings, noz, noz_in


# ------------------------------------------------------------------------------------------- engine
def engine_parts(xs, rs, scale_vis, z_throat_rel, sc, tvc: bool):
    """Engine geometry relative to the gimbal pivot (origin), unscaled (the assembly applies sc about the pivot).

    xs, rs: engine contour (m), xs relative to the throat (negative = chamber side). z = z_throat_rel - scale_vis*xs.
    The converging section is trimmed at 0.86 of the chamber radius so the scaled clusters stay inside the 0.6 m hull.
    """
    xs = np.asarray(xs, float)
    rs = np.asarray(rs, float)
    r = scale_vis * rs
    z = z_throat_rel - scale_vis * xs
    i_t = int(np.argmin(r))
    r_cap = 0.86 * r[0]
    keep = np.ones(len(r), bool)
    keep[:i_t] = r[:i_t] <= r_cap
    first = int(np.argmax(keep))
    if first > 0:  # insert the exact trim point
        zc = np.interp(r_cap, r[first - 1:first + 1][::-1], z[first - 1:first + 1][::-1])
        r = np.r_[r_cap, r[first:]]
        z = np.r_[zc, z[first:]]
    # resample by arc length to ~72 rows
    s = np.r_[0, np.cumsum(np.hypot(np.diff(r), np.diff(z)))]
    sq = np.unique(np.r_[np.linspace(0, s[-1], 72), np.interp(np.argmin(r), np.arange(len(s)), s)])
    r, z = np.interp(sq, s, r), np.interp(sq, s, z)
    # profile is ordered top (chamber) -> exit, i.e. walking downward; outer surface needs upward walk
    ztop, zexit = z[0], z[-1]
    v = (z - zexit) / (ztop - zexit)
    out = {}
    out["nozzle"] = revolve(r[::-1], z[::-1], 64, v=v[::-1])
    t_wall = 0.006
    out["nozzle_inner"] = revolve(r - t_wall, z, 64)
    r_e = r[-1]
    lip = revolve([r_e - t_wall, r_e + 0.012], [zexit, zexit], 64)  # normal -z
    lip.extend(revolve([r_e + 0.012, r_e + 0.012], [zexit, zexit + 0.035], 64))
    lip.extend(revolve([r_e + 0.012, r_e + 0.001], [zexit + 0.035, zexit + 0.04], 64))
    # two hat-band stiffeners on the diverging section
    i_t2 = int(np.argmin(r))
    for frac in (0.4, 0.75):
        zs = z[i_t2] + frac * (zexit - z[i_t2])
        rs_ = float(np.interp(-zs, -z[i_t2:], r[i_t2:]))
        lip.extend(revolve([rs_ - 0.002, rs_ + 0.008, rs_ + 0.008, rs_ - 0.002], [zs - 0.012, zs - 0.006, zs + 0.006, zs + 0.012], 64))
    out["nozzle_bands"] = lip
    # chamber, injector dome, powerhead, turbopump, gimbal block
    rc = r[0]
    zc0 = ztop
    ch = Mesh()
    ch.extend(revolve([rc, rc], [zc0, zc0 + 0.15], 48))
    dome_t = np.linspace(0, math.pi / 2, 8)
    ch.extend(revolve(0.17 + (rc - 0.17) * np.cos(dome_t), zc0 + 0.15 + 0.08 * np.sin(dome_t), 48))
    ch.extend(revolve([0.17, 0.17], [zc0 + 0.23, zc0 + 0.42], 48))
    ch.extend(revolve([0.17, 0.0], [zc0 + 0.42, zc0 + 0.42], 48))
    pump = Mesh()
    pump.extend(tube((-0.13, 0.11, zc0 + 0.30), (0.13, 0.11, zc0 + 0.30), 0.07, 24))
    pump.extend(tube((0.06, 0.11, zc0 + 0.30), (0.06, rc * 0.75, zc0 + 0.06), 0.03, 16))
    pump.extend(tube((-0.06, -0.12, zc0 + 0.38), (-0.06, -rc * 0.75, zc0 + 0.08), 0.028, 16))
    pump.extend(box((0, 0, zc0 + 0.47), np.eye(3), (0.22, 0.22, 0.08)))
    out["powerhead"] = ch
    out["plumbing"] = pump
    # flexible boot where the nozzle passes the heat shield (in this unscaled frame)
    z_hs = (S.Z_HEATSHIELD - (-3.0)) / sc
    r_at = lambda zz: float(np.interp(-zz, -z, r))
    nb = 10
    zb = np.linspace(z_hs + 0.002 / sc, z_hs - 0.07, nb)
    rb = np.array([r_at(q) for q in zb]) + np.linspace(0.075, 0.012, nb) + 0.008 * np.cos(np.linspace(0, 4 * math.pi, nb))
    out["boot"] = revolve(rb[::-1], zb[::-1], 64)
    if tvc:  # two gimbal actuators (gimbal_rcs only), from the heat shield to the nozzle collar
        act, rods = Mesh(), Mesh()
        z_att = -0.36
        for phi in (0.0, math.pi / 2):
            e = np.array([math.cos(phi), math.sin(phi), 0])
            p0 = 0.43 * e + np.array([0, 0, z_hs - 0.005])
            p1 = (r_at(z_att) + 0.03) * e + np.array([0, 0, z_att])
            d = p1 - p0
            act.extend(tube(p0, p0 + 0.55 * d, 0.03, 16))
            act.extend(sphere(p0, 0.035, 12, 6))
            act.extend(sphere(p1, 0.03, 12, 6))
            rods.extend(tube(p0 + 0.5 * d, p1, 0.016, 12))
        # collar on the nozzle where the actuators attach
        rr = r_at(z_att)
        act.extend(revolve([rr - 0.002, rr + 0.02, rr + 0.02, rr - 0.002], [z_att - 0.03, z_att - 0.02, z_att + 0.02, z_att + 0.03], 64))
        out["tvc_actuators"] = act
        out["tvc_rods"] = rods
    return out, (float(zexit), float(r_e))


# ------------------------------------------------------------------------------------------- site
def _site_height(x, y):
    """Visual terrain height: exactly 0 within 60 m of the pad, gentle undulation further out."""
    r = np.hypot(x, y)
    w = np.clip((r - 60.0) / 240.0, 0, 1)
    w = w * w * (3 - 2 * w)
    h = (0.9 * np.sin(x / 57.0 + 0.3) * np.cos(y / 71.0 - 1.1) + 0.6 * np.sin((x + 2 * y) / 131.0)
         + 0.4 * np.cos((2 * x - y) / 43.0))
    return w * h * (1.0 + 2.0 * np.clip((r - 600) / 1500, 0, 1))


def ground_mesh(r_max=2500.0, nseg=256):
    rings = np.r_[0.0, np.arange(3.0, 60.0, 3.0), np.geomspace(60.0, r_max, 40)]
    th = 2 * math.pi * np.arange(nseg + 1) / nseg
    X = rings[:, None] * np.cos(th)[None]
    Y = rings[:, None] * np.sin(th)[None]
    Z = _site_height(X, Y)
    P = np.stack([X, Y, Z], -1).reshape(-1, 3)
    # numerical normals
    eps = 0.5
    dzdx = (_site_height(X + eps, Y) - _site_height(X - eps, Y)) / (2 * eps)
    dzdy = (_site_height(X, Y + eps) - _site_height(X, Y - eps)) / (2 * eps)
    N = np.stack([-dzdx, -dzdy, np.ones_like(X)], -1).reshape(-1, 3)
    UV = np.stack([X, Y], -1).reshape(-1, 2) / S.GROUND_TILE_M
    m_, n_ = len(rings), nseg + 1
    j, k = np.meshgrid(np.arange(m_ - 1), np.arange(nseg), indexing="ij")
    a = j * n_ + k
    faces = np.stack([a, a + n_, a + n_ + 1, a + 1], -1).reshape(-1, 4)  # CCW seen from above
    return Mesh().add(P, N, UV, faces)


def ridges_mesh(r0=2500.0, r1=7000.0, nseg=512, nr=24):
    rng = np.random.default_rng(7)
    th = 2 * math.pi * np.arange(nseg + 1) / nseg
    # ridge crest height vs azimuth: a few ranges, low gaps
    crest = np.zeros_like(th)
    for k in range(1, 14):
        crest += rng.normal(0, 1) / k ** 1.1 * np.cos(k * th + rng.uniform(0, 2 * math.pi))
    crest = 160 + 140 * crest / np.abs(crest).max()
    crest += 25 * np.sin(th * 37) * np.sin(th * 11)
    crest = np.clip(crest, 30, None)
    rr = np.linspace(r0, r1, nr)
    prof = np.sin(np.clip((rr - r0) / (r1 - r0), 0, 1) * math.pi) ** 1.5
    X = rr[:, None] * np.cos(th)[None]
    Y = rr[:, None] * np.sin(th)[None]
    detail = 1 + 0.25 * np.sin(rr[:, None] / 97.0 + th[None] * 23) * np.sin(th[None] * 51 + rr[:, None] / 300)
    Z = prof[:, None] * crest[None] * detail + _site_height(X, Y) * (rr[:, None] <= r0 + 1)
    P = np.stack([X, Y, Z], -1)
    # normals from grid differences
    dPr = np.gradient(P, axis=0)
    dPt = np.gradient(P, axis=1)
    N = np.cross(dPr, dPt)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    N[N[..., 2] < 0] *= -1
    UV = np.stack([np.repeat((8 * np.arange(nseg + 1) / nseg)[None], nr, 0), Z / 450.0], -1)
    n_ = nseg + 1
    j, k = np.meshgrid(np.arange(nr - 1), np.arange(nseg), indexing="ij")
    a = j * n_ + k
    faces = np.stack([a, a + n_, a + n_ + 1, a + 1], -1).reshape(-1, 4)
    return Mesh().add(P.reshape(-1, 3), N.reshape(-1, 3), UV.reshape(-1, 2), faces)


def pad_mesh():
    R = S.PAD_R
    top = revolve(np.linspace(R - 0.06, 0.0, 6), np.full(6, S.PAD_TOP_Z), 192)
    edge = revolve([R, R - 0.06], [-0.02, S.PAD_TOP_Z], 192)
    side = revolve([R, R], [-0.25, -0.02], 192)
    m = Mesh().extend(top).extend(edge).extend(side)
    P = np.concatenate(m.P)
    m.UV = [np.concatenate([p[:, :2] for p in m.P]) / S.PAD_TEX_SPAN + 0.5]
    return m


def pad_props():
    fixtures, lenses, barriers = Mesh(), Mesh(), Mesh()
    for k in range(16):
        a = 2 * math.pi * (k + 0.5) / 16
        c = np.array([math.cos(a), math.sin(a), 0]) * (S.PAD_R + 0.45)
        fixtures.extend(tube(c + [0, 0, -0.05], c + [0, 0, 0.09], 0.075, 20))
        lenses.extend(tube(c + [0, 0, 0.09], c + [0, 0, 0.13], 0.05, 20))
    # a short arc of concrete jersey barriers at 24 m (scale reference / blast screen)
    prof_w = [0.30, 0.30, 0.20, 0.08, 0.08]
    for k in range(7):
        a = math.radians(200 + 6.2 * k)
        A = radial_axes(a)
        c = A[:, 0] * 24.0
        L = 2.5
        # trapezoid section: stack 2 boxes (base + top) for a jersey-barrier silhouette
        barriers.extend(box(c + [0, 0, 0.20], A, (0.60, L, 0.40)))
        barriers.extend(box(c + [0, 0, 0.62], A, (0.22, L, 0.45)))
        barriers.extend(box(c + [0, 0, 0.44], A, (0.40, L, 0.08)))
    return fixtures, lenses, barriers
