"""CFD initialisation (1D isentropic), exit-plane integration, QA and GCI.

Exit-plane stream thrust per full revolution:
    S = sum_f [ rho_f (U_f . Sf_f) U_x,f + p_f Sf_x,f ] * wedge_factor
    F(Pa) = S - Pa * A_exit              (attached, full-flowing nozzle)
Mass flow at inlet and outlet gives the conservation residual.
Because the inviscid perfect-gas solution scales with Pc (fixed T0),
F(throttle, Pa) = throttle * S_nom - Pa * A_exit and mdot = throttle * mdot_nom.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from rocket_dt.cfd import foam_io
from rocket_dt.propulsion import nozzle1d
from rocket_dt.propulsion.geometry import build_contour
from rocket_dt.schemas import EngineDesign


def init_isentropic(case: Path, d: EngineDesign) -> None:
    meta = json.loads((case / "case_meta.json").read_text())
    mesh = foam_io.Mesh(case)
    cc = mesh.cell_centres()
    c = build_contour(d)
    g, R = d.gas.gamma, d.gas.r_specific
    pc, tc = meta["chamber_pressure_pa"], meta["chamber_temperature_k"]
    rt = c.r_m[c.throat_index]
    p = np.empty(len(cc)); T = np.empty(len(cc)); U = np.zeros((len(cc), 3))
    for i, (x, _, _) in enumerate(cc):
        rw = np.interp(x, c.x_m, c.r_m)
        ar = max((rw / rt) ** 2, 1.0 + 1e-9)
        if x < 0:
            m = _subsonic_mach(ar, g)
        else:
            m = nozzle1d.supersonic_mach_from_area_ratio(ar, g)
        t = tc / (1 + 0.5 * (g - 1) * m * m)
        p[i] = pc * (t / tc) ** (g / (g - 1))
        T[i] = t
        U[i, 0] = m * math.sqrt(g * R * t)
    foam_io.set_internal(case / "0/p", p, "scalar")
    foam_io.set_internal(case / "0/T", T, "scalar")
    foam_io.set_internal(case / "0/U", U, "vector")


def _subsonic_mach(ar, g):
    from scipy.optimize import brentq
    return brentq(lambda m: nozzle1d.area_ratio_from_mach(m, g) - ar, 1e-6, 1.0)


def time_dirs(case: Path) -> list[str]:
    ts = []
    for p in Path(case).iterdir():
        try:
            if p.is_dir() and float(p.name) > 0:
                ts.append(p.name)
        except ValueError:
            pass
    return sorted(ts, key=float)


def integrate(case: Path, time: str, mesh: foam_io.Mesh | None = None) -> dict:
    meta = json.loads((Path(case) / "case_meta.json").read_text())
    mesh = mesh or foam_io.Mesh(case)
    wf = meta["wedge_to_full_factor"]
    out = {}
    for patch in ("inlet", "outlet"):
        sl = mesh.patch_slice(patch)
        n = sl.stop - sl.start
        own = mesh.owner[sl]
        sf = mesh.face_areas[sl]
        rho = foam_io.read_field_patch(case, time, "rho", patch, n, own)
        p = foam_io.read_field_patch(case, time, "p", patch, n, own)
        U = foam_io.read_field_patch(case, time, "U", patch, n, own)
        flux = rho * np.einsum("ij,ij->i", U, sf)
        out[patch] = {"mdot_kg_s": float(np.sum(flux) * wf),
                      "stream_thrust_n": float(np.sum(flux * U[:, 0] + p * sf[:, 0]) * wf),
                      "area_m2": float(np.sum(np.abs(sf[:, 0])) * wf),
                      "p_area_avg_pa": float(np.sum(p * np.abs(sf[:, 0])) / np.sum(np.abs(sf[:, 0])))}
    mi, mo = -out["inlet"]["mdot_kg_s"], out["outlet"]["mdot_kg_s"]  # inlet Sf points outward
    return {"time": float(time), "mdot_in": mi, "mdot_out": mo,
            "mass_imbalance": (mi - mo) / mo if mo else float("nan"),
            "stream_thrust_n": out["outlet"]["stream_thrust_n"], "exit_area_m2": out["outlet"]["area_m2"],
            "exit_p_avg_pa": out["outlet"]["p_area_avg_pa"]}


def history(case: Path) -> list[dict]:
    mesh = foam_io.Mesh(case)
    return [integrate(case, t, mesh) for t in time_dirs(case)]


def qa(case: Path, hist: list[dict], tol_thrust=2e-4, tol_mass=2e-3) -> dict:
    """Convergence QA: last two writes agree and mass is conserved inlet->outlet."""
    if len(hist) < 2:
        return {"status": "FAIL", "reason": "fewer than 2 written times"}
    a, b = hist[-2], hist[-1]
    d_s = abs(b["stream_thrust_n"] - a["stream_thrust_n"]) / abs(b["stream_thrust_n"])
    ok_s, ok_m = d_s < tol_thrust, abs(b["mass_imbalance"]) < tol_mass
    finite = all(np.isfinite([b["stream_thrust_n"], b["mdot_out"]]))
    return {"status": "PASS" if (ok_s and ok_m and finite) else "FAIL",
            "rel_change_stream_thrust_last_write": d_s, "mass_imbalance": b["mass_imbalance"],
            "tol_thrust": tol_thrust, "tol_mass": tol_mass}


def gci(f1: float, f2: float, f3: float, r21: float, r32: float, fs: float = 1.25) -> dict:
    """Celik et al. (2008) GCI on three grids, 1 = finest. Fixed-point solve for p."""
    e21, e32 = f2 - f1, f3 - f2
    if e21 == 0 or e32 == 0:
        return {"p": float("nan"), "gci_fine": 0.0, "f_extrap": f1, "monotone": True, "asymptotic_ratio": float("nan")}
    s = np.sign(e32 / e21)
    p = abs(math.log(abs(e32 / e21)) / math.log(r21))
    for _ in range(100):
        q = math.log((r21 ** p - s) / (r32 ** p - s))
        p = abs(math.log(abs(e32 / e21)) + q) / math.log(r21)
    f_ext = (r21 ** p * f1 - f2) / (r21 ** p - 1)
    ea21 = abs((f1 - f2) / f1)
    gci21 = fs * ea21 / (r21 ** p - 1)
    ea32 = abs((f2 - f3) / f2)
    gci32 = fs * ea32 / (r32 ** p - 1)
    return {"p": p, "f_extrap": f_ext, "gci_fine": gci21, "gci_medium": gci32,
            "asymptotic_ratio": gci32 / (r21 ** p * gci21) if gci21 else float("nan"),
            "monotone": bool(s > 0)}
