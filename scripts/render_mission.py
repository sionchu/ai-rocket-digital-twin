"""Webcast-style mission replay: vehicle close-up with exhaust, engine status panel, CFD inset, telemetry.

Kinematics come from the Python reference 6-DoF run (truth states). The exhaust drawn
outside the nozzle is illustrative: its length scales with thrust and its spread with
Pe/Pa from the ideal-1D model (under-expanded -> widens, over-expanded -> necks with
diamond markers). The CFD inset shows the actual rhoCentralFoam Mach field inside the
nozzle (cloud-v1912, throttle 1.0, sea level), which is the only part that is solved.

Usage: python scripts/render_mission.py --engine conical_A [--rcs-failed rcs_01 rcs_03 --fault-unknown --seed 1001]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rocket_dt.cfd import foam_io  # noqa: E402
from rocket_dt.dynamics.frames import isa_pressure, quat_to_rotmat, tilt_angle  # noqa: E402
from rocket_dt.evaluation import experiments as ex  # noqa: E402
from rocket_dt.evaluation.simulate import LANDING_LIMITS, run_scenario  # noqa: E402
from rocket_dt.propulsion import nozzle1d  # noqa: E402
from rocket_dt.propulsion.adapter import gimbal_direction  # noqa: E402
from rocket_dt.propulsion.geometry import build_contour  # noqa: E402

FPS = 25
BG, FG, DIM, ACC = "#0b0f17", "#e8edf5", "#7f8a9c", "#ffb347"
LABEL = "cloud-v1912 | Python reference 6-DoF + PD | exhaust outside nozzle is illustrative (1D Pe/Pa), CFD inset is solved"


def mach_field(engine: str):
    """Last-time Mach field of the L2 CFD case, cached as npz; None if no CFD for this engine."""
    cache = REPO / f"runs/video/cache/{engine}_L2_mach.npz"
    if cache.exists():
        d = np.load(cache)
        return d["x"], d["r"], d["ma"]
    case = REPO / f"runs/cfd/{engine}_L2"
    if not case.exists():
        return None
    _, eng, _ = ex.load_all(REPO)
    gas = eng[engine].gas
    mesh = foam_io.Mesh(case)
    cc = mesh.cell_centres()
    t = sorted((d.name for d in case.iterdir() if d.name[0].isdigit()), key=float)[-1]
    u = foam_io.read_internal(case, t, "U")
    temp = foam_io.read_internal(case, t, "T")
    ma = np.linalg.norm(u, axis=1) / np.sqrt(gas.gamma * gas.r_specific * temp)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, x=cc[:, 0], r=cc[:, 1], ma=ma)
    return cc[:, 0], cc[:, 1], ma


def cylinder_faces(a, b, radius, n=12):
    ax = b - a
    ax = ax / np.linalg.norm(ax)
    t = np.cross(ax, [1.0, 0, 0])
    if np.linalg.norm(t) < 1e-6:
        t = np.cross(ax, [0, 1.0, 0])
    t /= np.linalg.norm(t)
    s = np.cross(ax, t)
    ang = np.linspace(0, 2 * np.pi, n + 1)
    ring = [radius * (np.cos(q) * t + np.sin(q) * s) for q in ang]
    return [[a + ring[k], a + ring[k + 1], b + ring[k + 1], b + ring[k]] for k in range(n)]


def cone_faces(apex_base, tip, r_base, n=12, r_tip=0.0):
    return cylinder_faces_var(apex_base, tip, r_base, r_tip, n)


def cylinder_faces_var(a, b, ra, rb, n=12):
    ax = b - a
    ax = ax / np.linalg.norm(ax)
    t = np.cross(ax, [1.0, 0, 0])
    if np.linalg.norm(t) < 1e-6:
        t = np.cross(ax, [0, 1.0, 0])
    t /= np.linalg.norm(t)
    s = np.cross(ax, t)
    ang = np.linspace(0, 2 * np.pi, n + 1)
    return [[a + ra * (np.cos(ang[k]) * t + np.sin(ang[k]) * s), a + ra * (np.cos(ang[k + 1]) * t + np.sin(ang[k + 1]) * s),
             b + rb * (np.cos(ang[k + 1]) * t + np.sin(ang[k + 1]) * s), b + rb * (np.cos(ang[k]) * t + np.sin(ang[k]) * s)]
            for k in range(n)]


class Mission:
    def __init__(self, engine, sc, failed, title):
        veh, eng, maps = ex.load_all(REPO)
        self.veh, self.e, self.engine, self.sc, self.failed, self.title = veh.with_engine(engine), eng[engine], engine, sc, failed, title
        self.log, self.met, _ = run_scenario(sc, veh, eng[engine], eng[engine], maps[engine])
        self.n = len(self.log["t"])
        self.rcs = veh.layout.rcs
        self.half = veh.mass.structure_length_m / 2
        self.rad = veh.mass.structure_radius_m
        self.contour = build_contour(self.e)
        self.cfd = mach_field(engine)
        self.td = self.met.get("touchdown")

    def nozzle_point(self, i):
        thr = self.log["throttle"][i]
        if thr <= 0:
            return None
        return nozzle1d.evaluate(self.e, float(thr), isa_pressure(max(float(self.log["p"][i, 2]), 0.0)))

    def phase(self, i):
        t, h = self.log["t"][i], self.log["p"][i, 2]
        if self.td and t >= self.td["t"]:
            return "TOUCHDOWN / ENGINE CUTOFF"
        return "LANDING BURN" if h < 15 else "POWERED DESCENT"


def draw_scene(ax, m: Mission, i):
    lg = m.log
    ax.cla()
    ax.set_facecolor(BG)
    R = quat_to_rotmat(lg["q"][i])
    origin = lg["p"][i] - R @ lg["com_b"][i]
    c = lg["p"][i]
    span = 9.0
    ax.set_xlim(c[0] - span, c[0] + span)
    ax.set_ylim(c[1] - span, c[1] + span)
    z0 = max(c[2] - 11, -0.5)
    ax.set_zlim(z0, z0 + 2 * span)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=8, azim=-55 + 0.15 * i / FPS * 10)
    ax.set_axis_off()
    # ground + pad
    if z0 < 1:
        g = np.array([[c[0] - span, c[1] - span, 0], [c[0] + span, c[1] - span, 0], [c[0] + span, c[1] + span, 0], [c[0] - span, c[1] + span, 0]])
        ax.add_collection3d(Poly3DCollection([g], facecolor="#1d2633", edgecolor="none", alpha=0.9))
        th = np.linspace(0, 2 * np.pi, 60)
        tw = np.asarray(m.sc.target_w)
        ax.plot(tw[0] + 2 * np.cos(th), tw[1] + 2 * np.sin(th), 0.02, color="#ffd400", lw=1.5)
        ax.plot([tw[0] - 1, tw[0] + 1], [tw[1], tw[1]], 0.02, color="#ffd400", lw=1)
        ax.plot([tw[0], tw[0]], [tw[1] - 1, tw[1] + 1], 0.02, color="#ffd400", lw=1)
    # trail
    j0 = max(0, i - 200)
    ax.plot(lg["p"][j0:i + 1, 0], lg["p"][j0:i + 1, 1], lg["p"][j0:i + 1, 2], color="#3d7bd9", lw=0.8, alpha=0.6)
    # body
    bot, top = origin + R @ [0, 0, -m.half], origin + R @ [0, 0, m.half]
    ax.add_collection3d(Poly3DCollection(cylinder_faces(bot, top, m.rad), facecolor="#d9dee7", edgecolor="#9aa3b2", lw=0.2))
    nose = origin + R @ [0, 0, m.half + 1.2]
    ax.add_collection3d(Poly3DCollection(cylinder_faces_var(top, nose, m.rad, 0.02), facecolor="#c9ced8", edgecolor="#9aa3b2", lw=0.2))
    for f in m.veh.gear.foot_positions_body_m:
        a = origin + R @ [0, 0, -m.half + 1.0]
        b = origin + R @ np.asarray(f)
        ax.plot(*np.c_[a, b], color="#5b6577", lw=2)
    # engine bell + exhaust
    pivot = np.asarray(m.veh.layout.main_engine.gimbal_pivot_body_m, float)
    d = gimbal_direction(*lg["gimbal"][i])
    ce = m.contour
    L_n = float(ce.x_m[-1] - ce.x_m[0]) * 6.0  # drawn 6x for visibility
    r_e = float(ce.r_m[-1]) * 6.0
    throat_w = origin + R @ (pivot - 0.25 * d)
    exit_w = origin + R @ (pivot - (0.25 + L_n) * d)
    ax.add_collection3d(Poly3DCollection(cylinder_faces_var(throat_w, exit_w, 0.12, r_e), facecolor="#6b4a2b", edgecolor="#3a2a1a", lw=0.2))
    npnt = m.nozzle_point(i)
    if npnt is not None:
        dw = R @ d
        ratio = npnt.exit_pressure_pa / npnt.ambient_pressure_pa
        length = 2.0 + 9.0 * npnt.thrust_n / 30000.0
        spread = np.clip(ratio, 0.3, 2.5) ** 0.5
        flick = 1 + 0.06 * np.sin(1.7 * i) + 0.04 * np.sin(5.3 * i)
        layers = [(1.0, 1.6 * spread, "#ff4d00", 0.18), (0.75, 1.0 * spread, "#ff9a1f", 0.4), (0.45, 0.55 * spread, "#fff3c4", 0.85)]
        for fl, wf, col, al in layers:
            tip = exit_w - dw * length * fl * flick
            ax.add_collection3d(Poly3DCollection(cylinder_faces_var(exit_w, tip, r_e * (1.0 if wf > 1 else 0.9), r_e * wf, 10),
                                                 facecolor=col, edgecolor="none", alpha=al))
        if ratio < 1.0:  # over-expanded: shock diamonds (illustrative)
            for k in range(1, 4):
                pt = exit_w - dw * (0.9 * k * r_e * 2.2)
                ax.scatter(*pt, color="#ffffff", s=18, alpha=0.8, depthshade=False)
        # thrust vector
        tv = -dw * 3.0
        ax.quiver(*(origin + R @ pivot), *(-tv), color="#39d353", lw=1.5, arrow_length_ratio=0.25)
    # RCS jets
    frac = lg["rcs_frac"][i]
    for k, t in enumerate(m.rcs):
        pos = origin + R @ np.asarray(t.position_body_m)
        if m.failed[k]:
            ax.scatter(*pos, color="#ff3b3b", marker="x", s=30, depthshade=False)
        elif frac[k] > 0.05:
            jet = -(R @ np.asarray(t.force_direction_body)) * 1.6 * frac[k]
            ax.plot(*np.c_[pos, pos + jet], color="#bfe8ff", lw=3, alpha=0.9)


def draw_engine_panel(ax, m: Mission, i):
    lg = m.log
    ax.cla()
    ax.set_facecolor(BG)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    thr = lg["throttle"][i]
    on = thr > 0
    ax.add_patch(Circle((0, 0), 0.62, facecolor=ACC if on else "#2a3240", edgecolor=FG, lw=1.5, alpha=0.35 + 0.65 * thr if on else 1))
    gmax = m.veh.layout.main_engine.max_gimbal_rad
    gx, gy = lg["gimbal"][i]
    ax.add_patch(Circle((0, 0), 0.5, facecolor="none", edgecolor=DIM, lw=0.6, ls="--"))
    ax.plot([-0.5, 0.5], [0, 0], color=DIM, lw=0.4)
    ax.plot([0, 0], [-0.5, 0.5], color=DIM, lw=0.4)
    ax.add_patch(Circle((0.5 * gy / gmax, -0.5 * gx / gmax), 0.07, color="#39d353"))
    ax.text(0, -0.85, f"MAIN  {100 * thr:5.1f}%", color=FG, ha="center", fontsize=10, family="monospace", weight="bold")
    ax.text(0, -1.05, f"gimbal {np.degrees(gx):+4.1f}° / {np.degrees(gy):+4.1f}°", color=DIM, ha="center", fontsize=8, family="monospace")
    frac = lg["rcs_frac"][i]
    for k, t in enumerate(m.rcs):
        x, y = t.position_body_m[0], t.position_body_m[1]
        rr = np.hypot(x, y) or 1
        fd = np.asarray(t.force_direction_body)[:2]
        px, py = 1.1 * x / rr + 0.3 * fd[0], 1.1 * y / rr + 0.3 * fd[1]  # thrusters sharing a station are offset along their jet axis
        if m.failed[k]:
            ax.add_patch(Circle((px, py), 0.17, facecolor="#3a1010", edgecolor="#ff3b3b", lw=1.5))
            ax.text(px, py, "✕", color="#ff3b3b", ha="center", va="center", fontsize=11, weight="bold")
        else:
            lit = frac[k] > 0.05
            ax.add_patch(Circle((px, py), 0.17, facecolor="#7fd3ff" if lit else "#1f2733", edgecolor=FG, lw=1))
        ax.text(px, py - 0.3, t.id.replace("rcs_", "R"), color=DIM, ha="center", fontsize=7, family="monospace")
    ax.set_title(f"ENGINE STATUS — {m.engine}", color=FG, fontsize=10, loc="left", family="monospace")


def draw_cfd_field(ax, m: Mission):
    """Static: drawn once per video."""
    ax.cla()
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(DIM)
    if m.cfd is not None:
        x, r, ma = m.cfd
        xs, rs, ms = np.r_[x, x], np.r_[r, -r], np.r_[ma, ma]
        ax.tricontourf(xs * 100, rs * 100, ms, levels=np.linspace(0, ms.max(), 24), cmap="inferno")
        ax.set_title(f"CFD Mach (rhoCentralFoam L2, v1912)  Me={ma[x > x.max() - 2e-3].mean():.2f}", color=FG, fontsize=8, loc="left", family="monospace")
    else:
        ce = m.contour
        ax.fill_between(ce.x_m * 100, -ce.r_m * 100, ce.r_m * 100, color="#2a3240")
        ax.text(ce.x_m.mean() * 100, 0, "CFD NOT_RUN\n(1D only)", color=DIM, ha="center", va="center", fontsize=9, family="monospace")
        ax.set_title("nozzle contour (no CFD case)", color=FG, fontsize=8, loc="left", family="monospace")
    ce = m.contour
    ax.plot(ce.x_m * 100, ce.r_m * 100, color=FG, lw=1)
    ax.plot(ce.x_m * 100, -ce.r_m * 100, color=FG, lw=1)
    ax.set_aspect("equal")
    ax.tick_params(colors=DIM, labelsize=6)
    ax.set_xlabel("x [cm]", color=DIM, fontsize=6)


def draw_cfd_gauge(axb, m: Mission, i):
    axb.cla()
    axb.set_facecolor(BG)
    axb.set_xlim(0, 2.0)
    axb.set_ylim(0, 1)
    axb.set_yticks([])
    axb.tick_params(colors=DIM, labelsize=6)
    for s in axb.spines.values():
        s.set_color(DIM)
    npnt = m.nozzle_point(i)
    ratio = npnt.exit_pressure_pa / npnt.ambient_pressure_pa if npnt else 0.0
    col = "#ff3b3b" if (npnt and ratio < 0.4) else ("#ffb347" if ratio < 1 else "#39d353")
    axb.add_patch(Rectangle((0, 0.2), min(ratio, 2.0), 0.6, color=col))
    axb.axvline(0.4, color="#ff3b3b", lw=1, ls="--")
    axb.axvline(1.0, color=FG, lw=0.6, ls=":")
    state = "—" if not npnt else ("SEPARATION RISK" if ratio < 0.4 else ("over-expanded" if ratio < 1 else "under-expanded"))
    axb.set_title(f"Pe/Pa = {ratio:4.2f}  {state}", color=col if npnt else DIM, fontsize=8, loc="left", family="monospace")


def draw_telemetry(ax, m: Mission, i):
    lg = m.log
    ax.cla()
    ax.set_facecolor(BG)
    ax.axis("off")
    v = lg["v"][i]
    npnt = m.nozzle_point(i)
    rows = [("T+", f"{lg['t'][i]:6.2f} s"), ("ALTITUDE", f"{lg['p'][i, 2]:6.1f} m"), ("V-SPEED", f"{v[2]:+6.2f} m/s"),
            ("H-SPEED", f"{np.hypot(v[0], v[1]):6.2f} m/s"), ("TILT", f"{np.degrees(tilt_angle(lg['q'][i])):6.1f} °"),
            ("THRUST", f"{lg['thrust'][i] / 1000:6.2f} kN"), ("Isp", f"{npnt.isp_s:6.1f} s" if npnt else "    —"),
            ("PROPELLANT", f"{lg['m_prop'][i]:6.1f} kg"), ("MAP VALID", "YES" if lg["map_valid"][i] else "NO (sep. risk)")]
    for k, (a, b) in enumerate(rows):
        y = 1 - k * 0.11
        ax.text(0.0, y, a, color=DIM, fontsize=9, family="monospace", transform=ax.transAxes, va="top")
        warn = a == "MAP VALID" and not lg["map_valid"][i]
        ax.text(1.0, y, b, color="#ff3b3b" if warn else FG, fontsize=11, family="monospace", weight="bold",
                ha="right", transform=ax.transAxes, va="top")


def draw_strip(ax, m: Mission, i):
    lg = m.log
    ax.cla()
    ax.set_facecolor(BG)
    t = lg["t"]
    ax.plot(t, lg["p"][:, 2], color="#3d7bd9", lw=1, alpha=0.35)
    ax.plot(t[:i + 1], lg["p"][:i + 1, 2], color="#3d7bd9", lw=1.6, label="altitude [m]")
    if getattr(m, "_twin", None) is None:  # create once; a twinx per frame stacks axes and slows every draw
        m._twin = ax.twinx()
    ax2 = m._twin
    ax2.cla()
    ax2.plot(t, lg["thrust"] / 1000, color=ACC, lw=1, alpha=0.35)
    ax2.plot(t[:i + 1], lg["thrust"][:i + 1] / 1000, color=ACC, lw=1.6, label="thrust [kN]")
    ax2.plot(t[:i + 1], -lg["v"][:i + 1, 2] * 3, color="#39d353", lw=1.2, label="|vz| x3 [m/s]")
    ax.axvline(t[i], color=FG, lw=0.6)
    for a in (ax, ax2):
        a.tick_params(colors=DIM, labelsize=7)
        for s in a.spines.values():
            s.set_color("#2a3240")
    ax.set_xlim(0, t[-1])
    ax.set_ylim(0, max(lg["p"][:, 2].max(), 1) * 1.05)
    ax2.set_ylim(0, 35)
    ax.legend(loc="upper left", fontsize=7, facecolor=BG, labelcolor=FG, frameon=False)
    ax2.legend(loc="upper right", fontsize=7, facecolor=BG, labelcolor=FG, frameon=False)
    ax2.patch.set_visible(False)


def render(m: Mission, path: Path, tail_s=3.0):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
    a3 = fig.add_axes([0.0, 0.2, 0.62, 0.74], projection="3d")
    aeng = fig.add_axes([0.63, 0.62, 0.17, 0.32])
    atel = fig.add_axes([0.82, 0.60, 0.16, 0.33])
    acfd = fig.add_axes([0.655, 0.32, 0.32, 0.22])
    abar = fig.add_axes([0.655, 0.235, 0.32, 0.035])
    astrip = fig.add_axes([0.04, 0.05, 0.92, 0.13])
    top = fig.text(0.015, 0.965, "", color=FG, fontsize=15, weight="bold", family="monospace")
    sub = fig.text(0.015, 0.94, "", color=ACC, fontsize=11, family="monospace")
    res = fig.text(0.31, 0.24, "", color=FG, fontsize=18, weight="bold", ha="center", family="monospace")
    fig.text(0.5, 0.008, LABEL, color=DIM, fontsize=8, ha="center")
    frames = list(range(0, m.n, max(1, 50 // FPS))) + [m.n - 1] * int(tail_s * FPS)
    w = FFMpegWriter(fps=FPS, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "22"])
    draw_cfd_field(acfd, m)
    with w.saving(fig, str(path), dpi=100):
        for f in frames:
            draw_scene(a3, m, f)
            draw_engine_panel(aeng, m, f)
            draw_telemetry(atel, m, f)
            draw_cfd_gauge(abar, m, f)
            draw_strip(astrip, m, f)
            top.set_text(m.title)
            sub.set_text(m.phase(f))
            if m.td and m.log["t"][f] >= m.td["t"] + 0.5:
                ok = m.met["success"] and m.met["physics_valid"]
                txt = "LANDED" if m.met["success"] else "FAILED: " + ", ".join(m.met["failure_reasons"][:2])
                if m.met["success"] and not m.met["physics_valid"]:
                    txt += " (physics_valid = false)"
                res.set_text(txt + f"\nΔpos {m.td['pos_err']:.2f} m   vz {m.td['vz']:+.2f} m/s")
                res.set_color("#39d353" if ok else "#ff3b3b")
            else:
                res.set_text("")
            w.grab_frame(facecolor=BG)
    plt.close(fig)
    return len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="conical_A", choices=ex.ENGINES)
    ap.add_argument("--rcs-failed", nargs="*", default=[])
    ap.add_argument("--fault-unknown", action="store_true")
    ap.add_argument("--seed", type=int, default=None, help="randomized landing seed; default = configs/scenarios/landing.yaml")
    ap.add_argument("--out", default="runs/video/mission")
    a = ap.parse_args()
    base = ex._load_sc(REPO, "landing")
    sc = ex.randomized_scenarios(base, 1, a.seed)[0] if a.seed is not None else base
    sc = replace(sc, rcs_failed=list(a.rcs_failed), fault_known_to_controller=not a.fault_unknown)
    veh = ex.load_all(REPO)[0]
    failed = np.array([t.id in a.rcs_failed for t in veh.layout.rcs])
    tag = a.engine + (f"_rcsfail_{'-'.join(a.rcs_failed)}{'_unknown' if a.fault_unknown else '_known'}" if a.rcs_failed else "")
    tag += f"_s{a.seed}" if a.seed is not None else ""
    title = f"VTVL DIGITAL TWIN  |  {a.engine}  |  {sc.name}" + (f"  |  RCS FAULT {'+'.join(a.rcs_failed)} ({'unknown' if a.fault_unknown else 'known'} to controller)" if a.rcs_failed else "")
    m = Mission(a.engine, sc, failed, title)
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    nf = render(m, out / f"{tag}.mp4")
    (out / f"{tag}.json").write_text(json.dumps({"label": LABEL, "frames": nf, "limits": LANDING_LIMITS, "metrics": m.met},
                                                indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(tag, nf, m.met["success"], m.met["physics_valid"])


if __name__ == "__main__":
    main()
