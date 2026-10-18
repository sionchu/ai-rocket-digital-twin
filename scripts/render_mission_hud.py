"""Composite Isaac renders + per-step telemetry into a mission video with a webcast-style HUD (label `win-isaac`).

Inputs: runs/ppo_win-isaac/mission/<name>/{raw.mp4, telemetry.json} from scripts/isaac_mission_video.py.
Every number on screen comes from telemetry.json (the Isaac run) or from rocket_dt models evaluated at the
logged throttle / ambient pressure (Pe/Pa from nozzle1d, map validity from build_ideal_map), never typed in.

Panels: engine-status ring (main engine throttle + gimbal crosshair, 4 RCS dots), telemetry column,
CFD Mach inset (wsl-v2606 L3 field; bell_C has no CFD case), Pe/Pa gauge with the 0.4 separation line,
thrust-map operating point (validity cells), landing-limit gauges, strip chart, touchdown verdict.

usage (compositing venv with matplotlib + imageio):
  .venv-hud\\Scripts\\python scripts\\render_mission_hud.py --single bellC_ppo_m1 --out bell_C_separation.mp4
  .venv-hud\\Scripts\\python scripts\\render_mission_hud.py --pair pd_A_1004 ppo_A_1004 --out pd_vs_ppo_A.mp4
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from rocket_dt.evaluation.simulate import LANDING_LIMITS  # noqa: E402
from rocket_dt.propulsion import nozzle1d  # noqa: E402
from rocket_dt.propulsion.performance_map import build_ideal_map  # noqa: E402
from rocket_dt.schemas import load_engine, load_vehicle  # noqa: E402

MISSION = REPO / "runs/ppo_win-isaac/mission"
ASSETS = REPO / "runs/ppo_win-isaac/hud_assets"
BG, FG, DIM, ACC, OK, BAD, WARN = "#0b0f14", "#e8eef5", "#7d8a99", "#ffb347", "#3ddc84", "#ff5c5c", "#ffd54a"
SEP_RATIO = 0.4  # Summerfield-type criterion used for map validity (ledger: bell_C separation risk)
GMAX = math.radians(6.0)


class Run:
    def __init__(self, name):
        d = MISSION / name
        self.tel = json.loads((d / "telemetry.json").read_text())
        self.reader = imageio.get_reader(d / "raw.mp4")
        self.reader2 = imageio.get_reader(d / "raw_engine.mp4") if (d / "raw_engine.mp4").exists() else None
        self.plume = self.tel.get("plume", "cone")
        self.failed = [t in self.tel.get("rcs_failed", []) for t in ("rcs_01", "rcs_02", "rcs_03", "rcs_04")]
        self.steps = self.tel["steps"]
        self.engine = self.tel["engine"]
        eng = load_engine(REPO / f"configs/engines/{self.engine}.yaml")
        self.eng = eng
        self.map = build_ideal_map(eng)
        self.td_z = None
        veh = load_vehicle(REPO / "configs/vehicle_v0.yaml")
        self.feet_zmin = float(np.min(np.array(veh.gear.foot_positions_body_m)[:, 2]))
        self.frame_steps = [s for s in self.steps if s["frame"] is not None]
        self.t = np.array([s["t"] for s in self.steps])
        self.alt = np.array([s["p"][2] for s in self.steps])
        self.vz = np.array([s["v"][2] for s in self.steps])
        self.thr = np.array([s["throttle"] if s["engine_on"] else np.nan for s in self.steps])
        mach = ASSETS / f"mach_{self.engine}_L3.npz"
        self.mach = np.load(mach) if mach.exists() else None
        ep = self.tel["episode"] or {}
        self.episode = ep
        self.label = f"{'PPO ' + Path(self.tel['checkpoint']).parent.name if self.tel['controller'] == 'ppo' else 'PD baseline'}"
        self.pe_cache = {}

    def frame(self, i):
        return self.reader.get_data(i)

    def frame_engine(self, s):
        if self.reader2 is None or s.get("frame_engine") is None:
            return None
        return self.reader2.get_data(s["frame_engine"])

    def flame_note(self):
        how = {"flow": "NVIDIA Flow fire", "cone": "emissive cones", "both": "NVIDIA Flow + cones"}[self.plume]
        return (f"flame: {how}, length from thrust, spread from Pe/Pa (1D nozzle). Visual representation of the "
                "1D nozzle output, NOT a combustion simulation")

    def pe_pa(self, s):
        key = (round(s["throttle"], 3), round(s["pa"], -1))
        if key not in self.pe_cache:
            pt = nozzle1d.evaluate(self.eng, max(s["throttle"], self.eng.throttle_min), s["pa"])
            self.pe_cache[key] = pt.exit_pressure_pa / s["pa"]
        return self.pe_cache[key]


def txt(ax, x, y, s, size=11, color=FG, weight="normal", ha="left", va="center", family="monospace"):
    ax.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha, va=va, family=family, transform=ax.transAxes)


def panel(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_facecolor(BG)
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#26313d")
    return ax


def engine_ring(ax, s, title=True, failed=(False, False, False, False)):
    ax.set_xlim(-1.3, 1.3), ax.set_ylim(-1.3, 1.3), ax.set_aspect("equal")
    on = s["engine_on"]
    thr = s["throttle"] if on else 0.0
    glow = (1.0, 0.45 + 0.4 * thr, 0.1, 0.25 + 0.75 * thr) if on else (0.25, 0.25, 0.25, 1)
    ax.add_patch(Circle((0, 0), 0.62, color=glow))
    ax.add_patch(Circle((0, 0), 0.62, fill=False, ec=FG, lw=1.5))
    ax.plot([-0.62, 0.62], [0, 0], color=DIM, lw=0.6), ax.plot([0, 0], [-0.62, 0.62], color=DIM, lw=0.6)
    gx, gy = s["gimbal_rad"]
    ax.add_patch(Circle((0.62 * gy / GMAX, -0.62 * gx / GMAX), 0.08, color=FG))
    for k, ang in enumerate([0, 180, 180, 0]):  # rcs_01/02 at +x (y-dir +/-), rcs_03/04 at -x
        x = 1.0 * math.cos(math.radians(ang))
        y = 0.22 * (1 if k in (0, 3) else -1)
        lit = s["rcs_frac"][k] > 0 and not failed[k]
        ax.add_patch(Circle((x, y), 0.12, color=BAD if failed[k] else (WARN if lit else "#2b3440"), ec=FG, lw=0.8))
        if failed[k]:
            ax.text(x, y - 0.25, "FAIL", color=BAD, ha="center", fontsize=7, family="monospace")
    if title:
        ax.text(0, 1.18, "ENGINE", color=DIM, ha="center", fontsize=9, family="monospace")
    ax.text(0, -0.9, f"{thr * 100:5.1f} %  {s['f_main_n'] / 1000:5.2f} kN" if on else "CUTOFF", color=FG if on else BAD,
            ha="center", fontsize=10, family="monospace")
    ax.text(0, -1.15, f"gimbal x {math.degrees(gx):+4.1f}°  y {math.degrees(gy):+4.1f}°", color=DIM, ha="center",
            fontsize=8, family="monospace")


def telemetry(ax, run, s, compact=False):
    p, v = np.array(s["p"]), np.array(s["v"])
    h = p[2]
    h = p[2] - run.tel.get("td_com_z", 0.0)
    rows = [("T+", f"{s['t']:6.2f} s"), ("ALT (gear)", f"{h:6.2f} m"), ("V-SPEED", f"{v[2]:+6.2f} m/s"),
            ("H-SPEED", f"{np.linalg.norm(v[:2]):6.2f} m/s"), ("PAD ERR", f"{np.linalg.norm(p[:2]):6.2f} m"),
            ("TILT", f"{s['tilt_deg']:6.2f} °"), ("PROPELLANT", f"{s['m_prop']:6.1f} kg"),
            ("WIND", f"{np.linalg.norm(s['wind_w'][:2]):6.0f} N"),
            ("PLUME L/SPREAD", f"{s.get('plume_len_m', 0):4.1f} m ×{s.get('plume_spread', 1):.2f}")]
    n = len(rows)
    for i, (k, val) in enumerate(rows):
        y = 0.92 - i * (0.86 / n)
        txt(ax, 0.05, y, k, 10 if not compact else 9, DIM)
        txt(ax, 0.95, y, val, 12 if not compact else 10, FG, "bold", ha="right")


def gauge_bar(ax, y, label, value, limit, fmt, h=0.12, active=True):
    frac = min(value / (limit * 1.5), 1.0)
    ok = value <= limit
    col = (OK if ok else BAD) if active else "#4a5562"
    ax.add_patch(Rectangle((0.32, y - h / 2), 0.6, h, color="#1a222c", transform=ax.transAxes))
    ax.add_patch(Rectangle((0.32, y - h / 2), 0.6 * frac, h, color=col, transform=ax.transAxes))
    ax.plot([0.32 + 0.6 / 1.5] * 2, [y - h / 2 - 0.02, y + h / 2 + 0.02], color=FG, lw=1.2, transform=ax.transAxes)
    txt(ax, 0.02, y, label, 9, DIM)
    txt(ax, 0.98, y, fmt.format(value), 9, FG, ha="right")


def limits_panel(ax, run, s, td_step):
    """Live values while airborne (grey above 10 m gear height); at first contact the env's recorded
    pre-impact touchdown values (the ones the verdict uses) are frozen."""
    txt(ax, 0.02, 0.93, "LANDING LIMITS (ADR-0005)  bar = value, | = limit", 8, DIM)
    ep = run.episode
    if td_step is not None and ep:
        vals = [abs(ep["td_vz"]), ep["td_vh"], ep["td_tilt_deg"], ep["td_pos_err"]]
        active = True
    else:
        v = np.array(s["v"])
        vals = [abs(v[2]), float(np.linalg.norm(v[:2])), s["tilt_deg"], float(np.linalg.norm(np.array(s["p"][:2])))]
        active = s["p"][2] - run.tel.get("td_com_z", 0.0) < 10.0
    lims = [LANDING_LIMITS["vz_max_m_s"], LANDING_LIMITS["vh_max_m_s"], math.degrees(LANDING_LIMITS["tilt_max_rad"]),
            LANDING_LIMITS["pos_err_max_m"]]
    for i, (lab, val, lim, fmt) in enumerate(zip(("|vz|", "v_h", "tilt", "pad err"), vals, lims,
                                                 ("{:.2f} m/s", "{:.2f} m/s", "{:.1f} °", "{:.2f} m"))):
        gauge_bar(ax, 0.74 - i * 0.2, lab, val, lim, fmt, active=active)
    if td_step is not None:
        txt(ax, 0.02, 0.84, "frozen at first contact (pre-impact)", 8, WARN)
    elif not active:
        txt(ax, 0.02, 0.84, "armed below 10 m gear height", 8, DIM)


def pe_gauge(ax, run, s):
    r = run.pe_pa(s) if s["engine_on"] else float("nan")
    ax.set_xlim(0, 1.5), ax.set_ylim(0, 1)
    ax.add_patch(Rectangle((0, 0.35), SEP_RATIO, 0.3, color="#3a1a1a"))
    ax.add_patch(Rectangle((SEP_RATIO, 0.35), 1.5 - SEP_RATIO, 0.3, color="#16301f"))
    ax.axvline(SEP_RATIO, color=BAD, lw=1.5)
    if r == r:
        ax.plot([min(r, 1.49)] * 2, [0.28, 0.72], color=FG, lw=3)
    ax.text(0.01, 0.88, "Pe/Pa (1D nozzle at logged throttle, ISA Pa)", color=DIM, fontsize=8, family="monospace",
            transform=ax.transAxes)
    ax.text(SEP_RATIO / 1.5, 0.12, "0.4 separation risk", color=BAD, fontsize=8, family="monospace", ha="center",
            transform=ax.transAxes)
    ax.text(0.99, 0.12, f"{r:.3f}" if r == r else "off", color=FG, fontsize=10, family="monospace", ha="right",
            transform=ax.transAxes, weight="bold")


def map_panel(ax, run, s, upto):
    m = run.map
    thr, pa = m.throttle, m.ambient_pressure_pa / 1000
    ax.pcolormesh(thr, pa, np.where(m.valid.T, 1.0, 0.0), cmap=matplotlib.colors.ListedColormap(["#5a1f1f", "#18302a"]),
                  shading="nearest", vmin=0, vmax=1)
    tr = [(x["throttle"], x["pa"] / 1000, x["map_valid"]) for x in run.steps[:upto] if x["engine_on"]]
    if tr:
        a = np.array(tr)
        ax.plot(a[:, 0], a[:, 1], color=ACC, lw=1.0, alpha=0.8)
    if s["engine_on"]:
        ax.plot([s["throttle"]], [s["pa"] / 1000], "o", ms=7, color=FG if s["map_valid"] else BAD, mec="k")
    ax.set_xlim(thr[0] - 0.03, thr[-1] + 0.03), ax.set_ylim(95, 104)
    ax.tick_params(colors=DIM, labelsize=7)
    ax.set_xlabel("throttle", color=DIM, fontsize=8), ax.set_ylabel("ambient p [kPa]", color=DIM, fontsize=8)
    ax.set_title(f"{run.engine} thrust map (1D): red nodes = separation risk; red dot = lookup touches one", color=DIM, fontsize=8,
                 family="monospace")
    for sp in ax.spines.values():
        sp.set_color("#26313d")


def mach_panel(ax, run):
    if run.mach is None:
        txt(ax, 0.5, 0.5, f"{run.engine}: no CFD case\n(1D map only, ledger G1)", 10, DIM, ha="center")
        return
    d = run.mach
    lv = np.linspace(0, 3, 31)
    x, r, m = d["x"], d["r"], d["mach"]
    sc = ax.tricontourf(x, r, m, levels=lv, cmap="inferno")
    ax.tricontourf(x, -r, m, levels=lv, cmap="inferno")
    ax.tricontour(x, r, m, levels=[1.0], colors="w", linewidths=0.6)
    ax.set_aspect("equal")
    ax.set_title(f"{run.engine} nozzle Mach (OpenFOAM v2606 L3, sea level)  max {float(d['mach'].max()):.2f}",
                 color=DIM, fontsize=8, family="monospace")
    cb = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.01)
    cb.ax.tick_params(colors=DIM, labelsize=6)


def strip(ax, run, upto):
    t = run.t[:upto]
    ax.plot(t, run.alt[:upto], color="#6fb3ff", lw=1.2, label="alt [m]")
    ax.plot(t, run.vz[:upto] * 5, color=ACC, lw=1.0, label="vz ×5 [m/s]")
    ax.plot(t, run.thr[:upto] * 50, color=OK, lw=1.0, label="throttle ×50")
    ax.axhline(-LANDING_LIMITS["vz_max_m_s"] * 5, color=BAD, lw=0.6, ls="--")
    ax.set_xlim(0, max(run.t[-1], 1)), ax.set_ylim(-45, 85)
    ax.tick_params(colors=DIM, labelsize=7)
    ax.legend(loc="upper right", fontsize=7, facecolor=BG, labelcolor=FG, edgecolor="#26313d", ncol=3)
    ax.set_xlabel("t [s]", color=DIM, fontsize=8)
    for sp in ax.spines.values():
        sp.set_color("#26313d")


def verdict(ax, run, s):
    if not s["touchdown"]:
        return
    ep = run.episode
    ok = ep.get("success")
    msg = "LANDED: all LANDING_LIMITS met" if ok else f"FAIL: {ep.get('termination_reason')}"
    ax.text(0.5, 0.08, msg, transform=ax.transAxes, ha="center", va="center", fontsize=18, weight="bold",
            color=OK if ok else BAD, family="monospace",
            bbox=dict(facecolor="#000000aa", edgecolor=OK if ok else BAD, boxstyle="round,pad=0.4"))
    pv = ep.get("physics_valid")
    if pv is False:
        ax.text(0.5, 0.18, f"physics_valid = False ({ep.get('map_invalid_steps')} steps on separation-risk cells)",
                transform=ax.transAxes, ha="center", fontsize=11, color=WARN, family="monospace",
                bbox=dict(facecolor="#000000aa", edgecolor=WARN))


def header(ax, run, extra=""):
    ax.text(0.01, 0.5, f"{run.label}  |  engine {run.engine}  |  seed {run.tel['seed']}  {extra}", color=FG, fontsize=12,
            weight="bold", family="monospace", va="center", transform=ax.transAxes)
    ax.text(0.99, 0.5, "win-isaac · Isaac Lab 3.0 / Isaac Sim 6.0.1 PhysX · arrows: thrust 0.5 m/kN, net 2 m/kN, "
            "wind 10 m/kN", color=DIM, fontsize=8, family="monospace", ha="right", va="center", transform=ax.transAxes)


def pip(fig, run, s, rect, label="ENGINE CAM"):
    im = run.frame_engine(s)
    if im is None:
        return
    a = fig.add_axes(rect)
    a.imshow(im), a.set_xticks([]), a.set_yticks([])
    for sp in a.spines.values():
        sp.set_color(FG)
    a.text(0.02, 0.92, label, color=FG, fontsize=8, family="monospace", transform=a.transAxes,
           bbox=dict(facecolor="#000000aa", edgecolor="none"))


def flame_footer(fig, run, y=0.0, h=0.02):
    a = fig.add_axes([0, y, 1, h])
    a.axis("off")
    a.text(0.5, 0.5, run.flame_note(), color=WARN, fontsize=8, ha="center", va="center", family="monospace")


def first_td(run):
    for s in run.steps:
        if s["touchdown"]:
            return s
    return None


def render_single(name, out, fps=25):
    run = Run(name)
    td = first_td(run)
    w = imageio.get_writer(out, fps=fps, codec="libx264", quality=6, macro_block_size=8)
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
    for fs in run.frame_steps:
        fig.clf()
        k = run.steps.index(fs)
        hd = panel(fig, [0, 0.955, 1, 0.045])
        header(hd, run)
        vid = fig.add_axes([0.0, 0.29, 0.6667, 0.665])
        vid.imshow(run.frame(fs["frame"])), vid.axis("off")
        verdict(vid, run, fs)
        pip(fig, run, fs, [0.45, 0.30, 0.21, 0.21])
        flame_footer(fig, run)
        engine_ring(panel(fig, [0.675, 0.66, 0.15, 0.29]), fs, failed=run.failed)
        telemetry(panel(fig, [0.83, 0.66, 0.165, 0.29]), run, fs)
        mach_panel(panel(fig, [0.675, 0.43, 0.32, 0.22]), run)
        pe_gauge(panel(fig, [0.675, 0.30, 0.32, 0.12]), run, fs)
        strip(fig.add_axes([0.03, 0.05, 0.30, 0.2], facecolor=BG), run, k + 1)
        map_panel(fig.add_axes([0.38, 0.05, 0.27, 0.2], facecolor=BG), run, fs, k + 1)
        limits_panel(panel(fig, [0.675, 0.03, 0.32, 0.25]), run, fs, td if fs["touchdown"] else None)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        w.append_data(img)
    for _ in range(fps * 2):  # hold the verdict
        w.append_data(img)
    w.close()
    plt.close(fig)


def render_pair(name_a, name_b, out, fps=25):
    runs = [Run(name_a), Run(name_b)]
    tds = [first_td(r) for r in runs]
    n = max(len(r.frame_steps) for r in runs)
    w = imageio.get_writer(out, fps=fps, codec="libx264", quality=6, macro_block_size=8)
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
    for i in range(n):
        fig.clf()
        for c, (run, td) in enumerate(zip(runs, tds)):
            fs = run.frame_steps[min(i, len(run.frame_steps) - 1)]
            k = run.steps.index(fs)
            x0 = 0.5 * c
            hd = panel(fig, [x0, 0.955, 0.5, 0.045])
            hd.text(0.02, 0.5, f"{run.label}  |  {run.engine}  |  seed {run.tel['seed']}", color=FG, fontsize=12,
                    weight="bold", family="monospace", va="center", transform=hd.transAxes)
            vid = fig.add_axes([x0, 0.43, 0.5, 0.525])
            vid.imshow(run.frame(fs["frame"])), vid.axis("off")
            verdict(vid, run, fs)
            pip(fig, run, fs, [x0 + 0.34, 0.44, 0.155, 0.155])
            engine_ring(panel(fig, [x0 + 0.005, 0.2, 0.12, 0.225]), fs, failed=run.failed)
            telemetry(panel(fig, [x0 + 0.13, 0.2, 0.14, 0.225]), run, fs, compact=True)
            limits_panel(panel(fig, [x0 + 0.275, 0.2, 0.22, 0.225]), run, fs, td if fs["touchdown"] else None)
            strip(fig.add_axes([x0 + 0.04, 0.04, 0.44, 0.13], facecolor=BG), run, k + 1)
        ft = fig.add_axes([0, 0.016, 1, 0.016])
        ft.axis("off")
        ft.text(0.5, 0.5, "win-isaac · same Isaac env, same seed, same actuators/map/PhysX · "
                "arrows: thrust 0.5 m/kN (yellow), net 2 m/kN (magenta), wind 10 m/kN (cyan)", color=DIM, fontsize=8,
                ha="center", va="center", family="monospace")
        flame_footer(fig, runs[0], y=0.0, h=0.016)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        w.append_data(img)
    for _ in range(fps * 2):
        w.append_data(img)
    w.close()
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--single")
    g.add_argument("--pair", nargs=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = REPO / "runs/ppo_win-isaac/videos" / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    render_single(a.single, out) if a.single else render_pair(*a.pair, out)
    print("HUD_DONE", out, round(out.stat().st_size / 2 ** 20, 2), "MB")
