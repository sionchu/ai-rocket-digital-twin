"""Short, footage-first story video (LinkedIn cut), label `win-isaac`.

Chapters: title -> three nozzles -> training stages (16 parallel envs, same seeds) -> engine swap failure vs mixed
training -> one policy on three engines -> PD vs PPO -> three numbers. Text is limited to short labels; every number
is read from committed artefacts (grid.json, telemetry.json, eval JSON, rsl_rl logs, rocket_dt models).
Footage: scripts/isaac_grid_video.py and scripts/isaac_mission_video.py with --look story.
usage: .venv-hud\\Scripts\\python scripts\\render_story.py --out story_vtvl_isaac.mp4
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from rocket_dt.propulsion.geometry import build_contour  # noqa: E402
from rocket_dt.schemas import load_engine  # noqa: E402

RUNS = REPO / "runs/ppo_win-isaac"
W, H, FPS = 1920, 1080, 25
BG, FG, DIM, LINE = "#0d1117", "#f2f4f7", "#9aa4b2", "#3a4350"
OKC, BADC, ACC = "#5fd38d", "#ff6b6b", "#f5b14c"
ENG_COL = {"conical_A": "#8ab4f8", "bell_B": "#f5b14c", "bell_C": "#ff8a80"}
matplotlib.rcParams["font.family"] = ["Segoe UI", "Malgun Gothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def new_fig():
    return plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)


def grab(fig):
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def video(fig, img, rect=(0, 0, 1, 1)):
    ax = fig.add_axes(rect)
    ax.imshow(img, aspect="auto")
    ax.axis("off")
    return ax


def label(fig, text, x=0.035, y=0.93, size=24, color=FG):
    fig.text(x, y, text, color=color, fontsize=size, weight="light",
             bbox=dict(facecolor="#000000", alpha=0.35, edgecolor="none", boxstyle="round,pad=0.35"))


def band(fig, s, tel, x0=0.0, x1=1.0):
    """Minimal telemetry: altitude, speed, throttle bar, engine + RCS glyph."""
    ax = fig.add_axes([x0, 0.0, x1 - x0, 0.07])
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color="#000000", alpha=0.5))
    p, v = np.array(s["p"]), np.array(s["v"])
    alt = max(p[2] - tel.get("td_com_z", 0.0), 0.0)
    big = 18 if (x1 - x0) > 0.4 else 14
    for i, (k, val) in enumerate((("ALT", f"{alt:.1f} m"), ("SPEED", f"{np.linalg.norm(v):.1f} m/s"))):
        x = 0.04 + i * 0.26
        ax.text(x, 0.62, k, color=DIM, fontsize=9, transform=ax.transAxes)
        ax.text(x, 0.18, val, color=FG, fontsize=big, weight="light", transform=ax.transAxes)
    on = s["engine_on"]
    thr = s["throttle"] if on else 0.0
    ax.add_patch(plt.Rectangle((0.58, 0.38), 0.25, 0.18, color=LINE, transform=ax.transAxes))
    ax.add_patch(plt.Rectangle((0.58, 0.38), 0.25 * thr, 0.18, color=ACC if on else LINE, transform=ax.transAxes))
    ax.text(0.58, 0.68, "THROTTLE", color=DIM, fontsize=9, transform=ax.transAxes)
    asp = (W * (x1 - x0)) / (H * 0.07)
    rx = 0.25 / asp * 2.0
    gx = 0.93
    ax.add_patch(matplotlib.patches.Ellipse((gx, 0.5), rx, 0.6, color=ACC if on else LINE,
                                            alpha=0.3 + 0.7 * thr if on else 1.0, transform=ax.transAxes))
    failed = tel.get("rcs_failed", [])
    for k, (dx, dy) in enumerate([(1, 0.3), (1, -0.3), (-1, -0.3), (-1, 0.3)]):
        rid = f"rcs_{k + 1:02d}"
        col = BADC if rid in failed else (FG if s["rcs_frac"][k] > 0 else LINE)
        ax.add_patch(matplotlib.patches.Ellipse((gx + dx * rx * 0.85, 0.5 + dy), rx * 0.25, 0.14, color=col,
                                                transform=ax.transAxes))


def verdict(fig, ep, x, y=0.55, size=28):
    ok = ep.get("success")
    # tipped = passed the env's 60 deg tip-over limit (a video render may let the fall continue past it)
    tip = "tip" in ep.get("termination_reason", "") or ep.get("max_tilt_deg", 0.0) > 60.0
    txt = "LANDED" if ok else ("TIPPED OVER" if tip else "MISSED")
    fig.text(x, y, txt, color=OKC if ok else BADC, fontsize=size, ha="center", weight="bold")
    if ep.get("touchdown"):
        fig.text(x, y - 0.045, f"{abs(ep['td_vz']):.1f} m/s  ·  {ep['td_pos_err']:.1f} m from pad", color=FG,
                 fontsize=size * 0.5, ha="center")


class Mission:
    def __init__(self, name):
        d = RUNS / "mission" / name
        self.tel = json.loads((d / "telemetry.json").read_text())
        self.reader = imageio.get_reader(d / "raw.mp4")
        self.reader2 = imageio.get_reader(d / "raw_engine.mp4") if (d / "raw_engine.mp4").exists() else None
        self.fsteps = [s for s in self.tel["steps"] if s["frame"] is not None]
        self.ep = self.tel["episode"]
        # skip RTX warm-up frames (black) at the start of a capture
        k = 0
        while k < min(40, len(self.fsteps) - 1) and self.reader.get_data(self.fsteps[k]["frame"]).mean() < 8:
            k += 1
        self.fsteps = self.fsteps[k:]

    def at(self, i):
        s = self.fsteps[min(i, len(self.fsteps) - 1)]
        return self.reader.get_data(s["frame"]), s

    def engine(self, s):
        if self.reader2 is None or s.get("frame_engine") is None:
            return None
        return self.reader2.get_data(s["frame_engine"])

    def __len__(self):
        return len(self.fsteps)


class Grid:
    def __init__(self, name):
        d = RUNS / "grid" / name
        self.meta = json.loads((d / "grid.json").read_text())
        self.reader = imageio.get_reader(d / "raw.mp4")
        self.n = self.reader.count_frames()

    def at(self, i):
        i = min(i, self.n - 1)
        return self.reader.get_data(i), self.meta["frames"][min(i, len(self.meta["frames"]) - 1)]


def parse_log(name):
    rows, cur = [], None
    for line in (RUNS / f"{name}.log").read_text(encoding="utf-8", errors="ignore").splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
        m = re.match(r"Learning iteration (\d+)/", line)
        if m:
            cur = {"it": int(m.group(1))}
            rows.append(cur)
        elif cur is not None and line.startswith("Metrics/success_rate:"):
            cur["success"] = float(line.split(":")[1])
    return rows


def lineage():
    out = []
    for name, lo, hi in (("r0_base", 0, 149), ("r3_guided", 150, 399), ("m1_mix_ft", 400, 697)):
        out += [r for r in parse_log(name) if lo <= r["it"] <= hi and "success" in r]
    return out


def ev(path):
    return json.loads((RUNS / path).read_text())["summary"]


def counters(fig, st, n, x=0.965):
    fails = st["done"] - st["success"]
    for i, (val, lab, col) in enumerate(((st["success"], "landed", OKC), (fails, "failed", BADC))):
        xx = x - i * 0.08
        fig.text(xx, 0.905, f"{val}", color=col, fontsize=36, ha="right", weight="light")
        fig.text(xx, 0.875, lab, color=DIM, fontsize=11, ha="right")


# ----------------------------------------------------------------------------- chapters
def title(n, grid="g_m1_A", speed=4, head="Thrust-only rocket landing in Isaac Sim",
          sub="inspired by the cold-gas thruster package announced for the Tesla Roadster"):
    """Title over Isaac footage (16 envs landing), no black card."""
    g = Grid(grid)
    out = []
    for i in range(n):
        fig = new_fig()
        img, _ = g.at(g.n // 3 + i * speed)
        video(fig, (img * 0.55).astype(np.uint8))
        a = min(1.0, i / 10)
        fig.text(0.5, 0.54, head, color=FG, fontsize=40, ha="center", weight="light", alpha=a)
        fig.text(0.5, 0.45, sub, color=DIM, fontsize=17, ha="center", alpha=a)
        out.append(grab(fig))
    return out


def nozzles(n):
    cs = {k: build_contour(load_engine(REPO / f"configs/engines/{k}.yaml")) for k in ENG_COL}
    out = []
    for i in range(n):
        fig = new_fig()
        ax = fig.add_axes([0.12, 0.2, 0.76, 0.6], facecolor=BG)
        frac = min(1.0, (i + 1) / (n * 0.6))
        for k, c in cs.items():
            x = (c.x_m - c.x_m[c.throat_index]) * 1000
            m = max(2, int(len(x) * frac))
            ax.plot(x[:m], c.r_m[:m] * 1000, color=ENG_COL[k], lw=3)
            ax.plot(x[:m], -c.r_m[:m] * 1000, color=ENG_COL[k], lw=3)
            dy = {"conical_A": -5.0, "bell_B": 5.0, "bell_C": 0.0}[k]
            ax.text(x[-1] + 4, c.r_m[-1] * 1000 + dy, k, color=ENG_COL[k], fontsize=18, va="center",
                    alpha=1.0 if frac >= 1 else 0.0)
        ax.set_aspect("equal"), ax.axis("off")
        label(fig, "Three nozzles, one vehicle", size=26)
        out.append(grab(fig))
    return out


def training(stages, curve, seconds_each=5.0, speed=4):
    its = np.array([r["it"] for r in curve])
    succ = np.array([r["success"] for r in curve])
    out = []
    for gname, it, text in stages:
        g = Grid(gname)
        for i in range(int(seconds_each * FPS)):
            img, st = g.at(i * speed)
            fig = new_fig()
            video(fig, img)
            label(fig, text)
            counters(fig, st, g.meta["summary"]["n"])
            ax = fig.add_axes([0.035, 0.05, 0.26, 0.11])
            ax.patch.set_alpha(0)
            ax.plot(its, succ, color=OKC, lw=1.4)
            ax.axvline(it, color=FG, lw=1.2)
            ax.set_xlim(0, its.max()), ax.set_ylim(0, 1.05), ax.axis("off")
            fig.text(0.035, 0.165, "training success", color=DIM, fontsize=11)
            out.append(grab(fig))
    return out


def grid_pair(a, b, labs, text, seconds=6.5, speed=4):
    ga, gb = Grid(a), Grid(b)
    out = []
    for i in range(int(seconds * FPS)):
        fig = new_fig()
        for c, (g, lab) in enumerate(zip((ga, gb), labs)):
            img, st = g.at(i * speed)
            x0 = 0.5 * c
            video(fig, img, rect=(x0, 0.0, 0.5, 1.0))
            fig.text(x0 + 0.02, 0.06, lab, color=FG, fontsize=18, weight="light")
            counters(fig, st, g.meta["summary"]["n"], x=x0 + 0.48)
        fig.add_artist(plt.Line2D([0.5, 0.5], [0, 1], color=BG, lw=6))
        label(fig, text, y=0.95, size=20)
        out.append(grab(fig))
    return out


def missions(names, labs, text, speed=2, hold=2.0):
    ms = [Mission(n) for n in names]
    k = len(ms)
    n = max(len(m) for m in ms) // speed + int(FPS * hold)
    out = []
    for i in range(n):
        fig = new_fig()
        for c, (m, lab) in enumerate(zip(ms, labs)):
            img, s = m.at(i * speed)
            x0 = c / k
            crop = img if k < 3 else img[:, 200:1080]
            video(fig, crop, rect=(x0, 0.0, 1 / k, 1.0))
            band(fig, s, m.tel, x0=x0 + 0.003, x1=x0 + 1 / k - 0.003)
            im2 = m.engine(s)
            if im2 is not None and k >= 3:  # nozzle close-up: makes the three engines visibly different
                wpx = 0.55 / k
                a2 = fig.add_axes([x0 + 0.01, 0.13, wpx, wpx * W / H * 360 / 480])
                a2.imshow(im2[:, 80:560]), a2.axis("off")
            fig.text(x0 + 0.015, 0.09, lab, color=ENG_COL.get(lab.rstrip("*"), FG), fontsize=18, weight="light")
            if i * speed >= len(m) - 1:
                verdict(fig, m.ep, x=x0 + 0.5 / k, size=26 if k < 3 else 22)
        for c in range(1, k):
            fig.add_artist(plt.Line2D([c / k, c / k], [0, 1], color=BG, lw=6))
        label(fig, text, y=0.95, size=20)
        out.append(grab(fig))
    return out


def numbers(n, grid="g_m1_C", speed=2):
    def total(tpl, e):
        return sum(ev(tpl.format(s=s, e=e))["n_success"] for s in (1000, 2000))
    pd = [total("pd_isaac/seeds{s}/eval_{e}_ideal1d.json", e) for e in ENG_COL]
    m1 = [total("train/m1_mix_ft/model_697_eval_seeds{s}/eval_{e}_ideal1d.json", e) for e in ENG_COL]
    r3 = [ev(f"train/r3_guided/model_398_eval/eval_{e}_ideal1d.json")["n_success"]
          + ev(f"train/r3_guided/model_398_eval_seeds2000/eval_{e}_ideal1d.json")["n_success"] for e in ENG_COL]
    g = Grid(grid)
    out = []
    for i in range(n):
        fig = new_fig()
        img, _ = g.at(g.n - n * speed + i * speed)
        video(fig, (img * 0.35).astype(np.uint8))
        fig.text(0.5, 0.85, "landings out of 60, per engine", color=DIM, fontsize=18, ha="center")
        xs = [0.42, 0.6, 0.78]
        for x, e in zip(xs, ENG_COL):
            fig.text(x, 0.74, e + ("*" if e == "bell_C" else ""), color=ENG_COL[e], fontsize=18, ha="center")
        for j, (lab, vals) in enumerate((("PD controller", pd), ("PPO, one engine", r3), ("PPO, mixed engines", m1))):
            y = 0.6 - j * 0.14
            fig.text(0.1, y, lab, color=FG, fontsize=24, weight="light", va="center")
            for x, v in zip(xs, vals):
                fig.text(x, y, f"{v}", color=OKC if v >= 54 else (ACC if v >= 40 else BADC), fontsize=46,
                         ha="center", va="center", weight="light")
        fig.text(0.5, 0.1, "* bell_C uses separation-risk map cells  ·  flame = visual of 1D nozzle output  ·  Isaac Lab 3.0 / Isaac Sim 6.0.1",
                 color=DIM, fontsize=12, ha="center")
        out.append(grab(fig))
    return out


def quad(names, labs, text, speed=2, hold=2.0, notes=None):
    """2x2 split screen of single-vehicle Isaac renders (one trained policy or controller per pane)."""
    ms = [Mission(n) for n in names]
    n = max(len(m) for m in ms) // speed + int(FPS * hold)
    out = []
    rects = [(0, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5), (0, 0, 0.5, 0.5), (0.5, 0, 0.5, 0.5)]
    for i in range(n):
        fig = new_fig()
        for m, lab, (x0, y0, w, h), note in zip(ms, labs, rects, notes or [None] * 4):
            img, s_ = m.at(i * speed)
            video(fig, img, rect=(x0, y0, w, h))
            fig.text(x0 + 0.012, y0 + h - 0.05, lab, color=FG, fontsize=17, weight="light",
                     bbox=dict(facecolor="#000000", alpha=0.4, edgecolor="none", boxstyle="round,pad=0.3"))
            if note:
                fig.text(x0 + w - 0.012, y0 + h - 0.05, note, color=DIM, fontsize=13, ha="right",
                         bbox=dict(facecolor="#000000", alpha=0.4, edgecolor="none", boxstyle="round,pad=0.3"))
            if i * speed >= len(m) - 1:
                ok = m.ep.get("success")
                fig.text(x0 + w / 2, y0 + h * 0.5, "LANDED" if ok else ("TIPPED OVER" if "tip" in m.ep.get(
                    "termination_reason", "") else "MISSED"), color=OKC if ok else BADC, fontsize=26, ha="center",
                         weight="bold")
        fig.add_artist(plt.Line2D([0.5, 0.5], [0, 1], color=BG, lw=6))
        fig.add_artist(plt.Line2D([0, 1], [0.5, 0.5], color=BG, lw=6))
        fig.text(0.5, 0.5, text, color=FG, fontsize=18, ha="center", va="center", weight="light",
                 bbox=dict(facecolor=BG, alpha=0.9, edgecolor="none", boxstyle="round,pad=0.5"))
        out.append(grab(fig))
    return out


def layout_diagrams(n):
    """Closing card: top-down engine placement of the four layouts (from layouts.py), no footage."""
    from rocket_dt.isaac_env.layouts import layout_spec
    from rocket_dt.schemas import load_vehicle
    veh = load_vehicle(REPO / "configs/vehicle_v0.yaml")
    names = [("gimbal_rcs", "gimbaled engine + RCS"), ("tri_diff", "3 engines, differential throttle"),
             ("quad_diff", "4 engines, differential throttle"), ("coldgas", "fixed engine + cold-gas thrusters")]
    specs = [layout_spec(k, veh) for k, _ in names]
    out = []
    for i in range(n):
        fig = new_fig()
        a = min(1.0, (i + 1) / 12)
        for j, ((k, lab), sp) in enumerate(zip(names, specs)):
            ax = fig.add_axes([0.04 + j * 0.24, 0.25, 0.2, 0.45], facecolor=BG)
            ax.add_patch(plt.Circle((0, 0), 0.6, fill=False, ec=FG, lw=1.5, alpha=a))
            for e, sh in zip(sp.engine_pos_b, sp.engine_share):
                ax.add_patch(plt.Circle((e[0], e[1]), 0.1 + 0.25 * np.sqrt(sh) * 0.6, color=ACC, alpha=a))
                if sp.gimbal:
                    ax.add_patch(plt.Circle((e[0], e[1]), 0.36, fill=False, ec=ACC, ls="--", lw=1, alpha=a))
            for t in sp.rcs:
                x, y = t.position_body_m[0], t.position_body_m[1]
                dx, dy = t.force_direction_body[0], t.force_direction_body[1]
                col = BADC if t.id.startswith("cg_") else FG
                ax.arrow(x, y, -0.18 * dx, -0.18 * dy, width=0.015, color=col, alpha=a, length_includes_head=True)
            ax.set_xlim(-0.9, 0.9), ax.set_ylim(-0.9, 0.9), ax.set_aspect("equal"), ax.axis("off")
            fig.text(0.14 + j * 0.24, 0.2, lab, color=FG, fontsize=15, ha="center", weight="light", alpha=a)
        label(fig, "Four engine layouts, one airframe", size=26)
        fig.text(0.5, 0.08, "orange = main engines (dashed = gimbal)  ·  white = roll RCS  ·  red = cold-gas pitch/yaw",
                 color=DIM, fontsize=12, ha="center")
        out.append(grab(fig))
    return out


def results2(n, grid="g_m1_A", speed=2):
    def k(path):
        return sum(json.loads(Path(f).read_text())["summary"]["n_success"]
                   for f in sorted((RUNS / "eval2" / path).glob("*_*/eval_*.json")))
    def k2(path, ctl):
        return sum(json.loads(f.read_text())["summary"]["n_success"]
                   for f in sorted((RUNS / "eval2" / path).glob(f"{ctl}_*/eval_*.json")))
    rows = [("strong wind", "wind"), ("RCS failure", "rcs_fail"), ("sensor noise", "noise"), ("lateral approach", "lateral")]
    data = [(lab, k2(d, "pd"), k2(d, "r3"), k2(d, "own")) for lab, d in rows]
    lay = [(lab, k2(f"layout_{d}", "own")) for lab, d in (("3 engines", "tri_diff"), ("4 engines", "quad_diff"),
                                                          ("cold-gas", "coldgas"))]
    g = Grid(grid)
    out = []
    for i in range(n):
        fig = new_fig()
        img, _ = g.at(g.n - n * speed + i * speed)
        video(fig, (img * 0.3).astype(np.uint8))
        fig.text(0.5, 0.88, "landings out of 60", color=DIM, fontsize=18, ha="center")
        xs = [0.42, 0.57, 0.72]
        for x, h in zip(xs, ("PD", "RL nominal", "RL trained here")):
            fig.text(x, 0.8, h, color=DIM, fontsize=16, ha="center")
        for j, (lab, pd, r3, own) in enumerate(data):
            y = 0.7 - j * 0.09
            fig.text(0.12, y, lab, color=FG, fontsize=22, weight="light", va="center")
            best = max(pd, r3, own)
            for x, v in zip(xs, (pd, r3, own)):
                fig.text(x, y, f"{v}", color=OKC if v == best else (ACC if v >= 40 else BADC), fontsize=36,
                         ha="center", va="center", weight="light")
        fig.text(0.12, 0.27, "layouts (RL, nominal)", color=DIM, fontsize=16)
        for j, (lab, v) in enumerate(lay):
            fig.text(0.42 + j * 0.15, 0.27, f"{lab}  {v}", color=OKC if v >= 54 else ACC, fontsize=20, ha="center",
                     weight="light")
        fig.text(0.5, 0.12, "RL beats fixed-gain PD outside PD's tuning (strong wind, far offsets) when trained for it;"
                            "  noise-trained policy failed (17)", color=FG, fontsize=15, ha="center")
        out.append(grab(fig))
    return out


def results_v1(n, grid="g_v1_nom", speed=2):
    """Physics v1 landings out of 60 (runs/ppo_win-isaac/eval_v1/summary_v1.json) with the v0 number beside it."""
    s = json.loads((RUNS / "eval_v1/summary_v1.json").read_text())
    v0 = {"wind": (43, 20, 54), "rcs_fail": (60, 27, 57), "noise": (60, 60, 17), "lateral": (5, 28, 52)}
    rows = [("strong wind", "wind"), ("RCS failure", "rcs_fail"), ("sensor noise", "noise"),
            ("lateral approach", "lateral")]

    def k(sc, c):
        r = s[sc].get(c)
        return None if r is None else r["success"]
    if s.get("hop"):  # ground-start hop exists only in v1 (no v0 number, no re-tuned PD, no nominal-RL run)
        rows.append(("ground-start hop", "hop"))
        v0["hop"] = (None, None, None)
    data = [(lab, [k(sc, "pd"), k(sc, "pd_tuned"), k(sc, "rl_nominal"), k(sc, "rl_scenario")], v0[sc])
            for lab, sc in rows]
    ly = 0.28 if len(rows) == 4 else 0.235
    lay = [(lab, s["layouts"][d]["success"], v) for lab, d, v in (("3 engines", "tri_diff", 60),
                                                                  ("4 engines", "quad_diff", 60),
                                                                  ("cold-gas", "coldgas", 59))]
    g = Grid(grid)
    out = []
    for i in range(n):
        fig = new_fig()
        img, _ = g.at(g.n - n * speed + i * speed)
        video(fig, (img * 0.3).astype(np.uint8))
        fig.text(0.5, 0.9, "physics v1: landings out of 60  (v0 in grey)", color=DIM, fontsize=18, ha="center")
        xs = [0.38, 0.5, 0.62, 0.76]
        for x, h in zip(xs, ("PD", "PD re-tuned", "RL nominal", "RL trained here")):
            fig.text(x, 0.81, h, color=DIM, fontsize=15, ha="center")
        for j, (lab, vals, old) in enumerate(data):
            y = 0.71 - j * 0.09
            fig.text(0.08, y, lab, color=FG, fontsize=22, weight="light", va="center")
            best = max(v for v in vals if v is not None)
            olds = (old[0], None, old[1], old[2])
            for x, v, o in zip(xs, vals, olds):
                fig.text(x, y, "-" if v is None else f"{v}", color=OKC if v == best else (ACC if (v or 0) >= 40 else BADC),
                         fontsize=34, ha="center", va="center", weight="light")
                if o is not None:
                    fig.text(x + 0.033, y - 0.012, f"{o}", color=DIM, fontsize=13, ha="left", va="center")
        fig.text(0.08, ly, "layouts (RL, nominal)", color=DIM, fontsize=16)
        for j, (lab, v, o) in enumerate(lay):
            fig.text(0.42 + j * 0.17, ly, f"{lab}  {v}", color=OKC if v >= 54 else (ACC if v >= 40 else BADC),
                     fontsize=20, ha="center", weight="light")
            fig.text(0.42 + j * 0.17, ly - 0.035, f"v0 {o}", color=DIM, fontsize=12, ha="center")
        if s.get("hop"):
            fig.text(0.5, 0.165, "Ground-start hop (new): ignition, climb, 1 s hover hold, landing on a second pad "
                                 f"- PD {k('hop', 'pd')}/60, RL {k('hop', 'rl_scenario')}/60.", color=ACC,
                     fontsize=15, ha="center")
        fig.text(0.5, 0.13, "With aerodynamics, burning propellant and a CFD thrust map, the scenario-trained RL policy "
                            "still wins only in strong wind;", color=FG, fontsize=15, ha="center")
        fig.text(0.5, 0.095, "a re-tuned PD lands every far-lateral start. Fine-tunes for RCS failure, noise and the "
                             "three layouts fell short at the v0 iteration budget.", color=FG, fontsize=15, ha="center")
        fig.text(0.5, 0.04, "seeds 1000-1029 + 2000-2029  ·  Isaac Lab 3.0 / Isaac Sim 6.0.1  ·  win-isaac",
                 color=DIM, fontsize=11, ha="center")
        out.append(grab(fig))
    return out


def ecu_hero(name, head, sub, hold=2.0, title_s=3.5, caption="strong wind · RL policy trained for it · seed 1001",
             speed=1.0, slow_s=0.0):
    """Cold open: one landing at real time with the flight computer's outputs beside it, every value from the
    Isaac telemetry of that run (50 Hz control steps): engine state, throttle command, gimbal, 4 RCS, thrust, mass."""
    m = Mission(name)
    tel, steps = m.tel, m.tel["steps"]
    t_all = np.array([s["t"] for s in steps])
    thr = np.array([s["throttle"] if s["engine_on"] else 0.0 for s in steps])
    gim = np.degrees(np.array([s["gimbal_rad"] for s in steps]))
    rcs = np.array([s["rcs_frac"] for s in steps])
    fmain = np.array([s["f_main_n"] for s in steps]) / 1000
    mass = np.array([s.get("mass_kg", tel.get("mass_kg_fixed", 0.0)) for s in steps])
    aero = np.array([np.linalg.norm(s["wind_w"]) for s in steps])
    cut = next((s["t"] for s in steps if not s["engine_on"]), None)
    failed = tel.get("rcs_failed", [])
    win = 6.0
    slow = int(FPS * slow_s)  # real time for the first slow_s seconds (ignition), then `speed`
    idx = lambda i: i if i < slow else int(slow + (i - slow) * speed)  # noqa: E731
    n = slow + int(max(len(m) - slow, 0) / speed) + int(FPS * hold)
    out = []
    px = 0.685  # ECU panel left edge
    for i in range(n):
        img, s = m.at(idx(i))
        k = next(j for j, st in enumerate(steps) if st["k"] == s["k"])
        t = s["t"]
        fig = new_fig()
        video(fig, img, rect=(0, 0, px, 1))
        im2 = m.engine(s)
        if im2 is not None:
            ax2 = fig.add_axes([0.012, 0.03, 0.2, 0.2 * W / H * 360 / 640])
            ax2.imshow(im2), ax2.axis("off")
            fig.text(0.016, 0.03 + 0.2 * W / H * 360 / 640 - 0.03, "engine camera", color=FG, fontsize=11,
                     bbox=dict(facecolor="#000000", alpha=0.45, edgecolor="none"))
        if i < FPS * title_s:  # title over the opening seconds of footage
            a = min(1.0, i / 8, (FPS * title_s - i) / 8)
            fig.text(px / 2, 0.86, head, color=FG, fontsize=34, ha="center", weight="light", alpha=a,
                     bbox=dict(facecolor="#000000", alpha=0.35 * a, edgecolor="none", boxstyle="round,pad=0.4"))
            fig.text(px / 2, 0.8, sub, color=FG, fontsize=16, ha="center", alpha=a)
        else:
            label(fig, caption, size=16, y=0.95)
        if idx(i) >= len(m) - 1:
            verdict(fig, m.ep, x=px / 2, y=0.5)
        if speed != 1.0 and i >= slow:
            fig.text(px - 0.01, 0.95, f"{speed:g}x", color=DIM, fontsize=12, ha="right")
        # ---- ECU panel
        fig.add_artist(plt.Rectangle((px, 0), 1 - px, 1, color=BG, transform=fig.transFigure, zorder=-1))
        fig.text(px + 0.015, 0.955, "FLIGHT COMPUTER OUTPUTS", color=FG, fontsize=15, weight="bold")
        fig.text(px + 0.015, 0.93, f"50 Hz control step   t = {t:5.2f} s", color=DIM, fontsize=11,
                 family="monospace")
        on, kig = s["engine_on"], s.get("k_ign", 1.0)
        ign = tel.get("ignition")
        if not on:
            state, col = "MAIN ENGINE  CUTOFF", BADC
        elif ign and t < ign["cmd_s"]:
            state, col = f"ARMED   T-{ign['cmd_s'] - t:.1f} s", ACC
        elif kig <= 0.0:
            state, col = "IGNITION CMD", ACC
        elif kig < 0.95:
            state, col = f"THRUST RISE  {100 * kig:3.0f} %", ACC
        else:
            state, col = "MAIN ENGINE  ON", OKC
        fig.text(px + 0.015, 0.885, "●", color=col, fontsize=18)
        fig.text(px + 0.04, 0.888, state, color=FG, fontsize=15, family="monospace")
        if cut is not None and not on:
            fig.text(px + 0.04, 0.865, f"cutoff at touchdown, t = {cut:.2f} s", color=DIM, fontsize=10)
        elif ign:
            ph = ("ON PAD" if not s.get("airborne", True) else
                  ("CLIMB / HOVER" if s.get("hop_phase", 1) == 0 else "DESCENT TO LANDING PAD"))
            fig.text(px + 0.04, 0.865, f"{ph}   ·   ignition delay {ign['delay_s']} s, rise {ign['tau_s']} s (assumed)",
                     color=DIM, fontsize=10)
        y0 = 0.825
        rows = (("THROTTLE CMD", f"{100 * thr[k]:5.1f} %"), ("MAIN THRUST", f"{fmain[k]:5.1f} kN"),
                ("VEHICLE MASS", f"{mass[k]:7.1f} kg"), ("AERO FORCE", f"{aero[k]:6.0f} N"),
                ("TILT", f"{s['tilt_deg']:5.1f} °"))
        for j, (lab, val) in enumerate(rows):
            y = y0 - j * 0.035
            fig.text(px + 0.015, y, lab, color=DIM, fontsize=11)
            fig.text(0.985, y, val, color=FG, fontsize=13, ha="right", family="monospace")
        # gimbal crosshair (±6 deg) and RCS lamps
        ax = fig.add_axes([px + 0.02, 0.43, 0.12, 0.12 * W / H])
        ax.set_xlim(-7, 7), ax.set_ylim(-7, 7), ax.set_aspect("equal"), ax.axis("off")
        ax.add_patch(plt.Circle((0, 0), 6, fill=False, ec=LINE, lw=1.2))
        ax.plot([-6, 6], [0, 0], color=LINE, lw=0.8), ax.plot([0, 0], [-6, 6], color=LINE, lw=0.8)
        ax.plot(gim[max(0, k - 50):k + 1, 0], gim[max(0, k - 50):k + 1, 1], color=ACC, lw=1, alpha=0.5)
        ax.add_patch(plt.Circle((gim[k, 0], gim[k, 1]), 0.7, color=ACC))
        fig.text(px + 0.08, 0.415, f"GIMBAL  {gim[k, 0]:+.1f}° / {gim[k, 1]:+.1f}°", color=DIM, fontsize=10,
                 ha="center", family="monospace")
        for j in range(4):
            rid = f"rcs_{j + 1:02d}"
            f = rcs[k, j]
            col = BADC if rid in failed else (OKC if f > 0.05 else LINE)
            yy = 0.54 - j * 0.03
            fig.text(px + 0.17, yy, "●", color=col, fontsize=14)
            fig.text(px + 0.19, yy + 0.003, f"RCS {j + 1}  {100 * f:4.0f} %", color=FG if f > 0.05 else DIM,
                     fontsize=11, family="monospace")
        # strip charts, last `win` seconds
        sel = (t_all <= t) & (t_all >= t - win)
        charts = (("throttle cmd", thr, (0, 1), ACC), ("gimbal pitch / yaw [°]", gim, (-6.5, 6.5), None),
                  ("RCS duty", rcs, (0, 1), None))
        for j, (lab, arr, lim, col) in enumerate(charts):
            a3 = fig.add_axes([px + 0.02, 0.27 - j * 0.11, 0.29, 0.075], facecolor="#151b23")
            if arr.ndim == 1:
                a3.plot(t_all[sel], arr[sel], color=col, lw=1.4)
            else:
                cols = [ACC, "#8ab4f8"] if arr.shape[1] == 2 else [OKC, "#8ab4f8", "#f5b14c", "#ff8a80"]
                for c in range(arr.shape[1]):
                    a3.plot(t_all[sel], arr[sel, c], color=cols[c], lw=1.1)
            a3.set_xlim(t - win, t), a3.set_ylim(*lim)
            a3.set_xticks([]), a3.set_yticks([])
            for sp in a3.spines.values():
                sp.set_color(LINE)
            fig.text(px + 0.022, 0.27 - j * 0.11 + 0.08, lab, color=DIM, fontsize=10)
        fig.text(px + 0.015, 0.012, "Isaac Lab 3.0 · physics v1 · values from the run's telemetry",
                 color=DIM, fontsize=9)
        out.append(grab(fig))
    return out


def physics_card(n, grid="g_v1_nom", speed=2):
    """What changed between v0 and v1 (ADR-0011), over dimmed v1 footage."""
    g = Grid(grid)
    rows = [("aerodynamics", "crossflow strip theory, wind profile, rate damping"),
            ("variable mass", "propellant burn moves mass, CoM and inertia (compensating wrench)"),
            ("thrust map", "OpenFOAM CFD map for the conical engine instead of 1D nozzle theory"),
            ("PD baseline", "gains re-tuned per scenario on the reference integrator")]
    out = []
    for i in range(n):
        fig = new_fig()
        img, _ = g.at(g.n // 2 + i * speed)
        video(fig, (img * 0.3).astype(np.uint8))
        a = min(1.0, i / 10)
        fig.text(0.5, 0.78, "Physics v1", color=FG, fontsize=36, ha="center", weight="light", alpha=a)
        for j, (k, v) in enumerate(rows):
            y = 0.62 - j * 0.1
            fig.text(0.24, y, k, color=ACC, fontsize=22, weight="light", alpha=a)
            fig.text(0.42, y, v, color=FG, fontsize=18, weight="light", alpha=a)
        out.append(grab(fig))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="story_vtvl_isaac.mp4")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--cut", default="v1", choices=["v1", "v2", "v3", "v4", "v5"])
    a = ap.parse_args()
    out = RUNS / "videos" / a.out
    chapters = {
        "title": lambda: title(FPS * 4),
        "training": lambda: training([("g_r0_it0", 0, "Iteration 0"), ("g_r0_it50", 50, "Iteration 50"),
                                      ("g_r0_it149", 149, "Iteration 149"), ("g_r3_A", 398, "Iteration 398"),
                                      ("g_m1_A", 697, "Iteration 697")], lineage()),
        "swap": lambda: grid_pair("g_r3_C", "g_m1_C", ["trained on one engine", "trained on mixed engines"],
                                  "Engine swapped to bell_C"),
        "engines": lambda: missions(["s_m1_A_1004", "s_m1_B_1004", "s_m1_C_1004"], ["conical_A", "bell_B", "bell_C*"],
                                    "One policy, three engines", speed=3),
        "pd_vs_ppo": lambda: missions(["s_pd_A_1004", "s_m1_A_1004"], ["PD controller", "PPO policy"],
                                      "Classical control vs learned policy", speed=2),
        "numbers": lambda: numbers(FPS * 5),
        "nozzles": lambda: nozzles(FPS * 5),  # closing shot after the Isaac footage
    }
    if a.cut == "v2":  # split-screen cut: scenario-specific training and engine layouts
        chapters = {
            "title": lambda: title(FPS * 4),
            "training": lambda: training([("g_r0_it0", 0, "Iteration 0"), ("g_r0_it149", 149, "Iteration 149"),
                                          ("g_m1_A", 697, "Iteration 697")], lineage(), seconds_each=4.0),
            "scenarios": lambda: quad(["sc_wind_rl", "sc_rcs_rl", "sc_noise_nom", "sc_lat_rl"],
                                      ["strong wind", "RCS thruster failed", "sensor noise + latency", "far lateral start"],
                                      "one policy trained per scenario", notes=["RL trained here", "RL trained here",
                                                                                "nominal RL", "RL trained here"]),
            "rl_vs_pd": lambda: quad(["sc_wind_pd", "sc_wind_rl", "sc_lat_pd", "sc_lat_rl"],
                                     ["strong wind · PD", "strong wind · RL", "far lateral start · PD",
                                      "far lateral start · RL"], "where learning beats fixed-gain control"),
            "layouts": lambda: quad(["ly_a", "ly_b", "ly_c", "ly_d"],
                                    ["gimbaled engine", "3 engines", "4 engines", "cold-gas thrusters"],
                                    "four engine layouts, each trained in Isaac", speed=3),
            "results": lambda: results2(FPS * 6),
            "diagrams": lambda: layout_diagrams(FPS * 5),
        }
    if a.cut in ("v3", "v4", "v5"):  # physics v1 (ADR-0011) re-run, visual v1 assets; footage from runs/ppo_win-isaac/videos/render_v1.ps1
        chapters = {
            "title": lambda: title(FPS * 4, grid="g_v1_nom", head="Rocket landing in Isaac Sim, physics v1",
                                   sub="aerodynamics, burning propellant and a CFD thrust map"),
            "physics": lambda: physics_card(FPS * 5),
            "retrain": lambda: grid_pair("g_v1_r3", "g_v1_nom", ["v0 policy in v1 physics",
                                                                 "fine-tuned in v1 (300 iterations)"],
                                         "Same 16 starts, new physics"),
            "scenarios": lambda: quad(["v1_wind_rl", "v1_rcs_rl", "v1_noise_rl", "v1_lat_rl"],
                                      ["strong wind", "RCS thruster failed", "sensor noise + latency", "far lateral start"],
                                      "one policy fine-tuned per scenario",
                                      notes=["seed 1001", "seed 1000", "seed 1000", "seed 1002"]),
            "rl_vs_pd": lambda: quad(["v1_wind_pd", "v1_wind_rl", "v1_lat_pdt", "v1_lat_rl"],
                                     ["strong wind · PD", "strong wind · RL", "far lateral · re-tuned PD",
                                      "far lateral · RL"], "fixed-gain control vs learned policy",
                                     notes=["seed 1001", "seed 1001", "seed 1002", "seed 1002"]),
            "layouts": lambda: quad(["v1_ly_a", "v1_ly_b", "v1_ly_c", "v1_ly_d"],
                                    ["gimbaled engine", "3 engines", "4 engines", "cold-gas thrusters"],
                                    "four engine layouts, each trained in Isaac", speed=3,
                                    notes=["seed 1000"] * 4),
            "results": lambda: results_v1(FPS * 7),
            "diagrams": lambda: layout_diagrams(FPS * 5),
        }
    if a.cut in ("v4", "v5"):  # v3 reordered: footage first (cold open with flight-computer outputs), cards last
        v3 = chapters
        chapters = {
            "hero": lambda: ecu_hero("v1_hero_wind", "Rocket landing in Isaac Sim",
                                     "aerodynamics, burning propellant, CFD thrust map  ·  RL policy in strong wind"),
            "layouts": v3["layouts"],
            "rl_vs_pd": v3["rl_vs_pd"],
            "retrain": v3["retrain"],
            "scenarios": v3["scenarios"],
            "results": v3["results"],
            "physics": lambda: physics_card(int(FPS * 3.5)),
            "diagrams": lambda: layout_diagrams(int(FPS * 3.5)),
        }
    if a.cut == "v5":  # v4 + ground-start hop (ignition, liftoff, hover, landing on a second pad) as the cold open
        v4 = chapters
        chapters = {
            "hop": lambda: ecu_hero("v1_hero_hop", "Rocket hop in Isaac Sim",
                                    "ignition, liftoff, hover, landing on a second pad  ·  RL policy",
                                    caption="ground start hop · RL policy · seed 1000", speed=1.6, slow_s=2.5),
            "hero": lambda: ecu_hero("v1_hero_wind", "", "", title_s=0.0),
            **{k: v for k, v in v4.items() if k not in ("hero", "retrain", "scenarios", "results", "physics",
                                                          "diagrams")},
            # a real tip-over: default PD, far lateral start, seed 1000 (one of its 10/60 tip-overs in eval_v1)
            "tipover": lambda: missions(["v1_tip_pd", "v1_tip_rl"], ["default PD", "RL trained here"],
                                        "far lateral start, seed 1000", speed=2),
            **{k: v4[k] for k in ("retrain", "scenarios", "results", "physics", "diagrams")},
        }
    w = imageio.get_writer(out, fps=FPS, codec="libx264", quality=5, macro_block_size=8)
    total = 0
    for k in a.only or list(chapters):
        frs = chapters[k]()
        for f in frs:
            w.append_data(f)
        total += len(frs)
        print("chapter", k, len(frs), flush=True)
    w.close()
    print("STORY_DONE", out, round(out.stat().st_size / 2 ** 20, 2), "MB", round(total / FPS, 1), "s")


if __name__ == "__main__":
    main()
