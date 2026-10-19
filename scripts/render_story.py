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
    fig.text(x, y, "LANDED" if ok else "MISSED", color=OKC if ok else BADC, fontsize=size, ha="center", weight="bold")
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
def title(n, grid="g_m1_A", speed=4):
    """Title over Isaac footage (16 envs landing), no black card."""
    g = Grid(grid)
    out = []
    for i in range(n):
        fig = new_fig()
        img, _ = g.at(g.n // 3 + i * speed)
        video(fig, (img * 0.55).astype(np.uint8))
        a = min(1.0, i / 10)
        fig.text(0.5, 0.54, "Thrust-only rocket landing in Isaac Sim", color=FG, fontsize=40, ha="center",
                 weight="light", alpha=a)
        fig.text(0.5, 0.45, "inspired by the cold-gas thruster package announced for the Tesla Roadster",
                 color=DIM, fontsize=17, ha="center", alpha=a)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="story_vtvl_isaac.mp4")
    ap.add_argument("--only", nargs="*", default=None)
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
