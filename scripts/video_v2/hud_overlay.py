"""Video v2 HUD: composite raw Isaac renders with a webcast-style telemetry strip.

Inputs per run: runs/ppo_win-isaac/mission/<name>/{raw.mp4, telemetry.json[, raw_engine.mp4]}
(raw renders from scripts/isaac_mission_video.py on the PC; telemetry is per control step).
1–3 runs are placed side by side (engine swap = 3, PD vs PPO = 2).

  python scripts/video_v2/hud_overlay.py --runs pd_A_1004 --out runs/video/v2/clip_pd_A.mp4
  python scripts/video_v2/hud_overlay.py --runs swap_pd_A swap_pd_B swap_pd_C --title "Engine swap, PD"
  python scripts/video_v2/hud_overlay.py --grid grid_r0_m50 --title "r0 iteration 50: 16 envs"
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import ACCENT, BG, DIM, FAIL, FG, GRID, MID, H, W, Writer, new_fig  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MISSION = REPO / "runs/ppo_win-isaac/mission"
SEP_LIMIT = 0.4
CTRL_NAME = {"pd": "PD", "ppo": "PPO"}


def probe(path: Path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,nb_frames,r_frame_rate", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), int(s.get("nb_frames", 0) or 0), float(num) / float(den)


class Frames:
    """Sequential RGB frames from an mp4 via ffmpeg; repeats the last frame after the end."""

    def __init__(self, path: Path):
        self.w, self.h, self.n, self.fps = probe(path)
        self.p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "rawvideo",
                                   "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
        self.last = np.zeros((self.h, self.w, 3), np.uint8)

    def next(self):
        b = self.p.stdout.read(self.w * self.h * 3)
        if len(b) == self.w * self.h * 3:
            self.last = np.frombuffer(b, np.uint8).reshape(self.h, self.w, 3)
        return self.last


class Run:
    def __init__(self, name: str):
        d = MISSION / name
        self.name = name
        self.tel = json.loads((d / "telemetry.json").read_text())
        self.steps = self.tel["steps"]
        self.fe = int(self.tel.get("frame_every", 2))
        self.raw = Frames(d / "raw.mp4") if (d / "raw.mp4").exists() else None
        self.eng = Frames(d / "raw_engine.mp4") if (d / "raw_engine.mp4").exists() else None
        self.n_frames = int(self.tel.get("n_frames") or len(self.steps) // self.fe)
        self.failed = set(self.tel.get("rcs_failed") or [])
        self.td_k = next((i for i, s in enumerate(self.steps) if s.get("touchdown")), None)

    def step(self, f: int):
        return self.steps[min(len(self.steps) - 1, f * self.fe)]

    def title(self):
        c = CTRL_NAME.get(self.tel["controller"], self.tel["controller"])
        ck = self.tel.get("checkpoint")
        if ck:
            c += f" {Path(str(ck)).parent.name.split('_')[0]}"
        return f"{c}  ·  {self.tel['engine']}  ·  seed {self.tel['seed']}"


def label_value(fig, x, y, label, value, color=FG, size=26, alpha=1.0):
    fig.text(x, y + 0.038, label, fontsize=11, color=MID, alpha=alpha)
    fig.text(x, y, value, fontsize=size, color=color, family="DejaVu Sans Mono", alpha=alpha)


def engine_glyph(fig, rect, s, failed, ids=("rcs_01", "rcs_02", "rcs_03", "rcs_04")):
    ax = fig.add_axes(rect)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    on = s.get("engine_on", False)
    thr = float(s.get("throttle", 0.0)) if on else 0.0
    ax.add_patch(plt.Circle((0, 0), 0.55, fill=False, ec=GRID, lw=2))
    if thr > 0:
        th = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi * thr, 60)
        ax.plot(0.55 * np.cos(th), 0.55 * np.sin(th), color=ACCENT, lw=3)
    ax.add_patch(plt.Circle((0, 0), 0.22, color=ACCENT if on else GRID))
    # RCS ring at the nose: rcs_01/02 at +x, rcs_03/04 at -x (vehicle_v0.yaml)
    pos = {"rcs_01": (1.05, 0.25), "rcs_02": (1.05, -0.25), "rcs_03": (-1.05, -0.25), "rcs_04": (-1.05, 0.25)}
    frac = s.get("rcs_frac") or [0, 0, 0, 0]
    for k, rid in enumerate(ids):
        x, y = pos[rid]
        if rid in failed:
            ax.plot([x - 0.12, x + 0.12], [y - 0.12, y + 0.12], color=FAIL, lw=2)
            ax.plot([x - 0.12, x + 0.12], [y + 0.12, y - 0.12], color=FAIL, lw=2)
        else:
            lit = frac[k] > 0.01
            ax.add_patch(plt.Circle((x, y), 0.1, color=ACCENT if lit else GRID))


def strip(fig, run: Run, f: int, x0: float, w: float, compact: bool):
    s = run.step(f)
    fig.add_artist(plt.Rectangle((x0, 0), w, 0.13, color=BG, alpha=0.72, transform=fig.transFigure, lw=0))
    fig.add_artist(plt.Line2D([x0, x0 + w], [0.13, 0.13], color=GRID, lw=1, transform=fig.transFigure))
    v = np.asarray(s["v"])
    alt = s["p"][2] - run.tel.get("td_com_z", 0.0)
    pe = s.get("pe_pa")
    sep = pe is not None and s.get("engine_on") and pe < SEP_LIMIT
    vals = [("T+", f"{s['t']:5.1f}"), ("ALTITUDE", f"{max(0.0, alt):5.1f} m"),
            ("VERTICAL", f"{v[2]:+5.1f} m/s"), ("THROTTLE", f"{(s['throttle'] if s.get('engine_on') else 0):4.2f}"),
            ("Pe/Pa", f"{pe:4.2f}" if pe is not None and s.get("engine_on") else "  –"),
            ("TILT", f"{s.get('tilt_deg', 0):4.1f}°")]
    if compact:
        vals = [vals[1], vals[2], vals[4]]
    size = 22 if compact else 26
    step = (w - 0.12) / len(vals) if compact else 0.135
    for k, (lab, val) in enumerate(vals):
        col = FAIL if (lab == "Pe/Pa" and sep) else FG
        label_value(fig, x0 + 0.02 + k * step, 0.035, lab + ("  SEP RISK" if lab == "Pe/Pa" and sep else ""),
                    val, color=col, size=size)
    gh = 0.118
    gw = gh * H / W
    engine_glyph(fig, [x0 + w - gw - 0.015, 0.006, gw, gh], s, run.failed)
    if not compact:
        fig.text(x0 + w - gw - 0.03, 0.065, "MAIN  /  RCS", fontsize=11, color=MID, ha="right")


def verdict(fig, run: Run, f: int, x0: float, y: float):
    if run.td_k is None or f * run.fe < run.td_k:
        return
    e = run.tel["episode"]
    ok = e.get("success")
    head = "LANDED" if ok else f"FAILED · {e.get('termination_reason', '')}"
    phys = "" if e.get("physics_valid", True) else "   physics-invalid (nozzle in separation zone)"
    fig.text(x0, y, head, fontsize=22, color=ACCENT if ok else FAIL)
    fig.text(x0, y - 0.035, f"|vz| {abs(e['td_vz']):.2f} m/s   pad error {e['td_pos_err']:.2f} m   "
             f"fuel {e['fuel_used_kg']:.0f} kg{phys}", fontsize=13, color=FG)


def render_runs(names, out: Path, title: str, fps: int):
    runs = [Run(n) for n in names]
    n = max(r.n_frames for r in runs) + int(2.5 * fps)
    k = len(runs)
    with Writer(out, fps) as wr:
        for f in range(n):
            fig = new_fig()
            for i, r in enumerate(runs):
                x0 = i / k
                ax = fig.add_axes([x0, 0, 1 / k, 1])
                ax.axis("off")
                img = r.raw.next() if r.raw else np.zeros((H // 2, W // (2 * k), 3), np.uint8)
                ax.imshow(img, aspect="auto" if k > 1 else "equal", interpolation="bilinear")
                if k > 1:
                    fig.add_artist(plt.Line2D([x0, x0], [0, 1], color=BG, lw=3, transform=fig.transFigure))
                fig.text(x0 + 0.015, 0.95, r.title(), fontsize=15 if k > 1 else 17, color=FG)
                if r.failed:
                    fig.text(x0 + 0.015, 0.915, "failed: " + ", ".join(sorted(r.failed)), fontsize=13, color=FAIL)
                if r.eng is not None and k == 1:
                    eimg = r.eng.next()
                    pax = fig.add_axes([0.76, 0.62, 0.22, 0.22 * 16 / 9 * eimg.shape[0] / eimg.shape[1]])
                    pax.imshow(eimg)
                    pax.set_xticks([])
                    pax.set_yticks([])
                    for sp in pax.spines.values():
                        sp.set_color(MID)
                        sp.set_linewidth(0.8)
                    fig.text(0.76, 0.605, "engine camera", fontsize=11, color=MID)
                strip(fig, r, min(f, r.n_frames - 1), x0, 1 / k, compact=k > 1)
                verdict(fig, r, min(f, r.n_frames - 1), x0 + 0.015, 0.22)
            if title:
                fig.text(0.985, 0.95, title, fontsize=13, color=MID, ha="right")
            fig.text(0.985, 0.915, "win-isaac   flame = visual proxy of the 1D nozzle model", fontsize=11,
                     color=DIM, ha="right")
            wr.add(fig)
    return n


def render_grid(name: str, out: Path, title: str, caption: str, fps: int):
    d = MISSION / name
    meta = json.loads((d / "grid.json").read_text()) if (d / "grid.json").exists() else {}
    src = Frames(d / "raw.mp4")
    n = src.n or int(10 * fps)
    with Writer(out, fps) as wr:
        for f in range(n):
            fig = new_fig()
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")
            ax.imshow(src.next(), aspect="auto")
            fig.add_artist(plt.Rectangle((0, 0.88), 1, 0.12, color=BG, alpha=0.72, transform=fig.transFigure, lw=0))
            fig.text(0.02, 0.94, title, fontsize=20, color=FG)
            if caption:
                fig.text(0.02, 0.905, caption, fontsize=14, color=MID)
            outc = meta.get("outcomes")
            if outc:
                cnt = {kk: sum(1 for o in outc if o == kk) for kk in set(outc)}
                txt = "   ".join(f"{kk} {vv}" for kk, vv in sorted(cnt.items()))
                fig.text(0.98, 0.94, txt, fontsize=15, color=FG, ha="right")
            fig.text(0.98, 0.905, "win-isaac", fontsize=11, color=DIM, ha="right")
            wr.add(fig)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*")
    ap.add_argument("--grid")
    ap.add_argument("--title", default="")
    ap.add_argument("--caption", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=25)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.grid:
        n = render_grid(a.grid, out, a.title, a.caption, a.fps)
    else:
        n = render_runs(a.runs, out, a.title, a.fps)
    print(out, n, "frames")


if __name__ == "__main__":
    main()
