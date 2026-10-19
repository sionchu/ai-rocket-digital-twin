"""Video v2: narrative sections rendered from repo data (no Isaac needed).

Sections: motivation, three engines, Isaac validation, training, scoreboard, answer.
Every number on screen is read from a file in the repo; the source path is printed in the frame.
Usage: python scripts/video_v2/sections.py [--only s2_engines] [--fps 30]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rocket_dt.propulsion import nozzle1d  # noqa: E402
from rocket_dt.propulsion.geometry import build_contour  # noqa: E402
from rocket_dt.schemas import load_engine  # noqa: E402
from style import (ACCENT, BG, DIM, FAIL, FG, GRID, MID, W, H, Writer, ease,  # noqa: E402
                   frame_header, new_fig, style_axes)

OUT = REPO / "runs/video/v2"
ENGINES = ["conical_A", "bell_B", "bell_C"]
ENGINE_LABEL = {"conical_A": "A  conical 15°, ε 3.5", "bell_B": "B  bell 80%, ε 3.5",
                "bell_C": "C  bell 80%, ε 6.0"}
PA_SL = 101325.0


# ----------------------------------------------------------------------------- S1
def s1_motivation(wr: Writer, fps: int):
    lines = [
        (0.0, 0.70, 30, FG, "In October 2026 the Tesla Roadster demo lifted and rolled the car"),
        (0.0, 0.64, 30, FG, "on compressed-gas thrusters."),
        (2.2, 0.50, 22, MID, "Question"),
        (2.4, 0.44, 30, FG, "Can thruster-based thrust and attitude control be built,"),
        (2.4, 0.38, 30, FG, "checked against physics, and learned in Isaac Sim?"),
        (4.6, 0.24, 20, MID, "Test vehicle: 1.2 t, one gimballed throttleable main engine, four 100 N cold-gas roll"),
        (4.6, 0.20, 20, MID, "thrusters (Isp 65 s), 80 m powered descent to a pad. Three nozzle designs are compared."),
    ]
    dur = 9.0
    for i in range(int(dur * fps)):
        t = i / fps
        fig = new_fig()
        frame_header(fig, "01", "Motivation")
        for t0, y, size, col, txt in lines:
            a = ease((t - t0) / 0.6)
            if a > 0:
                fig.text(0.08, y, txt, fontsize=size, color=col, alpha=a)
        fig.text(0.08, 0.06, "Sources: techcrunch.com 2026-10-15, space.com (Roadster thruster demo). "
                 "Roadster figures are not used in the model.", fontsize=12, color=DIM)
        wr.add(fig)


# ----------------------------------------------------------------------------- S2
def _mach(engine: str):
    p = REPO / f"runs/ppo_win-isaac/hud_assets/mach_{engine}_L3.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return d["x"], d["r"], d["mach"]


def s2_engines(wr: Writer, fps: int):
    designs = {e: load_engine(REPO / f"configs/engines/{e}.yaml") for e in ENGINES}
    contours = {e: build_contour(d) for e, d in designs.items()}
    thr = np.linspace(0.40, 1.0, 61)
    curves = {}
    for e, d in designs.items():
        pts = [nozzle1d.evaluate(d, float(u), PA_SL) for u in thr]
        curves[e] = dict(F=np.array([p.thrust_n for p in pts]) / 1e3,
                         pr=np.array([p.exit_pressure_pa / PA_SL for p in pts]),
                         isp=np.array([p.isp_s for p in pts]),
                         state=[p.expansion_state for p in pts])
    xmax = max(c.x_m[-1] for c in contours.values())
    rmax = max(c.r_m.max() for c in contours.values())

    fig = new_fig()
    frame_header(fig, "02", "Three nozzle designs, same chamber (2.5 MPa, At 5.6 cm²)")
    col_axes = []
    for k, e in enumerate(ENGINES):
        ax = fig.add_axes([0.06 + k * 0.31, 0.50, 0.27, 0.33])
        style_axes(ax, grid=False)
        ax.set_xlim(contours[e].x_m[0] - 0.01, xmax * 1.9)
        ax.set_ylim(-rmax * 1.8, rmax * 1.8)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        m = _mach(e)
        if m is not None:
            x, r, ma = m
            for sgn in (1, -1):
                ax.scatter(x, sgn * r, c=ma, s=1.2, cmap="cividis", vmin=0, vmax=3.2,
                           linewidths=0, rasterized=True)
            note = "CFD rhoCentralFoam L3, Mach  (wsl-v2606 = cloud-v1912 < 1e-4)"
        else:
            note = "CFD NOT_RUN for C (1D model only)"
        c = contours[e]
        ax.plot(c.x_m, c.r_m, color=FG, lw=1.4)
        ax.plot(c.x_m, -c.r_m, color=FG, lw=1.4)
        fig.text(0.06 + k * 0.31, 0.86, ENGINE_LABEL[e], fontsize=18, color=FG)
        fig.text(0.06 + k * 0.31, 0.835, note, fontsize=11, color=DIM)
        col_axes.append(ax)
    sm = plt.cm.ScalarMappable(cmap="cividis", norm=plt.Normalize(0, 3.2))
    cax = fig.add_axes([0.94, 0.55, 0.008, 0.25])
    cb = fig.colorbar(sm, cax=cax)
    cb.ax.tick_params(colors=MID, labelsize=10)
    cb.outline.set_visible(False)
    cb.set_label("Mach", color=MID, fontsize=11)

    ax_pr = fig.add_axes([0.06, 0.10, 0.40, 0.28])
    ax_f = fig.add_axes([0.55, 0.10, 0.40, 0.28])
    for ax in (ax_pr, ax_f):
        style_axes(ax)
        ax.set_xlim(thr[0], thr[-1])
        ax.set_xlabel("throttle (sea level)", color=MID, fontsize=12)
    ax_pr.axhspan(0, 0.4, color=FAIL, alpha=0.10, lw=0)
    ax_pr.axhline(0.4, color=FAIL, lw=0.8)
    ax_pr.text(thr[0] + 0.01, 0.33, "flow-separation risk  Pe/Pa < 0.4", color=FAIL, fontsize=11)
    ax_pr.set_ylabel("Pe / Pa", color=MID, fontsize=12)
    ax_pr.set_ylim(0, 1.6)
    ax_f.set_ylabel("thrust [kN]", color=MID, fontsize=12)
    shades = {"conical_A": FG, "bell_B": MID, "bell_C": ACCENT}
    tag = {"conical_A": "A = B", "bell_B": None, "bell_C": "C"}
    for e in ENGINES:
        ax_pr.plot(thr, curves[e]["pr"], color=shades[e], lw=1.6)
        ax_f.plot(thr, curves[e]["F"], color=shades[e], lw=1.6)
        if tag[e]:
            ax_pr.text(thr[-1] - 0.005, curves[e]["pr"][-1] + 0.05, tag[e], color=shades[e],
                       fontsize=12, ha="right")
            ax_f.text(thr[-1] - 0.005, curves[e]["F"][-1] + (0.5 if e != "bell_C" else -1.4), tag[e],
                      color=shades[e], fontsize=12, ha="right")
    ax_f.text(thr[0] + 0.01, curves["conical_A"]["F"][-1] * 0.98,
              "A and B differ by < 1 % in 1D thrust; C is over-expanded at sea level", color=MID,
              fontsize=11)
    cur_pr = ax_pr.axvline(thr[0], color=FG, lw=0.8, alpha=0.6)
    cur_f = ax_f.axvline(thr[0], color=FG, lw=0.8, alpha=0.6)
    readouts = [fig.text(0.06 + k * 0.31, 0.45, "", fontsize=14, color=FG, family="DejaVu Sans Mono")
                for k in range(3)]
    thr_txt = fig.text(0.06, 0.415, "", fontsize=12, color=MID)
    fig.text(0.06, 0.03, "source: configs/engines/*.yaml, rocket_dt.propulsion.nozzle1d (ideal 1D, ISA sea level)"
             "   label cloud-v1912 (1D), wsl-v2606 (CFD)", fontsize=11, color=DIM)

    dur = 16.0
    n = int(dur * fps)
    for i in range(n):
        t = i / fps
        s = 0.5 - 0.5 * np.cos(np.pi * min(1.0, max(0.0, (t - 2.0) / 6.0)) * 2) if t < 14 else 0
        j = int(round(s * (len(thr) - 1)))
        cur_pr.set_xdata([thr[j]])
        cur_f.set_xdata([thr[j]])
        thr_txt.set_text(f"cursor: throttle {thr[j]:.2f}")
        for k, e in enumerate(ENGINES):
            c = curves[e]
            sep = c["pr"][j] < 0.4
            readouts[k].set_text(f"{c['F'][j]:5.2f} kN   Isp {c['isp'][j]:5.1f} s   Pe/Pa {c['pr'][j]:.2f}"
                                 + ("  SEP" if sep else ""))
            readouts[k].set_color(FAIL if sep else FG)
        wr.add(fig, close=False)
    plt.close(fig)


# ----------------------------------------------------------------------------- S3
def s3_isaac_check(wr: Writer, fps: int):
    m = json.loads((REPO / "runs/isaac_smoke_win-isaac/metrics.json").read_text())
    rows = []
    for c in m["cases"]:
        la, aa = c["lin_acc_w"], c["ang_acc_b"]
        lin = la.get("rel_err_vs_ref_fd")
        ang = aa.get("rel_err_vs_ref_fd")
        ang_s = f"{ang * 100:.3f} %" if ang is not None else f"abs {aa.get('abs_err_vs_ref_fd', 0):.1e}"
        ok = "PASS" if c["pass_lin"] and c["pass_ang"] else "FAIL"
        rows.append((c["case"], f"{lin * 100:.3f} %", ang_s, ok))
    dur = 8.0
    for i in range(int(dur * fps)):
        t = i / fps
        fig = new_fig()
        frame_header(fig, "03", "Is Isaac applying the forces we think it is?")
        fig.text(0.08, 0.76, "Same thrust and torque applied in Isaac Lab (PhysX) and in the RK4 reference; "
                 "compare accelerations.", fontsize=20, color=FG, alpha=ease(t / 0.6))
        hdr = ("case", "linear acc. error", "angular acc. error", "result")
        xs = (0.08, 0.38, 0.58, 0.80)
        for x, h in zip(xs, hdr):
            fig.text(x, 0.64, h, fontsize=15, color=MID)
        fig.add_artist(plt.Line2D([0.08, 0.9], [0.625, 0.625], color=GRID, lw=1, transform=fig.transFigure))
        for k, r in enumerate(rows):
            a = ease((t - 0.8 - 0.5 * k) / 0.4)
            if a <= 0:
                continue
            y = 0.57 - k * 0.07
            for x, v in zip(xs, r):
                col = (ACCENT if v == "PASS" else FAIL) if v in ("PASS", "FAIL") else FG
                fig.text(x, y, v, fontsize=17, color=col, alpha=a, family="DejaVu Sans Mono")
        a = ease((t - 3.5) / 0.6)
        fig.text(0.08, 0.22, f"limits: linear < {m['criteria']['lin_rel_err_lt'] * 100:.0f} %, angular < "
                 f"{m['criteria']['ang_rel_err_lt'] * 100:.0f} %.   mass {m['mass_model']['mass_kg']:.1f} kg, "
                 f"{m['num_envs']} envs, dt {m['dt_s']} s", fontsize=15, color=MID, alpha=a)
        fig.text(0.08, 0.17, "Found on the way: Isaac Lab 3.0 root pose is the link origin (not the CoM), quaternions "
                 "are xyzw, and set_masses only changes the readback, so mass is held fixed (ADR-0009).",
                 fontsize=15, color=MID, alpha=a)
        fig.text(0.08, 0.06, "source: runs/isaac_smoke_win-isaac/metrics.json   label win-isaac",
                 fontsize=11, color=DIM)
        wr.add(fig)


# ----------------------------------------------------------------------------- S4
def _series(d, run, key):
    s = d[run]["series"].get(key)
    if not s:
        return np.array([]), np.array([])
    a = np.array(s)
    return a[:, 0], a[:, 1]


def _smooth(y, k=7):
    if len(y) < k:
        return y
    ker = np.ones(k) / k
    yp = np.pad(y, (k // 2, k // 2), mode="edge")
    return np.convolve(yp, ker, mode="valid")


def s4_training(wr: Writer, fps: int):
    d = json.loads((OUT / "training_scalars.json").read_text())
    plan = [  # run, colour, lw, caption shown while this run is being drawn
        ("r0_base", FG, 1.8, "r0  from scratch, 4096 parallel envs. It first tips over, then drifts off, "
                             "then reaches the pad too fast."),
        ("r1_pos1", DIM, 1.0, "r1  position shaping 1 m: touches down, rarely within limits."),
        ("r2_pos2_ent", DIM, 1.0, "r2  position shaping 2 m, less entropy: no better."),
        ("r3_guided", FG, 1.8, "r3  guided: horizontal-speed reference from the PD law. Success climbs to 92 %."),
        ("m1_mix_ft", ACCENT, 1.8, "m1  continue r3 with engines A, B, C mixed per env."),
        ("m2_mix_scratch", ACCENT, 1.0, "m2  mixed engines from scratch: lands softly but burns 30 % more fuel."),
    ]
    fig = new_fig()
    frame_header(fig, "04", "Learning to land: 4096 envs per iteration, PPO (rsl-rl)")
    ax = fig.add_axes([0.06, 0.30, 0.56, 0.52])
    style_axes(ax)
    ax.set_xlim(0, 700)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("PPO iteration (r1–r3 start from r0 @149, m1 from r3 @398)", color=MID, fontsize=12)
    ax.set_ylabel("training success rate (stochastic policy)", color=MID, fontsize=12)
    lines = {}
    for run, col, lw, _ in plan:
        ls = "--" if run == "m2_mix_scratch" else "-"
        (lines[run],) = ax.plot([], [], color=col, lw=lw, ls=ls)
    tags = {}
    for run, col, _, _ in plan:
        tags[run] = ax.text(0, 0, run.split("_")[0], color=col, fontsize=12, alpha=0)

    axb = fig.add_axes([0.70, 0.30, 0.25, 0.52])
    style_axes(axb, grid=False)
    cats = ["tip over", "out of bounds", "hard / off-pad\ntouchdown", "landed"]
    cat_cols = [FAIL, FAIL, MID, ACCENT]
    bars = axb.barh(range(4), [0, 0, 0, 0], color=cat_cols, height=0.55)
    for b, a in zip(bars, (1.0, 0.55, 0.8, 1.0)):
        b.set_alpha(a)
    axb.set_yticks(range(4))
    axb.set_yticklabels(cats, color=FG, fontsize=13)
    axb.invert_yaxis()
    axb.set_xlim(0, 1.18)
    axb.set_xticks([0, 0.5, 1])
    axb.set_xticklabels(["0", "50 %", "100 %"], color=MID)
    axb.set_title("how episodes end", color=MID, fontsize=13, loc="left")
    bar_txt = [axb.text(0.02, k, "", color=FG, fontsize=12, va="center") for k in range(4)]
    run_txt = fig.text(0.70, 0.86, "", fontsize=14, color=FG)
    caption = fig.text(0.06, 0.20, "", fontsize=19, color=FG)
    fig.text(0.06, 0.06, "source: runs/ppo_win-isaac/train/*/events.out.tfevents (Isaac Lab 3.0.0, RTX 4070 Ti)"
             "   label win-isaac", fontsize=11, color=DIM)

    per_run_s = {"r0_base": 7.0, "r1_pos1": 2.5, "r2_pos2_ent": 2.5, "r3_guided": 6.0,
                 "m1_mix_ft": 4.5, "m2_mix_scratch": 4.5}
    for run, col, lw, cap in plan:
        xs, ys = _series(d, run, "success")
        ys = _smooth(ys)
        _, tip = _series(d, run, "tip_over")
        _, oob = _series(d, run, "oob")
        _, td = _series(d, run, "touchdown")
        _, succ = _series(d, run, "success")
        caption.set_text(cap)
        caption.set_color(col if col != DIM else FG)
        n = int(per_run_s[run] * fps)
        for i in range(n):
            j = max(1, int(round(ease(i / max(1, n - 1)) * len(xs))))
            lines[run].set_data(xs[:j], ys[:j])
            tags[run].set_position((xs[j - 1] + 4, ys[j - 1]))
            tags[run].set_alpha(1)
            k = j - 1
            tot = max(1e-9, tip[k] + oob[k] + td[k])
            vals = [tip[k] / tot, oob[k] / tot, max(0.0, td[k] - succ[k]) / tot, succ[k] / tot]
            for b, v, tx in zip(bars, vals, bar_txt):
                b.set_width(v)
                tx.set_text(f"{v * 100:.0f} %")
                tx.set_x(v + 0.02)
            run_txt.set_text(f"{run}   iteration {int(xs[k])}")
            wr.add(fig, close=False)
        wr.hold(fig, 0.6, close=False)
    caption.set_text("Training success is not evaluation: checkpoints are re-scored deterministically on held-out seeds.")
    caption.set_color(MID)
    wr.hold(fig, 3.0)


# ----------------------------------------------------------------------------- S5
SCORE = [  # controller, [(success/60, fuel kg, |vz| P95) per engine A, B, C]  from summary_mixed.md
    ("PD (cascaded, Isaac)", [(60, 70.3, 1.30), (60, 70.2, 1.30), (60, 72.8, 1.32)]),
    ("PPO r3 (trained on A)", [(57, 72.6, 1.97), (42, 66.9, 2.08), (9, 98.5, 1.39)]),
    ("PPO m1 (mixed, fine-tuned)", [(56, 84.9, 1.18), (57, 81.9, 1.26), (57, 90.5, 0.94)]),
    ("PPO m2 (mixed, scratch)", [(57, 106.9, 0.41), (59, 101.3, 0.43), (50, 105.2, 0.50)]),
]


def s5_scoreboard(wr: Writer, fps: int):
    dur = 13.0
    xs0, cw = 0.36, 0.20
    for i in range(int(dur * fps)):
        t = i / fps
        fig = new_fig()
        frame_header(fig, "05", "Same Isaac env, 60 seeds per cell, landing limits |vz| ≤ 2 m/s, pad error ≤ 2 m")
        for k, e in enumerate(ENGINES):
            fig.text(xs0 + k * cw, 0.78, ENGINE_LABEL[e], fontsize=16, color=FG)
        fig.text(xs0 + 2 * cw, 0.745, "physics-invalid: Pe/Pa < 0.4 on its path", fontsize=11, color=FAIL)
        fig.add_artist(plt.Line2D([0.06, 0.95], [0.73, 0.73], color=GRID, lw=1, transform=fig.transFigure))
        for r, (name, cells) in enumerate(SCORE):
            y = 0.64 - r * 0.13
            a_row = ease((t - 0.5 - 1.6 * r) / 0.5)
            if a_row <= 0:
                continue
            fig.text(0.06, y, name, fontsize=17, color=FG, alpha=a_row)
            for k, (s, fuel, vz) in enumerate(cells):
                a = ease((t - 0.7 - 1.6 * r - 0.25 * k) / 0.4)
                if a <= 0:
                    continue
                x = xs0 + k * cw
                frac = s / 60
                col = ACCENT if frac >= 0.9 else (MID if frac >= 0.6 else FAIL)
                fig.text(x, y, f"{s}/60", fontsize=24, color=col, alpha=a, family="DejaVu Sans Mono")
                fig.add_artist(plt.Rectangle((x, y - 0.025), 0.15 * frac * a, 0.006, color=col,
                                             transform=fig.transFigure, lw=0))
                fig.add_artist(plt.Rectangle((x, y - 0.025), 0.15, 0.006, color=GRID,
                                             transform=fig.transFigure, lw=0, zorder=-1))
                fig.text(x, y - 0.055, f"fuel {fuel:.0f} kg   |vz| P95 {vz:.2f}"
                         + ("   sim only" if k == 2 else ""), fontsize=11, color=MID, alpha=a)
        a = ease((t - 8.0) / 0.6)
        fig.text(0.06, 0.13, "r3 learned engine A and broke on a swap (bell_C 9/60). Mixing engines in training fixed "
                 "that (57/60), at +12 kg fuel.", fontsize=16, color=FG, alpha=a)
        fig.text(0.06, 0.09, "PD still lands every case with margin. No PPO run beats it except m2 on touchdown speed.",
                 fontsize=16, color=FG, alpha=a)
        fig.text(0.06, 0.04, "source: runs/ppo_win-isaac/summary_mixed.md   label win-isaac", fontsize=11, color=DIM)
        wr.add(fig)


# ----------------------------------------------------------------------------- S6
def s6_answer(wr: Writer, fps: int):
    items = [
        (FG, "Yes, in simulation. Isaac reproduces the thrust/torque physics within 0.3 %, and both a"),
        (FG, "cascaded PD and a PPO policy land the thruster-controlled vehicle."),
        (MID, "Not shown here and not claimed:"),
        (MID, "  real hardware, combustion, or plume physics (flames are a visual proxy of the 1D nozzle model);"),
        (MID, "  aerodynamics; propellant mass change inside Isaac (held fixed, ADR-0009);"),
        (MID, "  bell_C landings: its nozzle runs in the separation-risk zone, so they are not physical evidence."),
        (FG, "Next: Isaac version pin, variable mass, CFD-based maps for all three engines."),
    ]
    dur = 10.0
    for i in range(int(dur * fps)):
        t = i / fps
        fig = new_fig()
        frame_header(fig, "06", "Answer")
        for k, (col, txt) in enumerate(items):
            a = ease((t - 0.3 - 0.7 * k) / 0.5)
            if a > 0:
                fig.text(0.08, 0.72 - k * 0.075, txt, fontsize=21 if col == FG else 18, color=col, alpha=a)
        fig.text(0.08, 0.06, "repo: sionchu/ai-rocket-digital-twin (private)   PROJECT_LEDGER.md, docs/conclusion.md",
                 fontsize=11, color=DIM)
        wr.add(fig)


SECTIONS = {
    "s1_motivation": s1_motivation,
    "s2_engines": s2_engines,
    "s3_isaac_check": s3_isaac_check,
    "s4_training": s4_training,
    "s5_scoreboard": s5_scoreboard,
    "s6_answer": s6_answer,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, fn in SECTIONS.items():
        if a.only and name not in a.only:
            continue
        t0 = time.time()
        path = OUT / f"{name}.mp4"
        with Writer(path, a.fps) as wr:
            fn(wr, a.fps)
        h = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        manifest[name] = {"file": path.name, "frames": wr.n, "sha256_16": h,
                          "render_s": round(time.time() - t0, 1)}
        print(name, manifest[name], flush=True)
    mpath = OUT / "sections_manifest.json"
    old = json.loads(mpath.read_text()) if mpath.exists() else {}
    old.update(manifest)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                            text=True).stdout.strip()
    old["_meta"] = {"commit": commit, "matplotlib": matplotlib.__version__, "size": [W, H], "fps": a.fps}
    mpath.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main()
