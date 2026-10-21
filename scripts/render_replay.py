"""Render low-fidelity replay videos from the Python reference 6-DoF integrator.

These are kinematic replays of logged truth states (matplotlib 3D, not Isaac/USD).
Segments:
  1. engine_swap   - same landing scenario flown with conical_A / bell_B / bell_C (E2, 1D map)
  2. rcs_fault     - RCS rcs_01+rcs_03 failed, controller unaware vs aware, same seed
  3. cfd_vs_1d     - conical_A / bell_B flown with the 1D map vs the CFD map
Writes runs/video/<segment>.mp4, runs/video/overview.mp4 and manifest.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rocket_dt.dynamics.frames import quat_to_rotmat, tilt_angle  # noqa: E402
from rocket_dt.evaluation import experiments as ex  # noqa: E402
from rocket_dt.evaluation.artifacts import software_versions  # noqa: E402
from rocket_dt.evaluation.simulate import run_scenario  # noqa: E402
from rocket_dt.propulsion.adapter import gimbal_direction  # noqa: E402

FPS = 25
LABEL = "cloud-v1912 | Python reference 6-DoF (RK4) | PD baseline | low-fidelity replay, not Isaac"
COLORS = {"conical_A": "#2a6fdb", "bell_B": "#1f9d55", "bell_C": "#d9822b"}


def fly(sc, engine, map_source="ideal1d"):
    veh, eng, maps = ex.load_all(REPO, map_source)
    log, met, _ = run_scenario(sc, veh, eng[engine], eng[engine], maps[engine])
    return {"log": log, "metrics": met, "engine": engine, "veh": veh}


def body_geometry(run, i):
    log, veh = run["log"], run["veh"]
    R = quat_to_rotmat(log["q"][i])
    origin = log["p"][i] - R @ log["com_b"][i]
    half = veh.mass.structure_length_m / 2
    top, bot = origin + R @ [0, 0, half], origin + R @ [0, 0, -half]
    feet = [(origin + R @ [0, 0, -half + 1.0], origin + R @ f) for f in veh.gear.foot_positions_body_m]
    pivot = np.asarray(veh.layout.main_engine.gimbal_pivot_body_m, float)
    d = gimbal_direction(*log["gimbal"][i])
    exit_w = origin + R @ (pivot - 0.4 * d)
    plume = exit_w - R @ d * (log["thrust"][i] / 20000.0 * 5.0)
    return top, bot, feet, exit_w, plume


def draw_vehicle(ax, run, i, title, extra=""):
    log, met = run["log"], run["metrics"]
    ax.cla()
    p = log["p"]
    lim = 10
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(0, 65)
    ax.set_box_aspect((1, 1, 1.6))
    ax.view_init(elev=12, azim=-60)
    gx, gy = np.meshgrid([-lim, lim], [-lim, lim])
    ax.plot_surface(gx, gy, np.zeros_like(gx), color="#cfd6cf", alpha=0.35)
    ax.plot([0], [0], [0], marker="x", color="k", ms=8)
    ax.plot(p[: i + 1, 0], p[: i + 1, 1], p[: i + 1, 2], color="#888", lw=0.8)
    top, bot, feet, exit_w, plume = body_geometry(run, i)
    c = COLORS.get(run["engine"], "#444")
    ax.plot(*np.c_[bot, top], color=c, lw=6, solid_capstyle="round")
    for a, b in feet:
        ax.plot(*np.c_[a, b], color="#333", lw=1.5)
    if log["throttle"][i] > 0:
        pc = "#e0301e" if not log["map_valid"][i] else "#ff9a1f"
        ax.plot(*np.c_[exit_w, plume], color=pc, lw=4, alpha=0.8)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    t = log["t"][i]
    v = log["v"][i]
    txt = (f"t={t:5.1f}s  h={p[i, 2]:5.1f}m  vz={v[2]:5.2f}m/s\n"
           f"thr={log['throttle'][i]:.2f}  F={log['thrust'][i] / 1000:5.1f}kN  tilt={np.degrees(tilt_angle(log['q'][i])):4.1f}deg\n"
           f"fuel used={log['m_prop'][0] - log['m_prop'][i]:5.1f}kg  map valid={bool(log['map_valid'][i])}")
    ax.set_title(title, fontsize=10, color=c, pad=2)
    ax.text2D(0.02, 0.98, txt, transform=ax.transAxes, fontsize=7, va="top", family="monospace")
    if i == len(log["t"]) - 1 or (met.get("touchdown") and t >= met["touchdown"]["t"]):
        res = "LANDED" if met["success"] else "FAIL: " + ",".join(met.get("failure_reasons", []))
        valid = "" if met["physics_valid"] else "  (physics_valid=false)"
        ax.text2D(0.02, 0.05, res + valid + extra, transform=ax.transAxes, fontsize=8,
                  color="#1f7a1f" if met["success"] and met["physics_valid"] else "#b00020", weight="bold")


def render_segment(path, runs, titles, header, telemetry, n_tail_s=2.0):
    n = max(len(r["log"]["t"]) for r in runs)
    step = max(1, int(round(50 / FPS)))  # logs are at control rate (50 Hz)
    frames = list(range(0, n, step)) + [n - 1] * int(n_tail_s * FPS)
    k = len(runs)
    fig = plt.figure(figsize=(4.2 * k, 7.2), dpi=100)
    axes = [fig.add_subplot(2, k, j + 1, projection="3d") for j in range(k)]
    axt = fig.add_subplot(2, 1, 2)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.08, hspace=0.12)
    fig.suptitle(header, fontsize=12, weight="bold")
    fig.text(0.5, 0.005, LABEL, ha="center", fontsize=7, color="#555")
    w = FFMpegWriter(fps=FPS, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "23"])
    with w.saving(fig, str(path), dpi=100):
        for f in frames:
            for j, r in enumerate(runs):
                draw_vehicle(axes[j], r, min(f, len(r["log"]["t"]) - 1), titles[j])
            axt.cla()
            telemetry(axt, runs, f)
            w.grab_frame()
    plt.close(fig)
    return len(frames)


def tel_altitude_throttle(ax, runs, f):
    for r in runs:
        lg = r["log"]
        i = min(f, len(lg["t"]) - 1)
        c = COLORS.get(r["engine"], "#444")
        ax.plot(lg["t"][: i + 1], lg["thrust"][: i + 1] / 1000, color=c, label=f"{r['engine']} thrust [kN]")
    ax.set_xlim(0, max(r["log"]["t"][-1] for r in runs))
    ax.set_ylim(0, 40)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("main thrust [kN]")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)


def tel_roll(ax, runs, f):
    for r, ls in zip(runs, ("-", "--")):
        lg = r["log"]
        i = min(f, len(lg["t"]) - 1)
        ax.plot(lg["t"][: i + 1], lg["w"][: i + 1, 2], ls, label=f"{r['tag']}: roll rate wz [rad/s]")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlim(0, max(r["log"]["t"][-1] for r in runs))
    ax.set_ylim(-1.6, 1.6)
    ax.set_xlabel("time [s]")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)


def title_card(path, lines, seconds=3.0):
    fig = plt.figure(figsize=(12.6, 7.2), dpi=100)
    fig.text(0.5, 0.62, lines[0], ha="center", fontsize=22, weight="bold")
    for k, s in enumerate(lines[1:]):
        fig.text(0.5, 0.5 - 0.06 * k, s, ha="center", fontsize=13)
    fig.text(0.5, 0.04, LABEL, ha="center", fontsize=8, color="#555")
    w = FFMpegWriter(fps=FPS, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with w.saving(fig, str(path), dpi=100):
        for _ in range(int(seconds * FPS)):
            w.grab_frame()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/video")
    ap.add_argument("--seed-range", type=int, nargs=2, default=(1000, 1030))
    a = ap.parse_args()
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    base = ex._load_sc(REPO, "landing")
    man = {"label": LABEL, "software": software_versions(), "segments": {}}

    # 1. engine swap, nominal landing scenario
    runs = [fly(base, e) for e in ex.ENGINES]
    title_card(out / "t1.mp4", ["1. Engine swap (E2)", "Same vehicle, same PD controller, same landing scenario",
                                "conical_A  |  bell_B  |  bell_C  (ideal-1D maps)",
                                "Red plume = map cell flagged invalid (Pe/Pa < 0.4, separation risk)"])
    nf = render_segment(out / "engine_swap.mp4", runs, [f"{r['engine']}" for r in runs],
                        "Engine swap: descent and landing from 60 m", tel_altitude_throttle)
    man["segments"]["engine_swap"] = {"frames": nf, "scenario": base.name,
                                      "metrics": {r["engine"]: r["metrics"] for r in runs}}

    # 2. RCS double fault: first seed where the unaware controller fails
    pick = None
    for s in ex.randomized_scenarios(replace(base, rcs_failed=["rcs_01", "rcs_03"]), a.seed_range[1] - a.seed_range[0],
                                     a.seed_range[0]):
        un = fly(replace(s, fault_known_to_controller=False), "conical_A")
        if not un["metrics"]["success"]:
            pick = (s, un)
            break
    if pick:
        s, un = pick
        kn = fly(replace(s, fault_known_to_controller=True), "conical_A")
        un["tag"], kn["tag"] = "fault unknown", "fault known"
        title_card(out / "t2.mp4", ["2. RCS fault: rcs_01 + rcs_03 failed", f"seed {s.seed}, conical_A",
                                    "Left: controller unaware (no FDIR)   Right: controller aware (roll-damping fallback)",
                                    "Suite result over 30 seeds: unaware 19/30, aware 30/30"])
        nf = render_segment(out / "rcs_fault.mp4", [un, kn], ["fault unknown to controller", "fault known to controller"],
                            f"Positive-roll RCS lost (seed {s.seed})", tel_roll)
        man["segments"]["rcs_fault"] = {"frames": nf, "seed": s.seed,
                                        "metrics": {"unknown": un["metrics"], "known": kn["metrics"]}}

    # 3. 1D map vs CFD map
    cfd_engines = [e for e in ("conical_A", "bell_B") if (REPO / f"runs/cfd/maps/{e}_cfd.json").exists()]
    if cfd_engines:
        runs3 = []
        for e in cfd_engines:
            r1, r2 = fly(base, e, "ideal1d"), fly(base, e, "cfd")
            r1["engine"], r2["engine"] = e, e
            runs3 += [r1, r2]
        titles = [f"{e} {m}" for e in cfd_engines for m in ("1D map", "CFD map (v1912)")]
        title_card(out / "t3.mp4", ["3. Ideal-1D map vs CFD map", "Inviscid axisymmetric rhoCentralFoam, 3-grid GCI < 0.14%",
                                    "CFD thrust: conical_A -0.37%, bell_B -1.88% vs 1D"])
        nf = render_segment(out / "cfd_vs_1d.mp4", runs3, titles, "Same landing with 1D vs CFD performance maps",
                            tel_altitude_throttle)
        man["segments"]["cfd_vs_1d"] = {"frames": nf, "metrics": {t: r["metrics"] for t, r in zip(titles, runs3)}}

    parts = [p for p in ("t1", "engine_swap", "t2", "rcs_fault", "t3", "cfd_vs_1d") if (out / f"{p}.mp4").exists()]
    # segments differ in width (2-4 panels): pad each to a common canvas before concatenating
    inputs, chains = [], []
    for j, p in enumerate(parts):
        inputs += ["-i", str(out / f"{p}.mp4")]
        chains.append(f"[{j}:v]scale=1680:720:force_original_aspect_ratio=decrease,"
                      f"pad=1680:720:(ow-iw)/2:(oh-ih)/2:white,setsar=1,fps={FPS}[v{j}]")
    fc = ";".join(chains) + ";" + "".join(f"[v{j}]" for j in range(len(parts))) + f"concat=n={len(parts)}:v=1:a=0[o]"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", fc, "-map", "[o]",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out / "overview.mp4")], check=True)
    for p in ("t1", "t2", "t3"):
        (out / f"{p}.mp4").unlink(missing_ok=True)
    (out / "manifest.json").write_text(json.dumps(man, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(json.dumps({k: v.get("frames") for k, v in man["segments"].items()}))


if __name__ == "__main__":
    main()
