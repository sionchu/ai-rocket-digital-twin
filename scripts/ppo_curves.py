"""Learning curves from rsl_rl console logs (runs/ppo_win-isaac/<run>.log) -> curves.json + curves.png.

usage: python scripts/ppo_curves.py r0_base r1_pos1 r2_pos2_ent r3_guided
The TensorBoard event files in runs/ppo_win-isaac/train/<run>/ hold the same scalars.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
D = REPO / "runs/ppo_win-isaac"
KEYS = {"Mean reward": "mean_reward", "Mean episode length": "episode_length", "Metrics/success_rate": "success_rate",
        "Metrics/touchdown_rate": "touchdown_rate", "Metrics/td_vz_abs_mean": "td_vz_abs_mean",
        "Metrics/td_pos_err_mean": "td_pos_err_mean", "Metrics/fuel_used_mean": "fuel_used_mean"}
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def parse(path: Path):
    rows, cur = [], None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = ANSI.sub("", line).strip()
        m = re.match(r"Learning iteration (\d+)/(\d+)", line)
        if m:
            cur = {"iteration": int(m.group(1))}
            rows.append(cur)
            continue
        if cur is not None and ":" in line:
            k, _, v = line.rpartition(":")
            k = k.strip()
            if k in KEYS:
                try:
                    cur[KEYS[k]] = float(v)
                except ValueError:
                    pass
    return rows


def main(runs):
    curves = {r: parse(D / f"{r}.log") for r in runs}
    (D / "curves.json").write_text(json.dumps(curves))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing: wrote curves.json only")
        return
    panels = [("success_rate", "training success rate (stochastic policy)"), ("td_pos_err_mean", "touchdown position error [m]"),
              ("td_vz_abs_mean", "touchdown |vz| [m/s]"), ("mean_reward", "mean episode reward")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, (k, title) in zip(axes.ravel(), panels):
        for r, rows in curves.items():
            xs = [row["iteration"] for row in rows if k in row and row[k] == row[k]]
            ys = [row[k] for row in rows if k in row and row[k] == row[k]]
            ax.plot(xs, ys, label=r, lw=1.2)
        if k == "td_pos_err_mean":
            ax.axhline(2.0, color="gray", ls="--", lw=0.8)
        if k == "td_vz_abs_mean":
            ax.axhline(2.0, color="gray", ls="--", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("PPO iteration")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("VtvlEnv PPO, win-isaac (r1-r3 fine-tuned from r0 iteration 149); dashed = LANDING_LIMITS", fontsize=10)
    fig.savefig(D / "curves.png", dpi=110)
    print("wrote", D / "curves.png")


if __name__ == "__main__":
    main(sys.argv[1:])
