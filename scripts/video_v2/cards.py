"""Video v2, LinkedIn cut: a few short one-line cards between the Isaac footage.

  python scripts/video_v2/cards.py   -> runs/video/v2/cards/c*.mp4
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sections import OUT, _series, _smooth  # noqa: E402
from style import ACCENT, DIM, FG, MID, Writer, ease, new_fig, style_axes  # noqa: E402

CARDS = OUT / "cards"
FPS = 30


def fade(t, dur, tin=0.4, tout=0.4):
    return ease(t / tin) * ease((dur - t) / tout)


def c1_title(wr):
    dur = 3.5
    for i in range(int(dur * FPS)):
        t = i / FPS
        a = fade(t, dur)
        fig = new_fig()
        fig.text(0.5, 0.52, "Can Isaac Sim learn thruster control?", fontsize=46, color=FG, ha="center", alpha=a)
        fig.text(0.5, 0.44, "inspired by the Roadster cold-gas thruster demo", fontsize=20, color=MID,
                 ha="center", alpha=a)
        wr.add(fig)


def c2_training(wr):
    d = json.loads((OUT / "training_scalars.json").read_text())
    xs, ys = [], []
    for run in ("r0_base", "r3_guided", "m1_mix_ft"):
        x, y = _series(d, run, "success")
        xs.append(x)
        ys.append(_smooth(y))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    dur, draw = 6.0, 4.6
    fig = new_fig()
    ax = fig.add_axes([0.12, 0.18, 0.76, 0.56])
    style_axes(ax)
    ax.set_xlim(0, x[-1])
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "50%", "100%"], fontsize=14)
    ax.set_xticks([])
    (ln,) = ax.plot([], [], color=ACCENT, lw=2.5)
    big = fig.text(0.12, 0.80, "", fontsize=40, color=FG, family="DejaVu Sans Mono")
    fig.text(0.5, 0.80, "landing success while training · 4096 parallel envs", fontsize=18, color=MID)
    for i in range(int(dur * FPS)):
        t = i / FPS
        j = max(1, int(round(ease(t / draw) * len(x))))
        ln.set_data(x[:j], y[:j])
        big.set_text(f"{y[j - 1] * 100:3.0f}%")
        wr.add(fig, close=False)


def c3_swap(wr):
    dur = 4.0
    for i in range(int(dur * FPS)):
        t = i / FPS
        a = fade(t, dur)
        b = ease((t - 1.0) / 0.5) * ease((dur - t) / 0.4)
        fig = new_fig()
        fig.text(0.5, 0.62, "new engine, same policy", fontsize=22, color=MID, ha="center", alpha=a)
        fig.text(0.40, 0.45, "9/60", fontsize=64, color=DIM, ha="center", family="DejaVu Sans Mono", alpha=a)
        fig.text(0.5, 0.47, "→", fontsize=40, color=MID, ha="center", alpha=b)
        fig.text(0.60, 0.45, "57/60", fontsize=64, color=ACCENT, ha="center", family="DejaVu Sans Mono", alpha=b)
        fig.text(0.5, 0.34, "after training on mixed engines (simulation)", fontsize=20, color=MID, ha="center", alpha=b)
        wr.add(fig)


def c4_end(wr):
    dur = 3.0
    for i in range(int(dur * FPS)):
        t = i / FPS
        a = fade(t, dur)
        fig = new_fig()
        fig.text(0.5, 0.52, "Isaac Lab · PPO · NVIDIA Flow · OpenFOAM", fontsize=28, color=FG, ha="center", alpha=a)
        fig.text(0.5, 0.45, "simulation only · flame is a visual proxy", fontsize=16, color=DIM, ha="center",
                 alpha=a)
        wr.add(fig)


def main():
    CARDS.mkdir(parents=True, exist_ok=True)
    for fn in (c1_title, c2_training, c3_swap, c4_end):
        p = CARDS / f"{fn.__name__}.mp4"
        with Writer(p, FPS) as wr:
            fn(wr)
        print(p.name, wr.n)


if __name__ == "__main__":
    main()
