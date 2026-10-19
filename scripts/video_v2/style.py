"""Shared look for video v2: dark background, one typeface, one accent, red only for failure."""
from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

W, H, DPI = 1920, 1080, 100
BG = "#0b0b0c"
FG = "#e8e6e3"
MID = "#9a9893"
DIM = "#5c5a57"
GRID = "#2a2a2c"
ACCENT = "#f0a63a"
FAIL = "#e5484d"

plt.rcParams.update({
    "font.family": "Inter",
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "text.color": FG,
    "axes.edgecolor": GRID,
    "xtick.color": MID,
    "ytick.color": MID,
})


def ease(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def new_fig():
    return plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)


def frame_header(fig, num: str, title: str):
    fig.text(0.06, 0.93, num, fontsize=14, color=ACCENT, family="DejaVu Sans Mono")
    fig.text(0.09, 0.93, title.upper(), fontsize=14, color=MID)
    fig.add_artist(plt.Line2D([0.06, 0.95], [0.915, 0.915], color=GRID, lw=1, transform=fig.transFigure))


def style_axes(ax, grid: bool = True):
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MID, labelsize=11, length=3)
    if grid:
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)


class Writer:
    """Pipe matplotlib frames into ffmpeg (H.264, yuv420p)."""

    def __init__(self, path: Path, fps: int):
        self.path, self.fps, self.n = Path(path), fps, 0
        self.p = None

    def __enter__(self):
        self.p = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
             "-r", str(self.fps), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "23",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path)],
            stdin=subprocess.PIPE)
        return self

    def _grab(self, fig) -> bytes:
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        assert buf.shape[:2] == (H, W), buf.shape
        return buf.tobytes()

    def add(self, fig, close: bool = True):
        self.p.stdin.write(self._grab(fig))
        self.n += 1
        if close:
            plt.close(fig)

    def hold(self, fig, seconds: float, close: bool = True):
        b = self._grab(fig)
        for _ in range(int(seconds * self.fps)):
            self.p.stdin.write(b)
            self.n += 1
        if close:
            plt.close(fig)

    def __exit__(self, *exc):
        self.p.stdin.close()
        rc = self.p.wait()
        if rc != 0 and exc[0] is None:
            raise RuntimeError(f"ffmpeg failed ({rc}) for {self.path}")
