"""Run artifacts: manifest.json, metrics.json, trajectory.npz under runs/<run_id>/."""
from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy

from rocket_dt.schemas import RunManifest, stable_hash

REPO = Path(__file__).resolve().parents[3]


def software_versions() -> dict[str, str]:
    try:
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                               capture_output=True, text=True).stdout.strip() != ""
    except Exception:
        commit, dirty = "unknown", True
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "rocket_dt_commit": commit + ("-dirty" if dirty else "")}


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    raise TypeError(type(o))


def write_run(out_root: Path, manifest: RunManifest, metrics: dict, log: dict | None = None,
              episode: dict | None = None) -> Path:
    d = Path(out_root) / manifest.run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(asdict(manifest), indent=1, default=_default))
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1, default=_default))
    if episode is not None:
        (d / "episode.json").write_text(json.dumps(episode, indent=1, default=_default))
    if log is not None:
        np.savez_compressed(d / "trajectory.npz", **log)
    return d


def run_id(*parts, payload=None) -> str:
    h = stable_hash(payload)[:8] if payload is not None else ""
    return "_".join(str(p) for p in parts if p) + (f"_{h}" if h else "")
