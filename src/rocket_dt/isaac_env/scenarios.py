"""Fixed-seed evaluation scenarios for the VtvlEnv scenario set (ADR-0010), label `win-isaac`.

Base: evaluation.experiments.randomized_scenarios on configs/scenarios/landing.yaml (seed per episode), then
per scenario:
  nominal   unchanged
  wind      wind magnitude U(0, 1500) N (instead of U(0, 300)), random direction, sinusoidal gust +-40 %,
            period U(3, 6) s, random phase
  rcs_fail  one RCS thruster (uniform over the layout's thrusters) failed for the whole episode, not observed
  noise     unchanged initial state; the env adds measurement noise and 1-step (20 ms) latency (torch RNG,
            seeded by the evaluation script)
  lateral   start radius U(25, 45) m (uniform in the annulus) instead of <= 8 m, random bearing
"""
from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from rocket_dt.evaluation.experiments import randomized_scenarios
from rocket_dt.evaluation.simulate import Scenario

REPO = Path(__file__).resolve().parents[3]
SCENARIOS = ("nominal", "wind", "rcs_fail", "noise", "lateral")
KEYS = ("com_position_w", "velocity_w", "tilt_axis", "tilt_rad", "omega_b", "disturbance_force_w")


def eval_scenarios(name: str, n: int, seed0: int, n_rcs: int = 4) -> list[dict]:
    if name not in SCENARIOS:
        raise ValueError(name)
    base = Scenario(**yaml.safe_load(open(REPO / "configs/scenarios/landing.yaml")))
    out = []
    for sc in randomized_scenarios(base, n, seed0):
        d = {k: asdict(sc)[k] for k in KEYS}
        rng = np.random.default_rng(10_000 + sc.seed)  # independent stream: base draws stay identical
        if name == "wind":
            m, th = rng.uniform(0, 1500.0), rng.uniform(0, 2 * math.pi)
            d["disturbance_force_w"] = [float(m * math.cos(th)), float(m * math.sin(th)), 0.0]
            d["gust_period_s"], d["gust_phase"] = float(rng.uniform(3, 6)), float(rng.uniform(0, 2 * math.pi))
        elif name == "rcs_fail":
            d["rcs_failed"] = int(rng.integers(n_rcs))
        elif name == "lateral":
            r = math.sqrt(rng.uniform((25 / 45) ** 2, 1.0)) * 45
            th = rng.uniform(0, 2 * math.pi)
            d["com_position_w"] = [float(r * math.cos(th)), float(r * math.sin(th)), d["com_position_w"][2]]
        d["seed"] = sc.seed
        out.append(d)
    return out
