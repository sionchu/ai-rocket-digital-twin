"""rocket_dt CascadedPD (G3 baseline) driving VtvlEnv through its normalised action interface (label `win-isaac`)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from rocket_dt.control.baseline import CascadedPD, PDGains
from rocket_dt.dynamics.frames import isa_pressure
from rocket_dt.dynamics.mass import MassModel
from rocket_dt.propulsion.performance_map import EnginePerformanceMap, build_ideal_map
from rocket_dt.schemas import load_engine, load_vehicle

REPO = Path(__file__).resolve().parents[3]


class PDPolicy:
    """rocket_dt CascadedPD (G3 baseline, unchanged) on the Isaac state, mapped onto VtvlEnv's normalised actions.

    State handed to the PD: CoM position (env frame), CoM velocity, quaternion converted xyzw -> wxyz, body
    angular velocity, and the env's integrated propellant/RCS gas. The PD's own mass model therefore sees the
    propellant decrease although the Isaac body mass is fixed (ADR-0009), exactly as PPO cannot see it either.
    """

    def __init__(self, env, engine: str, map_source: str, gains: PDGains | None = None):
        if env.spec.name != "gimbal_rcs":
            raise NotImplementedError("CascadedPD allocates gimbal + roll RCS; other layouts have no PD baseline")
        veh = load_vehicle(REPO / "configs/vehicle_v0.yaml").with_engine(engine)
        eng = load_engine(REPO / f"configs/engines/{engine}.yaml")
        pm = build_ideal_map(eng) if map_source == "ideal1d" else EnginePerformanceMap.load(
            REPO / f"runs/cfd/maps/{engine}_cfd.json")
        self.env, self.mm = env, MassModel(veh, eng)
        self.ctrl = CascadedPD(veh, self.mm, pm, env.step_dt, gains)
        self.feet_zmin = float(np.min(np.array(veh.gear.foot_positions_body_m)[:, 2]))
        self.failed = np.zeros(len(veh.layout.rcs), dtype=bool)

    def __call__(self, obs):
        env = self.env
        # the controller sees the measured state (noise / latency in the "noise" scenario, else the true state)
        p, q, v, w_b = (t.detach().cpu().numpy().astype(float) for t in env.measured_state())
        mp, mr = env.m_prop.cpu().numpy().astype(float), env.m_rcs.cpu().numpy().astype(float)
        thr_act = env.act.throttle.cpu().numpy().astype(float)
        tmin, tmax, gmax, sdt = env.act.tmin, env.act.tmax, env.gmax, env.step_dt
        a = np.zeros((env.num_envs, 7))
        for i in range(env.num_envs):
            x = np.concatenate([p[i], v[i], [q[i, 3], q[i, 0], q[i, 1], q[i, 2]], w_b[i], [mp[i], mr[i]]])
            ms = self.mm.evaluate(max(mp[i], 0.0), max(mr[i], 0.0))
            cmd = self.ctrl(x, isa_pressure(max(p[i, 2], 0.0)), "descent", np.zeros(3),
                            ms.com_body_m[2] - self.feet_zmin, thr_act[i], self.failed)
            a[i, 0] = 2 * (cmd.throttle - tmin) / (tmax - tmin) - 1
            a[i, 1:3] = np.asarray(cmd.gimbal_rad) / gmax
            a[i, 3:7] = 2 * np.asarray(cmd.rcs_on_time_s) / sdt - 1
        return torch.tensor(np.clip(a, -1, 1), dtype=torch.float32, device=env.device)
