"""VtvlEnv smoke: random actions (no NaN, timing, VRAM, episode bookkeeping) + set_masses_index probe.

usage: C:\\dev\\IsaacLab\\isaaclab.bat -p scripts\\ppo_env_smoke.py --num_envs 256 --steps 600
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

REPO = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--out", default=str(REPO / "runs/ppo_win-isaac/env_smoke"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

sys.path.insert(0, str(REPO / "src"))
from rocket_dt.isaac_env.rocket_env import VtvlEnv, VtvlEnvCfg  # noqa: E402


def vram_used():
    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip()
    return int(q.splitlines()[0])


def main():
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vram0 = vram_used()
    cfg = VtvlEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = VtvlEnv(cfg)
    obs, _ = env.reset()
    n_nan, rew_sum, n_done = 0, 0.0, 0
    term = {"tip": 0, "oob": 0, "timeout": 0, "touchdown": 0, "success": 0}
    torch.cuda.synchronize()
    t0 = time.time()
    for k in range(args.steps):
        # random but biased to roughly hover thrust so some episodes reach the ground
        a = torch.rand(env.num_envs, 7, device=env.device) * 2 - 1
        a[:, 0] = 0.15 * a[:, 0] - 0.2
        a[:, 1:3] *= 0.3
        obs, rew, term_b, tout_b, _ = env.step(a)
        n_nan += int((~torch.isfinite(obs["policy"])).any(1).sum() + (~torch.isfinite(rew)).sum())
        rew_sum += float(rew.sum())
        d = term_b | tout_b
        n_done += int(d.sum())
        term["tip"] += int((env._tip & d).sum())
        term["oob"] += int((env._oob & d).sum())
        term["timeout"] += int(env._timeout.sum())
        term["touchdown"] += int((env._settled & d).sum())
        term["success"] += int((env._last_success & d).sum())
    torch.cuda.synchronize()
    wall = time.time() - t0
    vram1 = vram_used()

    # ---- probe: does a mass-only update through the tensor API take effect (and leave inertia alone)?
    rob = env.robot
    rv = rob.root_view
    m_before = rv.get_masses().numpy().ravel().copy()
    i_before = rv.get_inertias().numpy().reshape(env.num_envs, 9).copy()
    new_m = torch.full((env.num_envs, 1), env.mass, device=env.device)
    new_m[0, 0] = env.mass - 80.0
    rob.set_masses_index(masses=new_m)
    m_after = rv.get_masses().numpy().ravel()
    i_after = rv.get_inertias().numpy().reshape(env.num_envs, 9)
    # dynamic check: same world force on env 0 (lighter) and env 1, gravity on, one physics step
    pose = rob.data.root_link_pose_w.torch.clone()
    pose[:2, 2] = env.scene.env_origins[:2, 2] + 200.0
    pose[:2, 3:] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device)
    rob.write_root_link_pose_to_sim_index(root_pose=pose[:2], env_ids=torch.arange(2, device=env.device))
    rob.write_root_com_velocity_to_sim_index(root_velocity=torch.zeros(2, 6, device=env.device),
                                             env_ids=torch.arange(2, device=env.device))
    comp = rob.permanent_wrench_composer
    comp.reset()
    fz = 20000.0
    comp.set_forces_and_torques_index(forces=torch.tensor([[[0.0, 0.0, fz]]] * 2, device=env.device),
                                      torques=torch.zeros(2, 1, 3, device=env.device), positions=None,
                                      is_global=True, env_ids=torch.arange(2, device=env.device))
    rob.write_data_to_sim()
    env.sim.step(render=False)
    rob.update(env.physics_dt)
    vz = rob.data.root_com_lin_vel_w.torch[:2, 2].cpu().tolist()
    a_meas = [v / env.physics_dt for v in vz]
    a_exp = [fz / (env.mass - 80.0) - 9.80665, fz / env.mass - 9.80665]
    probe = {"mass_before": m_before[:2].tolist(), "mass_after": m_after[:2].tolist(),
             "inertia_unchanged": bool(abs(i_after - i_before).max() < 1e-3),
             "az_measured": a_meas, "az_expected": a_exp,
             "rel_err": [abs(m - e) / abs(e) for m, e in zip(a_meas, a_exp)]}
    probe["mass_only_update_works"] = bool(max(probe["rel_err"]) < 1e-3 and probe["inertia_unchanged"])

    res = {"label": "win-isaac", "num_envs": env.num_envs, "control_steps": args.steps, "wall_s": wall,
           "env_steps_per_s": args.steps * env.num_envs / wall, "nan_or_inf_count": n_nan,
           "episodes_finished": n_done, "terminations": term, "reward_sum": rew_sum,
           "vram_used_mib_before_app": vram0, "vram_used_mib_after_rollout": vram1,
           "vram_delta_mib": vram1 - vram0, "torch_max_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
           "asset": env.asset_info, "td_com_z": env.td_com_z, "map_hash": env.map_hash, "set_masses_probe": probe}
    (out / f"env_smoke_n{env.num_envs}.json").write_text(json.dumps(res, indent=1))
    print("ENV_SMOKE", json.dumps(res))
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
