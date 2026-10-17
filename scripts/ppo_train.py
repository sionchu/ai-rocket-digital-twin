"""PPO (rsl_rl, bundled with Isaac Lab 3.0) on VtvlEnv. Label `win-isaac`.

usage: C:\\dev\\IsaacLab\\isaaclab.bat -p scripts\\ppo_train.py --run_name base --num_envs 4096 --max_iterations 300
Logs/checkpoints: runs/ppo_win-isaac/train/<run_name>/ (TensorBoard events, model_*.pt, manifest.json)
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

REPO = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--run_name", required=True)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=300)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--entropy_coef", type=float, default=0.005)
parser.add_argument("--gamma", type=float, default=0.995)
parser.add_argument("--num_steps_per_env", type=int, default=32)
parser.add_argument("--init_std", type=float, default=0.5)
parser.add_argument("--rew_shaping_scale", type=float, default=None)
parser.add_argument("--rew_td_success", type=float, default=None)
parser.add_argument("--rew_fuel_per_throttle", type=float, default=None)
parser.add_argument("--phi_pos_m", type=float, default=None)
parser.add_argument("--phi_tilt_rad", type=float, default=None)
parser.add_argument("--shaping", default=None, choices=["basic", "guided"])
parser.add_argument("--resume_from", default=None, help="checkpoint to initialise from (fine-tune)")
parser.add_argument("--save_interval", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.configclass import configclass  # noqa: E402
from isaaclab_rl.rsl_rl import (RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg,  # noqa: E402
                                RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg)

sys.path.insert(0, str(REPO / "src"))
from rocket_dt.isaac_env.rocket_env import VtvlEnv, VtvlEnvCfg  # noqa: E402


@configclass
class VtvlPPOCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 300
    save_interval = 50
    experiment_name = "vtvl_landing"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(hidden_dims=[128, 128], activation="elu", obs_normalization=True,
                             distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5))
    critic = RslRlMLPModelCfg(hidden_dims=[128, 128], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
                                     entropy_coef=0.005, num_learning_epochs=5, num_mini_batches=4,
                                     learning_rate=3e-4, schedule="adaptive", gamma=0.995, lam=0.95,
                                     desired_kl=0.01, max_grad_norm=1.0)


def versions():
    out = {"python": platform.python_version(), "torch": torch.__version__}
    for p in ("rsl-rl-lib", "gymnasium", "warp-lang", "numpy"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    lab = Path(r"C:\dev\IsaacLab")
    out["isaaclab_repo_version"] = (lab / "VERSION").read_text().strip() if (lab / "VERSION").exists() else None
    out["isaacsim_version"] = (lab / "_isaac_sim/VERSION").read_text().strip() if (lab / "_isaac_sim/VERSION").exists() else None
    out["isaaclab_git_describe"] = subprocess.run(["git", "describe", "--tags", "--always"], cwd=lab,
                                                  capture_output=True, text=True).stdout.strip()
    out["rocket_dt_commit"] = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=REPO,
                                             capture_output=True, text=True).stdout.strip()
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                       capture_output=True, text=True).stdout.strip()
    out["gpu"] = q
    return out


def main():
    log_dir = REPO / "runs/ppo_win-isaac/train" / args.run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    env_cfg = VtvlEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    for k in ("rew_shaping_scale", "rew_td_success", "rew_fuel_per_throttle", "phi_pos_m", "phi_tilt_rad", "shaping"):
        if getattr(args, k) is not None:
            setattr(env_cfg, k, getattr(args, k))
    agent = VtvlPPOCfg()
    agent.seed = args.seed
    agent.max_iterations = args.max_iterations
    agent.num_steps_per_env = args.num_steps_per_env
    agent.save_interval = args.save_interval
    agent.run_name = args.run_name
    agent.algorithm.learning_rate = args.lr
    agent.algorithm.entropy_coef = args.entropy_coef
    agent.algorithm.gamma = args.gamma
    agent.actor.distribution_cfg.init_std = args.init_std
    agent = handle_deprecated_rsl_rl_cfg(agent, md.version("rsl-rl-lib"))
    env = VtvlEnv(env_cfg)
    manifest = {"label": "win-isaac", "argv": sys.argv, "software": versions(), "map_hash": env.map_hash,
                "engine": env_cfg.engine, "map_source": env_cfg.map_source, "asset": env.asset_info,
                "env_cfg": {k: getattr(env_cfg, k) for k in dir(env_cfg) if k.startswith(("rew_", "phi_")) or k in (
                    "shaping", "decimation", "episode_length_s", "terminate_at_touchdown", "touchdown_height_m", "throttle0")},
                "agent_cfg": agent.to_dict()}
    (log_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    wenv = RslRlVecEnvWrapper(env, clip_actions=None)
    runner = OnPolicyRunner(wenv, agent.to_dict(), log_dir=str(log_dir), device=agent.device)
    if args.resume_from:
        runner.load(args.resume_from)
        manifest["resumed_from"] = args.resume_from
    t0 = time.time()
    runner.learn(num_learning_iterations=agent.max_iterations, init_at_random_ep_len=False)  # 60 s episodes: random start would hit the timeout penalty
    manifest["train_wall_s"] = time.time() - t0
    manifest["torch_max_allocated_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
    (log_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    print("TRAIN_DONE", log_dir, manifest["train_wall_s"])
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
