"""Evaluate a VtvlEnv PPO checkpoint on the fixed landing suite (seeds 1000-1029) and optionally record video.

Same scenarios as evaluation.experiments.landing_suite (configs/scenarios/landing.yaml through
randomized_scenarios), same LANDING_LIMITS, deterministic policy (action mean), settle 3 s after cutoff.

usage:
  isaaclab.bat -p scripts\\ppo_eval.py --checkpoint runs\\ppo_win-isaac\\train\\r0_base\\model_149.pt --engine conical_A
  isaaclab.bat -p scripts\\ppo_eval.py --checkpoint ... --video --video_episodes 3 --enable_cameras
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from isaaclab.app import AppLauncher

REPO = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--engine", default="conical_A")
parser.add_argument("--map_source", default="ideal1d")
parser.add_argument("--n", type=int, default=30)
parser.add_argument("--seed0", type=int, default=1000)
parser.add_argument("--out", default=None)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_episodes", type=int, default=3)
parser.add_argument("--video_name", default="eval")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from rocket_dt.evaluation.experiments import randomized_scenarios, wilson  # noqa: E402
from rocket_dt.evaluation.simulate import LANDING_LIMITS, Scenario  # noqa: E402
from rocket_dt.isaac_env.rocket_env import VtvlEnv, VtvlEnvCfg  # noqa: E402


def summarize(recs):
    td = [r for r in recs if r["touchdown"]]
    k = sum(r["success"] for r in recs)

    def pct(xs, q):
        return float(np.percentile(xs, q)) if xs else None

    return {"n": len(recs), "n_success": k, "success_rate": k / len(recs), "wilson95": wilson(k, len(recs)),
            "n_physics_valid": sum(r["physics_valid"] for r in recs),
            "n_success_and_physics_valid": sum(r["success"] and r["physics_valid"] for r in recs),
            "td_vz_abs_p95": pct([abs(r["td_vz"]) for r in td], 95), "td_vh_p95": pct([r["td_vh"] for r in td], 95),
            "td_pos_err_median": pct([r["td_pos_err"] for r in td], 50),
            "td_pos_err_p95": pct([r["td_pos_err"] for r in td], 95),
            "td_pos_err_max": max([r["td_pos_err"] for r in td]) if td else None,
            "td_tilt_deg_max": max([r["td_tilt_deg"] for r in td]) if td else None,
            "fuel_mean_kg": float(np.mean([r["fuel_used_kg"] for r in recs])),
            "fuel_p95_kg": pct([r["fuel_used_kg"] for r in recs], 95),
            "termination_reasons": {t: sum(r["termination_reason"] == t for r in recs)
                                    for t in sorted({r["termination_reason"] for r in recs})}}


def main():
    base = Scenario(**yaml.safe_load(open(REPO / "configs/scenarios/landing.yaml")))
    scs = randomized_scenarios(base, args.n, args.seed0)
    keys = ("com_position_w", "velocity_w", "tilt_axis", "tilt_rad", "omega_b", "disturbance_force_w")
    cfg = VtvlEnvCfg()
    cfg.engine, cfg.map_source = args.engine, args.map_source
    cfg.__post_init__()  # regenerate the robot cfg for the selected engine's USD
    cfg.terminate_at_touchdown = False
    cfg.eval_scenarios = [{k: asdict(s)[k] for k in keys} for s in scs]
    n_episodes = args.video_episodes if args.video else args.n
    cfg.scene.num_envs = 1 if args.video else args.n
    if args.video:
        cfg.scene.env_spacing = 0.0
        cfg.viewer.origin_type = "world"
        cfg.viewer.eye, cfg.viewer.lookat = (70.0, -70.0, 40.0), (0.0, 0.0, 32.0)
    env = VtvlEnv(cfg, render_mode="rgb_array" if args.video else None)
    ck = Path(args.checkpoint)
    train_manifest = json.loads((ck.parent / "manifest.json").read_text())
    agent = train_manifest["agent_cfg"]
    wenv = RslRlVecEnvWrapper(env, clip_actions=None)
    runner = OnPolicyRunner(wenv, agent, log_dir=None, device=str(env.device))
    runner.load(str(ck))
    policy = runner.get_inference_policy(device=env.device)
    obs = wenv.get_observations()
    frames, steps = [], 0
    max_steps = int(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation)) * (n_episodes if args.video else 1) + 10
    while steps < max_steps:
        with torch.inference_mode():
            act = policy(obs)
        obs, _, _, _ = wenv.step(act)
        steps += 1
        if args.video and steps % 2 == 0:
            fr = env.render()
            if fr is not None:
                frames.append(np.asarray(fr)[..., :3])
        done_idx = {r["scenario_index"] for r in env.episode_results}
        if all(i in done_idx for i in range(n_episodes)):
            break
    recs = sorted([r for r in env.episode_results if r["scenario_index"] < n_episodes], key=lambda r: r["scenario_index"])
    seen, first = set(), []
    for r in recs:
        if r["scenario_index"] not in seen:
            seen.add(r["scenario_index"])
            r["seed"] = args.seed0 + r["scenario_index"]
            first.append(r)
    out = Path(args.out) if args.out else ck.parent / "eval"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.engine}_{args.map_source}" + ("_video" if args.video else "")
    res = {"label": "win-isaac", "checkpoint": str(ck), "engine": args.engine, "map_source": args.map_source,
           "map_hash": env.map_hash, "seeds": [args.seed0, args.seed0 + n_episodes - 1], "limits": LANDING_LIMITS,
           "settle_s": cfg.settle_s, "control_steps": steps, "summary": summarize(first) if first else None,
           "episodes": first}
    (out / f"eval_{tag}.json").write_text(json.dumps(res, indent=1))
    if args.video and frames:
        import imageio.v2 as imageio
        vdir = REPO / "runs/ppo_win-isaac/videos"
        vdir.mkdir(parents=True, exist_ok=True)
        path = vdir / f"{args.video_name}.mp4"
        imageio.mimwrite(path, frames, fps=25, codec="libx264", quality=6, macro_block_size=8)
        res["video"] = str(path)
        (out / f"eval_{tag}.json").write_text(json.dumps(res, indent=1))
    print("EVAL_RESULT", json.dumps({"summary": res["summary"], "n_frames": len(frames)}))
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
