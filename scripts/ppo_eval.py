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
parser.add_argument("--checkpoint", default=None, help="PPO checkpoint (required for --controller ppo)")
parser.add_argument("--controller", default="ppo", choices=["ppo", "pd"],
                    help="pd = rocket_dt CascadedPD (numpy, per env) driving the same Isaac env through its action API")
parser.add_argument("--engine", default="conical_A")
parser.add_argument("--scenario", default="nominal", choices=["nominal", "wind", "rcs_fail", "noise", "lateral"])
parser.add_argument("--layout", default="gimbal_rcs", choices=["gimbal_rcs", "tri_diff", "quad_diff", "coldgas"])
parser.add_argument("--map_source", default=None, help="ideal1d | cfd (default: ideal1d for v0, cfd for v1)")
parser.add_argument("--physics", default="v0", choices=["v0", "v1"],
                    help="v1 = aero + variable mass + CFD map (ADR-0011); v0 = the ADR-0009/0010 env")
parser.add_argument("--pd_gains", default=None, help="tuned PDGains JSON from scripts/tune_pd.py (controller pd)")
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
from rocket_dt.isaac_env.pd_policy import PDPolicy  # noqa: E402
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
    from rocket_dt.isaac_env.layouts import layout_spec
    from rocket_dt.isaac_env.scenarios import eval_scenarios
    from rocket_dt.schemas import load_vehicle
    n_rcs = len(layout_spec(args.layout, load_vehicle(REPO / "configs/vehicle_v0.yaml")).rcs)
    torch.manual_seed(args.seed0)  # measurement-noise stream of the "noise" scenario
    if args.map_source is None:
        args.map_source = "cfd" if args.physics == "v1" else "ideal1d"
    cfg = VtvlEnvCfg()
    cfg.engine, cfg.map_source = args.engine, args.map_source
    cfg.aero = cfg.variable_mass = args.physics == "v1"
    cfg.layout, cfg.scenario = args.layout, args.scenario
    cfg.apply_scenario()
    cfg.__post_init__()  # regenerate the robot cfg for the selected engine's USD
    cfg.terminate_at_touchdown = False
    cfg.eval_scenarios = eval_scenarios(args.scenario, args.n, args.seed0, n_rcs=n_rcs)
    n_episodes = args.video_episodes if args.video else args.n
    cfg.scene.num_envs = 1 if args.video else args.n
    if args.video:
        cfg.scene.env_spacing = 0.0
        cfg.viewer.origin_type = "world"
        cfg.viewer.eye, cfg.viewer.lookat = (70.0, -70.0, 40.0), (0.0, 0.0, 32.0)
    env = VtvlEnv(cfg, render_mode="rgb_array" if args.video else None)
    wenv = RslRlVecEnvWrapper(env, clip_actions=None)
    if args.controller == "ppo":
        ck = Path(args.checkpoint)
        agent = json.loads((ck.parent / "manifest.json").read_text())["agent_cfg"]
        runner = OnPolicyRunner(wenv, agent, log_dir=None, device=str(env.device))
        runner.load(str(ck))
        policy = runner.get_inference_policy(device=env.device)
    else:
        ck = REPO / "runs/ppo_win-isaac/pd_isaac" / "CascadedPD"
        gains = None
        if args.pd_gains:
            from tune_pd import gains_from
            gains = gains_from(json.loads(Path(args.pd_gains).read_text())["tuned_gains"])
            ck = ck.with_name("CascadedPD_tuned")
        policy = PDPolicy(env, args.engine, args.map_source, gains)
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
    out = Path(args.out) if args.out else (ck.parent / "eval" if args.controller == "ppo" else ck.parent)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.engine}_{args.map_source}" + ("" if args.layout == "gimbal_rcs" else f"_{args.layout}") + \
        ("" if args.scenario == "nominal" else f"_{args.scenario}") + ("" if args.physics == "v0" else f"_{args.physics}") + \
        ("_video" if args.video else "")
    res = {"label": "win-isaac", "physics": args.physics, "pd_gains": args.pd_gains, "layout": args.layout, "scenario": args.scenario, "controller": args.controller, "checkpoint": str(ck) if args.controller == "ppo" else None, "engine": args.engine, "map_source": args.map_source,
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
