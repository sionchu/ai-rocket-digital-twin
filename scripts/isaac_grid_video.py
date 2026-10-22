"""Render N parallel envs of one checkpoint in a grid (training-stage montage), label `win-isaac`.

Same fixed scenarios (seeds seed0..seed0+N-1 of evaluation.experiments.randomized_scenarios) for every checkpoint,
so early / mid / late policies are compared on identical initial conditions. Each env shows an emissive plume cone
(length from the actual thrust of the last physics step) and a pad that turns green / red when its first episode
ends (success per LANDING_LIMITS). Writes raw.mp4 + grid.json under runs/ppo_win-isaac/grid/<name>/.

usage: isaaclab.bat -p scripts\\isaac_grid_video.py --checkpoint runs\\...\\model_0.pt --name r0_it0 --enable_cameras
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

from isaaclab.app import AppLauncher

REPO = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--controller", choices=["ppo", "pd"], default="ppo")
parser.add_argument("--name", required=True)
parser.add_argument("--engine", default="conical_A")
parser.add_argument("--n", type=int, default=16)
parser.add_argument("--seed0", type=int, default=1000)
parser.add_argument("--seconds", type=float, default=24.0)
parser.add_argument("--spacing", type=float, default=16.0)
parser.add_argument("--cam", nargs=4, type=float, default=[78.0, -78.0, 42.0, 30.0], help="eye dx dy dz, target z")
parser.add_argument("--look", choices=["default", "story"], default="default")
parser.add_argument("--physics", default="v0", choices=["v0", "v1"], help="v1 = aero + variable mass + CFD map (ADR-0011)")
parser.add_argument("--visual", default="v0", choices=["v0", "v1"], help="v1 = assets/vtvl_v1 vehicle, pad and sky")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

sys.path.insert(0, str(REPO / "src"))
from rocket_dt.dynamics.mass import MassModel  # noqa: E402
from rocket_dt.evaluation.experiments import randomized_scenarios  # noqa: E402
from rocket_dt.evaluation.simulate import Scenario  # noqa: E402
from rocket_dt.isaac_env.pd_policy import PDPolicy  # noqa: E402
from rocket_dt.isaac_env.rocket_env import VtvlEnv, VtvlEnvCfg, _rotmat_xyzw  # noqa: E402
from rocket_dt.schemas import load_engine, load_vehicle  # noqa: E402


def quat_z_to(d):
    z = np.array([0.0, 0.0, 1.0])
    d = d / max(np.linalg.norm(d), 1e-9)
    c = float(np.dot(z, d))
    if c < -0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0])
    ax = np.cross(z, d)
    q = np.array([ax[0], ax[1], ax[2], 1.0 + c])
    return q / np.linalg.norm(q)


def main():
    base = Scenario(**yaml.safe_load(open(REPO / "configs/scenarios/landing.yaml")))
    scs = randomized_scenarios(base, args.n, args.seed0)
    keys = ("com_position_w", "velocity_w", "tilt_axis", "tilt_rad", "omega_b", "disturbance_force_w")
    cfg = VtvlEnvCfg()
    cfg.engine = args.engine
    cfg.map_source = "cfd" if args.physics == "v1" else "ideal1d"
    cfg.aero = cfg.variable_mass = args.physics == "v1"
    cfg.visual = args.visual
    cfg.__post_init__()
    cfg.terminate_at_touchdown = False
    cfg.eval_scenarios = [{k: asdict(s)[k] for k in keys} for s in scs]
    if args.look == "story":
        cfg.sky_color, cfg.sky_intensity, cfg.sun_intensity = (0.30, 0.40, 0.56), 1200.0, 2500.0
    cfg.scene.num_envs = args.n
    cfg.scene.env_spacing = args.spacing
    cfg.viewer.origin_type = "world"
    env = VtvlEnv(cfg, render_mode="rgb_array")
    wenv = RslRlVecEnvWrapper(env, clip_actions=None)
    if args.controller == "ppo":
        ck = Path(args.checkpoint)
        agent = json.loads((ck.parent / "manifest.json").read_text())["agent_cfg"]
        runner = OnPolicyRunner(wenv, agent, log_dir=None, device=str(env.device))
        runner.load(str(ck))
        policy = runner.get_inference_policy(device=env.device)
    else:
        policy = PDPolicy(env, args.engine, cfg.map_source)

    def mat(rgb, emis, op=1.0):
        return sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, emissive_color=emis, opacity=op, roughness=0.9)
    vis = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/grid", markers={
        "plume": sim_utils.ConeCfg(radius=0.45, height=1.0, visual_material=mat((1.0, 0.6, 0.2), (4.0, 1.8, 0.4))),
        "pad_idle": sim_utils.CylinderCfg(radius=2.0, height=0.02, visual_material=mat((0.7, 0.7, 0.7), (0.0, 0.0, 0.0))),
        "pad_ok": sim_utils.CylinderCfg(radius=2.0, height=0.02, visual_material=mat((0.2, 0.9, 0.4), (0.4, 1.6, 0.6))),
        "pad_fail": sim_utils.CylinderCfg(radius=2.0, height=0.02, visual_material=mat((1.0, 0.2, 0.2), (1.8, 0.2, 0.2))),
    }))
    from isaacsim.core.rendering_manager import ViewportManager
    org = env.scene.env_origins.cpu().numpy().astype(float)
    c = org.mean(0)
    cam_eye, cam_tgt = (c + np.array(args.cam[:3])).tolist(), (c + np.array([0.0, 0.0, args.cam[3]])).tolist()
    veh = load_vehicle(REPO / "configs/vehicle_v0.yaml").with_engine(args.engine)
    mm = MassModel(veh, load_engine(REPO / f"configs/engines/{args.engine}.yaml"))
    pivot = np.array(veh.layout.main_engine.gimbal_pivot_body_m, float)
    from rocket_dt.isaac_env.rocket_env import NOZZLE_VIS_SCALE
    from rocket_dt.propulsion.geometry import build_contour
    _c = build_contour(load_engine(REPO / f"configs/engines/{args.engine}.yaml"))
    # flame starts at the visible (scaled) nozzle exit; physics uses the real exit only through the thrust map
    exit_z = float(pivot[2] - veh.mass.engine_throat_below_pivot_m
                   - NOZZLE_VIS_SCALE * (_c.x_m[-1] - _c.x_m[_c.throat_index]))
    com_b = env.com_b.cpu().numpy().astype(float)
    f_ref = float(env.map.lookup(torch.tensor([1.0], device=env.device), torch.tensor([101325.0], device=env.device),
                                 torch.zeros(1, dtype=torch.long, device=env.device))[0])
    out = REPO / "runs/ppo_win-isaac/grid" / args.name
    out.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio
    w = imageio.get_writer(out / "raw.mp4", fps=25, codec="libx264", quality=7, macro_block_size=8)
    outcome = [None] * args.n  # first-episode result per env
    frames_status = []
    obs = wenv.get_observations()
    steps = int(args.seconds / env.step_dt)
    rob = env.robot
    parked = torch.zeros(args.n, 7, device=env.device)
    is_parked = torch.zeros(args.n, dtype=torch.bool, device=env.device)
    for k in range(steps):
        prev_pose = rob.data.root_link_pose_w.torch.clone()  # pose before this step (the env may auto-reset inside)
        with torch.inference_mode():
            act = policy(obs)
        obs, _, _, _ = wenv.step(act)
        for r in env.episode_results:
            i = r["env"]
            if r["scenario_index"] == i and outcome[i] is None:
                outcome[i] = r
                parked[i] = prev_pose[i]
                is_parked[i] = True
        if is_parked.any():  # freeze finished envs where their first episode ended (visual only; result recorded)
            ids = is_parked.nonzero().squeeze(-1)
            rob.write_root_link_pose_to_sim_index(root_pose=parked[ids], env_ids=ids)
            rob.write_root_com_velocity_to_sim_index(root_velocity=torch.zeros(len(ids), 6, device=env.device),
                                                     env_ids=ids)
            rob.update(0.0)
        p, q, _, _ = (t.detach().cpu().numpy().astype(float) for t in env._state())
        rot = _rotmat_xyzw(torch.tensor(q, dtype=torch.float32)).numpy().astype(float)
        f = env.last_f_main.cpu().numpy().astype(float)
        gim = env.last_gimbal.cpu().numpy().astype(float)
        trans, quats, scales, ids = [], [], [], []
        for i in range(args.n):
            com_w = p[i] + org[i]
            d_b = np.array([math.cos(gim[i, 0]) * math.sin(gim[i, 1]), -math.sin(gim[i, 0]),
                            math.cos(gim[i, 0]) * math.cos(gim[i, 1])])
            exit_w = com_w + rot[i] @ (pivot + d_b * (exit_z - pivot[2]) - com_b)
            d_w = rot[i] @ d_b
            h = (0.6 + 6.0 * f[i] / f_ref) if (f[i] > 1.0 and outcome[i] is None) else 1e-3
            trans.append(exit_w - d_w * h / 2)
            quats.append(quat_z_to(d_w))
            scales.append([1.0, 1.0, h])
            ids.append(0)
            trans.append(org[i] + np.array([0.0, 0.0, 0.011]))
            quats.append(np.array([0.0, 0.0, 0.0, 1.0]))
            scales.append([1.0, 1.0, 1.0])
            ids.append(1 if outcome[i] is None else (2 if outcome[i]["success"] else 3))
        vis.visualize(translations=torch.tensor(np.array(trans), dtype=torch.float32),
                      orientations=torch.tensor(np.array(quats), dtype=torch.float32),
                      scales=torch.tensor(np.array(scales), dtype=torch.float32), marker_indices=ids)
        if k % 2 == 0:  # the video recorder sets its own view on the first capture, so set ours every frame
            ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=cam_eye, target=cam_tgt)
            fr = env.render()
            if fr is not None and np.asarray(fr).size:
                w.append_data(np.asarray(fr)[..., :3])
                frames_status.append({"t": round((k + 1) * env.step_dt, 3),
                                      "done": sum(o is not None for o in outcome),
                                      "success": sum(bool(o and o["success"]) for o in outcome)})
        if all(o is not None for o in outcome) and k > steps // 2:
            break
    w.close()
    res = {"label": "win-isaac", "controller": args.controller, "checkpoint": args.checkpoint, "engine": args.engine,
           "map_hash": env.map_hash, "seeds": [args.seed0, args.seed0 + args.n - 1], "episodes": outcome,
           "frames": frames_status,
           "summary": {"n": args.n, "done": sum(o is not None for o in outcome),
                       "success": sum(bool(o and o["success"]) for o in outcome),
                       "reasons": {r: sum(1 for o in outcome if o and o["termination_reason"] == r)
                                   for r in sorted({o["termination_reason"] for o in outcome if o})}}}
    (out / "grid.json").write_text(json.dumps(res, indent=1))
    print("GRID_DONE", out, json.dumps(res["summary"]))
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
