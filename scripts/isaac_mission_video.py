"""Render one landing episode in Isaac with thrust visualisation + per-step telemetry (label `win-isaac`).

Scene additions (visual only, no physics): emissive plume cone on the nozzle exit (length and width from the
actual main thrust of the last physics step, axis from the actual gimbal), RCS puffs on firing thrusters, and
force arrows from the CoM (thrust, wind, net force incl. gravity; scales printed in the telemetry file).
Camera: chase camera at --cam_dist from the CoM. Output: raw mp4 + telemetry.json, composited later by
scripts/render_mission_hud.py.

usage: isaaclab.bat -p scripts\\isaac_mission_video.py --controller pd --engine conical_A --seed 1003 --name pd_A_1003 --enable_cameras
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
parser.add_argument("--controller", choices=["ppo", "pd"], required=True)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--engine", default="conical_A")
parser.add_argument("--seed", type=int, default=1003)
parser.add_argument("--name", required=True)
parser.add_argument("--cam_dist", type=float, default=24.0)
parser.add_argument("--cam_az_deg", type=float, default=-60.0)
parser.add_argument("--frame_every", type=int, default=2, help="control steps per video frame (2 -> 25 fps real time)")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--plume", choices=["cone", "flow", "both"], default="cone",
                    help="cone = emissive marker cones; flow = NVIDIA Flow fire (visual only)")
parser.add_argument("--flow_set", nargs="*", default=[], help="prim_rel:attr=value overrides for the Flow plume")
parser.add_argument("--flow_speed", nargs=2, type=float, default=[10.0, 15.0], help="box-local exhaust speed = a + b * thrust_frac")
parser.add_argument("--max_steps", type=int, default=3000)
parser.add_argument("--rcs_failed", nargs="*", default=[], help="RCS ids failed from t=0 (known to the PD allocator)")
parser.add_argument("--engine_cam", action="store_true", help="second camera close to the nozzle (picture-in-picture)")
parser.add_argument("--look", choices=["default", "story"], default="default")
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
from rocket_dt.dynamics.frames import G0  # noqa: E402
from rocket_dt.dynamics.mass import MassModel  # noqa: E402
from rocket_dt.evaluation.experiments import randomized_scenarios  # noqa: E402
from rocket_dt.evaluation.simulate import Scenario  # noqa: E402
from rocket_dt.isaac_env.pd_policy import PDPolicy  # noqa: E402
from rocket_dt.isaac_env.rocket_env import VtvlEnv, VtvlEnvCfg, _rotmat_xyzw  # noqa: E402
from rocket_dt.schemas import load_engine, load_vehicle  # noqa: E402

# arrow scales [m per N]; recorded in telemetry so the HUD legend is honest
ARROW_SCALE = {"thrust": 0.5e-3, "net": 2.0e-3, "wind": 10e-3}


def quat_z_to(d: np.ndarray) -> np.ndarray:
    """xyzw quaternion rotating +Z onto unit vector d."""
    z = np.array([0.0, 0.0, 1.0])
    d = d / max(np.linalg.norm(d), 1e-9)
    c = float(np.dot(z, d))
    if c < -0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0])
    ax = np.cross(z, d)
    q = np.array([ax[0], ax[1], ax[2], 1.0 + c])
    return q / np.linalg.norm(q)


def markers_cfg():
    def mat(rgb, emis, op=1.0):
        return sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, emissive_color=emis, opacity=op, roughness=0.9)
    return VisualizationMarkersCfg(prim_path="/Visuals/vtvl", markers={
        "plume_core": sim_utils.ConeCfg(radius=0.26, height=1.0, visual_material=mat((1.0, 0.85, 0.5), (4.0, 2.6, 0.9))),
        "plume_outer": sim_utils.ConeCfg(radius=0.50, height=1.0, visual_material=mat((1.0, 0.35, 0.05), (3.0, 0.7, 0.05), 0.45)),
        "rcs": sim_utils.ConeCfg(radius=0.10, height=1.0, visual_material=mat((0.9, 0.95, 1.0), (3.0, 3.0, 3.5), 0.8)),
        "shaft_thrust": sim_utils.CylinderCfg(radius=0.06, height=1.0, visual_material=mat((1.0, 0.75, 0.1), (1.5, 1.0, 0.1))),
        "head_thrust": sim_utils.ConeCfg(radius=0.16, height=0.4, visual_material=mat((1.0, 0.75, 0.1), (1.5, 1.0, 0.1))),
        "shaft_net": sim_utils.CylinderCfg(radius=0.06, height=1.0, visual_material=mat((0.9, 0.2, 0.9), (1.2, 0.2, 1.2))),
        "head_net": sim_utils.ConeCfg(radius=0.16, height=0.4, visual_material=mat((0.9, 0.2, 0.9), (1.2, 0.2, 1.2))),
        "shaft_wind": sim_utils.CylinderCfg(radius=0.06, height=1.0, visual_material=mat((0.2, 0.8, 1.0), (0.2, 1.0, 1.4))),
        "head_wind": sim_utils.ConeCfg(radius=0.16, height=0.4, visual_material=mat((0.2, 0.8, 1.0), (0.2, 1.0, 1.4))),
        "pad": sim_utils.CylinderCfg(radius=2.0, height=0.02, visual_material=mat((0.9, 0.9, 0.2), (0.3, 0.3, 0.0), 0.6)),
        "rcs_failed": sim_utils.SphereCfg(radius=0.14, visual_material=mat((1.0, 0.05, 0.05), (3.0, 0.0, 0.0))),
    })


MARKER_IDS = {k: i for i, k in enumerate(["plume_core", "plume_outer", "rcs", "shaft_thrust", "head_thrust", "shaft_net",
                                          "head_net", "shaft_wind", "head_wind", "pad", "rcs_failed"])}


def plume_shape(thr_frac: float, pe_pa: float) -> tuple[float, float]:
    """Visual mapping only (no combustion model): length from thrust fraction, spread from the 1D exit-to-ambient
    pressure ratio (under-expanded Pe/Pa > 1 widens, over-expanded < 1 narrows the jet)."""
    length = 0.6 + 6.0 * thr_frac
    spread = float(np.clip(math.sqrt(max(pe_pa, 1e-3)), 0.6, 1.8))
    return length, spread


def main():
    base = Scenario(**yaml.safe_load(open(REPO / "configs/scenarios/landing.yaml")))
    sc = randomized_scenarios(base, 1, args.seed)[0]
    keys = ("com_position_w", "velocity_w", "tilt_axis", "tilt_rad", "omega_b", "disturbance_force_w")
    cfg = VtvlEnvCfg()
    cfg.engine = args.engine
    cfg.__post_init__()
    cfg.terminate_at_touchdown = False
    cfg.eval_scenarios = [{k: asdict(sc)[k] for k in keys}]
    if args.look == "story":
        cfg.sky_color, cfg.sky_intensity, cfg.sun_intensity = (0.30, 0.40, 0.56), 1200.0, 2500.0
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 0.0
    cfg.viewer.origin_type = "world"
    cfg.video_recorder.window_width, cfg.video_recorder.window_height = args.width, args.height
    if args.plume != "cone":
        from rocket_dt.isaac_env.flow_plume import enable_flow
        enable_flow()
    env = VtvlEnv(cfg, render_mode="rgb_array")
    wenv = RslRlVecEnvWrapper(env, clip_actions=None)
    if args.controller == "ppo":
        ck = Path(args.checkpoint)
        agent = json.loads((ck.parent / "manifest.json").read_text())["agent_cfg"]
        runner = OnPolicyRunner(wenv, agent, log_dir=None, device=str(env.device))
        runner.load(str(ck))
        policy = runner.get_inference_policy(device=env.device)
    else:
        policy = PDPolicy(env, args.engine, "ideal1d")
    rcs_ids = [t.id for t in load_vehicle(REPO / "configs/vehicle_v0.yaml").layout.rcs]
    for rid in args.rcs_failed:
        env.act.failed[rcs_ids.index(rid)] = True
        if args.controller == "pd":
            policy.failed[rcs_ids.index(rid)] = True  # fault known to the allocator (simulate.py default)
    failed = env.act.failed.cpu().numpy().astype(bool)
    eng_design = load_engine(REPO / f"configs/engines/{args.engine}.yaml")
    from rocket_dt.propulsion import nozzle1d
    vis = VisualizationMarkers(markers_cfg())
    flow = None
    if args.plume != "cone":
        from rocket_dt.isaac_env.flow_plume import FlowPlume
        ov = {}
        for kv in args.flow_set:
            rel, val = kv.split("=")
            prim_rel, attr = rel.rsplit(":", 1)
            ov.setdefault(prim_rel, {})[attr] = float(val)
        flow = FlowPlume(sim_utils.get_current_stage(), REPO / "runs/ppo_win-isaac/mission" / args.name,
                         fps=50 / args.frame_every, overrides=ov, speed0=args.flow_speed[0], speed_gain=args.flow_speed[1])
    from isaacsim.core.rendering_manager import ViewportManager

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
    rcs_pos = env.act.pos_b.cpu().numpy().astype(float)
    rcs_dir = env.act.dir_b.cpu().numpy().astype(float)
    f_ref = float(env.map.lookup(torch.tensor([1.0], device=env.device), torch.tensor([101325.0], device=env.device),
                                 torch.zeros(1, dtype=torch.long, device=env.device))[0])
    out = REPO / "runs/ppo_win-isaac/mission" / args.name
    out.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio
    writer = imageio.get_writer(out / "raw.mp4", fps=50 // args.frame_every, codec="libx264", quality=7, macro_block_size=8)
    cam2 = None
    if args.engine_cam:
        import omni.replicator.core as rep
        from pxr import UsdGeom
        UsdGeom.Camera.Define(sim_utils.get_current_stage(), "/World/EngineCam").CreateFocalLengthAttr(18.0)
        rp2 = rep.create.render_product("/World/EngineCam", (640, 360))
        cam2 = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        cam2.attach([rp2])
        writer2 = imageio.get_writer(out / "raw_engine.mp4", fps=50 // args.frame_every, codec="libx264", quality=7,
                                     macro_block_size=8)
        n2 = 0
    tel = {"label": "win-isaac", "plume": args.plume, "flow_tuning": flow.tuning if flow else None, "controller": args.controller, "checkpoint": args.checkpoint, "engine": args.engine,
           "seed": args.seed, "scenario": cfg.eval_scenarios[0], "map_hash": env.map_hash, "control_dt": env.step_dt,
           "frame_every": args.frame_every, "arrow_scale_m_per_n": ARROW_SCALE, "thrust_ref_n_sl_full": f_ref,
           "mass_kg_fixed": env.mass, "td_com_z": env.td_com_z, "rcs_failed": [r for r, f in zip(rcs_ids, failed) if f],
           "plume_mapping": "length = 0.6 + 6.0 * F/F_ref [m]; spread = clip(sqrt(Pe/Pa), 0.6, 1.8); Pe from "
                            "rocket_dt nozzle1d at logged throttle and ISA Pa. Visual representation of the 1D "
                            "nozzle output, not a combustion simulation.",
           "steps": []}
    rng = np.random.default_rng(0)
    obs = wenv.get_observations()
    k, n_frames = 0, 0
    while k < args.max_steps:
        with torch.inference_mode():
            act = policy(obs)
        obs, _, _, _ = wenv.step(act)
        k += 1
        if env.episode_results:  # this step ended the episode and the env auto-reset: state is no longer ours
            break
        p, q, v, w = (t[0].detach().cpu().numpy().astype(float) for t in env._state())
        rot = _rotmat_xyzw(torch.tensor(q[None], dtype=torch.float32))[0].numpy().astype(float)
        f_main = float(env.last_f_main[0])
        gim = env.last_gimbal[0].cpu().numpy().astype(float)
        frac = env.last_rcs_frac[0].cpu().numpy().astype(float)
        F_b = env.last_F_b[0].cpu().numpy().astype(float)
        wind = env.wind_w[0].cpu().numpy().astype(float)
        thrust_w = rot @ F_b
        net_w = thrust_w + wind + np.array([0, 0, -env.mass * G0])
        step = {"k": k, "t": float(env.t[0]),
                "p": p.tolist(), "v": v.tolist(), "q_xyzw": q.tolist(), "w_b": w.tolist(),
                "tilt_deg": math.degrees(math.acos(max(-1.0, min(1.0, rot[2, 2])))),
                "throttle": float(env.act.throttle[0]), "engine_on": bool(env.act.engine_on[0]),
                "gimbal_rad": gim.tolist(), "rcs_frac": frac.tolist(), "f_main_n": f_main,
                "map_valid": bool(env.last_valid[0]), "pa": float(env.last_pa[0]),
                "m_prop": float(env.m_prop[0]),
                "thrust_w": thrust_w.tolist(), "wind_w": wind.tolist(), "net_w": net_w.tolist(),
                "touchdown": bool(env.td[0]), "frame": None}
        # ---- markers (world frame; env origin is the world origin with num_envs = 1)
        com_w = p
        d_b = np.array([math.cos(gim[0]) * math.sin(gim[1]), -math.sin(gim[0]), math.cos(gim[0]) * math.cos(gim[1])])
        exit_b = pivot + d_b * (exit_z - pivot[2])
        exit_w = com_w + rot @ (exit_b - com_b)
        d_w = rot @ d_b
        thr_frac = max(f_main, 0.0) / f_ref
        pt = nozzle1d.evaluate(eng_design, max(step["throttle"], eng_design.throttle_min), step["pa"])
        pe_pa = pt.exit_pressure_pa / step["pa"]
        step["pe_pa"] = float(pe_pa)
        L0, spread = plume_shape(thr_frac, pe_pa)
        step["plume_len_m"], step["plume_spread"] = (L0 if f_main > 1.0 else 0.0), spread
        L = L0 * (1.0 + 0.04 * rng.standard_normal()) if f_main > 1.0 else 1e-3
        trans, quats, scales, ids = [], [], [], []
        if flow is not None:
            flow.update(exit_w, d_w, L0 if f_main > 1.0 else 0.0, spread, thr_frac, step["engine_on"])
        for name, lf, wf in (("plume_core", 0.65, (0.7 + 0.5 * thr_frac) * spread),
                             ("plume_outer", 1.0, (0.8 + 0.6 * thr_frac) * spread)):
            h = L * lf if args.plume != "flow" else 1e-3
            trans.append(exit_w - d_w * h / 2)
            quats.append(quat_z_to(d_w))
            scales.append([wf, wf, h])
            ids.append(MARKER_IDS[name])
        for i in range(4):
            h = 0.9 if (frac[i] > 0 and not failed[i]) else 1e-3  # only thrusters that are actually firing
            fd = rot @ rcs_dir[i]
            pos = com_w + rot @ (rcs_pos[i] - com_b)
            trans.append(pos - fd * h / 2)
            quats.append(quat_z_to(fd))
            scales.append([1.0, 1.0, h])
            ids.append(MARKER_IDS["rcs"])
        for name, vec in (("thrust", thrust_w), ("net", net_w), ("wind", wind)):
            length = float(np.linalg.norm(vec)) * ARROW_SCALE[name]
            u = vec / max(np.linalg.norm(vec), 1e-9)
            if length < 0.05:
                length, u = 1e-3, np.array([0.0, 0.0, 1.0])
            trans.append(com_w + u * length / 2)
            quats.append(quat_z_to(u))
            scales.append([1.0, 1.0, length])
            ids.append(MARKER_IDS[f"shaft_{name}"])
            trans.append(com_w + u * (length + 0.2))
            quats.append(quat_z_to(u))
            scales.append([1.0, 1.0, 1.0 if length > 0.01 else 1e-3])
            ids.append(MARKER_IDS[f"head_{name}"])
        for i in np.flatnonzero(failed):
            trans.append(com_w + rot @ (rcs_pos[i] - com_b))
            quats.append(np.array([0.0, 0.0, 0.0, 1.0]))
            scales.append([1.0, 1.0, 1.0])
            ids.append(MARKER_IDS["rcs_failed"])
        trans.append(np.array([0.0, 0.0, 0.011]))
        quats.append(np.array([0.0, 0.0, 0.0, 1.0]))
        scales.append([1.0, 1.0, 1.0])
        ids.append(MARKER_IDS["pad"])
        vis.visualize(translations=torch.tensor(np.array(trans), dtype=torch.float32),
                      orientations=torch.tensor(np.array(quats), dtype=torch.float32),
                      scales=torch.tensor(np.array(scales), dtype=torch.float32), marker_indices=ids)
        if k % args.frame_every == 0:
            az = math.radians(args.cam_az_deg)
            eye = com_w + np.array([args.cam_dist * math.cos(az), args.cam_dist * math.sin(az), 4.0])
            eye[2] = max(eye[2], 2.0)
            target = com_w + np.array([0.0, 0.0, -0.5])
            ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=eye.tolist(), target=target.tolist())
            if cam2 is not None:  # side view of the nozzle, 7 m out, slightly above the exit plane
                side = np.array([math.cos(az + 1.2), math.sin(az + 1.2), 0.0])
                noz_mid = exit_w + d_w * 0.35  # centre the scaled nozzle, flame in the lower part of the frame
                e2 = noz_mid + side * 5.0 + np.array([0.0, 0.0, 0.4])
                e2[2] = max(e2[2], 0.6)
                ViewportManager.set_camera_view("/World/EngineCam", eye=e2.tolist(), target=(noz_mid - d_w * 0.8).tolist())
            fr = env.render()
            if fr is not None and np.asarray(fr).size:
                writer.append_data(np.asarray(fr)[..., :3])
                step["frame"] = n_frames
                n_frames += 1
                if cam2 is not None:
                    im2 = np.asarray(cam2.get_data())
                    if im2.size:
                        writer2.append_data(im2.reshape(360, 640, -1)[..., :3])
                        step["frame_engine"] = n2
                        n2 += 1
        tel["steps"].append(step)
    writer.close()
    if cam2 is not None:
        writer2.close()
    tel["episode"] = env.episode_results[0] if env.episode_results else None
    tel["n_frames"] = n_frames
    (out / "telemetry.json").write_text(json.dumps(tel))
    print("MISSION_DONE", out, n_frames, json.dumps(tel["episode"]))
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
