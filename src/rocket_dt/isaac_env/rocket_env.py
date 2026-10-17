"""Isaac Lab 3.0 DirectRLEnv for the VTVL landing task (label `win-isaac`).

Written against the installed Isaac Lab 3.0.0 (v3.0.0-beta2.patch1) + Isaac Sim 6.0.1, PhysX backend.

Single-integrator rule: PhysX integrates the rigid body. Propulsion is the repository's own model
mirrored in torch (torch_models.py): actuator lag/limits -> EnginePerformanceMap bilinear lookup ->
point forces -> wrench_bridge (sum F, tau about CoM) -> permanent_wrench_composer (body frame,
positions=None). Wind is a constant world-frame force at the CoM.

Conventions that differ from rocket_dt (checked in scripts/isaac_force_smoke.py):
- Isaac root quaternions are (x, y, z, w); rocket_dt uses (w, x, y, z).
- The Isaac root pose is the body origin (link frame); rocket_dt states are at the CoM.

ADR-0009: mass, CoM and inertia are authored in the vehicle USD at the full-propellant value and
held fixed for the episode; propellant is integrated only for observations, reward and fuel metrics.
Contact is PhysX rigid contact on four foot spheres, not the reference penalty spring.
Landing success uses evaluation/simulate.py LANDING_LIMITS (ADR-0005, provisional).
"""
from __future__ import annotations

import math
from dataclasses import MISSING
from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from rocket_dt.dynamics.frames import G0
from rocket_dt.dynamics.mass import MassModel
from rocket_dt.evaluation.simulate import LANDING_LIMITS
from rocket_dt.isaac_env.torch_models import TorchActuators, TorchPerfMap, gimbal_direction, isa_pressure
from rocket_dt.isaac_env.wrench_bridge import batched_wrench_torch
from rocket_dt.propulsion.performance_map import EnginePerformanceMap, build_ideal_map
from rocket_dt.schemas import load_engine, load_vehicle

REPO = Path(__file__).resolve().parents[3]

# Observation contract (design doc C2), fixed order, SI units, body frame unless noted.
OBSERVATION_SCHEMA_V1 = (
    ("relative_pad_position_body", 3), ("linear_velocity_body", 3), ("gravity_vector_body", 3),
    ("angular_velocity_body", 3), ("altitude", 1), ("vertical_velocity", 1), ("normalized_fuel", 1),
    ("engine_available_thrust_norm", 1), ("throttle_actual", 1), ("gimbal_actual", 2), ("previous_action", 7),
)
# Action contract: [throttle, gimbal_x, gimbal_y] in [-1, 1] + 4 RCS duty in [0, 1] (layout gimbal2_rcs4_roll_v0).
ACTION_SCHEMA_V1 = (("throttle", 1), ("gimbal", 2), ("rcs_duty", 4))
OBS_DIM = sum(n for _, n in OBSERVATION_SCHEMA_V1)
ACT_DIM = sum(n for _, n in ACTION_SCHEMA_V1)
# Fixed observation scaling (applied after the schema values are assembled)
OBS_SCALE = (1 / 10,) * 3 + (1 / 5,) * 3 + (1.0,) * 3 + (1.0,) * 3 + (1 / 50, 1 / 5, 1.0, 1.0, 1.0, 1.0, 1.0) + (1.0,) * 7


def build_vehicle_usd(path: Path, engine_name: str) -> dict:
    """Write a one-body USD: RigidBody + MassAPI (full propellant) + visual hull + 4 foot sphere colliders."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    veh = load_vehicle(REPO / "configs/vehicle_v0.yaml").with_engine(engine_name)
    eng = load_engine(REPO / f"configs/engines/{engine_name}.yaml")
    mm = MassModel(veh, eng)
    ms = mm.evaluate(mm.prop0, mm.rcs0)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Vehicle")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass.CreateMassAttr().Set(float(ms.mass_kg))
    mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(*map(float, ms.com_body_m)))
    mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*map(float, np.diag(ms.inertia_com_body))))
    mass.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    hull = UsdGeom.Cylinder.Define(stage, "/Vehicle/hull")
    hull.CreateRadiusAttr(veh.mass.structure_radius_m)
    hull.CreateHeightAttr(veh.mass.structure_length_m)
    hull.CreateAxisAttr("Z")
    hull.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.85, 0.9)])
    noz = UsdGeom.Cone.Define(stage, "/Vehicle/nozzle")
    noz.CreateRadiusAttr(0.35)
    noz.CreateHeightAttr(0.7)
    noz.CreateAxisAttr("Z")
    noz.AddTranslateOp().Set(Gf.Vec3d(0, 0, veh.layout.main_engine.gimbal_pivot_body_m[2] - 0.2))
    noz.CreateDisplayColorAttr([Gf.Vec3f(0.3, 0.3, 0.3)])
    r_foot = 0.15
    for k, f in enumerate(veh.gear.foot_positions_body_m):
        s = UsdGeom.Sphere.Define(stage, f"/Vehicle/foot_{k}")
        s.CreateRadiusAttr(r_foot)
        s.AddTranslateOp().Set(Gf.Vec3d(f[0], f[1], f[2] + r_foot))  # sphere bottom at the foot point
        s.CreateDisplayColorAttr([Gf.Vec3f(0.2, 0.2, 0.2)])
        UsdPhysics.CollisionAPI.Apply(s.GetPrim())
        leg = UsdGeom.Cylinder.Define(stage, f"/Vehicle/leg_{k}")  # visual strut only
        leg.CreateRadiusAttr(0.05)
        leg.CreateHeightAttr(float(np.linalg.norm([f[0], f[1], 1.5])))
        mid = Gf.Vec3d(f[0] / 2, f[1] / 2, f[2] + 0.75)
        leg.AddTranslateOp().Set(mid)
        rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(f[0] / 2, f[1] / 2, -1.5))
        leg.AddOrientOp().Set(Gf.Quatf(rot.GetQuat()))
    stage.GetRootLayer().Save()
    return {"mass_kg": float(ms.mass_kg), "com_body_m": ms.com_body_m.tolist(),
            "inertia_diag": np.diag(ms.inertia_com_body).tolist(), "foot_radius_m": r_foot,
            "propellant_kg": float(mm.prop0), "rcs_gas_kg": float(mm.rcs0)}


@configclass
class VtvlEnvCfg(DirectRLEnvCfg):
    # control 50 Hz on 200 Hz physics (matches evaluation/simulate.py Scenario defaults)
    decimation = 4
    episode_length_s = 60.0  # configs/scenarios/landing.yaml duration_s
    action_space = ACT_DIM
    observation_space = OBS_DIM
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200, render_interval=4, gravity=(0.0, 0.0, -G0),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.6, restitution=0.0))
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", collision_group=-1,
                                 physics_material=sim_utils.RigidBodyMaterialCfg(
                                     static_friction=0.6, dynamic_friction=0.6, restitution=0.0), debug_vis=False)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=40.0, replicate_physics=True)
    viewer: ViewerCfg = ViewerCfg(eye=(35.0, 35.0, 30.0), lookat=(0.0, 0.0, 15.0), origin_type="env", env_index=0,
                                  resolution=(960, 540))
    robot: RigidObjectCfg = MISSING  # filled in __post_init__ from the generated USD

    engine: str = "conical_A"
    map_source: str = "ideal1d"  # ideal1d | cfd (runs/cfd/maps/<engine>_cfd.json, cloud-v1912)
    asset_dir: str = str(REPO / "runs/ppo_win-isaac/assets")
    throttle0: float = 0.7  # evaluation/simulate.py ActuatorModel(throttle0=0.7)

    # episode logic
    terminate_at_touchdown: bool = True  # training: end at first contact; evaluation: False + settle
    settle_s: float = 3.0  # Scenario.settle_s
    touchdown_height_m: float = 0.02  # foot-bottom height that counts as first contact (pre-impact state)
    max_tilt_rad: float = math.radians(60)  # simulate.py tip-over termination
    oob_alt_m: float = 150.0
    oob_horizontal_m: float = 40.0

    # randomized reset = evaluation.experiments.randomized_scenarios distribution
    eval_scenarios: list | None = None  # list of dicts (com_position_w, velocity_w, ...) one per env

    # reward (see docs/decisions/ADR-0009)
    rew_shaping_scale: float = 2.0
    phi_pos_m: float = 4.0  # potential divisor for horizontal position error [m]
    # "basic": penalise |v_h| and tilt directly (r0-r2). "guided": penalise |v_h - v_h_ref(e_h)| with the PD
    # baseline's horizontal law v_h_ref = -(kp/kd) e_h (0.6/1.4), capped at 2 m/s, and a softer tilt weight.
    shaping: str = "basic"
    phi_tilt_rad: float = 0.1
    rew_fuel_per_throttle: float = 0.004
    rew_action_rate: float = 0.01
    rew_td_base: float = 5.0
    rew_td_success: float = 10.0
    rew_crash: float = -10.0
    rew_timeout: float = -10.0

    def __post_init__(self):
        usd = Path(self.asset_dir) / f"vtvl_v0_{self.engine}.usda"
        self.robot = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Vehicle",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(usd),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    linear_damping=0.0, angular_damping=0.0, max_linear_velocity=1e3, max_angular_velocity=1e5,
                    enable_gyroscopic_forces=True, max_depenetration_velocity=2.0)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 60.0)))


def _quat_axis_angle_xyzw(axis: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    s = torch.sin(ang / 2)[:, None]
    return torch.cat([axis * s, torch.cos(ang / 2)[:, None]], dim=-1)


def _rotmat_xyzw(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1).reshape(-1, 3, 3)


class VtvlEnv(DirectRLEnv):
    cfg: VtvlEnvCfg

    def __init__(self, cfg: VtvlEnvCfg, render_mode: str | None = None, **kwargs):
        self.asset_info = build_vehicle_usd(Path(cfg.robot.spawn.usd_path), cfg.engine)
        super().__init__(cfg, render_mode, **kwargs)
        if getattr(self, "_physics_handles_decimation", False):
            raise NotImplementedError("per-substep actuator/propulsion update requires physics-side decimation off")
        n, dev = self.num_envs, self.device
        f = dict(dtype=torch.float32, device=dev)
        self.veh = load_vehicle(REPO / "configs/vehicle_v0.yaml").with_engine(cfg.engine)
        self.eng = load_engine(REPO / f"configs/engines/{cfg.engine}.yaml")
        mm = MassModel(self.veh, self.eng)
        ms = mm.evaluate(mm.prop0, mm.rcs0)
        if cfg.map_source == "ideal1d":
            pm = build_ideal_map(self.eng)
        else:
            pm = EnginePerformanceMap.load(REPO / f"runs/cfd/maps/{cfg.engine}_cfd.json")
        self.map_hash = pm.content_hash()
        self.map = TorchPerfMap(pm, dev)
        self.act = TorchActuators(self.veh, self.eng, n, self.step_dt, dev)
        self.mass = float(ms.mass_kg)
        self.com_b = torch.tensor(ms.com_body_m, **f)
        self.pivot_b = torch.tensor(self.veh.layout.main_engine.gimbal_pivot_body_m, **f)
        feet = torch.tensor(self.veh.gear.foot_positions_body_m, **f).reshape(-1, 3)
        self.feet_b = feet
        self.td_com_z = float(ms.com_body_m[2] - feet[:, 2].min())  # CoM height when standing
        self.prop0, self.rcs0 = float(mm.prop0), float(mm.rcs0)
        self.gmax = self.veh.layout.main_engine.max_gimbal_rad
        self._target = torch.zeros(n, 3, **f)  # pad at env origin
        z = lambda *s: torch.zeros(n, *s, **f)  # noqa: E731
        self.m_prop, self.m_rcs, self.t, self.wind_w = z(), z(), z(), z(3)
        self.actions, self.prev_actions = z(ACT_DIM), z(ACT_DIM)
        self.td = torch.zeros(n, dtype=torch.bool, device=dev)
        self.td_t, self.td_vz, self.td_vh, self.td_tilt, self.td_pos = z(), z(), z(), z(), z()
        self.invalid_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.fuel_used = z()
        self.phi = z()
        self.max_tilt = z()
        self.obs_scale = torch.tensor(OBS_SCALE, **f)
        self.episode_results: list[dict] = []  # filled at reset (evaluation reads it)
        self._first_episode_done = torch.zeros(n, dtype=torch.bool, device=dev)
        self._episode_count = torch.zeros(n, dtype=torch.long, device=dev)

    # ---------------------------------------------------------------- scene
    def _setup_scene(self):
        self.robot = RigidObject(self.cfg.robot)
        self.scene.rigid_objects["robot"] = self.robot
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
        light.func("/World/Light", light)

    # ---------------------------------------------------------------- state helpers
    def _state(self):
        d = self.robot.data
        p = d.root_com_pos_w.torch - self.scene.env_origins
        q = d.root_link_quat_w.torch  # xyzw
        v = d.root_com_lin_vel_w.torch
        w_b = d.root_com_ang_vel_b.torch
        return p, q, v, w_b

    @staticmethod
    def _tilt(rot):
        return torch.acos(rot[:, 2, 2].clamp(-1.0, 1.0))

    def _foot_bottom_z(self, p, rot):
        r = self.feet_b[None] - self.com_b[None, None]  # (1, 4, 3)
        return p[:, None, 2] + (rot[:, None] @ r[..., None]).squeeze(-1)[..., 2]

    # ---------------------------------------------------------------- actions
    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone().clamp(-1.0, 1.0)
        a = self.actions
        tmin, tmax = self.act.tmin, self.act.tmax
        throttle = tmin + (a[:, 0] + 1.0) * 0.5 * (tmax - tmin)
        gimbal = a[:, 1:3] * self.gmax
        rcs_on = (a[:, 3:7] + 1.0) * 0.5 * self.step_dt
        self.act.command(self.t, throttle, gimbal, rcs_on)

    def _apply_action(self):
        dt = self.physics_dt
        p, q, v, w_b = self._state()
        rot = _rotmat_xyzw(q)
        # first contact: record the pre-impact state and cut the engine (simulate.py rule)
        new_td = (~self.td) & (self._foot_bottom_z(p, rot).min(dim=1).values <= self.cfg.touchdown_height_m)
        if new_td.any():
            self.td |= new_td
            self.td_t = torch.where(new_td, self.t, self.td_t)
            self.td_vz = torch.where(new_td, v[:, 2], self.td_vz)
            self.td_vh = torch.where(new_td, v[:, :2].norm(dim=1), self.td_vh)
            self.td_tilt = torch.where(new_td, self._tilt(rot), self.td_tilt)
            self.td_pos = torch.where(new_td, (p[:, :2] - self._target[:, :2]).norm(dim=1), self.td_pos)
            self.act.engine_on &= ~new_td
        thr, gim, frac = self.act.step(self.t, dt)
        pa = isa_pressure(p[:, 2].clamp(min=0.0))
        f_main, mdot_main, valid = self.map.lookup(thr, pa)
        main_on = self.act.engine_on & (self.m_prop > 0)
        f_main = torch.where(main_on, f_main, torch.zeros_like(f_main))
        mdot_main = torch.where(main_on, mdot_main, torch.zeros_like(mdot_main))
        self.invalid_steps += (main_on & ~valid).long()
        rcs_frac = torch.where((self.m_rcs > 0)[:, None], frac, torch.zeros_like(frac))
        n = self.num_envs
        force_b = torch.cat([(f_main[:, None] * gimbal_direction(gim))[:, None],
                             rcs_frac[..., None] * (self.act.fmax[:, None] * self.act.dir_b)[None]], dim=1)
        pos_b = torch.cat([self.pivot_b.expand(n, 1, 3), self.act.pos_b.expand(n, -1, 3)], dim=1)
        F, tau = batched_wrench_torch(force_b, pos_b, self.com_b.expand(n, 3))
        comp = self.robot.permanent_wrench_composer
        comp.set_forces_and_torques_index(forces=F, torques=tau, positions=None, is_global=False)
        comp.add_forces_and_torques_index(forces=self.wind_w[:, None], positions=None, is_global=True)
        self.m_prop = (self.m_prop - mdot_main * dt).clamp(min=0.0)
        self.m_rcs = (self.m_rcs - (rcs_frac * self.act.mdot_full).sum(1) * dt).clamp(min=0.0)
        self.fuel_used += mdot_main * dt
        self.t += dt

    # ---------------------------------------------------------------- obs / reward / dones
    def _available_thrust_norm(self, p):
        fmax, _, _ = self.map.lookup(torch.full_like(p[:, 2], self.act.tmax), isa_pressure(p[:, 2].clamp(min=0.0)))
        return fmax / (self.mass * G0)

    def _get_observations(self) -> dict:
        p, q, v, w_b = self._state()
        rot = _rotmat_xyzw(q)
        rt = rot.transpose(1, 2)
        rel = (rt @ (self._target - p)[..., None]).squeeze(-1)
        v_b = (rt @ v[..., None]).squeeze(-1)
        g_b = (rt @ torch.tensor([0.0, 0.0, -1.0], device=self.device).expand_as(p)[..., None]).squeeze(-1)
        obs = torch.cat([rel, v_b, g_b, w_b, p[:, 2:3], v[:, 2:3], (self.m_prop / self.prop0)[:, None],
                         self._available_thrust_norm(p)[:, None],
                         ((self.act.throttle - self.act.tmin) / (self.act.tmax - self.act.tmin))[:, None],
                         self.act.gimbal / self.gmax, self.actions], dim=-1) * self.obs_scale
        return {"policy": torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)}

    def _potential(self, p, v, rot, w_b):
        h = (p[:, 2] - self.td_com_z).clamp(min=0.0)
        vz_ref = -(0.3 * h).clamp(0.8, 6.0)  # same descent profile as the PD baseline (shaping only)
        e_h = p[:, :2] - self._target[:, :2]
        if self.cfg.shaping == "guided":
            v_ref = -(0.6 / 1.4) * e_h
            v_ref = v_ref * (2.0 / v_ref.norm(dim=1, keepdim=True).clamp(min=2.0))
            v_h_term = (v[:, :2] - v_ref).norm(dim=1)
        else:
            v_h_term = v[:, :2].norm(dim=1)
        return -(torch.abs(v[:, 2] - vz_ref) / 2.0 + e_h.norm(dim=1) / self.cfg.phi_pos_m
                 + v_h_term / 1.0 + self._tilt(rot) / self.cfg.phi_tilt_rad + w_b.norm(dim=1) / 0.5)

    def _get_dones(self):
        p, q, v, w_b = self._state()
        rot = _rotmat_xyzw(q)
        tilt = self._tilt(rot)
        self.max_tilt = torch.maximum(self.max_tilt, tilt)
        self._tip = tilt > self.cfg.max_tilt_rad
        self._oob = (p[:, 2] > self.cfg.oob_alt_m) | (p[:, :2].norm(dim=1) > self.cfg.oob_horizontal_m) | (p[:, 2] < -1.0)
        if self.cfg.terminate_at_touchdown:
            self._settled = self.td.clone()
        else:
            self._settled = self.td & (self.t - self.td_t >= self.cfg.settle_s)
        self._final_tilt = tilt
        terminated = self._tip | self._oob | self._settled
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self._timeout = time_out & ~terminated
        return terminated, self._timeout

    def _td_ok(self):
        L = LANDING_LIMITS
        return (self.td & (self.td_vz.abs() <= L["vz_max_m_s"]) & (self.td_vh <= L["vh_max_m_s"])
                & (self.td_tilt <= L["tilt_max_rad"]) & (self.td_pos <= L["pos_err_max_m"]))

    def _success(self):
        ok = self._td_ok() & self._settled & ~self._tip
        if not self.cfg.terminate_at_touchdown:
            ok &= self._final_tilt <= LANDING_LIMITS["final_tilt_max_rad"]
        return ok

    def _get_rewards(self) -> torch.Tensor:
        p, q, v, w_b = self._state()
        rot = _rotmat_xyzw(q)
        phi = self._potential(p, v, rot, w_b)
        airborne_prev = (self.episode_length_buf > 1)
        r = torch.where(airborne_prev, self.cfg.rew_shaping_scale * (phi - self.phi), torch.zeros_like(phi))
        if not self.cfg.terminate_at_touchdown:  # no shaping while the vehicle settles after cutoff
            r = torch.where(self.td & ~self._settled, torch.zeros_like(r), r)
        self.phi = phi
        r -= self.cfg.rew_fuel_per_throttle * self.act.throttle * self.act.engine_on.float()
        r -= self.cfg.rew_action_rate * (self.actions - self.prev_actions).pow(2).sum(1)
        L = LANDING_LIMITS
        td_quality = (self.cfg.rew_td_base
                      - 2.0 * (self.td_vz.abs() - L["vz_max_m_s"]).clamp(min=0.0)
                      - 2.0 * (self.td_vh - L["vh_max_m_s"]).clamp(min=0.0)
                      - 1.0 * (self.td_pos - L["pos_err_max_m"]).clamp(min=0.0)
                      - 20.0 * (self.td_tilt - L["tilt_max_rad"]).clamp(min=0.0)
                      - 0.5 * self.td_vz.abs() - 0.5 * self.td_pos)
        succ = self._success()
        r += torch.where(self._settled & ~self._tip, td_quality + self.cfg.rew_td_success * succ.float(), torch.zeros_like(r))
        r += torch.where(self._tip | self._oob, torch.full_like(r, self.cfg.rew_crash), torch.zeros_like(r))
        r += torch.where(self._timeout, torch.full_like(r, self.cfg.rew_timeout), torch.zeros_like(r))
        self._last_success = succ
        return r

    # ---------------------------------------------------------------- reset
    def _episode_record(self, ids):
        succ = self._last_success[ids] if hasattr(self, "_last_success") else torch.zeros(len(ids), dtype=torch.bool,
                                                                                           device=self.device)
        reason = []
        for k, i in enumerate(ids.tolist()):
            if bool(self._tip[i]):
                reason.append("tip_over_or_loss_of_control")
            elif bool(self._oob[i]):
                reason.append("out_of_bounds")
            elif bool(self._timeout[i]):
                reason.append("timeout")
            elif bool(succ[k]):
                reason.append("landed")
            else:
                reason.append("landing_limits")
        return [{"env": int(i), "scenario_index": int(i + (int(self._episode_count[i]) - 1) * self.num_envs),
                 "success": bool(succ[k]), "termination_reason": reason[k],
                 "touchdown": bool(self.td[i]), "td_vz": float(self.td_vz[i]), "td_vh": float(self.td_vh[i]),
                 "td_tilt_deg": float(torch.rad2deg(self.td_tilt[i])), "td_pos_err": float(self.td_pos[i]),
                 "final_tilt_deg": float(torch.rad2deg(self._final_tilt[i])),
                 "max_tilt_deg": float(torch.rad2deg(self.max_tilt[i])), "fuel_used_kg": float(self.fuel_used[i]),
                 "map_invalid_steps": int(self.invalid_steps[i]), "physics_valid": bool(self.invalid_steps[i] == 0),
                 "duration_s": float(self.t[i])} for k, i in enumerate(ids.tolist())]

    def _reset_idx(self, env_ids):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = env_ids.long()
        if hasattr(self, "_tip") and self.common_step_counter > 0:
            done_ids = env_ids[self.episode_length_buf[env_ids] > 0]
            if len(done_ids):
                if self.cfg.eval_scenarios is not None:
                    self.episode_results += self._episode_record(done_ids)
                self._first_episode_done[done_ids] = True
                succ = self._last_success[done_ids].float()
                tdm = self.td[done_ids]
                self.extras["log"] = {
                    "Metrics/success_rate": succ.mean().item(),
                    "Metrics/touchdown_rate": tdm.float().mean().item(),
                    "Metrics/td_vz_abs_mean": self.td_vz[done_ids][tdm].abs().mean().item() if tdm.any() else float("nan"),
                    "Metrics/td_pos_err_mean": self.td_pos[done_ids][tdm].mean().item() if tdm.any() else float("nan"),
                    "Metrics/fuel_used_mean": self.fuel_used[done_ids].mean().item(),
                    "Episode_Termination/tip_over": self._tip[done_ids].float().mean().item(),
                    "Episode_Termination/oob": self._oob[done_ids].float().mean().item(),
                    "Episode_Termination/timeout": self._timeout[done_ids].float().mean().item(),
                    "Metrics/physics_valid_rate": (self.invalid_steps[done_ids] == 0).float().mean().item()}
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        n = len(env_ids)
        dev = self.device
        if self.cfg.eval_scenarios is not None:
            # env i, episode e -> scenario (i + e * num_envs); evaluation keeps only each env's first episode
            sc = [self.cfg.eval_scenarios[(i + int(self._episode_count[i]) * self.num_envs) % len(self.cfg.eval_scenarios)]
                  for i in env_ids.tolist()]
            self._episode_count[env_ids] += 1
            pos = torch.tensor([s["com_position_w"] for s in sc], device=dev)
            vel = torch.tensor([s["velocity_w"] for s in sc], device=dev)
            axis = torch.tensor([s["tilt_axis"] for s in sc], device=dev)
            ang = torch.tensor([s["tilt_rad"] for s in sc], device=dev)
            w_b = torch.tensor([s["omega_b"] for s in sc], device=dev)
            wind = torch.tensor([s["disturbance_force_w"] for s in sc], device=dev)
        else:  # same distribution as evaluation.experiments.randomized_scenarios (torch RNG, not the seeds)
            u = lambda lo, hi: torch.rand(n, device=dev) * (hi - lo) + lo  # noqa: E731
            a = u(0, 2 * math.pi)
            pos = torch.stack([8 * torch.sqrt(u(0, 1)) * torch.cos(a), 8 * torch.sqrt(u(0, 1)) * torch.sin(a), u(40, 80)], 1)
            vel = torch.stack([u(-1.5, 1.5), u(-1.5, 1.5), u(-8, -4)], 1)
            axis = torch.stack([torch.cos(a + 1), torch.sin(a + 1), torch.zeros(n, device=dev)], 1)
            ang = torch.deg2rad(u(0, 4))
            w_b = torch.stack([torch.zeros(n, device=dev), torch.zeros(n, device=dev), u(-0.2, 0.2)], 1)
            wd, wm = u(0, 2 * math.pi), u(0, 300.0)
            wind = torch.stack([wm * torch.cos(wd), wm * torch.sin(wd), torch.zeros(n, device=dev)], 1)
        q = _quat_axis_angle_xyzw(axis, ang)
        rot = _rotmat_xyzw(q)
        p_w = pos + self.scene.env_origins[env_ids]
        link = p_w - (rot @ self.com_b[None, :, None].expand(n, 3, 1)).squeeze(-1)
        self.robot.write_root_link_pose_to_sim_index(root_pose=torch.cat([link, q], 1), env_ids=env_ids)
        w_w = (rot @ w_b[..., None]).squeeze(-1)
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=torch.cat([vel, w_w], 1), env_ids=env_ids)
        self.wind_w[env_ids] = wind
        self.act.reset(env_ids, self.cfg.throttle0)
        self.m_prop[env_ids] = self.prop0
        self.m_rcs[env_ids] = self.rcs0
        self.t[env_ids] = 0.0
        self.td[env_ids] = False
        for b in (self.td_t, self.td_vz, self.td_vh, self.td_tilt, self.td_pos, self.fuel_used, self.max_tilt):
            b[env_ids] = 0.0
        self.invalid_steps[env_ids] = 0
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.actions[env_ids, 0] = 2 * (self.cfg.throttle0 - self.act.tmin) / (self.act.tmax - self.act.tmin) - 1
        self.actions[env_ids, 3:] = -1.0
        self.phi[env_ids] = 0.0
