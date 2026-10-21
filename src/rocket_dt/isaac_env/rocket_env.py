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
import os
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

from rocket_dt.dynamics.aero import AeroModel, aero_wrench
from rocket_dt.dynamics.frames import G0
from rocket_dt.dynamics.mass import MassModel
from rocket_dt.dynamics.varmass import compensated_wrench
from rocket_dt.evaluation.simulate import LANDING_LIMITS
from rocket_dt.isaac_env.torch_models import TorchActuators, TorchPerfMap, gimbal_direction, isa_pressure
from rocket_dt.isaac_env.layouts import layout_spec
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


NOZZLE_VIS_SCALE = 4.0  # visual only: real exit radius is 79-103 mm on a 1.2 m hull
NOZZLE_COLOR = {"conical_A": (0.45, 0.62, 0.95), "bell_B": (0.95, 0.66, 0.25), "bell_C": (0.95, 0.45, 0.42)}


def build_vehicle_usd(path: Path, engine_name: str, layout: str = "gimbal_rcs", visual: str = "v0") -> dict:
    """Write a one-body USD: RigidBody + MassAPI (full propellant) + visual hull + 4 foot sphere colliders.
    visual "v1": the v0 placeholder geometry is replaced by the assets/vtvl_v1 vehicle (visual only; physics,
    colliders and mass are identical), and the collider spheres are hidden."""
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
    # visual nozzle: the engine's own contour revolved (all engines scaled by the same NOZZLE_VIS_SCALE so their
    # relative size and shape are preserved), tinted per engine; visual only, no collider / mass
    from rocket_dt.propulsion.geometry import build_contour
    c = build_contour(eng)
    xs, rs = c.x_m - c.x_m[c.throat_index], c.r_m
    z_throat = veh.layout.main_engine.gimbal_pivot_body_m[2] - veh.mass.engine_throat_below_pivot_m
    nseg = 36
    pts, faces, counts = [], [], []
    for j in range(len(xs)):
        for k in range(nseg):
            a_ = 2 * np.pi * k / nseg
            pts.append(Gf.Vec3f(float(NOZZLE_VIS_SCALE * rs[j] * np.cos(a_)), float(NOZZLE_VIS_SCALE * rs[j] * np.sin(a_)),
                                float(z_throat - NOZZLE_VIS_SCALE * xs[j])))
    for j in range(len(xs) - 1):
        for k in range(nseg):
            k2 = (k + 1) % nseg
            faces += [j * nseg + k, j * nseg + k2, (j + 1) * nseg + k2, (j + 1) * nseg + k]
            counts.append(4)
    spec = layout_spec(layout, veh)
    pivot_b = np.array(veh.layout.main_engine.gimbal_pivot_body_m, float)
    for e_i, (epos, share) in enumerate(zip(spec.engine_pos_b, spec.engine_share)):
        sc_ = float(np.sqrt(share))  # nozzle area ~ thrust share
        noz = UsdGeom.Mesh.Define(stage, f"/Vehicle/nozzle_{e_i}")
        noz.CreatePointsAttr([Gf.Vec3f(float(epos[0] + sc_ * q[0]), float(epos[1] + sc_ * q[1]),
                                       float(pivot_b[2] + sc_ * (q[2] - pivot_b[2]))) for q in pts])
        noz.CreateFaceVertexIndicesAttr(faces)
        noz.CreateFaceVertexCountsAttr(counts)
        noz.CreateDoubleSidedAttr(True)
        noz.CreateDisplayColorAttr([Gf.Vec3f(*NOZZLE_COLOR.get(engine_name, (0.3, 0.3, 0.3)))])
    for t in spec.rcs[4:]:  # layout d: radial cold-gas thrusters, drawn as small blocks
        blk = UsdGeom.Cube.Define(stage, f"/Vehicle/{t.id}")
        blk.CreateSizeAttr(0.16)
        blk.AddTranslateOp().Set(Gf.Vec3d(*map(float, t.position_body_m)))
        blk.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.3, 0.3)])
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
    if visual == "v1":
        from rocket_dt.isaac_env.visual_assets import author_vehicle_visuals
        for prim in list(stage.GetPrimAtPath("/Vehicle").GetChildren()):
            if prim.GetName().startswith("foot_"):
                UsdGeom.Imageable(prim).MakeInvisible()
            else:
                stage.RemovePrim(prim.GetPath())
        author_vehicle_visuals(stage, "/Vehicle", layout, engine_name)
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

    engine: str = "conical_A"  # also sets the rigid-body mass properties (USD) for every env
    # mixed-engine training: each env samples its thrust map from this list at reset. Mass properties stay
    # those of `engine` (A/B/C differ by <= 1.5 kg, 4 mm CoM, 0.4 % inertia; per-env mass is not settable, ADR-0009)
    engine_mix: list | None = None
    map_source: str = "ideal1d"  # ideal1d | cfd (runs/cfd/maps/<engine>_cfd.json, cloud-v1912)
    asset_dir: str = str(REPO / "runs/ppo_win-isaac/assets")
    throttle0: float = 0.7  # evaluation/simulate.py ActuatorModel(throttle0=0.7)
    # look only (render): dome light doubles as the sky colour
    sky_color: tuple = (0.8, 0.8, 0.8)
    sky_intensity: float = 2500.0
    sun_intensity: float = 0.0

    # engine layout (layouts.py) and scenario (ADR-0010); defaults reproduce the nominal v0 env exactly
    layout: str = "gimbal_rcs"
    scenario: str = "nominal"  # nominal | wind | rcs_fail | noise | lateral
    wind_max_n: float = 300.0  # nominal: randomized_scenarios; "wind" scenario raises it
    gust_amp: float = 0.0  # relative gust amplitude (sinusoid), "wind" scenario
    noise_pos_m: float = 0.0  # "noise" scenario: measurement noise (1 sigma) and 1-step latency
    noise_vel_m_s: float = 0.0
    noise_att_rad: float = 0.0
    noise_rate_rad_s: float = 0.0
    obs_delay_steps: int = 0
    lateral_r_m: tuple = (0.0, 8.0)  # start radius range; "lateral" scenario uses (25, 45)
    # physics v1 (ADR-0011); defaults reproduce v0. aero: quasi-steady aero (configs/aero_v0.yaml) and the wind
    # becomes a wind velocity at the reference height (the v0 force maps to the speed giving that force);
    # variable_mass: true mass/CoM/inertia from the burned propellant via a compensating wrench on the fixed body
    aero: bool = False
    variable_mass: bool = False
    # look only: "v1" = assets/vtvl_v1 vehicle, concrete pad + lakebed per env, baked sky dome + sun
    visual: str = "v0"

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
    rew_td_vz_target: float = 1.2  # touchdown |vz| above this is penalised (margin below the 2.0 m/s limit)
    rew_td_vz_margin: float = 0.0  # penalty per m/s above the target (0 = off, r0-r3)
    rew_crash: float = -10.0
    rew_timeout: float = -10.0

    def apply_scenario(self):
        """Set the scenario knobs from `scenario` (call after changing it)."""
        sc = self.scenario
        if sc == "wind":
            self.wind_max_n, self.gust_amp = 1500.0, 0.4
        elif sc == "noise":
            self.noise_pos_m, self.noise_vel_m_s, self.noise_att_rad, self.noise_rate_rad_s = 0.3, 0.15, 0.009, 0.02
            self.obs_delay_steps = 1
        elif sc == "lateral":
            self.lateral_r_m, self.oob_horizontal_m = (25.0, 45.0), 80.0
        elif sc not in ("nominal", "rcs_fail"):
            raise ValueError(f"unknown scenario {sc}")

    def __post_init__(self):
        from rocket_dt.isaac_env.layouts import layout_spec as _ls
        spec = _ls(self.layout, load_vehicle(REPO / "configs/vehicle_v0.yaml").with_engine(self.engine))
        self.action_space, self.observation_space = spec.act_dim, spec.obs_dim
        tag = "" if self.layout == "gimbal_rcs" else f"_{self.layout}"
        usd = Path(self.asset_dir) / f"vtvl_v0_{self.engine}{tag}.usda"
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


def _quat_mul_xyzw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], dim=-1)


def _rotmat_xyzw(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1).reshape(-1, 3, 3)


class VtvlEnv(DirectRLEnv):
    cfg: VtvlEnvCfg

    def __init__(self, cfg: VtvlEnvCfg, render_mode: str | None = None, **kwargs):
        # one USD per process: concurrent Isaac processes rewriting a shared file died silently at startup
        usd = Path(cfg.robot.spawn.usd_path)
        cfg.robot.spawn.usd_path = str(usd.with_name(f"{usd.stem}_{os.getpid()}{usd.suffix}"))
        self.asset_info = build_vehicle_usd(Path(cfg.robot.spawn.usd_path), cfg.engine, cfg.layout, cfg.visual)
        super().__init__(cfg, render_mode, **kwargs)
        if getattr(self, "_physics_handles_decimation", False):
            raise NotImplementedError("per-substep actuator/propulsion update requires physics-side decimation off")
        n, dev = self.num_envs, self.device
        f = dict(dtype=torch.float32, device=dev)
        self.veh = load_vehicle(REPO / "configs/vehicle_v0.yaml").with_engine(cfg.engine)
        self.eng = load_engine(REPO / f"configs/engines/{cfg.engine}.yaml")
        mm = MassModel(self.veh, self.eng)
        ms = mm.evaluate(mm.prop0, mm.rcs0)
        self.engine_names = list(cfg.engine_mix) if cfg.engine_mix else [cfg.engine]
        pms = []
        for name in self.engine_names:
            if cfg.map_source == "ideal1d":
                pms.append(build_ideal_map(load_engine(REPO / f"configs/engines/{name}.yaml")))
            else:
                pms.append(EnginePerformanceMap.load(REPO / f"runs/cfd/maps/{name}_cfd.json"))
        self.map_hashes = {name: pm.content_hash() for name, pm in zip(self.engine_names, pms)}
        self.map_hash = self.map_hashes[self.engine_names[0]] if len(pms) == 1 else self.map_hashes
        self.map = TorchPerfMap(pms, dev)
        self.engine_idx = torch.zeros(n, dtype=torch.long, device=dev)
        self.spec = layout_spec(cfg.layout, self.veh)
        self.act = TorchActuators(self.veh, self.eng, n, self.step_dt, dev, rcs=self.spec.rcs,
                                  n_main=self.spec.n_engines)
        self.eng_pos_b = torch.tensor(self.spec.engine_pos_b, **f)
        self.eng_share = torch.tensor(self.spec.engine_share, **f)
        self.mass = float(ms.mass_kg)
        self.com_b = torch.tensor(ms.com_body_m, **f)  # physics body CoM (full propellant, fixed)
        self.mm = mm
        self.i0 = torch.tensor(np.diag(ms.inertia_com_body), **f)
        self.unit = tuple(torch.eye(3, **f))
        self.com_true_b = self.com_b.expand(n, 3).clone()  # true CoM (= com_b unless variable_mass)
        self.d_dot_b = torch.zeros(n, 3, **f)  # d/dt of (true CoM - body CoM), body frame
        self.mass_true = torch.full((n,), self.mass, **f)
        self.aero_k = None
        if cfg.aero:
            self.aero_model = AeroModel.load(REPO / "configs/aero_v0.yaml", self.veh.mass.structure_radius_m,
                                             self.veh.mass.structure_length_m)
            self.aero_k = self.aero_model.arrays(lambda a: torch.tensor(a, **f))
        self.pivot_b = torch.tensor(self.veh.layout.main_engine.gimbal_pivot_body_m, **f)
        feet = torch.tensor(self.veh.gear.foot_positions_body_m, **f).reshape(-1, 3)
        self.feet_b = feet
        self.td_com_z = float(ms.com_body_m[2] - feet[:, 2].min())  # CoM height when standing
        self.prop0 = float(mm.prop0)
        self.rcs0 = float(self.spec.rcs_gas_kg if self.spec.rcs_gas_kg is not None else mm.rcs0)
        self.gmax = self.veh.layout.main_engine.max_gimbal_rad
        self._target = torch.zeros(n, 3, **f)  # pad at env origin
        z = lambda *s: torch.zeros(n, *s, **f)  # noqa: E731
        self.m_prop, self.m_rcs, self.t, self.wind_w = z(), z(), z(), z(3)
        self.actions, self.prev_actions = z(self.spec.act_dim), z(self.spec.act_dim)
        self.wind_base, self.gust_period, self.gust_phase = z(3), z(), z()
        self._meas_prev = None  # delayed measurement buffer ("noise" scenario)
        self.td = torch.zeros(n, dtype=torch.bool, device=dev)
        self.td_t, self.td_vz, self.td_vh, self.td_tilt, self.td_pos = z(), z(), z(), z(), z()
        self.invalid_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.fuel_used = z()
        self.phi = z()
        self.max_tilt = z()
        core = (1 / 10,) * 3 + (1 / 5,) * 3 + (1.0,) * 3 + (1.0,) * 3 + (1 / 50, 1 / 5, 1.0, 1.0)
        self.obs_scale = torch.tensor(core + (1.0,) * (self.spec.obs_dim - len(core)), **f)
        self.episode_results: list[dict] = []  # filled at reset (evaluation reads it)
        self._first_episode_done = torch.zeros(n, dtype=torch.bool, device=dev)
        self._episode_count = torch.zeros(n, dtype=torch.long, device=dev)
        self._update_mass_props()

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
        if self.cfg.visual == "v1":
            from rocket_dt.isaac_env.visual_assets import PAD_USD, SKY_HDR, SKY_SUN_AZIMUTH_DEG, SKY_SUN_ELEVATION_DEG
            pad = sim_utils.UsdFileCfg(usd_path=str(PAD_USD))
            pad.func("/World/envs/env_.*/Pad", pad)
            dome = sim_utils.DomeLightCfg(intensity=1000.0, texture_file=str(SKY_HDR), texture_format="latlong")
            dome.func("/World/Light", dome)
            # sun along the baked sky's sun direction: rotate the light's -Z onto -(sun direction)
            e, a = math.radians(SKY_SUN_ELEVATION_DEG), math.radians(SKY_SUN_AZIMUTH_DEG)
            to_sun = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
            axis = np.cross([0.0, 0.0, 1.0], to_sun)
            ang = math.acos(float(np.clip(to_sun[2], -1, 1)))
            axis = axis / np.linalg.norm(axis)
            sun = sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 0.93, 0.82), angle=0.53)
            sun.func("/World/Sun", sun, orientation=(math.cos(ang / 2), *(axis * math.sin(ang / 2))))
            return
        light = sim_utils.DomeLightCfg(intensity=self.cfg.sky_intensity, color=tuple(self.cfg.sky_color))
        light.func("/World/Light", light)
        if self.cfg.sun_intensity > 0:  # white key light so a tinted sky does not tint the vehicle
            sun = sim_utils.DistantLightCfg(intensity=self.cfg.sun_intensity, color=(1.0, 0.98, 0.95), angle=0.5)
            sun.func("/World/Sun", sun, orientation=(0.82, 0.25, 0.35, 0.38))

    # ---------------------------------------------------------------- state helpers
    def _state(self):
        """True CoM position/velocity (env frame), quaternion xyzw, body rates."""
        d = self.robot.data
        p = d.root_com_pos_w.torch - self.scene.env_origins
        q = d.root_link_quat_w.torch  # xyzw
        v = d.root_com_lin_vel_w.torch
        w_b = d.root_com_ang_vel_b.torch
        if self.cfg.variable_mass:  # PhysX tracks the fixed full-propellant CoM; shift to the true CoM
            rot = _rotmat_xyzw(q)
            dd = self.com_true_b - self.com_b
            p = p + (rot @ dd[..., None]).squeeze(-1)
            v = v + (rot @ (torch.linalg.cross(w_b, dd) + self.d_dot_b)[..., None]).squeeze(-1)
        return p, q, v, w_b

    @staticmethod
    def _tilt(rot):
        return torch.acos(rot[:, 2, 2].clamp(-1.0, 1.0))

    def _foot_bottom_z(self, p, rot):
        r = self.feet_b[None] - self.com_true_b[:, None]  # (n, 4, 3)
        return p[:, None, 2] + (rot[:, None] @ r[..., None]).squeeze(-1)[..., 2]

    # ---------------------------------------------------------------- actions
    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone().clamp(-1.0, 1.0)
        a = self.actions
        tmin, tmax = self.act.tmin, self.act.tmax
        nt = self.spec.n_throttle_actions
        throttle = tmin + (a[:, :nt] + 1.0) * 0.5 * (tmax - tmin)
        throttle = throttle[:, 0] if nt == 1 else throttle
        k = nt
        if self.spec.gimbal:
            gimbal = a[:, k:k + 2] * self.gmax
            k += 2
        else:
            gimbal = torch.zeros(self.num_envs, 2, device=self.device)
        rcs_on = (a[:, k:k + self.act.k] + 1.0) * 0.5 * self.step_dt
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
        thrs, gim, frac = self.act.step(self.t, dt)
        pa = isa_pressure(p[:, 2].clamp(min=0.0))
        n, E = self.num_envs, self.spec.n_engines
        f_e, md_e, val_e = self.map.lookup(thrs.reshape(-1), pa.repeat_interleave(E),
                                           self.engine_idx.repeat_interleave(E))
        f_e = f_e.reshape(n, E) * self.eng_share
        md_e = md_e.reshape(n, E) * self.eng_share
        valid = val_e.reshape(n, E).all(dim=1)
        main_on = self.act.engine_on & (self.m_prop > 0)
        f_e = torch.where(main_on[:, None], f_e, torch.zeros_like(f_e))
        f_main = f_e.sum(1)
        self.last_f_e = f_e  # per-engine thrust (telemetry / visualisation)
        mdot_main = torch.where(main_on, md_e.sum(1), torch.zeros_like(f_main))
        self.invalid_steps += (main_on & ~valid).long()
        rcs_frac = torch.where((self.m_rcs > 0)[:, None], frac, torch.zeros_like(frac))
        if self.spec.gimbal:
            dirs = gimbal_direction(gim)[:, None].expand(n, E, 3)
        else:
            dirs = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(n, E, 3)
        force_b = torch.cat([f_e[..., None] * dirs,
                             rcs_frac[..., None] * (self.act.fmax[:, None] * self.act.dir_b)[None]], dim=1)
        pos_b = torch.cat([self.eng_pos_b.expand(n, E, 3), self.act.pos_b.expand(n, -1, 3)], dim=1)
        if self.cfg.gust_amp > 0:  # sinusoidal gust on the constant wind (aero: speed x sqrt, so force x g)
            g = 1.0 + self.cfg.gust_amp * torch.sin(2 * math.pi * self.t / self.gust_period + self.gust_phase)
            self.wind_w = self.wind_base * (g.clamp(min=0.0).sqrt() if self.cfg.aero else g)[:, None]
        F, tau = batched_wrench_torch(force_b, pos_b, self.com_true_b)
        # last physics step, for telemetry/visualisation only
        self.last_f_main, self.last_valid, self.last_rcs_frac, self.last_pa = f_main, valid & main_on, rcs_frac, pa
        self.last_F_b, self.last_tau_b, self.last_gimbal = F[:, 0], tau[:, 0], gim
        mdot_rcs = (rcs_frac * self.act.mdot_full).sum(1)
        comp = self.robot.permanent_wrench_composer
        if not (self.cfg.aero or self.cfg.variable_mass):  # v0 path, unchanged
            comp.set_forces_and_torques_index(forces=F, torques=tau, positions=None, is_global=False)
            comp.add_forces_and_torques_index(forces=self.wind_w[:, None], positions=None, is_global=True)
        else:
            f_b, t_b = F[:, 0], tau[:, 0]
            wind_force_w = self.wind_w
            if self.aero_k is not None:  # wind_w holds the reference-height wind velocity here
                fa, ta = aero_wrench(self.aero_k, p[:, 2], rot, v, w_b, self.com_true_b, self.wind_w)
                f_b, t_b, wind_force_w = f_b + fa, t_b + ta, torch.zeros_like(self.wind_w)
                self.last_F_aero_b = fa
            f_w = (rot @ f_b[..., None]).squeeze(-1) + wind_force_w
            if self.cfg.variable_mass:
                i_true = torch.stack([self.i_true[0], self.i_true[0], self.i_true[1]], 1)
                cz_dot, cz_ddot = self.mm.com_z_rates(self.m_prop, self.m_rcs, mdot_main, mdot_rcs)
                z3 = torch.zeros_like(cz_dot)
                self.d_dot_b = torch.stack([z3, z3, cz_dot], 1)
                f_w, t_b, _ = compensated_wrench(f_w, t_b, w_b, rot, self.mass_true, i_true,
                                                 self.com_true_b - self.com_b, self.mass, self.i0, self.unit,
                                                 self.d_dot_b, torch.stack([z3, z3, cz_ddot], 1))
            f_body = (rot.transpose(1, 2) @ f_w[..., None]).squeeze(-1)
            comp.set_forces_and_torques_index(forces=f_body[:, None], torques=t_b[:, None], positions=None,
                                              is_global=False)
        self.m_prop = (self.m_prop - mdot_main * dt).clamp(min=0.0)
        self.m_rcs = (self.m_rcs - mdot_rcs * dt).clamp(min=0.0)
        self._update_mass_props()
        self.fuel_used += mdot_main * dt
        self.t += dt

    def _update_mass_props(self):
        """True mass properties from the remaining propellant (variable_mass only)."""
        if not self.cfg.variable_mass:
            return
        m, cz, ixx, izz = self.mm.batched(self.m_prop, self.m_rcs)
        self.mass_true = m
        self.i_true = (ixx, izz)
        self.com_true_b = torch.stack([torch.zeros_like(cz), torch.zeros_like(cz), cz], 1)

    # ---------------------------------------------------------------- obs / reward / dones
    def _available_thrust_norm(self, p):
        fmax, _, _ = self.map.lookup(torch.full_like(p[:, 2], self.act.tmax), isa_pressure(p[:, 2].clamp(min=0.0)),
                                     self.engine_idx)
        return fmax / (self.mass_true * G0)

    def measured_state(self, advance: bool = False):
        """State as seen by the controller: true state, or (noise scenario) noisy and delayed by obs_delay_steps.
        `advance=True` pushes the current true state into the delay buffer (once per control step)."""
        p, q, v, w_b = self._state()
        if self.cfg.noise_pos_m <= 0 and self.cfg.obs_delay_steps == 0:
            return p, q, v, w_b
        if advance or self._meas_prev is None:
            cur = (p.clone(), q.clone(), v.clone(), w_b.clone())
            out = cur if self._meas_prev is None or self.cfg.obs_delay_steps == 0 else self._meas_prev
            self._meas_prev = cur
            n, dev = self.num_envs, self.device
            pn = out[0] + torch.randn(n, 3, device=dev) * self.cfg.noise_pos_m
            vn = out[2] + torch.randn(n, 3, device=dev) * self.cfg.noise_vel_m_s
            wn = out[3] + torch.randn(n, 3, device=dev) * self.cfg.noise_rate_rad_s
            dq = _quat_axis_angle_xyzw(torch.randn(n, 3, device=dev),
                                       torch.randn(n, device=dev).abs() * self.cfg.noise_att_rad)
            qn = _quat_mul_xyzw(out[1], dq)
            self._meas = (pn, qn, vn, wn)
        return self._meas

    def _get_observations(self) -> dict:
        p, q, v, w_b = self.measured_state(advance=True)
        rot = _rotmat_xyzw(q)
        rt = rot.transpose(1, 2)
        rel = (rt @ (self._target - p)[..., None]).squeeze(-1)
        v_b = (rt @ v[..., None]).squeeze(-1)
        g_b = (rt @ torch.tensor([0.0, 0.0, -1.0], device=self.device).expand_as(p)[..., None]).squeeze(-1)
        thr_n = (self.act.throttles - self.act.tmin) / (self.act.tmax - self.act.tmin)
        act_state = [thr_n.mean(1, keepdim=True), self.act.gimbal / self.gmax] if self.spec.gimbal else [thr_n]
        obs = torch.cat([rel, v_b, g_b, w_b, p[:, 2:3], v[:, 2:3], (self.m_prop / self.prop0)[:, None],
                         self._available_thrust_norm(p)[:, None], *act_state, self.actions], dim=-1) * self.obs_scale
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
                      - 0.5 * self.td_vz.abs() - 0.5 * self.td_pos
                      - self.cfg.rew_td_vz_margin * (self.td_vz.abs() - self.cfg.rew_td_vz_target).clamp(min=0.0))
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
        return [{"env": int(i), "engine": self.engine_names[int(self.engine_idx[i])], "scenario_index": int(i + (int(self._episode_count[i]) - 1) * self.num_envs),
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
                if len(self.engine_names) > 1:
                    for k, name in enumerate(self.engine_names):
                        sel = self.engine_idx[done_ids] == k
                        if sel.any():
                            self.extras["log"][f"Metrics/success_rate_{name}"] = succ[sel].mean().item()
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
            gper = torch.tensor([s.get("gust_period_s", 4.0) for s in sc], device=dev)
            gph = torch.tensor([s.get("gust_phase", 0.0) for s in sc], device=dev)
            fail_i = torch.tensor([s.get("rcs_failed", -1) for s in sc], device=dev)
        else:  # same distribution as evaluation.experiments.randomized_scenarios (torch RNG, not the seeds)
            u = lambda lo, hi: torch.rand(n, device=dev) * (hi - lo) + lo  # noqa: E731
            a = u(0, 2 * math.pi)
            r0, r1 = self.cfg.lateral_r_m
            rr = torch.sqrt(u(r0 * r0 / max(r1 * r1, 1e-9), 1.0)) * r1  # uniform in the annulus r0..r1
            pos = torch.stack([rr * torch.cos(a), rr * torch.sin(a), u(40, 80)], 1)
            vel = torch.stack([u(-1.5, 1.5), u(-1.5, 1.5), u(-8, -4)], 1)
            axis = torch.stack([torch.cos(a + 1), torch.sin(a + 1), torch.zeros(n, device=dev)], 1)
            ang = torch.deg2rad(u(0, 4))
            w_b = torch.stack([torch.zeros(n, device=dev), torch.zeros(n, device=dev), u(-0.2, 0.2)], 1)
            wd, wm = u(0, 2 * math.pi), u(0, self.cfg.wind_max_n)
            wind = torch.stack([wm * torch.cos(wd), wm * torch.sin(wd), torch.zeros(n, device=dev)], 1)
            gper, gph = u(3.0, 6.0), u(0, 2 * math.pi)
            fail_i = (torch.randint(self.act.k if self.cfg.scenario == "rcs_fail" else 1, (n,), device=dev)
                      if self.cfg.scenario == "rcs_fail" else torch.full((n,), -1, device=dev))
        q = _quat_axis_angle_xyzw(axis, ang)
        rot = _rotmat_xyzw(q)
        p_w = pos + self.scene.env_origins[env_ids]
        link = p_w - (rot @ self.com_b[None, :, None].expand(n, 3, 1)).squeeze(-1)
        self.robot.write_root_link_pose_to_sim_index(root_pose=torch.cat([link, q], 1), env_ids=env_ids)
        w_w = (rot @ w_b[..., None]).squeeze(-1)
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=torch.cat([vel, w_w], 1), env_ids=env_ids)
        if self.cfg.aero:  # v0 wind force -> reference-height wind speed giving that force (ADR-0011)
            fm = wind.norm(dim=1, keepdim=True)
            wind = wind / fm.clamp(min=1e-9) * float(self.aero_model.wind_speed_for_force(1.0)) * fm.sqrt()
        self.wind_w[env_ids] = wind
        self.wind_base[env_ids] = wind
        self.gust_period[env_ids] = gper.float()
        self.gust_phase[env_ids] = gph.float()
        if len(self.engine_names) > 1 and self.cfg.eval_scenarios is None:
            self.engine_idx[env_ids] = torch.randint(len(self.engine_names), (n,), device=dev)
        self.act.reset(env_ids, self.cfg.throttle0)
        has_fail = fail_i >= 0
        if has_fail.any():  # one RCS thruster fails for the whole episode (not observed by the controller)
            fid = env_ids[has_fail]
            self.act.failed[fid] = False
            self.act.failed[fid, fail_i[has_fail].long()] = True
        self.rcs_failed_idx = getattr(self, "rcs_failed_idx", torch.full((self.num_envs,), -1, device=dev))
        self.rcs_failed_idx[env_ids] = fail_i.long()
        self._meas_prev = None
        self.m_prop[env_ids] = self.prop0
        self.m_rcs[env_ids] = self.rcs0
        self.d_dot_b[env_ids] = 0.0  # reset velocity is written for the body CoM = true CoM at full load
        self._update_mass_props()
        self.t[env_ids] = 0.0
        self.td[env_ids] = False
        for b in (self.td_t, self.td_vz, self.td_vh, self.td_tilt, self.td_pos, self.fuel_used, self.max_tilt):
            b[env_ids] = 0.0
        self.invalid_steps[env_ids] = 0
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        nt = self.spec.n_throttle_actions
        self.actions[env_ids, :nt] = 2 * (self.cfg.throttle0 - self.act.tmin) / (self.act.tmax - self.act.tmin) - 1
        self.actions[env_ids, nt + (2 if self.spec.gimbal else 0):] = -1.0
        self.phi[env_ids] = 0.0
