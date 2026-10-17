"""Isaac force smoke test: one rigid body per env, known (F, tau about CoM) vs Reference6DoF.

usage (Windows, Isaac Lab checkout):
    C:\\dev\\IsaacLab\\isaaclab.bat -p scripts\\isaac_force_smoke.py --headless --out runs\\isaac_smoke_win-isaac

Mass, CoM and inertia come from MassModel(vehicle_v0 + conical_A, full propellant) and are written
into PhysX after spawn. Point forces (main engine at the gimbal pivot, optional RCS) are reduced to a
body-frame wrench about the CoM by wrench_bridge.batched_wrench and applied with
permanent_wrench_composer.set_forces_and_torques_index(positions=None, is_global=False).
After n_steps physics steps the mean linear acceleration (world, CoM) and mean angular acceleration
(body) are compared with the same finite difference of Reference6DoF (RK4, same dt) and with
Reference6DoF.deriv at t0.

Frame conventions (checked here, not assumed):
- rocket_dt quaternions are (w, x, y, z); Isaac Lab 3.x root poses are (x, y, z, w).
- Isaac root pose is the link (body origin) frame; rocket_dt state position is the CoM.
- Forces/torques are body frame (is_global=False); torque is about the CoM.
- Linear acceleration is compared in world frame at the CoM; angular acceleration as d(omega_b)/dt in body frame.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

REPO = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--n_steps", type=int, default=10)
parser.add_argument("--physics_hz", type=float, default=200.0)  # SimConfig.physics_hz in rocket_dt
parser.add_argument("--out", default=str(REPO / "runs/isaac_smoke_win-isaac"))
parser.add_argument("--tol_lin", type=float, default=0.01)
parser.add_argument("--tol_ang", type=float, default=0.02)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObject, RigidObjectCfg  # noqa: E402

sys.path.insert(0, str(REPO / "src"))
from rocket_dt.dynamics.frames import G0, quat_from_axis_angle, quat_to_rotmat  # noqa: E402
from rocket_dt.dynamics.mass import MassModel  # noqa: E402
from rocket_dt.dynamics.reference_6dof import IDX_V, IDX_W, Reference6DoF, StepInputs, make_state  # noqa: E402
from rocket_dt.dynamics.wrench import PointForce  # noqa: E402
from rocket_dt.isaac_env.wrench_bridge import batched_wrench  # noqa: E402
from rocket_dt.schemas import load_engine, load_vehicle  # noqa: E402


def cases(vehicle, ms):
    """Per env: (name, q_wxyz, omega_b, point forces). Thrust 1.2 x weight at the gimbal pivot."""
    pivot = np.array(vehicle.layout.main_engine.gimbal_pivot_body_m, float)
    t = 1.2 * ms.mass_kg * G0

    def main(gx, gy):
        d = np.array([np.sin(gy), -np.sin(gx) * np.cos(gy), np.cos(gx) * np.cos(gy)])
        return PointForce("main", t * d, pivot)

    rcs = vehicle.layout.rcs[0]
    rcs_f = PointForce(rcs.id, rcs.max_thrust_n * np.array(rcs.force_direction_body, float),
                       np.array(rcs.position_body_m, float))
    out = [
        ("axial_upright", quat_from_axis_angle([1, 0, 0], 0.0), np.zeros(3), [main(0.0, 0.0)]),
        ("gimbal2deg_upright", quat_from_axis_angle([1, 0, 0], 0.0), np.zeros(3), [main(np.radians(2), 0.0)]),
        ("gimbal_rcs_tilted15", quat_from_axis_angle([1, 1, 0], np.radians(15)), np.zeros(3),
         [main(np.radians(1), np.radians(2)), rcs_f]),
        ("gimbal_rcs_tilted10_spinning", quat_from_axis_angle([0, 1, 0], np.radians(10)), np.array([0.2, -0.1, 0.8]),
         [main(np.radians(2), np.radians(-1)), rcs_f]),
    ]
    return [out[i % len(out)] for i in range(args.num_envs)]


def rel(a, b):
    nb = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b) / nb) if nb > 1e-9 else None


def gpu_info():
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
        name, drv, tot, used = [s.strip() for s in q.split(",")]
        return {"name": name, "driver": drv, "memory_total_mib": int(tot), "memory_used_mib_whole_gpu_at_end_all_processes": int(used)}
    except Exception as e:  # pragma: no cover
        return {"error": str(e)}


def versions():
    isaaclab_root = Path(sys.executable).resolve()
    lab = Path(r"C:\dev\IsaacLab")
    for p in [Path.cwd(), *isaaclab_root.parents]:
        if (p / "isaaclab.bat").exists():
            lab = p
    import isaaclab
    import warp
    v = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
         "warp": warp.__version__, "isaaclab_package": getattr(isaaclab, "__version__", "?"),
         "platform": platform.platform()}
    for key, f in (("isaaclab_repo_version", lab / "VERSION"), ("isaacsim_version", lab / "_isaac_sim/VERSION")):
        v[key] = f.read_text().strip() if f.exists() else None
    for key, cwd in (("isaaclab_git_describe", lab), ("rocket_dt_commit", REPO)):
        v[key] = subprocess.run(["git", "describe", "--tags", "--always", "--dirty"], cwd=cwd,
                                capture_output=True, text=True).stdout.strip()
    return v


def main():
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vehicle = load_vehicle(REPO / "configs/vehicle_v0.yaml")
    engine = load_engine(REPO / "configs/engines/conical_A.yaml")
    mm = MassModel(vehicle, engine)
    ms = mm.evaluate(mm.prop0, mm.rcs0)
    n, dt = args.num_envs, 1.0 / args.physics_hz
    cs = cases(vehicle, ms)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args.device, gravity=(0.0, 0.0, -G0)))
    for i in range(n):
        sim_utils.create_prim(f"/World/Env{i}", "Xform", translation=(20.0 * i, 0.0, 0.0))
    body = RigidObject(RigidObjectCfg(
        prim_path="/World/Env.*/Body",
        spawn=sim_utils.CuboidCfg(  # shape is visual only (no collider); mass props are overwritten below
            size=(1.2, 1.2, 7.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                linear_damping=0.0, angular_damping=0.0, max_linear_velocity=1e4, max_angular_velocity=1e6,
                enable_gyroscopic_forces=True, disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=ms.mass_kg)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 100.0))))
    # Mass properties are authored in USD before reset. Writing a full 3x3 inertia through the tensor API
    # (set_inertias_index) makes PhysX diagonalise it; with the axisymmetric I_xx == I_yy this produced an
    # invalid CoM frame (PxRigidDynamic::setCMassLocalPose pose is not valid) and wrong dynamics.
    from pxr import Gf, UsdPhysics
    stage = sim_utils.get_current_stage()
    i_diag = np.diag(ms.inertia_com_body)
    for i in range(n):
        api = UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(f"/World/Env{i}/Body"))
        api.CreateMassAttr().Set(float(ms.mass_kg))
        api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*map(float, ms.com_body_m)))
        api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*map(float, i_diag)))
        api.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    sim.reset()
    dev = body.device

    def tt(a):
        return torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)

    p_com0 = np.array([[20.0 * i, 0.0, 100.0] for i in range(n)])
    rot0 = [quat_to_rotmat(c[1]) for c in cs]
    link_pos = np.array([p_com0[i] - rot0[i] @ ms.com_body_m for i in range(n)])
    quat_xyzw = np.array([[*c[1][1:], c[1][0]] for c in cs])
    body.write_root_link_pose_to_sim_index(root_pose=tt(np.hstack([link_pos, quat_xyzw])))
    omega_w0 = np.array([rot0[i] @ cs[i][2] for i in range(n)])
    body.write_root_com_velocity_to_sim_index(root_velocity=tt(np.hstack([np.zeros((n, 3)), omega_w0])))
    body.reset()
    body.update(0.0)

    pf = [c[3] for c in cs]
    k = max(len(f) for f in pf)
    force_b, pos_b = np.zeros((n, k, 3)), np.zeros((n, k, 3))
    for i, fl in enumerate(pf):
        for j, f in enumerate(fl):
            force_b[i, j], pos_b[i, j] = f.force_body_n, f.position_body_m
    f_b, tau_b = batched_wrench(force_b, pos_b, np.tile(ms.com_body_m, (n, 1)))

    def read():
        d = body.data
        return (d.root_com_pos_w.torch.cpu().numpy().astype(float), d.root_com_lin_vel_w.torch.cpu().numpy().astype(float),
                d.root_com_ang_vel_b.torch.cpu().numpy().astype(float))

    rv = body.root_view  # read back from PhysX itself, not the Isaac Lab buffers
    readback = {"mass_kg": rv.get_masses().numpy().ravel().tolist(),
                "inertia_3x3": rv.get_inertias().numpy().reshape(n, 9).tolist(),
                "com_pose_b_xyz_qxyzw": rv.get_coms().numpy().reshape(n, 7).tolist()}
    p0, v0, w0 = read()
    t0 = time.time()
    for _ in range(args.n_steps):
        body.permanent_wrench_composer.set_forces_and_torques_index(
            forces=tt(f_b), torques=tt(tau_b), positions=None, is_global=False)
        body.write_data_to_sim()
        sim.step(render=False)
        body.update(dt)
    wall = time.time() - t0
    p1, v1, w1 = read()
    t_tot = args.n_steps * dt

    ref = Reference6DoF(mm, None)
    zero = np.zeros(3)
    rows, ok_all = [], True
    for i, (name, q, w_b, fl) in enumerate(cs):
        u = StepInputs(fl, 0.0, 0.0, zero, zero)
        x = make_state(p_com0[i], np.zeros(3), q, w_b, mm.prop0, mm.rcs0)
        d0 = ref.deriv(x, u)
        xr = x.copy()
        for _ in range(args.n_steps):
            xr = ref.step(xr, u, dt)
        a_isaac = (v1[i] - v0[i]) / t_tot
        al_isaac = (w1[i] - w0[i]) / t_tot
        a_ref = (xr[IDX_V] - x[IDX_V]) / t_tot
        al_ref = (xr[IDX_W] - x[IDX_W]) / t_tot
        e_lin, e_ang = rel(a_isaac, a_ref), rel(al_isaac, al_ref)
        ang_abs = float(np.linalg.norm(al_isaac - al_ref))
        ok_lin = e_lin is not None and e_lin < args.tol_lin
        ok_ang = (e_ang < args.tol_ang) if e_ang is not None else ang_abs < 1e-3
        ok_all &= ok_lin and ok_ang
        rows.append({
            "env": i, "case": name, "q_wxyz": q.tolist(), "omega0_b": w_b.tolist(),
            "force_b_n": f_b[i, 0].tolist(), "torque_b_about_com_nm": tau_b[i, 0].tolist(),
            "isaac_omega0_b_readback": w0[i].tolist(),
            "lin_acc_w": {"isaac": a_isaac.tolist(), "ref_rk4_fd": a_ref.tolist(), "ref_deriv_t0": d0[IDX_V].tolist(),
                          "rel_err_vs_ref_fd": e_lin, "rel_err_vs_deriv_t0": rel(a_isaac, d0[IDX_V])},
            "ang_acc_b": {"isaac": al_isaac.tolist(), "ref_rk4_fd": al_ref.tolist(), "ref_deriv_t0": d0[IDX_W].tolist(),
                          "rel_err_vs_ref_fd": e_ang, "abs_err_vs_ref_fd": ang_abs,
                          "rel_err_vs_deriv_t0": rel(al_isaac, d0[IDX_W])},
            "com_pos_err_m_after": float(np.linalg.norm(p1[i] - xr[0:3])),
            "pass_lin": ok_lin, "pass_ang": ok_ang})

    gpu = gpu_info()
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None
    metrics = {"label": "win-isaac", "status": "PASS" if ok_all else "FAIL",
               "criteria": {"lin_rel_err_lt": args.tol_lin, "ang_rel_err_lt": args.tol_ang,
                            "ang_abs_err_lt_when_ref_zero": 1e-3},
               "dt_s": dt, "n_steps": args.n_steps, "num_envs": n, "device": str(dev), "loop_wall_s": wall,
               "mass_model": {"mass_kg": ms.mass_kg, "com_body_m": ms.com_body_m.tolist(),
                              "inertia_diag": np.diag(ms.inertia_com_body).tolist(),
                              "inertia_offdiag_max_abs": float(np.abs(ms.inertia_com_body - np.diag(
                                  np.diag(ms.inertia_com_body))).max())},
               "physx_readback": readback,
               "frames": {"rocket_dt_quat": "wxyz", "isaac_root_pose_quat": "xyzw", "wrench": "body frame, about CoM",
                          "lin_acc": "world frame, CoM", "ang_acc": "d(omega_body)/dt, body frame"},
               "cases": rows}
    manifest = {"label": "win-isaac", "script": "scripts/isaac_force_smoke.py", "argv": sys.argv,
                "software": versions(), "gpu": gpu, "torch_cuda_max_allocated_mib": peak,
                "physics_backend": getattr(getattr(sim, "physics_manager", None), "__name__", "unknown")}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("SMOKE_RESULT", json.dumps({"status": metrics["status"], "cases": [
        {"case": r["case"], "lin": r["lin_acc_w"]["rel_err_vs_ref_fd"], "ang": r["ang_acc_b"]["rel_err_vs_ref_fd"],
         "ang_abs": r["ang_acc_b"]["abs_err_vs_ref_fd"]} for r in rows]}))


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
