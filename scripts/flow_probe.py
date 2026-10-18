"""Probe: NVIDIA Flow (omni.flowusd) fire as a downward exhaust in an Isaac Lab stage, headless RTX capture.

Writes runs/ppo_win-isaac/flow_probe/frame_XXX.png. Visual only (no coupling to the vehicle physics).
usage: isaaclab.bat -p scripts\\flow_probe.py --enable_cameras
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from isaaclab.app import AppLauncher

REPO = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=90)
parser.add_argument("--exhaust_speed", type=float, default=25.0)
parser.add_argument("--cell", type=float, default=0.06)
parser.add_argument("--set", nargs="*", default=[], help="prim_rel:attr=value overrides, e.g. flowEmitterSphere:fuel=1.5")
parser.add_argument("--tag", default="probe")
parser.add_argument("--descent", type=float, default=0.0, help="emitter descent speed [m/s] (moving vehicle test)")
parser.add_argument("--box", type=float, default=0.0, help=">0: use a FlowEmitterBox of this length [m] filling the plume")
parser.add_argument("--box_half_width", type=float, default=0.3)
parser.add_argument("--fps", type=float, default=25.0, help="Flow stepsPerSecond and frame rate (sim time)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.kit.app  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402

FLOW_EXT = "omni.flowusd"
OUT = REPO / "runs/ppo_win-isaac/flow_probe"  # per-tag subfolder


def metric_fire_usda(src: Path, dst: Path, cell: float):
    """Fire preset (cm, Y-up) -> metres, Z-up, exhaust tuning. Only numeric attributes we understand are changed."""
    t = src.read_text()
    t = t.replace("metersPerUnit = 0.01", "metersPerUnit = 1").replace('upAxis = "Y"', 'upAxis = "Z"')
    rep = {
        r"custom float radius = 10": "custom float radius = 0.25",
        r"custom float densityCellSize = 0.5": f"custom float densityCellSize = {cell}",
        r"float3 gravity = \(0, -100, 0\)": "float3 gravity = (0, 0, -9.81)",
        r"float buoyancyPerTemp = 6": "float buoyancyPerTemp = 0.3",
        r"custom bool velocityIsWorldSpace = 0": "custom bool velocityIsWorldSpace = 1",
    }
    for a, b in rep.items():
        t, n = re.subn(a, b, t)
        if n == 0:
            raise RuntimeError(f"preset pattern not found: {a}")
    dst.write_text(t)


def main():
    out = OUT / args.tag
    out.mkdir(parents=True, exist_ok=True)
    em = omni.kit.app.get_app().get_extension_manager()
    em.set_extension_enabled_immediate(FLOW_EXT, True)
    path = Path(em.get_extension_path_by_module("omni.flowusd") if hasattr(em, "get_extension_path_by_module") else "")
    src = next(Path(r"C:\isaacsim\extscache").glob("omni.flowusd-*")) / "data/presets/Fire/Fire.usda"
    dst = out / "fire_metric.usda"
    metric_fire_usda(src, dst, args.cell)
    s = carb.settings.get_settings()
    for k in ("/rtx/flow/enabled", "/rtx/flow/rayTracedTranslucencyEnabled", "/rtx/flow/rayTracedReflectionsEnabled",
              "/rtx/flow/pathTracingEnabled"):
        s.set(k, True)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / args.fps, device="cuda:0"))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    light = sim_utils.DomeLightCfg(intensity=800.0)
    light.func("/World/Light", light)
    from pxr import Gf, UsdGeom
    stage = sim_utils.get_current_stage()
    plume = stage.DefinePrim("/World/Plume", "Xform")
    plume.GetReferences().AddReference(str(dst))
    hull = UsdGeom.Cylinder.Define(stage, "/World/Hull")
    hull.CreateRadiusAttr(0.6), hull.CreateHeightAttr(7.0)
    hull_op = hull.AddTranslateOp()
    z0 = 4.6 + args.descent * args.frames / args.fps
    hull_op.Set(Gf.Vec3d(0, 0, z0 + 3.9))
    emitter = stage.GetPrimAtPath("/World/Plume/flowEmitterSphere")
    print("FLOW_PRIMS", [str(p.GetPath()) for p in stage.Traverse() if "Flow" in p.GetTypeName()][:8])
    emitter.GetAttribute("position").Set(Gf.Vec3f(0, 0, z0))
    stage.GetPrimAtPath("/World/Plume/flowSimulate").GetAttribute("stepsPerSecond").Set(float(args.fps))
    box_xf = None
    if args.box > 0:
        emitter.GetAttribute("enabled").Set(False)
        box_xf = UsdGeom.Xform.Define(stage, "/World/PlumeBox")
        box_op = box_xf.AddTranslateOp()
        box = stage.DefinePrim("/World/PlumeBox/box", "FlowEmitterBox")
        from pxr import Sdf
        F3, F, I, B = Sdf.ValueTypeNames.Float3, Sdf.ValueTypeNames.Float, Sdf.ValueTypeNames.Int, Sdf.ValueTypeNames.Bool
        for k, (tn, v) in {"halfSize": (F3, Gf.Vec3f(args.box_half_width, args.box_half_width, args.box / 2)),
                           "position": (F3, Gf.Vec3f(0, 0, -args.box / 2)),
                           "velocity": (F3, Gf.Vec3f(0, 0, -args.exhaust_speed)),
                           "velocityIsWorldSpace": (B, False), "enabled": (B, True), "layer": (I, 0),
                           "temperature": (F, 2.0), "fuel": (F, 2.0), "smoke": (F, 0.0), "burn": (F, 0.0),
                           "divergence": (F, 0.0), "coupleRateTemperature": (F, 10.0), "coupleRateFuel": (F, 10.0),
                           "coupleRateVelocity": (F, 10.0), "coupleRateSmoke": (F, 0.0), "coupleRateBurn": (F, 0.0),
                           "coupleRateDivergence": (F, 0.0), "allocationScale": (F, 1.0)}.items():
            box.CreateAttribute(k, tn).Set(v)
        print("BOX_ATTRS", [a.GetName() for a in box.GetAttributes()][:6], flush=True)
    emitter.GetAttribute("velocity").Set(Gf.Vec3f(0, 0, -args.exhaust_speed))
    for kv in args.set:
        rel, val = kv.split("=")
        prim_rel, attr = rel.rsplit(":", 1)
        a = stage.GetPrimAtPath(f"/World/Plume/{prim_rel}").GetAttribute(attr)
        a.Set(type(a.Get())(float(val)) if not isinstance(a.Get(), bool) else val.lower() in ("1", "true"))
        print("SET", prim_rel, attr, a.Get())
    sim.reset()
    import omni.replicator.core as rep
    from isaacsim.core.rendering_manager import ViewportManager
    ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=[14, -14, 6], target=[0, 0, 3.5])
    rp = rep.create.render_product("/OmniverseKit_Persp", (960, 540))
    ann = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    ann.attach([rp])
    import imageio.v2 as imageio
    for i in range(args.frames):
        z = z0 - args.descent * i / args.fps
        emitter.GetAttribute("position").Set(Gf.Vec3f(0, 0, z))
        if box_xf is not None:
            box_op.Set(Gf.Vec3d(0, 0, z))
        emitter.GetAttribute("velocity").Set(Gf.Vec3f(0, 0, -args.descent - args.exhaust_speed))
        hull_op.Set(Gf.Vec3d(0, 0, z + 3.9))
        ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=[14, -14, z + 2.0], target=[0, 0, z - 1.0])
        sim.step(render=True)
        omni.kit.app.get_app().update()
        img = np.asarray(ann.get_data())
        if img.size and i % 15 == 14:
            imageio.imwrite(out / f"frame_{i:03d}.png", img.reshape(540, 960, -1)[..., :3])
    print("FLOW_PROBE_DONE", out)


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
