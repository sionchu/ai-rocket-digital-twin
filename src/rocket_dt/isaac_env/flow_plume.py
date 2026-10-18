"""NVIDIA Flow (omni.flowusd) exhaust plume for Isaac renders. Visual only: it does not act on the vehicle and is
not a combustion model. It is a volumetric picture of the 1D nozzle output (thrust -> length, Pe/Pa -> spread).

Setup: the extension's Fire preset (cm, Y-up) supplies the FlowSimulate/FlowOffscreen/FlowRender prims, rewritten to
metres / Z-up. Its sphere emitter is disabled; a FlowEmitterBox under a per-frame Xform fills the plume envelope
instead (a point/sphere emitter on a vehicle descending at 5-6 m/s left the fire trailing above the nozzle; see
scripts/flow_probe.py tags dA-dC vs box5). Box local +Z = thrust axis, so the box extends along the exhaust.
Flow steps once per rendered frame, so stepsPerSecond is set to the video frame rate (sim time).
"""
from __future__ import annotations

import math
import re
from pathlib import Path

TUNING = {  # prim (relative to the plume root) : {attr: value}; found with short renders flow_bright / flow_bright2
    "flowSimulate/advection": {"coolingRate": 2.5, "burnPerTemp": 2.0, "fuelPerBurn": 0.1, "smokePerBurn": 0.3},
    "flowSimulate/advection/smoke": {"fade": 6.0},
    "flowSimulate/advection/temperature": {"fade": 4.0},  # fast decay: the box refills the plume every frame,
    "flowSimulate/advection/fuel": {"fade": 4.0},  # so old fire must not linger above a descending vehicle
    "flowRender/rayMarch": {"colorScale": 6.0},  # brighter than the preset under the env's 2500-nit dome light
    "flowOffscreen/colormap": {"colorScale": 6.0},
}
BOX = {"temperature": 2.0, "fuel": 2.0, "coupleRateTemperature": 10.0, "coupleRateFuel": 10.0,
       "coupleRateVelocity": 10.0}


def metric_fire_usda(src: Path, dst: Path, cell: float = 0.06) -> None:
    t = src.read_text()
    t = t.replace("metersPerUnit = 0.01", "metersPerUnit = 1").replace('upAxis = "Y"', 'upAxis = "Z"')
    for a, b in {r"custom float radius = 10": "custom float radius = 0.25",
                 r"custom float densityCellSize = 0.5": f"custom float densityCellSize = {cell}",
                 r"float3 gravity = \(0, -100, 0\)": "float3 gravity = (0, 0, -9.81)",
                 r"float buoyancyPerTemp = 6": "float buoyancyPerTemp = 0.3",
                 r"custom bool velocityIsWorldSpace = 0": "custom bool velocityIsWorldSpace = 1"}.items():
        t, n = re.subn(a, b, t)
        if n == 0:
            raise RuntimeError(f"Flow preset changed, pattern not found: {a}")
    dst.write_text(t)


def enable_flow():
    """Enable the extension and its RTX switches; call before the stage is populated."""
    import carb
    import omni.kit.app

    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.flowusd", True)
    s = carb.settings.get_settings()
    for k in ("/rtx/flow/enabled", "/rtx/flow/rayTracedTranslucencyEnabled", "/rtx/flow/rayTracedReflectionsEnabled",
              "/rtx/flow/pathTracingEnabled"):
        s.set(k, True)


def _quat_z_to(d):
    """(w, x, y, z) rotating +Z onto unit vector d."""
    n = math.sqrt(sum(c * c for c in d)) or 1.0
    d = [c / n for c in d]
    c = d[2]
    if c < -0.999999:
        return (0.0, 1.0, 0.0, 0.0)
    ax = (-d[1], d[0], 0.0)  # z x d
    w = 1.0 + c
    m = math.sqrt(w * w + ax[0] ** 2 + ax[1] ** 2)
    return (w / m, ax[0] / m, ax[1] / m, 0.0)


class FlowPlume:
    def __init__(self, stage, work_dir: Path, fps: float, root: str = "/World/FlowPlume", overrides: dict | None = None,
                 speed0: float = 10.0, speed_gain: float = 15.0, half_width: float = 0.3):
        from pxr import Gf, Sdf, UsdGeom

        self.Gf = Gf
        src = next(Path(r"C:\isaacsim\extscache").glob("omni.flowusd-*")) / "data/presets/Fire/Fire.usda"
        work_dir.mkdir(parents=True, exist_ok=True)
        dst = work_dir / "fire_metric.usda"
        metric_fire_usda(src, dst)
        prim = stage.DefinePrim(root, "Xform")
        prim.GetReferences().AddReference(str(dst))
        stage.GetPrimAtPath(f"{root}/flowEmitterSphere").GetAttribute("enabled").Set(False)
        tuning = {k: dict(v) for k, v in TUNING.items()}
        for rel, attrs in (overrides or {}).items():
            tuning.setdefault(rel, {}).update(attrs)
        for rel, attrs in tuning.items():
            if rel == "box":
                continue
            p = stage.GetPrimAtPath(f"{root}/{rel}")
            for k, v in attrs.items():
                p.GetAttribute(k).Set(type(p.GetAttribute(k).Get())(v))
        stage.GetPrimAtPath(f"{root}/flowSimulate").GetAttribute("stepsPerSecond").Set(float(fps))
        xf = UsdGeom.Xform.Define(stage, f"{root}Box")
        self.t_op, self.r_op = xf.AddTranslateOp(), xf.AddOrientOp()
        box = stage.DefinePrim(f"{root}Box/box", "FlowEmitterBox")  # schema attrs must be authored explicitly
        F3, F, I, B = Sdf.ValueTypeNames.Float3, Sdf.ValueTypeNames.Float, Sdf.ValueTypeNames.Int, Sdf.ValueTypeNames.Bool
        box_vals = dict(BOX)
        box_vals.update(tuning.get("box", {}))
        self.attr = {}
        for k, tn, v in [("halfSize", F3, Gf.Vec3f(half_width, half_width, 1.0)), ("position", F3, Gf.Vec3f(0, 0, -1)),
                         ("velocity", F3, Gf.Vec3f(0, 0, -speed0)), ("velocityIsWorldSpace", B, False),
                         ("enabled", B, False), ("layer", I, 0), ("smoke", F, 0.0), ("burn", F, 0.0),
                         ("divergence", F, 0.0), ("coupleRateSmoke", F, 0.0), ("coupleRateBurn", F, 0.0),
                         ("coupleRateDivergence", F, 0.0), ("allocationScale", F, 1.0)] + \
                [(k, F, float(v)) for k, v in box_vals.items()]:
            a = box.CreateAttribute(k, tn)
            a.Set(v)
            self.attr[k] = a
        tuning["box"] = box_vals
        self.tuning, self.speed0, self.speed_gain, self.half_width = tuning, speed0, speed_gain, half_width

    def update(self, exit_w, thrust_dir_w, length_m: float, spread: float, thrust_frac: float, engine_on: bool):
        Gf = self.Gf
        on = bool(engine_on and thrust_frac > 0.02 and length_m > 0.05)
        self.attr["enabled"].Set(on)
        if not on:
            return
        self.t_op.Set(Gf.Vec3d(*map(float, exit_w)))
        self.r_op.Set(Gf.Quatf(*map(float, _quat_z_to(thrust_dir_w))))
        hw = self.half_width * spread
        self.attr["halfSize"].Set(Gf.Vec3f(hw, hw, length_m / 2))
        self.attr["position"].Set(Gf.Vec3f(0, 0, -length_m / 2))
        self.attr["velocity"].Set(Gf.Vec3f(0, 0, -(self.speed0 + self.speed_gain * thrust_frac)))
        self.attr["fuel"].Set(float(0.8 + 1.2 * thrust_frac))
