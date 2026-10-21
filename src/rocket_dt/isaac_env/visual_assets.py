"""Visual-only USD assets for the VTVL vehicle and landing site (assets/vtvl_v1, built by scripts/assets/).

Nothing here touches physics: the referenced files carry no UsdPhysics schemas. Reference the vehicle visual under
the physics rigid body so it follows the body; the physics hull/foot colliders stay as they are.

Importing this module needs nothing beyond the standard library; pxr is imported inside the functions (no Isaac,
no torch).
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ASSET_DIR = REPO / "assets" / "vtvl_v1"
PAD_USD = ASSET_DIR / "pad.usd"
SKY_HDR = ASSET_DIR / "sky_2k.hdr"
# the sky bake has no sun disc; add a distant light from this direction (azimuth CCW from +X, Z-up world)
SKY_SUN_ELEVATION_DEG = 20.0
SKY_SUN_AZIMUTH_DEG = 95.0

LAYOUTS = ("gimbal_rcs", "tri_diff", "quad_diff", "coldgas")
ENGINES = ("conical_A", "bell_B", "bell_C")


def vehicle_usd(layout: str, engine: str) -> Path:
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; choose from {LAYOUTS}")
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}; choose from {ENGINES}")
    p = ASSET_DIR / f"vehicle_{layout}_{engine}.usd"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; build with: PYTHONPATH=src python3 scripts/assets/build_vtvl_assets.py")
    return p


def author_vehicle_visuals(stage, parent_path: str, layout: str, engine: str, name: str = "visual") -> list[str]:
    """Reference the visual vehicle (body frame: origin mid-length, +Z nose, metres) under parent_path/name.

    Returns the created prim paths: the visual root first, then its engine prims (one per nozzle).
    """
    from pxr import Sdf, UsdGeom

    asset = vehicle_usd(layout, engine)
    parent = stage.GetPrimAtPath(parent_path)
    if not parent.IsValid():
        UsdGeom.Xform.Define(stage, parent_path)
    root_path = Sdf.Path(parent_path).AppendChild(name)
    root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    root.GetReferences().ClearReferences()
    root.GetReferences().AddReference(str(asset))
    paths = [str(root_path)]
    engines = stage.GetPrimAtPath(root_path.AppendChild("engines"))
    if engines.IsValid():
        paths += [str(c.GetPath()) for c in engines.GetChildren()]
    return paths


def author_landing_site(stage, path: str = "/World/LandingSite") -> str:
    """Reference the visual pad + ground (pad top z=0.004, ground z=0 within 60 m). Keep the physics plane separate."""
    from pxr import UsdGeom

    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    prim.GetReferences().AddReference(str(PAD_USD))
    return path


def author_sky(stage, path: str = "/World/Sky", dome_intensity: float = 1.0, sun_intensity: float = 3000.0) -> list[str]:
    """DomeLight with sky_2k.hdr (lat-long) + a DistantLight at the baked sun direction.

    Not verified in Isaac Sim: dome orientation conventions differ between renderers, so the dome may need a
    rotation about +Z to line its bright side up with the distant light; intensities need tuning to the camera
    exposure.
    """
    import math

    from pxr import Gf, Sdf, UsdGeom, UsdLux

    UsdGeom.Xform.Define(stage, path)
    dome = UsdLux.DomeLight.Define(stage, f"{path}/dome")
    dome.CreateTextureFileAttr(Sdf.AssetPath(str(SKY_HDR)))
    dome.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
    dome.CreateIntensityAttr(dome_intensity)
    sun = UsdLux.DistantLight.Define(stage, f"{path}/sun")
    sun.CreateIntensityAttr(sun_intensity)
    sun.CreateAngleAttr(0.53)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.92, 0.82))
    # a DistantLight shines along its local -Z; aim -Z at minus the sun direction
    e, a = math.radians(SKY_SUN_ELEVATION_DEG), math.radians(SKY_SUN_AZIMUTH_DEG)
    d = Gf.Vec3d(math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))
    rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), d)
    UsdGeom.Xformable(sun.GetPrim()).AddOrientOp().Set(Gf.Quatf(rot.GetQuat()))
    return [f"{path}/dome", f"{path}/sun"]
