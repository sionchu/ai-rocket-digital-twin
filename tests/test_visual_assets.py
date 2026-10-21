"""Visual VTVL assets (assets/vtvl_v1): composition, dimensions and 'visual only' (no physics schemas)."""
from __future__ import annotations

import re

import pytest

pxr = pytest.importorskip("pxr")
from pxr import Usd, UsdGeom  # noqa: E402

from rocket_dt.isaac_env import visual_assets as va  # noqa: E402
from rocket_dt.isaac_env.layouts import LAYOUTS, layout_spec  # noqa: E402
from rocket_dt.schemas import load_vehicle  # noqa: E402

VEH = load_vehicle(va.REPO / "configs/vehicle_v0.yaml")


def _physics_schemas(prim) -> list[str]:
    return [s for s in prim.GetAppliedSchemas() if s.startswith("Physics") or "Physx" in s]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_vehicle_visual_reference(tmp_path, layout):
    stage = Usd.Stage.CreateNew(str(tmp_path / f"{layout}.usda"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, "/World/Vehicle")
    paths = va.author_vehicle_visuals(stage, "/World/Vehicle", layout, "conical_A")
    root = stage.GetPrimAtPath(paths[0])
    assert root.IsValid() and paths[0] == "/World/Vehicle/visual"

    # visual only: no physics API anywhere in the composed stage
    offenders = {str(p.GetPath()): _physics_schemas(p) for p in stage.Traverse() if _physics_schemas(p)}
    assert not offenders, offenders

    bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rng = bc.ComputeWorldBound(root).ComputeAlignedRange()
    assert abs(rng.GetMin()[2] - (-3.9)) < 0.05, rng  # foot pads rest on the physics contact plane
    assert abs(rng.GetMax()[2] - 3.5) < 0.01, rng

    hull = stage.GetPrimAtPath(paths[0] + "/airframe/hull")
    h = bc.ComputeWorldBound(hull).ComputeAlignedRange()
    r = VEH.mass.structure_radius_m
    for ax in (0, 1):
        assert abs(h.GetMax()[ax] - r) < 0.005 and abs(h.GetMin()[ax] + r) < 0.005, h
    assert abs((h.GetMax()[2] - h.GetMin()[2]) - VEH.mass.structure_length_m) < 0.005

    n_eng = layout_spec(layout, VEH).n_engines
    engines = [p for p in paths[1:] if re.search(r"/engines/engine_\d+$", p)]
    assert len(engines) == n_eng
    nozzles = [p for p in stage.Traverse() if p.GetName() == "nozzle" and p.IsA(UsdGeom.Mesh)]
    assert len(nozzles) == n_eng


def test_feet_match_physics_contacts(tmp_path):
    stage = Usd.Stage.CreateNew(str(tmp_path / "feet.usda"))
    va.author_vehicle_visuals(stage, "/V", "gimbal_rcs", "conical_A")
    bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    feet = stage.GetPrimAtPath("/V/visual/airframe/landing_legs/foot_pads")
    rng = bc.ComputeWorldBound(feet).ComputeAlignedRange()
    fx = max(abs(f[0]) for f in VEH.gear.foot_positions_body_m)
    assert abs(rng.GetMin()[2] - VEH.gear.foot_positions_body_m[0][2]) < 1e-3
    assert abs(rng.GetMax()[0] - fx) < 0.3 and abs(rng.GetMin()[0] + fx) < 0.3


def test_site_and_sky_files():
    assert va.PAD_USD.exists() and va.SKY_HDR.exists()
    stage = Usd.Stage.Open(str(va.PAD_USD))
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert not [p for p in stage.Traverse() if _physics_schemas(p)]
    bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    pad = bc.ComputeWorldBound(stage.GetPrimAtPath("/LandingSite/pad")).ComputeAlignedRange()
    assert pad.GetMax()[2] <= 0.0051
    ground = bc.ComputeWorldBound(stage.GetPrimAtPath("/LandingSite/ground")).ComputeAlignedRange()
    assert ground.GetSize()[0] >= 300 and ground.GetSize()[1] >= 300
    assert va.SKY_HDR.read_bytes()[:2] == b"#?"  # Radiance RGBE header
