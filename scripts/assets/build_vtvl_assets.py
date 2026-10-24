"""Build the VTVL v1 visual USD assets (vehicle per layout x engine, landing pad + ground) with pxr + numpy.

    cd <repo> && PYTHONPATH=src python3 scripts/assets/build_vtvl_assets.py [--skip-textures]

Writes assets/vtvl_v1/{parts/*.usdc, vehicle_<layout>_<engine>.usd, pad.usd, textures/*}. Visual only: no
UsdPhysics schema is applied anywhere. Dimensions are read from configs/vehicle_v0.yaml and the engine configs.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))

import vtvl_geometry as G  # noqa: E402
import vtvl_spec as S  # noqa: E402

OUT = REPO / "assets/vtvl_v1"
PARTS = OUT / "parts"
LAYOUTS = ("gimbal_rcs", "tri_diff", "quad_diff", "coldgas")
ENGINES = ("conical_A", "bell_B", "bell_C")
NOZZLE_VIS_SCALE = 4.0  # same as rocket_env.NOZZLE_VIS_SCALE

# --------------------------------------------------------------------------------------------- materials
# name -> UsdPreviewSurface inputs. Texture paths are relative to assets/vtvl_v1/textures.
MATERIALS = {
    "hull": dict(tex="hull_albedo.jpg", rough_tex="hull_roughness.jpg", metal_tex="hull_metallic.png", normal_tex="hull_normal.jpg"),
    "skirt_inner": dict(tex="soot_albedo.jpg", rough_tex="soot_roughness.jpg", normal_tex="soot_normal.jpg"),
    "heat_shield": dict(tex="soot_albedo.jpg", rough_tex="soot_roughness.jpg", normal_tex="soot_normal.jpg"),
    "leg_carbon": dict(color=(0.028, 0.029, 0.032), rough=0.38),
    "chrome": dict(color=(0.80, 0.80, 0.82), rough=0.12, metal=1.0),
    "fitting": dict(color=(0.30, 0.31, 0.33), rough=0.42, metal=1.0),
    "foot": dict(color=(0.06, 0.06, 0.065), rough=0.6, metal=0.3),
    "grid_fin": dict(color=(0.42, 0.40, 0.38), rough=0.38, metal=1.0),
    "paint_white": dict(color=(0.68, 0.675, 0.66), rough=0.38),
    "rcs_nozzle": dict(color=(0.30, 0.28, 0.27), rough=0.35, metal=1.0),
    "soot": dict(color=(0.02, 0.02, 0.02), rough=0.9),
    "collar": dict(color=(0.10, 0.105, 0.115), rough=0.45, metal=0.8),
    "nozzle": dict(tex="nozzle_albedo.jpg", rough_tex="nozzle_roughness.jpg", metal=1.0, normal_tex="nozzle_normal.jpg"),
    "nozzle_band": dict(color=(0.36, 0.36, 0.38), rough=0.4, metal=1.0),
    "chamber": dict(color=(0.45, 0.30, 0.20), rough=0.4, metal=1.0),
    "powerhead": dict(color=(0.55, 0.56, 0.58), rough=0.35, metal=1.0),
    "plumbing": dict(color=(0.62, 0.60, 0.56), rough=0.3, metal=1.0),
    "boot": dict(color=(0.035, 0.033, 0.03), rough=0.75),
    "lakebed": dict(tex="lakebed_albedo.jpg", rough_tex="lakebed_roughness.jpg", normal_tex="lakebed_normal.jpg"),
    "ridge": dict(tex="ridge_albedo.jpg", rough=0.95),
    "pad_concrete": dict(tex="pad_albedo.jpg", rough_tex="pad_roughness.jpg", normal_tex="pad_normal.jpg"),
    "fixture": dict(color=(0.05, 0.05, 0.05), rough=0.5, metal=0.5),
    "lens_amber": dict(color=(0.55, 0.30, 0.04), rough=0.15),
    "barrier_concrete": dict(color=(0.33, 0.315, 0.29), rough=0.88),
}


def vehicle_facts():
    """Physics dimensions from the repo configs (so the visuals cannot drift from the physics model)."""
    from rocket_dt.isaac_env.layouts import layout_spec
    from rocket_dt.propulsion.geometry import build_contour
    from rocket_dt.schemas import load_engine, load_vehicle

    veh = load_vehicle(REPO / "configs/vehicle_v0.yaml")
    facts = dict(
        radius=veh.mass.structure_radius_m, length=veh.mass.structure_length_m,
        pivot=tuple(veh.layout.main_engine.gimbal_pivot_body_m),
        throat_below_pivot=veh.mass.engine_throat_below_pivot_m,
        feet=[tuple(f) for f in veh.gear.foot_positions_body_m],
        layouts={L: layout_spec(L, veh) for L in LAYOUTS},
        contours={},
    )
    for e in ENGINES:
        c = build_contour(load_engine(REPO / f"configs/engines/{e}.yaml"))
        facts["contours"][e] = (c.x_m - c.x_m[c.throat_index], c.r_m)
    assert abs(facts["radius"] - S.HULL_R) < 1e-9 and abs(facts["length"] - S.HULL_LEN) < 1e-9
    return facts


# --------------------------------------------------------------------------------------------- USD helpers
def _pxr():
    from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdShade, Vt
    return Gf, Kind, Sdf, Usd, UsdGeom, UsdShade, Vt


def new_stage(path: Path, root: str, kind="component"):
    Gf, Kind, Sdf, Usd, UsdGeom, UsdShade, Vt = _pxr()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    st = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    rp = UsdGeom.Xform.Define(st, f"/{root}").GetPrim()
    st.SetDefaultPrim(rp)
    Usd.ModelAPI(rp).SetKind(kind)
    st.SetMetadata("comment", "VTVL v1 visual proxy (rocket_dt). Visual only: no physics schemas.")
    return st


def define_material(st, root: str, name: str, tex_rel_prefix: str):
    """UsdPreviewSurface material under /<root>/Looks/<name>. tex_rel_prefix: anchored path to textures/."""
    Gf, Kind, Sdf, Usd, UsdGeom, UsdShade, Vt = _pxr()
    spec = MATERIALS[name]
    mp = f"/{root}/Looks/{name}"
    mat = UsdShade.Material.Define(st, mp)
    pbr = UsdShade.Shader.Define(st, mp + "/PreviewSurface")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    reader = None

    def tex(fname, cs, prefix):
        nonlocal reader
        if reader is None:
            reader = UsdShade.Shader.Define(st, mp + "/stReader")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        t = UsdShade.Shader.Define(st, f"{mp}/{prefix}Tex")
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(tex_rel_prefix + fname))
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(cs)
        return t

    if "tex" in spec:
        t = tex(spec["tex"], "sRGB", "diffuse")
        t.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(t.ConnectableAPI(), "rgb")
    else:
        pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*spec["color"]))
    if "rough_tex" in spec:
        t = tex(spec["rough_tex"], "raw", "roughness")
        t.CreateOutput("r", Sdf.ValueTypeNames.Float)
        pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(t.ConnectableAPI(), "r")
    else:
        pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(spec.get("rough", 0.5)))
    if "metal_tex" in spec:
        t = tex(spec["metal_tex"], "raw", "metallic")
        t.CreateOutput("r", Sdf.ValueTypeNames.Float)
        pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).ConnectToSource(t.ConnectableAPI(), "r")
    else:
        pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(spec.get("metal", 0.0)))
    if "normal_tex" in spec:
        t = tex(spec["normal_tex"], "raw", "normal")
        t.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
        t.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
        t.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        pbr.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(t.ConnectableAPI(), "rgb")
    return mat


def write_mesh(st, path: str, mesh: G.Mesh, mat, double_sided=False):
    Gf, Kind, Sdf, Usd, UsdGeom, UsdShade, Vt = _pxr()
    P, N, UV, counts, idx = mesh.arrays()
    m = UsdGeom.Mesh.Define(st, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(P.astype(np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(counts.astype(np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(idx.astype(np.int32)))
    m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(N.astype(np.float32)))
    m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    m.CreateDoubleSidedAttr(bool(double_sided))
    pv = UsdGeom.PrimvarsAPI(m.GetPrim()).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    pv.Set(Vt.Vec2fArray.FromNumpy(UV.astype(np.float32)))
    m.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, P.min(0))), Gf.Vec3f(*map(float, P.max(0)))]))
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    return m


class PartWriter:
    def __init__(self, path: Path, root: str, tex_prefix: str, kind="component"):
        self.st = new_stage(path, root, kind)
        self.root = root
        self.prefix = tex_prefix
        self.mats = {}

    def mat(self, name):
        if name not in self.mats:
            self.mats[name] = define_material(self.st, self.root, name, self.prefix)
        return self.mats[name]

    def mesh(self, name, mesh, mat, group=None, double_sided=False):
        if mesh.n == 0:
            return
        parent = f"/{self.root}" + (f"/{group}" if group else "")
        if group:
            from pxr import UsdGeom
            UsdGeom.Xform.Define(self.st, parent)
        write_mesh(self.st, f"{parent}/{name}", mesh, self.mat(mat), double_sided)

    def save(self):
        self.st.GetRootLayer().Save()


# --------------------------------------------------------------------------------------------- parts
def build_airframe(facts):
    w = PartWriter(PARTS / "airframe.usdc", "airframe", "../textures/")
    w.mesh("hull", G.hull_mesh(), "hull")
    w.mesh("weld_rings", G.weld_rings(), "hull")
    w.mesh("skirt_inner", G.skirt_inner(), "skirt_inner")
    w.mesh("heat_shield", G.heat_shield(), "heat_shield")
    legs = G.leg_parts(facts["feet"])
    w.mesh("struts", legs["struts"], "leg_carbon", "landing_legs")
    w.mesh("pistons", legs["rods"], "chrome", "landing_legs")
    w.mesh("fittings", legs["fittings"], "fitting", "landing_legs")
    w.mesh("foot_pads", legs["feet"], "foot", "landing_legs")
    fins = G.Mesh()
    for a in S.FIN_ANGLES_DEG:
        fins.extend(G.grid_fin(math.radians(a)))
    w.mesh("grid_fins", fins, "grid_fin")
    tray, brackets = G.raceway()
    w.mesh("raceway", tray, "paint_white")
    w.mesh("raceway_brackets", brackets, "fitting")
    pods, noz, noz_in = G.roll_pods(facts["layouts"]["gimbal_rcs"].rcs)
    w.mesh("pods", pods, "paint_white", "roll_rcs")
    w.mesh("nozzles", noz, "rcs_nozzle", "roll_rcs")
    w.mesh("nozzle_interiors", noz_in, "soot", "roll_rcs")
    w.save()


def build_coldgas_ring(facts):
    w = PartWriter(PARTS / "coldgas_ring.usdc", "coldgas_ring", "../textures/")
    ring, housings, noz, noz_in = G.coldgas_ring(facts["layouts"]["coldgas"].rcs)
    w.mesh("collar", ring, "collar")
    w.mesh("valve_housings", housings, "fitting")
    w.mesh("nozzles", noz, "rcs_nozzle")
    w.mesh("nozzle_interiors", noz_in, "soot")
    w.save()


def engine_variant(layout: str) -> tuple[str, float, bool]:
    k = {"gimbal_rcs": 1, "coldgas": 1, "tri_diff": 3, "quad_diff": 4}[layout]
    tvc = layout == "gimbal_rcs"
    return (f"s{k}" + ("_tvc" if tvc else "")), math.sqrt(1.0 / k), tvc


def build_engine(facts, engine: str, tag: str, sc: float, tvc: bool):
    xs, rs = facts["contours"][engine]
    z_throat_rel = -facts["throat_below_pivot"]
    parts, (zexit, r_e) = G.engine_parts(xs, rs, NOZZLE_VIS_SCALE, z_throat_rel, sc, tvc)
    w = PartWriter(PARTS / f"engine_{engine}_{tag}.usdc", "engine", "../textures/")
    w.mesh("nozzle", parts["nozzle"], "nozzle")
    w.mesh("nozzle_interior", parts["nozzle_inner"], "soot")
    w.mesh("nozzle_bands", parts["nozzle_bands"], "nozzle_band")
    w.mesh("chamber", parts["powerhead"], "chamber")
    w.mesh("plumbing", parts["plumbing"], "plumbing")
    w.mesh("boot", parts["boot"], "boot")
    if tvc:
        w.mesh("tvc_actuators", parts["tvc_actuators"], "fitting")
        w.mesh("tvc_rods", parts["tvc_rods"], "chrome")
    w.st.GetDefaultPrim().SetCustomDataByKey("nozzle_exit_z_rel_pivot", zexit)
    w.st.GetDefaultPrim().SetCustomDataByKey("nozzle_exit_radius", r_e)
    w.save()


def build_vehicle(facts, layout: str, engine: str) -> Path:
    Gf, Kind, Sdf, Usd, UsdGeom, UsdShade, Vt = _pxr()
    path = OUT / f"vehicle_{layout}_{engine}.usd"
    st = new_stage(path, "Vehicle")
    root = st.GetDefaultPrim()
    root.SetCustomDataByKey("rocket_dt:layout", layout)
    root.SetCustomDataByKey("rocket_dt:engine", engine)
    root.SetCustomDataByKey("rocket_dt:visual_only", True)
    root.SetDocumentation("Visual proxy of the rocket_dt VTVL vehicle (body frame, origin mid-length, +Z nose). "
                          "No physics schemas; attach under the physics rigid body.")
    af = st.DefinePrim("/Vehicle/airframe", "Xform")
    af.GetReferences().AddReference("./parts/airframe.usdc")
    if layout == "coldgas":
        cg = st.DefinePrim("/Vehicle/coldgas_ring", "Xform")
        cg.GetReferences().AddReference("./parts/coldgas_ring.usdc")
    spec = facts["layouts"][layout]
    tag, sc, tvc = engine_variant(layout)
    UsdGeom.Xform.Define(st, "/Vehicle/engines")
    for i, (pos, share) in enumerate(zip(spec.engine_pos_b, spec.engine_share)):
        assert abs(math.sqrt(share) - sc) < 1e-9
        x = UsdGeom.Xform.Define(st, f"/Vehicle/engines/engine_{i}")
        x.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
        x.AddScaleOp().Set(Gf.Vec3f(sc, sc, sc))
        x.GetPrim().GetReferences().AddReference(f"./parts/engine_{engine}_{tag}.usdc")
    st.GetRootLayer().Save()
    return path


def build_pad() -> Path:
    path = OUT / "pad.usd"
    w = PartWriter(path, "LandingSite", "./textures/", kind="assembly")
    w.st.GetDefaultPrim().SetDocumentation(
        "Visual-only landing site: concrete pad (r=7 m, top z=0.004) + dry-lakebed ground (r=2.5 km, flat to 60 m) "
        "+ distant ridges. Keep the physics ground plane separate at z=0.")
    w.mesh("ground", G.ground_mesh(), "lakebed")
    w.mesh("ridges", G.ridges_mesh(), "ridge")
    w.mesh("pad", G.pad_mesh(), "pad_concrete")
    fixtures, lenses, barriers = G.pad_props()
    w.mesh("edge_light_bodies", fixtures, "fixture", "props")
    w.mesh("edge_light_lenses", lenses, "lens_amber", "props")
    w.mesh("blast_barriers", barriers, "barrier_concrete", "props")
    w.save()
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-textures", action="store_true")
    a = ap.parse_args()
    if not a.skip_textures:
        import vtvl_textures
        vtvl_textures.build_all(OUT / "textures")
    facts = vehicle_facts()
    build_airframe(facts)
    build_coldgas_ring(facts)
    variants = {engine_variant(L)[:2] + (engine_variant(L)[2],) for L in LAYOUTS}
    for e in ENGINES:
        for tag, sc, tvc in variants:
            build_engine(facts, e, tag, sc, tvc)
    made = [build_vehicle(facts, L, e) for L in LAYOUTS for e in ENGINES]
    made.append(build_pad())
    (OUT / "manifest.json").write_text(json.dumps({
        "vehicles": {f"{L}/{e}": f"vehicle_{L}_{e}.usd" for L in LAYOUTS for e in ENGINES},
        "pad": "pad.usd", "sky": "sky_2k.hdr"}, indent=2))
    for p in made:
        print("wrote", p.relative_to(REPO))


if __name__ == "__main__":
    main()
