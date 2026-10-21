"""Blender (4.0, Cycles CPU) preview renders + equirectangular sky bake for the VTVL v1 assets.

The scene is rebuilt from the USD files via dumps made by usd_to_blender_dump.py (this Blender build has no USD
importer and no numpy), so the previews show the USD geometry and UsdPreviewSurface materials as authored.

    blender -b --python scripts/assets/render_previews.py -- <shot> [--samples N] [--res WxH] [--ss 2]

No denoiser is available in this Blender build (no OpenImageDenoise), so the final previews are rendered at 2x
resolution (--ss 2) and box-downsampled by make_contact_sheet.py to reduce Monte Carlo noise.
    shots: hero | aft | layouts | site | sky
"""
from __future__ import annotations

import array
import json
import math
import os
import sys
import time

import bpy
from mathutils import Matrix, Vector

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "assets", "vtvl_v1")
DUMPS = os.path.join(OUT, "preview", "_dumps")

# lighting presets (degrees). Azimuth is measured CCW from +X; Nishita sun_rotation = 90 - azimuth.
GOLDEN = dict(elev=9.0, azim=105.0, sun_strength=6.0, sun_color=(1.0, 0.66, 0.42), sky_strength=0.12)
SKY_BAKE = dict(elev=20.0, azim=95.0)


# --------------------------------------------------------------------------------------------- scene
def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.use_denoising = False  # this Blender build has no OpenImageDenoise
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.015
    sc.cycles.max_bounces = 6
    sc.cycles.diffuse_bounces = 3
    sc.cycles.glossy_bounces = 3
    sc.cycles.sample_clamp_indirect = 4.0
    sc.cycles.blur_glossy = 0.6
    sc.render.threads_mode = "AUTO"
    sc.view_settings.view_transform = "AgX"
    try:
        sc.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        pass
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    return sc


def setup_world(sc, elev, azim, strength, sun_disc=False, ground_fill=None):
    w = bpy.data.worlds.new("sky")
    sc.world = w
    w.use_nodes = True
    nt = w.node_tree
    bg = nt.nodes["Background"]
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "NISHITA"
    sky.sun_disc = sun_disc
    sky.sun_elevation = math.radians(elev)
    sky.sun_rotation = math.radians(90.0 - azim)
    sky.altitude = 600.0  # high desert
    sky.air_density = 1.0
    sky.dust_density = 2.2  # a bit of desert haze
    sky.ozone_density = 1.0
    src = sky.outputs[0]
    if ground_fill is not None:  # lower hemisphere = dusty ground bounce instead of black
        tc = nt.nodes.new("ShaderNodeTexCoord")
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(tc.outputs["Generated"], sep.inputs[0])
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["From Min"].default_value = 0.0
        mr.inputs["From Max"].default_value = -0.03
        nt.links.new(sep.outputs["Z"], mr.inputs["Value"])
        mix = nt.nodes.new("ShaderNodeMixRGB")
        nt.links.new(mr.outputs[0], mix.inputs["Fac"])
        nt.links.new(sky.outputs[0], mix.inputs[1])
        mix.inputs[2].default_value = (*ground_fill, 1.0)
        src = mix.outputs[0]
    nt.links.new(src, bg.inputs["Color"])
    bg.inputs["Strength"].default_value = strength
    return w


def sun_dir(elev, azim):
    e, a = math.radians(elev), math.radians(azim)
    return Vector((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)))


def add_sun(sc, elev, azim, strength, color, angle_deg=0.6):
    ld = bpy.data.lights.new("sun", "SUN")
    ld.energy = strength
    ld.color = color
    ld.angle = math.radians(angle_deg)
    ob = bpy.data.objects.new("sun", ld)
    sc.collection.objects.link(ob)
    ob.rotation_euler = sun_dir(elev, azim).to_track_quat("Z", "Y").to_euler()
    return ob


def add_camera(sc, loc, target, lens=50.0, clip_end=12000.0):
    cd = bpy.data.cameras.new("cam")
    cd.lens = lens
    cd.clip_start = 0.05
    cd.clip_end = clip_end
    cd.sensor_width = 36.0
    ob = bpy.data.objects.new("cam", cd)
    sc.collection.objects.link(ob)
    ob.location = Vector(loc)
    ob.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    sc.camera = ob
    return ob


# --------------------------------------------------------------------------------------------- USD dump import
_MAT_CACHE: dict = {}
_IMG_CACHE: dict = {}


def _image(path, non_color):
    key = (path, non_color)
    if key not in _IMG_CACHE:
        img = bpy.data.images.load(path, check_existing=True)
        img.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
        _IMG_CACHE[key] = img
    return _IMG_CACHE[key]


def make_material(name, spec):
    """UsdPreviewSurface (as dumped) -> Principled BSDF."""
    if name in _MAT_CACHE:
        return _MAT_CACHE[name]
    m = bpy.data.materials.new(name.split("/")[-1])
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    uvn = nt.nodes.new("ShaderNodeUVMap")
    uvn.uv_map = "st"

    def tex_node(path, non_color):
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = _image(path, non_color)
        t.extension = "REPEAT"
        t.interpolation = "Cubic" if not non_color else "Linear"
        nt.links.new(uvn.outputs["UV"], t.inputs["Vector"])
        return t

    def scalar(inp_name, val):
        if isinstance(val, dict):
            t = tex_node(val["tex"], True)
            sepn = nt.nodes.new("ShaderNodeSeparateColor")
            nt.links.new(t.outputs["Color"], sepn.inputs["Color"])
            ch = {"r": "Red", "g": "Green", "b": "Blue"}.get(val["channel"], "Red")
            nt.links.new(sepn.outputs[ch], bsdf.inputs[inp_name])
        elif val is not None:
            bsdf.inputs[inp_name].default_value = val

    dc = spec.get("diffuseColor")
    if isinstance(dc, dict):
        t = tex_node(dc["tex"], False)
        nt.links.new(t.outputs["Color"], bsdf.inputs["Base Color"])
    elif dc is not None:
        bsdf.inputs["Base Color"].default_value = (*dc, 1.0)
    scalar("Roughness", spec.get("roughness"))
    scalar("Metallic", spec.get("metallic"))
    nm = spec.get("normal")
    if isinstance(nm, dict):
        t = tex_node(nm["tex"], True)
        nmap = nt.nodes.new("ShaderNodeNormalMap")
        nmap.uv_map = "st"
        nt.links.new(t.outputs["Color"], nmap.inputs["Color"])
        nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    _MAT_CACHE[name] = m
    return m


def load_dump(prefix, offset=(0, 0, 0), yaw_deg=0.0, collection=None, skip=(), pitch_deg=0.0):
    meta = json.load(open(prefix + ".json"))
    blob = open(prefix + ".bin", "rb").read()

    def arr(rec, code):
        a = array.array(code)
        a.frombytes(blob[rec[0]:rec[0] + rec[1]])
        return a

    col = collection or bpy.context.scene.collection
    R = Matrix.Rotation(math.radians(yaw_deg), 4, "Z") @ Matrix.Rotation(math.radians(pitch_deg), 4, "X")
    T = Matrix.Translation(Vector(offset))
    objs = []
    for mrec in meta["meshes"]:
        if any(s in mrec["path"] for s in skip):
            continue
        P, N, UV = arr(mrec["P"], "f"), arr(mrec["N"], "f"), arr(mrec["UV"], "f")
        counts, idx = arr(mrec["counts"], "i"), arr(mrec["idx"], "i")
        me = bpy.data.meshes.new(mrec["path"].split("/")[-1])
        nv, nf = mrec["n_points"], mrec["n_faces"]
        me.vertices.add(nv)
        me.vertices.foreach_set("co", P)
        me.loops.add(len(idx))
        me.loops.foreach_set("vertex_index", idx)
        me.polygons.add(nf)
        starts = array.array("i", [0] * nf)
        s = 0
        for i, c in enumerate(counts):
            starts[i] = s
            s += c
        me.polygons.foreach_set("loop_start", starts)
        try:
            me.polygons.foreach_set("loop_total", counts)
        except (AttributeError, TypeError, RuntimeError):
            pass
        me.update(calc_edges=True)
        uvl = me.uv_layers.new(name="st")
        luv = array.array("f", [0.0]) * (2 * len(idx))
        for li, vi in enumerate(idx):
            luv[2 * li] = UV[2 * vi]
            luv[2 * li + 1] = UV[2 * vi + 1]
        uvl.data.foreach_set("uv", luv)
        me.polygons.foreach_set("use_smooth", array.array("b", [1]) * nf)
        if hasattr(me, "use_auto_smooth"):
            me.use_auto_smooth = True
        normals = [(N[3 * i], N[3 * i + 1], N[3 * i + 2]) for i in range(nv)]
        me.normals_split_custom_set_from_vertices(normals)
        if mrec["material"]:
            me.materials.append(make_material(mrec["material"], meta["materials"][mrec["material"]]))
        me.validate(clean_customdata=False)
        ob = bpy.data.objects.new(me.name, me)
        ob.matrix_world = T @ R
        col.objects.link(ob)
        objs.append(ob)
    return objs


def dump_path(name):
    return os.path.join(DUMPS, name)


# --------------------------------------------------------------------------------------------- shots
VEH_Z = 3.9  # body origin height when the feet (z=-3.9) rest on the ground


def lit_scene(sc, preset=GOLDEN, site=True):
    setup_world(sc, preset["elev"], preset["azim"], preset["sky_strength"])
    add_sun(sc, preset["elev"], preset["azim"], preset["sun_strength"], preset["sun_color"])
    if site:
        load_dump(dump_path("pad"))


def shot_hero(sc):
    lit_scene(sc)
    load_dump(dump_path("vehicle_gimbal_rcs_conical_A"), (0, 0, VEH_Z))
    add_camera(sc, (-13.6, 9.2, 1.15), (0.0, 0.0, 3.45), lens=38)


def shot_aft(sc):
    lit_scene(sc)
    load_dump(dump_path("vehicle_gimbal_rcs_conical_A"), (0, 0, VEH_Z))
    add_camera(sc, (-2.35, 2.05, 0.16), (0.05, -0.05, 0.62), lens=24)


def shot_layouts(sc):
    lit_scene(sc, dict(GOLDEN, elev=14.0, azim=-115.0, sun_strength=3.6, sun_color=(1.0, 0.80, 0.62), sky_strength=0.35))
    names = ["gimbal_rcs", "tri_diff", "quad_diff", "coldgas"]
    alt = 14.0
    for i, n in enumerate(names):  # pitched so the aft end faces the camera (seen from below, 3/4)
        load_dump(dump_path(f"vehicle_{n}_conical_A"), (-6.9 + 4.6 * i, 0.0, alt), yaw_deg=0.0, pitch_deg=-62.0)
    add_camera(sc, (0.0, -23.5, alt - 1.6), (0.0, 0.0, alt - 0.6), lens=38)


def shot_nose(sc):  # inspection close-up of the coldgas collar / RCS pods / grid fins (not a deliverable)
    lit_scene(sc)
    load_dump(dump_path("vehicle_coldgas_conical_A"), (0, 0, VEH_Z))
    add_camera(sc, (-3.2, 2.6, 7.2), (0.0, 0.0, 6.4), lens=35)


def shot_site(sc):
    lit_scene(sc)
    load_dump(dump_path("vehicle_gimbal_rcs_conical_A"), (0, 0, VEH_Z))
    add_camera(sc, (-38.0, 22.0, 9.0), (2.0, -1.0, 2.0), lens=32)


def shot_sky(sc, samples):
    """Equirectangular sky bake (no sun disc; add a DistantLight at SKY_BAKE elev/azim in the simulator)."""
    setup_world(sc, SKY_BAKE["elev"], SKY_BAKE["azim"], 1.0, sun_disc=False, ground_fill=(0.36, 0.30, 0.23))
    cd = bpy.data.cameras.new("pano")
    cd.type = "PANO"
    try:
        cd.panorama_type = "EQUIRECTANGULAR"
    except AttributeError:
        cd.cycles.panorama_type = "EQUIRECTANGULAR"
    ob = bpy.data.objects.new("pano", cd)
    sc.collection.objects.link(ob)
    ob.rotation_euler = (math.pi / 2, 0, -math.pi / 2)  # image centre = +X, u=0.25 = +Y (Blender env convention)
    sc.camera = ob
    sc.render.resolution_x, sc.render.resolution_y = 2048, 1024
    sc.cycles.samples = samples
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.render.image_settings.file_format = "HDR"
    sc.render.filepath = os.path.join(OUT, "sky_2k.hdr")


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    shot = argv[0] if argv else "hero"
    samples = int(argv[argv.index("--samples") + 1]) if "--samples" in argv else 128
    res = argv[argv.index("--res") + 1] if "--res" in argv else "1280x720"
    ss = int(argv[argv.index("--ss") + 1]) if "--ss" in argv else 1  # supersample; make_contact_sheet downsizes
    sc = reset_scene()
    t0 = time.time()
    if shot == "sky":
        shot_sky(sc, min(samples, 32))
    else:
        {"hero": shot_hero, "aft": shot_aft, "nose": shot_nose, "layouts": shot_layouts, "site": shot_site}[shot](sc)
        sc.cycles.samples = samples
        sc.render.resolution_x, sc.render.resolution_y = (ss * int(v) for v in res.split("x"))
        name = f"{shot}_raw" if shot == "layouts" else shot
        sc.render.filepath = os.path.join(OUT, "preview", f"{name}_ss{ss}.png" if ss > 1 else f"{name}.png")
    t1 = time.time()
    bpy.ops.render.render(write_still=True)
    print(f"RENDER {shot}: build {t1 - t0:.1f}s render {time.time() - t1:.1f}s samples={sc.cycles.samples} -> {sc.render.filepath}")


if __name__ == "__main__":
    main()
