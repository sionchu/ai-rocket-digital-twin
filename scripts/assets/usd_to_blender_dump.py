"""Flatten a composed USD stage (meshes + UsdPreviewSurface materials) into a JSON + binary dump that Blender's
bundled Python (which has neither pxr nor numpy here) can load. Used only for the preview renders, so the previews
show exactly what the USD files contain.

    python3 scripts/assets/usd_to_blender_dump.py <in.usd> <out_prefix>   ->  out_prefix.json / out_prefix.bin
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdShade


def _input_value(shader: UsdShade.Shader, name: str):
    inp = shader.GetInput(name)
    if not inp:
        return None
    if inp.HasConnectedSource():
        src, out_name, _ = inp.GetConnectedSource()
        s = UsdShade.Shader(src.GetPrim())
        f = s.GetInput("file").Get()
        return {"tex": f.resolvedPath, "channel": out_name}
    v = inp.Get()
    try:
        return list(map(float, v))
    except TypeError:
        return float(v)


def dump(usd_path: str, out_prefix: str):
    st = Usd.Stage.Open(usd_path)
    xc = UsdGeom.XformCache()
    meshes, blobs, off = [], [], 0
    mats = {}

    def add(arr, dtype):
        nonlocal off
        b = np.ascontiguousarray(arr, dtype).tobytes()
        blobs.append(b)
        rec = [off, len(b)]
        off += len(b)
        return rec

    for prim in st.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        m = UsdGeom.Mesh(prim)
        M = np.array(xc.GetLocalToWorldTransform(prim), float)  # row-vector convention
        P = np.array(m.GetPointsAttr().Get(), float)
        P = P @ M[:3, :3] + M[3, :3]
        N = np.array(m.GetNormalsAttr().Get(), float)
        Nm = np.linalg.inv(M[:3, :3]).T
        N = N @ Nm
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
        uv = np.array(UsdGeom.PrimvarsAPI(prim).GetPrimvar("st").Get(), float)
        counts = np.array(m.GetFaceVertexCountsAttr().Get(), np.int32)
        idx = np.array(m.GetFaceVertexIndicesAttr().Get(), np.int32)
        mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        mname = str(mat.GetPath()) if mat else ""
        if mat and mname not in mats:
            surf = mat.ComputeSurfaceSource()[0]
            mats[mname] = {k: _input_value(surf, k) for k in ("diffuseColor", "roughness", "metallic", "normal", "emissiveColor")}
        meshes.append(dict(path=str(prim.GetPath()), material=mname, P=add(P, "<f4"), N=add(N, "<f4"), UV=add(uv, "<f4"),
                           counts=add(counts, "<i4"), idx=add(idx, "<i4"), n_points=len(P), n_faces=len(counts)))
    Path(out_prefix + ".bin").write_bytes(b"".join(blobs))
    Path(out_prefix + ".json").write_text(json.dumps({"source": usd_path, "meshes": meshes, "materials": mats}))


if __name__ == "__main__":
    dump(sys.argv[1], sys.argv[2])
