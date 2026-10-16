"""Minimal ASCII OpenFOAM polyMesh / field reader-writer (no OpenFOAM Python deps).

Used for: 1D-isentropic initial fields, exit-plane integration (thrust, mdot)
and conservation checks. Only the ASCII formats written by blockMesh and
rhoCentralFoam (writeFormat ascii) are supported.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_NUM = r"[-+0-9.eE]+"


def _body(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return text[text.index("}", text.index("FoamFile")) + 1:]


def _list_block(text: str):
    """Return (n, content) for the first top-level 'N\\n(' ... ')' list."""
    m = re.search(r"(\d+)\s*\(", text)
    n = int(m.group(1))
    start = m.end()
    depth, i = 1, start
    while depth:
        c = text[i]
        depth += c == "("
        depth -= c == ")"
        i += 1
    return n, text[start:i - 1]


def read_points(case: Path) -> np.ndarray:
    n, c = _list_block(_body((case / "constant/polyMesh/points").read_text()))
    pts = np.array(re.findall(rf"\(({_NUM})\s+({_NUM})\s+({_NUM})\)", c), float)
    assert len(pts) == n
    return pts


def read_faces(case: Path) -> list[np.ndarray]:
    n, c = _list_block(_body((case / "constant/polyMesh/faces").read_text()))
    faces = [np.array(f.split(), int) for f in re.findall(r"\d+\s*\(([\d\s]+)\)", c)]
    assert len(faces) == n
    return faces


def read_labels(case: Path, name: str) -> np.ndarray:
    n, c = _list_block(_body((case / f"constant/polyMesh/{name}").read_text()))
    arr = np.array(c.split(), int)
    assert len(arr) == n
    return arr


def read_boundary(case: Path) -> dict[str, dict]:
    b = _body((case / "constant/polyMesh/boundary").read_text())
    out = {}
    for name, body in re.findall(r"(\w+)\s*\{([^}]*)\}", b):
        d = dict(re.findall(r"(\w+)\s+([^;]+);", body))
        out[name] = {"type": d["type"], "nFaces": int(d["nFaces"]), "startFace": int(d["startFace"])}
    return out


class Mesh:
    def __init__(self, case: Path):
        self.case = Path(case)
        self.points = read_points(self.case)
        self.faces = read_faces(self.case)
        self.owner = read_labels(self.case, "owner")
        self.boundary = read_boundary(self.case)
        self.n_cells = int(self.owner.max()) + 1
        nb = read_labels(self.case, "neighbour")
        if len(nb):
            self.n_cells = max(self.n_cells, int(nb.max()) + 1)
        self.face_centres, self.face_areas = self._face_geometry()

    def _face_geometry(self):
        fc = np.zeros((len(self.faces), 3))
        sf = np.zeros((len(self.faces), 3))
        for i, f in enumerate(self.faces):
            p = self.points[f]
            c0 = p.mean(0)
            a = np.zeros(3)
            ca = np.zeros(3)
            for k in range(len(p)):
                tri = np.cross(p[k] - c0, p[(k + 1) % len(p)] - c0) * 0.5
                a += tri
                ca += np.linalg.norm(tri) * (p[k] + p[(k + 1) % len(p)] + c0) / 3
            fc[i] = ca / max(np.linalg.norm(a), 1e-300) if np.linalg.norm(a) > 0 else c0
            sf[i] = a
        return fc, sf

    def cell_centres(self) -> np.ndarray:
        """Approximate cell centres: area-weighted mean of owning/neighbouring face centres."""
        acc = np.zeros((self.n_cells, 3))
        w = np.zeros(self.n_cells)
        nb = read_labels(self.case, "neighbour")
        mag = np.linalg.norm(self.face_areas, axis=1)
        np.add.at(acc, self.owner, self.face_centres * mag[:, None])
        np.add.at(w, self.owner, mag)
        np.add.at(acc, nb, self.face_centres[:len(nb)] * mag[:len(nb), None])
        np.add.at(w, nb, mag[:len(nb)])
        return acc / w[:, None]

    def patch_slice(self, name: str) -> slice:
        b = self.boundary[name]
        return slice(b["startFace"], b["startFace"] + b["nFaces"])


def read_field_patch(case: Path, time: str, field: str, patch: str, n_faces: int, internal_cells=None):
    """Values of `field` on `patch` (uniform or nonuniform). zeroGradient patches fall back to owner cells."""
    text = _body((Path(case) / time / field).read_text())
    bf = text[text.index("boundaryField"):]
    m = re.search(rf"\b{patch}\b\s*\{{", bf)
    i, depth = m.end(), 1
    while depth:
        depth += bf[i] == "{"
        depth -= bf[i] == "}"
        i += 1
    block = bf[m.end():i - 1]
    vec = "vector" in text[:200] or "Vector" in text[:400]
    if "value" not in block:
        if internal_cells is None:
            raise ValueError(f"{field}/{patch} has no value entry")
        return read_internal(case, time, field)[internal_cells]
    vm = re.search(r"\bvalue\s+(uniform|nonuniform)", block)
    if vm.group(1) == "uniform":
        rest = block[vm.end():]
        if rest.lstrip().startswith("("):
            v = np.array(re.match(rf"\s*\(({_NUM})\s+({_NUM})\s+({_NUM})\)", rest).groups(), float)
            return np.tile(v, (n_faces, 1))
        return np.full(n_faces, float(re.match(rf"\s*({_NUM})", rest).group(1)))
    _, c = _list_block(block[vm.end():])
    if vec:
        return np.array(re.findall(rf"\(({_NUM})\s+({_NUM})\s+({_NUM})\)", c), float)
    return np.array(c.split(), float)


def read_internal(case: Path, time: str, field: str) -> np.ndarray:
    text = _body((Path(case) / time / field).read_text())
    im = re.search(r"internalField\s+(uniform|nonuniform)", text)
    rest = text[im.end():]
    if im.group(1) == "uniform":
        raise ValueError("uniform internal field; use the patch value instead")
    _, c = _list_block(rest)
    if c.lstrip().startswith("("):
        return np.array(re.findall(rf"\(({_NUM})\s+({_NUM})\s+({_NUM})\)", c), float)
    return np.array(c.split(), float)


def set_internal(path: Path, values: np.ndarray, kind: str) -> None:
    """Replace a field's internalField with a nonuniform list (kind 'scalar' | 'vector')."""
    text = Path(path).read_text()
    if kind == "scalar":
        lst = "\n".join(f"{v:.9g}" for v in values)
    else:
        lst = "\n".join(f"({v[0]:.9g} {v[1]:.9g} {v[2]:.9g})" for v in values)
    new = f"internalField nonuniform List<{kind}>\n{len(values)}\n(\n{lst}\n)\n;"
    text = re.sub(r"internalField\s+uniform\s+[^;]+;", lambda _: new, text, count=1)
    Path(path).write_text(text)
