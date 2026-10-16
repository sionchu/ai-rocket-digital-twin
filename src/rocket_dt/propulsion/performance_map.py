"""EnginePerformanceMap: the only thrust interface the flight track may use.

Bounded bilinear interpolation over (throttle, ambient_pressure_pa).
Queries outside the tabulated box raise OutOfMapRange (no extrapolation).
Cells touching an invalid node (e.g. separation risk) return valid=False.
The same class will hold CFD-derived maps; only `provenance` changes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rocket_dt.propulsion import nozzle1d
from rocket_dt.schemas import SCHEMA_VERSION, EngineDesign, stable_hash

VALUE_KEYS = ("thrust_n", "mass_flow_kg_s", "isp_s")


class OutOfMapRange(ValueError):
    pass


@dataclass
class EnginePerformanceMap:
    engine_design_id: str
    throttle: np.ndarray
    ambient_pressure_pa: np.ndarray
    values: dict[str, np.ndarray]  # each shape (n_throttle, n_pa)
    valid: np.ndarray  # bool (n_throttle, n_pa)
    provenance: dict = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    units: str = "SI"
    interpolation: str = "bounded_bilinear"
    allow_extrapolation: bool = False

    # ------------------------------------------------------------------ query
    def _locate(self, axis: np.ndarray, v: float, name: str):
        if not (axis[0] - 1e-9 <= v <= axis[-1] + 1e-9):
            raise OutOfMapRange(f"{name}={v} outside [{axis[0]}, {axis[-1]}] of map {self.engine_design_id}")
        i = int(np.clip(np.searchsorted(axis, v) - 1, 0, len(axis) - 2))
        t = (v - axis[i]) / (axis[i + 1] - axis[i])
        return i, float(np.clip(t, 0.0, 1.0))

    def lookup(self, throttle: float, ambient_pressure_pa: float) -> tuple[float, float, float, bool]:
        i, s = self._locate(self.throttle, throttle, "throttle")
        j, t = self._locate(self.ambient_pressure_pa, ambient_pressure_pa, "ambient_pressure_pa")
        w = np.array([(1 - s) * (1 - t), (1 - s) * t, s * (1 - t), s * t])
        out = []
        for k in VALUE_KEYS:
            a = self.values[k]
            out.append(float(w @ np.array([a[i, j], a[i, j + 1], a[i + 1, j], a[i + 1, j + 1]])))
        # a node with zero weight does not contaminate validity
        nodes = [self.valid[i, j], self.valid[i, j + 1], self.valid[i + 1, j], self.valid[i + 1, j + 1]]
        ok = all(v or wk < 1e-12 for v, wk in zip(nodes, w))
        return out[0], out[1], out[2], ok

    def throttle_for_thrust(self, thrust_n: float, ambient_pressure_pa: float) -> float:
        """Invert F(throttle) at fixed Pa by bisection on the interpolant (monotone in throttle)."""
        lo, hi = float(self.throttle[0]), float(self.throttle[-1])
        f_lo = self.lookup(lo, ambient_pressure_pa)[0]
        f_hi = self.lookup(hi, ambient_pressure_pa)[0]
        if thrust_n <= f_lo:
            return lo
        if thrust_n >= f_hi:
            return hi
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if self.lookup(mid, ambient_pressure_pa)[0] < thrust_n:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version, "units": self.units,
            "engine_design_id": self.engine_design_id,
            "dimensions": ["throttle", "ambient_pressure_pa"],
            "interpolation": self.interpolation, "allow_extrapolation": self.allow_extrapolation,
            "throttle": self.throttle.tolist(), "ambient_pressure_pa": self.ambient_pressure_pa.tolist(),
            "values": {k: v.tolist() for k, v in self.values.items()},
            "valid": self.valid.tolist(), "provenance": self.provenance,
        }

    def content_hash(self) -> str:
        d = self.to_dict()
        d.pop("provenance")
        return stable_hash(d)

    def save(self, path: str | Path) -> str:
        d = self.to_dict()
        d["map_hash"] = self.content_hash()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(d, indent=1))
        return d["map_hash"]

    @classmethod
    def load(cls, path: str | Path) -> "EnginePerformanceMap":
        d = json.loads(Path(path).read_text())
        if d.get("allow_extrapolation"):
            raise ValueError("extrapolating maps are not allowed")
        m = cls(d["engine_design_id"], np.array(d["throttle"]), np.array(d["ambient_pressure_pa"]),
                {k: np.array(v) for k, v in d["values"].items()}, np.array(d["valid"], dtype=bool),
                d.get("provenance", {}))
        if "map_hash" in d and d["map_hash"] != m.content_hash():
            raise ValueError("map_hash mismatch: file modified or corrupted")
        return m


def build_ideal_map(design: EngineDesign, n_throttle: int = 13,
                    pa_range=(50_000.0, 105_000.0), n_pa: int = 12) -> EnginePerformanceMap:
    thr = np.linspace(design.throttle_min, design.throttle_max, n_throttle)
    pa = np.linspace(*pa_range, n_pa)
    vals = {k: np.zeros((n_throttle, n_pa)) for k in VALUE_KEYS}
    valid = np.zeros((n_throttle, n_pa), dtype=bool)
    reasons = {}
    for i, t in enumerate(thr):
        for j, p in enumerate(pa):
            pt = nozzle1d.evaluate(design, float(t), float(p))
            vals["thrust_n"][i, j] = pt.thrust_n
            vals["mass_flow_kg_s"][i, j] = pt.mass_flow_kg_s
            vals["isp_s"][i, j] = pt.isp_s
            valid[i, j] = pt.valid
            if not pt.valid:
                reasons[f"{i},{j}"] = pt.invalid_reason
    prov = {"source": "ideal_1d_isentropic_v0", "engine_design_hash": design.content_hash(),
            "qa_status": "theoretical_not_cfd_verified", "invalid_reasons": reasons}
    return EnginePerformanceMap(design.engine_design_id, thr, pa, vals, valid, prov)
