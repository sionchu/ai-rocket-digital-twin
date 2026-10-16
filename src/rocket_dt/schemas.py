"""Versioned interface contracts (SSOT) between propulsion and flight tracks.

EngineDesign -> EnginePerformanceMap -> VehicleConfig/ThrusterLayout.
Every object carries schema_version and SI units; loaders reject unknown keys
and physically impossible values instead of silently fixing them.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

SCHEMA_VERSION = 1


class SchemaError(ValueError):
    pass


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SchemaError(msg)


def stable_hash(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def _check_keys(d: dict, allowed: set[str], where: str) -> None:
    unknown = set(d) - allowed
    _require(not unknown, f"{where}: unknown keys {sorted(unknown)}")


# --------------------------------------------------------------------------- engine
@dataclass(frozen=True)
class GasModel:
    gas_model_id: str
    gamma: float
    molar_mass_kg_mol: float

    @property
    def r_specific(self) -> float:
        return 8.314462618 / self.molar_mass_kg_mol


@dataclass(frozen=True)
class EngineDesign:
    engine_design_id: str
    family: str  # "conical" | "bell"
    throat_area_m2: float
    expansion_ratio: float  # Ae/At (area ratio, not radius ratio)
    chamber_pressure_pa: float  # nominal (throttle = 1)
    chamber_temperature_k: float
    gas: GasModel
    throttle_min: float
    throttle_max: float
    conical_half_angle_rad: float | None = None
    bell_length_fraction: float | None = None  # vs. 15 deg conical length
    bell_theta_n_rad: float | None = None  # initial wall angle after throat (Rao chart)
    bell_theta_e_rad: float | None = None  # exit wall angle (Rao chart)
    bell_divergence_efficiency: float | None = None  # momentum loss factor lambda
    engine_base_mass_kg: float = 0.0
    nozzle_wall_areal_density_kg_m2: float = 0.0
    schema_version: int = SCHEMA_VERSION
    units: str = "SI"

    @property
    def exit_area_m2(self) -> float:
        return self.throat_area_m2 * self.expansion_ratio

    def validate(self) -> "EngineDesign":
        _require(self.family in ("conical", "bell"), f"bad family {self.family}")
        _require(self.throat_area_m2 > 0, "throat_area_m2 must be > 0")
        _require(self.expansion_ratio > 1.0, "expansion_ratio must be > 1")
        _require(self.chamber_pressure_pa > 0, "chamber_pressure_pa must be > 0")
        _require(self.chamber_temperature_k > 0, "chamber_temperature_k must be > 0")
        _require(1.0 < self.gas.gamma < 1.7, "gamma outside (1, 1.7)")
        _require(0 < self.throttle_min < self.throttle_max <= 1.0, "bad throttle range")
        if self.family == "conical":
            a = self.conical_half_angle_rad
            _require(a is not None and 0 < a < np.radians(45), "conical needs half angle in (0, 45deg)")
        else:
            f = self.bell_length_fraction
            _require(f is not None and 0.6 <= f <= 1.0, "bell needs length fraction in [0.6, 1.0]")
            _require(self.bell_theta_n_rad is not None and self.bell_theta_e_rad is not None
                     and 0 < self.bell_theta_e_rad < self.bell_theta_n_rad < np.radians(50),
                     "bell needs 0 < theta_e < theta_n < 50deg")
            lam = self.bell_divergence_efficiency
            _require(lam is not None and 0.9 < lam <= 1.0, "bell needs divergence efficiency in (0.9, 1]")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    def content_hash(self) -> str:
        return stable_hash(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "EngineDesign":
        allowed = {f for f in cls.__dataclass_fields__} | {"exit_area_m2", "chamber_pressure_unit_note"}
        _check_keys(d, allowed, "EngineDesign")
        d = dict(d)
        _require(d.get("units", "SI") == "SI", "EngineDesign units must be SI")
        _require(int(d.get("schema_version", SCHEMA_VERSION)) == SCHEMA_VERSION, "schema_version mismatch")
        exit_area = d.pop("exit_area_m2", None)
        d.pop("chamber_pressure_unit_note", None)
        d["gas"] = GasModel(**d["gas"])
        if "conical_half_angle_deg" in d:  # pragma: no cover - guarded by _check_keys
            raise SchemaError("use conical_half_angle_rad")
        obj = cls(**d).validate()
        if exit_area is not None:
            _require(abs(exit_area - obj.exit_area_m2) / obj.exit_area_m2 < 1e-6,
                     "exit_area_m2 inconsistent with throat_area_m2*expansion_ratio")
        return obj


# --------------------------------------------------------------------------- thrusters
@dataclass(frozen=True)
class MainEngineMount:
    gimbal_pivot_body_m: tuple[float, float, float]
    thrust_axis_body: tuple[float, float, float]
    max_gimbal_rad: float
    max_gimbal_rate_rad_s: float
    throttle_time_constant_s: float


@dataclass(frozen=True)
class RcsThruster:
    id: str
    position_body_m: tuple[float, float, float]
    force_direction_body: tuple[float, float, float]
    max_thrust_n: float
    valve_delay_s: float
    min_on_time_s: float
    isp_s: float
    failed: bool = False


@dataclass(frozen=True)
class ThrusterLayout:
    layout_id: str
    main_engine: MainEngineMount
    rcs: tuple[RcsThruster, ...]

    def validate(self) -> "ThrusterLayout":
        ax = np.asarray(self.main_engine.thrust_axis_body, float)
        _require(abs(np.linalg.norm(ax) - 1) < 1e-9, "thrust_axis_body must be unit")
        _require(0 < self.main_engine.max_gimbal_rad < np.radians(20), "gimbal limit out of range")
        ids = [t.id for t in self.rcs]
        _require(len(ids) == len(set(ids)), "duplicate RCS ids")
        for t in self.rcs:
            d = np.asarray(t.force_direction_body, float)
            _require(abs(np.linalg.norm(d) - 1) < 1e-9, f"{t.id} direction must be unit")
            _require(t.max_thrust_n > 0 and t.isp_s > 0, f"{t.id} thrust/isp must be > 0")
        return self


# --------------------------------------------------------------------------- vehicle
@dataclass(frozen=True)
class MassProperties:
    structure_mass_kg: float  # dry structure without engine
    structure_length_m: float
    structure_radius_m: float
    propellant_mass_kg: float  # initial
    tank_bottom_z_m: float
    tank_top_z_m: float
    tank_radius_m: float
    rcs_gas_mass_kg: float
    rcs_gas_z_m: float
    engine_throat_below_pivot_m: float  # chamber length between gimbal pivot and throat


@dataclass(frozen=True)
class LandingGear:
    foot_positions_body_m: tuple[tuple[float, float, float], ...]
    stiffness_n_m: float
    damping_n_s_m: float
    friction_coeff: float


@dataclass(frozen=True)
class VehicleConfig:
    vehicle_id: str
    frame_convention: str
    engine_design_id: str
    mass: MassProperties
    layout: ThrusterLayout
    gear: LandingGear
    aero_model_id: str = "none_v0"
    schema_version: int = SCHEMA_VERSION
    units: str = "SI"

    def to_dict(self) -> dict:
        return asdict(self)

    def content_hash(self) -> str:
        return stable_hash(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "VehicleConfig":
        from rocket_dt.dynamics.frames import FRAME_CONVENTION

        _check_keys(d, set(cls.__dataclass_fields__), "VehicleConfig")
        _require(d.get("units", "SI") == "SI", "VehicleConfig units must be SI")
        _require(d["frame_convention"] == FRAME_CONVENTION,
                 f"frame_convention must be {FRAME_CONVENTION}")
        lay = d["layout"]
        layout = ThrusterLayout(
            layout_id=lay["layout_id"],
            main_engine=MainEngineMount(**{k: tuple(v) if isinstance(v, list) else v
                                           for k, v in lay["main_engine"].items()}),
            rcs=tuple(RcsThruster(**{k: tuple(v) if isinstance(v, list) else v for k, v in t.items()})
                      for t in lay["rcs"]),
        ).validate()
        gear = d["gear"]
        obj = cls(
            vehicle_id=d["vehicle_id"],
            frame_convention=d["frame_convention"],
            engine_design_id=d["engine_design_id"],
            mass=MassProperties(**d["mass"]),
            layout=layout,
            gear=LandingGear(
                foot_positions_body_m=tuple(tuple(p) for p in gear["foot_positions_body_m"]),
                stiffness_n_m=gear["stiffness_n_m"], damping_n_s_m=gear["damping_n_s_m"],
                friction_coeff=gear["friction_coeff"]),
            aero_model_id=d.get("aero_model_id", "none_v0"),
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
        )
        m = obj.mass
        _require(m.structure_mass_kg > 0 and m.propellant_mass_kg >= 0, "bad masses")
        _require(m.tank_top_z_m > m.tank_bottom_z_m, "tank top must be above bottom")
        _require(obj.schema_version == SCHEMA_VERSION, "schema_version mismatch")
        return obj

    def with_engine(self, engine_design_id: str) -> "VehicleConfig":
        from dataclasses import replace
        return replace(self, engine_design_id=engine_design_id)


# --------------------------------------------------------------------------- io
def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_engine(path: str | Path) -> EngineDesign:
    return EngineDesign.from_dict(load_yaml(path))


def load_vehicle(path: str | Path) -> VehicleConfig:
    return VehicleConfig.from_dict(load_yaml(path))


@dataclass
class RunManifest:
    run_id: str
    command: str
    engine_design_ids: list[str]
    vehicle_hash: str
    map_hashes: dict[str, str]
    software: dict[str, str]
    seed: int | None = None
    frame_convention: str = "world_Z_up_body_Z_forward"
    units: str = "SI"
    integrator: str = "python_rk4_reference"
    extra: dict = field(default_factory=dict)
