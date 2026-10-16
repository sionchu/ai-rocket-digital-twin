"""Time-varying mass, CoM and inertia from the component layout.

Components (all in body frame, axisymmetric about body Z):
- structure: thin cylindrical shell centred at the body origin
- engine: base mass at the gimbal pivot + nozzle shell at its wall centroid
  (point masses; the engine's own rotational inertia is neglected in P0)
- main propellant: rigid solid cylinder filling the tank from the bottom
  (no slosh), density fixed so the initial load fills the tank
- RCS gas: point mass on the axis at rcs_gas_z_m
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rocket_dt.propulsion.geometry import build_contour, nozzle_wall_centroid_x
from rocket_dt.schemas import EngineDesign, VehicleConfig


@dataclass(frozen=True)
class MassState:
    mass_kg: float
    com_body_m: np.ndarray
    inertia_com_body: np.ndarray  # 3x3 about CoM, body axes


def _point(m, z):
    return m, np.array([0.0, 0.0, z]), np.zeros((3, 3))


def _shift(m, c, i_c, ref):
    """Parallel-axis: inertia of (m, I_c at c) about point ref."""
    d = c - ref
    return i_c + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))


class MassModel:
    def __init__(self, vehicle: VehicleConfig, engine: EngineDesign):
        if vehicle.engine_design_id != engine.engine_design_id:
            raise ValueError(f"vehicle expects engine {vehicle.engine_design_id}, got {engine.engine_design_id}")
        mp = vehicle.mass
        self.vehicle = vehicle
        self.prop0 = mp.propellant_mass_kg
        self.rcs0 = mp.rcs_gas_mass_kg
        h = mp.tank_top_z_m - mp.tank_bottom_z_m
        self.rho_prop = self.prop0 / (np.pi * mp.tank_radius_m ** 2 * h) if self.prop0 > 0 else 1.0
        contour = build_contour(engine)
        pivot_z = vehicle.layout.main_engine.gimbal_pivot_body_m[2]
        throat_z = pivot_z - mp.engine_throat_below_pivot_m
        self.engine_base = _point(engine.engine_base_mass_kg, pivot_z)
        nozzle_mass = engine.nozzle_wall_areal_density_kg_m2 * contour.divergent_wall_area_m2
        self.nozzle = _point(nozzle_mass, throat_z - nozzle_wall_centroid_x(contour))
        self.engine_mass_kg = engine.engine_base_mass_kg + nozzle_mass
        self.nozzle_exit_z_m = throat_z - contour.nozzle_length_m
        m, r, l = mp.structure_mass_kg, mp.structure_radius_m, mp.structure_length_m
        i_shell = np.diag([m * (r * r / 2 + l * l / 12), m * (r * r / 2 + l * l / 12), m * r * r])
        self.structure = (m, np.zeros(3), i_shell)
        self.dry_mass_kg = m + self.engine_mass_kg

    def propellant_component(self, m_prop: float):
        mp = self.vehicle.mass
        if m_prop <= 0:
            return 0.0, np.array([0.0, 0.0, mp.tank_bottom_z_m]), np.zeros((3, 3))
        h = m_prop / (self.rho_prop * np.pi * mp.tank_radius_m ** 2)
        r = mp.tank_radius_m
        i_c = np.diag([m_prop * (r * r / 4 + h * h / 12)] * 2 + [m_prop * r * r / 2])
        return m_prop, np.array([0.0, 0.0, mp.tank_bottom_z_m + h / 2]), i_c

    def evaluate(self, m_prop: float, m_rcs: float) -> MassState:
        if m_prop < -1e-9 or m_rcs < -1e-9:
            raise ValueError("negative propellant mass")
        comps = [self.structure, self.engine_base, self.nozzle,
                 self.propellant_component(max(m_prop, 0.0)),
                 _point(max(m_rcs, 0.0), self.vehicle.mass.rcs_gas_z_m)]
        m_tot = sum(c[0] for c in comps)
        com = sum(c[0] * c[1] for c in comps) / m_tot
        inertia = sum(_shift(c[0], c[1], c[2], com) for c in comps)
        return MassState(m_tot, com, inertia)
