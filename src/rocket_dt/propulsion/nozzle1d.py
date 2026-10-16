"""Ideal 1D isentropic nozzle model (calorically perfect gas, frozen flow).

Reference equations: NASA Glenn "Rocket Thrust Equation" / "Thrust Equations
Summary" [S13, S14]. This is a theoretical reference, not a real-nozzle
prediction: no viscosity, boundary layer, shocks inside the nozzle, separation
or heat loss. Overexpanded points with Pe/Pa below the Summerfield-type
threshold are flagged invalid instead of being reported as attached-flow truth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from rocket_dt.dynamics.frames import G0
from rocket_dt.schemas import EngineDesign

# Summerfield-criterion style separation onset, Pe/Pa below this -> unreliable.
SEPARATION_PE_PA_RATIO = 0.4


def area_ratio_from_mach(mach: float, gamma: float) -> float:
    """A/A* for isentropic flow at Mach number `mach`."""
    t = (2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * mach * mach)
    return t ** ((gamma + 1.0) / (2.0 * (gamma - 1.0))) / mach


def supersonic_mach_from_area_ratio(eps: float, gamma: float) -> float:
    if eps <= 1.0:
        raise ValueError("supersonic branch needs eps > 1")
    return brentq(lambda m: area_ratio_from_mach(m, gamma) - eps, 1.0 + 1e-12, 100.0, xtol=1e-14)


def vandenkerckhove(gamma: float) -> float:
    """Gamma(gamma) in mdot = Pc*At*Gamma/sqrt(R*Tc) for a choked throat."""
    return np.sqrt(gamma) * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))


def divergence_efficiency(design: EngineDesign) -> float:
    if design.family == "conical":
        return 0.5 * (1.0 + np.cos(design.conical_half_angle_rad))
    return float(design.bell_divergence_efficiency)


@dataclass(frozen=True)
class NozzlePoint:
    throttle: float
    ambient_pressure_pa: float
    chamber_pressure_pa: float
    mass_flow_kg_s: float
    exit_mach: float
    exit_pressure_pa: float
    exit_temperature_k: float
    exit_velocity_m_s: float
    thrust_n: float
    isp_s: float
    thrust_coefficient: float
    expansion_state: str  # under | ideal | over
    valid: bool
    invalid_reason: str


def evaluate(design: EngineDesign, throttle: float, ambient_pressure_pa: float) -> NozzlePoint:
    if not (design.throttle_min - 1e-12 <= throttle <= design.throttle_max + 1e-12):
        raise ValueError(f"throttle {throttle} outside [{design.throttle_min}, {design.throttle_max}]")
    if ambient_pressure_pa < 0:
        raise ValueError("ambient pressure must be >= 0")
    g = design.gas.gamma
    r = design.gas.r_specific
    pc = design.chamber_pressure_pa * throttle  # throttling by Pc at fixed Tc (assumption)
    tc = design.chamber_temperature_k
    at, ae, eps = design.throat_area_m2, design.exit_area_m2, design.expansion_ratio

    mdot = pc * at * vandenkerckhove(g) / np.sqrt(r * tc)
    me = supersonic_mach_from_area_ratio(eps, g)
    pe = pc * (1.0 + 0.5 * (g - 1.0) * me * me) ** (-g / (g - 1.0))
    te = tc / (1.0 + 0.5 * (g - 1.0) * me * me)
    ve = np.sqrt(2.0 * g / (g - 1.0) * r * tc * (1.0 - (pe / pc) ** ((g - 1.0) / g)))
    lam = divergence_efficiency(design)
    thrust = lam * mdot * ve + (pe - ambient_pressure_pa) * ae
    isp = thrust / (mdot * G0)
    cf = thrust / (pc * at)

    if ambient_pressure_pa == 0 or pe > 1.02 * ambient_pressure_pa:
        state = "under"
    elif pe < 0.98 * ambient_pressure_pa:
        state = "over"
    else:
        state = "ideal"
    valid, reason = True, ""
    if ambient_pressure_pa > 0 and pe / ambient_pressure_pa < SEPARATION_PE_PA_RATIO:
        valid, reason = False, f"separation_risk Pe/Pa={pe / ambient_pressure_pa:.3f}"
    if thrust <= 0:
        valid, reason = False, "non_positive_thrust"
    return NozzlePoint(throttle, ambient_pressure_pa, pc, mdot, me, pe, te, ve, thrust, isp, cf,
                       state, valid, reason)
