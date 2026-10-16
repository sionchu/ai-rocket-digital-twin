"""Lightweight Python 6-DoF reference integrator (independent of Isaac).

State x = [p_com_w(3), v_com_w(3), q_wb(4), omega_b(3), m_prop, m_rcs].
Translational state is the CoM; mass properties are quasi-static per step
(CoM drift rate and jet-damping terms neglected, see ADR-0004). Actuator
forces are zero-order-held over each physics step; RK4 integrates gravity,
rotation of the thrust vector, contact and rigid-body rotation.
This integrator is authoritative only in runs that do not use Isaac.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rocket_dt.dynamics.frames import GRAVITY_WORLD, quat_derivative, quat_normalize, quat_to_rotmat
from rocket_dt.dynamics.mass import MassModel
from rocket_dt.dynamics.wrench import PointForce, total_wrench_body
from rocket_dt.schemas import LandingGear

IDX_P, IDX_V, IDX_Q, IDX_W, IDX_MP, IDX_MR = slice(0, 3), slice(3, 6), slice(6, 10), slice(10, 13), 13, 14


def make_state(p, v, q, w, m_prop, m_rcs) -> np.ndarray:
    return np.concatenate([p, v, quat_normalize(np.asarray(q, float)), w, [m_prop, m_rcs]]).astype(float)


@dataclass
class StepInputs:
    forces_body: list[PointForce]
    mdot_prop: float
    mdot_rcs: float
    disturbance_force_w: np.ndarray
    disturbance_torque_b: np.ndarray


class ContactModel:
    """Penalty spring-damper per foot against the plane z = 0 with regularized Coulomb friction."""

    def __init__(self, gear: LandingGear, v_eps: float = 0.05):
        self.feet = np.array(gear.foot_positions_body_m, float).reshape(-1, 3)
        self.k, self.c, self.mu, self.v_eps = gear.stiffness_n_m, gear.damping_n_s_m, gear.friction_coeff, v_eps

    def forces(self, p_com, v_com, rot, omega_b, com_b):
        f_w = np.zeros(3)
        tau_b = np.zeros(3)
        n_contact = 0
        for foot in self.feet:
            r_b = foot - com_b
            pos = p_com + rot @ r_b
            if pos[2] >= 0.0:
                continue
            vel = v_com + rot @ np.cross(omega_b, r_b)
            fn = max(0.0, -self.k * pos[2] - self.c * vel[2])
            vt = np.array([vel[0], vel[1], 0.0])
            ft = -self.mu * fn * vt / np.sqrt(vt @ vt + self.v_eps ** 2)
            f = ft + np.array([0.0, 0.0, fn])
            f_w += f
            tau_b += np.cross(r_b, rot.T @ f)
            n_contact += 1
        return f_w, tau_b, n_contact


class Reference6DoF:
    def __init__(self, mass_model: MassModel, gear: LandingGear | None):
        self.mass_model = mass_model
        self.contact = ContactModel(gear) if gear is not None else None
        self.last_contacts = 0

    def deriv(self, x: np.ndarray, u: StepInputs) -> np.ndarray:
        # RK4 stages may overshoot below zero near depletion; clamp mass inputs and flows there
        m_prop, m_rcs = max(x[IDX_MP], 0.0), max(x[IDX_MR], 0.0)
        ms = self.mass_model.evaluate(m_prop, m_rcs)
        q = quat_normalize(x[IDX_Q])
        rot = quat_to_rotmat(q)
        w = x[IDX_W]
        f_b, tau_b = total_wrench_body(u.forces_body, ms.com_body_m)
        f_w = rot @ f_b + ms.mass_kg * GRAVITY_WORLD + u.disturbance_force_w
        tau_b = tau_b + u.disturbance_torque_b
        if self.contact is not None:
            fc_w, tc_b, _ = self.contact.forces(x[IDX_P], x[IDX_V], rot, w, ms.com_body_m)
            f_w += fc_w
            tau_b += tc_b
        i_b = ms.inertia_com_body
        w_dot = np.linalg.solve(i_b, tau_b - np.cross(w, i_b @ w))
        dx = np.zeros_like(x)
        dx[IDX_P] = x[IDX_V]
        dx[IDX_V] = f_w / ms.mass_kg
        dx[IDX_Q] = quat_derivative(q, w)
        dx[IDX_W] = w_dot
        dx[IDX_MP] = -u.mdot_prop if m_prop > 0 else 0.0
        dx[IDX_MR] = -u.mdot_rcs if m_rcs > 0 else 0.0
        return dx

    def step(self, x: np.ndarray, u: StepInputs, dt: float) -> np.ndarray:
        k1 = self.deriv(x, u)
        k2 = self.deriv(x + 0.5 * dt * k1, u)
        k3 = self.deriv(x + 0.5 * dt * k2, u)
        k4 = self.deriv(x + dt * k3, u)
        xn = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        xn[IDX_Q] = quat_normalize(xn[IDX_Q])
        xn[IDX_MP] = max(xn[IDX_MP], 0.0)
        xn[IDX_MR] = max(xn[IDX_MR], 0.0)
        if self.contact is not None:
            ms = self.mass_model.evaluate(xn[IDX_MP], xn[IDX_MR])
            self.last_contacts = self.contact.forces(xn[IDX_P], xn[IDX_V], quat_to_rotmat(xn[IDX_Q]),
                                                     xn[IDX_W], ms.com_body_m)[2]
        return xn
