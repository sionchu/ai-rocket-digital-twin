"""Variable mass on a fixed-mass rigid body (ADR-0011).

Isaac Lab 3.0 does not apply per-env mass/inertia changes at run time on the GPU pipeline (ADR-0009), so the
PhysX body keeps the full-propellant properties (m0, I0, CoM c0). This module computes the wrench to apply to that
body so that the true CoM c(t) and the attitude follow the variable-mass rigid-body equations used by the
reference integrator (quasi-static mass properties, CoM drift rate and jet damping neglected, ADR-0004):

  true:     m a_c = F,          I w_dot + w x (I w) = tau        (F excludes gravity, tau about c)
  physics:  m0 a_c0 = F_app,    I0 w_dot + w x (I0 w) = tau_app  (PhysX adds m0 g at c0 itself)
  with c = c0 + R d,  d = c - c0 (body, moves along the axis as propellant drains):
      a_c = a_c0 + R (w_dot x d + w x (w x d) + 2 w x d_dot + d_ddot)

so  w_dot   = I^-1 (tau - w x I w)
    tau_app = I0 w_dot + w x (I0 w)
    F_app   = m0 (F / m - R (w_dot x d + w x (w x d) + 2 w x d_dot + d_ddot))
The true CoM then obeys m a_c = F exactly as in the reference integrator, and the structure (c0 point) moves
relative to it by -R d. At reset the c0-point velocity must be v_c - R (w x d + d_dot).

Backend-neutral (numpy or torch): inertia tensors are diagonal (axisymmetric vehicle), given as (n,3).
Contact forces are applied by PhysX to the m0/I0 body and are NOT corrected (only affects the post-touchdown
settle; touchdown metrics are taken before contact).
"""
from __future__ import annotations


def _cross(a, b):
    ax, ay, az = a[..., 0:1], a[..., 1:2], a[..., 2:3]
    bx, by, bz = b[..., 0:1], b[..., 1:2], b[..., 2:3]
    return ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx


def cross(a, b, e):
    """a x b with unit vectors e = (ex, ey, ez) of the caller's backend."""
    cx, cy, cz = _cross(a, b)
    return cx * e[0] + cy * e[1] + cz * e[2]


def compensated_wrench(force_w, tau_b, omega_b, rot, m, i_diag, d_b, m0, i0_diag, e, d_dot_b=None, d_ddot_b=None):
    """Wrench for the fixed-mass body.

    force_w (n,3) non-gravity external force on the true body (world); tau_b (n,3) torque about the true CoM
    (body); omega_b (n,3); rot (n,3,3) body->world; m (n,) true mass; i_diag (n,3) true inertia about the true CoM;
    d_b (n,3) true CoM minus physics CoM (body) and its body-frame rates d_dot_b, d_ddot_b (n,3, optional);
    m0 float; i0_diag (3,) physics inertia; e unit vectors.
    Returns (F_app_w (n,3), tau_app_b (n,3), w_dot_b (n,3)).
    """
    w_dot = (tau_b - cross(omega_b, i_diag * omega_b, e)) / i_diag
    tau_app = i0_diag * w_dot + cross(omega_b, i0_diag * omega_b, e)
    a_rel_b = cross(w_dot, d_b, e) + cross(omega_b, cross(omega_b, d_b, e), e)
    if d_dot_b is not None:
        a_rel_b = a_rel_b + 2 * cross(omega_b, d_dot_b, e)
    if d_ddot_b is not None:
        a_rel_b = a_rel_b + d_ddot_b
    a_rel_w = (rot * a_rel_b[:, None, :]).sum(-1)
    f_app = m0 * (force_w / m[:, None] - a_rel_w)
    return f_app, tau_app, w_dot
