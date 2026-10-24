"""Shared dimensions for the VTVL v1 visual assets (body frame, metres, Z-up, +Z = nose, origin mid-length).

Numbers that must match the physics model (hull radius/length, pivot, throat, feet) are read from the repo's
configs where possible (see build_vtvl_assets.vehicle_facts); the values here are the visual-only layout choices.
"""
from __future__ import annotations

import math

HULL_R = 0.6
HULL_LEN = 7.0
Z_BOT, Z_TOP = -HULL_LEN / 2, HULL_LEN / 2
NOSE_LEN = 1.0  # tangent ogive length (blunted tip, tip stays at Z_TOP)
NOSE_TIP_R = 0.07  # spherical blunting radius

# texture / paint zones along the hull (z, metres)
Z_SKIRT_TOP = -2.95  # bare stainless aft skirt below this
Z_BAND_BOT, Z_BAND_TOP = 1.90, 2.45  # dark avionics / interstage band
WELD_Z = (-2.95, -1.75, -0.55, 0.65, 1.90, 2.45)

Z_HEATSHIELD = -3.08  # recessed aft bulkhead; the skirt below it is open so all engine layouts are visible
SKIRT_INNER_R = 0.585

LEG_ANGLES_DEG = (0.0, 180.0, 90.0, 270.0)  # matches foot order (+x, -x, +y, -y)
FIN_ANGLES_DEG = (45.0, 135.0, 225.0, 315.0)
Z_FIN = 2.2
Z_LEG_HINGE = -1.9
RACEWAY_DEG = -67.5  # between the -y leg and the 315 deg grid fin
TEXT_DEG = 157.5  # 'VTVL-0' stencil, between +y... -x quadrant, visible in the hero shot

FOOT_PAD_R = 0.26
FOOT_PAD_H = 0.09

PAD_R = 7.0
PAD_TOP_Z = 0.004
PAD_TEX_SPAN = 14.4  # metres covered by the pad texture (square, centred)
GROUND_TILE_M = 40.0  # lakebed texture repeat


def deg(a: float) -> float:
    return math.radians(a)
