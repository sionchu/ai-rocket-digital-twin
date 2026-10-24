"""Procedural PBR textures for the VTVL v1 visual assets (numpy + scipy + PIL only, no downloads).

All tiling textures are made periodic (FFT-filtered noise, periodic Voronoi) so they repeat without seams.
Outputs go to assets/vtvl_v1/textures/. Albedo is written sRGB, everything else linear ('raw').
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

import vtvl_spec as S

RNG = np.random.default_rng(20261021)
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf"


# ----------------------------------------------------------------------------------------------- helpers
def to_srgb(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055)


def pnoise(h: int, w: int, period_px: float, rng=RNG, aniso: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    """Periodic band-limited noise, unit std, features ~period_px pixels. aniso stretches (rows, cols)."""
    f = np.fft.fft2(rng.standard_normal((h, w)))
    fy = np.fft.fftfreq(h)[:, None] * aniso[0]
    fx = np.fft.fftfreq(w)[None, :] * aniso[1]
    k = np.sqrt(fx ** 2 + fy ** 2) * period_px
    g = np.exp(-0.5 * (k / 0.6) ** 2)
    n = np.real(np.fft.ifft2(f * g))
    return (n - n.mean()) / (n.std() + 1e-9)


def fbm(h: int, w: int, periods, weights=None, rng=RNG, aniso=(1.0, 1.0)) -> np.ndarray:
    weights = weights or [0.5 ** i for i in range(len(periods))]
    out = sum(wt * pnoise(h, w, p, rng, aniso) for p, wt in zip(periods, weights))
    return out / np.sqrt(sum(wt ** 2 for wt in weights))


def periodic_voronoi(h: int, w: int, n_cells: int, warp: tuple[np.ndarray, np.ndarray] | None = None, rng=RNG):
    """Return (d1, d2, cell_id) for a periodic Voronoi in unit-square texture space (distances in pixels)."""
    pts = rng.random((n_cells, 2))
    tree = cKDTree(pts, boxsize=1.0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    u, v = (xx + 0.5) / w, (yy + 0.5) / h
    if warp is not None:
        u = u + warp[0] / w
        v = v + warp[1] / h
    q = np.stack([np.mod(u, 1.0).ravel(), np.mod(v, 1.0).ravel()], 1)
    d, i = tree.query(q, k=2, workers=-1)
    scale = 0.5 * (w + h)
    return (d[:, 0].reshape(h, w) * scale, d[:, 1].reshape(h, w) * scale, i[:, 0].reshape(h, w))


def height_to_normal(height_m: np.ndarray, px_u_m: float, px_v_m: float, wrap: bool = True) -> np.ndarray:
    """Tangent-space normal map (OpenGL, +G = +v). height in metres; image row 0 is v = 1."""
    if wrap:
        dhdu = (np.roll(height_m, -1, 1) - np.roll(height_m, 1, 1)) / (2 * px_u_m)
        dhdrow = (np.roll(height_m, -1, 0) - np.roll(height_m, 1, 0)) / (2 * px_v_m)
    else:
        dhdrow, dhdu = np.gradient(height_m, px_v_m, px_u_m)
    dhdv = -dhdrow
    n = np.stack([-dhdu, -dhdv, np.ones_like(height_m)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n * 0.5 + 0.5


def save_rgb(path: Path, lin_rgb: np.ndarray, srgb: bool = True, quality: int = 90):
    arr = to_srgb(lin_rgb) if srgb else np.clip(lin_rgb, 0, 1)
    img = Image.fromarray((arr * 255 + 0.5).astype(np.uint8), "RGB")
    if path.suffix == ".jpg":
        img.save(path, quality=quality, subsampling=0, optimize=True)
    else:
        img.save(path, optimize=True)


def save_gray(path: Path, x: np.ndarray, quality: int = 90):
    img = Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8), "L")
    if path.suffix == ".jpg":
        img.save(path, quality=quality, optimize=True)
    else:
        img.save(path, optimize=True)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def text_mask(text: str, height_px: int, font=FONT_BOLD, spacing: int = 0) -> np.ndarray:
    f = ImageFont.truetype(font, height_px)
    bbox = f.getbbox(text)
    im = Image.new("L", (bbox[2] - bbox[0] + 8 + spacing * len(text), bbox[3] - bbox[1] + 8), 0)
    d = ImageDraw.Draw(im)
    if spacing:
        x = 4
        for ch in text:
            d.text((x - bbox[0], 4 - bbox[1]), ch, fill=255, font=f)
            x += f.getbbox(ch)[2] + spacing
    else:
        d.text((4 - bbox[0], 4 - bbox[1]), text, fill=255, font=f)
    return np.asarray(im, np.float32) / 255.0


def stamp(dst: np.ndarray, mask: np.ndarray, row0: int, col0: int, wrap_cols: int | None = None):
    """Max-composite mask into dst at (row0, col0) (cols wrap if wrap_cols)."""
    h, w = mask.shape
    for r in range(h):
        rr = row0 + r
        if rr < 0 or rr >= dst.shape[0]:
            continue
        cols = (np.arange(col0, col0 + w) % wrap_cols) if wrap_cols else np.arange(col0, col0 + w)
        ok = (cols >= 0) & (cols < dst.shape[1])
        dst[rr, cols[ok]] = np.maximum(dst[rr, cols[ok]], mask[r, ok])


# ----------------------------------------------------------------------------------------------- hull
def hull_textures(out: Path):
    """Hull: u = theta/2pi (theta from +x, CCW), v = (z - z_bot)/length. 1024 (u) x 2048 (v)."""
    W, H = 1024, 2048
    px_u = 2 * math.pi * S.HULL_R / W
    px_v = S.HULL_LEN / H
    rows = np.arange(H)
    z = S.Z_BOT + S.HULL_LEN * (1 - (rows + 0.5) / H)
    th = 2 * math.pi * (np.arange(W) + 0.5) / W
    Z = np.repeat(z[:, None], W, 1)
    TH = np.repeat(th[None, :], H, 0)

    skirt = Z < S.Z_SKIRT_TOP
    band = (Z >= S.Z_BAND_BOT) & (Z < S.Z_BAND_TOP)
    paint = ~skirt & ~band

    lowf = fbm(H, W, [400, 160, 60], aniso=(1.0, 1.0))
    streak_v = pnoise(H, W, 30, aniso=(0.08, 1.0))  # long vertical streaks (grime runs)
    brush = pnoise(H, W, 6, aniso=(1.0, 0.03))  # circumferential brushing on bare metal
    fine = pnoise(H, W, 2)

    alb = np.zeros((H, W, 3), np.float32)
    rough = np.zeros((H, W), np.float32)
    metal = np.zeros((H, W), np.float32)
    height = np.zeros((H, W), np.float32)

    # white paint (slightly warm), subtle grime and roughness variation
    paint_col = np.array([0.72, 0.71, 0.69])
    grime = 1 - 0.025 * lowf - 0.02 * np.clip(streak_v, 0, None) - 0.006 * fine
    alb[paint] = (paint_col[None] * grime[paint][:, None])
    rough[paint] = 0.36 + 0.05 * lowf[paint] + 0.03 * streak_v[paint]
    # bare stainless skirt with circumferential brushing + soot toward the bottom edge
    steel = np.array([0.56, 0.56, 0.57])
    soot = smoothstep(S.Z_SKIRT_TOP - 0.05, S.Z_BOT, Z) ** 1.6 * (0.75 + 0.25 * lowf)
    s_alb = steel[None, None] * (1 + 0.05 * brush[..., None] + 0.02 * lowf[..., None])
    s_alb = s_alb * (1 - 0.65 * np.clip(soot, 0, 1)[..., None])
    alb[skirt] = s_alb[skirt]
    rough[skirt] = (0.26 + 0.05 * np.abs(brush) + 0.35 * np.clip(soot, 0, 1))[skirt]
    metal[skirt] = 1.0 - 0.3 * np.clip(soot, 0, 1)[skirt]
    # dark avionics band (carbon composite look)
    weave = 0.5 + 0.5 * np.sign(np.sin(TH * W / 8) * np.sin(Z / px_v * 2 * math.pi / 8))
    alb[band] = (np.array([0.032, 0.033, 0.036])[None] * (1 + 0.15 * weave[band][:, None] + 0.08 * lowf[band][:, None]))
    rough[band] = 0.5 + 0.06 * weave[band] + 0.04 * lowf[band]

    # weld rings: raised bead, slightly darker/greyer paint over the weld
    for zw in S.WELD_Z:
        d = np.abs(Z - zw)
        bead = np.exp(-0.5 * (d / 0.006) ** 2)
        height += 0.0008 * bead
        alb *= (1 - 0.18 * bead)[..., None]
        rough += 0.08 * bead
    # vertical panel seams (thin grooves) on the painted tank only
    for a in (30.0, 150.0, 270.0):
        dth = np.angle(np.exp(1j * (TH - math.radians(a)))) * S.HULL_R
        g = np.exp(-0.5 * (dth / 0.002) ** 2) * ((Z > S.Z_SKIRT_TOP) & (Z < S.Z_BAND_BOT))
        height -= 0.0005 * g
        alb *= (1 - 0.25 * g)[..., None]
    # avionics band: access panel outlines + fastener rows
    for a0 in range(0, 360, 45):
        c = math.radians(a0 + 22.5)
        dth = np.angle(np.exp(1j * (TH - c))) * S.HULL_R
        inside = (np.abs(dth) < 0.16) & (Z > S.Z_BAND_BOT + 0.10) & (Z < S.Z_BAND_TOP - 0.10)
        edge = inside & ~((np.abs(dth) < 0.16 - 0.004) & (Z > S.Z_BAND_BOT + 0.104) & (Z < S.Z_BAND_TOP - 0.104))
        height -= 0.0006 * edge
        alb[edge] *= 0.5
    for zf in (S.Z_BAND_BOT + 0.03, S.Z_BAND_TOP - 0.03, S.Z_SKIRT_TOP - 0.03):
        n_f = int(2 * math.pi * S.HULL_R / 0.045)
        phase = np.mod(TH / (2 * math.pi) * n_f, 1.0) - 0.5
        r_px = np.sqrt((phase * 0.045) ** 2 + (Z - zf) ** 2)
        dot = r_px < 0.0045
        height += 0.0004 * dot
        alb[dot] = np.array([0.45, 0.45, 0.46])
        metal[dot] = 1.0
        rough[dot] = 0.35

    # ---- markings (black paint) on the white tank: stencil, roll-orientation checks, photogrammetry targets
    ink = np.zeros((H, W), np.float32)
    # 'VTVL-0' stencil running vertically (reads bottom-to-top), letters ~0.30 m tall
    letter_px = int(0.30 / px_u)
    m = text_mask("VTVL-0", letter_px, spacing=int(0.05 / px_u))
    m = np.rot90(m, 1)  # reads bottom->top along +z
    col0 = int(math.radians(S.TEXT_DEG) / (2 * math.pi) * W) - m.shape[1] // 2
    zc = -0.25
    row0 = int((1 - (zc - S.Z_BOT) / S.HULL_LEN) * H) - m.shape[0] // 2
    stamp(ink, m, row0, col0, wrap_cols=W)
    # roll-orientation checker ring just below the band: 8 sectors x 2 rows, alternating
    z_r0, z_r1 = 1.40, 1.80
    sec = np.floor(TH / (2 * math.pi) * 8).astype(int)
    rowk = np.where(Z > 0.5 * (z_r0 + z_r1), 1, 0)
    chk = ((sec + rowk) % 2 == 0) & (Z > z_r0) & (Z < z_r1)
    ink = np.maximum(ink, chk.astype(np.float32))
    # thin tick marks every 0.25 m at theta = 0 and 180 deg (tracking reference)
    for a in (0.0, 180.0, 90.0, 270.0):
        dth = np.angle(np.exp(1j * (TH - math.radians(a)))) * S.HULL_R
        zz = np.mod(Z + 3.5, 0.25)
        major = np.mod(Z + 3.5, 1.0) < 0.02
        tick = (np.abs(dth) < np.where(major, 0.05, 0.025)) & (zz < 0.012) & (Z > -2.8) & (Z < 1.3)
        ink = np.maximum(ink, tick.astype(np.float32))
    # quarter-checker photogrammetry targets
    for a, zt in ((112.5, 0.9), (247.5, 0.9), (22.5, -2.2), (202.5, -2.2), (337.5, 0.2)):
        dth = np.angle(np.exp(1j * (TH - math.radians(a)))) * S.HULL_R
        dz = Z - zt
        rr = np.hypot(dth, dz)
        disc = rr < 0.07
        q = (np.sign(dth) * np.sign(dz)) > 0
        ink = np.maximum(ink, (disc & q).astype(np.float32))
        ring = (rr > 0.07) & (rr < 0.08)
        ink = np.maximum(ink, ring.astype(np.float32))
    wear = np.clip(1.15 + 0.25 * fbm(H, W, [40, 12]), 0, 1)  # very light wear
    ink = ink * wear * paint
    alb = alb * (1 - ink[..., None]) + np.array([0.025, 0.025, 0.027])[None, None] * ink[..., None]
    rough = rough * (1 - ink) + 0.45 * ink
    height += 0.00005 * ink

    save_rgb(out / "hull_albedo.jpg", alb, quality=92)
    save_gray(out / "hull_roughness.jpg", np.clip(rough, 0.05, 1))
    save_gray(out / "hull_metallic.png", metal)
    save_rgb(out / "hull_normal.jpg", height_to_normal(height, px_u, px_v), srgb=False, quality=95)


# ----------------------------------------------------------------------------------------------- nozzle
def nozzle_textures(out: Path):
    """Nozzle: u = theta/2pi, v = 0 at the exit lip, 1 at the chamber top. Heat-tint gradient toward the exit."""
    W, H = 512, 1024
    v = 1 - (np.arange(H) + 0.5) / H  # row 0 is v = 1 (chamber)
    t = 1 - v  # 0 at chamber, 1 at exit
    # (t, linear F0 colour, roughness): copper/bronze chamber -> straw -> violet -> blue -> grey-blue lip
    keys = [(0.00, (0.30, 0.19, 0.12), 0.42), (0.22, (0.34, 0.22, 0.13), 0.38), (0.40, (0.42, 0.33, 0.18), 0.34),
            (0.55, (0.30, 0.20, 0.30), 0.32), (0.72, (0.19, 0.22, 0.40), 0.30), (0.90, (0.30, 0.33, 0.40), 0.34),
            (1.00, (0.40, 0.41, 0.43), 0.40)]
    kt = np.array([k[0] for k in keys])
    col = np.stack([np.interp(t, kt, [k[1][c] for k in keys]) for c in range(3)], -1)
    rgh = np.interp(t, kt, [k[2] for k in keys])
    streak = pnoise(H, W, 10, aniso=(0.04, 1.0))
    mott = fbm(H, W, [80, 20])
    tubes = 0.5 + 0.5 * np.cos(np.arange(W) / W * 2 * math.pi * 96)  # faint regen-tube lines
    alb = col[:, None, :] * (1 + 0.07 * streak[..., None] + 0.05 * mott[..., None] - 0.04 * tubes[None, :, None])
    rough = rgh[:, None] + 0.04 * streak + 0.03 * mott
    height = 0.0004 * tubes[None, :] * np.ones((H, 1))
    save_rgb(out / "nozzle_albedo.jpg", alb, quality=92)
    save_gray(out / "nozzle_roughness.jpg", rough)
    save_rgb(out / "nozzle_normal.jpg", height_to_normal(height, 2 * math.pi * 0.34 / W, 0.9 / H), srgb=False)


# ----------------------------------------------------------------------------------------------- small tiling
def soot_texture(out: Path):
    """Tiling dark ablative / soot mottling (heat shield, skirt interior). 512^2, 1 m repeat."""
    N = 512
    m = fbm(N, N, [120, 40, 10, 3])
    alb = np.array([0.045, 0.042, 0.040])[None, None] * (1 + 0.25 * m[..., None])
    rough = 0.82 + 0.06 * m
    save_rgb(out / "soot_albedo.jpg", alb)
    save_gray(out / "soot_roughness.jpg", rough)
    save_rgb(out / "soot_normal.jpg", height_to_normal(0.0006 * fbm(N, N, [6, 2]), 1 / N, 1 / N), srgb=False)


# ----------------------------------------------------------------------------------------------- lakebed
def lakebed_textures(out: Path):
    """Seamless dry-lakebed (playa) ground, 2048^2 covering GROUND_TILE_M metres."""
    N = 2048
    px = S.GROUND_TILE_M / N
    warp = (fbm(N, N, [60, 15]) * 2.0, fbm(N, N, [60, 15]) * 2.0)
    d1, d2, cid = periodic_voronoi(N, N, int((S.GROUND_TILE_M / 1.3) ** 2), warp)
    crack_big = np.exp(-0.5 * ((d2 - d1) / 0.9) ** 2)
    warp2 = (fbm(N, N, [25, 6]) * 1.5, fbm(N, N, [25, 6]) * 1.5)
    e1, e2, cid2 = periodic_voronoi(N, N, int((S.GROUND_TILE_M / 0.45) ** 2), warp2)
    crack_small = np.exp(-0.5 * ((e2 - e1) / 0.6) ** 2)
    edge_curl = np.exp(-((d2 - d1) / 6.0)) - crack_big  # slightly raised plate edges
    lum_cell = RNG.normal(0, 1, cid.max() + 1)[cid]
    macro = fbm(N, N, [1024, 512, 256])
    meso = fbm(N, N, [96, 32])
    fine = pnoise(N, N, 1.5)
    base_a = np.array([0.49, 0.385, 0.27])  # pale tan
    base_b = np.array([0.41, 0.335, 0.25])  # greyer silt
    mix = np.clip(0.5 + 0.35 * macro + 0.15 * meso, 0, 1)[..., None]
    alb = base_a * (1 - mix) + base_b * mix
    alb = alb * (1 + 0.035 * lum_cell[..., None] + 0.04 * meso[..., None] + 0.03 * fine[..., None])
    alb = alb * (1 - 0.55 * crack_big[..., None] - 0.22 * crack_small[..., None]) * (1 + 0.06 * edge_curl[..., None])
    rough = 0.9 + 0.04 * meso - 0.05 * edge_curl
    height = -0.012 * crack_big - 0.004 * crack_small + 0.002 * edge_curl + 0.0015 * meso + 0.0004 * fine
    save_rgb(out / "lakebed_albedo.jpg", alb, quality=88)
    save_gray(out / "lakebed_roughness.jpg", rough, quality=85)
    save_rgb(out / "lakebed_normal.jpg", height_to_normal(height, px, px), srgb=False, quality=90)


def ridge_texture(out: Path):
    """Distant ridge rock/scree strata, u wraps around the horizon, v = height. 1024 x 512."""
    W, H = 1024, 512
    v = 1 - (np.arange(H) + 0.5) / H
    warpv = 0.05 * fbm(H, W, [200, 60])
    vv = v[:, None] + warpv
    strata = np.sin(vv * 2 * math.pi * 9) * 0.5 + 0.5
    m = fbm(H, W, [120, 30, 8])
    c0 = np.array([0.30, 0.22, 0.16])
    c1 = np.array([0.40, 0.31, 0.22])
    alb = c0 * (1 - strata[..., None]) + c1 * strata[..., None]
    alb = alb * (1 + 0.12 * m[..., None])
    save_rgb(out / "ridge_albedo.jpg", alb)


# ----------------------------------------------------------------------------------------------- pad
def pad_textures(out: Path):
    """Concrete landing pad (square texture covering PAD_TEX_SPAN m, centred on the pad).

    Markings: outer ring, cross, inner ring (white paint, worn); slab saw-cuts; centre scorch with radial streaks."""
    N = 2048
    span = S.PAD_TEX_SPAN
    px = span / N
    c = (np.arange(N) + 0.5) / N * span - span / 2
    X = c[None, :] * np.ones((N, 1))
    Y = c[::-1, None] * np.ones((1, N))  # row 0 is +y (v = 1)
    R = np.hypot(X, Y)
    A = np.arctan2(Y, X)
    mott = fbm(N, N, [300, 90, 25])
    fine = pnoise(N, N, 1.5)
    spk = RNG.random((N, N))
    aggregate = (spk > 0.985).astype(np.float32) * RNG.normal(0, 1, (N, N))
    concrete = np.array([0.40, 0.385, 0.36])
    alb = concrete[None, None] * (1 + 0.05 * mott[..., None] + 0.04 * fine[..., None] + 0.10 * aggregate[..., None])
    rough = 0.82 + 0.05 * mott + 0.03 * fine
    height = 0.0004 * fine + 0.0003 * aggregate
    # slab saw cuts every 3.5 m (offset so no joint runs under the centre)
    for axis in (X, Y):
        g = np.abs(np.mod(axis + 1.75, 3.5) - 1.75)
        cut = np.exp(-0.5 * ((g - 1.75) / 0.004) ** 2)  # joints at +-1.75, +-5.25
        height -= 0.004 * cut
        alb *= (1 - 0.45 * cut)[..., None]
    # rim: darker trowelled edge band
    rim = smoothstep(S.PAD_R - 0.18, S.PAD_R - 0.05, R)
    alb *= (1 - 0.15 * rim)[..., None]
    # markings
    paint = np.zeros((N, N), np.float32)
    paint = np.maximum(paint, ((R > 5.0) & (R < 5.35)).astype(np.float32))
    paint = np.maximum(paint, ((R > 0.9) & (R < 1.05)).astype(np.float32))
    arm = ((np.abs(X) < 0.125) | (np.abs(Y) < 0.125)) & (R > 1.6) & (R < 4.6)
    paint = np.maximum(paint, arm.astype(np.float32))
    # foot target squares at r = 1.6 on the axes (where the gear lands at zero yaw)
    for fx, fy in ((1.6, 0), (-1.6, 0), (0, 1.6), (0, -1.6)):
        sq = (np.abs(X - fx) < 0.42) & (np.abs(Y - fy) < 0.42) & ~((np.abs(X - fx) < 0.36) & (np.abs(Y - fy) < 0.36))
        paint = np.maximum(paint, sq.astype(np.float32) * 0.85)
    wear = np.clip(0.95 + 0.45 * fbm(N, N, [80, 25, 8]), 0, 1)
    paint *= wear
    paint_col = np.array([0.72, 0.71, 0.68])
    # scorch: radial soot with streaks, stronger at the centre
    s_ang = gaussian_filter(RNG.normal(0, 1, 360), 1.5, mode="wrap")
    s_ang /= s_ang.std()
    streaks = 0.5 + 0.25 * np.interp(np.mod(A, 2 * math.pi), np.linspace(0, 2 * math.pi, 361), np.r_[s_ang, s_ang[0]])
    sc = np.exp(-(R / 2.0) ** 2) * (0.75 + 0.25 * streaks) + 0.25 * np.exp(-(R / 4.0) ** 2) * (0.6 + 0.4 * mott)
    sc = np.clip(sc * (0.9 + 0.15 * mott), 0, 1)
    alb = alb * (1 - paint[..., None]) + paint_col[None, None] * paint[..., None]
    rough = rough * (1 - paint) + 0.6 * paint
    alb = alb * (1 - 0.78 * sc[..., None])
    rough = rough * (1 - 0.2 * sc)
    height += 0.0002 * paint
    save_rgb(out / "pad_albedo.jpg", alb, quality=92)
    save_gray(out / "pad_roughness.jpg", rough)
    save_rgb(out / "pad_normal.jpg", height_to_normal(height, px, px, wrap=False), srgb=False, quality=92)


def build_all(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for fn in (hull_textures, nozzle_textures, soot_texture, lakebed_textures, ridge_texture, pad_textures):
        fn(out)
        print("textures:", fn.__name__, "done", flush=True)


if __name__ == "__main__":
    build_all(Path(__file__).resolve().parents[2] / "assets/vtvl_v1/textures")
