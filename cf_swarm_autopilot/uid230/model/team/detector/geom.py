"""Geometric victim proposals from the depth frame alone — no network.

The learned depth screen has to recognise a mannequin from 2-5 px of appearance,
and measured on real flight that is where it struggles: per-frame recall 0.41-0.52
on city/village/forest at thresholds we can afford, with a false-alarm rate that
drains the 40-frame RGB budget before the victim ever shows up.

A victim is also a *shape in space*: a small thing standing on the ground, under
2 m tall and well under a metre across. Depth gives that directly. So instead of
asking "does this look like a person", ask "is anything here the size of one" —
and hand what survives to the colour model, which is the part that is actually
good at telling a person from a bollard.

    heights()  back-project the frame, subtract a LOCAL ground surface
    propose()  connected components of the "small object on the ground" mask

Ground is estimated per world-XY cell as the lowest point in a `ground_win`-cell
neighbourhood, not from the altimeter. That matters and is not a detail: the
altimeter measures whatever is under the DRONE — a rooftop, a canopy — and a ring
of pixels around a detection measures the victim's own legs (that attempt lost to
the altimeter 3.49 m vs 1.13 m). A neighbourhood minimum is immune to both,
because a 0.5 m victim cannot be the lowest thing within several metres of itself.

Everything is vectorised: this runs per drone per detector tick inside a 500 ms
act() budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .localize import (CAM_FWD, CAM_UP, DEPTH_MAX_M, DEPTH_MIN_M, FOV_DEG, RES,
                       depth_to_meters)

_RAY_CACHE: dict = {}


@dataclass
class GeomConfig:
    # A mannequin is ~1.7 m standing, ~0.5 m lying, and ~0.5 m across. The bands
    # are generous on purpose — this is a PROPOSAL stage whose job is not to miss;
    # colour does the discriminating.
    min_height: float = 0.35      # below this is ground noise and kerbs
    max_height: float = 2.2       # above this is a lamp post, a wall, a tree
    max_extent: float = 2.4       # longest horizontal span (a lying victim is 1.7 m)
    max_width: float = 1.2        # the SHORTER horizontal span
    min_pixels: int = 4           # 4 px is a 0.5 m target at ~25 m
    max_pixels: int = 4000        # anything this big is scenery
    max_range: float = 28.0       # inside the 30 m depth ceiling
    cell: float = 1.0             # ground grid, metres
    ground_win: int = 5           # neighbourhood for the ground minimum, in cells
    stride: int = 1               # pixel decimation (2 halves the cost, halves the range)
    # Depth normalisation ceiling, which is per FAMILY: SAR stores over 30 m
    # (SAR_DEPTH_MAX_M), autopilot over 20 m (DEPTH_MAX_M). Wrong value = every
    # range silently scaled, so it travels with the config rather than the module.
    depth_max: float = DEPTH_MAX_M


@dataclass
class Proposal:
    centre: np.ndarray            # (3,) world centroid of the object's points
    box: tuple                    # (u0, v0, u1, v1) pixels
    height: float                 # metres above local ground
    extent: float                 # longest horizontal span, metres
    width: float                  # shorter horizontal span, metres
    n_px: int
    rng: float                    # median range to the object, metres


def _rays(res: int, fov_deg: float, stride: int) -> np.ndarray:
    """Unit ray per pixel in the DRONE BODY frame (x fwd, y left, z up), cached."""
    key = (res, round(float(fov_deg), 4), stride)
    if key not in _RAY_CACHE:
        half = np.tan(np.radians(fov_deg) * 0.5)
        idx = np.arange(0, res, stride)
        a = (2.0 * (idx + 0.5) / res - 1.0) * half        # + right, columns
        b = (1.0 - 2.0 * (idx + 0.5) / res) * half        # + up, rows
        n = len(idx)
        d = np.empty((n, n, 3), dtype=np.float32)
        d[..., 0] = 1.0
        d[..., 1] = -a[None, :]                            # right is -Y in body frame
        d[..., 2] = b[:, None]
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        _RAY_CACHE[key] = d
    return _RAY_CACHE[key]


def heights(pos, R, depth_img, cfg: Optional[GeomConfig] = None,
            fov_deg: float = FOV_DEG, res: int = RES):
    """(points, height_above_ground, valid) for every sampled pixel.

    `points` is (H,W,3) world coordinates, `height` is (H,W) metres above the
    local ground surface, `valid` masks out saturated and clipped depth.
    """
    cfg = cfg or GeomConfig()
    pts, valid, rng = _cloud(pos, R, depth_img, cfg, fov_deg, res)
    if pts is None:
        return None, None, valid

    ground = _ground_surface(pts, valid, cfg)
    if ground is None:
        return pts, None, valid
    gi, gj, gz = ground                       # gi/gj are per VALID pixel
    h = np.full(rng.shape, np.nan, dtype=np.float32)
    h[valid] = pts[..., 2][valid] - gz[gi, gj]
    return pts, h, valid


def terrain_ahead(pos, R, depth_img, cfg: Optional[GeomConfig] = None,
                  fov_deg: float = FOV_DEG, res: int = RES,
                  span: float = 20.0, k: int = 8, lateral: float = 1.2,
                  fine_cell: float = 0.4, overhead_margin: float = 2.0):
    """`[(distance_m, surface_z), ...]` along the heading — the teacher's terrain
    fan, rebuilt from the depth frame instead of privileged ray casts.

    `GroundTruth.ahead_samples` drops rays down from above each sample point and
    reports what they hit; it is the input the whole flight envelope is built on
    (`geometry.required_clearance_z` turns it into "how high must I be by the time
    I get there"). It is also the one privileged input the teacher's own comments
    call a stand-in for exactly this image: "the student's depth camera sees this
    slope out to 30 m".

    The height per cell is the MAXIMUM back-projected z, not the minimum used for
    victim segmentation, and the difference is the whole point: a downward probe
    onto a rooftop reports the ROOF, and a fan that reported the street under it
    would fly the drone through the building.

    Cells the camera cannot see contribute nothing, which matches the privileged
    version's behaviour when a ray finds no geometry — a void is not an obstacle.
    """
    cfg = cfg or GeomConfig()
    pts, valid, _rng = _cloud(pos, R, depth_img, cfg, fov_deg, res)
    if pts is None or not valid.any():
        return ()
    # A FINE grid, point-sampled — the privileged ray this replaces has zero
    # width. Coarse cells or a neighbourhood max turn "there is a gap here" into
    # "there is a tree near here", and in a forest there is a tree within a metre
    # of everything: measured on seed 1027, the privileged ray found ground at
    # 0.2 m where a 1 m neighbourhood max reported 10.0 m of canopy. The teacher
    # then climbs to clear canopy it was meant to fly under, and goes blind.
    cell = min(cfg.cell, fine_cell)
    grid = _cell_extreme(pts, valid, cfg, mode="max", cell=cell)
    if grid is None:
        return ()
    zmax, x0, y0, ni, nj = grid

    R = np.asarray(R, dtype=float)
    fwd = R[:, 0].copy()
    fwd[2] = 0.0
    n = float(np.linalg.norm(fwd))
    if n < 1e-6:
        return ()
    fwd /= n
    left = np.array([-fwd[1], fwd[0], 0.0])
    pos = np.asarray(pos, dtype=float)
    offsets = (0.0,) if lateral <= 0 else (-lateral, 0.0, lateral)

    # A PROGRESSIVE CLIMB WINDOW: only report surfaces within reach overhead.
    #
    # `required_clearance_z` answers every sample with "be 5 m above that", and
    # near samples dominate. A camera that honestly reports forest canopy 2.5 m
    # ahead at z=9.4 therefore demands the drone climb to 13.15 — out of the band
    # it is supposed to sweep in. The privileged ray only escapes this by luck:
    # being zero-width it slips between trees and reports the ground beneath.
    #
    # So the honest fan is the wrong input for this teacher, whose forest strategy
    # is explicitly to fly at 6 m AGL UNDER the canopy. Reporting only what is
    # within `overhead_margin` above us keeps the drone reacting to terrain,
    # rooftops and walls it can actually clear, while ignoring the overhead
    # structure it is meant to fly through — and because the window rises with the
    # drone, a genuinely tall obstacle still comes into view progressively as it
    # climbs, instead of demanding treetop altitude in one jump.
    ceiling = float(pos[2]) + float(overhead_margin) if np.isfinite(overhead_margin) \
        else np.inf

    out = []
    for j in range(1, max(1, k) + 1):
        dist = span * j / max(1, k)
        for o in offsets:
            q = pos + fwd * dist + left * o
            i = int((q[0] - x0) / cell)
            m = int((q[1] - y0) / cell)
            if not (0 <= i < ni and 0 <= m < nj):
                continue
            z = zmax[i, m]
            if np.isfinite(z) and z <= ceiling:
                out.append((float(dist), float(z)))
    return tuple(out)


def _cloud(pos, R, depth_img, cfg: GeomConfig, fov_deg: float, res: int):
    """(points, valid, range) for the sampled pixels — shared by every consumer."""
    d = np.asarray(depth_img, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    d = d[::cfg.stride, ::cfg.stride]
    rng = depth_to_meters(d, cfg.depth_max).astype(np.float32)
    valid = (rng < min(cfg.max_range, cfg.depth_max - 0.25)) & (rng > DEPTH_MIN_M + 1e-3)
    if not valid.any():
        return None, valid, rng
    R = np.asarray(R, dtype=np.float32)
    body = _rays(res, fov_deg, cfg.stride)
    dirs = body @ R.T
    fwd, up = R[:, 0], R[:, 2]
    cam = np.asarray(pos, dtype=np.float32) + CAM_FWD * fwd + CAM_UP * up
    # The stored depth is PLANAR — pybullet's buffer measures along the optical
    # axis, and _process_depth's inversion (far*near/(far - buf*(far-near)))
    # preserves that. Multiplying a unit ray by it therefore lands every pixel
    # short by cos(angle-off-axis): nothing at the frame centre, 19% at the
    # bottom rows of a 90 deg frame. Verified against p.rayTest — pixel (110,64)
    # read 4.14 m where the true ray was 5.14 m, and 4.14/cos = 5.11.
    # body[..., 0] is the ray's forward component, which IS cos(theta).
    euclid = rng / np.maximum(body[..., 0], 1e-6)
    return cam[None, None, :] + euclid[..., None] * dirs, valid, euclid


def _cell_extreme(pts, valid, cfg: GeomConfig, mode: str = "min", cell=None):
    """Per world-XY cell min (ground) or max (obstacle top) of the point cloud."""
    cell = float(cfg.cell if cell is None else cell)
    x, y, z = pts[..., 0][valid], pts[..., 1][valid], pts[..., 2][valid]
    if x.size < 16:
        return None
    x0, y0 = float(x.min()), float(y.min())
    gi = ((x - x0) / cell).astype(np.int32)
    gj = ((y - y0) / cell).astype(np.int32)
    ni, nj = int(gi.max()) + 1, int(gj.max()) + 1
    if ni * nj > 4_000_000:
        return None
    flat = gi * nj + gj
    fill = np.inf if mode == "min" else -np.inf
    g = np.full(ni * nj, fill, dtype=np.float32)
    (np.minimum.at if mode == "min" else np.maximum.at)(g, flat, z)
    g = g.reshape(ni, nj)
    g[~np.isfinite(g)] = np.nan
    return g, x0, y0, ni, nj


def _ground_surface(pts, valid, cfg: GeomConfig):
    """Lowest world Z per XY cell, then a minimum over a cell neighbourhood.

    The neighbourhood is what makes it work: a per-cell minimum alone would call
    the victim its own ground wherever the victim fills a cell, but nothing 0.5 m
    across can be the lowest point within `ground_win` metres of itself.
    """
    from scipy import ndimage

    x, y, z = pts[..., 0][valid], pts[..., 1][valid], pts[..., 2][valid]
    if x.size < 16:
        return None
    x0, y0 = float(x.min()), float(y.min())
    gi = ((x - x0) / cfg.cell).astype(np.int32)
    gj = ((y - y0) / cfg.cell).astype(np.int32)
    ni, nj = int(gi.max()) + 1, int(gj.max()) + 1
    if ni * nj > 4_000_000:                       # degenerate frame, skip
        return None
    flat = gi * nj + gj
    gz = np.full(ni * nj, np.inf, dtype=np.float32)
    np.minimum.at(gz, flat, z)
    gz = gz.reshape(ni, nj)
    empty = ~np.isfinite(gz)
    if empty.any():                               # fill holes so the filter behaves
        gz[empty] = np.nanmax(gz[~empty]) if (~empty).any() else 0.0
    gz = ndimage.minimum_filter(gz, size=cfg.ground_win, mode="nearest")
    return gi, gj, gz


def propose(pos, R, depth_img, cfg: Optional[GeomConfig] = None,
            fov_deg: float = FOV_DEG, res: int = RES) -> List[Proposal]:
    """Victim-sized objects standing on the ground, as image+world regions."""
    from scipy import ndimage

    cfg = cfg or GeomConfig()
    pts, h, valid = heights(pos, R, depth_img, cfg, fov_deg, res)
    if h is None:
        return []
    # Segment on "above ground" ONLY, and apply max_height to the finished object.
    # Putting the ceiling in the mask instead cuts a lamp post off at 2.2 m and
    # hands back its lower half, which is indistinguishable from a person — a
    # 6 m post measured 2.09 m tall and passed every filter. An object has to be
    # measured whole to be rejected for being too tall.
    mask = valid & np.isfinite(h) & (h > cfg.min_height)
    if not mask.any():
        return []

    lab, n = ndimage.label(mask)
    if n == 0:
        return []
    out: List[Proposal] = []
    objs = ndimage.find_objects(lab)
    for k, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        sub = lab[sl] == k
        npx = int(sub.sum())
        if npx < cfg.min_pixels or npx > cfg.max_pixels:
            continue
        p = pts[sl][sub]
        xy = p[:, :2]
        # principal horizontal spans, so a lying victim is measured along its own
        # axis rather than along the world axes
        c = xy.mean(0)
        q = xy - c
        try:
            _u, s, _vt = np.linalg.svd(q, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        span = 2.0 * s / max(1.0, np.sqrt(len(q)))
        extent, width = float(span[0]), float(span[1] if len(span) > 1 else 0.0)
        hh = float(np.nanmax(h[sl][sub]))
        if hh > cfg.max_height or extent > cfg.max_extent or width > cfg.max_width:
            continue
        rows, cols = np.nonzero(sub)
        r0, c0 = sl[0].start, sl[1].start
        out.append(Proposal(
            centre=p.mean(0),
            box=(float((c0 + cols.min()) * cfg.stride), float((r0 + rows.min()) * cfg.stride),
                 float((c0 + cols.max()) * cfg.stride), float((r0 + rows.max()) * cfg.stride)),
            height=hh, extent=extent, width=width, n_px=npx,
            rng=float(np.median(np.linalg.norm(p - np.asarray(pos, float), axis=1))),
        ))
    return out
