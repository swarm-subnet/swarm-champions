"""Pick a flight direction by asking which directions are actually free.

Our obstacle handling reasons in one dimension (`_clearance`: is something ahead)
or two (`_steer_around`: which BEARING is clear). Neither can express "up and to
the left, between those two trunks and under that branch" — and in a forest the
free path is almost always diagonal. The replays showed the consequence: drones
squiggling beside a pad, knotting up against a trunk, or climbing over the canopy
because `up` was the only escape the controller could represent.

This answers a different question, in full 3D: for each of ~K^2 candidate
directions, how far could a sphere of the drone's radius travel along it before
touching anything in the depth frame? Then take the direction that is clear
enough and closest in angle to where we wanted to go.

    range along a candidate d to obstacle point p:
        proj      = d . p                      (how far along d the point lies)
        lateral^2 = |p|^2 - proj^2             (perpendicular miss distance)
        hit       iff proj > 0 and lateral^2 < r_eff^2
        distance  = proj - sqrt(r_eff^2 - lateral^2)   (sphere touches, not centre)

which is a swept-sphere test, vectorised over every (direction, point) pair as a
single matmul. No learning, no weights — the geometry is exact given the depth.

Depth here is PLANAR (measured along the optical axis), so the euclidean range is
`planar / ray_x`; getting that backwards puts every point at the wrong distance
and was a real bug in this codebase once already (team/detector/geom.py).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

_RAY_CACHE: dict = {}


def _pool_min(d: np.ndarray, k: int) -> np.ndarray:
    """Min-pool to (k,k). MIN, not mean: a cell containing a branch must report
    the branch, not the sky behind it. Averaging is how a thin obstacle vanishes."""
    h, w = d.shape
    kh, kw = h // k, w // k
    if kh < 1 or kw < 1:
        return d
    return d[:kh * k, :kw * k].reshape(k, kh, k, kw).min(axis=(1, 3))


def _rays(k: int, fov_deg: float) -> np.ndarray:
    """(k*k, 3) unit rays in the BODY frame — x forward, y left, z up.

    Same convention as team/detector/geom.py::_rays, so the two agree about where
    a pixel points.
    """
    key = (k, round(float(fov_deg), 4))
    if key not in _RAY_CACHE:
        half = np.tan(np.radians(fov_deg) * 0.5)
        idx = np.arange(k)
        a = (2.0 * (idx + 0.5) / k - 1.0) * half        # + right, columns
        b = (1.0 - 2.0 * (idx + 0.5) / k) * half        # + up, rows
        d = np.empty((k, k, 3), dtype=np.float32)
        d[..., 0] = 1.0
        d[..., 1] = -a[None, :]                          # right is -Y in body
        d[..., 2] = b[:, None]
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        _RAY_CACHE[key] = d.reshape(-1, 3).astype(np.float32)
    return _RAY_CACHE[key]


def free_direction(depth_img,
                   v_des_body,
                   depth_max: float,
                   fov_deg: float = 90.0,
                   grid: int = 24,
                   radius: float = 0.12,
                   margin: float = 0.35,
                   want_clear: float = 6.0,
                   lookahead: float = 10.0,
                   depth_min: float = 0.5) -> Optional[np.ndarray]:
    """Unit direction in the BODY frame: clear if possible, else least-bad.

    `v_des_body` is where we would like to go, also body frame. Returns None when
    the frame carries no usable geometry, so the caller can keep its own command.
    """
    d = np.asarray(depth_img, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    if d.ndim != 2 or d.size == 0:
        return None
    want = np.asarray(v_des_body, dtype=np.float32).reshape(3)
    n = float(np.linalg.norm(want))
    if n < 1e-6:
        return None
    want = want / n

    dirs = _rays(grid, fov_deg)                       # (M,3) unit, body frame
    pooled = _pool_min(d, grid).reshape(-1)           # (M,) normalised planar
    planar = pooled * (depth_max - depth_min) + depth_min
    euclid = planar / np.maximum(dirs[:, 0], 1e-6)    # planar -> euclidean

    pts = dirs * euclid[:, None]                      # (M,3) surface points
    r_eff = float(radius + margin)

    # Only points that could matter: anything past the lookahead cannot be hit
    # inside it, and dropping them shrinks the matmul.
    keep = euclid <= (lookahead + r_eff)
    if not np.any(keep):
        return want                                   # nothing near: go as asked
    P = pts[keep]                                     # (Q,3)

    proj = dirs @ P.T                                 # (M,Q)
    lat2 = np.maximum(np.sum(P * P, axis=1)[None, :] - proj * proj, 0.0)
    hit = (proj > 0.0) & (lat2 < r_eff * r_eff)
    dist = np.full(proj.shape, np.inf, dtype=np.float32)
    if np.any(hit):
        # distance to the sphere touching, not to the centre passing
        dist[hit] = np.maximum(proj[hit] - np.sqrt(r_eff * r_eff - lat2[hit]), 0.0)
    clear = np.minimum(dist.min(axis=1), lookahead).astype(np.float32)

    ang = np.arccos(np.clip(dirs @ want, -1.0, 1.0))
    safe = np.flatnonzero(clear >= want_clear)
    if safe.size:
        # among directions that are clear enough, the one closest to what we
        # wanted; ties broken toward more room
        best = safe[np.lexsort((-clear[safe], ang[safe]))[0]]
    else:
        # nothing is clear enough: take the roomiest, ties toward our heading
        best = int(np.lexsort((ang, -clear))[0])
    return dirs[int(best)].astype(np.float32)
