\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\

from __future__ import annotations

from dataclasses import dataclass

from typing import List, Optional

import numpy as np

from .localize import (CAM_FWD, CAM_UP, DEPTH_MAX_M, DEPTH_MIN_M, FOV_DEG, RES,
                       depth_to_meters)

_RAY_CACHE: dict = {}

@dataclass
class GeomConfig:

    min_height: float = 0.35
    max_height: float = 2.2
    max_extent: float = 2.4
    max_width: float = 1.2
    min_pixels: int = 4
    max_pixels: int = 4000
    max_range: float = 28.0
    cell: float = 1.0
    ground_win: int = 5
    stride: int = 1

    depth_max: float = DEPTH_MAX_M

def _rays(res: int, fov_deg: float, stride: int) -> np.ndarray:

    key = (res, round(float(fov_deg), 4), stride)
    if key not in _RAY_CACHE:
        half = np.tan(np.radians(fov_deg) * 0.5)
        idx = np.arange(0, res, stride)
        a = (2.0 * (idx + 0.5) / res - 1.0) * half
        b = (1.0 - 2.0 * (idx + 0.5) / res) * half
        n = len(idx)
        d = np.empty((n, n, 3), dtype=np.float32)
        d[..., 0] = 1.0
        d[..., 1] = -a[None, :]
        d[..., 2] = b[:, None]
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        _RAY_CACHE[key] = d
    return _RAY_CACHE[key]

def heights(pos, R, depth_img, cfg: Optional[GeomConfig] = None,
            fov_deg: float = FOV_DEG, res: int = RES):
\
\
\
\

    cfg = cfg or GeomConfig()
    pts, valid, rng = _cloud(pos, R, depth_img, cfg, fov_deg, res)
    if pts is None:
        return None, None, valid

    ground = _ground_surface(pts, valid, cfg)
    if ground is None:
        return pts, None, valid
    gi, gj, gz = ground
    h = np.full(rng.shape, np.nan, dtype=np.float32)
    h[valid] = pts[..., 2][valid] - gz[gi, gj]
    return pts, h, valid

def _cloud(pos, R, depth_img, cfg: GeomConfig, fov_deg: float, res: int):

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

    euclid = rng / np.maximum(body[..., 0], 1e-6)
    return cam[None, None, :] + euclid[..., None] * dirs, valid, euclid

def _ground_surface(pts, valid, cfg: GeomConfig):
\
\
\
\
\

    from scipy import ndimage

    x, y, z = pts[..., 0][valid], pts[..., 1][valid], pts[..., 2][valid]
    if x.size < 16:
        return None
    x0, y0 = float(x.min()), float(y.min())
    gi = ((x - x0) / cfg.cell).astype(np.int32)
    gj = ((y - y0) / cfg.cell).astype(np.int32)
    ni, nj = int(gi.max()) + 1, int(gj.max()) + 1
    if ni * nj > 4_000_000:
        return None
    flat = gi * nj + gj
    gz = np.full(ni * nj, np.inf, dtype=np.float32)
    np.minimum.at(gz, flat, z)
    gz = gz.reshape(ni, nj)
    empty = ~np.isfinite(gz)
    if empty.any():
        gz[empty] = np.nanmax(gz[~empty]) if (~empty).any() else 0.0
    gz = ndimage.minimum_filter(gz, size=cfg.ground_win, mode="nearest")
    return gi, gj, gz

