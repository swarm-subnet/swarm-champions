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
\

from __future__ import annotations

from typing import Optional

import numpy as np

_RAY_CACHE: dict = {}

def _pool_min(d: np.ndarray, k: int) -> np.ndarray:
\

    h, w = d.shape
    kh, kw = h // k, w // k
    if kh < 1 or kw < 1:
        return d
    return d[:kh * k, :kw * k].reshape(k, kh, k, kw).min(axis=(1, 3))

def _rays(k: int, fov_deg: float) -> np.ndarray:
\
\
\
\

    key = (k, round(float(fov_deg), 4))
    if key not in _RAY_CACHE:
        half = np.tan(np.radians(fov_deg) * 0.5)
        idx = np.arange(k)
        a = (2.0 * (idx + 0.5) / k - 1.0) * half
        b = (1.0 - 2.0 * (idx + 0.5) / k) * half
        d = np.empty((k, k, 3), dtype=np.float32)
        d[..., 0] = 1.0
        d[..., 1] = -a[None, :]
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
\
\
\
\

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

    dirs = _rays(grid, fov_deg)
    pooled = _pool_min(d, grid).reshape(-1)
    planar = pooled * (depth_max - depth_min) + depth_min
    euclid = planar / np.maximum(dirs[:, 0], 1e-6)

    pts = dirs * euclid[:, None]
    r_eff = float(radius + margin)

    keep = euclid <= (lookahead + r_eff)
    if not np.any(keep):
        return want
    P = pts[keep]

    proj = dirs @ P.T
    lat2 = np.maximum(np.sum(P * P, axis=1)[None, :] - proj * proj, 0.0)
    hit = (proj > 0.0) & (lat2 < r_eff * r_eff)
    dist = np.full(proj.shape, np.inf, dtype=np.float32)
    if np.any(hit):

        dist[hit] = np.maximum(proj[hit] - np.sqrt(r_eff * r_eff - lat2[hit]), 0.0)
    clear = np.minimum(dist.min(axis=1), lookahead).astype(np.float32)

    ang = np.arccos(np.clip(dirs @ want, -1.0, 1.0))
    safe = np.flatnonzero(clear >= want_clear)
    if safe.size:

        best = safe[np.lexsort((-clear[safe], ang[safe]))[0]]
    else:

        best = int(np.lexsort((ang, -clear))[0])
    return dirs[int(best)].astype(np.float32)
