from __future__ import annotations
from typing import Optional
import numpy as np
_RAY_CACHE: dict = {}
def _pool_min(d: np.ndarray, k: int) -> np.ndarray:
    h, w = d.shape
    kh, kw = h // k, w // k
    if kh < 1 or kw < 1:
        return d
    return d[:kh * k, :kw * k].reshape(k, kh, k, kw).min(axis=(1, 3))
def _rays(k: int, fov_deg: float) -> np.ndarray:
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
def _edges(n: int, k: int) -> np.ndarray:
    return np.round(np.arange(k + 1) * n / k).astype(int)


def _pool_min_aligned(d: np.ndarray, k: int) -> np.ndarray:
    """Min-pool the whole frame into k x k blocks of 5-6 px (the shipped _pool_min crops to
    k*(n//k) px and never reads the last rows/columns)."""
    h, w = d.shape
    er, ec = _edges(h, k), _edges(w, k)
    rows = np.minimum.reduceat(d, er[:-1], axis=0)
    return np.minimum.reduceat(rows, ec[:-1], axis=1)


_RAY_CACHE_AL: dict = {}


def _rays_aligned(h: int, w: int, k: int, fov_deg: float) -> np.ndarray:
    """One ray through the centre of each pooled block. The shipped _rays places ray j at
    (j+0.5)/k of the full frame while its block sits at (j+0.5)*(n//k)/n, so pooled obstacles are
    re-projected 3-6 deg right of and below where they are."""
    key = (h, w, k, round(float(fov_deg), 4))
    if key not in _RAY_CACHE_AL:
        half = np.tan(np.radians(fov_deg) * 0.5)
        er, ec = _edges(h, k), _edges(w, k)
        uc = 0.5 * (ec[:-1] + ec[1:]) / w
        vr = 0.5 * (er[:-1] + er[1:]) / h
        a = (2.0 * uc - 1.0) * half
        b = (1.0 - 2.0 * vr) * half
        dd = np.empty((k, k, 3), dtype=np.float32)
        dd[..., 0] = 1.0
        dd[..., 1] = -a[None, :]
        dd[..., 2] = b[:, None]
        dd /= np.linalg.norm(dd, axis=-1, keepdims=True)
        _RAY_CACHE_AL[key] = dd.reshape(-1, 3).astype(np.float32)
    return _RAY_CACHE_AL[key]


def _grid(d: np.ndarray, grid: int, fov_deg: float, aligned: bool):
    if aligned:
        h, w = d.shape
        return _rays_aligned(h, w, grid, fov_deg), _pool_min_aligned(d, grid).reshape(-1)
    return _rays(grid, fov_deg), _pool_min(d, grid).reshape(-1)


LAST_CLEAR = [None]   # tube clearance of the last free_direction pick (read right after the call)


def free_direction(depth_img,
                   v_des_body,
                   depth_max: float,
                   fov_deg: float = 90.0,
                   grid: int = 24,
                   radius: float = 0.12,
                   margin: float = 0.35,
                   want_clear: float = 6.0,
                   lookahead: float = 10.0,
                   depth_min: float = 0.5,
                   half_deg: float = 0.0,
                   aligned: bool = False,
                   el_max_world: Optional[float] = None,
                   R: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
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
    dirs, pooled = _grid(d, grid, fov_deg, aligned)
    planar = pooled * (depth_max - depth_min) + depth_min
    euclid = planar / np.maximum(dirs[:, 0], 1e-6)    
    pts = dirs * euclid[:, None]                      
    r_eff = float(radius + margin)
    keep = euclid <= (lookahead + r_eff)
    if not np.any(keep):
        LAST_CLEAR[0] = float(lookahead)
        return want                                   
    P = pts[keep]                                     
    proj = dirs @ P.T                                 
    lat2 = np.maximum(np.sum(P * P, axis=1)[None, :] - proj * proj, 0.0)
    hit = (proj > 0.0) & (lat2 < r_eff * r_eff)
    dist = np.full(proj.shape, np.inf, dtype=np.float32)
    if np.any(hit):
        dist[hit] = np.maximum(proj[hit] - np.sqrt(r_eff * r_eff - lat2[hit]), 0.0)
    clear = np.minimum(dist.min(axis=1), lookahead).astype(np.float32)
    raw_clear = clear
    ang = np.arccos(np.clip(dirs @ want, -1.0, 1.0))
    if el_max_world is not None and R is not None:
        # rays climbing faster than the ceiling allows are not flyable (the caller clamps the climb)
        wz = (np.asarray(R, np.float32) @ dirs.T).T[:, 2]
        clear = np.where(wz <= float(np.sin(el_max_world)), clear, -1.0).astype(np.float32)
    if half_deg > 0.0:
        # only headings at least (fov/2 - half_deg) inside the frame edge: an obstacle just
        # outside the view cannot be seen, so a path along the edge is half blind
        az = np.degrees(np.abs(np.arctan2(dirs[:, 1], dirs[:, 0])))
        clear = np.where(az <= half_deg, clear, -1.0).astype(np.float32)
    safe = np.flatnonzero(clear >= want_clear)
    if safe.size:
        best = safe[np.lexsort((-clear[safe], ang[safe]))[0]]
    else:
        best = int(np.lexsort((ang, -clear))[0])
    LAST_CLEAR[0] = float(raw_clear[int(best)])
    return dirs[int(best)].astype(np.float32)
def ray_clearance(depth_img,
                  dir_body,
                  depth_max: float,
                  fov_deg: float = 90.0,
                  grid: int = 24,
                  r_eff: float = 0.47,
                  lookahead: float = 12.0,
                  depth_min: float = 0.5,
                  aligned: bool = False) -> float:
    """Distance along one body-frame direction to the first depth point within r_eff of that
    ray (the same pooled point cloud free_direction uses), capped at lookahead."""
    d = np.asarray(depth_img, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    u = np.asarray(dir_body, dtype=np.float32).reshape(3)
    n = float(np.linalg.norm(u))
    if d.ndim != 2 or d.size == 0 or n < 1e-6:
        return float(lookahead)
    u = u / n
    dirs, pooled = _grid(d, grid, fov_deg, aligned)
    planar = pooled * (depth_max - depth_min) + depth_min
    euclid = planar / np.maximum(dirs[:, 0], 1e-6)
    keep = euclid <= (lookahead + r_eff)
    if not np.any(keep):
        return float(lookahead)
    P = dirs[keep] * euclid[keep][:, None]
    proj = P @ u
    lat2 = np.maximum(np.sum(P * P, axis=1) - proj * proj, 0.0)
    hit = (proj > 0.0) & (lat2 < r_eff * r_eff)
    if not np.any(hit):
        return float(lookahead)
    dist = np.maximum(proj[hit] - np.sqrt(r_eff * r_eff - lat2[hit]), 0.0)
    return float(min(float(dist.min()), lookahead))
