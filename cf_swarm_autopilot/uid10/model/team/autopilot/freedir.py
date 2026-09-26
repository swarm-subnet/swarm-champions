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
                   R: Optional[np.ndarray] = None,
                   el_min_world: Optional[float] = None,
                   extra_pts: Optional[np.ndarray] = None,
                   lat_target: float = 0.0,
                   lat_w: float = 1.0,
                   extra_wide: Optional[np.ndarray] = None,
                   wide_add: float = 0.0) -> Optional[np.ndarray]:
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
    P = pts[keep]
    if extra_pts is not None:
        # extra body-frame obstacle points (extrude_pts): part of the cloud for every ray's tube test
        P = _with_extra(P, extra_pts, lookahead + r_eff)
    if P.shape[0] == 0:
        LAST_CLEAR[0] = float(lookahead)
        return want
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
    if el_min_world is not None and R is not None:
        # collide FND: rays descending steeper than el_min_world are not flyable
        wz = (np.asarray(R, np.float32) @ dirs.T).T[:, 2]
        clear = np.where(wz >= float(np.sin(el_min_world)), clear, -1.0).astype(np.float32)
    if half_deg > 0.0:
        # only headings at least (fov/2 - half_deg) inside the frame edge: an obstacle just
        # outside the view cannot be seen, so a path along the edge is half blind
        az = np.degrees(np.abs(np.arctan2(dirs[:, 1], dirs[:, 0])))
        clear = np.where(az <= half_deg, clear, -1.0).astype(np.float32)
    safe = np.flatnonzero(clear >= want_clear)
    if safe.size and lat_target > 0.0:
        # DEV wide pass: among the safe rays prefer the one whose closest point over the next
        # want_clear metres lies farthest to the side, traded against the angle off the command
        near_pts = (proj > 0.0) & (proj < want_clear)
        lat_min = np.where(near_pts, np.sqrt(lat2), np.inf).min(axis=1)
        pen = np.maximum(0.0, 1.0 - np.minimum(lat_min, lat_target) / lat_target)
        score = ang / np.radians(45.0) + float(lat_w) * pen
        best = safe[np.lexsort((-clear[safe], score[safe]))[0]]
    elif safe.size:
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
                  aligned: bool = False,
                  extra_pts: Optional[np.ndarray] = None,
                  extra_wide: Optional[np.ndarray] = None,
                  wide_add: float = 0.0) -> float:
    """Distance along one body-frame direction to the first depth point within r_eff of that
    ray (the same pooled point cloud free_direction uses, plus extra_pts), capped at lookahead."""
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
    P = dirs[keep] * euclid[keep][:, None]
    if extra_pts is not None:
        P = _with_extra(P, extra_pts, lookahead + r_eff)
    best = float(lookahead)
    if extra_wide is not None and float(wide_add) > 0.0:
        # collide BNH: roof-only extruded points with the wider tube
        rw = r_eff + float(wide_add)
        Pw = _with_extra(np.zeros((0, 3), np.float32), extra_wide, lookahead + rw)
        if Pw.shape[0]:
            pj = Pw @ u
            l2 = np.maximum(np.sum(Pw * Pw, axis=1) - pj * pj, 0.0)
            hw = (pj > 0.0) & (l2 < rw * rw)
            if np.any(hw):
                best = min(best, float(np.maximum(pj[hw] - np.sqrt(rw * rw - l2[hw]), 0.0).min()))
    if P.shape[0] == 0:
        return best
    proj = P @ u
    lat2 = np.maximum(np.sum(P * P, axis=1) - proj * proj, 0.0)
    hit = (proj > 0.0) & (lat2 < r_eff * r_eff)
    if not np.any(hit):
        return best
    dist = np.maximum(proj[hit] - np.sqrt(r_eff * r_eff - lat2[hit]), 0.0)
    return float(min(float(dist.min()), best))


def _with_extra(P: np.ndarray, extra_pts, r_max: float) -> np.ndarray:
    """P with the extra body-frame points within r_max of the eye appended (float32)."""
    E = np.asarray(extra_pts, dtype=np.float32).reshape(-1, 3)
    if E.shape[0] == 0:
        return P
    E = E[np.linalg.norm(E, axis=1) <= float(r_max)]
    if E.shape[0] == 0:
        return P
    return np.vstack([P, E]) if P.shape[0] else E


def _cloud(d: np.ndarray, grid: int, fov_deg: float, depth_max: float, depth_min: float, aligned: bool):
    dirs, pooled = _grid(d, grid, fov_deg, aligned)
    euclid = (pooled * (depth_max - depth_min) + depth_min) / np.maximum(dirs[:, 0], 1e-6)
    return dirs, euclid


def extrude_pts(depth_img, R, eye_z: float,
                band=(2.0, 3.2), rng: float = 10.0, dz=(0.0, 0.8, 1.6, 2.4),
                vox: float = 0.3, cap: int = 200, grid: int = 24, fov_deg: float = 90.0,
                depth_max: float = 20.0, depth_min: float = 0.5, aligned: bool = True,
                max_drop: float = 4.0) -> Optional[np.ndarray]:
    """CG awning extrusion (research/beat_f8kw/city_coll). Pooled depth points whose WORLD height
    (eye_z + world-frame offset) lies in band and within rng m horizontally, and at most max_drop m
    below the eye (<= 0: no limit), are copied as a vertical column at dz m above the eye (one per
    vox x vox m cell, the cap nearest cells). Returns body-frame (N, 3) float32 points for
    free_direction / ray_clearance extra_pts, or None. City collision hulls are convex: building-n's
    2.5 m awning becomes a slab 1.7-2.7 m in front of the drawn wall at cruise height, which the
    camera never sees; the column makes the awning plate a wall at the drone's height."""
    d = np.asarray(depth_img, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    if d.ndim != 2 or d.size == 0:
        return None
    dirs, eu = _cloud(d, grid, fov_deg, depth_max, depth_min, aligned)
    P = dirs * eu[:, None]
    Rm = np.asarray(R, dtype=float)
    W = (Rm @ P.T).T
    zw = float(eye_z) + W[:, 2]
    m = (zw >= float(band[0])) & (zw <= float(band[1])) & (np.hypot(W[:, 0], W[:, 1]) <= float(rng))
    if max_drop is not None and float(max_drop) > 0.0:
        m &= (-W[:, 2] <= float(max_drop))
    if not np.any(m):
        return None
    V0 = W[m]
    if float(vox) > 0.0:
        cell = np.round(V0[:, :2] / float(vox)).astype(np.int64)
        _, idx = np.unique(cell, axis=0, return_index=True)
        V0 = V0[idx]
    if int(cap) > 0 and len(V0) > int(cap):
        V0 = V0[np.argsort(np.hypot(V0[:, 0], V0[:, 1]), kind="stable")[:int(cap)]]
    V = np.concatenate([np.column_stack([V0[:, 0], V0[:, 1], np.full(len(V0), float(h))])
                        for h in tuple(dz)], axis=0)
    return (Rm.T @ V.T).T.astype(np.float32)


def near_dir(depth_img, r_near: float, grid: int = 24, fov_deg: float = 90.0,
             depth_max: float = 20.0, depth_min: float = 0.5, aligned: bool = True) -> Optional[np.ndarray]:
    """CG saturation escape: unit horizontal body-frame direction toward the pooled depth points
    within r_near m of the eye (1/d weighted), or None."""
    d = np.asarray(depth_img, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    if d.ndim != 2 or d.size == 0:
        return None
    dirs, eu = _cloud(d, grid, fov_deg, depth_max, depth_min, aligned)
    close = eu <= float(r_near)
    if not np.any(close):
        return None
    w = 1.0 / np.maximum(eu[close], 0.05)
    m = (dirs[close] * w[:, None]).sum(axis=0).astype(float)
    m[2] = 0.0
    n = float(np.linalg.norm(m))
    return None if n < 1e-6 else m / n


def extrude_pts2(depth_img, R, eye_z: float,
                 band=(2.0, 3.2), rng: float = 10.0, dz=(0.0, 0.8, 1.6, 2.4),
                 vox: float = 0.3, cap: int = 200, grid: int = 24, fov_deg: float = 90.0,
                 depth_max: float = 20.0, depth_min: float = 0.5, aligned: bool = True,
                 max_drop: float = 4.0, above: float = 0.4, cell: float = 0.6):
    """collide BNH: extrude_pts (same points, same order) plus the columns of the ROOF-ONLY band points: those with
    no pooled point higher than band[1] + above within one cell (3 x 3 cells of `cell` m) - a low roof / podium /
    awning top, not the foot of a wall. Returns (all_pts or None, roof_pts or None), body frame float32."""
    allp = extrude_pts(depth_img, R, eye_z, band=band, rng=rng, dz=dz, vox=vox, cap=cap, grid=grid,
                       fov_deg=fov_deg, depth_max=depth_max, depth_min=depth_min, aligned=aligned,
                       max_drop=max_drop)
    if allp is None:
        return None, None
    d = np.asarray(depth_img, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    dirs, eu = _cloud(d, grid, fov_deg, depth_max, depth_min, aligned)
    P = dirs * eu[:, None]
    Rm = np.asarray(R, dtype=float)
    W = (Rm @ P.T).T
    zw = float(eye_z) + W[:, 2]
    valid = eu * dirs[:, 0] < depth_max - 0.3
    m = valid & (zw >= float(band[0])) & (zw <= float(band[1])) & (np.hypot(W[:, 0], W[:, 1]) <= float(rng))
    if max_drop is not None and float(max_drop) > 0.0:
        m &= (-W[:, 2] <= float(max_drop))
    if not np.any(m):
        return allp, None
    hi = valid & (zw > float(band[1]) + float(above))
    V0 = W[m]
    if np.any(hi):
        H = W[hi]
        ch = set(map(tuple, np.floor(H[:, :2] / float(cell)).astype(np.int64).tolist()))
        c0 = np.floor(V0[:, :2] / float(cell)).astype(np.int64)
        keep = np.ones(len(V0), bool)
        for k in range(len(V0)):
            cx, cy = int(c0[k, 0]), int(c0[k, 1])
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    if (cx + ox, cy + oy) in ch:
                        keep[k] = False
                        break
                if not keep[k]:
                    break
        V0 = V0[keep]
    if len(V0) == 0:
        return allp, None
    if float(vox) > 0.0:
        cc = np.round(V0[:, :2] / float(vox)).astype(np.int64)
        _, idx = np.unique(cc, axis=0, return_index=True)
        V0 = V0[idx]
    if int(cap) > 0 and len(V0) > int(cap):
        V0 = V0[np.argsort(np.hypot(V0[:, 0], V0[:, 1]), kind="stable")[:int(cap)]]
    V = np.concatenate([np.column_stack([V0[:, 0], V0[:, 1], np.full(len(V0), float(h))])
                        for h in tuple(dz)], axis=0)
    return allp, (Rm.T @ V.T).T.astype(np.float32)
