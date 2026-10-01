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
@dataclass
class Proposal:
    centre: np.ndarray            
    box: tuple                    
    height: float                 
    extent: float                 
    width: float                  
    n_px: int
    rng: float                    
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
def terrain_ahead(pos, R, depth_img, cfg: Optional[GeomConfig] = None,
                  fov_deg: float = FOV_DEG, res: int = RES,
                  span: float = 20.0, k: int = 8, lateral: float = 1.2,
                  fine_cell: float = 0.4, overhead_margin: float = 2.0):
    cfg = cfg or GeomConfig()
    pts, valid, _rng = _cloud(pos, R, depth_img, cfg, fov_deg, res)
    if pts is None or not valid.any():
        return ()
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
def _cell_extreme(pts, valid, cfg: GeomConfig, mode: str = "min", cell=None):
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
def propose(pos, R, depth_img, cfg: Optional[GeomConfig] = None,
            fov_deg: float = FOV_DEG, res: int = RES) -> List[Proposal]:
    from scipy import ndimage
    cfg = cfg or GeomConfig()
    pts, h, valid = heights(pos, R, depth_img, cfg, fov_deg, res)
    if h is None:
        return []
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
