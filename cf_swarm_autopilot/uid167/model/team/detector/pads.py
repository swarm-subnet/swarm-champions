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

from .geom import GeomConfig, heights

AP_RES = 128
AP_FOV_DEG = 90.0
AP_DEPTH_MAX_M = 20.0
PAD_RADIUS = 0.6

@dataclass
class PadConfig:

    min_height: float = 0.12
    max_height: float = 3.2

    max_extent: float = 3.0
    min_extent: float = 0.3
    aspect_max: float = 3.0

    flatness_max: float = 0.28
    min_pixels: int = 5
    max_pixels: int = 6000

    net_max_extent: float = 3.2

    net_max_depth_spread: float = 2.0

    net_max_aspect: float = 2.6

    net_min_short: float = 0.20

    net_max_range: float = 28.3
    max_range: float = 28.3
    cell: float = 0.8
    ground_win: int = 7
    stride: int = 1

@dataclass
class PadProposal:
    centre: np.ndarray
    height: float
    extent: float
    flatness: float
    n_px: int
    rng: float
    score: float

def _geom_cfg(cfg: PadConfig) -> GeomConfig:

    return GeomConfig(min_height=cfg.min_height, max_height=cfg.max_height,
                      min_pixels=cfg.min_pixels, max_pixels=cfg.max_pixels,
                      max_range=cfg.max_range, cell=cfg.cell,
                      ground_win=cfg.ground_win, stride=cfg.stride,
                      depth_max=AP_DEPTH_MAX_M)

def propose_from_mask(pos, R, depth_img, mask, cfg: Optional[PadConfig] = None,
                      fov_deg: float = AP_FOV_DEG,
                      res: int = AP_RES) -> List[PadProposal]:
\
\
\
\
\

    from scipy import ndimage

    from .geom import _cloud

    cfg = cfg or PadConfig()
    gcfg = _geom_cfg(cfg)
    d = np.asarray(depth_img, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    pts, valid, _rng = _cloud(pos, R, d, gcfg, fov_deg, res)
    if pts is None:
        return []
    m = np.asarray(mask, dtype=bool)
    if m.shape != valid.shape:
        return []
    m = m & valid
    if not m.any():
        return []
    lab, n = ndimage.label(m)
    out: List[PadProposal] = []
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        sub = lab[sl] == k
        npx = int(sub.sum())
        if npx < cfg.min_pixels:
            continue
        p = pts[sl][sub]
        xy = p[:, :2]
        c = np.median(xy, axis=0)
        spread = float(np.median(np.linalg.norm(xy - c, axis=1))) * 2.0
        d_pt = np.linalg.norm(p - np.asarray(pos, float), axis=1)
        rng = float(np.median(d_pt))
        if rng > cfg.net_max_range or spread > cfg.net_max_extent:
            continue

        if float(np.percentile(d_pt, 90) - np.percentile(d_pt, 10))\
                > cfg.net_max_depth_spread:
            continue

        if float(np.std(p[:, 2])) > cfg.flatness_max * 2.0:
            continue

        d_xy = xy - xy.mean(axis=0)
        ev = np.linalg.eigvalsh(np.cov(d_xy.T) + 1e-9 * np.eye(2))
        short = float(2.0 * np.sqrt(max(float(ev.min()), 0.0)))
        aspect = float(np.sqrt(max(float(ev.max()), 1e-9) / max(float(ev.min()), 1e-9)))
        if aspect > cfg.net_max_aspect or short < cfg.net_min_short:
            continue
        centre = np.array([c[0], c[1], float(np.median(p[:, 2]))])
        out.append(PadProposal(centre=centre, height=0.0, extent=spread,
                               flatness=float(np.std(p[:, 2])), n_px=npx, rng=rng,
                               score=float(min(1.0, npx / 30.0))))
    out.sort(key=lambda q_: -q_.score)
    return out

class PadNetDetector:
\
\
\
\
\

    def __init__(self, ckpt: str, device: str = "cpu", thr: float = 0.5,
                 cfg: Optional[PadConfig] = None):
        import torch

        from .padnet import PadNet

        blob = torch.load(ckpt, map_location=device)
        self.net = PadNet(width=int(blob.get("width", 16)))
        self.net.load_state_dict(blob["model"])
        self.net.eval().to(device)
        self.device = device
        self.thr = float(thr)
        self.cfg = cfg or PadConfig()
        self._torch = torch

    def masks(self, depth_batch) -> np.ndarray:

        torch = self._torch
        d = np.asarray(depth_batch, dtype=np.float32)
        if d.ndim == 4:
            d = d[..., 0]
        with torch.no_grad():
            x = torch.from_numpy(d)[:, None].to(self.device)
            pr = torch.sigmoid(self.net(x))[:, 0].cpu().numpy()
        return pr > self.thr

    def propose_batch(self, poses, rots, depth_batch) -> List[List[PadProposal]]:
        m = self.masks(depth_batch)
        d = np.asarray(depth_batch, dtype=np.float32)
        if d.ndim == 4:
            d = d[..., 0]
        return [propose_from_mask(poses[i], rots[i], d[i], m[i], self.cfg)
                for i in range(len(poses))]

def propose(pos, R, depth_img, cfg: Optional[PadConfig] = None,
            fov_deg: float = AP_FOV_DEG, res: int = AP_RES) -> List[PadProposal]:
\
\
\
\

    from scipy import ndimage

    cfg = cfg or PadConfig()
    gcfg = _geom_cfg(cfg)
    pts, h, valid = heights(pos, R, depth_img, gcfg, fov_deg, res)
    if h is None:
        return []

    mask = valid & np.isfinite(h) & (h > cfg.min_height)
    if not mask.any():
        return []
    lab, n = ndimage.label(mask)
    if n == 0:
        return []

    out: List[PadProposal] = []
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        sub = lab[sl] == k
        npx = int(sub.sum())
        if npx < cfg.min_pixels or npx > cfg.max_pixels:
            continue
        p = pts[sl][sub]
        hh = h[sl][sub]
        hh = hh[np.isfinite(hh)]
        if hh.size < cfg.min_pixels:
            continue
        top = float(np.nanmax(hh))
        if top > cfg.max_height:
            continue
        xy = p[:, :2]
        c = xy.mean(0)
        q = xy - c
        try:
            _u, s, _vt = np.linalg.svd(q, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        span = 2.0 * s / max(1.0, np.sqrt(len(q)))
        extent = float(span[0])
        width = float(span[1]) if len(span) > 1 else 0.0
        if extent > cfg.max_extent or extent < cfg.min_extent:
            continue
        if width > 1e-6 and extent / width > cfg.aspect_max:
            continue
        flat = float(np.std(hh))
        if flat > cfg.flatness_max:
            continue
        rng = float(np.median(np.linalg.norm(p - np.asarray(pos, float), axis=1)))

        centre = np.array([c[0], c[1], float(np.nanmax(p[:, 2]))])

        score = (1.0 - min(1.0, flat / cfg.flatness_max))\
            * (1.0 - min(1.0, abs(extent - 2.0 * PAD_RADIUS) / cfg.max_extent))\
            * min(1.0, npx / 40.0)
        out.append(PadProposal(centre=centre, height=top, extent=extent,
                               flatness=flat, n_px=npx, rng=rng, score=float(score)))
    out.sort(key=lambda q_: -q_.score)
    return out
