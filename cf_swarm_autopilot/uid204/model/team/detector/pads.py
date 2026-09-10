"""Landing-pad proposals from the depth frame alone — cf_swarm_autopilot.

The family gives one shared, noisy fix on the pad CENTROID (std ~10.6 m/axis) and
scatters the pads around it: only 29% lie within 30 m of the clue and the median
pairwise spread is 90 m. The optimal drone->pad assignment still has a median leg
of 33 m against a 20 m depth ceiling, so no policy can see its pad at t=0. The
pads have to be found, and this is what finds them.

Unlike the victim, a pad is a *good* geometric target, and measured
(probe_pad_camera.py, delete-and-diff against the real renderer) it stays visible
essentially to the sensor's edge:

    open 21.6 m / 0.61 m contrast    city     21.6 m / 1.36 m
    mountain 21.6 m / 0.58 m         forest   15.6 m / 2.24 m  (canopy)

So no network. A pad is a flat horizontal disc ~1.2 m across standing a little
proud of the ground, and depth says all three of those directly. The work is
reused wholesale from `geom.heights`, which already back-projects the frame and
subtracts a LOCAL ground surface (a neighbourhood minimum, immune to the
altimeter sitting on a roof — the mistake that cost the victim detector 3.49 m).

What this adds over `geom.propose` is the discriminator that suits a pad and not
a person: FLATNESS. A pad's top is a plane at constant world Z, so the spread of
heights across its pixels is millimetres where a bush, a rock or a parked car
varies by tens of centimetres. That one test does most of the rejecting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .geom import GeomConfig, heights

# The autopilot camera, which is NOT the SAR one: 128 px over the same 90 deg,
# and depth normalised over 20 m rather than 30 (swarm.constants.DEPTH_MAX_M vs
# SAR_DEPTH_MAX_M). Getting depth_max wrong scales every range by 1.5.
AP_RES = 128
AP_FOV_DEG = 90.0
AP_DEPTH_MAX_M = 20.0
PAD_RADIUS = 0.6              # swarm.constants.LANDING_PLATFORM_RADIUS


@dataclass
class PadConfig:
    # Height above local ground. Measured pad heights by map: open 0.20, village
    # 0.33, city 0.63, forest 2.43 (mountain pads sit on terrain 30-60 m up, but
    # the ground surface is local so the RELATIVE height is small there too).
    # The floor has to clear ground noise without losing an open-map pad at
    # 0.20 m, which is the tightest case on the board.
    min_height: float = 0.12
    max_height: float = 3.2       # forest pads reach 2.43 m; above this is scenery
    # A pad is 1.2 m across. Allow generously for partial views and for the disc
    # merging with whatever it stands on at long range.
    max_extent: float = 3.0       # longest horizontal span
    min_extent: float = 0.3
    aspect_max: float = 3.0       # a disc seen obliquely, not a wall or a kerb
    # THE discriminator: a pad top is a plane. Heights across its pixels should
    # agree to a few cm; a bush or a rock will not.
    flatness_max: float = 0.28     # std of height-above-ground within the blob, m
    min_pixels: int = 5           # measured: 7-14 px at 18-21 m, 240 px up close
    max_pixels: int = 6000
    # ── sanity checks applied to a LEARNED mask ──────────────────────────────
    # The net decides "pad-like appearance"; these decide "physically a pad".
    # Dropping them when the net was swapped in put a white proposal stalk on
    # every pole, log and roof edge in village 8003 — the net's own precision is
    # per-PIXEL, and a handful of stray pixels still back-projects to a confident
    # world position unless the shape is checked too.
    net_max_extent: float = 3.2    # a 1.2 m disc, generous for oblique views
    # A pad is at ONE range: its pixels span its own diameter. A pole or a tree
    # trunk spans many metres of depth in the same blob, which is the cheapest
    # way to tell them apart and needs no ground estimate.
    net_max_depth_spread: float = 2.0
    # Beyond this the back-projection is too coarse to be worth flying to; the
    # sweep will pass closer soon enough.
    # SHAPE. The pad is a disc of LANDING_PLATFORM_RADIUS = 0.6 m; a branch is a
    # stick. Until now the net path had a MAXIMUM extent test and no minimum and
    # no shape test at all, so a thin sliver of branch back-projected to a
    # confident proposal — which is what the forest replays kept showing.
    #
    # Isotropic extent cannot see this (pads p50=0.78 m, clutter p50=0.80), but
    # the ratio of the blob's principal axes can. Measured on delete-and-diff
    # ground truth, 6 seeds x 50 poses per map:
    #
    #                            keeps pads   keeps clutter
    #   forest   aspect <= 1.6       89%          60%
    #            short  >= 0.45      91%          65%
    #   village  aspect <= 1.6       88%          77%
    #
    # Forest clutter reaches p95 aspect 7.57 against 2.24 for pads. Defaults here
    # are the conservative global setting; the forest detector is built with a
    # stricter PadConfig, since that is where the branches are.
    net_max_aspect: float = 2.6    # longest/shortest principal axis of the blob
    # THE SHORT-AXIS GATE IS RANGE-DEPENDENT, and applying it as a constant is
    # what caps recall at 0.40.
    #
    # `short` is 2*sqrt(lambda_min) of the blob's point cloud -- an estimate made
    # from however many pixels the pad happens to subtend. A 1.2 m disc spans
    # 19.5 px at 5 m but only 4.9 px at 20 m, and one pixel's ground footprint at
    # 20 m is range*2*tan(45)/128 = 0.31 m. So at 20 m the threshold 0.30 m sits
    # exactly at the quantisation limit: the estimator cannot return a value above
    # it however real the pad is. The gate stops discriminating and starts acting
    # as a hard range cut at ~15 m.
    #
    # Measured over 14k frames, relaxing this one constant to 0.10 moved the real
    # network from R 0.398 -> 0.492 while precision went UP, 0.909 -> 0.915: it was
    # discarding true pads, not false ones. Aspect already rejects the slivers this
    # was meant to catch, which is why precision does not suffer.
    #
    # Scaling the gate with range was tried first and is WORSE (F1 0.585 vs 0.665):
    # it assumes the estimate only degrades far away, but partial views and
    # occlusion shrink the short axis at every range, so the gate misfires
    # throughout. Lowering the constant is the correct fix.
    #
    # THIS CHANGE IS MARGINAL AND THE FIRST JUSTIFICATION FOR IT WAS WRONG.
    #
    # On the first validation set it looked decisive -- a cliff between 0.25 and
    # 0.20 worth +0.087 F1, which read as a real structural threshold. On a
    # disjoint set of seeds the cliff does not exist:
    #
    #            set A (14k frames)      set B (29k frames, disjoint seeds)
    #     0.30   P .9088 R .3982 F1 .5537    P .8227 R .4058 F1 .5435   <- shipped
    #     0.25   P .9078 R .4149 F1 .5695    P .8199 R .4195 F1 .5551
    #     0.20   P .9190 R .4913 F1 .6403    P .8182 R .4245 F1 .5589
    #     0.15   P .9177 R .4921 F1 .6407    P .8160 R .4257 F1 .5595
    #     0.10   P .9154 R .4921 F1 .6401    P .8127 R .4261 F1 .5591
    #
    # So the honest size of this effect is +0.015 F1, not +0.087, and the "plateau
    # not a peak" argument was itself fitted to set A. It is kept because it is
    # positive on both sets and costs 0.005 precision, not because the cliff is
    # real. 0.20 over 0.15 is the conservative end; they are indistinguishable.
    net_min_short: float = 0.20    # metres across the SHORT axis
    # The sensor reaches 20 m and this threw away everything past 17, which is a
    # quarter of all the frames in which a pad is genuinely visible. Measured on
    # village with delete-and-diff truth, same model, same frames:
    #
    #     15-20 m detection   21% -> 48%      overall  66% -> 72%
    #
    # It does not recover the whole band — at 18 m a 1.2 m pad is ~8 px — and
    # that residue is a training problem, not a threshold one. But keeping the
    # cap below the sensor range was costing recall for nothing.
    # 19.5 was still the PLANAR ceiling used as a EUCLIDEAN cap. DEPTH_MAX_M = 20 m
    # is measured along the optical axis; euclidean range is planar / ray_x, and at
    # the corner of a 90 deg frame ray_x = 0.577 (54.7 deg off axis), so a pad is
    # genuinely visible out to 20/0.577 = 34.6 m. Capping at 19.5 discards the
    # entire outer band of the frustum.
    #
    # This is the same planar-vs-euclidean confusion this file's own docstring
    # warns about ("getting depth_max wrong scales every range by 1.5") -- here
    # applied to the range cap rather than the depth scale. Their note that the
    # 15-20 m residue is "a training problem, not a threshold one" is what the
    # measurement contradicts: it is still a threshold problem.
    #
    # 28.3 m = 20 / cos(45 deg): the euclidean visibility limit at the EDGE MIDPOINT
    # of the frame. This is the change that carries the gain, and unlike the
    # short-axis one it reproduces on disjoint seeds -- same size, and saturating
    # at the same place, which is what a geometric bound should do:
    #
    #            set A (14k frames)      set B (29k frames, disjoint seeds)
    #     19.5   P .9088 R .3982 F1 .5537    P .8229 R .4056 F1 .5434   <- shipped
    #     22.0   P .8928 R .4232 F1 .5742    P .8152 R .4344 F1 .5668
    #     25.0   P .8887 R .4313 F1 .5807    P .8137 R .4513 F1 .5806
    #     28.3   P .8887 R .4324 F1 .5818    P .8138 R .4535 F1 .5824  <- saturated
    #     34.6   P .8887 R .4326 F1 .5819    P .8138 R .4536 F1 .5825
    #
    # +0.028 / +0.039 F1. Recall saturates at the predicted bound on BOTH sets and
    # the corner limit of 34.6 buys nothing, so this takes the geometric value the
    # data confirms rather than the largest one available. It costs ~0.01
    # precision, which the end-to-end test has to justify: their own notes attribute
    # 24 of 43 drone deaths to descending onto a misidentified pad.
    net_max_range: float = 28.3
    max_range: float = 28.3
    cell: float = 0.8             # ground grid, metres
    ground_win: int = 7           # neighbourhood for the ground minimum, in cells
    stride: int = 1


@dataclass
class PadProposal:
    centre: np.ndarray            # (3,) world centroid; z is the LANDING SURFACE
    height: float                 # metres above local ground
    extent: float                 # longest horizontal span, metres
    flatness: float               # std of height within the blob, metres
    n_px: int
    rng: float                    # median range, metres
    score: float                  # higher = more pad-like


def _geom_cfg(cfg: PadConfig) -> GeomConfig:
    """The shared back-projection settings, carrying the autopilot depth ceiling."""
    return GeomConfig(min_height=cfg.min_height, max_height=cfg.max_height,
                      min_pixels=cfg.min_pixels, max_pixels=cfg.max_pixels,
                      max_range=cfg.max_range, cell=cfg.cell,
                      ground_win=cfg.ground_win, stride=cfg.stride,
                      depth_max=AP_DEPTH_MAX_M)


def propose_from_mask(pos, R, depth_img, mask, cfg: Optional[PadConfig] = None,
                      fov_deg: float = AP_FOV_DEG,
                      res: int = AP_RES) -> List[PadProposal]:
    """Turn a per-pixel pad mask into world-space proposals.

    Shares the back-projection with `propose` — same `_cloud`, same planar-depth
    correction — so a learned mask and a geometric one land in the same world
    coordinates and are directly comparable. Only the segmentation differs.
    """
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
        c = np.median(xy, axis=0)              # median: robust to a ragged edge
        spread = float(np.median(np.linalg.norm(xy - c, axis=1))) * 2.0
        d_pt = np.linalg.norm(p - np.asarray(pos, float), axis=1)
        rng = float(np.median(d_pt))
        if rng > cfg.net_max_range or spread > cfg.net_max_extent:
            continue
        # Depth spread within the blob: a pad's pixels are all at one range, a
        # pole's are not. This is what rejects the stalks.
        if float(np.percentile(d_pt, 90) - np.percentile(d_pt, 10)) \
                > cfg.net_max_depth_spread:
            continue
        # A pad is horizontal. Its pixels' world Z should agree; a sloped or
        # vertical surface will not, and costs nothing extra to check here.
        if float(np.std(p[:, 2])) > cfg.flatness_max * 2.0:
            continue
        # A pad is ROUND and at least a pad wide. A branch is neither.
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
    """The trained segmenter, batched over the fleet.

    One forward pass for every drone rather than N passes: measured 168 ms on CPU
    single-threaded for 8 drones against a 500 ms per-act budget, where looping
    would not fit.
    """

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
        """(B,H,W) depth in [0,1] -> (B,H,W) bool pad mask."""
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
    """Flat, pad-sized discs standing on the local ground, as world positions.

    `pos` is the drone's world position (state[0:3]) and `R` its body->world
    rotation (from state[3:6]); both are observation, so this is deployable.
    """
    from scipy import ndimage

    cfg = cfg or PadConfig()
    gcfg = _geom_cfg(cfg)
    pts, h, valid = heights(pos, R, depth_img, gcfg, fov_deg, res)
    if h is None:
        return []

    # Segment on "above ground" only and apply the ceiling to the FINISHED blob:
    # putting max_height in the mask slices a wall off at 3.2 m and hands back a
    # flat-looking lid, which is exactly a pad's signature. Whole objects get
    # rejected; halves of objects get accepted. (The same trap cost the victim
    # detector a 6 m lamp post that measured 2.09 m tall.)
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
        # Land ON the surface, so the centre's z is the pad top, not the blob
        # centroid — half a pad thickness low is half a pad thickness into it.
        centre = np.array([c[0], c[1], float(np.nanmax(p[:, 2]))])
        # Prefer flat, pad-sized, well-resolved things. Flatness dominates because
        # it is the one property clutter does not share.
        score = (1.0 - min(1.0, flat / cfg.flatness_max)) \
            * (1.0 - min(1.0, abs(extent - 2.0 * PAD_RADIUS) / cfg.max_extent)) \
            * min(1.0, npx / 40.0)
        out.append(PadProposal(centre=centre, height=top, extent=extent,
                               flatness=flat, n_px=npx, rng=rng, score=float(score)))
    out.sort(key=lambda q_: -q_.score)
    return out
