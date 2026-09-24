"""Exact Monte-Carlo posterior over goal-pad positions, from the task generator's structure.

What a swarm_autopilot seed exposes at t=0 is the fleet's start pads (each drone's position)
and one shared clue = centroid(goal pads) + U(-R, R)^2 with R ~ U(5, 30).  The generator
(swarm/validator/task_gen.py) draws each goal from ITS OWN start: a uniform angle and a
radius uniform in the map's distance band, capped so the goal stays inside the world box.
That is a fully known generative model, so it is sampled directly rather than learned:

  * per drone i: M independent goal samples g_i from goal_from_start(s_i)
  * nothing found:  joint draw k -> centroid_k -> weight p(clue | centroid_k)
  * pads found:     each found pad is the goal of exactly one drone.  Enumerate the injective
                    assignments found->drone, weight each by the per-drone match likelihood,
                    and for the drones left over re-weight their samples by the clue
                    likelihood given the found pads' contribution to the centroid.

The result is the expected number of NOT-YET-FOUND pads per grid cell.  Nothing here uses
the seed, pad coordinates, or anything outside the observation.
"""
from __future__ import annotations

import itertools
import math
from typing import Optional

import os
import numpy as np

# swarm.constants: TYPE_x_WORLD_RANGE, R_MIN/R_MAX, VILLAGE_R_*.  Mountain's box is
# 250*gs*0.6 with gs in [0.6, 0.8] -> 90..120; unknown per seed, so take the middle.
MAP_PARAMS = {
    "city":     dict(world=75.0,  r_min=22.0, r_max=45.0),
    "open":     dict(world=60.0,  r_min=28.0, r_max=72.0),
    "mountain": dict(world=105.0, r_min=65.0, r_max=100.0),
    "village":  dict(world=40.0,  r_min=28.0, r_max=56.0),
    "forest":   dict(world=42.0,  r_min=22.0, r_max=45.0),
}
CLUE_R_MIN, CLUE_R_MAX = 5.0, 30.0
PAD_SPACING = 4.0

# Distance bands the seed templates actually use (swarm challenge_families templates):
# screening slots (epoch indices 0-299) draw every goal from one band per map, benchmark
# slots (300-999) from one of three equal sub-bands of the full range, and every drone in
# a seed shares that band. Weights are each band's share of the epoch's seeds for the map.
BAND_MIX = {
    "mountain": ((65.0, 95.0, 0.292), (65.0, 76.7, 0.250), (76.7, 88.3, 0.250), (88.3, 100.0, 0.208)),
    "open":     ((28.0, 65.0, 0.292), (28.0, 42.7, 0.250), (42.7, 57.3, 0.250), (57.3, 72.0, 0.208)),
}
# City and forest keep the single MAP_PARAMS band: the mixture did not help there (300-seed pod
# test, 2026-09-22). Village goals are drawn at 28-56 m, but placement then enforces the mountain
# 65-100 m ring inside a +-40 m box (platform_placement.py:370), so real start->pad distances land
# at ~55-77 m (p10-p90, median 67 m). SWARM_DEV_VILLAGE_BAND="lo,hi" switches the village band in
# both pad models; unset keeps the champion (route prior 65-100 m, this module 28-56 m).
BAND_MIX["village"] = ((57.0, 79.0, 1.0),)   # measured start->pad distances (160-seed pod test: +0.012)
_VB = os.environ.get("SWARM_DEV_VILLAGE_BAND", "").strip()
if _VB:
    _lo, _hi = (float(x) for x in _VB.split(","))
    BAND_MIX["village"] = ((_lo, _hi, 1.0),)


def band_mix(kind: str):
    """(r_lo, r_hi, weight) per band hypothesis for this map; weights sum to 1."""
    mix = BAND_MIX.get(kind)
    if not mix:
        p = MAP_PARAMS.get(kind, MAP_PARAMS["open"])
        return ((p["r_min"], p["r_max"], 1.0),)
    tot = sum(w for _, _, w in mix)
    return tuple((lo, hi, w / tot) for lo, hi, w in mix)


def sample_goals_banded(S: np.ndarray, kind: str, M: int, rng: np.random.Generator):
    """Index-aligned joint goal samples with one shared band per joint draw.

    Returns g (n, M, 2), band index bk (M,), and the band mixture. Joint draw k puts every
    drone's goal in band bk[k], matching the generator, where all goals of a seed share the
    slot's distance band."""
    p = MAP_PARAMS.get(kind, MAP_PARAMS["open"])
    mix = band_mix(kind)
    S = np.asarray(S, np.float64)[:, :2]
    n = S.shape[0]
    bk = rng.choice(len(mix), size=M, p=np.array([w for _, _, w in mix]))
    g = np.empty((n, M, 2), np.float32)
    for b, (lo, hi, _w) in enumerate(mix):
        idx = np.flatnonzero(bk == b)
        if idx.size == 0:
            continue
        for i in range(n):
            g[i, idx] = _sample_goals_one(float(S[i, 0]), float(S[i, 1]), p["world"], lo, hi, idx.size, rng)
    return g, bk, mix


def band_match(g: np.ndarray, bk: np.ndarray, n_bands: int, found_xy: np.ndarray, sigma: float) -> np.ndarray:
    """Lb[b, i, f]: kernel likelihood that found pad f is drone i's goal under band b."""
    n, M, _ = g.shape
    k = found_xy.shape[0]
    s2 = 2.0 * sigma ** 2
    Lb = np.full((n_bands, n, k), 1e-6)
    for f in range(k):
        e = np.exp(-((g - found_xy[f][None, None, :]) ** 2).sum(-1) / s2)   # (n, M)
        for b in range(n_bands):
            sel = bk == b
            if sel.any():
                Lb[b, :, f] = e[:, sel].mean(axis=1) + 1e-6
    return Lb


def assignment_weights(Lb: np.ndarray, band_w: np.ndarray, perms: np.ndarray):
    """log weight of each found->drone assignment, marginalised over the shared band."""
    kk = perms.shape[1]
    # per band: sum of log match over the assignment; then log-sum-exp over bands with priors
    lb = np.log(Lb[:, perms, np.arange(kk)[None, :]]).sum(axis=2)          # (B, P)
    lb = lb + np.log(np.maximum(band_w, 1e-12))[:, None]
    m = lb.max(axis=0)
    return m + np.log(np.exp(lb - m[None, :]).sum(axis=0))                   # (P,)


def _sample_goals_one(sx: float, sy: float, world: float, r_min: float, r_max: float,
                      M: int, rng: np.random.Generator) -> np.ndarray:
    """(M, 2) goal samples for one start, replicating task_gen._goal_from_start."""
    buf, need, tries = [], M, 0
    while need > 0 and tries < 12:
        tries += 1
        m = int(need * 1.6) + 64
        ang = rng.uniform(0.0, 2.0 * math.pi, m)
        ca, sa = np.cos(ang), np.sin(ang)
        with np.errstate(divide="ignore", invalid="ignore"):
            mrx = np.where(np.abs(ca) > 1e-8, np.where(ca > 0, (world - sx) / ca, (-world - sx) / ca), np.inf)
            mry = np.where(np.abs(sa) > 1e-8, np.where(sa > 0, (world - sy) / sa, (-world - sy) / sa), np.inf)
        max_r = np.minimum(np.minimum(mrx, mry), r_max)
        ok = max_r >= r_min
        if not ok.any():
            continue
        hi = np.minimum(max_r[ok] * 0.999, r_max)
        rad = rng.uniform(r_min, np.maximum(hi, r_min + 1e-3))
        buf.append(np.stack([sx + rad * ca[ok], sy + rad * sa[ok]], axis=1))
        need -= int(ok.sum())
    g = np.concatenate(buf, axis=0)[:M] if buf else np.zeros((0, 2), np.float32)
    if g.shape[0] < M:      # deep-corner start: the generator's clamp fallback
        k = M - g.shape[0]
        ang = rng.uniform(0.0, 2.0 * math.pi, k)
        rad = rng.uniform(r_min, r_max, k)
        g = np.concatenate([g, np.stack([np.clip(sx + rad * np.cos(ang), -world, world),
                                         np.clip(sy + rad * np.sin(ang), -world, world)], axis=1)], axis=0)
    return g.astype(np.float32)


def clue_log_likelihood(centroid_xy: np.ndarray, clue_xy: np.ndarray) -> np.ndarray:
    """log p(clue | centroid): per-axis uniform box noise of half-width R, R ~ U(5, 30)."""
    m = np.max(np.abs(centroid_xy - clue_xy[None, :]), axis=1)
    lo = np.maximum(m, CLUE_R_MIN)
    dens = np.where(m <= CLUE_R_MAX, np.maximum(1.0 / lo - 1.0 / CLUE_R_MAX, 0.0), 0.0)
    with np.errstate(divide="ignore"):
        return np.log(dens)


class GeoPosterior:
    """Drop-in for the champion's ActorVisibleGoalPosterior: reset(), observe_reset(obs),
    score(starts, found, clue, xy, map_kind) -> log expected (unfound) pads per query cell."""

    def __init__(self, K: int = 4000, cell: float = 4.0, seed: int = 0,
                 match_sigma: float = 4.0, max_assign: int = 48):
        self.M = int(K)
        self.cell = float(cell)
        self.seed = int(seed)
        self.match_sigma = float(match_sigma)
        self.max_assign = int(max_assign)
        self.reset()

    def reset(self):
        self._starts = None
        self._clue = None
        self._g = None            # (n, M, 2) per-drone samples
        self._kind = None
        self._found_key = None
        self._grid = None
        self._bound = None

    def observe_reset(self, observation) -> None:
        if self._starts is not None:
            return
        st = np.asarray(observation["state"], np.float32).reshape(-1, 190)
        self._starts = st[:, 0:3].astype(np.float64).copy()
        self._clue = (st[0, 0:3] + st[0, 138:141]).astype(np.float64)

    # -- sampling ------------------------------------------------------------------------
    def _ensure(self, map_kind: str):
        kind = map_kind if map_kind in MAP_PARAMS else "open"
        if self._g is not None and self._kind == kind:
            return
        if self._starts is None:
            raise RuntimeError("reset observation not supplied")
        p = MAP_PARAMS[kind]
        rng = np.random.default_rng(self.seed)
        S = self._starts[:, :2]
        n = S.shape[0]
        g, bk, mix = sample_goals_banded(S, kind, self.M, rng)
        for i in range(n):
            gi = g[i]
            # a goal pad never lands within PAD_SPACING of any start pad: resample those
            # (within the same band, so the joint draw keeps one band for every drone)
            d2 = ((gi[:, None, :] - S[None, :, :].astype(np.float32)) ** 2).sum(-1).min(axis=1)
            bad = np.flatnonzero(d2 < PAD_SPACING ** 2)
            for j in bad:
                same = np.flatnonzero(bk == bk[j])
                gi[j] = gi[same[rng.integers(0, same.size)]]
            g[i] = gi
        self._g = g
        self._bk = bk
        self._band_w = np.array([w for _, _, w in mix])
        self._kind = kind
        self._bound = float(p["world"]) + 2 * self.cell
        self._found_key = None
        self._grid = None

    # -- posterior grid --------------------------------------------------------------------
    def _hist(self, pts: np.ndarray, w: np.ndarray, grid: np.ndarray):
        b, c = self._bound, self.cell
        nb = grid.shape[0]
        ix = np.clip(((pts[:, 0] + b) / c).astype(int), 0, nb - 1)
        iy = np.clip(((pts[:, 1] + b) / c).astype(int), 0, nb - 1)
        np.add.at(grid, (iy, ix), w)

    def _compute_grid(self, found_xy: np.ndarray) -> np.ndarray:
        g = self._g
        n, M, _ = g.shape
        nb = int(math.ceil(2 * self._bound / self.cell))
        grid = np.zeros((nb, nb), np.float64)
        clue = self._clue[:2]
        k = int(found_xy.shape[0])
        if k == 0:
            cen = g.mean(axis=0)                                   # (M, 2) index-aligned joint draws
            w = np.exp(clue_log_likelihood(cen, clue) - 0.0)
            w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
            if w.sum() <= 0:
                w = np.ones(M)
            w = w / w.sum()
            for i in range(n):
                self._hist(g[i], w, grid)
            return grid
        # per-band, per-drone match likelihood Lb[b, i, f]; all found pads share one band
        bk, band_w = self._bk, self._band_w
        kk = min(k, n)
        Lb = band_match(g, bk, len(band_w), found_xy[:kk], self.match_sigma)
        perms = np.array(list(itertools.permutations(range(n), kk)), dtype=np.int64)   # (P, kk)
        logw = assignment_weights(Lb, band_w, perms)
        order = np.argsort(-logw)[: self.max_assign]
        wa = np.exp(logw[order] - logw[order[0]])
        wa /= wa.sum()
        found_sum = found_xy[:kk].sum(axis=0)
        for a, wA in zip(order, wa):
            used = set(perms[a].tolist())
            U = [i for i in range(n) if i not in used]
            if not U:
                continue
            cen = (found_sum[None, :] + g[U].sum(axis=0)) / n       # (M, 2)
            # band posterior for this assignment, applied to each joint draw through its band
            band_lik = np.prod(Lb[:, perms[a], np.arange(kk)], axis=1)          # (B,)
            w = np.exp(clue_log_likelihood(cen, clue)) * band_lik[bk]
            w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
            if w.sum() <= 0:
                w = np.ones(M)
            w = w / w.sum() * wA
            for i in U:
                self._hist(g[i], w, grid)
        return grid

    def _apply_found(self, found_xy: np.ndarray):
        key = tuple(np.round(found_xy.reshape(-1), 1).tolist())
        if key == self._found_key and self._grid is not None:
            return
        self._found_key = key
        self._grid = self._compute_grid(np.asarray(found_xy, np.float32).reshape(-1, 2))

    def expected_count_grid(self, bound: Optional[float] = None, cell: Optional[float] = None):
        """(xs, ys, grid[y, x]) expected number of unfound pads per cell."""
        if self._grid is None:
            self._apply_found(np.zeros((0, 2), np.float32))
        nb = self._grid.shape[0]
        xs = -self._bound + (np.arange(nb) + 0.5) * self.cell
        return xs, xs.copy(), self._grid

    def score(self, starts, found, clue, xy, map_kind=None):
        """Champion interface: log expected unfound-pad count in the 8x8 m block around xy."""
        del starts, clue
        self._ensure(str(map_kind or "open"))
        f = (np.asarray(found, np.float32).reshape(-1, 3)[:, :2] if found is not None and len(found)
             else np.zeros((0, 2), np.float32))
        self._apply_found(f)
        grid = self._grid
        nb = grid.shape[0]
        q = np.asarray(xy, np.float32).reshape(-1, 2)
        b, c = self._bound, self.cell
        ix = np.clip(((q[:, 0] + b) / c).astype(int), 0, nb - 1)
        iy = np.clip(((q[:, 1] + b) / c).astype(int), 0, nb - 1)
        # 2x2-cell block sum (an 8 m planner cell) so the planner's coarser grid sees whole mass
        val = np.zeros(len(q))
        for dy in (0, 1):
            for dx in (0, 1):
                val += grid[np.clip(iy + dy - 1, 0, nb - 1), np.clip(ix + dx - 1, 0, nb - 1)]
        return np.log(val + 1e-4).astype(np.float32)
