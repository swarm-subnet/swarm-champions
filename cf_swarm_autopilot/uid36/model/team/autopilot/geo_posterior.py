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
        g = np.empty((n, self.M, 2), np.float32)
        for i in range(n):
            gi = _sample_goals_one(float(S[i, 0]), float(S[i, 1]), p["world"], p["r_min"], p["r_max"], self.M, rng)
            # a goal pad never lands within PAD_SPACING of any start pad: resample those
            d2 = ((gi[:, None, :] - S[None, :, :].astype(np.float32)) ** 2).sum(-1).min(axis=1)
            bad = np.flatnonzero(d2 < PAD_SPACING ** 2)
            if bad.size:
                gi[bad] = gi[rng.integers(0, self.M, bad.size)]
            g[i] = gi
        self._g = g
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
        # per-drone match likelihood L[i, f]
        s2 = 2.0 * self.match_sigma ** 2
        L = np.empty((n, k))
        for f in range(k):
            d2 = ((g - found_xy[f][None, None, :]) ** 2).sum(-1)   # (n, M)
            L[:, f] = np.exp(-d2 / s2).mean(axis=1) + 1e-5
        kk = min(k, n)
        perms = np.array(list(itertools.permutations(range(n), kk)), dtype=np.int64)   # (P, kk)
        logw = np.log(L[perms, np.arange(kk)[None, :]]).sum(axis=1)
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
            w = np.exp(clue_log_likelihood(cen, clue))
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
