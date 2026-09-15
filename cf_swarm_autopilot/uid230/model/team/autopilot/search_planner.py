"""Fleet search planner: optimise the drones' waypoint routes against the exact generator
posterior with a fast surrogate of the episode, at t=0 of every seed.

Surrogate (vectorised over candidate plans x posterior layouts, dt = 0.5 s):
  * a searching drone flies its polyline at cruise speed after a short climb;
  * a pad is discovered when it lies within `det_range` of a flying drone and within
    +-`sector` of the drone's heading (the camera is forward-facing; the agent's yaw scan
    widens the swept sector);
  * a discovered pad is claimed by the nearest free drone, which diverts to it at cruise
    speed and lands `land_delay` after arrival (the pool is shared, a landed drone stops);
  * the seed score is the mean per-drone 0.45*success + 0.45*time_factor, with the
    generator's target time (1.06*d/3 + (n-1) s from the pad nearest each start).
Cross-entropy optimisation over the waypoints, warm-started from a given plan (the
champion's learned route), so with a faithful surrogate the result can only improve.
Everything used here is observable: starts, the clue, the map kind.
"""
from __future__ import annotations

import itertools
import math
from typing import Optional

import numpy as np

from .geo_posterior import MAP_PARAMS, _sample_goals_one, clue_log_likelihood

SPEED = 3.0
DT = 0.5
T_MAX = 58.0
DEFAULTS = dict(det_range=22.0, sector_deg=95.0, climb_delay=0.5, land_delay=3.0, speed=SPEED)


def sample_layouts(starts_xy: np.ndarray, clue_xy: np.ndarray, kind: str, L: int,
                   M: int = 3000, seed: int = 0) -> np.ndarray:
    """(L, n, 2) pad layouts drawn from the exact generator posterior (importance
    resampling of independent per-drone goal samples under the clue likelihood)."""
    p = MAP_PARAMS[kind]
    rng = np.random.default_rng(seed)
    n = starts_xy.shape[0]
    g = np.stack([_sample_goals_one(float(starts_xy[i, 0]), float(starts_xy[i, 1]),
                                    p["world"], p["r_min"], p["r_max"], M, rng) for i in range(n)])  # (n, M, 2)
    cen = g.mean(axis=0)                                            # index-aligned joint draws
    w = np.exp(clue_log_likelihood(cen, clue_xy))
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    if w.sum() <= 0:
        w = np.ones(M)
    idx = rng.choice(M, size=L, replace=True, p=w / w.sum())
    return np.transpose(g[:, idx, :], (1, 0, 2)).astype(np.float32)   # (L, n, 2)


def sample_layouts_conditioned(starts_xy: np.ndarray, clue_xy: np.ndarray, kind: str, found_xy: np.ndarray,
                               L: int, M: int = 3000, seed: int = 0, match_sigma: float = 4.0,
                               max_assign: int = 24):
    """Posterior layouts of the pads NOT yet found, given the found pads' positions.
    Enumerates which start owns each found pad (radius-band likelihood, like geo_posterior),
    then draws the remaining starts' pads from their annuli under the clue likelihood with the
    found pads fixed. Returns (layouts (L, n-F, 2), owner_free (L, n-F) start indices)."""
    p = MAP_PARAMS[kind]
    rng = np.random.default_rng(seed)
    S = np.asarray(starts_xy, np.float32)[:, :2]
    n = S.shape[0]
    found = np.asarray(found_xy, np.float32).reshape(-1, 2)
    F = found.shape[0]
    if F == 0:
        return sample_layouts(S, clue_xy, kind, L, M=M, seed=seed), np.tile(np.arange(n), (L, 1))
    g = np.stack([_sample_goals_one(float(S[i, 0]), float(S[i, 1]), p["world"], p["r_min"], p["r_max"], M, rng)
                  for i in range(n)])                                          # (n, M, 2)
    s2 = 2.0 * match_sigma ** 2
    Lk = np.empty((n, F))
    for f in range(F):
        d2 = ((g - found[f][None, None, :]) ** 2).sum(-1)                      # (n, M)
        Lk[:, f] = np.exp(-d2 / s2).mean(axis=1) + 1e-6
    kk = min(F, n)
    perms = np.array(list(itertools.permutations(range(n), kk)), dtype=np.int64) if n <= 8 else None
    if perms is None or len(perms) == 0:
        return None, None
    logw = np.log(Lk[perms, np.arange(kk)[None, :]]).sum(axis=1)
    order = np.argsort(-logw)[:max_assign]
    wa = np.exp(logw[order] - logw[order[0]])
    wa /= wa.sum()
    found_sum = found[:kk].sum(axis=0)
    n_free = n - kk
    if n_free <= 0:
        return np.zeros((L, 0, 2), np.float32), np.zeros((L, 0), np.int64)
    # draw layouts: choose an assignment by wa, then a joint sample m by the clue weight
    counts = rng.multinomial(L, wa)
    out = np.empty((L, n_free, 2), np.float32)
    own = np.empty((L, n_free), np.int64)
    pos = 0
    for a, cnt in zip(order, counts):
        if cnt == 0:
            continue
        used = set(perms[a].tolist())
        U = [i for i in range(n) if i not in used]
        cen = (found_sum[None, :] + g[U].sum(axis=0)) / n                      # (M, 2)
        w = np.exp(clue_log_likelihood(cen, np.asarray(clue_xy, np.float32)[:2]))
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        if w.sum() <= 0:
            w = np.ones(M)
        idx = rng.choice(M, size=int(cnt), replace=True, p=w / w.sum())
        out[pos:pos + cnt] = np.transpose(g[U][:, idx, :], (1, 0, 2))
        own[pos:pos + cnt] = np.asarray(U)[None, :]
        pos += cnt
    return out, own


def target_times(starts_xy: np.ndarray, layouts: np.ndarray) -> np.ndarray:
    """(L, n) generator target time per drone: nearest pad (XY) to its start."""
    d = np.linalg.norm(layouts[:, None, :, :] - starts_xy[None, :, None, :], axis=-1)  # (L, n_drone, n_pad)
    n = starts_xy.shape[0]
    return 1.06 * d.min(axis=2) / SPEED + 1.0 * (n - 1)


def simulate(plans: np.ndarray, starts_xy: np.ndarray, layouts: np.ndarray, tt: np.ndarray,
             det_range: float, sector_deg: float, climb_delay: float, land_delay: float,
             world: float, speed: float = SPEED, t0: float = 0.0,
             pad_found0: Optional[np.ndarray] = None) -> np.ndarray:
    """Expected seed score of every plan. plans (C, n_d, K, 2); layouts (L, n_p, 2).
    t0: sim time at which the rollout starts (drones already at starts_xy, clocks absolute);
    pad_found0: (n_p,) bool, pads already detected (claimable at once) but not yet claimed."""
    C, n, K, _ = plans.shape
    L, n_p = layouts.shape[0], layouts.shape[1]
    B = C * L
    routes = np.repeat(plans, L, axis=0)                              # (B, n, K, 2)
    pads = np.tile(layouts, (C, 1, 1))                                # (B, n_p, 2)
    ttime = np.tile(tt, (C, 1))                                       # (B, n)
    pos = np.repeat(starts_xy[None, :, :], B, axis=0).astype(np.float32)
    wp_idx = np.zeros((B, n), dtype=np.int64)
    status = np.zeros((B, n), dtype=np.int8)        # 0 searching, 1 approaching, 2 landed
    target = np.full((B, n), -1, dtype=np.int64)
    pad_state = np.zeros((B, n_p), dtype=np.int8)   # 0 unfound, 1 found, 2 claimed, 3 landed
    survive = np.ones((B, n_p), dtype=np.float32)   # probability the pad is still undetected
    if pad_found0 is not None:
        f0 = np.asarray(pad_found0, bool)
        pad_state[:, f0] = 1
        survive[:, f0] = 0.0
    land_t = np.full((B, n), np.inf, dtype=np.float32)
    arrive_t = np.full((B, n), np.inf, dtype=np.float32)
    first = routes[:, :, 0, :] - pos
    heading = (first / (np.linalg.norm(first, axis=-1, keepdims=True) + 1e-6)).astype(np.float32)
    cos_sector = math.cos(math.radians(sector_deg))
    bi = np.arange(B)[:, None]
    t = float(t0)
    while t < T_MAX:
        t += DT
        moving = t > climb_delay
        # --- route following ---------------------------------------------------------
        cur_wp = np.take_along_axis(routes, np.clip(wp_idx, 0, K - 1)[:, :, None, None].repeat(2, axis=3), axis=2)[:, :, 0, :]
        to_wp = cur_wp - pos
        d_wp = np.linalg.norm(to_wp, axis=-1) + 1e-6
        step = speed * DT if moving else 0.0
        exhausted = wp_idx >= K          # past the last waypoint: continue straight, bounce off the walls
        reach = (d_wp <= step) & (status == 0) & ~exhausted
        wp_idx = np.where(reach, wp_idx + 1, wp_idx)
        dirn = np.where(exhausted[..., None], heading, to_wp / d_wp[..., None])
        mvlen = np.where(exhausted, step, np.minimum(step, d_wp))
        mv = mvlen[..., None] * dirn
        srch = status == 0
        newpos = pos + mv
        # reflect the heading at the world box for exhausted drones
        out_x = np.abs(newpos[..., 0]) > world
        out_y = np.abs(newpos[..., 1]) > world
        dirn = dirn.copy()
        dirn[..., 0] = np.where(out_x & exhausted, -dirn[..., 0], dirn[..., 0])
        dirn[..., 1] = np.where(out_y & exhausted, -dirn[..., 1], dirn[..., 1])
        newpos = np.clip(newpos, -world, world)
        pos = np.where(srch[..., None], newpos, pos)
        heading = np.where(srch[..., None], dirn, heading)
        # --- approaching drones ----------------------------------------------------
        appr = status == 1
        if appr.any():
            tgt_xy = pads[bi, np.clip(target, 0, n_p - 1)]
            to_t = tgt_xy - pos
            d_t = np.linalg.norm(to_t, axis=-1) + 1e-6
            mv2 = np.minimum(step, d_t)[..., None] * to_t / d_t[..., None]
            pos = np.where(appr[..., None], pos + mv2, pos)
            heading = np.where(appr[..., None], to_t / d_t[..., None], heading)
            arrived = appr & (d_t <= step + 0.5) & ~np.isfinite(arrive_t)
            arrive_t = np.where(arrived, t, arrive_t)
            done = appr & (t >= arrive_t + land_delay)
            if done.any():
                status = np.where(done, 2, status)
                land_t = np.where(done, t, land_t)
                pad_state[bi, np.clip(target, 0, n_p - 1)] = np.where(
                    done, 3, pad_state[bi, np.clip(target, 0, n_p - 1)])
        # --- detection ---------------------------------------------------------------
        flying = status != 2
        rel = pads[:, None, :, :] - pos[:, :, None, :]                     # (B, drone, pad, 2)
        dist = np.linalg.norm(rel, axis=-1) + 1e-6
        cosang = (rel * heading[:, :, None, :]).sum(-1) / dist
        # per-step detection probability: certain inside det_range*0.55, fading to ~0.15 at
        # det_range; halved beyond the camera's own +-45 deg (only the yaw scan reaches there)
        # measured on the champion (open, set B): pads are first detected at 20-25 m (p50 20.7,
        # p90 24.8) -> near-certain out to det_range-3, fading to 0.3 at det_range
        p_r = np.clip(0.9 - 0.6 * (dist - (det_range - 3.0)) / 3.0, 0.3, 0.9)
        p_r = np.where(dist <= det_range, p_r, 0.0)
        p_r = np.where(cosang >= 0.7071, p_r, p_r * 0.5)
        p_r = np.where(cosang >= cos_sector, p_r, 0.0)
        p_r = np.where(flying[:, :, None] & moving, p_r, 0.0)
        survive = survive * np.prod(1.0 - p_r * 0.8, axis=1)               # 0.8: exposure per 0.5 s
        newly = (survive < 0.5) & (pad_state == 0)                          # median detection time
        pad_state = np.where(newly, 1, pad_state)
        # --- claiming: nearest free drone per found pad, one pad per drone -----------
        free = status == 0
        avail = pad_state == 1
        if (free.any() and avail.any()):
            dmat = np.linalg.norm(pads[:, None, :, :] - pos[:, :, None, :], axis=-1)   # (B, drone, pad)
            for _ in range(min(n, n_p)):
                dm = np.where(free[:, :, None] & avail[:, None, :], dmat, np.inf)
                flat = dm.reshape(B, -1)
                k = flat.argmin(axis=1)
                ok = np.isfinite(flat[np.arange(B), k])
                if not ok.any():
                    break
                di, pj = k // n_p, k % n_p
                rows = np.flatnonzero(ok)
                status[rows, di[rows]] = 1
                target[rows, di[rows]] = pj[rows]
                free[rows, di[rows]] = False
                avail[rows, pj[rows]] = False
                pad_state[rows, pj[rows]] = 2
    # --- score --------------------------------------------------------------------------
    succ = np.isfinite(land_t)
    lt = np.where(succ, land_t, 60.0)
    tf = np.where(lt <= ttime, 1.0, np.clip(1.0 - (lt - ttime) / np.maximum(60.0 - ttime, 1e-6), 0.0, 1.0))
    per = np.where(succ, 0.45 + 0.45 * tf + 0.10 * 0.5, 0.01)   # safety term ~ half credit on average
    return per.mean(axis=1).reshape(C, L).mean(axis=1)


def _kmeans(X: np.ndarray, k: int, rng, iters: int = 30) -> np.ndarray:
    C = X[rng.choice(len(X), k, replace=False)].copy()
    lab = np.zeros(len(X), np.int64)
    for _ in range(iters):
        lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
    return lab


def _arc_plan(starts_xy: np.ndarray, lay_sub: np.ndarray, world: float, K: int = 5, spread: float = 1.0) -> np.ndarray:
    """Every drone sweeps the arc (around its own start) covered by its pad samples in lay_sub (M, n, 2)."""
    n = starts_xy.shape[0]
    plan = np.zeros((n, K, 2), np.float32)
    for i in range(n):
        v = lay_sub[:, i, :] - starts_xy[i][None]
        r = np.linalg.norm(v, axis=1)
        th = np.arctan2(v[:, 1], v[:, 0])
        m = math.atan2(float(np.sin(th).mean()), float(np.cos(th).mean()))
        d = np.angle(np.exp(1j * (th - m)))
        sd = max(float(d.std()), 0.15)
        rm = float(np.median(r))
        for k, a in enumerate(m + np.linspace(-spread * sd, spread * sd, K)):
            plan[i, k] = starts_xy[i] + rm * np.array([math.cos(a), math.sin(a)], np.float32)
    return np.clip(plan, -world, world)


def arc_candidates(starts_xy: np.ndarray, layouts: np.ndarray, world: float, rng, K: int = 5, ks=(1, 2, 3)) -> list:
    """Global candidate sweep plans from the posterior's modes: cluster the joint layouts into k
    hypotheses; 'top' sends every drone along its arc under the heaviest hypothesis, 'split' gives
    drone i the i-th heaviest hypothesis so the fleet tests several at once."""
    n = starts_xy.shape[0]
    X = layouts.reshape(len(layouts), -1)
    out = []
    for k in ks:
        if k > 1 and len(X) < 4 * k:
            continue
        lab = _kmeans(X, k, rng) if k > 1 else np.zeros(len(X), np.int64)
        order = np.argsort([-(lab == j).sum() for j in range(k)])
        out.append(_arc_plan(starts_xy, layouts[lab == order[0]], world, K))
        if k >= 2 and n >= 2:
            plan = np.zeros((n, K, 2), np.float32)
            for i in range(n):
                plan[i] = _arc_plan(starts_xy, layouts[lab == order[i % k]], world, K)[i]
            out.append(plan)
    return out


class SearchPlanner:
    """Incremental CEM over per-drone waypoints against the surrogate.

        pl.start(starts, clue, kind, init_plan)   # sample layouts, seed the search
        pl.step(budget_sec)                        # run iterations until the budget is spent
        pl.best()                                  # (n, K, 2) best plan so far
    """

    def __init__(self, K: int = 5, layouts: int = 24, pop: int = 32, iters: int = 10, elite: int = 6,
                 sigma0: float = 6.0, seed: int = 0, fixed_prefix: int = 2, arc_n: int = 0, **params):
        self.K, self.L, self.pop, self.iters, self.elite, self.sigma0 = K, layouts, pop, iters, elite, sigma0
        self.fixed_prefix = int(fixed_prefix)
        self.arc_n = int(arc_n)
        self.seed = seed
        self.params = dict(DEFAULTS, **params)
        self.reset()

    def reset(self):
        self._cond_args = None
        self._cond_phase = 0
        self._mean = None
        self._sigma = None
        self._best = None
        self._best_score = -np.inf
        self._init_score = None
        self._it = 0
        self._lay = None
        self._tt = None
        self._starts = None
        self._kw = None
        self._rng = None
        self._cand = None

    @property
    def done(self) -> bool:
        return self._mean is None or self._it >= self.iters

    def start(self, starts_xy, clue_xy, kind: str, init_plan: Optional[np.ndarray] = None):
        starts_xy = np.asarray(starts_xy, np.float32)[:, :2]
        clue_xy = np.asarray(clue_xy, np.float32)[:2]
        n = starts_xy.shape[0]
        p = MAP_PARAMS[kind]
        W = float(p["world"])
        self._lay = sample_layouts(starts_xy, clue_xy, kind, self.L, seed=self.seed)
        self._tt = target_times(starts_xy, self._lay)
        self._lay2 = sample_layouts(starts_xy, clue_xy, kind, self.L, seed=self.seed + 7919)   # held-out layouts for an unbiased gain
        self._tt2 = target_times(starts_xy, self._lay2)
        self._init_plan = None
        self._hold = None
        self._rng = np.random.default_rng(self.seed + 1)
        self._starts = starts_xy
        self._kw = dict(det_range=self.params["det_range"], sector_deg=self.params["sector_deg"],
                        climb_delay=self.params["climb_delay"], land_delay=self.params["land_delay"], world=W,
                        speed=float(self.params.get("speed", SPEED)))
        if init_plan is None:
            mean = np.zeros((n, self.K, 2), np.float32)
            for i in range(n):
                for k in range(self.K):
                    mean[i, k] = starts_xy[i] + (k + 1) / self.K * (clue_xy - starts_xy[i]) * 1.5
        else:
            init = np.asarray(init_plan, np.float32)[:, :, :2]
            if init.shape[1] >= self.K:
                mean = init[:, : self.K, :].copy()
            else:
                mean = np.concatenate([init, np.repeat(init[:, -1:, :], self.K - init.shape[1], axis=1)], axis=1)
        self._mean = np.clip(mean, -W, W)
        self._init_plan = self._mean.copy()
        self._best = self._mean.copy()
        self._best_score = float(simulate(self._best[None], starts_xy, self._lay, self._tt, **self._kw)[0])
        self._init_score = self._best_score
        if self.arc_n >= n and init_plan is not None:
            # global candidates from the posterior modes; warm-start from the best of them
            cands = arc_candidates(starts_xy, self._lay, W, self._rng, K=self.K)
            if cands:
                sc = simulate(np.stack(cands).astype(np.float32), starts_xy, self._lay, self._tt, **self._kw)
                j = int(np.argmax(sc))
                if float(sc[j]) > self._best_score:
                    self._best_score = float(sc[j])
                    self._best = np.clip(np.asarray(cands[j], np.float32), -W, W)
                    self._mean = self._best.copy()
        self._sigma = np.full_like(self._mean, self.sigma0)
        self._sigma[:, : self.fixed_prefix, :] = 0.0      # keep the early legs of the warm start
        self._it = 0

    def start_conditioned_light(self, cur_xy, orig_free_xy, orig_starts_xy, clue_xy, kind: str, found_xy,
                                init_plan: np.ndarray, t0: float, M: int = 1500):
        """Same as start_conditioned but split into three cheap phases so no single act pays for
        both samplings and the first evaluation: call start_conditioned_phase() once per act
        until it returns True (ready) or None (cannot re-plan)."""
        self._cond_args = dict(cur_xy=np.asarray(cur_xy, np.float32)[:, :2], orig_free_xy=np.asarray(orig_free_xy, np.float32)[:, :2],
                               orig_starts_xy=np.asarray(orig_starts_xy, np.float32)[:, :2], clue_xy=np.asarray(clue_xy, np.float32)[:2],
                               kind=kind, found=np.asarray(found_xy, np.float32).reshape(-1, 2),
                               init_plan=np.asarray(init_plan, np.float32)[:, :, :2], t0=float(t0), M=int(M))
        self._cond_phase = 0
        self._lay = None
        return self.start_conditioned_phase()

    def start_conditioned_phase(self):
        a = self._cond_args
        if a is None:
            return None
        if self._cond_phase == 0:
            lay, _ = sample_layouts_conditioned(a["orig_starts_xy"], a["clue_xy"], a["kind"], a["found"], self.L, M=a["M"], seed=self.seed)
            if lay is None or lay.shape[1] == 0:
                self._cond_args = None
                return None
            self._lay = lay
            self._cond_phase = 1
            return False
        if self._cond_phase == 1:
            lay2, _ = sample_layouts_conditioned(a["orig_starts_xy"], a["clue_xy"], a["kind"], a["found"], self.L, M=a["M"], seed=self.seed + 7919)
            if lay2 is None or lay2.shape[1] == 0:
                self._cond_args = None
                return None
            self._lay2 = lay2
            self._cond_phase = 2
            return False
        # phase 2: target times, warm start, first evaluation
        p = MAP_PARAMS[a["kind"]]
        W = float(p["world"])
        found = a["found"]
        ofs = a["orig_free_xy"]
        n_all = int(a["orig_starts_xy"].shape[0])
        def _tt(layouts):
            allp = np.concatenate([layouts, np.repeat(found[None], layouts.shape[0], axis=0)], axis=1) if found.shape[0] else layouts
            d = np.linalg.norm(allp[:, None, :, :] - ofs[None, :, None, :], axis=-1)
            return (1.06 * d.min(axis=2) / SPEED + 1.0 * (n_all - 1)).astype(np.float32)
        self._tt = _tt(self._lay)
        self._tt2 = _tt(self._lay2)
        self._rng = np.random.default_rng(self.seed + 1)
        self._starts = a["cur_xy"]
        self._kw = dict(det_range=self.params["det_range"], sector_deg=self.params["sector_deg"],
                        climb_delay=0.0, land_delay=self.params["land_delay"], world=W,
                        speed=float(self.params.get("speed", SPEED)), t0=a["t0"])
        init = a["init_plan"]
        if init.shape[1] >= self.K:
            mean = init[:, : self.K, :].copy()
        else:
            mean = np.concatenate([init, np.repeat(init[:, -1:, :], self.K - init.shape[1], axis=1)], axis=1)
        self._mean = np.clip(mean, -W, W)
        self._init_plan = self._mean.copy()
        self._best = self._mean.copy()
        self._best_score = float(simulate(self._best[None], self._starts, self._lay, self._tt, **self._kw)[0])
        self._init_score = self._best_score
        self._hold = None
        self._sigma = np.full_like(self._mean, self.sigma0)
        self._it = 0
        self._cand = None
        self._cond_phase = 3
        self._cond_args = None
        return True

    def start_conditioned(self, cur_xy, orig_free_xy, orig_starts_xy, clue_xy, kind: str, found_xy,
                          init_plan: np.ndarray, t0: float):
        """Re-plan for the free drones from their CURRENT positions, against the posterior of
        the pads not yet found (found pads fixed). orig_free_xy: (n_free, 2) the free drones'
        start pads (their generator target time uses the nearest pad to the START).
        init_plan: (n_free, K, 2) their remaining waypoints."""
        cur_xy = np.asarray(cur_xy, np.float32)[:, :2]
        n_free = cur_xy.shape[0]
        p = MAP_PARAMS[kind]
        W = float(p["world"])
        found = np.asarray(found_xy, np.float32).reshape(-1, 2)
        lay, own = sample_layouts_conditioned(orig_starts_xy, clue_xy, kind, found, self.L, seed=self.seed)
        lay2, _ = sample_layouts_conditioned(orig_starts_xy, clue_xy, kind, found, self.L, seed=self.seed + 7919)
        if lay is None or lay2 is None or lay.shape[1] == 0:
            return False
        self._lay, self._lay2 = lay, lay2
        ofs = np.asarray(orig_free_xy, np.float32)[:, :2]
        n_all = int(np.asarray(orig_starts_xy).shape[0])
        def _tt(layouts):
            allp = np.concatenate([layouts, np.repeat(found[None], layouts.shape[0], axis=0)], axis=1) if found.shape[0] else layouts
            d = np.linalg.norm(allp[:, None, :, :] - ofs[None, :, None, :], axis=-1)      # (L, n_free, n_pads)
            return (1.06 * d.min(axis=2) / SPEED + 1.0 * (n_all - 1)).astype(np.float32)
        self._tt = _tt(lay)
        self._tt2 = _tt(lay2)
        self._rng = np.random.default_rng(self.seed + 1)
        self._starts = cur_xy
        self._kw = dict(det_range=self.params["det_range"], sector_deg=self.params["sector_deg"],
                        climb_delay=0.0, land_delay=self.params["land_delay"], world=W,
                        speed=float(self.params.get("speed", SPEED)), t0=float(t0))
        init = np.asarray(init_plan, np.float32)[:, :, :2]
        if init.shape[1] >= self.K:
            mean = init[:, : self.K, :].copy()
        else:
            mean = np.concatenate([init, np.repeat(init[:, -1:, :], self.K - init.shape[1], axis=1)], axis=1)
        self._mean = np.clip(mean, -W, W)
        self._init_plan = self._mean.copy()
        self._best = self._mean.copy()
        self._best_score = float(simulate(self._best[None], cur_xy, self._lay, self._tt, **self._kw)[0])
        self._init_score = self._best_score
        self._hold = None
        self._sigma = np.full_like(self._mean, self.sigma0)
        self._it = 0
        self._cand = None
        return True

    def _new_population(self):
        n, K = self._mean.shape[0], self.K
        W = self._kw["world"]
        cand = self._mean[None] + self._sigma[None] * self._rng.standard_normal((self.pop - 1, n, K, 2)).astype(np.float32)
        self._cand = np.clip(np.concatenate([self._mean[None], cand], axis=0), -W, W)
        self._cand_sc = np.full(self.pop, np.nan, dtype=np.float32)
        self._cand_pos = 0

    def step(self, budget_sec: float = 0.4, chunk: int = 4) -> int:
        """Evaluate candidates in chunks until the time budget is spent (every act pays for at
        most a few surrogate rollouts); a CEM generation completes once all are scored."""
        import time
        t0 = time.perf_counter()
        done_iters = 0
        if getattr(self, "_cand", None) is None and not self.done:
            self._new_population()
        while not self.done:
            lo = self._cand_pos
            hi = min(lo + chunk, self.pop)
            self._cand_sc[lo:hi] = simulate(self._cand[lo:hi], self._starts, self._lay, self._tt, **self._kw)
            self._cand_pos = hi
            if hi >= self.pop:
                sc = self._cand_sc
                order = np.argsort(-sc)
                if sc[order[0]] > self._best_score:
                    self._best_score, self._best = float(sc[order[0]]), self._cand[order[0]].copy()
                el = self._cand[order[: self.elite]]
                self._mean = el.mean(axis=0)
                self._sigma = np.maximum(el.std(axis=0), 2.0) * 0.9
                self._sigma[:, : self.fixed_prefix, :] = 0.0
                self._it += 1
                done_iters += 1
                self._cand = None
                if not self.done:
                    self._new_population()
            if time.perf_counter() - t0 >= budget_sec:
                break
        return done_iters

    def best(self) -> np.ndarray:
        return None if self._best is None else self._best.copy()

    @property
    def scores(self):
        return (self._init_score, self._best_score)

    @property
    def holdout_scores(self):
        """(init, best) expected scores on layouts the optimiser never saw; cached per best plan."""
        if self._best is None or self._init_plan is None:
            return (None, None)
        key = (self._it, float(self._best_score))
        if self._hold is None or self._hold[0] != key:
            sc = simulate(np.stack([self._init_plan, self._best]).astype(np.float32), self._starts, self._lay2, self._tt2, **self._kw)
            self._hold = (key, (float(sc[0]), float(sc[1])))
        return self._hold[1]
