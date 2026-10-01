"""Fleet obstacle memory for the round-5 city collision filter (candHc OM2; every key default off).

Why (research/round5/CITY.md): in city about 40 % of the crashes hit a building while the camera pointed more than
45 deg away from the travel direction, and about half while sinking faster than 0.6 m/s toward a surface below the
camera's view. free_direction sees only the current frame; the OM point memory keeps each drone's own points for
5 s within 8 m and bumps only in a thin slab at the drone's height.

ObstacleGrid is one world-frame 2.5-D grid for the whole fleet (0.5 m cells, +-100 m), fed from the pooled depth
rays of every live drone:
  top[c]   highest surface point any camera saw in the cell,
  bot[c]   lowest surface point seen in the cell (interval mode, forest canopies),
  free[c]  lowest height at which a depth ray crossed the cell without a hit (the surface there is below it).
Nothing decays (the world is static). Points within mask_r of a moving teammate are dropped before a frame is
folded in, so drones never leave ghost obstacles. The camera eye sits CAMERA_EYE_FWD_M ahead of and
CAMERA_EYE_UP_M above the body centre (swarm/constants.py); the rays use team/autopilot/freedir's aligned
min-pooled blocks (a block's nearest surface on its centre ray).
"""
from __future__ import annotations

import math

import numpy as np

from team.autopilot import freedir as FD

EYE_FWD_M = 0.13
EYE_UP_M = 0.05
_NEG = np.float32(-1e9)
_POS = np.float32(1e9)


def _group_minmax(k, z):
    """Unique keys of k with the max and min of z per key (no duplicate-index writes, which numpy leaves unordered)."""
    o = np.lexsort((z, k))
    ks, zs = k[o], z[o]
    cut = np.flatnonzero(ks[1:] != ks[:-1])
    last = np.append(cut, len(ks) - 1)
    first = np.insert(cut + 1, 0, 0)
    return ks[last], zs[last], zs[first]


def stop_speed(room, a, lat):
    """Largest speed that still stops within `room` metres with deceleration a after latency lat (array ok)."""
    room = np.maximum(room, 0.0)
    return a * (-lat + np.sqrt(lat * lat + 2.0 * room / max(a, 1e-3)))


class ObstacleGrid:
    def __init__(self, half: float = 100.0, cell: float = 0.5):
        self.cell = float(cell)
        self.half = float(half)
        self.size = int(round(2.0 * self.half / self.cell))
        self.top = np.full((self.size, self.size), _NEG, np.float32)
        self.bot = np.full((self.size, self.size), _POS, np.float32)
        self.free = np.full((self.size, self.size), _POS, np.float32)
        self.frames = 0

    def _ij(self, x, y):
        i = np.floor((x + self.half) / self.cell).astype(np.int64)
        j = np.floor((y + self.half) / self.cell).astype(np.int64)
        ok = (i >= 0) & (i < self.size) & (j >= 0) & (j < self.size)
        return i, j, ok

    def window(self, x: float, y: float, r: float):
        """Index box (i0, i1, j0, j1) of the cells within r m (square) of (x, y), clipped to the grid."""
        i0 = max(int(math.floor((x - r + self.half) / self.cell)), 0)
        i1 = min(int(math.floor((x + r + self.half) / self.cell)) + 1, self.size)
        j0 = max(int(math.floor((y - r + self.half) / self.cell)), 0)
        j1 = min(int(math.floor((y + r + self.half) / self.cell)) + 1, self.size)
        return i0, i1, j0, j1

    def add_frame(self, pos, R, depth, depth_max: float, fov_deg: float, hit_grid: int = 32,
                  carve_grid: int = 16, carve_step: float = 0.5, carve_stop: float = 0.35, carve_max: float = 12.0,
                  carve_el=(-25.0, 10.0), mask=None, mask_r: float = 0.6, zmin=None):
        """Fold one depth frame of a drone at pos with body rotation R (world <- body). Returns (hits, carved)."""
        d = np.asarray(depth, np.float32)
        if d.ndim == 3:
            d = d[..., 0]
        R = np.asarray(R, float)
        eye = np.asarray(pos, float) + R[:, 0] * EYE_FWD_M + R[:, 2] * EYE_UP_M
        n_hit = n_carve = 0
        if hit_grid > 0:
            dirs, pooled = FD._grid(d, int(hit_grid), fov_deg, True)
            planar = pooled * (depth_max - 0.5) + 0.5
            hit = planar < depth_max - 0.3
            if np.any(hit):
                eu = planar[hit] / np.maximum(dirs[hit, 0], 1e-6)
                P = eye[None, :] + (dirs[hit] * eu[:, None]) @ R.T
                keep = np.ones(len(P), bool)
                if mask is not None and len(mask):
                    M = np.asarray(mask, float).reshape(-1, 3)
                    dm = np.linalg.norm(P[:, None, :] - M[None, :, :], axis=2).min(axis=1)
                    keep &= dm > mask_r
                if zmin is not None:
                    keep &= P[:, 2] >= zmin
                if np.any(keep):
                    P = P[keep]
                    i, j, ok = self._ij(P[:, 0], P[:, 1])
                    if np.any(ok):
                        k = i[ok] * self.size + j[ok]
                        z = P[ok, 2].astype(np.float32)
                        ku, zhi, zlo = _group_minmax(k, z)
                        ft, fb = self.top.reshape(-1), self.bot.reshape(-1)
                        ft[ku] = np.maximum(ft[ku], zhi)
                        fb[ku] = np.minimum(fb[ku], zlo)
                        n_hit = int(len(z))
        if carve_grid > 0:
            dirs2, pooled2 = FD._grid(d, int(carve_grid), fov_deg, True)
            W0 = dirs2 @ R.T
            el = np.degrees(np.arcsin(np.clip(W0[:, 2], -1.0, 1.0)))
            rows = (el >= float(carve_el[0])) & (el <= float(carve_el[1]))     # rays near the horizon only
            dirs2, pooled2 = dirs2[rows], pooled2[rows]
            planar2 = pooled2 * (depth_max - 0.5) + 0.5
            eu2 = np.minimum(planar2 / np.maximum(dirs2[:, 0], 1e-6) - carve_stop, carve_max)
            L = float(eu2.max()) if len(eu2) else 0.0
            if L > carve_step:
                K = int(L / carve_step)
                s = (np.arange(K, dtype=np.float64) + 1.0) * carve_step
                W = dirs2 @ R.T
                ok = s[None, :] <= eu2[:, None]
                ri, si = np.nonzero(ok)
                if len(ri):
                    pts = eye[None, :] + W[ri] * s[si][:, None]
                    i, j, okg = self._ij(pts[:, 0], pts[:, 1])
                    if np.any(okg):
                        k = i[okg] * self.size + j[okg]
                        z = pts[okg, 2].astype(np.float32)
                        ku, _zmax, zmin = _group_minmax(k, z)
                        ff = self.free.reshape(-1)
                        ff[ku] = np.minimum(ff[ku], zmin)
                        n_carve = int(len(z))
        self.frames += 1
        return n_hit, n_carve
