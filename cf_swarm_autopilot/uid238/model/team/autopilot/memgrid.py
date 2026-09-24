"""A short local memory of where the obstacles were.

``free_direction`` decides from the current depth frame alone, so a drone whose
camera is yawed away from its flight path (a search scan swings 50 deg off track)
has no evidence at all about what it is flying into -- and the shipped code
responds by leaving the velocity untouched. This keeps a coarse world-frame
height map built from the same pooled depth rays, so the direction of travel can
be checked against what was seen a second ago, and a heading behind the camera
can be evaluated at all.

The map stores one number per cell: the height of the tallest thing seen there.
Static obstacles are what matters, so nothing decays; a cell only counts as
blocking when something in it stands above the drone's own altitude.
"""
from __future__ import annotations

import numpy as np

CELL_M = 2.0
HALF_EXTENT_M = 140.0
_NEG = -1e9


class HeightMemory:
    """World-frame max-height grid per drone, fed from pooled depth rays."""

    def __init__(self, n: int, cell: float = CELL_M, half: float = HALF_EXTENT_M):
        self.cell = float(cell)
        self.half = float(half)
        self.size = int(2 * self.half / self.cell) + 1
        self.grid = np.full((n, self.size, self.size), _NEG, dtype=np.float32)

    def reset(self) -> None:
        self.grid.fill(_NEG)

    def _idx(self, xy):
        return np.clip(((np.asarray(xy, float) + self.half) / self.cell).astype(int),
                       0, self.size - 1)

    def update(self, i: int, pos, rot, depth, depth_max: float, fov_deg: float,
               grid_n: int = 16, max_range: float = 22.0) -> None:
        """Fold one frame's pooled rays into drone ``i``'s map."""
        d = np.asarray(depth, dtype=np.float32)
        if d.ndim == 3:
            d = d[..., 0]
        h, w = d.shape
        kh, kw = h // grid_n, w // grid_n
        if kh < 1 or kw < 1:
            return
        pooled = d[:kh * grid_n, :kw * grid_n].reshape(
            grid_n, kh, grid_n, kw).min(axis=(1, 3))
        half_t = np.tan(np.radians(fov_deg) * 0.5)
        idx = (2.0 * (np.arange(grid_n) + 0.5) / grid_n - 1.0) * half_t
        # body-frame ray directions, matching freedir's convention
        bx = np.ones((grid_n, grid_n), dtype=np.float32)
        by = -np.broadcast_to(idx[None, :], (grid_n, grid_n)).astype(np.float32)
        bz = np.broadcast_to(-idx[:, None], (grid_n, grid_n)).astype(np.float32)
        dirs = np.stack([bx, by, bz], axis=-1).reshape(-1, 3)
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
        planar = pooled.reshape(-1) * (depth_max - 0.5) + 0.5
        rng = planar / np.maximum(dirs[:, 0], 1e-6)
        keep = (rng > 0.6) & (rng <= max_range)
        if not np.any(keep):
            return
        pts = np.asarray(pos, float)[None, :] + (
            (np.asarray(rot, float) @ dirs[keep].T).T * rng[keep][:, None])
        ix, iy = self._idx(pts[:, 0]), self._idx(pts[:, 1])
        np.maximum.at(self.grid[i], (ix, iy), pts[:, 2].astype(np.float32))

    def blocked_range(self, i: int, pos, direction, lookahead: float,
                      clear_below: float = 0.5) -> float:
        """Distance to the first remembered obstacle along ``direction``.

        Returns ``lookahead`` when the way is clear. Only things standing above the
        drone's own altitude count -- terrain it is flying over is not in the way.
        """
        d = np.asarray(direction, float)[:2]
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            return lookahead
        d = d / n
        p = np.asarray(pos, float)
        steps = np.arange(self.cell, lookahead + self.cell, self.cell)
        xs = p[0] + d[0] * steps
        ys = p[1] + d[1] * steps
        hs = self.grid[i][self._idx(xs), self._idx(ys)]
        hit = np.flatnonzero(hs > (p[2] - clear_below))
        return float(steps[hit[0]]) if hit.size else float(lookahead)
