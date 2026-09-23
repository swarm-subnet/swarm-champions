# Derived from Swarm champion cf_swarm_sar/uid253 (benchmark score 0.9482, crowned 2026-09-19),
# itself derived from uid173 / uid184 / uid137 / uid66. Source:
# https://github.com/swarm-subnet/swarm-champions/tree/main/cf_swarm_sar/uid253
#
# MIT License -- Copyright (c) 2026 Swarm
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction. The above copyright notice and this permission
# notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
#
# Build ort26: obstacle-memory barrier (forest/city), Bayesian colour veto, posterior-gated
# take-off sweep (village/city/open), mountain far-pad prior, depth heads fine-tuned on synthetic
# frames (prone/buried/seated poses), team election hygiene (elect only drones on the candidate,
# confirmer follow, parallel self-confirm, supersede guards), lane tour advance-on-arrival and
# close-hit verify on mountain, image-depth range clamp, per-episode state reset, dwell-kept tour
# advance, mountain depth head fine-tuned on 41k rare-pose frames.
# See agents/swarm_sar/README.md in the working tree.
# swarm_sar v29e resubmission 2026-09-02 (re-roll marker: comment only, code identical to the benchmarked build)
from __future__ import annotations

class _FlatModule:

    def __init__(self, _flat_name, /, **attrs):
        self.__name__ = _flat_name
        self.__dict__.update(attrs)

    def __repr__(self):
        return "<flattened module %r>" % (self.__name__,)

import numpy as np

BOX_B = np.array([25.0, 30.0, 30.0, 25.0, 12.0, 20.0], dtype=np.float64)

TYPE_VALID = np.array([True, True, True, True, False, True])
CLUE_R_FULL = 80.0
CLUE_R_NREF = 8.0
PAD0_R = 80.0
RELAX_BOX = 1.5

MARGIN = 3.0
P_MIN = 0.02
B_FLOOR = 25.0
FLOOR_FRAC = 0.02
MIN_AREA = 200.0

PAD_TRIG = 0.15

PAD_SLACK = 30.0
GRID_HALF = 48.0
GRID_MAX_CELLS = 4096

TAKEOFF_SEC = 3.0
TRANSIT_V = 2.6
SCAN_V = float(__import__('os').environ.get('NEWDET_SCAN_V', '2.4'))   # NEWDET: lane-geometry override
USABLE_SEC = 52.0
CAP_LO, CAP_HI = 4.0, 45.0
WP_SEP = float(__import__('os').environ.get('NEWDET_WP_SEP', '14.0'))   # NEWDET: lane-geometry override
WP_MAX = int(__import__('os').environ.get('NEWDET_WP_MAX', '6'))   # NEWDET: lane-geometry override

ADVANCE_R = 8.0
PASS_OFFSET_M = float(__import__('os').environ.get('NEWDET_PASS_OFFSET', '0'))   # NEWDET: lateral shift of repeated lane passes

HOLD_SEC = float(__import__('os').environ.get('NEWDET_HOLD_SEC', '2.0'))   # NEWDET: lane-geometry override
CENTRE_EPS = 3.0
SCAN_R_REF = float(__import__('os').environ.get('NEWDET_SCAN_R_REF', '26.0'))   # NEWDET: lane-geometry override
SCAN_R_MIN = 0.45
import os

# --- MOUNTAIN FAR-PAD PRIOR (MFP, 2026-09-21): see tools/patch_mfp.py.
MFP_ON = os.environ.get('SWSAR_MFP', '1') == '1'
MFP_F = float(os.environ.get('SWSAR_MFP_F', '0.057'))            # valid fraction: near/far split, P(far) = exp(-f N(80))
MFP_F_FAR = float(os.environ.get('SWSAR_MFP_FFAR', '0.037'))  # valid fraction for the far shape (clustered terrain -> smaller)
MFP_AIN_MAX = float(os.environ.get('SWSAR_MFP_AIN', '0.2'))  # only when this little of the box is within 80 m of pad0
MFP_SLACK = float(os.environ.get('SWSAR_MFP_SLACK', '50.0'))  # pad-disc slack beyond the nearest box point
MFP_P_MTN = float(os.environ.get('SWSAR_MFP_PMTN', '0.5'))
MFP_N_IN, MFP_B_IN, MFP_N_OUT, MFP_B_OUT = 100.0, 30.0, 50.0, 45.0

_TWO_PI = 2.0 * np.pi

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))

def _as_xy(xy) -> tuple[np.ndarray, bool]:
    a = np.asarray(xy, dtype=np.float64)
    if a.ndim == 1:
        return a.reshape(1, 2), True
    return a.reshape(-1, 2), False

class Posterior:

    def __init__(self, grid_m: float = 2.0):
        self.grid_m = float(grid_m)

        self.clue_xy: np.ndarray | None = None
        self.pad0_xy: np.ndarray | None = None
        self.n_drones = 0
        self.r_clue = CLUE_R_FULL
        self.r_pad = PAD0_R
        self.pad_overlap = 1.0

        p = TYPE_VALID.astype(np.float64)
        self.p_env = p / p.sum()

        self.version = 0
        self.rung = 0
        self.rung_name = "uninitialised"
        self._b_eff: float | None = float(BOX_B[TYPE_VALID].max())
        self._b_relax: float | None = RELAX_BOX * float(BOX_B[TYPE_VALID].max())
        self._use_clue = False
        self._use_pad0 = False
        self._mfp = False
        self.mfp_p_far = 0.0
        self._build_grid(GRID_HALF, self.grid_m)
        self._recompute()

    def _build_grid(self, half: float, cell: float) -> None:
        half = float(half)
        cell = float(cell)
        n_side = int(np.floor(half / cell)) * 2 + 1
        if n_side * n_side > GRID_MAX_CELLS:
            cell = half * 2.0 / np.sqrt(GRID_MAX_CELLS)
            n_side = int(np.floor(half / cell)) * 2 + 1
        ax = (np.arange(n_side, dtype=np.float64) - (n_side - 1) * 0.5) * cell
        gx, gy = np.meshgrid(ax, ax, indexing="xy")
        self._grid_half = half
        self._grid_cell = cell
        self._cell_area = cell * cell
        self._gx = gx.ravel()
        self._gy = gy.ravel()

    def set_geometry(self, clue_world_xy, pad0_xy, n_drones) -> None:
        c = np.asarray(clue_world_xy, dtype=np.float64).reshape(2).copy()
        p0 = np.asarray(pad0_xy, dtype=np.float64).reshape(2).copy()
        if not np.all(np.isfinite(c)):
            c = np.zeros(2)
        if not np.all(np.isfinite(p0)):
            p0 = np.zeros(2)
        n = int(max(1, int(n_drones)))
        self.clue_xy = c
        self.pad0_xy = p0
        self.n_drones = n
        self.r_clue = CLUE_R_FULL * np.sqrt(min(n, int(CLUE_R_NREF)) / CLUE_R_NREF)
        self._recompute()

    def set_type_belief(self, p_env) -> None:
        p = np.asarray(p_env, dtype=np.float64).reshape(-1)
        if p.size < 6:
            p = np.pad(p, (0, 6 - p.size))
        p = np.nan_to_num(p[:6], nan=0.0, posinf=0.0, neginf=0.0)
        p = np.clip(p, 0.0, None)
        p[~TYPE_VALID] = 0.0
        s = p.sum()
        if s <= 1e-12:
            p = TYPE_VALID.astype(np.float64)
            s = p.sum()
        self.p_env = p / s
        self._recompute()

    def _box_half(self) -> float:
        live = (self.p_env >= P_MIN) & TYPE_VALID
        if not live.any():
            live = TYPE_VALID
        return float(max(BOX_B[live].max(), B_FLOOR))

    def _hard_mask(self, b_eff, use_clue, use_pad0, gx, gy) -> np.ndarray:
        ok = np.ones(gx.shape, dtype=bool)
        if b_eff is not None:
            lim = b_eff + MARGIN
            ok &= (np.abs(gx) <= lim) & (np.abs(gy) <= lim)
        if use_clue and self.clue_xy is not None:
            r = self.r_clue + MARGIN
            ok &= ((gx - self.clue_xy[0]) ** 2 + (gy - self.clue_xy[1]) ** 2) <= r * r
        if use_pad0 and self.pad0_xy is not None:
            r = self.r_pad + MARGIN
            ok &= ((gx - self.pad0_xy[0]) ** 2 + (gy - self.pad0_xy[1]) ** 2) <= r * r
        return ok

    def _pad_radius(self, b_nominal: float) -> float:
        if self.pad0_xy is None:
            return PAD0_R
        lim = float(b_nominal)
        dx = max(abs(float(self.pad0_xy[0])) - lim, 0.0)
        dy = max(abs(float(self.pad0_xy[1])) - lim, 0.0)
        slack = MFP_SLACK if getattr(self, '_mfp', False) else PAD_SLACK
        return float(max(PAD0_R, np.hypot(dx, dy) + slack))

    def _mfp_active(self) -> bool:
        if not MFP_ON or self.pad0_xy is None:
            return False
        try:
            return float(self.p_env[2]) >= MFP_P_MTN and float(self.pad_overlap) < MFP_AIN_MAX
        except Exception:
            return False

    def _mfp_weights(self, sx, sy, use_clue, w_default):
        """Mixture weights over the support cells (sx, sy): near (uniform inside the 80 m pad disc)
        and far (closest valid candidate beyond it), see tools/patch_mfp.py."""
        p0 = self.pad0_xy
        # expected candidate count within distance d of pad0, on a 1 m lattice of the 45 m box
        ax = np.arange(-MFP_B_OUT + 0.5, MFP_B_OUT, 1.0)
        lx, ly = np.meshgrid(ax, ax, indexing="xy")
        lx, ly = lx.ravel(), ly.ravel()
        dens = MFP_N_OUT / (2.0 * MFP_B_OUT) ** 2 + np.where(
            (np.abs(lx) <= MFP_B_IN) & (np.abs(ly) <= MFP_B_IN), MFP_N_IN / (2.0 * MFP_B_IN) ** 2, 0.0)
        ld = np.hypot(lx - p0[0], ly - p0[1])
        srt = np.argsort(ld, kind="stable")
        ld_s = ld[srt]
        n_cum = np.cumsum(dens[srt])
        def N_of(d):
            j = np.searchsorted(ld_s, d, side="right")
            return np.where(j > 0, n_cum[np.clip(j - 1, 0, n_cum.size - 1)], 0.0)
        p_far = float(np.clip(np.exp(-MFP_F * float(N_of(np.array([PAD0_R]))[0])), 0.0, 0.95))
        d = np.hypot(sx - p0[0], sy - p0[1])
        inside = _sigmoid(3.0 * (MFP_B_IN - np.abs(sx)) / MARGIN) * _sigmoid(3.0 * (MFP_B_IN - np.abs(sy)) / MARGIN)
        rho = MFP_N_OUT / (2.0 * MFP_B_OUT) ** 2 + inside * MFP_N_IN / (2.0 * MFP_B_IN) ** 2
        clue = np.ones(sx.shape)
        if use_clue and self.clue_xy is not None:
            dc = np.hypot(sx - self.clue_xy[0], sy - self.clue_xy[1])
            clue = _sigmoid(3.0 * (self.r_clue - dc) / MARGIN)
        near = _sigmoid(3.0 * (PAD0_R - d) / MARGIN)
        w_near = rho * clue * near
        w_far = rho * clue * (1.0 - near) * np.exp(-MFP_F_FAR * np.maximum(N_of(d) - N_of(np.array([PAD0_R]))[0], 0.0))
        sn, sf = float(w_near.sum()), float(w_far.sum())
        if sn <= 1e-12 and sf <= 1e-12:
            return w_default
        if sn <= 1e-12:
            return w_far / sf
        if sf <= 1e-12:
            return w_near / sn
        self.mfp_p_far = p_far
        return (1.0 - p_far) * w_near / sn + p_far * w_far / sf

    def _pad_overlap_frac(self, b: float) -> float:
        if self.pad0_xy is None:
            return 1.0
        inbox = (np.abs(self._gx) <= b) & (np.abs(self._gy) <= b)
        if not inbox.any():
            return 1.0
        d2 = ((self._gx[inbox] - self.pad0_xy[0]) ** 2
              + (self._gy[inbox] - self.pad0_xy[1]) ** 2)
        return float((d2 <= PAD0_R * PAD0_R).mean())

    def _density(self, gx, gy) -> np.ndarray:
        d = np.zeros(gx.shape, dtype=np.float64)
        for t in range(6):
            pt = self.p_env[t]
            if pt <= 0.0:
                continue
            b = BOX_B[t]
            sx = _sigmoid(3.0 * (b - np.abs(gx)) / MARGIN)
            sy = _sigmoid(3.0 * (b - np.abs(gy)) / MARGIN)
            d += pt * (2.0 * b) ** -2 * sx * sy
        peak = d.max()
        if peak <= 0.0:
            return np.ones(gx.shape, dtype=np.float64)
        return d / peak + FLOOR_FRAC

    def _soft_factors(self, gx, gy, use_clue, use_pad0) -> np.ndarray:
        f = np.ones(gx.shape, dtype=np.float64)
        if use_clue and self.clue_xy is not None:
            d = np.hypot(gx - self.clue_xy[0], gy - self.clue_xy[1])
            f *= _sigmoid(3.0 * (self.r_clue - d) / MARGIN)
        if use_pad0 and self.pad0_xy is not None:
            d = np.hypot(gx - self.pad0_xy[0], gy - self.pad0_xy[1])
            f *= _sigmoid(3.0 * (self.r_pad - d) / MARGIN)
        return f

    def _rung_ladder(self, b0: float, b_relax: float):
        return (
            (b0, True, True, "box+clue+pad0"),
            (b_relax, True, True, "box1.5+clue+pad0"),
            (b_relax, True, False, "box1.5+clue"),
            (None, True, True, "clue+pad0"),
            (None, True, False, "clue"),
            (b_relax, False, False, "box1.5"),
            (None, False, False, "grid"),
        )

    def _recompute(self) -> None:
        if self._grid_half != GRID_HALF or self._grid_cell != self.grid_m:
            self._build_grid(GRID_HALF, self.grid_m)
        b = self._box_half()

        self.r_pad = PAD0_R
        self.pad_overlap = self._pad_overlap_frac(b)
        self._mfp = self._mfp_active()
        self.mfp_p_far = 0.0
        b0 = RELAX_BOX * b if self.pad_overlap < P1_PAD_TRIG else b
        ladder = self._rung_ladder(b0, RELAX_BOX * b)
        chosen = None
        for k, (b_eff, uc, up, name) in enumerate(ladder):
            self.r_pad = self._pad_radius(b) if up else PAD0_R
            if b_eff is None and (uc or up):

                self._regrid_for_disks(uc, up)
            elif self._grid_half != GRID_HALF or self._grid_cell != self.grid_m:
                self._build_grid(GRID_HALF, self.grid_m)
            mask = self._hard_mask(b_eff, uc, up, self._gx, self._gy)

            need = MIN_AREA if up else self._cell_area
            if mask.sum() * self._cell_area >= need or k == len(ladder) - 1:
                chosen = (k, b_eff, uc, up, name, mask)
                break
        assert chosen is not None
        k, b_eff, uc, up, name, mask = chosen
        if not mask.any():
            mask = np.ones_like(mask)
            name, b_eff, uc, up = "grid", None, False, False
        self.rung = int(k)
        self.rung_name = name
        self._b_eff = None if b_eff is None else float(b_eff)

        self._b_relax = (None if b_eff is None
                         else float(max(RELAX_BOX * b, float(b_eff))))
        self._use_clue = bool(uc)
        self._use_pad0 = bool(up)

        idx = np.flatnonzero(mask)
        sx = self._gx[idx]
        sy = self._gy[idx]
        w = self._density(sx, sy) * self._soft_factors(sx, sy, uc, up)
        if self._mfp and up:
            w = self._mfp_weights(sx, sy, uc, w)
        tot = w.sum()
        w = w / tot if tot > 0 else np.full(idx.size, 1.0 / idx.size)
        self._sup = np.stack([sx, sy], axis=1)
        self._w = w
        self._area = float(idx.size * self._cell_area)
        self._centroid = np.array([float((w * sx).sum()), float((w * sy).sum())])
        self.version += 1

    def _regrid_for_disks(self, use_clue: bool, use_pad0: bool) -> None:
        half = GRID_HALF
        if use_clue and self.clue_xy is not None:
            half = max(half, float(np.abs(self.clue_xy).max()) + self.r_clue + MARGIN)
        if use_pad0 and self.pad0_xy is not None:
            half = max(half, float(np.abs(self.pad0_xy).max()) + self.r_pad + MARGIN)
        half = min(half, 260.0)
        cell = max(self.grid_m, half * 2.0 / np.sqrt(GRID_MAX_CELLS))
        self._build_grid(half, cell)

    def contains(self, xy) -> bool:
        q, single = _as_xy(xy)
        ok = np.ones(q.shape[0], dtype=bool)
        finite = np.all(np.isfinite(q), axis=1)
        qf = np.where(finite[:, None], q, 0.0)
        if self._b_eff is not None:
            lim = self._b_eff + MARGIN
            ok &= (np.abs(qf[:, 0]) <= lim) & (np.abs(qf[:, 1]) <= lim)
        if self._use_clue and self.clue_xy is not None:
            r = self.r_clue + MARGIN
            d2 = (qf[:, 0] - self.clue_xy[0]) ** 2 + (qf[:, 1] - self.clue_xy[1]) ** 2
            ok &= d2 <= r * r
        if self._use_pad0 and self.pad0_xy is not None:
            r = self.r_pad + MARGIN
            d2 = (qf[:, 0] - self.pad0_xy[0]) ** 2 + (qf[:, 1] - self.pad0_xy[1]) ** 2
            ok &= d2 <= r * r
        ok &= finite
        return bool(ok[0]) if single else ok

    def dist_outside(self, xy):
        return self._dist_outside_with(xy, self._b_eff)

    def dist_outside_relaxed(self, xy):
        return self._dist_outside_with(xy, self._b_relax)

    def contains_relaxed(self, xy) -> bool:
        d = self._dist_outside_with(xy, self._b_relax)
        return bool(d <= 0.0) if np.isscalar(d) or np.ndim(d) == 0 else (d <= 0.0)

    def _dist_outside_with(self, xy, b_eff):
        q, single = _as_xy(xy)
        finite = np.all(np.isfinite(q), axis=1)
        qf = np.where(finite[:, None], q, 0.0)
        d = np.zeros(q.shape[0], dtype=np.float64)
        if b_eff is not None:
            lim = float(b_eff) + MARGIN
            d = np.maximum(d, np.abs(qf[:, 0]) - lim)
            d = np.maximum(d, np.abs(qf[:, 1]) - lim)
        if self._use_clue and self.clue_xy is not None:
            r = self.r_clue + MARGIN
            d = np.maximum(d, np.hypot(qf[:, 0] - self.clue_xy[0],
                                       qf[:, 1] - self.clue_xy[1]) - r)
        if self._use_pad0 and self.pad0_xy is not None:
            r = self.r_pad + MARGIN
            d = np.maximum(d, np.hypot(qf[:, 0] - self.pad0_xy[0],
                                       qf[:, 1] - self.pad0_xy[1]) - r)
        d = np.where(finite, np.maximum(d, 0.0), 1e3)
        return float(d[0]) if single else d

    def support_xy(self) -> np.ndarray:
        return self._sup.copy()

    def weights(self) -> np.ndarray:
        return self._w.copy()

    def area_m2(self) -> float:
        return self._area

    def centroid(self) -> np.ndarray:
        return self._centroid.copy()

    def cell_area(self) -> float:
        return float(self._cell_area)

    def core_mask(self, mass_frac: float = 0.90) -> np.ndarray:
        order = np.argsort(-self._w, kind="stable")
        cum = np.cumsum(self._w[order])
        keep = int(np.searchsorted(cum, float(mass_frac), side="left")) + 1
        m = np.zeros(self._w.size, dtype=bool)
        m[order[:min(keep, self._w.size)]] = True
        return m

    def core_xy(self, mass_frac: float = 0.90) -> np.ndarray:
        return self._sup[self.core_mask(mass_frac)].copy()

    def core_area_m2(self, mass_frac: float = 0.90) -> float:
        return float(self.core_mask(mass_frac).sum() * self._cell_area)

class LanePlanner:
    advance_r = 0.0          # P3: > 0 -> the lane clock snaps forward on early arrival (set by the team on mountain)
    advance_hold = 2.0       # P3: seconds of the scheduled dwell kept at the reached waypoint
    n_advanced = 0

    def __init__(self, posterior: Posterior):
        self.P = posterior
        self._n = 0
        self._version = -1
        self._key: bytes | None = None
        self._labels = np.zeros(0, dtype=np.int64)
        self._cell_lane = np.zeros(0, dtype=np.int64)
        self._lane_cells: list[np.ndarray] = []
        self._lane_w: list[np.ndarray] = []
        self._lane_centroid = np.zeros((0, 2))
        self._tours: list[np.ndarray] = []
        self._dur: list[np.ndarray] = []
        self._dur_rev: list[np.ndarray] = []
        self._t0 = np.zeros(0)
        self._scan_r = np.zeros(0)
        self._lane_width = np.zeros(0)
        self._cap = np.zeros(0)

    def assign(self, pos_xy) -> np.ndarray:
        P = np.asarray(pos_xy, dtype=np.float64).reshape(-1, 2)
        P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
        n = int(P.shape[0])
        if n <= 0:
            return np.zeros(0, dtype=np.int64)
        key = P.tobytes()
        if self._version == self.P.version and self._key == key and self._n == n:
            return self._labels.copy()

        sup = self.P._sup
        w = self.P._w
        mu = self.P._centroid
        k = int(sup.shape[0])

        canon = np.lexsort((P[:, 1], P[:, 0]))
        Pc = P[canon]

        dv = Pc - mu
        dist = np.hypot(dv[:, 0], dv[:, 1])
        psi = np.arctan2(dv[:, 1], dv[:, 0])
        synth = dist < CENTRE_EPS
        if synth.any():
            psi = np.where(synth, -np.pi + _TWO_PI * np.arange(n) / n, psi)
        order = np.argsort(psi, kind="stable")
        phi0 = psi[order[0]] - np.pi / n

        phi = np.arctan2(sup[:, 1] - mu[1], sup[:, 0] - mu[0])
        ckey = np.mod(phi - phi0, _TWO_PI)
        srt = np.argsort(ckey, kind="stable")

        meas = np.full(k, 1.0 / max(k, 1))
        F = np.cumsum(meas[srt])

        cap = np.ones(n)
        cell_lane = np.zeros(k, dtype=np.int64)
        lane_centroid = np.zeros((n, 2))
        for _ in range(2):
            q = np.cumsum(cap[order]) / cap.sum()
            q[-1] = 1.0
            lane_rank = np.clip(np.searchsorted(q, F, side="left"), 0, n - 1)
            lane_rank = self._repair(lane_rank, n, k)
            cell_lane[srt] = lane_rank
            t_arr = np.zeros(n)
            for r in range(n):
                sel = lane_rank == r
                if not sel.any():
                    lane_centroid[r] = mu
                else:
                    cw = w[srt][sel]
                    cc = sup[srt][sel]
                    s = cw.sum()
                    lane_centroid[r] = (cw @ cc) / s if s > 0 else cc.mean(axis=0)
                di = np.hypot(*(lane_centroid[r] - Pc[order[r]]))
                t_arr[r] = TAKEOFF_SEC + di / TRANSIT_V
                cap[order[r]] = np.clip(USABLE_SEC - t_arr[r], CAP_LO, CAP_HI)

        labels_canon = np.zeros(n, dtype=np.int64)
        labels_canon[order] = np.arange(n, dtype=np.int64)
        labels = np.zeros(n, dtype=np.int64)
        labels[canon] = labels_canon

        self._lane_cells = []
        self._lane_w = []
        self._tours = []
        self._dur = []
        self._dur_rev = []
        t0 = np.zeros(n)
        scan_r = np.ones(n)
        lane_width = np.zeros(n)
        for r in range(n):
            sel = cell_lane == r
            cells = sup[sel]
            cw = w[sel]
            if cells.shape[0] == 0:
                j = int(np.argmin(np.sum((sup - lane_centroid[r]) ** 2, axis=1)))
                cells = sup[j:j + 1]
                cw = w[j:j + 1]
            self._lane_cells.append(cells)
            self._lane_w.append(cw)
            start = Pc[order[r]]
            tour = self._build_tour(cells, cw, start)
            self._tours.append(tour)
            dur = np.full(tour.shape[0], HOLD_SEC)
            if tour.shape[0] > 1:
                dur[1:] += np.hypot(np.diff(tour[:, 0]), np.diff(tour[:, 1])) / SCAN_V
            self._dur.append(dur)

            self._dur_rev.append(dur if dur.size < 2 else
                                 np.concatenate([dur[:1], dur[:0:-1]]))
            t0[r] = TAKEOFF_SEC + float(np.hypot(*(tour[0] - start))) / TRANSIT_V
            lw = self._minor_extent(cells)
            lane_width[r] = lw
            scan_r[r] = float(np.clip(min(lw, SCAN_R_REF) / SCAN_R_REF, SCAN_R_MIN, 1.0))

        self._n = n
        self._version = self.P.version
        self._key = key
        self._labels = labels
        self._cell_lane = cell_lane
        self._lane_centroid = lane_centroid
        self._t0 = t0
        self._scan_r = scan_r
        self._lane_width = lane_width
        self._cap = cap[order].copy()
        return labels.copy()

    @staticmethod
    def _repair(lane_rank: np.ndarray, n: int, k: int) -> np.ndarray:
        if k < n:
            return lane_rank
        out = lane_rank.copy()
        counts = np.bincount(out, minlength=n)
        for r in np.flatnonzero(counts == 0):
            target = int(round((r + 0.5) * k / n))
            best = -1
            for off in range(k):
                for j in (target + off, target - off):
                    if 0 <= j < k and counts[out[j]] > 1:
                        best = j
                        break
                if best >= 0:
                    break
            if best < 0:
                continue
            counts[out[best]] -= 1
            out[best] = r
            counts[r] += 1
        return out

    @staticmethod
    def _build_tour(cells: np.ndarray, w: np.ndarray, start_xy: np.ndarray) -> np.ndarray:
        order = np.argsort(-w, kind="stable")
        wps: list[np.ndarray] = []
        for i in order:
            p = cells[i]
            if all(np.hypot(p[0] - q[0], p[1] - q[1]) >= WP_SEP for q in wps):
                wps.append(p)
                if len(wps) >= WP_MAX:
                    break
        if not wps:
            wps = [cells[int(order[0])]]
        rem = list(range(len(wps)))
        tour: list[np.ndarray] = []
        cur = np.asarray(start_xy, dtype=np.float64)
        while rem:
            d = [float(np.hypot(wps[j][0] - cur[0], wps[j][1] - cur[1])) for j in rem]
            j = rem[int(np.argmin(d))]
            tour.append(wps[j])
            cur = wps[j]
            rem.remove(j)
        return np.asarray(tour, dtype=np.float64).reshape(-1, 2)

    @staticmethod
    def _minor_extent(cells: np.ndarray) -> float:
        if cells.shape[0] < 3:
            return SCAN_R_REF
        c = cells - cells.mean(axis=0)
        cov = (c.T @ c) / max(cells.shape[0] - 1, 1)
        evals, evecs = np.linalg.eigh(cov)
        proj = c @ evecs[:, 0]
        return float(proj.max() - proj.min())

    def waypoint(self, lane_id, pos_xy, t_sec, locked) -> np.ndarray:
        r = int(lane_id)
        if not self._tours or r < 0 or r >= len(self._tours):
            return self.P.centroid()
        tour = self._tours[r]
        m = int(tour.shape[0])
        p = np.asarray(pos_xy, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(p)):
            p = self._lane_centroid[r]
        if m == 1:
            return tour[0].copy()

        if bool(locked):
            d = np.hypot(tour[:, 0] - p[0], tour[:, 1] - p[1])
            return tour[int(np.argmin(d))].copy()

        t = float(t_sec) if np.isfinite(t_sec) else 0.0
        elapsed = t - float(self._t0[r])
        if elapsed <= 0.0:
            # P3: the first waypoint reached before its scheduled time (transit faster than TRANSIT_V) -- start
            # the tour clock now so the arrival branch below can advance (b1-625: 40.7 -> 37.2 s)
            if self.advance_r > 0.0 and float(np.hypot(tour[0, 0] - p[0], tour[0, 1] - p[1])) < self.advance_r:
                self._t0[r] = t
                elapsed = 0.0
            else:
                return tour[0].copy()
        cyc = float(np.sum(self._dur[r]))
        if cyc <= 1e-6:
            return tour[0].copy()

        npass = int(elapsed // cyc)
        phase = elapsed - npass * cyc
        rev = bool(npass & 1)
        cum = np.cumsum(self._dur_rev[r] if rev else self._dur[r])
        j = int(np.clip(np.searchsorted(cum, phase, side="right"), 0, m - 1))
        if PASS_OFFSET_M > 0.0 and npass >= 1:
            # NEWDET: a repeated pass is flown on a laterally shifted line (alternating side) so the same ground is
            # seen from a different viewpoint instead of re-tracing the exact first pass
            tour = self._pass_shifted(r, tour, npass)
        if self.advance_r > 0.0 and j < m - 1:
            # P3: arrived early at the scheduled waypoint -- snap the lane clock to the next one
            idx = m - 1 - j if rev else j
            if float(np.hypot(tour[idx, 0] - p[0], tour[idx, 1] - p[1])) < self.advance_r:
                # keep the scheduled dwell (advance_hold s) at the waypoint -- the local spiral / look-around
                # during the dwell is coverage; snapping the whole leg lost b3 304/610/665/679 (never framed)
                remaining = float(cum[j]) - phase - float(self.advance_hold)
                if remaining > 0.0:
                    self._t0[r] -= remaining
                    self.n_advanced += 1
                    if self.advance_hold <= 0.0:
                        j += 1
        return tour[m - 1 - j].copy() if rev else tour[j].copy()

    def _pass_shifted(self, r, tour, npass):
        key = (int(r), int(npass))
        cache = getattr(self, "_shift_cache", None)
        if cache is None:
            cache = self._shift_cache = {}
        if key in cache:
            return cache[key]
        m = int(tour.shape[0])
        side = 1.0 if (npass % 2 == 1) else -1.0
        mag = PASS_OFFSET_M * (1.0 + 0.5 * ((npass - 1) // 2))
        out = tour.copy()
        for k in range(m):
            a = tour[max(k - 1, 0)]
            b = tour[min(k + 1, m - 1)]
            d = b - a
            n = float(np.hypot(d[0], d[1]))
            if n < 1e-6:
                c = self._lane_centroid[r] if r < len(self._lane_centroid) else tour[k]
                d = tour[k] - c
                n = float(np.hypot(d[0], d[1]))
                if n < 1e-6:
                    continue
            perp = np.array([-d[1], d[0]]) / n
            out[k] = tour[k] + side * mag * perp
        cache[key] = out
        return out

    def lane_of(self, lane_id) -> np.ndarray:
        r = int(lane_id)
        if not self._lane_cells or r < 0 or r >= len(self._lane_cells):
            return self.P.support_xy()
        return self._lane_cells[r].copy()

    def n_lanes(self) -> int:
        return int(self._n)

    def tour(self, lane_id) -> np.ndarray:
        r = int(lane_id)
        if not self._tours or r < 0 or r >= len(self._tours):
            return self.P.centroid().reshape(1, 2)
        return self._tours[r].copy()

    def lane_centroid(self, lane_id) -> np.ndarray:
        r = int(lane_id)
        if self._lane_centroid.shape[0] == 0 or r < 0 or r >= self._lane_centroid.shape[0]:
            return self.P.centroid()
        return self._lane_centroid[r].copy()

    def lane_width(self, lane_id) -> float:
        r = int(lane_id)
        if self._lane_width.size == 0 or r < 0 or r >= self._lane_width.size:
            return SCAN_R_REF
        return float(self._lane_width[r])

    def scan_r_scale(self, lane_id) -> float:
        r = int(lane_id)
        if self._scan_r.size == 0 or r < 0 or r >= self._scan_r.size:
            return 1.0
        return float(self._scan_r[r])

    def lane_area_m2(self, lane_id) -> float:
        return float(self.lane_of(lane_id).shape[0] * self.P.cell_area())

    def lane_mass(self, lane_id) -> float:
        r = int(lane_id)
        if not self._lane_w or r < 0 or r >= len(self._lane_w):
            return 1.0
        return float(self._lane_w[r].sum())

    def lane_arrival_t(self, lane_id) -> float:
        r = int(lane_id)
        if self._t0.size == 0 or r < 0 or r >= self._t0.size:
            return TAKEOFF_SEC
        return float(self._t0[r])

    def lane_capacity(self, lane_id) -> float:
        r = int(lane_id)
        if self._cap.size == 0 or r < 0 or r >= self._cap.size:
            return CAP_HI
        return float(self._cap[r])

    def cell_lane(self) -> np.ndarray:
        return self._cell_lane.copy()

_FLATMOD_posterior = _FlatModule(
    'posterior',
    Posterior=Posterior, LanePlanner=LanePlanner)

import json
import os
import sys
from pathlib import Path
import onnxruntime as ort
SIM_DT = 1.0 / 50.0
DEPTH_RES = 256
DEPTH_MIN_M, DEPTH_MAX_M = (0.5, 30.0)
CAMERA_OFFSET_M = 0.13
CAMERA_UP_OFFSET_M = 0.05
HALF_TAN = 1.0
MAX_RAY_M = 20.0
SPEED_LIMIT = 3.0
IDX_POS = slice(0, 3)
IDX_RPY = slice(3, 6)
IDX_VEL = slice(6, 9)
IDX_ACT_HIST = slice(12, 162)
IDX_ALT = 162
IDX_CLUE = slice(163, 165)

def _repo_fallback(rel: str) -> Path:
    here = Path(__file__).resolve()
    root = here.parents[3] if len(here.parents) > 3 else here.parent
    return root / rel
# --- NEWDET knobs (dev): swap the champion depth head for the per-map GoalDetector heads
NEWDET_ON = os.environ.get('NEWDET', '1') == '1'
NEWDET_MAPS = tuple(m for m in os.environ.get('NEWDET_MAPS', '').split(',') if m)   # empty = all maps
NEWDET_UNKNOWN = os.environ.get('NEWDET_UNKNOWN', '124')     # head before the map latch ('' = champion)
NEWDET_H_MODE = os.environ.get('NEWDET_H', 'champ')          # 'champ' | 'const'
NEWDET_H_CONST = float(os.environ.get('NEWDET_HC', '1.6'))
NEWDET_T = float(os.environ.get('NEWDET_T', '0'))            # new head's own STRONG_P-equivalent (0 = raw)
# per-map bar: NEWDET_T_<map> overrides NEWDET_T (the new head's p that should count as STRONG_P)
NEWDET_T_MAP = {m: float(os.environ.get('NEWDET_T_' + m.upper(), '0')) for m in ('city', 'open', 'mountain', 'village', 'forest')}
NEWDET_N_LATCH = float(os.environ.get('NEWDET_NLATCH', '0'))   # >0: consistent hits needed to latch when the new head owns the frame
NEWDET_LR_OFF_MAPS = tuple(m for m in os.environ.get('NEWDET_LR_OFF', '').split(',') if m)   # maps where the long-range head is disabled
NEWDET_MIX = os.environ.get('NEWDET_MIX', 'new')             # 'new' | 'max' (take whichever head is surer)
# fusion of the two detector families (default off = new head alone):
NEWDET_AND_P = float(os.environ.get('NEWDET_AND_P', '0'))     # >0: beyond NEWDET_AND_R the new head's hit counts only if the champion head has p >= this
NEWDET_AND_R = float(os.environ.get('NEWDET_AND_R', '15'))
NEWDET_NEAR_M = float(os.environ.get('NEWDET_NEAR', '0'))     # >0: inside this range take the champion head's u/v/cz when it is strong (better near-range accuracy)
NEWDET_AVG_M = float(os.environ.get('NEWDET_AVG', '0'))       # >0: when both heads are strong and agree within this, average their world points
NEWDET_MAXP = os.environ.get('NEWDET_MAXP', '0') == '1'       # p = max(p_new, p_old) (OR of the heads)
DET_NAME = 'victim_depth.onnx'
DET_FALLBACK = _repo_fallback('RL/victim_detect/out/victim_depth.onnx')
RGB_NAME = 'victim_rgb.onnx'
RGB_FALLBACK = _repo_fallback('RL/victim_detect/out/victim_rgb.onnx')
RGB_RES = 128
RGB_REQUEST_PERIOD = 40
RGB_MIN_INTERVAL = 12
RGB_TARGET_TRIGGER = 0.55
RGB_CAP = 40
RGB_WEIGHT = 0.5
RGB_PRESENT_EPS = 0.005
RGB_CONFIRM_THRESH = 0.5
RANGE_VETO = os.environ.get('SAR_RANGE_VETO', '1') == '1'

# --- RGBV2 (2026-09-20): Bayesian colour veto. In flight the colour head says yes on 85-100 % of true
# detections within COLOUR_TRUST_M and no on ~90 % of phantoms, so one "no" leaves ~69 % person odds
# on a p>=0.85 depth track; the champion's hard 6 s veto on a single "no" (which also suppresses the
# on-demand re-check) cost village slot 162 its rescue. A "no" now has to repeat within RGBV2_WINDOW
# ticks at the same spot before it vetoes, unless the depth head itself was weak.
RGBV2_ON = os.environ.get('SWSAR_RGBV2', '1') == '1'
RGBV2_NO_NEEDED = int(os.environ.get('SWSAR_RGBV2_N', '2'))
RGBV2_WINDOW = int(os.environ.get('SWSAR_RGBV2_WIN', '150'))
RGBV2_STRONG_P = float(os.environ.get('SWSAR_RGBV2_P', '0.85'))
RGBV3_TRACK_MIN = float(os.environ.get('SWSAR_RGBV3_TRK', '3.0'))
# village-only floor for the depth-strength below which a single colour 'no' may fast-veto a
# detection (global stays 0.85). Measured standalone on the full 3-base 423-seed village panel:
# +0.0022, 8 seeds changed (7 up / 1 down) -- the narrowest positive arm found.
RGBV2_P_VILLAGE = float(os.environ.get('SWSAR_RGBV2_P_VILLAGE', '0.6'))   # single colour 'no' vetoes only tracks weaker than this
RGBV3_HOLD = os.environ.get('SWSAR_RGBV3_HOLD', '1') == '1'
COLOUR_TRUST_M = float(os.environ.get('SAR_COLOUR_TRUST', '14.0'))
RGB_CONFIRM_WINDOW = 60
GAIN = 0.25
SIGMA_INNOVATION_M = 8.0
P_ACCEPT = 0.657
K_P = 12.0
CONF_DECAY = 0.999
STRONG_P = 0.7
CONSIST_M = 5.0
N_LATCH = 6.0
N_CAP = 14.0
STALE_TICKS = 50
STALE_DECAY = 0.94
STRONG_GAIN = 0.35
TRACK_GAIN = 0.05
MJ_FLOOR = float(os.environ.get('SWSAR_MJ_FLOOR', '0.02'))     # decayed track strength below this is 0 (MSP/MCV gate on strong <= 0)
MJ_WP_LOCK = os.environ.get('SWSAR_MJ_WP', '1') == '1'            # lane waypoint snaps only on a real lock
EDGE_GUARD = os.environ.get('SWSAR_EDGE', '0') == '1'      # p195: cone-edge measurements must not jerk a saturated track
EDGE_EL_DEG = float(os.environ.get('SWSAR_EDGE_EL', '40.0'))  # measured point more than this far below the horizon = bottom 5 deg of the 90 deg cone
EDGE_STRONG_MIN = float(os.environ.get('SWSAR_EDGE_SMIN', '6.0'))
EDGE_STEP_M = float(os.environ.get('SWSAR_EDGE_STEP', '1.0'))
EDGE_DROP_M = float(os.environ.get('SWSAR_EDGE_DROP', '0.6'))
COMMIT_CONF = 0.6
COMMIT_CONF_MTN = float(os.environ.get('SAR220_MTN', '0.45'))
COMMIT_CLUTTER_BONUS = 0.25
LOST_CONF = 0.3
LATCH_CONF = 0.95
LOST_VIS_TICKS = 20
FREEZE_NEAR_M = 6.0
SEARCH_ALT_M = 10.0
HOVER_ABOVE_TOP_M = 2.0
HOVER_SETTLE_MARGIN_M = 0.5
R_HOVER_M = 3.0
CONFIRM_SPEED = 0.9 / SPEED_LIMIT
SPIRAL_R0_M = 4.0
SPIRAL_RMAX_M = 26.0
SPIRAL_GROW_TICKS = 2000
SPIRAL_ANGULAR = 0.011
SPIRAL_PITCH_M = 12.0
SPIRAL_SPEED_MPS = 2.6
SPIRAL_ARC_RMAX_M = 22.0
CTRL_HZ = 50.0
ARC_SPIRAL = os.environ.get('SAR_ARC_SPIRAL', '') == '1'
AVOID_LOOKAHEAD_M = 8.0
AVOID_SAFETY_M = 1.2
AVOID_STRENGTH = 0.7
AVOID_SLOW_M = 3.0
AVOID_MEMORY_DECAY = 0.9
CLUTTER_EDGE_THRESH = 0.05
CLUTTER_EDGE_LO = 0.01
CLUTTER_EDGE_HI = 0.05
CLUTTER_ALT_BONUS_M = 4.0
CLUTTER_EMA = 0.98
MOUNTAIN_SEARCH_ALT_M = float(os.environ.get('SAR_MTN_ALT', '4.0'))
MTN_WP_RING_RADII = (5.0, 11.0, 17.0, 23.0)
MTN_WP_PER_RING = 6
MTN_WP_REACH_M = 3.5
ALT_TD_K = float(os.environ.get('SAR_ALT_K', '0.010'))
ALT_TD_D = float(os.environ.get('SAR_ALT_D', '0.18'))
ALT_TD_VMAX = float(os.environ.get('SAR_ALT_VMAX', '0.10'))
MTN_CLIMB_LOOKAHEAD_M = float(os.environ.get('SAR_MTN_CLIMB', '12.0'))
MTN_CLIMB_GAIN = 1.2
MTN_STEER_DAMP = 0.4
MTN_DET_NAME = 'victim_depth_m.onnx'
MTN_DET_FALLBACK = _repo_fallback('my_pool/victim_detect_mountain/out_v3/mountain_victim_depth.onnx')
MTN_RGB_NAME = 'victim_rgb_m.onnx'
MTN_RGB_FALLBACK = _repo_fallback('my_pool/victim_detect_mountain/out_rgb/mountain_victim_rgb.onnx')
T_OLD_DEPTH = 0.657
T_MTN_DEPTH = 0.586
T_MTN_RGB = 0.713
MTN_RGB_PRIMARY = True
MTN_RGB_UV_TOL = 0.20
MTN_RGB_IMG_WIN = 2
MTN_RGB_IMG_MAX_M = 29.0
MTN_RGB_RANGE_MAX_M = 20.0
MTN_RGB_PROMOTE_H_MAX = 1.0
MTN_RGB_OUTER_RAW = True
MTN_RGB_CHASE_TICKS = 36
MTN_RGB_PACE = True
MTN_RGB_PACE_HORIZON = 2900
MTN_RGB_PACE_CAP = 34
MTN_RGB_PACE_AFTER = 10
MAP_LABELS = ('city', 'open', 'mountain', 'village', 'warehouse', 'forest')

# --- TAKE-OFF SWEEP (TKS, 2026-09-20). See tools/patch_tks.py. Posterior-gated look-around at take-off.
TKS_ON = os.environ.get('SWSAR_TKS', '1') == '1'
TKS_MAPS = tuple(x for x in os.environ.get('SWSAR_TKS_MAPS', 'village,city,open').split(',') if x)
TKS_R_NEAR = float(os.environ.get('SWSAR_TKS_RNEAR', '10.0'))
TKS_P_HOLD = float(os.environ.get('SWSAR_TKS_PHOLD', '0.12'))
TKS_DEG_FREE = float(os.environ.get('SWSAR_TKS_DEGFREE', '150.0'))
TKS_DEG_HOLD = float(os.environ.get('SWSAR_TKS_DEGHOLD', '300.0'))
TKS_RATE = float(os.environ.get('SWSAR_TKS_RATE', '60.0'))     # setpoint ramp deg/s (the airframe follows at ~43)
TKS_MAX_TICKS = int(os.environ.get('SWSAR_TKS_MAXT', '500'))   # never sweep past 10 s
TKS_T0 = 16                                                     # first tick after the map classification
TKS_HOLD_AGL_MAX = float(os.environ.get('SWSAR_TKS_HOLD_AGLMAX', '4.5'))
TKS_VERIFY_COMMIT = int(os.environ.get('SWSAR_TKS_VCOMMIT', '2'))   # strong sightings during a verify that force a commit at its end
TKS_HOLD_ABOVE_PAD = float(os.environ.get('SWSAR_TKS_HOLD_PAD', '3.3'))  # hold height above the take-off position: a 1.2 m village person under the pad sits 2.7 m below it (band [2,4]); the ray on the person's head made an AGL hold sit above the band (slot 68)
TKS_VERIFY_MIN = int(os.environ.get('SWSAR_TKS_VMIN', '25'))       # ticks the camera stays on a sighting at least
TKS_VERIFY_MAX = int(os.environ.get('SWSAR_TKS_VMAX', '75'))       # ... and at most
TKS_VERIFY_N = int(os.environ.get('SWSAR_TKS_VN', '2'))            # verifies per sweep at most
TKS_LEAD = float(os.environ.get('SWSAR_TKS_LEAD', '40.0'))         # yaw setpoint lead over the achieved sweep angle


def _tks_pnear(map_name, n, clue_xy, pad0_xy, starts_xy, r_near):
    """Posterior mass (box x clue disc x pad0 disc, uniform) within r_near of each start."""
    try:
        idx = MAP_LABELS.index(map_name)
    except ValueError:
        return None
    b = float(BOX_B[idx])
    ax = np.arange(-b + 0.5, b, 1.0)
    gx, gy = np.meshgrid(ax, ax, indexing="xy")
    gx = gx.ravel()
    gy = gy.ravel()
    r_clue = CLUE_R_FULL * np.sqrt(min(int(n), int(CLUE_R_NREF)) / CLUE_R_NREF)
    m = np.hypot(gx - clue_xy[0], gy - clue_xy[1]) <= r_clue
    m &= np.hypot(gx - pad0_xy[0], gy - pad0_xy[1]) <= PAD0_R
    tot = int(m.sum())
    if tot <= 0:
        return [0.0] * len(starts_xy)
    out = []
    for s in starts_xy:
        near = m & (np.hypot(gx - s[0], gy - s[1]) <= r_near)
        out.append(float(near.sum()) / tot)
    return out
XGB_NAME = 'map_xgboost_state_depth_stats_model.json'
XGB_STATE_DIM = 141
XGB_EVERY = 5
XGB_MIN_PRED = 3
XGB_MTN_PROB = 0.7
XGB_MAX_TICK = 600
WAYPOINT_TERRAINS = frozenset((t.strip() for t in os.environ.get('SAR_WP_TERRAINS', 'warehouse').split(',') if t.strip()))
WAYPOINT_LATCH_PROB = 0.5
WAREHOUSE_SIDE_LATCH_PROB = 0.15
WAREHOUSE_SIDE_LATCH_MIN_PRED = 40
HIGH_WP_RING_RADII = (8.0, 18.0)
HIGH_WP_PER_RING = 6
HIGH_WP_REACH_M = 4.5
WAYPOINT_FLAT_COMMIT = True
HOVER_GIVEUP_TICKS = 300
HOVER_STABLE_M = 1.5
PHANTOM_VETO_M = CONSIST_M
RGB_VETO_TTL = 300
FOREST_ALT_M = float(os.environ.get('SAR_FOREST_ALT', '5.0'))
# --- FOREST TAKE-OFF / DECK HEIGHT (user, 2026-09-17: "set taking off height as 3 m"). The champion climbs
# to ground + 10 m until its per-drone forest latch (1 s if xgb p >= 0.9 and clutter >= 0.5, else 5 s,
# or never in sparse spots) and then flies the 5 m deck: measured peaks median 6.3 m, p75 10.2 m,
# p90 19 m, 7.6 s to settle. Here, as soon as the map classifier says forest (tick 15), every drone
# climbs to and searches at ground + TK_H, before and after the latch. SWSAR101_TK=0 = parent.
TK_ON = os.environ.get('SWSAR101_TK', '1') == '1'
TK_H = float(os.environ.get('SWSAR101_TK_H', '3.0'))


def _forest_alt():
    return TK_H if (TK_ON and _MAPSW_STATE.get("map") == 'forest') else FOREST_ALT_M
# --- MOUNTAIN TRACK-QUALITY CONFIRMER + DEEPER SWEEP (201, 2026-09-17). Measured on the champion's 445
# mountain seeds: a real person is re-detected on 38 % of the ticks it sits in the camera cone while the
# drone flies at it, a rock on 3 %; real approaches still go blind for 3.6 s (p75) / 6.8 s (p90) behind
# terrain, and the 1 s stale + 0.94/tick decay ends the approach 1.7 s after the last hit. Head estimates
# run 2-4 m high in the terminal failures and the sweep stops at estimate + 1.4. Mountain only.
MQ_ON = os.environ.get('SWSAR201_MQ', '0') == '1'      # v2: neutral (-1) -> off
MQ_VIEW_MIN = int(os.environ.get('SWSAR201_MQ_VIEW', '100'))     # in-view ticks before a verdict
MQ_LOW = float(os.environ.get('SWSAR201_MQ_LOW', '0.0'))          # v1 0.08: 71 refutes, 7 real approaches broken, 0 phantom fails converted -> OFF
MQ_HOLD_VIEW = int(os.environ.get('SWSAR201_MQ_HOLDVIEW', '50'))  # in-view ticks before patience applies
MQ_HIGH = float(os.environ.get('SWSAR201_MQ_HIGH', '0.20'))       # hit rate above this = person -> patience
MQ_HOLD_TICKS = int(os.environ.get('SWSAR201_MQ_HOLD', '250'))    # extra blind ticks the track survives (5 s)
MQ_REFUTE_TTL = int(os.environ.get('SWSAR201_MQ_TTL', '500'))     # refuted spot ignored for 10 s
MQ_RANGE_M = 25.0
MQ_DEP_DEG = 42.0
MQ_PROTECT_TICKS = int(os.environ.get('SWSAR201_MQ_PROTECT', '900'))  # hover protected from team deny this long
MSW_ON = os.environ.get('SWSAR201_MSW', '0') == '1'    # v2: neutral (-1) -> off
MSW_LO = float(os.environ.get('SWSAR201_MSW_LO', '-2.0'))         # sweep floor relative to the head estimate (champion 1.4)
MSW_MPS = float(os.environ.get('SWSAR201_MSW_MPS', '0.8'))
MSW_HI = float(os.environ.get('SWSAR201_MSW_HI', '4.6'))           # v1 stopped at +2.5 on the way up and lost the champion's late catches (251099, 251702)
MSW_TOP = 2.5                                                    # sweep starts and ends at estimate + 2.5


def _mq_on():
    # gated per drone by the champion's own mountain latch (mtn.is_mountain) at the call sites
    return MQ_ON
# --- MOUNTAIN TAKE-OFF 3 m + FULL SPEED (user, 2026-09-17: "focus on saving time"). Champion mountain
# drones climb toward the 10 m deck at 0.6 speed (median peak +8.7 m above the start at 9 s, 71 % never
# back under +5 m within 10 s) and then search the 4 m deck; only 41 % of their transit ticks run at
# >= 2.8 m/s horizontally, the ramp-down within 8 m of every ring waypoint (24 of them) eats the rest.
# Here: take off to ground + MTK_H at full speed, and no ramp-down at search waypoints. Mountain only.
MTK_ON = os.environ.get('SWSAR201_MTK', '1') == '1'
MTK_H = float(os.environ.get('SWSAR201_MTK_H', '3.0'))
MTK_CLIMB_SPEED = float(os.environ.get('SWSAR201_MTK_CLIMB', '1.0'))
MTK_SEARCH_H = float(os.environ.get('SWSAR201_MTK_SEARCH', '0'))   # > 0 also sets the mountain search deck
MTK_FULL_SPEED = os.environ.get('SWSAR201_MTK_FULL', '1') == '1'   # no waypoint ramp-down in search mode
INIT_EXIT_MARGIN = float(os.environ.get('NEWDET_INIT_MARGIN', '1.0'))   # NEWDET: mountain take-off exits 'initial' when within this of the deck
INIT_EXIT_TICKS = int(os.environ.get('NEWDET_INIT_TICKS', '0'))          # NEWDET: >0 replaces the 150-tick fallback on mountain
# NEWDET: fleet-size-conditional overrides (small fleets are coverage-bound on mountain); 0 = off
N2_MAX = int(os.environ.get('NEWDET_N2_MAX', '2'))
N2_MTN_ALT = float(os.environ.get('NEWDET_N2_MTN_ALT', '4.5'))
N2_INIT_TICKS = int(os.environ.get('NEWDET_N2_INIT_TICKS', '300'))
N2_INIT_MARGIN = float(os.environ.get('NEWDET_N2_INIT_MARGIN', '0.3'))
_N2_DEFAULTS = None
NBIG_MIN = int(os.environ.get('NEWDET_NBIG_MIN', '6'))            # NEWDET: n >= this -> mountain joins the OMB barrier maps
NBIG_MTN_RSAFE = float(os.environ.get('NEWDET_NBIG_MTN_RSAFE', '0.5'))
_NBIG_DEFAULTS = None
NF_MAX = int(os.environ.get('NEWDET_NF_MAX', '3'))                # NEWDET: n <= this -> forest deck override
NF_TKH = float(os.environ.get('NEWDET_NF_TKH', '2.7'))
_NF_DEFAULT = None
SEP_MIN_AGL = float(os.environ.get('NEWDET_SEP_MIN_AGL', '0'))          # NEWDET: mountain: skip the separation push below this AGL (0 = off)
# --- MOUNTAIN HOVER HEIGHT (user, 2026-09-17: "many failed hoverings for the real person"). The head
# estimate = tracked point z + half the net's height output; on mountain that height output runs 4.7-9.6 m
# for 0.8-2.2 m people (hub replays of 251416 / 250457 / 250259 / 250566), so the hover sits 2-4 m above
# the 2-4 m rescue band and the sweep never reaches it. Clamp the height to a human range and, instead
# of replacing the hover target with the downward-ray + 3.2 m (which is feet + 3.2 = below the band
# for a tall person when the ray misses the head), take the higher of the two. Mountain only.
MH_ON = os.environ.get('SWSAR201_MH', '1') == '1'
MH_HMAX = float(os.environ.get('SWSAR201_MH_HMAX', '2.2'))
MH_HMIN = 0.3
MH_BIDIR = os.environ.get('SWSAR201_MH_BIDIR', '0') == '1'         # 1 = champion's ray-replaces-target
# --- MOUNTAIN LOW APPROACH (user, 2026-09-17: "drones don't check the person's real position until they
# fly to him"). The champion navigates at the 4 m search deck and only descends once it is over the
# estimate -- from there the person is below the 45-degree camera cone, so the last sightings are from
# 10-20 m away and every metre of estimate error is baked in. Here a locked mountain drone descends to
# the hover height while still MA_R m out, so it keeps re-detecting the person from close range until
# the hover starts. The team's 2.5 m AGL guard still holds it off the ground.
MA_ON = os.environ.get('SWSAR201_MA', '1') == '1'
MA_R = float(os.environ.get('SWSAR302_MA_R', os.environ.get('SWSAR201_MA_R', '15.0')))   # 302: sighted approach from 15 m
# --- 302 FIXED HOVER (user, 2026-09-18: "the drone should know the person's exact (x, y, z) for hovering").
# In 302's 35 mountain fails, 8-10 seeds had the person seen from <= 8 m with 0.3-0.5 m accuracy (27-380 hits) and
# still no rescue: the hover sat on a stale far-range estimate 2-4 m too high (250259, 251722 hovered 4.0-4.6 m
# above the top, band ends at 4.0), or a blind give-up had blacklisted the person (250289: 380 accurate hits
# afterwards, no re-lock allowed). Here: hits from <= FIX_RANGE_M are collected; once FIX_MIN of them arrive within
# FIX_WINDOW ticks, the hover point is FROZEN on their median (x, y, top = z + half height) instead of the tracker's
# estimate, the drone freezes as soon as it is within R_HOVER of it (not only after losing sight), and a spot with
# >= FIX_EVIDENCE accurate close hits is never blacklisted by a give-up. Worlds: FIX_WORLDS.
FIX_ON = os.environ.get('SWSAR302_FIX', '1') == '1'
FIX_WORLDS = tuple(x for x in os.environ.get('SWSAR302_FIX_WORLDS', 'mountain').split(',') if x)
FIX_RANGE_M = float(os.environ.get('SWSAR302_FIX_RANGE', '8.0'))
FIX_MIN = int(os.environ.get('SWSAR302_FIX_MIN', '10'))
FIX_WINDOW = int(os.environ.get('SWSAR302_FIX_WINDOW', '50'))
FIX_EVIDENCE = int(os.environ.get('SWSAR302_FIX_EVIDENCE', '20'))
FIX_NEAR_M = 6.0                        # fixes are only applied once the drone is this close to its target


def _fix_world(ctrl):
    if not FIX_ON:
        return False
    if ctrl.mtn.is_mountain:
        return 'mountain' in FIX_WORLDS
    if _forest_hov(ctrl):
        return True                                 # 304: forest hover overhaul
    return _MAPSW_STATE.get("map") in FIX_WORLDS
# --- FOREST TAKE-OFF LOOK-AROUND (user, 2026-09-18: "at the start of the episode the drones should rotate the
# camera and find the person near the start"). 18 of the 301 lane's 30 forest fails have the person within 15 m of
# a drone start, 16 of them outside the initial +-45 deg camera cone (every drone starts facing +x), and the
# detector never runs in 'initial' mode. The 900 g airframe yaws ~35-40 deg/s when hovering (yaw torque clipped)
# and ~200 deg/s only at 3 m/s, so a turn on the spot costs time (360 deg = 10 s): variant 1 keeps it to +-45 deg.
# Variant 1 (HUB, 60 forest seeds paired: 21 of 30 fails rescued / 1 of 30 near-start wins broken): from the
# forest classification until TK_SWEEP_TICKS the yaw target sits TK_SWEEP_DEG right of the take-off heading for
# TK_SWEEP_TURN ticks, then TK_SWEEP_DEG left; once the initial climb is over the drone holds its xy until the
# window ends. A strong depth output, a lock or leaving search mode ends the hold at once (TK_SWEEP_STOP).
# Variant 2: the same right/left yaw sweep with NO hold (the search leg flies on under a swinging camera; the
# champion's blind-speed rule slows it when the camera is off the travel axis and a trunk is near). Variant 3: a
# forced 301 look-around turn as soon as the frame ahead reads >= TKB_CLEAR_M, aborting inside TKB_ABORT_M -- note
# the 170 deg/s commanded turn is a +-50 deg wobble at search speed (the airframe cannot follow). Forest only; 0 = 302.
TK_SWEEP = int(os.environ.get('SWSAR303_TKSWEEP', '1'))          # 304: ON — the user chose 'A + take-off sweep' (2026-09-18)
TK_SWEEP_TICKS = int(os.environ.get('SWSAR303_TK_TICKS', '200'))     # window end, ticks from episode start (4 s)
TK_SWEEP_TURN = int(os.environ.get('SWSAR303_TK_TURN', '75'))        # right leg length before the left leg
TK_SWEEP_DEG = float(os.environ.get('SWSAR303_TK_DEG', '60.0'))      # yaw target offset (real swing ~45 deg)
TK_SWEEP_STOP = os.environ.get('SWSAR303_TK_STOP', '1') == '1'
TKB_CLEAR_M = float(os.environ.get('SWSAR303_TKB_CLEAR', '6.0'))
TKB_ABORT_M = float(os.environ.get('SWSAR303_TKB_ABORT', '3.0'))

def _tk_sweep_world():
    return TK_SWEEP > 0 and _MAPSW_STATE.get("map") == 'forest'

# --- FOREST HOVER OVERHAUL (user, 2026-09-18: "the drones detect the person but don't try hovering, or hover at the
# wrong position"). 22 of 303's 28 forest fails saw the person; 15 never locked on it, 2 hovered 2 m off, 3 locked
# without a hover. Three mechanisms: (1) navigation keeps the deck height (local ground + 3 m) while closing, so a
# person on lower ground slides under the 90-degree frame before the 6-hit lock completes (250889: 4.5 m above the
# head); (2) the lock takes one position estimate, 1-2 m off at 7-13 m, the hover sits outside the 2 m cylinder,
# the give-up blacklists the spot within 5 m and the real person is vetoed for the rest of the episode (250716,
# 251131); (3) drones hold phantom locks for 30-45 s while the real sightings arrive.
# FA: a navigating drone within FA_R of its candidate (locked or not) descends to the candidate's head + 2.5 m
# (floor: local ground + FA_FLOOR) instead of the deck, keeping the head in the frame.
# HOV=1: 302's fixed hover on forest (mass centre of the close hits, freeze on the fix, no blacklist / veto lifted
# on >= 20 close hits), hits collected from FIXF_RANGE_M and applied from FIXF_NEAR_M (the forest sightings are
# farther than mountain's). HOV=2: the centroid keeps updating while the drone approaches (longer window, no freeze
# on the fix -- the champion's lost-sight freeze applies), and a give-up on a spot with >= 20 close hits re-hovers
# once on the centroid instead of unlocking. Forest only; other worlds byte-identical to 302.
FA_ON = os.environ.get('SWSAR304_FA', '1') == '1'
FA_R = float(os.environ.get('SWSAR304_FA_R', '12.0'))
FA_FLOOR = float(os.environ.get('SWSAR304_FA_FLOOR', '1.5'))
FA_EL_DEG = float(os.environ.get('SWSAR304_FA_EL', '30.0'))    # descend before closing when the head is below this depression angle
HOV_MODE = int(os.environ.get('SWSAR304_HOV', '1'))
FIXF_RANGE_M = float(os.environ.get('SWSAR304_FIXF_RANGE', '12.0'))
FIXF_NEAR_M = float(os.environ.get('SWSAR304_FIXF_NEAR', '8.0'))
FIXF_WINDOW = int(os.environ.get('SWSAR304_FIXF_WINDOW', '150'))
FIXF_RETRY = int(os.environ.get('SWSAR304_FIXF_RETRY', '1'))

def _forest_here(ctrl):
    return (not ctrl.mtn.is_mountain) and (_MAPSW_STATE.get("map") == 'forest' or bool(getattr(ctrl, 'forest_low', False)))

def _forest_hov(ctrl):
    return HOV_MODE > 0 and _forest_here(ctrl)
# --- MOUNTAIN RGB FOR CHECKING (user, 2026-09-17: "don't waste the RGB camera, use it for checking").
# The champion spends the 40-frame budget on a periodic scan (every 40 ticks, then paced to 34 over
# 58 s): mountain fails have 35 of 40 gone by the end, drone 2 of 251608 burnt 35 frames in one
# approach. Here the periodic scan is off on mountain; frames go to (a) the depth-triggered
# on-demand look (kept), (b) a CHECK cadence while a locked drone approaches its target within
# MRGB_CHECK_R m (every MRGB_CHECK_GAP ticks), where the person is in the picture.
# --- MOUNTAIN LOOK-AROUND WHILE FLYING (user, 2026-09-17: "if the drone didn't find the person and the
# forward direction is safe, rotate 360 degrees and keep flying"). The camera is bolted to the body:
# a searching drone only ever sees the 90-degree cone ahead of its track, and 10 of the champion's 39
# mountain failures flew within 0.3-9 m of the person without a detection, 14 more never got a look.
# Here a searching mountain drone with no track keeps its velocity command and turns its yaw through a
# full circle once every MSP_PERIOD ticks, only when the depth sectors read >= MSP_CLEAR_M ahead and the
# downward ray >= MSP_AGL_MIN; the turn stops at once on a strong depth output, a lock, terrain inside
# the look-ahead, or leaving search mode. No stand-still (the forest spin-on-the-spot cost -3).
MSP_ON = os.environ.get('SWSAR201_MSP', '1') == '1'
MSP_PERIOD = int(os.environ.get('SWSAR201_MSP_PERIOD', '400'))      # ticks between turns (8 s)
MSP_T0 = int(os.environ.get('SWSAR201_MSP_T0', '150'))              # no turn before 3 s
MSP_RATE = float(os.environ.get('SWSAR201_MSP_RATE', '170.0'))      # deg/s commanded (cap 180)
MSP_CLEAR_M = float(os.environ.get('SWSAR201_MSP_CLEAR', '12.0'))
MSP_AGL_MIN = float(os.environ.get('SWSAR201_MSP_AGL', '4.0'))
MSP_MIN_SPEED = 0.2

# --- MOUNTAIN BLIND-TURN CAP (MBT, 2026-09-21): see tools/patch_mbt.py
MBT_ON = os.environ.get('SWSAR_MBT', '0') == '1'    # off by default: with the dwell kept in the tour advance the hard turns are gone; the cap only cost time (panel 10.63 vs 9.77)
MBT_DEG = float(os.environ.get('SWSAR_MBT_DEG', '60.0'))
MBT_SPEED = float(os.environ.get('SWSAR_MBT_V', '0.5'))
MBT_WIN = int(os.environ.get('SWSAR_MBT_WIN', '100'))       # ticks after a waypoint jump during which the cap applies
MBT_JUMP_M = float(os.environ.get('SWSAR_MBT_JUMP', '3.0'))

# --- MOUNTAIN CLOSE-HIT VERIFY (MCV, 2026-09-21): see tools/patch_mcv.py
MCV_ON = os.environ.get('SWSAR_MCV', '1') == '1'
MCV_R = float(os.environ.get('SWSAR_MCV_R', '8.0'))
MCV_TICKS = int(os.environ.get('SWSAR_MCV_T', '60'))
MCV_SPEED = float(os.environ.get('SWSAR_MCV_V', '0.25'))
MCV_GAP = int(os.environ.get('SWSAR_MCV_GAP', '150'))
MRGB_ON = os.environ.get('SWSAR201_MRGB', '1') == '1'
# --- 301: the look-around while flying ALSO on forest (user, 2026-09-17). Forest flies the 3 m deck
# between trunks, so the gates are its own: clear-ahead FSP_CLEAR_M, downward ray >= FSP_AGL_MIN, and the
# turn aborts as soon as any depth sector reads < FSP_ABORT_M (the avoider is blind ahead while yawed).
FSP_ON = os.environ.get('SWSAR301_FSP', '1') == '1'
FSP_CLEAR_M = float(os.environ.get('SWSAR301_FSP_CLEAR', '12.0'))
FSP_AGL_MIN = float(os.environ.get('SWSAR301_FSP_AGL', '2.5'))
FSP_ABORT_M = float(os.environ.get('SWSAR301_FSP_ABORT', '8.0'))
FSP_PERIOD = int(os.environ.get('SWSAR301_FSP_PERIOD', '400'))
# --- 301 VILLAGE absolute altitude cap (user, 2026-09-17: "max absolute altitude 8 m; the drones should compute
# the absolute altitude"). Village ground is z 0.10 m and the person stands at z ~0.7; the champion's village deck is
# 7 m plus a clutter bonus of up to 4 m, and the 2.5 m ray floor lifts drones over roofs: wins fly a median 6.9 m
# absolute, fails 7.2 m, with 21 % of fail ticks above 8 m (wins 11 %). The search / navigation target z is
# clamped to VCAP_ABS_M absolute (the drone reads its own z directly); the hover is untouched and the team's
# ray floor still wins over a tall roof.
VCAP_ON = os.environ.get('SWSAR301_VCAP', '0') == '1'      # village lane 420 vs 420 = 0 (7/7): off
# --- 301 VILLAGE FAST COMMIT (user, 2026-09-17: "the person was detected but the drones didn't try hovering").
# 14 of the champion's 25 village fails had the person on the depth net and never a hover; in 11 the person gave
# >= 4 real hits (250038 9 hits at 3 s, 251189 22, 250349 73, 251590 280) and no confirmer was ever elected: the
# person shows between houses in bursts of 1-3 frames, the track decays 6 %/tick after 1 s and never reaches the
# 4-hit commit / election bar. Village wins finish at 16 s median against a ~37 s target, so a wasted check is
# cheap there. Village only: slower decay through gaps, commit at 2 consistent hits, election at > 2 hits.
VFC_ON = os.environ.get('SWSAR301_VFC', '1') == '1'
VFC_DECAY = float(os.environ.get('SWSAR301_VFC_DECAY', '0.985'))
VFC_COMMIT = float(os.environ.get('SWSAR301_VFC_COMMIT', '0.35'))    # conf = strong / 6 -> 2.1 hits
VFC_ELECT_NDET = float(os.environ.get('SWSAR301_VFC_NDET', '2.0'))


def _vfc_on():
    return VFC_ON and _MAPSW_STATE.get("map") == 'village'
VCAP_ABS_M = float(os.environ.get('SWSAR301_VCAP_M', '8.0'))
MRGB_CHECK_R = float(os.environ.get('SWSAR201_MRGB_R', '20.0'))
MRGB_CHECK_GAP = int(os.environ.get('SWSAR201_MRGB_GAP', '15'))
FOREST_CLUTTER_MIN = 0.5
FOREST_EARLY_TICK = 50
FOREST_EARLY_PROB = 0.9
FOREST_EARLY_WH = 0.05
FOREST_MIN_TICK = 250
FOREST_LATE_PROB = 0.5
FOREST_SOLO_CLUTTER = 0.95
FOREST_WH_VETO = 0.25
FOREST_SPEED = 1.0
FOREST_GROUND_TRACK_M = 2.5
BLIND_ANG_COS = float(os.environ.get('SAR_BLIND_COS', '0.71'))
BLIND_NEAR_M = float(os.environ.get('SAR_BLIND_NEAR', '3.0'))
BLIND_SPEED = float(os.environ.get('SAR_BLIND_SPEED', '0.0'))
YAWGATE_DEG = float(os.environ.get('SAR350_YAWGATE', '35'))
YAWGATE_SPEED = float(os.environ.get('SAR350_YAWSPEED', '0.35'))
YAWGATE_MAPS = tuple(x for x in os.environ.get('SAR351_MAPS', 'forest,city').split(',') if x)
YAWGATE_SPEED_CITY = float(os.environ.get('SAR350_YAWSPEED_CITY', '0.15'))
CITY_ALT_M = float(os.environ.get('SAR870_CITY_ALT', '7.0'))
VILLAGE_ALT_M = float(os.environ.get('SAR871_VILLAGE_ALT', '8.0'))
YAWGATE_MODES = tuple(x for x in os.environ.get('SAR351_MODES', 'search').split(',') if x)
V9_BRAKE_NARROW = os.environ.get('SAR700_BRAKE_NARROW', '1') == '1'
V9_RGBF_CLUTTER = os.environ.get('SAR_V9_RGBF_CLUTTER', '1') == '1'
URBAN_LOW = os.environ.get('SAR_URBAN_LOW', '') == '1'
URBAN_ALT_M = float(os.environ.get('SAR_URBAN_ALT', '6.0'))
CENTRE_REACH_M = 3.5
CENTRE_MAX_TICKS = 600
INVESTIGATE_M = 8.0
GROUND_GATE_M = float(os.environ.get('SAR_GROUND_GATE', '1.05'))
INVESTIGATE_FACTOR = 0.75
HOVER_DESCENT_SPEED = float(os.environ.get('NEWDET_HDS', '0.6'))   # NEWDET override
HOVER_DESCENT_M = 2.0
HOVZ_FREEZE_R = float(os.environ.get('SAR179_FREEZE_R', '2.5'))
STAR_MODE = os.environ.get('SAR_STAR', '') == '1'
STAR_RADIUS_M = float(os.environ.get('SAR_STAR_R', '20.0'))
_star_env = os.environ.get('SAR_STAR_ORDER', '')
STAR_ORDER_DEG = tuple((float(v) for v in _star_env.split(','))) if _star_env else (0.0, 135.0, -90.0, 45.0, 180.0, -45.0, 90.0, -135.0)
STAR_REACH_M = 3.5
HOVER_GIVEUP_BAND = (1.5, 4.5)
HOVER_GIVEUP_FALLBACK_TICKS = 1200
HOVSWEEP = os.environ.get('SAR191_SWEEP', '1') == '1'
HOVSWEEP_TRIG = int(os.environ.get('SAR191_TRIG', '150'))
MSR_ON = os.environ.get('SWSAR_MSR', '1') == '1'          # mountain hover sweep relative to the AGL ray (slot 126: head 2 m high)
MSR_HI = float(os.environ.get('SWSAR_MSR_HI', '6.0'))
HOVSWEEP_GIVEUP = int(os.environ.get('SAR191_GIVEUP', '700'))
HOVSWEEP_MPS = float(os.environ.get('SAR191_MPS', '0.4'))
HOVSWEEP_LO = float(os.environ.get('SAR191_LO', '1.4'))
HOVSWEEP_HI = float(os.environ.get('SAR191_HI', '4.6'))

def _recalibrate(p: float, t_own: float, t_target: float) -> float:
    lo = p * (t_target / t_own)
    hi = t_target + (p - t_own) * ((1.0 - t_target) / (1.0 - t_own))
    return float(np.clip(min(lo, hi), 0.0, 1.0))

def camera_axes(rpy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, p, y = (float(rpy[0]), float(rpy[1]), float(rpy[2]))
    cr, sr, cp, sp, cy, sy = (np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y))
    fwd = np.array([cy * cp, sy * cp, -sp], dtype=np.float64)
    up = np.array([cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr], dtype=np.float64)
    right = np.cross(fwd, up)
    return (fwd, up, right)

def measurement_world(u: float, v: float, cz: float, pos: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    fwd, up, right = camera_axes(rpy)
    cam = pos + fwd * CAMERA_OFFSET_M + up * CAMERA_UP_OFFSET_M
    return (cam + right * (u * HALF_TAN * cz) + up * (v * HALF_TAN * cz) + fwd * cz).astype(np.float64)

def _unit(v: np.ndarray, eps: float=1e-09) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > eps else np.zeros_like(v)

def _map_depth_2d(depth: np.ndarray) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        arr = np.reshape(arr, arr.shape[:2])
    return np.clip(arr, 0.0, 1.0)

def _map_depth_feature_values(depth: np.ndarray) -> list:
    d = _map_depth_2d(depth)
    gx = np.abs(np.diff(d, axis=1))
    gy = np.abs(np.diff(d, axis=0))
    pc = np.percentile(d, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    values = [float(d.min()), float(d.mean()), float(d.std()), float(d.max())]
    values += [float(p) for p in pc]
    values += [float((d <= 0.1).mean()), float((d <= 0.25).mean()), float((d >= 0.95).mean()), float((d >= 0.999).mean()), float((d <= 0.001).mean()), float(gx.mean() + gy.mean()), float(((gx > 0.05).mean() + (gy > 0.05).mean()) / 2.0)]
    h, w = d.shape
    tm, tmin, tmax = ([], [], [])
    for y0 in np.linspace(0, h, 5, dtype=int)[:-1]:
        y1 = int(y0 + h // 4)
        for x0 in np.linspace(0, w, 5, dtype=int)[:-1]:
            x1 = int(x0 + w // 4)
            tile = d[y0:y1, x0:x1]
            tm.append(float(tile.mean()))
            tmin.append(float(tile.min()))
            tmax.append(float(tile.max()))
    return values + tm + tmin + tmax

class _XGBMapPredictor:

    def __init__(self, model_path: Path) -> None:
        self.enabled = False
        self.num_class = 0
        self.base_score = None
        self.tree_info = []
        self.trees = []
        if not model_path.exists():
            return
        model = json.loads(model_path.read_text())
        learner = model['learner']
        params = learner['learner_model_param']
        self.num_class = int(params['num_class'])
        self.base_score = np.asarray(json.loads(params['base_score']), dtype=np.float32)
        booster = learner['gradient_booster']['model']
        self.tree_info = [int(v) for v in booster['tree_info']]
        for tree in booster['trees']:
            self.trees.append({'left': np.asarray(tree['left_children'], dtype=np.int32), 'right': np.asarray(tree['right_children'], dtype=np.int32), 'split_idx': np.asarray(tree['split_indices'], dtype=np.int32), 'split_cond': np.asarray(tree['split_conditions'], dtype=np.float32), 'default_left': np.asarray(tree['default_left'], dtype=np.int8), 'weights': np.asarray(tree['base_weights'], dtype=np.float32)})
        self.enabled = self.num_class == len(MAP_LABELS) and len(self.trees) == len(self.tree_info)

    def predict_proba(self, features: np.ndarray):
        if not self.enabled:
            return None
        x = np.asarray(features, dtype=np.float32).reshape(-1)
        base = self.base_score
        if base is None:
            return None
        margins = base.astype(np.float32).copy()
        for tree, class_idx in zip(self.trees, self.tree_info):
            node = 0
            left, right = (tree['left'], tree['right'])
            split_idx, split_cond = (tree['split_idx'], tree['split_cond'])
            default_left, weights = (tree['default_left'], tree['weights'])
            while left[node] != -1:
                fi = int(split_idx[node])
                val = x[fi] if fi < x.size else np.nan
                if np.isnan(val):
                    node = int(left[node] if default_left[node] else right[node])
                elif float(val) < float(split_cond[node]):
                    node = int(left[node])
                else:
                    node = int(right[node])
            margins[class_idx] += weights[node]
        margins -= np.max(margins)
        probs = np.exp(margins)
        denom = float(np.sum(probs))
        return (probs / denom).astype(np.float32) if denom > 1e-12 else None

class MountainClassifier:

    def __init__(self) -> None:
        p = Path(__file__).resolve().parent / XGB_NAME
        self.pred = _XGBMapPredictor(p)
        self.reset()

    def reset(self) -> None:
        self.prob_sum = np.zeros(len(MAP_LABELS), dtype=np.float64)
        self.count = 0
        self.is_mountain = False
        self.label = None
        self.latched = set()

    def update(self, tick: int, state: np.ndarray, depth: np.ndarray) -> None:
        if not self.pred.enabled or tick > XGB_MAX_TICK or tick % XGB_EVERY != 0:
            return
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        if s.size < XGB_STATE_DIM:
            sf = np.zeros(XGB_STATE_DIM, dtype=np.float32)
            sf[:s.size] = s
        else:
            sf = s[:XGB_STATE_DIM]
        feats = [float(tick), float(tick) * SIM_DT] + sf.tolist() + _map_depth_feature_values(depth)
        probs = self.pred.predict_proba(np.asarray(feats, dtype=np.float32))
        if probs is None:
            return
        self.prob_sum += probs
        self.count += 1
        avg = self.prob_sum / self.count
        idx = int(np.argmax(avg))
        self.label = MAP_LABELS[idx]
        if self.label == 'mountain' or self.label not in WAYPOINT_TERRAINS:
            bar = XGB_MTN_PROB
        else:
            bar = WAYPOINT_LATCH_PROB
        if self.count >= XGB_MIN_PRED and float(avg[idx]) >= bar:
            self.latched.add(self.label)
            if self.label == 'mountain':
                self.is_mountain = True
        if self.count >= WAREHOUSE_SIDE_LATCH_MIN_PRED and float(avg[MAP_LABELS.index('warehouse')]) >= WAREHOUSE_SIDE_LATCH_PROB:
            self.latched.add('warehouse')

    def settled(self, label: str) -> bool:
        return label in self.latched

    def prob_of(self, label: str) -> float:
        if self.count <= 0:
            return 0.0
        return float(self.prob_sum[MAP_LABELS.index(label)] / self.count)

class AltitudeTD:

    def __init__(self) -> None:
        self.reset()

    def reset(self, z0: float | None=None) -> None:
        self.z = None if z0 is None else float(z0)
        self.v = 0.0

    def __call__(self, z_target: float) -> float:
        if self.z is None:
            self.z = float(z_target)
            self.v = 0.0
            return self.z
        self.v = float(np.clip(self.v + ALT_TD_K * (z_target - self.z) - ALT_TD_D * self.v, -ALT_TD_VMAX, ALT_TD_VMAX))
        self.z += self.v
        return self.z

class VictimDetector:

    def __init__(self) -> None:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1

        def _load(name, fallback):
            p = Path(__file__).resolve().parent / name
            if not p.exists():
                p = fallback
            s = ort.InferenceSession(str(p), so, providers=['CPUExecutionProvider'])
            return (s, s.get_inputs()[0].name)

        def _load_optional(name, fallback):
            if not (Path(__file__).resolve().parent / name).exists() and (not Path(fallback).exists()):
                print(f'[sar_v21] {name} not found (looked in this directory and {fallback}) -- mountain maps will use the mixed-terrain head', file=sys.stderr, flush=True)
                return (None, None)
            return _load(name, fallback)
        self.sess, self.iname = _load(DET_NAME, DET_FALLBACK)
        # --- NEWDET: the per-map-type GoalDetector heads (depth + state -> champion 5-vector + P_w)
        self.nd_sess = {}
        if NEWDET_ON:
            for _k, _f in (('124', 'new_victim_124.onnx'), ('forest', 'new_victim_forest.onnx'), ('mountain', 'new_victim_mountain.onnx')):
                _p = Path(__file__).resolve().parent / _f
                if _p.exists():
                    _s = ort.InferenceSession(str(_p), so, providers=['CPUExecutionProvider'])
                    self.nd_sess[_k] = (_s, [i.name for i in _s.get_inputs()])
        self.nd_calls = 0
        self.nd_used = 0
        self.rgb_sess, self.rgb_iname = _load(RGB_NAME, RGB_FALLBACK)
        self.mtn_sess, self.mtn_iname = _load_optional(MTN_DET_NAME, MTN_DET_FALLBACK)
        self.mtn_rgb_sess, self.mtn_rgb_iname = _load_optional(MTN_RGB_NAME, MTN_RGB_FALLBACK)
        self.lr_sess = None
        self.lr_iname = None
        if LR_ON:
            _lp = Path(__file__).resolve().parent / LR_NAME
            if _lp.exists():
                try:
                    self.lr_sess = ort.InferenceSession(str(_lp), so, providers=['CPUExecutionProvider'])
                    self.lr_iname = self.lr_sess.get_inputs()[0].name
                except Exception:
                    self.lr_sess = None
        self.reset()

    def reset(self) -> None:
        self.vic = np.zeros(3, dtype=np.float64)
        self.height = 0.0
        self.conf = 0.0
        self.strong = 0.0
        self.last_strong_t = -10 ** 9
        self.cand = np.zeros(3, dtype=np.float64)
        self.cand_strong = 0.0
        self.t = 0
        self.last_p_depth = 0.0
        self.seen = False
        self.rgb_fresh = False
        self.rgb_prob = 0.0
        self.rgb_vec = None
        self.rgb_raw = 0.0
        self.rgb_promoted = False
        self.rgb_promotions = 0
        self.rgb_primary_ok = True
        self.rgb_primary_near_only = False
        self.rgb_src = 0
        self.mtn_head_used = False
        self.blacklist = []
        self.veto_spots = []
        self.close_hits = []                     # 2026-09-21: leaked across episodes (t restarts at 0, old hits pass the window test)
        self.rgb_no_log = []
        self.last_z = np.zeros(3, dtype=np.float64)
        self.lr_tick = 0
        self.lr_ok = False
        self.lr_owned = False
        self.lr_fired = 0
        self.lr_last_t = -10 ** 9
        self.lr_calls = 0
        self.lr_consults = 0
        self.lr_b_own = 0
        self.lr_b_ev = 0
        self.lr_b_p = 0
        self.lr_pmax = 0.0

    def location_refuted(self, z: np.ndarray) -> bool:
        if any((self.t - tk <= RGB_VETO_TTL and float(np.linalg.norm(z - p)) <= CONSIST_M for p, tk in self.veto_spots)):
            return True
        for b in list(self.blacklist):
            if float(np.linalg.norm(z - b)) <= PHANTOM_VETO_M:
                if getattr(self, "fix_world", False) and self.evidence_at(b) >= FIX_EVIDENCE:
                    self.blacklist.remove(b)                 # 302: a spot seen this well from this close is a person
                    self.fix_lifted = getattr(self, "fix_lifted", 0) + 1
                    continue
                return True
        return False

    def note_close_hit(self, z: np.ndarray, pos: np.ndarray) -> None:
        """302: remember strong hits measured from within FIX_RANGE_M (any lock state) for the fixed hover."""
        if float(np.linalg.norm(z - pos)) > float(getattr(self, "fix_range", FIX_RANGE_M)):
            return
        ch = getattr(self, "close_hits", None)
        if ch is None:
            ch = self.close_hits = []
        ch.append((self.t, float(z[0]), float(z[1]), float(z[2])))
        if len(ch) > 400:
            del ch[:len(ch) - 400]

    def evidence_at(self, xy, window: int = 500, radius: float = 3.0) -> int:
        ch = getattr(self, "close_hits", None)
        if not ch:
            return 0
        x, y = float(xy[0]), float(xy[1])
        return sum(1 for t, hx, hy, hz in ch if self.t - t <= window and math.hypot(hx - x, hy - y) <= radius)

    def fix_near(self, xy, window: int = FIX_WINDOW, min_n: int = FIX_MIN, radius: float = 3.0):
        ch = getattr(self, "close_hits", None)
        if not ch:
            return None
        x, y = float(xy[0]), float(xy[1])
        pts = [(hx, hy, hz) for t, hx, hy, hz in ch if self.t - t <= window and math.hypot(hx - x, hy - y) <= radius]
        if len(pts) < min_n:
            return None
        return np.median(np.array(pts, dtype=np.float64), axis=0)

    def _rgb_prob(self, rgb: np.ndarray, is_mountain: bool=False) -> float | None:
        rgb = np.asarray(rgb, dtype=np.float32)
        if rgb.size == 0 or float(np.mean(np.abs(rgb))) < RGB_PRESENT_EPS:
            return None
        r = rgb.reshape(256, 256, 3).reshape(RGB_RES, 2, RGB_RES, 2, 3).mean(axis=(1, 3))
        r = np.ascontiguousarray(r, dtype=np.float32)
        if is_mountain and self.mtn_rgb_sess is not None:
            out = self.mtn_rgb_sess.run(None, {self.mtn_rgb_iname: r})[0]
            self.rgb_vec = np.asarray(out, dtype=np.float64).reshape(5)
            return _recalibrate(float(_sigmoid(float(out.reshape(5)[0]))), T_MTN_RGB, RGB_CONFIRM_THRESH)
        out = self.rgb_sess.run(None, {self.rgb_iname: r})[0]
        return float(_sigmoid(float(out.reshape(5)[0])))

    def _depth_heads(self, x: np.ndarray, is_mountain: bool):
        out = self.sess.run(None, {self.iname: x})[0].reshape(5)
        p_depth = float(_sigmoid(float(out[0])))
        self.mtn_head_used = False
        if not (is_mountain and self.mtn_sess is not None):
            return (p_depth, float(out[1]), float(out[2]), float(out[3]), float(out[4]))
        mout = self.mtn_sess.run(None, {self.mtn_iname: x})[0].reshape(5)
        p_mtn = _recalibrate(float(_sigmoid(float(mout[0]))), T_MTN_DEPTH, T_OLD_DEPTH)
        if p_mtn > p_depth:
            self.mtn_head_used = True
            return (p_mtn, float(mout[1]), float(mout[2]), float(mout[3]), float(mout[4]))
        return (p_depth, float(out[1]), float(out[2]), float(out[3]), float(out[4]))

    def _nd_key(self, is_mountain):
        m = _MAPSW_STATE.get("map")
        if is_mountain or m == 'mountain':
            k = 'mountain'
        elif m == 'forest':
            k = 'forest'
        elif m in ('city', 'open', 'village'):
            k = '124'
        else:
            k = NEWDET_UNKNOWN
        if k not in self.nd_sess:
            return None
        if NEWDET_MAPS and (m or 'unknown') not in NEWDET_MAPS:
            return None
        return k

    def _depth_heads_nd(self, x, is_mountain, pos, rpy):
        """NEWDET: p / u / v / log_cz from the map-matched new head; height from the champion head."""
        k = self._nd_key(is_mountain) if self.nd_sess else None
        if k is None:
            return self._depth_heads(x, is_mountain)
        self.nd_calls += 1
        st = np.zeros(165, dtype=np.float32)
        st[0:3] = pos[0:3]
        st[3:6] = rpy[0:3]
        sess, names = self.nd_sess[k]
        out = sess.run(None, {names[0]: x, names[1]: st})[0].reshape(5)
        p_new = float(_sigmoid(float(out[0])))
        _bar = NEWDET_T_MAP.get(_MAPSW_STATE.get("map") or '', 0.0) or NEWDET_T
        if _bar > 0.0:
            p_new = _recalibrate(p_new, _bar, STRONG_P)
        u_n, v_n, lcz_n = float(out[1]), float(out[2]), float(out[3])
        if NEWDET_H_MODE == 'champ':
            p_old, u_o, v_o, lcz_o, h_old = self._depth_heads(x, is_mountain)
            if NEWDET_MIX == 'max' and p_old > p_new:
                return (p_old, u_o, v_o, lcz_o, h_old)
            h = h_old
            cz_n, cz_o = float(np.exp(lcz_n)), float(np.exp(lcz_o))
            if NEWDET_MAXP:
                p_new = max(p_new, p_old)
            if NEWDET_AND_P > 0.0 and cz_n >= NEWDET_AND_R and p_new >= STRONG_P and p_old < NEWDET_AND_P:
                p_new = min(p_new, STRONG_P - 0.05)          # far hit without the champion's agreement: keep it sub-strong
                self.nd_and_veto = getattr(self, 'nd_and_veto', 0) + 1
            if NEWDET_NEAR_M > 0.0 and cz_n < NEWDET_NEAR_M and p_old >= STRONG_P:
                u_n, v_n, lcz_n = u_o, v_o, lcz_o                # near range: the champion head places the person better
                self.nd_near = getattr(self, 'nd_near', 0) + 1
            elif NEWDET_AVG_M > 0.0 and p_old >= STRONG_P and p_new >= STRONG_P:
                z_n = measurement_world(u_n, v_n, cz_n, pos, rpy)
                z_o = measurement_world(u_o, v_o, cz_o, pos, rpy)
                if float(np.linalg.norm(z_n - z_o)) <= NEWDET_AVG_M:
                    z_m = 0.5 * (z_n + z_o)
                    fwd, up, right = camera_axes(rpy)
                    cam = pos + fwd * CAMERA_OFFSET_M + up * CAMERA_UP_OFFSET_M
                    d = z_m - cam; cz_m = max(float(d @ fwd), 1e-3)
                    u_n, v_n, lcz_n = float(d @ right) / (HALF_TAN * cz_m), float(d @ up) / (HALF_TAN * cz_m), float(np.log(cz_m))
                    self.nd_avg = getattr(self, 'nd_avg', 0) + 1
        else:
            self.mtn_head_used = False
            h = NEWDET_H_CONST
        self.nd_used += 1
        return (p_new, u_n, v_n, lcz_n, float(h))

    def _lr_allowed(self) -> bool:
        if self.lr_sess is None:
            return False
        if NEWDET_LR_OFF_MAPS and (_MAPSW_STATE.get("map") or 'unknown') in NEWDET_LR_OFF_MAPS:
            return False
        if self.lr_owned and (self.strong <= 0.0
                              or self.t - self.lr_last_t > LR_OWN_TTL):
            self.lr_owned = False
        if self.lr_owned and LR_TRACK:
            return True
        if not self.lr_ok:
            return False
        ok = (self.strong <= LR_STRONG_MAX and self.cand_strong <= LR_STRONG_MAX
              and self.t - self.last_strong_t >= LR_QUIET_TICKS)
        if not ok:
            self.lr_b_ev += 1
        return ok

    def _lr_heads(self, x: np.ndarray, pos: np.ndarray, rpy: np.ndarray):
        out = self.lr_sess.run(None, {self.lr_iname: x})[0].reshape(5)
        p_lr = float(_sigmoid(float(out[0])))
        cz = float(np.exp(min(float(out[3]), 6.0)))
        z_lr = measurement_world(float(out[1]), float(out[2]), cz, pos, rpy)
        self.lr_calls += 1
        if p_lr > self.lr_pmax:
            self.lr_pmax = p_lr
        return (p_lr, z_lr, float(out[4]))

    def update(self, depth: np.ndarray, pos: np.ndarray, rpy: np.ndarray, rgb: np.ndarray | None=None, is_mountain: bool=False):
        self.t += 1
        x = np.asarray(depth, dtype=np.float32).reshape(DEPTH_RES, DEPTH_RES, 1)
        p_depth, u, v, log_cz, h_meas = self._depth_heads_nd(x, is_mountain, pos, rpy)
        self.last_p_depth = p_depth
        cz = float(np.exp(log_cz))
        if DC_ON and is_mountain and cz >= DC_MIN_R and p_depth >= 0.3:
            # DC D1: ground the learned range on the depth image at the predicted pixel
            _px = min(DEPTH_RES - 1, max(0, int(round((u + 1.0) * 0.5 * DEPTH_RES - 0.5))))
            _py = min(DEPTH_RES - 1, max(0, int(round((1.0 - v) * 0.5 * DEPTH_RES - 0.5))))
            _win = x[max(0, _py - 2):_py + 3, max(0, _px - 2):_px + 3, 0]
            _d_img = float(np.median(_win)) * (DEPTH_MAX_M - DEPTH_MIN_M) + DEPTH_MIN_M
            if _d_img < DEPTH_MAX_M - 0.5 and (_d_img + 0.6) < cz <= (_d_img + 4.0):
                cz = _d_img + 0.25
                self.dc_clamped = getattr(self, "dc_clamped", 0) + 1
        z_world = measurement_world(u, v, cz, pos, rpy)
        self.last_z = z_world.copy()
        if getattr(self, "fix_world", False) and p_depth >= STRONG_P:
            self.note_close_hit(z_world, pos)
        _lr_bar = getattr(self, 'strong_p_override', 0.0) or STRONG_P
        if p_depth >= _lr_bar:
            self.lr_owned = False
            self.lr_b_own += 1
        elif self._lr_allowed():
            self.lr_tick += 1
            if self.lr_tick % LR_STRIDE == 0:
                self.lr_consults += 1
                p_lr, z_lr, h_lr = self._lr_heads(x, pos, rpy)
                if p_lr >= LR_P_BAR and h_lr <= LR_H_MAX:
                    p_depth = p_lr
                    z_world = np.asarray(z_lr, dtype=np.float64)
                    h_meas = h_lr
                    self.last_p_depth = p_depth
                    self.last_z = z_world.copy()
                    self.lr_owned = True
                    self.lr_fired += 1
                    self.lr_last_t = self.t
                else:
                    self.lr_b_p += 1
        self.rgb_fresh = False
        p_rgb = None
        if rgb is not None:
            p_rgb = self._rgb_prob(rgb, is_mountain)
            if p_rgb is not None:
                self.rgb_fresh = True
                self.rgb_prob = p_rgb
        colour_blind = RANGE_VETO and float(np.linalg.norm(z_world - pos)) > COLOUR_TRUST_M
        _rgb_vetoed_now = False
        if p_rgb is not None and p_depth >= RGB_TARGET_TRIGGER:
            if p_rgb < RGB_CONFIRM_THRESH:
                if not colour_blind:
                    if RGBV2_ON:
                        _log = getattr(self, "rgb_no_log", None)
                        if _log is None:
                            _log = self.rgb_no_log = []
                        _n_no = 1 + sum(1 for _p, _tk in _log if self.t - _tk <= RGBV2_WINDOW
                                        and float(np.linalg.norm(z_world - _p)) <= CONSIST_M)
                        _log.append((z_world.copy(), self.t))
                        if len(_log) > 64:
                            del _log[:len(_log) - 64]
                        # RGBV3 R1: the single-'no' fast path fires only when the depth TRACK is weak, not when
                        # one frame's p is below 0.85 (village 364 / b2-315: 4-9 strong hits on the person, one colour 'no' erased them)
                        _trk = max(float(self.strong), float(self.cand_strong))
                        _pmin = RGBV2_STRONG_P
                        if RGBV2_P_VILLAGE >= 0.0 and _MAPSW_STATE.get('map') == 'village':
                            _pmin = RGBV2_P_VILLAGE     # village only; every other map keeps 0.85
                        if _n_no >= RGBV2_NO_NEEDED or (p_depth < _pmin and _trk < RGBV3_TRACK_MIN):
                            self.veto_spots.append((z_world.copy(), self.t))
                            _rgb_vetoed_now = True
                            self.rgbv2_vetoes = getattr(self, "rgbv2_vetoes", 0) + 1
                        else:
                            self.rgbv2_deferred = getattr(self, "rgbv2_deferred", 0) + 1
                    else:
                        self.veto_spots.append((z_world.copy(), self.t))
                        _rgb_vetoed_now = True
            else:
                self.veto_spots = [(p, tk) for p, tk in self.veto_spots if float(np.linalg.norm(z_world - p)) > CONSIST_M]
                if RGBV2_ON and getattr(self, "rgb_no_log", None):
                    self.rgb_no_log = [(p, tk) for p, tk in self.rgb_no_log if float(np.linalg.norm(z_world - p)) > CONSIST_M]
        if len(self.veto_spots) > 64:
            self.veto_spots = self.veto_spots[-64:]

        strong_bar = getattr(self, 'strong_p_override', 0.0) or STRONG_P
        is_strong = p_depth >= strong_bar
        if is_strong and p_rgb is not None and (p_rgb < RGB_CONFIRM_THRESH) and (not colour_blind):
            if (not RGBV2_ON) or _rgb_vetoed_now:
                is_strong = False
        self.rgb_promoted = False
        _rgb_try = False
        if is_mountain and p_rgb is not None and self.rgb_vec is not None:
            _rv = self.rgb_vec
            _rgb_raw = float(_sigmoid(float(_rv[0])))
            self.rgb_raw = _rgb_raw
            _ur, _vr, _hr = float(_rv[1]), float(_rv[2]), float(_rv[4])
            _duv = float(np.hypot(u - _ur, v - _vr))
            if MTN_RGB_OUTER_RAW and is_strong and p_rgb >= RGB_CONFIRM_THRESH and _duv <= MTN_RGB_UV_TOL:
                self.last_p_depth = max(float(self.last_p_depth), _rgb_raw)
            if (MTN_RGB_PRIMARY and (not is_strong) and p_rgb >= RGB_CONFIRM_THRESH and _hr <= MTN_RGB_PROMOTE_H_MAX
                    and self.rgb_primary_ok):
                if _duv <= MTN_RGB_UV_TOL:
                    _zr, _src = z_world, 1
                else:
                    _px = min(DEPTH_RES - 1, max(0, int(round((_ur + 1.0) * 0.5 * DEPTH_RES - 0.5))))
                    _py = min(DEPTH_RES - 1, max(0, int(round((1.0 - _vr) * 0.5 * DEPTH_RES - 0.5))))
                    _win = x[max(0, _py - MTN_RGB_IMG_WIN):_py + MTN_RGB_IMG_WIN + 1, max(0, _px - MTN_RGB_IMG_WIN):_px + MTN_RGB_IMG_WIN + 1, 0]
                    _cz_i = float(np.median(_win)) * (DEPTH_MAX_M - DEPTH_MIN_M) + DEPTH_MIN_M
                    _zr, _src = (measurement_world(_ur, _vr, _cz_i, pos, rpy), 2) if _cz_i <= MTN_RGB_IMG_MAX_M else (None, 0)
                if (_zr is not None and float(np.linalg.norm(_zr - pos)) <= MTN_RGB_RANGE_MAX_M
                        and ((not self.rgb_primary_near_only) or float(np.linalg.norm(_zr - self.vic)) <= CONSIST_M)):
                    z_world = _zr
                    h_meas = _hr
                    self.rgb_src = _src
                    is_strong = True
                    _rgb_try = True
                    self.last_z = z_world.copy()
                    self.last_p_depth = max(float(self.last_p_depth), _rgb_raw if MTN_RGB_OUTER_RAW else float(p_rgb))
                    self.veto_spots = [(p, tk) for p, tk in self.veto_spots if float(np.linalg.norm(z_world - p)) > CONSIST_M]
        if is_strong and self.location_refuted(z_world):
            is_strong = False
            if (RGBV3_HOLD and self.strong >= N_LATCH
                    and float(np.linalg.norm(z_world - self.vic)) <= CONSIST_M):
                # RGBV3 R2: a colour 'no' on a spot this track already latched holds the track (no decay,
                # no growth) -- the lock stays where the depth evidence put it
                self.last_strong_t = self.t
                self.rgbv3_held = getattr(self, "rgbv3_held", 0) + 1
        if _rgb_try and is_strong:
            self.rgb_promoted = True
            self.rgb_promotions += 1
        d_inc = float(np.linalg.norm(z_world - self.vic))
        d_cand = float(np.linalg.norm(z_world - self.cand))
        _edge = False
        if EDGE_GUARD and self.strong >= EDGE_STRONG_MIN and d_inc <= CONSIST_M:
            _hzd = float(np.hypot(z_world[0] - pos[0], z_world[1] - pos[1]))
            _el = math.degrees(math.atan2(float(z_world[2] - pos[2]), max(_hzd, 0.1)))
            _step = float(np.hypot(z_world[0] - self.vic[0], z_world[1] - self.vic[1]))
            if _el <= -EDGE_EL_DEG and (_step > EDGE_STEP_M or float(z_world[2] - self.vic[2]) < -EDGE_DROP_M):
                # EG: the person is leaving the bottom of the frame -- what is left in view is the lower body /
                # the ground at the feet, 2-3 m off and 1-2 m low: keep the saturated track, do not re-centre
                _edge = True
                self.edge_hits = getattr(self, 'edge_hits', 0) + 1
        if _edge:
            if is_strong:
                self.strong = min(N_CAP, self.strong + 1.0)
                self.last_strong_t = self.t
        elif is_strong:
            if self.strong > 0.0 and d_inc <= CONSIST_M:
                self.strong = min(N_CAP, self.strong + 1.0)
                self.last_strong_t = self.t
                self.vic += (z_world - self.vic) * STRONG_GAIN
                self.height += (h_meas - self.height) * STRONG_GAIN
            elif self.cand_strong > 0.0 and d_cand <= CONSIST_M:
                self.cand_strong = min(N_CAP, self.cand_strong + 1.0)
                self.cand += (z_world - self.cand) * STRONG_GAIN
                if self.cand_strong > self.strong:
                    self.vic, self.strong = (self.cand.copy(), self.cand_strong)
                    self.height = h_meas
                    self.last_strong_t = self.t
                    self.cand_strong = 0.0
            elif self.strong == 0.0:
                self.vic, self.strong, self.last_strong_t = (z_world.copy(), 1.0, self.t)
                self.height = h_meas
            else:
                self.cand, self.cand_strong = (z_world.copy(), 1.0)
        elif self.strong > 0.0 and d_inc <= CONSIST_M:
            self.vic += (z_world - self.vic) * TRACK_GAIN
            self.height += (h_meas - self.height) * TRACK_GAIN
        if self.t - self.last_strong_t > STALE_TICKS:
            _dec = VFC_DECAY if (_vfc_on() and not is_mountain) else STALE_DECAY
            self.strong *= _dec
            self.cand_strong *= _dec
            if MJ_FLOOR > 0.0:
                if self.strong < MJ_FLOOR:
                    self.strong = 0.0
                if self.cand_strong < MJ_FLOOR:
                    self.cand_strong = 0.0
        self.conf = float(np.clip(self.strong / (NEWDET_N_LATCH if (NEWDET_N_LATCH > 0.0 and self.nd_sess) else N_LATCH), 0.0, 1.0))
        self.seen = bool(is_strong and self.strong > 0.0 and (float(np.linalg.norm(z_world - self.vic)) <= CONSIST_M))
        if MH_ON and is_mountain:
            head_z = self.vic[2] + 0.5 * float(np.clip(self.height, MH_HMIN, MH_HMAX))
        else:
            head_z = self.vic[2] + 0.5 * self.height
        return (self.conf, self.vic.copy(), head_z)

class ObstacleAvoider:

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.mem_bias = np.zeros(3, dtype=np.float64)
        self.mem_slow = 1.0

    def sectors(self, depth: np.ndarray) -> tuple[float, float, float]:
        d = np.asarray(depth, dtype=np.float32).reshape(DEPTH_RES, DEPTH_RES)
        d = d * (DEPTH_MAX_M - DEPTH_MIN_M) + DEPTH_MIN_M
        if getattr(self, 'wide', False):

            band = d[96:160, :]
            left = float(band[:, 0:96].min())
            centre = float(band[:, 80:176].min())
            right = float(band[:, 160:256].min())
            return (left, centre, right)
        band = d[96:160, :]
        left = float(band[:, 32:112].min())
        centre = float(band[:, 96:160].min())
        right = float(band[:, 144:224].min())
        return (left, centre, right)

    def _narrow_centre(self, depth: np.ndarray) -> float:
        d = np.asarray(depth, dtype=np.float32).reshape(DEPTH_RES, DEPTH_RES)
        d = d * (DEPTH_MAX_M - DEPTH_MIN_M) + DEPTH_MIN_M
        return float(d[96:160, 96:160].min())

    def steer(self, direction: np.ndarray, speed: float, depth: np.ndarray, rpy: np.ndarray) -> tuple[np.ndarray, float]:
        left, centre, right = self.sectors(depth)
        nearest = min(left, centre, right)
        brake_ref = nearest
        if getattr(self, 'wide', False):
            brake_ref = (self._narrow_centre(depth)
                         if (V9_BRAKE_NARROW and _MAPSW_STATE.get("map") == 'forest')
                         else min(self._narrow_centre(depth), left, right))
        self.mem_bias *= AVOID_MEMORY_DECAY
        self.mem_slow = 1.0 - (1.0 - self.mem_slow) * AVOID_MEMORY_DECAY
        if nearest <= AVOID_LOOKAHEAD_M:
            fwd, _, right_ax = camera_axes(rpy)
            urgency = float(np.clip((AVOID_LOOKAHEAD_M - nearest) / max(AVOID_LOOKAHEAD_M - AVOID_SAFETY_M, 0.001), 0.0, 1.0))
            push = (right_ax if left < right else -right_ax) * (AVOID_STRENGTH * urgency)
            if centre < AVOID_SAFETY_M:
                push = push - fwd * urgency
            if float(push @ push) > float(self.mem_bias @ self.mem_bias):
                self.mem_bias = push
            self.mem_slow = min(self.mem_slow, float(np.clip(brake_ref / AVOID_SLOW_M, 0.25, 1.0)))
        if float(self.mem_bias @ self.mem_bias) < 1e-06:
            return (direction, speed)
        new_dir = _unit(direction + self.mem_bias)
        return (new_dir, speed * self.mem_slow)

class TrackingDifferentiator:
    ALPHA = 0.3
    BETA = 0.2
    A_MAX = 0.02

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.v = np.zeros(3)
        self.a = np.zeros(3)

    def __call__(self, direction: np.ndarray, speed: float) -> tuple[np.ndarray, float]:
        v_sp = _unit(direction) * float(np.clip(speed, 0.0, 1.0))
        self.a = np.clip(self.ALPHA * self.a + self.BETA * (v_sp - self.v), -self.A_MAX, self.A_MAX)
        self.v = self.v + self.a
        s = float(np.linalg.norm(self.v))
        if s < 1e-05:
            return (np.zeros(3), 0.0)
        return (self.v / s, min(s, 1.0))

class _V21Controller:

    def __init__(self) -> None:
        self.detector = VictimDetector()
        self.avoider = ObstacleAvoider()
        self.smoother = TrackingDifferentiator()
        self.mtn = MountainClassifier()
        self.alt_td = AltitudeTD()
        self.reset()

    def reset(self) -> None:
        self.detector.reset()
        self.avoider.reset()
        self.smoother.reset()
        self.mtn.reset()
        self.alt_td.reset()
        self.wp_list = []
        self.wp_idx = 0
        self.star_list = []
        self.star_idx = 0
        self.centre_visited = False
        self._hov_z_lock = None
        self._sweep_t0 = None
        self.search_t0 = None
        self.spiral_t0 = 0
        self.spiral_legacy = None
        self.spiral_theta = 0.0
        self.spiral_r = SPIRAL_R0_M
        self.mode = 'initial'
        self.tick = 0
        self.clutter = 0.0
        self.rgb_requests = 0
        self.last_rgb_req = -10 ** 9
        self.rgb_confirm_tick = -10 ** 9
        self.rgb_hit_tick = -10 ** 9
        self.locked = False
        self.locked_vic = np.zeros(3)
        self.locked_head_z = 0.0
        self.frozen = False
        self.lost_ticks = 0
        self.hover_stable_ticks = 0
        self.hover_total_ticks = 0
        self.forest_low = False
        self.forest_latch_tick = 0
        self._rgbf_hits = 0
        self._rgbf_evals = 0
        self.ground_min = None
        self._last_agl = None
        self.mq_view = 0            # ticks the tracked spot sat in the camera cone during this approach
        self.mq_hits = 0            # ticks the net re-fired on it while in the cone
        self.mq_hold_left = 0       # blind ticks the track may still survive (patience budget)
        self.mq_refuted = []        # [(xyz, tick)] spots refuted by a low hit rate
        self.mq_stats = {"mq_refute": 0, "mq_hold": 0, "mq_protect": 0, "msw_giveup": 0, "mq_approach": 0}
        self._msp_turned = None         # radians turned in the current look-around, None = not turning
        self._fix_hits = []             # (tick, x, y, z) of close-range hits on the current lock
        self._fix_pt = None             # frozen hover point [x, y, z_centre] from the close hits
        self._fix_evidence = 0          # accurate close hits collected on the current lock
        self._msp_yaw = 0.0
        self._msp_last_end = -10 ** 9
        self._mq_spot = None
        self._tks_pnear = None          # TKS: posterior mass within TKS_R_NEAR of this drone's start (team sets it)
        self._tks_yaw0 = None
        self._tks_t0 = None
        self._tks_dir = 1.0
        self._tks_done = False
        self._tks_deg = 0.0
        self._tks_phase = 'sweep'
        self._tks_ck_t = None
        self._mcv_until = -1
        self._mcv_end = -10 ** 6
        self._mcv_pt = None
        self._tks_ck_deg = 0.0
        self._tks_ver_t0 = 0
        self._tks_sweep_t0 = None
        self._tks_prev_yaw = None
        self._tks_pad_z = None
        self._tk_yaw0 = None            # take-off heading the 303 sweep is measured from
        self._tk_t0 = None              # tick the sweep window opened
        self._tk_done = False           # sweep over (window, stop rule, or turn started)
        self._tkb_forced = False        # a variant-2 forced turn is in progress

    def _mq_step(self, pos, rpy, vic, conf):
        """Track-quality bookkeeping for one approach on mountain. Returns True when the track was refuted."""
        det = self.detector
        if self.mode != 'navigation' or self.frozen or conf <= 0.0:
            if self.mode != 'navigation':
                self._mq_spot = None
            return False
        if self._mq_spot is None or math.hypot(vic[0] - self._mq_spot[0], vic[1] - self._mq_spot[1], vic[2] - self._mq_spot[2]) > CONSIST_M:
            self._mq_spot = vic.copy()
            self.mq_view = 0
            self.mq_hits = 0
            self.mq_hold_left = MQ_HOLD_TICKS
            self.mq_stats["mq_approach"] += 1
        dx, dy, dzz = float(vic[0] - pos[0]), float(vic[1] - pos[1]), float(vic[2] - pos[2])
        horiz = math.hypot(dx, dy)
        rng = math.hypot(horiz, dzz)
        if 2.0 <= rng <= MQ_RANGE_M and horiz > 1e-3:
            rel = abs((math.atan2(dy, dx) - float(rpy[2]) + math.pi) % (2.0 * math.pi) - math.pi)
            dep = math.degrees(math.atan2(-dzz, horiz))
            if rel <= 0.7854 and dep <= MQ_DEP_DEG:
                self.mq_view += 1
                if det.seen:
                    self.mq_hits += 1
        if MQ_LOW > 0.0 and self.mq_view >= MQ_VIEW_MIN and self.mq_hits < MQ_LOW * self.mq_view:
            self.mq_refuted.append((vic.copy(), self.tick))
            if len(self.mq_refuted) > 16:
                self.mq_refuted.pop(0)
            self.mq_stats["mq_refute"] += 1
            det.vic = np.zeros(3, dtype=np.float64)
            det.cand = np.zeros(3, dtype=np.float64)
            det.strong = 0.0
            det.cand_strong = 0.0
            det.conf = 0.0
            det.seen = False
            self.locked = False
            self.frozen = False
            self.lost_ticks = 0
            self.mode = 'search'
            self._mq_spot = None
            return True
        return False

    def _msp_step(self, yaw, depth, rpy, speed, clear_m=None, agl_min=None, abort_m=None, period=None):
        """Look-around while flying: returns the yaw command (normalised, /pi) for this tick."""
        st = self.mq_stats
        clear_m = MSP_CLEAR_M if clear_m is None else clear_m
        agl_min = MSP_AGL_MIN if agl_min is None else agl_min
        abort_m = MTN_CLIMB_LOOKAHEAD_M if abort_m is None else abort_m
        period = MSP_PERIOD if period is None else period
        searching = (self.mode == 'search' and (not self.locked) and (not self.frozen)
                     and self.detector.strong <= 0.0)
        if self._msp_turned is not None:
            stop = None
            if not searching:
                stop = "msp_stop_mode"
            elif self.detector.last_p_depth >= STRONG_P:
                stop = "msp_stop_seen"
            else:
                try:
                    if min(self.avoider.sectors(depth)) < abort_m:
                        stop = "msp_stop_terrain"
                except Exception:
                    stop = "msp_stop_terrain"
            if stop is None and self._msp_turned >= 2.0 * np.pi:
                stop = "msp_done"
            if stop is not None:
                st[stop] = st.get(stop, 0) + 1
                self._msp_turned = None
                self._msp_last_end = self.tick
                return yaw
            step = np.radians(MSP_RATE) / CTRL_HZ
            self._msp_turned += step
            self._msp_yaw = (self._msp_yaw + step + np.pi) % (2.0 * np.pi) - np.pi
            st["msp_ticks"] = st.get("msp_ticks", 0) + 1
            return float(self._msp_yaw / np.pi)
        if (searching and self.tick >= MSP_T0 and self.tick - self._msp_last_end >= period
                and speed >= MSP_MIN_SPEED and self._last_agl is not None and self._last_agl >= agl_min):
            try:
                clear = min(self.avoider.sectors(depth)) >= clear_m
            except Exception:
                clear = False
            if clear:
                self._msp_turned = 0.0
                self._msp_yaw = float(rpy[2])
                st["msp_turns"] = st.get("msp_turns", 0) + 1
        return yaw

    def _mq_hold(self, strong_before):
        """Patience: a high-quality track keeps its strength through a blind gap of up to MQ_HOLD_TICKS."""
        det = self.detector
        if self.mode != 'navigation' or self.frozen or det.seen or self.mq_hold_left <= 0:
            return
        if self.mq_view < MQ_HOLD_VIEW or self.mq_hits < MQ_HIGH * self.mq_view:
            return
        if det.strong < strong_before:
            det.strong = strong_before
            det.conf = float(np.clip(det.strong / N_LATCH, 0.0, 1.0))
            self.mq_hold_left -= 1
            self.mq_stats["mq_hold"] += 1

    def _mq_spot_refuted(self, z):
        if not self.mq_refuted:
            return False
        return any((self.tick - tk <= MQ_REFUTE_TTL and float(np.linalg.norm(z - p)) <= CONSIST_M) for p, tk in self.mq_refuted)

    def _clutter_frac(self) -> float:
        return float(np.clip((self.clutter - CLUTTER_EDGE_LO) / (CLUTTER_EDGE_HI - CLUTTER_EDGE_LO), 0.0, 1.0))

    def _update_clutter(self, depth: np.ndarray) -> None:
        d = np.asarray(depth, dtype=np.float32).reshape(DEPTH_RES, DEPTH_RES)
        gx = np.abs(np.diff(d[64:192, :], axis=1))
        edge_frac = float((gx > CLUTTER_EDGE_THRESH).mean())
        self.clutter = CLUTTER_EMA * self.clutter + (1.0 - CLUTTER_EMA) * edge_frac

    def _search_alt(self) -> float:
        if TK_ON and _MAPSW_STATE.get("map") == 'forest' and not self.mtn.is_mountain:
            return TK_H
        if CITY_ALT_M > 0.0 and _MAPSW_STATE.get("map") == 'city' and not self.mtn.is_mountain:
            return CITY_ALT_M + CLUTTER_ALT_BONUS_M * self._clutter_frac()
        if VILLAGE_ALT_M > 0.0 and _MAPSW_STATE.get("map") == 'village' and not self.mtn.is_mountain:
            return VILLAGE_ALT_M + CLUTTER_ALT_BONUS_M * self._clutter_frac()
        if self.mtn.settled('warehouse') and not self.mtn.is_mountain:

            return float(os.environ.get('HYB_WH_ALT', str(SEARCH_ALT_M)))\
                + CLUTTER_ALT_BONUS_M * self._clutter_frac()
        return SEARCH_ALT_M + CLUTTER_ALT_BONUS_M * self._clutter_frac()

    def _commit_ok(self, conf: float, near: bool=False) -> bool:
        if WAYPOINT_FLAT_COMMIT and self._use_waypoint_search() and (not self.mtn.is_mountain):

            bar = float(os.environ.get('HYB_WH_COMMIT', '0.6'))

            late = os.environ.get('HYB_WH_LATEBAR', '')
            if late:
                lt, lbar = (float(x) for x in late.split(','))
                if self.tick >= lt * CTRL_HZ:
                    bar = min(bar, lbar)
        elif self.forest_low:
            bar = COMMIT_CONF
        elif self.mtn.is_mountain:
            bar = COMMIT_CONF_MTN + COMMIT_CLUTTER_BONUS * self._clutter_frac()
        else:
            bar = COMMIT_CONF + COMMIT_CLUTTER_BONUS * self._clutter_frac()
            if _vfc_on() and not self.mtn.is_mountain:
                bar = VFC_COMMIT
                self.mq_stats["vfc_bar"] = self.mq_stats.get("vfc_bar", 0) + 1
        if near:
            bar *= INVESTIGATE_FACTOR
        return conf >= bar

    def _true_ground(self, ground_z: float) -> float:
        gmin = self.ground_min
        if gmin is None or ground_z - gmin < FOREST_GROUND_TRACK_M:
            return ground_z
        return gmin

    def _target_z(self, ground_z: float) -> float:
        if self.mtn.is_mountain:
            return self.alt_td(ground_z + (MTK_SEARCH_H if (MTK_ON and MTK_SEARCH_H > 0.0) else MOUNTAIN_SEARCH_ALT_M))
        if self.forest_low:
            return self.alt_td(self._true_ground(ground_z) + _forest_alt())
        if URBAN_LOW:
            return self.alt_td(self._true_ground(ground_z) + URBAN_ALT_M)
        z = ground_z + self._search_alt()
        if VCAP_ON and _MAPSW_STATE.get("map") == 'village' and z > VCAP_ABS_M:
            self.mq_stats["vcap"] = self.mq_stats.get("vcap", 0) + 1
            z = VCAP_ABS_M
        return z

    def _build_waypoints(self, centre_xy: np.ndarray) -> None:
        if self.mtn.is_mountain:
            radii, per_ring = (MTN_WP_RING_RADII, MTN_WP_PER_RING)
        else:
            radii, per_ring = (HIGH_WP_RING_RADII, HIGH_WP_PER_RING)
        wps = [np.asarray(centre_xy, dtype=np.float64).copy()]
        for j, r in enumerate(radii):
            phase = np.pi / per_ring * j
            for k in range(per_ring):
                ang = 2.0 * np.pi * k / per_ring + phase
                wps.append(centre_xy + r * np.array([np.cos(ang), np.sin(ang)]))
        self.wp_list = wps
        self.wp_idx = 0

    def _initial_policy(self, pos, ground_z):
        if MTK_ON and self.mtn.is_mountain:
            target = np.array([pos[0], pos[1], ground_z + MTK_H])
            return (target, MTK_CLIMB_SPEED)
        if self.forest_low:
            target = np.array([pos[0], pos[1], self._true_ground(ground_z) + _forest_alt()])
            return (target, 0.6)
        if URBAN_LOW and (not self.mtn.is_mountain):
            target = np.array([pos[0], pos[1], self._true_ground(ground_z) + URBAN_ALT_M])
            return (target, 0.6)
        _z0 = ground_z + self._search_alt()
        if VCAP_ON and _MAPSW_STATE.get("map") == 'village' and _z0 > VCAP_ABS_M:
            _z0 = VCAP_ABS_M                                   # 301: the take-off climb honours the village cap too
        target = np.array([pos[0], pos[1], _z0])
        return (target, 0.6)

    def _use_waypoint_search(self) -> bool:
        if self.mtn.is_mountain:
            return True
        return any((self.mtn.settled(t) for t in WAYPOINT_TERRAINS))

    def _search_policy(self, pos, clue, ground_z):
        if STAR_MODE:
            if not self.star_list:
                base_dir = _unit(np.array([clue[0], clue[1], 0.0]))[0:2]
                self.star_list = [np.zeros(2)]
                for deg in STAR_ORDER_DEG:
                    a = np.deg2rad(deg)
                    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
                    self.star_list.append(rot @ base_dir * STAR_RADIUS_M)
            centre_xy = pos[0:2] + clue
            wp = centre_xy + self.star_list[self.star_idx]
            if float(np.linalg.norm(wp - pos[0:2])) < STAR_REACH_M:
                self.star_idx = (self.star_idx + 1) % len(self.star_list)
                wp = centre_xy + self.star_list[self.star_idx]
            speed = FOREST_SPEED if self.forest_low else 1.0
            return (np.array([wp[0], wp[1], self._target_z(ground_z)]), speed)
        if self._use_waypoint_search():
            if not self.wp_list:

                if not self.mtn.is_mountain and self.mtn.settled('warehouse')\
                        and os.environ.get('HYB_WH_ORIGIN', '1') == '1':
                    rings = tuple(float(x) for x in os.environ.get('HYB_WH_RINGS', '4,10,16').split(','))
                    wps = [np.zeros(2, dtype=np.float64)]
                    for j, r in enumerate(rings):
                        phase = np.pi / HIGH_WP_PER_RING * j
                        for k in range(HIGH_WP_PER_RING):
                            a = 2.0 * np.pi * k / HIGH_WP_PER_RING + phase
                            wps.append(r * np.array([np.cos(a), np.sin(a)]))

                    sc = pos[0:2] + clue
                    kept = [w for w in wps if float(np.linalg.norm(w - sc)) <= 28.0]
                    self.wp_list = kept if kept else wps
                    self.wp_idx = 0
                else:
                    self._build_waypoints(pos[0:2] + clue)
            reach = MTN_WP_REACH_M if self.mtn.is_mountain else HIGH_WP_REACH_M
            wp = self.wp_list[self.wp_idx]
            if float(np.linalg.norm(wp - pos[0:2])) < reach:
                self.wp_idx = (self.wp_idx + 1) % len(self.wp_list)
                wp = self.wp_list[self.wp_idx]
            return (np.array([wp[0], wp[1], self._target_z(ground_z)]), 1.0)
        centre_xy = pos[0:2] + clue
        speed = FOREST_SPEED if self.forest_low else 1.0
        if not self.centre_visited and (not self.forest_low):
            if self.search_t0 is None:
                self.search_t0 = self.tick
                if self._clutter_frac() >= FOREST_CLUTTER_MIN:
                    self.centre_visited = True
            if not self.centre_visited:
                if float(np.linalg.norm(clue)) < CENTRE_REACH_M or self.tick - self.search_t0 >= CENTRE_MAX_TICKS:
                    self.centre_visited = True
                    self.spiral_t0 = self.tick
                else:
                    return (np.array([centre_xy[0], centre_xy[1], self._target_z(ground_z)]), speed)
        if self.spiral_legacy is None:
            self.spiral_legacy = bool(not ARC_SPIRAL or self.forest_low or self._clutter_frac() >= FOREST_CLUTTER_MIN)
        if self.spiral_legacy:
            ts = self.tick - self.spiral_t0
            rad = SPIRAL_R0_M + (SPIRAL_RMAX_M - SPIRAL_R0_M) * min(ts / SPIRAL_GROW_TICKS, 1.0)
            if self.forest_low:
                t0f = min(max(self.forest_latch_tick - self.spiral_t0, 0), ts)
                ang = t0f * SPIRAL_ANGULAR + (ts - t0f) * SPIRAL_ANGULAR * FOREST_SPEED
            else:
                ang = ts * SPIRAL_ANGULAR
        else:
            v = SPIRAL_SPEED_MPS * (FOREST_SPEED if self.forest_low else 1.0)
            self.spiral_r = min(SPIRAL_R0_M + SPIRAL_PITCH_M * self.spiral_theta / (2.0 * np.pi), SPIRAL_ARC_RMAX_M)
            self.spiral_theta += v / CTRL_HZ / max(self.spiral_r, SPIRAL_R0_M)
            rad, ang = (self.spiral_r, self.spiral_theta)
        target = np.array([centre_xy[0] + rad * np.cos(ang), centre_xy[1] + rad * np.sin(ang), self._target_z(ground_z)])
        return (target, speed)

    def _navigation_policy(self, pos, vic, ground_z):
        target = np.array([vic[0], vic[1], self._target_z(ground_z)])
        if FA_ON and _forest_here(self) and (not self.frozen):
            _horiz = float(np.hypot(vic[0] - pos[0], vic[1] - pos[1]))
            _head = float(self.locked_head_z) if self.locked else getattr(self, '_nav_head_z', None)
            if _horiz <= FA_R and _head is not None and np.isfinite(_head):
                # 304: approach at the candidate's height so the person stays in the frame while closing
                target[2] = max(_head + HOVER_ABOVE_TOP_M + HOVER_SETTLE_MARGIN_M, ground_z + FA_FLOOR)
                self.mq_stats["fa_low"] = self.mq_stats.get("fa_low", 0) + 1
                _el = math.degrees(math.atan2(_head - pos[2], max(_horiz, 0.1)))
                if _el < -FA_EL_DEG and pos[2] > target[2] + 0.3:
                    # the head is near the bottom of the 90-degree frame: descend first, close afterwards
                    target[0] = pos[0] + 0.1 * (vic[0] - pos[0])
                    target[1] = pos[1] + 0.1 * (vic[1] - pos[1])
                    self.mq_stats["fa_hold"] = self.mq_stats.get("fa_hold", 0) + 1
        if MA_ON and self.mtn.is_mountain and self.locked and (not self.frozen):
            horiz = float(np.hypot(vic[0] - pos[0], vic[1] - pos[1]))
            if horiz <= MA_R:
                hover_z = float(self.locked_head_z) + HOVER_ABOVE_TOP_M + HOVER_SETTLE_MARGIN_M
                target[2] = max(hover_z, ground_z + 2.5)
                self.mq_stats["ma_low"] = self.mq_stats.get("ma_low", 0) + 1
        return (target, FOREST_SPEED if self.forest_low else 1.0)

    def _mountain_avoid(self, direction, speed, depth, rpy):
        left, centre, right = self.avoider.sectors(depth)
        nearest = min(left, centre, right)
        if nearest >= MTN_CLIMB_LOOKAHEAD_M:
            return (direction, speed)
        urg = float(np.clip((MTN_CLIMB_LOOKAHEAD_M - nearest) / MTN_CLIMB_LOOKAHEAD_M, 0.0, 1.0))
        _, _, right_ax = camera_axes(rpy)
        d = direction + np.array([0.0, 0.0, MTN_CLIMB_GAIN * urg])
        d = d + (right_ax if left < right else -right_ax) * (MTN_STEER_DAMP * urg)
        return (_unit(d), max(speed, 0.4))

    def _hover_policy(self, vic, head_z, pos):
        if getattr(self, '_sweep_t0', None) is not None:
            if MSW_ON and self.mtn.is_mountain:
                # 201: one V-cycle estimate+2.5 -> estimate+MSW_LO -> estimate+2.5 at MSW_MPS; the AGL floor
                # (team guard, 2.5 m over the downward ray) still bounds the descent
                ph = (self.tick - self._sweep_t0) / CTRL_HZ * MSW_MPS
                _d0 = MSW_TOP - MSW_LO
                if ph <= _d0:
                    dz = MSW_TOP - ph
                elif ph <= _d0 + (MSW_HI - MSW_LO):
                    dz = MSW_LO + (ph - _d0)
                else:
                    dz = MSW_HI
                target = np.array([vic[0], vic[1], head_z + dz])
                self._dbg_hover_tz = float(target[2])
                return (target, MSW_MPS / SPEED_LIMIT)
            ph = (self.tick - self._sweep_t0) / CTRL_HZ * HOVSWEEP_MPS
            _d0 = 2.5 - HOVSWEEP_LO
            _base, _hi = head_z, HOVSWEEP_HI
            if MSR_ON and self.mtn.is_mountain and self._last_agl is not None and np.isfinite(self._last_agl):
                # sweep relative to the downward ray (the body or the ground beside it), not the far-range
                # head estimate: covers the 2-4 m band for a 0.3-1.9 m tall victim either way
                _base, _hi = float(pos[2]) - float(self._last_agl), MSR_HI
                self.mq_stats['msr_ticks'] = self.mq_stats.get('msr_ticks', 0) + 1
            if ph <= _d0:
                dz = 2.5 - ph
            elif ph <= _d0 + (_hi - HOVSWEEP_LO):
                dz = HOVSWEEP_LO + (ph - _d0)
            else:
                dz = _hi
            target = np.array([vic[0], vic[1], _base + dz])
            self._dbg_hover_tz = float(target[2])             # probe: sweep target
            return (target, HOVSWEEP_MPS / SPEED_LIMIT)

        boost = float(os.environ.get('HYB_MTN_HOVZ', '0.0')) if self.mtn.is_mountain else 0.0
        target = np.array([vic[0], vic[1], head_z + HOVER_ABOVE_TOP_M + HOVER_SETTLE_MARGIN_M + boost])
        agl_sp = float(os.environ.get('HYB_MTN_AGLHOV', '3.2'))
        if self.mtn.is_mountain and agl_sp > 0.0 and self._last_agl is not None:

            _ray_z = pos[2] - (self._last_agl - agl_sp)

            _bidir_on = os.environ.get('SAR150_BIDIR', '1') == '1'
            if MH_ON and not MH_BIDIR:
                _bidir_on = False                              # 201: max(head + 2.5, ray + 3.2)
            if (_bidir_on and os.environ.get('SAR_BIDIR_OPENGATE', '1') == '1'
                    and _MAPSW_STATE["map"] == 'open'):
                _bidir_on = False
            if _bidir_on:
                target[2] = _ray_z
            else:
                target[2] = max(target[2], _ray_z)

        if HOVZ_FREEZE_R > 0.0:
            _hzd = float(np.linalg.norm(pos[0:2] - target[0:2]))
            if _hzd <= HOVZ_FREEZE_R:
                if self._hov_z_lock is None:
                    self._hov_z_lock = float(target[2])
                target[2] = self._hov_z_lock
            else:
                self._hov_z_lock = None
        self._dbg_hover_tz = float(target[2])                 # probe: what the hover asked for
        if pos[2] - target[2] > HOVER_DESCENT_M:
            return (target, HOVER_DESCENT_SPEED)
        return (target, CONFIRM_SPEED)

    def act(self, observation) -> np.ndarray:
        self.tick += 1
        state = np.asarray(observation['state'], dtype=np.float64).reshape(-1)
        depth = np.asarray(observation['depth'], dtype=np.float32)
        pos = state[IDX_POS]
        rpy = state[IDX_RPY]
        rgb = np.asarray(observation.get('rgb', np.zeros(0)), dtype=np.float32)
        alt = float(state[IDX_ALT])
        self._last_agl = alt * MAX_RAY_M if alt > 1e-6 else None
        clue = state[IDX_CLUE]
        ground_z = pos[2] - alt * MAX_RAY_M
        if self.tick <= 1 or getattr(self, '_tks_pad_z', None) is None:
            self._tks_pad_z = float(pos[2])
        self._update_clutter(depth)
        self.avoider.wide = bool(getattr(self, 'forest_low', False))\
            and os.environ.get('SAR150_WIDE', '1') == '1'
        self.mtn.update(self.tick, state, depth)
        self.detector.strong_p_override = (
            float(os.environ.get('HYB_WH_STRONGP', '0'))
            if (self.mtn.settled('warehouse') and not self.mtn.is_mountain) else 0.0)
        if self.ground_min is None:
            self.ground_min = ground_z
        self.ground_min = min(self.ground_min, ground_z)
        if not self.forest_low and (not self.mtn.is_mountain):
            wp_settled = any((self.mtn.settled(t) for t in WAYPOINT_TERRAINS))
            cf = self._clutter_frac()
            p_fo = self.mtn.prob_of('forest')
            p_wh = self.mtn.prob_of('warehouse')
            early = (not wp_settled) and self.tick >= FOREST_EARLY_TICK and p_fo >= FOREST_EARLY_PROB and (p_wh <= FOREST_EARLY_WH) and (cf >= FOREST_CLUTTER_MIN)
            late = (not wp_settled) and self.tick >= FOREST_MIN_TICK and p_wh <= FOREST_WH_VETO and (p_fo >= FOREST_LATE_PROB and cf >= FOREST_CLUTTER_MIN or cf >= FOREST_SOLO_CLUTTER)

            rgbf = False
            if getattr(self, '_rgbf_evals', 0) < RGBT_MAX_EVALS and rgb.size >= 49152 and rgb.size % 3 == 0:
                try:
                    side = int(round((rgb.size // 3) ** 0.5))
                    fr = rgb.reshape(side, side, 3)
                    if float(np.abs(fr[::16, ::16]).mean()) > 0.005:
                        pr = _rgbt_proba(fr)
                        self._rgbf_evals = getattr(self, '_rgbf_evals', 0) + 1
                        if pr[5] >= RGBT_P_FOREST:
                            self._rgbf_hits = getattr(self, '_rgbf_hits', 0) + 1
                except Exception:
                    self._rgbf_evals = RGBT_MAX_EVALS
                rgbf = (getattr(self, '_rgbf_hits', 0) >= RGBT_HITS_NEEDED
                        and self.tick >= FOREST_EARLY_TICK and p_wh <= FOREST_WH_VETO
                        and (not V9_RGBF_CLUTTER or _MAPSW_STATE.get("map") != 'forest'
                             or cf >= FOREST_CLUTTER_MIN))
            if early or late or rgbf:
                self.forest_low = True
                self.forest_latch_tick = self.tick
        _tks_det = (TKS_ON and (not self._tks_done) and (not self.mtn.is_mountain)
                    and _MAPSW_STATE.get("map") in TKS_MAPS and self.tick >= TKS_T0)
        if self.mode == 'initial' and not _tks_det:
            conf, vic, head_z = (0.0, pos.copy(), ground_z)
        elif self.frozen:
            conf, vic, head_z = (1.0, self.locked_vic, self.locked_head_z)
        else:
            self.detector.rgb_primary_ok = self.mode in ('search', 'navigation')
            self.detector.rgb_primary_near_only = bool(self.locked)
            _strong_before = float(self.detector.strong)
            conf, vic, head_z = self.detector.update(depth, pos, rpy, rgb=rgb, is_mountain=self.mtn.is_mountain)
            if self.detector.rgb_fresh and self.detector.rgb_prob >= RGB_CONFIRM_THRESH:
                self.rgb_confirm_tick = self.tick
            if self.detector.rgb_promoted:
                self.rgb_hit_tick = self.tick
            if _mq_on() and self.mtn.is_mountain:
                if self.detector.seen and self._mq_spot_refuted(vic):
                    # a refuted rock keeps firing: do not let it re-arm the track for MQ_REFUTE_TTL
                    self.detector.strong = 0.0
                    self.detector.cand_strong = 0.0
                    self.detector.conf = 0.0
                    self.detector.seen = False
                    conf = 0.0
                else:
                    self._mq_hold(_strong_before)
                    if self._mq_step(pos, rpy, vic, conf):
                        conf, vic, head_z = (0.0, pos.copy(), ground_z)
                    else:
                        conf = float(self.detector.conf)
        self._nav_head_z = float(head_z) if (self.mode != 'initial' and np.isfinite(head_z)) else None   # 304 FA
        elevated = not self.mtn.is_mountain and vic[2] - ground_z > _mapsw_gate() and (float(np.linalg.norm(vic[0:2] - pos[0:2])) > INVESTIGATE_M)
        if self.mode != 'initial' and (not self.frozen) and (not self.locked) and (conf >= LATCH_CONF) and (not elevated):
            self.locked = True
            self.locked_vic = vic.copy()
            self.locked_head_z = head_z
            self.lost_ticks = 0
            self._fix_hits = []
            self._fix_pt = None
            self._fix_evidence = 0
            self._fixf_retries = 0
        _fixw = _fix_world(self)
        self.detector.fix_world = _fixw
        _fh = _forest_hov(self)
        self.detector.fix_range = FIXF_RANGE_M if _fh else FIX_RANGE_M
        _fix_near_m = FIXF_NEAR_M if _fh else FIX_NEAR_M
        if self.locked and (not self.frozen):
            _seen = self.detector.seen
            if (_seen and TEL_LOCKGUARD and float(np.hypot(vic[0] - self.locked_vic[0], vic[1] - self.locked_vic[1])) > CONSIST_M
                    and float(self.detector.strong) < TEL_LG_N):
                # TEL D: the detector's primary track moved to another object (veto / stronger candidate);
                # keep the lock where it was unless the new track is fully established
                _seen = False
                self.mq_stats["tel_lockguard"] = self.mq_stats.get("tel_lockguard", 0) + 1
            if _seen:
                if _fixw and float(np.linalg.norm(self.locked_vic[0:2] - pos[0:2])) <= _fix_near_m:
                    _fp = self.detector.fix_near(self.locked_vic[0:2], window=(FIXF_WINDOW if (_fh and HOV_MODE == 2) else FIX_WINDOW))
                    if _fp is not None:
                        self._fix_pt = _fp
                        self.mq_stats["fix_set"] = self.mq_stats.get("fix_set", 0) + 1
                if self._fix_pt is not None:
                    # the frozen point wins over the tracker's estimate; the head is the measured centre + half height
                    self.locked_vic = np.array([self._fix_pt[0], self._fix_pt[1], self._fix_pt[2]], dtype=np.float64)
                    self.locked_head_z = float(self._fix_pt[2]) + 0.5 * float(np.clip(self.detector.height, MH_HMIN, MH_HMAX))
                else:
                    self.locked_vic = vic.copy()
                    self.locked_head_z = head_z
                self.lost_ticks = 0
                if (_fixw and self._fix_pt is not None and float(np.linalg.norm(self.locked_vic[0:2] - pos[0:2])) < R_HOVER_M
                        and not (_fh and HOV_MODE == 2)):   # 304 mode 2: keep updating the centroid until sight is lost
                    self.frozen = True              # freeze on a good fix, not only after losing sight
                    self.mq_stats["fix_freeze"] = self.mq_stats.get("fix_freeze", 0) + 1
            elif float(np.linalg.norm(self.locked_vic[0:2] - pos[0:2])) < FREEZE_NEAR_M:
                self.lost_ticks += 1
                if self.lost_ticks >= LOST_VIS_TICKS:
                    self.frozen = True
        if self.mode == 'hover' and self.frozen and (float(np.linalg.norm(self.locked_vic[0:2] - pos[0:2])) < HOVER_STABLE_M):
            self.hover_total_ticks += 1
            dz = pos[2] - self.locked_head_z
            if HOVER_GIVEUP_BAND[0] <= dz <= HOVER_GIVEUP_BAND[1]:
                self.hover_stable_ticks += 1
        else:
            self.hover_stable_ticks = 0
            self.hover_total_ticks = 0
        _giveup_n = HOVER_GIVEUP_TICKS
        if HOVSWEEP and self.mtn.is_mountain and self.frozen:
            if getattr(self, '_sweep_t0', None) is None and self.hover_stable_ticks >= HOVSWEEP_TRIG:
                self._sweep_t0 = self.tick
            if getattr(self, '_sweep_t0', None) is not None:
                _giveup_n = HOVSWEEP_GIVEUP
        _msw_done = False
        if (MSW_ON and self.mtn.is_mountain and getattr(self, '_sweep_t0', None) is not None
                and (self.tick - self._sweep_t0) / CTRL_HZ * MSW_MPS >= (MSW_TOP - MSW_LO) + (MSW_HI - MSW_LO) + 1.0):
            _msw_done = True
            self.mq_stats["msw_giveup"] += 1
        _rehover = False
        if self.hover_stable_ticks >= _giveup_n or self.hover_total_ticks >= HOVER_GIVEUP_FALLBACK_TICKS or _msw_done:
            if (_fh and HOV_MODE == 2 and getattr(self, '_fixf_retries', 0) < FIXF_RETRY
                    and self.detector.evidence_at(self.locked_vic[0:2]) >= FIX_EVIDENCE):
                _cp = self.detector.fix_near(self.locked_vic[0:2], window=500, min_n=FIX_EVIDENCE)
                if _cp is not None:
                    # 304: the spot is a person seen from close range -- hover again on the mass centre of the hits
                    self._fixf_retries = getattr(self, '_fixf_retries', 0) + 1
                    self.locked_vic = np.array([_cp[0], _cp[1], _cp[2]], dtype=np.float64)
                    self.locked_head_z = float(_cp[2]) + 0.5 * float(np.clip(self.detector.height, MH_HMIN, MH_HMAX))
                    self._fix_pt = _cp
                    self.frozen = True
                    self.hover_stable_ticks = 0
                    self.hover_total_ticks = 0
                    self.mq_stats["fix_rehover"] = self.mq_stats.get("fix_rehover", 0) + 1
                    _rehover = True
        if _rehover:
            pass
        elif self.hover_stable_ticks >= _giveup_n or self.hover_total_ticks >= HOVER_GIVEUP_FALLBACK_TICKS or _msw_done:
            if _fix_world(self) and self.detector.evidence_at(self.locked_vic[0:2]) >= FIX_EVIDENCE:
                self.mq_stats["fix_noblack"] = self.mq_stats.get("fix_noblack", 0) + 1   # a person, not a phantom: re-approach later
            else:
                self.detector.blacklist.append(self.locked_vic.copy())
            self._fix_hits = []
            self._fix_pt = None
            self._fix_evidence = 0
            self.detector.vic = np.zeros(3, dtype=np.float64)
            self.detector.strong = 0.0
            self.detector.cand = np.zeros(3, dtype=np.float64)
            self.detector.cand_strong = 0.0
            self.detector.conf = 0.0
            self.locked = False
            self.frozen = False
            self.lost_ticks = 0
            self.hover_stable_ticks = 0
            self.hover_total_ticks = 0
            self.mode = 'search'
            self._sweep_t0 = None
        if self.mode == 'initial':
            _deck = MTK_H if (MTK_ON and self.mtn.is_mountain) else self._search_alt()
            _mg, _tk = (INIT_EXIT_MARGIN, INIT_EXIT_TICKS) if (self.mtn.is_mountain and INIT_EXIT_TICKS > 0) else (1.0, 150)
            if pos[2] >= ground_z + _deck - _mg or self.tick > _tk or (_tks_det and conf >= COMMIT_CONF):
                self.mode = 'search'
        elif self.locked:
            horiz = float(np.linalg.norm(self.locked_vic[0:2] - pos[0:2]))
            self.mode = 'hover' if horiz < R_HOVER_M else 'navigation'
        else:
            near = float(np.linalg.norm(vic[0:2] - pos[0:2])) < INVESTIGATE_M
            if self._commit_ok(conf, near=near) and (not elevated):
                self.mode = 'navigation'
            elif self.mode == 'navigation' and conf < LOST_CONF:
                self.mode = 'search'
        if self.mode == 'initial':
            target, speed = self._initial_policy(pos, ground_z)
        elif self.mode == 'search':
            target, speed = self._search_policy(pos, clue, ground_z)
        elif self.mode == 'navigation':
            nav_vic = self.locked_vic if self.locked else vic
            target, speed = self._navigation_policy(pos, nav_vic, ground_z)
        else:
            target, speed = self._hover_policy(self.locked_vic, self.locked_head_z, pos)
        if (_tks_det and self.mode in ('initial', 'search') and (not self.locked) and (not self.frozen)
                and ((self._tks_pnear if self._tks_pnear is not None else 0.0) >= TKS_P_HOLD
                     or getattr(self, '_tks_phase', 'sweep') == 'verify')):
            # holding sweep: hover TKS_HOLD_ABOVE_PAD over the take-off point while the camera goes round
            _pz = self._tks_pad_z if self._tks_pad_z is not None else float(pos[2])
            # pad-relative (a person under the pad sits in the band) but never above ground + TKS_HOLD_AGL_MAX
            # (open pads are 5-10 m platforms: a hold at pad + 3.3 would put a ground victim 10 m below the cone)
            _hz = min(_pz + TKS_HOLD_ABOVE_PAD, ground_z + TKS_HOLD_AGL_MAX)
            target = np.array([pos[0], pos[1], max(_hz, ground_z + 2.6)], dtype=np.float64)
            speed = 0.6
            self.mq_stats["tks_hold"] = self.mq_stats.get("tks_hold", 0) + 1
        delta = target - pos
        dist = float(np.linalg.norm(delta))
        direction = _unit(delta)
        if MTK_ON and MTK_FULL_SPEED and self.mtn.is_mountain and self.mode == 'search':
            self.mq_stats["mtk_full"] = self.mq_stats.get("mtk_full", 0) + 1
        elif P1_RAMPK > 0.0 and (self.locked or self.frozen) and self.mode in ('navigation', 'hover'):
            speed = min(speed, float(np.clip(dist * P1_RAMPK, 0.0, 1.0)))
        else:
            speed = min(speed, float(np.clip(dist * 0.125, 0.0, 1.0)))
        if (YAWGATE_DEG > 0.0 and _MAPSW_STATE.get("map") in YAWGATE_MAPS
                and self.mode in YAWGATE_MODES and not self.frozen
                and float(np.hypot(direction[0], direction[1])) > 0.1):
            _mis = abs((float(np.arctan2(direction[1], direction[0]))
                        - float(rpy[2]) + np.pi) % (2.0 * np.pi) - np.pi)
            if _mis > np.radians(YAWGATE_DEG):
                speed = min(speed, YAWGATE_SPEED_CITY if _MAPSW_STATE.get("map") == 'city' else YAWGATE_SPEED)
        if MBT_ON and self.mtn.is_mountain and self.mode == 'search' and (not self.locked) and (not self.frozen):
            # v2: only in the MBT_WIN ticks after the lane waypoint jumped (the tour advance / a new leg) -- a
            # continuous gate fired 600-2500 ticks per episode on the local spiral and cost more than it saved
            _wp = np.array([float(pos[0]) + float(clue[0]), float(pos[1]) + float(clue[1])])
            _pw = getattr(self, '_mbt_prev_wp', None)
            if _pw is not None and float(np.hypot(_wp[0] - _pw[0], _wp[1] - _pw[1])) > MBT_JUMP_M:
                self._mbt_until = self.tick + MBT_WIN
            self._mbt_prev_wp = _wp
            if (getattr(self, '_mbt_until', -1) >= self.tick and self._msp_turned is None
                    and getattr(self, '_mcv_until', -1) < self.tick
                    and float(np.hypot(direction[0], direction[1])) > 0.1):
                _mis = abs((float(np.arctan2(direction[1], direction[0]))
                            - float(rpy[2]) + np.pi) % (2.0 * np.pi) - np.pi)
                if _mis > np.radians(MBT_DEG):
                    speed = min(speed, MBT_SPEED)
                    self.mq_stats["mbt_ticks"] = self.mq_stats.get("mbt_ticks", 0) + 1
        if self.mode != 'hover':
            if self.mtn.is_mountain:
                direction, speed = self._mountain_avoid(direction, speed, depth, rpy)
            else:
                direction, speed = self.avoider.steer(direction, speed, depth, rpy)

                if (BLIND_SPEED > 0.0 and speed > BLIND_SPEED
                        and (_MAPSW_STATE.get("map") == 'forest'
                             or bool(getattr(self, 'forest_low', False)))):
                    _fw, _up, _rt = camera_axes(rpy)
                    if float(direction @ _fw) < BLIND_ANG_COS:
                        _dm = depth if depth.ndim == 2 else depth[0]
                        _near = float(_dm.min()) * (DEPTH_MAX_M - DEPTH_MIN_M) + DEPTH_MIN_M
                        if _near < BLIND_NEAR_M:
                            speed = BLIND_SPEED
                if (os.environ.get('HYB_WH_CLIMB', '0') == '1'
                        and self.mtn.settled('warehouse') and self.mode != 'navigation'):

                    left, centre, right = self.avoider.sectors(depth)
                    nearest = min(left, centre, right)
                    if nearest < 6.0:
                        urg = float(np.clip((6.0 - nearest) / 6.0, 0.0, 1.0))
                        direction = _unit(direction + np.array([0.0, 0.0, 0.9 * urg]))
                        speed = min(speed, max(0.35, 1.0 - 0.7 * urg))
        direction, speed = self.smoother(direction, speed)

        ramp = os.environ.get('HYB_WH_RAMP', '')
        if ramp and self.mtn.settled('warehouse') and not self.mtn.is_mountain\
                and self.mode == 'search':
            rt, rcap = (float(x) for x in ramp.split(','))

            if self.tick < rt:
                frac = float(np.clip((self.tick - (rt - 50.0)) / 50.0, 0.0, 1.0))
                speed = min(speed, rcap + (1.0 - rcap) * frac)
        if speed > 0.001:
            yaw = np.arctan2(direction[1], direction[0]) / np.pi
        else:
            yaw = float(rpy[2]) / np.pi

        sweep_amp = float(os.environ.get('HYB_MTN_YAWSWEEP', '0.0'))
        if sweep_amp > 0.0 and self.mtn.is_mountain and self.mode == 'search'\
                and not self.locked and not self.frozen:

            sweep_delay = float(os.environ.get('HYB_MTN_YAWSWEEP_DELAY', '1000'))
            if self.tick >= sweep_delay and self.detector.last_strong_t < 0:
                yaw = float(np.clip(yaw + sweep_amp * np.sin(self.tick * 0.045), -1.0, 1.0))
        if MSP_ON and self.mtn.is_mountain:
            yaw = self._msp_step(yaw, depth, rpy, speed)
        if MCV_ON and self.mtn.is_mountain and self.mode == 'search' and (not self.locked) and (not self.frozen):
            _det = self.detector
            if (self._mcv_until < self.tick and float(_det.last_p_depth) >= STRONG_P
                    and self.tick - self._mcv_end >= MCV_GAP):
                _mz = np.asarray(_det.last_z, dtype=np.float64)
                if (float(np.hypot(_mz[0] - pos[0], _mz[1] - pos[1])) <= MCV_R and float(_det.strong) > 0.0
                        and not _det.location_refuted(_mz)):
                    self._mcv_until = self.tick + MCV_TICKS
                    self._mcv_pt = _mz.copy()
                    self._msp_turned = None
                    self.mq_stats["mcv_started"] = self.mq_stats.get("mcv_started", 0) + 1
            if self._mcv_until >= self.tick:
                _mz = np.asarray(_det.vic if float(_det.strong) > 0.0 else self._mcv_pt, dtype=np.float64)
                yaw = float(np.clip(math.atan2(_mz[1] - pos[1], _mz[0] - pos[0]) / np.pi, -1.0, 1.0))
                speed = min(speed, MCV_SPEED)
                self._mcv_end = self.tick
                self.mq_stats["mcv_ticks"] = self.mq_stats.get("mcv_ticks", 0) + 1
        elif FSP_ON and (not self.mtn.is_mountain) and (_MAPSW_STATE.get("map") == 'forest' or self.forest_low):
            if (TK_SWEEP == 3 and _tk_sweep_world() and (not self._tk_done) and self.mode == 'search'
                    and self._msp_turned is None and (not self.locked) and (not self.frozen)):
                if self.tick > TK_SWEEP_TICKS:
                    self._tk_done = True
                else:
                    try:
                        _clear = min(self.avoider.sectors(depth)) >= TKB_CLEAR_M
                    except Exception:
                        _clear = False
                    if _clear:
                        self._msp_turned = 0.0                 # 303: forced take-off turn, no 12 m / period gate
                        self._msp_yaw = float(rpy[2])
                        self._tkb_forced = True
                        self._tk_done = True
                        self.mq_stats["tk_turns"] = self.mq_stats.get("tk_turns", 0) + 1
            _abort = TKB_ABORT_M if self._tkb_forced else FSP_ABORT_M
            yaw = self._msp_step(yaw, depth, rpy, speed, FSP_CLEAR_M, FSP_AGL_MIN, _abort, FSP_PERIOD)
            if self._msp_turned is None:
                self._tkb_forced = False
            self.mq_stats["fsp_ticks"] = self.mq_stats.get("fsp_ticks", 0) + (1 if self._msp_turned is not None else 0)
        if (TKS_ON and (not self._tks_done) and (not self.mtn.is_mountain)
                and _MAPSW_STATE.get("map") in TKS_MAPS and self.tick >= TKS_T0):
            _busy = self.locked or self.frozen or self.mode not in ('initial', 'search')
            if self._tks_yaw0 is None:
                self._tks_yaw0 = float(rpy[2])
                self._tks_t0 = self.tick
                self._tks_phase = 'sweep'
                # sweep toward the side holding the box centre (world origin)
                _b = math.atan2(-pos[1], -pos[0])
                _rel = (_b - self._tks_yaw0 + np.pi) % (2.0 * np.pi) - np.pi
                self._tks_dir = 1.0 if _rel >= 0.0 else -1.0
                self.mq_stats["tks_start"] = self.mq_stats.get("tks_start", 0) + 1
            _pn = self._tks_pnear if self._tks_pnear is not None else 0.0
            _hold = _pn >= TKS_P_HOLD
            _total = TKS_DEG_HOLD if _hold else ((P1_TKS_OUT if (P1_TKS_OUT > 0.0 and _pn < 0.02 and _MAPSW_STATE.get("map") in P1_TKS_OUT_MAPS) else TKS_DEG_FREE))
            # unwrapped progress: accumulate the yaw actually turned in the sweep direction
            _prev = getattr(self, '_tks_prev_yaw', None)
            if _prev is not None and self._tks_phase == 'sweep':
                _d = ((float(rpy[2]) - _prev + np.pi) % (2.0 * np.pi)) - np.pi
                self._tks_deg += math.degrees(_d) * self._tks_dir
            self._tks_prev_yaw = float(rpy[2])
            if getattr(self, '_tks_ck_t', None) is None or self.tick - self._tks_ck_t >= 50:
                if getattr(self, '_tks_ck_t', None) is None or self._tks_deg - self._tks_ck_deg >= 3.0:
                    self._tks_ck_t, self._tks_ck_deg = self.tick, float(self._tks_deg)
            _strong_now = float(self.detector.last_p_depth) >= STRONG_P
            if _busy or (self.tick - self._tks_t0 >= TKS_MAX_TICKS):
                self._tks_done = True
                self.mq_stats["tks_stop_busy" if _busy else "tks_timeout"] = self.mq_stats.get("tks_stop_busy" if _busy else "tks_timeout", 0) + 1
            elif self._tks_phase == 'verify':
                # camera parked on the sighting; wait for the track to commit or die
                if self.detector.seen:
                    self._tks_ver_hits = getattr(self, '_tks_ver_hits', 0) + 1
                _ver_end = None
                if float(self.detector.strong) <= 0.0 and self.tick - self._tks_ver_t0 >= TKS_VERIFY_MIN:
                    _ver_end = "tks_verify_lost"
                elif self.tick - self._tks_ver_t0 >= TKS_VERIFY_MAX:
                    _ver_end = "tks_verify_timeout"
                if _ver_end is not None:
                    self.mq_stats[_ver_end] = self.mq_stats.get(_ver_end, 0) + 1
                    if getattr(self, '_tks_ver_hits', 0) >= TKS_VERIFY_COMMIT and float(self.detector.strong) > 0.0:
                        # repeated sightings from a still camera: go and look instead of resuming the sweep
                        self.detector.strong = max(float(self.detector.strong), N_LATCH * COMMIT_CONF + 0.1)
                        self.detector.conf = float(np.clip(self.detector.strong / N_LATCH, 0.0, 1.0))
                        self._tks_done = True
                        self.mq_stats["tks_verify_commit"] = self.mq_stats.get("tks_verify_commit", 0) + 1
                    else:
                        self._tks_phase = 'sweep'
                        self._tks_sweep_t0 = self.tick
                    self._tks_ver_hits = 0
                else:
                    _z = self.detector.vic if float(self.detector.strong) > 0.0 else self.detector.last_z
                    yaw = float(math.atan2(_z[1] - pos[1], _z[0] - pos[0]) / np.pi)
                    self.mq_stats["tks_verify_ticks"] = self.mq_stats.get("tks_verify_ticks", 0) + 1
            if (not self._tks_done) and self._tks_phase == 'sweep':
                if (_strong_now or float(self.detector.strong) > 0.0) and self.mq_stats.get("tks_verify", 0) < TKS_VERIFY_N:
                    self._tks_phase = 'verify'
                    self._tks_ver_t0 = self.tick
                    self._tks_ver_hits = 1 if self.detector.seen else 0
                    self._tks_deg_at_verify = self._tks_deg
                    self.mq_stats["tks_verify"] = self.mq_stats.get("tks_verify", 0) + 1
                    _z = self.detector.vic if float(self.detector.strong) > 0.0 else self.detector.last_z
                    yaw = float(math.atan2(_z[1] - pos[1], _z[0] - pos[0]) / np.pi)
                elif self._tks_deg >= _total - 5.0:
                    self._tks_done = True
                    self.mq_stats["tks_done"] = self.mq_stats.get("tks_done", 0) + 1
                elif (P1_TKS and self._tks_deg > 20.0 and getattr(self, '_tks_ck_t', None) is not None
                        and self.tick - self._tks_ck_t >= 50 and self._tks_deg - self._tks_ck_deg < 3.0):
                    # P1: the airframe cannot track the setpoint any further (steady-state yaw error at
                    # cruise); end the sweep instead of holding the camera off-track for 10 s
                    self._tks_done = True
                    self.mq_stats["tks_stall"] = self.mq_stats.get("tks_stall", 0) + 1
                else:
                    # setpoint leads the yaw actually achieved by a bounded angle (a lead past 180 deg would
                    # flip the shortest-path direction and turn the drone back)
                    _ramp = min(_total, max(self._tks_deg, 0.0) + TKS_LEAD)
                    yaw = float((((self._tks_yaw0 + self._tks_dir * np.radians(_ramp)) + np.pi) % (2.0 * np.pi) - np.pi) / np.pi)
                    self.mq_stats["tks_ticks"] = self.mq_stats.get("tks_ticks", 0) + 1
        if TK_SWEEP in (1, 2) and _tk_sweep_world() and (not self._tk_done):
            if self._tk_yaw0 is None:
                self._tk_yaw0 = float(rpy[2])
                self._tk_t0 = self.tick
            _stop = self.tick > TK_SWEEP_TICKS or (TK_SWEEP_STOP and (
                self.locked or self.frozen or self.mode not in ('initial', 'search')
                or float(self.detector.last_p_depth) >= STRONG_P))
            if _stop:
                self._tk_done = True
                self.mq_stats["tk_stop"] = self.mq_stats.get("tk_stop", 0) + 1
            else:
                _off = np.radians(TK_SWEEP_DEG) * (1.0 if self.tick - self._tk_t0 < TK_SWEEP_TURN else -1.0)
                yaw = float(((self._tk_yaw0 + _off + np.pi) % (2.0 * np.pi) - np.pi) / np.pi)
                if TK_SWEEP == 1 and self.mode != 'initial':     # variant 1: the climb continues, the search leg waits
                    direction = np.zeros(3, dtype=np.float32)
                    speed = 0.0
                    self.mq_stats["tk_hold"] = self.mq_stats.get("tk_hold", 0) + 1
        rgb_req = 0.0
        ondemand = self.detector.last_p_depth > RGB_TARGET_TRIGGER and (not self.detector.location_refuted(self.detector.last_z))
        _periodic = (self.tick - getattr(self, '_rgb_phase', 0)) % RGB_REQUEST_PERIOD == 0
        if self.mtn.is_mountain:
            if MTN_RGB_PACE:
                _left = MTN_RGB_PACE_CAP - self.rgb_requests
                if _left <= 0:
                    _periodic = False
                elif self.rgb_requests >= MTN_RGB_PACE_AFTER:
                    _interval = max(RGB_MIN_INTERVAL, (MTN_RGB_PACE_HORIZON - self.tick) // _left)
                    _periodic = (self.tick - self.last_rgb_req >= _interval) and ((self.tick - getattr(self, '_rgb_phase', 0)) % RGB_MIN_INTERVAL == 0)
            if MTN_RGB_CHASE_TICKS > 0 and self.tick - self.rgb_hit_tick <= MTN_RGB_CHASE_TICKS:
                ondemand = True
            if MRGB_ON:
                _periodic = False                              # 201: no blind periodic scan on mountain
                if (self.locked and (not self.frozen) and self.mode == 'navigation'
                        and float(np.hypot(self.locked_vic[0] - pos[0], self.locked_vic[1] - pos[1])) <= MRGB_CHECK_R
                        and self.tick - self.last_rgb_req >= MRGB_CHECK_GAP):
                    ondemand = True                             # check the target on the way in
                    self.mq_stats["mrgb_check"] = self.mq_stats.get("mrgb_check", 0) + 1
        if self.mode in ('search', 'navigation') and (not self.frozen) and (self.rgb_requests < RGB_CAP) and (self.tick - self.last_rgb_req >= RGB_MIN_INTERVAL) and (_periodic or ondemand):
            rgb_req = 1.0
            self.rgb_requests += 1
            self.last_rgb_req = self.tick
        action = np.zeros(6, dtype=np.float32)
        action[0:3] = direction.astype(np.float32)
        action[3] = float(np.clip(speed, 0.0, 1.0))
        action[4] = float(np.clip(yaw, -1.0, 1.0))
        action[5] = rgb_req
        return action
UID137_MODEL = 'policy.onnx'
UID137_MEMORY_NAME = 'memory_tensor'
UID137_MEMORY_SHAPE = (63,)
UID137_STATE_LEN = 165
UID137_DEPTH_SHAPE = (256, 256, 1)
ACTION_QUANT_STEP = 9.5367431640625e-07
RGB_REQUEST_THRESHOLD = 0.5
ACTION_LOWER = np.array([-1.0, -1.0, -1.0, 0.0, -1.0, 0.0], dtype=np.float32)
ACTION_UPPER = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
RGB_REQUEST_INDEX = 5

RGBT_NAME = 'rgb_terrain_flat.npz'
RGBT_P_FOREST = 0.9
RGBT_HITS_NEEDED = 2
RGBT_MAX_EVALS = 12
_RGBT = None

LR_NAME = 'victim_lr.onnx'
LR_ON = os.environ.get('SAR_LR', '1') == '1'
LR_STRIDE = int(os.environ.get('SAR_LR_STRIDE', '2'))
LR_P_BAR = float(os.environ.get('SAR_LR_P', '0.70'))
LR_H_MAX = float(os.environ.get('SAR_LR_HMAX', '0.95'))
LR_STRONG_MAX = float(os.environ.get('SAR_LR_SMAX', '0.5'))
LR_QUIET_TICKS = int(os.environ.get('SAR_LR_QUIET', '50'))
LR_T_MIN = int(os.environ.get('SAR_LR_TMIN', '0'))
LR_TRACK = os.environ.get('SAR_LR_TRACK', '1') == '1'
LR_OWN_TTL = int(os.environ.get('SAR_LR_TTL', '150'))
_LR_NOCAND = os.environ.get('SAR_LR_NOCAND', '1') == '1'

def _rgbt_load():
    global _RGBT
    if _RGBT is None:
        z = np.load(str(Path(__file__).resolve().parent / RGBT_NAME))
        _RGBT = (z['feat'], z['thr'], z['lc'], z['rc'], z['leaf'], z['roots'],
                 z['bias'].astype(np.float64), int(z['n_class']), z['classes'].tolist())
    return _RGBT

def _rgbt_hsv(f):
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    mx = np.max(f, axis=-1)
    mn = np.min(f, axis=-1)
    d = mx - mn + 1e-9
    h = np.zeros_like(mx)
    m = mx == r
    h[m] = ((g - b)[m] / d[m]) % 6.0
    m = mx == g
    h[m] = (b - r)[m] / d[m] + 2.0
    m = mx == b
    h[m] = (r - g)[m] / d[m] + 4.0
    h /= 6.0
    s = d / (mx + 1e-9)
    return h, s, mx

def _rgbt_features(frame):
    f = np.clip(np.asarray(frame, np.float32), 0.0, 1.0)
    h, s, v = _rgbt_hsv(f)
    feats = []
    w = (s * v).ravel()
    hh, _ = np.histogram(h.ravel(), bins=12, range=(0, 1), weights=w)
    feats += list(hh / (w.sum() + 1e-6))
    feats += list(np.histogram(s.ravel(), bins=6, range=(0, 1))[0] / s.size)
    feats += list(np.histogram(v.ravel(), bins=6, range=(0, 1))[0] / v.size)
    green = ((h > 0.18) & (h < 0.45) & (s > 0.15) & (v > 0.08)).mean()
    grey = (s < 0.12).mean()
    brown = ((h > 0.02) & (h < 0.13) & (s > 0.15)).mean()
    blue = ((h > 0.5) & (h < 0.75) & (s > 0.15)).mean()
    white = ((v > 0.8) & (s < 0.15)).mean()
    dark = (v < 0.15).mean()
    feats += [green, grey, brown, blue, white, dark]
    H = f.shape[0]
    for band in (slice(0, H // 3), slice(H // 3, 2 * H // 3), slice(2 * H // 3, H)):
        feats += [float(v[band].mean()), float(s[band].mean()),
                  float(((h[band] > 0.18) & (h[band] < 0.45) & (s[band] > 0.15)).mean())]
    gx = np.abs(np.diff(v, axis=1)).mean()
    gy = np.abs(np.diff(v, axis=0)).mean()
    g4 = np.abs(np.diff(v[::4, ::4], axis=1)).mean()
    feats += [float(gx), float(gy), float(g4), float(v.std()), float(s.std())]
    return np.asarray(feats, np.float64)

def _rgbt_proba(frame):
    F, T, L, R, V, roots, bias, n_class, _ = _rgbt_load()
    x = _rgbt_features(frame)
    m = np.zeros(n_class)
    for t in range(len(roots)):
        i = int(roots[t])
        while F[i] >= 0:
            i = int(L[i]) if x[F[i]] < T[i] else int(R[i])
        m[t % n_class] += V[i]
    m += bias
    e = np.exp(m - m.max())
    return e / e.sum()
ORT_INTRA_OP_THREADS = 2
ORT_INTER_OP_THREADS = 1
_UID137_SAFE_ACTION = np.zeros(6, dtype=np.float32)

def _uid137_session_options() -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.intra_op_num_threads = ORT_INTRA_OP_THREADS
    options.inter_op_num_threads = ORT_INTER_OP_THREADS
    options.enable_mem_pattern = False
    options.add_session_config_entry('session.use_deterministic_compute', '1')
    return options

def _uid137_canonicalize(raw: np.ndarray) -> np.ndarray:
    action = np.asarray(raw, dtype=np.float32).reshape(-1)[:6]
    if not np.isfinite(action).all():
        return _UID137_SAFE_ACTION.copy()
    action = np.clip(action, ACTION_LOWER, ACTION_UPPER).astype(np.float32, copy=True)
    action = (np.rint(action / ACTION_QUANT_STEP) * ACTION_QUANT_STEP).astype(np.float32)
    action[RGB_REQUEST_INDEX] = 1.0 if action[RGB_REQUEST_INDEX] > RGB_REQUEST_THRESHOLD else 0.0
    return action

class _UID137Policy:

    def __init__(self) -> None:
        model_path = Path(__file__).resolve().parent / UID137_MODEL
        if not model_path.is_file():
            raise FileNotFoundError(f'{UID137_MODEL} must sit beside drone_agent.py')
        self._session = ort.InferenceSession(model_path.read_bytes(), sess_options=_uid137_session_options(), providers=['CPUExecutionProvider'])
        self._input_names = {i.name for i in self._session.get_inputs()}
        self._output_names = [o.name for o in self._session.get_outputs()]
        self._memory = np.zeros(UID137_MEMORY_SHAPE, dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        self._memory = np.zeros(UID137_MEMORY_SHAPE, dtype=np.float32)

    def act(self, observation) -> np.ndarray:
        depth = np.ascontiguousarray(np.asarray(observation['depth'], dtype=np.float32).reshape(UID137_DEPTH_SHAPE))
        state = np.asarray(observation['state'], dtype=np.float32).reshape(-1)
        if state.size != UID137_STATE_LEN:
            fitted = np.zeros(UID137_STATE_LEN, dtype=np.float32)
            fitted[:min(UID137_STATE_LEN, state.size)] = state[:UID137_STATE_LEN]
            state = fitted
        state = np.ascontiguousarray(state)
        feeds = {'depth': depth, 'state': state, UID137_MEMORY_NAME: self._memory}
        feeds = {k: v for k, v in feeds.items() if k in self._input_names}
        try:
            outputs = self._session.run(None, feeds)
        except Exception:
            return _UID137_SAFE_ACTION.copy()
        named = dict(zip(self._output_names, outputs))
        action = np.asarray(named.get('action', outputs[0]), dtype=np.float32)
        memory_out = named.get('memory_tensor_out')
        if memory_out is not None:
            memory_out = np.asarray(memory_out, dtype=np.float32).reshape(-1)
            if memory_out.size == UID137_MEMORY_SHAPE[0] and np.isfinite(memory_out).all():
                self._memory = np.ascontiguousarray(memory_out)
        return _uid137_canonicalize(action)

class _KingStackSingleDrone:
    ROUTE_TO_V21 = ('mountain', 'warehouse')

    def __init__(self) -> None:
        self._v21 = _V21Controller()
        self._uid137 = _UID137Policy()
        self.route = 'uid137'

    def reset(self) -> None:
        self._v21.reset()
        self._uid137.reset()
        self.route = 'uid137'

    def _use_v21(self) -> bool:
        mtn = getattr(self._v21, 'mtn', None)
        if mtn is None:
            return False
        if bool(getattr(mtn, 'is_mountain', False)):
            return True
        return any((mtn.settled(label) for label in self.ROUTE_TO_V21))

    def act(self, observation) -> np.ndarray:
        a_v21 = self._v21.act(observation)
        a_137 = self._uid137.act(observation)
        if self._use_v21():
            self.route = 'v21'
            return np.asarray(a_v21, dtype=np.float32).reshape(6)
        self.route = 'uid137'
        return np.asarray(a_137, dtype=np.float32).reshape(6)

_FLATMOD_king_stack = _FlatModule(
    'king_stack',
    SIM_DT=SIM_DT, DEPTH_RES=DEPTH_RES, DEPTH_MIN_M=DEPTH_MIN_M,
    DEPTH_MAX_M=DEPTH_MAX_M, CAMERA_OFFSET_M=CAMERA_OFFSET_M,
    CAMERA_UP_OFFSET_M=CAMERA_UP_OFFSET_M, HALF_TAN=HALF_TAN,
    MAX_RAY_M=MAX_RAY_M, SPEED_LIMIT=SPEED_LIMIT, IDX_POS=IDX_POS,
    IDX_RPY=IDX_RPY, IDX_VEL=IDX_VEL, IDX_ACT_HIST=IDX_ACT_HIST,
    IDX_ALT=IDX_ALT, IDX_CLUE=IDX_CLUE, _repo_fallback=_repo_fallback,
    DET_NAME=DET_NAME, DET_FALLBACK=DET_FALLBACK, RGB_NAME=RGB_NAME,
    RGB_FALLBACK=RGB_FALLBACK, RGB_RES=RGB_RES,
    RGB_REQUEST_PERIOD=RGB_REQUEST_PERIOD,
    RGB_MIN_INTERVAL=RGB_MIN_INTERVAL,
    RGB_TARGET_TRIGGER=RGB_TARGET_TRIGGER, RGB_CAP=RGB_CAP,
    RGB_WEIGHT=RGB_WEIGHT, RGB_PRESENT_EPS=RGB_PRESENT_EPS,
    RGB_CONFIRM_THRESH=RGB_CONFIRM_THRESH, RANGE_VETO=RANGE_VETO,
    COLOUR_TRUST_M=COLOUR_TRUST_M, RGB_CONFIRM_WINDOW=RGB_CONFIRM_WINDOW,
    GAIN=GAIN, SIGMA_INNOVATION_M=SIGMA_INNOVATION_M, P_ACCEPT=P_ACCEPT,
    K_P=K_P, CONF_DECAY=CONF_DECAY, STRONG_P=STRONG_P, CONSIST_M=CONSIST_M,
    N_LATCH=N_LATCH, N_CAP=N_CAP, STALE_TICKS=STALE_TICKS,
    STALE_DECAY=STALE_DECAY, STRONG_GAIN=STRONG_GAIN, TRACK_GAIN=TRACK_GAIN,
    COMMIT_CONF=COMMIT_CONF, COMMIT_CLUTTER_BONUS=COMMIT_CLUTTER_BONUS,
    LOST_CONF=LOST_CONF, LATCH_CONF=LATCH_CONF,
    LOST_VIS_TICKS=LOST_VIS_TICKS, FREEZE_NEAR_M=FREEZE_NEAR_M,
    SEARCH_ALT_M=SEARCH_ALT_M, HOVER_ABOVE_TOP_M=HOVER_ABOVE_TOP_M,
    HOVER_SETTLE_MARGIN_M=HOVER_SETTLE_MARGIN_M, R_HOVER_M=R_HOVER_M,
    CONFIRM_SPEED=CONFIRM_SPEED, SPIRAL_R0_M=SPIRAL_R0_M,
    SPIRAL_RMAX_M=SPIRAL_RMAX_M, SPIRAL_GROW_TICKS=SPIRAL_GROW_TICKS,
    SPIRAL_ANGULAR=SPIRAL_ANGULAR, SPIRAL_PITCH_M=SPIRAL_PITCH_M,
    SPIRAL_SPEED_MPS=SPIRAL_SPEED_MPS, SPIRAL_ARC_RMAX_M=SPIRAL_ARC_RMAX_M,
    CTRL_HZ=CTRL_HZ, ARC_SPIRAL=ARC_SPIRAL,
    AVOID_LOOKAHEAD_M=AVOID_LOOKAHEAD_M, AVOID_SAFETY_M=AVOID_SAFETY_M,
    AVOID_STRENGTH=AVOID_STRENGTH, AVOID_SLOW_M=AVOID_SLOW_M,
    AVOID_MEMORY_DECAY=AVOID_MEMORY_DECAY,
    CLUTTER_EDGE_THRESH=CLUTTER_EDGE_THRESH,
    CLUTTER_EDGE_LO=CLUTTER_EDGE_LO, CLUTTER_EDGE_HI=CLUTTER_EDGE_HI,
    CLUTTER_ALT_BONUS_M=CLUTTER_ALT_BONUS_M, CLUTTER_EMA=CLUTTER_EMA,
    MOUNTAIN_SEARCH_ALT_M=MOUNTAIN_SEARCH_ALT_M,
    MTN_WP_RING_RADII=MTN_WP_RING_RADII, MTN_WP_PER_RING=MTN_WP_PER_RING,
    MTN_WP_REACH_M=MTN_WP_REACH_M, ALT_TD_K=ALT_TD_K, ALT_TD_D=ALT_TD_D,
    ALT_TD_VMAX=ALT_TD_VMAX, MTN_CLIMB_LOOKAHEAD_M=MTN_CLIMB_LOOKAHEAD_M,
    MTN_CLIMB_GAIN=MTN_CLIMB_GAIN, MTN_STEER_DAMP=MTN_STEER_DAMP,
    MTN_DET_NAME=MTN_DET_NAME, MTN_DET_FALLBACK=MTN_DET_FALLBACK,
    MTN_RGB_NAME=MTN_RGB_NAME, MTN_RGB_FALLBACK=MTN_RGB_FALLBACK,
    T_OLD_DEPTH=T_OLD_DEPTH, T_MTN_DEPTH=T_MTN_DEPTH, T_MTN_RGB=T_MTN_RGB,
    MAP_LABELS=MAP_LABELS, XGB_NAME=XGB_NAME, XGB_STATE_DIM=XGB_STATE_DIM,
    XGB_EVERY=XGB_EVERY, XGB_MIN_PRED=XGB_MIN_PRED,
    XGB_MTN_PROB=XGB_MTN_PROB, XGB_MAX_TICK=XGB_MAX_TICK,
    WAYPOINT_TERRAINS=WAYPOINT_TERRAINS,
    WAYPOINT_LATCH_PROB=WAYPOINT_LATCH_PROB,
    WAREHOUSE_SIDE_LATCH_PROB=WAREHOUSE_SIDE_LATCH_PROB,
    WAREHOUSE_SIDE_LATCH_MIN_PRED=WAREHOUSE_SIDE_LATCH_MIN_PRED,
    HIGH_WP_RING_RADII=HIGH_WP_RING_RADII,
    HIGH_WP_PER_RING=HIGH_WP_PER_RING, HIGH_WP_REACH_M=HIGH_WP_REACH_M,
    WAYPOINT_FLAT_COMMIT=WAYPOINT_FLAT_COMMIT,
    HOVER_GIVEUP_TICKS=HOVER_GIVEUP_TICKS, HOVER_STABLE_M=HOVER_STABLE_M,
    PHANTOM_VETO_M=PHANTOM_VETO_M, RGB_VETO_TTL=RGB_VETO_TTL,
    FOREST_ALT_M=FOREST_ALT_M, FOREST_CLUTTER_MIN=FOREST_CLUTTER_MIN,
    FOREST_EARLY_TICK=FOREST_EARLY_TICK,
    FOREST_EARLY_PROB=FOREST_EARLY_PROB, FOREST_EARLY_WH=FOREST_EARLY_WH,
    FOREST_MIN_TICK=FOREST_MIN_TICK, FOREST_LATE_PROB=FOREST_LATE_PROB,
    FOREST_SOLO_CLUTTER=FOREST_SOLO_CLUTTER, FOREST_WH_VETO=FOREST_WH_VETO,
    FOREST_SPEED=FOREST_SPEED, FOREST_GROUND_TRACK_M=FOREST_GROUND_TRACK_M,
    URBAN_LOW=URBAN_LOW, URBAN_ALT_M=URBAN_ALT_M,
    CENTRE_REACH_M=CENTRE_REACH_M, CENTRE_MAX_TICKS=CENTRE_MAX_TICKS,
    INVESTIGATE_M=INVESTIGATE_M, GROUND_GATE_M=GROUND_GATE_M,
    INVESTIGATE_FACTOR=INVESTIGATE_FACTOR,
    HOVER_DESCENT_SPEED=HOVER_DESCENT_SPEED,
    HOVER_DESCENT_M=HOVER_DESCENT_M, STAR_MODE=STAR_MODE,
    STAR_RADIUS_M=STAR_RADIUS_M, _star_env=_star_env,
    STAR_ORDER_DEG=STAR_ORDER_DEG, STAR_REACH_M=STAR_REACH_M,
    HOVER_GIVEUP_BAND=HOVER_GIVEUP_BAND,
    HOVER_GIVEUP_FALLBACK_TICKS=HOVER_GIVEUP_FALLBACK_TICKS,
    _recalibrate=_recalibrate, camera_axes=camera_axes,
    measurement_world=measurement_world, _unit=_unit,
    _map_depth_2d=_map_depth_2d,
    _map_depth_feature_values=_map_depth_feature_values,
    _XGBMapPredictor=_XGBMapPredictor,
    MountainClassifier=MountainClassifier, AltitudeTD=AltitudeTD,
    VictimDetector=VictimDetector, ObstacleAvoider=ObstacleAvoider,
    TrackingDifferentiator=TrackingDifferentiator,
    _V21Controller=_V21Controller, UID137_MODEL=UID137_MODEL,
    UID137_MEMORY_NAME=UID137_MEMORY_NAME,
    UID137_MEMORY_SHAPE=UID137_MEMORY_SHAPE,
    UID137_STATE_LEN=UID137_STATE_LEN,
    UID137_DEPTH_SHAPE=UID137_DEPTH_SHAPE,
    ACTION_QUANT_STEP=ACTION_QUANT_STEP,
    RGB_REQUEST_THRESHOLD=RGB_REQUEST_THRESHOLD, ACTION_LOWER=ACTION_LOWER,
    ACTION_UPPER=ACTION_UPPER, RGB_REQUEST_INDEX=RGB_REQUEST_INDEX,
    ORT_INTRA_OP_THREADS=ORT_INTRA_OP_THREADS,
    ORT_INTER_OP_THREADS=ORT_INTER_OP_THREADS,
    _UID137_SAFE_ACTION=_UID137_SAFE_ACTION,
    _uid137_session_options=_uid137_session_options,
    _uid137_canonicalize=_uid137_canonicalize, _UID137Policy=_UID137Policy,
    _KingStackSingleDrone=_KingStackSingleDrone)

import dis
import math
import types

_HERE = os.path.dirname(os.path.abspath(__file__))

LOCAL_R_M = float(os.environ.get("SWSAR_V21_LOCAL_R", "5.0"))
LOCAL_R0_M = float(os.environ.get("SWSAR_V21_LOCAL_R0", "1.5"))
LOCAL_GROW_TICKS = float(os.environ.get("SWSAR_V21_GROW", "250"))
LOCAL_ANGULAR = float(os.environ.get("SWSAR_V21_ANG", "0.012"))
LOCAL_REACH_M = float(os.environ.get("SWSAR_V21_REACH", "3.5"))
TRANSIT_MAX_TICKS = float(os.environ.get("SWSAR_V21_TRANSIT", "900"))
WP_RESET_M = 0.75
STANDOFF_R_M = 7.0
STANDOFF_SPEED = 0.6
STANDOFF_TTL = 900
SOFT_DENY_TICKS = 500
TEAM_DENY_FIX = True
TEAM_DENY_PRIVATE_TICKS = 600
TEAM_COOLDOWN_EXEMPT = True
TEAM_SUPERSEDE = True
TEAM_SUPERSEDE_LOCK = 0.9499
TEAM_SUPERSEDE_FROZEN_TICKS = 300
TEAM_SUPERSEDE_STALL_TICKS = 150
TEAM_SUPERSEDE_MAX = 2

# --- FAR-RANGE FIXES (DC, 2026-09-21): see tools/patch_dc.py
DC_ON = os.environ.get('SWSAR_DC', '1') == '1'          # mountain only: on village the pixel lands on walls/roof edges (b1-92 0.89 -> 0.01)
DC_MIN_R = float(os.environ.get('SWSAR_DC_MINR', '8.0'))
DC_BAND = os.environ.get('SWSAR_DC_BAND', '1') == '1'

# --- PILOT FIXES (P1, 2026-09-21): see tools/patch_p1.py
P1_TKS = os.environ.get('SWSAR_P1_TKS', '1') == '1'
P1_RAMPK = float(os.environ.get('SWSAR_P1_RAMPK', '0.25'))   # locked approach/hover ramp gain (champion 0.125 = 0.375*dist m/s); 0.25: -0.84 s per late confirm, 21/22 seeds same or better, 0 new collisions
P1_ADV = os.environ.get('SWSAR_P1_ADV', '1') == '1'
P1_ADV_R = float(os.environ.get('SWSAR_P1_ADVR', '4.0'))
P1_ADV_HOLD = float(os.environ.get('SWSAR_P1_ADVHOLD', '0.0'))   # dwell kept at a reached waypoint (0 = snap the whole leg)
P1_MSW = os.environ.get('SWSAR_P1_MSW', '1') == '1'
P1_U66 = os.environ.get('SWSAR_P1_U66', '1') == '1'
P1_TKS_OUT = float(os.environ.get('SWSAR_P1_TKSOUT', '300.0'))     # free-sweep degrees for drones with ~no posterior mass near the pad (0 = champion)
P1_TKS_OUT_MAPS = tuple(x for x in os.environ.get('SWSAR_P1_TKSOUT_MAPS', 'city').split(',') if x)   # village b2-241/305/339 broke with the long sweep
P1_PAD_TRIG = float(os.environ.get('SWSAR_P1_PADTRIG', '0.15'))    # Posterior: relaxed 1.5x box while pad overlap < this (champion 0.15)

# --- TEAM ELECTION (TEL, 2026-09-21): see tools/patch_team.py
TEL_SUPERSEDE_ALL = os.environ.get('SWSAR_TEL_SUP', '1') == '1'
TEL_HANDOVER_TICKS = int(os.environ.get('SWSAR_TEL_HO', '150'))
TEL_HANDOVER_GAIN = float(os.environ.get('SWSAR_TEL_GAIN', '1.5'))
TEL_HANDOVER_R = float(os.environ.get('SWSAR_TEL_HOR', '12.0'))
TEL_HANDOVER_MAX = int(os.environ.get('SWSAR_TEL_HOMAX', '1'))
TEL_LOCKGUARD = os.environ.get('SWSAR_TEL_LG', '1') == '1'
TEL_LG_N = float(os.environ.get('SWSAR_TEL_LGN', '6.0'))     # a re-pointed track this strong is followed (N_CAP=14 held 38: true person seen mid-approach)

# --- TEAM ELECTION v3 (TEL2, 2026-09-21): see tools/patch_team2.py
TEL2_E1 = os.environ.get('SWSAR_TEL2_E1', '1') == '1'
TEL2_E2 = os.environ.get('SWSAR_TEL2_E2', '1') == '1'
TEL2_E3 = os.environ.get('SWSAR_TEL2_E3', '1') == '1'
TEL2_E4 = os.environ.get('SWSAR_TEL2_E4', '1') == '1'
TEL2_E7 = os.environ.get('SWSAR_TEL2_E7', '1') == '1'
TEL2_E8 = os.environ.get('SWSAR_TEL2_E8', '1') == '1'
TEL2_E5 = os.environ.get('SWSAR_TEL2_E5', '1') == '1'
TEL2_E6 = os.environ.get('SWSAR_TEL2_E6', '1') == '1'
TEL2_SELF_M = float(os.environ.get('SWSAR_TEL2_SELFM', '6.0'))
TEL_SUP_GUARD = os.environ.get('SWSAR_TEL_SUPG', '1') == '1'   # never supersede a confirmer that is physically at its spot (b1-452)
TEL_SUP_QUAL = os.environ.get('SWSAR_TEL_SUPQ', '1') == '1'    # a challenger record must be comparable to the candidate (b1-432: 1-drone 87-hit rock beat a 3-drone 441-hit person)
TEAM_BLACKLIST_DENY_TICKS = 3000
XGB_OWNER_STALE = 25

OUTER_ODDS = 9.17
# v600 PERSISTENT-EVIDENCE OVERRIDE of the posterior-support veto. Traces on the champion base
# (pd_mountain.json): in 6/25 failed mountain seeds the depth net reports p>=0.7 on 100-900 frames
# with the estimate within 1 m of the true victim, no RGB veto, and strong stays 0 — the victim
# spawned via the validator's >80 m / out-of-box fallback (16/48 mountain victims in the audit),
# so it sits outside the team's support and _gated_refuted throws every frame away (hard veto, or
# the 0.955 outer bar). Override: a detection that stays strong and spatially consistent for
# OUTER_STREAK consecutive detector frames is admitted regardless of support.
OUTER_STREAK = int(os.environ.get('SAR_OUTER_STREAK', '12'))
OUTER_STREAK_MAPS = tuple(os.environ.get('SAR_OUTER_STREAK_MAPS', 'mountain,city,forest,village,open').split(','))
def _outer_bar(strong_p, odds):
    o = odds * (strong_p / max(1e-9, 1.0 - strong_p))
    return float(o / (1.0 + o))
DT = 1.0 / 50.0
MEM_SIZE = 80
ACT_DIM = 6

# --- c225.swarmsar.101 FOREST TAKE-OFF LOOK-AROUND (user, 2026-09-16) -- forest world only.
# In the champion's forest failures the person is within 20 m of a drone's start in 23 of 40 seeds
# (within 5 m in 6), and the forward-only camera never looks behind the drone at take-off. The
# 2026-09-14 version (candidate 103: a 1-turn / 2 s spin DURING the climb) rescued 20 forest seeds but
# broke 31: 7 were the champion's no-detection rescues (the person under the take-off hover, the
# spinning climb broke the 2 s hold) and most others were early catches where the fast spin smeared
# the depth track and delayed the approach. This version: a drone that has seen nothing looks around
# ONCE, after the climb has settled (from SPIN_T0), slowly (SPIN_RATE deg/s), HOLDING POSITION at the
# deck (a still hover over the start is exactly the no-detection rescue), and the turn stops the
# instant its depth head fires strong, it locks, or the team holds a candidate.
SPIN_ON = os.environ.get('SWSAR101_SPIN', '0') == '1'   # forest lane 2026-09-16: 376 vs 379 (16 rescued / 19 broken) -> off by default

# --- c225.swarmsar.101 FOREST GAP-SEEKING HOVER (user, 2026-09-16) -- forest world only.
# 9 of the champion's 40 forest failures had a drone within 3 m of the person for >= 2 s but 4-18 m
# above the head: the drone climbed over the canopy, the downward ray reads a crown, and the 2.5 m
# AGL floor blocks the descent (250716 18.5 m, 250177 12.2, 250489 9.8, 251635 7.2, 250469 5.9,
# 250854 4.7, 251754 4.5, 250637 4.4, 251936 4.0). Descending blindly through a crown is a
# collision (the 1 m search lost 33 seeds), so this lever moves the hover point sideways instead:
# a drone hovering > GAP_HIGH_M above the estimated head with the ray reading a crown (AGL <=
# GAP_AGL_MAX) visits GAP_N points on a GAP_RING_M ring around the estimate (plus the estimate
# itself), at its current altitude, and where the ray reads >= GAP_CLEAR_M of clear air below it
# adopts that point as the hover spot -- the champion's own hover then descends there. One try per
# lock; the AGL floor is never relaxed.
GAP_ON = os.environ.get('SWSAR101_GAP', '0') == '1'     # never fired on the too-high seeds: the ray below was CLEAR (see LAD_*)
# What the too-high hovers really are (instrumented hub runs of 250489/250854/251635/250469): the ray
# below is clear; the drone either (a) descends slowly from the 10 m no-latch deck (250489: 14.5 m
# altitude at lock, 1.3 m/s down), (b) hovers at estimated-head + 2.5 with the head estimate 1.8 m
# too high (250854: hover 4.3-4.6 m over the real head, the band ends at 4 m), or (c) is not locked at
# all (searching at 7-11 m, never saw the person). So instead of a sideways gap search: a LADDER.
# A locked, frozen drone hovering within 1.5 m of its estimate with the ray clear (>= LAD_AGL_MIN)
# and no rescue after LAD_WAIT ticks lowers its head estimate by LAD_STEP every LAD_EVERY ticks, at
# most LAD_MAX in all -- the champion's own hover follows it down; the AGL floor still protects.
LAD_ON = os.environ.get('SWSAR101_LAD', '0') == '1'   # forest lane: v1 374 vs 379 (3/8), v2 376 vs 379 (1/4) -> off
LAD_WAIT = int(os.environ.get('SWSAR101_LAD_WAIT', '100'))
LAD_EVERY = int(os.environ.get('SWSAR101_LAD_EVERY', '50'))
LAD_STEP = float(os.environ.get('SWSAR101_LAD_STEP', '0.5'))
LAD_MAX = float(os.environ.get('SWSAR101_LAD_MAX', '2.5'))
LAD_AGL_MIN = float(os.environ.get('SWSAR101_LAD_AGL', '2.5'))
# v1 lane (419 forest seeds): 374 vs 379 = -5 (3 rescued / 8 broken). Every break diverged at the first
# ladder step of a hover over a PHANTOM: lowering the head estimate takes the drone out of the
# champion's give-up band [1.5, 4.5] above the (lowered) head, its stable-hover clock stops, and the
# drone sits on the phantom for up to 24 s instead of 6 -- the late catches were lost. v2: the give-up
# clock keeps running against the ORIGINAL head estimate, the ladder lasts at most LAD_TOTAL ticks and
# then restores the original estimate so the champion's give-up proceeds unchanged.
LAD_TOTAL = int(os.environ.get('SWSAR101_LAD_TOTAL', '300'))    # 6 s of ladder at most

# --- FOREST APPROACH CHECK WITH A SKIN-COLOUR VOTE (user, 2026-09-17) -- forest world only.
# Measured on 8 forest seeds replayed in-process (220 colour frames with a depth candidate in the
# picture): the skin-pixel fraction of a crop around the depth detection is 2-15 % on the person
# (median 5.6 %) and 0 on most phantoms, but brown bark / soil floods the crop on a quarter of them,
# so the vote is a BAND: skin in [SKN_LO, SKN_HI] -> yes. Per frame: person 74 %, phantom 8 %; the
# champion's colour net on the same frames says yes on 33 % / 3.5 %. Either one = a yes vote. The
# heads' height output does not separate (phantoms are person-sized by construction).
# The check runs ONLY while a drone flies toward its own locked spot (navigation mode): bars, the
# hover and the champion's colour-frame schedule are untouched (those three lost every earlier forest
# test). Verdict phantom = >= SKN_MIN_LOOKS looks with <= 1/3 yes AND the depth head silent on the spot
# (SKN_SILENT in-view served frames with no hit, or SKN_CLOSE_SILENT such frames inside 8 m) -> drop
# the lock, ignore the spot for SKN_MEMORY ticks, keep searching. Verdict person = >= 2/3 yes or 3
# depth hits inside 8 m -> stop checking (no behaviour change).
# --- NO-WAIT STAND-OFF (user, 2026-09-17) -- forest only. A drone that holds a lock on the team's
# candidate but is not the confirmer stands off at 7 m for up to STANDOFF_TTL = 18 s before it
# soft-denies the spot and resumes its tour. Measured on the champion's flights: 3.3 s of such
# waiting per forest WIN and 6.8 s per forest FAIL (24 % of forest seeds > 5 s; 2.1 s of it at a
# phantom), 0.5-1.2 s elsewhere. Here the wait ends after NW_TTL ticks: the spot is soft-denied for
# NW_DENY ticks and the drone searches on (it can still be elected if it re-sees the person).
# --- CO-HOVER (user, 2026-09-17): "max 2 drones can try hovering, only when two drones are already near
# the strong point". Measured on the champion's flights: failures where the confirmer hovered AT the
# person >= 2 s and still failed = 14 of 111, of which 6 had a second drone within 12 m; seeds with a
# >= 2 s PHANTOM hover and a second drone within 12 m = 122 (forest 57). So the second hover is gated
# hard: it starts only after the confirmer has been on the candidate for COH_WAIT ticks without a
# rescue (a converting hover never needs it), only for a drone that already stands off within
# COH_NEAR m holding its own lock on the same spot, at most one co-hoverer, offset COH_OFF m from the
# candidate towards its own side and COH_DZ m LOWER than the confirmer (a different height is the
# point: the first hover fails high or off), for at most COH_MAX ticks, then it soft-denies and
# leaves. The team's keep-out and separation exempt the co-hoverer; a 2 m 3-D guard against the
# confirmer is kept in the pilot.
COH_ON = os.environ.get('SWSAR101_COH', '1') == '1'
COH_WAIT = int(os.environ.get('SWSAR101_COH_WAIT', '200'))
COH_NEAR = float(os.environ.get('SWSAR101_COH_NEAR', '12.0'))
COH_OFF = float(os.environ.get('SWSAR101_COH_OFF', '1.2'))
COH_DZ = float(os.environ.get('SWSAR101_COH_DZ', '1.2'))
COH_MAX = int(os.environ.get('SWSAR101_COH_MAX', '400'))
COH_GUARD = 2.0
COH_SPEED = 0.45
NW_ON = os.environ.get('SWSAR101_NW', '0') == '1'   # forest lane: 368 vs 379 (9/20) -> off: the waiting drone is the backup the team elects next
NW_TTL = int(os.environ.get('SWSAR101_NW_TTL', '50'))
NW_DENY = int(os.environ.get('SWSAR101_NW_DENY', '500'))
SKN_ON = os.environ.get('SWSAR101_SKN', '1') == '1'
SKN_MAPS = tuple(x for x in os.environ.get('SWSAR101_SKN_MAPS', 'forest,village').split(',') if x)
SKN_PERSON_H = {'city': 1.1, 'village': 1.1}          # the builder scales the characters 0.6x on city/village
SKN_LO = float(os.environ.get('SWSAR101_SKN_LO', '0.02'))
SKN_HI = float(os.environ.get('SWSAR101_SKN_HI', '0.15'))
SKN_RGB_M = 14.0
SKN_MIN_LOOKS = int(os.environ.get('SWSAR101_SKN_LOOKS', '3'))
SKN_YES = 0.66
SKN_NO = 0.34
SKN_SILENT = int(os.environ.get('SWSAR101_SKN_SILENT', '10'))
SKN_CLOSE_SILENT = int(os.environ.get('SWSAR101_SKN_CLOSE_SILENT', '20'))
SKN_REQ_GAP = 12
SKN_REQ_MAX = 8
SKN_MEMORY = int(os.environ.get('SWSAR101_SKN_MEMORY', '500'))
SKN_FOV = 0.9
SKN_OCC_M = 3.0


def _skin_frac(img):
    """Fraction of skin-coloured pixels (Peer RGB rule AND a YCbCr box) in an RGB crop."""
    a = np.asarray(img, dtype=np.float64)
    if a.ndim != 3 or a.shape[2] < 3 or a.size == 0:
        return 0.0
    if a.max() <= 1.0:
        a = a * 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    m = ((r > 95) & (g > 40) & (b > 20) & ((mx - mn) > 15) & (np.abs(r - g) > 15) & (r > g) & (r > b)
         & (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173))
    return float(m.mean())
GAP_HIGH_M = float(os.environ.get('SWSAR101_GAP_HIGH', '4.5'))
GAP_AGL_MAX = float(os.environ.get('SWSAR101_GAP_AGL', '3.0'))
GAP_CLEAR_M = float(os.environ.get('SWSAR101_GAP_CLEAR', '5.0'))
GAP_RING_M = float(os.environ.get('SWSAR101_GAP_RING', '1.5'))
GAP_N = int(os.environ.get('SWSAR101_GAP_N', '8'))
GAP_WAIT = int(os.environ.get('SWSAR101_GAP_WAIT', '100'))      # ticks too high before the search starts (2 s)
GAP_VISIT = int(os.environ.get('SWSAR101_GAP_VISIT', '75'))     # ticks per candidate at most (1.5 s)
GAP_SPEED = float(os.environ.get('SWSAR101_GAP_SPEED', '0.4'))
GAP_REACH_M = 0.7
SPIN_T0 = int(os.environ.get('SWSAR101_SPIN_T0', '200'))
SPIN_RATE = float(os.environ.get('SWSAR101_SPIN_RATE', '75.0'))      # deg/s -> one turn in 4.8 s
SPIN_TURNS = float(os.environ.get('SWSAR101_SPIN_TURNS', '1.0'))
SPIN_STOP_P = float(os.environ.get('SWSAR101_SPIN_STOP_P', '0.70'))  # a fresh depth output at/above this ends the turn
SPIN_MAX_T = int(os.environ.get('SWSAR101_SPIN_MAX_T', '500'))       # never start a turn after this tick (10 s)


def _forest():
    return (SPIN_ON or GAP_ON or LAD_ON) and _MAPSW_STATE.get("map") == 'forest'


def _skin_world():
    return SKN_ON and _MAPSW_STATE.get("map") in SKN_MAPS

_LOGGED_V21 = set()

def _log_v21(key, msg):
    if key in _LOGGED_V21 or len(_LOGGED_V21) > 32:
        return
    _LOGGED_V21.add(key)
    try:
        sys.stderr.write("[swarm_sar.v21] %s\n" % msg)
        sys.stderr.flush()
    except Exception:
        pass

def _load_king(*_args, **_kwargs):
    return _FLATMOD_king_stack

_ks = _load_king()

def _store_attrs(fn):
    try:
        out = set()
        last = None
        for ins in dis.get_instructions(fn):
            if ins.opname == "STORE_ATTR":
                if last == "self":
                    out.add(ins.argval)
            elif ins.opname in ("LOAD_FAST", "LOAD_DEREF", "LOAD_FAST_CHECK",
                                "LOAD_NAME", "LOAD_GLOBAL"):
                last = ins.argval
        return out
    except Exception:
        return set()

_CTRL_ATTRS = _store_attrs(_ks._V21Controller.__init__)
_DET_ATTRS = _store_attrs(_ks.VictimDetector.__init__)
_DET_SESSION_KEYS = ("sess", "iname", "rgb_sess", "rgb_iname", "lr_sess", "lr_iname",
                     "mtn_sess", "mtn_iname", "mtn_rgb_sess", "mtn_rgb_iname",
                     "nd_sess", "nd_calls", "nd_used")   # NEWDET: keep the shared-session assembly path
_CTRL_ASSEMBLED = ("detector", "avoider", "smoother", "mtn", "alt_td")
_ASSEMBLY_OK = (set(_CTRL_ASSEMBLED) >= _CTRL_ATTRS
                and set(_DET_SESSION_KEYS) >= _DET_ATTRS)
if not _ASSEMBLY_OK:
    _log_v21("assembly",
         "king_stack ctor attributes moved (ctrl=%r det=%r); building private "
         "sessions per drone" % (sorted(_CTRL_ATTRS), sorted(_DET_ATTRS)))

_EMPTY_RGB = np.zeros(0, dtype=np.float32)
_ZERO6 = np.zeros(6, dtype=np.float32)

class _MapBus:

    def __init__(self, real):
        self.real = real
        self.owner = 0
        self.last_t = -10 ** 9
        self.serves = 0
        self._next_id = 0

    def claim_id(self):
        i = self._next_id
        self._next_id += 1
        return i

    def reset(self):
        self.real.reset()
        self.owner = 0
        self.last_t = -10 ** 9
        self.serves = 0

    def update(self, uid, tick, state, depth):
        if uid != self.owner:
            if tick - self.last_t < XGB_OWNER_STALE:
                return
            self.owner = uid
        self.last_t = tick
        self.serves += 1
        self.real.update(tick, state, depth)

    def p_env(self):
        r = self.real
        if int(r.count) <= 0:
            return _ZERO6.copy()
        p = np.asarray(r.prob_sum / float(r.count), dtype=np.float32).reshape(-1)
        return p[:6] if p.size >= 6 else _ZERO6.copy()

class _MtnView:

    __slots__ = ("bus", "uid")

    def __init__(self, bus, uid):
        self.bus = bus
        self.uid = int(uid)

    def update(self, tick, state, depth):
        self.bus.update(self.uid, tick, state, depth)

    def reset(self):
        return None

    def settled(self, label):
        return self.bus.real.settled(label)

    def prob_of(self, label):
        return self.bus.real.prob_of(label)

    @property
    def is_mountain(self):
        return self.bus.real.is_mountain

    @property
    def label(self):
        return self.bus.real.label

    @property
    def latched(self):
        return self.bus.real.latched

    @property
    def count(self):
        return int(self.bus.real.count)

    @property
    def pred(self):
        return self.bus.real.pred

def _swarm_search_policy(self, pos, clue, ground_z):
    centre = pos[0:2] + clue
    z = self._target_z(ground_z)
    speed = _ks.FOREST_SPEED if self.forest_low else 1.0
    if not self.centre_visited:
        if self.search_t0 is None:
            self.search_t0 = self.tick
        if (float(math.hypot(clue[0], clue[1])) < LOCAL_REACH_M
                or (self.tick - self.search_t0) >= TRANSIT_MAX_TICKS):
            self.centre_visited = True
            self.spiral_t0 = self.tick
        else:
            return (np.array([centre[0], centre[1], z]), speed)
    scale = float(getattr(self, "_swarm_r_scale", 1.0))
    ts = max(0, self.tick - self.spiral_t0)
    r_max = max(LOCAL_R0_M, LOCAL_R_M * scale)
    rad = LOCAL_R0_M + (r_max - LOCAL_R0_M) * min(ts / max(LOCAL_GROW_TICKS, 1.0), 1.0)
    ang = ts * LOCAL_ANGULAR * (_ks.FOREST_SPEED if self.forest_low else 1.0)
    return (np.array([centre[0] + rad * math.cos(ang),
                      centre[1] + rad * math.sin(ang), z]), speed)

def _to_np(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        try:
            return x.detach().cpu().numpy()
        except Exception:
            pass
    return np.asarray(x)

def _depth2d(depth):
    a = np.asarray(_to_np(depth), dtype=np.float32)
    if a.size == 256 * 256 and a.shape != (256, 256):
        a = a.reshape(256, 256)
    return a

def _rgb_present(rgb):
    if rgb is None:
        return False
    a = np.asarray(_to_np(rgb), dtype=np.float32)
    if a.size < 3:
        return False
    try:
        return bool(float(np.abs(a.reshape(256, 256, 3)[::16, ::16]).mean())
                    >= _ks.RGB_PRESENT_EPS)
    except Exception:
        return True

def _unit_or(vec, fallback):
    n = float(math.hypot(vec[0], vec[1]))
    if n < 1e-9:
        return fallback
    return (vec[0] / n, vec[1] / n)


# --- OBSTACLE-MEMORY BARRIER (OMB, 2026-09-20). Measured on the champion: forest crashes happen in search
# mode against a canopy/trunk that has just left the 90-degree camera cone (the avoider's memory decays in
# 0.4 s and the next waypoint turn steers into it); 28 %% of forest seeds lose a drone, 17 of 19 forest
# failures involve a crash, and forest safety averages 0.58. Here every near depth point is back-projected
# and remembered for OMB_MEM_S seconds, and the commanded velocity is constrained so the approach speed
# toward any remembered point q at distance r never exceeds OMB_K * (r - OMB_R_SAFE).
OMB_ON = os.environ.get('SWSAR_OMB', '1') == '1'
# village added to the default: the obstacle-memory velocity barrier was never active on village,
# where 14/423 seeds executed a perfect rescue and then forfeited the whole 0.10 safety term to a
# 1-4 cm graze (min_clearance -0.001..-0.037). Barrier r_safe 1.0 m == SAFETY_DISTANCE_SAFE, so it
# targets exactly the clearance the score rewards. Measured village +0.0102 (n=196, 82 up / 2 down,
# collisions 9->1). Other maps keep their existing membership, so their behaviour is unchanged.
OMB_MAPS = tuple(x for x in os.environ.get('SWSAR_OMB_MAPS', 'forest,city,village').split(',') if x)
OMB_R_SAFE = float(os.environ.get('SWSAR_OMB_RSAFE', '1.0'))
OMB_K = float(os.environ.get('SWSAR_OMB_K', '1.5'))
OMB_SEE_M = float(os.environ.get('SWSAR_OMB_SEE', '6.0'))
OMB_MEM_S = float(os.environ.get('SWSAR_OMB_MEM', '10.0'))
OMB_INFL_M = float(os.environ.get('SWSAR_OMB_INFL', '3.5'))
OMB_CAP_NEAR = float(os.environ.get('SWSAR_OMB_CAP_NEAR', '2.0'))    # total speed cap engages inside this
OMB_CAP_MIN = float(os.environ.get('SWSAR_OMB_CAP_MIN', '0.3'))      # never below this fraction of full speed
OMB_KEEP_M = float(os.environ.get('SWSAR_OMB_KEEP', '15.0'))     # forget points farther than this
OMB_BLIND_V = float(os.environ.get('SWSAR_OMB_BLIND_V', '1.0'))  # m/s cap when moving > BLIND_DEG off the camera axis
OMB_BLIND_DEG = float(os.environ.get('SWSAR_OMB_BLIND_DEG', '120.0'))   # reversing only: 60 deg caught the champion's own spiral turns (1000-2300 ticks/seed)
OMB_BLIND_MAPS = tuple(x for x in os.environ.get('SWSAR_OMB_BLIND_MAPS', 'forest').split(',') if x)
OMB_R_SAFE_BY_MAP = {'forest': float(os.environ.get('SWSAR_OMB_RSAFE_FOREST', '0.75')),
                     'city': float(os.environ.get('SWSAR_OMB_RSAFE_CITY', '1.2'))}   # best_v3: city rooftop grazes 4->1 (+0.004 pooled, 460 eps); village 1.2 was -0.012 on fresh seeds
OMB_SKIP_NAV_MAPS = tuple(x for x in os.environ.get('SWSAR_OMB_SKIP_NAV', '').split(',') if x)   # NEWDET: maps where the barrier is off during 'navigation'
if os.environ.get('SWSAR_OMB_RSAFE_MOUNTAIN'):
    OMB_R_SAFE_BY_MAP['mountain'] = float(os.environ['SWSAR_OMB_RSAFE_MOUNTAIN'])   # NEWDET: mountain-specific barrier radius
OMB_SLIDE_V = float(os.environ.get('SWSAR_OMB_SLIDE_V', '1.2'))   # tangential speed when the barrier blocks the command
OMB_SLIDE_HOLD = 50                                              # ticks a slide side is kept (hysteresis)
OMB_STUCK_TICKS = int(os.environ.get('SWSAR_OMB_STUCK', '150'))  # no progress this long -> barrier off for STUCK_FREE
OMB_STUCK_FREE = 100
OMB_VOX = 0.4
OMB_BLOCK = 8
OMB_ALL_MAPS = os.environ.get('SWSAR_OMB_ALLMAPS', '0') == '1'

# --- FOREST BLOCKED-APPROACH DETOUR (FBD, 2026-09-21): see tools/patch_fbd.py
FBD_ON = os.environ.get('SWSAR_FBD', '1') == '1'
FBD_MAPS = tuple(x for x in os.environ.get('SWSAR_FBD_MAPS', 'forest').split(',') if x)
FBD_STALL = int(os.environ.get('SWSAR_FBD_STALL', '100'))
FBD_NEAR = float(os.environ.get('SWSAR_FBD_NEAR', '2.0'))
FBD_MAX_H = float(os.environ.get('SWSAR_FBD_MAXH', '25'))
FBD_MIN_H = float(os.environ.get('SWSAR_FBD_MINH', '2.0'))
FBD_HOLD = int(os.environ.get('SWSAR_FBD_HOLD', '200'))
FBD_V = float(os.environ.get('SWSAR_FBD_V', '1.2'))
FBD_MAX = int(os.environ.get('SWSAR_FBD_MAX', '2'))
FBD_PROBE = float(os.environ.get('SWSAR_FBD_PROBE', '2.5'))
_OMB_GRID = None


def _omb_grid():
    global _OMB_GRID
    if _OMB_GRID is None:
        nb = DEPTH_RES // OMB_BLOCK
        # pixel centre -> normalised image coords (u right, v up), as measurement_world expects
        px = (np.arange(DEPTH_RES, dtype=np.float64) + 0.5) / DEPTH_RES * 2.0 - 1.0
        u = np.broadcast_to(px[None, :], (DEPTH_RES, DEPTH_RES))
        v = np.broadcast_to(-px[:, None], (DEPTH_RES, DEPTH_RES))
        _OMB_GRID = (nb, np.ascontiguousarray(u), np.ascontiguousarray(v))
    return _OMB_GRID


def _omb_world():
    m = _MAPSW_STATE.get("map")
    if OMB_ALL_MAPS:
        return True
    return m in OMB_MAPS


class _ObstacleMemory:
    """Voxelised memory of near depth points, with a velocity barrier against them."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.keys = np.zeros(0, dtype=np.int64)     # voxel ids (ix, iy, iz packed)
        self.pts = np.zeros((0, 3), dtype=np.float64)
        self.seen_t = np.zeros(0, dtype=np.int64)
        self.tick = 0
        self.stats = {"omb_pts": 0, "omb_ticks": 0, "omb_limited": 0, "omb_capped": 0, "omb_min_r": 99.0}
        self.slide_side = 0.0
        self.slide_until = -1
        self.prog_pos = None
        self.prog_tick = 0
        self.free_until = -1
        self._P = None
        self._P_tick = -1

    @property
    def vox(self):
        return self.keys.size

    def observe(self, depth, pos, rpy):
        self.tick += 1
        nb, U, V = _omb_grid()
        d = np.asarray(depth, dtype=np.float32).reshape(DEPTH_RES, DEPTH_RES)
        dm = d * (DEPTH_MAX_M - DEPTH_MIN_M) + DEPTH_MIN_M
        blk = dm.reshape(nb, OMB_BLOCK, nb, OMB_BLOCK).transpose(0, 2, 1, 3).reshape(nb, nb, OMB_BLOCK * OMB_BLOCK)
        arg = blk.argmin(axis=2)
        mn = np.take_along_axis(blk, arg[:, :, None], axis=2)[:, :, 0]
        sel = mn <= OMB_SEE_M
        if sel.any():
            by, bx = np.nonzero(sel)
            a = arg[by, bx]
            py = by * OMB_BLOCK + a // OMB_BLOCK
            px = bx * OMB_BLOCK + a % OMB_BLOCK
            cz = mn[by, bx].astype(np.float64)
            u = U[py, px]
            v = V[py, px]
            fwd, up, right = camera_axes(rpy)
            cam = np.asarray(pos, dtype=np.float64) + fwd * CAMERA_OFFSET_M + up * CAMERA_UP_OFFSET_M
            pts = cam[None, :] + right[None, :] * (u * cz)[:, None] + up[None, :] * (v * cz)[:, None] + fwd[None, :] * cz[:, None]
            ijk = np.floor(pts / OMB_VOX).astype(np.int64) + 500000
            keys = (ijk[:, 0] * 1000003 + ijk[:, 1]) * 1000003 + ijk[:, 2]
            # merge: new points win over remembered ones with the same voxel id
            allk = np.concatenate([keys, self.keys])
            allp = np.concatenate([pts, self.pts])
            allt = np.concatenate([np.full(keys.size, self.tick, dtype=np.int64), self.seen_t])
            _, first = np.unique(allk, return_index=True)
            self.keys, self.pts, self.seen_t = allk[first], allp[first], allt[first]
            self.stats["omb_pts"] += int(len(pts))
        self._prune(pos)

    def _prune(self, pos=None):
        if self.tick % 25 or self.keys.size == 0:
            return
        ttl = int(OMB_MEM_S * CTRL_HZ)
        keep = (self.tick - self.seen_t) <= ttl
        if pos is not None:
            keep &= np.hypot(self.pts[:, 0] - float(pos[0]), self.pts[:, 1] - float(pos[1])) <= OMB_KEEP_M
        if not keep.all():
            self.keys, self.pts, self.seen_t = self.keys[keep], self.pts[keep], self.seen_t[keep]

    def constrain(self, a, pos, rpy=None):
        """Apply the barrier to action a (dir, speed, ...) in place. Returns True when it bit."""
        changed = False
        if rpy is not None and OMB_BLIND_V > 0.0 and _MAPSW_STATE.get("map") in OMB_BLIND_MAPS:
            # blind-motion cap: moving where the camera is not looking (stand-off retreats, keep-out
            # pushes, lane turns) is limited to OMB_BLIND_V m/s -- forest slot 148 backed into a tree at 2.8 m/s
            v0 = _vel_of(a) * SPEED_LIMIT
            sh = math.hypot(v0[0], v0[1])
            if sh > OMB_BLIND_V:
                yaw = float(rpy[2])
                head = math.atan2(v0[1], v0[0])
                mis = abs((head - yaw + math.pi) % (2.0 * math.pi) - math.pi)
                if mis > math.radians(OMB_BLIND_DEG):
                    v0 = v0 * (OMB_BLIND_V / sh)
                    _store_vel(a, v0 / SPEED_LIMIT)
                    self.stats["omb_blind"] = self.stats.get("omb_blind", 0) + 1
                    changed = True
        if self.keys.size == 0:
            return changed
        P = self.pts
        x = np.asarray(pos, dtype=np.float64)
        D = x[None, :] - P
        R = np.sqrt((D * D).sum(axis=1))
        near = R <= OMB_INFL_M
        if not near.any():
            return changed
        D = D[near]
        R = R[near]
        order = np.argsort(R)
        D = D[order]
        R = R[order]
        v_cmd = _vel_of(a) * SPEED_LIMIT
        v = v_cmd.copy()
        r_safe = OMB_R_SAFE_BY_MAP.get(_MAPSW_STATE.get("map"), OMB_R_SAFE)
        # stuck escape: commanded to move but not moving for OMB_STUCK_TICKS -> barrier off briefly
        sp_cmd = float(math.hypot(v_cmd[0], v_cmd[1]))
        x2 = (float(x[0]), float(x[1]))
        if self.prog_pos is None or math.hypot(x2[0] - self.prog_pos[0], x2[1] - self.prog_pos[1]) > 1.0 or sp_cmd < 0.5:
            self.prog_pos = x2
            self.prog_tick = self.tick
        elif self.tick - self.prog_tick > OMB_STUCK_TICKS and self.tick > self.free_until:
            self.free_until = self.tick + OMB_STUCK_FREE
            self.prog_tick = self.tick
            self.stats["omb_stuck"] = self.stats.get("omb_stuck", 0) + 1
        if self.tick <= self.free_until:
            r_safe = 0.35

        def _project(v):
            for _pass in range(2):
                for i in range(min(len(R), 24)):
                    r = R[i]
                    n = D[i] / max(r, 1e-6)
                    lim = OMB_K * (r - r_safe)
                    v_toward = -float(v @ n)
                    if v_toward > lim:
                        v = v + (v_toward - lim) * n
            return v
        v = _project(v)
        if not np.allclose(v, v_cmd, atol=1e-6):
            changed = True
        # slide: the barrier ate most of the horizontal command -> move along the nearest obstacle's tangent
        sp_h = float(math.hypot(v[0], v[1]))
        if sp_cmd > 0.3 and sp_h < 0.35 * sp_cmd and self.tick > self.free_until:
            toward = -D[0][:2]
            tn = float(math.hypot(toward[0], toward[1]))
            if tn > 1e-6:
                toward = toward / tn
                tL = np.array([-toward[1], toward[0]])
                tR = -tL
                if self.tick <= self.slide_until and self.slide_side != 0.0:
                    side = self.slide_side
                else:
                    dot = float(v_cmd[0] * tL[0] + v_cmd[1] * tL[1]) / max(sp_cmd, 1e-6)
                    if abs(dot) > 0.2:
                        side = 1.0 if dot > 0 else -1.0
                    else:
                        # head-on: pick the side with the clearer 2 m probe against the memory
                        pl = x[:2] + tL * 2.0
                        pr = x[:2] + tR * 2.0
                        dl = float(np.min(np.hypot(P[:, 0] - pl[0], P[:, 1] - pl[1]))) if len(P) else 9.0
                        dr = float(np.min(np.hypot(P[:, 0] - pr[0], P[:, 1] - pr[1]))) if len(P) else 9.0
                        side = 1.0 if dl >= dr else -1.0
                    self.slide_side = side
                    self.slide_until = self.tick + OMB_SLIDE_HOLD
                t = tL if side > 0 else tR
                vs = min(sp_cmd, OMB_SLIDE_V)
                v2 = np.array([t[0] * vs, t[1] * vs, v_cmd[2]], dtype=np.float64)
                v = _project(v2)
                self.stats["omb_slide"] = self.stats.get("omb_slide", 0) + 1
                changed = True
        rmin = float(R[0])
        if rmin < self.stats["omb_min_r"]:
            self.stats["omb_min_r"] = rmin
        s = float(np.linalg.norm(v))
        cap = SPEED_LIMIT
        if rmin < OMB_CAP_NEAR:
            cap = SPEED_LIMIT * max(OMB_CAP_MIN, min(1.0, (rmin - r_safe * 0.5) / (OMB_CAP_NEAR - r_safe * 0.5)))
        if s > cap:
            v = v * (cap / s)
            changed = True
            self.stats["omb_capped"] += 1
        if changed:
            self.stats["omb_limited"] += 1
            _store_vel(a, v / SPEED_LIMIT)
        return changed


class PerDronePilot:

    MEM = MEM_SIZE

    @staticmethod
    def load_nets(models_dir=None):
        here = models_dir if models_dir else _HERE
        if not os.path.exists(os.path.join(_HERE, _ks.DET_NAME)):
            _log_v21("det_missing",
                 "%s is not next to king_stack.py (models_dir=%r); falling back to "
                 "the repo path, which will NOT exist inside the validator"
                 % (_ks.DET_NAME, here))
        ref = _ks.VictimDetector()
        bundle = {k: getattr(ref, k, None) for k in _DET_SESSION_KEYS}
        n_sess = sum(1 for k in ("sess", "rgb_sess", "mtn_sess", "mtn_rgb_sess")
                     if bundle.get(k) is not None)
        real_mtn = _ks.MountainClassifier()
        if not real_mtn.pred.enabled:
            _log_v21("xgb_off",
                 "%s missing or unreadable -- no map label, so v21 will fly the flat "
                 "10 m deck on mountain too" % _ks.XGB_NAME)
        return {"v21_sessions": bundle,
                "v21_bus": _MapBus(real_mtn),
                "n_sessions": int(n_sess),
                "labels": tuple(_ks.MAP_LABELS)}

    def __init__(self, models_dir, nets=None):
        if nets is None:
            nets = self.load_nets(models_dir)
        self.nets = nets
        self.bundle = nets["v21_sessions"]
        self.bus = nets["v21_bus"]
        self.uid = self.bus.claim_id()
        self.ctrl = self._build_controller()

        self._in_support = None
        self._support_src = None
        self._support_fn = None
        self._support_relax = None
        self._outer_admits = 0
        self._ostreak_n = 0                      # 2026-09-21: outer-streak state leaked across episodes (harness/RPC reuse)
        self._ostreak_t = -1
        self._ostreak_z = None
        self._ostreak_lt = -99
        self._stride_ok = True
        self._last_wp = None
        self._team_deny = None
        self._soft_deny = []
        self._omb = _ObstacleMemory()
        self._last_pos = np.zeros(3, dtype=np.float64)
        self._standoff_ticks = 0
        self._skipped = 0
        self._served = 0
        self._spin = None
        self._gap = None
        self._fbd = None
        self.spin_stats = {"spin_ticks": 0, "spin_started": 0, "spin_done": 0, "spin_stopped_seen": 0,
                           "spin_stopped_team": 0, "spin_skipped": 0,
                           "gap_started": 0, "gap_found": 0, "gap_none": 0, "gap_ticks": 0, "gap_high_ticks": 0,
                           "lad_started": 0, "lad_steps": 0, "lad_ticks": 0, "lad_ended": 0,
                           "skn_checks": 0, "skn_looks": 0, "skn_yes": 0, "skn_req": 0, "skn_confirm": 0, "skn_reject": 0,
                           "nw_leave": 0, "coh_started": 0, "coh_ticks": 0, "coh_timeout": 0, "coh_guard": 0}
        self._lad = None
        self._skn = None
        self._co = None
        self._co_state = None
        self.last_action = np.zeros(ACT_DIM, dtype=np.float32)
        self.p_env = _ZERO6.copy()
        self.est = np.zeros(3, dtype=np.float64)
        self.lock = 0.0
        self.n_det = 0.0

    def _build_controller(self):
        c = None
        if _ASSEMBLY_OK:
            try:
                c = _ks._V21Controller.__new__(_ks._V21Controller)
                c.detector = self._build_detector()
                c.avoider = _ks.ObstacleAvoider()
                c.smoother = _ks.TrackingDifferentiator()
                c.mtn = _MtnView(self.bus, self.uid)
                c.alt_td = _ks.AltitudeTD()
                c.reset()
            except Exception as exc:
                _log_v21("ctrl_assembly", "shared assembly failed (%r); full ctor" % (exc,))
                c = None
        if c is None:
            c = _ks._V21Controller()
            c.mtn = _MtnView(self.bus, self.uid)
            c.reset()

        c._search_policy = types.MethodType(_swarm_search_policy, c)
        c._swarm_r_scale = 1.0

        c._rgb_phase = (int(self.uid) * 5) % _ks.RGB_REQUEST_PERIOD

        det = c.detector
        self._det_update = _ks.VictimDetector.update.__get__(det)
        self._det_refuted = _ks.VictimDetector.location_refuted.__get__(det)
        det.update = self._gated_update
        det.location_refuted = self._gated_refuted
        return c

    def _build_detector(self):
        b = self.bundle
        det = _ks.VictimDetector.__new__(_ks.VictimDetector)
        for k in _DET_SESSION_KEYS:
            setattr(det, k, b.get(k))
        det.reset()
        return det

    def reset(self):
        self.ctrl.reset()
        self.bus.reset()
        try:
            self.ctrl._nav_head_z = None          # 2026-09-21: survived reset()
            self.ctrl._swarm_r_scale = 1.0
        except Exception:
            pass
        self._in_support = None
        self._support_src = None
        self._support_fn = None
        self._support_relax = None
        self._outer_admits = 0
        self._ostreak_n = 0                      # 2026-09-21: outer-streak state leaked across episodes (harness/RPC reuse)
        self._ostreak_t = -1
        self._ostreak_z = None
        self._ostreak_lt = -99
        self._stride_ok = True
        self._last_wp = None
        self._team_deny = None
        self._soft_deny = []
        self._omb.reset()
        self._last_pos = np.zeros(3, dtype=np.float64)
        self._standoff_ticks = 0
        self._skipped = 0
        self._served = 0
        self._spin = None
        self._gap = None
        self._fbd = None
        self.spin_stats = {"spin_ticks": 0, "spin_started": 0, "spin_done": 0, "spin_stopped_seen": 0,
                           "spin_stopped_team": 0, "spin_skipped": 0,
                           "gap_started": 0, "gap_found": 0, "gap_none": 0, "gap_ticks": 0, "gap_high_ticks": 0,
                           "lad_started": 0, "lad_steps": 0, "lad_ticks": 0, "lad_ended": 0,
                           "skn_checks": 0, "skn_looks": 0, "skn_yes": 0, "skn_req": 0, "skn_confirm": 0, "skn_reject": 0,
                           "nw_leave": 0, "coh_started": 0, "coh_ticks": 0, "coh_timeout": 0, "coh_guard": 0}
        self._lad = None
        self._skn = None
        self._co = None
        self._co_state = None
        self.last_action = np.zeros(ACT_DIM, dtype=np.float32)
        self.p_env = _ZERO6.copy()
        self.est = np.zeros(3, dtype=np.float64)
        self.lock = 0.0
        self.n_det = 0.0

    def warm(self):
        if self.nets.get("_warmed"):
            return
        b = self.bundle
        try:
            x = np.zeros((256, 256, 1), dtype=np.float32)
            if b.get("sess") is not None:
                b["sess"].run(None, {b["iname"]: x})
            if b.get("mtn_sess") is not None:
                b["mtn_sess"].run(None, {b["mtn_iname"]: x})
            r = np.zeros((_ks.RGB_RES, _ks.RGB_RES, 3), dtype=np.float32)
            if b.get("rgb_sess") is not None:
                b["rgb_sess"].run(None, {b["rgb_iname"]: r})
            if b.get("mtn_rgb_sess") is not None:
                b["mtn_rgb_sess"].run(None, {b["mtn_rgb_iname"]: r})
        except Exception as exc:
            _log_v21("warm", "warm-up pass failed (%r)" % (exc,))
        self.nets["_warmed"] = True

    def _gated_update(self, depth, pos, rpy, rgb=None, is_mountain=False):
        det = self.ctrl.detector
        if self._stride_ok or _rgb_present(rgb):
            self._served += 1
            return self._det_update(depth, pos, rpy, rgb=rgb, is_mountain=is_mountain)
        self._skipped += 1
        det.t += 1
        det.rgb_fresh = False
        det.rgb_promoted = False
        det.seen = False
        if det.t - det.last_strong_t > _ks.STALE_TICKS:
            _dec = VFC_DECAY if _vfc_on() else _ks.STALE_DECAY     # VFC_* are this module's globals, not the king bag
            det.strong *= _dec
            det.cand_strong *= _dec
        det.conf = float(np.clip(det.strong / _ks.N_LATCH, 0.0, 1.0))
        return (det.conf, det.vic.copy(), det.vic[2] + 0.5 * det.height)

    def _gated_refuted(self, z):
        try:
            if self._det_refuted(z):
                return True
        except Exception:
            pass
        zxy = (float(z[0]), float(z[1]))
        if OUTER_STREAK > 0 and _MAPSW_STATE.get("map") in OUTER_STREAK_MAPS:
            try:
                det = self.ctrl.detector
                _bar = float(getattr(det, "strong_p_override", 0.0) or _ks.STRONG_P)
                _tk = int(det.t)
                if _tk != getattr(self, '_ostreak_t', -1):
                    self._ostreak_t = _tk
                    _lz = getattr(self, '_ostreak_z', None)
                    if float(det.last_p_depth) >= _bar and _lz is not None and math.hypot(zxy[0] - _lz[0], zxy[1] - _lz[1]) <= _ks.CONSIST_M and _tk - getattr(self, '_ostreak_lt', -99) <= 3:
                        self._ostreak_n = getattr(self, '_ostreak_n', 0) + 1
                    else:
                        self._ostreak_n = 1 if float(det.last_p_depth) >= _bar else 0
                    self._ostreak_z = zxy
                    self._ostreak_lt = _tk
                if getattr(self, '_ostreak_n', 0) >= OUTER_STREAK:
                    self._outer_admits += 1
                    return False
            except Exception:
                pass
        d = self._team_deny
        if d is not None and math.hypot(zxy[0] - d[0], zxy[1] - d[1]) <= _ks.CONSIST_M:
            return True
        if self._soft_deny:
            t = self.ctrl.tick
            for xy, exp_t in self._soft_deny:
                if t <= exp_t and math.hypot(zxy[0] - xy[0], zxy[1] - xy[1]) <= _ks.CONSIST_M:
                    return True
        f = self._in_support
        if f is not None:
            q = np.array(zxy, dtype=np.float64)
            try:
                v = f(q)
                inside = bool(v) if isinstance(v, (bool, np.bool_)) else float(v) >= 0.5
            except Exception:
                _log_v21("support", "in_support raised; failing open")
                self._in_support = None
                return False
            if not inside:
                return self._outer_refuted(q)
        return False

    def _outer_refuted(self, q):
        g = self._support_relax
        if g is None:
            return True
        try:
            v = g(q)
            in_relax = bool(v) if isinstance(v, (bool, np.bool_)) else float(v) >= 0.5
        except Exception:
            _log_v21("support", "relaxed support raised; reverting to the hard veto")
            self._support_relax = None
            return True
        if not in_relax:
            return True
        det = self.ctrl.detector
        bar = float(getattr(det, "strong_p_override", 0.0) or _ks.STRONG_P)
        if float(det.last_p_depth) < _outer_bar(bar, OUTER_ODDS):
            return True
        self._outer_admits += 1
        return False

    def act(self, depth, state165, rgb=None, spoof_clue_xy=None, in_support=None,
            allow_confirm=True, stride_ok=True, *, rgb_allowed=True):
        c = self.ctrl
        s = np.asarray(_to_np(state165), dtype=np.float64).reshape(-1)
        if s.size < 165:
            pad = np.zeros(165, dtype=np.float64)
            pad[:s.size] = s
            s = pad
        else:
            s = np.array(s[:165], dtype=np.float64)
        if spoof_clue_xy is not None:
            wp = np.asarray(_to_np(spoof_clue_xy), dtype=np.float64).reshape(2)
            s[163] = wp[0] - s[0]
            s[164] = wp[1] - s[1]
        else:
            wp = np.array([s[0] + s[163], s[1] + s[164]], dtype=np.float64)
        self._rearm_on_waypoint(wp)
        self._in_support = in_support
        if in_support is not self._support_fn:

            self._support_relax = None
        self._stride_ok = bool(stride_ok)
        self._last_pos = s[0:3]
        self._apply_deny()

        obs = {"state": s,
               "depth": _depth2d(depth),
               "rgb": _EMPTY_RGB if rgb is None else np.asarray(_to_np(rgb),
                                                               dtype=np.float32)}
        pre_req = (int(c.rgb_requests), int(c.last_rgb_req))
        a = np.asarray(c.act(obs), dtype=np.float32).reshape(ACT_DIM).copy()
        if a[5] > 0.5 and not rgb_allowed:

            a[5] = 0.0
            c.rgb_requests, c.last_rgb_req = pre_req
        if _skin_world():
            try:
                a = self._skin_step(a, s, rgb, obs["depth"], pre_req, rgb_allowed)
            except Exception as exc:
                _log_v21("skin_step", "skin check raised %r; champion action this tick" % (exc,))
        if _forest():
            if SPIN_ON:
                try:
                    a = self._spin_step(a, s)
                except Exception as exc:
                    _log_v21("spin_step", "take-off look-around raised %r; champion action this tick" % (exc,))
            if GAP_ON:
                try:
                    a = self._gap_step(a, s)
                except Exception as exc:
                    _log_v21("gap_step", "gap-seeking hover raised %r; champion action this tick" % (exc,))
            if LAD_ON:
                try:
                    self._ladder_step(s)
                except Exception as exc:
                    _log_v21("ladder_step", "hover ladder raised %r" % (exc,))
        if not allow_confirm:
            a = self._standoff(a, s)
        else:
            self._standoff_ticks = 0
        if FBD_ON and OMB_ON and _MAPSW_STATE.get("map") in FBD_MAPS:
            try:
                a = self._detour_step(a, s)
            except Exception as exc:
                _log_v21("detour_step", "blocked-approach detour raised %r; champion action this tick" % (exc,))
        if OMB_ON and _omb_world():
            try:
                self._omb.observe(obs["depth"], s[0:3], s[3:6])
                self._omb.stats["omb_ticks"] += 1
                if not (OMB_SKIP_NAV_MAPS and _MAPSW_STATE.get('map') in OMB_SKIP_NAV_MAPS and self.ctrl.mode == 'navigation'):
                    self._omb.constrain(a, s[0:3], s[3:6])   # NEWDET: optionally no barrier while approaching a candidate
            except Exception as exc:
                _log_v21("omb", "obstacle memory raised %r; champion action this tick" % (exc,))
        if not np.all(np.isfinite(a)):
            a = self.last_action.copy()
            a[5] = 0.0
        self._publish()
        self.last_action = a
        return a

    def step(self, depth, state165, mem=None, rgb=None, mates=None, region=None,
             waypoint=None, stride_ok=True, allow_confirm=False, rgb_allowed=True,
             scan_r_scale=None, deny_xy=None, det_stride_phase=None, lr_allow=None,
             in_support=None, spoof_clue_xy=None, co_hover=None, **_ignored):
        m = _to_np(mem)
        self._co = co_hover if (COH_ON and isinstance(co_hover, dict) and co_hover.get("ok")) else None
        if scan_r_scale is not None:
            try:
                self.ctrl._swarm_r_scale = float(np.clip(float(scan_r_scale), 0.2, 1.0))
            except Exception:
                pass
        if deny_xy is not None:
            self.set_team_blacklist(deny_xy)
        _det_lr = self.ctrl.detector
        if getattr(_det_lr, 'lr_sess', None) is not None:
            _det_lr.lr_ok = bool(lr_allow) and not self.ctrl.locked\
                and not self.ctrl.frozen and self.ctrl.tick >= LR_T_MIN
        if in_support is None and region is not None:
            if region is not self._support_src:
                self._support_src = region
                self._support_fn = self.support_fn(region)
                self._support_relax = self.support_relax_fn(region)
            in_support = self._support_fn
        if spoof_clue_xy is None and waypoint is not None:
            spoof_clue_xy = waypoint
        a = self.act(depth, state165, rgb=rgb, spoof_clue_xy=spoof_clue_xy,
                     in_support=in_support, allow_confirm=allow_confirm,
                     stride_ok=stride_ok, rgb_allowed=rgb_allowed)
        if m is not None:
            self._write_mem(m)
        return a, (m if m is not None else mem)

    @staticmethod
    def support_fn(region):
        if region is None:
            return None
        for nm in ("contains", "in_support", "inside", "support"):
            f = getattr(region, nm, None)
            if callable(f):
                return f
        d_out = getattr(region, "dist_outside", None) or getattr(region, "d_out", None)
        if callable(d_out):
            return lambda xy: 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0,
                                     3.0 - float(d_out(xy))))))
        if callable(region):
            return region
        return None

    @staticmethod
    def support_relax_fn(region):
        if region is None:
            return None
        for nm in ("contains_relaxed", "in_support_relaxed"):
            f = getattr(region, nm, None)
            if callable(f):
                return f
        d_out = getattr(region, "dist_outside_relaxed", None)
        if callable(d_out):
            return lambda xy: float(d_out(xy)) <= 0.0
        return None

    def _rearm_on_waypoint(self, wp):
        prev = self._last_wp
        if prev is not None and float(math.hypot(wp[0] - prev[0], wp[1] - prev[1])) <= WP_RESET_M:
            return
        self._last_wp = np.array(wp, dtype=np.float64)
        if prev is None:
            return
        c = self.ctrl
        if c.locked or c.frozen:
            return
        c.centre_visited = False
        c.search_t0 = None
        c.spiral_t0 = c.tick
        c.spiral_theta = 0.0
        c.spiral_r = _ks.SPIRAL_R0_M
        c.wp_list = []
        c.wp_idx = 0
        c.star_list = []
        c.star_idx = 0

    def _apply_deny(self):
        c = self.ctrl
        if not (c.locked or c.frozen):
            return
        v = c.locked_vic
        d = self._team_deny
        hit = d is not None and math.hypot(v[0] - d[0], v[1] - d[1]) <= _ks.CONSIST_M
        if not hit and self._soft_deny:
            t = c.tick
            hit = any(t <= exp_t and math.hypot(v[0] - xy[0], v[1] - xy[1]) <= _ks.CONSIST_M
                      for xy, exp_t in self._soft_deny)
        if hit:
            if (_mq_on() and c.mtn.is_mountain and c.frozen and c.mode == 'hover'
                    and c.hover_total_ticks <= MQ_PROTECT_TICKS
                    and float(np.hypot(v[0] - self._last_pos[0], v[1] - self._last_pos[1])) <= 2.5):
                c.mq_stats["mq_protect"] += 1
                return
            if _fix_world(c) and c.detector.evidence_at(v[0:2]) >= FIX_EVIDENCE:
                c.mq_stats["fix_denyskip"] = c.mq_stats.get("fix_denyskip", 0) + 1   # 302: evidence beats a team deny
                return
            self._drop_lock()

    def _drop_lock(self):
        c = self.ctrl
        det = c.detector
        c.locked = False
        c.frozen = False
        c.lost_ticks = 0
        c.hover_stable_ticks = 0
        c.hover_total_ticks = 0
        c.locked_vic = np.zeros(3, dtype=np.float64)
        c.locked_head_z = 0.0
        det.vic = np.zeros(3, dtype=np.float64)
        det.cand = np.zeros(3, dtype=np.float64)
        det.strong = 0.0
        det.cand_strong = 0.0
        det.conf = 0.0
        det.seen = False
        if c.mode in ("navigation", "hover"):
            c.mode = "search"

        c.centre_visited = False
        c.search_t0 = None
        c.spiral_t0 = c.tick
        c.spiral_theta = 0.0
        c.spiral_r = _ks.SPIRAL_R0_M
        self._standoff_ticks = 0

    def _standoff(self, a, s):
        c = self.ctrl
        if not (c.locked or c.frozen):
            self._standoff_ticks = 0
            return a
        tgt = c.locked_vic
        if TEL2_E4:
            # TEL2 E4: only a lock on the team candidate is stood off; a lock on a distinct spot is this
            # drone's own find and it confirms it in parallel (the team deny already handles the case
            # where the spot IS the candidate).
            _d = self._team_deny
            if _d is None or math.hypot(float(tgt[0]) - float(_d[0]), float(tgt[1]) - float(_d[1])) > TEL2_SELF_M:
                self._standoff_ticks = 0
                self.spin_stats["tel2_self"] = self.spin_stats.get("tel2_self", 0) + 1
                return a
        dx = float(s[0] - tgt[0])
        dy = float(s[1] - tgt[1])
        h = float(math.hypot(dx, dy))
        if h > STANDOFF_R_M + 1.5:
            self._standoff_ticks = 0
            return a
        self._standoff_ticks += 1
        co = getattr(self, "_co", None)
        if co is not None:
            try:
                res = self._co_hover(a, s, tgt, co)
                if res is not None:
                    return res
            except Exception as exc:
                _log_v21("co_hover", "co-hover raised %r; stand-off this tick" % (exc,))
        else:
            self._co_state = None
        _ttl, _deny = STANDOFF_TTL, SOFT_DENY_TICKS
        if NW_ON and _MAPSW_STATE.get("map") == 'forest':
            _ttl, _deny = NW_TTL, NW_DENY
            if self._standoff_ticks > _ttl:
                self.spin_stats["nw_leave"] += 1
        if self._standoff_ticks > _ttl:
            self._soft_deny.append((np.array([tgt[0], tgt[1]], dtype=np.float64),
                                    c.tick + _deny))
            if len(self._soft_deny) > 8:
                self._soft_deny.pop(0)
            self._drop_lock()
            return a
        out = np.array(a, dtype=np.float32)
        ux, uy = _unit_or((dx, dy), (math.cos(0.7853 * (self.uid + 1)),
                                     math.sin(0.7853 * (self.uid + 1))))
        vz = max(0.0, float(a[2]) * float(a[3]))
        if h < STANDOFF_R_M:

            spd = min(STANDOFF_SPEED, 0.15 + 0.35 * (STANDOFF_R_M - h))
            vx, vy = ux * spd, uy * spd
        else:
            vx = vy = 0.0
        mag = float(math.sqrt(vx * vx + vy * vy + vz * vz))
        if mag > 1e-9:
            out[0] = vx / mag
            out[1] = vy / mag
            out[2] = vz / mag
            out[3] = min(mag, 1.0)
        else:
            out[0] = out[1] = out[2] = out[3] = 0.0

        out[4] = float(np.clip(math.atan2(-dy, -dx) / math.pi, -1.0, 1.0))
        return out

    def _spin_step(self, a, s):
        """Forest take-off look-around: one slow turn on the spot after the climb, only while this
        drone has seen nothing and the team holds no candidate; ends at the first strong depth
        output. The action keeps the champion's vertical command, zeroes the horizontal one and
        drives the yaw target round."""
        c = self.ctrl
        det = c.detector
        sp = self._spin
        if sp is not None and sp.get("done"):
            return a
        busy = (c.locked or c.frozen or c.mode not in ('initial', 'search')
                or self._team_deny is not None or float(det.strong) > 0.0)
        if sp is None:
            if c.tick < SPIN_T0 or c.tick > SPIN_MAX_T:
                return a
            if busy:
                self.spin_stats["spin_skipped"] += 1
                self._spin = {"done": True}
                return a
            sp = self._spin = {"t0": int(c.tick), "yaw0": float(s[5]), "done": False}
            self.spin_stats["spin_started"] += 1
        # stop conditions: a strong depth output this tick, a track, a lock, or a team candidate
        if busy or float(det.last_p_depth) >= SPIN_STOP_P:
            sp["done"] = True
            self.spin_stats["spin_stopped_seen" if (float(det.last_p_depth) >= SPIN_STOP_P or float(det.strong) > 0.0) else "spin_stopped_team"] += 1
            return a
        ang = math.radians(SPIN_RATE) * (c.tick - sp["t0"]) * DT
        if ang >= 2.0 * math.pi * SPIN_TURNS:
            sp["done"] = True
            self.spin_stats["spin_done"] += 1
            return a
        target = sp["yaw0"] + ang
        target = (target + math.pi) % (2.0 * math.pi) - math.pi
        out = np.array(a, dtype=np.float32)
        vz = float(a[2]) * float(a[3])                # keep the champion's climb / descent
        out[0] = 0.0
        out[1] = 0.0
        out[2] = 1.0 if vz > 0.0 else (-1.0 if vz < 0.0 else 0.0)
        out[3] = float(np.clip(abs(vz), 0.0, 1.0))
        out[4] = float(np.clip(target / math.pi, -1.0, 1.0))
        self.spin_stats["spin_ticks"] += 1
        return out

    def _detour_step(self, a, s):
        """FBD: commit to a tangential detour when the locked approach has stalled against remembered obstacles."""
        c = self.ctrl
        f = getattr(self, "_fbd", None)
        if not (c.locked or c.frozen) or c.mode not in ('navigation', 'hover'):
            self._fbd = None
            return a
        est = np.asarray(c.locked_vic, dtype=np.float64)
        pos = np.asarray(s[0:3], dtype=np.float64)
        if f is not None and math.hypot(f["est"][0] - est[0], f["est"][1] - est[1]) > _ks.CONSIST_M:
            f = None                                       # a new lock: fresh bookkeeping
        if f is None:
            f = self._fbd = {"est": est.copy(), "best_h": 1e9, "best_t": int(c.tick), "side": 0.0,
                             "until": -1, "t0": -1, "n": 0}
        h = math.hypot(est[0] - pos[0], est[1] - pos[1])
        if h < f["best_h"] - 0.3:
            f["best_h"], f["best_t"] = h, int(c.tick)
        active = c.tick <= f["until"]
        if h < FBD_MIN_H or h > FBD_MAX_H:
            if active:
                f["until"] = -1
                self.spin_stats["fbd_end_h"] = self.spin_stats.get("fbd_end_h", 0) + 1
            return a
        P = self._omb.pts
        gx, gy = est[0] - pos[0], est[1] - pos[1]
        gn = math.hypot(gx, gy)
        if gn < 1e-6:
            return a
        gx, gy = gx / gn, gy / gn
        if not active:
            if f["n"] >= FBD_MAX or c.tick - f["best_t"] < FBD_STALL or len(P) == 0:
                return a
            dx, dy = P[:, 0] - pos[0], P[:, 1] - pos[1]
            r = np.hypot(dx, dy)
            ahead = (dx * gx + dy * gy) > 0.0
            if not bool(np.any((r <= FBD_NEAR) & ahead)):
                return a
            tl = (-gy, gx)
            best_side, best_cnt = 1.0, None
            for side in (1.0, -1.0):
                qx, qy = pos[0] + side * tl[0] * FBD_PROBE, pos[1] + side * tl[1] * FBD_PROBE
                cnt = int(np.sum(np.hypot(P[:, 0] - qx, P[:, 1] - qy) <= 1.5))
                if best_cnt is None or cnt < best_cnt:
                    best_side, best_cnt = side, cnt
            f["side"], f["until"], f["t0"], f["n"] = best_side, int(c.tick) + FBD_HOLD, int(c.tick), f["n"] + 1
            self.spin_stats["fbd_started"] = self.spin_stats.get("fbd_started", 0) + 1
            active = True
        elif c.tick - f["t0"] >= 50 and len(P):
            # straight line to the lock clear of memory points (within 1.2 m of the segment) -> resume
            dx, dy = P[:, 0] - pos[0], P[:, 1] - pos[1]
            proj = np.clip(dx * gx + dy * gy, 0.0, gn)
            perp = np.hypot(dx - proj * gx, dy - proj * gy)
            if not bool(np.any(perp <= 1.2)):
                f["until"] = -1
                f["best_t"] = int(c.tick)
                self.spin_stats["fbd_clear"] = self.spin_stats.get("fbd_clear", 0) + 1
                return a
        if not active:
            return a
        if c.tick > f["until"]:
            f["until"] = -1
            f["best_t"] = int(c.tick)
            self.spin_stats["fbd_timeout"] = self.spin_stats.get("fbd_timeout", 0) + 1
            return a
        self.spin_stats["fbd_ticks"] = self.spin_stats.get("fbd_ticks", 0) + 1
        side = f["side"]
        tx, ty = -gy * side, gx * side
        ux, uy = 0.85 * tx + 0.15 * gx, 0.85 * ty + 0.15 * gy
        un = math.hypot(ux, uy)
        ux, uy = ux / un, uy / un
        out = np.array(a, dtype=np.float32)
        vz = float(a[2]) * float(a[3])
        vs = FBD_V / SPEED_LIMIT
        lx, ly = ux * vs, uy * vs
        mag = math.sqrt(lx * lx + ly * ly + vz * vz)
        if mag > 1e-6:
            out[0], out[1], out[2] = lx / mag, ly / mag, vz / mag
            out[3] = float(min(mag, 1.0))
        out[4] = float(np.clip(math.atan2(uy, ux) / math.pi, -1.0, 1.0))
        return out

    def _gap_step(self, a, s):
        """Forest gap-seeking hover: a locked drone stuck high over its estimate (ray on a crown)
        tries the ring around the estimate at its current altitude and moves the hover spot to the
        first point with clear air below. The vertical command stays the champion's."""
        c = self.ctrl
        g = self._gap
        if not (c.locked or c.frozen):
            self._gap = None
            return a
        est = np.asarray(c.locked_vic, dtype=np.float64)
        if g is not None and math.hypot(g["est"][0] - est[0], g["est"][1] - est[1]) > _ks.CONSIST_M:
            g = self._gap = None                          # a new lock: a new try
        pos = np.asarray(s[0:3], dtype=np.float64)
        agl = getattr(c, "_last_agl", None)
        above = float(pos[2]) - float(c.locked_head_z)
        if g is None:
            if c.mode != 'hover':
                return a
            if above > GAP_HIGH_M and agl is not None and np.isfinite(agl) and float(agl) <= GAP_AGL_MAX:
                self.spin_stats["gap_high_ticks"] += 1
                hi = getattr(self, "_gap_hi", 0) + 1
                self._gap_hi = hi
                if hi < GAP_WAIT:
                    return a
                ring = [est[:2].copy()] + [est[:2] + GAP_RING_M * np.array([math.cos(2 * math.pi * i / GAP_N),
                                                                             math.sin(2 * math.pi * i / GAP_N)])
                                           for i in range(GAP_N)]
                ring.sort(key=lambda q: math.hypot(q[0] - pos[0], q[1] - pos[1]))
                g = self._gap = {"est": est.copy(), "ring": ring, "i": 0, "t0": int(c.tick), "done": False}
                self.spin_stats["gap_started"] += 1
            else:
                self._gap_hi = 0
                return a
        if g["done"]:
            return a
        self.spin_stats["gap_ticks"] += 1
        q = g["ring"][g["i"]]
        d = math.hypot(q[0] - pos[0], q[1] - pos[1])
        if d <= GAP_REACH_M and agl is not None and np.isfinite(agl) and float(agl) >= GAP_CLEAR_M:
            c.locked_vic[0] = float(q[0])                 # clear air below: hover here, the champion descends
            c.locked_vic[1] = float(q[1])
            c._hov_z_lock = None
            g["done"] = True
            self.spin_stats["gap_found"] += 1
            return a
        if (d <= GAP_REACH_M and c.tick - g["t0"] >= 10) or c.tick - g["t0"] >= GAP_VISIT:
            g["i"] += 1
            g["t0"] = int(c.tick)
            if g["i"] >= len(g["ring"]):
                g["done"] = True
                self.spin_stats["gap_none"] += 1
                return a
            q = g["ring"][g["i"]]
            d = math.hypot(q[0] - pos[0], q[1] - pos[1])
        out = np.array(a, dtype=np.float32)
        vz = float(a[2]) * float(a[3])
        if d > 1e-3:
            lx, ly = (q[0] - pos[0]) / d * GAP_SPEED, (q[1] - pos[1]) / d * GAP_SPEED
        else:
            lx = ly = 0.0
        mag = math.sqrt(lx * lx + ly * ly + vz * vz)
        if mag > 1e-6:
            out[0], out[1], out[2] = lx / mag, ly / mag, vz / mag
            out[3] = float(min(mag, 1.0))
        return out

    def _skin_step(self, a, s, rgb, depth, pre_req, rgb_allowed):
        c = self.ctrl
        det = c.detector
        K = getattr(self, "_skn", None)
        if not (c.locked and not c.frozen and c.mode == 'navigation'):
            if K is not None and c.mode != 'navigation' and not c.locked:
                self._skn = None
            return a
        est = np.asarray(c.locked_vic, dtype=np.float64)
        if K is not None and math.hypot(K["est"][0] - est[0], K["est"][1] - est[1]) > _ks.CONSIST_M:
            K = None
        if K is None:
            K = self._skn = {"est": est.copy(), "looks": 0, "yes": 0, "served": -1, "inview": 0, "close_inview": 0,
                             "dyes": 0, "close_hits": 0, "req": 0, "verdict": 0}
            self.spin_stats["skn_checks"] += 1
        if K["verdict"] != 0:
            return a
        pos = np.asarray(s[0:3], dtype=np.float64)
        rpy = np.asarray(s[3:6], dtype=np.float64)
        fwd, up, right = _ks.camera_axes(rpy)
        cam = pos + fwd * _ks.CAMERA_OFFSET_M + up * _ks.CAMERA_UP_OFFSET_M
        d = est - cam
        z = float(d @ fwd)
        rng = float(np.linalg.norm(d))
        in_view = visible = False
        u = v = 0.0
        if z > 0.5:
            u = float(d @ right) / z
            v = float(d @ up) / z
            if abs(u) <= SKN_FOV and abs(v) <= SKN_FOV:
                in_view = True
                dm = np.asarray(depth, dtype=np.float32)
                if dm.ndim == 2 and dm.shape[0] == _ks.DEPTH_RES:
                    px = min(_ks.DEPTH_RES - 1, max(0, int(round((u + 1.0) * 0.5 * _ks.DEPTH_RES - 0.5))))
                    py = min(_ks.DEPTH_RES - 1, max(0, int(round((1.0 - v) * 0.5 * _ks.DEPTH_RES - 0.5))))
                    win = dm[max(0, py - 2):py + 3, max(0, px - 2):px + 3]
                    dz_m = float(np.median(win)) * (_ks.DEPTH_MAX_M - _ks.DEPTH_MIN_M) + _ks.DEPTH_MIN_M
                    visible = dz_m >= rng - SKN_OCC_M
        served = int(self._served)
        if served != K["served"]:
            K["served"] = served
            if in_view and visible:
                K["inview"] += 1
                if rng < 8.0:
                    K["close_inview"] += 1
            if det.seen:
                K["dyes"] += 1
                if rng < 8.0:
                    K["close_hits"] += 1
        near = rng <= SKN_RGB_M
        if det.rgb_fresh and in_view and visible and near and rgb is not None:
            try:
                fr = np.asarray(_to_np(rgb), dtype=np.float32).reshape(256, 256, 3)
                H = 256
                px = int((u + 1.0) * 0.5 * H)
                py = int((1.0 - v) * 0.5 * H)
                ph_m = SKN_PERSON_H.get(_MAPSW_STATE.get("map"), 1.8)
                half = max(3, int(64.0 * ph_m / max(rng, 1.0)))     # half the person's height in pixels (256 px / 90 deg)
                crop = fr[max(0, py - half):min(H, py + half), max(0, px - half // 2 - 2):min(H, px + half // 2 + 2)]
                sk = _skin_frac(crop)
            except Exception:
                sk = None
            if sk is not None:
                K["looks"] += 1
                self.spin_stats["skn_looks"] += 1
                if (SKN_LO <= sk <= SKN_HI) or float(det.rgb_prob) >= _ks.RGB_CONFIRM_THRESH:
                    K["yes"] += 1
                    self.spin_stats["skn_yes"] += 1
        if (in_view and visible and near and K["req"] < SKN_REQ_MAX and a[5] < 0.5 and rgb_allowed
                and c.rgb_requests < _ks.RGB_CAP and (c.tick - c.last_rgb_req) >= SKN_REQ_GAP):
            a[5] = 1.0
            c.rgb_requests += 1
            c.last_rgb_req = c.tick
            K["req"] += 1
            self.spin_stats["skn_req"] += 1
        looks = K["looks"]
        frac = K["yes"] / looks if looks else 0.0
        if K["close_hits"] >= 3 or (looks >= SKN_MIN_LOOKS and frac >= SKN_YES):
            K["verdict"] = 1
            self.spin_stats["skn_confirm"] += 1
        elif looks >= SKN_MIN_LOOKS and frac <= SKN_NO and K["close_hits"] == 0 \
                and (K["close_inview"] >= SKN_CLOSE_SILENT or (K["inview"] >= 3 * SKN_SILENT and K["dyes"] <= 1)):
            K["verdict"] = -1
            self.spin_stats["skn_reject"] += 1
            self._soft_deny.append((np.array([est[0], est[1]], dtype=np.float64), c.tick + SKN_MEMORY))
            if len(self._soft_deny) > 8:
                self._soft_deny.pop(0)
            self._drop_lock()
            self._skn = None
        return a

    def _ladder_step(self, s):
        """Forest hover ladder: a frozen hover over the estimate that does not rescue lowers the head
        estimate step by step (ray clear below), so the champion's hover target sinks into the band."""
        c = self.ctrl
        L = getattr(self, "_lad", None)
        if not (c.locked and c.frozen and c.mode == 'hover'):
            self._lad = None
            return
        est = np.asarray(c.locked_vic, dtype=np.float64)
        pos = np.asarray(s[0:3], dtype=np.float64)
        if L is not None and math.hypot(L["est"][0] - est[0], L["est"][1] - est[1]) > _ks.CONSIST_M:
            L = self._lad = None
        if L is None:
            L = self._lad = {"est": est.copy(), "ticks": 0, "lowered": 0.0, "last": int(c.tick),
                             "head0": float(c.locked_head_z), "active": 0, "ended": False}
        if L["ended"]:
            return
        agl = getattr(c, "_last_agl", None)
        near = math.hypot(pos[0] - est[0], pos[1] - est[1]) <= _ks.HOVER_STABLE_M
        clear = agl is not None and np.isfinite(agl) and float(agl) >= LAD_AGL_MIN
        if L["lowered"] > 0.0:
            L["active"] += 1
            # keep the champion's give-up clock honest against the ORIGINAL head estimate
            dz0 = float(pos[2]) - L["head0"]
            if near and _ks.HOVER_GIVEUP_BAND[0] <= dz0 <= _ks.HOVER_GIVEUP_BAND[1] \
                    and not (_ks.HOVER_GIVEUP_BAND[0] <= float(pos[2]) - float(c.locked_head_z) <= _ks.HOVER_GIVEUP_BAND[1]):
                c.hover_stable_ticks += 1
            if L["active"] >= LAD_TOTAL:
                c.locked_head_z = L["head0"]              # ladder over: restore, the champion decides
                c._hov_z_lock = None
                L["ended"] = True
                self.spin_stats["lad_ended"] += 1
                return
        if not (near and clear):
            return
        L["ticks"] += 1
        self.spin_stats["lad_ticks"] += 1
        if L["ticks"] < LAD_WAIT or L["lowered"] >= LAD_MAX or c.tick - L["last"] < LAD_EVERY:
            return
        c.locked_head_z = float(c.locked_head_z) - LAD_STEP
        c._hov_z_lock = None                              # the hover z target is frozen near the spot; release it
        L["lowered"] += LAD_STEP
        L["last"] = int(c.tick)
        if L["lowered"] <= LAD_STEP + 1e-6:
            self.spin_stats["lad_started"] += 1
        self.spin_stats["lad_steps"] += 1

    def _co_hover(self, a, s, tgt, co):
        """Second hoverer: hold a point COH_OFF m from the candidate on the own side, COH_DZ m below
        the confirmer, facing the candidate; a 2 m 3-D guard against the confirmer; time-boxed."""
        c = self.ctrl
        st = getattr(self, "_co_state", None)
        if st is None or math.hypot(st["tgt"][0] - tgt[0], st["tgt"][1] - tgt[1]) > _ks.CONSIST_M:
            st = self._co_state = {"tgt": np.array([tgt[0], tgt[1]], dtype=np.float64), "t0": int(c.tick), "ticks": 0}
            self.spin_stats["coh_started"] += 1
        st["ticks"] += 1
        self.spin_stats["coh_ticks"] += 1
        if st["ticks"] > COH_MAX:
            self.spin_stats["coh_timeout"] += 1
            self._soft_deny.append((np.array([tgt[0], tgt[1]], dtype=np.float64), c.tick + SOFT_DENY_TICKS))
            if len(self._soft_deny) > 8:
                self._soft_deny.pop(0)
            self._drop_lock()
            self._co_state = None
            return None
        pos = np.asarray(s[0:3], dtype=np.float64)
        cpos = np.asarray(co.get("conf_pos"), dtype=np.float64)
        cand = np.asarray(co.get("cand"), dtype=np.float64)
        dxy = pos[:2] - cand[:2]
        nrm = float(np.hypot(dxy[0], dxy[1]))
        u = dxy / nrm if nrm > 1e-3 else np.array([math.cos(0.7853 * (self.uid + 1)), math.sin(0.7853 * (self.uid + 1))])
        goal = np.array([cand[0] + u[0] * COH_OFF, cand[1] + u[1] * COH_OFF,
                         max(float(cpos[2]) - COH_DZ, float(c.locked_head_z) + _ks.HOVER_ABOVE_TOP_M)], dtype=np.float64)
        v = goal - pos
        dist = float(np.linalg.norm(v))
        g = pos - cpos
        gd = float(np.linalg.norm(g))
        if gd < COH_GUARD:
            self.spin_stats["coh_guard"] += 1
            push = (g / gd) if gd > 1e-3 else np.array([u[0], u[1], -0.5])
            v = v + push * (COH_GUARD - gd) * 2.0
            dist = float(np.linalg.norm(v))
        out = np.array(a, dtype=np.float32)
        if dist > 1e-3:
            d = v / dist
            spd = min(COH_SPEED, dist * 0.6)
            out[0], out[1], out[2] = d[0], d[1], d[2]
            out[3] = float(np.clip(spd, 0.0, 1.0))
        else:
            out[0] = out[1] = out[2] = out[3] = 0.0
        out[4] = float(np.clip(math.atan2(cand[1] - pos[1], cand[0] - pos[0]) / math.pi, -1.0, 1.0))
        return out

    def _belief(self):
        c = self.ctrl
        det = c.detector
        if c.frozen:
            return np.asarray(c.locked_vic, dtype=np.float64), 1.0, float(det.strong)
        if c.locked:
            return (np.asarray(c.locked_vic, dtype=np.float64),
                    max(float(det.conf), _ks.LATCH_CONF), float(det.strong))
        return np.asarray(det.vic, dtype=np.float64), float(det.conf), float(det.strong)

    def _publish(self):
        est, lock, ndet = self._belief()
        self.est = est
        self.lock = float(lock)
        self.n_det = float(ndet)
        self.p_env = self.bus.p_env()

    def _write_mem(self, m):
        try:
            c = self.ctrl
            m[0:3] = np.asarray(self.est, dtype=np.float32)
            m[3] = np.float32(self.n_det)
            m[4] = np.float32(self.lock)
            m[5] = np.float32(1.0 if c.mode in ("navigation", "hover") else 0.0)
            m[10] = np.float32(c.tick * DT)
            m[23:29] = np.asarray(self.p_env, dtype=np.float32).reshape(6)
            m[30] = np.float32(c.rgb_requests)
        except Exception:
            _log_v21("mem_write", "team memory buffer is not writable; using the hooks")

    def peek_estimate(self):
        est, lock, ndet = self._belief()
        return (np.asarray(est[0:2], dtype=np.float32).copy(), float(lock), float(ndet))

    def get_map_latches(self):
        return {"p_env": np.asarray(self.p_env, dtype=np.float32).copy(),
                "air_t": float(self.ctrl.tick) * DT,
                "mtn_f": 1.0 if bool(self.ctrl.mtn.is_mountain) else 0.0}

    def set_map_latches(self, latches=None, **kw):
        return None

    def set_team_blacklist(self, xy):
        try:
            p = np.asarray(_to_np(xy), dtype=np.float64).reshape(2)
        except Exception:
            return
        d = self._team_deny
        if d is not None and math.hypot(p[0] - d[0], p[1] - d[1]) <= 0.5:
            return
        self._team_deny = p

    def clear_team_blacklist(self):
        self._team_deny = None

    def add_soft_deny(self, xy, until_tick):
        try:
            p = np.asarray(_to_np(xy), dtype=np.float64).reshape(2)
        except Exception:
            return
        self._soft_deny.append((p.copy(), int(until_tick)))
        if len(self._soft_deny) > 8:
            self._soft_deny.pop(0)

    def telemetry(self):
        c = self.ctrl
        return {"est": np.asarray(self.est, dtype=np.float64),
                "lock": float(self.lock),
                "n_det": float(self.n_det),
                "p_env": np.asarray(self.p_env, dtype=np.float32),
                "mode": c.mode,
                "tick": int(c.tick),
                "map": self.bus.real.label,
                "mtn": bool(c.mtn.is_mountain),
                "xgb_owner": int(self.bus.owner),
                "xgb_serves": int(self.bus.serves),
                "det_served": int(self._served),
                "det_skipped": int(self._skipped),
                "standoff": int(self._standoff_ticks),
                "outer_admits": int(self._outer_admits),
                "outer_ready": bool(self._support_relax is not None),
                "rgb_req": int(c.rgb_requests)}

    @property
    def scan_r_scale(self):
        return float(getattr(self.ctrl, "_swarm_r_scale", 1.0))

    @scan_r_scale.setter
    def scan_r_scale(self, s):
        try:
            self.ctrl._swarm_r_scale = float(np.clip(float(s), 0.2, 1.0))
        except Exception:
            pass

_FLATMOD_pilot_v21 = _FlatModule(
    'pilot_v21',
    _HERE=_HERE, LOCAL_R_M=LOCAL_R_M, LOCAL_R0_M=LOCAL_R0_M,
    LOCAL_GROW_TICKS=LOCAL_GROW_TICKS, LOCAL_ANGULAR=LOCAL_ANGULAR,
    LOCAL_REACH_M=LOCAL_REACH_M, TRANSIT_MAX_TICKS=TRANSIT_MAX_TICKS,
    WP_RESET_M=WP_RESET_M, STANDOFF_R_M=STANDOFF_R_M,
    STANDOFF_SPEED=STANDOFF_SPEED, STANDOFF_TTL=STANDOFF_TTL,
    SOFT_DENY_TICKS=SOFT_DENY_TICKS, XGB_OWNER_STALE=XGB_OWNER_STALE,
    OUTER_ODDS=OUTER_ODDS, _outer_bar=_outer_bar, DT=DT, MEM_SIZE=MEM_SIZE,
    ACT_DIM=ACT_DIM, _LOGGED_V21=_LOGGED_V21, _log_v21=_log_v21, _ks=_ks,
    _store_attrs=_store_attrs, _CTRL_ATTRS=_CTRL_ATTRS,
    _DET_ATTRS=_DET_ATTRS, _DET_SESSION_KEYS=_DET_SESSION_KEYS,
    _CTRL_ASSEMBLED=_CTRL_ASSEMBLED, _ASSEMBLY_OK=_ASSEMBLY_OK,
    _EMPTY_RGB=_EMPTY_RGB, _ZERO6=_ZERO6, _MapBus=_MapBus,
    _MtnView=_MtnView, _swarm_search_policy=_swarm_search_policy,
    _to_np=_to_np, _depth2d=_depth2d, _rgb_present=_rgb_present,
    _unit_or=_unit_or, PerDronePilot=PerDronePilot)

import inspect
import time
import warnings

try:
    import torch
except Exception:
    torch = None

if torch is not None:

    warnings.filterwarnings("ignore", message=".*non-writable.*")

    try:
        torch.set_num_threads(int(os.environ.get("SWSAR_THREADS", "2")))
        torch.set_grad_enabled(False)
    except Exception:
        pass

MAX_DRONES = 8
STATE_DIM = 214
PILOT_STATE_DIM = 165

S_POS = slice(0, 3)
S_RPY = slice(3, 6)
S_VEL = slice(6, 9)
S_AGL = 162
S_CLUE = slice(163, 165)
S_MATES = slice(165, 214)
N_MATE_SLOTS = 7

M_EST = slice(0, 3)
M_NDET = 3
M_LOCK = 4
M_COMMIT = 5
M_AIR_T = 10
M_PENV = slice(23, 29)
M_REQ_USED = 30
M_ANTI2 = slice(35, 37)
M_ANTI2_VALID = 37
M_GU_CD = 38
M_BIN_DEC = 66
M_DET_CH = 68
M_FMODE = 69
M_MTN_F = 70
M_LANE_IDX = 73
M_CONFIRMER = 74
M_ARRIVED = 75
M_LANE_WP = slice(76, 78)
M_WP_DWELL = 78

_LOGGED = set()

def _log(key, msg):
    if key in _LOGGED or len(_LOGGED) > 64:
        return
    _LOGGED.add(key)
    try:
        sys.stderr.write("[swarm_sar.team] %s\n" % msg)
        sys.stderr.flush()
    except Exception:
        pass

def _vel_of(a):
    return np.array([a[0] * a[3], a[1] * a[3], a[2] * a[3]], dtype=np.float64)

def _store_vel(a, v):
    s = float(math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))
    if s > 1e-9:
        a[0] = v[0] / s
        a[1] = v[1] / s
        a[2] = v[2] / s
        a[3] = min(s, 1.0)
    else:
        a[0] = 0.0
        a[1] = 0.0
        a[2] = 0.0
        a[3] = 0.0

def _enforce_vz_min(a, vz_min):
    vz_min = max(-1.0, min(1.0, float(vz_min)))
    v = _vel_of(a)
    if v[2] >= vz_min - 1e-9:
        return False
    v[2] = vz_min
    if vz_min > 0.0:
        h = math.hypot(v[0], v[1])
        h_max = math.sqrt(max(0.0, 1.0 - vz_min * vz_min))
        if h > h_max:
            s = h_max / max(h, 1e-9)
            v[0] *= s
            v[1] *= s
    _store_vel(a, v)
    return True

def _enforce_vz_max(a, vz_max):
    vz_max = max(-1.0, min(1.0, float(vz_max)))
    v = _vel_of(a)
    if v[2] <= vz_max + 1e-9:
        return False
    v[2] = vz_max
    if vz_max < 0.0:
        h = math.hypot(v[0], v[1])
        h_max = math.sqrt(max(0.0, 1.0 - vz_max * vz_max))
        if h > h_max:
            s = h_max / max(h, 1e-9)
            v[0] *= s
            v[1] *= s
    _store_vel(a, v)
    return True

def _horiz(p, q):
    return float(math.hypot(p[0] - q[0], p[1] - q[1]))

def _mem_get(mem, idx):
    if mem is None:
        return None
    try:
        if isinstance(mem, np.ndarray):
            v = mem[idx]
            return float(v) if not isinstance(v, np.ndarray) else v.astype(np.float64, copy=False)
        v = mem[idx]
    except Exception:
        return None
    if torch is not None and isinstance(v, torch.Tensor):
        if v.dim() == 0:
            return float(v.item())
        return v.detach().cpu().numpy().astype(np.float64, copy=False)
    if isinstance(v, np.ndarray):
        return v.astype(np.float64, copy=False)
    try:
        return float(v)
    except Exception:
        return None

def _mem_set(mem, idx, val):
    if mem is None:
        return False
    try:
        if isinstance(mem, np.ndarray):
            mem[idx] = val
            return True
        if isinstance(idx, slice):
            arr = np.asarray(val, dtype=np.float32).ravel()
            if torch is not None and isinstance(mem, torch.Tensor):
                mem[idx] = torch.from_numpy(arr)
            else:
                mem[idx] = arr
        else:
            mem[idx] = float(val)
        return True
    except Exception:
        return False

def _new_mem():
    return np.zeros(MEM_SIZE, dtype=np.float32)

def _as_mem(x):
    if x is None or isinstance(x, np.ndarray):
        return x
    if torch is not None and isinstance(x, torch.Tensor):
        try:
            t = x.detach()
            if not t.is_contiguous():
                t = t.contiguous()
            return t.numpy()
        except Exception:
            try:
                return np.asarray(x.detach().cpu().numpy(), dtype=np.float32)
            except Exception:
                return None
    try:
        return np.asarray(x, dtype=np.float32)
    except Exception:
        return None

class _PilotAdapter:

    _KW_ALIASES = {
        "rgb": ("rgb", "rgb_frame"),
        "mates": ("mates", "teammates", "mate_slots"),
        "region": ("region", "region_t", "mask"),
        "waypoint": ("waypoint", "wp", "lane_wp"),
        "stride_ok": ("stride_ok", "det_ok", "run_detector"),
        "allow_confirm": ("allow_confirm", "confirmer", "is_confirmer"),
        "rgb_allowed": ("rgb_allowed", "rgb_ok", "may_request_rgb"),
        "scan_r_scale": ("scan_r_scale", "r_scale"),
        "deny_xy": ("deny_xy", "anti2_xy", "blacklist_xy"),
        "lr_allow": ("lr_allow",),
        "co_hover": ("co_hover",),
        "det_stride_phase": ("det_stride_phase",),
    }

    def __init__(self, pilot):
        self.p = pilot
        self.mode = "torch" if torch is not None else "numpy"
        self.fn = None
        self.accepts = set()
        self.var_kw = False
        self.n_pos = 3
        self.kw_map = {}
        self.attr_keys = ()
        self._resolve()
        self._bind()

    def _resolve(self):
        for name in ("step", "act", "forward"):
            f = getattr(self.p, name, None)
            if callable(f):
                self.fn = f
                break
        if self.fn is None and callable(self.p):
            self.fn = self.p
        if self.fn is None:
            raise TypeError("pilot object is not callable and has no step/act/forward")
        try:
            sig = inspect.signature(self.fn)
        except Exception:
            self.var_kw = True
            return
        pos = 0
        for prm in sig.parameters.values():
            if prm.kind == prm.VAR_KEYWORD:
                self.var_kw = True
            elif prm.kind == prm.VAR_POSITIONAL:
                pos = 3
            else:
                self.accepts.add(prm.name)
                if prm.kind in (prm.POSITIONAL_ONLY, prm.POSITIONAL_OR_KEYWORD):
                    pos += 1
        self.n_pos = pos

    _ATTR_KEYS = ("stride_ok", "det_stride_phase", "allow_confirm", "rgb_allowed",
                  "scan_r_scale", "deny_xy", "waypoint")

    def _bind(self):
        self.kw_map = {}
        for key, aliases in self._KW_ALIASES.items():
            for alias in aliases:
                if self.var_kw or alias in self.accepts:
                    self.kw_map[key] = alias
                    break

        self.attr_keys = tuple(k for k in self._ATTR_KEYS if k not in self.kw_map)

    def _kw(self, want):
        km = self.kw_map
        return {km[k]: v for k, v in want.items() if k in km}

    def _push_attrs(self, want):
        for key in self.attr_keys:
            if key not in want:
                continue
            for alias in self._KW_ALIASES.get(key, (key,)):
                try:
                    setattr(self.p, alias, want[key])
                except Exception:
                    pass
                break

    def _wrap(self, depth, state165, mem):
        if self.mode == "torch":
            try:
                d = torch.from_numpy(depth) if not isinstance(depth, torch.Tensor) else depth
                s = torch.from_numpy(state165) if not isinstance(state165, torch.Tensor) else state165

                m = torch.from_numpy(mem) if isinstance(mem, np.ndarray) else mem
                return d, s, m
            except Exception:
                self.mode = "numpy"
        return depth, state165, _as_mem(mem)

    def hook(self, name):
        f = getattr(self.p, name, None)
        return f if callable(f) else None

    def reset(self):
        f = getattr(self.p, "reset", None)
        if callable(f):
            try:
                f()
            except Exception:
                _log("pilot_reset", "pilot.reset() raised; continuing")

    def warm(self):
        f = getattr(self.p, "warm", None)
        if callable(f):
            try:
                f()
            except Exception:
                pass

    def __call__(self, depth, state165, mem, want):
        self._push_attrs(want)
        kw = self._kw(want)
        d, s, m = self._wrap(depth, state165, mem)
        args = (d, s, m) if self.n_pos >= 3 or self.var_kw else (d, s)
        try:
            r = self.fn(*args, **kw)
        except TypeError:

            r = self._retry(d, s, m, kw, depth, state165, mem, want)
        return self._unpack(r, mem)

    def _retry(self, d, s, m, kw, depth, state165, mem, want):

        try:
            r = self.fn(d, s, m)
            self.accepts = set()
            self.var_kw = False
            self._bind()
            _log("pilot_kw", "pilot rejected keywords; falling back to attribute handoff")
            return r
        except TypeError:
            pass

        try:
            r = self.fn(d, s, **kw)
            self.n_pos = 2
            _log("pilot_mem", "pilot takes no mem argument; team mem slots inert")
            return r
        except TypeError:
            pass

        if self.mode == "torch":
            self.mode = "numpy"
            _log("pilot_np", "pilot rejected torch inputs; switching to numpy")
            d2, s2, m2 = self._wrap(depth, state165, mem)
            return self.fn(d2, s2, m2, **kw)
        raise

    @staticmethod
    def _unpack(r, mem):
        new_mem = mem
        act = r
        if isinstance(r, (tuple, list)) and len(r) == 2:
            act, new_mem = r[0], _as_mem(r[1])
            if new_mem is None:
                new_mem = mem
        if torch is not None and isinstance(act, torch.Tensor):
            act = act.detach().cpu().numpy()
        a = np.asarray(act, dtype=np.float32).ravel()
        if a.size < ACT_DIM:
            a = np.concatenate([a, np.zeros(ACT_DIM - a.size, np.float32)])
        return a[:ACT_DIM].astype(np.float32, copy=False), new_mem

def _load_posterior_module(*_args, **_kwargs):
    return _FLATMOD_posterior

def _call_flexible(fn, pool, order):
    try:
        sig = inspect.signature(fn)
    except Exception:
        return fn(*[pool[k] for k in order if k in pool])
    names, var_kw = set(), False
    for prm in sig.parameters.values():
        if prm.kind == prm.VAR_KEYWORD:
            var_kw = True
        elif prm.kind != prm.VAR_POSITIONAL:
            names.add(prm.name)
    kw = {}
    for key, val in pool.items():
        if key in names or (var_kw and key in order):
            kw[key] = val
    if kw:
        try:
            return fn(**kw)
        except TypeError:
            pass
    return fn(*[pool[k] for k in order if k in pool])

class _ClueRegion:

    FLOOR_HALF = 45.0
    PAD_R = 80.0
    RING_R = 12.0
    ADVANCE_R = 8.0
    DWELL_S = 9.0

    def __init__(self, n, clue_xy, pad0_xy, p_type=None):
        self.n = int(max(1, n))
        self.clue = np.asarray(clue_xy, dtype=np.float64).ravel()[:2].copy()
        self.pad0 = np.asarray(pad0_xy, dtype=np.float64).ravel()[:2].copy()
        self.r_clue = 80.0 * math.sqrt(self.n / 8.0)
        self.rung = 9

        d = float(np.hypot(*self.clue))
        self.mu = self.clue * (min(1.0, 30.0 / d) if d > 1e-6 else 0.0)

    def dist_outside(self, xy):
        x, y = float(xy[0]), float(xy[1])
        return max(abs(x) - self.FLOOR_HALF, abs(y) - self.FLOOR_HALF,
                   math.hypot(x - self.clue[0], y - self.clue[1]) - self.r_clue,
                   math.hypot(x - self.pad0[0], y - self.pad0[1]) - self.PAD_R, 0.0)

    def contains(self, xy):
        return self.dist_outside(xy) <= 0.0

    def centroid(self):
        return self.mu.copy()

    def assign(self, pos_xy):
        self.n = int(np.asarray(pos_xy).reshape(-1, 2).shape[0])
        return np.arange(self.n, dtype=np.int64)

    def tour(self, lane_id):
        th = 2.0 * math.pi * (int(lane_id) % max(self.n, 1)) / max(self.n, 1)
        return np.array([[self.mu[0] + self.RING_R * math.cos(th + k),
                          self.mu[1] + self.RING_R * math.sin(th + k)]
                         for k in (0.0, 2.094, 4.189)])

    def waypoint(self, lane_id, pos_xy, t_sec, locked):
        tour = self.tour(lane_id)
        p = np.asarray(pos_xy, dtype=np.float64).reshape(2)
        d = np.hypot(tour[:, 0] - p[0], tour[:, 1] - p[1])
        if bool(locked):
            return tour[int(np.argmin(d))].copy()
        j = int((float(t_sec) // self.DWELL_S) % tour.shape[0])
        if d[j] < self.ADVANCE_R:
            j = int(np.argmax(d))
        return tour[j].copy()

    def scan_r_scale(self, lane_id):
        return 1.0

class _RegionAdapter:

    def __init__(self, mod, n, clue_xy, pad0_xy, p_type):
        self.source = "clue-degenerate"
        self.native = None
        self.planner = None
        self.n = int(n)
        self.lane_ids = np.arange(self.n, dtype=np.int64)
        self._wp_fn = None
        self._rs_fn = None
        self._tour_fn = None
        self.tours = {}
        if mod is not None:
            self._build_native(mod, n, clue_xy, pad0_xy, p_type)
        if self.native is None:
            self.native = _ClueRegion(n, clue_xy, pad0_xy, p_type)
        self._d_out = self._pick(self.native, ("dist_outside", "d_out", "distance_outside"))
        host = self.planner if self.planner is not None else self.native
        self._wp_fn = self._pick(host, ("waypoint",))
        self._rs_fn = self._pick(host, ("scan_r_scale", "r_scale", "rosette_scale"))
        self._tour_fn = self._pick(host, ("tour", "waypoints", "lane_waypoints"))
        self.payload = getattr(self.native, "region_t", None)
        if self.payload is None:
            self.payload = self.native

    def _build_native(self, mod, n, clue_xy, pad0_xy, p_type):
        cls = getattr(mod, "Posterior", None)
        if cls is None:
            return
        try:
            post = None
            setg = None
            try:
                post = cls()
                setg = self._pick(post, ("set_geometry",))
            except Exception:
                post = None
            if post is not None and setg is not None:
                setg(np.asarray(clue_xy, dtype=np.float64).ravel()[:2],
                     np.asarray(pad0_xy, dtype=np.float64).ravel()[:2], int(n))
                if p_type is not None:
                    fn = self._pick(post, ("set_type_belief", "set_prior"))
                    if fn is not None:
                        fn(np.asarray(p_type, dtype=np.float64).ravel()[:6])
            else:
                pool = {"n": n, "n_drones": int(n), "num_drones": int(n),
                        "clue_xy": clue_xy, "clue": clue_xy, "c": clue_xy,
                        "pad0_xy": pad0_xy, "pad0": pad0_xy,
                        "p_type": p_type, "p_env": p_type, "prior": p_type}
                post = _call_flexible(cls, pool, ("n_drones", "clue_xy", "pad0_xy", "p_type"))
            if self._pick(post, ("dist_outside", "d_out", "distance_outside")) is None:
                raise TypeError("Posterior has no dist_outside/d_out")
            self.native = post
            self.source = "posterior.py"
        except Exception as exc:
            _log("post_ctor", "Posterior(...) failed (%r); using the fallback region" % (exc,))
            self.native = None
            return
        lp = getattr(mod, "LanePlanner", None)
        if lp is not None:
            try:
                self.planner = _call_flexible(
                    lp, {"posterior": self.native, "post": self.native,
                         "region": self.native, "n_drones": int(n)}, ("posterior",))
            except Exception as exc:
                _log("lane_ctor", "LanePlanner(...) failed (%r)" % (exc,))
                self.planner = None

    def set_type_belief(self, p_type):
        if p_type is None:
            return False
        fn = self._pick(self.native, ("set_type_belief", "set_prior"))
        if fn is None:
            return False
        try:
            fn(np.asarray(p_type, dtype=np.float64).ravel()[:6])
            return True
        except Exception as exc:
            _log("relatch", "set_type_belief failed (%r)" % (exc,))
            return False

    @staticmethod
    def _pick(obj, names):
        for nm in names:
            f = getattr(obj, nm, None)
            if callable(f):
                return f
        return None

    @property
    def rung(self):
        return int(getattr(self.native, "rung", -1))

    @property
    def centroid(self):
        v = getattr(self.native, "centroid", None)
        try:
            if callable(v):
                v = v()
            if v is not None:
                return np.asarray(v, dtype=np.float64).ravel()[:2]
        except Exception:
            pass
        v = getattr(self.native, "mu", None)
        try:
            return np.asarray(v, dtype=np.float64).ravel()[:2]
        except Exception:
            return np.zeros(2)

    def d_out(self, xy):
        if self._d_out is None:
            return 0.0
        try:
            return float(self._d_out(np.asarray(xy, dtype=np.float64).ravel()[:2]))
        except Exception:
            return 0.0

    def assign(self, pos_xy):
        pos = np.asarray(pos_xy, dtype=np.float64).reshape(-1, 2)
        n = int(pos.shape[0])
        self.n = n
        self.lane_ids = np.arange(n, dtype=np.int64)
        host = self.planner if self.planner is not None else self.native
        fn = self._pick(host, ("assign", "plan", "partition", "assign_lanes"))
        if fn is not None:
            try:
                res = fn(pos)
                if isinstance(res, dict):
                    self.tours = {int(k): np.asarray(v, dtype=np.float64).reshape(-1, 2)
                                  for k, v in res.items()}
                elif res is not None:
                    ids = np.asarray(res, dtype=np.int64).ravel()
                    if ids.size == n:
                        self.lane_ids = ids
            except Exception as exc:
                _log("lane_assign", "lane assign failed (%r); ring fallback" % (exc,))
        if not self.tours:
            for i in range(n):
                self.tours[i] = self._tour_of(int(self.lane_ids[i]), i, n)
        return self.tours

    def _tour_of(self, lane_id, i, n):
        if self._tour_fn is not None:
            try:
                a = np.asarray(self._tour_fn(lane_id), dtype=np.float64).reshape(-1, 2)
                if a.shape[0] >= 1 and np.all(np.isfinite(a)):
                    return a
            except Exception:
                pass

        mu = self.centroid
        r = 12.0 if n > 1 else 0.0
        th = 2.0 * math.pi * i / max(n, 1)
        return np.array([[mu[0] + r * math.cos(th), mu[1] + r * math.sin(th)]])

    def lane_of(self, i):
        return int(self.lane_ids[i]) if i < self.lane_ids.size else int(i)

    def waypoint(self, i, pos_xy, t_sec, locked):
        lane = self.lane_of(i)
        if self._wp_fn is not None:
            try:
                wp = np.asarray(self._wp_fn(lane, np.asarray(pos_xy, dtype=np.float64)[:2],
                                            float(t_sec), bool(locked)),
                                dtype=np.float64).ravel()[:2]
                if wp.size == 2 and np.all(np.isfinite(wp)):
                    return wp
            except Exception as exc:
                _log("lane_wp", "waypoint() failed (%r); using the cached tour" % (exc,))
        tour = self.tours.get(int(i))
        if tour is None or len(tour) == 0:
            return self.centroid.copy()
        p = np.asarray(pos_xy, dtype=np.float64)[:2]
        d = np.hypot(tour[:, 0] - p[0], tour[:, 1] - p[1])
        if bool(locked) or tour.shape[0] == 1:
            return tour[int(np.argmin(d))].copy()
        j = int((float(t_sec) // 9.0) % tour.shape[0])
        if d[j] < 8.0:
            j = int(np.argmax(d))
        return tour[j].copy()

    def scan_r_scale(self, i):
        if self._rs_fn is None:
            return 1.0
        try:
            return float(np.clip(float(self._rs_fn(self.lane_of(i))), 0.30, 1.0))
        except Exception:
            return 1.0

class _TeamState:

    SEP_R = 3.4
    SEP_R_WRECK = 4.5
    SEP_HARD = 1.6
    SEP_GAIN = 1.5
    SEP_CAP = 0.75

    LEAD_SEC = 0.40

    FREEZE_TICKS = 75
    FREEZE_TICKS_COLD = 250

    AGL_FLOOR = 2.5
    AGL_TTL = 0.4
    CLIMB_CMD = 0.5

    CONF_AGL_HI_FOREST = float(os.environ.get('SAR660_AGLHI', '4.6'))
    CONF_NEAR = 6.0
    CONF_Z_MARGIN_FOREST = float(os.environ.get('SAR660_ZMARGIN', '3.2'))
    CONF_AGL_HI_BASE = 4.10
    CONF_Z_MARGIN_BASE = 2.8

    @property
    def CONF_AGL_HI(self):
        if os.environ.get('SAR710_FOREST_ONLY', '1') != '1' \
                or _MAPSW_STATE.get("map") == 'forest':
            return self.CONF_AGL_HI_FOREST
        return self.CONF_AGL_HI_BASE

    @property
    def CONF_Z_MARGIN(self):
        if os.environ.get('SAR710_FOREST_ONLY', '1') != '1' \
                or _MAPSW_STATE.get("map") == 'forest':
            return self.CONF_Z_MARGIN_FOREST
        return self.CONF_Z_MARGIN_BASE

    CONF_VZ_DN = 0.55
    KEEPOUT_R = 5.0
    KEEPOUT_SOFT = 1.5
    KEEPOUT_PUSH = 0.6
    OWN_EST_R = 3.0

    SPEED_LIMIT = 3.0
    CORR_RATE_H = 0.030
    CORR_RATE_DN = 0.020

    CF_D_XY = 0.2
    CF_D_Z = 0.5
    CF_WEIGHT = 0.027 * 9.8
    TILT_SAFE = 0.6981
    THRUST_KEEP = 0.65

    AGL_ENV_GAIN = 1.0 / 1.2

    ELECT_LOCK = 0.55
    ELECT_NDET_BASE = 4.0
    WEAK_LOCK = 0.30

    @property
    def ELECT_NDET(self):
        return VFC_ELECT_NDET if _vfc_on() else self.ELECT_NDET_BASE
    RELEASE_LOCK = 0.25
    MERGE_R = 4.0
    MIN_DWELL = 50
    LOST_TICKS = 75
    STALL_TICKS = 700
    COOLDOWN = 150
    MAX_RECORDS = 4
    RECORD_TTL = 600
    BLACKLIST_R = 4.0

    DET_CAP = 4
    DET_STRIDE = int(os.environ.get('NEWDET_DET_STRIDE', '3'))   # NEWDET: detector cadence per drone (ticks)
    DET_LOCK_THR = 0.3
    DET_STARVE = 9

    RELATCH_TICK = 150
    RELATCH_TICK_MAX = 300
    LATCH_BCAST_TICK = 405

    RGB_RESERVE = 6

    WP_REACH = 8.0
    WP_PERIOD = 5

    def __init__(self, post_mod):
        self.post_mod = post_mod
        self.tick = 0
        self.n = 0
        self.last_pos = None
        self.pad0 = None
        self.origins = None
        self.frozen = np.zeros(MAX_DRONES, dtype=bool)
        self.still = np.zeros(MAX_DRONES, dtype=np.int32)
        self.flew = np.zeros(MAX_DRONES, dtype=bool)

        self.region = None
        self.region_latched = False
        self.region_final = False
        self.tours = {}
        self.wp_idx = np.zeros(MAX_DRONES, dtype=np.int32)
        self.wp_dwell = np.zeros(MAX_DRONES, dtype=np.float64)
        self.arrived = np.zeros(MAX_DRONES, dtype=np.float64)
        self.last_wp = np.full((MAX_DRONES, 2), np.nan)
        self.wp_locked = np.zeros(MAX_DRONES, dtype=np.float64)
        self.scan_r = np.ones(MAX_DRONES, dtype=np.float64)
        self.clue_world = np.zeros(2)
        self.clue_spread = 0.0

        self.p_env_team = None
        self.type_bcast = False

        self.records = []
        self.cand = None
        self.blacklist = []
        self.confirmer = -1
        self.elect_tick = -1
        self.mtn = False
        self.released_xy = np.full((MAX_DRONES, 2), np.nan)
        self.private_pending = np.zeros(MAX_DRONES, dtype=bool)
        self.n_supersede = 0
        self.elect_t0 = -1
        self.conf_frozen_t = -1
        self.blacklist_pending = []
        self.lost = np.zeros(MAX_DRONES, dtype=np.int32)
        self.cooldown = np.zeros(MAX_DRONES, dtype=np.int64)
        self.best_h = np.full(MAX_DRONES, 1e9)
        self.offcand = np.zeros(MAX_DRONES, dtype=np.int32)
        self.cohover = -1                        # the one drone allowed to hover beside the confirmer

        self.det_ok = np.zeros(MAX_DRONES, dtype=bool)
        self.det_served = np.full(MAX_DRONES, -1000, dtype=np.int64)
        self.det_rr = 0

        self.rgb_used = np.zeros(MAX_DRONES, dtype=np.int32)

        self.last_out = np.zeros((MAX_DRONES, ACT_DIM), dtype=np.float32)

        self.corr = np.zeros((MAX_DRONES, 3), dtype=np.float64)

        self.stats = {
            "region_source": "none", "region_rung": -1, "clue_spread": 0.0,
            "guard_agl": 0, "guard_keepout": 0, "guard_band": 0, "guard_own_est": 0,
            "guard_band_agl_incons": 0,
            "sep_push": 0, "det_calls": 0, "det_steps": 0, "det_max_gap": 0,
            "confirmer_multi": 0, "elections": 0, "releases": 0,
            "corr_clamped": 0, "tilt_capped": 0, "tel_handover": 0, "tel_offcand": 0, "tel2_follow": 0, "tel2_repoint": 0,
            "rgb_requests": 0, "rgb_blocked": 0, "deadline_hits": 0,
            "pilot_exceptions": 0, "shape_fixups": 0, "blacklisted": 0,
            "teleport_resets": 0,
            "superseded": 0, "cooldown_exempt": 0, "deny_clear_ticks": 0, "private_deny": 0, "readopt": 0,
            "coh_elect": 0,
        }

    def grow(self, n):
        if n <= self.frozen.shape[0]:
            return
        pad = n - self.frozen.shape[0]
        self.frozen = np.concatenate([self.frozen, np.zeros(pad, bool)])
        self.still = np.concatenate([self.still, np.zeros(pad, np.int32)])
        self.flew = np.concatenate([self.flew, np.zeros(pad, bool)])
        self.wp_idx = np.concatenate([self.wp_idx, np.zeros(pad, np.int32)])
        self.wp_dwell = np.concatenate([self.wp_dwell, np.zeros(pad)])
        self.arrived = np.concatenate([self.arrived, np.zeros(pad)])
        self.last_wp = np.concatenate([self.last_wp, np.full((pad, 2), np.nan)])
        self.wp_locked = np.concatenate([self.wp_locked, np.zeros(pad)])
        self.scan_r = np.concatenate([self.scan_r, np.ones(pad)])
        self.lost = np.concatenate([self.lost, np.zeros(pad, np.int32)])
        self.cooldown = np.concatenate([self.cooldown, np.zeros(pad, np.int64)])
        self.best_h = np.concatenate([self.best_h, np.full(pad, 1e9)])
        self.offcand = np.concatenate([self.offcand, np.zeros(pad, np.int32)])
        self.released_xy = np.concatenate([self.released_xy, np.full((pad, 2), np.nan)])
        self.private_pending = np.concatenate([self.private_pending, np.zeros(pad, bool)])
        self.det_ok = np.concatenate([self.det_ok, np.zeros(pad, bool)])
        self.det_served = np.concatenate([self.det_served, np.full(pad, -1000, np.int64)])
        self.rgb_used = np.concatenate([self.rgb_used, np.zeros(pad, np.int32)])
        self.last_out = np.concatenate([self.last_out, np.zeros((pad, ACT_DIM), np.float32)])
        self.corr = np.concatenate([self.corr, np.zeros((pad, 3))])

    def update_frozen(self, pos, vel):
        n = self.n
        if self.last_pos is None:
            self.origins = pos.copy()
            self.last_pos = pos.copy()
            return
        m = min(n, self.last_pos.shape[0])
        moved = np.linalg.norm(pos[:m] - self.last_pos[:m], axis=1)
        speed = np.linalg.norm(vel[:m], axis=1)
        for i in range(m):
            self.still[i] = self.still[i] + 1 if (moved[i] < 1e-4 and speed[i] < 1e-3) else 0
            if not self.flew[i]:
                if float(np.linalg.norm(pos[i] - self.origins[i])) > 0.6 or speed[i] > 0.4:
                    self.flew[i] = True
            if self.frozen[i]:
                continue

            if self.still[i] > self.FREEZE_TICKS and self.flew[i]:
                self.frozen[i] = True
            elif self.still[i] > self.FREEZE_TICKS_COLD:
                self.frozen[i] = True

    def latch_region(self, n, pos, clue_rel, p_type, force=False):
        if self.region_final and not force:
            return
        cl = np.asarray(pos[:, 0:2], dtype=np.float64) + np.asarray(clue_rel, dtype=np.float64)
        c = cl.mean(axis=0)
        if n > 1:
            spread = float(np.abs(cl - c).max())
            self.clue_spread = spread
            self.stats["clue_spread"] = spread
            if spread > 1e-3:

                _log("clue_spread",
                     "clue XY disagreement across drones: %.4f m (using the mean)" % spread)
        self.clue_world = c
        pad0 = self.pad0 if self.pad0 is not None else pos[0, 0:2]
        if self.region is None:
            self.region = _RegionAdapter(self.post_mod, n, c, np.asarray(pad0)[:2], p_type)
        elif p_type is not None:

            self.region.set_type_belief(p_type)
        self.stats["region_source"] = self.region.source
        self.stats["region_rung"] = self.region.rung
        try:
            self.stats["mfp_p_far"] = float(getattr(self.region.native, "mfp_p_far", 0.0))
        except Exception:
            pass
        self.tours = self.region.assign(np.asarray(pos[:, 0:2], dtype=np.float64))
        for i in range(n):
            self.scan_r[i] = self.region.scan_r_scale(i)
        self.wp_idx[:n] = 0
        self.wp_dwell[:n] = 0.0
        self.arrived[:n] = 0.0
        self.last_wp = np.full((max(n, MAX_DRONES), 2), np.nan)
        self.wp_locked = np.zeros(max(n, MAX_DRONES), dtype=np.float64)
        self.region_latched = True

    def maybe_latch_region(self, n, pos, clue_rel, p_type, airborne):
        if not self.region_latched:
            self.latch_region(n, pos, clue_rel, None)
            return
        if self.region_final:
            return
        ready = (self.tick >= self.RELATCH_TICK and airborne) or self.tick >= self.RELATCH_TICK_MAX
        if ready:
            self.latch_region(n, pos, clue_rel, p_type, force=True)
            self.region_final = True

    def waypoint(self, i, pos_xy, locked=False):
        if self.region is None:
            return np.array(self.clue_world, dtype=np.float64)
        prev = self.last_wp[i]
        have = bool(np.all(np.isfinite(prev)))
        lk = 1.0 if locked else 0.0

        if (not have) or lk != self.wp_locked[i] or ((self.tick + i) % self.WP_PERIOD == 0):
            if P1_ADV and self.mtn and not getattr(self, "_adv_set", False):
                try:
                    _lp = getattr(self.region, "planner", None)
                    if _lp is not None:
                        _lp.advance_r = P1_ADV_R
                        _lp.advance_hold = P1_ADV_HOLD
                        _lp.n_advanced = 0
                        self.stats["p1_adv_armed"] = 1
                except Exception:
                    pass
                self._adv_set = True
            wp = self.region.waypoint(i, pos_xy, self.tick * DT, locked)
            self.wp_locked[i] = lk
        else:
            wp = prev
        if (not have) or float(np.hypot(*(wp - prev))) > 0.5:
            self.wp_idx[i] = (int(self.wp_idx[i]) + 1) % 64
            self.wp_dwell[i] = 0.0
        else:
            self.wp_dwell[i] += DT
        self.last_wp[i] = wp
        if float(math.hypot(wp[0] - pos_xy[0], wp[1] - pos_xy[1])) < self.WP_REACH:
            self.arrived[i] = 1.0
        return wp

    def plan_detector_budget(self, live, locks):
        self.det_ok[:] = False
        if not live:
            return self.det_ok
        t = self.tick
        starved, locked, other = [], [], []
        for k in range(len(live)):
            i = live[(self.det_rr + k) % len(live)]
            if t - int(self.det_served[i]) >= self.DET_STARVE:
                starved.append(i)
            elif float(locks[i]) >= self.DET_LOCK_THR:
                locked.append(i)
            elif (t + i) % self.DET_STRIDE == 0:
                other.append(i)
        grant, seen = [], set()
        for i in starved + locked + other:
            if i in seen:
                continue
            seen.add(i)
            grant.append(i)
            if len(grant) >= self.DET_CAP:
                break
        for i in grant:
            self.det_ok[i] = True
            self.det_served[i] = t
        self.det_rr = (self.det_rr + max(1, len(grant))) % max(1, len(live))
        self.stats["det_calls"] += len(grant)
        self.stats["det_steps"] += 1
        gap = max(int(t - self.det_served[i]) for i in live)
        if gap > self.stats["det_max_gap"]:
            self.stats["det_max_gap"] = gap
        return self.det_ok

    def rgb_allowed(self, i):
        cap = RGB_CAP if i == self.confirmer else max(0, RGB_CAP - self.RGB_RESERVE)
        return int(self.rgb_used[i]) < cap

    def account_rgb(self, out, live):
        for i in live:
            if out[i, 5] > 0.5:
                if not self.rgb_allowed(i):
                    out[i, 5] = 0.0
                    self.stats["rgb_blocked"] += 1
                else:
                    self.rgb_used[i] += 1
                    self.stats["rgb_requests"] += 1
        return out

    def _has_cand(self):
        return self.cand is not None and any(r is self.cand for r in self.records)

    def _drop_cand(self):
        self.records = [r for r in self.records if r is not self.cand]

    def _blacklisted(self, xy):
        for b in self.blacklist:
            if _horiz(b, xy) < self.BLACKLIST_R:
                return True
        return False

    def update_candidate(self, beliefs, pos, live):
        t = self.tick
        for i in live:
            b = beliefs[i]
            if b is None or b["lock"] < self.ELECT_LOCK or b["n_det"] <= self.ELECT_NDET:
                continue
            est = b["est"]
            if not np.all(np.isfinite(est)) or self._blacklisted(est):
                continue
            best, best_d = None, 1e9
            for rec in self.records:
                d = _horiz(rec["xy"], est)
                if d < self.MERGE_R and d < best_d:
                    best, best_d = rec, d
            if best is None:
                if len(self.records) >= self.MAX_RECORDS:
                    self.records.sort(key=lambda r: (len(r["contrib"]), r["hits"]))
                    weak = self.records[0]
                    if weak is self.cand and self.confirmer >= 0:
                        continue
                    self.records.pop(0)
                self.records.append({"xy": np.asarray(est[:2], dtype=np.float64).copy(),
                                     "z": float(est[2]), "hits": 1.0,
                                     "contrib": {int(i)}, "t_first": t, "t_last": t,
                                     "failed": set()})
            else:
                w = min(best["hits"], 20.0)
                best["xy"] = (best["xy"] * w + np.asarray(est[:2], dtype=np.float64)) / (w + 1.0)
                best["z"] = (best["z"] * w + float(est[2])) / (w + 1.0)
                best["hits"] += 1.0
                best["contrib"].add(int(i))
                best["t_last"] = t

        self.records = [r for r in self.records
                        if (t - r["t_last"]) <= self.RECORD_TTL
                        or (self.confirmer >= 0 and r is self.cand)]

        if self.confirmer >= 0 and self._has_cand():
            if TEL2_E3:
                # TEL2 E3: the confirmer's own strong lock is on another spot (elected off-candidate, or its
                # track was re-pointed): make that spot the candidate instead of releasing it -- v1's
                # off-candidate release rotated the role every 2-3 s around phantom records.
                c = self.confirmer
                bc = beliefs[c] if 0 <= c < len(beliefs) else None
                if (bc is not None and bc["lock"] >= self.ELECT_LOCK and bc["n_det"] > self.ELECT_NDET
                        and np.all(np.isfinite(bc["est"])) and _horiz(bc["est"], self.cand["xy"]) > self.CONF_NEAR
                        and not self._blacklisted(bc["est"])):
                    rec, rd = None, 1e9
                    for r in self.records:
                        d = _horiz(r["xy"], bc["est"])
                        if r is not self.cand and d < self.MERGE_R and d < rd and not self._blacklisted(r["xy"]):
                            rec, rd = r, d
                    if rec is not None:
                        self.cand = rec
                        self.elect_tick = t
                        self.lost[c] = 0
                        self.best_h[c] = _horiz(pos[c], rec["xy"])
                        self.conf_frozen_t = -1
                        self.stats["tel2_follow"] += 1
                        _log("tel2_follow", "confirmer %d follows its own lock to a new candidate" % c)
            if TEAM_SUPERSEDE and (self.mtn or TEL_SUPERSEDE_ALL) and self.n_supersede < TEAM_SUPERSEDE_MAX:
                hit = self._superseder(beliefs, pos, live, t)
                if hit is not None:
                    rec, j = hit
                    self._release("superseded", blame=False)
                    self.cand = rec
                    self.confirmer = int(j)
                    self.elect_tick = self.elect_t0 = t
                    self.conf_frozen_t = -1
                    self.lost[j] = 0
                    self.best_h[j] = _horiz(pos[j], rec["xy"])
                    self.n_supersede += 1
                    self.stats["superseded"] += 1
                    self.stats["elections"] += 1
            return
        if not self.records:
            self.cand = None
            return
        if TEL2_E8:
            self.cand = max(self.records, key=lambda r: (len(r["contrib"] - r["failed"]), r["hits"], -r["t_first"]))
        else:
            self.cand = max(self.records, key=lambda r: (len(r["contrib"]), r["hits"], -r["t_first"]))

    def _superseder(self, beliefs, pos, live, t):
        c = self.confirmer
        bc = beliefs[c] if 0 <= c < len(beliefs) else None
        if bc is None or bc["n_det"] > self.ELECT_NDET:
            self.conf_frozen_t = -1
            return None
        frozen = bc["lock"] >= 1.0
        if frozen and self.conf_frozen_t < 0:
            self.conf_frozen_t = t
        if not frozen:
            self.conf_frozen_t = -1
        if (t - self.elect_t0) < self.MIN_DWELL:
            return None
        if frozen:
            if (t - self.conf_frozen_t) < TEAM_SUPERSEDE_FROZEN_TICKS:
                return None
            if TEL_SUP_GUARD and float(self.best_h[c]) <= HOVER_STABLE_M:
                # TEL A guard: a confirmer at its spot has the victim below the cone, so n_det ~ 0 is the
                # expected state of a good hover, not phantom evidence -- leave the exit to the hover give-up
                self.stats["tel_supguard"] = self.stats.get("tel_supguard", 0) + 1
                return None
        elif (t - self.elect_tick) < TEAM_SUPERSEDE_STALL_TICKS:
            return None
        cxy = self.cand["xy"]
        best, best_key, best_j = None, None, -1
        for j in live:
            if j == c or j >= len(beliefs) or beliefs[j] is None:
                continue
            b = beliefs[j]
            est = b["est"]
            if b["lock"] < TEAM_SUPERSEDE_LOCK or not np.all(np.isfinite(est)):
                continue
            if self._blacklisted(est) or not self._electable(j, t, beliefs):
                continue
            if _horiz(est, cxy) <= self.CONF_NEAR:
                continue
            rec, rd = None, 1e9
            for r in self.records:
                d = _horiz(r["xy"], est)
                if r is not self.cand and d < self.MERGE_R and d < rd:
                    rec, rd = r, d
            if rec is None or rec["hits"] < self.ELECT_NDET or self._blacklisted(rec["xy"]):
                continue
            if TEL_SUP_QUAL and not (rec["hits"] >= 0.5 * float(self.cand["hits"])
                                     or len(rec["contrib"]) >= len(self.cand["contrib"])):
                continue
            key = (b["n_det"] > self.ELECT_NDET, float(rec["hits"]), float(b["lock"]), -int(j))
            if best is None or key > best_key:
                best, best_key, best_j = rec, key, j
        return (best, best_j) if best is not None else None

    def _release(self, reason, blame=True):
        i = self.confirmer
        if i < 0:
            return
        self.cooldown[i] = self.tick + self.COOLDOWN
        if self.mtn and self.cand is not None and np.all(np.isfinite(self.cand["xy"])):
            self.released_xy[i] = np.asarray(self.cand["xy"], dtype=np.float64).copy()
            self.private_pending[i] = bool(TEAM_DENY_FIX)
        else:
            self.released_xy[i] = np.nan
        self.lost[i] = 0
        self.confirmer = -1
        self.stats["releases"] += 1
        if blame and self.cand is not None:
            self.cand["failed"].add(int(i))
            if len(self.cand["failed"]) >= 2:

                if self.mtn:
                    self.blacklist_pending.append(self.cand["xy"].copy())
                self.blacklist.append(self.cand["xy"].copy())
                if len(self.blacklist) > 8:
                    self.blacklist.pop(0)
                self._drop_cand()
                self.cand = None
                self.stats["blacklisted"] += 1
        _log("release_%s" % reason, "confirmer %d released (%s)" % (i, reason))

    def _electable(self, i, t, beliefs):
        if self.cooldown[i] <= t:
            return True
        if not (TEAM_COOLDOWN_EXEMPT and self.mtn):
            return False
        b = beliefs[i] if i < len(beliefs) else None
        rx = self.released_xy[i]
        if b is None or not np.all(np.isfinite(rx)) or not np.all(np.isfinite(b["est"])):
            return False
        return _horiz(b["est"], rx) > self.CONF_NEAR

    def elect_confirmer(self, pos, live, beliefs):
        t = self.tick
        c = self.confirmer
        if c >= 0:
            b = beliefs[c] if c < len(beliefs) else None
            if c not in live or self.frozen[c]:
                self._release("frozen", blame=False)
            elif self.cand is None:
                self._release("no_candidate", blame=False)
            elif self._blacklisted(self.cand["xy"]):
                self._release("blacklisted", blame=False)
            else:
                lock = b["lock"] if b else 0.0
                self.lost[c] = self.lost[c] + 1 if lock < self.RELEASE_LOCK else 0
                h = _horiz(pos[c], self.cand["xy"])
                if h < self.best_h[c] - 0.5:
                    self.best_h[c] = h
                    self.elect_tick = t
                held = t - self.elect_tick
                _handed = False
                # TEL B: one hand-over per candidate, from a confirmer that is still navigating (not hovering)
                # and has made no progress for TEL_HANDOVER_TICKS, to a non-frozen drone with a lock on the
                # same candidate that is >= TEL_HANDOVER_GAIN m closer and within TEL_HANDOVER_R m of it.
                # (v1 also released 'off-candidate' confirmers and handed over repeatedly: the role rotated
                # every 2-3 s around phantom candidates, nobody hovered long enough to blacklist them, and
                # village/city seeds that used to confirm timed out -- b1 -0.0135, b2 -0.0104.)
                _conf_frozen = bool(b is not None and b["lock"] >= 1.0)
                if (TEL_HANDOVER_TICKS > 0 and held >= self.MIN_DWELL and (t - self.elect_t0) >= self.MIN_DWELL
                        and held > TEL_HANDOVER_TICKS and not _conf_frozen
                        and int(self.cand.get("handovers", 0)) < TEL_HANDOVER_MAX):
                    cxy = self.cand["xy"]
                    best_j, best_hj = -1, min(float(self.best_h[c]), h)
                    for j in live:
                        if j == c or self.frozen[j] or j >= len(beliefs) or beliefs[j] is None:
                            continue
                        bj = beliefs[j]
                        if bj["lock"] < self.ELECT_LOCK or bj["n_det"] <= self.ELECT_NDET or not np.all(np.isfinite(bj["est"])):
                            continue
                        if _horiz(bj["est"], cxy) >= self.CONF_NEAR:
                            continue
                        hj = _horiz(pos[j], cxy)
                        if hj <= TEL_HANDOVER_R and hj < best_hj - TEL_HANDOVER_GAIN:
                            best_j, best_hj = int(j), hj
                    if best_j >= 0:
                        self.cand["handovers"] = int(self.cand.get("handovers", 0)) + 1
                        if True:
                            self._release("handover", blame=False)
                            self.confirmer = best_j
                            self.elect_tick = self.elect_t0 = t
                            self.conf_frozen_t = -1
                            self.lost[best_j] = 0
                            self.best_h[best_j] = best_hj
                            self.stats["elections"] += 1
                            self.stats["tel_handover"] += 1
                            _handed = True
                if (not _handed) and TEL2_E7 and (t - self.elect_t0) >= self.MIN_DWELL and self.lost[c] > self.LOST_TICKS:
                    self._release("lock_lost")
                elif (not _handed) and held >= self.MIN_DWELL:
                    if self.lost[c] > self.LOST_TICKS:
                        self._release("lock_lost")
                    elif held > self.STALL_TICKS:
                        self._release("no_progress")

        if self.confirmer < 0 and self.cand is not None:
            pool = [i for i in live
                    if beliefs[i] is not None
                    and beliefs[i]["lock"] >= self.ELECT_LOCK
                    and beliefs[i]["n_det"] > self.ELECT_NDET
                    and self._electable(i, t, beliefs)]
            if not pool:

                pool = [i for i in live
                        if beliefs[i] is not None
                        and beliefs[i]["lock"] >= self.WEAK_LOCK
                        and self.cooldown[i] <= t]

            near = [i for i in pool
                    if np.all(np.isfinite(beliefs[i]["est"]))
                    and _horiz(beliefs[i]["est"], self.cand["xy"]) < self.CONF_NEAR]
            if near:
                pool = near
            elif TEL2_E1 and pool:
                # TEL2 E1: nobody in the pool is on this candidate. Electing one of them (the champion's
                # fallback) sends a drone to confirm its OWN spot while the team keeps measuring progress
                # toward the old record. Re-point the candidate to the best drone's own record instead.
                pool.sort(key=lambda i: (-round(float(beliefs[i]["lock"]), 1), -float(beliefs[i]["n_det"]), int(i)))
                moved = False
                for i in pool:
                    e = beliefs[i]["est"]
                    if not np.all(np.isfinite(e)) or self._blacklisted(e):
                        continue
                    rec, rd = None, 1e9
                    for r in self.records:
                        d = _horiz(r["xy"], e)
                        if d < self.MERGE_R and d < rd and not self._blacklisted(r["xy"]):
                            rec, rd = r, d
                    if rec is not None:
                        self.cand = rec
                        pool = [i]
                        moved = True
                        self.stats["tel2_repoint"] += 1
                        break
                if not moved:
                    pool = []
            if pool:
                if TEL2_E2:
                    cxy = self.cand["xy"]
                    pool.sort(key=lambda i: (-round(float(beliefs[i]["lock"]), 1), float(_horiz(pos[i], cxy)),
                                             -float((beliefs[i] or {}).get("fix_ev", 0.0)), int(i)))
                else:
                    pool.sort(key=lambda i: (-float(beliefs[i]["lock"]), -float((beliefs[i] or {}).get("fix_ev", 0.0)), int(i)))
                self.confirmer = int(pool[0])
                self.elect_tick = t
                self.elect_t0 = t
                self.conf_frozen_t = -1
                self.lost[self.confirmer] = 0
                self.best_h[self.confirmer] = _horiz(pos[self.confirmer], self.cand["xy"])
                self.stats["elections"] += 1
                if self.mtn and self.cooldown[self.confirmer] > t:
                    self.stats["cooldown_exempt"] += 1
                if self.mtn and self.cand["failed"]:
                    self.stats["readopt"] += 1
        return self.confirmer

    def inject_blacklist(self, mems, pilots, live):
        xy = None if self.cand is None else self.cand["xy"]
        for i in live:
            mine = (i == self.confirmer)
            _mem_set(mems[i], M_CONFIRMER, 1.0 if mine else 0.0)
            if TEAM_DENY_FIX and self.mtn and (self.private_pending[i] or self.blacklist_pending):
                p = pilots[i] if i < len(pilots) else None
                hook = p.hook("add_soft_deny") if p is not None else None
                if self.private_pending[i]:
                    self.private_pending[i] = False
                    rx = self.released_xy[i]
                    if hook is not None and np.all(np.isfinite(rx)):
                        try:
                            hook(rx, self.tick + TEAM_DENY_PRIVATE_TICKS)
                            self.stats["private_deny"] += 1
                        except Exception:
                            pass
                if hook is not None:
                    for xy_b in self.blacklist_pending:
                        try:
                            hook(xy_b, self.tick + TEAM_BLACKLIST_DENY_TICKS)
                        except Exception:
                            pass
            if TEAM_DENY_FIX and self.mtn and (xy is None or self.confirmer < 0):
                p = pilots[i] if i < len(pilots) else None
                hook = p.hook("clear_team_blacklist") if p is not None else None
                if hook is not None:
                    try:
                        hook()
                        self.stats["deny_clear_ticks"] += 1
                        continue
                    except Exception:
                        pass
                _mem_set(mems[i], M_ANTI2_VALID, 0.0)
                continue
            if xy is None:
                continue
            p = pilots[i] if i < len(pilots) else None
            hook = p.hook("clear_team_blacklist" if mine else "set_team_blacklist")\
                if p is not None else None
            if hook is not None:
                try:
                    hook() if mine else hook(xy)
                    continue
                except Exception:
                    pass
            if mine:
                _mem_set(mems[i], M_ANTI2_VALID, 0.0)
            else:
                _mem_set(mems[i], M_ANTI2, xy)
                _mem_set(mems[i], M_ANTI2_VALID, 1.0)
        self.blacklist_pending = []

    LATCH_KEYS = ("bin_dec", "det_ch", "fmode", "mtn_f")
    LATCH_DISCRETE = ("bin_dec", "fmode")
    LATCH_SLOT = {"bin_dec": M_BIN_DEC, "det_ch": M_DET_CH, "fmode": M_FMODE,
                  "mtn_f": M_MTN_F}

    def broadcast_latches(self, mems, pilots, live, p_team):
        if self.type_bcast or self.tick < self.LATCH_BCAST_TICK or not live:
            return
        if p_team is not None:
            self.p_env_team = np.asarray(p_team, dtype=np.float64).ravel()[:6]

        getters = [(i, pilots[i].hook("get_map_latches")) for i in live
                   if i < len(pilots)]
        rows = []
        for i, g in getters:
            if g is None:
                continue
            try:
                rows.append(g())
            except Exception:
                pass
        team = {}
        if rows:
            for k in self.LATCH_KEYS:
                vals = [float(r[k]) for r in rows if k in r and np.isfinite(r[k])]
                if vals:
                    v = float(np.median(vals))
                    team[k] = float(round(v)) if k in self.LATCH_DISCRETE else v
        if self.p_env_team is not None:
            team["p_env"] = self.p_env_team.astype(np.float32)

        wrote = False
        for i in live:
            s = pilots[i].hook("set_map_latches") if i < len(pilots) else None
            if s is None:
                continue
            try:
                s(dict(team))
                wrote = True
            except Exception:
                pass

        if team:
            if "p_env" in team:
                for i in live:
                    _mem_set(mems[i], M_PENV, team["p_env"])
            for k, slot in self.LATCH_SLOT.items():
                if k in team:
                    for i in live:
                        _mem_set(mems[i], slot, team[k])
        if not wrote and not team:
            for k, slot in self.LATCH_SLOT.items():
                vals = [_mem_get(mems[i], slot) for i in live]
                vals = [v for v in vals if v is not None and np.isfinite(v)]
                if not vals:
                    continue
                v = float(np.median(vals))
                if k in self.LATCH_DISCRETE:
                    v = float(round(v))
                for i in live:
                    _mem_set(mems[i], slot, v)
        self.stats["latch_bcast"] = {k: round(float(v), 4) for k, v in team.items()
                                     if k != "p_env"}
        self.type_bcast = True

    def team_p_type_from(self, get_p_env, live):
        rows = []
        for i in live:
            r = get_p_env(i)
            if r is not None and np.isfinite(r).all() and float(np.sum(r)) > 1e-6:
                rows.append(r)
        if not rows:
            return self.p_env_team
        p = np.mean(np.stack(rows, axis=0), axis=0)
        s = float(p.sum())
        return p / s if s > 1e-6 else None

    def vz_envelope(self, h, climb=None):
        c = self.CLIMB_CMD if climb is None else climb
        h = float(h)
        if h < 0.0:
            return c
        v = -h * self.AGL_ENV_GAIN
        return -1.0 if v < -1.0 else v

    def guard_agl_floor(self, out, agl, live):
        for i in live:
            floor = self.vz_envelope(float(agl[i]) - self.AGL_FLOOR)
            if floor <= -1.0:
                continue
            if _enforce_vz_min(out[i], floor):
                self.stats["guard_agl"] += 1
        return out

    def guard_keepout(self, out, pos, vel, live, beliefs):
        cand = None if self.cand is None else self.cand["xy"]
        for i in live:
            if i == self.confirmer or (COH_ON and i == self.cohover):
                continue
            tgt, radius, own_mode = None, 0.0, False
            if cand is not None:
                h = _horiz(pos[i], cand)
                if h < self.KEEPOUT_R:
                    tgt, radius = cand, self.KEEPOUT_R
            if tgt is None:
                b = beliefs[i]
                if b is not None and b["lock"] > self.DET_LOCK_THR and np.all(np.isfinite(b["est"])):

                    h = _horiz(pos[i], b["est"])
                    _own_ok = (not TEL2_E4) or cand is None or _horiz(b["est"], cand) <= TEL2_SELF_M
                    if h < self.OWN_EST_R and (cand is not None or self.confirmer >= 0) and _own_ok:
                        tgt, radius, own_mode = b["est"][:2], self.OWN_EST_R, True
                        self.stats["guard_own_est"] += 1
            if tgt is None:
                continue
            d = np.array([pos[i][0] - tgt[0], pos[i][1] - tgt[1]], dtype=np.float64)
            r = float(np.hypot(d[0], d[1]))
            if r < 1e-6:

                ang = 2.0 * math.pi * (i % max(self.n, 1)) / max(self.n, 1)
                d = np.array([math.cos(ang), math.sin(ang)])
                r = 1.0
            ux, uy = d[0] / r, d[1] / r

            rdot = float(vel[i][0]) * ux + float(vel[i][1]) * uy
            r_eff = r + self.LEAD_SEC * min(rdot, 0.0)

            w = (radius - r_eff) / self.KEEPOUT_SOFT
            w = 0.0 if w < 0.0 else (1.0 if w > 1.0 else w)
            if w <= 0.0:
                continue
            v = _vel_of(out[i])
            v_rad = v[0] * ux + v[1] * uy
            if v_rad < 0.0:
                v[0] -= w * v_rad * ux
                v[1] -= w * v_rad * uy
            v[0] += w * self.KEEPOUT_PUSH * ux
            v[1] += w * self.KEEPOUT_PUSH * uy
            if own_mode:
                v[2] = max(v[2], 0.4 * w)
            _store_vel(out[i], v)
            self.stats["guard_keepout"] += 1
        return out

    def guard_confirmer_band(self, out, pos, agl, live, beliefs=None):
        i = self.confirmer
        if i < 0 or i not in live:
            return out
        ref_xy, ref_z, h = None, None, 1e9
        if self.cand is not None:
            ref_xy, ref_z = self.cand["xy"], float(self.cand["z"])
            h = _horiz(pos[i], ref_xy)
        b = None if beliefs is None else beliefs[i]
        if (b is not None and b["lock"] > self.DET_LOCK_THR
                and np.all(np.isfinite(b["est"]))):
            hb = _horiz(pos[i], b["est"])
            if hb < h:
                ref_xy, ref_z, h = b["est"][:2], float(b["est"][2]), hb
        if ref_xy is None or h > self.CONF_NEAR:
            return out
        a = float(agl[i])
        changed = False

        dz_est = float(pos[i][2]) - ref_z
        _margin = self.CONF_Z_MARGIN
        if not ((a - 1.35) <= dz_est <= (a + 0.35)):
            self.stats["guard_band_agl_incons"] += 1
            if DC_BAND and self.mtn and dz_est < (a - 1.35):
                # DC D2: the head estimate sits above where the ray says the top can be -- bound the floor by the ray
                ref_z = min(ref_z, (float(pos[i][2]) - a) + 2.0)
                _margin = 1.5
                self.stats["dc_band"] = self.stats.get("dc_band", 0) + 1
        elif a > self.CONF_AGL_HI:

            sink = -(a - self.CONF_AGL_HI) * self.AGL_ENV_GAIN
            changed |= _enforce_vz_max(out[i], max(sink, -self.CONF_VZ_DN))
        lo = self.vz_envelope(a - self.AGL_FLOOR)
        if lo > -1.0:
            changed |= _enforce_vz_min(out[i], lo)

        z_above = float(pos[i][2]) - (ref_z + _margin)
        lo_z = self.vz_envelope(z_above, climb=0.3)
        if lo_z > -1.0:
            changed |= _enforce_vz_min(out[i], lo_z)
        if changed:
            self.stats["guard_band"] += 1
        return out

    def separate(self, out, pos, vel, n, live):
        if n < 2 or not live:
            return out
        P = pos[:n]
        V = vel[:n]
        dx = P[:, 0:1] - P[None, :, 0]
        dy = P[:, 1:2] - P[None, :, 1]
        dz = np.abs(P[:, 2:3] - P[None, :, 2])
        r = np.hypot(dx, dy)
        rs = np.maximum(r, 1e-9)

        rdot = ((dx * (V[:, 0:1] - V[None, :, 0])
                 + dy * (V[:, 1:2] - V[None, :, 1])) / rs)
        r_eff = np.maximum(r + self.LEAD_SEC * np.minimum(rdot, 0.0), 0.05)

        rad = np.where(self.frozen[:n][None, :], self.SEP_R_WRECK, self.SEP_R)
        near = (r > 1e-6) & (r_eff <= rad) & (dz <= rad)
        np.fill_diagonal(near, False)
        w = np.minimum((rad - r_eff) / max(self.SEP_R - self.SEP_HARD, 1e-3), 2.0)
        wr = np.where(near, w / np.maximum(r, 1e-9), 0.0)
        px = (wr * dx).sum(axis=1)
        py = (wr * dy).sum(axis=1)
        mag = np.hypot(px, py)
        _agl_now = getattr(self, '_agl_now', None)
        for i in live:
            if i == self.confirmer or (COH_ON and i == self.cohover):
                continue
            if SEP_MIN_AGL > 0.0 and self.mtn and _agl_now is not None and i < len(_agl_now) and float(_agl_now[i]) < SEP_MIN_AGL:
                continue                                   # NEWDET: no lateral push while still near the pad on mountain
            m = float(mag[i])
            if m < 1e-6:
                continue

            g = min(min(m, 2.0) * self.SEP_GAIN * 0.5, self.SEP_CAP) / m
            v = _vel_of(out[i])
            v[0] += float(px[i]) * g
            v[1] += float(py[i]) * g
            _store_vel(out[i], v)
            self.stats["sep_push"] += 1
        return out

    def limit_correction(self, out, base, live):
        for i in live:
            vb = _vel_of(base[i])
            want = _vel_of(out[i]) - vb
            prev = self.corr[i]
            d = want - prev
            dh = math.hypot(d[0], d[1])
            if dh > self.CORR_RATE_H:
                s = self.CORR_RATE_H / dh
                d[0] *= s
                d[1] *= s
                self.stats["corr_clamped"] += 1
            if d[2] < -self.CORR_RATE_DN:
                d[2] = -self.CORR_RATE_DN
                self.stats["corr_clamped"] += 1
            c = prev + d
            self.corr[i] = c
            _store_vel(out[i], vb + c)
        return out

    def cap_commanded_tilt(self, out, vel, live):
        S = self.SPEED_LIMIT
        tan_max = math.tan(self.TILT_SAFE)
        ez_min = -(1.0 - self.THRUST_KEEP) * self.CF_WEIGHT / self.CF_D_Z
        for i in live:
            v = _vel_of(out[i]) * S
            va = np.asarray(vel[i], dtype=np.float64)
            changed = False
            ez = v[2] - va[2]
            if ez < ez_min:
                v[2] = va[2] + ez_min
                if v[2] > S:
                    v[2] = S
                ez = v[2] - va[2]
                h_max = math.sqrt(max(0.0, S * S - v[2] * v[2]))
                h = math.hypot(v[0], v[1])
                if h > h_max:
                    s = h_max / max(h, 1e-9)
                    v[0] *= s
                    v[1] *= s
                changed = True

            den = max(self.CF_WEIGHT + self.CF_D_Z * ez,
                      self.THRUST_KEEP * self.CF_WEIGHT)
            lim = tan_max * den / self.CF_D_XY
            ex, ey = v[0] - va[0], v[1] - va[1]
            eh = math.hypot(ex, ey)
            if eh > lim:
                s = lim / eh
                v[0] = va[0] + ex * s
                v[1] = va[1] + ey * s
                changed = True
            if changed:

                h_max = math.sqrt(max(0.0, S * S - v[2] * v[2]))
                h = math.hypot(v[0], v[1])
                if h > h_max:
                    s = h_max / max(h, 1e-9)
                    v[0] *= s
                    v[1] *= s
                self.stats["tilt_capped"] += 1
                _store_vel(out[i], v / S)
        return out

    def cheap_fallback(self, i):
        a = self.last_out[i].copy()
        a[3] *= 0.8
        a[5] = 0.0
        return a

class TeamLayer:

    DEADLINE = float(os.environ.get('SWSAR_DEADLINE', '0.380'))
    TELEPORT_M = 3.0

    def __init__(self, models_dir, pilot_factory, eager=True, max_drones=MAX_DRONES):
        self.models_dir = models_dir
        self.pilot_factory = pilot_factory
        self.max_drones = int(max_drones)
        self.pilots = []
        self.nets = None
        self.m = [_new_mem() for _ in range(self.max_drones)]
        self.post_mod = _load_posterior_module()
        if self.post_mod is None:
            _log("no_posterior",
                 "posterior.py not importable -- degrading to the clue-centred stand-in")
        self.timing = {"last_ms": 0.0, "team_ms": 0.0, "pilot_ms": 0.0,
                       "max_ms": 0.0, "max_team_ms": 0.0}
        self.T = _TeamState(self.post_mod)
        if eager:

            self._ensure(self.max_drones)
            for p in self.pilots:
                p.warm()
        self.reset()

    def _make_pilot(self):
        try:
            p = self.pilot_factory(self.models_dir, self.nets)
        except TypeError:
            p = self.pilot_factory(self.models_dir)
        if self.nets is None:
            self.nets = getattr(p, "nets", None)
        return _PilotAdapter(p)

    def _ensure(self, n):
        while len(self.pilots) < n:
            self.pilots.append(self._make_pilot())
        if n > len(self.m):
            self.m.extend(_new_mem() for _ in range(n - len(self.m)))

    def reset(self) -> None:
        for i in range(len(self.m)):
            try:
                self.m[i][:] = 0.0
            except Exception:
                self.m[i] = _new_mem()
        for p in self.pilots:
            p.reset()

        self.T = _TeamState(self.post_mod)

    def _lock_of(self, i):
        v = _mem_get(self.m[i], M_LOCK) if i < len(self.m) else None
        if v is None or not np.isfinite(v):
            p = self.pilots[i].p if i < len(self.pilots) else None
            v = getattr(p, "lock", 0.0) if p is not None else 0.0
            try:
                v = float(v)
            except Exception:
                v = 0.0
        return float(v)

    def _p_env_of(self, i):
        v = _mem_get(self.m[i], M_PENV) if i < len(self.m) else None
        if v is None:
            p = self.pilots[i].p if i < len(self.pilots) else None
            v = getattr(p, "p_env", None) if p is not None else None
        try:
            v = np.asarray(v, dtype=np.float64).ravel()[:6] if v is not None else None
        except Exception:
            return None
        return v if (v is not None and v.size == 6) else None

    def _belief(self, i):
        mem = self.m[i] if i < len(self.m) else None
        est = _mem_get(mem, M_EST)
        lock = _mem_get(mem, M_LOCK)
        ndet = _mem_get(mem, M_NDET)
        penv = _mem_get(mem, M_PENV)

        if (lock is None or est is None) and i < len(self.pilots):
            pk = self.pilots[i].hook("peek_estimate")
            if pk is not None:
                try:
                    exy, lk, nd = pk()
                    est = np.array([float(exy[0]), float(exy[1]),
                                    float(est[2]) if est is not None else 0.0])
                    lock, ndet = float(lk), float(nd)
                except Exception:
                    pass
        p = None if (lock is not None and est is not None) else\
            (self.pilots[i].p if i < len(self.pilots) else None)
        if p is not None:
            tel = getattr(p, "telemetry", None)
            if callable(tel):
                try:
                    d = tel() or {}
                    est = d.get("est", est)
                    lock = d.get("lock", lock)
                    ndet = d.get("n_det", ndet)
                    penv = d.get("p_env", penv)
                except Exception:
                    pass
            if lock is None:
                lock = getattr(p, "lock", None)
            if est is None:
                est = getattr(p, "est", None)
            if ndet is None:
                ndet = getattr(p, "n_det", None)
            if penv is None:
                penv = getattr(p, "p_env", None)
        try:
            est = np.asarray(est, dtype=np.float64).ravel()[:3] if est is not None else None
        except Exception:
            est = None
        if est is None or est.size < 3 or not np.all(np.isfinite(est)):
            est = np.array([np.nan, np.nan, np.nan])
        try:
            penv = np.asarray(penv, dtype=np.float64).ravel()[:6] if penv is not None else None
            if penv is not None and penv.size < 6:
                penv = None
        except Exception:
            penv = None
        try:
            _c = (getattr(self.pilots[i], "ctrl", None) or getattr(getattr(self.pilots[i], "p", None), "ctrl", None)) if (TEL2_E6 and i < len(self.pilots)) else (self.pilots[i].ctrl if i < len(self.pilots) else None)
            _fe = float(_c.detector.evidence_at(est[0:2])) if (_c is not None and est is not None and _fix_world(_c)) else 0.0
        except Exception:
            _fe = 0.0
        return {"est": est,
                "lock": float(lock) if lock is not None and np.isfinite(lock) else 0.0,
                "n_det": float(ndet) if ndet is not None and np.isfinite(ndet) else 0.0,
                "p_env": penv, "fix_ev": _fe}

    def act(self, obs) -> np.ndarray:
        t0 = time.perf_counter()
        n = 0
        try:
            state = np.asarray(obs["state"], dtype=np.float32)
            if state.ndim == 1:
                state = state.reshape(1, -1)
            n = int(state.shape[0])
            out = self._act_inner(obs, state, n, t0)
        except Exception as exc:
            _log("act_exc", "act() firewall caught %r" % (exc,))
            self.T.stats["pilot_exceptions"] += 1
            if n <= 0:
                try:
                    n = int(np.asarray(obs["state"]).reshape(-1, STATE_DIM).shape[0])
                except Exception:
                    n = 1
            out = np.zeros((n, ACT_DIM), dtype=np.float32)
        ms = (time.perf_counter() - t0) * 1e3
        self.timing["last_ms"] = ms
        if ms > self.timing["max_ms"]:
            self.timing["max_ms"] = ms
        return out

    def _act_inner(self, obs, state, n, t0):
        T = self.T
        depth = np.asarray(obs["depth"], dtype=np.float32).reshape(n, 256, 256)
        rgb_in = obs.get("rgb") if hasattr(obs, "get") else None
        rgb = None
        if rgb_in is not None:
            try:
                rgb = np.asarray(rgb_in, dtype=np.float32).reshape(n, 256, 256, 3)
            except Exception:
                rgb = None

        pos = state[:, S_POS].astype(np.float64)
        vel = state[:, S_VEL].astype(np.float64)
        agl = state[:, S_AGL].astype(np.float64) * 20.0

        if T.tick > 0 and T.last_pos is not None and T.last_pos.shape[0] == n:
            if float(np.abs(pos - T.last_pos).max()) > self.TELEPORT_M:
                self.reset()
                T = self.T
                T.stats["teleport_resets"] += 1
        self._ensure(n)
        T.grow(n)
        T.n = n
        if NF_MAX > 0 and NF_TKH > 0.0:
            global _NF_DEFAULT, TK_H
            if _NF_DEFAULT is None:
                _NF_DEFAULT = TK_H
            TK_H = NF_TKH if n <= NF_MAX else _NF_DEFAULT
        if NBIG_MIN > 0:
            global _NBIG_DEFAULTS, OMB_MAPS
            if _NBIG_DEFAULTS is None:
                _NBIG_DEFAULTS = (OMB_MAPS, dict(OMB_R_SAFE_BY_MAP))
            if n >= NBIG_MIN:
                if 'mountain' not in OMB_MAPS:
                    OMB_MAPS = tuple(OMB_MAPS) + ('mountain',)
                OMB_R_SAFE_BY_MAP['mountain'] = NBIG_MTN_RSAFE
            else:
                OMB_MAPS = _NBIG_DEFAULTS[0]
                OMB_R_SAFE_BY_MAP.clear(); OMB_R_SAFE_BY_MAP.update(_NBIG_DEFAULTS[1])
        if N2_MAX > 0:
            global _N2_DEFAULTS, MOUNTAIN_SEARCH_ALT_M, INIT_EXIT_TICKS, INIT_EXIT_MARGIN
            if _N2_DEFAULTS is None:
                _N2_DEFAULTS = (MOUNTAIN_SEARCH_ALT_M, INIT_EXIT_TICKS, INIT_EXIT_MARGIN)
            if n <= N2_MAX:
                MOUNTAIN_SEARCH_ALT_M = N2_MTN_ALT if N2_MTN_ALT > 0.0 else _N2_DEFAULTS[0]
                INIT_EXIT_TICKS = N2_INIT_TICKS if N2_INIT_TICKS > 0 else _N2_DEFAULTS[1]
                INIT_EXIT_MARGIN = N2_INIT_MARGIN if N2_INIT_TICKS > 0 else _N2_DEFAULTS[2]
            else:
                MOUNTAIN_SEARCH_ALT_M, INIT_EXIT_TICKS, INIT_EXIT_MARGIN = _N2_DEFAULTS

        if T.tick == 0:
            T.pad0 = pos[0, 0:2].copy()
            T._tks_starts = pos[:, 0:2].copy()
            T._tks_set = False
        T.update_frozen(pos, vel)
        live = [i for i in range(n) if not T.frozen[i]]
        try:
            T.mtn = bool(self.pilots[0].p.ctrl.mtn.is_mountain)
        except Exception:
            T.mtn = False

        need_belief = (not T.region_final) or (not T.type_bcast)
        if need_belief:
            p_type = T.team_p_type_from(self._p_env_of, live)
            if p_type is None:
                p_type = T.p_env_team
        else:
            p_type = T.p_env_team
        airborne = bool(np.any(agl[:n] > 1.0))
        T.maybe_latch_region(n, pos, state[:, S_CLUE].astype(np.float64),
                             p_type, airborne)
        if TKS_ON and (not getattr(T, "_tks_set", False)) and _MAPSW_STATE.get("map") is not None:
            try:
                if T.tick == 0 or getattr(T, "_tks_starts", None) is None:
                    T._tks_starts = pos[:, 0:2].copy()
                _pn = _tks_pnear(_MAPSW_STATE.get("map"), n, T.clue_world, np.asarray(T.pad0)[:2],
                                 T._tks_starts, TKS_R_NEAR)
                if _pn is not None:
                    for _i in range(min(n, len(self.pilots))):
                        self.pilots[_i].p.ctrl._tks_pnear = float(_pn[_i])
                    T.stats["tks_pnear_max"] = float(max(_pn))
                T._tks_set = True
            except Exception as exc:
                _log("tks", "take-off sweep prior failed %r" % (exc,))
                T._tks_set = True
        T.broadcast_latches(self.m, self.pilots, live, p_type)

        locks = np.zeros(max(n, 1))
        for i in range(n):
            locks[i] = self._lock_of(i)
        T.plan_detector_budget(live, locks)

        out = np.zeros((n, ACT_DIM), dtype=np.float32)
        pilot_ms = 0.0
        for i in live:
            if (time.perf_counter() - t0) > self.DEADLINE:
                out[i] = T.cheap_fallback(i)
                T.stats["deadline_hits"] += 1
                continue
            s = np.array(state[i, 0:PILOT_STATE_DIM], dtype=np.float32)

            wp = T.waypoint(i, pos[i], locks[i] >= (LATCH_CONF if MJ_WP_LOCK else T.DET_LOCK_THR))
            s[163] = np.float32(wp[0] - pos[i, 0])
            s[164] = np.float32(wp[1] - pos[i, 1])
            _mem_set(self.m[i], M_LANE_WP, wp)
            _mem_set(self.m[i], M_LANE_IDX, float(T.wp_idx[i]))
            _mem_set(self.m[i], M_WP_DWELL, float(T.wp_dwell[i]))
            _mem_set(self.m[i], M_ARRIVED, float(T.arrived[i]))
            mates = None
            if state.shape[1] >= STATE_DIM:
                mates = state[i, S_MATES].reshape(N_MATE_SLOTS, 7).astype(np.float32).copy()
            want = {
                "rgb": None if rgb is None else rgb[i],
                "mates": mates,
                "region": None if T.region is None else T.region.payload,
                "waypoint": wp.astype(np.float64),
                "stride_ok": bool(T.det_ok[i]),
                "det_stride_phase": int((T.tick + i) % T.DET_STRIDE),
                "allow_confirm": bool(i == T.confirmer),
                "rgb_allowed": bool(T.rgb_allowed(i)),
                "scan_r_scale": float(T.scan_r[i]),
                "deny_xy": None if (T.cand is None or i == T.confirmer
                                    or ((TEL2_E5 or (TEAM_DENY_FIX and T.mtn)) and T.confirmer < 0)) else T.cand["xy"],
                "lr_allow": bool(T.confirmer < 0
                                 and (T.cand is None or not _LR_NOCAND)),
                "co_hover": ({"ok": True, "conf_pos": pos[T.confirmer].astype(np.float64).tolist(),
                              "cand": [float(T.cand["xy"][0]), float(T.cand["xy"][1])]}
                             if (COH_ON and i == T.cohover and T.confirmer >= 0 and T.cand is not None) else None),
            }
            tp = time.perf_counter()
            try:
                a, mem = self.pilots[i](depth[i], s, self.m[i], want)
                if mem is not None:
                    self.m[i] = mem
                if a.shape != (ACT_DIM,):
                    T.stats["shape_fixups"] += 1
                out[i] = a
            except Exception as exc:
                _log("pilot_exc", "pilot %d raised %r" % (i, exc))
                T.stats["pilot_exceptions"] += 1
                out[i] = T.cheap_fallback(i)
            pilot_ms += (time.perf_counter() - tp) * 1e3

        # elect the co-hoverer for the NEXT tick: the confirmer has been on the candidate long enough
        # without a rescue, and one other live drone with its own lock on the same spot stands within
        # COH_NEAR m. Cleared as soon as the confirmer or the candidate changes.
        if COH_ON:
            c_ = T.confirmer
            keep = -1
            if c_ >= 0 and T.cand is not None and (T.tick - T.elect_t0) >= COH_WAIT and T.best_h[c_] <= 3.0:
                cxy = T.cand["xy"]
                if T.cohover >= 0 and T.cohover in live and T.cohover != c_ and _horiz(pos[T.cohover], cxy) <= COH_NEAR + 3.0:
                    keep = T.cohover
                else:
                    best, bd = -1, 1e9
                    for j in live:
                        if j == c_ or T.frozen[j]:
                            continue
                        pj = getattr(self.pilots[j], "p", None) if j < len(self.pilots) else None
                        cj = getattr(pj, "ctrl", None)
                        if cj is None or not (cj.locked or cj.frozen):
                            continue
                        if _horiz(cj.locked_vic, cxy) > _ks.CONSIST_M:
                            continue
                        dj = _horiz(pos[j], cxy)
                        if dj <= COH_NEAR and dj < bd:
                            best, bd = j, dj
                    keep = best
                    if best >= 0 and best != T.cohover:
                        T.stats["coh_elect"] += 1
            T.cohover = keep
        beliefs = [self._belief(i) for i in range(n)]
        T.update_candidate(beliefs, pos, live)
        T.elect_confirmer(pos, live, beliefs)
        T.inject_blacklist(self.m, self.pilots, live)

        n_conf = 0
        for i in live:
            f = _mem_get(self.m[i], M_CONFIRMER)
            if f is not None and f > 0.5:
                n_conf += 1
        if n_conf > 1:
            T.stats["confirmer_multi"] += 1
        T.stats["confirm_flags"] = n_conf

        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        base = out.copy()
        T._agl_now = agl
        out = T.guard_agl_floor(out, agl, live)
        out = T.guard_keepout(out, pos, vel, live, beliefs)
        out = T.guard_confirmer_band(out, pos, agl, live, beliefs)
        out = T.separate(out, pos, vel, n, live)

        out = T.limit_correction(out, base, live)
        out = T.guard_agl_floor(out, agl, live)
        if OMB_ON and _omb_world():
            # final safety pass: the team guards (keep-out pushes, separation) move drones where the
            # camera is not looking; constrain the guarded command against the memory too
            for i in live:
                try:
                    _om = getattr(self.pilots[i].p, "_omb", None)
                    if _om is not None:
                        if not (OMB_SKIP_NAV_MAPS and _MAPSW_STATE.get('map') in OMB_SKIP_NAV_MAPS and getattr(getattr(self.pilots[i].p, 'ctrl', None), 'mode', '') == 'navigation'):
                            _om.constrain(out[i], pos[i], state[i, 3:6])
                except Exception:
                    pass

        out = T.cap_commanded_tilt(out, vel, live)
        out = T.account_rgb(out, live)

        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(out[:, 0:5], -1.0, 1.0, out=out[:, 0:5])
        np.clip(out[:, 5:6], 0.0, 1.0, out=out[:, 5:6])
        out = np.ascontiguousarray(out, dtype=np.float32)
        if out.shape != (n, ACT_DIM):
            T.stats["shape_fixups"] += 1
            fixed = np.zeros((n, ACT_DIM), dtype=np.float32)
            flat = out.ravel()
            fixed.ravel()[:min(flat.size, n * ACT_DIM)] = flat[:n * ACT_DIM]
            out = fixed
        for i in range(n):
            T.last_out[i] = out[i]
        T.last_pos = pos.copy()
        T.tick += 1

        team_ms = (time.perf_counter() - t0) * 1e3 - pilot_ms
        self.timing["pilot_ms"] = pilot_ms
        self.timing["team_ms"] = team_ms
        if team_ms > self.timing["max_team_ms"]:
            self.timing["max_team_ms"] = team_ms
        return out

    @property
    def stats(self):
        d = dict(self.T.stats)
        d.update(tick=self.T.tick, confirmer=self.T.confirmer,
                 n_records=len(self.T.records), n_blacklist=len(self.T.blacklist),
                 frozen=int(self.T.frozen[:max(self.T.n, 1)].sum()),
                 rgb_used=[int(x) for x in self.T.rgb_used[:max(self.T.n, 1)]],
                 team_ms=self.timing["team_ms"], last_ms=self.timing["last_ms"],
                 max_team_ms=self.timing["max_team_ms"])
        return d

_FLATMOD_team = _FlatModule(
    'team',
    MAX_DRONES=MAX_DRONES, STATE_DIM=STATE_DIM,
    PILOT_STATE_DIM=PILOT_STATE_DIM, S_POS=S_POS, S_RPY=S_RPY, S_VEL=S_VEL,
    S_AGL=S_AGL, S_CLUE=S_CLUE, S_MATES=S_MATES, N_MATE_SLOTS=N_MATE_SLOTS,
    M_EST=M_EST, M_NDET=M_NDET, M_LOCK=M_LOCK, M_COMMIT=M_COMMIT,
    M_AIR_T=M_AIR_T, M_PENV=M_PENV, M_REQ_USED=M_REQ_USED, M_ANTI2=M_ANTI2,
    M_ANTI2_VALID=M_ANTI2_VALID, M_GU_CD=M_GU_CD, M_BIN_DEC=M_BIN_DEC,
    M_DET_CH=M_DET_CH, M_FMODE=M_FMODE, M_MTN_F=M_MTN_F,
    M_LANE_IDX=M_LANE_IDX, M_CONFIRMER=M_CONFIRMER, M_ARRIVED=M_ARRIVED,
    M_LANE_WP=M_LANE_WP, M_WP_DWELL=M_WP_DWELL, _LOGGED=_LOGGED, _log=_log,
    _vel_of=_vel_of, _store_vel=_store_vel, _enforce_vz_min=_enforce_vz_min,
    _enforce_vz_max=_enforce_vz_max, _horiz=_horiz, _mem_get=_mem_get,
    _mem_set=_mem_set, _new_mem=_new_mem, _as_mem=_as_mem,
    _PilotAdapter=_PilotAdapter, _call_flexible=_call_flexible,
    _ClueRegion=_ClueRegion, _RegionAdapter=_RegionAdapter,
    _TeamState=_TeamState, TeamLayer=TeamLayer)

import traceback
AGL_FLOOR_M = 2.5
HOVER_CLIMB = 0.35

ACT_LO = np.array([-1.0, -1.0, -1.0, 0.0, -1.0, 0.0], dtype=np.float32)
ACT_HI = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

def _log_once(key, msg, exc=None):
    if key in _LOGGED or len(_LOGGED) > 32:
        return
    _LOGGED.add(key)
    try:
        sys.stderr.write("[swarm_sar] %s\n" % msg)
        if exc is not None:
            sys.stderr.write(traceback.format_exc()[-2000:] + "\n")
        sys.stderr.flush()
    except Exception:
        pass

_TAG = "_swsar_%x" % (abs(hash(_HERE)) & 0xFFFFFFF)
_pilot_mod = _FLATMOD_pilot_v21
_team_mod = _FLATMOD_team

def _pilot_factory(models_dir, nets=None):
    return _pilot_mod.PerDronePilot(models_dir, nets)

class DroneFlightController:

    def __init__(self):
        self.team = None
        self._n_last = 2
        self._build_tries = 0
        self._build()

    def _build(self):
        self._build_tries += 1
        if _team_mod is None or _pilot_mod is None:
            _log_once("no_modules", "team.py / pilot_v21.py unavailable -- hover only")
            return
        try:
            self.team = _team_mod.TeamLayer(_HERE, _pilot_factory)
        except Exception as exc:
            self.team = None
            _log_once("team_ctor", "TeamLayer construction failed: %r" % (exc,), exc)

    def reset(self):
        if self.team is None and self._build_tries < 2:
            self._build()
        if self.team is None:
            return
        try:
            self.team.reset()
        except Exception as exc:
            _log_once("team_reset", "TeamLayer.reset() failed: %r" % (exc,), exc)
            self.team = None
            if self._build_tries < 3:
                self._build()

    def act(self, obs):
        state = None
        n = self._n_last
        try:
            state = np.asarray(obs["state"], dtype=np.float32)
            if state.ndim == 1:
                state = state.reshape(1, -1)
            elif state.ndim > 2:
                state = state.reshape(-1, state.shape[-1])
            n = int(state.shape[0])
            self._n_last = n
        except Exception as exc:
            _log_once("bad_state", "unreadable obs['state']: %r" % (exc,), exc)

        if self.team is not None:
            try:
                out = self.team.act(obs)
                out = np.ascontiguousarray(np.asarray(out, dtype=np.float32))
                if out.shape == (n, ACT_DIM) and np.all(np.isfinite(out)):
                    np.clip(out, ACT_LO, ACT_HI, out=out)
                    return out
                _log_once("bad_out",
                          "TeamLayer returned shape %r finite=%s; hovering"
                          % (out.shape, bool(np.all(np.isfinite(out)))))
            except Exception as exc:
                _log_once("team_act", "TeamLayer.act() raised %r; hovering"
                          % (exc,), exc)
        return self._hover(n, state)

    @staticmethod
    def _hover(n, state):
        n = max(1, int(n))
        out = np.zeros((n, ACT_DIM), dtype=np.float32)
        if state is None:
            return out
        try:
            rows = min(n, int(state.shape[0]))
            w = int(state.shape[1])
            for i in range(rows):
                if w > 5:

                    out[i, 4] = np.float32(np.clip(state[i, 5] / np.pi, -1.0, 1.0))
                if w > 162 and float(state[i, 162]) * 20.0 < AGL_FLOOR_M:
                    out[i, 2] = np.float32(1.0)
                    out[i, 3] = np.float32(HOVER_CLIMB)
        except Exception:
            pass
        return out

    @property
    def stats(self):
        return {} if self.team is None else self.team.stats

    @property
    def timing(self):
        return {} if self.team is None else self.team.timing

_np = np

_MAPSW_MAPS = ('city', 'open', 'mountain', 'village', 'forest')
_MAPSW_TICK = 15
_MAPSW_GATE_BY_MAP = {'open': 4.0, 'mountain': 4.0, 'city': 1.05, 'village': 1.05, 'forest': 1.05}
_MAPSW_BASE_GATE = GROUND_GATE_M

_MAPSW_MU = _np.asarray([
    0.0, 0.5284554960834771, 0.3745140772851924, 0.9924079641808693, 0.043162963024530286, 0.0779888399466429,
    0.09507136588164339, 0.1737553367272529, 0.5676305832299602, 0.8626663857654823, 0.9571032424592814, 0.9770041689942508,
    0.9877353893849666, 0.2795016955421086, 0.4179571508528689, 0.41816642658294195, 0.4152402557494146, 0.009232208615257625,
    0.008954477323310228, 0.011014554056214505, 0.8478243828861494, 0.8772399018420746, 0.8765674408960713, 0.8493092620840076,
    0.7391448580302479, 0.7789580330575875, 0.7778495677271363, 0.7387758959586487, 0.36821442574909885, 0.3771091295744653,
    0.37657931592033783, 0.37120170634484334, 0.12191333673794098, 0.11496125055489041, 0.11499087327953206, 0.12464855338783783,
    0.7035410013780015, 0.7361203322983966, 0.7382645024508107, 0.700534022280561, 0.5308797840015368, 0.5549859334522812,
    0.552073963611045, 0.5373569248918091, 0.16983645393846714, 0.16682185930145055, 0.16600507644541865, 0.17017913422041053,
    0.0649883074891532, 0.0, 0.0, 0.06392025950004922, 0.955189369880725, 0.9664862017009264,
    0.9660238251312603, 0.9524141291284621, 0.9103354540244366, 0.9356330431116274, 0.9352821140447826, 0.9088539541655432,
    0.8043638855109023, 0.8545735301677821, 0.8551779583300919, 0.804851100408365, 0.2040040798220253, 0.20475438345034305,
    0.2071206096151685, 0.20677485909056909,
], dtype=_np.float64)

_MAPSW_SD = _np.asarray([
    1e-06, 0.14796869270979982, 0.04838867560815698, 0.042460300852126286, 0.06333050049158864, 0.0751780336209004,
    0.08713027607483, 0.14336632648167696, 0.3267546659113263, 0.18130884292773336, 0.09538511821844654, 0.07385451030346481,
    0.05633327100246892, 0.17750195943169106, 0.17242580991437886, 0.1435944622844582, 0.142995190880814, 0.00838908274290572,
    0.009085742043220213, 0.011866490474070188, 0.1779903279132858, 0.16273738298493795, 0.16219161068416793, 0.17418655783594889,
    0.23090547224878663, 0.2102360042573781, 0.2064234749484013, 0.2288588505300987, 0.21947237416013388, 0.21415255579721626,
    0.21492515786365804, 0.22058027894452778, 0.10109220039237099, 0.09433925995368644, 0.09532287416320118, 0.10374131555940347,
    0.2828637557820449, 0.27741706480562045, 0.27482170765069824, 0.28385600726901594, 0.3176018439788455, 0.31495232701166387,
    0.3163016750832633, 0.3196916076446118, 0.14463620303835836, 0.1415439625153799, 0.14220249090722578, 0.1455672231825242,
    0.06808740757990761, 1e-06, 1e-06, 0.06828182021976466, 0.10051627908796974, 0.08982172937352072,
    0.08518935379757617, 0.09870193222859049, 0.1520244581135138, 0.12905167651880872, 0.12520122811728923, 0.1498381687979377,
    0.242648329415076, 0.21694582143808358, 0.2131819309194473, 0.23908069121865705, 0.1585487587921501, 0.1573378922400863,
    0.15919099386353897, 0.16341199232632128,
], dtype=_np.float64)

_MAPSW_W = _np.asarray([
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.10031318563091189, 0.1466688501106067, 0.051855311034851206, -0.20761189147105716, -0.09122545530531266],
    [-0.13598003578655066, -0.03984162009724184, 0.053732831441958215, 0.2872823011791377, -0.1651934767373035],
    [-0.04491397588656192, 0.0007746867759536978, 0.01781382807811099, -0.02904438546238306, 0.05536984649488024],
    [0.2257487813190064, 0.009688016573652595, -0.11841698278376993, -0.07854835016244457, -0.03847146494644454],
    [0.1498322978183922, 0.25297876157781635, -0.16472592982432752, -0.15400186660512397, -0.08408326296675699],
    [0.09840021481030209, 0.14469270789450578, -0.01554376823940462, -0.15662482618817608, -0.07092432827722706],
    [-0.03126071430131841, 0.18076540084641363, 0.038151748754252625, -0.15446974716111772, -0.03318668813823008],
    [0.21624749537574028, 0.28110775109509434, -0.007710595487580529, -0.42049089494782615, -0.06915375603542803],
    [0.10454856922059146, -0.018885950665012793, 0.015048827728078843, 0.05698749632706428, -0.15769894261072173],
    [0.07051279330873356, -0.013848359569791087, -0.02287783804503417, 0.023972828496961373, -0.057759424190869726],
    [0.02984182023767696, -0.00653620411620648, -0.001762253858668013, -0.024650177604530584, 0.0031068153417281373],
    [0.005405347976176577, -0.0034893838684027785, -0.17277706275379953, 0.09888461996449828, 0.07197647868152729],
    [-0.7591501881089957, 0.2169294088629925, 0.27045286586142425, 0.33265326022073705, -0.06088534683615828],
    [-0.11718049859747322, -0.13266352033844062, -0.024957213514908403, 0.35801217622141795, -0.08321094377059582],
    [-0.17929054036580855, 0.3563496286354841, 0.08697853543997054, -0.04219339392371331, -0.2218442297859327],
    [-0.19699618768546756, 0.38221785106659906, 0.06885678544116317, -0.031168942812082313, -0.22290950601021217],
    [-0.009522800190108285, 0.008144503794529748, 0.06827006143138033, -0.21985193451399182, 0.15296016947819013],
    [-0.12363397649441037, -0.17512657848061494, -0.1479837434469645, -0.13499380871081607, 0.5817381071328058],
    [0.0012420583215399087, -0.1934591504753011, -0.348374344110351, -0.16169168552707808, 0.7022831217911906],
    [-0.10727327954175066, 0.14877863362227453, 0.0770506991143967, 0.09884273885956195, -0.2173987920544825],
    [-0.06674348646468557, 0.05271673046554978, 0.2191496411036555, 0.016500326581720886, -0.22162321168624055],
    [-0.04777120823028581, 0.058725079301422727, 0.21715438268115708, 0.03590991397918042, -0.26401816773147413],
    [-0.06858375968385937, 0.10719197749237229, 0.12283106658018415, 0.08909742162382456, -0.25053670601252154],
    [0.1580198523074397, 0.24864187787389114, -0.15225875887839788, -0.23111157989552578, -0.023291391407407366],
    [0.18047008190434127, 0.10654141266766004, -0.05677049208269936, -0.13911573597724208, -0.09112526651205984],
    [0.22444905132286513, 0.11850404418167984, -0.07295924523965407, -0.1428086688424617, -0.12718518142242893],
    [0.19394056752287994, 0.19476816947018227, -0.1753303676205141, -0.1955844803884755, -0.01779388898407256],
    [0.07426139784163108, 0.22869433682096654, 0.05388069969561533, -0.34417915672681554, -0.012657277631397715],
    [0.15818043812519647, 0.10466967503256398, 0.04367290366665487, -0.2980352827913341, -0.008487734033081257],
    [0.19420727025920823, 0.06324668412673193, 0.05513353111079758, -0.30096114712161903, -0.011626338375118447],
    [0.18294206305898789, 0.1499274505990976, 0.08929537689785885, -0.3865594111040519, -0.0356054794518926],
    [-0.03338223487306928, 0.09239649661598932, 0.1922152451256118, -0.24708204803991166, -0.004147458828620045],
    [-0.028542751282220245, 0.05389136993225217, 0.09992721751629641, -0.19184042651013639, 0.06656459034380806],
    [-0.011325427487883018, -0.029183339411964306, 0.1715219785307579, -0.20166175990846538, 0.0706485482775549],
    [-0.0301127335195141, 0.040673132545318416, 0.20357692099430513, -0.2378734073349064, 0.02373608731479694],
    [0.006609842627934969, 0.23634714371110174, 0.16298048778330682, -0.08129971860706629, -0.3246377555152775],
    [0.031887414304955196, 0.11119535703946755, 0.13644389021561493, -7.694634785672567e-05, -0.279449715212181],
    [-0.061197545051270515, 0.12433331403668733, 0.17381108796045777, 0.0638004460651763, -0.3007473030110508],
    [-0.007892004532453015, 0.23654371801271412, 0.04344933997751499, 0.06357746911059428, -0.3356785225683701],
    [-0.2858678499902974, 0.9150300502270919, -0.15823232976981713, -0.31769115165940615, -0.1532387188075706],
    [-0.1461810458856364, 0.6211116267150001, -0.1706608336718111, -0.12210972666034245, -0.1821600204972098],
    [-0.0071551639819553365, 0.5240877087298308, -0.18929673604343647, -0.19811286500374833, -0.12952294370069037],
    [-0.14369416067574317, 0.7000676122005757, -0.12630936934227466, -0.24189234978129406, -0.18817173240126409],
    [-0.009866316021474234, 0.23305171092282279, 0.05037118408492015, -0.174379863718479, -0.09917671526778989],
    [0.004213866142404243, 0.20341005869523465, 0.010546706078924502, -0.13864624673107026, -0.07952438418549305],
    [0.08380540952256503, 0.0895284204949665, 0.04096755978108175, -0.13665992984355219, -0.07764145995506096],
    [0.051834087471095185, 0.15130175238974186, 0.07173787057586596, -0.16753298219137058, -0.1073407282453323],
    [0.026346420481702708, 0.12554832866471513, 0.03249478355429826, -0.15120483435928078, -0.03318469834143534],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.04344878755215431, 0.08673401374059549, 0.056291785283076136, -0.14444053991893657, -0.04203404665688942],
    [-0.2220360821331481, 0.01988811249481536, 0.11146167978220918, -0.018049078273374344, 0.10873536812949784],
    [-0.08947717839863077, 0.014089797478259659, 0.0969725698226046, -0.09234439918917045, 0.07075921028693688],
    [-0.08644236891343544, -0.0025072558103636255, 0.1199662625659793, -0.10635143460433882, 0.07533479676215861],
    [-0.15835826626610255, 0.03658554226602956, 0.0945705353658098, 0.044293634957756914, -0.01709144632349366],
    [-0.08782413394221482, 0.04542329372636941, -0.27175884491065383, 0.07613199961920275, 0.23802768550729622],
    [-0.04212292363157163, 0.006820688574332036, -0.08222405378649501, -0.10379491728903488, 0.22132120613276954],
    [-0.08384222934913185, -0.003747072929426531, -0.08380963432722947, -0.0746976181413381, 0.24609655474712583],
    [-0.1558065575755141, 0.02541113707474868, -0.19561501279920118, -0.057483981987904255, 0.3834944152878705],
    [0.28557339013610783, 0.14658011128459522, -0.7086009096138735, 0.14947420858794438, 0.12697319960522588],
    [0.2712290289926475, 0.008643371871126548, -0.5896366689448167, 0.2709196934110955, 0.03884457466994727],
    [0.27685486863707465, 0.012298291148075745, -0.6526187354390154, 0.3382840538476607, 0.02518152180620399],
    [0.2464687488005347, 0.050811630264936715, -0.5916850361945009, 0.0703424287441897, 0.22406222838483988],
    [-0.3801263655992552, 0.13210118293150777, 0.3689107791268328, -0.3216805670098573, 0.20079497055077172],
    [-0.3014532767292272, 0.04075260695811357, 0.2612075609154013, -0.24397103034951179, 0.2434641392052245],
    [-0.2822691417161962, -0.0076140935777390594, 0.2921963320752433, -0.2697092196714107, 0.26739612289010195],
    [-0.2883360901990163, 0.02936597145845228, 0.4009828609443823, -0.3089133374328609, 0.16690059522904283],
], dtype=_np.float64)

_MAPSW_B = _np.asarray([2.2915123565291373, -3.7159032612579033, 1.558257490446752, -1.594573430524937, 1.4607068448069505], dtype=_np.float64)

_MAPSW_STATE = {"map": None}

def _mapsw_gate():
    if os.environ.get('SAR_MAPSW', '1') != '1':
        return _MAPSW_BASE_GATE
    if _MAPSW_STATE["map"] in ('city', 'village', 'forest'):
        return 1.05
    return float(os.environ.get('SAR_MAPSW_HI', '4.0'))

def _mapsw_classify(obs):
    try:
        st = _np.asarray(obs["state"], _np.float32)
        st = st.reshape(1, -1) if st.ndim == 1 else st.reshape(-1, st.shape[-1])
        n = st.shape[0]
        dp = _np.asarray(obs["depth"], _np.float32).reshape(n, DEPTH_RES, DEPTH_RES, 1)

        feats = [_np.asarray(_map_depth_feature_values(dp[i]), _np.float64) for i in range(n)]
        f = _np.mean(feats, axis=0)
        if f.shape[0] != _MAPSW_MU.shape[0]:
            return None
        z = (f - _MAPSW_MU) / _MAPSW_SD
        full = _MAPSW_MAPS[int(_np.argmax(z @ _MAPSW_W + _MAPSW_B))]
        if P1_MSW and n > 1:
            # P4: forest seeds with tree-top pads (10-22 m starts) tip the averaged vote to 'mountain' and switch
            # the OMB barrier off (b1-331). Re-vote from the low drones only and take that answer only when it
            # says forest or city (both carry the OMB barrier) and the low drones sit near the base plane (z <= 5 m;
            # mountain pads at 12-20 m voted 'city' on b2-674 and the seed was lost) -- mountain pads also spread 30+ m in z (b2-170/180 went 'open'/'village' under a
            # plain low vote and collided).
            zs = st[:, 2].astype(_np.float64)
            low = [i for i in range(n) if zs[i] <= float(zs.min()) + 6.0]
            if 0 < len(low) < n and full == 'mountain' and float(zs.min()) <= 5.0:
                fl = _np.mean([feats[i] for i in low], axis=0)
                zl = (fl - _MAPSW_MU) / _MAPSW_SD
                lowvote = _MAPSW_MAPS[int(_np.argmax(zl @ _MAPSW_W + _MAPSW_B))]
                if lowvote in ('forest', 'city'):
                    return lowvote
        return full
    except Exception:
        return None

_MapswBase = DroneFlightController

class DroneFlightController(_MapswBase):

    def reset(self):
        _MAPSW_STATE["map"] = None
        self._mapsw_tick = 0
        return _MapswBase.reset(self)

    def act(self, observation):
        t = getattr(self, "_mapsw_tick", 0)

        if t == _MAPSW_TICK and _MAPSW_STATE["map"] is None:
            _MAPSW_STATE["map"] = _mapsw_classify(observation)
        self._mapsw_tick = t + 1
        return _MapswBase.act(self, observation)

_UID66_TERMINAL_BASE = DroneFlightController

def _uid66_public_terminal_candidates(team):
    out = []
    if team is None:
        return out
    for i in range(len(getattr(team, "pilots", []))):
        try:
            b = team._belief(i)
            e = np.asarray(b.get("est", [np.nan] * 3), dtype=np.float64)
            if np.all(np.isfinite(e)) and float(b.get("n_det", 0.0)) > 0.0:
                out.append((e, i, "pilot"))
        except Exception:
            pass
    T = getattr(team, "T", None)
    for j, r in enumerate(list(getattr(T, "records", []) or [])):
        try:
            xy = np.asarray(r.get("xy", [np.nan] * 2), dtype=np.float64)
            z = float(r.get("z", np.nan))
            if np.all(np.isfinite(xy)) and np.isfinite(z):
                out.append((np.r_[xy, z], j, "record"))
        except Exception:
            pass
    return out

def _uid66_consensus_horizontal(ctrl, action, observation):
    team = getattr(ctrl, "team", None)
    T = getattr(team, "T", None) if team is not None else None
    confirmer = int(getattr(T, "confirmer", -1)) if T is not None else -1
    state = np.asarray(observation["state"], dtype=np.float32)
    state = state.reshape(1, -1) if state.ndim == 1 else state.reshape(-1, state.shape[-1])
    cs = _uid66_public_terminal_candidates(team)
    if not (0 <= confirmer < len(state)) or len(cs) < 2:
        return action
    xy = np.asarray([x[0][:2] for x in cs], dtype=np.float64)
    med = np.median(xy, axis=0)
    keep = np.flatnonzero(np.linalg.norm(xy - med, axis=1) <= 2.0)
    if len(keep) < 2:
        return action
    chosen = [cs[i] for i in keep]
    med = np.median(np.asarray([x[0][:2] for x in chosen]), axis=0)
    if max(np.linalg.norm(x[0][:2] - med) for x in chosen) > 2.0:
        return action
    pos = np.asarray(state[:, 0:3], dtype=np.float64)
    est = np.median(np.asarray([x[0] for x in chosen]), axis=0)
    if np.linalg.norm(pos[confirmer, :2] - est[:2]) > 2.0:
        return action
    head = []
    for _e, i, src in chosen:
        if src == "pilot":
            try:
                head.append(float(team.pilots[i].p.ctrl.locked_head_z))
            except Exception:
                pass
    head_z = float(np.median(head)) if head and np.all(np.isfinite(head)) else float(est[2] + 0.5)
    target = np.r_[est[:2], head_z + 3.0]
    old = np.asarray(action[confirmer, :3], dtype=np.float64) * float(action[confirmer, 3]) * 3.0
    v = (target - pos[confirmer]) / 2.0
    v[2] = old[2]
    n = float(np.linalg.norm(v))
    if P1_U66 and abs(float(pos[confirmer, 2]) - float(target[2])) > 1.5:
        # P5: still well above the band -- cap the horizontal residual only, keep the descent
        nh = float(np.hypot(v[0], v[1]))
        if nh > 1.0:
            v[0] /= nh
            v[1] /= nh
        n = float(np.linalg.norm(v))
        if n > 3.0:
            v *= 3.0 / n
            n = 3.0
    elif n > 1.0:
        v *= 1.0 / n
        n = 1.0
    if n > 1e-12:
        action[confirmer, :3] = (v / n).astype(np.float32)
        action[confirmer, 3] = np.float32(n / 3.0)
    else:
        action[confirmer, :4] = 0.0
    return action

_UID66_RESID_ON = os.environ.get('SAR66_RESID', '1') == '1'
_UID66_RESID_SKIP = tuple(x for x in os.environ.get('SAR66_SKIPMAPS', 'village').split(',') if x)


_SPIN_DBG_KEYS = ("spin_ticks", "spin_started", "spin_done", "spin_stopped_seen", "spin_stopped_team", "spin_skipped",
                  "gap_started", "gap_found", "gap_none", "gap_ticks", "gap_high_ticks",
                  "lad_started", "lad_steps", "lad_ticks", "lad_ended",
                  "skn_checks", "skn_looks", "skn_yes", "skn_req", "skn_confirm", "skn_reject", "nw_leave",
                  "coh_started", "coh_ticks", "coh_timeout", "coh_guard")


class DroneFlightController(_UID66_TERMINAL_BASE):
    def debug_state(self):
        """Probe channels: the look-around counters summed over the pilots (spin_*), and every drone's
        latest depth-head verdict with the world point it measured (det_p<k>, det_z<k>)."""
        team = getattr(self, "team", None)
        if team is None:
            return {}
        try:
            T = team.T
            out = {"forest": 1.0 if (_forest() or _skin_world()) else 0.0, "confirmer": float(T.confirmer),
                   "pilot_exceptions": float(T.stats.get("pilot_exceptions", 0)),
                   "deadline_hits": float(T.stats.get("deadline_hits", 0))}
            nan = float("nan")
            seen_t = getattr(self, "_dbg_det_t", None)
            if seen_t is None:
                seen_t = self._dbg_det_t = {}
            n = max(1, min(int(T.n), len(team.pilots)))
            agg = {k: 0.0 for k in _SPIN_DBG_KEYS}
            mq = {"mq_refute": 0.0, "mq_hold": 0.0, "mq_protect": 0.0, "msw_giveup": 0.0, "mq_approach": 0.0, "mtk_full": 0.0, "ma_low": 0.0, "mrgb_check": 0.0, "msp_turns": 0.0, "msp_ticks": 0.0, "msp_done": 0.0, "msp_stop_seen": 0.0, "msp_stop_terrain": 0.0, "msp_stop_mode": 0.0, "fsp_ticks": 0.0, "vcap": 0.0, "vfc_bar": 0.0, "fix_set": 0.0, "fix_freeze": 0.0, "fix_noblack": 0.0, "fix_denyskip": 0.0, "tk_hold": 0.0, "tk_stop": 0.0, "tk_turns": 0.0, "fa_low": 0.0, "fa_hold": 0.0, "fix_rehover": 0.0, "tks_start": 0.0, "tks_ticks": 0.0, "tks_hold": 0.0, "tks_done": 0.0, "tks_stop_busy": 0.0, "tks_verify": 0.0, "tks_verify_ticks": 0.0, "tks_verify_lost": 0.0, "tks_verify_timeout": 0.0, "tks_timeout": 0.0, "tks_verify_commit": 0.0}
            for k in range(n):
                p = getattr(team.pilots[k], "p", None)
                det = getattr(getattr(p, "ctrl", None), "detector", None)
                if det is None:
                    continue
                ss = getattr(p, "spin_stats", None) or {}
                for kk in _SPIN_DBG_KEYS:
                    agg[kk] += float(ss.get(kk, 0))
                served = int(getattr(p, "_served", 0))
                fresh = served != seen_t.get(k, -1)
                seen_t[k] = served
                pd = float(getattr(det, "last_p_depth", 0.0)) if fresh else nan
                out["det_p%d" % k] = pd
                zz = getattr(det, "last_z", None)
                if fresh and zz is not None and pd >= 0.70:
                    out["det_z%d" % k] = [float(zz[0]), float(zz[1]), float(zz[2])]
                else:
                    out["det_z%d" % k] = [nan, nan, nan]
                c = p.ctrl
                agl = getattr(c, "_last_agl", None)
                out["hov%d" % k] = [float(c.locked_head_z) if (c.locked or c.frozen) else nan,
                                    float(agl) if (agl is not None and np.isfinite(agl)) else nan,
                                    float({'initial': 0, 'search': 1, 'navigation': 2, 'hover': 3}.get(c.mode, -1))
                                    + (10.0 if c.locked else 0.0) + (20.0 if c.frozen else 0.0)]
                _sw = getattr(c, "_sweep_t0", None)
                out["hov2%d" % k] = [float(getattr(c, "_dbg_hover_tz", nan)) if c.mode == 'hover' else nan,
                                     float(c.tick - _sw) / CTRL_HZ if _sw is not None else nan,
                                     float(getattr(c, "hover_stable_ticks", 0)), float(getattr(c, "hover_total_ticks", 0))]
                for kk in mq:
                    mq[kk] += float((getattr(c, "mq_stats", None) or {}).get(kk, 0))
                _om = getattr(p, "_omb", None)
                if _om is not None:
                    for kk, vv in _om.stats.items():
                        if kk == "omb_min_r":
                            mq[kk] = min(mq.get(kk, 99.0), float(vv))
                        else:
                            mq[kk] = mq.get(kk, 0.0) + float(vv)
                out["mq%d" % k] = [float(getattr(c, "mq_view", 0)), float(getattr(c, "mq_hits", 0)), float(getattr(c, "mq_hold_left", 0))]
                mq["fix_lifted"] = mq.get("fix_lifted", 0.0) + float(getattr(c.detector, "fix_lifted", 0))
                _ms = getattr(c, "mq_stats", None) or {}
                for kk in ("mcv_started", "mcv_ticks", "tks_stall", "tel_lockguard", "lockjump_held", "mbt_ticks"):
                    mq[kk] = mq.get(kk, 0.0) + float(_ms.get(kk, 0))
                for kk in ("dc_clamped", "edge_hits", "rgbv3_held", "rgbv2_vetoes", "rgbv2_deferred"):
                    mq[kk] = mq.get(kk, 0.0) + float(getattr(c.detector, kk, 0))
                for kk in ("fbd_started", "fbd_ticks", "fbd_clear", "fbd_timeout", "tel2_self"):
                    mq[kk] = mq.get(kk, 0.0) + float(ss.get(kk, 0))
            for kk in _SPIN_DBG_KEYS:
                out[kk] = agg[kk]
            out.update(mq)
            return out
        except Exception:
            return {}

    def act(self, observation):
        action = _UID66_TERMINAL_BASE.act(self, observation)
        if not _UID66_RESID_ON:
            return action
        if _UID66_RESID_SKIP and _MAPSW_STATE.get("map") in _UID66_RESID_SKIP:
            return action
        try:
            return _uid66_consensus_horizontal(self, action, observation)
        except Exception:
            return action

