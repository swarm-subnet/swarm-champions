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
SCAN_V = float(__import__('os').environ.get('NEWDET_SCAN_V', '2.4'))
USABLE_SEC = 52.0
CAP_LO, CAP_HI = 4.0, 45.0
WP_SEP = float(__import__('os').environ.get('NEWDET_WP_SEP', '14.0'))
WP_MAX = int(__import__('os').environ.get('NEWDET_WP_MAX', '6'))
ADVANCE_R = 8.0
PASS_OFFSET_M = float(__import__('os').environ.get('NEWDET_PASS_OFFSET', '10'))
HOLD_SEC = float(__import__('os').environ.get('NEWDET_HOLD_SEC', '2.0'))
CENTRE_EPS = 3.0
SCAN_R_REF = float(__import__('os').environ.get('NEWDET_SCAN_R_REF', '26.0'))
SCAN_R_MIN = 0.45
import os
MFP_ON = os.environ.get('SWSAR_MFP', '1') == '1'
MFP_F = float(os.environ.get('SWSAR_MFP_F', '0.057'))
MFP_F_FAR = float(os.environ.get('SWSAR_MFP_FFAR', '0.037'))
MFP_AIN_MAX = float(os.environ.get('SWSAR_MFP_AIN', '0.2'))
MFP_SLACK = float(os.environ.get('SWSAR_MFP_SLACK', '50.0'))
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
        p0 = self.pad0_xy
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
    advance_r = 0.0
    advance_hold = 2.0
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
            tour = self._pass_shifted(r, tour, npass)
        if self.advance_r > 0.0 and j < m - 1:
            idx = m - 1 - j if rev else j
            if float(np.hypot(tour[idx, 0] - p[0], tour[idx, 1] - p[1])) < self.advance_r:
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
NEWDET_ON = os.environ.get('NEWDET', '1') == '1'
NEWDET_MAPS = tuple(m for m in os.environ.get('NEWDET_MAPS', '').split(',') if m)
NEWDET_UNKNOWN = os.environ.get('NEWDET_UNKNOWN', '124')
NEWDET_H_MODE = os.environ.get('NEWDET_H', 'champ')
NEWDET_H_CONST = float(os.environ.get('NEWDET_HC', '1.6'))
NEWDET_T = float(os.environ.get('NEWDET_T', '0'))
NEWDET_T_MAP = {m: float(os.environ.get('NEWDET_T_' + m.upper(), '0')) for m in ('city', 'open', 'mountain', 'village', 'forest')}
NEWDET_N_LATCH = float(os.environ.get('NEWDET_NLATCH', '0'))
NEWDET_LR_OFF_MAPS = tuple(m for m in os.environ.get('NEWDET_LR_OFF', '').split(',') if m)
NEWDET_MIX = os.environ.get('NEWDET_MIX', 'new')
NEWDET_AND_P = float(os.environ.get('NEWDET_AND_P', '0'))
NEWDET_AND_R = float(os.environ.get('NEWDET_AND_R', '15'))
NEWDET_NEAR_M = float(os.environ.get('NEWDET_NEAR', '0'))
NEWDET_AVG_M = float(os.environ.get('NEWDET_AVG', '0'))
NEWDET_MAXP = os.environ.get('NEWDET_MAXP', '0') == '1'
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
RGBV2_ON = os.environ.get('SWSAR_RGBV2', '1') == '1'
RGBV2_NO_NEEDED = int(os.environ.get('SWSAR_RGBV2_N', '2'))
RGBV2_WINDOW = int(os.environ.get('SWSAR_RGBV2_WIN', '150'))
RGBV2_STRONG_P = float(os.environ.get('SWSAR_RGBV2_P', '0.85'))
RGBV3_TRACK_MIN = float(os.environ.get('SWSAR_RGBV3_TRK', '3.0'))
RGBV2_P_VILLAGE = float(os.environ.get('SWSAR_RGBV2_P_VILLAGE', '0.6'))
RGBV3_HOLD = os.environ.get('SWSAR_RGBV3_HOLD', '1') == '1'
RGBM_P = float(os.environ.get('SARZ_RGBM_P', '0.70'))
RGBM_HOLD = os.environ.get('SARZ_RGBM_HOLD', '0') == '1'
RGBM_HOLD_N = float(os.environ.get('SARZ_RGBM_HOLD_N', '4.0'))
RGBM_WIN = int(os.environ.get('SARZ_RGBM_WIN', '150'))
_RGBM = {}
def _rgbm_reset():
    _RGBM.clear()
    _RGBM.update(defer=0, t0=-1.0, x0=-9999.0, y0=-9999.0, latch=0, lt=-1.0, lx=-9999.0, ly=-9999.0, reveto=0, hold=0)
_rgbm_reset()
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
MSWZ_Z = float(os.environ.get('SWSAR_MSWZ_Z', '5.0'))
MSWZ_TO = os.environ.get('SWSAR_MSWZ_TO', 'village')
MZL_Z = float(os.environ.get('SWSAR_MZL_Z', '25.0'))
MZL_MODE = os.environ.get('SWSAR_MZL_MODE', 'late')
MSWF_Z = float(os.environ.get('SWSAR_MSWF_Z', '11.0'))
MSWF_ZLO = float(os.environ.get('SWSAR_MSWF_ZLO', '3.5'))
MTYP_ON = os.environ.get('SWSAR_MOUNTAIN_TYP', '0') == '1'
MTYP_MODE = os.environ.get('SWSAR_MOUNTAIN_TYP_MODE', 'support')
MTYP_RELABEL = os.environ.get('SWSAR_MOUNTAIN_TYP_RELABEL', '1') == '1'
MTYP_ZMIN = float(os.environ.get('SWSAR_MOUNTAIN_TYP_ZMIN', '5.5'))
MTYP_ZMAX = float(os.environ.get('SWSAR_MOUNTAIN_TYP_ZMAX', '12.0'))
def _mtyp_or(a, b):
    if isinstance(a, (bool, np.bool_)) and isinstance(b, (bool, np.bool_)):
        return bool(a) or bool(b)
    return np.logical_or(np.asarray(a, dtype=bool), np.asarray(b, dtype=bool))
class _MtypUnionRegion:
    def __init__(self, base, alt):
        self._mtyp_base = base
        self._mtyp_alt = alt
    def contains(self, xy):
        return _mtyp_or(self._mtyp_base.contains(xy), self._mtyp_alt.contains(xy))
    def contains_relaxed(self, xy):
        return _mtyp_or(self._mtyp_base.contains_relaxed(xy), self._mtyp_alt.contains_relaxed(xy))
    def __getattr__(self, k):
        return getattr(self.__dict__["_mtyp_base"], k)
MTN_RECON_ON = os.environ.get('SWSAR_MTN_RECON', '1') == '1'
MTN_RECON_TICK = int(os.environ.get('SWSAR_MTN_RECON_TICK', '150'))
MTN_RECON_ZMIN = float(os.environ.get('SWSAR_MTN_RECON_ZMIN', '6.0'))
MTN_RECON_NMAX = int(os.environ.get('SWSAR_MTN_RECON_NMAX', '0'))
MJ_FLOOR = float(os.environ.get('SWSAR_MJ_FLOOR', '0.02'))
MJ_WP_LOCK = os.environ.get('SWSAR_MJ_WP', '1') == '1'
EDGE_GUARD = os.environ.get('SWSAR_EDGE', '0') == '1'
EDGE_EL_DEG = float(os.environ.get('SWSAR_EDGE_EL', '40.0'))
EDGE_STRONG_MIN = float(os.environ.get('SWSAR_EDGE_SMIN', '6.0'))
EDGE_STEP_M = float(os.environ.get('SWSAR_EDGE_STEP', '1.0'))
EDGE_DROP_M = float(os.environ.get('SWSAR_EDGE_DROP', '0.6'))
VEG_ON = os.environ.get('SWSAR_VEG', '1') == '1'
VEG_MAPS = tuple(x for x in os.environ.get('SWSAR_VEG_MAPS', 'village').split(',') if x)
VEG_EL = float(os.environ.get('SWSAR_VEG_EL', '42.0'))
VEG_J = float(os.environ.get('SWSAR_VEG_J', '1.5'))
VEG_DZ = float(os.environ.get('SWSAR_VEG_DZ', '1.0'))
VEG_SMIN = float(os.environ.get('SWSAR_VEG_SMIN', '5.7'))
VEG_WEAK = os.environ.get('SWSAR_VEG_WEAK', '1') == '1'
_VEG = {"arm": 0, "drop": 0, "weak": 0}
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
LTD_ON     = os.environ.get('SWSAR_LTD', '1') == '1'
LTD_MARGIN = float(os.environ.get('SWSAR_LTD_MARGIN', '4.0'))
LTD_CAP    = float(os.environ.get('SWSAR_LTD_CAP', '14.0'))
LTD_DMIN   = float(os.environ.get('SWSAR_LTD_DMIN', '8.0'))
LTD_DMAX   = float(os.environ.get('SWSAR_LTD_DMAX', '28.0'))
LTD_PCT    = float(os.environ.get('SWSAR_LTD_PCT', '85.0'))
LTD_STRIDE = int(os.environ.get('SWSAR_LTD_STRIDE', '8'))
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
TKS_ON = os.environ.get('SWSAR_TKS', '1') == '1'
TKS_MAPS = tuple(x for x in os.environ.get('SWSAR_TKS_MAPS', 'village,city,open').split(',') if x)
TKS_R_NEAR = float(os.environ.get('SWSAR_TKS_RNEAR', '10.0'))
TKS_P_HOLD = float(os.environ.get('SWSAR_TKS_PHOLD', '0.12'))
TKS_DEG_FREE = float(os.environ.get('SWSAR_TKS_DEGFREE', '150.0'))
TKS_DEG_HOLD = float(os.environ.get('SWSAR_TKS_DEGHOLD', '300.0'))
TKS_RATE = float(os.environ.get('SWSAR_TKS_RATE', '60.0'))
TKS_MAX_TICKS = int(os.environ.get('SWSAR_TKS_MAXT', '500'))
TKS_T0 = 16
TKS_HOLD_AGL_MAX = float(os.environ.get('SWSAR_TKS_HOLD_AGLMAX', '4.5'))
TKS_VERIFY_COMMIT = int(os.environ.get('SWSAR_TKS_VCOMMIT', '2'))
TKS_HOLD_ABOVE_PAD = float(os.environ.get('SWSAR_TKS_HOLD_PAD', '3.3'))
TKS_VERIFY_MIN = int(os.environ.get('SWSAR_TKS_VMIN', '25'))
TKS_VERIFY_MAX = int(os.environ.get('SWSAR_TKS_VMAX', '75'))
TKS_VERIFY_N = int(os.environ.get('SWSAR_TKS_VN', '2'))
TKS_LEAD = float(os.environ.get('SWSAR_TKS_LEAD', '40.0'))
def _tks_pnear(map_name, n, clue_xy, pad0_xy, starts_xy, r_near):
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
FHS_ON = os.environ.get('SWSAR_FHS', '1') == '1'
FHS_MAPS = tuple(x for x in os.environ.get('SWSAR_FHS_MAPS', 'forest').split(',') if x)
FHS_T0 = int(os.environ.get('SWSAR_FHS_T0', '150'))
FHS_STEP = int(os.environ.get('SWSAR_FHS_STEP', '100'))
FHS_DZ = float(os.environ.get('SWSAR_FHS_DZ', '0.5'))
_FHZ_DZ = float(os.environ.get('SWSAR_FHZ_DZ', '0.0'))
HOVER_STABLE_M = 1.5
PHANTOM_VETO_M = CONSIST_M
RGB_VETO_TTL = 300
FOREST_ALT_M = float(os.environ.get('SAR_FOREST_ALT', '5.0'))
TK_ON = os.environ.get('SWSAR101_TK', '1') == '1'
TK_H = float(os.environ.get('SWSAR101_TK_H', '3.0'))
def _forest_alt():
    return TK_H if (TK_ON and _MAPSW_STATE.get("map") == 'forest') else FOREST_ALT_M
MQ_ON = os.environ.get('SWSAR201_MQ', '0') == '1'
MQ_VIEW_MIN = int(os.environ.get('SWSAR201_MQ_VIEW', '100'))
MQ_LOW = float(os.environ.get('SWSAR201_MQ_LOW', '0.0'))
MQ_HOLD_VIEW = int(os.environ.get('SWSAR201_MQ_HOLDVIEW', '50'))
MQ_HIGH = float(os.environ.get('SWSAR201_MQ_HIGH', '0.20'))
MQ_HOLD_TICKS = int(os.environ.get('SWSAR201_MQ_HOLD', '250'))
MQ_REFUTE_TTL = int(os.environ.get('SWSAR201_MQ_TTL', '500'))
MQ_RANGE_M = 25.0
MQ_DEP_DEG = 42.0
MQ_PROTECT_TICKS = int(os.environ.get('SWSAR201_MQ_PROTECT', '900'))
MSW_ON = os.environ.get('SWSAR201_MSW', '0') == '1'
MSW_LO = float(os.environ.get('SWSAR201_MSW_LO', '-2.0'))
MSW_MPS = float(os.environ.get('SWSAR201_MSW_MPS', '0.8'))
MSW_HI = float(os.environ.get('SWSAR201_MSW_HI', '4.6'))
MSW_TOP = 2.5
def _mq_on():
    return MQ_ON
MTK_ON = os.environ.get('SWSAR201_MTK', '1') == '1'
MTK_H = float(os.environ.get('SWSAR201_MTK_H', '3.0'))
MTK_CLIMB_SPEED = float(os.environ.get('SWSAR201_MTK_CLIMB', '1.0'))
MTK_SEARCH_H = float(os.environ.get('SWSAR201_MTK_SEARCH', '0'))
MTK_FULL_SPEED = os.environ.get('SWSAR201_MTK_FULL', '1') == '1'
INIT_EXIT_MARGIN = float(os.environ.get('NEWDET_INIT_MARGIN', '1.0'))
INIT_EXIT_TICKS = int(os.environ.get('NEWDET_INIT_TICKS', '0'))
N2_MAX = int(os.environ.get('NEWDET_N2_MAX', '2'))
N2_MTN_ALT = float(os.environ.get('NEWDET_N2_MTN_ALT', '4.5'))
N2_INIT_TICKS = int(os.environ.get('NEWDET_N2_INIT_TICKS', '300'))
N2_INIT_MARGIN = float(os.environ.get('NEWDET_N2_INIT_MARGIN', '0.3'))
_N2_DEFAULTS = None
NBIG_MIN = int(os.environ.get('NEWDET_NBIG_MIN', '6'))
NBIG_MTN_RSAFE = float(os.environ.get('NEWDET_NBIG_MTN_RSAFE', '0.5'))
_NBIG_DEFAULTS = None
NF_MAX = int(os.environ.get('NEWDET_NF_MAX', '3'))
NF_TKH = float(os.environ.get('NEWDET_NF_TKH', '2.7'))
_NF_DEFAULT = None
SEP_MIN_AGL = float(os.environ.get('NEWDET_SEP_MIN_AGL', '0'))
MSS_ON = os.environ.get('SWSAR_MSS', '0') == '1'
MSS_A_ON = float(os.environ.get('SWSAR_MSS_A_ON', '2.0'))
MSS_RATE = float(os.environ.get('SWSAR_MSS_RATE', '0.4'))
MSS_VH = float(os.environ.get('SWSAR_MSS_VH', '1.5'))
MSS_A_REL = float(os.environ.get('SWSAR_MSS_A_REL', '3.0'))
MSS_HOLD = int(os.environ.get('SWSAR_MSS_HOLD', '15'))
MSS_HMIN = float(os.environ.get('SWSAR_MSS_HMIN', '0.8'))
MSS_DH = float(os.environ.get('SWSAR_MSS_DH', '0.08'))
MSS_DHREL = float(os.environ.get('SWSAR_MSS_DHREL', '0.04'))
MSS_DZUP = float(os.environ.get('SWSAR_MSS_DZUP', '0.10'))
MSS_DZDN = float(os.environ.get('SWSAR_MSS_DZDN', '0.07'))
MSS_EZUP = float(os.environ.get('SWSAR_MSS_EZUP', '0.5'))
MSS_TOL = float(os.environ.get('SWSAR_MSS_TOL', '0.05'))
MSS_ALPHA = float(os.environ.get('SWSAR_MSS_ALPHA', '0.15'))
MH_ON = os.environ.get('SWSAR201_MH', '1') == '1'
MH_HMAX = float(os.environ.get('SWSAR201_MH_HMAX', '2.2'))
MH_HMIN = 0.3
MH_BIDIR = os.environ.get('SWSAR201_MH_BIDIR', '0') == '1'
MA_ON = os.environ.get('SWSAR201_MA', '1') == '1'
MA_R = float(os.environ.get('SWSAR302_MA_R', os.environ.get('SWSAR201_MA_R', '15.0')))
SARX_MAF = os.environ.get('SARX_MAF', '0') == '1'
SARX_MAG_DEG = float(os.environ.get('SARX_MAG_DEG', '0'))
SARX_MTKY = os.environ.get('SARX_MTKY', '1') == '1'
MEG_ON = os.environ.get('SWSAR_MTN_ENDGAME', os.environ.get('SWSAR_MEG', '1')) == '1'
MEG_A = os.environ.get('SWSAR_MTN_ENDGAME_A', '0') == '1'
MEG_HDS = float(os.environ.get('SWSAR_MTN_ENDGAME_HDS', '0.9'))
MEG_BRK = float(os.environ.get('SWSAR_MTN_ENDGAME_BRK', '0'))
MEG_BRK_M = float(os.environ.get('SWSAR_MTN_ENDGAME_BRK_M', '0.3'))
MEG_B = os.environ.get('SWSAR_MTN_ENDGAME_B', '1') == '1'
MEG_R = float(os.environ.get('SWSAR_MTN_ENDGAME_R', '25.0'))
MEG_GAP = float(os.environ.get('SWSAR_MTN_ENDGAME_GAP', '2.0'))
MEG_FRZ = os.environ.get('SWSAR_MTN_ENDGAME_FRZ', '1') == '1'
MEG_ZMIN = float(os.environ.get('SWSAR_MTN_ENDGAME_ZMIN', '2.0'))
MEG_FRZ_ZMIN = float(os.environ.get('SWSAR_MTN_ENDGAME_FRZ_ZMIN', '1.0'))
MEG_C = os.environ.get('SWSAR_MTN_ENDGAME_C', '0') == '1'
MEG_RC_H = float(os.environ.get('SWSAR_MTN_ENDGAME_RC_H', '0.8'))
MEG_RC_N = int(os.environ.get('SWSAR_MTN_ENDGAME_RC_N', '10'))
MEG_RC_TOL = float(os.environ.get('SWSAR_MTN_ENDGAME_RC_TOL', '0.15'))
MEG_RC_GATE = float(os.environ.get('SWSAR_MTN_ENDGAME_RC_GATE', '1.5'))
MEG_RC_LO = float(os.environ.get('SWSAR_MTN_ENDGAME_RC_LO', '2.4'))
MEG_RC_HI = float(os.environ.get('SWSAR_MTN_ENDGAME_RC_HI', '3.8'))
MEG_RC_AT = float(os.environ.get('SWSAR_MTN_ENDGAME_RC_AT', '3.0'))
MEG_RC_HMIN = float(os.environ.get('SWSAR_MTN_ENDGAME_RC_HMIN', '2.0'))
MEG_RC_HL = float(os.environ.get('SWSAR_MTN_ENDGAME_RC_HL', '0.5'))
MEG_RC_DN = os.environ.get('SWSAR_MTN_ENDGAME_RC_DN', '0') == '1'
MEG_RC_DNG = float(os.environ.get('SWSAR_MTN_ENDGAME_RC_DNG', '0.8'))
MEG_RC_UPG = float(os.environ.get('SWSAR_MTN_ENDGAME_RC_UPG', '1.0'))
MEG_RC_UPMAX = float(os.environ.get('SWSAR_MTN_ENDGAME_RC_UPMAX', '3.5'))
MEG_KEYS = ("meg_fast", "meg_glide", "meg_floor", "meg_frz", "meg_rc", "meg_rc_up", "meg_rc_dn")
def _meg_dbg_sum(team, key):
    try:
        tot = 0.0
        for _k in range(max(1, min(int(team.T.n), len(team.pilots)))):
            _c = getattr(getattr(team.pilots[_k], "p", None), "ctrl", None)
            if _c is not None:
                tot += float((getattr(_c, "mq_stats", None) or {}).get(key, 0))
        return tot
    except Exception:
        return 0.0
FIX_ON = os.environ.get('SWSAR302_FIX', '1') == '1'
FIX_WORLDS = tuple(x for x in os.environ.get('SWSAR302_FIX_WORLDS', 'mountain').split(',') if x)
FIX_RANGE_M = float(os.environ.get('SWSAR302_FIX_RANGE', '8.0'))
FIX_MIN = int(os.environ.get('SWSAR302_FIX_MIN', '10'))
FIX_WINDOW = int(os.environ.get('SWSAR302_FIX_WINDOW', '50'))
FIX_EVIDENCE = int(os.environ.get('SWSAR302_FIX_EVIDENCE', '20'))
FIX_NEAR_M = 6.0
def _fix_world(ctrl):
    if not FIX_ON:
        return False
    if ctrl.mtn.is_mountain:
        return 'mountain' in FIX_WORLDS
    if _forest_hov(ctrl):
        return True
    return _MAPSW_STATE.get("map") in FIX_WORLDS
TK_SWEEP = int(os.environ.get('SWSAR303_TKSWEEP', '1'))
TK_SWEEP_TICKS = int(os.environ.get('SWSAR303_TK_TICKS', '200'))
TK_SWEEP_TURN = int(os.environ.get('SWSAR303_TK_TURN', '75'))
TK_SWEEP_DEG = float(os.environ.get('SWSAR303_TK_DEG', '60.0'))
TK_SWEEP_STOP = os.environ.get('SWSAR303_TK_STOP', '1') == '1'
TKALIGN_ON = os.environ.get('SWSAR_TKALIGN', '1') == '1'
TKALIGN_LAST = int(os.environ.get('SWSAR_TKALIGN_LAST', '40'))
TKALIGN_LEAD = float(os.environ.get('SWSAR_TKALIGN_LEAD', '90.0'))
TKA_SIGN_ON = os.environ.get('SWSAR_TKA_SIGN', '0') == '1'
TKA_AMB = float(os.environ.get('SWSAR_TKA_AMB', '150'))
PDH_ON = os.environ.get('SWSAR_PDH', '1') == '1'
PDH_Z0 = float(os.environ.get('SWSAR_PDH_Z0', '6.0'))
PDH_DZ = float(os.environ.get('SWSAR_PDH_DZ', '3.0'))
PDH_TMAX = float(os.environ.get('SWSAR_PDH_TMAX', '20.0'))
PDH_GND = os.environ.get('SWSAR_PDH_GND', '1') == '1'
PDH_GND_TMAX = float(os.environ.get('SWSAR_PDH_GND_TMAX', '20.0'))
PDH_GND_CONF = os.environ.get('SWSAR_PDH_GND_CONF', '1') == '1'
PDH_DEG = float(os.environ.get('SWSAR_PDH_DEG', '120.0'))
PDH_REL = float(os.environ.get('SWSAR_PDH_REL', '45.0'))
PDH_VMIN = float(os.environ.get('SWSAR_PDH_VMIN', '0.3'))
PDH_RAMP = max(1, int(os.environ.get('SWSAR_PDH_RAMP', '25')))
PDH_VZ = float(os.environ.get('SWSAR_PDH_VZ', '0.5'))
PDH_MAXHOLD = int(os.environ.get('SWSAR_PDH_MAXHOLD', '250'))
PDH_BUDGET = int(os.environ.get('SWSAR_PDH_BUDGET', '400'))
PDH_LEAD = float(os.environ.get('SWSAR_PDH_LEAD', '90.0'))
PDH_AMB = float(os.environ.get('SWSAR_PDH_AMB', '120.0'))
PDH_TKC = os.environ.get('SWSAR_PDH_TKC', '0') == '1'
PDH_LEG = os.environ.get('SWSAR_PDH_LEG', '1') == '1'
PDH_DIAG = os.environ.get('SWSAR_PDH_DIAG', '0') == '1'
TKB_CLEAR_M = float(os.environ.get('SWSAR303_TKB_CLEAR', '6.0'))
TKB_ABORT_M = float(os.environ.get('SWSAR303_TKB_ABORT', '3.0'))
def _tk_sweep_world():
    return TK_SWEEP > 0 and _MAPSW_STATE.get("map") == 'forest'
FA_ON = os.environ.get('SWSAR304_FA', '1') == '1'
FA_R = float(os.environ.get('SWSAR304_FA_R', '12.0'))
FA_FLOOR = float(os.environ.get('SWSAR304_FA_FLOOR', '1.5'))
FA_EL_DEG = float(os.environ.get('SWSAR304_FA_EL', '30.0'))
HOV_MODE = int(os.environ.get('SWSAR304_HOV', '1'))
FIXF_RANGE_M = float(os.environ.get('SWSAR304_FIXF_RANGE', '12.0'))
FIXF_NEAR_M = float(os.environ.get('SWSAR304_FIXF_NEAR', '8.0'))
FIXF_WINDOW = int(os.environ.get('SWSAR304_FIXF_WINDOW', '150'))
FIXF_RETRY = int(os.environ.get('SWSAR304_FIXF_RETRY', '1'))
def _forest_here(ctrl):
    return (not ctrl.mtn.is_mountain) and (_MAPSW_STATE.get("map") == 'forest' or bool(getattr(ctrl, 'forest_low', False)))
def _forest_hov(ctrl):
    return HOV_MODE > 0 and _forest_here(ctrl)
MSP_ON = os.environ.get('SWSAR201_MSP', '1') == '1'
MSP_PERIOD = int(os.environ.get('SWSAR201_MSP_PERIOD', '400'))
MSP_T0 = int(os.environ.get('SWSAR201_MSP_T0', '150'))
MSP_RATE = float(os.environ.get('SWSAR201_MSP_RATE', '170.0'))
MSP_CLEAR_M = float(os.environ.get('SWSAR201_MSP_CLEAR', '12.0'))
MSP_AGL_MIN = float(os.environ.get('SWSAR201_MSP_AGL', '4.0'))
MSP_MIN_SPEED = 0.2
MBT_ON = os.environ.get('SWSAR_MBT', '0') == '1'
MBT_DEG = float(os.environ.get('SWSAR_MBT_DEG', '60.0'))
MBT_SPEED = float(os.environ.get('SWSAR_MBT_V', '0.5'))
MBT_WIN = int(os.environ.get('SWSAR_MBT_WIN', '100'))
MBT_JUMP_M = float(os.environ.get('SWSAR_MBT_JUMP', '3.0'))
MCV_ON = os.environ.get('SWSAR_MCV', '1') == '1'
MCV_R = float(os.environ.get('SWSAR_MCV_R', '8.0'))
OFV_ON = os.environ.get('SARZ_OFV', '0') == '1'
OFV_RNG = float(os.environ.get('SARZ_OFV_RNG', '14.0'))
OFV_WIN = int(os.environ.get('SARZ_OFV_WIN', '300'))
OFV_R = float(os.environ.get('SARZ_OFV_R', '2.0'))
OFV_AGL = float(os.environ.get('SARZ_OFV_AGL', '3.2'))
OFV_V = float(os.environ.get('SARZ_OFV_V', '0.8'))
OFV_HOLD = int(os.environ.get('SARZ_OFV_HOLD', '130'))
OFV_MAX = int(os.environ.get('SARZ_OFV_MAX', '2'))
def _ofv_end(c, st):
    if st["pt"] is not None:
        st["done"].append(np.asarray(st["pt"], dtype=np.float64).copy())
    st["n"] += 1
    st["active"] = 0
    st["pt"] = None
def _ofv_step(c, pos, target, speed):
    if not (c.mtn.is_mountain or _MAPSW_STATE.get("map") == 'mountain'):
        return target, speed
    det = c.detector
    st = getattr(c, '_ofv', None)
    if st is None:
        st = c._ofv = {"dt": -1, "pt": None, "t": -10 ** 9, "ta": 0, "active": 0, "good": 0, "done": [], "n": 0}
    if c.locked or c.frozen or c.mode != 'search':
        if st["active"]:
            _ofv_end(c, st)
        return target, speed
    if det.t != st["dt"]:
        st["dt"] = det.t
        z = getattr(det, "last_z", None)
        if z is not None and float(det.last_p_depth) >= STRONG_P and st["n"] < OFV_MAX and not st["active"]:
            z = np.asarray(z, dtype=np.float64)
            if (float(np.linalg.norm(z - pos)) <= OFV_RNG and not det.location_refuted(z)
                    and all(math.hypot(z[0] - q[0], z[1] - q[1]) > 3.0 for q in st["done"])):
                st["pt"] = z.copy()
                st["t"] = int(c.tick)
    pt = st["pt"]
    if pt is None:
        return target, speed
    if not st["active"]:
        if c.tick - st["t"] > OFV_WIN:
            st["pt"] = None
            return target, speed
        if math.hypot(pos[0] - pt[0], pos[1] - pt[1]) <= OFV_R and c._last_agl is not None:
            st["active"] = 1
            st["good"] = 0
            st["ta"] = int(c.tick)
            c.mq_stats['ofv_start'] = c.mq_stats.get('ofv_start', 0) + 1
        else:
            return target, speed
    if c._last_agl is None:
        return target, speed
    tz = float(pos[2]) - (float(c._last_agl) - OFV_AGL)
    c.mq_stats['ofv_ticks'] = c.mq_stats.get('ofv_ticks', 0) + 1
    if math.hypot(pos[0] - pt[0], pos[1] - pt[1]) <= 1.0 and abs(float(pos[2]) - tz) <= 0.4:
        st["good"] += 1
    if st["good"] >= OFV_HOLD or c.tick - st["ta"] >= 400:
        _ofv_end(c, st)
        return target, speed
    return np.array([pt[0], pt[1], tz], dtype=np.float64), min(float(speed), OFV_V / SPEED_LIMIT)
MCV_TICKS = int(os.environ.get('SWSAR_MCV_T', '60'))
MCV_SPEED = float(os.environ.get('SWSAR_MCV_V', '0.25'))
MCV_GAP = int(os.environ.get('SWSAR_MCV_GAP', '150'))
MRGB_ON = os.environ.get('SWSAR201_MRGB', '1') == '1'
MPL_ON = os.environ.get('SWSAR_MPL', '0') == '1'
MPL_N = float(os.environ.get('SWSAR_MPL_N', '6.0'))
MPL_NLOG = 4
_MPL = {"boost": 0, "swap": 0, "now": 0, "log": []}
SKL_NLOG = 6
def _veg_dbg():
    if not VEG_ON:
        return {"veg_on": 0.0}
    return {"veg_on": 1.0, "veg_arm": float(_VEG["arm"]), "veg_drop": float(_VEG["drop"]), "veg_weak": float(_VEG["weak"])}
SKG_ON = os.environ.get('SWSAR_SKG', '1') == '1'
SKG_CDP = int(os.environ.get('SWSAR_SKG_CDP', '15'))
SKG_MAPS = tuple(x for x in os.environ.get('SWSAR_SKG_MAPS', 'forest').split(',') if x)
_SKL = {"log": [], "n": 0}
SKL_FIELDS = ("t", "x", "y", "looks", "yes", "inview", "cinview", "dyes", "chits", "dp", "cdp", "pmax")
def _skl_dbg():
    out = {"skl_n": float(_SKL.get("n", 0)), "skg_spared": float(_SKL.get("skg", 0))}
    for i, e in enumerate(_SKL.get("log", [])[:SKL_NLOG]):
        for j, nm in enumerate(SKL_FIELDS):
            out["skl_%d_%s" % (i, nm)] = float(e[j])
    return out
def _mpl_dbg():
    out = {"mpl_on": 1.0 if MPL_ON else 0.0}
    if not MPL_ON:
        return out
    out["mpl_boost"] = float(_MPL.get("boost", 0))
    out["mpl_swap"] = float(_MPL.get("swap", 0))
    for i, e in enumerate(_MPL.get("log", [])[:MPL_NLOG]):
        for j, nm in enumerate(("t", "x", "y")):
            out["mpl_b%d_%s" % (i, nm)] = float(e[j])
    return out
MCS2G_ON = os.environ.get('SWSAR_MCS2_PACED_G', '0') == '1'
MCS2G_CAP = int(os.environ.get('SWSAR_MCS2_PACED_G_CAP', '28'))
MCS2G_T0 = int(os.environ.get('SWSAR_MCS2_PACED_G_T0', '150'))
MCS2G_TEND = int(os.environ.get('SWSAR_MCS2_PACED_G_TEND', '2500'))
MCS2G_GMIN = int(os.environ.get('SWSAR_MCS2_PACED_G_GMIN', '60'))
MCS2G_GFRAC = float(os.environ.get('SWSAR_MCS2_PACED_G_GFRAC', '0.20'))
MCS2G_PADR = float(os.environ.get('SWSAR_MCS2_PACED_G_PADR', '2.0'))
MCS2G_PADZ = float(os.environ.get('SWSAR_MCS2_PACED_G_PADZ', '2.5'))
MCS2G_NLOG = 8
MCS2G_YO = os.environ.get('SWSAR_MCS2_PACED_G_YO', '1') == '1'
_MCS2G = {"pads": None, "scans": 0, "skip": 0, "padveto": 0, "promo": 0, "promo_scan": 0, "log": [], "starved": 0}
def _mcs2g_useful(depth):
    try:
        d = np.asarray(depth, dtype=np.float32).reshape(DEPTH_RES, DEPTH_RES)
        m = d[DEPTH_RES // 2:, :] * (DEPTH_MAX_M - DEPTH_MIN_M) + DEPTH_MIN_M
        return float(np.mean((m >= 6.0) & (m <= MTN_RGB_RANGE_MAX_M))) >= MCS2G_GFRAC
    except Exception:
        return True
def _mcs2g_on_pad(zr):
    pads = _MCS2G.get("pads")
    if pads is None:
        return False
    try:
        dxy = np.hypot(pads[:, 0] - float(zr[0]), pads[:, 1] - float(zr[1]))
        dz = np.abs(pads[:, 2] - float(zr[2]))
        return bool(np.any((dxy <= MCS2G_PADR) & (dz <= MCS2G_PADZ)))
    except Exception:
        return False
def _mcs2g_dbg(T=None):
    out = {"mcs2g_on": 1.0 if MCS2G_ON else 0.0, "mcs2g_starved": float(_MCS2G.get("starved", 0))}
    if not MCS2G_ON:
        return out
    out["mcs2g_yo_skips"] = float(_MCS2G.get("yo_skips", 0))
    out["mcs2g_yo_quiet"] = float(_MCS2G.get("yo_quiet", 0))
    for k in ("scans", "skip", "padveto", "promo", "promo_scan"):
        out["mcs2g_" + k] = float(_MCS2G.get(k, 0))
    try:
        _u = np.asarray(T.rgb_used[:int(T.n)], dtype=np.float64)
        out["mcs2g_rgb_blocked"] = float(T.stats.get("rgb_blocked", 0))
        out["mcs2g_rgb_used_max"] = float(_u.max()) if _u.size else 0.0
        out["mcs2g_rgb_used_sum"] = float(_u.sum())
        out["mcs2g_rgb_at_cap"] = float(np.sum(_u >= RGB_CAP - T.RGB_RESERVE))
    except Exception:
        pass
    for i, e in enumerate(_MCS2G.get("log", [])[:MCS2G_NLOG]):
        for j, nm in enumerate(("t", "x", "y", "z", "r", "s")):
            out["mcs2g_p%d_%s" % (i, nm)] = float(e[j])
    return out
FSP_ON = os.environ.get('SWSAR301_FSP', '1') == '1'
FSP_CLEAR_M = float(os.environ.get('SWSAR301_FSP_CLEAR', '12.0'))
FSP_AGL_MIN = float(os.environ.get('SWSAR301_FSP_AGL', '2.5'))
FSP_ABORT_M = float(os.environ.get('SWSAR301_FSP_ABORT', '8.0'))
FSP_PERIOD = int(os.environ.get('SWSAR301_FSP_PERIOD', '400'))
VCAP_ON = os.environ.get('SWSAR301_VCAP', '0') == '1'
VFC_ON = os.environ.get('SWSAR301_VFC', '1') == '1'
VFC_DECAY = float(os.environ.get('SWSAR301_VFC_DECAY', '0.985'))
VFC_COMMIT = float(os.environ.get('SWSAR301_VFC_COMMIT', '0.35'))
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
HOVER_DESCENT_SPEED = float(os.environ.get('NEWDET_HDS', '0.6'))
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
HOVSWEEP_TRIG = int(os.environ.get('SAR191_TRIG', '75'))
MSR_ON = os.environ.get('SWSAR_MSR', '1') == '1'
MSR_HI = float(os.environ.get('SWSAR_MSR_HI', '6.0'))
HSD_ON = os.environ.get('SARZ_HSD', '1') == '1'
HSD_TOP = float(os.environ.get('SARZ_HSD_TOP', '4.4'))
HSD_LO = float(os.environ.get('SARZ_HSD_LO', '1.6'))
HSD_HOLD = float(os.environ.get('SARZ_HSD_HOLD', '2.5'))
HSD_MPS = float(os.environ.get('SARZ_HSD_MPS', '0.35'))
HSD_ORB_R = float(os.environ.get('SARZ_HSD_ORB', '1.0'))
HSD_ORB_V = float(os.environ.get('SARZ_HSD_ORBV', '0.25'))
HSD_SETTLE = os.environ.get('SARZ_HSD_SETTLE', '1') == '1'
HSD_SETTLE_VZ = float(os.environ.get('SARZ_HSD_SETTLE_VZ', '0.3'))
HSD_ONV = float(os.environ.get('SARZ_HSD_ONV', '0'))
HSD_ONV_DZ = float(os.environ.get('SARZ_HSD_ONV_DZ', '0.6'))
HOVSWEEP_GIVEUP = int(os.environ.get('SAR191_GIVEUP', '700'))
SARX_HGU_N = int(os.environ.get('SARX_HGU_N', '0'))
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
        so.add_session_config_entry('session.intra_op.allow_spinning', '0')
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
        self.close_hits = []
        self._veg_armed = False
        self.rgb_no_log = []
        self._rgbm_spots = []
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
                    self.blacklist.remove(b)
                    self.fix_lifted = getattr(self, "fix_lifted", 0) + 1
                    continue
                return True
        return False
    def note_close_hit(self, z: np.ndarray, pos: np.ndarray) -> None:
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
                p_new = min(p_new, STRONG_P - 0.05)
                self.nd_and_veto = getattr(self, 'nd_and_veto', 0) + 1
            if NEWDET_NEAR_M > 0.0 and cz_n < NEWDET_NEAR_M and p_old >= STRONG_P:
                u_n, v_n, lcz_n = u_o, v_o, lcz_o
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
        if (MCS2G_ON and MCS2G_YO and getattr(self, 'mcs2g_yes_only', False) and p_rgb is not None
                and p_depth >= RGB_TARGET_TRIGGER and p_rgb < RGB_CONFIRM_THRESH and not colour_blind):
            _MCS2G["yo_skips"] = _MCS2G.get("yo_skips", 0) + 1
            colour_blind = True
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
                        _trk = max(float(self.strong), float(self.cand_strong))
                        _pmin = RGBV2_STRONG_P
                        if RGBV2_P_VILLAGE >= 0.0 and _MAPSW_STATE.get('map') == 'village':
                            _pmin = RGBV2_P_VILLAGE
                        _mtn_here = bool(is_mountain) or _MAPSW_STATE.get('map') == 'mountain'
                        if RGBM_P >= 0.0 and _mtn_here:
                            _pmin = RGBM_P
                        _two = _n_no >= RGBV2_NO_NEEDED
                        _fast = p_depth < _pmin and _trk < RGBV3_TRACK_MIN
                        if (_two and RGBM_HOLD and RGBM_P >= 0.0 and _mtn_here and self.strong >= RGBM_HOLD_N
                                and self.t - self.last_strong_t <= 3 and float(np.linalg.norm(z_world - self.vic)) <= CONSIST_M):
                            _two = False
                            _RGBM["hold"] += 1
                        if _two or _fast:
                            self.veto_spots.append((z_world.copy(), self.t))
                            _rgb_vetoed_now = True
                            self.rgbv2_vetoes = getattr(self, "rgbv2_vetoes", 0) + 1
                            if _two and any(self.t - tk <= RGBM_WIN and float(np.linalg.norm(z_world - p)) <= CONSIST_M
                                            for p, tk in getattr(self, "_rgbm_spots", ())):
                                _RGBM["reveto"] += 1
                        else:
                            self.rgbv2_deferred = getattr(self, "rgbv2_deferred", 0) + 1
                            if RGBM_P >= 0.0 and _mtn_here and p_depth < RGBV2_STRONG_P and _trk < RGBV3_TRACK_MIN:
                                _RGBM["defer"] += 1
                                if _RGBM["t0"] < 0.0:
                                    _RGBM.update(t0=self.t / 50.0, x0=float(z_world[0]), y0=float(z_world[1]))
                                _sp = self.__dict__.setdefault("_rgbm_spots", [])
                                _sp.append((z_world.copy(), self.t))
                                del _sp[:-16]
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
                if MCS2G_ON and _zr is not None and _mcs2g_on_pad(_zr):
                    _MCS2G["padveto"] += 1
                    _zr = None
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
                self.last_strong_t = self.t
                self.rgbv3_held = getattr(self, "rgbv3_held", 0) + 1
        if _rgb_try and is_strong:
            self.rgb_promoted = True
            self.rgb_promotions += 1
        _veg = False
        if (VEG_ON and (not is_mountain) and _MAPSW_STATE.get("map") in VEG_MAPS and self.strong > 0.0
                and (self.strong >= VEG_SMIN or getattr(self, "_veg_armed", False))):
            try:
                _vf, _vu, _vr = camera_axes(rpy)
                _vcam = np.asarray(pos, dtype=np.float64) + _vf * CAMERA_OFFSET_M + _vu * CAMERA_UP_OFFSET_M
                _vd = self.vic - _vcam
                _vzc = float(_vd @ _vf)
                _vup = float(_vd @ _vu)
                if _vzc <= 0.1 or _vup < -math.tan(math.radians(VEG_EL)) * _vzc:
                    if not getattr(self, "_veg_armed", False):
                        self._veg_armed = True
                        self._veg_snap = self.vic.copy()
                        _VEG["arm"] += 1
                elif getattr(self, "_veg_armed", False) and _vup > -math.tan(math.radians(VEG_EL - 3.0)) * _vzc:
                    self._veg_armed = False
                if getattr(self, "_veg_armed", False):
                    _oh = float(np.hypot(z_world[0] - self.vic[0], z_world[1] - self.vic[1]))
                    _ov = abs(float(z_world[2] - self.vic[2]))
                    if is_strong and (_oh > VEG_J or _ov > VEG_DZ):
                        is_strong = False
                        _veg = True
                        self.last_strong_t = self.t
                        _VEG["drop"] += 1
                    elif (not is_strong) and VEG_WEAK:
                        _sn = getattr(self, "_veg_snap", self.vic)
                        if (float(np.hypot(z_world[0] - _sn[0], z_world[1] - _sn[1])) > VEG_J
                                or abs(float(z_world[2] - _sn[2])) > VEG_DZ):
                            _veg = True
                            _VEG["weak"] += 1
            except Exception:
                pass
        if self.strong <= 0.0 and getattr(self, "_veg_armed", False):
            self._veg_armed = False
        d_inc = float("inf") if _veg else float(np.linalg.norm(z_world - self.vic))
        d_cand = float(np.linalg.norm(z_world - self.cand))
        _edge = False
        if EDGE_GUARD and self.strong >= EDGE_STRONG_MIN and d_inc <= CONSIST_M:
            _hzd = float(np.hypot(z_world[0] - pos[0], z_world[1] - pos[1]))
            _el = math.degrees(math.atan2(float(z_world[2] - pos[2]), max(_hzd, 0.1)))
            _step = float(np.hypot(z_world[0] - self.vic[0], z_world[1] - self.vic[1]))
            if _el <= -EDGE_EL_DEG and (_step > EDGE_STEP_M or float(z_world[2] - self.vic[2]) < -EDGE_DROP_M):
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
        if MPL_ON and is_mountain and self.rgb_promoted and (not self.rgb_primary_near_only):
            _b = False
            if self.strong > 0.0 and float(np.linalg.norm(z_world - self.vic)) <= CONSIST_M:
                if self.strong < MPL_N:
                    self.strong = min(N_CAP, MPL_N)
                    self.last_strong_t = self.t
                    _b = True
            elif self.cand_strong > 0.0 and float(np.linalg.norm(z_world - self.cand)) <= CONSIST_M:
                self.cand_strong = min(N_CAP, max(self.cand_strong, MPL_N))
                if self.cand_strong > self.strong:
                    self.vic, self.strong = (self.cand.copy(), self.cand_strong)
                    self.height = h_meas
                    self.last_strong_t = self.t
                    self.cand_strong = 0.0
                    _MPL["swap"] += 1
                _b = True
            if _b:
                _MPL["boost"] += 1
                if len(_MPL["log"]) < MPL_NLOG:
                    _MPL["log"].append((_MPL["now"] * SIM_DT, float(z_world[0]), float(z_world[1])))
        self.conf = float(np.clip(self.strong / (NEWDET_N_LATCH if (NEWDET_N_LATCH > 0.0 and self.nd_sess) else N_LATCH), 0.0, 1.0))
        if getattr(self, "_rgbm_spots", None) and self.conf >= LATCH_CONF:
            _keep = []
            for _p, _tk in self._rgbm_spots:
                if self.t - _tk > RGBM_WIN:
                    continue
                if float(np.linalg.norm(self.vic - _p)) <= CONSIST_M:
                    _RGBM["latch"] += 1
                    if _RGBM["lt"] < 0.0:
                        _RGBM.update(lt=self.t / 50.0, lx=float(self.vic[0]), ly=float(self.vic[1]))
                else:
                    _keep.append((_p, _tk))
            self._rgbm_spots = _keep
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
        self._hsd_settle = 0
        self.hover_total_ticks = 0
        self.forest_low = False
        self.forest_latch_tick = 0
        self._rgbf_hits = 0
        self._rgbf_evals = 0
        self.ground_min = None
        self._last_agl = None
        self._ofv = None
        self.mq_view = 0
        self.mq_hits = 0
        self.mq_hold_left = 0
        self.mq_refuted = []
        self.mq_stats = {"mq_refute": 0, "mq_hold": 0, "mq_protect": 0, "msw_giveup": 0, "mq_approach": 0}
        self._msp_turned = None
        self._fix_hits = []
        self._fix_pt = None
        self._fix_evidence = 0
        self._msp_yaw = 0.0
        self._msp_last_end = -10 ** 9
        self._mq_spot = None
        self._tks_pnear = None
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
        self._tk_yaw0 = None
        self._tka_sign = None
        self._tk_t0 = None
        self._tk_done = False
        self._tkb_forced = False
        self._pdh_z0 = None
        self._pdh_tk_sign = 0.0
        self._pdh_dbg = None
        self._pdh_leg = None
    def _mq_step(self, pos, rpy, vic, conf):
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
    def _terrain_ahead(self, depth, pos, rpy):
        try:
            _dm = depth if getattr(depth, 'ndim', 2) == 2 else depth[0]
            rs = np.arange(DEPTH_RES // 2, DEPTH_RES, LTD_STRIDE)
            cs = np.arange(DEPTH_RES // 3, 2 * DEPTH_RES // 3, LTD_STRIDE)
            d = _dm[rs][:, cs].astype(np.float64) * (DEPTH_MAX_M - DEPTH_MIN_M) + DEPTH_MIN_M
            v = 1.0 - 2.0 * (rs + 0.5) / DEPTH_RES
            u = 2.0 * (cs + 0.5) / DEPTH_RES - 1.0
            U, V = np.meshgrid(u, v)
            fwd, up, right = camera_axes(np.asarray(rpy, dtype=np.float64))
            nrm = np.sqrt(1.0 + (U * HALF_TAN) ** 2 + (V * HALF_TAN) ** 2)
            zdir = (fwd[2] + (U * HALF_TAN) * right[2] + (V * HALF_TAN) * up[2]) / nrm
            zw = float(pos[2]) + d * zdir
            m = (d >= LTD_DMIN) & (d <= LTD_DMAX)
            if int(m.sum()) < 8:
                return None
            return float(np.percentile(zw[m], LTD_PCT))
        except Exception:
            return None
    def _meg_terrain_near(self, depth, pos, rpy, max_h):
        try:
            _dm = depth if getattr(depth, 'ndim', 2) == 2 else depth[0]
            rs = np.arange(DEPTH_RES // 2, DEPTH_RES, LTD_STRIDE)
            cs = np.arange(DEPTH_RES // 3, 2 * DEPTH_RES // 3, LTD_STRIDE)
            d = _dm[rs][:, cs].astype(np.float64) * (DEPTH_MAX_M - DEPTH_MIN_M) + DEPTH_MIN_M
            v = 1.0 - 2.0 * (rs + 0.5) / DEPTH_RES
            u = 2.0 * (cs + 0.5) / DEPTH_RES - 1.0
            U, V = np.meshgrid(u, v)
            fwd, up, right = camera_axes(np.asarray(rpy, dtype=np.float64))
            nrm = np.sqrt(1.0 + (U * HALF_TAN) ** 2 + (V * HALF_TAN) ** 2)
            xd = (fwd[0] + (U * HALF_TAN) * right[0] + (V * HALF_TAN) * up[0]) / nrm
            yd = (fwd[1] + (U * HALF_TAN) * right[1] + (V * HALF_TAN) * up[1]) / nrm
            zd = (fwd[2] + (U * HALF_TAN) * right[2] + (V * HALF_TAN) * up[2]) / nrm
            zw = float(pos[2]) + d * zd
            hr = d * np.sqrt(xd * xd + yd * yd)
            m = (d >= LTD_DMIN) & (d <= LTD_DMAX) & (hr < float(max_h))
            if int(m.sum()) < 8:
                return None
            return float(np.percentile(zw[m], LTD_PCT))
        except Exception:
            return None
    def _target_z(self, ground_z: float) -> float:
        if self.mtn.is_mountain:
            _deck = ground_z + (MTK_SEARCH_H if (MTK_ON and MTK_SEARCH_H > 0.0) else MOUNTAIN_SEARCH_ALT_M)
            _ah = getattr(self, '_ltd_h', None)
            if LTD_ON and _ah is not None and np.isfinite(_ah):
                _want = min(_ah + LTD_MARGIN, ground_z + LTD_CAP)
                if _want > _deck:
                    _deck = _want
                    self.mq_stats["ltd_lift"] = self.mq_stats.get("ltd_lift", 0) + 1
            return self.alt_td(_deck)
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
            _z0 = VCAP_ABS_M
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
                target[2] = max(_head + HOVER_ABOVE_TOP_M + HOVER_SETTLE_MARGIN_M + (_FHZ_DZ if self.locked else 0.0), ground_z + FA_FLOOR)
                self.mq_stats["fa_low"] = self.mq_stats.get("fa_low", 0) + 1
                _el = math.degrees(math.atan2(_head - pos[2], max(_horiz, 0.1)))
                if _el < -FA_EL_DEG and pos[2] > target[2] + 0.3:
                    target[0] = pos[0] + 0.1 * (vic[0] - pos[0])
                    target[1] = pos[1] + 0.1 * (vic[1] - pos[1])
                    self.mq_stats["fa_hold"] = self.mq_stats.get("fa_hold", 0) + 1
        if MA_ON and self.mtn.is_mountain and self.locked and (SARX_MAF or not self.frozen):
            horiz = float(np.hypot(vic[0] - pos[0], vic[1] - pos[1]))
            if horiz <= MA_R:
                hover_z = float(self.locked_head_z) + HOVER_ABOVE_TOP_M + HOVER_SETTLE_MARGIN_M
                if SARX_MAG_DEG > 0.0:
                    _g = hover_z + math.tan(math.radians(SARX_MAG_DEG)) * max(horiz - R_HOVER_M, 0.0)
                    target[2] = max(ground_z + 2.5, min(_g, max(float(target[2]), hover_z)))
                else:
                    target[2] = max(hover_z, ground_z + 2.5)
                self.mq_stats["ma_low"] = self.mq_stats.get("ma_low", 0) + 1
        if MEG_ON and MA_ON and self.mtn.is_mountain and self.locked:
            _mh = float(np.hypot(vic[0] - pos[0], vic[1] - pos[1]))
            _mhz = float(self.locked_head_z) + HOVER_ABOVE_TOP_M + HOVER_SETTLE_MARGIN_M
            _zab = float(pos[2]) - _mhz
            if self.frozen and MEG_FRZ and (getattr(self, '_meg_fz', False) or _zab >= MEG_FRZ_ZMIN):
                self._meg_fz = True
                target[2] = max(_mhz, ground_z + 2.5)
                self.mq_stats["meg_frz"] = self.mq_stats.get("meg_frz", 0) + 1
            elif ((not self.frozen) and MEG_B and MA_R < _mh <= MEG_R
                  and (getattr(self, '_meg_gl', False) or _zab >= MEG_ZMIN)):
                self._meg_gl = True
                _fl = ground_z + 2.5
                _ob = getattr(self, '_meg_obs', None)
                if LTD_ON and _ob is not None:
                    _tn = self._meg_terrain_near(_ob[0], pos, _ob[1], _mh - MEG_GAP)
                    if _tn is not None and np.isfinite(_tn):
                        _tw = min(_tn + LTD_MARGIN, ground_z + LTD_CAP)
                        if _tw > max(_fl, _mhz):
                            self.mq_stats["meg_floor"] = self.mq_stats.get("meg_floor", 0) + 1
                        _fl = max(_fl, _tw)
                _nz = max(_mhz, _fl)
                if _nz < float(target[2]):
                    target[2] = _nz
                    self.mq_stats["meg_glide"] = self.mq_stats.get("meg_glide", 0) + 1
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
                _base, _hi = float(pos[2]) - float(self._last_agl), MSR_HI
                self.mq_stats['msr_ticks'] = self.mq_stats.get('msr_ticks', 0) + 1
            if HSD_ON and self.mtn.is_mountain and MSR_ON and self._last_agl is not None and np.isfinite(self._last_agl):
                ts = (self.tick - self._sweep_t0) / CTRL_HZ
                span = HSD_TOP - HSD_LO
                leg = span / max(HSD_MPS, 1e-3)
                if ts <= HSD_HOLD:
                    dz = HSD_TOP
                else:
                    u = (ts - HSD_HOLD) % (2.0 * leg)
                    dz = HSD_TOP - HSD_MPS * u if u <= leg else HSD_LO + HSD_MPS * (u - leg)
                if HSD_ONV > 0.0 and dz > HSD_ONV and float(_base) >= float(vic[2]) - HSD_ONV_DZ:
                    dz = HSD_ONV
                    self.mq_stats['hsd_onv'] = self.mq_stats.get('hsd_onv', 0) + 1
                self.mq_stats['hsd_ticks'] = self.mq_stats.get('hsd_ticks', 0) + 1
                ox = oy = 0.0
                if HSD_ORB_R > 0.0 and ts > HSD_HOLD:
                    ang = (ts - HSD_HOLD) * HSD_ORB_V / HSD_ORB_R
                    ox, oy = HSD_ORB_R * math.cos(ang), HSD_ORB_R * math.sin(ang)
                target = np.array([vic[0] + ox, vic[1] + oy, _base + dz])
                self._dbg_hover_tz = float(target[2])
                return (target, max(HSD_MPS, 0.2) / SPEED_LIMIT)
            if ph <= _d0:
                dz = 2.5 - ph
            elif ph <= _d0 + (_hi - HOVSWEEP_LO):
                dz = HOVSWEEP_LO + (ph - _d0)
            else:
                dz = _hi
            target = np.array([vic[0], vic[1], _base + dz])
            self._dbg_hover_tz = float(target[2])
            return (target, HOVSWEEP_MPS / SPEED_LIMIT)
        boost = float(os.environ.get('HYB_MTN_HOVZ', '0.0')) if self.mtn.is_mountain else 0.0
        target = np.array([vic[0], vic[1], head_z + HOVER_ABOVE_TOP_M + HOVER_SETTLE_MARGIN_M + boost])
        if _FHZ_DZ != 0.0 and _forest_here(self):
            target[2] += _FHZ_DZ
        agl_sp = float(os.environ.get('HYB_MTN_AGLHOV', '3.2'))
        if self.mtn.is_mountain and agl_sp > 0.0 and self._last_agl is not None:
            _ray_z = pos[2] - (self._last_agl - agl_sp)
            _bidir_on = os.environ.get('SAR150_BIDIR', '1') == '1'
            if MH_ON and not MH_BIDIR:
                _bidir_on = False
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
        if (FHS_ON and (not self.mtn.is_mountain) and _MAPSW_STATE.get("map") in FHS_MAPS and self.frozen
                and self.hover_stable_ticks >= FHS_T0):
            _ph = (self.hover_stable_ticks - FHS_T0) // max(1, FHS_STEP)
            _fd = min(FHS_DZ, 0.3) if _FHZ_DZ > 0.0 else FHS_DZ
            _off = _fd if _ph == 0 else (-_fd if _ph == 1 else 0.0)
            if _off != 0.0:
                target[2] = float(target[2]) + _off
                self.mq_stats["fhs_ticks"] = self.mq_stats.get("fhs_ticks", 0) + 1
        self._dbg_hover_tz = float(target[2])
        if MEG_ON and self.mtn.is_mountain:
            if MEG_C:
                if self._hov_z_lock is None or not self.frozen:
                    self._meg_rc_hist = []
                    self._meg_rc_done = False
                elif not getattr(self, '_meg_rc_done', False):
                    _rh = float(np.hypot(pos[0] - target[0], pos[1] - target[1]))
                    _ag = self._last_agl
                    if _rh < MEG_RC_H and _ag is not None and np.isfinite(_ag):
                        _hist = getattr(self, '_meg_rc_hist', None) or []
                        _hist.append(float(pos[2]) - float(_ag))
                        _hist = _hist[-MEG_RC_N:]
                        self._meg_rc_hist = _hist
                        _rz = _hist[-1]
                        if (len(_hist) >= MEG_RC_N and max(_hist) - min(_hist) <= 2.0 * MEG_RC_TOL
                                and abs(_rz - float(head_z)) < MEG_RC_GATE):
                            _tza = float(self._hov_z_lock) - _rz
                            _new = None
                            if _tza < MEG_RC_LO and _rz - float(head_z) < MEG_RC_UPG:
                                _new = min(_rz + MEG_RC_AT, float(head_z) + MEG_RC_UPMAX)
                            elif MEG_RC_DN and _tza > MEG_RC_HI and _rh < MEG_RC_HL and abs(_rz - float(head_z)) < MEG_RC_DNG:
                                _new = max(_rz + MEG_RC_AT, float(head_z) + MEG_RC_HMIN)
                            if _new is not None and abs(_new - float(self._hov_z_lock)) > 0.05:
                                _key = "meg_rc_up" if _new > float(self._hov_z_lock) else "meg_rc_dn"
                                self.mq_stats[_key] = self.mq_stats.get(_key, 0) + 1
                                self.mq_stats["meg_rc"] = self.mq_stats.get("meg_rc", 0) + 1
                                self._hov_z_lock = float(_new)
                                target[2] = self._hov_z_lock
                                self._dbg_hover_tz = float(target[2])
                                self._meg_rc_done = True
                    else:
                        self._meg_rc_hist = []
            if MEG_A and pos[2] - target[2] > HOVER_DESCENT_M:
                self.mq_stats["meg_fast"] = self.mq_stats.get("meg_fast", 0) + 1
                _v = MEG_HDS
                if MEG_BRK > 0.0:
                    _room = float(pos[2]) - (float(head_z) + _CSC_DZ_HI) - MEG_BRK_M
                    _v = min(_v, math.sqrt(_CSC_V ** 2 + 2.0 * MEG_BRK * max(_room, 0.0)) / SPEED_LIMIT)
                return (target, _v)
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
        if self.tick <= 1:
            self._pdh_z0 = float(pos[2])
        self._update_clutter(depth)
        if LTD_ON and self.mtn.is_mountain and self.mode in ('search', 'navigation'):
            self._ltd_h = self._terrain_ahead(depth, pos, rpy)
        else:
            self._ltd_h = None
        if MEG_ON:
            self._meg_obs = (depth, rpy)
            if not self.locked:
                self._meg_gl = False
                self._meg_fz = False
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
        if _PADV_ON:
            self.detector.padv_live = bool(self.mtn.is_mountain or _MAPSW_STATE.get("map") == 'mountain')
            self.detector.padv_cam = np.asarray(pos, dtype=np.float64)
        _tks_det = (TKS_ON and (not self._tks_done) and (not self.mtn.is_mountain)
                    and _MAPSW_STATE.get("map") in TKS_MAPS and self.tick >= TKS_T0)
        if self.mode == 'initial' and not _tks_det:
            conf, vic, head_z = (0.0, pos.copy(), ground_z)
        elif self.frozen:
            conf, vic, head_z = (1.0, self.locked_vic, self.locked_head_z)
        else:
            self.detector.rgb_primary_ok = self.mode in ('search', 'navigation')
            self.detector.mcs2g_yes_only = bool(MCS2G_ON and MCS2G_YO
                                                and self.tick - 1 == getattr(self, '_mcs2g_pure_t', -99))
            self.detector.rgb_primary_near_only = bool(self.locked)
            _strong_before = float(self.detector.strong)
            conf, vic, head_z = self.detector.update(depth, pos, rpy, rgb=rgb, is_mountain=self.mtn.is_mountain)
            if self.detector.rgb_fresh and self.detector.rgb_prob >= RGB_CONFIRM_THRESH:
                self.rgb_confirm_tick = self.tick
            if self.detector.rgb_promoted:
                self.rgb_hit_tick = self.tick
                if MCS2G_ON and self.mtn.is_mountain:
                    _sc = 1 if self.tick - getattr(self, '_mcs2g_scan_t', -99) <= 3 else 0
                    _MCS2G["promo"] += 1
                    _MCS2G["promo_scan"] += _sc
                    try:
                        if len(_MCS2G["log"]) < MCS2G_NLOG:
                            _lz = np.asarray(self.detector.last_z, dtype=np.float64).reshape(-1)
                            _MCS2G["log"].append((self.tick * SIM_DT, _lz[0], _lz[1], _lz[2],
                                                  float(np.linalg.norm(_lz[0:3] - np.asarray(pos[0:3], dtype=np.float64))), _sc))
                    except Exception:
                        pass
            if _mq_on() and self.mtn.is_mountain:
                if self.detector.seen and self._mq_spot_refuted(vic):
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
        self._nav_head_z = float(head_z) if (self.mode != 'initial' and np.isfinite(head_z)) else None
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
                _seen = False
                self.mq_stats["tel_lockguard"] = self.mq_stats.get("tel_lockguard", 0) + 1
            if _seen:
                if _fixw and float(np.linalg.norm(self.locked_vic[0:2] - pos[0:2])) <= _fix_near_m:
                    _fp = self.detector.fix_near(self.locked_vic[0:2], window=(FIXF_WINDOW if (_fh and HOV_MODE == 2) else FIX_WINDOW))
                    if _fp is not None:
                        self._fix_pt = _fp
                        self.mq_stats["fix_set"] = self.mq_stats.get("fix_set", 0) + 1
                if self._fix_pt is not None:
                    self.locked_vic = np.array([self._fix_pt[0], self._fix_pt[1], self._fix_pt[2]], dtype=np.float64)
                    self.locked_head_z = float(self._fix_pt[2]) + 0.5 * float(np.clip(self.detector.height, MH_HMIN, MH_HMAX))
                else:
                    self.locked_vic = vic.copy()
                    self.locked_head_z = head_z
                self.lost_ticks = 0
                if (_fixw and self._fix_pt is not None and float(np.linalg.norm(self.locked_vic[0:2] - pos[0:2])) < R_HOVER_M
                        and not (_fh and HOV_MODE == 2)):
                    self.frozen = True
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
                if HSD_SETTLE and abs(float(state[IDX_VEL][2])) < HSD_SETTLE_VZ:
                    self._hsd_settle = getattr(self, '_hsd_settle', 0) + 1
        else:
            self.hover_stable_ticks = 0
            self._hsd_settle = 0
            self.hover_total_ticks = 0
        _giveup_n = HOVER_GIVEUP_TICKS
        if HOVSWEEP and self.mtn.is_mountain and self.frozen:
            if (getattr(self, '_sweep_t0', None) is None and self.hover_stable_ticks >= HOVSWEEP_TRIG
                    and (not HSD_SETTLE or getattr(self, '_hsd_settle', 0) >= HOVSWEEP_TRIG)):
                self._sweep_t0 = self.tick
            if getattr(self, '_sweep_t0', None) is not None:
                _giveup_n = HOVSWEEP_GIVEUP
                if SARX_HGU_N > 0 and self.hover_stable_ticks >= SARX_HGU_N:
                    _lt = getattr(self, '_sarx_lock_t', None)
                    _dt = int(self.detector.t)
                    _w = (_dt - int(_lt) + 1) if (_lt is not None and int(_lt) <= _dt) else 500
                    if self.detector.evidence_at(self.locked_vic[0:2], window=max(_w, 1)) < FIX_EVIDENCE:
                        _giveup_n = SARX_HGU_N
                        self.mq_stats["sarx_hgu"] = self.mq_stats.get("sarx_hgu", 0) + 1
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
                    self._fixf_retries = getattr(self, '_fixf_retries', 0) + 1
                    self.locked_vic = np.array([_cp[0], _cp[1], _cp[2]], dtype=np.float64)
                    self.locked_head_z = float(_cp[2]) + 0.5 * float(np.clip(self.detector.height, MH_HMIN, MH_HMAX))
                    self._fix_pt = _cp
                    self.frozen = True
                    self.hover_stable_ticks = 0
                    self._hsd_settle = 0
                    self.hover_total_ticks = 0
                    self.mq_stats["fix_rehover"] = self.mq_stats.get("fix_rehover", 0) + 1
                    _rehover = True
        if _rehover:
            pass
        elif self.hover_stable_ticks >= _giveup_n or self.hover_total_ticks >= HOVER_GIVEUP_FALLBACK_TICKS or _msw_done:
            if _fix_world(self) and self.detector.evidence_at(self.locked_vic[0:2]) >= FIX_EVIDENCE:
                self.mq_stats["fix_noblack"] = self.mq_stats.get("fix_noblack", 0) + 1
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
            self._hsd_settle = 0
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
        if OFV_ON:
            try:
                target, speed = _ofv_step(self, pos, target, speed)
            except Exception as exc:
                _log_once("sarz_ofv", "overflight verify failed", exc)
        if (_tks_det and self.mode in ('initial', 'search') and (not self.locked) and (not self.frozen)
                and ((self._tks_pnear if self._tks_pnear is not None else 0.0) >= TKS_P_HOLD
                     or getattr(self, '_tks_phase', 'sweep') == 'verify')):
            _pz = self._tks_pad_z if self._tks_pad_z is not None else float(pos[2])
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
        if (SARX_MTKY and self.mode == 'initial'
                and (self.mtn.is_mountain or _MAPSW_STATE.get("map") == 'mountain')):
            _c = np.asarray(clue, dtype=np.float64)
            if float(np.hypot(_c[0], _c[1])) > 2.0:
                yaw = float(np.arctan2(_c[1], _c[0]) / np.pi)
                self.mq_stats["sarx_mtky"] = self.mq_stats.get("sarx_mtky", 0) + 1
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
                        self._msp_turned = 0.0
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
                _b = math.atan2(-pos[1], -pos[0])
                _rel = (_b - self._tks_yaw0 + np.pi) % (2.0 * np.pi) - np.pi
                self._tks_dir = 1.0 if _rel >= 0.0 else -1.0
                self.mq_stats["tks_start"] = self.mq_stats.get("tks_start", 0) + 1
            _pn = self._tks_pnear if self._tks_pnear is not None else 0.0
            _hold = _pn >= TKS_P_HOLD
            _total = TKS_DEG_HOLD if _hold else ((P1_TKS_OUT if (P1_TKS_OUT > 0.0 and _pn < 0.02 and _MAPSW_STATE.get("map") in P1_TKS_OUT_MAPS) else TKS_DEG_FREE))
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
                    self._tks_done = True
                    self.mq_stats["tks_stall"] = self.mq_stats.get("tks_stall", 0) + 1
                else:
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
                if TKALIGN_ON and self.mode != 'initial':
                    _hd = float(math.hypot(float(direction[0]), float(direction[1])))
                    if _hd > 1e-3:
                        _ctr = float(math.atan2(float(direction[1]), float(direction[0])))
                        if self.tick >= TK_SWEEP_TICKS - TKALIGN_LAST:
                            _off = 0.0
                        _err = float(((_ctr + _off - float(rpy[2])) + np.pi) % (2.0 * np.pi) - np.pi)
                        if TKA_SIGN_ON and getattr(self, '_tk_yaw0', None) is not None:
                            if getattr(self, '_tka_sign', None) is None:
                                _e0 = float(((_ctr - self._tk_yaw0) + np.pi) % (2.0 * np.pi) - np.pi)
                                self._tka_sign = ((1.0 if _e0 >= 0.0 else -1.0) if abs(_e0) >= np.radians(TKA_AMB) else 0.0)
                            if self._tka_sign and _err * self._tka_sign < 0.0 and abs(_err) > np.radians(90.0):
                                _err += 2.0 * np.pi * self._tka_sign
                                self.mq_stats["tka_sign"] = self.mq_stats.get("tka_sign", 0) + 1
                        _lead = np.radians(TKALIGN_LEAD)
                        _set = float(rpy[2]) + max(-_lead, min(_lead, _err))
                        yaw = float(((_set + np.pi) % (2.0 * np.pi) - np.pi) / np.pi)
                        self.mq_stats["tkalign"] = self.mq_stats.get("tkalign", 0) + 1
                if (PDH_ON and PDH_TKC and getattr(self, '_pdh_z0', None) is not None
                        and self._pdh_z0 >= PDH_Z0):
                    _lx = float(target[0]) - float(pos[0])
                    _ly = float(target[1]) - float(pos[1])
                    if math.hypot(_lx, _ly) <= 1.0:
                        _lx, _ly = float(clue[0]), float(clue[1])
                    if math.hypot(_lx, _ly) > 1.0:
                        _err = float(((math.atan2(_ly, _lx) - float(rpy[2])) + np.pi) % (2.0 * np.pi) - np.pi)
                        _sg = float(getattr(self, '_pdh_tk_sign', 0.0))
                        if abs(_err) < math.radians(PDH_AMB) or _sg == 0.0:
                            _sg = 1.0 if _err >= 0.0 else -1.0
                        elif _err * _sg < 0.0:
                            _err += 2.0 * np.pi * _sg
                        self._pdh_tk_sign = _sg
                        _lead = math.radians(PDH_LEAD)
                        _set = float(rpy[2]) + max(-_lead, min(_lead, _err))
                        yaw = float(((_set + np.pi) % (2.0 * np.pi) - np.pi) / np.pi)
                        self.mq_stats["pdh_tkc"] = self.mq_stats.get("pdh_tkc", 0) + 1
                if TK_SWEEP == 1 and self.mode != 'initial':
                    direction = np.zeros(3, dtype=np.float32)
                    speed = 0.0
                    self.mq_stats["tk_hold"] = self.mq_stats.get("tk_hold", 0) + 1
        if PDH_ON:
            try:
                _tx, _ty = float(target[0]) - float(pos[0]), float(target[1]) - float(pos[1])
                self._pdh_leg = (math.atan2(_ty, _tx), math.hypot(_tx, _ty))
            except Exception:
                self._pdh_leg = None
        if PDH_DIAG:
            try:
                _tx, _ty = float(target[0]) - float(pos[0]), float(target[1]) - float(pos[1])
                _mb = self.avoider.mem_bias
                self._pdh_dbg = [math.degrees(math.atan2(_ty, _tx)), math.hypot(_tx, _ty), float(_mb[0]), float(_mb[1]),
                                 math.degrees(math.atan2(float(direction[1]), float(direction[0])))
                                 if math.hypot(float(direction[0]), float(direction[1])) > 1e-6 else float('nan'),
                                 math.degrees(math.atan2(float(clue[1]), float(clue[0])))]
            except Exception:
                self._pdh_dbg = None
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
                _periodic = False
                if (MCS2G_ON and self.mode == 'search' and (not self.locked) and (not self.frozen)
                        and MCS2G_T0 <= self.tick < MCS2G_TEND and self.rgb_requests < MCS2G_CAP):
                    _left = max(1, MCS2G_CAP - self.rgb_requests)
                    _gap = max(MCS2G_GMIN, (MCS2G_TEND - self.tick) // _left)
                    if self.tick - self.last_rgb_req >= _gap:
                        if _mcs2g_useful(depth):
                            _periodic = True
                            self._mcs2g_scan_t = self.tick
                            _MCS2G["scans"] += 1
                        else:
                            _MCS2G["skip"] += 1
                if (self.locked and (not self.frozen) and self.mode == 'navigation'
                        and float(np.hypot(self.locked_vic[0] - pos[0], self.locked_vic[1] - pos[1])) <= MRGB_CHECK_R
                        and self.tick - self.last_rgb_req >= MRGB_CHECK_GAP):
                    ondemand = True
                    self.mq_stats["mrgb_check"] = self.mq_stats.get("mrgb_check", 0) + 1
        if (self.mtn.is_mountain and ondemand and self.rgb_requests >= RGB_CAP and (not self.frozen)
                and self.mode in ('search', 'navigation') and self.tick - self.last_rgb_req >= RGB_MIN_INTERVAL
                and (self.tick - self.last_rgb_req) % RGB_MIN_INTERVAL == 0):
            _MCS2G["starved"] += 1
        if self.mode in ('search', 'navigation') and (not self.frozen) and (self.rgb_requests < RGB_CAP) and (self.tick - self.last_rgb_req >= RGB_MIN_INTERVAL) and (_periodic or ondemand):
            rgb_req = 1.0
            self.rgb_requests += 1
            self.last_rgb_req = self.tick
            if MCS2G_ON and _periodic and (not ondemand) and self.mtn.is_mountain:
                self._mcs2g_pure_t = self.tick
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
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
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
FLTE_ON = os.environ.get('SWSAR_FROZEN_LOCK_', '0') == '1'
FLTE_U66 = FLTE_ON and os.environ.get('SWSAR_FROZEN_LOCK_U66', '1') == '1'
FLTE_U66_MIN = float(os.environ.get('SWSAR_FROZEN_LOCK_MIN', '1.0'))
FLTE_ZRESET = os.environ.get('SWSAR_FROZEN_LOCK_ZRESET', '1') == '1'
FLTE_DWELL = int(os.environ.get('SWSAR_FROZEN_LOCK_DWELL', '0'))
FLTE_Z0 = float(os.environ.get('SWSAR_FROZEN_LOCK_Z0', '12.0'))
FLTE_PDN = FLTE_ON and os.environ.get('SWSAR_FROZEN_LOCK_PDN', '1') == '1'
FLTE_PDN_H = float(os.environ.get('SWSAR_FROZEN_LOCK_PDN_H', '12.0'))
FLTE_PDN_TICKS = int(os.environ.get('SWSAR_FROZEN_LOCK_PDN_TICKS', '0'))
_FLTE_STATE = {}
def _flte_reset():
    _FLTE_STATE.clear()
    _FLTE_STATE.update({"u66_ticks": 0.0, "u66_would": 0.0, "u66_latch": 0.0, "u66_shift": 0.0,
                        "u66_dwell_skip": 0.0, "band": {},
                        "pdn_would": 0.0, "pdn_skip": 0.0, "pdn_short": 0.0, "z0_block": 0.0})
def _flte_z0_ok():
    return float(_FLTE_STATE.get("z0", 99.0)) >= FLTE_Z0
_flte_reset()
TEAM_COOLDOWN_EXEMPT = True
TEAM_SUPERSEDE = True
TEAM_SUPERSEDE_LOCK = 0.9499
TEAM_SUPERSEDE_FROZEN_TICKS = 300
TEAM_SUPERSEDE_STALL_TICKS = 150
TEAM_SUPERSEDE_MAX = 2
DC_ON = os.environ.get('SWSAR_DC', '1') == '1'
DC_MIN_R = float(os.environ.get('SWSAR_DC_MINR', '8.0'))
DC_BAND = os.environ.get('SWSAR_DC_BAND', '1') == '1'
P1_TKS = os.environ.get('SWSAR_P1_TKS', '1') == '1'
P1_RAMPK = float(os.environ.get('SWSAR_P1_RAMPK', '0.25'))
P1_ADV = os.environ.get('SWSAR_P1_ADV', '1') == '1'
P1_ADV_R = float(os.environ.get('SWSAR_P1_ADVR', '4.0'))
P1_ADV_HOLD = float(os.environ.get('SWSAR_P1_ADVHOLD', '0.0'))
P1_MSW = os.environ.get('SWSAR_P1_MSW', '1') == '1'
P1_U66 = os.environ.get('SWSAR_P1_U66', '1') == '1'
P1_TKS_OUT = float(os.environ.get('SWSAR_P1_TKSOUT', '300.0'))
P1_TKS_OUT_MAPS = tuple(x for x in os.environ.get('SWSAR_P1_TKSOUT_MAPS', 'city').split(',') if x)
P1_PAD_TRIG = float(os.environ.get('SWSAR_P1_PADTRIG', '0.15'))
TEL_SUPERSEDE_ALL = os.environ.get('SWSAR_TEL_SUP', '1') == '1'
TEL_HANDOVER_TICKS = int(os.environ.get('SWSAR_TEL_HO', '150'))
TEL_HANDOVER_GAIN = float(os.environ.get('SWSAR_TEL_GAIN', '1.5'))
TEL_HANDOVER_R = float(os.environ.get('SWSAR_TEL_HOR', '12.0'))
TEL_HANDOVER_MAX = int(os.environ.get('SWSAR_TEL_HOMAX', '1'))
_DHO_ON = os.environ.get('SARZ_DHO', '1') == '1'
_DHO_GAIN = float(os.environ.get('SARZ_DHO_GAIN', '9.0'))
_DHO_R = float(os.environ.get('SARZ_DHO_R', '20.0'))
HOC_ON = os.environ.get('SWSAR_HOC', '1') == '1'
HOC_GAIN = float(os.environ.get('SWSAR_HOC_GAIN', '6.0'))
HOC_MAPS = tuple(x for x in os.environ.get('SWSAR_HOC_MAPS', 'forest').split(',') if x)
_SUPK_ON = os.environ.get('SARZ_SUPK', '1') == '1'
_SUPK_R = float(os.environ.get('SARZ_SUPK_R', '3.0'))
_SUPK_MAPS = tuple(x for x in os.environ.get('SARZ_SUPK_MAPS', 'mountain').split(',') if x)
_PADV_ON = os.environ.get('SARZ_PADV', '1') == '1'
_PADV_R = float(os.environ.get('SARZ_PADV_R', '1.5'))
_PADV_DZLO = float(os.environ.get('SARZ_PADV_DZLO', '2.5'))
_PADV_DZHI = float(os.environ.get('SARZ_PADV_DZHI', '1.5'))
_PADV_RAY = os.environ.get('SARZ_PADV_RAY', '0') == '1'
TEL_LOCKGUARD = os.environ.get('SWSAR_TEL_LG', '1') == '1'
TEL_LG_N = float(os.environ.get('SWSAR_TEL_LGN', '6.0'))
TEL2_E1 = os.environ.get('SWSAR_TEL2_E1', '1') == '1'
TEL2_E2 = os.environ.get('SWSAR_TEL2_E2', '1') == '1'
TEL2_E3 = os.environ.get('SWSAR_TEL2_E3', '1') == '1'
TEL2_E4 = os.environ.get('SWSAR_TEL2_E4', '1') == '1'
TEL2_E7 = os.environ.get('SWSAR_TEL2_E7', '1') == '1'
TEL2_E8 = os.environ.get('SWSAR_TEL2_E8', '1') == '1'
TEL2_E5 = os.environ.get('SWSAR_TEL2_E5', '1') == '1'
TEL2_E6 = os.environ.get('SWSAR_TEL2_E6', '1') == '1'
TEL2_SELF_M = float(os.environ.get('SWSAR_TEL2_SELFM', '6.0'))
TEL_SUP_GUARD = os.environ.get('SWSAR_TEL_SUPG', '1') == '1'
TEL_SUP_QUAL = os.environ.get('SWSAR_TEL_SUPQ', '1') == '1'
TEAM_BLACKLIST_DENY_TICKS = 3000
XGB_OWNER_STALE = 25
OUTER_ODDS = 9.17
OUTER_STREAK = int(os.environ.get('SAR_OUTER_STREAK', '12'))
OUTER_STREAK_MAPS = tuple(os.environ.get('SAR_OUTER_STREAK_MAPS', 'mountain,city,forest,village,open').split(','))
def _outer_bar(strong_p, odds):
    o = odds * (strong_p / max(1e-9, 1.0 - strong_p))
    return float(o / (1.0 + o))
DT = 1.0 / 50.0
MEM_SIZE = 80
ACT_DIM = 6
SPIN_ON = os.environ.get('SWSAR101_SPIN', '0') == '1'
GAP_ON = os.environ.get('SWSAR101_GAP', '0') == '1'
LAD_ON = os.environ.get('SWSAR101_LAD', '0') == '1'
LAD_WAIT = int(os.environ.get('SWSAR101_LAD_WAIT', '100'))
LAD_EVERY = int(os.environ.get('SWSAR101_LAD_EVERY', '50'))
LAD_STEP = float(os.environ.get('SWSAR101_LAD_STEP', '0.5'))
LAD_MAX = float(os.environ.get('SWSAR101_LAD_MAX', '2.5'))
LAD_AGL_MIN = float(os.environ.get('SWSAR101_LAD_AGL', '2.5'))
LAD_TOTAL = int(os.environ.get('SWSAR101_LAD_TOTAL', '300'))
COH_ON = os.environ.get('SWSAR101_COH', '1') == '1'
COH_WAIT = int(os.environ.get('SWSAR101_COH_WAIT', '200'))
COH_NEAR = float(os.environ.get('SWSAR101_COH_NEAR', '12.0'))
COH_OFF = float(os.environ.get('SWSAR101_COH_OFF', '1.2'))
COH_DZ = float(os.environ.get('SWSAR101_COH_DZ', '1.2'))
COH_MAX = int(os.environ.get('SWSAR101_COH_MAX', '400'))
COH_GUARD = 2.0
COH_SPEED = 0.45
NW_ON = os.environ.get('SWSAR101_NW', '0') == '1'
NW_TTL = int(os.environ.get('SWSAR101_NW_TTL', '50'))
NW_DENY = int(os.environ.get('SWSAR101_NW_DENY', '500'))
SKN_ON = os.environ.get('SWSAR101_SKN', '1') == '1'
SKN_MAPS = tuple(x for x in os.environ.get('SWSAR101_SKN_MAPS', 'forest,village').split(',') if x)
SKN_PERSON_H = {'city': 1.1, 'village': 1.1}
SKN_LO = float(os.environ.get('SWSAR101_SKN_LO', '0.02'))
SKN_HI = float(os.environ.get('SWSAR101_SKN_HI', '0.15'))
SKN_RGB_M = 14.0
SKN_MIN_LOOKS = int(os.environ.get('SWSAR101_SKN_LOOKS', '3'))
SKN_YES = 0.66
SKN_NO = 0.34
SKN_REJ_MAXYES = int(os.environ.get('SWSAR_SKN_MAXYES', '2'))
SKN_SILENT = int(os.environ.get('SWSAR101_SKN_SILENT', '10'))
SKN_CLOSE_SILENT = int(os.environ.get('SWSAR101_SKN_CLOSE_SILENT', '20'))
SKN_REQ_GAP = 12
SKN_REQ_MAX = 8
SKN_MEMORY = int(os.environ.get('SWSAR101_SKN_MEMORY', '500'))
SKN_FOV = 0.9
SKN_OCC_M = 3.0
def _skin_frac(img):
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
GAP_WAIT = int(os.environ.get('SWSAR101_GAP_WAIT', '100'))
GAP_VISIT = int(os.environ.get('SWSAR101_GAP_VISIT', '75'))
GAP_SPEED = float(os.environ.get('SWSAR101_GAP_SPEED', '0.4'))
GAP_REACH_M = 0.7
SPIN_T0 = int(os.environ.get('SWSAR101_SPIN_T0', '200'))
SPIN_RATE = float(os.environ.get('SWSAR101_SPIN_RATE', '75.0'))
SPIN_TURNS = float(os.environ.get('SWSAR101_SPIN_TURNS', '1.0'))
SPIN_STOP_P = float(os.environ.get('SWSAR101_SPIN_STOP_P', '0.70'))
SPIN_MAX_T = int(os.environ.get('SWSAR101_SPIN_MAX_T', '500'))
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
                     "nd_sess", "nd_calls", "nd_used")
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
OMB_ON = os.environ.get('SWSAR_OMB', '1') == '1'
OMB_MAPS = tuple(x for x in os.environ.get('SWSAR_OMB_MAPS', 'forest,city,village').split(',') if x)
OMB_R_SAFE = float(os.environ.get('SWSAR_OMB_RSAFE', '1.0'))
OMB_K = float(os.environ.get('SWSAR_OMB_K', '1.5'))
OMB_SEE_M = float(os.environ.get('SWSAR_OMB_SEE', '6.0'))
OMB_MEM_S = float(os.environ.get('SWSAR_OMB_MEM', '10.0'))
OMB_INFL_M = float(os.environ.get('SWSAR_OMB_INFL', '3.5'))
OMB_CAP_NEAR = float(os.environ.get('SWSAR_OMB_CAP_NEAR', '2.0'))
OMB_CAP_MIN = float(os.environ.get('SWSAR_OMB_CAP_MIN', '0.3'))
OMB_KEEP_M = float(os.environ.get('SWSAR_OMB_KEEP', '15.0'))
OMB_BLIND_V = float(os.environ.get('SWSAR_OMB_BLIND_V', '1.0'))
OMB_BLIND_DEG = float(os.environ.get('SWSAR_OMB_BLIND_DEG', '120.0'))
OMB_BLIND_MAPS = tuple(x for x in os.environ.get('SWSAR_OMB_BLIND_MAPS', 'forest').split(',') if x)
OMB_R_SAFE_BY_MAP = {'forest': float(os.environ.get('SWSAR_OMB_RSAFE_FOREST', '0.75')),
                     'city': float(os.environ.get('SWSAR_OMB_RSAFE_CITY', '1.2'))}
OMB_SKIP_NAV_MAPS = tuple(x for x in os.environ.get('SWSAR_OMB_SKIP_NAV', '').split(',') if x)
if os.environ.get('SWSAR_OMB_RSAFE_MOUNTAIN'):
    OMB_R_SAFE_BY_MAP['mountain'] = float(os.environ['SWSAR_OMB_RSAFE_MOUNTAIN'])
OMB_SLIDE_V = float(os.environ.get('SWSAR_OMB_SLIDE_V', '1.2'))
OMB_SLIDE_HOLD = 50
OMB_STUCK_TICKS = int(os.environ.get('SWSAR_OMB_STUCK', '150'))
OMB_STUCK_FREE = 100
OMB_VOX = 0.4
OMB_BLOCK = 8
OMB_ALL_MAPS = os.environ.get('SWSAR_OMB_ALLMAPS', '0') == '1'
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
FPP_ON = os.environ.get('SWSAR_FPP', '1') == '1'
FPP_MAPS = tuple(x for x in os.environ.get('SWSAR_FPP_MAPS', 'forest').split(',') if x)
FPP_RES = float(os.environ.get('SWSAR_FPP_RES', '0.5'))
FPP_PAD = float(os.environ.get('SWSAR_FPP_PAD', '10.0'))
FPP_INFL = float(os.environ.get('SWSAR_FPP_INFL', '1.0'))
FPP_ZLO = float(os.environ.get('SWSAR_FPP_ZLO', '0.8'))
FPP_ZHI = float(os.environ.get('SWSAR_FPP_ZHI', '1.2'))
FPP_EVERY = int(os.environ.get('SWSAR_FPP_EVERY', '10'))
FPP_LOOK = float(os.environ.get('SWSAR_FPP_LOOK', '3.0'))
FPP_MIN_H = float(os.environ.get('SWSAR_FPP_MINH', '3.0'))
FPP_MAX_H = float(os.environ.get('SWSAR_FPP_MAXH', '35.0'))
FPP_MAXCELLS = int(os.environ.get('SWSAR_FPP_MAXCELLS', '16000'))
FPP_MAXEXP = int(os.environ.get('SWSAR_FPP_MAXEXP', '12000'))
FPP_YAW = os.environ.get('SWSAR_FPP_YAW', 'path')
FPP_STALL = int(os.environ.get('SWSAR_FPP_STALL', '75'))
FPP_BLIND_DEG = float(os.environ.get('SWSAR_FPP_BLINDDEG', '60'))
FPP_BLIND_V = float(os.environ.get('SWSAR_FPP_BLINDV', '1.0'))
_FPP_STATS = {"plans": 0, "ticks": 0, "nopath": 0, "ms_max": 0.0}
APZ_ON = os.environ.get('SWSAR_APZ', '1') == '1'
APZ_MAPS = tuple(x for x in os.environ.get('SWSAR_APZ_MAPS', 'forest').split(',') if x)
APZ_STALL = int(os.environ.get('SWSAR_APZ_STALL', '50'))
APZ_HMAX = float(os.environ.get('SWSAR_APZ_HMAX', '7.0'))
APZ_OFFS = tuple(float(x) for x in os.environ.get('SWSAR_APZ_OFFS', '-0.5,-1.0,0.5,-1.5,1.0,2.0').split(',') if x)
APZ_CLR = float(os.environ.get('SWSAR_APZ_CLR', '0.6'))
APZ_ZB = float(os.environ.get('SWSAR_APZ_ZB', '0.4'))
APZ_AGL = float(os.environ.get('SWSAR_APZ_AGL', '2.5'))
APZ_TOP = float(os.environ.get('SWSAR_APZ_TOP', '1.3'))
APZ_V = float(os.environ.get('SWSAR_APZ_V', '1.5'))
APZ_REL = float(os.environ.get('SWSAR_APZ_REL', '1.0'))
APZ_TMAX = int(os.environ.get('SWSAR_APZ_TMAX', '300'))
APZ_PSTALL = int(os.environ.get('SWSAR_APZ_PSTALL', '60'))
APZ_MAXN = int(os.environ.get('SWSAR_APZ_MAXN', '3'))
APZ_UNFRZ = os.environ.get('SWSAR_APZ_UNFRZ', '1') == '1'
APZ_ECL = os.environ.get('SWSAR_APZ_ECL', '1') == '1'
APZ_ECL_H = float(os.environ.get('SWSAR_APZ_ECLH', '3.0'))
APZ_TOPMAX = float(os.environ.get('SWSAR_APZ_TOPMAX', '0'))
_APZ = {"start": 0, "done": 0, "abort": 0, "nofree": 0, "ticks": 0}
def _apz_seg_free(pts, p0, p1, z, clr, zb):
    if pts is None or len(pts) == 0:
        return True
    m = np.abs(pts[:, 2] - z) <= zb
    if not m.any():
        return True
    q = pts[m, 0:2]
    a = np.asarray(p0[0:2], dtype=np.float64)
    b = np.asarray(p1[0:2], dtype=np.float64)
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-9:
        d = np.hypot(q[:, 0] - a[0], q[:, 1] - a[1])
    else:
        t = np.clip(((q - a) @ ab) / L2, 0.0, 1.0)
        d = np.hypot(q[:, 0] - (a[0] + t * ab[0]), q[:, 1] - (a[1] + t * ab[1]))
    return bool(np.all(d > clr))
def _apz_col_free(pts, xy, z0, z1, clr, zb):
    if pts is None or len(pts) == 0:
        return True
    lo, hi = min(z0, z1) - zb, max(z0, z1) + zb
    m = (pts[:, 2] >= lo) & (pts[:, 2] <= hi)
    if not m.any():
        return True
    d = np.hypot(pts[m, 0] - xy[0], pts[m, 1] - xy[1])
    return bool(np.all(d > clr))
def _fpp_plan(pos, goal, pts):
    path = _fpp_plan1(pos, goal, pts, FPP_PAD)
    if path is None:
        path = _fpp_plan1(pos, goal, pts, FPP_PAD * 1.6)
    return path
def _fpp_plan1(pos, goal, pts, pad):
    import heapq
    res = FPP_RES
    x0 = min(pos[0], goal[0]) - pad
    x1 = max(pos[0], goal[0]) + pad
    y0 = min(pos[1], goal[1]) - pad
    y1 = max(pos[1], goal[1]) + pad
    nx = int(math.ceil((x1 - x0) / res)) + 1
    ny = int(math.ceil((y1 - y0) / res)) + 1
    while nx * ny > FPP_MAXCELLS:
        res *= 1.25
        nx = int(math.ceil((x1 - x0) / res)) + 1
        ny = int(math.ceil((y1 - y0) / res)) + 1
    occ = np.zeros((ny, nx), dtype=bool)
    if pts is not None and len(pts):
        m = (pts[:, 2] >= pos[2] - FPP_ZLO) & (pts[:, 2] <= pos[2] + FPP_ZHI)
        q = pts[m]
        if len(q):
            ix = np.floor((q[:, 0] - x0) / res).astype(np.int64)
            iy = np.floor((q[:, 1] - y0) / res).astype(np.int64)
            ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            occ[iy[ok], ix[ok]] = True
    k = int(math.ceil(FPP_INFL / res))
    if k > 0 and occ.any():
        base = occ.copy()
        for dy in range(-k, k + 1):
            for dx in range(-k, k + 1):
                if dx * dx + dy * dy > k * k:
                    continue
                sy0, sy1 = max(0, dy), ny + min(0, dy)
                sx0, sx1 = max(0, dx), nx + min(0, dx)
                occ[sy0:sy1, sx0:sx1] |= base[sy0 - dy:sy1 - dy, sx0 - dx:sx1 - dx]
    sx = int((pos[0] - x0) / res)
    sy = int((pos[1] - y0) / res)
    gx = int((goal[0] - x0) / res)
    gy = int((goal[1] - y0) / res)
    kc = max(1, int(math.ceil(0.6 / res)))
    occ[max(0, sy - kc):sy + kc + 1, max(0, sx - kc):sx + kc + 1] = False
    kg = max(1, int(math.ceil(1.5 / res)))
    occ[max(0, gy - kg):gy + kg + 1, max(0, gx - kg):gx + kg + 1] = False
    start, tgt = (sy, sx), (gy, gx)
    g = {start: 0.0}
    came = {}
    h0 = math.hypot(gx - sx, gy - sy)
    heap = [(h0, 0.0, start)]
    moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, 1.41421356), (1, -1, 1.41421356), (-1, 1, 1.41421356), (-1, -1, 1.41421356)]
    exp = 0
    found = False
    while heap:
        f, gc, cur = heapq.heappop(heap)
        if cur == tgt:
            found = True
            break
        if gc > g.get(cur, 1e18) + 1e-9:
            continue
        exp += 1
        if exp > FPP_MAXEXP:
            break
        cy, cx = cur
        for dy, dx, w in moves:
            yy, xx = cy + dy, cx + dx
            if yy < 0 or yy >= ny or xx < 0 or xx >= nx or occ[yy, xx]:
                continue
            if dy and dx and (occ[cy, xx] or occ[yy, cx]):
                continue
            ng = gc + w
            if ng < g.get((yy, xx), 1e18):
                g[(yy, xx)] = ng
                came[(yy, xx)] = cur
                heapq.heappush(heap, (ng + math.hypot(gx - xx, gy - yy), ng, (yy, xx)))
    if not found:
        return None
    path = [tgt]
    while path[-1] != start:
        path.append(came[path[-1]])
    path.reverse()
    return np.array([[x0 + (cx + 0.5) * res, y0 + (cy + 0.5) * res] for cy, cx in path], dtype=np.float64)
def _fpp_dbg():
    if not FPP_ON:
        return {"fpp_on": 0.0}
    return {"apz_start": float(_APZ["start"]), "apz_done": float(_APZ["done"]), "apz_abort": float(_APZ["abort"]),
            "apz_nofree": float(_APZ["nofree"]), "apz_ticks": float(_APZ["ticks"]),
            "fpp_on": 1.0, "fpp_eng": float(_FPP_STATS.get("eng", 0)), "fpp_plans": float(_FPP_STATS["plans"]), "fpp_ticks": float(_FPP_STATS["ticks"]),
            "fpp_nopath": float(_FPP_STATS["nopath"]), "fpp_ms_max": float(_FPP_STATS["ms_max"])}
_OMB_GRID = None
def _omb_grid():
    global _OMB_GRID
    if _OMB_GRID is None:
        nb = DEPTH_RES // OMB_BLOCK
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
    def __init__(self):
        self.reset()
    def reset(self):
        self.keys = np.zeros(0, dtype=np.int64)
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
        changed = False
        if rpy is not None and OMB_BLIND_V > 0.0 and _MAPSW_STATE.get("map") in OMB_BLIND_MAPS:
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
        self._ostreak_n = 0
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
        self._bkp_t0 = None
        self._bkp_last = -99
        self._fpp = None
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
            self.ctrl._nav_head_z = None
            self._apz = None
            self._apz_n = 0
            self._apz_prog = None
            self.ctrl._swarm_r_scale = 1.0
        except Exception:
            pass
        self._in_support = None
        self._support_src = None
        self._support_fn = None
        self._support_relax = None
        self._outer_admits = 0
        self._ostreak_n = 0
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
        self._bkp_t0 = None
        self._bkp_last = -99
        self._fpp = None
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
        if (MCS2G_ON and MCS2G_YO and (not self._stride_ok) and getattr(det, 'mcs2g_yes_only', False)
                and _rgb_present(rgb)):
            try:
                _pq = det._rgb_prob(rgb, is_mountain)
            except Exception:
                _pq = None
            if _pq is not None and _pq < RGB_CONFIRM_THRESH:
                _MCS2G["yo_quiet"] = _MCS2G.get("yo_quiet", 0) + 1
                rgb = None
        if self._stride_ok or _rgb_present(rgb):
            self._served += 1
            return self._det_update(depth, pos, rpy, rgb=rgb, is_mountain=is_mountain)
        self._skipped += 1
        det.t += 1
        det.rgb_fresh = False
        det.rgb_promoted = False
        det.seen = False
        if det.t - det.last_strong_t > _ks.STALE_TICKS:
            _dec = VFC_DECAY if _vfc_on() else _ks.STALE_DECAY
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
        if FPP_ON and OMB_ON and allow_confirm and _MAPSW_STATE.get("map") in FPP_MAPS:
            try:
                a = self._fpp_step(a, s)
            except Exception as exc:
                _log_v21("fpp_step", "forest path planner raised %r; champion action this tick" % (exc,))
        if APZ_ON and OMB_ON and allow_confirm and _MAPSW_STATE.get("map") in APZ_MAPS and not self.ctrl.mtn.is_mountain:
            try:
                a = self._apz_step(a, s)
            except Exception as exc:
                _log_v21("apz_step", "approach corridor raised %r; champion action this tick" % (exc,))
        elif getattr(self, "_apz", None) is not None:
            self._apz = None
        if OMB_ON and _omb_world():
            try:
                self._omb.observe(obs["depth"], s[0:3], s[3:6])
                self._omb.stats["omb_ticks"] += 1
                if not (OMB_SKIP_NAV_MAPS and _MAPSW_STATE.get('map') in OMB_SKIP_NAV_MAPS and self.ctrl.mode == 'navigation'):
                    self._omb.constrain(a, s[0:3], s[3:6])
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
                c.mq_stats["fix_denyskip"] = c.mq_stats.get("fix_denyskip", 0) + 1
                return
            self._drop_lock()
    def _drop_lock(self):
        c = self.ctrl
        det = c.detector
        c.locked = False
        c.frozen = False
        c.lost_ticks = 0
        c.hover_stable_ticks = 0
        c._hsd_settle = 0
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
        vz = float(a[2]) * float(a[3])
        out[0] = 0.0
        out[1] = 0.0
        out[2] = 1.0 if vz > 0.0 else (-1.0 if vz < 0.0 else 0.0)
        out[3] = float(np.clip(abs(vz), 0.0, 1.0))
        out[4] = float(np.clip(target / math.pi, -1.0, 1.0))
        self.spin_stats["spin_ticks"] += 1
        return out
    def _apz_step(self, a, s):
        c = self.ctrl
        pos = np.asarray(s[0:3], dtype=np.float64)
        est = np.asarray(c.locked_vic, dtype=np.float64)
        h = math.hypot(est[0] - pos[0], est[1] - pos[1])
        E = getattr(self, "_apz", None)
        if E is not None:
            if not (c.locked and (c.frozen or APZ_UNFRZ) and c.mode in ('navigation', 'hover')) \
                    or math.hypot(E["est"][0] - est[0], E["est"][1] - est[1]) > 1.5:
                self._apz = None
                _APZ["abort"] += 1
                return a
            if h <= APZ_REL:
                self._apz = None
                _APZ["done"] += 1
                return a
            if h < E["best_h"] - 0.3:
                E["best_h"], E["best_t"] = h, int(c.tick)
            if c.tick - E["t0"] > APZ_TMAX or (E["phase"] == 1 and c.tick - E["best_t"] > APZ_PSTALL):
                self._apz = None
                self._apz_prog = None
                _APZ["abort"] += 1
                return a
            S = SPEED_LIMIT
            if APZ_ECL and E["phase"] == 1 and h < APZ_ECL_H:
                E["z"] = max(E["z"], float(c.locked_head_z) + HOVER_ABOVE_TOP_M + HOVER_SETTLE_MARGIN_M + _FHZ_DZ)
                _APZ["ecl"] = _APZ.get("ecl", 0) + 1
            dz = E["z"] - float(pos[2])
            out = np.array(a, dtype=np.float32)
            if E["phase"] == 0:
                if abs(dz) <= 0.2:
                    E["phase"], E["best_t"] = 1, int(c.tick)
                v = np.array([0.0, 0.0, float(np.clip(1.5 * dz, -0.8, 0.8))]) / S
            if E["phase"] == 1:
                ux, uy = (est[0] - pos[0]) / max(h, 1e-6), (est[1] - pos[1]) / max(h, 1e-6)
                vh = min(APZ_V, 0.6 * h + 0.2)
                v = np.array([ux * vh, uy * vh, float(np.clip(1.5 * dz, -0.6, 0.6))]) / S
            _store_vel(out, v)
            _APZ["ticks"] += 1
            return out
        if not (c.locked and (c.frozen or APZ_UNFRZ) and c.mode == 'navigation'):
            self._apz_prog = None
            return a
        P = getattr(self, "_apz_prog", None)
        if P is None or math.hypot(P["est"][0] - est[0], P["est"][1] - est[1]) > 1.5:
            P = self._apz_prog = {"est": est.copy(), "best_h": h, "best_t": int(c.tick)}
        if h < P["best_h"] - 0.3:
            P["best_h"], P["best_t"] = h, int(c.tick)
        if h < R_HOVER_M or h > APZ_HMAX or c.tick - P["best_t"] < APZ_STALL:
            return a
        if int(getattr(self, "_apz_n", 0)) >= APZ_MAXN:
            return a
        pts = getattr(self._omb, "pts", None)
        agl = getattr(c, "_last_agl", None)
        gz = float(pos[2]) - float(agl) if (agl is not None and np.isfinite(agl)) else 0.0
        zmin = max(gz + APZ_AGL, float(c.locked_head_z) + APZ_TOP)
        u = (est[0:2] - pos[0:2]) / max(h, 1e-6)
        end = est[0:2] - 0.5 * u
        for off in APZ_OFFS:
            zc = float(pos[2]) + off
            if zc < zmin:
                continue
            if APZ_TOPMAX > 0.0 and zc > float(c.locked_head_z) + APZ_TOPMAX:
                continue
            if _apz_col_free(pts, pos[0:2], float(pos[2]), zc, APZ_CLR, APZ_ZB) and \
                    _apz_seg_free(pts, pos[0:2], end, zc, APZ_CLR, APZ_ZB):
                self._apz = {"z": zc, "phase": 0, "t0": int(c.tick), "est": est.copy(), "best_h": h, "best_t": int(c.tick)}
                self._apz_n = int(getattr(self, "_apz_n", 0)) + 1
                _APZ["start"] += 1
                return self._apz_step(a, s)
        _APZ["nofree"] += 1
        P["best_t"] = int(c.tick)
        return a
    def _fpp_step(self, a, s):
        c = self.ctrl
        if not (c.locked and (not c.frozen) and c.mode == 'navigation'):
            self._fpp = None
            return a
        pos = np.asarray(s[0:3], dtype=np.float64)
        est = np.asarray(c.locked_vic, dtype=np.float64)
        h = math.hypot(est[0] - pos[0], est[1] - pos[1])
        if h < FPP_MIN_H or h > FPP_MAX_H:
            return a
        st = getattr(self, "_fpp", None)
        if st is not None and math.hypot(st["est"][0] - est[0], st["est"][1] - est[1]) > _ks.CONSIST_M:
            st = None
        if st is None:
            st = self._fpp = {"est": est.copy(), "path": None, "t": -10 ** 9, "best_h": 1e9, "best_t": int(c.tick),
                              "eng": FPP_STALL <= 0}
        if h < st["best_h"] - 0.3:
            st["best_h"], st["best_t"] = h, int(c.tick)
        if not st["eng"]:
            if c.tick - st["best_t"] < FPP_STALL:
                return a
            st["eng"] = True
            _FPP_STATS["eng"] = _FPP_STATS.get("eng", 0) + 1
        if c.tick - st["t"] >= FPP_EVERY:
            import time as _time
            _t0 = _time.perf_counter()
            path = _fpp_plan(pos, est, getattr(self._omb, "pts", None))
            _ms = (_time.perf_counter() - _t0) * 1000.0
            _FPP_STATS["ms_max"] = max(_FPP_STATS["ms_max"], _ms)
            _FPP_STATS["plans"] += 1
            st["t"] = int(c.tick)
            st["est"] = est.copy()
            if path is None or len(path) < 2:
                _FPP_STATS["nopath"] += 1
                st["path"] = None
                return a
            st["path"] = path
        path = st["path"]
        if path is None:
            return a
        d = np.hypot(path[:, 0] - pos[0], path[:, 1] - pos[1])
        i0 = int(np.argmin(d))
        seg = np.hypot(np.diff(path[i0:, 0]), np.diff(path[i0:, 1]))
        acc = np.concatenate([[0.0], np.cumsum(seg)])
        j = i0 + int(np.searchsorted(acc, FPP_LOOK))
        j = min(j, len(path) - 1)
        wx, wy = path[j, 0] - pos[0], path[j, 1] - pos[1]
        wn = math.hypot(wx, wy)
        if wn < 0.3:
            return a
        v = _vel_of(a)
        sp_h = math.hypot(v[0], v[1])
        if sp_h < 1e-6:
            return a
        if FPP_BLIND_DEG > 0.0 and FPP_YAW != 'path':
            _yaw = float(s[5])
            _mis = abs((math.atan2(wy, wx) - _yaw + math.pi) % (2.0 * math.pi) - math.pi)
            if _mis > math.radians(FPP_BLIND_DEG):
                sp_h = min(sp_h, FPP_BLIND_V / SPEED_LIMIT)
        v[0], v[1] = wx / wn * sp_h, wy / wn * sp_h
        out = np.array(a, dtype=np.float32)
        _store_vel(out, v)
        if FPP_YAW == 'path':
            out[4] = float(np.clip(math.atan2(wy, wx) / math.pi, -1.0, 1.0))
        self._fbd = None
        _FPP_STATS["ticks"] += 1
        return out
    def _detour_step(self, a, s):
        c = self.ctrl
        f = getattr(self, "_fbd", None)
        if not (c.locked or c.frozen) or c.mode not in ('navigation', 'hover'):
            self._fbd = None
            return a
        est = np.asarray(c.locked_vic, dtype=np.float64)
        pos = np.asarray(s[0:3], dtype=np.float64)
        if f is not None and math.hypot(f["est"][0] - est[0], f["est"][1] - est[1]) > _ks.CONSIST_M:
            f = None
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
        c = self.ctrl
        g = self._gap
        if not (c.locked or c.frozen):
            self._gap = None
            return a
        est = np.asarray(c.locked_vic, dtype=np.float64)
        if g is not None and math.hypot(g["est"][0] - est[0], g["est"][1] - est[1]) > _ks.CONSIST_M:
            g = self._gap = None
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
            c.locked_vic[0] = float(q[0])
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
    def _skg_spare(self, K):
        if not K.get("skg"):
            K["skg"] = True
            _SKL["skg"] = _SKL.get("skg", 0) + 1
        return True
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
            try:
                _pr = float(getattr(det, "last_p_depth", 0.0))
                if in_view and visible:
                    K["pmax"] = max(K.get("pmax", 0.0), _pr)
                    if _pr >= 0.7:
                        K["dp"] = K.get("dp", 0) + 1
                        if rng < 8.0:
                            K["cdp"] = K.get("cdp", 0) + 1
            except Exception:
                pass
        near = rng <= SKN_RGB_M
        if det.rgb_fresh and in_view and visible and near and rgb is not None:
            try:
                fr = np.asarray(_to_np(rgb), dtype=np.float32).reshape(256, 256, 3)
                H = 256
                px = int((u + 1.0) * 0.5 * H)
                py = int((1.0 - v) * 0.5 * H)
                ph_m = SKN_PERSON_H.get(_MAPSW_STATE.get("map"), 1.8)
                half = max(3, int(64.0 * ph_m / max(rng, 1.0)))
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
        elif looks >= SKN_MIN_LOOKS and frac <= SKN_NO and K["close_hits"] == 0 and not (K["yes"] >= SKN_REJ_MAXYES and _MAPSW_STATE.get("map") == 'forest') \
                and (K["close_inview"] >= SKN_CLOSE_SILENT or (K["inview"] >= 3 * SKN_SILENT and K["dyes"] <= 1)) \
                and not (SKG_ON and _MAPSW_STATE.get("map") in SKG_MAPS and K.get("cdp", 0) >= SKG_CDP
                         and self._skg_spare(K)):
            K["verdict"] = -1
            self.spin_stats["skn_reject"] += 1
            try:
                _SKL["n"] = _SKL.get("n", 0) + 1
                if len(_SKL["log"]) < SKL_NLOG:
                    _SKL["log"].append((_MPL["now"] * SIM_DT, float(est[0]), float(est[1]), K["looks"], K["yes"],
                                        K["inview"], K["close_inview"], K["dyes"], K["close_hits"], K.get("dp", 0),
                                        K.get("cdp", 0), K.get("pmax", 0.0)))
            except Exception:
                pass
            self._soft_deny.append((np.array([est[0], est[1]], dtype=np.float64), c.tick + SKN_MEMORY))
            if len(self._soft_deny) > 8:
                self._soft_deny.pop(0)
            self._drop_lock()
            self._skn = None
        return a
    def _ladder_step(self, s):
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
            dz0 = float(pos[2]) - L["head0"]
            if near and _ks.HOVER_GIVEUP_BAND[0] <= dz0 <= _ks.HOVER_GIVEUP_BAND[1] \
                    and not (_ks.HOVER_GIVEUP_BAND[0] <= float(pos[2]) - float(c.locked_head_z) <= _ks.HOVER_GIVEUP_BAND[1]):
                c.hover_stable_ticks += 1
            if L["active"] >= LAD_TOTAL:
                c.locked_head_z = L["head0"]
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
        c._hov_z_lock = None
        L["lowered"] += LAD_STEP
        L["last"] = int(c.tick)
        if L["lowered"] <= LAD_STEP + 1e-6:
            self.spin_stats["lad_started"] += 1
        self.spin_stats["lad_steps"] += 1
    def _co_hover(self, a, s, tgt, co):
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
    DET_STRIDE = int(os.environ.get('NEWDET_DET_STRIDE', '3'))
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
        self.cohover = -1
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
            hov = bool(_SUPK_ON and (self.mtn or _MAPSW_STATE.get("map") in _SUPK_MAPS)
                       and b["lock"] >= 1.0 and _horiz(pos[j], rec["xy"]) <= _SUPK_R)
            key = (b["n_det"] > self.ELECT_NDET or hov, float(rec["hits"]), float(b["lock"]), hov,
                   (-float(_horiz(pos[j], rec["xy"])) if hov else 0.0), -int(j))
            if best is None or key > best_key:
                best, best_key, best_j = rec, key, j
        if best is not None and best_key[3]:
            self.stats["sarz_supk"] = self.stats.get("sarz_supk", 0) + 1
        return (best, best_j) if best is not None else None
    def _release(self, reason, blame=True):
        i = self.confirmer
        if i < 0:
            return
        self.cooldown[i] = self.tick + self.COOLDOWN
        if self.mtn and self.cand is not None and np.all(np.isfinite(self.cand["xy"])):
            self.released_xy[i] = np.asarray(self.cand["xy"], dtype=np.float64).copy()
            self.private_pending[i] = bool(TEAM_DENY_FIX)
            if reason == "lock_lost" and self.private_pending[i] and float(self.best_h[i]) > FLTE_PDN_H:
                _FLTE_STATE["pdn_would"] = _FLTE_STATE.get("pdn_would", 0.0) + 1.0
                if FLTE_PDN and not _flte_z0_ok():
                    _FLTE_STATE["z0_block"] = _FLTE_STATE.get("z0_block", 0.0) + 1.0
                elif FLTE_PDN:
                    if FLTE_PDN_TICKS > 0:
                        self._flte_short = getattr(self, "_flte_short", set())
                        self._flte_short.add(int(i))
                        _FLTE_STATE["pdn_short"] = _FLTE_STATE.get("pdn_short", 0.0) + 1.0
                    else:
                        self.private_pending[i] = False
                        _FLTE_STATE["pdn_skip"] = _FLTE_STATE.get("pdn_skip", 0.0) + 1.0
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
                _conf_frozen = bool(b is not None and b["lock"] >= 1.0)
                _stall = bool(held >= self.MIN_DWELL and held > TEL_HANDOVER_TICKS)
                _hoc = bool(HOC_ON and (not _stall) and _MAPSW_STATE.get("map") in HOC_MAPS)
                if (TEL_HANDOVER_TICKS > 0 and (_stall or _hoc) and (t - self.elect_t0) >= self.MIN_DWELL
                        and not _conf_frozen
                        and int(self.cand.get("handovers", 0)) < TEL_HANDOVER_MAX):
                    cxy = self.cand["xy"]
                    best_j, best_hj = -1, min(float(self.best_h[c]), h)
                    if _hoc:
                        best_hj = h - (HOC_GAIN - TEL_HANDOVER_GAIN)
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
                            if _hoc:
                                self.stats["hoc_handover"] = self.stats.get("hoc_handover", 0) + 1
                            _handed = True
                if ((not _handed) and _DHO_ON and self.mtn and not _conf_frozen
                        and (t - self.elect_t0) >= self.MIN_DWELL and int(self.cand.get("dho", 0)) < 1):
                    cxy = self.cand["xy"]
                    best_j, best_hj = -1, h - _DHO_GAIN
                    for j in live:
                        if j == c or self.frozen[j] or j >= len(beliefs) or beliefs[j] is None:
                            continue
                        bj = beliefs[j]
                        if bj["lock"] < self.ELECT_LOCK or bj["n_det"] <= self.ELECT_NDET or not np.all(np.isfinite(bj["est"])):
                            continue
                        if _horiz(bj["est"], cxy) >= self.CONF_NEAR:
                            continue
                        hj = _horiz(pos[j], cxy)
                        if hj <= _DHO_R and hj < best_hj:
                            best_j, best_hj = int(j), hj
                    if best_j >= 0:
                        self.cand["dho"] = 1
                        self._release("handover", blame=False)
                        self.confirmer = best_j
                        self.elect_tick = self.elect_t0 = t
                        self.conf_frozen_t = -1
                        self.lost[best_j] = 0
                        self.best_h[best_j] = best_hj
                        self.stats["elections"] += 1
                        self.stats["sarz_dho"] = self.stats.get("sarz_dho", 0) + 1
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
                            _ttl = TEAM_DENY_PRIVATE_TICKS
                            if FLTE_PDN and int(i) in getattr(self, "_flte_short", set()):
                                _ttl = min(_ttl, FLTE_PDN_TICKS)
                                self._flte_short.discard(int(i))
                            hook(rx, self.tick + _ttl)
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
        if APZ_ON:
            try:
                if getattr(self._apz_pilots[i], "_apz", None) is not None:
                    return out
            except Exception:
                pass
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
                continue
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
        _MPL["now"] = T.tick
        if T.tick == 0:
            _flte_reset()
            _MPL.update(boost=0, swap=0, log=[])
            _FPP_STATS.update(plans=0, ticks=0, nopath=0, ms_max=0.0, eng=0)
            _APZ.update(start=0, done=0, abort=0, nofree=0, ticks=0)
            _SKL.update(log=[], n=0, skg=0)
            _VEG.update(arm=0, drop=0, weak=0)
            _VC.update(vgr_ref=0, vgr_pz=0, pause=0, resume=0, catchup=0, err=0)
            try:
                _FLTE_STATE["z0"] = float(np.max(pos[:n, 2]))
            except Exception:
                pass
            T.pad0 = pos[0, 0:2].copy()
            _MCS2G["pads"] = np.asarray(pos[:n, 0:3], dtype=np.float64).copy()
            _MCS2G.update(scans=0, skip=0, padveto=0, promo=0, promo_scan=0, log=[], starved=0, padv=0)
            _MCS2G["yo_skips"] = 0
            _rgbm_reset()
            _MCS2G["yo_quiet"] = 0
            T._tks_starts = pos[:, 0:2].copy()
            T._tks_set = False
        T.update_frozen(pos, vel)
        live = [i for i in range(n) if not T.frozen[i]]
        try:
            T.mtn = bool(self.pilots[0].p.ctrl.mtn.is_mountain)
        except Exception:
            T.mtn = False
        if (MTYP_ON and MTYP_RELABEL and MTN_RECON_ON and T.tick == MTN_RECON_TICK and not T.mtn
                and _MAPSW_STATE.get("map") == 'city'
                and float(_MSWF_STATE.get("zmin", -1.0)) > MTYP_ZMIN
                and float(_MSWF_STATE.get("zmax", -1.0)) > MTYP_ZMAX
                and getattr(T, "_mrecon_zmax", None) is not None
                and T._mrecon_zmax >= MTN_RECON_ZMIN
                and (MTN_RECON_NMAX <= 0 or n <= MTN_RECON_NMAX)):
            _MAPSW_STATE["map"] = 'mountain'
            T.stats["mtyp_relabel"] = 1
        if MTN_RECON_ON:
            if T.tick == 0:
                T._mrecon_zmax = float(np.max(pos[:n, 2]))
            elif (T.tick == MTN_RECON_TICK and not T.mtn
                    and (_MAPSW_STATE.get("map") == 'mountain'
                         or (MZL_Z > 0.0 and MZL_MODE == 'late' and getattr(T, "_mrecon_zmax", None) is not None
                             and T._mrecon_zmax >= MZL_Z))
                    and getattr(T, "_mrecon_zmax", None) is not None
                    and T._mrecon_zmax >= MTN_RECON_ZMIN
                    and (MTN_RECON_NMAX <= 0 or n <= MTN_RECON_NMAX)):
                _hit = False
                for _i in range(min(n, len(self.pilots))):
                    try:
                        _m = self.pilots[_i].p.ctrl.mtn
                        _real = getattr(getattr(_m, "bus", None), "real", _m)
                        if not bool(_real.is_mountain):
                            _real.is_mountain = True
                            _real.latched.add('mountain')
                        _hit = True
                    except Exception:
                        pass
                if _hit:
                    T.mtn = True
                    T.stats["mtn_recon"] = 1
                    if _MAPSW_STATE.get("map") != 'mountain':
                        _MAPSW_STATE["map"] = 'mountain'
                        _MSWF_STATE["mzl"] = 2
                    T.stats["mtn_recon_t"] = float(T.tick) * DT
        if MTYP_ON and T.stats.get("mtyp_relabel", 0) == 1 and not T.mtn:
            _MAPSW_STATE["map"] = 'city'
            T.stats["mtyp_relabel"] = 2
        _VC['mtn'] = bool(T.mtn)
        need_belief = (not T.region_final) or (not T.type_bcast)
        if need_belief:
            p_type = T.team_p_type_from(self._p_env_of, live)
            if p_type is None:
                p_type = T.p_env_team
        else:
            p_type = T.p_env_team
        airborne = bool(np.any(agl[:n] > 1.0))
        _mtyp_p = p_type
        _mtyp_fire = False
        if (MTYP_ON and MTYP_MODE in ('relatch', 'support') and T.stats.get("mtn_recon", 0)
                and T.region_latched and not T.region_final and p_type is not None):
            try:
                _pr = np.asarray(p_type, dtype=np.float64).ravel().copy()
                if _pr.size >= 3 and float(_pr[2]) < MFP_P_MTN:
                    _mtyp_fire = True
                    T.stats["mtyp_pmtn"] = float(_pr[2])
                    _pr[:] = 0.0
                    _pr[2] = 1.0
                    if MTYP_MODE == 'relatch':
                        _mtyp_p = _pr
            except Exception:
                _mtyp_fire = False
        T.maybe_latch_region(n, pos, state[:, S_CLUE].astype(np.float64),
                             _mtyp_p, airborne)
        if _mtyp_fire and T.region_final:
            if MTYP_MODE == 'relatch':
                T.stats["mtyp_relatch"] = T.stats.get("mtyp_relatch", 0) + 1
            else:
                try:
                    _post = T.region.native
                    _alt = type(_post)()
                    _alt.set_geometry(np.asarray(T.clue_world, dtype=np.float64)[:2],
                                      np.asarray(T.pad0 if T.pad0 is not None else pos[0, 0:2],
                                                 dtype=np.float64)[:2], int(n))
                    _alt.set_type_belief(_pr)
                    T.region.payload = _MtypUnionRegion(T.region.payload, _alt)
                    T.stats["mtyp_union"] = T.stats.get("mtyp_union", 0) + 1
                except Exception as exc:
                    T.stats["mtyp_err"] = T.stats.get("mtyp_err", 0) + 1
                    _log("mtyp", "union support failed %r" % (exc,))
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
        if APZ_ON:
            try:
                T._apz_pilots = [getattr(pl, "p", None) for pl in self.pilots]
            except Exception:
                pass
        out = T.guard_agl_floor(out, agl, live)
        out = T.guard_keepout(out, pos, vel, live, beliefs)
        out = T.guard_confirmer_band(out, pos, agl, live, beliefs)
        out = T.separate(out, pos, vel, n, live)
        out = T.limit_correction(out, base, live)
        out = T.guard_agl_floor(out, agl, live)
        if OMB_ON and _omb_world():
            for i in live:
                try:
                    _om = getattr(self.pilots[i].p, "_omb", None)
                    if _om is not None:
                        if not (OMB_SKIP_NAV_MAPS and _MAPSW_STATE.get('map') in OMB_SKIP_NAV_MAPS and getattr(getattr(self.pilots[i].p, 'ctrl', None), 'mode', '') == 'navigation'):
                            _om.constrain(out[i], pos[i], state[i, 3:6])
                except Exception:
                    pass
        if MSS_ON and T.mtn:
            _S = float(T.SPEED_LIMIT)
            _mst = getattr(T, "_mss_st", None)
            if _mst is None:
                _mst = T._mss_st = {}
                T._mss_last = {}
                T._mss_rate = {}
                T._mss_seen = set()
            for i in range(n):
                if i not in live:
                    _mst.pop(i, None)
            for i in live:
                try:
                    a_i = float(agl[i])
                    if not np.isfinite(a_i):
                        continue
                    _l = T._mss_last.get(i)
                    if _l is not None:
                        T._mss_rate[i] = (1.0 - MSS_ALPHA) * T._mss_rate.get(i, 0.0) + MSS_ALPHA * (a_i - _l) / DT
                    T._mss_last[i] = a_i
                    _st = _mst.get(i)
                    _c = getattr(getattr(self.pilots[i], "p", None), "ctrl", None) if i < len(self.pilots) else None
                    _elig = (_c is not None and i != T.confirmer and not (COH_ON and i == T.cohover)
                             and getattr(_c, "mode", "") in ("search", "navigation")
                             and not bool(getattr(_c, "locked", False)) and not bool(getattr(_c, "frozen", False)))
                    _va = np.asarray(vel[i], dtype=np.float64)
                    _trig = bool(_elig and a_i < MSS_A_ON and T._mss_rate.get(i, 0.0) < -MSS_RATE
                                 and math.hypot(float(_va[0]), float(_va[1])) >= MSS_VH)
                    if _st is None:
                        if not _trig:
                            continue
                        _st = _mst[i] = {"on": False, "eng": False, "hold": 0, "g": None}
                    _vp = _vel_of(out[i]) * _S
                    if _trig:
                        if not _st["on"]:
                            _st["on"] = True
                            _st["g"] = _vp.copy()
                            T.stats["mss_fires"] = T.stats.get("mss_fires", 0) + 1
                            if i not in T._mss_seen:
                                T._mss_seen.add(i)
                                T.stats["mss_drones"] = T.stats.get("mss_drones", 0) + 1
                        _st["eng"] = True
                        _st["hold"] = MSS_HOLD
                    elif _st["eng"]:
                        _st["hold"] -= 1
                        if not _elig or (_st["hold"] <= 0 and a_i >= MSS_A_REL):
                            _st["eng"] = False
                    if not _st["on"]:
                        _mst.pop(i, None)
                        continue
                    _g = _st["g"]
                    _hp = float(math.hypot(float(_vp[0]), float(_vp[1])))
                    if _st["eng"]:
                        T.stats["mss_eng_ticks"] = T.stats.get("mss_eng_ticks", 0) + 1
                        if _hp > 1e-6:
                            _ux, _uy = float(_vp[0]) / _hp, float(_vp[1]) / _hp
                        else:
                            _gh = float(math.hypot(float(_g[0]), float(_g[1])))
                            _ux, _uy = (float(_g[0]) / _gh, float(_g[1]) / _gh) if _gh > 1e-6 else (0.0, 0.0)
                        _ht = min(_hp, MSS_HMIN)
                        _tgt = np.array([_ux * _ht, _uy * _ht, max(float(_vp[2]), math.sqrt(max(0.0, _S * _S - _ht * _ht)))])
                        _dhm = MSS_DH
                    else:
                        _tgt = _vp.copy()
                        _dhm = MSS_DHREL
                    _d = _tgt - _g
                    _dh = float(math.hypot(float(_d[0]), float(_d[1])))
                    if _dh > _dhm:
                        _d[0] *= _dhm / _dh
                        _d[1] *= _dhm / _dh
                    _d[2] = min(max(float(_d[2]), -MSS_DZDN), MSS_DZUP)
                    _g = _g + _d
                    if _g[2] > float(_va[2]) + MSS_EZUP and _g[2] > float(_vp[2]):
                        _g[2] = max(float(_vp[2]), float(_va[2]) + MSS_EZUP)
                    _sp = float(np.linalg.norm(_g))
                    if _sp > _S:
                        _hz = float(math.hypot(float(_g[0]), float(_g[1])))
                        if _hz >= _S:
                            _g[0] *= _S / _hz
                            _g[1] *= _S / _hz
                            _g[2] = 0.0
                        else:
                            _g[2] = math.copysign(math.sqrt(_S * _S - _hz * _hz), float(_g[2]))
                    _st["g"] = _g
                    if (not _st["eng"]) and float(np.linalg.norm(_g - _vp)) <= MSS_TOL:
                        _mst.pop(i, None)
                        continue
                    _store_vel(out[i], _g / _S)
                    T.stats["mss_ticks"] = T.stats.get("mss_ticks", 0) + 1
                except Exception:
                    _mst.pop(i, None)
        if PDH_ON:
            try:
                self._pdh_step(out, pos, state, live, n)
            except Exception as exc:
                _log("pdh", "perch-descent hold raised %r" % (exc,))
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
    def _pdh_step(self, out, pos, state, live, n):
        T = self.T
        P = getattr(T, "_pdh", None)
        if P is None or P["z0"].shape[0] != n:
            if T.tick != 0:
                return
            P = T._pdh = {"z0": np.array(pos[:n, 2], dtype=np.float64), "hold": np.zeros(n, bool),
                          "f": np.ones(n), "sign": np.zeros(n), "hdg": np.full(n, np.nan),
                          "t0": np.zeros(n, np.int64), "used": np.zeros(n, np.int64), "leg": np.full(n, np.nan)}
        if _MAPSW_STATE.get("map") != 'forest':
            return
        S = SPEED_LIMIT
        t_s = float(T.tick) * DT
        for i in live:
            _gnd = bool(P["z0"][i] < PDH_Z0)
            if _gnd and not PDH_GND:
                continue
            a = out[i]
            v = _vel_of(a)
            yaw_i = float(state[i, 5])
            vh = math.hypot(float(v[0]), float(v[1])) * S
            if vh >= PDH_VMIN:
                P["hdg"][i] = math.atan2(float(v[1]), float(v[0]))
            c = getattr(getattr(self.pilots[i], "p", None), "ctrl", None) if i < len(self.pilots) else None
            ok_mode = (c is not None and getattr(c, "mode", "") == 'search' and not getattr(c, "locked", False)
                       and not getattr(c, "frozen", False) and i != T.confirmer)
            _lg = getattr(c, "_pdh_leg", None) if c is not None else None
            if _lg is not None and _lg[1] > 1.0:
                P["leg"][i] = _lg[0]
            elif not P["hold"][i]:
                P["leg"][i] = np.nan
            _wr = lambda x: float(((x - yaw_i) + np.pi) % (2.0 * np.pi) - np.pi)
            hd = P["hdg"][i]
            err_c = _wr(hd) if np.isfinite(hd) else 0.0
            err_l = _wr(P["leg"][i]) if np.isfinite(P["leg"][i]) else 0.0
            err = err_l if (PDH_LEG and np.isfinite(P["leg"][i])) else err_c
            leg_behind = (not PDH_LEG) or (np.isfinite(P["leg"][i]) and abs(err_l) >= math.radians(PDH_DEG))
            _gnd_ok = (not _gnd) or (t_s <= PDH_GND_TMAX and ((not PDH_GND_CONF) or T.confirmer >= 0))
            if P["hold"][i]:
                end = None
                if not ok_mode or (_gnd and PDH_GND_CONF and T.confirmer < 0):
                    end = "pdh_abort"
                elif abs(err) <= math.radians(PDH_REL) or abs(err_c) <= math.radians(PDH_REL):
                    end = "pdh_rel"
                elif T.tick - P["t0"][i] >= PDH_MAXHOLD or P["used"][i] >= PDH_BUDGET:
                    end = "pdh_timeout"
                if end is not None:
                    P["hold"][i] = False
                    T.stats[end] = T.stats.get(end, 0) + 1
            elif (ok_mode and _gnd_ok and vh >= PDH_VMIN and abs(err_c) >= math.radians(PDH_DEG) and leg_behind and t_s <= PDH_TMAX
                    and float(pos[i, 2]) >= P["z0"][i] - PDH_DZ and P["used"][i] < PDH_BUDGET):
                P["hold"][i] = True
                P["t0"][i] = T.tick
                P["sign"][i] = 1.0 if err >= 0.0 else -1.0
                T.stats["pdh_trig"] = T.stats.get("pdh_trig", 0) + 1
                if _gnd:
                    T.stats["pdh_gnd_trig"] = T.stats.get("pdh_gnd_trig", 0) + 1
            if P["hold"][i]:
                P["f"][i] = max(0.0, P["f"][i] - 1.0 / PDH_RAMP)
                P["used"][i] += 1
                T.stats["pdh_ticks"] = T.stats.get("pdh_ticks", 0) + 1
                if np.isfinite(hd) or np.isfinite(P["leg"][i]):
                    sg = P["sign"][i]
                    if abs(err) < math.radians(PDH_AMB):
                        sg = 1.0 if err >= 0.0 else -1.0
                    elif err * sg < 0.0:
                        err += 2.0 * np.pi * sg
                    P["sign"][i] = sg
                    lead = math.radians(PDH_LEAD)
                    sp = yaw_i + max(-lead, min(lead, err))
                    a[4] = float(((sp + np.pi) % (2.0 * np.pi) - np.pi) / np.pi)
                try:
                    _om = getattr(self.pilots[i].p, "_omb", None)
                    if _om is not None:
                        _om.prog_pos = None
                except Exception:
                    pass
            elif P["f"][i] < 1.0:
                P["f"][i] = min(1.0, P["f"][i] + 1.0 / PDH_RAMP)
            if P["f"][i] < 1.0:
                v[0] *= P["f"][i]
                v[1] *= P["f"][i]
                if P["hold"][i] and v[2] < -PDH_VZ / S:
                    v[2] = -PDH_VZ / S
                _store_vel(a, v)
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
_MSWF_STATE = {"fired": 0, "zmax": -1.0, "zmin": -1.0}
def _mswf_guard(vote, st):
    try:
        zs = st[:, 2]
        zmax = float(_np.max(zs))
        zmin = float(_np.min(zs))
    except Exception:
        return vote
    _MSWF_STATE["zmax"] = zmax
    _MSWF_STATE["zmin"] = zmin
    if (MSWF_Z > 0.0 and vote == 'city' and zmax > MSWF_Z
            and (MSWF_ZLO <= 0.0 or zmin <= MSWF_ZLO)):
        _MSWF_STATE["fired"] = 1
        return 'forest'
    return vote
def _mapsw_classify(obs):
    vote = _mapsw_classify_depth(obs)
    if MZL_Z > 0.0 and MZL_MODE == 'early':
        try:
            st = _np.asarray(obs["state"], _np.float32)
            st = st.reshape(1, -1) if st.ndim == 1 else st.reshape(-1, st.shape[-1])
            if st.shape[0] > 0 and float(_np.max(st[:, 2])) >= MZL_Z:
                _MSWF_STATE["mzl"] = 1 if vote == 'mountain' else 2
                return 'mountain'
        except Exception:
            pass
    return vote
def _mapsw_classify_depth(obs):
    try:
        st = _np.asarray(obs["state"], _np.float32)
        st = st.reshape(1, -1) if st.ndim == 1 else st.reshape(-1, st.shape[-1])
        n = st.shape[0]
        dp = _np.asarray(obs["depth"], _np.float32).reshape(n, DEPTH_RES, DEPTH_RES, 1)
        feats = [_np.asarray(_map_depth_feature_values(dp[i]), _np.float64) for i in range(n)]
        f = _np.mean(feats, axis=0)
        if WD_CLF_ON:
            _WD["f56"] = feats
        if f.shape[0] != _MAPSW_MU.shape[0]:
            return None
        z = (f - _MAPSW_MU) / _MAPSW_SD
        full = _MAPSW_MAPS[int(_np.argmax(z @ _MAPSW_W + _MAPSW_B))]
        if MSWZ_Z > 0.0 and full == 'mountain' and float(_np.max(st[:, 2])) <= MSWZ_Z:
            full = MSWZ_TO
        if P1_MSW and n > 1:
            zs = st[:, 2].astype(_np.float64)
            low = [i for i in range(n) if zs[i] <= float(zs.min()) + 6.0]
            if 0 < len(low) < n and full == 'mountain' and float(zs.min()) <= 5.0:
                fl = _np.mean([feats[i] for i in low], axis=0)
                zl = (fl - _MAPSW_MU) / _MAPSW_SD
                lowvote = _MAPSW_MAPS[int(_np.argmax(zl @ _MAPSW_W + _MAPSW_B))]
                if lowvote in ('forest', 'city'):
                    return _mswf_guard(lowvote, st)
        return _mswf_guard(full, st)
    except Exception:
        return None
WD_CLF_ON = os.environ.get('SWSAR_WD_CLF', '1') == '1'
WD_TAU = float(os.environ.get('SWSAR_WD_TAU', '0.97'))
WD_PAIRS = tuple(x for x in os.environ.get('SWSAR_WD_PAIRS', 'city>mountain,city>forest,city>open,mountain>city').split(',') if x)
WD_RGB = os.environ.get('SWSAR_WD_RGB', '0') == '1'
WD_RQ = int(os.environ.get('SWSAR_WD_RQ', '12'))
WD_RGB_MARGIN = float(os.environ.get('SWSAR_WD_RGB_MARGIN', '0'))
WD_MODEL_NAME = os.environ.get('SWSAR_WD_MODEL', 'wd_clf.json')
WD_GATE = os.environ.get('SWSAR_WD_GATE', '1') == '1'
WD_HCITY = float(os.environ.get('SWSAR_WD_HCITY', '5.5'))
WD_HOPEN = float(os.environ.get('SWSAR_WD_HOPEN', '10.5'))
WD_HFOREST = float(os.environ.get('SWSAR_WD_HFOREST', '22.5'))
WD_HMTN = float(os.environ.get('SWSAR_WD_HMTN', '5.5'))
_WD_NAMES = ('city', 'open', 'mountain', 'village', 'forest')
_WD_CODE = {"city": 0, "open": 1, "mountain": 2, "village": 3, "forest": 4}
_WD = {"arm": 0, "from": -1, "to": -1, "pred": -1, "conf": -1.0, "model": -1, "req": 0, "got": 0, "m56": -1.0,
       "ms": 0.0, "err": 0, "S0": None, "R": None, "reqd": None, "f56": None, "prov": -1, "cand": 0}
_WD_MODEL = None
def _wd_reset():
    _WD.update(arm=0, pred=-1, conf=-1.0, model=-1, req=0, got=0, m56=-1.0, ms=0.0, S0=None, R=None, reqd=None, f56=None,
               prov=-1, cand=0, mask=0, x=None)
    _WD["from"] = -1
    _WD["to"] = -1
def _wd_model():
    global _WD_MODEL
    if _WD_MODEL is None:
        try:
            _WD_MODEL = json.loads((Path(__file__).resolve().parent / WD_MODEL_NAME).read_text())
            for _k in ('d', 'r'):
                if _k in _WD_MODEL:
                    _WD_MODEL[_k]["_np"] = [(int(t['k']), np.asarray(t['f'], np.int64), np.asarray(t['t'], np.float64),
                                            np.asarray(t['l'], np.int64), np.asarray(t['r'], np.int64),
                                            np.asarray(t['v'], np.float64)) for t in _WD_MODEL[_k]['trees']]
        except Exception:
            _WD_MODEL = {}
            _WD["err"] += 1
    return _WD_MODEL
def _wd_predict(m, x):
    F = np.asarray(m['base'], np.float64).copy()
    for k, f, thr, l, r, v in m['_np']:
        nd = 0
        while f[nd] >= 0:
            nd = l[nd] if x[f[nd]] < thr[nd] else r[nd]
        F[k] += v[nd]
    F -= F.max()
    p = np.exp(F)
    return p / p.sum()
def _wd_candidate(lab, zmax0):
    if 'all' not in WD_PAIRS and not any(p.split('>')[0] == lab for p in WD_PAIRS):
        return False
    if not WD_GATE:
        return True
    if lab == 'mountain':
        return zmax0 <= WD_HMTN
    return True
def _wd_consistent(new, zmax0):
    if not WD_GATE:
        return True
    return {'city': zmax0 <= WD_HCITY, 'village': zmax0 <= WD_HCITY, 'open': zmax0 <= WD_HOPEN,
            'forest': zmax0 <= WD_HFOREST, 'mountain': zmax0 > WD_HMTN}.get(new, False)
def _wd_logits56(obs):
    st = _np.asarray(obs["state"], _np.float32)
    st = st.reshape(1, -1) if st.ndim == 1 else st.reshape(-1, st.shape[-1])
    n = st.shape[0]
    dp = _np.asarray(obs["depth"], _np.float32).reshape(n, DEPTH_RES, DEPTH_RES, 1)
    f = _np.mean([_np.asarray(_map_depth_feature_values(dp[i]), _np.float64) for i in range(n)], axis=0)
    return ((f - _MAPSW_MU) / _MAPSW_SD) @ _MAPSW_W + _MAPSW_B
_WD_GRID = None
def _wd_grid():
    global _WD_GRID
    if _WD_GRID is None:
        px = (np.arange(64) * 4 + 2 + 0.5) / 256.0 * 2.0 - 1.0
        u = np.broadcast_to(px[None, :], (64, 64)); v = np.broadcast_to(-px[:, None], (64, 64))
        ub = u.reshape(16, 4, 16, 4).transpose(0, 2, 1, 3).reshape(256, 16)
        vb = v.reshape(16, 4, 16, 4).transpose(0, 2, 1, 3).reshape(256, 16)
        uc = ub.mean(axis=1, keepdims=True); vc = vb.mean(axis=1, keepdims=True)
        A = np.stack([ub[0] - uc[0], vb[0] - vc[0], np.ones(16)], axis=1)
        P = np.linalg.pinv(A)
        _WD_GRID = (A, P, uc[:, 0], vc[:, 0])
    return _WD_GRID
_WD_SLOPE_EDGES = np.array([3.0, 10.0, 20.0, 35.0, 60.0])
def wd_depth_feats(d):
    d = np.asarray(d, np.float64)
    valid = (d > 0.003) & (d < 0.995)
    sky = d >= 0.995
    z = 0.5 + 29.5 * d
    out = [float(sky.mean()), float(sky[:32].mean()), float(sky[32:].mean()), float((valid & (z < 2.0)).mean()),
           float((valid & (z < 6.0)).mean())]
    zv = z[valid]
    out += [float(np.median(zv)) if zv.size else 30.0, float(np.percentile(zv, 90)) if zv.size else 30.0]
    col_sky = sky.cumprod(axis=0).sum(axis=0)
    out += [float(col_sky.mean() / 64.0), float(col_sky.std() / 64.0)]
    A, P, uc, vc = _wd_grid()
    w = 1.0 / z
    wb = w.reshape(16, 4, 16, 4).transpose(0, 2, 1, 3).reshape(256, 16)
    vb = valid.reshape(16, 4, 16, 4).transpose(0, 2, 1, 3).reshape(256, 16).all(axis=1)
    nvb = int(vb.sum())
    out.append(nvb / 256.0)
    if nvb >= 3:
        W = wb[vb]
        coef = W @ P.T
        res = W - coef @ A.T
        rms = np.sqrt((res ** 2).mean(axis=1))
        rel = rms / np.maximum(W.mean(axis=1), 1e-6)
        a, b = coef[:, 0], coef[:, 1]
        c = coef[:, 2] - a * uc[vb] - b * vc[vb]
        nrm = np.sqrt(a * a + b * b + c * c) + 1e-12
        slope = np.degrees(np.arccos(np.clip(np.abs(b) / nrm, 0.0, 1.0)))
        planar = rel < 0.002
        out += [float(np.percentile(rms, 25)), float(np.percentile(rel, 10)), float(np.percentile(rel, 25)),
                float(np.percentile(rel, 50)), float(planar.mean()), float((rel < 0.0005).mean()), float((rel < 0.01).mean())]
        sp = slope[planar] if planar.any() else slope
        h = np.histogram(sp, bins=np.r_[0.0, _WD_SLOPE_EDGES, 90.01])[0] / max(1, sp.size)
        out += [float(x) for x in h]
        out += [float(np.median(slope)), float(np.mean(slope < 3.0)), float(np.mean(slope > 60.0))]
        lo = np.repeat(np.arange(16) >= 8, 16)[vb]
        out += [float(np.median(rel[lo])) if lo.any() else 1.0, float(np.median(slope[lo])) if lo.any() else 90.0]
    else:
        out += [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0] + [0.0] * 6 + [90.0, 0.0, 0.0] + [1.0, 90.0]
    gx = np.abs(np.diff(d, axis=1)); gy = np.abs(np.diff(d, axis=0))
    out += [float(gx.mean() + gy.mean()), float(((gx > 0.05).mean() + (gy > 0.05).mean()) / 2.0)]
    return out
def wd_rgb_feats(rgb, d):
    x = np.asarray(rgb, np.float64) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx = x.max(axis=2); mn = x.min(axis=2)
    s = (mx - mn) / np.maximum(mx, 1e-6)
    pad = (r > 0.45) & (g < 0.15) & (b < 0.15)
    sky = np.asarray(d, np.float64) >= 0.995
    m = (~pad) & (~sky)
    frac = float(m.mean())
    if m.sum() < 20:
        return [frac] + [0.0] * 21
    rr, gg, bb, ss, vv = r[m], g[m], b[m], s[m], mx[m]
    tot = rr + gg + bb + 1e-6
    green = (gg > rr * 1.08) & (gg > bb * 1.08)
    grey = ss < 0.12
    white = (vv > 0.8) & (ss < 0.12)
    dark = vv < 0.25
    brown = (rr >= gg) & (gg >= bb) & (ss > 0.2)
    blue = (bb > rr * 1.05) & (bb > gg * 1.02)
    hue = np.zeros_like(vv)
    dd = np.maximum(vv - np.minimum(np.minimum(rr, gg), bb), 1e-6)
    ir = vv == rr; ig = (vv == gg) & ~ir; ib = ~(ir | ig)
    hue[ir] = ((gg - bb)[ir] / dd[ir]) % 6.0
    hue[ig] = ((bb - rr)[ig] / dd[ig]) + 2.0
    hue[ib] = ((rr - gg)[ib] / dd[ib]) + 4.0
    sat = ss > 0.15
    hh = np.histogram(hue[sat], bins=6, range=(0.0, 6.0))[0] / max(1, int(sat.sum()))
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    gxl = np.abs(np.diff(lum, axis=1))[:, :][m[:, 1:]]
    out = [frac, float(green.mean()), float(grey.mean()), float(white.mean()), float(dark.mean()), float(brown.mean()),
           float(blue.mean()), float((rr / tot).mean()), float((gg / tot).mean()), float(vv.mean()), float(vv.std()),
           float(ss.mean()), float(gxl.mean()) if gxl.size else 0.0, float(sat.mean())]
    out += [float(v) for v in hh]
    out += [float(np.median(ss)), float(np.percentile(vv, 90))]
    return out
def wd_state_feats(S0, S15):
    S0 = np.asarray(S0, np.float64); S15 = np.asarray(S15, np.float64)
    n = S0.shape[0]
    z0 = S0[:, 2]; xy = S0[:, 0:2]
    cen = xy.mean(axis=0)
    dcen = np.hypot(*(xy - cen).T)
    pair = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    clue = np.hypot(S0[:, 163], S0[:, 164])
    dz15 = S15[:, 2] - z0
    return [float(n), float(z0.min()), float(z0.max()), float(z0.mean()), float(z0.std()), float(np.median(z0)),
            float((z0 > 12.0).mean()), float((z0 < 2.0).mean()), float(np.abs(xy).max()), float(np.hypot(*cen)),
            float(dcen.max()), float(pair.max()), float(clue.mean()), float(dz15.mean()), float(S15[:, 162].mean() * 20.0)]
def wd_team_feats(S0, S15, D15, R13, L):
    n = S0.shape[0]
    Dq = np.asarray(D15, np.float32).astype(np.float16).astype(np.float64)
    pd = np.asarray([wd_depth_feats(Dq[i]) for i in range(n)])
    f = wd_state_feats(S0, S15)
    f += list(pd.mean(axis=0)) + list(pd.min(axis=0)) + list(pd.max(axis=0))
    if R13 is not None:
        pr = np.asarray([wd_rgb_feats(R13[i], Dq[i]) for i in range(n)])
        f += list(pr.mean(axis=0)) + list(pr.min(axis=0)) + list(pr.max(axis=0))
    L = np.asarray(L, np.float64)
    Ls = L - L.max()
    p = np.exp(Ls) / np.exp(Ls).sum()
    srt = np.sort(L)
    f += list(L) + list(p) + [float(srt[-1] - srt[-2])]
    return np.asarray(f, np.float64)
WD_DEPTH_N, WD_RGB_N, WD_STATE_N = 30, 22, 15
def _wd_decide(observation):
    t0 = time.perf_counter()
    try:
        m = _wd_model()
        lab = _MAPSW_STATE.get("map")
        f56 = _WD.get("f56")
        if not m or lab is None or f56 is None or _WD.get("S0") is None:
            return
        st = np.asarray(observation["state"], np.float32)
        st = st.reshape(1, -1) if st.ndim == 1 else st.reshape(-1, st.shape[-1])
        n = st.shape[0]
        if _WD["S0"].shape[0] != n:
            return
        zmax0 = float(np.max(_WD["S0"][:, 2]))
        if not _wd_candidate(lab, zmax0):
            return
        _WD["cand"] = 1
        dp = np.asarray(observation["depth"], np.float32).reshape(n, DEPTH_RES, DEPTH_RES)[:, 2::4, 2::4]
        f = np.mean(np.asarray(f56, np.float32).astype(np.float64), axis=0)
        L = [round(float(v), 4) for v in (((f - _MAPSW_MU) / _MAPSW_SD) @ _MAPSW_W + _MAPSW_B)]
        R = _WD.get("R")
        use_r = 'r' in m and R is not None and R.shape[0] == n
        key = 'r' if use_r else ('d' if 'd' in m else None)
        if key is None:
            return
        x = wd_team_feats(_WD["S0"][:, :214], st[:, :214], dp, R if use_r else None, L)
        _WD["x"] = x
        p = _wd_predict(m[key], x)
        k = int(np.argmax(p))
        _WD.update(model=1 if use_r else 0, pred=k, conf=float(p[k]))
        new = _WD_NAMES[k]
        if (new != lab and float(p[k]) >= WD_TAU and _wd_consistent(new, zmax0)
                and ('all' in WD_PAIRS or f'{lab}>{new}' in WD_PAIRS)):
            _WD["from"] = _WD_CODE.get(lab, -1)
            _WD["to"] = k
            _WD["arm"] = 1
            _MAPSW_STATE["map"] = new
    except Exception:
        _WD["err"] += 1
    finally:
        _WD["ms"] = (time.perf_counter() - t0) * 1e3
_MapswBase = DroneFlightController
class DroneFlightController(_MapswBase):
    def reset(self):
        _MAPSW_STATE["map"] = None
        _MSWF_STATE["fired"] = 0
        _MSWF_STATE["zmax"] = -1.0
        _MSWF_STATE["zmin"] = -1.0
        _MSWF_STATE["mzl"] = 0
        _flte_reset()
        self._mapsw_tick = 0
        return _MapswBase.reset(self)
    def act(self, observation):
        t = getattr(self, "_mapsw_tick", 0)
        if t == _MAPSW_TICK and _MAPSW_STATE["map"] is None:
            _MAPSW_STATE["map"] = _mapsw_classify(observation)
        if WD_CLF_ON and t == _MAPSW_TICK and _MAPSW_STATE["map"] is not None and _WD["pred"] < 0:
            _wd_decide(observation)
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
    try:
        _c = team.pilots[confirmer].p.ctrl
        if _c.mtn.is_mountain:
            _FLTE_STATE["u66_ticks"] = _FLTE_STATE.get("u66_ticks", 0.0) + 1.0
            _band = False
            try:
                _dze = float(pos[confirmer, 2]) - float(_c.locked_head_z)
                _spd = float(np.linalg.norm(np.asarray(state[confirmer, 6:9], dtype=np.float64)))
                _band = bool((_c.locked or _c.frozen) and _spd <= 1.0 and 2.0 <= _dze <= 4.0)
            except Exception:
                _band = False
            _bd = _FLTE_STATE.setdefault("band", {})
            _bd[confirmer] = (_bd.get(confirmer, 0) + 1) if _band else 0
            if _c.locked or _c.frozen:
                _sh = float(np.hypot(float(_c.locked_vic[0]) - float(est[0]), float(_c.locked_vic[1]) - float(est[1])))
                if _sh > FLTE_U66_MIN:
                    _FLTE_STATE["u66_would"] = _FLTE_STATE.get("u66_would", 0.0) + 1.0
                    _ev = [float(getattr(T, "tick", -1)) * DT, float(_c.locked_vic[0]), float(_c.locked_vic[1]),
                           float(est[0]), float(est[1])]
                    if "u66_first" not in _FLTE_STATE:
                        _FLTE_STATE["u66_first"] = _ev
                    _FLTE_STATE["u66_last"] = _ev
                    if FLTE_U66 and not _flte_z0_ok():
                        _FLTE_STATE["z0_block"] = _FLTE_STATE.get("z0_block", 0.0) + 1.0
                    elif FLTE_U66 and FLTE_DWELL > 0 and _band and _bd[confirmer] < FLTE_DWELL:
                        _FLTE_STATE["u66_dwell_skip"] = _FLTE_STATE.get("u66_dwell_skip", 0.0) + 1.0
                    elif FLTE_U66:
                        _c.locked_vic = np.array([float(est[0]), float(est[1]), float(_c.locked_vic[2])], dtype=np.float64)
                        if getattr(_c, "_fix_pt", None) is not None:
                            _fp = np.array(_c._fix_pt, dtype=np.float64).copy()
                            _fp[0], _fp[1] = float(est[0]), float(est[1])
                            _c._fix_pt = _fp
                        if FLTE_ZRESET:
                            _c._hov_z_lock = None
                        _FLTE_STATE["u66_latch"] = _FLTE_STATE.get("u66_latch", 0.0) + 1.0
                        _FLTE_STATE["u66_shift"] = max(_FLTE_STATE.get("u66_shift", 0.0), _sh)
    except Exception:
        pass
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
        team = getattr(self, "team", None)
        if team is None:
            return {}
        try:
            T = team.T
            out = {"forest": 1.0 if (_forest() or _skin_world()) else 0.0, "confirmer": float(T.confirmer),
                   "pilot_exceptions": float(T.stats.get("pilot_exceptions", 0)),
                   "deadline_hits": float(T.stats.get("deadline_hits", 0)),
                   "pdh_trig": float(T.stats.get("pdh_trig", 0)), "pdh_ticks": float(T.stats.get("pdh_ticks", 0)),
                   "pdh_gnd_trig": float(T.stats.get("pdh_gnd_trig", 0)),
                   "pdh_rel": float(T.stats.get("pdh_rel", 0)), "pdh_timeout": float(T.stats.get("pdh_timeout", 0)),
                   "pdh_abort": float(T.stats.get("pdh_abort", 0)),
                   "pdh_tkc": float(sum(float((getattr(getattr(getattr(_pp, "p", None), "ctrl", None), "mq_stats", None) or {}).get("pdh_tkc", 0))
                                        for _pp in team.pilots)),
                   "mss_on": 1.0 if MSS_ON else 0.0,
                   **_mpl_dbg(),
                   **_fpp_dbg(),
                   **_skl_dbg(),
                   **_veg_dbg(),
                   **_vc_dbg(),
                   "hoc_handover": float(T.stats.get("hoc_handover", 0)),
                   "fl_on": 1.0 if FLTE_ON else 0.0,
                   "fl_u66_ticks": float(_FLTE_STATE.get("u66_ticks", 0.0)),
                   "fl_u66_would": float(_FLTE_STATE.get("u66_would", 0.0)),
                   "fl_u66_latch": float(_FLTE_STATE.get("u66_latch", 0.0)),
                   "fl_u66_shift": float(_FLTE_STATE.get("u66_shift", 0.0)),
                   "fl_u66_dwell_skip": float(_FLTE_STATE.get("u66_dwell_skip", 0.0)),
                   **{"fl_u66_%s%s" % (_fk[4], _nm): float(_FLTE_STATE.get(_fk, [-1.0, 0.0, 0.0, 0.0, 0.0])[_ix])
                      for _fk in ("u66_first", "u66_last")
                      for _ix, _nm in enumerate(("t", "lx", "ly", "ex", "ey"))},
                   "fl_pdn_would": float(_FLTE_STATE.get("pdn_would", 0.0)),
                   "fl_pdn_skip": float(_FLTE_STATE.get("pdn_skip", 0.0)),
                   "fl_pdn_short": float(_FLTE_STATE.get("pdn_short", 0.0)),
                   "fl_z0": float(_FLTE_STATE.get("z0", -1.0)),
                   "fl_z0_block": float(_FLTE_STATE.get("z0_block", 0.0)),
                   **_mcs2g_dbg(T),
                   "mtyp_on": 1.0 if MTYP_ON else 0.0,
                   "mtyp_mode": float({"relatch": 1, "support": 2, "none": 0}.get(MTYP_MODE, -1)) if MTYP_ON else 0.0,
                   "mtyp_relabel": float(T.stats.get("mtyp_relabel", 0)),
                   "mtyp_relatch": float(T.stats.get("mtyp_relatch", 0)),
                   "mtyp_union": float(T.stats.get("mtyp_union", 0)),
                   "mtyp_err": float(T.stats.get("mtyp_err", 0)),
                   "mtyp_pmtn": float(T.stats.get("mtyp_pmtn", -1.0)),
                   "meg_on": 1.0 if MEG_ON else 0.0,
                   "meg_c_on": 1.0 if (MEG_ON and MEG_C) else 0.0,
                   "meg_fast": _meg_dbg_sum(team, "meg_fast"),
                   "meg_glide": _meg_dbg_sum(team, "meg_glide"),
                   "meg_floor": _meg_dbg_sum(team, "meg_floor"),
                   "meg_frz": _meg_dbg_sum(team, "meg_frz"),
                   "meg_rc": _meg_dbg_sum(team, "meg_rc"),
                   "meg_rc_up": _meg_dbg_sum(team, "meg_rc_up"),
                   "meg_rc_dn": _meg_dbg_sum(team, "meg_rc_dn"),
                   "mss_fires": float(T.stats.get("mss_fires", 0)),
                   "mss_ticks": float(T.stats.get("mss_ticks", 0)),
                   "mss_eng_ticks": float(T.stats.get("mss_eng_ticks", 0)),
                   "mss_drones": float(T.stats.get("mss_drones", 0)),
                   "mswf_armed": 1.0 if MSWF_Z > 0.0 else 0.0,
                   "mswf_fired": float(_MSWF_STATE.get("fired", 0)),
                   "mswf_zmax": float(_MSWF_STATE.get("zmax", -1.0)),
                   "mswf_zmin": float(_MSWF_STATE.get("zmin", -1.0)),
                   "mtn_recon": float(T.stats.get("mtn_recon", 0)),
                   "mtn_recon_t": float(T.stats.get("mtn_recon_t", -1.0)),
                   "mapsw_code": float({"city": 0, "open": 1, "mountain": 2, "village": 3, "forest": 4}.get(_MAPSW_STATE.get("map"), -1))}
            nan = float("nan")
            seen_t = getattr(self, "_dbg_det_t", None)
            if seen_t is None:
                seen_t = self._dbg_det_t = {}
            n = max(1, min(int(T.n), len(team.pilots)))
            agg = {k: 0.0 for k in _SPIN_DBG_KEYS}
            mq = {"mq_refute": 0.0, "mq_hold": 0.0, "mq_protect": 0.0, "msw_giveup": 0.0, "mq_approach": 0.0, "mtk_full": 0.0, "ma_low": 0.0, "mrgb_check": 0.0, "msp_turns": 0.0, "msp_ticks": 0.0, "msp_done": 0.0, "msp_stop_seen": 0.0, "msp_stop_terrain": 0.0, "msp_stop_mode": 0.0, "fsp_ticks": 0.0, "vcap": 0.0, "vfc_bar": 0.0, "fix_set": 0.0, "fix_freeze": 0.0, "fix_noblack": 0.0, "fix_denyskip": 0.0, "tk_hold": 0.0, "tkalign": 0.0, "tk_stop": 0.0, "tk_turns": 0.0, "fa_low": 0.0, "fa_hold": 0.0, "fix_rehover": 0.0, "tks_start": 0.0, "tks_ticks": 0.0, "tks_hold": 0.0, "tks_done": 0.0, "tks_stop_busy": 0.0, "tks_verify": 0.0, "tks_verify_ticks": 0.0, "tks_verify_lost": 0.0, "tks_verify_timeout": 0.0, "tks_timeout": 0.0, "tks_verify_commit": 0.0, "fhs_ticks": 0.0}
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
                if PDH_ON:
                    _P = getattr(T, "_pdh", None)
                    if _P is not None and k < _P["z0"].shape[0]:
                        out["pdh%d" % k] = [float(_P["hold"][k]), float(_P["f"][k]),
                                            float(np.degrees(_P["hdg"][k])) if np.isfinite(_P["hdg"][k]) else nan,
                                            float(_P["sign"][k]),
                                            float(np.degrees(_P["leg"][k])) if np.isfinite(_P["leg"][k]) else nan]
                    if PDH_DIAG:
                        _dd = getattr(c, "_pdh_dbg", None)
                        out["pdhd%d" % k] = [float(x) for x in _dd] if _dd is not None else [nan] * 6
                mq["fix_lifted"] = mq.get("fix_lifted", 0.0) + float(getattr(c.detector, "fix_lifted", 0))
                _ms = getattr(c, "mq_stats", None) or {}
                for kk in ("mcv_started", "mcv_ticks", "tks_stall", "tel_lockguard", "lockjump_held", "mbt_ticks",
                           "ltd_lift"):
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
_SARX_GEO = os.environ.get('SARX_GEO', '1') == '1'
_SARX_GEO_M = float(os.environ.get('SARX_GEO_M', '4.0'))
_SARX_GEO_MTN_ONLY = os.environ.get('SARX_GEO_MTN_ONLY', '1') == '1'
_SARX_CLF_MODE = os.environ.get('SARX_CLF_MODE', 'geoall')
_SARX_BLG_MODE = os.environ.get('SARX_BLG_MODE', 'win')
_SARX_BLG_V = float(os.environ.get('SARX_BLG_V', '0.3'))
_SARX_BLG_A0 = float(os.environ.get('SARX_BLG_A0', '45'))
_SARX_BLG_A1 = float(os.environ.get('SARX_BLG_A1', '90'))
_SARX_TRAIL = os.environ.get('SARX_TRAIL', '1') == '1'
_SARX_TRAIL_SEC = float(os.environ.get('SARX_TRAIL_SEC', '12.0'))
_SARX_TRAIL_LOOK = float(os.environ.get('SARX_TRAIL_LOOK', '2.5'))
_SARX_TRAIL_R = float(os.environ.get('SARX_TRAIL_R', '1.0'))
_SARX_CLF_D = float(os.environ.get('SARX_CLF_D', '7.0'))
_SARX_CLF_MIN = float(os.environ.get('SARX_CLF_MIN', '2.0'))
_SARX_CLF_VZ = float(os.environ.get('SARX_CLF_VZ', '2.0'))
_SARX_WIN_POST = int(os.environ.get('SARX_WIN_POST', '300'))
_SARX_WIN_PAD_R = float(os.environ.get('SARX_WIN_PAD_R', '12.0'))
_SARX_WIN_MAX = int(os.environ.get('SARX_WIN_MAX', '600'))
_SARX_ODT = os.environ.get('SARX_ODT', '0') == '1'
_SARX_ODT_M = float(os.environ.get('SARX_ODT_M', '32.0'))
_SARX_STATS = os.environ.get('SARX_STATS', '1') == '1'
_SARX_MQ_KEYS = ('ltd_lift', 'ma_low', 'msw_giveup', 'fix_noblack', 'fix_rehover', 'fix_freeze',
                 'msp_ticks', 'mcv_ticks', 'sarx_hgu', 'sarx_mtky', 'meg_glide', 'meg_frz', 'meg_floor', 'meg_fast', 'hsd_onv', 'ofv_start', 'ofv_ticks', 'tka_sign', 'fa_low', 'fhs_ticks')
def _sarx_pilot(team, i):
    try:
        return team.pilots[i].p
    except Exception:
        return None
def _sarx_mtn(cc):
    try:
        if bool(cc.mtn.is_mountain):
            return True
    except Exception:
        pass
    return _MAPSW_STATE.get("map") == 'mountain'
def _sarx_state(observation):
    st_ = np.asarray(observation["state"], dtype=np.float64)
    return st_.reshape(1, -1) if st_.ndim == 1 else st_.reshape(-1, st_.shape[-1])
def _sarx_geofence(ctrl):
    team = getattr(ctrl, "team", None)
    T = getattr(team, "T", None)
    if T is None or not getattr(T, "region_latched", False):
        return
    n = int(getattr(T, "n", 0))
    if n <= 0:
        return
    if _SARX_GEO_MTN_ONLY and not (bool(getattr(T, "mtn", False)) or _MAPSW_STATE.get("map") == 'mountain'):
        return
    c = np.asarray(T.clue_world, dtype=np.float64)[:2]
    r = 80.0 * math.sqrt(min(max(n, 1), 8) / 8.0) + _SARX_GEO_M
    def out(xy):
        xy = np.asarray(xy, dtype=np.float64)
        return bool(np.all(np.isfinite(xy[:2]))) and float(math.hypot(xy[0] - c[0], xy[1] - c[1])) > r
    st = T.stats
    for i in range(min(n, len(team.pilots))):
        p = _sarx_pilot(team, i)
        cc = getattr(p, "ctrl", None)
        if cc is None:
            continue
        det = cc.detector
        if (cc.locked or cc.frozen) and out(cc.locked_vic):
            det.blacklist.append(np.asarray(cc.locked_vic, dtype=np.float64).copy())
            p._drop_lock()
            st["sarx_geo_lock"] = st.get("sarx_geo_lock", 0) + 1
            continue
        if float(det.strong) > 0.0 and out(det.vic):
            det.strong = 0.0
            det.conf = 0.0
            det.seen = False
            st["sarx_geo_trk"] = st.get("sarx_geo_trk", 0) + 1
        if float(det.cand_strong) > 0.0 and out(det.cand):
            det.cand_strong = 0.0
    if T.cand is not None and out(T.cand["xy"]):
        if T.confirmer >= 0:
            T._release("geofence", blame=False)
        T._drop_cand()
        T.cand = None
        st["sarx_geo_cand"] = st.get("sarx_geo_cand", 0) + 1
    T.records = [rec for rec in T.records if not out(rec.get("xy", [np.nan, np.nan]))]
def _sarx_odt(ctrl, observation):
    team = getattr(ctrl, "team", None)
    T = getattr(team, "T", None)
    if T is None or not getattr(T, "region_latched", False):
        return
    st_ = _sarx_state(observation)
    n = min(st_.shape[0], len(team.pilots))
    c = np.asarray(T.clue_world, dtype=np.float64)[:2]
    lim = 80.0 * math.sqrt(min(max(n, 1), 8) / 8.0) + _SARX_ODT_M
    for i in range(n):
        cc = getattr(_sarx_pilot(team, i), "ctrl", None)
        if cc is None or getattr(cc, "_msp_turned", None) is not None or not bool(cc.mtn.is_mountain):
            continue
        if float(math.hypot(st_[i, 0] - c[0], st_[i, 1] - c[1])) > lim:
            cc._msp_last_end = int(cc.tick) - MSP_PERIOD + 2
            T.stats["sarx_odt"] = T.stats.get("sarx_odt", 0) + 1
def _sarx_on_trail(ctrl, i, pos, vdir):
    tr = getattr(ctrl, "_sarx_trail", None)
    if tr is None or i >= len(tr) or len(tr[i]) < 3:
        return False
    pts = np.asarray(tr[i], dtype=np.float64)
    for dist in (0.5 * _SARX_TRAIL_LOOK, _SARX_TRAIL_LOOK):
        q = pos + vdir * dist
        d = np.sqrt(((pts - q) ** 2).sum(axis=1))
        if float(d.min()) > _SARX_TRAIL_R:
            return False
    return True
def _sarx_window(ctrl, t, i, pos):
    if t > _SARX_WIN_MAX:
        return False
    ex = ctrl._sarx_exit[i]
    if ex is None or t <= ex + _SARX_WIN_POST:
        return True
    s = ctrl._sarx_starts[i]
    return float(math.hypot(pos[0] - s[0], pos[1] - s[1])) <= _SARX_WIN_PAD_R
_SARX_CLF_GEO = _SARX_CLF_MODE in ('geo', 'geoall')
if _SARX_CLF_GEO:
    _SARX_CLF_MODE = 'win' if _SARX_CLF_MODE == 'geo' else 'always'
_GCL_A = float(os.environ.get('SARZ_GCL_A', '6.0'))
_GCL_W = float(os.environ.get('SARZ_GCL_W', '1.5'))
_GCL_CLR = float(os.environ.get('SARZ_GCL_CLR', '0.8'))
_GCL_MARG = math.radians(float(os.environ.get('SARZ_GCL_MARG', '8.0')))
_GCL_R = np.arange(48, 256, 2)
_GCL_C = np.arange(48, 208, 2)
_GCL_U = ((_GCL_C + 0.5) / 128.0 - 1.0)[None, :] * np.ones((len(_GCL_R), 1))
_GCL_V = (1.0 - (_GCL_R + 0.5) / 128.0)[:, None] * np.ones((1, len(_GCL_C)))
def _gcl_cap(v, depth_i, pos, rpy):
    vh = float(math.hypot(v[0], v[1]))
    if vh < 0.3:
        return v, False
    m = np.array([v[0] / vh, v[1] / vh])
    d = np.asarray(depth_i, dtype=np.float32)[np.ix_(_GCL_R, _GCL_C)].astype(np.float64)
    d = d * (DEPTH_MAX_M - DEPTH_MIN_M) + DEPTH_MIN_M
    ok = d < DEPTH_MAX_M - 0.5
    if not ok.any():
        return v, False
    fwd, up, right = camera_axes(rpy)
    cam = np.asarray(pos, dtype=np.float64) + fwd * CAMERA_OFFSET_M + up * CAMERA_UP_OFFSET_M
    dz, uu, vv = d[ok], _GCL_U[ok], _GCL_V[ok]
    R = (cam - np.asarray(pos, dtype=np.float64))[None, :] + dz[:, None] * (
        fwd[None, :] + uu[:, None] * right[None, :] * HALF_TAN + vv[:, None] * up[None, :] * HALF_TAN)
    a = R[:, 0] * m[0] + R[:, 1] * m[1]
    c = R[:, 0] * m[1] - R[:, 1] * m[0]
    sel = (a > 0.3) & (a < _GCL_A) & (np.abs(c) < _GCL_W)
    if not sel.any():
        return v, False
    need = float(np.max(np.arctan2(R[sel, 2] + _GCL_CLR, a[sel]))) + _GCL_MARG
    gam = math.atan2(float(v[2]), vh)
    if gam >= need:
        return v, False
    sp = max(float(np.linalg.norm(v)), 1.0)
    need = min(need, math.radians(89.0))
    out = np.array([m[0] * sp * math.cos(need), m[1] * sp * math.cos(need), sp * math.sin(need)])
    return out, True
def _sarx_motion_guards(ctrl, action, observation):
    team = getattr(ctrl, "team", None)
    T = getattr(team, "T", None)
    st_ = _sarx_state(observation)
    n = st_.shape[0]
    k = int(getattr(ctrl, "_sarx_tick", 0))
    ctrl._sarx_tick = k + 1
    if getattr(ctrl, "_sarx_starts", None) is None or len(ctrl._sarx_starts) != n:
        ctrl._sarx_starts = st_[:, 0:3].copy()
        ctrl._sarx_exit = [None] * n
        ctrl._sarx_trail = [[] for _ in range(n)]
    if _SARX_TRAIL and k % 10 == 0:
        keep = int(_SARX_TRAIL_SEC * CTRL_HZ / 10)
        for j in range(n):
            tr = ctrl._sarx_trail[j]
            tr.append(st_[j, 0:3].copy())
            if len(tr) > keep:
                del tr[0]
    if T is None:
        return action
    t = int(getattr(T, "tick", k))
    conf = int(getattr(T, "confirmer", -1))
    stats = T.stats
    depth = None
    for i in range(min(n, action.shape[0], len(team.pilots))):
        cc = getattr(_sarx_pilot(team, i), "ctrl", None)
        if cc is None:
            continue
        mode = getattr(cc, "mode", "search")
        if ctrl._sarx_exit[i] is None and mode != 'initial':
            ctrl._sarx_exit[i] = t
        if mode == 'hover' or not _sarx_mtn(cc):
            continue
        pos = st_[i, 0:3]
        win = _sarx_window(ctrl, t, i, pos)
        do_clf = _SARX_CLF_MODE == 'always' or (_SARX_CLF_MODE == 'win' and win)
        do_blg = _SARX_BLG_MODE == 'always' or (_SARX_BLG_MODE == 'win' and win)
        if not (do_clf or do_blg):
            continue
        a = action[i]
        v = np.asarray(a[0:3], dtype=np.float64) * float(a[3]) * SPEED_LIMIT
        yaw = float(st_[i, 5])
        fwd = np.array([math.cos(yaw), math.sin(yaw)])
        changed = False
        if do_clf and i != conf and _SARX_CLF_GEO:
            near_lock = False
            if cc.locked or cc.frozen:
                lv = np.asarray(cc.locked_vic, dtype=np.float64)
                near_lock = float(math.hypot(lv[0] - pos[0], lv[1] - pos[1])) <= MA_R
            if not near_lock:
                if depth is None:
                    depth = np.asarray(observation["depth"], dtype=np.float32).reshape(n, DEPTH_RES, DEPTH_RES)
                v2, hit = _gcl_cap(v, depth[i], pos, st_[i, 3:6])
                if hit:
                    v = v2
                    changed = True
                    stats["sarx_clf"] = stats.get("sarx_clf", 0) + 1
        elif do_clf and i != conf:
            comp = float(v[0] * fwd[0] + v[1] * fwd[1])
            near_lock = False
            if cc.locked or cc.frozen:
                lv = np.asarray(cc.locked_vic, dtype=np.float64)
                near_lock = float(math.hypot(lv[0] - pos[0], lv[1] - pos[1])) <= MA_R
            if comp > 0.0 and not near_lock:
                if depth is None:
                    depth = np.asarray(observation["depth"], dtype=np.float32).reshape(n, DEPTH_RES, DEPTH_RES)
                try:
                    near = float(min(cc.avoider.sectors(depth[i])))
                except Exception:
                    near = 99.0
                if near < _SARX_CLF_D:
                    s = float(np.clip((near - _SARX_CLF_MIN) / max(_SARX_CLF_D - _SARX_CLF_MIN, 1e-3), 0.0, 1.0))
                    v[0:2] -= (1.0 - s) * comp * fwd
                    v[2] = max(v[2], _SARX_CLF_VZ * (1.0 - s))
                    changed = True
                    stats["sarx_clf"] = stats.get("sarx_clf", 0) + 1
        if do_blg:
            vh = float(math.hypot(v[0], v[1]))
            if vh > _SARX_BLG_V and not (_SARX_TRAIL and _sarx_on_trail(
                    ctrl, i, pos, v / max(float(np.linalg.norm(v)), 1e-9))):
                mis = abs((math.atan2(v[1], v[0]) - yaw + math.pi) % (2.0 * math.pi) - math.pi)
                f = float(np.clip((math.degrees(mis) - _SARX_BLG_A0) / max(_SARX_BLG_A1 - _SARX_BLG_A0, 1e-3), 0.0, 1.0))
                if f > 0.0:
                    v[0:2] *= (vh * (1.0 - f) + _SARX_BLG_V * f) / vh
                    changed = True
                    stats["sarx_blg"] = stats.get("sarx_blg", 0) + 1
        if changed:
            stats["sarx_win_ticks" if win else "sarx_mid_ticks"] = stats.get("sarx_win_ticks" if win else "sarx_mid_ticks", 0) + 1
            nv = float(np.linalg.norm(v))
            if nv > 1e-9:
                action[i, 0:3] = (v / nv).astype(np.float32)
                action[i, 3] = np.float32(min(nv / SPEED_LIMIT, 1.0))
            else:
                action[i, 0:4] = 0.0
    return action
def _sarx_bookkeep(ctrl):
    team = getattr(ctrl, "team", None)
    T = getattr(team, "T", None)
    if T is None:
        return
    n = min(int(getattr(T, "n", 0)), len(team.pilots))
    prev = getattr(ctrl, "_sarx_lockprev", None)
    if prev is None or len(prev) != n:
        prev = ctrl._sarx_lockprev = [False] * n
    tot = dict.fromkeys(_SARX_MQ_KEYS, 0.0)
    for i in range(n):
        cc = getattr(_sarx_pilot(team, i), "ctrl", None)
        if cc is None:
            continue
        lk = bool(cc.locked)
        if lk and not prev[i]:
            cc._sarx_lock_t = int(getattr(cc.detector, "t", 0))
        prev[i] = lk
        if _SARX_STATS:
            ms = getattr(cc, "mq_stats", None) or {}
            for kk in _SARX_MQ_KEYS:
                tot[kk] += float(ms.get(kk, 0))
    if _SARX_STATS:
        for kk, vv in tot.items():
            T.stats["sarx_mq_" + kk] = vv
        T.stats["sarx_mtn"] = 1.0 if getattr(T, "mtn", False) else 0.0
        T.stats["sarx_recon"] = float(T.stats.get("mtn_recon", 0))
        T.stats["sarx_mapsw_mtn"] = 1.0 if _MAPSW_STATE.get("map") == 'mountain' else 0.0
        T.stats["sarx_mzl"] = float(_MSWF_STATE.get("mzl", 0))
        T.stats["sarz_padv"] = float(_MCS2G.get("padv", 0))
        for kk, vv in _RGBM.items():
            T.stats["sarz_rgbm_" + kk] = float(vv)
        try:
            for kk in ("plans", "nopath", "ms_max", "ticks"):
                T.stats["sarf_fpp_" + kk] = float(_FPP_STATS.get(kk, 0))
            for kk, vv in _APZ.items():
                T.stats["sarf_apz_" + kk] = float(vv)
            T.stats["sarf_omb_stuck"] = float(sum(float((getattr(getattr(_sarx_pilot(team, i), "_omb", None), "stats", None)
                                                        or {}).get("omb_stuck", 0)) for i in range(n)))
        except Exception:
            pass
_SARX_BASE = DroneFlightController
class DroneFlightController(_SARX_BASE):
    def reset(self):
        self._sarx_trail = None
        self._sarx_tick = 0
        self._sarx_starts = None
        self._sarx_exit = None
        self._sarx_lockprev = None
        return _SARX_BASE.reset(self)
    def act(self, observation):
        if _SARX_GEO:
            try:
                _sarx_geofence(self)
            except Exception as exc:
                _log_once("sarx_geo", "geofence failed", exc)
        if _SARX_ODT:
            try:
                _sarx_odt(self, observation)
            except Exception as exc:
                _log_once("sarx_odt", "odt failed", exc)
        action = _SARX_BASE.act(self, observation)
        if _SARX_CLF_MODE != 'off' or _SARX_BLG_MODE != 'off':
            try:
                action = _sarx_motion_guards(self, np.array(action, dtype=np.float32, copy=True), observation)
            except Exception as exc:
                _log_once("sarx_mg", "motion guards failed", exc)
        try:
            _sarx_bookkeep(self)
        except Exception as exc:
            _log_once("sarx_bk", "bookkeeping failed", exc)
        return action
_CYW_ON = os.environ.get('SARY_CYW', '1') == '1'
_CYW_CELL = float(os.environ.get('SARY_CYW_CELL', '2.0'))
_CYW_RMAX = float(os.environ.get('SARY_CYW_RMAX', '25.0'))
_CYW_RMIN = float(os.environ.get('SARY_CYW_RMIN', '4.0'))
_CYW_HALF = float(os.environ.get('SARY_CYW_HALF', '40.0'))
_CYW_HYST = float(os.environ.get('SARY_CYW_HYST', '1.3'))
_CYW_FWD_BONUS = float(os.environ.get('SARY_CYW_FWD', '0.15'))
_CYW_MIN_FRAC = float(os.environ.get('SARY_CYW_MINF', '0.15'))
_CYW_VBLIND = float(os.environ.get('SARY_CYW_VBLIND', '1.0'))
_CYW_BLIND_A0 = float(os.environ.get('SARY_CYW_BLIND0', '30.0'))
_CYW_BLIND_A1 = float(os.environ.get('SARY_CYW_BLIND1', '45.0'))
_CYW_CLEAR_M = float(os.environ.get('SARY_CYW_CLEAR', '2.5'))
_CYW_HOLD = int(os.environ.get('SARY_CYW_HOLD', '100'))
_CYW_TRK = os.environ.get('SARY_CYW_TRK', '1') == '1'
_CYW_OCC = os.environ.get('SARY_CYW_OCC', '1') == '1'
_CYW_STRIDE = int(os.environ.get('SARY_CYW_STRIDE', '8'))
_CYW_T0 = int(os.environ.get('SARY_CYW_T0', '0'))
_CYW_MAXOFF = float(os.environ.get('SARY_CYW_MAXOFF', '40.0'))
_CYW_MAXOFF_V = float(os.environ.get('SARY_CYW_MAXOFF_V', '0.5'))
_CYW_PD_R = np.array([0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 31.0])
_CYW_PD_P = np.array([0.95, 0.95, 0.85, 0.78, 0.60, 0.45, 0.0])
_CYW_NB = 72
_CYW_HW = min(max(int(round(_CYW_HALF / (360.0 / _CYW_NB))), 1), (_CYW_NB - 1) // 2)
_CYW_KER = 1.0 - np.abs(np.arange(-_CYW_HW, _CYW_HW + 1)) / (_CYW_HW + 1.0)
_CYW_CENTRES = -math.pi + (np.arange(_CYW_NB) + 0.5) * (2.0 * math.pi / _CYW_NB)
_cyw_r = np.arange(_CYW_STRIDE // 2, DEPTH_RES, _CYW_STRIDE)
_CYW_UU, _CYW_VV = np.meshgrid(2.0 * (_cyw_r + 0.5) / DEPTH_RES - 1.0, 1.0 - 2.0 * (_cyw_r + 0.5) / DEPTH_RES)
def _cyw_pd(r):
    return np.interp(r, _CYW_PD_R, _CYW_PD_P)
def _cyw_wrap(x):
    return (x + math.pi) % (2.0 * math.pi) - math.pi
def _cyw_init(ctrl, T, n, st_):
    c = np.asarray(T.clue_world, dtype=np.float64)[:2]
    R = 80.0 * math.sqrt(min(max(n, 1), 8) / 8.0)
    half = R + _CYW_RMAX + 10.0
    m = int(math.ceil(2.0 * half / _CYW_CELL))
    x0, y0 = c[0] - half, c[1] - half
    gx, gy = np.meshgrid(x0 + (np.arange(m) + 0.5) * _CYW_CELL, y0 + (np.arange(m) + 0.5) * _CYW_CELL, indexing="ij")
    w = (np.hypot(gx - c[0], gy - c[1]) <= R + 2.0).astype(np.float32)
    w /= max(float(w.sum()), 1.0)
    nn = max(n, 8)
    ctrl._cyw = {"x0": x0, "y0": y0, "m": m, "w": w, "q": np.zeros((m, m), np.float32),
                 "h": np.full((m, m), -1e9, np.float32), "gx": gx, "gy": gy, "n": n,
                 "yaw": [None] * nn, "last": [-10 ** 9] * nn,
                 "start": st_[:, 0:3].copy(), "exit": [None] * nn}
    return ctrl._cyw
def _cyw_update(G, depth_i, pos, rpy):
    fwd, up, right = camera_axes(rpy)
    cam = pos + fwd * CAMERA_OFFSET_M + up * CAMERA_UP_OFFSET_M
    d = depth_i[_cyw_r][:, _cyw_r].astype(np.float64) * (DEPTH_MAX_M - DEPTH_MIN_M) + DEPTH_MIN_M
    ok = (d >= 1.0) & (d <= DEPTH_MAX_M - 1.5)
    if not ok.any():
        return
    d = d[ok]
    U = _CYW_UU[ok] * HALF_TAN
    V = _CYW_VV[ok] * HALF_TAN
    px = cam[0] + d * (fwd[0] + U * right[0] + V * up[0])
    py = cam[1] + d * (fwd[1] + U * right[1] + V * up[1])
    pz = cam[2] + d * (fwd[2] + U * right[2] + V * up[2])
    rng = d * np.sqrt(1.0 + U * U + V * V)
    ix = np.floor((px - G["x0"]) / _CYW_CELL).astype(np.int64)
    iy = np.floor((py - G["y0"]) / _CYW_CELL).astype(np.int64)
    m = G["m"]
    inb = (ix >= 0) & (ix < m) & (iy >= 0) & (iy < m)
    if not inb.any():
        return
    ix, iy = ix[inb], iy[inb]
    np.maximum.at(G["q"], (ix, iy), _cyw_pd(rng[inb]).astype(np.float32))
    np.maximum.at(G["h"], (ix, iy), pz[inb].astype(np.float32))
def _cyw_hmap(G, x, y):
    ix = np.floor((x - G["x0"]) / _CYW_CELL).astype(np.int64)
    iy = np.floor((y - G["y0"]) / _CYW_CELL).astype(np.int64)
    m = G["m"]
    inb = (ix >= 0) & (ix < m) & (iy >= 0) & (iy < m)
    h = np.full(np.shape(x), -1e9, np.float32)
    h[inb] = G["h"][ix[inb], iy[inb]]
    return h
def _cyw_best_yaw(G, i, pos, agl, motion_yaw):
    m = G["m"]
    cx = int((pos[0] - G["x0"]) / _CYW_CELL)
    cy = int((pos[1] - G["y0"]) / _CYW_CELL)
    k = int(math.ceil(_CYW_RMAX / _CYW_CELL)) + 1
    a0, a1 = max(cx - k, 0), min(cx + k + 1, m)
    b0, b1 = max(cy - k, 0), min(cy + k + 1, m)
    if a0 >= a1 or b0 >= b1:
        return None, 0.0, 0.0
    wv = G["w"][a0:a1, b0:b1]
    dx = G["gx"][a0:a1, b0:b1] - pos[0]
    dy = G["gy"][a0:a1, b0:b1] - pos[1]
    r = np.hypot(dx, dy)
    rmin = max(_CYW_RMIN, float(agl) if (agl is not None and np.isfinite(agl)) else _CYW_RMIN)
    sel = (r >= rmin) & (r <= _CYW_RMAX) & (wv > 0.0)
    if not sel.any():
        return None, 0.0, 0.0
    pdr = _cyw_pd(r[sel])
    add = wv[sel] * np.clip(pdr - G["q"][a0:a1, b0:b1][sel], 0.0, None)
    full = wv[sel] * pdr
    if _CYW_OCC:
        cz = float(pos[2])
        cell_z = G["h"][a0:a1, b0:b1][sel]
        cell_z = np.where(cell_z > -1e8, cell_z, cz - 4.0)
        occ = np.zeros(add.shape, bool)
        for f in (0.35, 0.65):
            xs = pos[0] + dx[sel] * f
            ys = pos[1] + dy[sel] * f
            hz = _cyw_hmap(G, xs, ys)
            lz = cz + (cell_z - cz) * f
            occ |= (hz > -1e8) & (hz > lz + 0.5)
        add = np.where(occ, 0.2 * add, add)
    b = np.arctan2(dy[sel], dx[sel])
    bins = ((b + math.pi) / (2.0 * math.pi) * _CYW_NB).astype(np.int64) % _CYW_NB
    def wedge(vals):
        hist = np.bincount(bins, weights=vals, minlength=_CYW_NB)
        ext = np.concatenate([hist[_CYW_NB - _CYW_HW:], hist, hist[:_CYW_HW]])
        return np.convolve(ext, _CYW_KER, mode="valid")
    gain, gain0 = wedge(add), wedge(full)
    if motion_yaw is not None:
        fb = 1.0 + _CYW_FWD_BONUS * np.cos(_CYW_CENTRES - motion_yaw)
        gain, gain0 = gain * fb, gain0 * fb
        if _CYW_MAXOFF > 0.0:
            off = np.abs((_CYW_CENTRES - motion_yaw + np.pi) % (2.0 * np.pi) - np.pi)
            allowed = off <= math.radians(_CYW_MAXOFF)
            gain = np.where(allowed, gain, -1.0)
            gain0 = np.where(allowed, gain0, 1e-12)
    kbest = int(np.argmax(gain))
    cur = G["yaw"][i]
    if cur is not None:
        kc = int(((cur + math.pi) / (2.0 * math.pi) * _CYW_NB)) % _CYW_NB
        if float(gain[kc]) >= 0.0 and float(gain[kbest]) < _CYW_HYST * float(gain[kc]) and float(gain[kc]) >= _CYW_MIN_FRAC * float(gain0[kc]):
            return cur, float(gain[kc]), float(gain0[kc])
    return float(_CYW_CENTRES[kbest]), float(gain[kbest]), float(gain0[kbest])
def _cyw_track_window(G, cc, i, t, pos):
    if G["exit"][i] is None and cc is not None and getattr(cc, "mode", "initial") != 'initial':
        G["exit"][i] = t
    if t > _SARX_WIN_MAX:
        return False
    ex = G["exit"][i]
    if ex is None or t - ex <= _SARX_WIN_POST:
        return True
    return float(math.hypot(pos[0] - G["start"][i][0], pos[1] - G["start"][i][1])) <= _SARX_WIN_PAD_R
def _cyw_blind_cap(G, T, action, i, pos, rpy, vel_all, stats):
    a = action[i]
    v = np.asarray(a[0:3], dtype=np.float64) * float(a[3]) * SPEED_LIMIT
    vh = float(math.hypot(v[0], v[1]))
    if vh <= _CYW_VBLIND:
        return
    mis = math.degrees(abs(_cyw_wrap(math.atan2(v[1], v[0]) - float(rpy[2]))))
    if mis <= _CYW_BLIND_A0:
        return
    vdir = v[0:2] / vh
    ss = np.array([2.0, 4.0, 6.0, 9.0, MTN_CLIMB_LOOKAHEAD_M])
    hs = _cyw_hmap(G, pos[0] + vdir[0] * ss, pos[1] + vdir[1] * ss)
    known = hs > -1e8
    clear = bool(known.all()) and float(hs.max()) < float(pos[2]) - _CYW_CLEAR_M
    changed = False
    if known.any():
        hk = float(hs[known].max())
        need = hk + _CYW_CLEAR_M + 0.5 - float(pos[2])
        if need > 0.0:
            v[2] = max(v[2], min(1.5, need))
            changed = True
    if not clear:
        f = min(1.0, (mis - _CYW_BLIND_A0) / max(_CYW_BLIND_A1 - _CYW_BLIND_A0, 1e-3))
        cap = vh * (1.0 - f) + _CYW_VBLIND * f
        if cap < vh:
            v[0:2] *= cap / vh
            changed = True
            stats["sary_cyw_slow"] = stats.get("sary_cyw_slow", 0) + 1
    if changed:
        nv = float(np.linalg.norm(v))
        if nv > 1e-9:
            action[i, 0:3] = (v / nv).astype(np.float32)
            action[i, 3] = np.float32(min(nv / SPEED_LIMIT, 1.0))
        try:
            T.cap_commanded_tilt(action, vel_all, [i])
        except Exception:
            pass
def _cyw_own_yaw(cc):
    if getattr(cc, "_msp_turned", None) is not None:
        cc._msp_turned = None
    try:
        cc._msp_last_end = int(cc.tick)
    except Exception:
        pass
def _cyw_step(ctrl, action, observation):
    team = getattr(ctrl, "team", None)
    T = getattr(team, "T", None)
    if T is None or not getattr(T, "region_latched", False):
        return action
    _mp = _MAPSW_STATE.get("map")
    if _mp is not None and _mp != 'mountain' and not bool(getattr(T, "mtn", False)):
        return action
    st_ = _sarx_state(observation)
    n = st_.shape[0]
    G = getattr(ctrl, "_cyw", None)
    if G is None or G["n"] != n:
        G = _cyw_init(ctrl, T, n, st_)
    depth = np.asarray(observation["depth"], dtype=np.float32).reshape(n, DEPTH_RES, DEPTH_RES)
    vel_all = st_[:, 6:9]
    t = int(getattr(T, "tick", 0))
    stats = T.stats
    for i in range(min(n, action.shape[0], len(team.pilots))):
        if T.frozen[i]:
            continue
        pos = st_[i, 0:3]
        rpy = st_[i, 3:6]
        _cyw_update(G, depth[i], pos, rpy)
        p = _sarx_pilot(team, i)
        cc = getattr(p, "ctrl", None)
        in_win = _cyw_track_window(G, cc, i, t, pos)
        if cc is None or not _sarx_mtn(cc):
            G["yaw"][i] = None
            continue
        det = cc.detector
        release = (cc.mode != 'search' or cc.locked or cc.frozen or getattr(cc, "_mcv_until", -1) >= cc.tick
                   or i == T.confirmer or in_win or t < _CYW_T0)
        yaw = None
        if not release and _CYW_TRK and (float(det.strong) > 0.0 or float(getattr(det, "cand_strong", 0.0)) > 0.0):
            tgt = det.vic if float(det.strong) > 0.0 else getattr(det, "cand", None)
            if tgt is not None and np.all(np.isfinite(np.asarray(tgt, dtype=np.float64)[:2])):
                tgt = np.asarray(tgt, dtype=np.float64)
                if float(math.hypot(tgt[0] - pos[0], tgt[1] - pos[1])) > 1.0:
                    yaw = math.atan2(tgt[1] - pos[1], tgt[0] - pos[0])
                    G["yaw"][i] = None
                    stats["sary_cyw_trk"] = stats.get("sary_cyw_trk", 0) + 1
        elif not release:
            a = action[i]
            v = np.asarray(a[0:3], dtype=np.float64) * float(a[3]) * SPEED_LIMIT
            vh = float(math.hypot(v[0], v[1]))
            motion_yaw = math.atan2(v[1], v[0]) if vh > max(0.2, _CYW_MAXOFF_V) else None
            y2, g, g0 = _cyw_best_yaw(G, i, pos, getattr(cc, "_last_agl", None), motion_yaw)
            if y2 is not None and g0 > 0.0 and g >= _CYW_MIN_FRAC * g0:
                yaw = y2
                G["yaw"][i] = y2
                stats["sary_cyw"] = stats.get("sary_cyw", 0) + 1
            else:
                G["yaw"][i] = None
        else:
            G["yaw"][i] = None
        if yaw is not None:
            action[i, 4] = np.float32(yaw / math.pi)
            _cyw_own_yaw(cc)
            G["last"][i] = t
        if yaw is not None or (t - G["last"][i]) <= _CYW_HOLD:
            if not (cc.locked or cc.frozen or cc.mode == 'hover'):
                _cyw_blind_cap(G, T, action, i, pos, rpy, vel_all, stats)
    try:
        stats["sary_cov"] = float((G["w"] * G["q"]).sum())
    except Exception:
        pass
    return action
_SARY_BASE = DroneFlightController
class DroneFlightController(_SARY_BASE):
    def reset(self):
        self._cyw = None
        return _SARY_BASE.reset(self)
    def act(self, observation):
        action = _SARY_BASE.act(self, observation)
        if _CYW_ON:
            try:
                action = _cyw_step(self, np.array(action, dtype=np.float32, copy=True), observation)
            except Exception as exc:
                _log_once("sary_cyw", "coverage yaw failed", exc)
        return action
_CSC_ON = os.environ.get('SARZ_CSC', '1') == '1'
_CSC_V = float(os.environ.get('SARZ_CSC_V', '0.9'))
_CSC_R = float(os.environ.get('SARZ_CSC_R', '2.5'))
_CSC_DZ_LO = float(os.environ.get('SARZ_CSC_DZLO', '1.2'))
_CSC_DZ_HI = float(os.environ.get('SARZ_CSC_DZHI', '4.8'))
_CSC_FOREST = os.environ.get('SARZ_CSC_FOREST', '0') == '1'
def _csc_step(ctrl, action, observation):
    team = getattr(ctrl, "team", None)
    if team is None:
        return action
    st_ = _sarx_state(observation)
    n = min(st_.shape[0], action.shape[0], len(team.pilots))
    for i in range(n):
        cc = getattr(_sarx_pilot(team, i), "ctrl", None)
        if cc is None or getattr(cc, "mode", None) != 'hover' or not cc.locked or not (_sarx_mtn(cc) or (_CSC_FOREST and _MAPSW_STATE.get('map') == 'forest')):
            continue
        lv = np.asarray(cc.locked_vic, dtype=np.float64)
        if not np.all(np.isfinite(lv[:2])) or float(math.hypot(lv[0] - st_[i, 0], lv[1] - st_[i, 1])) > _CSC_R:
            continue
        dzh = float(st_[i, 2]) - float(getattr(cc, "locked_head_z", lv[2]))
        if not (_CSC_DZ_LO <= dzh <= _CSC_DZ_HI):
            continue
        a = action[i]
        v = np.asarray(a[0:3], dtype=np.float64) * float(a[3]) * SPEED_LIMIT
        nv = float(np.linalg.norm(v))
        if nv > _CSC_V:
            v *= _CSC_V / nv
            action[i, 0:3] = (v / _CSC_V).astype(np.float32)
            action[i, 3] = np.float32(_CSC_V / SPEED_LIMIT)
            T = getattr(team, "T", None)
            if T is not None:
                T.stats["sarz_csc"] = T.stats.get("sarz_csc", 0) + 1
    return action
_CSC_BASE = DroneFlightController
class DroneFlightController(_CSC_BASE):
    def act(self, observation):
        action = _CSC_BASE.act(self, observation)
        if _CSC_ON:
            try:
                action = _csc_step(self, np.array(action, dtype=np.float32, copy=True), observation)
            except Exception as exc:
                _log_once("sarz_csc", "confirm speed cap failed", exc)
        return action
_LEG_ON = os.environ.get('SARZ_LEG', '0') == '1'
_LEG_HITS = int(os.environ.get('SARZ_LEG_HITS', '25'))
_LEG_D = float(os.environ.get('SARZ_LEG_D', '10.0'))
_LEG_T = int(os.environ.get('SARZ_LEG_T', '250'))
_LEG_KEEP = 1500
def _leg_state(ctrl, n):
    G = getattr(ctrl, "_leg", None)
    if G is None or len(G) != n:
        G = ctrl._leg = [{"hits": [], "lock0": None, "fz0": None, "flag": False, "fired": False,
                          "wl": False, "wf": False} for _ in range(n)]
    return G
def _leg_pre(ctrl):
    team = getattr(ctrl, "team", None)
    G = getattr(ctrl, "_leg", None)
    if team is None or G is None:
        return
    k = int(getattr(ctrl, "_leg_k", 0))
    for i in range(min(len(G), len(team.pilots))):
        g = G[i]
        if not g["flag"] or g["fz0"] is None or k - g["fz0"] < _LEG_T:
            continue
        cc = getattr(_sarx_pilot(team, i), "ctrl", None)
        if cc is None or not cc.frozen or getattr(cc, "mode", None) != 'hover':
            continue
        cc.hover_stable_ticks = max(int(cc.hover_stable_ticks), 10 ** 6)
        cc.hover_total_ticks = max(int(cc.hover_total_ticks), 10 ** 6)
        if not g["fired"]:
            g["fired"] = True
            T = getattr(team, "T", None)
            if T is not None:
                T.stats["sarz_leg"] = T.stats.get("sarz_leg", 0) + 1
def _leg_post(ctrl, observation):
    team = getattr(ctrl, "team", None)
    if team is None:
        return
    st_ = _sarx_state(observation)
    n = min(st_.shape[0], len(team.pilots))
    G = _leg_state(ctrl, n)
    k = int(getattr(ctrl, "_leg_k", 0)) + 1
    ctrl._leg_k = k
    for i in range(n):
        cc = getattr(_sarx_pilot(team, i), "ctrl", None)
        if cc is None or not _sarx_mtn(cc):
            continue
        g = G[i]
        det = cc.detector
        z = getattr(det, "last_z", None)
        if float(getattr(det, "last_p_depth", 0.0)) >= 0.7 and z is not None:
            z = np.asarray(z, dtype=np.float64)
            if z.shape[0] >= 2 and np.all(np.isfinite(z[:2])):
                g["hits"].append((k, float(z[0]), float(z[1]), float(st_[i, 0]), float(st_[i, 1])))
        while g["hits"] and k - g["hits"][0][0] > _LEG_KEEP:
            g["hits"].pop(0)
        lk, fz = bool(cc.locked), bool(cc.frozen)
        if lk and not g["wl"]:
            g["lock0"], g["fz0"], g["flag"], g["fired"] = k, None, False, False
        if fz and not g["wf"] and lk:
            lv = np.asarray(cc.locked_vic, dtype=np.float64)
            k0 = g["lock0"] if g["lock0"] is not None else k - _LEG_KEEP
            hs = [h for h in g["hits"] if h[0] >= k0 and math.hypot(h[1] - lv[0], h[2] - lv[1]) < 3.0]
            last_d = math.hypot(hs[-1][3] - lv[0], hs[-1][4] - lv[1]) if hs else 99.0
            g["fz0"] = k
            g["flag"] = bool(np.all(np.isfinite(lv[:2]))) and len(hs) < _LEG_HITS and last_d > _LEG_D
        if not lk and not fz:
            g["fz0"], g["flag"] = None, False
        g["wl"], g["wf"] = lk, fz
_LEG_BASE = DroneFlightController
class DroneFlightController(_LEG_BASE):
    def reset(self):
        self._leg = None
        self._leg_k = 0
        return _LEG_BASE.reset(self)
    def act(self, observation):
        if _LEG_ON:
            try:
                _leg_pre(self)
            except Exception as exc:
                _log_once("sarz_leg_pre", "low-evidence give-up failed", exc)
        action = _LEG_BASE.act(self, observation)
        if _LEG_ON:
            try:
                _leg_post(self, observation)
            except Exception as exc:
                _log_once("sarz_leg_post", "low-evidence bookkeeping failed", exc)
        return action
_OAC_ON = os.environ.get('SARZ_OAC', '1') == '1'
_OAC_K = int(os.environ.get('SARZ_OAC_K', '6'))
_OAC_P = float(os.environ.get('SARZ_OAC_P', '0.85'))
_OAC_W = int(os.environ.get('SARZ_OAC_W', '200'))
_OAC_PREV = PerDronePilot._outer_refuted
def _oac_outer_refuted(self, q):
    if _OAC_ON and _MAPSW_STATE.get("map") == 'mountain':
        try:
            det = self.ctrl.detector
            g = self._support_relax
            in_relax = True
            if g is not None:
                v = g(q)
                in_relax = bool(v) if isinstance(v, (bool, np.bool_)) else float(v) >= 0.5
            if in_relax and float(det.last_p_depth) >= _OAC_P:
                tk = int(det.t)
                log = getattr(self, "_oac_log", None)
                if log is None:
                    log = self._oac_log = []
                qx, qy = float(np.asarray(q).ravel()[0]), float(np.asarray(q).ravel()[1])
                if not log or log[-1][0] != tk:
                    log.append((tk, qx, qy))
                    while log and tk - log[0][0] > _OAC_W:
                        log.pop(0)
                k = sum(1 for t0, x0, y0 in log if math.hypot(x0 - qx, y0 - qy) <= CONSIST_M)
                if k >= _OAC_K:
                    self._outer_admits += 1
                    self._oac_admits = getattr(self, "_oac_admits", 0) + 1
                    return False
        except Exception as exc:
            _log_once("sarz_oac", "outer accumulation failed", exc)
    return _OAC_PREV(self, q)
PerDronePilot._outer_refuted = _oac_outer_refuted
_RLK_ON = os.environ.get('SARZ_RLK', '0') == '1'
_RLK_T = float(os.environ.get('SARZ_RLK_T', '7.0'))
_RLK_HITS = int(os.environ.get('SARZ_RLK_HITS', '3'))
_RLK_DMIN = float(os.environ.get('SARZ_RLK_DMIN', '6.0'))
_RLK_DMAX = float(os.environ.get('SARZ_RLK_DMAX', '11.0'))
_RLK_V = float(os.environ.get('SARZ_RLK_V', '2.0'))
_RLK_LOOK = int(os.environ.get('SARZ_RLK_LOOK', '150'))
_RLK_NEED = int(os.environ.get('SARZ_RLK_NEED', '3'))
_RLK_BACK_MAX = int(os.environ.get('SARZ_RLK_BACK_MAX', '400'))
def _rlk_cmd(action, i, vel, yaw):
    v = np.asarray(vel, dtype=np.float64)
    nv = float(np.linalg.norm(v))
    if nv > 1e-6:
        action[i, 0:3] = (v / nv).astype(np.float32)
        action[i, 3] = np.float32(min(nv / SPEED_LIMIT, 1.0))
    else:
        action[i, 0:4] = 0.0
    action[i, 4] = np.float32(((yaw + math.pi) % (2.0 * math.pi) - math.pi) / math.pi)
def _rlk_step(ctrl, action, observation):
    team = getattr(ctrl, "team", None)
    T = getattr(team, "T", None)
    if team is None:
        return action
    st_ = _sarx_state(observation)
    n = min(st_.shape[0], action.shape[0], len(team.pilots))
    G = getattr(ctrl, "_rlk", None)
    if G is None or len(G) != n:
        G = ctrl._rlk = [{"hits": [], "wl": False, "lock0": None, "done": set(), "phase": None} for _ in range(n)]
    k = int(getattr(ctrl, "_rlk_k", 0)) + 1
    ctrl._rlk_k = k
    stats = T.stats if T is not None else {}
    depth = None
    for i in range(n):
        cc = getattr(_sarx_pilot(team, i), "ctrl", None)
        if cc is None or not _sarx_mtn(cc):
            continue
        g = G[i]
        det = cc.detector
        pos = st_[i, 0:3]
        z = getattr(det, "last_z", None)
        if g["phase"] is None and float(getattr(det, "last_p_depth", 0.0)) >= 0.7 and z is not None:
            z = np.asarray(z, dtype=np.float64)
            if z.shape[0] >= 3 and np.all(np.isfinite(z[:3])):
                g["hits"].append((k, float(z[0]), float(z[1]), float(pos[0]), float(pos[1]), float(pos[2])))
        while g["hits"] and k - g["hits"][0][0] > 1500:
            g["hits"].pop(0)
        lk = bool(cc.locked)
        if lk and not g["wl"]:
            g["lock0"] = k
            g["trail"] = []
            g["hover0"] = None
        if lk and cc.frozen and getattr(cc, "mode", None) == 'hover' and g.get("hover0") is None:
            g["hover0"] = k
        g["wl"] = lk
        if lk and g["phase"] is None and k % 10 == 0:
            tr_ = g.setdefault("trail", [])
            tr_.append(np.asarray(pos, dtype=np.float64).copy())
            if len(tr_) > 250:
                del tr_[0]
        if g["phase"] is not None and not (cc.locked and cc.frozen):
            g["phase"] = None
            continue
        if g["phase"] is None:
            if not (lk and cc.frozen and getattr(cc, "mode", None) == 'hover'):
                continue
            h0 = g.get("hover0")
            if h0 is None or (k - h0) / CTRL_HZ < _RLK_T:
                continue
            lv = np.asarray(cc.locked_vic, dtype=np.float64).copy()
            key = (round(float(lv[0]), 0), round(float(lv[1]), 0))
            if key in g["done"] or not np.all(np.isfinite(lv[:3])):
                continue
            g["done"].add(key)
            k0 = g["lock0"] if g["lock0"] is not None else k - 1500
            hl = [h for h in g["hits"] if h[0] >= k0 and math.hypot(h[1] - lv[0], h[2] - lv[1]) < 3.0]
            nh = len(hl)
            if nh < _RLK_HITS:
                stats["sarz_rlk_skip"] = stats.get("sarz_rlk_skip", 0) + 1
                continue
            vp = None
            for h in reversed(hl):
                if math.hypot(h[3] - lv[0], h[4] - lv[1]) >= _RLK_DMIN - 1.0:
                    vp = np.array([h[3], h[4], h[5]], dtype=np.float64)
                    break
            if vp is None and g.get("trail"):
                for q in reversed(g["trail"]):
                    dq = math.hypot(float(q[0]) - lv[0], float(q[1]) - lv[1])
                    if _RLK_DMIN <= dq <= _RLK_DMAX:
                        vp = np.asarray(q, dtype=np.float64).copy()
                        break
            if vp is None:
                stats["sarz_rlk_notrail"] = stats.get("sarz_rlk_notrail", 0) + 1
                continue
            g.update(phase="back", lv=lv, vp=vp, t0=k, got=[])
            stats["sarz_rlk_start"] = stats.get("sarz_rlk_start", 0) + 1
        lv, vp = g["lv"], g["vp"]
        yaw_to = math.atan2(lv[1] - pos[1], lv[0] - pos[0])
        agl = float(st_[i, 162]) * 20.0 if st_.shape[1] > 162 else 99.0
        if g["phase"] == "back":
            dv = vp - pos
            dist = float(np.linalg.norm(dv))
            if dist < 1.0 or k - g["t0"] > _RLK_BACK_MAX:
                g["phase"], g["t1"] = "look", k
            else:
                vel = dv / max(dist, 1e-6) * min(_RLK_V, dist)
                if agl < 2.5:
                    vel[2] = max(vel[2], 0.5)
                _rlk_cmd(action, i, vel, yaw_to)
                continue
        if g["phase"] == "look":
            vel = np.zeros(3)
            if agl < 2.5:
                vel[2] = 0.5
            _rlk_cmd(action, i, vel, yaw_to)
            yerr = abs((yaw_to - float(st_[i, 5]) + math.pi) % (2.0 * math.pi) - math.pi)
            if (k - g["t1"]) % 2 == 0 and yerr < math.radians(25.0):
                try:
                    if depth is None:
                        depth = np.asarray(observation["depth"], dtype=np.float32).reshape(-1, DEPTH_RES, DEPTH_RES)
                    out = det._depth_heads_nd(depth[i].reshape(DEPTH_RES, DEPTH_RES, 1), True, pos, st_[i, 3:6])
                    pp = float(out[0])
                    if pp >= STRONG_P:
                        zz = measurement_world(float(out[1]), float(out[2]), float(np.exp(min(float(out[3]), 6.0))), pos, st_[i, 3:6])
                        if math.hypot(zz[0] - lv[0], zz[1] - lv[1]) <= CONSIST_M:
                            g["got"].append(np.asarray(zz, dtype=np.float64))
                except Exception as exc:
                    _log_once("sarz_rlk_det", "re-look detection failed", exc)
            if len(g["got"]) >= _RLK_NEED + 3 or k - g["t1"] > _RLK_LOOK:
                if len(g["got"]) >= _RLK_NEED:
                    new = np.median(np.asarray(g["got"]), axis=0)
                    cc.locked_head_z = float(cc.locked_head_z) + float(new[2] - lv[2])
                    cc.locked_vic = np.array([new[0], new[1], new[2]], dtype=np.float64)
                    cc.hover_stable_ticks = 0
                    cc._hsd_settle = 0
                    cc.hover_total_ticks = 0
                    cc._sweep_t0 = None
                    try:
                        if T is not None and T.cand is not None and math.hypot(T.cand["xy"][0] - lv[0], T.cand["xy"][1] - lv[1]) <= CONSIST_M:
                            T.cand["xy"] = np.array([new[0], new[1]], dtype=np.float64)
                    except Exception:
                        pass
                    stats["sarz_rlk_moved"] = stats.get("sarz_rlk_moved", 0) + 1
                    stats["sarz_rlk_shift_cm"] = stats.get("sarz_rlk_shift_cm", 0) + int(100 * math.hypot(new[0] - lv[0], new[1] - lv[1]))
                else:
                    stats["sarz_rlk_none"] = stats.get("sarz_rlk_none", 0) + 1
                g["phase"] = None
    return action
_RLK_BASE = DroneFlightController
class DroneFlightController(_RLK_BASE):
    def reset(self):
        self._rlk = None
        self._rlk_k = 0
        return _RLK_BASE.reset(self)
    def act(self, observation):
        action = _RLK_BASE.act(self, observation)
        if _RLK_ON:
            try:
                action = _rlk_step(self, np.array(action, dtype=np.float32, copy=True), observation)
            except Exception as exc:
                _log_once("sarz_rlk", "re-look failed", exc)
        return action
_FTC_ON = os.environ.get('SARZ_FTC', '1') == '1'
_TRC_DEG = float(os.environ.get('SARZ_TRC', '38'))
_FTC_BASE = DroneFlightController
class DroneFlightController(_FTC_BASE):
    def act(self, observation):
        action = _FTC_BASE.act(self, observation)
        if _FTC_ON and _MAPSW_STATE.get("map") == 'mountain':
            try:
                T = getattr(getattr(self, "team", None), "T", None)
                if T is not None:
                    st_ = _sarx_state(observation)
                    n = min(st_.shape[0], np.asarray(action).shape[0])
                    act2 = np.array(action, dtype=np.float32, copy=True)
                    T.cap_commanded_tilt(act2, st_[:n, 6:9], list(range(n)))
                    if _TRC_DEG > 0.0:
                        for i in range(n):
                            tl = max(abs(float(st_[i, 3])), abs(float(st_[i, 4])))
                            if tl > math.radians(_TRC_DEG):
                                va = st_[i, 6:9].astype(np.float64)
                                v = np.array([va[0], va[1], float(np.clip(va[2] + 0.2, -1.0, 1.0))])
                                _store_vel(act2[i], v / SPEED_LIMIT)
                                T.stats["sarz_trc"] = T.stats.get("sarz_trc", 0) + 1
                    action = act2
            except Exception as exc:
                _log_once("sarz_ftc", "final tilt cap failed", exc)
        return action
_PDX_ON = os.environ.get('SARZ_PDX', '1') == '1'
_PDX_K = int(os.environ.get('SARZ_PDX_K', '6'))
_PDX_P = float(os.environ.get('SARZ_PDX_P', '0.85'))
_PDX_W = int(os.environ.get('SARZ_PDX_W', '200'))
_PDX_PREV = PerDronePilot._gated_refuted
def _pdx_gated_refuted(self, z):
    if _PDX_ON and _MAPSW_STATE.get("map") == 'mountain':
        try:
            det = self.ctrl.detector
            tk = int(det.t)
            zz = np.asarray(z, dtype=np.float64).ravel()
            zx, zy = float(zz[0]), float(zz[1])
            log = getattr(self, "_pdx_log", None)
            if log is None:
                log = self._pdx_log = []
            if float(det.last_p_depth) >= _PDX_P and (not log or log[-1][0] != tk):
                log.append((tk, zx, zy))
            while log and tk - log[0][0] > _PDX_W:
                log.pop(0)
            if self._soft_deny:
                ev = sum(1 for e in log if math.hypot(e[1] - zx, e[2] - zy) <= CONSIST_M)
                if ev >= _PDX_K:
                    keep = [(xy, et) for xy, et in self._soft_deny
                            if math.hypot(zx - float(xy[0]), zy - float(xy[1])) > CONSIST_M]
                    if len(keep) < len(self._soft_deny):
                        self._soft_deny = keep
                        self._pdx_lifts = getattr(self, "_pdx_lifts", 0) + 1
        except Exception as exc:
            _log_once("sarz_pdx", "soft-deny lift failed", exc)
    return _PDX_PREV(self, z)
PerDronePilot._gated_refuted = _pdx_gated_refuted
_RFZ_ON = os.environ.get('SARZ_RFZ', '1') == '1'
_RFZ_R = float(os.environ.get('SARZ_RFZ_R', '5.0'))
_RFZ_MIN = int(os.environ.get('SARZ_RFZ_MIN', '3'))
def _rfz_step(ctrl, observation):
    team = getattr(ctrl, "team", None)
    if team is None:
        return
    n = len(team.pilots)
    G = getattr(ctrl, "_rfz", None)
    if G is None or len(G) != n:
        G = ctrl._rfz = [{"hits": [], "wl": False, "wf": False} for _ in range(n)]
    T = getattr(team, "T", None)
    for i in range(n):
        cc = getattr(_sarx_pilot(team, i), "ctrl", None)
        if cc is None or not _sarx_mtn(cc):
            continue
        g = G[i]
        lk, fz = bool(cc.locked), bool(cc.frozen)
        k = int(cc.tick)
        det = cc.detector
        z = getattr(det, "last_z", None)
        if not fz and float(getattr(det, "last_p_depth", 0.0)) >= 0.7 and z is not None:
            z = np.asarray(z, dtype=np.float64)
            if z.shape[0] >= 3 and np.all(np.isfinite(z[:3])):
                g["hits"].append((k, z[:3].copy()))
        while g["hits"] and k - g["hits"][0][0] > 1500:
            g["hits"].pop(0)
        if fz and lk and not g["wf"] and g["hits"]:
            lv = np.asarray(cc.locked_vic, dtype=np.float64)
            H = np.asarray([h[1] for h in g["hits"]])
            near = H[np.hypot(H[:, 0] - lv[0], H[:, 1] - lv[1]) <= _RFZ_R]
            if near.shape[0] >= _RFZ_MIN and np.all(np.isfinite(lv[:3])):
                med = np.median(near, axis=0)
                cc.locked_head_z = float(cc.locked_head_z) + float(med[2] - lv[2])
                cc.locked_vic = np.array([med[0], med[1], med[2]], dtype=np.float64)
                try:
                    if T is not None and T.cand is not None and math.hypot(T.cand["xy"][0] - lv[0], T.cand["xy"][1] - lv[1]) <= CONSIST_M:
                        T.cand["xy"] = np.array([med[0], med[1]], dtype=np.float64)
                    if T is not None:
                        T.stats["sarz_rfz"] = T.stats.get("sarz_rfz", 0) + 1
                        T.stats["sarz_rfz_cm"] = T.stats.get("sarz_rfz_cm", 0) + int(100 * math.hypot(med[0] - lv[0], med[1] - lv[1]))
                except Exception:
                    pass
        g["wl"], g["wf"] = lk, fz
_RFZ_BASE = DroneFlightController
class DroneFlightController(_RFZ_BASE):
    def reset(self):
        self._rfz = None
        return _RFZ_BASE.reset(self)
    def act(self, observation):
        action = _RFZ_BASE.act(self, observation)
        if _RFZ_ON:
            try:
                _rfz_step(self, observation)
            except Exception as exc:
                _log_once("sarz_rfz", "robust freeze failed", exc)
        return action
_GEOD_ON = os.environ.get('SARZ_GEOD', '1') == '1'
_GEOD_P = float(os.environ.get('SARZ_GEOD_P', '0.9'))
_GEOD_THR = float(os.environ.get('SARZ_GEOD_THR', '0.03'))
_GEOD_AMAX = float(os.environ.get('SARZ_GEOD_AMAX', '0.3'))
_GEOD_DMAX = float(os.environ.get('SARZ_GEOD_DMAX', '16.0'))
_GEOD_DLAB = float(os.environ.get('SARZ_GEOD_DLAB', '22.0'))
_GEOD_MINPX = int(os.environ.get('SARZ_GEOD_MINPX', '12'))
_GEOD_EXT = float(os.environ.get('SARZ_GEOD_EXT', '3.0'))
_GEOD_LIN = float(os.environ.get('SARZ_GEOD_LIN', '10.0'))
_GEOD_BORDER = int(os.environ.get('SARZ_GEOD_BORDER', '3'))
_GEOD_PADS = []
_GEOD_MATES = []
_GDV_ON = os.environ.get('SARZ_GDV', '0') == '1'
_GDV_R = float(os.environ.get('SARZ_GDV_R', '1.0'))
try:
    from scipy import ndimage as _geod_ndi
except Exception:
    _geod_ndi = None
_GEOD_ST8 = np.ones((3, 3), bool)
def _geod_detect(depth01):
    D = np.asarray(depth01, dtype=np.float32).reshape(DEPTH_RES, DEPTH_RES) * (DEPTH_MAX_M - DEPTH_MIN_M) + DEPTH_MIN_M
    w = 1.0 / D
    A = np.zeros_like(w)
    A[1:-1, :] += np.abs(w[:-2, :] - 2.0 * w[1:-1, :] + w[2:, :])
    A[:, 1:-1] += np.abs(w[:, :-2] - 2.0 * w[:, 1:-1] + w[:, 2:])
    A *= D * D
    M = (A > _GEOD_THR) & (D < _GEOD_DLAB)
    if not M.any():
        return None
    lab, n = _geod_ndi.label(M, structure=_GEOD_ST8)
    cnt = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    mx = np.asarray(_geod_ndi.maximum(A, lab, np.arange(1, n + 1)))
    cand = np.nonzero((cnt >= _GEOD_MINPX) & (mx > _GEOD_AMAX))[0]
    if cand.size == 0:
        return None
    sl = _geod_ndi.find_objects(lab)
    best = None
    last = DEPTH_RES - 1
    for j in cand:
        s_ = sl[j]
        r0, r1, c0, c1 = s_[0].start, s_[0].stop, s_[1].start, s_[1].stop
        if min(r0, c0, last - (r1 - 1), last - (c1 - 1)) < _GEOD_BORDER:
            continue
        sub = lab[s_] == j + 1
        rr, cc = np.nonzero(sub)
        dm = float(np.median(D[s_][sub]))
        if dm >= _GEOD_DMAX:
            continue
        if max(r1 - r0, c1 - c0) * dm * 2.0 / DEPTH_RES > _GEOD_EXT:
            continue
        ev = np.linalg.eigvalsh(np.cov(np.vstack([rr, cc]).astype(np.float64)) + 1e-6 * np.eye(2))
        if math.sqrt(ev[1] / max(ev[0], 1e-6)) > _GEOD_LIN:
            continue
        sc = float(cnt[j]) * float(mx[j])
        if best is None or sc > best[0]:
            best = (sc, float(rr.mean() + r0), float(cc.mean() + c0), dm)
    return best
_GEOD_PREV = VictimDetector._depth_heads_nd
def _geod_heads(self, x, is_mountain, pos, rpy):
    out = _GEOD_PREV(self, x, is_mountain, pos, rpy)
    if not (_GEOD_ON and is_mountain) or _geod_ndi is None:
        return out
    try:
        if float(out[0]) >= STRONG_P:
            return out
        g = _geod_detect(x)
        if g is None:
            return out
        _sc, r, c, dm = g
        u = (c + 0.5) / (0.5 * DEPTH_RES) - 1.0
        v = 1.0 - (r + 0.5) / (0.5 * DEPTH_RES)
        z = measurement_world(u, v, dm, np.asarray(pos, dtype=np.float64), np.asarray(rpy, dtype=np.float64))
        for px, py in _GEOD_PADS:
            if math.hypot(z[0] - px, z[1] - py) < 2.0:
                return out
        if _GDV_ON:
            for qx, qy, qz in _GEOD_MATES:
                if (qx - pos[0]) ** 2 + (qy - pos[1]) ** 2 + (qz - pos[2]) ** 2 < 0.25:
                    continue
                if (z[0] - qx) ** 2 + (z[1] - qy) ** 2 + (z[2] - qz) ** 2 < _GDV_R ** 2:
                    self.gdv_drops = getattr(self, "gdv_drops", 0) + 1
                    return out
        self.geod_hits = getattr(self, "geod_hits", 0) + 1
        return (_GEOD_P, u, v, float(math.log(max(dm, 0.5))), out[4])
    except Exception as exc:
        _log_once("sarz_geod", "geometric detector failed", exc)
        return out
VictimDetector._depth_heads_nd = _geod_heads
_GEOD_BASE = DroneFlightController
class DroneFlightController(_GEOD_BASE):
    def reset(self):
        del _GEOD_PADS[:]
        del _GEOD_MATES[:]
        return _GEOD_BASE.reset(self)
    def act(self, observation):
        if not _GEOD_PADS:
            try:
                st_ = _sarx_state(observation)
                _GEOD_PADS.extend((float(q[0]), float(q[1])) for q in st_[:, 0:2])
            except Exception:
                pass
        if _GDV_ON:
            try:
                _GEOD_MATES[:] = [(float(q[0]), float(q[1]), float(q[2])) for q in _sarx_state(observation)[:, 0:3]]
            except Exception:
                del _GEOD_MATES[:]
        action = _GEOD_BASE.act(self, observation)
        try:
            T = getattr(getattr(self, "team", None), "T", None)
            if T is not None:
                T.stats["sarz_geod"] = float(sum(getattr(pw.p.ctrl.detector, "geod_hits", 0) for pw in self.team.pilots))
                T.stats["sarz_gdv"] = float(sum(getattr(pw.p.ctrl.detector, "gdv_drops", 0) for pw in self.team.pilots))
        except Exception:
            pass
        return action
_PADV_PREV = VictimDetector.location_refuted
def _padv_near(z, cam):
    pads = _MCS2G.get("pads")
    if pads is None:
        return False
    rx, ry = z[0] - pads[:, 0], z[1] - pads[:, 1]
    dz = z[2] - (pads[:, 2] - 0.121)
    okz = (dz >= -_PADV_DZLO) & (dz <= _PADV_DZHI)
    if np.any(okz & (np.hypot(rx, ry) <= _PADV_R)):
        return True
    if _PADV_RAY and cam is not None:
        ux, uy = z[0] - cam[0], z[1] - cam[1]
        L = math.hypot(ux, uy)
        if L > 1.0:
            ux, uy = ux / L, uy / L
            along = rx * ux + ry * uy
            lat = np.abs(rx * uy - ry * ux)
            return bool(np.any(okz & (lat <= 1.0) & (along >= -3.5) & (along <= 1.0)))
    return False
def _padv_refuted(self, z):
    if _PADV_ON and getattr(self, "padv_live", False):
        try:
            if _padv_near(np.asarray(z, dtype=np.float64), getattr(self, "padv_cam", None)):
                _MCS2G["padv"] = _MCS2G.get("padv", 0) + 1
                return True
        except Exception:
            pass
    return _PADV_PREV(self, z)
VictimDetector.location_refuted = _padv_refuted
_BKP_ON = os.environ.get('SARZ_BKP', '0') == '1'
_BKP_TICKS = int(os.environ.get('SARZ_BKP_TICKS', '125'))
_BKP_MAPS = tuple(x for x in os.environ.get('SARZ_BKP_MAPS', 'mountain').split(',') if x)
_BKP_DH = float(os.environ.get('SARZ_BKP_DH', '1.8'))
_BKP_DZ = (float(os.environ.get('SARZ_BKP_DZLO', '1.8')), float(os.environ.get('SARZ_BKP_DZHI', '4.2')))
_BKP_PREV = PerDronePilot._apply_deny
def _bkp_apply_deny(self):
    if _BKP_ON:
        try:
            c = self.ctrl
            if (_MAPSW_STATE.get("map") in _BKP_MAPS or bool(c.mtn.is_mountain)) and (c.locked or c.frozen) \
                    and c.mode == 'hover':
                v = np.asarray(c.locked_vic, dtype=np.float64)
                p = np.asarray(self._last_pos, dtype=np.float64)
                dz = float(p[2]) - float(c.locked_head_z)
                if math.hypot(v[0] - p[0], v[1] - p[1]) <= _BKP_DH and _BKP_DZ[0] <= dz <= _BKP_DZ[1]:
                    t = int(c.tick)
                    last = int(getattr(self, "_bkp_last", -99))
                    if t - last > 5 or t < last:
                        self._bkp_t0 = t
                    self._bkp_last = t
                    t0 = getattr(self, "_bkp_t0", None)
                    if t0 is not None and 0 <= t - int(t0) <= _BKP_TICKS:
                        d = self._team_deny
                        hit = (d is not None and math.hypot(v[0] - d[0], v[1] - d[1]) <= _ks.CONSIST_M) or any(
                            t <= e and math.hypot(v[0] - xy[0], v[1] - xy[1]) <= _ks.CONSIST_M for xy, e in self._soft_deny)
                        if not hit:
                            k = "bkp_inband"
                        elif _fix_world(c) and c.detector.evidence_at(v[0:2]) >= FIX_EVIDENCE:
                            k = "bkp_fixcov"
                        else:
                            k = "bkp_saves"
                        self.spin_stats[k] = self.spin_stats.get(k, 0) + 1
                        return
        except Exception:
            pass
    return _BKP_PREV(self)
PerDronePilot._apply_deny = _bkp_apply_deny
_FBG_ON = os.environ.get('SARF_FBG', '1') == '1'
_FBG_A0 = float(os.environ.get('SARF_FBG_A0', '60.0'))
_FBG_A1 = float(os.environ.get('SARF_FBG_A1', '90.0'))
_FBG_V = float(os.environ.get('SARF_FBG_V', '0.35'))
_FBG_MODES = tuple(x for x in os.environ.get('SARF_FBG_MODES', 'search,navigation').split(',') if x)
_FBG_VNAV = float(os.environ.get('SARF_FBG_VNAV', str(_FBG_V)))
_FBG_A0NAV = float(os.environ.get('SARF_FBG_A0NAV', str(_FBG_A0)))
def _fbg_step(ctrl, action, observation):
    if _MAPSW_STATE.get("map") != 'forest':
        return action
    team = getattr(ctrl, "team", None)
    T = getattr(team, "T", None)
    if T is None or bool(getattr(T, "mtn", False)):
        return action
    st_ = _sarx_state(observation)
    n = min(st_.shape[0], action.shape[0], len(team.pilots))
    capped = []
    for i in range(n):
        if T.frozen[i]:
            continue
        cc = getattr(_sarx_pilot(team, i), "ctrl", None)
        if cc is None or getattr(cc, "mode", None) not in _FBG_MODES:
            continue
        a = action[i]
        v = np.asarray(a[0:3], dtype=np.float64) * float(a[3]) * SPEED_LIMIT
        vh = float(math.hypot(v[0], v[1]))
        nav = cc.mode == 'navigation'
        vcap = _FBG_VNAV if nav else _FBG_V
        a0 = _FBG_A0NAV if nav else _FBG_A0
        if vh <= vcap:
            continue
        mis = math.degrees(abs(_cyw_wrap(math.atan2(v[1], v[0]) - float(st_[i, 5]))))
        if mis <= a0:
            continue
        f = min(1.0, (mis - a0) / max(_FBG_A1 - a0, 1e-3))
        cap = vh * (1.0 - f) + vcap * f
        v[0:2] *= cap / vh
        nv = float(np.linalg.norm(v))
        if nv > 1e-9:
            action[i, 0:3] = (v / nv).astype(np.float32)
            action[i, 3] = np.float32(min(nv / SPEED_LIMIT, 1.0))
        capped.append(i)
        T.stats["sarf_fbg"] = T.stats.get("sarf_fbg", 0) + 1
    if capped:
        try:
            T.cap_commanded_tilt(action, st_[:, 6:9], capped)
        except Exception:
            pass
    return action
_FBG_BASE = DroneFlightController
class DroneFlightController(_FBG_BASE):
    def act(self, observation):
        action = _FBG_BASE.act(self, observation)
        if _FBG_ON:
            try:
                action = _fbg_step(self, np.array(action, dtype=np.float32, copy=True), observation)
            except Exception as exc:
                _log_once("sarf_fbg", "forest blind guard failed", exc)
        return action
MD_VDR_ON = os.environ.get('SWSAR_MD_VDR', '1') == '1'
MD_VDR_FILE = os.environ.get('SWSAR_MD_VDR_FILE', 'vdr02_full.onnx')
MD_VDR_THR = float(os.environ.get('SWSAR_MD_VDR_THR', '0.72'))
MD_VDR_WIN = 12
MD_VDR_MLOGIT = 2.0
MD_VDR_PCT = 25.0
MD_VDR_LPCH = (2, 4)
MD_VDR_ZMAX = float(os.environ.get('SWSAR_MD_VDR_ZMAX', '29.3'))
MD_VDR_RMAX = float(os.environ.get('SWSAR_MD_VDR_RMAX', '25.0'))
MD_VDR_PADR = float(os.environ.get('SWSAR_MD_VDR_PADR', '4.0'))
MD_VDR_LPONLY = os.environ.get('SWSAR_MD_VDR_LPONLY', '1') == '1'
MD_VDR_T0 = int(os.environ.get('SWSAR_MD_VDR_T0', '1000'))
MD_VDR_FT = float(os.environ.get('SWSAR_MD_VDR_FT', '0.4'))
MD_VDR_IV = int(os.environ.get('SWSAR_MD_VDR_IV', '50'))
MD_VDR_SMAX = int(os.environ.get('SWSAR_MD_VDR_SMAX', '12'))
MD_VDR_BASEMAX = int(os.environ.get('SWSAR_MD_VDR_BASEMAX', '4'))
MD_VDR_RES = int(os.environ.get('SWSAR_MD_VDR_RES', '0'))
MD_VDR_TILT = float(os.environ.get('SWSAR_MD_VDR_TILT', '15.0'))
MD_VDR_MOVE = float(os.environ.get('SWSAR_MD_VDR_MOVE', '2.5'))
MD_VDR_TURN = float(os.environ.get('SWSAR_MD_VDR_TURN', '12.0'))
MD_VDR_GLO = 3.0
MD_VDR_GHI = 24.0
MD_VDR_GFRAC = float(os.environ.get('SWSAR_MD_VDR_GFRAC', '0.25'))
MD_VDR_K = int(os.environ.get('SWSAR_MD_VDR_K', '2'))
MD_VDR_RUNMAX = int(os.environ.get('SWSAR_MD_VDR_RUNMAX', '3'))
MD_VDR_VGAP = int(os.environ.get('SWSAR_MD_VDR_VGAP', '12'))
MD_VDR_VN = int(os.environ.get('SWSAR_MD_VDR_VN', '3'))
MD_VDR_AGREE = float(os.environ.get('SWSAR_MD_VDR_AGREE', '2.5'))
MD_VDR_TTL = int(os.environ.get('SWSAR_MD_VDR_TTL', '600'))
MD_VDR_PIGGY = os.environ.get('SWSAR_MD_VDR_PIGGY', '1') == '1'
MD_VDR_COMMIT = os.environ.get('SWSAR_MD_VDR_COMMIT', '1') == '1'
MD_VDR_NBOOST = float(os.environ.get('SWSAR_MD_VDR_NBOOST', '6.0'))
MD_VDR_LPH = 0.6
MD_VDR_UPH = 1.7
MD_VDR_INJ_TTL = 10
MD_VDR_THREADS = int(os.environ.get('SWSAR_MD_VDR_THREADS', '1'))
MD_VDR_ENV_CAP = 40
MD_VDR_ABUF_LAST = 161
MD_VDR_ABUF_STEP = 6
MD_VDR_ABUF_N = 25
_MD_VDR_KEYS = ('arm', 'scan', 'ver', 'run', 'hit', 'piggy', 'phit', 'commit', 'refine', 'refuted', 'busy', 'dup',
                'over', 'pad', 'far', 'up', 'low', 'noframe', 'vfail', 'abuf_miss', 'inj', 'inj_drop', 'err', 'ms_max',
                'ms_tick_max', 'fmax', 't1', 'lt1', 'frames', 'tgt_tk', 'gate_tk')
_MD_VDR_STATS = {}
_MD_VDR_SESS = [None, False]
def _md_vdr_stat(k, d=1):
    _MD_VDR_STATS[k] = _MD_VDR_STATS.get(k, 0) + d
def _md_vdr_session():
    if _MD_VDR_SESS[0] is None and not _MD_VDR_SESS[1]:
        _MD_VDR_SESS[1] = True
        try:
            so = ort.SessionOptions()
            so.intra_op_num_threads = MD_VDR_THREADS
            so.inter_op_num_threads = 1
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            so.add_session_config_entry('session.intra_op.allow_spinning', '0')
            p = Path(__file__).resolve().parent / MD_VDR_FILE
            sess = ort.InferenceSession(str(p), so, providers=['CPUExecutionProvider'])
            o = sess.run(None, {'rgb': np.zeros((256, 256, 3), dtype=np.float32)})
            if tuple(o[0].shape) != (64, 64) or tuple(o[3].shape) != (128, 128):
                raise RuntimeError('vdr02 outputs %r' % ([x.shape for x in o],))
            _MD_VDR_SESS[0] = sess
        except Exception as exc:
            _log_once("md_vdr_load", "vdr02 session failed: %r" % (exc,), exc)
            _MD_VDR_SESS[0] = None
    return _MD_VDR_SESS[0]
def _md_vdr_hit(sess, rgb, dep01, st, pads):
    x = np.asarray(rgb, dtype=np.float32).reshape(256, 256, 3)
    if float(np.max(x)) <= 0.0:
        return None, 'noframe'
    t0 = time.perf_counter()
    h, o, c, m = sess.run(None, {'rgb': x})
    ms = (time.perf_counter() - t0) * 1e3
    _md_vdr_stat('run')
    if ms > _MD_VDR_STATS.get('ms_max', 0.0):
        _MD_VDR_STATS['ms_max'] = ms
    i = int(np.argmax(h))
    gy, gx = divmod(i, 64)
    pk = float(h[gy, gx])
    if pk < MD_VDR_THR:
        return None, 'low'
    cx = 4.0 * (gx + 0.5 + float(o[0, gy, gx]))
    cy = 4.0 * (gy + 0.5 + float(o[1, gy, gx]))
    plp = float(sum(c[j, gy, gx] for j in MD_VDR_LPCH))
    w = MD_VDR_WIN
    M = np.repeat(np.repeat(m, 2, 0), 2, 1)
    dep = np.asarray(dep01, dtype=np.float64).reshape(256, 256) * (DEPTH_MAX_M - DEPTH_MIN_M) + DEPTH_MIN_M
    py = min(max(gy * 4 + 2, w), 255 - w)
    px = min(max(gx * 4 + 2, w), 255 - w)
    L = M[py - w:py + w + 1, px - w:px + w + 1]
    Dw = dep[py - w:py + w + 1, px - w:px + w + 1]
    pm = L > MD_VDR_MLOGIT
    if not pm.any():
        pm = L >= L.max()
    zc = float(np.percentile(Dw[pm], MD_VDR_PCT))
    if zc >= MD_VDR_ZMAX:
        return None, 'far'
    pos = np.asarray(st[0:3], dtype=np.float64)
    rpy = np.asarray(st[3:6], dtype=np.float64)
    fw, up, rt = camera_axes(rpy)
    cam = pos + fw * CAMERA_OFFSET_M + up * CAMERA_UP_OFFSET_M
    tan = HALF_TAN
    u = cx / 128.0 - 1.0
    v = 1.0 - cy / 128.0
    P = cam + fw * zc + rt * (u * tan * zc) + up * (v * tan * zc)
    W = max(w, min(64, int(round(3.0 * 128.0 / max(zc, 1.0)))))
    y0, y1 = max(0, int(cy) - W), min(256, int(cy) + W + 1)
    x0, x1 = max(0, int(cx) - W), min(256, int(cx) + W + 1)
    L2 = M[y0:y1, x0:x1]
    D2 = dep[y0:y1, x0:x1]
    sel = (L2 > MD_VDR_MLOGIT) & (D2 >= zc - 1.0) & (D2 <= zc + 1.5)
    top = None
    if int(sel.sum()) >= 3:
        rr, cc = np.nonzero(sel)
        zz = D2[sel].astype(np.float64)
        uu = ((cc + x0) + 0.5) / 128.0 - 1.0
        vv = 1.0 - ((rr + y0) + 0.5) / 128.0
        Z = cam[2] + fw[2] * zz + rt[2] * (uu * tan * zz) + up[2] * (vv * tan * zz)
        top = float(np.percentile(Z, 95))
    lp = bool(plp >= 0.5)
    if top is None or top < P[2]:
        top = float(P[2]) + (0.4 if lp else 0.9)
    r = float(np.linalg.norm(P - cam))
    if r > MD_VDR_RMAX:
        return None, 'far'
    if pads is not None:
        for q in pads:
            if math.hypot(float(P[0]) - float(q[0]), float(P[1]) - float(q[1])) < MD_VDR_PADR and abs(float(P[2]) - float(q[2])) < 3.0:
                return None, 'pad'
    if MD_VDR_LPONLY and not lp:
        return None, 'up'
    return dict(x=float(P[0]), y=float(P[1]), z=float(P[2]), top=float(top), r=r, lp=lp, pk=pk), 'hit'
_MD_VDR_PREV_GATED = PerDronePilot._gated_update
def _md_vdr_gated_update(self, depth, pos, rpy, rgb=None, is_mountain=False):
    out = _MD_VDR_PREV_GATED(self, depth, pos, rpy, rgb=rgb, is_mountain=is_mountain)
    det = self.ctrl.detector
    inj = getattr(det, '_md_vdr_inj', None)
    if inj is None:
        return out
    det._md_vdr_inj = None
    try:
        z = np.asarray(inj['z'], dtype=np.float64).reshape(3)
        if det.location_refuted(z):
            _md_vdr_stat('refuted')
            return out
        hgt = float(inj['h'])
        cons = float(det.strong) > 0.0 and float(np.linalg.norm(z - det.vic)) <= CONSIST_M
        det.vic = z.copy()
        det.height = hgt
        det.strong = min(N_CAP, max(float(det.strong) if cons else 0.0, MD_VDR_NBOOST))
        det.last_strong_t = det.t
        if float(det.cand_strong) > 0.0 and float(np.linalg.norm(z - det.cand)) <= CONSIST_M:
            det.cand_strong = 0.0
        det.last_z = z.copy()
        det.seen = True
        det.conf = float(np.clip(det.strong / (NEWDET_N_LATCH if (NEWDET_N_LATCH > 0.0 and det.nd_sess) else N_LATCH), 0.0, 1.0))
        if MH_ON and is_mountain:
            head_z = det.vic[2] + 0.5 * float(np.clip(det.height, MH_HMIN, MH_HMAX))
        else:
            head_z = det.vic[2] + 0.5 * det.height
        _md_vdr_stat('inj')
        return (det.conf, det.vic.copy(), head_z)
    except Exception:
        _md_vdr_stat('err')
        return out
if MD_VDR_ON:
    PerDronePilot._gated_update = _md_vdr_gated_update
def _md_vdr_ctrl(fc, i):
    try:
        return fc.team.pilots[i].p.ctrl
    except Exception:
        return None
def _md_vdr_gate(T):
    return bool(MD_VDR_ON and _MAPSW_STATE.get('map') == 'mountain' and bool(getattr(T, 'mtn', False)))
def _md_vdr_gate_tick(T, R):
    g = R.get('gate_tk')
    if g is not None:
        return g
    g = MD_VDR_T0
    try:
        n = max(1, int(R['n']))
        pads = R['pads']
        cx = float(np.mean([q[0] for q in pads]))
        cy = float(np.mean([q[1] for q in pads]))
        c = np.asarray(T.clue_world, dtype=np.float64).reshape(-1)[:2]
        r = 80.0 * math.sqrt(min(n, 8) / 8.0)
        tgt = 1.03 * (math.hypot(cx - float(c[0]), cy - float(c[1])) / 3.0 + 0.70 * math.pi * r * r / (24.0 * 3.0) / n + 2.0)
        tgt = min(tgt, 0.95 * 60.0)
        tk = tgt * 50.0
        g = max(MD_VDR_T0, int(math.ceil(MD_VDR_FT * tk)))
        _MD_VDR_STATS['tgt_tk'] = float(tk)
    except Exception:
        _md_vdr_stat('err')
    R['gate_tk'] = g
    _MD_VDR_STATS['gate_tk'] = float(g)
    return g
def _md_vdr_useful(dep01):
    try:
        d = np.asarray(dep01, dtype=np.float32).reshape(256, 256)[128:, :] * (DEPTH_MAX_M - DEPTH_MIN_M) + DEPTH_MIN_M
        return float(np.mean((d >= MD_VDR_GLO) & (d <= MD_VDR_GHI))) >= MD_VDR_GFRAC
    except Exception:
        return False
def _md_vdr_new_state(n):
    return {'n': n, 'pend': {}, 'piggy': [], 'req': [], 'base': [0] * n, 'basesrv': [0] * n, 'env': [0] * n,
            'ours': [0] * n, 'last_scan': [-10 ** 6] * n, 'lpose': [None] * n, 'cand': {}, 'hits': [], 'spots': [],
            'pads': None}
def _md_vdr_eligible(fc, T, i):
    c = _md_vdr_ctrl(fc, i)
    try:
        tf = bool(T.frozen[i])
    except Exception:
        tf = False
    if c is None or tf or c.frozen or c.locked or c.mode not in ('search', 'navigation'):
        return None
    return c
def _md_vdr_process(fc, T, R, i, hit, kind, t):
    cd = R['cand'].get(i)
    if hit is None:
        if kind == 'verify' and cd is not None:
            cd['left'] -= 1
            cd['tl'] = t
            _md_vdr_stat('vfail')
            if cd['left'] <= 0:
                R['cand'].pop(i, None)
        return
    _md_vdr_stat('hit')
    if kind == 'piggy':
        _md_vdr_stat('phit')
    if _MD_VDR_STATS.get('lt1', 0) <= 0:
        _MD_VDR_STATS['lt1'] = float(t)
    h = (t, i, hit['x'], hit['y'], hit['top'], hit['lp'])
    for sp in R['spots']:
        if math.hypot(hit['x'] - sp['x'], hit['y'] - sp['y']) <= MD_VDR_AGREE:
            c = _md_vdr_ctrl(fc, sp['j'])
            if c is not None and c.locked and (not c.frozen) and \
                    math.hypot(float(c.locked_vic[0]) - sp['x'], float(c.locked_vic[1]) - sp['y']) <= CONSIST_M:
                if sp['j'] == i and MD_VDR_COMMIT:
                    sp['pts'].append(h)
                    del sp['pts'][:-4]
                    _md_vdr_inject(c, sp['pts'], t)
                    _md_vdr_stat('refine')
                else:
                    _md_vdr_stat('dup')
                R['hits'].append(h)
                del R['hits'][:-64]
                return
    agree = [g for g in R['hits'] if (g[0], g[1]) != (t, i) and t - g[0] <= MD_VDR_TTL
             and math.hypot(g[2] - hit['x'], g[3] - hit['y']) <= MD_VDR_AGREE]
    R['hits'].append(h)
    del R['hits'][:-64]
    if not agree:
        if kind in ('scan', 'verify', 'piggy') and _md_vdr_eligible(fc, T, i) is not None:
            R['cand'][i] = {'t': t, 'tl': t, 'left': MD_VDR_VN, 'x': hit['x'], 'y': hit['y']}
        return
    R['cand'].pop(i, None)
    if not MD_VDR_COMMIT:
        return
    c = _md_vdr_eligible(fc, T, i)
    if c is None:
        _md_vdr_stat('busy')
        return
    pts = agree[-3:] + [h]
    _md_vdr_inject(c, pts, t)
    R['spots'].append({'x': float(np.mean([g[2] for g in pts])), 'y': float(np.mean([g[3] for g in pts])),
                       'j': i, 't': t, 'pts': pts})
    del R['spots'][:-8]
    _cl = _MD_VDR_STATS.setdefault('clog', [])
    if len(_cl) < 3:
        _cl.append([float(t), R['spots'][-1]['x'], R['spots'][-1]['y'], float(i)])
    _md_vdr_stat('commit')
    _md_vdr_stat('arm')
    if _MD_VDR_STATS.get('t1', 0) <= 0:
        _MD_VDR_STATS['t1'] = float(t)
def _md_vdr_inject(c, pts, t):
    x = float(np.mean([g[2] for g in pts]))
    y = float(np.mean([g[3] for g in pts]))
    top = float(np.median([g[4] for g in pts]))
    lp = sum(1 for g in pts if g[5]) * 2 >= len(pts)
    hh = MD_VDR_LPH if lp else MD_VDR_UPH
    c.detector._md_vdr_inj = {'z': [x, y, top - 0.5 * hh], 'h': hh, 't': t}
_MD_VDR_BASE = DroneFlightController
class DroneFlightController(_MD_VDR_BASE):
    def __init__(self):
        _MD_VDR_BASE.__init__(self)
        self._md_vdr = None
        if MD_VDR_ON:
            _md_vdr_session()
    def reset(self):
        self._md_vdr = None
        _MD_VDR_STATS.clear()
        out = _MD_VDR_BASE.reset(self)
        try:
            for _pp in (getattr(getattr(self, 'team', None), 'pilots', None) or []):
                _d = getattr(getattr(getattr(_pp, 'p', None), 'ctrl', None), 'detector', None)
                if _d is not None:
                    _d._md_vdr_inj = None
        except Exception:
            pass
        return out
    def debug_state(self):
        out = _MD_VDR_BASE.debug_state(self)
        try:
            if isinstance(out, dict) and out:
                out['mdvdr_on'] = 1.0 if MD_VDR_ON else 0.0
                for _k in _MD_VDR_KEYS:
                    out['mdvdr_' + _k] = float(_MD_VDR_STATS.get(_k, 0))
                _cl = _MD_VDR_STATS.get('clog') or []
                for _q in range(3):
                    _c = _cl[_q] if _q < len(_cl) else (0.0, 0.0, 0.0, -1.0)
                    for _j, _nm in enumerate(('t', 'x', 'y', 'd')):
                        out['mdvdr_c%d%s' % (_q + 1, _nm)] = float(_c[_j])
        except Exception:
            pass
        return out
    def act(self, observation):
        _act_t0 = time.perf_counter()
        if not MD_VDR_ON:
            return _MD_VDR_BASE.act(self, observation)
        team = getattr(self, 'team', None)
        T = getattr(team, 'T', None)
        if T is None:
            return _MD_VDR_BASE.act(self, observation)
        obs = observation
        R = None
        try:
            t = int(T.tick)
            st = np.asarray(observation['state'], dtype=np.float64)
            st = st.reshape(-1, st.shape[-1]) if st.ndim > 1 else st.reshape(1, -1)
            n = int(st.shape[0])
            R = self._md_vdr
            if t == 0 or R is None or R['n'] != n:
                if t == 0:
                    _MD_VDR_STATS.clear()
                    try:
                        for _pp in (team.pilots or []):
                            _pp.p.ctrl.detector._md_vdr_inj = None
                    except Exception:
                        pass
                R = self._md_vdr = _md_vdr_new_state(n)
            if R['pads'] is None:
                R['pads'] = [tuple(float(v) for v in st[k, 0:3]) for k in range(n)]
            for _pp in (team.pilots or [])[:n]:
                _d = getattr(getattr(getattr(_pp, 'p', None), 'ctrl', None), 'detector', None)
                _in = getattr(_d, '_md_vdr_inj', None) if _d is not None else None
                if _in is not None and t - int(_in.get('t', t)) > MD_VDR_INJ_TTL:
                    _d._md_vdr_inj = None
                    _md_vdr_stat('inj_drop')
            ours_now, R['pend'] = R['pend'], {}
            pig_now, R['piggy'] = R['piggy'], []
            if ours_now or pig_now:
                tk0 = time.perf_counter()
                sess = _md_vdr_session()
                raw = np.asarray(observation['rgb'])
                rgb = raw.reshape(n, 256, 256, 3)
                dep = np.asarray(observation['depth'], dtype=np.float32).reshape(n, 256, 256)
                runs = 0
                for i, kind in sorted(ours_now.items()):
                    if kind == 'verify':
                        _md_vdr_stat('ver')
                    if sess is None or runs >= MD_VDR_RUNMAX:
                        continue
                    runs += 1
                    hit, why = _md_vdr_hit(sess, rgb[i], dep[i], st[i], R['pads'])
                    if hit is None:
                        _md_vdr_stat(why)
                    _md_vdr_process(self, T, R, i, hit, kind, t)
                for i in pig_now:
                    if sess is None or runs >= MD_VDR_RUNMAX or i in ours_now:
                        continue
                    if float(np.max(rgb[i])) <= 0.0:
                        continue
                    runs += 1
                    _md_vdr_stat('piggy')
                    hit, why = _md_vdr_hit(sess, rgb[i], dep[i], st[i], R['pads'])
                    if hit is None:
                        _md_vdr_stat(why)
                    _md_vdr_process(self, T, R, i, hit, 'piggy', t)
                if ours_now:
                    rgb2 = np.array(raw, copy=True)
                    v2 = rgb2.reshape(n, -1)
                    for i in ours_now:
                        v2[i] = 0
                    obs = dict(observation)
                    obs['rgb'] = rgb2
                ms = (time.perf_counter() - tk0) * 1e3
                if ms > _MD_VDR_STATS.get('ms_tick_max', 0.0):
                    _MD_VDR_STATS['ms_tick_max'] = ms
            if R['req']:
                R['req'] = [q for q in R['req'] if t - q[0] <= MD_VDR_ABUF_N]
                if R['req']:
                    s0 = np.asarray(observation['state'])
                    s2 = np.array(s0, copy=True)
                    v2 = s2.reshape(-1, s2.shape[-1]) if s2.ndim > 1 else s2.reshape(1, -1)
                    for (ts, i, a5) in R['req']:
                        k = MD_VDR_ABUF_LAST - MD_VDR_ABUF_STEP * (t - ts - 1)
                        if 0 <= k < v2.shape[1] and i < v2.shape[0]:
                            if abs(float(v2[i, k]) - 1.0) <= 1e-6:
                                v2[i, k] = a5
                            else:
                                _md_vdr_stat('abuf_miss')
                    if obs is observation:
                        obs = dict(observation)
                    obs['state'] = s2
        except Exception:
            _md_vdr_stat('err')
            if R is not None:
                R['pend'] = {}
        _dl_set = False
        try:
            _el = time.perf_counter() - _act_t0
            if team is not None and _el > 0.0:
                team.DEADLINE = max(0.05, float(type(team).DEADLINE) - _el)
                _dl_set = True
        except Exception:
            _dl_set = False
        try:
            action = _MD_VDR_BASE.act(self, obs)
        finally:
            if _dl_set:
                try:
                    del team.DEADLINE
                except Exception:
                    pass
        if R is None:
            return action
        try:
            out = np.array(action, dtype=np.float32, copy=True)
            flat = out.reshape(-1, out.shape[-1])
            n = min(flat.shape[0], R['n'])
            if flat.shape[1] < 6:
                return action
            gate = _md_vdr_gate(T) and t >= MD_VDR_T0 and t >= _md_vdr_gate_tick(T, R)
            for i in range(n):
                if flat[i, 5] > 0.5:
                    R['base'][i] += 1
                    if R['basesrv'][i] < MD_VDR_ENV_CAP:
                        R['basesrv'][i] += 1
                        if R['env'][i] >= MD_VDR_ENV_CAP:
                            _md_vdr_stat('over')
                            _md_vdr_stat('arm')
                    if R['env'][i] < MD_VDR_ENV_CAP:
                        R['env'][i] += 1
                    if gate and MD_VDR_PIGGY:
                        R['piggy'].append(i)
            if not gate:
                return action
            dep = None
            k_new = 0
            changed = False
            for jj in range(n):
                i = (t + jj) % n
                if flat[i, 5] > 0.5:
                    continue
                if R['ours'][i] >= MD_VDR_SMAX or R['base'][i] > MD_VDR_BASEMAX:
                    continue
                if R['env'][i] + 1 > MD_VDR_ENV_CAP - MD_VDR_RES:
                    continue
                c = _md_vdr_eligible(self, T, i)
                if c is None:
                    R['cand'].pop(i, None)
                    continue
                kind = None
                cd = R['cand'].get(i)
                if cd is not None and t - cd['t'] > MD_VDR_TTL:
                    R['cand'].pop(i, None)
                    cd = None
                if cd is not None:
                    if t - cd['tl'] >= MD_VDR_VGAP:
                        kind = 'verify'
                elif c.mode == 'search' and k_new < MD_VDR_K and t - R['last_scan'][i] >= MD_VDR_IV:
                    sti = st[i]
                    if max(abs(float(sti[3])), abs(float(sti[4]))) > math.radians(MD_VDR_TILT):
                        continue
                    lp_ = R['lpose'][i]
                    if lp_ is not None:
                        mv = float(np.linalg.norm(sti[0:3] - lp_[0]))
                        dy = abs((math.degrees(float(sti[5]) - lp_[1]) + 180.0) % 360.0 - 180.0)
                        if mv < MD_VDR_MOVE and dy < MD_VDR_TURN:
                            continue
                    if dep is None:
                        dep = np.asarray(observation['depth'], dtype=np.float32).reshape(-1, 256, 256)
                    if not _md_vdr_useful(dep[i]):
                        continue
                    kind = 'scan'
                    k_new += 1
                if kind is None:
                    continue
                R['req'].append((t, i, float(flat[i, 5])))
                flat[i, 5] = 1.0
                changed = True
                R['pend'][i] = kind
                R['ours'][i] += 1
                R['env'][i] += 1
                _md_vdr_stat('frames')
                if kind == 'scan':
                    R['last_scan'][i] = t
                    R['lpose'][i] = (np.asarray(st[i, 0:3], dtype=np.float64).copy(), float(st[i, 5]))
                    _md_vdr_stat('scan')
                else:
                    cd['tl'] = t
                _md_vdr_stat('arm')
                if R['ours'][i] > _MD_VDR_STATS.get('fmax', 0):
                    _MD_VDR_STATS['fmax'] = float(R['ours'][i])
            return out if changed else action
        except Exception:
            _md_vdr_stat('err')
            return action
N_VB_ON = os.environ.get('SWSAR_N_VB', '1') == '1'
N_VF_ON = os.environ.get('SWSAR_N_VF', '1') == '1'
N_PLC = os.environ.get('SWSAR_N_PLC', '0') == '1'
N_PLC_EPS = float(os.environ.get('SWSAR_N_PLC_EPS', '0.001'))
N_VB_IV = int(os.environ.get('SWSAR_N_VB_IV', '100'))
N_VB_SMAX = int(os.environ.get('SWSAR_N_VB_SMAX', '8'))
N_VB_VN = int(os.environ.get('SWSAR_N_VB_VN', '3'))
N_VF_VN = int(os.environ.get('SWSAR_N_VF_VN', '4'))
N_VF_TTL = int(os.environ.get('SWSAR_N_VF_TTL', '300'))
N_VF_SMAX = int(os.environ.get('SWSAR_N_VF_SMAX', '8'))
N_X_VGAP = int(os.environ.get('SWSAR_N_X_VGAP', '12'))
N_X_RES = int(os.environ.get('SWSAR_N_X_RES', '8'))
N_X_K = int(os.environ.get('SWSAR_N_X_K', '1'))
_N_KEYS = ('vb_arm', 'vf_arm', 'vb_t1', 'vf_t1', 'vb_over', 'vf_over', 'vb_blk', 'vf_blk', 'vb_fr', 'vf_fr', 'vb_hit',
           'vf_hit', 'plc_t', 'plc_src', 'plc_sup', 'busy', 'dup', 'runs', 'ms_max', 'err')
_N_STATS = {}
def _n_stat(k, d=1):
    _N_STATS[k] = _N_STATS.get(k, 0) + d
def _n_new(n):
    return {'n': n, 'pend': {}, 'ours': [0] * n, 'vb': [0] * n, 'vbs': [0] * n, 'vf': [0] * n,
            'lastfr': [-10 ** 6] * n, 'lscan': [-10 ** 6] * n, 'lpose': [None] * n, 'vbc': {}, 'vfc': {},
            'hits': [], 'envt': [0] * n, 'envb': [0] * n, 'plc_done': False, 'nudge': None}
def _n_hit(sess, rgb, dep, st, pads):
    k_run, k_ms = _MD_VDR_STATS.get('run'), _MD_VDR_STATS.get('ms_max')
    t0 = time.perf_counter()
    try:
        return _md_vdr_hit(sess, rgb, dep, st, pads)
    finally:
        ms = (time.perf_counter() - t0) * 1e3
        for _k, _v in (('run', k_run), ('ms_max', k_ms)):
            if _v is None:
                _MD_VDR_STATS.pop(_k, None)
            else:
                _MD_VDR_STATS[_k] = _v
        _n_stat('runs')
        if ms > _N_STATS.get('ms_max', 0.0):
            _N_STATS['ms_max'] = ms
def _n_on_spot(R, x, y):
    for sp in R['spots']:
        if math.hypot(x - sp['x'], y - sp['y']) <= MD_VDR_AGREE:
            return True
    return False
def _n_commit(fc, T, R, X, i, pts, t, srcs):
    c = _md_vdr_eligible(fc, T, i)
    if c is None:
        _n_stat('busy')
        return False
    if N_PLC and X['plc_done']:
        _n_stat('plc_sup')
        return False
    for sc in sorted(srcs):
        _n_stat(sc + '_arm')
        if _N_STATS.get(sc + '_t1', 0) <= 0:
            _N_STATS[sc + '_t1'] = float(t)
    if N_PLC:
        X['plc_done'] = True
        X['nudge'] = i
        _N_STATS['plc_t'] = float(t)
        _N_STATS['plc_src'] = float((1 if 'vb' in srcs else 0) + (2 if 'vf' in srcs else 0))
        return True
    _md_vdr_inject(c, pts, t)
    R['spots'].append({'x': float(np.mean([g[2] for g in pts])), 'y': float(np.mean([g[3] for g in pts])),
                       'j': i, 't': t, 'pts': pts})
    del R['spots'][:-8]
    return True
def _n_process(fc, T, R, X, i, hit, kind, t):
    src = 'vf' if kind == 'vf' else 'vb'
    cmap = X['vfc'] if kind == 'vf' else X['vbc']
    if hit is None:
        cd = cmap.get(i)
        if kind in ('vf', 'vbv') and cd is not None:
            cd['left'] -= 1
            if cd['left'] <= 0:
                cmap.pop(i, None)
        return
    _n_stat(src + '_hit')
    if _n_on_spot(R, hit['x'], hit['y']):
        _n_stat('dup')
        return
    h = (t, i, hit['x'], hit['y'], hit['top'], hit['lp'], src)
    agree = [g for g in list(R['hits']) + X['hits'] if (g[0], g[1]) != (t, i) and t - g[0] <= MD_VDR_TTL
             and math.hypot(g[2] - hit['x'], g[3] - hit['y']) <= MD_VDR_AGREE]
    X['hits'].append(h)
    del X['hits'][:-64]
    if not agree:
        if kind == 'vb':
            X['vbc'][i] = {'t': t, 'left': N_VB_VN, 'x': hit['x'], 'y': hit['y']}
        else:
            cd = cmap.get(i)
            if cd is not None:
                cd['left'] -= 1
                if cd['left'] <= 0:
                    cmap.pop(i, None)
        return
    X['vbc'].pop(i, None)
    X['vfc'].pop(i, None)
    pts = [g[:6] for g in agree[-3:]] + [h[:6]]
    srcs = {src} | {g[6] for g in agree[-3:] if len(g) > 6}
    _n_commit(fc, T, R, X, i, pts, t, srcs)
_N_BASE = DroneFlightController
class DroneFlightController(_N_BASE):
    def __init__(self):
        _N_BASE.__init__(self)
        self._n_x = None
    def reset(self):
        self._n_x = None
        _N_STATS.clear()
        return _N_BASE.reset(self)
    def debug_state(self):
        out = _N_BASE.debug_state(self)
        try:
            if isinstance(out, dict) and out and (N_VB_ON or N_VF_ON):
                out['n_vb_on'] = 1.0 if N_VB_ON else 0.0
                out['n_vf_on'] = 1.0 if N_VF_ON else 0.0
                out['n_plc_on'] = 1.0 if N_PLC else 0.0
                for _k in _N_KEYS:
                    out['n_' + _k] = float(_N_STATS.get(_k, 0))
        except Exception:
            pass
        return out
    def act(self, observation):
        if not (N_VB_ON or N_VF_ON):
            return _N_BASE.act(self, observation)
        team = getattr(self, 'team', None)
        T = getattr(team, 'T', None)
        if T is None:
            return _N_BASE.act(self, observation)
        tn0 = time.perf_counter()
        obs = observation
        X = None
        R = None
        t = -1
        hit0 = com0 = 0
        st = None
        try:
            t = int(T.tick)
            st = np.asarray(observation['state'], dtype=np.float64)
            st = st.reshape(-1, st.shape[-1]) if st.ndim > 1 else st.reshape(1, -1)
            n = int(st.shape[0])
            X = self._n_x
            if t == 0 or X is None or X['n'] != n:
                if t == 0:
                    _N_STATS.clear()
                X = self._n_x = _n_new(n)
            R = getattr(self, '_md_vdr', None)
            pend, X['pend'] = X['pend'], {}
            if pend and R is not None and R.get('pads') is not None:
                sess = _md_vdr_session()
                raw = np.asarray(observation['rgb'])
                rgb = raw.reshape(n, 256, 256, 3)
                dep = np.asarray(observation['depth'], dtype=np.float32).reshape(n, 256, 256)
                if sess is not None:
                    for i, kind in sorted(pend.items()):
                        hit, why = _n_hit(sess, rgb[i], dep[i], st[i], R['pads'])
                        _n_process(self, T, R, X, i, hit, kind, t)
                rgb2 = np.array(raw, copy=True)
                v2 = rgb2.reshape(n, -1)
                for i in pend:
                    v2[i] = 0
                obs = dict(observation)
                obs['rgb'] = rgb2
            hit0 = int(_MD_VDR_STATS.get('hit', 0))
            com0 = int(_MD_VDR_STATS.get('commit', 0))
        except Exception:
            _n_stat('err')
            if X is not None:
                X['pend'] = {}
        el = time.perf_counter() - tn0
        cls = None
        dl0 = None
        try:
            if team is not None and el > 0.0:
                cls = next((k for k in type(team).__mro__ if 'DEADLINE' in k.__dict__), None)
                if cls is not None:
                    dl0 = cls.__dict__['DEADLINE']
                    cls.DEADLINE = max(0.05, float(dl0) - el)
        except Exception:
            dl0 = None
        try:
            action = _N_BASE.act(self, obs)
        finally:
            if dl0 is not None:
                try:
                    cls.DEADLINE = dl0
                except Exception:
                    pass
        if X is None:
            return action
        if R is None:
            R = getattr(self, '_md_vdr', None)
            if R is None:
                return action
        try:
            out = np.array(action, dtype=np.float32, copy=True)
            flat = out.reshape(-1, out.shape[-1])
            n = min(flat.shape[0], X['n'], R['n'])
            if flat.shape[1] < 6:
                return action
            changed = False
            k_new = int(_MD_VDR_STATS.get('hit', 0)) - hit0
            r2_committed = int(_MD_VDR_STATS.get('commit', 0)) > com0
            if k_new > 0 and R['hits']:
                for h in list(R['hits'][-k_new:]):
                    if h[0] != t or _n_on_spot(R, h[2], h[3]):
                        continue
                    agree = [g for g in X['hits'] if (g[0], g[1]) != (h[0], h[1]) and t - g[0] <= MD_VDR_TTL
                             and math.hypot(g[2] - h[2], g[3] - h[3]) <= MD_VDR_AGREE]
                    if agree and not r2_committed:
                        pts = [g[:6] for g in agree[-3:]] + [tuple(h[:6])]
                        if _n_commit(self, T, R, X, int(h[1]), pts, t, {g[6] for g in agree[-3:]}):
                            X['vfc'].pop(int(h[1]), None)
                            continue
                    if N_VF_ON and not r2_committed:
                        X['vfc'][int(h[1])] = {'t': t, 'left': N_VF_VN, 'x': h[2], 'y': h[3]}
            gate = (_md_vdr_gate(T) and t >= MD_VDR_T0 and R.get('gate_tk') is not None and t >= R['gate_tk'])
            if gate and not (N_PLC and X['plc_done']):
                k_req = 0
                dep = None
                for jj in range(n):
                    if k_req >= N_X_K:
                        break
                    i = (t + jj) % n
                    if flat[i, 5] > 0.5:
                        continue
                    if X['envt'][i] + 1 > MD_VDR_ENV_CAP - N_X_RES:
                        continue
                    c = _md_vdr_eligible(self, T, i)
                    if c is None:
                        continue
                    if t - X['lastfr'][i] < N_X_VGAP:
                        continue
                    kind = None
                    cd = X['vfc'].get(i)
                    if cd is not None and (t - cd['t'] > N_VF_TTL or cd['left'] <= 0):
                        X['vfc'].pop(i, None)
                        cd = None
                    cb = X['vbc'].get(i)
                    if cb is not None and (t - cb['t'] > MD_VDR_TTL or cb['left'] <= 0):
                        X['vbc'].pop(i, None)
                        cb = None
                    if N_VF_ON and cd is not None and X['vf'][i] < N_VF_SMAX:
                        kind = 'vf'
                    elif N_VB_ON and cb is not None and X['vb'][i] < N_VB_SMAX + N_VB_VN:
                        kind = 'vbv'
                    elif (N_VB_ON and c.mode == 'search' and R['ours'][i] >= MD_VDR_SMAX
                          and R['base'][i] <= MD_VDR_BASEMAX and X['vbs'][i] < N_VB_SMAX
                          and t - X['lscan'][i] >= N_VB_IV and t - R['last_scan'][i] >= N_VB_IV):
                        sti = st[i] if (st is not None and i < st.shape[0]) else None
                        if sti is None or max(abs(float(sti[3])), abs(float(sti[4]))) > math.radians(MD_VDR_TILT):
                            continue
                        lp_ = X['lpose'][i] if X['lpose'][i] is not None else R['lpose'][i]
                        if lp_ is not None:
                            mv = float(np.linalg.norm(sti[0:3] - lp_[0]))
                            dy = abs((math.degrees(float(sti[5]) - lp_[1]) + 180.0) % 360.0 - 180.0)
                            if mv < MD_VDR_MOVE and dy < MD_VDR_TURN:
                                continue
                        if dep is None:
                            dep = np.asarray(observation['depth'], dtype=np.float32).reshape(-1, 256, 256)
                        if not _md_vdr_useful(dep[i]):
                            continue
                        kind = 'vb'
                    if kind is None:
                        continue
                    R['req'].append((t, i, float(flat[i, 5])))
                    flat[i, 5] = 1.0
                    changed = True
                    X['pend'][i] = kind
                    X['ours'][i] += 1
                    k_req += 1
                    if kind == 'vf':
                        X['vf'][i] += 1
                        _n_stat('vf_fr')
                    else:
                        X['vb'][i] += 1
                        _n_stat('vb_fr')
                        if kind == 'vb':
                            X['vbs'][i] += 1
                            X['lscan'][i] = t
                            X['lpose'][i] = (np.asarray(st[i, 0:3], dtype=np.float64).copy(), float(st[i, 5]))
            if X['nudge'] is not None:
                j = int(X['nudge'])
                X['nudge'] = None
                if 0 <= j < n:
                    sp_ = float(flat[j, 3])
                    flat[j, 3] = np.float32(sp_ - N_PLC_EPS if sp_ + N_PLC_EPS > 1.0 else sp_ + N_PLC_EPS)
                    changed = True
            for i in range(n):
                if flat[i, 5] <= 0.5:
                    continue
                X['lastfr'][i] = t
                if i in X['pend']:
                    X['envt'][i] += 1
                    continue
                srv_t = X['envt'][i] < MD_VDR_ENV_CAP
                srv_b = X['envb'][i] < MD_VDR_ENV_CAP
                if srv_t:
                    X['envt'][i] += 1
                if srv_b:
                    X['envb'][i] += 1
                if srv_b and not srv_t:
                    kk = 'blk' if i in R['pend'] else 'over'
                    if X['vb'][i] > 0:
                        _n_stat('vb_' + kk)
                    if X['vf'][i] > 0:
                        _n_stat('vf_' + kk)
            return out if changed else action
        except Exception:
            _n_stat('err')
            return action
MI_INV_ON = os.environ.get('SWSAR_MI_INV', '1') == '1'
MI_TAIL_ON = os.environ.get('SWSAR_MI_TAIL', '1') == '1'
MI_CMP = os.environ.get('SWSAR_MI_CMP', '0') == '1'
MI_TAIL_N = int(os.environ.get('SWSAR_MI_TAIL_N', '4'))
MI_TAIL_NOFF = frozenset(int(_x) for _x in os.environ.get('SWSAR_MI_TAIL_NOFF', '').split(',') if _x.strip())
MI_TAIL_T = float(os.environ.get('SWSAR_MI_TAIL_T', '38.0'))
MI_HANDBACK = os.environ.get('SWSAR_MI_HANDBACK', '1') == '1'
MI_T0 = int(os.environ.get('SWSAR_MI_T0', '500'))
MI_TCAL = int(os.environ.get('SWSAR_MI_TCAL', '400'))
MI_BUDGET = float(os.environ.get('SWSAR_MI_BUDGET', '100'))
MI_GATE = float(os.environ.get('SWSAR_MI_GATE', '0.20'))
MI_FIN = int(os.environ.get('SWSAR_MI_FIN', '40'))
MI_EVERY = int(os.environ.get('SWSAR_MI_EVERY', '10'))
MI_MAXATT = int(os.environ.get('SWSAR_MI_MAXATT', '30'))
MI_PASS2 = int(os.environ.get('SWSAR_MI_PASS2', '0'))
MI_LIB_FILE = 'mi_lib.npz'
_MI_ANY = MI_INV_ON or MI_CMP
_MI_KEYS = ('arm', 'ntail', 'act', 'swm', 'hbm', 'fov', 'tcal', 'rel', 'nbod', 'nkeep', 'att', 'plan', 'cov', 'treg', 'tfin', 'cost', 'wall',
            'pre_max', 'post_max', 'tail_max', 'frames', 'npts', 'inband', 'sweep', 'dtk', 'err')
_MI_STATS = {}
_MI_COST = {'cal': (0.0, 1.35e-04), 'ing': (8.5, 0.0), 'c': (2.269, 8.020e-05), 'r': (8.666, 2.588e-04), 'g': (-0.420, 4.625e-05), 'x': (-0.420, 4.625e-05),
            's': (0.0, 2.536e-05), 'l': (0.035, 2.692e-06), 'p': (4.315, 2.520e-03)}
_MI_SRC = "import math, time\nimport numpy as np\ntry:\n    import scipy.fft as sfft\n    from scipy.optimize import least_squares\n    from scipy import ndimage\n    HAVE_SCIPY = True\nexcept Exception:\n    sfft = least_squares = ndimage = None\n    HAVE_SCIPY = False\nKEYS = ['mountain_peak'] + [str(i) for i in (2, 3, 4, 5, 6, 7, 8, 9, 10)]\nHILLS = KEYS[1:]\nLIB = {}\nRATIO = 145.0 / 95.0\ndef load_lib(path):\n    if LIB:\n        return True\n    L = np.load(path)\n    for k in KEYS:\n        LIB[k] = (L[k + '_x'].astype(np.float64), L[k + '_y'].astype(np.float64), L[k + '_z'].astype(np.float64),\n                  float(L[k + '_zmin']), float(L[k + '_zmax']))\n    return True\ndef sv_of(key, s):\n    if key == 'mountain_peak':\n        return (s, s, s * RATIO)\n    return (s, s, round(s * 0.55 * 2) / 2)\ndef interp(key, xu, yu):\n    xs, ys, Z = LIB[key][:3]\n    fx = (xu - xs[0]) / (xs[1] - xs[0]); fy = (yu - ys[0]) / (ys[1] - ys[0])\n    ix = np.floor(fx).astype(np.int64); iy = np.floor(fy).astype(np.int64)\n    ok = (ix >= 0) & (iy >= 0) & (ix < xs.size - 1) & (iy < ys.size - 1)\n    ix = np.clip(ix, 0, xs.size - 2); iy = np.clip(iy, 0, ys.size - 2); tx = fx - ix; ty = fy - iy\n    z = (Z[iy, ix] * (1 - tx) * (1 - ty) + Z[iy, ix + 1] * tx * (1 - ty) + Z[iy + 1, ix] * (1 - tx) * ty + Z[iy + 1, ix + 1] * tx * ty)\n    return np.where(ok, z, np.nan)\ndef body_z(b, X, Y):\n    key, th, s, x0, y0, z0 = b[:6]\n    sv = b[6] if len(b) > 6 else sv_of(key, s)\n    t = math.radians(th); c, sn = math.cos(t), math.sin(t)\n    dx = np.asarray(X, float) - x0; dy = np.asarray(Y, float) - y0\n    a = c * dx + sn * dy; bb = -sn * dx + c * dy\n    return z0 + sv[1] * interp(key, a / sv[0], bb / sv[2])\ndef body_aabb_z(b):\n    key, th, s, x0, y0, z0 = b[:6]\n    sv = b[6] if len(b) > 6 else sv_of(key, s)\n    return z0 + sv[1] * LIB[key][3], z0 + sv[1] * LIB[key][4]\ndef body_foot_radius(b):\n    key, th, s = b[:3]\n    xs, ys = LIB[key][:2]; sv = sv_of(key, s)\n    return math.hypot(max(abs(xs[0]), abs(xs[-1])) * sv[0], max(abs(ys[0]), abs(ys[-1])) * sv[2])\nG = 4.0\n_MCACHE = {}\ndef model_raster(key, s, th, g=G):\n    k = (key, s, th, g)\n    r = _MCACHE.get(k)\n    if r is not None:\n        return r\n    xs, ys = LIB[key][:2]; sv = sv_of(key, s)\n    cu = np.array([[xs[0], ys[0]], [xs[-1], ys[0]], [xs[0], ys[-1]], [xs[-1], ys[-1]]])\n    loc = np.c_[cu[:, 0] * sv[0], cu[:, 1] * sv[2]]\n    c, sn = math.cos(math.radians(th)), math.sin(math.radians(th))\n    wx = c * loc[:, 0] - sn * loc[:, 1]; wy = sn * loc[:, 0] + c * loc[:, 1]\n    mx0 = math.floor(wx.min() / g) * g; my0 = math.floor(wy.min() / g) * g\n    Wn = int(math.ceil((wx.max() - mx0) / g)) + 1; Hn = int(math.ceil((wy.max() - my0) / g)) + 1\n    X = mx0 + (np.arange(Wn) + 0.5) * g; Y = my0 + (np.arange(Hn) + 0.5) * g\n    XX, YY = np.meshgrid(X, Y)\n    Z = body_z((key, th, s, 0.0, 0.0, 0.0), XX, YY)\n    Mm = np.isfinite(Z).astype(np.float64); M = np.where(Mm > 0, Z, 0.0)\n    r = (mx0, my0, M, Mm)\n    if g == G:\n        _MCACHE[k] = r\n    return r\ndef patch_raster(P, g=G):\n    x0 = math.floor(P[:, 0].min() / g) * g; y0 = math.floor(P[:, 1].min() / g) * g\n    j = np.floor((P[:, 0] - x0) / g).astype(int); i = np.floor((P[:, 1] - y0) / g).astype(int)\n    H, W = i.max() + 1, j.max() + 1\n    lin = i * W + j\n    cx = x0 + (j + 0.5) * g; cy = y0 + (i + 0.5) * g\n    dx = P[:, 0] - cx; dy = P[:, 1] - cy; z = P[:, 2]\n    n = np.bincount(lin, minlength=H * W).astype(float)\n    sx = np.bincount(lin, dx, H * W); sy = np.bincount(lin, dy, H * W); sz = np.bincount(lin, z, H * W)\n    sxx = np.bincount(lin, dx * dx, H * W); syy = np.bincount(lin, dy * dy, H * W); sxy = np.bincount(lin, dx * dy, H * W)\n    sxz = np.bincount(lin, dx * z, H * W); syz = np.bincount(lin, dy * z, H * W)\n    Pm = np.zeros(H * W); Wm = (n >= 3).astype(float)\n    for k in np.flatnonzero(n >= 3):\n        A = np.array([[sxx[k], sxy[k], sx[k]], [sxy[k], syy[k], sy[k]], [sx[k], sy[k], n[k]]])\n        rhs = np.array([sxz[k], syz[k], sz[k]])\n        try:\n            sol = np.linalg.solve(A + np.eye(3) * 1e-6, rhs)\n            Pm[k] = sol[2]\n        except np.linalg.LinAlgError:\n            Pm[k] = sz[k] / n[k]\n    return x0, y0, Pm.reshape(H, W), Wm.reshape(H, W)\ndef coarse_g(key, P, ths, ss, ntop=10, lam=4.0, g=None, chunk=24):\n    g = G if g is None else g\n    px0, py0, Pm, Wm = patch_raster(P, g)\n    Hp, Wp = Pm.shape; nW = Wm.sum()\n    if nW < 4:\n        return []\n    cands = []\n    for s in ss:\n        mods = [(th,) + model_raster(key, s, th, g) for th in ths]\n        Hm = max(m[4].shape[0] for m in mods); Wmm = max(m[4].shape[1] for m in mods)\n        sh = (sfft.next_fast_len(Hp + Hm, real=True), sfft.next_fast_len(Wp + Wmm, real=True))\n        fW = sfft.rfft2(Wm, sh); fWP = sfft.rfft2(Wm * Pm, sh); fWPP = sfft.rfft2(Wm * Pm * Pm, sh)\n        for q, (th, mx0, my0, M, Mm) in enumerate(mods):\n            fM = np.conj(sfft.rfft2(M, sh)); fMm = np.conj(sfft.rfft2(Mm, sh)); fM2 = np.conj(sfft.rfft2(M * M, sh))\n            stack = np.stack([fW * fMm, fWP * fMm, fWPP * fMm, fW * fM, fW * fM2, fWP * fM])\n            n, SP, SPP, SM, SMM, SPM = sfft.irfft2(stack, sh)\n            ok = n > 0.5\n            nn = np.where(ok, n, 1.0)\n            res = (SPP - 2 * SPM + SMM - (SP - SM) ** 2 / nn) / nn\n            cost = np.where(ok, np.maximum(res, 0) + lam * (nW - n) / nW * 25.0, np.inf)\n            k = int(np.argmin(cost)); ti, tj = np.unravel_index(k, sh)\n            if ti >= sh[0] - Hm:\n                ti -= sh[0]\n            if tj >= sh[1] - Wmm:\n                tj -= sh[1]\n            x0 = px0 + tj * g - mx0; y0 = py0 + ti * g - my0\n            zb = float((SP.flat[k] - SM.flat[k]) / max(n.flat[k], 1e-9))\n            cands.append((float(cost.flat[k]), key, th, s, x0, y0, zb))\n            if (q + 1) % chunk == 0:\n                yield ('c', chunk * sh[0] * sh[1])\n        yield ('c', (len(mods) % chunk) * sh[0] * sh[1] + 1)\n    cands.sort(key=lambda c: c[0])\n    return cands[:ntop]\ndef _resid(b, P, up_w=0.15):\n    z = body_z(b, P[:, 0], P[:, 1])\n    r = P[:, 2] - z\n    r = np.where(np.isfinite(r), r, 3.0)\n    return np.where(r > 0.3, 0.3 + (r - 0.3) * up_w, r)\n_WORK = [0.0]\ndef refine(c, P, free_scale=True):\n    _, key, th, s, x0, y0, zb = c\n    is_peak = key == 'mountain_peak'\n    if is_peak and free_scale:\n        def f(q):\n            return _resid((key, q[0], q[1], q[2], q[3], q[4]), P)\n        q0 = np.array([th, s, x0, y0, zb]); lb = [th - 10, s * 0.85, x0 - 10, y0 - 10, zb - 12]; ub = [th + 10, s * 1.15, x0 + 10, y0 + 10, zb + 12]\n    else:\n        def f(q):\n            return _resid((key, q[0], s, q[1], q[2], q[3]), P)\n        q0 = np.array([th, x0, y0, zb]); lb = [th - 10, x0 - 10, y0 - 10, zb - 12]; ub = [th + 10, x0 + 10, y0 + 10, zb + 12]\n    try:\n        r = least_squares(f, q0, bounds=(lb, ub), loss='soft_l1', f_scale=0.1, max_nfev=120)\n        q = r.x\n        _WORK[0] = float((int(r.nfev) + int(r.njev or 0) * (len(q0) + 1)) * len(P))\n    except Exception:\n        q = q0\n        _WORK[0] = float(len(P))\n    if is_peak and free_scale:\n        b = (key, float(q[0]), float(q[1]), float(q[2]), float(q[3]), float(q[4]))\n    else:\n        b = (key, float(q[0]), float(s), float(q[1]), float(q[2]), float(q[3]))\n    rr = _resid(b, P)\n    return b, float(np.sqrt(np.mean(np.minimum(rr ** 2, 1.0))))\ndef global_check(b, Q, F, tol_in=0.12, tol_v=0.25):\n    zq = body_z(b, Q[:, 0], Q[:, 1]); rq = Q[:, 2] - zq\n    fin = np.isfinite(rq)\n    inl = fin & (np.abs(rq) < tol_in)\n    vp = int((fin & (rq < -tol_v)).sum())\n    vf = 0\n    if F is not None and len(F):\n        zf = body_z(b, F[:, 0], F[:, 1]); rf = F[:, 2] - zf\n        vf = int((np.isfinite(rf) & (rf < -tol_v)).sum())\n    return inl, vp, vf\nclass IncInverter:\n    def __init__(self, centre, roi_r=40.0, R=12.0, max_attempts=18, rng=0, peak_rot=6.0, hill_rot=8.0, nscale=10,\n                 ntop_peak=8, ntop_hill=8, min_inl=150, vfrac=0.02, vabs=10, seed_mode='mixed', zband=20.0, n_high=6,\n                 R_low=9.0, r_low_pick=25.0, grow_iters=3, reopen=1.5):\n        self.centre = np.asarray(centre, float)\n        self.roi_r = roi_r; self.R = R; self.max_attempts = max_attempts\n        self.rng = np.random.RandomState(rng)\n        self.peak_ths = np.arange(0, 360, peak_rot); self.hill_ths = np.arange(0, 360, hill_rot)\n        A0, A1 = 95 * 0.6 * 0.85, 95 * 0.8 * 1.25\n        self.peak_ss = np.round(np.exp(np.linspace(math.log(A0), math.log(A1), nscale)), 2)\n        self.hill_ss = np.round(np.linspace(4.0, 17.5, nscale) * 2) / 2\n        self.ntop_peak = ntop_peak; self.ntop_hill = ntop_hill; self.min_inl = min_inl; self.vfrac = vfrac; self.vabs = vabs\n        self.seed_mode = seed_mode; self.zband = zband\n        self.n_high = n_high; self.R_low = R_low; self.r_low_pick = r_low_pick; self.grow_iters = grow_iters\n        self.reopen = reopen\n        self.bodies = []; self.log = []; self.tried_discs = []\n        self.Q = np.zeros((0, 3)); self.F = None\n        self.explained = np.zeros(0, bool); self.tried = np.zeros(0, bool)\n        self.dc = np.zeros(0); self.in_roi = np.zeros(0, bool)\n        self.n_refresh = 0; self.n_small = 0\n    def set_data_g(self, Q, F):\n        self.Q = Q; self.F = F; self.n_refresh += 1\n        d = np.hypot(Q[:, 0] - self.centre[0], Q[:, 1] - self.centre[1]); self.dc = d\n        self.in_roi = d <= self.roi_r\n        tr = np.zeros(len(Q), bool)\n        keep = []\n        for (x, y, r, n0) in self.tried_discs:\n            m = np.hypot(Q[:, 0] - x, Q[:, 1] - y) <= r\n            if self.reopen and n0 > 0 and m.sum() >= self.reopen * n0:\n                continue\n            tr |= m\n            keep.append((x, y, r, n0))\n        self.tried_discs = keep\n        self.tried = tr\n        ex = np.zeros(len(Q), bool)\n        for b in self.bodies:\n            zq = body_z(b, Q[:, 0], Q[:, 1]); rq = Q[:, 2] - zq\n            ex |= np.isfinite(rq) & (np.abs(rq) < 0.12)\n            yield ('x', len(Q))\n        self.explained = ex\n    def _try(self, s, r):\n        m = np.hypot(self.Q[:, 0] - s[0], self.Q[:, 1] - s[1]) <= r\n        self.tried |= m\n        self.tried_discs.append((float(s[0]), float(s[1]), float(r), int(m.sum())))\n    def _seed(self):\n        cand = np.flatnonzero(self.in_roi & ~self.explained & ~self.tried)\n        if cand.size < 50:\n            return None\n        if self.seed_mode == 'centre':\n            k = cand[np.argsort(self.dc[cand])[:max(1, cand.size // 20)]]\n        elif self.seed_mode == 'mixed' and len(self.log) >= self.n_high:\n            near = cand[self.dc[cand] <= self.r_low_pick]\n            if near.size < 30:\n                near = cand\n            return int(self.rng.choice(near))\n        else:\n            k = cand[np.argsort(-self.Q[cand, 2])[:max(1, cand.size // 200)]]\n        return int(self.rng.choice(k))\n    def attempt_g(self):\n        si = self._seed()\n        if si is None:\n            return False\n        s = self.Q[si].copy()\n        d = np.hypot(self.Q[:, 0] - s[0], self.Q[:, 1] - s[1])\n        R = self.R_low if (self.seed_mode == 'mixed' and len(self.log) >= self.n_high) else self.R\n        nb = np.flatnonzero((d <= R) & ~self.explained & (self.Q[:, 2] >= s[2] - self.zband))\n        if nb.size < 60:\n            self._try(s, 4.0)\n            self.n_small += 1\n            return True\n        Pn = self.Q[nb]\n        if len(Pn) > 900:\n            Pn = Pn[self.rng.choice(len(Pn), 900, replace=False)]\n        yield ('s', len(self.Q))\n        cands = yield from coarse_g('mountain_peak', Pn, self.peak_ths, self.peak_ss, ntop=self.ntop_peak)\n        hc = []\n        for key in HILLS:\n            hc += yield from coarse_g(key, Pn, self.hill_ths, self.hill_ss, ntop=3)\n        hc.sort(key=lambda c: c[0]); cands += hc[:self.ntop_hill]\n        Pr = Pn if len(Pn) <= 400 else Pn[self.rng.choice(len(Pn), 400, replace=False)]\n        best = None\n        for c in cands:\n            key = c[1]\n            if key == 'mountain_peak':\n                trials = [refine(c, Pr)]\n                yield ('r', _WORK[0])\n            else:\n                ss = np.array([c[3] + ds for ds in np.arange(-2.0, 2.01, 0.5) if 4.0 <= c[3] + ds <= 17.5])\n                ths = (c[2] + np.arange(-12.0, 12.01, 2.0)) % 360\n                loc = yield from coarse_g(key, Pn, ths, ss, ntop=3, g=2.0)\n                trials = []\n                for lc in loc:\n                    trials.append(refine(lc, Pr))\n                    yield ('r', _WORK[0])\n                if not trials:\n                    trials = [refine(c, Pr)]\n                    yield ('r', _WORK[0])\n                if trials:\n                    bb = min(trials, key=lambda t: t[1])[0]\n                    for ds in (-0.5, 0.5, -1.0, 1.0):\n                        if 4.0 <= bb[2] + ds <= 17.5:\n                            trials.append(refine((0.0, key, bb[1], bb[2] + ds, bb[3], bb[4], bb[5]), Pr))\n                            yield ('r', _WORK[0])\n            for b, rms in trials:\n                inl, vp, vf = global_check(b, self.Q, self.F)\n                ni = int(inl.sum()); nv = vp + vf\n                sc = ni - 5 * nv\n                if best is None or sc > best[0]:\n                    best = (sc, b, ni, vp, vf, inl, rms)\n                yield ('g', len(self.Q) + (len(self.F) if self.F is not None else 0))\n        if best is not None and self.grow_iters > 0:\n            for _ in range(self.grow_iters):\n                b0 = best[1]\n                zq = body_z(b0, self.Q[:, 0], self.Q[:, 1]); rq = self.Q[:, 2] - zq\n                wide = np.flatnonzero(np.isfinite(rq) & (np.abs(rq) < 0.6) & ~self.explained & (d <= 45.0))\n                if wide.size < 60:\n                    break\n                Pw = self.Q[wide] if wide.size <= 600 else self.Q[self.rng.choice(wide, 600, replace=False)]\n                cands2 = [refine((0.0,) + tuple(b0[:6]), Pw)]\n                yield ('r', _WORK[0])\n                if b0[0] != 'mountain_peak':\n                    for ds in (-0.5, 0.5, -1.0, 1.0):\n                        if 4.0 <= b0[2] + ds <= 17.5:\n                            cands2.append(refine((0.0, b0[0], b0[1], b0[2] + ds, b0[3], b0[4], b0[5]), Pw))\n                            yield ('r', _WORK[0])\n                improved = False\n                for b, rms in cands2:\n                    inl, vp, vf = global_check(b, self.Q, self.F)\n                    ni = int(inl.sum()); sc = ni - 5 * (vp + vf)\n                    if sc > best[0] + 5:\n                        best = (sc, b, ni, vp, vf, inl, rms); improved = True\n                    yield ('g', len(self.Q) + (len(self.F) if self.F is not None else 0))\n                if not improved:\n                    break\n        ok = best is not None and best[2] >= self.min_inl and (best[3] + best[4]) <= self.vfrac * best[2] + self.vabs\n        rec = dict(seed_pt=[round(float(v), 2) for v in s], nb=int(nb.size))\n        if best is not None:\n            rec.update(body=[best[1][0]] + [round(v, 4) for v in best[1][1:]], inl=best[2], vp=best[3], vf=best[4], rms=round(best[6], 3), acc=bool(ok))\n        self.log.append(rec)\n        if ok:\n            self.bodies.append(best[1])\n            self.explained |= best[5]\n            self._try(s, 3.0)\n        else:\n            self._try(s, 6.0)\n        return True\n    def run_all(self):\n        for _ in range(self.max_attempts):\n            g = self.attempt_g()\n            try:\n                while True:\n                    next(g)\n            except StopIteration as e:\n                if e.value is False:\n                    break\n        return self.bodies\n_OFF = [(0.0, 0.0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]\ndef compose(bodies, X, Y):\n    if not bodies:\n        return np.full(np.shape(X), np.nan), np.full(np.shape(X), -1)\n    Z = np.stack([body_z(b, X, Y) for b in bodies])\n    allnan = np.all(np.isnan(Z), axis=0)\n    top = np.where(allnan, -1, np.argmax(np.where(np.isnan(Z), -np.inf, Z), axis=0))\n    Zt = np.where(allnan, np.nan, np.nanmax(np.where(np.isnan(Z), -np.inf, Z), axis=0))\n    return Zt, top\ndef _plane_slope(X, Y, zc, bodies):\n    out = []\n    for r in (0.4, 1.0):\n        xs = []; ys = []; zs = []\n        for dx, dy in _OFF:\n            Xo = X + dx * r; Yo = Y + dy * r\n            Zo, _ = compose(bodies, Xo, Yo)\n            xs.append(Xo); ys.append(Yo); zs.append(Zo)\n        xs = np.stack(xs); ys = np.stack(ys); zs = np.stack(zs)\n        keep = np.isfinite(zs) & (np.abs(zs - zc[None, :]) <= 2.0)\n        w = keep.astype(float)\n        n = w.sum(0)\n        mx = (w * xs).sum(0) / np.maximum(n, 1); my = (w * ys).sum(0) / np.maximum(n, 1); mz = (w * np.nan_to_num(zs)).sum(0) / np.maximum(n, 1)\n        dx_ = xs - mx; dy_ = ys - my; dz_ = np.nan_to_num(zs) - mz\n        sxx = (w * dx_ * dx_).sum(0); syy = (w * dy_ * dy_).sum(0); sxy = (w * dx_ * dy_).sum(0)\n        sxz = (w * dx_ * dz_).sum(0); syz = (w * dy_ * dz_).sum(0)\n        det = sxx * syy - sxy * sxy\n        a = np.where(np.abs(det) > 1e-12, (sxz * syy - syz * sxy) / np.where(np.abs(det) > 1e-12, det, 1), 0.0)\n        bq = np.where(np.abs(det) > 1e-12, (syz * sxx - sxz * sxy) / np.where(np.abs(det) > 1e-12, det, 1), 0.0)\n        ang = np.degrees(np.arccos(1.0 / np.sqrt(1 + a * a + bq * bq)))\n        ang = np.where(n >= 3, ang, 0.0)\n        out.append(ang)\n    return np.maximum(out[0], out[1])\ndef predict_valid(bodies, X, Y, nz_min=0.70, slope_max=22.0):\n    X = np.asarray(X, float); Y = np.asarray(Y, float)\n    Zt, top = compose(bodies, X, Y)\n    valid = np.zeros(X.shape, bool)\n    if not bodies:\n        return valid, top, Zt\n    ok = np.isfinite(Zt)\n    h = 0.05\n    nzv = np.zeros(X.shape)\n    upper = np.zeros(X.shape, bool)\n    for bi, b in enumerate(bodies):\n        m = ok & (top == bi)\n        if not m.any():\n            continue\n        zx1 = body_z(b, X[m] + h, Y[m]); zx0 = body_z(b, X[m] - h, Y[m]); zy1 = body_z(b, X[m], Y[m] + h); zy0 = body_z(b, X[m], Y[m] - h)\n        gx = (zx1 - zx0) / (2 * h); gy = (zy1 - zy0) / (2 * h)\n        nzv[m] = np.nan_to_num(1.0 / np.sqrt(1 + gx * gx + gy * gy), nan=0.0)\n        lo, hi = body_aabb_z(b)\n        upper[m] = (np.abs(Zt[m] - hi) <= 0.2) | (Zt[m] >= lo + 0.5 * max(0.5, hi - lo))\n    cand = ok & (nzv >= nz_min) & upper\n    if cand.any():\n        sl = _plane_slope(X[cand], Y[cand], Zt[cand], bodies)\n        vv = np.zeros(X.shape, bool); vv[cand] = sl <= slope_max\n        valid = vv\n    return valid, top, Zt\ndef cam_axes(r, pch, y):\n    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(pch), np.sin(pch), np.cos(y), np.sin(y)\n    fwd = np.array([cy * cp, sy * cp, -sp]); up = np.array([cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr])\n    return fwd, up, np.cross(fwd, up)\nFOV_GRID = np.tan(np.radians(np.linspace(43.8, 46.2, 25)))\nclass DepthAcc:\n    HALF = 100.0; C = 0.5; NPIX = 32\n    DMIN, DMAX = 0.5, 30.0\n    NEAR = 0.05; CAM_F = 0.13; CAM_U = 0.05; MATE_R = 0.5\n    def __init__(self):\n        NG = self.NG = int(2 * self.HALF / self.C)\n        self.PX = np.full((NG, NG), np.nan, np.float64); self.PY = self.PX.copy(); self.PZ = self.PX.copy()\n        self.seen = np.zeros((NG, NG), bool)\n        self.FZ = np.full((NG, NG), np.inf); self.FX = np.full((NG, NG), np.nan); self.FY = self.FX.copy()\n        st = self.NPIX\n        pix = np.arange(st) * (256 // st) + (256 // st) // 2\n        rr, cc = np.meshgrid(pix, pix, indexing='ij')\n        self.rr = rr.ravel(); self.cc = cc.ravel()\n        self.u = self.cc / 128.0 - 1.0\n        self.v = 1.0 - (self.rr + 1.0) / 128.0\n        self.frames = 0; self.npts = 0; self.version = 0\n        self.s = None; self.raw = []; self.cal = None; self.ready = False\n    def add_frame(self, dep01, pos, rpy, mates):\n        d = np.asarray(dep01, dtype=np.float32).reshape(256, 256)[self.rr, self.cc].copy()\n        fwd, up, right = cam_axes(float(rpy[0]), float(rpy[1]), float(rpy[2]))\n        cam = np.asarray(pos, float) + fwd * self.CAM_F + up * self.CAM_U\n        rec = (cam, fwd, up, right, d, np.asarray(mates, float).reshape(-1, 3).copy())\n        if self.s is None:\n            self.raw.append(rec)\n        else:\n            self._ingest(*rec)\n    def _spread(self, A, B, f, s):\n        P = A + s * B\n        ix = np.floor(P[:, 0] / self.C).astype(np.int64); iy = np.floor(P[:, 1] / self.C).astype(np.int64)\n        key = ix * 1000003 + iy\n        o = np.lexsort((f, key)); key = key[o]; z = P[o, 2]; ff = f[o]\n        if key.size < 2:\n            return 9.0\n        st_ = np.r_[0, np.flatnonzero(key[1:] != key[:-1]) + 1]\n        en = np.r_[st_[1:], key.size] - 1\n        multi = ff[st_] != ff[en]\n        if not multi.any():\n            return 9.0\n        sp = (np.maximum.reduceat(z, st_) - np.minimum.reduceat(z, st_))[multi]\n        return float(np.median(sp))\n    def calib_g(self):\n        As = []; Bs = []; fs = []\n        for k, (cam, fwd, up, right, d, mates) in enumerate(self.raw):\n            zc = d.astype(np.float64) * (self.DMAX - self.DMIN) + self.DMIN\n            hit = (d < 0.9999) & (d > 0.0)\n            D1 = self.u[:, None] * right[None, :] + self.v[:, None] * up[None, :]\n            A = cam[None, :] + zc[:, None] * fwd[None, :]\n            B = zc[:, None] * D1\n            P = A + B\n            m = hit.copy()\n            for mp in mates:\n                m &= np.sum((P - mp[None, :]) ** 2, axis=1) >= 1.0\n            As.append(A[m]); Bs.append(B[m]); fs.append(np.full(int(m.sum()), k))\n        A = np.concatenate(As) if As else np.zeros((0, 3)); B = np.concatenate(Bs) if Bs else np.zeros((0, 3))\n        f = np.concatenate(fs) if fs else np.zeros(0, int)\n        yield ('cal', A.shape[0])\n        sc = []\n        for s in FOV_GRID:\n            sc.append(self._spread(A, B, f, float(s)))\n            yield ('cal', A.shape[0])\n        sc = np.asarray(sc)\n        k = int(np.argmin(sc))\n        s_best = float(FOV_GRID[k])\n        if 0 < k < len(FOV_GRID) - 1:\n            y0, y1, y2 = sc[k - 1], sc[k], sc[k + 1]\n            den = y0 - 2 * y1 + y2\n            if den > 1e-9:\n                off = 0.5 * (y0 - y2) / den\n                s_best = float(FOV_GRID[k] + max(-1.0, min(1.0, off)) * (FOV_GRID[k + 1] - FOV_GRID[k]))\n        if not np.isfinite(s_best) or A.shape[0] < 2000:\n            s_best = 1.0\n        self.s = s_best\n        self.cal = dict(s=s_best, k=k, n=int(A.shape[0]), frames=len(self.raw), fov=float(2 * np.degrees(np.arctan(s_best))))\n        raw, self.raw = self.raw, []\n        for rec in raw:\n            self._ingest(*rec)\n            yield ('ing', 1)\n        self.ready = True\n    def _ingest(self, cam, fwd, up, right, d32, mates):\n        d = d32.astype(np.float64)\n        zc = d * (self.DMAX - self.DMIN) + self.DMIN\n        s = self.s\n        dirs = fwd[None, :] + s * (self.u[:, None] * right[None, :] + self.v[:, None] * up[None, :])\n        near = d <= 0.0\n        hit = (d < 0.9999) & ~near\n        P = cam[None, :] + zc[:, None] * dirs\n        HALF, C, NG = self.HALF, self.C, self.NG\n        isd = np.zeros(len(d), bool)\n        for mp in mates:\n            isd |= np.sum((P - mp[None, :]) ** 2, axis=1) < self.MATE_R ** 2\n        ok = hit & ~isd\n        if ok.any():\n            q = P[ok]\n            ix = np.floor((q[:, 0] + HALF) / C).astype(int); iy = np.floor((q[:, 1] + HALF) / C).astype(int)\n            m = (ix >= 0) & (ix < NG) & (iy >= 0) & (iy < NG)\n            q, ix, iy = q[m], ix[m], iy[m]\n            new = ~self.seen[iy, ix]\n            if new.any():\n                lin = iy[new] * NG + ix[new]; _, first = np.unique(lin, return_index=True)\n                sel = np.flatnonzero(new)[first]\n                self.PX[iy[sel], ix[sel]] = q[sel, 0]; self.PY[iy[sel], ix[sel]] = q[sel, 1]; self.PZ[iy[sel], ix[sel]] = q[sel, 2]\n                self.seen[iy[sel], ix[sel]] = True\n                self.npts += int(sel.size)\n        Ld = np.linalg.norm(dirs, axis=1); dn = dirs / Ld[:, None]\n        A0 = cam[None, :] + self.NEAR * dirs\n        Lf = np.where(hit, (zc - self.NEAR) * Ld - 0.4, (self.DMAX - self.NEAR) * Ld)\n        Lf = np.where(near, 0.0, Lf)\n        nsm = np.maximum(np.floor(Lf / 0.5).astype(int), 0); tot = int(nsm.sum())\n        if tot > 0:\n            ridx = np.repeat(np.arange(len(nsm)), nsm)\n            off = np.arange(tot) - np.repeat(np.cumsum(nsm) - nsm, nsm)\n            s_ = A0[ridx] + dn[ridx] * (0.5 * (off + 1))[:, None]\n            ix = np.floor((s_[:, 0] + HALF) / C).astype(int); iy = np.floor((s_[:, 1] + HALF) / C).astype(int)\n            m = (ix >= 0) & (ix < NG) & (iy >= 0) & (iy < NG)\n            s_, ix, iy = s_[m], ix[m], iy[m]\n        if tot > 0 and s_.shape[0] > 0:\n            lin = iy * NG + ix\n            o = np.lexsort((s_[:, 2], lin))\n            lo = o[np.r_[True, lin[o][1:] != lin[o][:-1]]]\n            cz = s_[lo, 2]; cl = lin[lo]; cy_, cx_ = np.divmod(cl, NG)\n            better = cz < self.FZ[cy_, cx_]\n            self.FZ[cy_[better], cx_[better]] = cz[better]; self.FX[cy_[better], cx_[better]] = s_[lo[better], 0]\n            self.FY[cy_[better], cx_[better]] = s_[lo[better], 1]\n        self.frames += 1; self.version += 1\n    def snapshot(self, centre, R=90.0):\n        Q = np.c_[self.PX[self.seen], self.PY[self.seen], self.PZ[self.seen]]\n        okf = np.isfinite(self.FZ)\n        F = np.c_[self.FX[okf], self.FY[okf], self.FZ[okf]]\n        if centre is not None:\n            Q = Q[np.hypot(Q[:, 0] - centre[0], Q[:, 1] - centre[1]) <= R]\n            F = F[np.hypot(F[:, 0] - centre[0], F[:, 1] - centre[1]) <= R]\n        return Q, F\nH_SWEEP = 3.0; V_SWEEP = 0.85; V_MAX3 = 0.95; V_TRANSIT = 2.6; H_TRANSIT = 3.5; SWATH = 1.7\nSIM_DT = 1.0 / 50.0; SPEED_LIMIT = 3.0\nclass Surface:\n    def __init__(self, lo, hi, step=0.5):\n        self.lo = np.asarray(lo, float); self.step = step\n        xs = np.arange(lo[0], hi[0] + 1e-6, step); ys = np.arange(lo[1], hi[1] + 1e-6, step)\n        self.XX, self.YY = np.meshgrid(xs, ys)\n        self.Z = np.full(self.XX.shape, np.nan)\n        self.cx, self.cy = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]); self.rad = 0.5 * math.hypot(hi[0] - lo[0], hi[1] - lo[1])\n        self.nx = xs.size; self.ny = ys.size\n    def add_body(self, b):\n        if math.hypot(b[3] - self.cx, b[4] - self.cy) > self.rad + body_foot_radius(b):\n            return\n        zb = body_z(b, self.XX, self.YY)\n        self.Z = np.fmax(self.Z, zb)\n    def add_pts(self, pts, dil=1):\n        if pts is not None and len(pts):\n            lo, step = self.lo, self.step\n            ix = np.floor((pts[:, 0] - lo[0]) / step + 0.5).astype(int); iy = np.floor((pts[:, 1] - lo[1]) / step + 0.5).astype(int)\n            m = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)\n            Zp = np.full(self.Z.shape, -np.inf); np.maximum.at(Zp, (iy[m], ix[m]), pts[m, 2])\n            Zp = np.where(np.isfinite(Zp), Zp, np.nan)\n            Zd = Zp.copy()\n            for dy in range(-dil, dil + 1):\n                for dx in range(-dil, dil + 1):\n                    Zd = np.fmax(Zd, np.roll(np.roll(Zp, dy, 0), dx, 1))\n            self.Z = np.fmax(self.Z, Zd)\n        self.XX = self.YY = None\n    def at(self, x, y):\n        ix = int(round((x - self.lo[0]) / self.step)); iy = int(round((y - self.lo[1]) / self.step))\n        if 0 <= ix < self.nx and 0 <= iy < self.ny:\n            v = self.Z[iy, ix]\n            return float(v) if np.isfinite(v) else None\n        return None\n    def max_disc(self, x, y, r):\n        k = int(math.ceil(r / self.step))\n        ix = int(round((x - self.lo[0]) / self.step)); iy = int(round((y - self.lo[1]) / self.step))\n        a = self.Z[max(0, iy - k):iy + k + 1, max(0, ix - k):ix + k + 1]\n        if a.size == 0 or not np.isfinite(a).any():\n            return None\n        return float(np.nanmax(a))\n    def max_path(self, x0, y0, x1, y1, r=2.0):\n        L = math.hypot(x1 - x0, y1 - y0); n = max(2, int(L / 1.0) + 1); best = None\n        for t in np.linspace(0, 1, n):\n            v = self.max_disc(x0 + t * (x1 - x0), y0 + t * (y1 - y0), r)\n            if v is not None:\n                best = v if best is None else max(best, v)\n        return best\nclass Tail:\n    def __init__(self, plan_xy, plan_w, surf, tail_ids, t0):\n        self.P = np.asarray(plan_xy, float).reshape(-1, 2); self.W = np.asarray(plan_w, float).reshape(-1)\n        self.cov = np.zeros(len(self.P), bool); self.surf = surf\n        self.ids = list(tail_ids); self.active = set(tail_ids)\n        self.tgt = {i: None for i in tail_ids}; self.mode = {i: 'transit' for i in tail_ids}\n        self.ztr = {i: None for i in tail_ids}\n        self.t0 = t0; self.drone_ticks = 0; self.sweep_ticks = 0; self.inband_ticks = 0\n        self.handback = {}; self.released = []\n        self.dwell = {}; self.gaveup = 0\n        self.dscale = 15.0\n        self.vcmd = {}; self.acc_xy = 1.5; self.acc_z = 2.0\n        self.wq = {i: [] for i in tail_ids}; self.blob_of = {i: None for i in tail_ids}; self.first_wp = {i: False for i in tail_ids}\n        self._make_blobs()\n    def _make_blobs(self):\n        self.blobs = []\n        if len(self.P) == 0:\n            return\n        ix = np.round(self.P[:, 0]).astype(int); iy = np.round(self.P[:, 1]).astype(int)\n        x0, y0 = ix.min() - 2, iy.min() - 2\n        G_ = np.zeros((iy.max() - y0 + 3, ix.max() - x0 + 3), bool); G_[iy - y0, ix - x0] = True\n        lab, nl = ndimage.label(ndimage.binary_dilation(G_, np.ones((3, 3), bool)), np.ones((3, 3), int))\n        pl = lab[iy - y0, ix - x0]\n        for b in range(1, nl + 1):\n            idx = np.flatnonzero(pl == b)\n            if idx.size == 0:\n                continue\n            Q = self.P[idx]; c = Q.mean(0)\n            if idx.size >= 3:\n                ev, evec = np.linalg.eigh(np.cov((Q - c).T))\n                u = evec[:, 1]\n            else:\n                u = np.array([1.0, 0.0])\n            v = np.array([-u[1], u[0]])\n            pu = (Q - c) @ u; pv = (Q - c) @ v\n            if pv.max() - pv.min() < 3.0:\n                vks = [0.5 * (pv.max() + pv.min())]\n            else:\n                vks = list(np.arange(pv.min() + 1.5, pv.max() + 1.51, 3.0))\n            lines = []\n            for vk in vks:\n                m = np.abs(pv - vk) <= 1.7\n                if not m.any():\n                    continue\n                a, e = pu[m].min() - 0.5, pu[m].max() + 0.5\n                us = np.linspace(a, e, max(2, int(np.ceil((e - a) / 2.0)) + 1))\n                lines.append([c + uu * u + vk * v for uu in us])\n            self.blobs.append(dict(idx=idx, w=float(self.W[idx].sum()), lines=lines, taken=None))\n    def _next_blob(self, i, pos):\n        best = None\n        for bi, b in enumerate(self.blobs):\n            unc = ~self.cov[b['idx']]\n            if not unc.any() or not b['lines']:\n                continue\n            ends = np.array([ln[0] for ln in b['lines']] + [ln[-1] for ln in b['lines']])\n            d = float(np.min(np.hypot(ends[:, 0] - pos[0], ends[:, 1] - pos[1])))\n            wv = float(self.W[b['idx'][unc]].sum()) / max(self.W.sum(), 1e-9)\n            sc = wv * math.exp(-d / self.dscale) * (0.25 if (b['taken'] is not None and b['taken'] != i and b['taken'] in self.active) else 1.0)\n            if best is None or sc > best[0]:\n                best = (sc, bi)\n        if best is None:\n            return False\n        b = self.blobs[best[1]]; b['taken'] = i; self.blob_of[i] = best[1]\n        L = [list(map(np.asarray, ln)) for ln in b['lines']]\n        first_end = min(((k, e) for k in (0, len(L) - 1) for e in (0, -1)), key=lambda ke: float(np.hypot(*(L[ke[0]][ke[1]] - pos[:2]))))\n        order = range(len(L)) if first_end[0] == 0 else range(len(L) - 1, -1, -1)\n        wq = []; rev = first_end[1] == -1\n        for k in order:\n            ln = L[k][::-1] if rev else L[k]\n            wq.extend(ln); rev = not rev\n        self.wq[i] = wq; self.first_wp[i] = True\n        return True\n    def act(self, i, pos, vel, yaw, agl_ray, others):\n        while self.wq[i] and float(np.hypot(self.wq[i][0][0] - pos[0], self.wq[i][0][1] - pos[1])) < (1.0 if self.mode[i] == 'sweep' else 0.8):\n            self.wq[i].pop(0)\n            if self.mode[i] == 'transit':\n                self.mode[i] = 'sweep'\n        b = self.blob_of[i]\n        if b is not None and self.cov[self.blobs[b]['idx']].all():\n            self.wq[i] = []\n        if not self.wq[i]:\n            if b is not None:\n                self.blobs[b]['taken'] = None\n            if not self._next_blob(i, pos):\n                return None\n            self.mode[i] = 'transit'\n        tx, ty = self.wq[i][0]\n        self.tgt[i] = None\n        dxy = np.array([tx - pos[0], ty - pos[1]]); dist = float(np.hypot(*dxy))\n        last = len(self.wq[i]) == 1\n        sz = self.surf.at(pos[0], pos[1])\n        if self.mode[i] == 'transit' and dist <= 3.0:\n            self.mode[i] = 'sweep'\n        elif self.mode[i] == 'sweep' and dist > 6.0:\n            self.mode[i] = 'transit'; self.ztr[i] = None\n        if self.mode[i] == 'transit':\n            u = dxy / max(dist, 1e-6)\n            la = pos[:2] + u * min(dist, 8.0)\n            zm = self.surf.max_path(pos[0], pos[1], la[0], la[1], 1.5)\n            zt_ = self.surf.at(tx, ty)\n            if zm is None:\n                zref = pos[2] + 1.0\n            else:\n                zref = zm + H_TRANSIT\n                if dist < 10.0 and zt_ is not None:\n                    zref = max(zm + 2.5, min(zref, zt_ + H_SWEEP + 0.3 * dist))\n            self.ztr[i] = zref\n            ez = zref - pos[2]\n            vz = float(np.clip(1.2 * ez, -2.5, 2.0))\n            vh = V_TRANSIT if ez < 1.0 else max(0.3, V_TRANSIT * (1.0 - (ez - 1.0) / 2.0))\n            vxy = u * min(vh, 0.8 * dist + 0.3)\n        else:\n            look = pos[:2] + (dxy / max(dist, 1e-6)) * min(2.0, dist)\n            zl = self.surf.at(look[0], look[1]); zd = self.surf.max_disc(pos[0], pos[1], 0.8)\n            cands = [z + H_SWEEP for z in (sz,) if z is not None] + [z + 2.2 for z in (zl, zd) if z is not None]\n            zref = max(cands) if cands else pos[2]\n            ez = zref - pos[2]\n            if ez < -0.8:\n                vz = float(max(1.2 * ez, -2.5)); vh = min(0.5, 0.6 * dist + 0.25)\n            else:\n                vz = float(np.clip(0.8 * ez, -0.45, 0.9))\n                vh_max = math.sqrt(max(V_MAX3 ** 2 - vz ** 2, 0.01))\n                slow = 1.0 if ez <= 0.3 else max(0.1, 1.0 - (ez - 0.3) / 0.7)\n                vh = min(V_SWEEP * slow, vh_max, (0.6 * dist + 0.25) if last else V_SWEEP)\n            u = dxy / max(dist, 1e-6)\n            vxy = u * vh\n        if agl_ray < 2.0:\n            vz = max(vz, 0.8); vxy = vxy * min(1.0, 0.3 / max(float(np.hypot(*vxy)), 1e-6))\n        for j, pj in others:\n            dd = pos - pj\n            d3 = float(np.linalg.norm(dd))\n            if d3 < 3.0:\n                h = dd[:2] / max(float(np.hypot(*dd[:2])), 1e-3)\n                vxy = vxy + h * (3.0 - d3) * 0.8\n                if abs(dd[2]) < 1.5 and i > j:\n                    vz = max(vz, 0.4)\n        v = np.array([vxy[0], vxy[1], vz]); sp = float(np.linalg.norm(v))\n        cap = V_MAX3 if (self.mode[i] == 'sweep' and ez >= -0.8) else SPEED_LIMIT * 0.95\n        if sp > cap:\n            v *= cap / sp; sp = cap\n        vc = self.vcmd.get(i)\n        if vc is None:\n            vc = np.asarray(vel, float).copy()\n        dv = v - vc\n        amax = np.array([self.acc_xy, self.acc_xy, self.acc_z]) * SIM_DT\n        dxy_ = dv[:2]; nxy = float(np.hypot(*dxy_))\n        if nxy > amax[0]:\n            dv[:2] = dxy_ * amax[0] / nxy\n        dv[2] = float(np.clip(dv[2], -amax[2], amax[2]))\n        v = vc + dv; self.vcmd[i] = v.copy(); sp = float(np.linalg.norm(v))\n        hd = math.atan2(v[1], v[0]) if float(np.hypot(v[0], v[1])) > 0.05 else float(yaw)\n        self.drone_ticks += 1\n        spd = float(np.linalg.norm(vel))\n        if self.mode[i] == 'sweep':\n            self.sweep_ticks += 1\n        dw = self.dwell.setdefault(i, {})\n        if sz is not None and 2.1 <= pos[2] - sz <= 3.9 and spd <= 1.0:\n            self.inband_ticks += 1\n            near = np.flatnonzero(np.hypot(self.P[:, 0] - pos[0], self.P[:, 1] - pos[1]) <= 2.0)\n            nd = {}\n            for k in near:\n                nd[int(k)] = dw.get(int(k), 0.0) + SIM_DT\n                if nd[int(k)] >= 2.1:\n                    self.cov[k] = True\n            self.dwell[i] = nd\n        else:\n            self.dwell[i] = {}\n        return v / max(sp, 1e-9), sp, hd\nLAT = np.arange(-46.0, 46.0 + 1e-9, 1.0)\ndef support_cells_g(sup, cell, chunk=600):\n    XX, YY = np.meshgrid(LAT, LAT)\n    xr = XX.ravel(); yr = YY.ravel()\n    out = np.full(xr.size, -1, np.int64)\n    for a in range(0, xr.size, chunk):\n        x = xr[a:a + chunk]; y = yr[a:a + chunk]\n        j = np.argmin((x[:, None] - sup[None, :, 0]) ** 2 + (y[:, None] - sup[None, :, 1]) ** 2, axis=1)\n        ins = (np.abs(x - sup[j, 0]) <= cell / 2 + 1e-6) & (np.abs(y - sup[j, 1]) <= cell / 2 + 1e-6)\n        out[a:a + chunk] = np.where(ins, j, -1)\n        yield ('l', x.size * len(sup))\n    return out.reshape(XX.shape), XX, YY\ndef recheck_g(inv, Q, F):\n    keep = []\n    for b in inv.bodies:\n        inl, vp, vf = global_check(b, Q, F)\n        ni = int(inl.sum())\n        if ni >= inv.min_inl and (vp + vf) <= inv.vfrac * ni + inv.vabs:\n            keep.append(b)\n        yield ('g', len(Q) + (len(F) if F is not None else 0))\n    return keep\ndef predict_g(bodies, X, Y, chunk=500):\n    v = np.zeros(X.shape, bool)\n    for a in range(0, X.size, chunk):\n        vv, _, _ = predict_valid(bodies, X[a:a + chunk], Y[a:a + chunk])\n        v[a:a + chunk] = vv\n        yield ('p', min(chunk, X.size - a) * max(1, len(bodies)))\n    return v\n"
_MI_MOD = [None, False]
def _mi_core():
    if _MI_MOD[0] is None and not _MI_MOD[1]:
        _MI_MOD[1] = True
        try:
            import types as _mi_types
            m = _mi_types.ModuleType('_swsar_mi_core')
            exec(compile(_MI_SRC, '<swsar_mi_core>', 'exec'), m.__dict__)
            if not m.HAVE_SCIPY:
                raise RuntimeError('scipy unavailable')
            m.load_lib(str(Path(__file__).resolve().parent / MI_LIB_FILE))
            _MI_MOD[0] = m
        except Exception as exc:
            _log_once("mi_core", "stream MB core unavailable: %r" % (exc,), exc)
            _MI_MOD[0] = None
    return _MI_MOD[0]
def _mi_stat(k, d=1):
    _MI_STATS[k] = _MI_STATS.get(k, 0) + d
def _mi_max(k, v):
    if v > _MI_STATS.get(k, 0.0):
        _MI_STATS[k] = float(v)
def _mi_gate(T):
    return bool(_MAPSW_STATE.get('map') == 'mountain' and bool(getattr(T, 'mtn', False)))
def _mi_ctrl(fc, i):
    try:
        return fc.team.pilots[i].p.ctrl
    except Exception:
        return None
def _mi_busy(fc, T, i):
    c = _mi_ctrl(fc, i)
    if c is None:
        return True
    return bool(c.locked or c.frozen or c.mode in ('navigation', 'hover') or int(T.confirmer) == int(i))
def _mi_free(fc, T, i):
    c = _mi_ctrl(fc, i)
    try:
        tf = bool(T.frozen[i])
    except Exception:
        tf = True
    return bool(c is not None and not tf and not c.locked and not c.frozen and c.mode in ('initial', 'search')
                and int(T.confirmer) != int(i))
def _mi_new(n):
    return {'n': n, 'ok': (n >= MI_TAIL_N and n <= 8 and n not in MI_TAIL_NOFF), 'acc': None, 'inv': None, 'job': None, 'calg': None,
            'fin': None, 'phase': 'collect', 'plan': None, 'surf': None, 'tail': None, 'idle_until': -1,
            'why': '', 'test': None}
def _mi_job(X):
    inv = X['inv']
    npass = 0
    while True:
        while len(inv.log) + inv.n_small < inv.max_attempts:
            Q, F = X['acc'].snapshot(inv.centre, 90.0)
            yield ('s', 3 * len(Q))
            yield from inv.set_data_g(Q, F)
            r = yield from inv.attempt_g()
            if r is False:
                X['idle_until'] = X['t'] + 50
                yield ('idle', 0)
        if MI_PASS2 > 0 and npass == 0:
            npass = 1
            if X['t'] < MI_PASS2:
                X['idle_until'] = MI_PASS2
                yield ('idle', 0)
            inv.log_p1 = inv.log
            inv.log = []
            inv.n_small = 0
            continue
        X['why'] = 'max'
        return
def _mi_finalize(X, T, pos):
    core = _MI_MOD[0]
    inv = X['inv']
    post = T.region.native
    sup = np.asarray(post._sup, dtype=np.float64); w = np.asarray(post._w, dtype=np.float64); cell = float(post._grid_cell)
    centre = inv.centre if inv is not None else np.asarray(post._centroid, dtype=np.float64)
    if X['calg'] is not None:
        yield from X['calg']
        X['calg'] = None
    elif X['acc'].s is None:
        yield from X['acc'].calib_g()
    if X['acc'].cal is not None:
        _MI_STATS['fov'] = float(X['acc'].cal['fov'])
    Q, F = X['acc'].snapshot(centre, 90.0)
    yield ('s', 3 * len(Q))
    bodies = []
    if not MI_CMP and inv is not None:
        bodies = yield from core.recheck_g(inv, Q, F)
    X['bodies'] = bodies
    S, XX, YY = yield from core.support_cells_g(sup, cell)
    m = S >= 0
    if MI_CMP:
        pv = m
    else:
        v = yield from core.predict_g(bodies, XX[m], YY[m])
        pv = np.zeros(XX.shape, bool)
        pv[m] = v
    X['plan'] = (np.c_[XX[pv], YY[pv]], w[S[pv]])
    lo = np.minimum(pos[:, 0:2].min(0), -46.0) - 8.0
    hi = np.maximum(pos[:, 0:2].max(0), 46.0) + 8.0
    surf = core.Surface(lo, hi)
    yield ('s', 1)
    for b in bodies:
        surf.add_body(b)
        yield ('x', surf.nx * surf.ny)
    X['surf'] = surf
def _mi_run(X, gen, t_post0, t_act0, modelled):
    spent = 0.0
    while True:
        if modelled:
            if spent >= MI_BUDGET:
                return True
        else:
            now = time.perf_counter()
            if (now - t_post0) * 1e3 >= MI_BUDGET or (now - t_act0) >= MI_GATE:
                return True
        u0 = time.perf_counter()
        try:
            u = next(gen)
        except StopIteration:
            return False
        dt = (time.perf_counter() - u0) * 1e3
        _mi_stat('wall', dt)
        if u[0] == 'idle':
            return True
        a, b = _MI_COST.get(u[0], (1.0, 0.0))
        c = max(0.2, a + b * float(u[1]))
        _mi_stat('cost', c)
        spent += c if modelled else dt
_MS_NS = frozenset(int(_x) for _x in os.environ.get('SWSAR_MI_SMALL_NS', '3').split(',') if _x.strip())
_MS_K = {2: int(os.environ.get('SWSAR_MI_SMALL_K2', '1')), 3: int(os.environ.get('SWSAR_MI_SMALL_K3', '1'))}
_MS_T = {2: float(os.environ.get('SWSAR_MI_SMALL_T2', '0')), 3: float(os.environ.get('SWSAR_MI_SMALL_T3', '0'))}
_MS_THR = {2: float(os.environ.get('SWSAR_MI_SMALL_THR2', '1.0')), 3: float(os.environ.get('SWSAR_MI_SMALL_THR3', '1.0'))}
_MS_RULE = os.environ.get('SWSAR_MI_SMALL_RULE', 'mrem')
_MS_G = {2: os.environ.get('SWSAR_MI_SMALL_G2', ''), 3: os.environ.get('SWSAR_MI_SMALL_G3', '')}
_MS_FORCE = None
_MS_LOG = {}
if _MS_NS:
    _MI_KEYS = _MI_KEYS + ('s_arm', 's_ids', 's_nfree', 's_m0', 's_m1', 's_m2', 's_st0', 's_st1', 's_st2',
                           's_rs0', 's_rs1', 's_rs2', 's_err')
def _ms_new(X):
    if _MS_NS and int(X['n']) in _MS_NS:
        X['small'] = True
        X['ok'] = True
def _ms_ttail(X, t_tail):
    if X.get('small'):
        tt = float(_MS_T.get(int(X['n']), 0.0))
        if tt > 0.0:
            return int(round(tt * 50.0))
    return t_tail
def _ms_lane(T, t, n, pos):
    st = [4] * n; mrem = [-1.0] * n; rs = [0.0] * n; dcen = [0.0] * n
    reg = getattr(T, 'region', None)
    lp = getattr(reg, 'planner', None) if reg is not None else None
    post = getattr(reg, 'native', None) if reg is not None else None
    try:
        cen = np.asarray(post._centroid, dtype=np.float64).ravel()[:2]
    except Exception:
        cen = None
    for i in range(n):
        if cen is not None:
            dcen[i] = float(np.hypot(pos[i, 0] - cen[0], pos[i, 1] - cen[1]))
    tours = getattr(lp, '_tours', None) if lp is not None else None
    if not tours:
        return st, mrem, rs, dcen
    wtot = float(sum(float(np.sum(w_)) for w_ in lp._lane_w)) or 1.0
    tsec = float(t) * DT
    for i in range(n):
        r = int(reg.lane_of(i))
        if r < 0 or r >= len(tours):
            continue
        tour = np.asarray(tours[r], dtype=np.float64).reshape(-1, 2)
        dur = np.asarray(lp._dur[r], dtype=np.float64).ravel()
        m = int(tour.shape[0]); cyc = float(np.sum(dur))
        el = tsec - float(lp._t0[r])
        if m <= 1 or cyc <= 1e-6:
            st[i] = 3; mrem[i] = 0.0
            continue
        if el <= 0.0:
            j = 0; rs[i] = cyc - el
        elif int(el // cyc) >= 1:
            st[i] = 2; mrem[i] = 0.0
            continue
        else:
            j = int(np.clip(np.searchsorted(np.cumsum(dur), el, side='right'), 0, m - 1)); rs[i] = cyc - el
        st[i] = 1
        cells = np.asarray(lp._lane_cells[r], dtype=np.float64).reshape(-1, 2)
        cw = np.asarray(lp._lane_w[r], dtype=np.float64).ravel()
        if cells.shape[0] == 0:
            mrem[i] = 0.0
            continue
        near = np.argmin((cells[:, None, 0] - tour[None, :, 0]) ** 2 + (cells[:, None, 1] - tour[None, :, 1]) ** 2, axis=1)
        mrem[i] = float(cw[near >= j].sum()) / wtot
    return st, mrem, rs, dcen
def _ms_upl(X, pos, n):
    out = [0.0] * n
    plan = X.get('plan')
    if plan is None or len(plan[0]) == 0:
        return out
    P = np.asarray(plan[0], dtype=np.float64).reshape(-1, 2); Wp = np.asarray(plan[1], dtype=np.float64).ravel()
    tot = float(Wp.sum()) or 1.0
    for i in range(n):
        d = np.hypot(P[:, 0] - pos[i, 0], P[:, 1] - pos[i, 1])
        out[i] = float((Wp * np.exp(-d / 15.0)).sum()) / tot
    return out
def _ms_pick(fc, T, X, st, t, ids):
    if not X.get('small'):
        return ids
    try:
        n = int(X['n'])
        pos = np.asarray(st, dtype=np.float64)[:n, 0:3]
        lst, mrem, rs, dcen = _ms_lane(T, t, n, pos)
        upl = _ms_upl(X, pos, n)
        _MS_LOG.clear()
        _MS_LOG.update(t=int(t), free=list(ids), st=lst, mrem=mrem, rs=rs, dcen=dcen, upl=upl)
        if _MS_FORCE is not None:
            sel = [i for i in ids if i in _MS_FORCE]
        else:
            thr = float(_MS_THR.get(n, 1.0))
            m_ = [(x if x >= 0.0 else 1.0) for x in mrem]
            cand = [i for i in ids if m_[i] <= thr + 1e-12]
            if _MS_G.get(n) == 'busyfin':
                live = [j for j in range(n) if not bool(T.frozen[j])]
                if all(j in ids for j in live):
                    cand = [i for i in cand if lst[i] in (2, 3)]
            if _MS_RULE == 'far':
                cand.sort(key=lambda i: (-dcen[i], i))
            elif _MS_RULE == 'plan':
                cand.sort(key=lambda i: (-upl[i], i))
            else:
                cand.sort(key=lambda i: (m_[i], -dcen[i], i))
            sel = cand[:max(0, int(_MS_K.get(n, 1)))]
        _MS_LOG['sel'] = list(sel)
        _MI_STATS['s_arm'] = float(t)
        _MI_STATS['s_ids'] = float(sum(1 << i for i in sel))
        _MI_STATS['s_nfree'] = float(len(ids))
        for i in range(min(n, 3)):
            _MI_STATS['s_m%d' % i] = float(round(mrem[i] * 1000.0, 1))
            _MI_STATS['s_st%d' % i] = float(lst[i])
            _MI_STATS['s_rs%d' % i] = float(round(rs[i], 2))
        return sel
    except Exception:
        _mi_stat('s_err')
        return []
_MI_BASE = DroneFlightController
class DroneFlightController(_MI_BASE):
    def __init__(self):
        _MI_BASE.__init__(self)
        self._mi = None
        if _MI_ANY:
            _mi_core()
    def reset(self):
        self._mi = None
        _MI_STATS.clear()
        if _MI_ANY:
            _mi_core()
        return _MI_BASE.reset(self)
    def debug_state(self):
        out = _MI_BASE.debug_state(self)
        try:
            if isinstance(out, dict) and out and _MI_ANY:
                out['mi_on'] = float((1 if MI_INV_ON else 0) + (2 if MI_TAIL_ON else 0) + (4 if MI_CMP else 0))
                for _k in _MI_KEYS:
                    out['mi_' + _k] = float(_MI_STATS.get(_k, 0))
        except Exception:
            pass
        return out
    def act(self, observation):
        if not _MI_ANY or _MI_MOD[0] is None:
            return _MI_BASE.act(self, observation)
        team = getattr(self, 'team', None)
        T = getattr(team, 'T', None)
        if T is None:
            return _MI_BASE.act(self, observation)
        t_act0 = time.perf_counter()
        X = None
        st = None
        t = -1
        try:
            t = int(T.tick)
            st = np.asarray(observation['state'], dtype=np.float64)
            st = st.reshape(-1, st.shape[-1]) if st.ndim > 1 else st.reshape(1, -1)
            n = int(st.shape[0])
            X = self._mi
            if t == 0 or X is None or X['n'] != n:
                if t == 0:
                    _MI_STATS.clear()
                X = self._mi = _mi_new(n)
                _ms_new(X)
            X['t'] = t
            if not X['ok']:
                return _MI_BASE.act(self, observation)
            if X['test'] is None:
                X['test'] = bool(float(type(team).DEADLINE) >= 100.0)
            t_tail = int(round(MI_TAIL_T * 50.0))
            t_tail = _ms_ttail(X, t_tail)
            X['t_tail'] = t_tail
            if X['phase'] in ('collect', 'reg', 'fin') and t < t_tail and _mi_gate(T):
                i = t % MI_EVERY
                if i < n:
                    if X['acc'] is None:
                        X['acc'] = _MI_MOD[0].DepthAcc()
                    dep = np.asarray(observation['depth'], dtype=np.float32).reshape(n, 256, 256)
                    X['acc'].add_frame(dep[i], st[i, 0:3], st[i, 3:6], st[:, 0:3])
                    _MI_STATS['frames'] = float(X['acc'].frames)
                    _MI_STATS['npts'] = float(X['acc'].npts)
        except Exception:
            _mi_stat('err')
            if X is not None:
                X['phase'] = 'off'
        el = time.perf_counter() - t_act0
        _mi_max('pre_max', el * 1e3)
        cls = None
        dl0 = None
        try:
            if team is not None and el > 0.0:
                cls = next((k for k in type(team).__mro__ if 'DEADLINE' in k.__dict__), None)
                if cls is not None:
                    dl0 = cls.__dict__['DEADLINE']
                    cls.DEADLINE = max(0.05, float(dl0) - el)
        except Exception:
            dl0 = None
        try:
            action = _MI_BASE.act(self, observation)
        finally:
            if dl0 is not None:
                try:
                    cls.DEADLINE = dl0
                except Exception:
                    pass
        if X is None or not X['ok'] or X['phase'] in ('off', 'done'):
            return action
        t_post0 = time.perf_counter()
        try:
            if X['phase'] == 'tail':
                action = self._mi_tail_step(X, T, st, action, t)
                _mi_max('tail_max', (time.perf_counter() - t_post0) * 1e3)
                return action
            core = _MI_MOD[0]
            t_tail = X['t_tail']
            modelled = bool(X['test'])
            if not modelled and (t_post0 - t_act0) >= MI_GATE:
                work = False
            else:
                work = True
            if (X['phase'] in ('collect', 'reg') and work and X['acc'] is not None and not X['acc'].ready
                    and (X['calg'] is not None or (t >= MI_TCAL and X['acc'].s is None)) and t < t_tail - MI_FIN):
                if X['calg'] is None:
                    X['calg'] = X['acc'].calib_g()
                if not _mi_run(X, X['calg'], t_post0, t_act0, modelled):
                    X['calg'] = None
                    _MI_STATS['tcal'] = float(t)
                    _MI_STATS['fov'] = float(X['acc'].cal['fov'])
            if X['phase'] == 'collect' and t >= MI_T0 and _mi_gate(T) and X['acc'] is not None and X['acc'].ready:
                post = getattr(getattr(T, 'region', None), 'native', None)
                if getattr(T, 'region_final', False) and post is not None and hasattr(post, '_sup'):
                    if MI_CMP:
                        X['phase'] = 'reg'
                        X['why'] = 'cmp'
                        _MI_STATS['treg'] = float(t)
                    else:
                        X['inv'] = core.IncInverter(np.asarray(post._centroid, dtype=np.float64), max_attempts=MI_MAXATT)
                        X['job'] = _mi_job(X)
                        X['phase'] = 'reg'
            if X['phase'] == 'collect' and t >= t_tail - MI_FIN and X['acc'] is not None:
                X['phase'] = 'fin'
            if X['phase'] == 'reg':
                if t >= t_tail - MI_FIN:
                    X['phase'] = 'fin'
                    if X['inv'] is not None:
                        _MI_STATS['treg'] = float(t)
                        _MI_STATS['att'] = float(len(X['inv'].log) + X['inv'].n_small)
                        _MI_STATS['nbod'] = float(len(X['inv'].bodies))
                elif work and X['job'] is not None and t >= X['idle_until']:
                    if not _mi_run(X, X['job'], t_post0, t_act0, modelled):
                        X['job'] = None
                        _MI_STATS['treg'] = float(t)
            if X['phase'] == 'fin' and work:
                if X['fin'] is None:
                    if X['inv'] is None and not MI_CMP:
                        X['phase'] = 'done'
                        return action
                    X['fin'] = _mi_finalize(X, T, st[:, 0:3])
                if not _mi_run(X, X['fin'], t_post0, t_act0, modelled):
                    X['phase'] = 'ready'
                    _MI_STATS['tfin'] = float(t)
                    _MI_STATS['nkeep'] = float(len(X.get('bodies') or []))
                    _MI_STATS['plan'] = float(len(X['plan'][0]))
            if X['phase'] == 'ready' and t >= t_tail:
                if not (MI_TAIL_ON and _mi_gate(T)) or len(X['plan'][0]) == 0:
                    X['phase'] = 'done'
                    return action
                ids = [i for i in range(X['n']) if _mi_free(self, T, i)]
                ids = _ms_pick(self, T, X, st, t, ids)
                if not ids:
                    X['phase'] = 'done'
                    return action
                acc = X['acc']
                Q, _F = acc.snapshot(None)
                X['surf'].add_pts(Q, dil=3 if MI_CMP else 1)
                X['tail'] = core.Tail(X['plan'][0], X['plan'][1], X['surf'], ids, t)
                X['phase'] = 'tail'
                _MI_STATS['arm'] = float(t)
                _MI_STATS['ntail'] = float(len(ids))
                action = self._mi_tail_step(X, T, st, action, t)
            if t >= t_tail + 50 and X['phase'] in ('reg', 'fin'):
                X['phase'] = 'done'
        except Exception:
            _mi_stat('err')
            X['phase'] = 'off'
        if t < X.get('t_tail', 10 ** 9):
            _mi_max('post_max', (time.perf_counter() - t_post0) * 1e3)
        return action
    def _mi_tail_step(self, X, T, st, action, t):
        tail = X['tail']
        out = np.array(action, dtype=np.float32, copy=True)
        flat = out.reshape(-1, out.shape[-1])
        n = min(flat.shape[0], X['n'], st.shape[0])
        pos = st[:n, 0:3]; vel = st[:n, 6:9]
        live = [j for j in range(n) if not bool(T.frozen[j])]
        changed = False
        for i in sorted(tail.active):
            if i >= n or bool(T.frozen[i]):
                tail.active.discard(i)
                continue
            if MI_HANDBACK and _mi_busy(self, T, i):
                tail.active.discard(i)
                tail.handback[i] = t
                continue
            others = [(j, pos[j]) for j in live if j != i]
            r = tail.act(i, pos[i], vel[i], float(st[i, 5]), float(st[i, 162]) * 20.0, others)
            if r is None:
                tail.active.discard(i)
                tail.released.append(i)
                continue
            u, sp, hd = r
            flat[i, 0:3] = np.asarray(u, dtype=np.float32)
            flat[i, 3] = np.float32(sp / 3.0)
            flat[i, 4] = np.float32(hd / math.pi)
            changed = True
        _MI_STATS['act'] = float(sum(1 << i for i in tail.active))
        _MI_STATS['swm'] = float(sum(1 << i for i in tail.active if tail.mode.get(i) == 'sweep'))
        _MI_STATS['hbm'] = float(sum(1 << i for i in tail.handback))
        _MI_STATS['rel'] = float(len(tail.released))
        _MI_STATS['cov'] = float(tail.cov.sum())
        _MI_STATS['inband'] = float(tail.inband_ticks)
        _MI_STATS['sweep'] = float(tail.sweep_ticks)
        _MI_STATS['dtk'] = float(tail.drone_ticks)
        if not tail.active:
            X['phase'] = 'done'
        if not changed:
            return action
        np.clip(flat, ACT_LO, ACT_HI, out=flat)
        return out
VS_PG_ON = os.environ.get('SWSAR_VS_PG', '1') == '1'
VS_PLC = os.environ.get('SWSAR_VS_PLC', '0') == '1'
VS_PLC_EPS = float(os.environ.get('SWSAR_VS_PLC_EPS', '0.001'))
VS_PG_T0 = int(os.environ.get('SWSAR_VS_PG_T0', '100'))
VS_PG_IV = int(os.environ.get('SWSAR_VS_PG_IV', '100'))
VS_PG_SMAX = int(os.environ.get('SWSAR_VS_PG_SMAX', '9'))
VS_PG_VN = int(os.environ.get('SWSAR_VS_PG_VN', '3'))
VS_PG_VGAP = int(os.environ.get('SWSAR_VS_PG_VGAP', '12'))
VS_PG_RES = int(os.environ.get('SWSAR_VS_PG_RES', '8'))
VS_PG_K = int(os.environ.get('SWSAR_VS_PG_K', '1'))
VS_PG_TTL = int(os.environ.get('SWSAR_VS_PG_TTL', '1500'))
VS_PG_BUSYR = float(os.environ.get('SWSAR_VS_PG_BUSYR', '5.0'))
VS_PG_NMAX = int(os.environ.get('SWSAR_VS_PG_NMAX', '5'))
_VS_KEYS = ('arm', 't1', 'fr', 'hit', 'ver', 'over', 'busy', 'skip', 'dup', 'runs', 'ms_max', 'gate', 'err',
            'c1t', 'c1x', 'c1y', 'c1d', 'plc_t', 'plc_sup', 'pend', 'ngate')
_VS_STATS = {}
def _vs_stat(k, d=1):
    _VS_STATS[k] = _VS_STATS.get(k, 0) + d
def _vs_new(n):
    return {'n': n, 'pend': {}, 'scan': [0] * n, 'ver': [0] * n, 'lscan': [-10 ** 6] * n, 'lastfr': [-10 ** 6] * n,
            'lpose': [None] * n, 'cand': {}, 'hits': [], 'envt': [0] * n, 'envb': [0] * n, 'gate': None,
            'plc_done': False, 'nudge': None, 'done': []}
def _vs_hit(sess, rgb, dep, st, pads):
    k_run, k_ms = _MD_VDR_STATS.get('run'), _MD_VDR_STATS.get('ms_max')
    t0 = time.perf_counter()
    try:
        return _md_vdr_hit(sess, rgb, dep, st, pads)
    finally:
        ms = (time.perf_counter() - t0) * 1e3
        for _k, _v in (('run', k_run), ('ms_max', k_ms)):
            if _v is None:
                _MD_VDR_STATS.pop(_k, None)
            else:
                _MD_VDR_STATS[_k] = _v
        _vs_stat('runs')
        if ms > _VS_STATS.get('ms_max', 0.0):
            _VS_STATS['ms_max'] = ms
def _vs_on_spot(R, X, x, y):
    for sp in R['spots']:
        if math.hypot(x - sp['x'], y - sp['y']) <= MD_VDR_AGREE:
            return True
    for sp in X['done']:
        if math.hypot(x - sp[0], y - sp[1]) <= MD_VDR_AGREE:
            return True
    return False
def _vs_team_on(fc, n, x, y):
    for k in range(n):
        c = _md_vdr_ctrl(fc, k)
        if c is None or not (c.locked or c.frozen):
            continue
        try:
            v = c.locked_vic
            if math.hypot(float(v[0]) - x, float(v[1]) - y) <= VS_PG_BUSYR:
                return True
        except Exception:
            continue
    return False
def _vs_commit(fc, T, R, X, st, pts, t, prefer):
    x = float(np.mean([g[2] for g in pts]))
    y = float(np.mean([g[3] for g in pts]))
    n = X['n']
    if _vs_on_spot(R, X, x, y):
        _vs_stat('dup')
        return True
    if _vs_team_on(fc, n, x, y):
        _vs_stat('skip')
        X['done'].append((x, y))
        return True
    c = None
    j = -1
    if prefer is not None and 0 <= prefer < n:
        c = _md_vdr_eligible(fc, T, prefer)
        j = prefer if c is not None else -1
    if c is None:
        best = None
        for k in range(n):
            ck = _md_vdr_eligible(fc, T, k)
            if ck is None:
                continue
            d = math.hypot(float(st[k, 0]) - x, float(st[k, 1]) - y)
            if best is None or d < best[0]:
                best = (d, k, ck)
        if best is not None:
            j, c = best[1], best[2]
    if c is None:
        _vs_stat('busy')
        return False
    if VS_PLC and X['plc_done']:
        _vs_stat('plc_sup')
        return True
    _vs_stat('arm')
    if _VS_STATS.get('t1', 0) <= 0:
        _VS_STATS['t1'] = float(t)
        _VS_STATS['c1t'], _VS_STATS['c1x'], _VS_STATS['c1y'], _VS_STATS['c1d'] = float(t), x, y, float(j)
    if VS_PLC:
        X['plc_done'] = True
        X['nudge'] = j
        _VS_STATS['plc_t'] = float(t)
        return True
    _md_vdr_inject(c, [g[:6] for g in pts], t)
    R['spots'].append({'x': x, 'y': y, 'j': j, 't': t, 'pts': [g[:6] for g in pts]})
    del R['spots'][:-8]
    X['done'].append((x, y))
    return True
def _vs_process(X, i, hit, kind, t):
    if hit is None:
        cd = X['cand'].get(i)
        if kind == 'ver' and cd is not None:
            cd['left'] -= 1
            if cd['left'] <= 0:
                X['cand'].pop(i, None)
        return
    _vs_stat('hit')
    h = (t, i, hit['x'], hit['y'], hit['top'], hit['lp'], 'vs')
    agree = [g for g in X['hits'] if (g[0], g[1]) != (t, i) and math.hypot(g[2] - h[2], g[3] - h[3]) <= MD_VDR_AGREE]
    X['hits'].append(h)
    del X['hits'][:-64]
    if agree:
        X['cand'].pop(i, None)
        return
    X['cand'][i] = {'t': t, 'left': VS_PG_VN}
_VS_BASE = DroneFlightController
class DroneFlightController(_VS_BASE):
    def __init__(self):
        _VS_BASE.__init__(self)
        self._vs_x = None
    def reset(self):
        self._vs_x = None
        _VS_STATS.clear()
        return _VS_BASE.reset(self)
    def debug_state(self):
        out = _VS_BASE.debug_state(self)
        try:
            if isinstance(out, dict) and out and VS_PG_ON:
                out['vs_pg_on'] = 1.0
                out['vs_plc_on'] = 1.0 if VS_PLC else 0.0
                for _k in _VS_KEYS:
                    out['vs_pg_' + _k] = float(_VS_STATS.get(_k, 0))
        except Exception:
            pass
        return out
    def act(self, observation):
        if not VS_PG_ON:
            return _VS_BASE.act(self, observation)
        team = getattr(self, 'team', None)
        T = getattr(team, 'T', None)
        if T is None:
            return _VS_BASE.act(self, observation)
        if VS_PG_NMAX < 8:
            try:
                _st0 = np.asarray(observation['state'])
                _n0 = int(_st0.shape[0]) if _st0.ndim > 1 else 1
            except Exception:
                _n0 = 0
            if _n0 > VS_PG_NMAX:
                _VS_STATS['ngate'] = float(_n0)
                return _VS_BASE.act(self, observation)
        tv0 = time.perf_counter()
        obs = observation
        X = None
        R = None
        t = -1
        st = None
        try:
            t = int(T.tick)
            st = np.asarray(observation['state'], dtype=np.float64)
            st = st.reshape(-1, st.shape[-1]) if st.ndim > 1 else st.reshape(1, -1)
            n = int(st.shape[0])
            X = self._vs_x
            if t == 0 or X is None or X['n'] != n:
                if t == 0:
                    _VS_STATS.clear()
                X = self._vs_x = _vs_new(n)
            R = getattr(self, '_md_vdr', None)
            pend, X['pend'] = X['pend'], {}
            if pend and R is not None and R.get('pads') is not None:
                sess = _md_vdr_session()
                raw = np.asarray(observation['rgb'])
                rgb = raw.reshape(n, 256, 256, 3)
                dep = np.asarray(observation['depth'], dtype=np.float32).reshape(n, 256, 256)
                if sess is not None:
                    for i, kind in sorted(pend.items()):
                        if kind == 'ver':
                            _vs_stat('ver')
                        hit, why = _vs_hit(sess, rgb[i], dep[i], st[i], R['pads'])
                        _vs_process(X, i, hit, kind, t)
                rgb2 = np.array(raw, copy=True)
                v2 = rgb2.reshape(n, -1)
                for i in pend:
                    v2[i] = 0
                obs = dict(observation)
                obs['rgb'] = rgb2
        except Exception:
            _vs_stat('err')
            if X is not None:
                X['pend'] = {}
        el = time.perf_counter() - tv0
        cls = None
        dl0 = None
        try:
            if team is not None and el > 0.0:
                cls = next((k for k in type(team).__mro__ if 'DEADLINE' in k.__dict__), None)
                if cls is not None:
                    dl0 = cls.__dict__['DEADLINE']
                    cls.DEADLINE = max(0.05, float(dl0) - el)
        except Exception:
            dl0 = None
        try:
            action = _VS_BASE.act(self, obs)
        finally:
            if dl0 is not None:
                try:
                    cls.DEADLINE = dl0
                except Exception:
                    pass
        if X is None:
            return action
        if R is None:
            R = getattr(self, '_md_vdr', None)
            if R is None or R.get('pads') is None:
                return action
        try:
            out = np.array(action, dtype=np.float32, copy=True)
            flat = out.reshape(-1, out.shape[-1])
            n = min(flat.shape[0], X['n'], R['n'])
            if flat.shape[1] < 6:
                return action
            changed = False
            mgate = _md_vdr_gate(T)
            g = R.get('gate_tk')
            if g is not None and _VS_STATS.get('gate', 0) <= 0:
                _VS_STATS['gate'] = float(g)
            if mgate and g is not None and t >= g and X['hits'] and not (VS_PLC and X['plc_done']):
                live = [h for h in X['hits'] if t - h[0] <= VS_PG_TTL]
                new = [h for h in R['hits'] if h[0] == t]
                _nx = getattr(self, '_n_x', None)
                if _nx is not None and isinstance(_nx.get('hits'), list):
                    new += [h for h in _nx['hits'] if h[0] == t]
                for h in new:
                    agree = [q for q in live if math.hypot(q[2] - h[2], q[3] - h[3]) <= MD_VDR_AGREE]
                    if agree:
                        pts = [q[:6] for q in agree[-3:]] + [tuple(h[:6])]
                        if _vs_commit(self, T, R, X, st, pts, t, int(h[1])):
                            live = [q for q in live if q not in agree]
                            X['hits'] = [q for q in X['hits'] if q not in agree]
                used = set()
                for a_ in range(len(live)):
                    if a_ in used:
                        continue
                    ha = live[a_]
                    grp = [b_ for b_ in range(len(live)) if b_ not in used and (live[b_][0], live[b_][1]) != (ha[0], ha[1])
                           and math.hypot(live[b_][2] - ha[2], live[b_][3] - ha[3]) <= MD_VDR_AGREE]
                    if not grp:
                        continue
                    pts = [live[b_][:6] for b_ in grp[-3:]] + [ha[:6]]
                    if _vs_commit(self, T, R, X, st, pts, t, None):
                        used.add(a_)
                        used.update(grp)
                if used:
                    gone = set(live[u] for u in used)
                    X['hits'] = [q for q in X['hits'] if q not in gone]
            if mgate and VS_PG_T0 <= t < (g if g is not None else MD_VDR_T0) and not (VS_PLC and X['plc_done']):
                k_req = 0
                dep = None
                for jj in range(n):
                    if k_req >= VS_PG_K:
                        break
                    i = (t + jj) % n
                    if flat[i, 5] > 0.5:
                        continue
                    if X['envt'][i] + 1 > MD_VDR_ENV_CAP - VS_PG_RES:
                        continue
                    c = _md_vdr_eligible(self, T, i)
                    if c is None:
                        X['cand'].pop(i, None)
                        continue
                    kind = None
                    cd = X['cand'].get(i)
                    if cd is not None and (cd['left'] <= 0 or t - cd['t'] > MD_VDR_TTL):
                        X['cand'].pop(i, None)
                        cd = None
                    if cd is not None:
                        if t - X['lastfr'][i] >= VS_PG_VGAP and X['ver'][i] < VS_PG_SMAX:
                            kind = 'ver'
                    elif c.mode == 'search' and X['scan'][i] < VS_PG_SMAX and t - X['lscan'][i] >= VS_PG_IV:
                        sti = st[i]
                        if max(abs(float(sti[3])), abs(float(sti[4]))) > math.radians(MD_VDR_TILT):
                            continue
                        lp_ = X['lpose'][i]
                        if lp_ is not None:
                            mv = float(np.linalg.norm(sti[0:3] - lp_[0]))
                            dy = abs((math.degrees(float(sti[5]) - lp_[1]) + 180.0) % 360.0 - 180.0)
                            if mv < MD_VDR_MOVE and dy < MD_VDR_TURN:
                                continue
                        if dep is None:
                            dep = np.asarray(observation['depth'], dtype=np.float32).reshape(-1, 256, 256)
                        if not _md_vdr_useful(dep[i]):
                            continue
                        kind = 'scan'
                    if kind is None:
                        continue
                    R['req'].append((t, i, float(flat[i, 5])))
                    flat[i, 5] = 1.0
                    changed = True
                    X['pend'][i] = kind
                    k_req += 1
                    _vs_stat('fr')
                    if kind == 'scan':
                        X['scan'][i] += 1
                        X['lscan'][i] = t
                        X['lpose'][i] = (np.asarray(st[i, 0:3], dtype=np.float64).copy(), float(st[i, 5]))
                    else:
                        X['ver'][i] += 1
            if X['nudge'] is not None:
                j = int(X['nudge'])
                X['nudge'] = None
                if 0 <= j < n:
                    sp_ = float(flat[j, 3])
                    flat[j, 3] = np.float32(sp_ - VS_PLC_EPS if sp_ + VS_PLC_EPS > 1.0 else sp_ + VS_PLC_EPS)
                    changed = True
            for i in range(n):
                if flat[i, 5] <= 0.5:
                    continue
                X['lastfr'][i] = t
                if i in X['pend']:
                    X['envt'][i] += 1
                    continue
                srv_t = X['envt'][i] < MD_VDR_ENV_CAP
                srv_b = X['envb'][i] < MD_VDR_ENV_CAP
                if srv_t:
                    X['envt'][i] += 1
                if srv_b:
                    X['envb'][i] += 1
                if srv_b and not srv_t:
                    _vs_stat('over')
            _VS_STATS['pend'] = float(len(X['hits']))
            return out if changed else action
        except Exception:
            _vs_stat('err')
            return action
_WD_BASE = DroneFlightController
class DroneFlightController(_WD_BASE):
    def reset(self):
        _wd_reset()
        self._wd_t = 0
        if WD_CLF_ON:
            _wd_model()
        return _WD_BASE.reset(self)
    def act(self, observation):
        t = getattr(self, "_wd_t", 0)
        self._wd_t = t + 1
        if not WD_CLF_ON:
            return _WD_BASE.act(self, observation)
        obs = observation
        want_rgb = False
        try:
            if t == 0:
                _wd_reset()
                st = np.asarray(observation["state"], np.float32)
                _WD["S0"] = (st.reshape(1, -1) if st.ndim == 1 else st.reshape(-1, st.shape[-1])).copy()
            if WD_RGB and t == WD_RQ and _WD.get("S0") is not None:
                lg = _wd_logits56(observation)
                srt = np.sort(lg)
                _WD["m56"] = float(srt[-1] - srt[-2])
                _WD["prov"] = int(np.argmax(lg))
                want_rgb = (_wd_candidate(_WD_NAMES[int(np.argmax(lg))], float(np.max(_WD["S0"][:, 2])))
                            or _WD["m56"] < WD_RGB_MARGIN)
            if WD_RGB and t == WD_RQ + 1 and _WD.get("reqd") is not None and hasattr(observation, "get") \
                    and observation.get("rgb") is not None:
                n = _WD["S0"].shape[0]
                rgb = np.asarray(observation["rgb"], np.float32).reshape(n, 256, 256, 3)
                sub = np.clip(rgb[:, 2::4, 2::4] * 255.0 + 0.5, 0, 255).astype(np.uint8)
                ok = sub.reshape(n, -1).max(axis=1) > 0
                _WD["got"] = int(ok.sum())
                if bool(ok.all()):
                    _WD["R"] = sub
                mine = [i for i in _WD["reqd"] if i < n]
                if mine:
                    raw = np.asarray(observation["rgb"])
                    hid = np.array(raw, copy=True)
                    hid.reshape(n, -1)[mine] = 0
                    obs = dict(observation)
                    obs["rgb"] = hid
            if WD_RGB and _WD.get("reqd") and WD_RQ + 1 <= t <= WD_RQ + 25:
                k = 161 - 6 * (t - WD_RQ - 1)
                n = _WD["S0"].shape[0]
                st = np.array(observation["state"], copy=True)
                st2 = st.reshape(n, -1)
                hit = 0
                for i in _WD["reqd"]:
                    if i < n and st2.shape[1] > k and st2[i, k] == 1.0:
                        st2[i, k] = 0.0
                        hit += 1
                if hit:
                    if obs is observation:
                        obs = dict(observation)
                    obs["state"] = st
                    _WD["mask"] = _WD.get("mask", 0) + hit
        except Exception:
            _WD["err"] += 1
            obs = observation
        action = _WD_BASE.act(self, obs)
        if want_rgb:
            try:
                a = np.array(action, dtype=np.float32, copy=True)
                n = _WD["S0"].shape[0]
                a2 = a.reshape(n, -1)
                mine = [i for i in range(n) if not (a2[i, 5] > 0.5)]
                a2[mine, 5] = 1.0
                _WD["reqd"] = mine
                _WD["req"] = len(mine)
                action = a
            except Exception:
                _WD["err"] += 1
        return action
    def debug_state(self):
        out = _WD_BASE.debug_state(self)
        try:
            if isinstance(out, dict) and out and WD_CLF_ON:
                out['wd_on'] = 1.0
                for _k in ('arm', 'from', 'to', 'pred', 'conf', 'model', 'req', 'got', 'm56', 'ms', 'err', 'prov', 'cand', 'mask'):
                    out['wd_' + _k] = float(_WD.get(_k, -1))
        except Exception:
            pass
        return out
_IDEMP_BASE = DroneFlightController
class DroneFlightController(_IDEMP_BASE):
    def reset(self):
        self._idem_key = None
        self._idem_act = None
        return _IDEMP_BASE.reset(self)
    def act(self, observation):
        try:
            st = np.asarray(observation["state"])
            key = (st.shape, hash(st.tobytes()))
        except Exception:
            key = None
        if key is not None and key == getattr(self, "_idem_key", None) and getattr(self, "_idem_act", None) is not None:
            return self._idem_act.copy()
        action = _IDEMP_BASE.act(self, observation)
        if key is not None:
            self._idem_key = key
            self._idem_act = np.array(action, dtype=np.float32, copy=True)
        return action
VC_VGR_ON = os.environ.get('SWSAR_VC_VGR', '1') == '1'
VC_VGR_ZHI = float(os.environ.get('SWSAR_VC_VGR_ZHI', '2.0'))
VC_VGR_ZLO = float(os.environ.get('SWSAR_VC_VGR_ZLO', '-1.0'))
VC_PAUSE_ON = os.environ.get('SWSAR_VC_PAUSE', '0') == '1'
VC_CATCHUP = os.environ.get('SWSAR_VC_CATCHUP', '1') == '1'
VC_REACH = float(os.environ.get('SWSAR_VC_REACH', '8.0'))
VC_DEFER = os.environ.get('SWSAR_VC_DEFER', '1') == '1'
_VC = {"mtn": False, "vgr_ref": 0, "vgr_pz": 0, "pause": 0, "resume": 0, "catchup": 0, "err": 0}
def _vc_village():
    return _MAPSW_STATE.get("map") == 'village' and not _VC.get("mtn")
def _vc_dbg():
    return {"vc_on": float((1 if VC_VGR_ON else 0) + (2 if VC_PAUSE_ON else 0)),
            "vc_vgr_ref": float(_VC["vgr_ref"]), "vc_vgr_pz": float(_VC["vgr_pz"]),
            "vc_pause": float(_VC["pause"]), "vc_resume": float(_VC["resume"]),
            "vc_catchup": float(_VC["catchup"]), "vc_err": float(_VC["err"])}
def _vc_off_ground(z):
    if not _vc_village():
        return False
    try:
        zz = float(z[2])
        return math.isfinite(zz) and (zz > VC_VGR_ZHI or zz < VC_VGR_ZLO)
    except Exception:
        _VC["err"] += 1
        return False
_VC_VD_REFUTED = VictimDetector.location_refuted
_VC_VD_UPDATE = VictimDetector.update
def _vc_location_refuted(self, z):
    if VC_VGR_ON and _vc_off_ground(z):
        _VC["vgr_ref"] += 1
        return True
    return _VC_VD_REFUTED(self, z)
def _vc_update(self, depth, pos, rpy, rgb=None, is_mountain=False):
    r = _VC_VD_UPDATE(self, depth, pos, rpy, rgb, is_mountain)
    if VC_VGR_ON:
        z = getattr(self, "last_z", None)
        if z is not None and _vc_off_ground(z):
            try:
                if float(self.last_p_depth) != 0.0:
                    _VC["vgr_pz"] += 1
            except Exception:
                _VC["err"] += 1
            self.last_p_depth = 0.0
    return r
VictimDetector.location_refuted = _vc_location_refuted
VictimDetector.update = _vc_update
_VC_TWP = _TeamState.waypoint
def _vc_team_waypoint(self, i, pos_xy, locked=False):
    if VC_PAUSE_ON and self.region is not None and self.region_final and _vc_village():
        try:
            lp = getattr(self.region, "planner", None)
            if lp is not None and getattr(lp, "_t0", None) is not None:
                r = int(self.region.lane_of(int(i)))
                if 0 <= r < lp._t0.size:
                    pz = getattr(self, "_vc_paused", None)
                    if pz is None or pz.size != lp._t0.size:
                        pz = self._vc_paused = np.zeros(lp._t0.size, dtype=bool)
                    pend = getattr(self, "_vc_pend", None)
                    if pend is None or pend.size != lp._t0.size:
                        pend = self._vc_pend = np.zeros(lp._t0.size, dtype=np.float64)
                    _div = bool(locked) or int(i) == int(self.confirmer)
                    if _div and VC_DEFER:
                        pend[r] += DT
                        _VC["pause"] += 1
                    elif _div:
                        lp._t0[r] += DT
                        pz[r] = True
                        _VC["pause"] += 1
                    elif pend[r] > 0.0:
                        lp._t0[r] += pend[r]
                        pend[r] = 0.0
                        pz[r] = True
                        _VC["resume"] += 1
                    elif VC_CATCHUP and pz[r]:
                        t = self.tick * DT
                        wp = np.asarray(self.region.waypoint(int(i), np.asarray(pos_xy, dtype=np.float64)[:2], t,
                                                             False), dtype=np.float64)
                        if float(math.hypot(wp[0] - pos_xy[0], wp[1] - pos_xy[1])) > VC_REACH:
                            lp._t0[r] += DT
                            _VC["catchup"] += 1
                        else:
                            pz[r] = False
                            if t < float(lp._t0[r]):
                                lp._t0[r] = t
        except Exception:
            _VC["err"] += 1
    return _VC_TWP(self, i, pos_xy, locked)
_TeamState.waypoint = _vc_team_waypoint
