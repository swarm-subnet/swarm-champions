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
SCAN_V = 2.4                                                
USABLE_SEC = 52.0                                               
CAP_LO, CAP_HI = 4.0, 45.0
WP_SEP = 14.0                                             
WP_MAX = 6
                                                                                
                                                                                   
                                                                                
ADVANCE_R = 8.0
                                                                             
                                                                                 
                                                                            
                                             
HOLD_SEC = 2.0
CENTRE_EPS = 3.0                                                                   
SCAN_R_REF = 26.0                                         
SCAN_R_MIN = 0.45

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
        return float(max(PAD0_R, np.hypot(dx, dy) + PAD_SLACK))

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
        b0 = RELAX_BOX * b if self.pad_overlap < PAD_TRIG else b
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
            return tour[0].copy()                                                  
        cyc = float(np.sum(self._dur[r]))
        if cyc <= 1e-6:
            return tour[0].copy()
                                                                                    
                                                                                   
                                                                              
        npass = int(elapsed // cyc)
        phase = elapsed - npass * cyc
        rev = bool(npass & 1)
        cum = np.cumsum(self._dur_rev[r] if rev else self._dur[r])
        j = int(np.clip(np.searchsorted(cum, phase, side="right"), 0, m - 1))
        return tour[m - 1 - j].copy() if rev else tour[j].copy()

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
COMMIT_CONF = 0.6
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
MTN_DET_FALLBACK = _repo_fallback('sotaUID220/victim_detect_mountain/out_v3/mountain_victim_depth.onnx')
MTN_RGB_NAME = 'victim_rgb_m.onnx'
MTN_RGB_FALLBACK = _repo_fallback('sotaUID220/victim_detect_mountain/out_rgb/mountain_victim_rgb.onnx')
T_OLD_DEPTH = 0.657
T_MTN_DEPTH = 0.586
T_MTN_RGB = 0.649
MAP_LABELS = ('city', 'open', 'mountain', 'village', 'warehouse', 'forest')
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
URBAN_LOW = os.environ.get('SAR_URBAN_LOW', '') == '1'
URBAN_ALT_M = float(os.environ.get('SAR_URBAN_ALT', '6.0'))
CENTRE_REACH_M = 3.5
CENTRE_MAX_TICKS = 600
INVESTIGATE_M = 8.0
GROUND_GATE_M = float(os.environ.get('SAR_GROUND_GATE', '1.05'))
INVESTIGATE_FACTOR = 0.75
HOVER_DESCENT_SPEED = 0.6
HOVER_DESCENT_M = 2.0
STAR_MODE = os.environ.get('SAR_STAR', '') == '1'
STAR_RADIUS_M = float(os.environ.get('SAR_STAR_R', '20.0'))
_star_env = os.environ.get('SAR_STAR_ORDER', '')
STAR_ORDER_DEG = tuple((float(v) for v in _star_env.split(','))) if _star_env else (0.0, 135.0, -90.0, 45.0, 180.0, -45.0, 90.0, -135.0)
STAR_REACH_M = 3.5
HOVER_GIVEUP_BAND = (1.5, 4.5)
HOVER_GIVEUP_FALLBACK_TICKS = 1200

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
        self.mtn_head_used = False
        self.blacklist = []
        self.veto_spots = []
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
        return any((float(np.linalg.norm(z - b)) <= PHANTOM_VETO_M for b in self.blacklist))

    def _rgb_prob(self, rgb: np.ndarray, is_mountain: bool=False) -> float | None:
        rgb = np.asarray(rgb, dtype=np.float32)
        if rgb.size == 0 or float(np.mean(np.abs(rgb))) < RGB_PRESENT_EPS:
            return None
        r = rgb.reshape(256, 256, 3).reshape(RGB_RES, 2, RGB_RES, 2, 3).mean(axis=(1, 3))
        r = np.ascontiguousarray(r, dtype=np.float32)
        if is_mountain and self.mtn_rgb_sess is not None:
            out = self.mtn_rgb_sess.run(None, {self.mtn_rgb_iname: r})[0]
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

    def _lr_allowed(self) -> bool:
        if self.lr_sess is None:
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
        p_depth, u, v, log_cz, h_meas = self._depth_heads(x, is_mountain)
        self.last_p_depth = p_depth
        cz = float(np.exp(log_cz))
        z_world = measurement_world(u, v, cz, pos, rpy)
        self.last_z = z_world.copy()
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
        if p_rgb is not None and p_depth >= RGB_TARGET_TRIGGER:
            if p_rgb < RGB_CONFIRM_THRESH:
                if not colour_blind:
                    self.veto_spots.append((z_world.copy(), self.t))
            else:
                self.veto_spots = [(p, tk) for p, tk in self.veto_spots if float(np.linalg.norm(z_world - p)) > CONSIST_M]
        if len(self.veto_spots) > 64:
            self.veto_spots = self.veto_spots[-64:]
                                                                          
                                                                         
                                                                           
                                                                           
                                                                   
        strong_bar = getattr(self, 'strong_p_override', 0.0) or STRONG_P
        is_strong = p_depth >= strong_bar
        if is_strong and p_rgb is not None and (p_rgb < RGB_CONFIRM_THRESH) and (not colour_blind):
            is_strong = False
        if is_strong and self.location_refuted(z_world):
            is_strong = False
        d_inc = float(np.linalg.norm(z_world - self.vic))
        d_cand = float(np.linalg.norm(z_world - self.cand))
        if is_strong:
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
            self.strong *= STALE_DECAY
            self.cand_strong *= STALE_DECAY
        self.conf = float(np.clip(self.strong / N_LATCH, 0.0, 1.0))
        self.seen = bool(is_strong and self.strong > 0.0 and (float(np.linalg.norm(z_world - self.vic)) <= CONSIST_M))
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
            # v150 (= our unpublished v116): the stock cone reads cols 32..224
            # of 256 -- the outer 12.5% each side is blind, and at the 5 m
            # forest deck an edge trunk stays invisible until nearly central.
            # Widen STEERING to the full frame; braking stays on the narrow
            # centre via brake_ref below (28% fewer forest crashes, no slowdown).
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
            brake_ref = min(self._narrow_centre(depth), left, right)
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

    def _clutter_frac(self) -> float:
        return float(np.clip((self.clutter - CLUTTER_EDGE_LO) / (CLUTTER_EDGE_HI - CLUTTER_EDGE_LO), 0.0, 1.0))

    def _update_clutter(self, depth: np.ndarray) -> None:
        d = np.asarray(depth, dtype=np.float32).reshape(DEPTH_RES, DEPTH_RES)
        gx = np.abs(np.diff(d[64:192, :], axis=1))
        edge_frac = float((gx > CLUTTER_EDGE_THRESH).mean())
        self.clutter = CLUTTER_EMA * self.clutter + (1.0 - CLUTTER_EMA) * edge_frac

    def _search_alt(self) -> float:
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
        else:
            bar = COMMIT_CONF + COMMIT_CLUTTER_BONUS * self._clutter_frac()
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
            return self.alt_td(ground_z + MOUNTAIN_SEARCH_ALT_M)
        if self.forest_low:
            return self.alt_td(self._true_ground(ground_z) + FOREST_ALT_M)
        if URBAN_LOW:
            return self.alt_td(self._true_ground(ground_z) + URBAN_ALT_M)
        return ground_z + self._search_alt()

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
        if self.forest_low:
            target = np.array([pos[0], pos[1], self._true_ground(ground_z) + FOREST_ALT_M])
            return (target, 0.6)
        if URBAN_LOW and (not self.mtn.is_mountain):
            target = np.array([pos[0], pos[1], self._true_ground(ground_z) + URBAN_ALT_M])
            return (target, 0.6)
        target = np.array([pos[0], pos[1], ground_z + self._search_alt()])
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
                                                                             
                                                                          
                                                                           
                                                                           
                                                                          
        boost = float(os.environ.get('HYB_MTN_HOVZ', '0.0')) if self.mtn.is_mountain else 0.0
        target = np.array([vic[0], vic[1], head_z + HOVER_ABOVE_TOP_M + HOVER_SETTLE_MARGIN_M + boost])
        agl_sp = float(os.environ.get('HYB_MTN_AGLHOV', '3.2'))
        if self.mtn.is_mountain and agl_sp > 0.0 and self._last_agl is not None:
                                                                            
                                                                         
                                                                           
                                                                        
                                                                         
                                                                       
            # v150: bidirectional AGL servo (our v104/v105 fix -- the max() can
            # only RAISE the hover target; census showed misses sit ABOVE the band)
            _ray_z = pos[2] - (self._last_agl - agl_sp)
            if os.environ.get('SAR150_BIDIR', '1') == '1':
                target[2] = _ray_z
            else:
                target[2] = max(target[2], _ray_z)
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
        self._update_clutter(depth)
        self.avoider.wide = bool(getattr(self, 'forest_low', False)) \
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
                        and self.tick >= FOREST_EARLY_TICK and p_wh <= FOREST_WH_VETO)
            if early or late or rgbf:
                self.forest_low = True
                self.forest_latch_tick = self.tick
        if self.mode == 'initial':
            conf, vic, head_z = (0.0, pos.copy(), ground_z)
        elif self.frozen:
            conf, vic, head_z = (1.0, self.locked_vic, self.locked_head_z)
        else:
            conf, vic, head_z = self.detector.update(depth, pos, rpy, rgb=rgb, is_mountain=self.mtn.is_mountain)
            if self.detector.rgb_fresh and self.detector.rgb_prob >= RGB_CONFIRM_THRESH:
                self.rgb_confirm_tick = self.tick
        elevated = not self.mtn.is_mountain and vic[2] - ground_z > GROUND_GATE_M and (float(np.linalg.norm(vic[0:2] - pos[0:2])) > INVESTIGATE_M)
        if self.mode != 'initial' and (not self.frozen) and (not self.locked) and (conf >= LATCH_CONF) and (not elevated):
            self.locked = True
            self.locked_vic = vic.copy()
            self.locked_head_z = head_z
            self.lost_ticks = 0
        if self.locked and (not self.frozen):
            if self.detector.seen:
                self.locked_vic = vic.copy()
                self.locked_head_z = head_z
                self.lost_ticks = 0
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
        if self.hover_stable_ticks >= HOVER_GIVEUP_TICKS or self.hover_total_ticks >= HOVER_GIVEUP_FALLBACK_TICKS:
            self.detector.blacklist.append(self.locked_vic.copy())
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
        if self.mode == 'initial':
            if pos[2] >= ground_z + SEARCH_ALT_M - 1.0 or self.tick > 150:
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
        delta = target - pos
        dist = float(np.linalg.norm(delta))
        direction = _unit(delta)
        speed = min(speed, float(np.clip(dist * 0.125, 0.0, 1.0)))
        if self.mode != 'hover':
            if self.mtn.is_mountain:
                direction, speed = self._mountain_avoid(direction, speed, depth, rpy)
            else:
                direction, speed = self.avoider.steer(direction, speed, depth, rpy)
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
        rgb_req = 0.0
        ondemand = self.detector.last_p_depth > RGB_TARGET_TRIGGER and (not self.detector.location_refuted(self.detector.last_z))
        if self.mode in ('search', 'navigation') and (not self.frozen) and (self.rgb_requests < RGB_CAP) and (self.tick - self.last_rgb_req >= RGB_MIN_INTERVAL) and ((self.tick - getattr(self, '_rgb_phase', 0)) % RGB_REQUEST_PERIOD == 0 or ondemand):
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
XGB_OWNER_STALE = 25                                                         

                                                                             
                                                                               
                                                                              
                                                                                 
                                                                            
                                                                            
                                                                            
                                                                              
                                                                             
 
                                                                                 
                                                                             
                                                                                
                                                                                
                                                    
                                                      
                                                                              
                          
                                                    
                                                   
                                                     
                                                                                 
                                                                                 
                                                                                 
                                                                              
                                                                               
                                                                              
                                                      
OUTER_ODDS = 9.17
def _outer_bar(strong_p, odds):
    o = odds * (strong_p / max(1e-9, 1.0 - strong_p))
    return float(o / (1.0 + o))
DT = 1.0 / 50.0
MEM_SIZE = 80
ACT_DIM = 6

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
                     "mtn_sess", "mtn_iname", "mtn_rgb_sess", "mtn_rgb_iname")
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
        self._stride_ok = True
        self._last_wp = None
        self._team_deny = None
        self._soft_deny = []
        self._standoff_ticks = 0
        self._skipped = 0
        self._served = 0
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
        self._in_support = None
        self._support_src = None
        self._support_fn = None
        self._support_relax = None
        self._outer_admits = 0
        self._stride_ok = True
        self._last_wp = None
        self._team_deny = None
        self._soft_deny = []
        self._standoff_ticks = 0
        self._skipped = 0
        self._served = 0
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
        det.seen = False
        if det.t - det.last_strong_t > _ks.STALE_TICKS:
            det.strong *= _ks.STALE_DECAY
            det.cand_strong *= _ks.STALE_DECAY
        det.conf = float(np.clip(det.strong / _ks.N_LATCH, 0.0, 1.0))
        return (det.conf, det.vic.copy(), det.vic[2] + 0.5 * det.height)

    def _gated_refuted(self, z):
        try:
            if self._det_refuted(z):
                return True
        except Exception:
            pass
        zxy = (float(z[0]), float(z[1]))
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
        if not allow_confirm:
            a = self._standoff(a, s)                         
        else:
            self._standoff_ticks = 0
        if not np.all(np.isfinite(a)):
            a = self.last_action.copy()
            a[5] = 0.0
        self._publish()
        self.last_action = a
        return a

                                                                                 
    def step(self, depth, state165, mem=None, rgb=None, mates=None, region=None,
             waypoint=None, stride_ok=True, allow_confirm=False, rgb_allowed=True,
             scan_r_scale=None, deny_xy=None, det_stride_phase=None, lr_allow=None,
             in_support=None, spoof_clue_xy=None, **_ignored):
        m = _to_np(mem)
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
        dx = float(s[0] - tgt[0])
        dy = float(s[1] - tgt[1])
        h = float(math.hypot(dx, dy))
        if h > STANDOFF_R_M + 1.5:
            self._standoff_ticks = 0
            return a
        self._standoff_ticks += 1
        if self._standoff_ticks > STANDOFF_TTL:
            self._soft_deny.append((np.array([tgt[0], tgt[1]], dtype=np.float64),
                                    c.tick + SOFT_DENY_TICKS))
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
                                                                  
                                                                      
                                                                             
                                          
                                                 
                                                                          
                                                                          
                                                                           
                                                                             
                                                                                
                                                                             
                                                                              
                                                                           
                                                                   
                                                                               
                                                                               
                                                                         
    CONF_AGL_HI = 4.10
    CONF_NEAR = 6.0
    CONF_Z_MARGIN = 2.8
                                                                               
                                                                               
                                                                          
                                                                               
                                                                        
                                                                                 
                                                                                 
                                                                                 
                                                                                
                                                                               
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
    ELECT_NDET = 4.0
    WEAK_LOCK = 0.30
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
    DET_STRIDE = 3
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
        self.lost = np.zeros(MAX_DRONES, dtype=np.int32)
        self.cooldown = np.zeros(MAX_DRONES, dtype=np.int64)
        self.best_h = np.full(MAX_DRONES, 1e9)
                         
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
            "corr_clamped": 0, "tilt_capped": 0,
            "rgb_requests": 0, "rgb_blocked": 0, "deadline_hits": 0,
            "pilot_exceptions": 0, "shape_fixups": 0, "blacklisted": 0,
            "teleport_resets": 0,
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
            return
        if not self.records:
            self.cand = None
            return
        self.cand = max(self.records, key=lambda r: (len(r["contrib"]), r["hits"], -r["t_first"]))

                                                                                
                                                                              
                                                                             
                                                            
     
                                                                                
                                                                                  
                                                                           
                                                                                
                                                                               
                                                                                 
                                                                                 
                                                                                 
                                                                               
                                                                 
                                                                             
                                                                              
                                                                            
                                                       
     
                                                                                  
                                                                                
                                                                                  
                                                                   

    def _release(self, reason, blame=True):
        i = self.confirmer
        if i < 0:
            return
        self.cooldown[i] = self.tick + self.COOLDOWN
        self.lost[i] = 0
        self.confirmer = -1
        self.stats["releases"] += 1
        if blame and self.cand is not None:
            self.cand["failed"].add(int(i))
            if len(self.cand["failed"]) >= 2:
                                                                             
                                                         
                self.blacklist.append(self.cand["xy"].copy())
                if len(self.blacklist) > 8:
                    self.blacklist.pop(0)
                self._drop_cand()
                self.cand = None
                self.stats["blacklisted"] += 1
        _log("release_%s" % reason, "confirmer %d released (%s)" % (i, reason))

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
                if held >= self.MIN_DWELL:
                    if self.lost[c] > self.LOST_TICKS:
                        self._release("lock_lost")
                    elif held > self.STALL_TICKS:
                        self._release("no_progress")
                                                                        
                                                                                  
                                                                                  
                                                                             
                     
                                                                                 
                                                                          
                                                                                
                                                                               
                                                                             
                                                                          
                                                                             
                                                                                 
                                                                                
                                                                                
                                                                                  
                                                  
                     
                                                                                
                                                                                
                                                                              
                                                                                
                                                                                 
                                                                              
        if self.confirmer < 0 and self.cand is not None:
            pool = [i for i in live
                    if beliefs[i] is not None
                    and beliefs[i]["lock"] >= self.ELECT_LOCK
                    and beliefs[i]["n_det"] > self.ELECT_NDET
                    and self.cooldown[i] <= t]
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
            if pool:
                pool.sort(key=lambda i: (-float(beliefs[i]["lock"]), int(i)))
                self.confirmer = int(pool[0])
                self.elect_tick = t
                self.lost[self.confirmer] = 0
                self.best_h[self.confirmer] = _horiz(pos[self.confirmer], self.cand["xy"])
                self.stats["elections"] += 1
        return self.confirmer

    def inject_blacklist(self, mems, pilots, live):
        xy = None if self.cand is None else self.cand["xy"]
        for i in live:
            mine = (i == self.confirmer)
            _mem_set(mems[i], M_CONFIRMER, 1.0 if mine else 0.0)
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
            if i == self.confirmer:
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
                    if h < self.OWN_EST_R and (cand is not None or self.confirmer >= 0):
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
        if not ((a - 1.35) <= dz_est <= (a + 0.35)):
            self.stats["guard_band_agl_incons"] += 1
        elif a > self.CONF_AGL_HI:                                                    
                                                                        
                                                                              
                                                                          
                                                                              
                                                                               
                                                                              
                                                                               
                                                                            
                                                                               
                                                                         
            sink = -(a - self.CONF_AGL_HI) * self.AGL_ENV_GAIN
            changed |= _enforce_vz_max(out[i], max(sink, -self.CONF_VZ_DN))
        lo = self.vz_envelope(a - self.AGL_FLOOR)
        if lo > -1.0:
            changed |= _enforce_vz_min(out[i], lo)
                                                                                 
                                                                               
                                                                                 
                                                                                
                                                                                 
                                                                         
        z_above = float(pos[i][2]) - (ref_z + self.CONF_Z_MARGIN)
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
        for i in live:
            if i == self.confirmer:
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

    DEADLINE = 0.380                                                          
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
        return {"est": est,
                "lock": float(lock) if lock is not None and np.isfinite(lock) else 0.0,
                "n_det": float(ndet) if ndet is not None and np.isfinite(ndet) else 0.0,
                "p_env": penv}

                                                                               
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

                        
        if T.tick == 0:
            T.pad0 = pos[0, 0:2].copy()                                              
        T.update_frozen(pos, vel)
        live = [i for i in range(n) if not T.frozen[i]]

                                                                            
                                                                            
                                                                       
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
                                                                              
                                                             
            wp = T.waypoint(i, pos[i], locks[i] >= T.DET_LOCK_THR)
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
                "deny_xy": None if (T.cand is None or i == T.confirmer) else T.cand["xy"],
                "lr_allow": bool(T.confirmer < 0
                                 and (T.cand is None or not _LR_NOCAND)),
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
        out = T.guard_agl_floor(out, agl, live)
        out = T.guard_keepout(out, pos, vel, live, beliefs)
        out = T.guard_confirmer_band(out, pos, agl, live, beliefs)
        out = T.separate(out, pos, vel, n, live)
                                                                            
                                                                           
                                                                            
                                                                              
                                                                               
                                                                             
        out = T.limit_correction(out, base, live)
        out = T.guard_agl_floor(out, agl, live)
                                                                               
                                                                           
                                                         
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
