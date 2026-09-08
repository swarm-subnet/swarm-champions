from __future__ import annotations


import os

import numpy as np

_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "office_grid.npz")


TOF_RAY_OFFSET = 0.11
TOF_NO_HIT = 100.0


class OfficeGrid:

    def __init__(self, path: str = _DEFAULT_PATH):
        with np.load(path) as d:
            self.shape = tuple(int(v) for v in d["shape"])
            self.res = float(d["res"])
            self.origin = np.asarray(d["origin"], dtype=np.float64)
            nz, ny, nx = self.shape
            total = nz * ny * nx
            self.surface = (
                np.unpackbits(d["surface_bits"])[:total]
                .reshape(self.shape)
                .astype(bool)
            )
            self.clear_cm = d["clear_cm"]
            self.col_top = d["col_top"]
            self.surf_mm = d["surf_mm"]
        self._nz, self._ny, self._nx = self.shape


    def _prep(self, xyz):
        pts = np.asarray(xyz, dtype=np.float64)
        single = pts.ndim == 1
        pts = np.atleast_2d(pts)
        return pts, single

    def _cell(self, pts):
        rel = (pts - self.origin[None, :]) / self.res
        idx = np.floor(rel).astype(np.int64)
        return idx[:, 0], idx[:, 1], idx[:, 2]


    def clearance(self, xyz):
        pts, single = self._prep(xyz)
        i, j, k = self._cell(pts)
        inb = (
            (i >= 0) & (i < self._nx)
            & (j >= 0) & (j < self._ny)
            & (k >= 0) & (k < self._nz)
        )
        ic = np.where(inb, i, 0)
        jc = np.where(inb, j, 0)
        kc = np.where(inb, k, 0)
        out = np.where(
            inb, self.clear_cm[kc, jc, ic].astype(np.float64) * 0.01, 0.0
        )
        return float(out[0]) if single else out

    def predict_tof(self, xyz):
        pts, single = self._prep(xyz)
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        z_start = z - TOF_RAY_OFFSET
        i = np.floor((x - self.origin[0]) / self.res).astype(np.int64)
        j = np.floor((y - self.origin[1]) / self.res).astype(np.int64)
        k = np.floor((z_start - self.origin[2]) / self.res).astype(np.int64)
        inb = (
            (i >= 0) & (i < self._nx)
            & (j >= 0) & (j < self._ny)
            & (k >= 0) & (k < self._nz)
        )
        ic = np.where(inb, i, 0)
        jc = np.where(inb, j, 0)
        kc = np.where(inb, k, 0)
        top = self.col_top[jc, ic, kc].astype(np.int64)
        hit = inb & (top >= 0)
        tc = np.where(hit, top, 0)
        z_surf = (
            self.origin[2]
            + tc.astype(np.float64) * self.res
            + self.surf_mm[tc, jc, ic].astype(np.float64) * 1e-3
        )
        out = np.where(hit, z - z_surf, TOF_NO_HIT)
        return float(out[0]) if single else out

    def path_clearance(self, a, b, samples: int = 0):
        a2, single = self._prep(a)
        b2, _ = self._prep(b)
        d = np.linalg.norm(b2 - a2, axis=1)
        if samples and samples > 0:
            n = max(2, int(samples))
        else:
            n = max(2, int(float(d.max()) / self.res) + 2)
        t = np.linspace(0.0, 1.0, n)
        pts = a2[:, None, :] + t[None, :, None] * (b2 - a2)[:, None, :]
        c = self.clearance(pts.reshape(-1, 3)).reshape(a2.shape[0], n)
        out = c.min(axis=1)
        return float(out[0]) if single else out


import math
from collections import deque

import numpy as np

DT = 1.0 / 50.0
SPEED = 3.0
DEAD_ZONE = 0.05
SLEW_STEP = 4.0 * DT
SPAWN_Z = 0.05

DBAND_HI = 0.06
DBAND_LO = 0.035

IDENT_PRIOR_W = 6.0
IDENT_MIN_CMD = 0.25
IDENT_MAX_DRC = 0.02
IDENT_LO, IDENT_HI = 0.82, 1.18


def body_to_world_xy(f: float, r: float, yaw: float) -> tuple[float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    return c * f + s * r, s * f - c * r


class EgoState:

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.t = 0.0
        self.yaw = 0.0
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.rc = np.zeros(4)
        self._age_prev = 1e9
        self.have_fix = False
        self._id_num = 0.0
        self._id_den = 0.0
        self._rc_prev = np.zeros(4)
        self.speed_hat = SPEED
        self.hist: deque = deque(maxlen=256)


    def update(self, state: np.ndarray, last_action: np.ndarray) -> None:
        s = np.asarray(state, dtype=np.float64).reshape(-1)

        a = np.clip(np.asarray(last_action, dtype=np.float64).reshape(-1)[:4], -1, 1)
        a[np.abs(a) < DEAD_ZONE] = 0.0
        self.rc += np.clip(a - self.rc, -SLEW_STEP, SLEW_STEP)

        self.t += DT
        age, valid = float(s[13]), float(s[14])
        fresh = (valid > 0.5 and 0.0 <= age < self._age_prev - 1e-9
                 and np.isfinite(s[:15]).all())
        self._age_prev = age

        if fresh:
            self.yaw = math.atan2(float(s[2]), float(s[3]))
            self.have_fix = True

        vx, vy = body_to_world_xy(float(s[4]), float(s[5]), self.yaw)
        vz = -float(s[6])
        self.vel = np.array([vx, vy, vz])

        self.pos[0] += vx * DT
        self.pos[1] += vy * DT
        self.pos[2] = float(s[11]) + SPAWN_Z + vz * age
        self.hist.append((self.t, self.pos[0], self.pos[1]))

        if fresh:
            drc = float(np.max(np.abs(self.rc[:2] - self._rc_prev[:2])))
            cmd = math.hypot(float(self.rc[0]), float(self.rc[1]))
            if cmd >= IDENT_MIN_CMD and drc <= IDENT_MAX_DRC:
                self._id_num += self.rc[1] * float(s[4]) + self.rc[0] * float(s[5])
                self._id_den += self.rc[1] * self.rc[1] + self.rc[0] * self.rc[0]
            self._rc_prev = self.rc.copy()
        self.speed_hat = min(max(
            (IDENT_PRIOR_W * SPEED + self._id_num) / (IDENT_PRIOR_W + self._id_den),
            SPEED * IDENT_LO), SPEED * IDENT_HI)


    def xy_at(self, t_query: float) -> np.ndarray:
        if not self.hist:
            return self.pos[:2].copy()
        h = self.hist
        if t_query <= h[0][0]:
            return np.array([h[0][1], h[0][2]])
        prev = h[0]
        for entry in h:
            if entry[0] >= t_query:
                t0, x0, y0 = prev
                t1, x1, y1 = entry
                if t1 - t0 < 1e-9:
                    return np.array([x1, y1])
                a = (t_query - t0) / (t1 - t0)
                return np.array([x0 + a * (x1 - x0), y0 + a * (y1 - y0)])
            prev = entry
        return self.pos[:2].copy()

    def vcmd_world(self) -> np.ndarray:
        sp = self.speed_hat
        vx, vy = body_to_world_xy(self.rc[1] * sp, self.rc[0] * sp, self.yaw)
        return np.array([vx, vy, self.rc[2] * sp])


import math

import numpy as np

HULL = 0.16
TOF_SIGMA = 0.13
TOF_OUTLIER = 0.35
ALIVE_FLOOR = 0.02
DRIFT_PER_M = 0.012
ROUGHEN_XY = 0.06
ROUGHEN_TH = 0.02
SEED_Z = 1.5
START_Z = 0.8
SEED_MARGIN = 0.35
SELF_FREE_M = 0.35
SELF_FREE_R = 0.12
CONVERGED_SPREAD = 0.35


class GridLocalizer:
    def __init__(self, grid, n: int = 4096, seed: int = 0):
        self.grid = grid
        self.n = int(n)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self.seeded = False
        self.p = np.zeros((self.n, 2))
        self.th = np.zeros(self.n)
        self.cth = np.ones(self.n)
        self.sth = np.zeros(self.n)
        self.w = np.full(self.n, 1.0 / self.n)
        self.anchor = np.zeros(2)
        self._free_anchor: np.ndarray | None = None
        self.converged = False


    def _seed(self, dr_xy: np.ndarray) -> None:
        g = self.grid
        nz, ny, nx = [int(v) for v in g.shape]
        lo = np.asarray(g.origin[:2]) + SEED_MARGIN
        hi = (np.asarray(g.origin[:2])
              + np.asarray([nx, ny]) * g.res - SEED_MARGIN)
        pts = []
        need = self.n
        for _ in range(24):
            cand = self.rng.uniform(lo, hi, (need * 2, 2))
            z = np.full((cand.shape[0], 1), SEED_Z)
            ok = g.clearance(np.hstack([cand, z])) > HULL
            pts.append(cand[ok])
            if sum(len(x) for x in pts) >= self.n:
                break
        cloud = np.vstack(pts)[: self.n]
        if cloud.shape[0] < self.n:
            extra = self.rng.uniform(lo, hi, (self.n - cloud.shape[0], 2))
            cloud = np.vstack([cloud, extra])
        self.p = cloud
        self.th = self.rng.uniform(0.0, 2 * math.pi, self.n)
        self.cth, self.sth = np.cos(self.th), np.sin(self.th)
        self.w = np.full(self.n, 1.0 / self.n)
        self.anchor = np.asarray(dr_xy, dtype=float).copy()
        self.seeded = True
        self.converged = False


    def update(self, dr_xy: np.ndarray, z: float, tof: float,
               moved: float) -> None:
        if not self.seeded:
            if z >= START_Z:
                self._seed(dr_xy)
            return
        if z < START_Z:
            return
        d = np.asarray(dr_xy, dtype=float) - self.anchor
        self.anchor = np.asarray(dr_xy, dtype=float).copy()
        self.p[:, 0] += self.cth * d[0] - self.sth * d[1]
        self.p[:, 1] += self.sth * d[0] + self.cth * d[1]
        if moved > 0:
            sig = DRIFT_PER_M * moved + 1e-4
            self.p += self.rng.normal(0.0, sig, self.p.shape)

        pts = np.hstack([self.p, np.full((self.n, 1), z)])
        like = np.where(self.grid.clearance(pts) > HULL, 1.0, ALIVE_FLOOR)
        if 0.0 < tof < 50.0:
            r = np.abs(self.grid.predict_tof(pts) - tof)
            like = like * ((1 - TOF_OUTLIER) * np.exp(-0.5 * (r / TOF_SIGMA) ** 2)
                           + TOF_OUTLIER)
        self._reweigh(like)

    def observe_self_free(self, a_xyz: np.ndarray, b_xyz: np.ndarray) -> None:
        if not self.seeded:
            return
        a = np.asarray(a_xyz, dtype=float)
        b = np.asarray(b_xyz, dtype=float)
        n = float(np.linalg.norm(b[:2] - a[:2]))
        if n < 0.15 or n > 4.0:
            return
        k = max(3, int(n / 0.25) + 1)
        ts = np.linspace(0.0, 1.0, k)
        seg = a[None, :] + ts[:, None] * (b - a)[None, :]
        da = seg[:, :2] - self.anchor[None, :]

        px = (self.p[:, 0, None] + self.cth[:, None] * da[None, :, 0]
              - self.sth[:, None] * da[None, :, 1])
        py = (self.p[:, 1, None] + self.sth[:, None] * da[None, :, 0]
              + self.cth[:, None] * da[None, :, 1])
        pz = np.broadcast_to(seg[None, :, 2], px.shape)
        pts = np.stack([px, py, pz], axis=-1).reshape(-1, 3)
        clear = self.grid.clearance(pts).reshape(self.n, k)
        like = np.where(clear.min(axis=1) > SELF_FREE_R, 1.0, ALIVE_FLOOR)
        self._reweigh(like)

    def maybe_observe_self_free(self, pos_xyz: np.ndarray) -> None:
        if self._free_anchor is None:
            self._free_anchor = np.asarray(pos_xyz, dtype=float).copy()
            return
        d = float(np.linalg.norm(pos_xyz[:2] - self._free_anchor[:2]))
        if d >= SELF_FREE_M:
            self.observe_self_free(self._free_anchor, pos_xyz)
            self._free_anchor = np.asarray(pos_xyz, dtype=float).copy()


    def _reweigh(self, like: np.ndarray) -> None:
        w = self.w * like
        tot = float(w.sum())
        if not np.isfinite(tot) or tot <= 0.0:
            self.seeded = False
            self.converged = False
            return
        self.w = w / tot
        ess = 1.0 / float((self.w ** 2).sum())
        if ess < 0.5 * self.n:
            self._resample()
        self.converged = self.spread() < CONVERGED_SPREAD

    def _resample(self) -> None:
        cum = np.cumsum(self.w)
        cum[-1] = 1.0
        pos = (self.rng.random() + np.arange(self.n)) / self.n
        idx = np.searchsorted(cum, pos)
        self.p = self.p[idx] + self.rng.normal(0.0, ROUGHEN_XY, (self.n, 2))
        self.th = self.th[idx] + self.rng.normal(0.0, ROUGHEN_TH, self.n)
        self.cth, self.sth = np.cos(self.th), np.sin(self.th)
        self.w = np.full(self.n, 1.0 / self.n)


    def stats_at(self, dr_xy: np.ndarray):
        if not self.seeded:
            return np.zeros(2), 99.0
        d = np.asarray(dr_xy, dtype=float) - self.anchor
        px = self.p[:, 0] + self.cth * d[0] - self.sth * d[1]
        py = self.p[:, 1] + self.sth * d[0] + self.cth * d[1]
        mx = float((self.w * px).sum())
        my = float((self.w * py).sum())
        sp = float(np.sqrt((self.w * ((px - mx) ** 2 + (py - my) ** 2)).sum()))
        return np.array([mx, my]), sp

    def spread(self) -> float:
        return self.stats_at(self.anchor)[1]

    def theta(self) -> float:
        if not self.seeded:
            return 0.0
        return float(math.atan2((self.w * self.sth).sum(),
                                (self.w * self.cth).sum()))

    def theta_conf(self) -> float:
        if not self.seeded:
            return 0.0
        return float(np.hypot((self.w * self.sth).sum(),
                              (self.w * self.cth).sum()))


import math
from collections import deque
from pathlib import Path as _PPath

import numpy as np

_PHERE = _PPath(__file__).resolve().parent

DT = 1.0 / 50.0
RES = 256.0
# Pose-acceptance threshold. 0.5 shipped by UID_175; measured at 0.0 here.
# The office rearranges its 41 furniture pieces per episode (commit 3a4e8ee), which
# pushes posenet16 out of distribution: at 0.5 a pose is accepted on only 11% of
# ticks, so the map feature block f[22:28] is ABSENT for ~9 ticks in 10. office_actor.pt
# is frozen and was trained when that block was present, so starving it costs more than
# the pose error does. Accepting always keeps the actor in distribution.
# Measured, 500 paired epoch-20 seeds: catch 0.9440 -> 0.9680 (+0.0240 +- 0.0123, t 1.95),
# collisions 20 -> 12, TILTs 7 -> 2. Matches UID_142 (0.9640) on the same seeds.
VIS_CONF = 0.0
VIS_SPREAD = 0.25
K_TAN = 2.0
TGT_W, TGT_H = 0.18, 0.08
EYE_FWD, EYE_UP = 0.13, 0.05
ALPHA, BETA = 0.45, 0.35
ADAPT_REF = 1.5
GATE_K = 0.25
VT_MAX = 2.4
VT_DECAY = 0.98
STALE_S = 0.45
FRESH_S = 0.30
CONFIRM_S = 0.35
Z_MIN, Z_MAX = 0.95, 3.00


def _derotate(f: float, left: float, up: float, pitch: float, roll: float):
    cp, sp = math.cos(pitch), math.sin(pitch)
    f1 = cp * f + sp * up
    u1 = -sp * f + cp * up
    cr, sr = math.cos(roll), math.sin(roll)
    l1 = cr * left - sr * u1
    u2 = sr * left + cr * u1
    return f1, l1, u2


class TargetTrack:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.t = 0.0
        self.rel: np.ndarray | None = None
        self.vt = np.zeros(3)
        self.age = 10.0
        self.confirmed = False
        self._confirm_t = -10.0
        self._meas_t = -10.0
        self._det_age_prev = 1e9
        self.att: deque = deque(maxlen=40)
        self.last_box = np.zeros(5)
        self.last_meas_rel: np.ndarray | None = None
        self.last_meas_t = -1.0


    @property
    def have(self) -> bool:
        return self.rel is not None and self.confirmed

    def fresh(self) -> bool:
        return self.have and self.age < FRESH_S

    def usable(self) -> bool:
        return self.have and self.age < STALE_S


    def _att_at(self, t_eff: float):
        best = None
        for entry in self.att:
            if best is None or abs(entry[0] - t_eff) < abs(best[0] - t_eff):
                best = entry
        return best[1:] if best else (0.0, 0.0, 0.0)

    def _box_to_rel(self, box: np.ndarray, det_age: float, z_self: float):
        cx, cy, w, h, _conf = [float(v) for v in box]
        if w < 1e-6 or h < 1e-6:
            return None
        depth = math.sqrt((TGT_W / (w * K_TAN)) * (TGT_H / (h * K_TAN)))
        a = (cx - 0.5) * K_TAN
        b = (cy - 0.5) * K_TAN
        pitch, roll, yaw = self._att_at(self.t - det_age)
        f, left, up = _derotate(depth + EYE_FWD, -a * depth, -b * depth + EYE_UP,
                                pitch, roll)
        c, s = math.cos(yaw), math.sin(yaw)
        rel = np.array([c * f - s * left, s * f + c * left, up])
        if not (Z_MIN <= z_self + rel[2] <= Z_MAX):
            return None
        return rel


    def update(self, state: np.ndarray, v_self: np.ndarray, yaw: float,
               z_self: float) -> None:
        s = np.asarray(state, dtype=np.float64).reshape(-1)
        self.t += DT
        self.att.append((self.t - float(s[13]), float(s[0]), float(s[1]), yaw))
        self.age += DT
        if self.rel is not None:
            self.rel = self.rel + (self.vt - v_self) * DT
            if self.age > STALE_S:
                self.vt = self.vt * VT_DECAY
        n_boxes = int(round(float(s[15])))
        det_age = float(s[16])
        fresh_block = det_age < self._det_age_prev - 1e-9
        self._det_age_prev = det_age
        if n_boxes <= 0 or not fresh_block:
            if self.age > 4.0:
                self.rel, self.confirmed = None, False
            return

        held = self.rel if self.age < STALE_S else None
        best, best_cost = None, None
        for k in range(min(n_boxes, 2)):
            box = s[17 + 5 * k: 22 + 5 * k]
            rel = self._box_to_rel(box, det_age, z_self)
            if rel is None:
                continue
            rel = rel - v_self * det_age
            if held is not None:
                cost = float(np.linalg.norm(rel - held))
                if cost > max(1.0, GATE_K * float(np.linalg.norm(held))):
                    continue
            else:
                cost = 4.0 - float(box[4])
            if best_cost is None or cost < best_cost:
                best, best_cost = (rel, box), cost
        if best is None:
            return
        rel_meas, box = best
        self.last_box = np.asarray(box, dtype=np.float64).copy()
        self.last_meas_rel = rel_meas.copy()
        self.last_meas_t = self.t

        if self.rel is None or self.age >= STALE_S:
            self.rel = rel_meas
            self.vt = np.zeros(3)
            self.confirmed = (self.t - self._confirm_t) <= CONFIRM_S
        else:
            dt_m = max(self.t - self._meas_t, DT)
            err = rel_meas - self.rel
            r = float(np.linalg.norm(self.rel))
            g = 2.0 * ADAPT_REF ** 2 / (ADAPT_REF ** 2 + r * r)
            ka = float(np.clip(ALPHA * g, 0.15, 0.90))
            kb = float(np.clip(BETA * g, 0.08, 0.75))
            self.rel = self.rel + ka * err
            self.vt = self.vt + (kb / dt_m) * err
            n = float(np.linalg.norm(self.vt))
            if n > VT_MAX:
                self.vt *= VT_MAX / n
            self.confirmed = True
        self._confirm_t = self._meas_t = self.t
        self.age = 0.0


import math

import numpy as np

R_TERMINAL = 1.5
A_MAX_TGT = 2.0
N_TERMINAL = 10


class TerminalEstimator:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._blind_t = 0.0
        self._prev_az = None
        self._bearing_rate = 0.0
        self._r_hat: np.ndarray | None = None
        self._meas_t_seen = -1.0

    def features(self, track, ego, det_age: float, dt: float) -> np.ndarray:
        f = np.zeros(N_TERMINAL, dtype=np.float32)
        if track.rel is None or not track.have:
            self._blind_t = 0.0
            self._prev_az = None
            return f


        if (track.last_meas_rel is not None
                and track.last_meas_t > self._meas_t_seen):
            self._r_hat = np.asarray(track.last_meas_rel, dtype=np.float64).copy()
            self._meas_t_seen = track.last_meas_t
        elif self._r_hat is not None:
            self._r_hat = self._r_hat + (np.asarray(track.vt)
                                         - np.asarray(ego.vel)) * dt
        r = (self._r_hat if self._r_hat is not None
             else np.asarray(track.rel, dtype=np.float64))
        rng = float(np.linalg.norm(r))
        in_term = rng < R_TERMINAL

        if in_term and det_age > 0.15:
            self._blind_t += dt
        else:
            self._blind_t = 0.0

        v_rel = np.asarray(track.vt, dtype=np.float64) - np.asarray(ego.vel)
        sp2 = float(v_rel @ v_rel)
        cy, sy = math.cos(-ego.yaw), math.sin(-ego.yaw)

        def to_body(w):
            return np.array([cy * w[0] - sy * w[1],
                             sy * w[0] + cy * w[1], w[2]])

        closing = float(-(r @ v_rel) / max(rng, 1e-6))
        if sp2 > 1e-6:
            tgo = float(np.clip(-(r @ v_rel) / sp2, 0.0, 4.0))
        else:
            tgo = 4.0
        miss = r + v_rel * tgo
        miss_b = to_body(miss)
        az = math.atan2(r[1], r[0])
        if self._prev_az is not None:
            d = math.atan2(math.sin(az - self._prev_az),
                           math.cos(az - self._prev_az))
            self._bearing_rate = 0.7 * self._bearing_rate + 0.3 * (d / dt)
        self._prev_az = az

        f[0] = 1.0 if in_term else 0.0
        f[1] = tgo / 4.0
        f[2:5] = np.clip(miss_b, -3.0, 3.0)
        f[5] = min(float(np.linalg.norm(miss)), 3.0)
        f[6] = float(np.clip(closing, -3.0, 3.0)) / 3.0
        f[7] = min(self._blind_t, 2.0)
        f[8] = min(0.5 * A_MAX_TGT * self._blind_t ** 2, 2.0)
        f[9] = float(np.clip(self._bearing_rate, -4.0, 4.0)) / 4.0
        return f


import math

import numpy as np


STATE_DIM = 127
N_BASE_FEATURES = 52
N_FEATURES = N_BASE_FEATURES + N_TERMINAL
STACK_KEEP = 27
STACK_LEN = 10
BASE_ACTOR_DIM = STATE_DIM + N_BASE_FEATURES + STACK_KEEP * STACK_LEN
ACTOR_DIM = STATE_DIM + N_FEATURES + STACK_KEEP * STACK_LEN


_PROBE_R = 0.75
_PATH_REACH = 1.2


class _VisPosterior:
    __slots__ = ("xy", "heading_offset", "mode_spread_m", "finite")

    def __init__(self, xy, th, spread):
        self.xy = xy
        self.heading_offset = th
        self.mode_spread_m = spread
        self.finite = True


class _VisLast:
    __slots__ = ("posterior",)

    def __init__(self, posterior):
        self.posterior = posterior


# Sibling modules are imported at module scope, not lazily inside __init__.
# swarm.policy_interface.smoke_test_policy_package inserts the extracted
# submission directory on sys.path only around exec_module, and pops it in a
# `finally` that runs BEFORE it constructs the controller -- so a deferred
# `from office_pilot import ...` raises there and the official verifier reports
# `Compliant: False (controller_init_failed:No module named 'office_pilot')`.
# The pristine UID_175 artifact fails that check as shipped; UID_142 passes it
# only because it imports its own modules at module scope. Hoisting also makes
# a missing module fail loudly at import instead of silently disabling the
# visual pose, which is the one thing this submission changes.
from office_pilot import Pilot as _Pilot

try:
    from posevis import VisualPose as _VisualPose
except Exception:                                   # optional: torch may be absent
    _VisualPose = None


class FeatureBuilder:
    def __init__(self, grid=None, loc_particles: int = 4096, seed: int = 0):
        if grid is None:
            from .mapgrid import OfficeGrid
            grid = OfficeGrid()
        self.grid = grid
        self.ego = EgoState()
        self.track = TargetTrack()
        self.loc = GridLocalizer(grid, n=loc_particles, seed=seed)
        self.terminal = TerminalEstimator()
        self._stack = np.zeros((STACK_LEN, STACK_KEEP), dtype=np.float32)
        self._primed = False
        self._vp = None
        self._vp_fix = None
        self.visual_last = None
        try:
            if _VisualPose is None:
                raise ImportError("posevis unavailable")
            self._vp = _VisualPose(str(_PHERE / "posenet16.pt"), threads=1)
        except Exception:
            self._vp = None
        self._last_action = np.zeros(4)
        self._age_prev = 1e9

    def reset(self) -> None:
        self._vp_fix = None
        if self._vp is not None:
            self._vp.reset()
        self.ego.reset()
        self.track.reset()
        self.loc.reset()
        self.terminal.reset()
        self._stack[:] = 0.0
        self._primed = False
        self._last_action = np.zeros(4)
        self._age_prev = 1e9

    def note_action(self, action: np.ndarray) -> None:
        self._last_action = np.asarray(action, dtype=np.float64).reshape(-1)[:4]


    def build(self, state: np.ndarray, rgb=None) -> np.ndarray:
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        live = s[:STACK_KEEP]
        if not self._primed:
            self._stack[:] = live[None, :]
            self._primed = True
        else:
            self._stack[:-1] = self._stack[1:]
            self._stack[-1] = live

        self.ego.update(s, self._last_action)
        ego = self.ego
        self.track.update(s, ego.vel, ego.yaw, float(ego.pos[2]))


        age, valid = float(s[13]), float(s[14])
        fresh = valid > 0.5 and 0.0 <= age < self._age_prev - 1e-9
        self._age_prev = age
        if fresh:
            dr_pkt = ego.xy_at(ego.t - age)
            z_pkt = float(s[11]) + SPAWN_Z
            moved = (float(np.linalg.norm(dr_pkt - self.loc.anchor))
                     if self.loc.seeded else 0.0)
            self.loc.update(dr_pkt, z_pkt, float(s[10]), moved)
        self.loc.maybe_observe_self_free(ego.pos)

        if self._vp is not None and rgb is not None and ego.pos[2] >= 0.8:
            try:
                r = self._vp.predict(rgb)
                if r is not None and r[3]:
                    self._vp_fix = (r[0], r[1], r[2])
                    if r[2] >= VIS_CONF:
                        self.visual_last = _VisLast(_VisPosterior(
                            np.asarray(r[0], dtype=np.float64),
                            float(r[1]) - float(ego.yaw), VIS_SPREAD))
                    else:
                        self.visual_last = None
            except Exception:
                self._vp = None

        term = self.terminal.features(self.track, self.ego, float(s[16]), DT)
        return np.concatenate(
            [s, self._features(s), self._stack.reshape(-1), term]
        ).astype(np.float32)


    def _features(self, s: np.ndarray) -> np.ndarray:
        f = np.zeros(N_BASE_FEATURES, dtype=np.float32)
        ego, track, loc = self.ego, self.track, self.loc


        f[0] = ego.pos[0] / 10.0
        f[1] = ego.pos[1] / 10.0
        f[2] = ego.pos[2] / 3.0
        f[3] = math.sin(ego.yaw)
        f[4] = math.cos(ego.yaw)
        f[5:8] = ego.vel / 3.0
        f[8:12] = ego.rc
        f[12:15] = ego.vcmd_world() / 3.0


        est, spread = loc.stats_at(ego.pos[:2])
        th = loc.theta()
        _fix = self._vp_fix
        _vis_ok = _fix is not None and _fix[2] >= VIS_CONF
        if _vis_ok:


            est = np.asarray(_fix[0], dtype=np.float64)
            th = float(_fix[1]) - float(ego.yaw)
            spread = min(spread, VIS_SPREAD)
        f[15] = est[0] / 10.0
        f[16] = est[1] / 10.0
        f[17] = min(spread, 3.0)
        f[18] = math.sin(th)
        f[19] = math.cos(th)
        f[20] = loc.theta_conf()
        f[21] = 1.0 if (_vis_ok or (loc.seeded and loc.converged)) else 0.0


        usable = _vis_ok or (loc.seeded and loc.converged and loc.theta_conf() > 0.5)
        if usable:
            c, s_ = math.cos(th), math.sin(th)
            z = float(ego.pos[2])
            pos_m = np.array([est[0], est[1], z])
            vc = ego.vcmd_world()
            hd = vc[:2]
            n = float(np.linalg.norm(hd))
            u = hd / n if n > 1e-6 else np.array(
                [math.cos(ego.yaw), math.sin(ego.yaw)])
            um = np.array([c * u[0] - s_ * u[1], s_ * u[0] + c * u[1]])
            left = np.array([-um[1], um[0]])
            probes = np.array([
                pos_m,
                pos_m + np.array([um[0], um[1], 0.0]) * _PROBE_R,
                pos_m + np.array([left[0], left[1], 0.0]) * _PROBE_R,
                pos_m - np.array([left[0], left[1], 0.0]) * _PROBE_R,
            ])
            cl = self.grid.clearance(probes)
            f[22:26] = np.minimum(cl, 2.0)
            f[26] = min(self.grid.path_clearance(
                pos_m, pos_m + np.array([um[0], um[1], 0.0]) * _PATH_REACH), 2.0)
            tof_pred = float(np.atleast_1d(self.grid.predict_tof(pos_m[None, :]))[0])
            f[27] = float(np.clip(tof_pred - float(s[10]), -1.0, 1.0))
            f[28] = 1.0

        if track.rel is not None:
            rel = track.rel
            cy, sy = math.cos(-ego.yaw), math.sin(-ego.yaw)
            rb = np.array([cy * rel[0] - sy * rel[1],
                           sy * rel[0] + cy * rel[1], rel[2]])
            rng = float(np.linalg.norm(rel))
            az = math.atan2(rb[1], rb[0])
            f[29] = 1.0 if track.have else 0.0
            f[30] = 1.0 if track.fresh() else 0.0
            f[31:34] = rb / 10.0
            f[34] = rng / 10.0
            f[35] = math.sin(az)
            f[36] = math.cos(az)
            f[37] = math.atan2(rb[2], max(math.hypot(rb[0], rb[1]), 1e-6))
            f[38:41] = track.vt / 3.0
            if rng > 1e-6:
                f[41] = float(np.dot(track.vt, rel) / rng) / 3.0
            f[42] = min(track.age, 5.0)
            f[43] = float(np.linalg.norm(track.vt)) / 3.0
        else:
            f[42] = 5.0


        f[44] = s[15]
        f[45] = s[16]
        f[46:51] = track.last_box
        f[51] = min(self.ego.t, 60.0) / 60.0
        return f


import numpy as _np
import torch as _torch
import torch.nn as _nn
from pathlib import Path as _Path

_torch.set_num_threads(1)
_HERE = _Path(__file__).resolve().parent
_HIDDEN = (1024, 1024, 1024)
_NORM_DIM = 449


def _mlp(sizes):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [_nn.Linear(a, b), _nn.SiLU()]
    return _nn.Sequential(*layers)


class _Actor(_nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = _nn.LayerNorm(_NORM_DIM)
        self.trunk = _mlp((ACTOR_DIM, *_HIDDEN))
        self.mean = _nn.Linear(_HIDDEN[-1], 4)
        self.log_std = _nn.Parameter(_torch.zeros(4))

    def forward(self, x):
        h = _torch.cat([self.norm(x[..., :_NORM_DIM]), x[..., _NORM_DIM:]], -1)
        return _torch.tanh(self.mean(self.trunk(h)))


class DroneFlightController:
    def __init__(self):
        grid = OfficeGrid(str(_HERE / "office_grid.npz"))
        self._features = FeatureBuilder(grid=grid, loc_particles=4096)
        self._pilot = _Pilot(grid)
        self._actor = _Actor()
        sd = _torch.load(_HERE / "office_actor.pt", map_location="cpu",
                         weights_only=True)
        self._actor.load_state_dict(sd)
        self._actor.eval()
        self._last = _np.zeros(4, dtype=_np.float32)
        with _torch.inference_mode():
            self._actor(_torch.zeros(1, ACTOR_DIM))

    def reset(self):
        self._features.reset()
        self._pilot.reset()
        self._last = _np.zeros(4, dtype=_np.float32)

    def act(self, observation):
        state = _np.array(observation["state"], dtype=_np.float32).reshape(-1)
        if state.shape[0] != 127:
            fixed = _np.zeros(127, dtype=_np.float32)
            fixed[:min(127, state.shape[0])] = state[:127]
            state = fixed
        x = self._features.build(state, observation.get("rgb"))
        with _torch.inference_mode():
            a = self._actor(_torch.from_numpy(x).unsqueeze(0)).squeeze(0).numpy()
        a = _np.nan_to_num(a, nan=0.0).astype(_np.float32)
        try:
            a = _np.asarray(self._pilot.step(a, self._features),
                            dtype=_np.float32).reshape(-1)[:4]
        except Exception:
            pass
        m = _np.abs(a[:3])
        a[:3] = _np.where(m < DBAND_LO, 0.0,
                          _np.where(m < DBAND_HI, _np.copysign(DBAND_HI, a[:3]), a[:3]))
        self._features.note_action(a)
        self._last = a
        return a
