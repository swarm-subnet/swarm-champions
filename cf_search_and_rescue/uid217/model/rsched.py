
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

_DMIN, _DMAX = 0.5, 30.0
_CAMF, _CAMU = 0.13, 0.05
CELL = 1.5
R_DISC = 30.0
VIC_Z = 0.55
SLACK = 5
BOX = {"city": 25.0, "village": 25.0}
R_MAX = 34.0
P_PX_R = np.array([0.0, 20.0, 26.0, 30.0, 34.0])
P_PX_P = np.array([1.0, 1.0, 0.8, 0.55, 0.3])
G_WIN = 300
FEATS = ["e_new", "e_rel", "m_vis", "e_near", "e_mid", "e_far", "unseen", "open", "d_clue", "agl", "spd",
         "pitch", "yawr", "f_left", "t_left", "t_since", "city"]
T_END = 3000


def _axes(rpy):
    r, p, y = (float(rpy[0]), float(rpy[1]), float(rpy[2]))
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y))
    fwd = np.array([cy * cp, sy * cp, -sp], np.float64)
    up = np.array([cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr], np.float64)
    return fwd, up, np.cross(fwd, up)


def _vmax(d, k):
    p = np.pad(d, ((k, k), (0, 0)), mode="edge")
    out = p[0:256].copy()
    for i in range(1, 2 * k + 1):
        np.maximum(out, p[i:i + 256], out=out)
    return out


class Tracker:

    def __init__(self, clue, mtype):
        self.clue = np.asarray(clue, np.float64).reshape(-1)[0:2].copy()
        g = np.arange(-R_DISC, R_DISC + 1e-9, CELL)
        X, Y = np.meshgrid(g, g)
        m = X ** 2 + Y ** 2 <= (R_DISC + 0.5 * CELL) ** 2
        self.xy = np.stack([X[m] + self.clue[0], Y[m] + self.clue[1]], 1)
        w = np.ones(len(self.xy))
        b = BOX.get(mtype)
        if b:
            w[(np.abs(self.xy) > b + 1.0).any(1)] = 0.1
        self.prior = w / w.sum()
        self.cov = np.zeros(len(self.xy))
        self.city = 1.0 if mtype == "city" else 0.0
        self.gh = []
        self.g = None
        self.last_frame = None
        self.n_frames = 0

    def ground(self, tick, z, agl):
        if agl < 19.5:
            self.gh.append((tick, float(z) - float(agl)))
        while self.gh and tick - self.gh[0][0] > G_WIN:
            self.gh.pop(0)
        if self.gh:
            self.g = min(v for _, v in self.gh)
        elif self.g is None:
            self.g = float(z) - min(float(agl), 19.5)
        return self.g

    def view(self, pos, rpy, depth):
        fwd, up, right = _axes(rpy)
        cam = np.asarray(pos, np.float64)[0:3] + fwd * _CAMF + up * _CAMU
        n = len(self.xy)
        P = np.empty((n, 3))
        P[:, 0:2] = self.xy - cam[0:2]
        P[:, 2] = (self.g if self.g is not None else float(pos[2]) - 5.0) + VIC_Z - cam[2]
        zc = P @ fwd
        ok = zc > 1.0
        zs = np.where(ok, zc, 1.0)
        u = (P @ right) / zs
        v = (P @ up) / zs
        ok &= (np.abs(u) <= 1.0) & (np.abs(v) <= 1.0) & (zc <= R_MAX)
        d = np.asarray(depth, np.float32).reshape(256, 256) * (_DMAX - _DMIN) + _DMIN
        col = np.clip(np.round((u + 1.0) * 128.0 - 0.5), 0, 255).astype(np.int64)
        row = np.clip(np.round((1.0 - v) * 128.0 - 0.5), 0, 255).astype(np.int64)
        dm = _vmax(d, SLACK)[row, col]
        ok &= (dm >= zc - 1.0) | (dm >= 29.0)
        w = np.where(ok, np.interp(zc, P_PX_R, P_PX_P), 0.0)
        return w, zc, d

    def frame(self, w):
        self.cov = 1.0 - (1.0 - self.cov) * (1.0 - w)

    def step(self, tick, st, depth, frames_used, delivered):
        pos, rpy, vel, angv = st[0:3], st[3:6], st[6:9], st[9:12]
        self.ground(tick, pos[2], st[12])
        w, zc, d = self.view(pos, rpy, depth)
        if delivered:
            self.frame(w)
            self.last_frame = tick
            self.n_frames += 1
        pu = self.prior * (1.0 - self.cov)
        e = pu * w
        unseen = float(pu.sum())
        e_new = float(e.sum())
        f = [e_new, e_new / max(unseen, 1e-4), float((self.prior * w).sum()),
             float(e[zc < 15.0].sum()), float(e[(zc >= 15.0) & (zc < 25.0)].sum()), float(e[zc >= 25.0].sum()),
             unseen, float((d >= 29.0).mean()),
             float(np.hypot(float(pos[0]) - self.clue[0], float(pos[1]) - self.clue[1])),
             float(st[12]), float(np.hypot(float(vel[0]), float(vel[1]))), float(rpy[1]), float(angv[2]),
             float(40 - frames_used), float(T_END - tick) / 50.0,
             min(60.0, (tick - self.last_frame) / 50.0) if self.last_frame is not None else 60.0, self.city]
        return np.asarray(f, np.float64), w


class GBM:

    def __init__(self, d):
        self.base = float(d["base"])
        self.trees = [(np.asarray(t["feat"], np.int64), np.asarray(t["thr"], np.float64),
                       np.asarray(t["left"], np.int64), np.asarray(t["right"], np.int64),
                       np.asarray(t["leaf"], bool), np.asarray(t["value"], np.float64),
                       np.asarray(t["nan_left"], bool)) for t in d["trees"]]

    def logit(self, x):
        s = self.base
        for feat, thr, left, right, leaf, val, nl in self.trees:
            i = 0
            while not leaf[i]:
                xv = x[feat[i]]
                i = left[i] if (xv <= thr[i] or (nl[i] and xv != xv)) else right[i]
            s += val[i]
        return s

    def prob(self, x):
        return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, self.logit(x)))))


class Sched:

    def __init__(self, clue, mtype, model_path, gap=10, max_gap=150, p_min=0.02, reserve=6, kappa=1.0):
        d = json.loads(Path(model_path).read_text())
        self.gbm = GBM(d["gbm"])
        self.q = np.asarray(d["p_quantiles"], np.float64)
        self.tr = Tracker(clue, mtype)
        self.gap, self.max_gap, self.p_min, self.reserve = int(gap), int(max_gap), float(p_min), int(reserve)
        self.kappa = float(kappa)
        self.p = 0.0
        self.theta = 1.0
        self.n_want = 0
        self.n_force = 0

    def observe(self, tick, st13, depth, frames_used, delivered):
        self.x, self.w = self.tr.step(tick, st13, depth, frames_used, delivered)
        self.p = self.gbm.prob(self.x)
        return self.p

    def want(self, tick, last_req, frames_left):
        since = tick - last_req
        if since < self.gap or frames_left <= self.reserve:
            return False
        ticks_left = max(1, T_END - tick)
        a = min(1.0, self.kappa * (frames_left - self.reserve) * self.gap / ticks_left)
        k = len(self.q) - 1
        self.theta = max(self.p_min, float(np.interp(1.0 - a, np.linspace(0.0, 1.0, k + 1), self.q)))
        if self.p >= self.theta:
            self.n_want += 1
            return True
        if since >= self.max_gap:
            self.n_force += 1
            return True
        return False
