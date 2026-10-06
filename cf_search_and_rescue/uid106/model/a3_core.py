from __future__ import annotations
import math
import os
import time
import numpy as np
try:
    from scipy.ndimage import distance_transform_edt as _EDT
except Exception:
    _EDT = None
NG = 95
XS = np.arange(-47, 48, 1.0)
GX, GY = np.meshgrid(XS, XS)
AX = np.maximum(np.abs(GX), np.abs(GY))
B30 = AX <= 30.0
B45 = AX <= 45.0
DT = 0.02
DMIN, DMAX = (0.5, 30.0)
CAMF, CAMU = (0.13, 0.05)
_PX = (np.arange(0, 256, 4) + 0.5) / 128.0 - 1.0
U64 = np.broadcast_to(_PX[None, :], (64, 64)).ravel().copy()
V64 = np.broadcast_to(-_PX[:, None], (64, 64)).ravel().copy()
GXf = GX.ravel()
GYf = GY.ravel()
DEF = dict(V=2.9, H=4.5, H2=7.5, H_lp=4.0, radii=(10.0, 14.0, 18.0, 22.0), n_az=8, K=5, lat=10.0, n_yaw=8, tau=0.8, cap=0.85, floor=0.001, q=0.094, dil=0.5, lp_share=0.27, unk_vis=0.4, g0=0.03, gk=2.0, reserve=4, gap=12, yaw_tol=5.0, tilt_tol=15.0, fr_move=2.5, fr_turn=12.0, w_vdd=0.5, tabu=0.3, w_map=0.5, sub3=1, vdd_every=5, vdd_tau=0.5, vdd_cap=0.5, vdd_move=4.0, commit_m=0.9, abort_m=0.3, soft_t=45.0, soft_m=0.6, hail_t=50.0, fb_dsc=80.0, transit_r=30.0, replan=25, offnose_v=2.0, map_every=5, ne_rays=1500, gain_cells=300, fp_rgb=0.001, lr_vdd=200.0, vp_clear=3.0, occ_drop=2.5, h_unk=1.33, term_to=6.0, c_off=3.0, lookahead2=1, standoff=1, negev=1, fbfirst=1, lattice=1, direct=1, zmax=30.0, tan=1.0, vis_tol=0.8, occ_tol=1.0, map_dmax=29.0, slice_sched=((0, 1, 2), (3, 4, 5, 6)), pad_r=3.0, map_n=128, vis_seen=1, tilt_guard=36.0)

def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def angd(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)

def axes(rpy):
    r, p, y = (float(rpy[0]), float(rpy[1]), float(rpy[2]))
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y))
    fw = np.array([cy * cp, sy * cp, -sp])
    up = np.array([cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr])
    return (fw, up, np.cross(fw, up))

def cam_of(pos, rpy):
    fw, up, rt = axes(rpy)
    return (np.asarray(pos, np.float64) + fw * CAMF + up * CAMU, fw, up, rt)

def cell_of(x, y):
    return (int(np.clip(round(y + 47), 0, NG - 1)), int(np.clip(round(x + 47), 0, NG - 1)))

class PDM:

    def __init__(self, d):
        self.RB = np.array(d['RB'], np.float64)
        self.EB = np.array(d['EB'], np.float64)
        self.tab = {}
        self.marg = {}
        for det in ('rgb', 'dep'):
            T = d['tables'][det]
            for ci, cls in enumerate(('up', 'lp')):
                r = np.array(T[cls]['rate'], np.float64)
                n = np.array(T[cls]['n'], np.float64)
                mg = np.array([x[0] if x[0] is not None else 0.0 for x in T[cls]['marg_nv3']], np.float64)
                k = np.where(r >= 0, r * n, 0.0)
                sm = (k + 4.0 * mg[:, None]) / (n + 4.0)
                sm[:, 0] = np.minimum(sm[:, 0], 0.02)
                sm = np.maximum.accumulate(sm, axis=1)
                self.tab[det, ci] = sm
                self.marg[det, ci] = mg

    def rbin(self, r):
        return np.clip(np.searchsorted(self.RB, r, side='right') - 1, 0, len(self.RB) - 2)

    def p_true(self, det, ci, r, nv):
        if nv < 1 or r >= self.RB[-1]:
            return 0.0
        ri = int(self.rbin(r))
        ei = int(np.clip(np.searchsorted(self.EB, nv, side='right') - 1, 0, len(self.EB) - 2))
        return float(self.tab[det, ci][ri, ei])

    def p_plan(self, det, ci, r):
        r = np.asarray(r, np.float64)
        out = self.marg[det, ci][self.rbin(r)]
        return np.where(r < self.RB[-1], out, 0.0)

def slope3(top):
    t = np.pad(top, 1, mode='edge')
    a = (t[:-2, 2:] + t[1:-1, 2:] + t[2:, 2:] - (t[:-2, :-2] + t[1:-1, :-2] + t[2:, :-2])) / 6.0
    b = (t[2:, :-2] + t[2:, 1:-1] + t[2:, 2:] - (t[:-2, :-2] + t[:-2, 1:-1] + t[:-2, 2:])) / 6.0
    return np.degrees(np.arctan(np.hypot(a, b)))

def maxf3(a):
    p = np.pad(a, 1, mode='constant', constant_values=0.0)
    out = a.copy()
    for dj in (0, 1, 2):
        for di in (0, 1, 2):
            out = np.maximum(out, p[dj:dj + NG, di:di + NG])
    return out

def seen9(seen):
    p = np.pad(seen, 1, mode='edge').astype(np.int8)
    n = np.zeros((NG, NG), np.int8)
    for dj in (0, 1, 2):
        for di in (0, 1, 2):
            n += p[dj:dj + NG, di:di + NG]
    return n

def box_sum(b, r):
    out = np.zeros_like(b)
    R = int(math.ceil(r))
    n = b.shape[0]
    for dj in range(-R, R + 1):
        for di in range(-R, R + 1):
            if di * di + dj * dj > r * r:
                continue
            ys0, ys1 = (max(0, dj), min(n, n + dj))
            xs0, xs1 = (max(0, di), min(n, n + di))
            out[ys0 - dj:ys1 - dj, xs0 - di:xs1 - di] += b[ys0:ys1, xs0:xs1]
    return out

def nearest_idx(unseen):
    if _EDT is not None:
        _, (jj, ii) = _EDT(unseen, return_indices=True)
        return (jj, ii)
    n = unseen.shape[0]
    J, I = np.indices(unseen.shape)
    bj = np.where(~unseen, J, -10 ** 6)
    bi = np.where(~unseen, I, -10 ** 6)
    step = 64
    while step >= 1:
        for dj in (-step, 0, step):
            for di in (-step, 0, step):
                if dj == 0 and di == 0:
                    continue
                cj = np.full_like(bj, -10 ** 6)
                ci = np.full_like(bi, -10 ** 6)
                ys0, ys1 = (max(0, -dj), min(n, n - dj))
                xs0, xs1 = (max(0, -di), min(n, n - di))
                cj[ys0:ys1, xs0:xs1] = bj[ys0 + dj:ys1 + dj, xs0 + di:xs1 + di]
                ci[ys0:ys1, xs0:xs1] = bi[ys0 + dj:ys1 + dj, xs0 + di:xs1 + di]
                d0 = (bj - J) ** 2 + (bi - I) ** 2
                d1 = (cj - J) ** 2 + (ci - I) ** 2
                m = d1 < d0
                bj = np.where(m, cj, bj)
                bi = np.where(m, ci, bi)
        step //= 2
    return (np.clip(bj, 0, n - 1), np.clip(bi, 0, n - 1))

class Belief:

    def __init__(self, start_xy, clue_xy, par):
        self.par = par
        self.S = np.asarray(start_xy, np.float64)
        self.C = np.asarray(clue_xy, np.float64)
        self.dS = np.hypot(GX - self.S[0], GY - self.S[1])
        self.inr = self.dS <= 80.0
        self.o = np.argsort(self.dS.ravel(), kind='stable')
        dc = np.hypot(GX - self.C[0], GY - self.C[1])
        self.clue = 1.0 / (1.0 + np.exp(-(30.0 - dc) * 2.0))
        self.pi = np.array([1.0 - par['lp_share'], par['lp_share']])
        self.LL = np.zeros((2, NG, NG))
        self.prior = None
        self.reg = None
        self.b = None
        self.ms = None

    def set_val(self, val):
        V = val
        n30 = float((V * (B30 & self.inr)).sum())
        n45 = float((V * (B45 & self.inr)).sum())
        ms = self.ms
        if ms is None:
            f1 = (1.0 - min(n30, 3720.0) / 3721.0) ** 100
            f2 = (1.0 - min(n45, 8280.0) / 8281.0) ** 50
        else:
            f1 = (1.0 - min(n30 * ms[0], 3720.0) / 3721.0) ** 100
            f2 = (1.0 - min(n45 * ms[0], 8280.0) / 8281.0) ** 50
        dS_ = np.zeros_like(GX)
        dR = np.zeros_like(GX)
        if n30 > 0:
            dS_ = (1.0 - f1) * V * (B30 & self.inr) / n30
        if n45 > 0:
            dR = f1 * (1.0 - f2) * V * (B45 & self.inr) / n45
        lam = V * (100.0 * B30 / 3721.0 + 50.0 * B45 / 8281.0) * ~self.inr
        lo = lam.ravel()[self.o]
        fb = np.zeros(lam.size)
        if ms is None:
            fb[self.o] = lo * np.exp(-(np.cumsum(lo) - lo))
        else:
            fb[self.o] = lo * np.exp(-ms[1] * (np.cumsum(lo) - lo))
        dF = f1 * f2 * fb.reshape(lam.shape)
        comp = np.stack([dS_, dR, dF]) * self.clue[None]
        tot = float(comp.sum())
        self.Z = tot
        self.reg = comp.sum(axis=(1, 2)) / max(tot, 1e-12)
        self.prior = comp.sum(0) / max(tot, 1e-12)
        self.fbmap = comp[2] / max(tot, 1e-12)
        self._mk()

    def _mk(self):
        w = self.pi[:, None, None] * self.prior[None] * np.exp(self.LL - self.LL.max())
        s = float(w.sum())
        self.b = w / s if s > 0 else w

    def neg(self, ci, js, is_, pd, tau, cap):
        if len(js) == 0:
            return
        d = np.maximum(tau * np.log(np.clip(1.0 - pd, 1e-09, 1.0)), math.log(1.0 - cap))
        np.add.at(self.LL[ci], (js, is_), d)
        np.maximum(self.LL[ci], math.log(self.par['floor']), out=self.LL[ci])

    def pos(self, x, y, lr, sig=1.0):
        d2 = (GX - x) ** 2 + (GY - y) ** 2
        m = d2 <= (3.0 * sig) ** 2
        self.LL[:, m] += np.log1p(lr * np.exp(-d2[m] / (2 * sig * sig)))[None]
        self._mk()

    def pos_mass(self, x, y, r=2.0):
        m = np.hypot(GX - x, GY - y) <= r
        return (float(self.b[:, m].sum()), self.b[:, m].sum(axis=1))

def map_los(Hm, seen, hfill, cam, tgt, unk_vis, S=12):
    t = np.linspace(0.08, 0.92, S)
    P = cam[:, None, None, :] + (tgt[None, :, None, :] - cam[:, None, None, :]) * t[None, None, :, None]
    ix = np.rint(P[..., 0] + 47).astype(np.int32)
    iy = np.rint(P[..., 1] + 47).astype(np.int32)
    ins = (ix >= 0) & (ix < NG) & (iy >= 0) & (iy < NG)
    iyc = np.clip(iy, 0, NG - 1)
    ixc = np.clip(ix, 0, NG - 1)
    sn = seen[iyc, ixc] & ins
    h = np.where(sn, Hm[iyc, ixc], -1000000000.0)
    hk = np.any(h > P[..., 2] + 0.25, axis=-1)
    hu = np.where(ins & ~sn, hfill[iyc, ixc], -1000000000.0)
    uk = np.any(hu > P[..., 2] + 1.0, axis=-1)
    return np.where(hk, 0.0, np.where(uk, unk_vis, 1.0))

class Map1m:

    def __init__(self, par):
        self.par = par
        self.seen = np.zeros((NG, NG), bool)
        self.Hm = np.full((NG, NG), -1000000000.0)
        self.occ = np.zeros((NG, NG), np.int16)
        self.version = 0
        self.dirty = True
    _UV = {}

    @classmethod
    def uv(cls, n):
        if n not in cls._UV:
            px = (np.arange(0, 256, 256 // n) + 0.5) / 128.0 - 1.0
            cls._UV[n] = (np.broadcast_to(px[None, :], (n, n)).ravel().copy(), np.broadcast_to(-px[:, None], (n, n)).ravel().copy())
        return cls._UV[n]

    def add(self, z64, cam, fw, up, rt, hf, zmap=None):
        par = self.par
        tan = par['tan']
        zm = z64 if zmap is None else zmap
        Uq, Vq = self.uv(zm.shape[0])
        z = zm.ravel().astype(np.float64)
        hit = z <= par['map_dmax']
        if hit.any():
            zz = z[hit]
            u = Uq[hit] * tan * zz
            v = Vq[hit] * tan * zz
            P = cam[None, :] + zz[:, None] * fw[None, :] + u[:, None] * rt[None, :] + v[:, None] * up[None, :]
            ix = np.rint(P[:, 0] + 47).astype(np.int64)
            iy = np.rint(P[:, 1] + 47).astype(np.int64)
            ok = (ix >= 0) & (ix < NG) & (iy >= 0) & (iy < NG)
            if ok.any():
                ix = ix[ok]
                iy = iy[ok]
                pz = P[ok, 2]
                new = ~self.seen[iy, ix]
                np.maximum.at(self.Hm, (iy, ix), pz)
                if new.any():
                    self.seen[iy[new], ix[new]] = True
                    self.dirty = True
                self.version += 1
        us = ~self.seen.ravel()
        if hf is not None and us.any():
            idx = np.nonzero(us & (np.abs(GXf - cam[0]) <= 30.0) & (np.abs(GYf - cam[1]) <= 30.0))[0]
            if len(idx):
                Pt = np.stack([GXf[idx], GYf[idx], hf.ravel()[idx] + 0.3], 1)
                d = Pt - cam[None, :]
                x = d @ fw
                a = d @ rt
                b = d @ up
                m = (x > 0.5) & (x <= par['map_dmax']) & (np.abs(a) <= x * tan) & (np.abs(b) <= x * tan)
                if m.any():
                    xs = x[m]
                    uu = a[m] / (xs * tan)
                    vv = b[m] / (xs * tan)
                    col = np.clip(np.floor((uu + 1.0) * 32.0), 0, 63).astype(np.int64)
                    row = np.clip(np.floor((1.0 - vv) * 32.0), 0, 63).astype(np.int64)
                    zp = z64[row, col]
                    bad = zp < xs - par['occ_tol']
                    if bad.any():
                        c = idx[np.nonzero(m)[0][bad]]
                        bj, bi = np.divmod(c, NG)
                        self.occ[bj, bi] += 1
                        self.version += 1

def depth_vis(z64, cam, fw, up, rt, pts, tan, tol):
    d = pts - cam[None, :]
    x = d @ fw
    xs = np.maximum(x, 1e-06)
    uu = d @ rt / (xs * tan)
    vv = d @ up / (xs * tan)
    inimg = (x > 0.05) & (np.abs(uu) <= 1.0) & (np.abs(vv) <= 1.0)
    col = np.clip(np.floor((uu + 1.0) * 32.0), 0, 63).astype(np.int64)
    row = np.clip(np.floor((1.0 - vv) * 32.0), 0, 63).astype(np.int64)
    zp = z64[row, col]
    return inimg & ((zp >= x - tol) | (zp >= DMAX - 0.05))

class Brain:
    STAGES = ('takeoff', 'transit', 'search', 'confirm', 'term', 'hail')

    def __init__(self, par, pdm, vcal, fl, vdd_on):
        self.par = dict(par)
        self.pdm = pdm
        self.VBINS = np.array(vcal['bins'], np.float64)
        self.VPV = np.array(vcal['p_valid'], np.float64)
        self.fl = fl
        self.vdd_on = bool(vdd_on)
        self.a3lp = os.environ.get('KT_A3LP', '1') == '1'
        if not self.vdd_on:
            self.par['w_vdd'] = 0.0
        self.k = 0
        self.t = 0.0
        self.stage = 'takeoff'
        self.B = None
        self.pending = None
        self.frames = 0
        self.est = None
        self.commit = None
        self.a3t = None
        self.a3t_n = 0
        self.a3t_t1 = None
        self.mfar = None
        self.mfar_on = False
        self.log = {'events': [], 'plan_ms': [], 'slice_ms': [], 'frames': [], 'commit': None, 'det': None, 'stage_t': {}, 'n_plans': 0, 'n_cancel': 0, 'vdd_fires': 0, 'vdd_ev': 0}

    def _init(self, st):
        par = self.par
        pos = st[0:3].astype(np.float64)
        self.pad = pos.copy()
        clue = pos[0:2] + st[163:165].astype(np.float64)
        self.clue = clue
        self.dsc = float(np.hypot(*clue - pos[0:2]))
        self.B = Belief(pos[0:2], clue, par)
        if self.mfar is not None and self.dsc >= self.mfar['d']:
            self.B.ms = (self.mfar['s1'], self.mfar['s2'])
            self.mfar_on = True
        self.M = Map1m(par)
        self._hf = None
        self._hf_k = -1
        self.ground_last = float(pos[2]) - 1.0
        self.ground_now = None
        self._update_val(force=True)
        self.b0 = self.B.b.sum(0).copy()
        self.reg0 = self.B.reg.copy()
        self.frames = 0
        self.last_f = -999
        self.last_fpose = None
        self.pending = None
        self.vdd_hist = []
        self.vdd_hits = 0
        self.vdd_ev = 0
        self.vdd_lastpos = np.full((NG, NG, 3), 1000000000.0)
        self.goal = None
        self.gyaw = None
        self.plan_k = -999
        self.visited = []
        self.arrive_k = None
        self.track = None
        self.est = None
        self.commit = None
        self.nref = 0
        self.job = None
        self.job_k = None
        self.frame_req = 0.0

    def _update_val(self, force=False):
        M = self.M
        if not (M.dirty or force):
            return
        par = self.par
        q = par['q']
        H = np.where(M.seen, M.Hm, 0.0)
        s = slope3(H)
        pv = self.VPV[np.clip(np.searchsorted(self.VBINS, s, side='right') - 1, 0, len(self.VPV) - 1)]
        res = seen9(M.seen) >= 9
        v = np.where(res, pv, np.where(M.seen, 1.6 * q, q))
        vres = np.where(res, pv, 0.0)
        val = np.maximum(v, par['dil'] * maxf3(vres))
        if self.mfar_on and self.B.ms is None:
            self.B.ms = (self.mfar['s1'], self.mfar['s2'])
        self.B.set_val(val)
        M.dirty = False

    def _hfill(self):
        M = self.M
        if self._hf is not None and self._hf_k == M.version:
            return self._hf
        if M.seen.all():
            hf = M.Hm.copy()
        elif not M.seen.any():
            hf = np.full((NG, NG), self.ground_last)
        else:
            jj, ii = nearest_idx(~M.seen)
            hf = M.Hm[jj, ii]
            hf = hf - self.par['occ_drop'] * ((M.occ >= 1) & ~M.seen)
        self._hf = hf
        self._hf_k = M.version
        return hf

    def _gaim(self, x, y):
        g = self._gnd(x, y)
        if g is not None:
            return g
        hf = self._hfill() + self.par['occ_drop'] * ((self.M.occ >= 1) & ~self.M.seen)
        if -47.5 <= x <= 47.5 and -47.5 <= y <= 47.5:
            j, i = cell_of(x, y)
            return float(hf[j, i])
        return self.ground_last

    def _gest(self, x, y, hf=None):
        if -47.5 <= x <= 47.5 and -47.5 <= y <= 47.5:
            j, i = cell_of(x, y)
            return float((self._hfill() if hf is None else hf)[j, i])
        return self.ground_last

    def _gnd(self, x, y):
        if -47.5 <= x <= 47.5 and -47.5 <= y <= 47.5:
            j, i = cell_of(x, y)
            if self.M.seen[j, i]:
                return float(self.M.Hm[j, i])
        return None

    def _hest(self):
        return np.where(self.M.seen, self.M.Hm, self._hfill())

    def _vis_cells(self, cam, fw, up, rt, z64, dmax, nmax):
        par = self.par
        tan = par['tan']
        b = self.B.b.sum(0).ravel()
        he = self._hest().ravel()
        P = np.stack([GXf, GYf, he + 0.3], 1)
        d = P - cam[None, :]
        x = d @ fw
        m = (x > 1.0) & (np.abs(d @ rt) <= x * tan) & (np.abs(d @ up) <= x * tan) & (x <= dmax)
        idx = np.nonzero(m)[0]
        if len(idx) == 0:
            return idx
        idx = idx[b[idx] > 1e-06 * float(b.max())]
        if len(idx) > nmax:
            idx = idx[np.argsort(-b[idx])[:nmax]]
        if len(idx) == 0:
            return idx
        if par['vis_seen']:
            idx = idx[self.M.seen.ravel()[idx]]
            if len(idx) == 0:
                return idx
        tg = P[idx].copy()
        tg[:, 2] += 0.3
        return idx[depth_vis(z64, cam, fw, up, rt, tg, tan, par['vis_tol'])]

    def _neg_frame(self, cam, idx, det, layers, tau, cap):
        if len(idx) == 0 or not self.par['negev']:
            return
        js, is_ = np.divmod(idx, NG)
        he = self._hest()
        P = np.stack([XS[is_], XS[js], he[js, is_] + 0.3], 1)
        r = np.linalg.norm(P - cam, axis=1)
        for ci in layers:
            self.B.neg(ci, js, is_, self.pdm.p_plan(det, ci, r), tau, cap)
        self.B._mk()

    def _frame_gain(self, cam, fw, up, rt, z64):
        idx = self._vis_cells(cam, fw, up, rt, z64, 30.0, 500)
        if len(idx) == 0:
            return (0.0, idx)
        js, is_ = np.divmod(idx, NG)
        he = self._hest()
        P = np.stack([XS[is_], XS[js], he[js, is_] + 0.3], 1)
        r = np.linalg.norm(P - cam, axis=1)
        g = 0.0
        for ci in (0, 1):
            g += float((self.B.b[ci, js, is_] * self.pdm.p_plan('rgb', ci, r)).sum())
        return (g, idx)

    def _gmin(self, t):
        par = self.par
        fu = self.frames / float(40 - par['reserve'])
        tu = min(1.0, t / 45.0)
        return par['g0'] * math.exp(par['gk'] * (fu - tu))

    def _clusters(self, b=None):
        B = self.B.b if b is None else b
        bt = B.sum(0)
        G = box_sum(bt, 4.0)
        out = []
        Gc = G.copy()
        for _ in range(self.par['K']):
            j, i = np.unravel_index(int(np.argmax(Gc)), Gc.shape)
            if Gc[j, i] <= 0.0001:
                break
            m = np.hypot(GX - XS[i], GY - XS[j]) <= 6.0
            lpf = float(B[1][m].sum() / max(bt[m].sum(), 1e-12))
            out.append((XS[i], XS[j], float(G[j, i]), lpf))
            Gc[np.hypot(GX - XS[i], GY - XS[j]) < 8.0] = 0.0
        return out

    def _gest_v(self, x, y, hf):
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        ins = (x >= -47.5) & (x <= 47.5) & (y >= -47.5) & (y <= 47.5)
        j = np.clip(np.rint(y + 47), 0, NG - 1).astype(np.int64)
        i = np.clip(np.rint(x + 47), 0, NG - 1).astype(np.int64)
        return np.where(ins, hf[j, i], self.ground_last)

    def _plan_gen(self, pos, yaw_now):
        par = self.par
        pos = np.asarray(pos, np.float64).copy()
        c0 = time.perf_counter()
        Bb = self.B.b
        bt = Bb.sum(0)
        flat = bt.ravel()
        tidx = np.argsort(-flat)[:par['gain_cells']]
        tidx = tidx[flat[tidx] > 1e-06 * flat[tidx[0]]]
        wt = np.ones(len(tidx))
        if par['sub3']:
            lat = np.zeros((NG, NG), bool)
            lat[1::3, 1::3] = True
            lm = lat.ravel() & (flat > 1e-06 * flat.max())
            lm[tidx] = False
            extra = np.nonzero(lm)[0]
            tidx = np.concatenate([tidx, extra])
            wt = np.concatenate([wt, np.full(len(extra), 9.0)])
        tj, ti = np.divmod(tidx, NG)
        hf = self._hfill()
        seen = self.M.seen.copy()
        HsP = np.full((NG + 2, NG + 2), -1000000000.0, np.float32)
        HuP = np.full((NG + 2, NG + 2), -1000000000.0, np.float32)
        HsP[1:-1, 1:-1] = np.where(seen, self.M.Hm, -1000000000.0)
        HuP[1:-1, 1:-1] = np.where(seen, -1000000000.0, hf)
        tgt = np.stack([XS[ti], XS[tj], hf[tj, ti] + 0.6], -1)
        unk = ~seen[tj, ti]
        bw = np.stack([Bb[0, tj, ti], Bb[1, tj, ti]]) * wt[None]
        cl = self._clusters(Bb)
        yield ((time.perf_counter() - c0) * 1000.0)
        c0 = time.perf_counter()
        PX, PY, PZ, CY, CK, CP = ([], [], [], [], [], [])
        npos = 0
        az = 2 * np.pi * np.arange(par['n_az']) / par['n_az']
        radii = np.asarray(par['radii'] if par['standoff'] else (0.0,), np.float64)
        for cx, cy, m, lpf in cl:
            gz = float(self._gest_v(cx, cy, hf))
            hs = (par['H_lp'], par['H']) if lpf > 0.5 else (par['H'], par['H2'])
            for r in radii:
                azs = az if r > 0 else az[:1]
                vx = cx + r * np.cos(azs)
                vy = cy + r * np.sin(azs)
                gv = self._gest_v(vx, vy, hf)
                yawr = np.arctan2(cy - vy, cx - vx)
                for q in range(len(azs)):
                    for h in hs:
                        if r > 0 and r < h:
                            continue
                        PX.append(vx[q])
                        PY.append(vy[q])
                        PZ.append(max(gz + h, float(gv[q]) + par['vp_clear']))
                        CY.append(yawr[q])
                        CK.append(0)
                        CP.append(npos)
                        npos += 1
        nring = npos
        if par['lattice']:
            o = np.argsort(-flat)
            cs = np.cumsum(flat[o]) / max(flat.sum(), 1e-12)
            sel = o[:int(np.searchsorted(cs, 0.95)) + 1]
            sj, si = np.divmod(sel, NG)
            x0, x1 = (XS[si].min() - 12, XS[si].max() + 12)
            y0, y1 = (XS[sj].min() - 12, XS[sj].max() + 12)
            sp = par['lat']
            lx = np.arange(x0, x1 + 0.1, sp)
            ly = np.arange(y0, y1 + 0.1, sp)
            LX = np.repeat(lx, len(ly))
            LY = np.tile(ly, len(lx))
            LZ = self._gest_v(LX, LY, hf) + par['H']
            ny = par['n_yaw']
            PX += LX.tolist()
            PY += LY.tolist()
            PZ += LZ.tolist()
            CY += (2 * np.pi * np.arange(ny) / ny).tolist() * len(LX)
            CK += [1] * (len(LX) * ny)
            CP += np.repeat(np.arange(npos, npos + len(LX)), ny).tolist()
            npos += len(LX)
        gd = self.ground_now if self.ground_now is not None else float(self._gest_v(pos[0], pos[1], hf))
        zh = float(np.clip(pos[2], gd + par['H'], gd + par['H2']))
        PX.append(pos[0])
        PY.append(pos[1])
        PZ.append(zh)
        ny = par['n_yaw']
        CY += (2 * np.pi * np.arange(ny) / ny).tolist()
        CK += [2] * ny
        CP += [npos] * ny
        npos += 1
        Pp = np.stack([np.asarray(PX), np.asarray(PY), np.asarray(PZ)], 1)
        pc = np.asarray(CP, np.int64)
        yaw_v = np.asarray(CY, np.float64)
        kind = np.asarray(CK, np.int64)
        nC = len(pc)
        half = math.radians(90.0 / 2.0 - 1.0)
        ch_ = math.cos(half)
        th_ = math.tan(half)
        frames_ok = 1.0 if self.frames < 40 - par['reserve'] else 0.0
        uw = np.where(unk, 0.7, 1.0)
        kk = np.arange(46, dtype=np.float64) + 0.5
        pru = self.pdm.p_plan('rgb', 0, kk)
        prl = self.pdm.p_plan('rgb', 1, kk)
        pdu = self.pdm.p_plan('dep', 0, kk)
        W = frames_ok * (pru[:, None] * (bw[0] * uw)[None] + prl[:, None] * (bw[1] * uw)[None])
        if par['w_vdd'] > 0:
            W = W + par['w_vdd'] * pdu[:, None] * (bw[0] * uw)[None]
            if self.a3lp:
                W = W + par['w_vdd'] * self.pdm.p_plan('dep', 1, kk)[:, None] * (bw[1] * uw)[None]
        W = W + par['w_map'] * (np.arange(46) <= 28)[:, None] * ((bw[0] + bw[1]) * unk)[None]
        W = W.astype(np.float32)
        T = len(tidx)
        tg32 = tgt.astype(np.float32)
        Pp32 = Pp.astype(np.float32)
        Np_ = len(Pp)
        dx = np.empty((Np_, T), np.float32)
        dy = np.empty((Np_, T), np.float32)
        base = np.empty((Np_, T), np.float32)
        horc = np.empty((Np_, T), np.float32)
        tcol = np.arange(T, dtype=np.int32)[None, :]
        cyv = np.cos(yaw_v).astype(np.float32)
        syv = np.sin(yaw_v).astype(np.float32)
        G0 = np.zeros(nC)
        yield ((time.perf_counter() - c0) * 1000.0)
        hp = (Np_ + 1) // 2
        for part in range(2):
            c0 = time.perf_counter()
            a0, a1 = (part * hp, min(Np_, (part + 1) * hp))
            if a1 > a0:
                ddx = tg32[None, :, 0] - Pp32[a0:a1, None, 0]
                ddy = tg32[None, :, 1] - Pp32[a0:a1, None, 1]
                ddz = tg32[None, :, 2] - Pp32[a0:a1, None, 2]
                hor2 = ddx * ddx + ddy * ddy
                hor = np.sqrt(hor2)
                r3 = np.sqrt(hor2 + ddz * ddz)
                ri = np.minimum(r3, 45.0).astype(np.int32)
                bb = W.ravel()[ri * T + tcol]
                bb *= (np.abs(ddz) <= np.maximum(hor, 0.001) * th_) & (r3 >= 2.0)
                dx[a0:a1] = ddx
                dy[a0:a1] = ddy
                base[a0:a1] = bb
                horc[a0:a1] = hor * ch_
            yield ((time.perf_counter() - c0) * 1000.0)
        c0 = time.perf_counter()
        if nring:
            al = dx[:nring] * cyv[:nring, None] + dy[:nring] * syv[:nring, None]
            G0[:nring] = (((al >= horc[:nring]) & (al <= 29.0)) * base[:nring]).sum(1, dtype=np.float64)
        ny = par['n_yaw']
        cy8 = np.cos(2 * np.pi * np.arange(ny) / ny).astype(np.float32)
        sy8 = np.sin(2 * np.pi * np.arange(ny) / ny).astype(np.float32)
        P1 = nring
        P2 = Np_
        for a in range(ny):
            al = dx[P1:P2] * cy8[a] + dy[P1:P2] * sy8[a]
            G0[nring + a:nC:ny] = (((al >= horc[P1:P2]) & (al <= 29.0)) * base[P1:P2]).sum(1, dtype=np.float64)
        del al
        yield ((time.perf_counter() - c0) * 1000.0)
        cam = Pp[pc]
        tb = np.zeros(nC, bool)
        if self.visited:
            Vp = np.array(self.visited)
            dd = np.linalg.norm(cam[:, None, 0:2] - Vp[None, :, 0:2], axis=-1)
            dy_ = np.abs((yaw_v[:, None] - Vp[None, :, 2] + np.pi) % (2 * np.pi) - np.pi)
            tb = np.any((dd < 4.0) & (dy_ < math.radians(30.0)), axis=1)
            G0 = np.where(tb, G0 * par['tabu'], G0)
        d3 = np.linalg.norm(cam - pos[None], axis=1)
        trav = d3 / par['V']
        yawt = np.degrees(np.abs((yaw_v - yaw_now + np.pi) % (2 * np.pi) - np.pi)) / 40.0
        cost = trav + 1.0 + np.maximum(0.0, yawt - 0.5 * trav)
        sc0 = G0 / (cost + par['c_off'])
        top = np.argsort(-sc0)[:48]
        q = pc[top]
        along = dx[q] * cyv[top, None] + dy[q] * syv[top, None]
        U0t = (((along >= horc[q]) & (along <= 29.0)) * base[q]).astype(np.float64)
        del dx, dy, base, horc
        camt = cam[top]
        nt = len(top)
        h2 = (nt + 1) // 2
        los = np.empty((nt, T))
        los[:h2] = self._los_fast(HsP, HuP, camt[:h2], tgt, par['unk_vis'])
        yield ((time.perf_counter() - c0) * 1000.0)
        c0 = time.perf_counter()
        if nt > h2:
            los[h2:] = self._los_fast(HsP, HuP, camt[h2:], tgt, par['unk_vis'])
        U = U0t * los
        G = U.sum(1)
        G = np.where(tb[top], G * par['tabu'], G)
        sc = G / (cost[top] + par['c_off'])
        o = np.argsort(-sc)[:10]
        best = top[o[0]]
        if par['lookahead2'] and len(o) > 1:
            Ut = U[o]
            pdc = np.clip(Ut / np.maximum(bw.sum(0)[None], 1e-12), 0.0, 1.0)
            bestp = -1.0
            for a in range(len(o)):
                va = top[o[a]]
                g2 = ((1.0 - pdc[a])[None] * Ut).sum(1)
                c2 = np.linalg.norm(cam[top[o]] - cam[va][None], axis=1) / par['V'] + 1.0
                ps = (G[o[a]] + g2) / (cost[va] + c2 + 2.0 * par['c_off'])
                ps[a] = G[o[a]] / (cost[va] + par['c_off'])
                k2 = int(np.argmax(ps))
                if ps[k2] > bestp:
                    bestp = ps[k2]
                    best = va
        cb = cam[best]
        self._plan_out = dict(goal=np.array(cb[0:3]), yaw=float(yaw_v[best]), kind=int(kind[best]), gain=float(G0[best]))
        yield ((time.perf_counter() - c0) * 1000.0)

    @staticmethod
    def _los_fast(HsP, HuP, cam, tgt, unk_vis, S=12):
        t = np.linspace(0.08, 0.92, S).astype(np.float32)
        c = cam.astype(np.float32)
        g = tgt.astype(np.float32)
        Px = c[:, None, None, 0] + (g[None, :, None, 0] - c[:, None, None, 0]) * t
        Py = c[:, None, None, 1] + (g[None, :, None, 1] - c[:, None, None, 1]) * t
        Pz = c[:, None, None, 2] + (g[None, :, None, 2] - c[:, None, None, 2]) * t
        ix = np.clip(np.rint(Px + 48.0), 0, NG + 1).astype(np.int32)
        iy = np.clip(np.rint(Py + 48.0), 0, NG + 1).astype(np.int32)
        fi = iy * (NG + 2) + ix
        hk = (HsP.ravel()[fi] - Pz > 0.25).any(-1)
        uk = (HuP.ravel()[fi] - Pz > 1.0).any(-1)
        return np.where(hk, 0.0, np.where(uk, unk_vis, 1.0))

    def _plan_ref(self, pos, yaw_now):
        par = self.par
        pos = np.asarray(pos, np.float64).copy()
        c0 = time.perf_counter()
        self._update_val()
        Bb = self.B.b
        bt = Bb.sum(0)
        flat = bt.ravel()
        tidx = np.argsort(-flat)[:par['gain_cells']]
        tidx = tidx[flat[tidx] > 1e-06 * flat[tidx[0]]]
        wt = np.ones(len(tidx))
        if par['sub3']:
            lat = np.zeros((NG, NG), bool)
            lat[1::3, 1::3] = True
            lm = lat.ravel() & (flat > 1e-06 * flat.max())
            lm[tidx] = False
            extra = np.nonzero(lm)[0]
            tidx = np.concatenate([tidx, extra])
            wt = np.concatenate([wt, np.full(len(extra), 9.0)])
        tj, ti = np.divmod(tidx, NG)
        hf = self._hfill()
        Hm = self.M.Hm.copy()
        seen = self.M.seen.copy()
        tgt = np.stack([XS[ti], XS[tj], hf[tj, ti] + 0.6], -1)
        unk = ~seen[tj, ti]
        bw = np.stack([Bb[0, tj, ti], Bb[1, tj, ti]]) * wt[None]
        cl = self._clusters(Bb)
        yield ((time.perf_counter() - c0) * 1000.0)
        c0 = time.perf_counter()
        cands = []
        for cx, cy, m, lpf in cl:
            gz = self._gest(cx, cy, hf)
            hs = (par['H_lp'], par['H']) if lpf > 0.5 else (par['H'], par['H2'])
            for r in par['radii'] if par['standoff'] else (0.0,):
                for a in range(par['n_az'] if r > 0 else 1):
                    az = 2 * math.pi * a / par['n_az']
                    vx, vy = (cx + r * math.cos(az), cy + r * math.sin(az))
                    for h in hs:
                        if r > 0 and r < h:
                            continue
                        cands.append((vx, vy, max(gz + h, self._gest(vx, vy, hf) + par['vp_clear']), math.atan2(cy - vy, cx - vx), 0))
        if par['lattice']:
            o = np.argsort(-flat)
            cs = np.cumsum(flat[o]) / max(flat.sum(), 1e-12)
            sel = o[:int(np.searchsorted(cs, 0.95)) + 1]
            sj, si = np.divmod(sel, NG)
            x0, x1 = (XS[si].min() - 12, XS[si].max() + 12)
            y0, y1 = (XS[sj].min() - 12, XS[sj].max() + 12)
            sp = par['lat']
            for vx in np.arange(x0, x1 + 0.1, sp):
                for vy in np.arange(y0, y1 + 0.1, sp):
                    z = self._gest(vx, vy, hf) + par['H']
                    for a in range(par['n_yaw']):
                        cands.append((vx, vy, z, 2 * math.pi * a / par['n_yaw'], 1))
        gd = self.ground_now if self.ground_now is not None else self._gest(pos[0], pos[1], hf)
        zh = float(np.clip(pos[2], gd + par['H'], gd + par['H2']))
        for a in range(par['n_yaw']):
            cands.append((pos[0], pos[1], zh, 2 * math.pi * a / par['n_yaw'], 2))
        C = np.array(cands, np.float64)
        cam = C[:, 0:3]
        yaw_v = C[:, 3]
        nC = len(C)
        half = math.radians(90.0 / 2.0 - 1.0)
        frames_ok = 1.0 if self.frames < 40 - par['reserve'] else 0.0
        uw = np.where(unk, 0.7, 1.0)[None]
        G0 = np.zeros(nC)
        U0rows = {}
        yield ((time.perf_counter() - c0) * 1000.0)
        nh = (nC + 1) // 2
        for part in range(2):
            c0 = time.perf_counter()
            s0, s1 = (part * nh, min(nC, (part + 1) * nh))
            if s1 > s0:
                dvec = tgt[None, :, :] - cam[s0:s1, None, :]
                hor = np.hypot(dvec[..., 0], dvec[..., 1])
                rng3 = np.sqrt(hor * hor + dvec[..., 2] * dvec[..., 2])
                dyaw = (np.arctan2(dvec[..., 1], dvec[..., 0]) - yaw_v[s0:s1, None] + np.pi) % (2 * np.pi) - np.pi
                el = np.arctan2(dvec[..., 2], np.maximum(hor, 0.001))
                infov = (np.abs(dyaw) <= half) & (np.abs(el) <= half) & (hor * np.cos(dyaw) <= 29.0) & (rng3 >= 2.0)
                Urgb = (self.pdm.p_plan('rgb', 0, rng3) * bw[0][None] + self.pdm.p_plan('rgb', 1, rng3) * bw[1][None]) * infov * uw
                U0 = frames_ok * Urgb
                if par['w_vdd'] > 0:
                    U0 = U0 + par['w_vdd'] * (self.pdm.p_plan('dep', 0, rng3) * bw[0][None] * infov * uw)
                    if self.a3lp:
                        U0 = U0 + par['w_vdd'] * (self.pdm.p_plan('dep', 1, rng3) * bw[1][None] * infov * uw)
                U0 = U0 + par['w_map'] * ((bw[0] + bw[1])[None] * (infov & (rng3 <= 29.0)) * unk[None])
                G0[s0:s1] = U0.sum(1)
                U0rows[part] = (s0, U0)
            yield ((time.perf_counter() - c0) * 1000.0)
        c0 = time.perf_counter()
        tb = np.zeros(nC, bool)
        if self.visited:
            Vp = np.array(self.visited)
            dd = np.linalg.norm(cam[:, None, 0:2] - Vp[None, :, 0:2], axis=-1)
            dy = np.abs((yaw_v[:, None] - Vp[None, :, 2] + np.pi) % (2 * np.pi) - np.pi)
            tb = np.any((dd < 4.0) & (dy < math.radians(30.0)), axis=1)
            G0 = np.where(tb, G0 * par['tabu'], G0)
        d3 = np.linalg.norm(cam - pos[None], axis=1)
        trav = d3 / par['V']
        yawt = np.degrees(np.abs((yaw_v - yaw_now + np.pi) % (2 * np.pi) - np.pi)) / 40.0
        cost = trav + 1.0 + np.maximum(0.0, yawt - 0.5 * trav)
        sc0 = G0 / (cost + par['c_off'])
        top = np.argsort(-sc0)[:48]
        U0t = np.empty((len(top), len(tidx)))
        for part, (s0, U0) in U0rows.items():
            m = (top >= s0) & (top < s0 + len(U0))
            if m.any():
                U0t[m] = U0[top[m] - s0]
        del U0rows
        los = map_los(Hm, seen, hf, cam[top], tgt, par['unk_vis'])
        yield ((time.perf_counter() - c0) * 1000.0)
        c0 = time.perf_counter()
        U = U0t * los
        G = U.sum(1)
        G = np.where(tb[top], G * par['tabu'], G)
        sc = G / (cost[top] + par['c_off'])
        o = np.argsort(-sc)[:10]
        best = top[o[0]]
        if par['lookahead2'] and len(o) > 1:
            Ut = U[o]
            pdc = np.clip(Ut / np.maximum(bw.sum(0)[None], 1e-12), 0.0, 1.0)
            bestp = -1.0
            for a in range(len(o)):
                va = top[o[a]]
                g2 = ((1.0 - pdc[a])[None] * Ut).sum(1)
                c2 = np.linalg.norm(cam[top[o]] - cam[va][None], axis=1) / par['V'] + 1.0
                ps = (G[o[a]] + g2) / (cost[va] + c2 + 2.0 * par['c_off'])
                ps[a] = G[o[a]] / (cost[va] + par['c_off'])
                k2 = int(np.argmax(ps))
                if ps[k2] > bestp:
                    bestp = ps[k2]
                    best = va
        cb = C[best]
        self._plan_out = dict(goal=np.array(cb[0:3]), yaw=float(cb[3]), kind=int(cb[4]), gain=float(G0[best]))
        yield ((time.perf_counter() - c0) * 1000.0)

    def _job_step(self):
        grp = self.par['slice_sched'][min(self._job_g, len(self.par['slice_sched']) - 1)]
        self._job_g += 1
        ms = 0.0
        for _ in grp:
            try:
                ms += next(self.job)
            except StopIteration:
                break
        self.log['slice_ms'].append((self._job_g - 1, round(ms, 2)))
        self._job_ms += ms
        if self._job_g >= len(self.par['slice_sched']):
            for _ in self.job:
                pass
            self.job = None
            self.log['plan_ms'].append(round(self._job_ms, 2))
            return self._plan_out
        return None

    def _transit_goal(self, pos):
        src = self.B.fbmap if self.par['fbfirst'] and self.dsc >= self.par['fb_dsc'] and (float(self.B.reg[2]) > 0.3) else self.B.b.sum(0)
        f = src.ravel()
        o = np.argsort(-f)
        cs = np.cumsum(f[o]) / max(f.sum(), 1e-12)
        sel = o[:int(np.searchsorted(cs, 0.5)) + 1]
        js, is_ = np.divmod(sel, NG)
        d = np.hypot(XS[is_] - pos[0], XS[js] - pos[1])
        k = int(np.argmin(d))
        return (np.array([XS[is_[k]], XS[js[k]]]), float(d[k]))

    def _set_stage(self, s):
        if s != self.stage:
            self.log['events'].append((round(self.t, 2), 'stage', s))
            self.log['stage_t'].setdefault(s, round(self.t, 2))
        self.stage = s

    def _ev(self, *a):
        self.log['events'].append((round(self.t, 2),) + tuple(a))

    def _search_logic(self, pos, yaw_now, t):
        par = self.par
        if t >= par['soft_t']:
            cl = self._clusters()
            if cl:
                m2 = [(self.B.pos_mass(c[0], c[1], 2.0)[0], c) for c in cl]
                mbest, cb = max(m2, key=lambda z: z[0])
                if mbest >= par['soft_m'] or t >= par['hail_t']:
                    if t >= par['hail_t']:
                        best = None
                        bs = -1
                        for mm, c in m2:
                            dd = math.hypot(c[0] - pos[0], c[1] - pos[1])
                            reach = 58.0 - t - (dd / 2.7 + 4.5)
                            s = mm * (1.0 if reach > 0 else 0.2)
                            if s > bs:
                                bs = s
                                best = (mm, c)
                        mbest, cb = best
                    self._commit_blind(cb, mbest, 'soft' if t < par['hail_t'] else 'hail')
                    return
        if self.stage == 'transit':
            tg, dd = self._transit_goal(pos)
            if dd <= par['transit_r']:
                self._set_stage('search')
                self.plan_k = -999
            else:
                self.goal = np.array([tg[0], tg[1]])
                self.gyaw = math.atan2(tg[1] - pos[1], tg[0] - pos[0])
                return
        if self.arrive_k is not None and self.goal is not None and (len(self.goal) > 2) and ((self.k - self.arrive_k) * DT >= 1.0) and (angd(self.gyaw, yaw_now) < math.radians(8.0)):
            self.visited.append((self.goal[0], self.goal[1], self.gyaw))
            self.arrive_k = None
            self.plan_k = -999
        due = self.k - self.plan_k >= par['replan'] or self.goal is None or len(self.goal) < 3
        if self.job is not None and self.plan_k == -999 and (self.job_k is not None) and (self.job_k != self.k):
            self.job = None
            self.log['n_cancel'] += 1
        if self.job is None and due:
            self.job = self._plan_gen(pos, yaw_now)
            self.job_k = self.k
            self._job_ms = 0.0
            self._job_g = 0
            self.plan_k = self.k
            self.log['n_plans'] += 1
        if self.job is not None:
            pl = self._job_step()
            if pl is not None:
                if self.goal is None or len(self.goal) < 3 or float(np.linalg.norm(pl['goal'][0:2] - self.goal[0:2])) > 1.0 or (angd(pl['yaw'], self.gyaw if self.gyaw is not None else 0.0) > 0.2):
                    self.arrive_k = None
                self.goal = pl['goal']
                self.gyaw = pl['yaw']

    def _on_hit(self, hit, t):
        pm, _ = self.B.pos_mass(hit['x'], hit['y'], 2.0)
        self._ev('hit', round(hit['x'], 1), round(hit['y'], 1), round(pm, 3), round(hit['r'], 1))
        if self.log['det'] is None:
            self.log['det'] = [round(t, 2), round(hit['r'], 1)]
        consistent = self.track is not None and self.track.get('src') == 'rgb' and (math.hypot(self.track['x'] - hit['x'], self.track['y'] - hit['y']) <= 1.5)
        if pm >= self.par['commit_m'] or consistent:
            self._commit_xy(hit['x'], hit['y'], hit['top'], pm, 'det' if not consistent else 'confirm')
        else:
            self._commit_xy(hit['x'], hit['y'], hit['top'], pm, 'hit', verified=False)

    def _new_term(self):
        fl = self.fl
        fl.gi = 0
        fl.n_frames = 0
        fl.last_frame_k = -10 ** 6
        fl.lmap = None
        fl.hoff = np.zeros(3)
        fl.hoff_k = -10 ** 6
        fl.fb_key = None
        fl.fb_i = 0
        fl.settled = 0

    def _commit_xy(self, x, y, top, post, kind, up_only=False, verified=True):
        if top is None:
            top = self._gaim(x, y) + self.par['h_unk']
        self.est = dict(x=x, y=y, z=top + 3.0, n=1 if kind != 'vdd' else 0, sx=x, sy=y, stop=top, verified=bool(verified))
        self._ev('commit', kind, round(x, 2), round(y, 2), round(top, 2), round(post, 3))
        if self.commit is None and self.log['commit'] is None:
            self.log['commit'] = [round(self.t, 2), kind, round(post, 3)]
        self.commit = kind
        self.nref = 0
        self.term_k = self.k
        self.job = None
        self._new_term()
        self._set_stage('term')

    def _commit_blind(self, c, mass, kind):
        cx, cy = (c[0], c[1])
        gz = self._gaim(cx, cy)
        _, mc = self.B.pos_mass(cx, cy, 2.0)
        plp = float(mc[1] / max(mc.sum(), 1e-12))
        hz = 0.81 if plp > 0.8 else 1.84 if plp < 0.2 else self.par['h_unk']
        self.est = dict(x=cx, y=cy, z=gz + hz + 3.0, n=0, sx=cx, sy=cy, stop=gz + hz, verified=True)
        self._ev('commit', kind, round(cx, 2), round(cy, 2), round(gz + hz, 2), round(mass, 3))
        if self.log['commit'] is None:
            self.log['commit'] = [round(self.t, 2), kind, round(mass, 3)]
        self.commit = kind
        self.nref = 0
        self.term_k = self.k
        self.job = None
        self._new_term()
        self._set_stage('hail')

    def _abort(self, why):
        self._ev(why)
        self._set_stage('search')
        self.commit = None
        self.est = None
        self.plan_k = -999
        self.track = None
        self._new_term()

    def frame_result(self, hit, st, z64, zmap_fn=None):
        pend = self.pending
        self.pending = None
        if pend is None:
            return
        pos = st[0:3].astype(np.float64)
        rpy = st[3:6].astype(np.float64)
        cam, fw, up, rt = cam_of(pos, rpy)
        par = self.par
        t = self.t
        purpose = pend['purpose']
        if par['vis_seen']:
            self.M.add(z64, cam, fw, up, rt, self._hf, zmap=zmap_fn() if zmap_fn is not None and par['map_n'] != 64 else None)
            self._mapped_k = self.k + 1
        nm = 500 if purpose in ('search', 'hail') else par['ne_rays']
        vis_idx = self._vis_cells(cam, fw, up, rt, z64, 30.0, nm)
        self._neg_frame(cam, vis_idx, 'rgb', (0, 1), par['tau'], par['cap'])
        if hit is not None:
            pd = float(self.pdm.p_plan('rgb', 1 if hit['lp'] else 0, np.array([hit['r']]))[0])
            pfa = par['fp_rgb'] * 12.6 / 400.0
            self.B.pos(hit['x'], hit['y'], max(pd, 0.3) / pfa)
        self.log['frames'].append((round(t, 2), purpose, None if hit is None else [round(hit['x'], 2), round(hit['y'], 2), round(hit['r'], 1)]))
        if purpose == 'search':
            self.plan_k = -999
            if hit is not None and self.stage in ('transit', 'search'):
                self._on_hit(hit, t)
            return
        if purpose == 'hail':
            if hit is not None and self.stage == 'hail':
                self._commit_xy(hit['x'], hit['y'], hit['top'], 0.99, 'retarget')
            return
        if purpose == 'refine':
            e = self.est
            if e is None or self.stage not in ('term', 'hail'):
                if hit is not None and self.stage in ('transit', 'search'):
                    self._on_hit(hit, t)
                return
            self.nref += 1
            if hit is not None and math.hypot(hit['x'] - e['x'], hit['y'] - e['y']) <= 3.0:
                if e['n'] == 0:
                    e['sx'], e['sy'], e['stop'], e['n'] = (hit['x'], hit['y'], hit['top'], 1)
                else:
                    e['sx'] += hit['x']
                    e['sy'] += hit['y']
                    e['stop'] += hit['top']
                    e['n'] += 1
                n = e['n']
                e['x'] = e['sx'] / n
                e['y'] = e['sy'] / n
                e['z'] = e['stop'] / n + 3.0
                e['verified'] = True
                e['rgb_ok'] = True
                if self.log['det'] is None:
                    self.log['det'] = [round(t, 2), round(hit['r'], 1)]
                self._ev('refine_hit', round(e['x'], 2), round(e['y'], 2), round(e['stop'] / n, 2), round(hit['r'], 1))
            elif hit is not None:
                self._ev('refine_other')
                self._commit_xy(hit['x'], hit['y'], hit['top'], 0.99, 'retarget')
            else:
                pm, _ = self.B.pos_mass(e['x'], e['y'], 2.0)
                self._ev('refine_miss', round(pm, 3))
                if pm < par['abort_m'] and t < 50.0:
                    self._abort('abort')

    def _vdd_update(self, fires, cam, fw, up, rt, z64):
        par = self.par
        if fires is None:
            fires = []
        elif isinstance(fires, tuple):
            fires = [fires]
        self.vdd_hist.append(list(fires))
        self.vdd_hist = self.vdd_hist[-5:]
        self.vdd_ev += 1
        self.log['vdd_ev'] += 1
        self.log['vdd_fires'] += len(fires)
        idx = self._vis_cells(cam, fw, up, rt, z64, 29.0, par['ne_rays'])
        if len(idx):
            js, is_ = np.divmod(idx, NG)
            mv = np.linalg.norm(self.vdd_lastpos[js, is_] - cam, axis=1) > par['vdd_move']
            idx = idx[mv]
            if len(idx):
                self._neg_frame(cam, idx, 'dep', (0, 1) if self.a3lp else (0,), par['vdd_tau'], par['vdd_cap'])
                js, is_ = np.divmod(idx, NG)
                self.vdd_lastpos[js, is_] = cam
        best = None
        for f in fires:
            sup = [f]
            for prev in self.vdd_hist[:-1]:
                near = [g for g in prev if math.hypot(g[0] - f[0], g[1] - f[1]) <= 2.0]
                if near:
                    sup.append(min(near, key=lambda g: math.hypot(g[0] - f[0], g[1] - f[1])))
            if len(sup) >= 2 and (best is None or len(sup) > len(best)):
                best = sup
        if best is not None:
            self.vdd_hits += 1
            x = float(np.mean([g[0] for g in best]))
            y = float(np.mean([g[1] for g in best]))
            top = float(np.median([g[2] for g in best]))
            return ((x, y, top), len(best))
        return None

    def tick(self, st, z64, vdd_fn=None, zmap_fn=None):
        par = self.par
        self.k += 1
        k = self.k
        self.t = t = k * DT
        st = np.asarray(st, np.float64).reshape(-1)
        if self.B is None:
            self._init(st)
        pos = st[0:3]
        vel = st[6:9]
        rpy = st[3:6]
        agl = float(st[162]) * 20.0
        yaw_now = float(rpy[2])
        ground = pos[2] - agl if agl < 19.9 else None
        self.ground_now = ground
        if ground is not None:
            self.ground_last = ground
        tilt = math.degrees(max(abs(rpy[0]), abs(rpy[1])))
        cam, fw, up, rt = cam_of(pos, rpy)
        if k % par['map_every'] == 0 or k == 1:
            self.M.add(z64, cam, fw, up, rt, self._hf, zmap=zmap_fn() if zmap_fn is not None and par['map_n'] != 64 else None)
            self._mapped_k = k
        if k % par['replan'] == 0:
            self._update_val()
        vres = None
        if self.vdd_on and vdd_fn is not None and (k % par['vdd_every'] == 0) and (self.stage != 'takeoff'):
            vres = self._vdd_update(vdd_fn(), cam, fw, up, rt, z64)
        if self.stage == 'takeoff':
            if agl >= 4.0 or t > 3.0:
                self._set_stage('transit')
            else:
                return (self.fl.cmd_takeoff(), None)
        if vres is not None and self.stage in ('transit', 'search', 'hail'):
            fire, n = vres
            if self.est is None or math.hypot(self.est['x'] - fire[0], self.est['y'] - fire[1]) > 3.0:
                self._ev('vdd', round(fire[0], 1), round(fire[1], 1), n)
                if self.frames < 40 and par['direct']:
                    self.B.pos(fire[0], fire[1], par['lr_vdd'], sig=1.5)
                    pm, _ = self.B.pos_mass(fire[0], fire[1], 2.0)
                    self._commit_xy(fire[0], fire[1], fire[2], pm, 'vdd', up_only=True, verified=False)
                elif n >= 3:
                    self._commit_xy(fire[0], fire[1], fire[2], 0.9, 'vdd', up_only=True, verified=True)
        if self.stage in ('transit', 'search'):
            self._search_logic(pos, yaw_now, t)
        if self.stage in ('term', 'hail'):
            return self._terminal(st, pos, rpy, vel, yaw_now, tilt, t, cam, fw, up, rt, z64)
        goal = self.goal if self.goal is not None else pos
        d = goal[0:2] - pos[0:2]
        dist = float(np.linalg.norm(d))
        hd = d / max(dist, 1e-06)
        if self.stage == 'transit':
            vmag = min(par['V'], max(1.0, 0.5 * dist))
            yaw = math.atan2(hd[1], hd[0]) if dist > 1.0 else yaw_now
        else:
            vmag = min(par['V'], max(0.3, 0.6 * dist))
            vhd = math.atan2(hd[1], hd[0])
            if self.gyaw is None:
                self.gyaw = yaw_now
            gy = self.gyaw
            if dist > 6.0:
                dy = wrap(gy - vhd)
                yaw = vhd + float(np.clip(dy, -math.radians(45.0), math.radians(45.0)))
            else:
                yaw = gy
                if angd(yaw, vhd) > math.radians(45.0) and dist > 1.0:
                    vmag = min(vmag, par['offnose_v'])
            if dist < 2.0 and self.arrive_k is None and (len(goal) > 2):
                self.arrive_k = k
        zgoal = None
        if self.goal is not None and len(self.goal) > 2 and (self.stage == 'search') and (dist < 12.0):
            zgoal = float(self.goal[2])
            if ground is not None:
                zgoal = float(np.clip(zgoal, ground + 3.0, ground + par['H2'] + 3.0))
        if self.a3t is not None and self.stage == 'transit' and (self.dsc >= self.a3t['d']) and (math.hypot(float(pos[0]) - float(self.clue[0]), float(pos[1]) - float(self.clue[1])) > self.a3t['r']):
            self.a3t_n += 1
            self.a3t_t1 = round(t, 2)
            a = self.fl.cmd_far(hd, vmag, yaw, par['H'], dist, par['V'], self.a3t)
        else:
            a = self.fl.cmd_search(hd, vmag, yaw, par['H'], zgoal, dist)
        purpose = None
        if self.stage in ('transit', 'search') and self.frames < 40 - par['reserve'] and (k - self.last_f >= par['gap']) and (self.pending is None):
            yerr = angd(yaw, yaw_now)
            newview = True
            if self.last_fpose is not None:
                newview = float(np.linalg.norm(pos - self.last_fpose[0])) >= par['fr_move'] or angd(yaw_now, self.last_fpose[1]) >= math.radians(par['fr_turn'])
            if newview and math.degrees(yerr) < par['yaw_tol'] and (tilt < par['tilt_tol']):
                self._map_now(z64, cam, fw, up, rt)
                self._update_val()
                g, _ = self._frame_gain(cam, fw, up, rt, z64)
                if g >= self._gmin(t):
                    purpose = 'search'
        if purpose is not None:
            self._request(purpose, pos, yaw_now)
            a[5] = 1.0
        else:
            a[5] = 0.0
        return (a, purpose)

    def _map_now(self, z64, cam, fw, up, rt):
        if self.par['vis_seen'] and getattr(self, '_mapped_k', -1) != self.k:
            self.M.add(z64, cam, fw, up, rt, self._hf)
            self._mapped_k = self.k

    def _request(self, purpose, pos, yaw_now):
        self.frames += 1
        self.last_f = self.k
        self.last_fpose = (np.asarray(pos, np.float64).copy(), float(yaw_now))
        self.pending = dict(purpose=purpose, k=self.k)

    def _terminal(self, st, pos, rpy, vel, yaw_now, tilt, t, cam, fw, up, rt, z64):
        par = self.par
        e = self.est
        dh = float(math.hypot(e['x'] - pos[0], e['y'] - pos[1]))
        if par['term_to'] > 0 and self.commit in ('hit', 'vdd') and (not e.get('rgb_ok')) and (t < 52.0):
            if dh < 3.0:
                e.setdefault('near_k', self.k)
                if (self.k - e['near_k']) * DT > par['term_to'] and (not self.fl._dp_gate('tto', pos, vel, (e['x'], e['y']), e['z'] - 3.0)):
                    self._abort('term_timeout')
                    return (self.fl.cmd_hover(), None)
        a = self.fl.cmd_approach(e['x'], e['y'], e['z'] - 3.0)
        purpose = None
        if self.pending is None and self.frames < 40:
            if self.fl.want_frame:
                purpose = 'refine'
            elif self.stage == 'hail' and self.k - self.last_f >= par['gap'] and (dh > 14.0) and (tilt < par['tilt_tol']):
                self._map_now(z64, cam, fw, up, rt)
                g, _ = self._frame_gain(cam, fw, up, rt, z64)
                if g >= 0.5 * self._gmin(t):
                    purpose = 'hail'
        if purpose is not None:
            self._request(purpose, pos, yaw_now)
            a[5] = 1.0
        else:
            a[5] = 0.0
            if self.fl.want_frame:
                self.fl.n_frames = max(0, self.fl.n_frames - 1)
        return (a, purpose)

def make_flight_class(FL):
    Flight, SafetyGrid, vel_action = (FL.Flight, FL.SafetyGrid, FL.vel_action)

    class SearchFlight(Flight):
        look_s = 16.0

        def observe(self, st, depth):
            st = np.asarray(st, np.float64).reshape(-1)
            self.k += 1
            self.want_frame = False
            self.st = st
            pos = st[0:3]
            rpy = st[3:6]
            self.agl = float(st[162]) * 20.0 if st.size > 162 else 20.0
            if self.grid is None:
                self.pad = pos[0:2].copy()
                self.z0 = float(pos[2])
                self.grid = SafetyGrid(pos[0:2])
            try:
                self.P = self.grid.add_depth(depth, pos, rpy)
            except Exception:
                self.P = None
            self.ground = self._ground(pos, self.agl)

        def cmd_hover(self):
            st = self.st
            pos = st[0:3]
            vel = st[6:9]
            g = self.ground
            zt = g + self.H if g is not None else pos[2]
            c = self.sh([0.0, 0.0, float(np.clip(1.2 * (zt - pos[2]), -1.0, 2.0))], vel)
            return vel_action(c, float(st[5]))

        def cmd_takeoff(self):
            st = self.st
            self._set_phase('takeoff')
            c = self.sh([0.0, 0.0, 3.0], st[6:9])
            return vel_action(c, float(st[5]))

        def cmd_search(self, hd, vmag, yaw, H, zgoal, dist):
            st = self.st
            pos = st[0:3]
            vel = st[6:9]
            g = self.ground
            self._set_phase('search')
            margin = max(2.5, 0.6 * H)
            z_req = zgoal if zgoal is not None else g + H if g is not None else pos[2] - 1.0
            hd = np.asarray(hd, np.float64)
            try:
                if dist > 2.0:
                    al, hm, unseen = self.grid.corridor(pos[0:2], hd, self.look_s, self.half_w)
                    need = hm + margin
                    fl = max(float(need.max()), g + margin if g is not None else -1000000000.0)
                    vmag = self._climb_cap(al, need, pos[2], vmag)
                    if unseen > 0.6:
                        vmag = min(vmag, 2.0)
                        self.stats['n_unseen_slow'] += 1
                else:
                    fl = max(self.grid.disc_max(pos[0:2], 4.0) + margin, g + margin if g is not None else -1000000000.0)
                if fl > z_req:
                    z_req = fl
                    self.stats['n_floor'] += 1
            except Exception:
                pass
            vz = float(np.clip(1.2 * (z_req - pos[2]), -2.0, 2.5))
            vh = hd * vmag
            c = self.sh([vh[0], vh[1], vz], vel)
            self.dbg = {'ph': 'search', 'dh': dist, 'z_des': z_req, 'vcap': vmag}
            return vel_action(c, yaw)
        a3t_exc = 0

        def cmd_far(self, hd, vmag, yaw, H, dist, V, cfg):
            vm0 = vmag
            if not cfg['vz']:
                return self.cmd_search(hd, float(cfg['v']) if vmag >= V - 1e-09 else vmag, yaw, H, None, dist)
            try:
                vm = float(cfg['v']) if vmag >= V - 1e-09 else float(vmag)
                st = self.st
                pos = st[0:3]
                vel = st[6:9]
                g = self.ground
                hd = np.asarray(hd, np.float64)
                margin = max(2.5, 0.6 * H)
                z = float(pos[2])
                al, hm, unseen = self.grid.corridor(pos[0:2], hd, self.look_s, self.half_w)
                vm = self._climb_cap(al, hm + margin, z, vm)
                if unseen > 0.6:
                    vm = min(vm, 2.0)
                    self.stats['n_unseen_slow'] += 1
                alL, hmL, _ = self.grid.corridor(pos[0:2], hd, float(cfg['look']), self.half_w)
                zs = g + H if g is not None else z - 1.0
                vz = float(np.clip(1.2 * (zs - z), -float(cfg['dn']), float(cfg['up'])))
                vn = float(math.hypot(float(vel[0]), float(vel[1])))
                vh = max(vm, vn)
                ok = hmL > -100000000.0
                if ok.any():
                    k = float(np.max((hmL[ok] + margin - z) / np.maximum(alL[ok] - float(cfg['b']), 1.0)))
                    vz = max(vz, k * vh)
                if vn > 1.0:
                    hv = np.array([float(vel[0]), float(vel[1])]) / vn
                    if float(hv @ hd) < 0.985:
                        alV, hmV, _ = self.grid.corridor(pos[0:2], hv, self.look_s, self.half_w)
                        okV = hmV > -100000000.0
                        if okV.any():
                            kV = float(np.max((hmV[okV] + margin - z) / np.maximum(alV[okV] - float(cfg['b']), 1.0)))
                            vz = max(vz, kV * vh)
                if g is not None and z < g + margin:
                    vz = max(vz, 1.2 * (g + margin - z))
                vz = float(np.clip(vz, -2.0, self.vz_max))
                if not (math.isfinite(vz) and math.isfinite(vm)):
                    raise FloatingPointError('a3t')
            except Exception:
                self.a3t_exc += 1
                return self.cmd_search(hd, vm0, yaw, H, None, dist)
            self._set_phase('search')
            vh_ = hd * vm
            c = self.sh([vh_[0], vh_[1], vz], vel)
            self.dbg = {'ph': 'search', 'dh': dist, 'z_des': z + vz / 1.2, 'vcap': vm}
            return vel_action(c, yaw)

        def cmd_approach(self, x, y, vtop):
            st = self.st
            pos = st[0:3]
            rpy = st[3:6]
            vel = st[6:9]
            yaw_now = float(rpy[2])
            axy = np.array([float(x), float(y)])
            e = axy - pos[0:2]
            dh = float(np.linalg.norm(e))
            tgt_hd = math.atan2(e[1], e[0]) if dh > 1e-06 else yaw_now
            if self.phase not in ('appr', 'hold'):
                self._set_phase('appr')
            return self._approach(pos, vel, rpy, yaw_now, self.ground, axy, dh, e, tgt_hd, float(vtop))
    return SearchFlight
FBG_ON = os.environ.get('KT_X_FBG', '1').strip() == '1'
FBG_A = 0.25
FBG_MIN = 0.4
_fbg_orig_set_val = Belief.set_val

def _fbg_set_val(self, val):
    _fbg_orig_set_val(self, val)
    if not FBG_ON:
        return
    w = getattr(self, '_fbg_w', None)
    if w is None:
        cb = self.clue * B45
        s = float(cb.sum())
        oor = float((cb * ~self.inr).sum() / s) if s > 0 else 0.0
        w = self._fbg_w = min(1.0, FBG_A * oor) if oor > FBG_MIN else 0.0
    if not w > 0.0:
        return
    flat = np.asarray(val, np.float64) * self.clue * B45
    s = float(flat.sum())
    if not s > 0.0:
        return
    self.prior = (1.0 - w) * self.prior + w * (flat / s)
    self._mk()
Belief.set_val = _fbg_set_val
_fbg_orig_uv = Brain._update_val

def _fbg_update_val(self, force=False):
    _fbg_orig_uv(self, force)
    B = self.B
    if B is not None:
        w = float(getattr(B, '_fbg_w', 0.0) or 0.0)
        self.x43_fbg_gate = int(w > 0.0)
        self.x43_fbg_w = round(w, 4)
Brain._update_val = _fbg_update_val
