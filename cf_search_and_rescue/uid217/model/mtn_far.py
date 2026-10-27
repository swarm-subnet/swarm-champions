import math
import os
import types
from collections import Counter

import numpy as np

FAR_PK = float(os.environ.get('KTM_FAR_PK', '0.9'))
FAR_SMAX = float(os.environ.get('KTM_FAR_SMAX', '60.0'))
FAR_KMIN = float(os.environ.get('KTM_FAR_KMIN', '0'))


class _Cache:

    def __init__(self, sess):
        self.sess = sess
        self.last = None

    def run(self, *a, **k):
        self.last = self.sess.run(*a, **k)
        return self.last

    def __getattr__(self, name):
        return getattr(self.sess, name)


def march(cam, D, ground, s0, s1, step=0.5, clear=0.3):
    s, P, _ = march_k(cam, D, ground, s0, s1, step, clear)
    return s, P


def march_k(cam, D, ground, s0, s1, step=0.5, clear=0.3, back=2.0):
    s = s0
    hist = []
    while s <= s1:
        P = cam + s * D
        g = ground(float(P[0]), float(P[1]))
        if g is not None:
            hist.append((s, float(P[2]) - g))
            if P[2] <= g + clear:
                prev = [h for h in hist if h[0] <= s - back + 1e-9]
                k = (prev[-1][1] - hist[-1][1]) / (s - prev[-1][0]) if prev else None
                return s, P, k
        s += step
    return None, None, None


def far_hit(a3, outputs, st):
    vc = a3.vdr_cfg
    h, o, c, m = outputs
    i = int(np.argmax(h))
    gy, gx = divmod(i, 64)
    pk = float(h[gy, gx])
    if pk < max(vc['thr'], FAR_PK):
        return None, 'weak'
    cx = 4.0 * (gx + 0.5 + float(o[0, gy, gx]))
    cy = 4.0 * (gy + 0.5 + float(o[1, gy, gx]))
    plp = float(sum((c[j, gy, gx] for j in vc['lp_ch'])))
    pos = st[0:3].astype(np.float64)
    rpy = st[3:6].astype(np.float64)
    cam, fw, up, rt = a3.C.cam_of(pos, rpy)
    B = a3.brain
    tan = B.par['tan']
    u = cx / 128.0 - 1.0
    v = 1.0 - cy / 128.0
    D = fw + rt * (u * tan) + up * (v * tan)
    s, P, k = march_k(cam, D, B._gaim, float(vc['zmax']), FAR_SMAX)
    if s is None:
        return None, 'no_ray'
    if FAR_KMIN > 0 and (k is None or k < FAR_KMIN):
        return None, 'grazing'
    pad = B.pad
    if math.hypot(P[0] - pad[0], P[1] - pad[1]) < a3.par['pad_r'] and abs(P[2] - pad[2]) < 3.0:
        return None, 'pad'
    top = float(P[2]) + (0.4 if plp >= 0.5 else 0.9)
    r = float(np.linalg.norm(P - cam))
    return dict(x=float(P[0]), y=float(P[1]), top=top, r=r, lp=bool(plp >= 0.5), peak=pk, zc=float(s), n_ms=0,
                far=True), None


def install(a3):
    if getattr(a3, 'mtn_far_on', False):
        return
    a3.mtn_far_on = True
    a3.mtn_far_stats = Counter()
    a3.vdr = _Cache(a3.vdr)
    orig = a3._vdr_hit

    def vdr_hit(self, rgb, dep, st):
        n0 = self.stats.get('vdr_far', 0)
        hit = orig(rgb, dep, st)
        if hit is not None or self.stats.get('vdr_far', 0) == n0:
            return hit
        B = self.brain
        pend = B.pending
        if pend is None or pend.get('purpose') not in ('search', 'hail') or B.stage not in ('transit', 'search', 'hail'):
            return None
        try:
            hit, why = far_hit(self, self.vdr.last, st)
        except Exception:
            self.mtn_far_stats['exc'] += 1
            return None
        if hit is None:
            self.mtn_far_stats[why] += 1
            return None
        self.mtn_far_stats['hit'] += 1
        B.mtn_far_stats = self.mtn_far_stats
        B._ev('far_hit', round(hit['x'], 2), round(hit['y'], 2), round(hit['top'], 2), round(hit['r'], 1),
              round(hit['peak'], 3))
        return hit

    a3._vdr_hit = types.MethodType(vdr_hit, a3)
