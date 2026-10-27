import math
import os
import types
from collections import Counter

import numpy as np

RAY_DEG = float(os.environ.get('KTM_RAY_DEG', '3.0'))
RAY_BEHIND = float(os.environ.get('KTM_RAY_BEHIND', '1.5'))
RAY_SEP = float(os.environ.get('KTM_RAY_SEP', '2.0'))
KEEP_T = float(os.environ.get('KTM_RAY_KEEP_T', '1.5'))
RGB_KINDS = ('hit', 'det', 'confirm', 'retarget')


def behind_on_ray(cam, near, far, deg=RAY_DEG, behind=RAY_BEHIND):
    u = np.asarray(near, float) - cam
    w = np.asarray(far, float) - cam
    nu, nw = float(np.linalg.norm(u)), float(np.linalg.norm(w))
    if nu < 1.0 or nw < nu + behind:
        return False
    return float(u @ w) / (nu * nw) >= math.cos(math.radians(deg))


def install(brain, C):
    orig_vdd, orig_frame = brain._vdd_update, brain.frame_result
    brain.mtn_ray_stats = Counter()
    brain.mtn_ray_dbg = []
    brain.mtn_ray_last = None

    def est_top(e):
        return float(e['z']) - 3.0

    def vdd_update(self, fires, cam, fw, up, rt, z64):
        res = orig_vdd(fires, cam, fw, up, rt, z64)
        if res is None:
            return res
        (x, y, top), n = res
        self.mtn_ray_last = (self.t, x, y)
        e = self.est
        if self.stage != 'term' or e is None or e.get('ray') or n < 2:
            return res
        rays = e.get('rgb_rays') or []
        if not rays or not (self.commit in RGB_KINDS or e.get('rgb_ok')):
            return res
        if math.dist((x, y, top), (e['x'], e['y'], est_top(e))) <= RAY_SEP:
            return res
        if not any(behind_on_ray(np.asarray(c, float), p, (x, y, top)) for c, p in rays):
            return res
        old = (round(e['x'], 2), round(e['y'], 2), round(est_top(e), 2))
        self.B.pos(x, y, self.par['lr_vdd'], sig=1.5)
        pm, _ = self.B.pos_mass(x, y, 2.0)
        self._ev('ray_rt', round(x, 2), round(y, 2), round(top, 2), n, old)
        self._commit_xy(x, y, top, pm, 'vdd', up_only=True, verified=True)
        self.est['ray'] = old
        self.mtn_ray_stats['retarget'] += 1
        if len(self.mtn_ray_dbg) < 8:
            self.mtn_ray_dbg.append((round(self.t, 2), 'rt', (round(x, 2), round(y, 2), round(top, 2)), old))
        return res

    def depth_placed(self, e):
        if e.get('ray'):
            return True
        last = self.mtn_ray_last
        return (self.commit == 'vdd' and last is not None and self.t - last[0] <= KEEP_T
                and math.hypot(last[1] - e['x'], last[2] - e['y']) <= 2.0)

    def frame_result(self, hit, st, z64, zmap_fn=None):
        pend, e = self.pending, self.est
        cam = None
        if hit is not None and pend is not None:
            st64 = np.asarray(st, np.float64).reshape(-1)
            cam = np.asarray(C.cam_of(st64[0:3], st64[3:6])[0], float)
        given = hit
        if (cam is not None and pend.get('purpose') == 'refine' and e is not None
                and self.stage in ('term', 'hail') and math.hypot(hit['x'] - e['x'], hit['y'] - e['y']) > 3.0
                and depth_placed(self, e)
                and behind_on_ray(cam, (hit['x'], hit['y'], hit['top']), (e['x'], e['y'], est_top(e)))):
            self.mtn_ray_stats['keep'] += 1
            self._ev('ray_keep', round(hit['x'], 2), round(hit['y'], 2), round(hit['r'], 1))
            n = int(e.get('n', 0))
            top = (e['stop'] / n) if n else est_top(e)
            hit = dict(hit, x=float(e['x']), y=float(e['y']), top=float(top))
        out = orig_frame(hit, st, z64, zmap_fn)
        e2 = self.est
        if (cam is not None and hit is given and e2 is not None
                and math.hypot(given['x'] - e2['x'], given['y'] - e2['y']) <= 3.0):
            rays = e2.setdefault('rgb_rays', [])
            if len(rays) < 8:
                rays.append((tuple(float(v) for v in cam), (float(given['x']), float(given['y']), float(given['top']))))
        return out

    brain._vdd_update = types.MethodType(vdd_update, brain)
    brain.frame_result = types.MethodType(frame_result, brain)
