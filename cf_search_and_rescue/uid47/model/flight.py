"""mtn_a3 FLIGHT LAYER (numpy only, no models, no truth): protoC wrapped into a reusable support module (M1).

Pieces (A3-BIP+ plan, section 2 "Take-off", "Flight layer", "Terminal"):
  Shaper       acceleration slew 3 m/s^2, velocity-heading rate 35 deg/s, vz band [-0.25, +0.6] around the CURRENT vz,
               demanded tilt <= 35 deg (the SAR VEL action is a pure velocity-D law: F = 0.2*dv_h, 0.5*dv_z + m g).
  SafetyGrid   2 m max-obstacle-height grid (+ seen flag) over 400 x 400 m, filled from every depth frame (64x64).
  LocalMap     0.5 m max-height map (20 x 20 m) around the victim estimate, filled during the approach.
  Flight       take-off: vertical climb to 4 m AGL, yaw toward the target.
               goto (search legs, aim={'xy','agl'}): AGL-hold cruise, 15 m x +-3 m corridor look-ahead on the grid
                     (margin max(2.5, 0.6 H)), climb-time speed cap, nose within +-40 deg of the velocity unless mapped.
               approach (aim={'xy','vtop'}): 3-D line of sight to the hover point H = (x, y, vtop + 3.0) with a 3-D
                     braking profile (2.5 m/s^2 to 0.9 m/s at 1.8 m), a glide cone (<= 0.45 (dh - 1) above H), the
                     corridor/ground terrain FLOOR as a hard lower bound on z (margin 2.0 m in transit tapering to 1.5 m
                     inside 5 m; the victim box +-1.2 m excluded; the 2-4 m band wins inside 3 m), spare 3-D speed
                     given to the horizontal when the floor/cone moves vz off the line of sight, nose on the victim,
                     RGB refine frames at range gates 18/13/9/6.5 m while the victim projects near the image centre.
               hover placement: inside 14 m the hover point may move <= 1.3 m / +-0.6 m inside the confirm cylinder to
                     the candidate with the largest LocalMap clearance (only when the estimate's own clearance < 1.6 m).
               hold + dwell fallback: 3-D P-hold; settled 2.6 s at the hover point without the episode ending ->
                     the aim is wrong -> step through hover offsets (dz -0.8, +0.8, dx/dy +-1.3, dz -1.5, +1.5).
Interface (the planner / harness supplies the target; this module never reads truth):
    fl = Flight(); fl.reset()
    a = fl.step(state, depth, aim)   # aim: None (climb/hover) | {'xy':(x,y), 'agl':H} (goto) | {'xy':(x,y), 'vtop':z}
    fl.want_frame                      # True on a tick where the terminal requested an RGB refine frame (a[5] = 1)
Action: a[0:3] unit world velocity, a[3] = |v|/3, a[4] = absolute yaw/pi, a[5] = RGB request.
M1 measurements (wnew/a3p/m1, harness-only arm agents/mtn_a3_m1): see runs/m1_summary.json.
"""
from __future__ import annotations

import math

import numpy as np

DT = 0.02
MG = 0.027 * 9.8
TILT_LIM = math.radians(35.0)
DZ_DOWN = 0.25
DZ_UP = 0.6
DMIN, DMAX = 0.5, 30.0
CAMF, CAMU = 0.13, 0.05
VMAX = 3.0


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def axes(rpy):
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    fw = np.array([cy * cp, sy * cp, -sp])
    up = np.array([cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr])
    return fw, up, np.cross(fw, up)          # forward, up, right


def vel_action(v, yaw: float, rgb: float = 0.0) -> np.ndarray:
    v = np.asarray(v, np.float64)
    n = float(np.linalg.norm(v))
    a = np.zeros(6, np.float32)
    if n > 1e-9:
        a[0:3] = (v / n).astype(np.float32)
        a[3] = np.float32(min(1.0, n / VMAX))
    a[4] = np.float32(wrap(float(yaw)) / math.pi)
    a[5] = np.float32(rgb)
    return a


class Shaper:
    """Acceleration/heading-rate slew + tilt-envelope guard on the world velocity command (protoC)."""

    def __init__(self, A=3.0, W=math.radians(35.0), vmax=VMAX):
        self.A, self.W, self.vmax = A, W, vmax
        self.cmd = np.zeros(3); self.n_guard = 0; self.max_demand = 0.0

    def reset(self):
        self.cmd = np.zeros(3); self.n_guard = 0; self.max_demand = 0.0

    def __call__(self, want, vel):
        want = np.asarray(want, np.float64).copy()
        n = float(np.linalg.norm(want))
        if n > self.vmax:                                   # keep the vertical part, shrink horizontal first
            vz = float(np.clip(want[2], -self.vmax, self.vmax)); hmax = math.sqrt(max(0.0, self.vmax ** 2 - vz ** 2))
            hn = float(np.linalg.norm(want[0:2]))
            if hn > hmax:
                want[0:2] = want[0:2] * (hmax / hn)
            want[2] = vz
        ch = self.cmd[0:2]; wh = want[0:2]; nc = float(np.linalg.norm(ch)); nw = float(np.linalg.norm(wh))
        if nc > 0.5 and nw > 0.5:                           # velocity-heading rate limit
            a0 = math.atan2(ch[1], ch[0]); a1 = math.atan2(wh[1], wh[0])
            da = wrap(a1 - a0); lim = self.W * DT
            if abs(da) > lim:
                a = a0 + math.copysign(lim, da); want[0:2] = nw * np.array([math.cos(a), math.sin(a)])
        dh = want[0:2] - self.cmd[0:2]; dn = float(np.linalg.norm(dh)); lim = self.A * DT
        if dn > lim:
            dh = dh * (lim / dn)
        c = np.array([self.cmd[0] + dh[0], self.cmd[1] + dh[1], want[2]])
        v = np.asarray(vel, np.float64)
        ez = float(np.clip(c[2] - v[2], -DZ_DOWN, DZ_UP)); c[2] = v[2] + ez
        eh = c[0:2] - v[0:2]; ehn = float(np.linalg.norm(eh))
        allow = math.tan(TILT_LIM) * (MG + 0.5 * ez) / 0.2
        if ehn > allow:
            c[0:2] = v[0:2] + eh * (allow / ehn); self.n_guard += 1
        dem = math.degrees(math.atan2(0.2 * float(np.linalg.norm(c[0:2] - v[0:2])), MG + 0.5 * (c[2] - v[2])))
        self.max_demand = max(self.max_demand, dem)
        self.cmd = c
        return c


class SafetyGrid:
    """Max observed obstacle-top height per 2 m cell over a 400 x 400 m window centred on the pad."""

    def __init__(self, centre_xy, half=200.0, res=2.0, step=4):
        self.res = float(res); self.half = float(half)
        self.n = int(round(2 * half / res))
        self.o = np.asarray(centre_xy, np.float64)[0:2] - half
        self.h = np.full((self.n, self.n), -1e9, np.float32)
        self.seen = np.zeros((self.n, self.n), bool)
        px = (np.arange(0, 256, step) + 0.5) / 128.0 - 1.0
        self.step = step
        self.U = np.broadcast_to(px[None, :], (len(px), len(px))).ravel()
        self.V = np.broadcast_to(-px[:, None], (len(px), len(px))).ravel()

    def ij(self, xy):
        xy = np.asarray(xy, np.float64)
        i = np.floor((xy[..., 0] - self.o[0]) / self.res).astype(np.int64)
        j = np.floor((xy[..., 1] - self.o[1]) / self.res).astype(np.int64)
        ok = (i >= 0) & (i < self.n) & (j >= 0) & (j < self.n)
        return np.clip(i, 0, self.n - 1), np.clip(j, 0, self.n - 1), ok

    def add_depth(self, depth, pos, rpy, tan=1.0):
        d = np.asarray(depth, np.float32).reshape(256, 256)[::self.step, ::self.step].ravel().astype(np.float64)
        d = d * (DMAX - DMIN) + DMIN
        fw, up, rt = axes(rpy)
        cam = np.asarray(pos, np.float64) + fw * CAMF + up * CAMU
        u = self.U * tan * d; v = self.V * tan * d
        P = cam[None, :] + d[:, None] * fw[None, :] + u[:, None] * rt[None, :] + v[:, None] * up[None, :]
        hit = d < DMAX - 1.0
        i, j, ok = self.ij(P[:, 0:2])
        self.seen[j[ok], i[ok]] = True
        m = hit & ok
        if m.any():
            np.maximum.at(self.h, (j[m], i[m]), P[m, 2].astype(np.float32))
        return P[hit]

    def corridor(self, p_xy, hd, length, half_w, excl_xy=None, excl_r=0.0, d_step=1.0):
        """per along-distance max obstacle height over the corridor ahead (5 lateral samples), and the unseen share."""
        al = np.arange(d_step, length + 1e-6, d_step)
        la = np.linspace(-half_w, half_w, 5)
        A, Lt = np.meshgrid(al, la, indexing="ij")
        hd = np.asarray(hd, np.float64); nr = np.array([-hd[1], hd[0]])
        X = p_xy[0] + A * hd[0] + Lt * nr[0]; Y = p_xy[1] + A * hd[1] + Lt * nr[1]
        i, j, ok = self.ij(np.stack([X, Y], -1))
        H = np.where(ok, self.h[j, i], -1e9)
        S = np.where(ok, self.seen[j, i], False)
        if excl_xy is not None and excl_r > 0.0:
            ex = np.hypot(X - excl_xy[0], Y - excl_xy[1]) <= excl_r
            H = np.where(ex, -1e9, H)
        return al, H.max(axis=1), float(1.0 - S.mean())

    def disc_max(self, xy, r):
        k = int(math.ceil(r / self.res))
        i0, j0, ok = self.ij(np.asarray(xy, np.float64))
        if not ok:
            return -1e9
        i0 = int(i0); j0 = int(j0)
        sl = self.h[max(0, j0 - k):j0 + k + 1, max(0, i0 - k):i0 + k + 1]
        return float(sl.max()) if sl.size else -1e9


class LocalMap:
    """0.5 m max-height map (20 x 20 m) around the victim estimate, filled from every depth frame of the approach; used
    to place the hover point where the mapped obstacles (rocks, cliff faces next to the victim) are >= 1 m away."""

    def __init__(self, centre_xy, half=10.0, res=0.5):
        self.c = np.asarray(centre_xy, np.float64)[0:2].copy(); self.res = res; self.half = half
        self.n = int(round(2 * half / res))
        self.h = np.full((self.n, self.n), -1e9, np.float32)
        g = (np.arange(self.n) + 0.5) * res - half
        self.GX, self.GY = np.meshgrid(g + self.c[0], g + self.c[1])       # [j, i]

    def add(self, P):
        if P is None or len(P) == 0:
            return
        i = np.floor((P[:, 0] - self.c[0] + self.half) / self.res).astype(np.int64)
        j = np.floor((P[:, 1] - self.c[1] + self.half) / self.res).astype(np.int64)
        ok = (i >= 0) & (i < self.n) & (j >= 0) & (j < self.n)
        if ok.any():
            np.maximum.at(self.h, (j[ok], i[ok]), P[ok, 2].astype(np.float32))

    def clearance(self, Q, r_query=3.5, body=0.1):
        """estimated clearance of candidate points Q (M x 3) to the mapped column tops within r_query of the centre."""
        m = (self.h > -1e8) & (np.hypot(self.GX - self.c[0], self.GY - self.c[1]) <= r_query + 1.5)
        if not m.any():
            return np.full(len(Q), 9.0)
        C = np.stack([self.GX[m], self.GY[m], self.h[m]], 1)
        dxy = np.maximum(0.0, np.hypot(Q[:, None, 0] - C[None, :, 0], Q[:, None, 1] - C[None, :, 1]) - 0.5 * self.res)
        dz = Q[:, None, 2] - C[None, :, 2]
        d = np.where(dz > 0.0, np.sqrt(dxy ** 2 + dz ** 2), dxy)
        return d.min(axis=1) - body


class Flight:
    """Take-off / goto cruise / 3-D approach + hold on a supplied target. Numpy only; ~1 ms per tick."""

    def __init__(self, H=6.0, V=3.0, a_brake=2.5, v_e=0.9, d_e=1.8, k_hold=0.6, up=3.0,
                 m_cruise=2.0, m_term=1.5, look=15.0, half_w=3.0, nose_cone=math.radians(40.0), off_nose_v=2.0,
                 takeoff_agl=4.0, gates=(18.0, 13.0, 9.0, 6.5), refine_every=5, refine_max=4, vz_max=2.5,
                 cone_tan=0.45, kz_hold=1.2, view_tan=0.85, fb=True, fb_settle=2.6,
                 fb_seq=((0.0, 0.0, -0.8), (0.0, 0.0, 0.8), (1.3, 0.0, 0.0), (-1.3, 0.0, 0.0), (0.0, 1.3, 0.0),
                         (0.0, -1.3, 0.0), (0.0, 0.0, -1.5), (0.0, 0.0, 1.5)), h_budget=True,
                 excl_r=1.2, hover_opt=True, hover_rmax=1.3, hover_clr=1.6, takeoff_normal=False):
        self.H, self.V = H, V
        self.a_brake, self.v_e, self.d_e, self.k_hold, self.up = a_brake, v_e, d_e, k_hold, up
        self.m_cruise = max(2.5, 0.6 * H) if m_cruise is None else m_cruise
        self.m_term, self.look, self.half_w = m_term, look, half_w
        self.nose_cone, self.off_nose_v, self.takeoff_agl = nose_cone, off_nose_v, takeoff_agl
        self.gates, self.refine_every, self.refine_max = tuple(gates), refine_every, refine_max
        self.vz_max, self.cone_tan, self.kz_hold, self.view_tan = vz_max, cone_tan, kz_hold, view_tan
        self.fb, self.fb_settle, self.fb_seq = fb, fb_settle, tuple(tuple(x) for x in fb_seq)
        self.h_budget = h_budget
        self.excl_r, self.hover_opt, self.hover_rmax, self.hover_clr = excl_r, hover_opt, hover_rmax, hover_clr
        self.takeoff_normal = takeoff_normal
        self.sh = Shaper()
        self.reset()

    def reset(self):
        self.sh.reset()
        self.grid = None; self.k = 0; self.phase = "takeoff"; self.pad = None; self.z0 = None
        self.want_frame = False; self.n_frames = 0; self.last_frame_k = -10 ** 6; self.gi = 0
        self.ground = None; self.settled = 0; self.fb_i = 0; self.fb_key = None
        self.lmap = None; self.hoff = np.zeros(3); self.hoff_k = -10 ** 6; self.to_dir = None; self.P = None
        self.stats = {"phase_k": {}, "n_climbcap": 0, "n_offnose": 0, "n_unseen_slow": 0, "n_floor": 0,
                      "max_floor_lift": 0.0, "n_frames": 0, "frame_dh": [], "fb_steps": 0, "to_tilt": None,
                      "hoff": None, "hover_clr0": None, "hover_clr": None}
        self.dbg = {}

    # ------------------------------------------------------------------------------------------------ helpers
    def _set_phase(self, ph):
        if ph != self.phase:
            self.phase = ph
            self.stats["phase_k"].setdefault(ph, self.k)

    def _ground(self, pos, agl):
        """ground height under the drone: the AGL ray (<= 20 m), else the grid max around the drone's xy."""
        if agl < 19.9:
            return pos[2] - agl
        g = self.grid.disc_max(pos[0:2], 2.0)
        return g if g > -1e8 else None

    def _nose(self, tgt_hd, yaw_now, hd, vmag, mapped):
        """keep the velocity within +-nose_cone of the nose unless the corridor is mapped (then <= off_nose_v)."""
        off = wrap(tgt_hd - yaw_now)
        if abs(off) <= self.nose_cone:
            return hd, vmag
        self.stats["n_offnose"] += 1
        if mapped:
            return hd, min(vmag, self.off_nose_v)
        vdir = yaw_now + math.copysign(self.nose_cone, off)
        return np.array([math.cos(vdir), math.sin(vdir)]), vmag * max(0.0, math.cos(wrap(tgt_hd - vdir)))

    def _fit_takeoff(self, pos):
        """terrain plane under/around the pad from the tick-1 depth points within 3.5 m -> take-off direction = normal."""
        self.to_p0 = np.asarray(pos, np.float64).copy()
        P = self.P
        if P is None or len(P) < 40:
            return
        r = np.hypot(P[:, 0] - pos[0], P[:, 1] - pos[1])
        m = (r < 3.5) & (P[:, 2] < pos[2] + 1.5)
        if int(m.sum()) < 40:
            return
        A = np.stack([P[m, 0] - pos[0], P[m, 1] - pos[1], np.ones(int(m.sum()))], 1)
        coef, res, rk, _ = np.linalg.lstsq(A, P[m, 2], rcond=None)
        rms = float(np.sqrt(np.mean((A @ coef - P[m, 2]) ** 2)))
        n = np.array([-coef[0], -coef[1], 1.0]); n /= np.linalg.norm(n)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(n[2])))))
        self.stats["to_tilt"] = round(tilt, 1)
        if rms > 0.35 or tilt < 10.0:
            return
        if tilt > 40.0:                           # cap the lean of the take-off direction
            h = n[0:2] / max(1e-9, float(np.linalg.norm(n[0:2])))
            n = np.array([h[0] * math.sin(math.radians(40.0)), h[1] * math.sin(math.radians(40.0)), math.cos(math.radians(40.0))])
        self.to_dir = n

    def _hover_offset(self, axy, az):
        """choose the hover point inside the cylinder (<= hover_rmax from the estimate, az +- 0.6) with the largest
        mapped clearance; the estimate itself unless its clearance is below hover_clr."""
        L = self.lmap
        Q0 = np.array([[axy[0], axy[1], az]])
        c0 = float(L.clearance(Q0)[0])
        self.stats["hover_clr0"] = round(c0, 2)
        if c0 >= self.hover_clr:
            self.stats["hover_clr"] = round(c0, 2)
            return np.zeros(3)
        cand = [(0.0, 0.0)]
        for rr in (0.5, 0.9, self.hover_rmax):
            for a in range(8):
                cand.append((rr * math.cos(a * math.pi / 4), rr * math.sin(a * math.pi / 4)))
        cand = np.array(cand)
        dzs = np.array([0.0, 0.3, 0.6, -0.3])
        Q = np.array([[axy[0] + x, axy[1] + y, az + z] for x, y in cand for z in dzs])
        c = L.clearance(Q)
        off = Q - np.array([axy[0], axy[1], az])
        sc = np.minimum(c, 2.0) - 0.25 * np.hypot(off[:, 0], off[:, 1]) - 0.2 * np.abs(off[:, 2])
        b = int(np.argmax(sc))
        # hysteresis: keep the current offset unless the best candidate scores clearly higher
        cur = self.hoff
        cc = float(L.clearance((np.array([axy[0], axy[1], az]) + cur)[None, :])[0])
        scur = min(cc, 2.0) - 0.25 * float(np.hypot(cur[0], cur[1])) - 0.2 * abs(float(cur[2]))
        if float(sc[b]) < scur + 0.15:
            self.stats["hover_clr"] = round(cc, 2)
            return cur
        self.stats["hover_clr"] = round(float(c[b]), 2)
        return off[b] if c[b] > c0 + 0.1 else np.zeros(3)

    def _climb_cap(self, al, need, z, vcap):
        dz = need - z
        m = dz > 0.3
        if m.any():
            vc = np.maximum(0.6, (al[m] - 1.0) / (dz[m] / 2.0))
            vcm = float(vc.min())
            if vcm < vcap:
                self.stats["n_climbcap"] += 1
                return vcm
        return vcap

    # ------------------------------------------------------------------------------------------------ main step
    def step(self, st, depth, aim):
        st = np.asarray(st, np.float64).reshape(-1)
        self.k += 1; self.want_frame = False
        pos = st[0:3]; rpy = st[3:6]; vel = st[6:9]
        agl = float(st[162]) * 20.0 if st.size > 162 else 20.0
        yaw_now = float(rpy[2])
        if self.grid is None:
            self.pad = pos[0:2].copy(); self.z0 = float(pos[2])
            self.grid = SafetyGrid(pos[0:2])
        try:
            self.P = self.grid.add_depth(depth, pos, rpy)
        except Exception:
            self.P = None
        if self.k == 1 and self.takeoff_normal:
            try:
                self._fit_takeoff(pos)
            except Exception:
                self.to_dir = None
        g = self._ground(pos, agl)
        self.ground = g

        if aim is None:
            zt = (g + self.H) if g is not None else pos[2]
            c = self.sh([0.0, 0.0, float(np.clip(1.2 * (zt - pos[2]), -1.0, 2.0))], vel)
            return vel_action(c, yaw_now)

        axy = np.asarray(aim["xy"], np.float64)[0:2]
        e = axy - pos[0:2]; dh = float(np.linalg.norm(e))
        tgt_hd = math.atan2(e[1], e[0]) if dh > 1e-6 else yaw_now
        vtop = aim.get("vtop")

        # ---------------- take-off: vertical climb, yaw toward the target
        if self.phase == "takeoff":
            if agl >= self.takeoff_agl or self.k * DT > 3.0:
                self._set_phase("cruise" if vtop is None else "appr")
            else:
                w = [0.0, 0.0, 3.0]
                if self.to_dir is not None and float(np.linalg.norm(pos - self.to_p0)) < 1.3:
                    w = list(3.0 * self.to_dir)     # leave a sloped pad along its normal (clearance at the grace end)
                c = self.sh(w, vel)
                return vel_action(c, tgt_hd if dh > 2.5 else yaw_now)
        if vtop is None:
            return self._goto(pos, vel, yaw_now, g, axy, dh, e, tgt_hd, float(aim.get("agl", self.H)))
        return self._approach(pos, vel, rpy, yaw_now, g, axy, dh, e, tgt_hd, float(vtop))

    # ------------------------------------------------------------------------------------------------ goto (search legs)
    def _goto(self, pos, vel, yaw_now, g, axy, dh, e, tgt_hd, H):
        self._set_phase("cruise")
        hd = e / dh if dh > 1e-6 else np.array([math.cos(yaw_now), math.sin(yaw_now)])
        margin = max(2.5, 0.6 * H)
        z_des = (g + H) if g is not None else pos[2]
        vmag = min(self.V, 1.2 * dh)
        mapped = False
        try:
            al, hm, unseen = self.grid.corridor(pos[0:2], hd, self.look, self.half_w)
            need = hm + margin
            fl = max(float(need.max()), (g + margin) if g is not None else -1e9)
            if fl > z_des:
                z_des = fl; self.stats["n_floor"] += 1
            vmag = self._climb_cap(al, need, pos[2], vmag)
            mapped = unseen < 0.2
            if unseen > 0.6:
                vmag = min(vmag, 2.0); self.stats["n_unseen_slow"] += 1
        except Exception:
            pass
        hd, vmag = self._nose(tgt_hd, yaw_now, hd, vmag, mapped)
        vz = float(np.clip(1.2 * (z_des - pos[2]), -2.0, self.vz_max))
        vh = hd * vmag
        c = self.sh([vh[0], vh[1], vz], vel)
        self.dbg = {"ph": self.phase, "dh": dh, "z_des": z_des, "vcap": vmag}
        return vel_action(c, tgt_hd if dh > 2.5 else yaw_now)

    # ------------------------------------------------------------------------------------------------ approach + hold
    def _approach(self, pos, vel, rpy, yaw_now, g, axy, dh, e, tgt_hd, vtop):
        az = vtop + self.up
        # local 0.5 m map around the estimate (re-centred when the estimate moves) + hover-point placement
        if self.hover_opt and dh < 25.0:
            try:
                if self.lmap is None or float(np.linalg.norm(self.lmap.c - axy)) > 3.0:
                    self.lmap = LocalMap(axy)
                    if self.grid is not None:
                        pass
                self.lmap.add(self.P)
                if dh < 14.0 and self.k - self.hoff_k >= 10:
                    self.hoff = self._hover_offset(axy, az); self.hoff_k = self.k
                    self.stats["hoff"] = [round(float(x), 2) for x in self.hoff]
            except Exception:
                self.hoff = np.zeros(3)
        T = np.array([axy[0], axy[1], az]) + (self.hoff if self.hover_opt else 0.0)
        # dwell fallback: settled at the aim (|v| < 0.8, within 0.5 m) for fb_settle s and the episode still runs ->
        # the predicate is not met (aim error) -> step through small offsets of the hover point (vertical first)
        key = (round(float(axy[0]), 2), round(float(axy[1]), 2), round(float(vtop), 2))
        if key != self.fb_key:                     # a new estimate restarts the ladder
            self.fb_key = key; self.fb_i = 0; self.settled = 0
        if self.fb and self.fb_i > 0:
            T = T + np.asarray(self.fb_seq[(self.fb_i - 1) % len(self.fb_seq)])
        if self.fb and float(np.linalg.norm(T - pos)) < 0.5 and float(np.linalg.norm(vel)) < 0.8:
            self.settled += 1
            if self.settled * DT > self.fb_settle:
                self.fb_i += 1; self.settled = 0; self.stats["fb_steps"] += 1
                T = np.array([axy[0], axy[1], az]) + (self.hoff if self.hover_opt else 0.0) \
                    + np.asarray(self.fb_seq[(self.fb_i - 1) % len(self.fb_seq)])
        else:
            self.settled = 0
        az = float(T[2]); e = T[0:2] - pos[0:2]; dh = float(np.linalg.norm(e))
        tgt_hd = math.atan2(e[1], e[0]) if dh > 1e-6 else yaw_now
        d = T - pos; D3 = float(np.linalg.norm(d))
        self._set_phase("hold" if D3 < self.d_e else "appr")
        hd = e / dh if dh > 1e-6 else np.array([math.cos(yaw_now), math.sin(yaw_now)])
        # 3-D speed along the line of sight: braking profile to (v_e at d_e), then P-hold
        if D3 > self.d_e:
            s = min(self.V, math.sqrt(self.v_e ** 2 + 2.0 * self.a_brake * (D3 - self.d_e)))
            v = d / max(D3, 1e-6) * s
            vmag = float(np.linalg.norm(v[0:2])); vz = float(v[2])
        else:                                      # hold: horizontal P, stiffer vertical P (the band is +-1 m)
            vmag = min(self.v_e, self.k_hold * dh); vz = float(np.clip(self.kz_hold * d[2], -self.v_e, self.v_e))
        # optional glide cone: never be more than cone_tan * (dh - 1) above the hover point
        if self.cone_tan is not None and dh > 1.0:
            zc = az + self.cone_tan * (dh - 1.0)
            if pos[2] > zc:
                vz = min(vz, 1.2 * (zc - pos[2]))
        # terrain floor (hard lower bound on z)
        w = float(np.clip((dh - 5.0) / 10.0, 0.0, 1.0))
        margin = self.m_term + (self.m_cruise - self.m_term) * w
        L = min(self.look, dh + 1.0); hw = 2.0 + (self.half_w - 2.0) * w
        vcap = self.V; mapped = False; floor = -1e9
        try:
            al, hm, unseen = self.grid.corridor(pos[0:2], hd, max(L, 2.0), hw, excl_xy=axy, excl_r=self.excl_r)
            need = hm + margin
            floor = max(float(need.max()) if need.size else -1e9, (g + margin) if g is not None else -1e9)
            vcap = self._climb_cap(al, need, pos[2], vcap)
            mapped = unseen < 0.2
            if unseen > 0.6 and dh > 15.0:
                vcap = min(vcap, 2.0); self.stats["n_unseen_slow"] += 1
        except Exception:
            pass
        if dh <= 3.0:
            floor = min(floor, az - 0.3)          # the 2-4 m band has priority over the terminal margin
        if floor > pos[2] + vz * 0.5:
            lift = 1.2 * (floor - pos[2])
            if lift > vz:
                self.stats["n_floor"] += 1; self.stats["max_floor_lift"] = max(self.stats["max_floor_lift"], floor - pos[2])
                vz = lift
        vz = float(np.clip(vz, -self.vz_max, self.vz_max))
        if self.h_budget and D3 > self.d_e:
            # the floor / cone moved vz off the line of sight: give the rest of the 3-D speed budget to the horizontal,
            # capped by a horizontal braking profile so a held-up drone cannot overshoot the hover point
            vmag = max(vmag, min(math.sqrt(max(0.0, s * s - vz * vz)),
                                 math.sqrt(self.v_e ** 2 + 2.0 * self.a_brake * max(0.0, dh - self.d_e))))
        vmag = min(vmag, vcap)
        if dh > 2.5:
            hd, vmag = self._nose(tgt_hd, yaw_now, hd, vmag, mapped)
        if D3 < self.d_e + 0.7:                    # the dwell needs |v| < 1.0 (3-D)
            n3 = math.hypot(vmag, vz)
            if n3 > self.v_e:
                vmag *= self.v_e / n3; vz *= self.v_e / n3
        vh = hd * vmag
        c = self.sh([vh[0], vh[1], vz], vel)
        yaw_cmd = tgt_hd if dh > 2.5 else yaw_now
        # refine frames at range gates while the victim projects near the image centre
        if self.n_frames < self.refine_max and self.gi < len(self.gates) and dh <= self.gates[self.gi] \
                and self.k - self.last_frame_k >= self.refine_every:
            try:
                fw, up, rt = axes(rpy)
                cam = pos + fw * CAMF + up * CAMU
                q = np.array([axy[0], axy[1], vtop - 0.5]) - cam
                x = float(q @ fw)
                if x > 1.0 and abs(float(q @ rt)) / x < self.view_tan and abs(float(q @ up)) / x < self.view_tan:
                    self.want_frame = True; self.n_frames += 1; self.last_frame_k = self.k
                    self.stats["n_frames"] = self.n_frames; self.stats["frame_dh"].append(round(dh, 1))
                    while self.gi < len(self.gates) and dh <= self.gates[self.gi]:
                        self.gi += 1
            except Exception:
                pass
        self.dbg = {"ph": self.phase, "dh": dh, "z_des": floor, "vcap": vcap}
        return vel_action(c, yaw_cmd, 1.0 if self.want_frame else 0.0)
