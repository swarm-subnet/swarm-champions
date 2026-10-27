import math, time

import numpy as np

N = 256
EYE_F, EYE_U = 0.13, 0.05
DMIN, DSPAN = 0.5, 29.5
DU, DV = 0.0, 1.0
SNOW_Z = 0.10
TILE, GLO, GHI, NT = 6.0, -50.0, 52.0, 17
NC = 102
SUBR = 0.25
NS = 408
HRANGE = 33.0
SLOPE = 0.08
HOUSE_Z = 3.2
DT = 0.02
T_LAST = 58.0
SEARCH_R, SPEED_LIMIT, DWELL, HORIZON = 30.0, 3.0, 2.0, 60.0

GENERATOR_FACTS = dict(
    tile=6.0, lattice_origin=-47.0, tiles_per_axis=17, grid=[-50.0, 52.0],
    outer_ring="the road lines at -47 m and +49 m are complete on both axes on every map (tile index 0 and 16)",
    base_line_spacing=[18.0, 24.0, 30.0], last_gap_may_be=[36.0, 42.0],
    base_lines_never_at=[-41.0, -35.0, 37.0, 43.0],
    segments="an interior segment between two crossing lines is kept (80%), dropped (5%, only spans of 18 m) or SHIFTED sideways by "
             "one tile (15%). Shifted segments put road runs 6 m beside their base line: also on the lattice lines -35 and +37 m "
             "(88 of 346 maps) and only 12 m from the neighbouring line; so do not rule out any lattice line and do not assume "
             "18 m between parallel roads [recognise_check R17]",
    victim_square=25.0, clue_radius=30.0, camera="forward, no tilt, field of view 90 +- 2 degrees, far plane 30 m",
    confirm="within 2 m horizontally, 2 to 4 m above the victim's top, 1 m/s or less, for 2 s; the band must be entered by 58 s",
    infeasible="the flight is cut when (60 - t) < max(0, distance to the clue - 30) / 3 + 2 s",
)

_POP16 = np.array([bin(i).count("1") for i in range(16)], np.uint8)


def target_time(start_xy, clue_xy):
    d = math.hypot(start_xy[0] - clue_xy[0], start_xy[1] - clue_xy[1])
    sweep = 0.70 * math.pi * SEARCH_R ** 2 / (24.0 * SPEED_LIMIT)
    return min(1.03 * (d / SPEED_LIMIT + sweep + DWELL), HORIZON * 0.95)


def _ro(a):
    a = np.array(a, copy=True)
    a.setflags(write=False)
    return a


class LivePublic:
    __slots__ = ("t", "t_search", "t_takeover", "t_target", "t_last", "x", "y", "z", "yaw", "speed", "clue", "start", "fov_deg",
                 "tiles", "ground_seen", "ground_range", "obst_h", "path", "fed", "facts")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw[k])

    def tile_centres_known_road(self):
        i, j = np.nonzero(self.tiles == 1)
        return np.stack([GLO + TILE * (i + 0.5), GLO + TILE * (j + 0.5)], axis=1)


class LiveVillageState:

    def __init__(self, vote_tick0=100, step=2, fov_step=4, fov_freeze=300, fov_freeze_mass=20000.0, min_votes=40.0, thr=0.5,
                 house_min=10.0, house_frac=0.05, sub_min=4, ring=False, keep_cost=True):
        self.vote_tick0, self.step, self.fov_step = int(vote_tick0), int(step), int(fov_step)
        self.fov_freeze, self.fov_freeze_mass = int(fov_freeze), float(fov_freeze_mass)
        self.min_votes, self.thr, self.house_min, self.house_frac = float(min_votes), float(thr), float(house_min), float(house_frac)
        self.sub_min, self.ring, self.keep_cost = int(sub_min), bool(ring), bool(keep_cost)
        u0 = (np.arange(N) + DU) / N * 2.0 - 1.0
        v0 = 1.0 - (np.arange(N) + DV) / N * 2.0
        self._U, self._V = np.meshgrid(u0[::self.step], v0[::self.step])
        self._Uf, self._Vf = np.meshgrid(u0[::self.fov_step], v0[::self.fov_step])
        self._wf = max(1, 3 // self.step)
        self._flo, self._fhi, self._fbin = 0.94, 1.06, 0.0002
        self._fnb = int(round((self._fhi - self._flo) / self._fbin))
        self._fH = np.zeros(self._fnb, np.float64)
        self._fmass = 0.0
        self.T = None; self.fov_frozen = False
        self._r = np.zeros(NT * NT, np.float64); self._s = np.zeros(NT * NT, np.float64); self._h = np.zeros(NT * NT, np.float64)
        self._G = np.zeros((NS, NS), bool)
        self._Gf = self._G.reshape(-1)
        self._Gc = np.zeros(NC * NC, np.int16)
        self._R = np.full(NC * NC, np.inf, np.float32)
        self._H = np.zeros(NC * NC, np.float32)
        self.tick = None; self.t = 0.0; self._last_map_tick = -1
        self.x = self.y = self.z = self.roll = self.pitch = self.yaw = 0.0; self.speed = 0.0; self.have_pose = False
        self.clue = None; self.start = None; self.t_target = None
        self._path = np.zeros((700, 4), np.float32); self._np = 0; self._last_path_k = -1
        self.n_upd = 0; self.n_frames = 0; self.n_err = 0; self.n_skip = 0; self.last_err = None
        self.cost_ms = []
        self._pub_tick = None; self._pub_arrays = None

    def update(self, tick, obs, state=None, clue=None, start=None):
        t0 = time.perf_counter()
        try:
            with np.errstate(all="ignore"):
                self._update(tick, obs, state, clue, start)
        except Exception as e:
            self.n_err += 1
            self.last_err = repr(e)[:200]
        if self.keep_cost:
            self.cost_ms.append(1000.0 * (time.perf_counter() - t0))

    def _update(self, tick, obs, state, clue, start):
        self.n_upd += 1
        tick = (0 if self.tick is None else self.tick + 1) if tick is None else int(tick)
        if state is None and obs is not None and hasattr(obs, "get"):
            state = obs.get("state")
        if state is None:
            self.n_skip += 1
            return
        st = np.asarray(state, np.float64).reshape(-1)
        if st.size < 9 or not np.all(np.isfinite(st[0:9])):
            self.n_skip += 1
            return
        new_tick = self.tick is None or tick > self.tick
        self.tick = tick if new_tick else self.tick
        self.t = DT * self.tick
        self.x, self.y, self.z = float(st[0]), float(st[1]), float(st[2])
        self.roll, self.pitch, self.yaw = float(st[3]), float(st[4]), float(st[5])
        self.speed = math.hypot(float(st[6]), float(st[7]))
        self.have_pose = True
        if self.start is None:
            if start is not None and np.all(np.isfinite(np.asarray(start, np.float64).reshape(-1)[:3])):
                self.start = tuple(float(v) for v in np.asarray(start, np.float64).reshape(-1)[:3])
            else:
                self.start = (self.x, self.y, self.z)
        if self.clue is None:
            c = np.asarray(clue, np.float64).reshape(-1) if clue is not None else (st[0:2] + st[163:165] if st.size >= 165 else np.zeros(0))
            if c.size >= 2 and np.all(np.isfinite(c[:2])):
                self.clue = (float(c[0]), float(c[1]))
                self.t_target = target_time(self.start, self.clue)
        if not new_tick:
            return
        k = self.tick // 5
        if self.tick % 5 == 0 and k != self._last_path_k:
            if self._np >= len(self._path):
                self._path = np.concatenate([self._path, np.zeros_like(self._path)], axis=0)
            self._path[self._np] = (self.t, self.x, self.y, self.yaw)
            self._np += 1; self._last_path_k = k
        dep = None if obs is None or not hasattr(obs, "get") else obs.get("depth")
        if dep is None:
            return
        dep = np.asarray(dep)
        if dep.size != N * N:
            return
        dep = dep.reshape(N, N)
        pos = st[0:3]; rpy = st[3:6]
        R = self._rot(rpy)
        fwd, up = R[:, 0], R[:, 2]; right = -R[:, 1]
        eye = pos + EYE_F * fwd + EYE_U * up
        if not self.fov_frozen:
            self._fov_add(dep, eye, fwd, right, up)
        if self.T is None or self.tick < self.vote_tick0:
            return
        self._vote(dep, pos, eye, fwd, right, up)
        self.n_frames += 1
        self._last_map_tick = self.tick

    @staticmethod
    def _rot(rpy):
        r, p, y = (float(v) for v in rpy)
        cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
        return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                         [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                         [-sp, cp * sr, cp * cr]])

    def _fov_add(self, dep, eye, fwd, right, up):
        s = self.fov_step
        ze = np.asarray(dep[::s, ::s], np.float64) * DSPAN + DMIN
        a = eye[2] + ze * fwd[2]
        b = ze * (self._Uf * right[2] + self._Vf * up[2])
        m = (ze < 29.4) & (ze > 1.0) & (b < -1.0)
        if m.any():
            tt = (SNOW_Z - a[m]) / b[m]
            q = np.floor((tt - self._flo) / self._fbin)
            q = q[(q >= 0) & (q < self._fnb)].astype(np.int64)
            if q.size:
                self._fH += np.bincount(q, minlength=self._fnb)
                self._fmass += float(q.size)
        T = self._fov_tan()
        if T is not None:
            self.T = T
            if self.tick is not None and self.tick >= self.fov_freeze and self._fmass >= self.fov_freeze_mass:
                self.fov_frozen = True

    def _fov_tan(self, frac=0.3):
        H = self._fH
        if H.sum() < 2000:
            return None
        sm = np.convolve(H, np.ones(5) / 5.0, mode="same")
        top = sm.max()
        i = np.arange(2, len(sm) - 2)
        pk = i[(sm[i] >= frac * top) & (sm[i] >= sm[i - 1]) & (sm[i] >= sm[i + 1])]
        k = int(pk[0]) if pk.size else int(np.argmax(sm))
        lo, hi = max(0, k - 4), min(len(sm), k + 5)
        w = H[lo:hi]
        c = self._flo + self._fbin * (np.arange(lo, hi) + 0.5)
        return float((w * c).sum() / max(w.sum(), 1e-9))

    def fov_deg(self):
        return None if self.T is None else 2.0 * math.degrees(math.atan(self.T))

    def _vote(self, dep, pos, eye, fwd, right, up):
        s = self.step; T = self.T
        ze = np.asarray(dep[::s, ::s], np.float64) * DSPAN + DMIN
        ok = (ze < DMIN + DSPAN - 0.05) & (ze > DMIN + 0.05)
        U = self._U * T; V = self._V * T
        dx = fwd[0] + U * right[0] + V * up[0]
        dy = fwd[1] + U * right[1] + V * up[1]
        dz = fwd[2] + U * right[2] + V * up[2]
        X = eye[0] + ze * dx; Y = eye[1] + ze * dy; Z = eye[2] + ze * dz
        hr2 = (X - pos[0]) ** 2 + (Y - pos[1]) ** 2
        ok &= hr2 < HRANGE * HRANGE
        lanek = ok & (Z >= 0.12) & (Z < 0.24)
        obj = ok & (Z >= 0.24)
        w = self._wf
        rise = np.zeros(Z.shape, bool)
        rise[w:] = ok[:-w] & ((Z[:-w] - Z[w:]) > 0.08)
        foot = rise & lanek
        lanek &= ~foot
        snow = ok & (Z < 0.12) & (Z > 0.03)
        steep = (pos[2] - Z) > SLOPE * np.sqrt(hr2)
        roadv = lanek & steep
        snowv = snow & steep
        fx = (X - GLO) / TILE; fy = (Y - GLO) / TILE
        i = np.floor(fx); j = np.floor(fy)
        e = SUBR / TILE
        ing = (i >= 0) & (i < NT) & (j >= 0) & (j < NT)
        inner = (fx - i > e) & (fx - i < 1 - e) & (fy - j > e) & (fy - j < 1 - e)
        kt = (np.clip(i, 0, NT - 1) * NT + np.clip(j, 0, NT - 1)).astype(np.int64)
        m = ing & inner & roadv
        if m.any():
            self._r += np.bincount(kt[m], minlength=NT * NT)
        m = ing & inner & snowv
        if m.any():
            self._s += np.bincount(kt[m], minlength=NT * NT)
        m = ing & ok & (Z > HOUSE_Z)
        if m.any():
            self._h += np.bincount(kt[m], minlength=NT * NT)
        g = roadv | snowv
        if g.any():
            gx = X[g]; gy = Y[g]
            a4 = np.floor((gx - GLO) / SUBR).astype(np.int64); b4 = np.floor((gy - GLO) / SUBR).astype(np.int64)
            inb = (a4 >= 0) & (a4 < NS) & (b4 >= 0) & (b4 < NS)
            if inb.any():
                a4 = a4[inb]; b4 = b4[inb]
                sl = a4 * NS + b4
                new = ~self._Gf[sl]
                if new.any():
                    u = np.unique(sl[new])
                    self._Gf[u] = True
                    self._Gc += np.bincount((u // NS // 4) * NC + (u % NS) // 4, minlength=NC * NC).astype(np.int16)
                rng = (ze[g] * np.sqrt(dx[g] ** 2 + dy[g] ** 2 + dz[g] ** 2))[inb].astype(np.float32)
                np.minimum.at(self._R, (a4 // 4) * NC + (b4 // 4), rng)
        if obj.any():
            ox = np.floor(X[obj] - GLO).astype(np.int64); oy = np.floor(Y[obj] - GLO).astype(np.int64)
            inb = (ox >= 0) & (ox < NC) & (oy >= 0) & (oy < NC)
            if inb.any():
                np.maximum.at(self._H, ox[inb] * NC + oy[inb], Z[obj][inb].astype(np.float32))

    def tiles(self):
        tot = self._r + self._s
        frac = np.where(tot > 0, self._r / np.maximum(tot, 1.0), 0.0)
        out = np.zeros(NT * NT, np.int8)
        known = tot >= self.min_votes
        out[known & (frac >= self.thr)] = 1
        out[known & (frac < self.thr)] = -1
        out[self._h >= np.maximum(self.house_min, self.house_frac * self._r)] = -1
        out = out.reshape(NT, NT)
        if self.ring:
            out[0, :] = 1; out[NT - 1, :] = 1; out[:, 0] = 1; out[:, NT - 1] = 1
        return out

    def ground_seen(self):
        return self._Gc.reshape(NC, NC) >= self.sub_min

    def ground_range(self, seen=None):
        seen = self.ground_seen() if seen is None else seen
        return np.where(seen, self._R.reshape(NC, NC), np.inf).astype(np.float32)

    def obst_h(self):
        h = self._H.reshape(NC, NC)
        return np.where(h > 0.5, h, 0.0).astype(np.float32)

    def path(self):
        return self._path[:self._np].copy()

    def _arrays(self):
        if self._pub_tick != self._last_map_tick or self._pub_arrays is None:
            gs = self.ground_seen()
            self._pub_arrays = (_ro(self.tiles()), _ro(gs), _ro(self.ground_range(gs)), _ro(self.obst_h()))
            self._pub_tick = self._last_map_tick
        return self._pub_arrays

    def public(self, t_takeover=None, fed=None, z=None):
        tiles, gs, gr, oh = self._arrays()
        t = float(self.t)
        tt = t if t_takeover is None else float(t_takeover)
        clue = self.clue if self.clue is not None else (float("nan"), float("nan"))
        start = self.start if self.start is not None else (self.x, self.y, self.z)
        return LivePublic(t=t, t_search=t - tt, t_takeover=tt, t_target=(self.t_target if self.t_target is not None else float("nan")),
                          t_last=T_LAST, x=float(self.x), y=float(self.y), z=float(self.z if z is None else z), yaw=float(self.yaw),
                          speed=float(self.speed), clue=clue, start=start, fov_deg=self.fov_deg(), tiles=tiles, ground_seen=gs,
                          ground_range=gr, obst_h=oh, path=_ro(self.path()), fed=fed, facts=GENERATOR_FACTS)

    def snapshot(self):
        gs = self.ground_seen()
        return dict(tick=-1 if self.tick is None else int(self.tick), tiles=self.tiles(), ground_seen=gs, ground_range=self.ground_range(gs),
                    obst_h=self.obst_h(), pose=np.array([self.t, self.x, self.y, self.z, self.roll, self.pitch, self.yaw, self.speed], np.float64),
                    fov=np.float64(self.fov_deg() if self.T is not None else np.nan), tvote=np.array([self._r, self._s, self._h]).astype(np.float32),
                    gsub=np.packbits(self._G.reshape(-1)))

    def nbytes(self):
        return int(sum(a.nbytes for a in (self._r, self._s, self._h, self._G, self._Gc, self._R, self._H, self._path, self._fH, self._U, self._V, self._Uf, self._Vf)))
