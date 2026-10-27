import math

import numpy as np

FALLBACK = "FALLBACK"
STOCK = "STOCK"
TILE, GLO, NT, NC = 6.0, -50.0, 17, 102
DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def tile_centre(t):
    return GLO + TILE * (t[0] + 0.5), GLO + TILE * (t[1] + 0.5)


class RoadPolicy3:
    CONST = dict(
        a_min=6.0,
        reach=3.5,
        skip_r=4.0,
        skip_cross=12.0,
        skip_bound=8.0,
        skip_frac=0.8,
        z_high=8.5,
        z_low=4.5,
        near=9.0,
        lim_low=3.3,
        half=2.0,
        ahead=4.0,
        clear_up=1.0,
        feed=8.0,
        cross_first=False,
        merge=True,
        coverage=True,
        look=18.0,
        v_r=10.0,
        v_d0=8.0,
        v_min=20.0,
        v_check=0.5,
        stock_first_s=0.0,
    )

    def __init__(self, **const):
        self.c = dict(self.CONST)
        for k, v in const.items():
            if k not in self.c:
                raise TypeError(f"unknown constant {k}")
            self.c[k] = v
        self._clear()

    def _clear(self):
        self.mode = "new"; self.decisions = []; self.start_info = None; self.error = None
        self.leg_kind = None; self.turn = None; self.spin = None; self._ready = False
        self.stats = {"hops": 0, "joins": 0, "given_up": 0, "rejoined": 0, "back_forbidden": 0, "rule8": 0, "replans": 0,
                      "second_round_t": None, "spins": 0, "spin_s": 0.0, "spin_skipped": 0, "wps": 0, "wp_cross": 0, "wp_bound": 0,
                      "wp_reached": 0, "wp_skipped": 0, "low_s": 0.0}
        self.done = set(); self.cur = None; self.cur_kind = None; self.z_target = None
        self._tkey = None; self._wps = {}; self.S = None; self.order = []; self._K = None; self.pub = None; self.hold = False
        self._mem = {}; self.A1 = None; self._vt = -1.0; self._U = None; self._Ut = None

    def reset(self, pub):
        self._clear()

    def _prepare(self, pub):
        cx, cy = pub.clue
        g = GLO + (np.arange(NC) + 0.5)
        X, Y = np.meshgrid(g, g, indexing="ij")
        area = (np.hypot(X - cx, Y - cy) <= 30.0) & (np.abs(X) <= 25.0) & (np.abs(Y) <= 25.0)
        self.S = area.reshape(NT, 6, NT, 6).sum(axis=(1, 3)) >= self.c["a_min"]
        self.A1 = area
        self._cx, self._cy = g, g

    def tables(self, pub):
        T = np.asarray(pub.tiles)
        key = T.tobytes()
        if key == self._tkey:
            return
        self._tkey = key
        K = (T == 1); S = self.S; W = {}
        for i in range(NT):
            for j in range(NT):
                if not (K[i, j] and S[i, j]):
                    continue
                nb = 0; out = False
                for dx, dy in DIRS:
                    a, b = i + dx, j + dy
                    if 0 <= a < NT and 0 <= b < NT and K[a, b]:
                        nb += 1
                        if not S[a, b]:
                            out = True
                if out:
                    W[(i, j)] = "boundary"
                elif nb >= 3:
                    W[(i, j)] = "crossroads"
        mem = {w: [w] for w in W}
        if self.c["merge"]:
            W2, mem, seen = {}, {}, set()
            for w0, kind in W.items():
                if w0 in seen:
                    continue
                grp, stack = [], [w0]; seen.add(w0)
                while stack:
                    q = stack.pop(); grp.append(q)
                    for dx, dy in DIRS:
                        n_ = (q[0] + dx, q[1] + dy)
                        if n_ not in seen and W.get(n_) == kind:
                            seen.add(n_); stack.append(n_)
                mx = sum(q[0] for q in grp) / len(grp); my = sum(q[1] for q in grp) / len(grp)
                rep = min(sorted(grp), key=lambda q: (q[0] - mx) ** 2 + (q[1] - my) ** 2)
                W2[rep] = kind; mem[rep] = grp
            W = W2
        self._wps = W; self._K = K; self._mem = mem
        self.stats["wps"] = len(W)
        self.stats["wp_cross"] = sum(1 for v in W.values() if v == "crossroads")
        self.stats["wp_bound"] = sum(1 for v in W.values() if v == "boundary")

    def _seen_around(self, pub, t):
        x, y = tile_centre(t); r = self.c["skip_r"]
        i0, i1 = int(max(0, math.floor(x - r - GLO))), int(min(NC, math.ceil(x + r - GLO)))
        j0, j1 = int(max(0, math.floor(y - r - GLO))), int(min(NC, math.ceil(y + r - GLO)))
        R = np.asarray(pub.ground_range)[i0:i1, j0:j1]; H = np.asarray(pub.obst_h)[i0:i1, j0:j1]
        gx = GLO + np.arange(i0, i1) + 0.5; gy = GLO + np.arange(j0, j1) + 0.5
        inside = (np.hypot(gx[:, None] - x, gy[None, :] - y) <= r) & (H < 0.5)
        n = int(inside.sum())
        if n == 0:
            return True
        rng = self.c["skip_bound"] if self._wps.get(t) == "boundary" else self.c["skip_cross"]
        return float(((R <= rng) & inside).sum()) / n >= self.c["skip_frac"]

    def _is_done(self, w):
        return w in self.done or any(m in self.done for m in self._mem.get(w, ()))

    def _set_done(self, w):
        self.done.add(w); self.done.update(self._mem.get(w, ()))

    def mark(self, t, x, y, yaw, z, pub=None):
        for w in self._wps:
            if not self._is_done(w):
                for m in self._mem.get(w, (w,)):
                    wx, wy = tile_centre(m)
                    if math.hypot(wx - x, wy - y) <= self.c["reach"]:
                        self._set_done(w); self.stats["wp_reached"] += 1
                        if w == self.cur:
                            self.cur = None
                        break

    def _unlooked(self, pub):
        if self._Ut != pub.t:
            self._U = self.A1 & (np.asarray(pub.ground_range) > self.c["look"]) & (np.asarray(pub.obst_h) < 0.5)
            self._Ut = pub.t
        return self._U

    def value(self, pub, w):
        U = self._unlooked(pub); x, y = tile_centre(w); r = self.c["v_r"]
        i0, i1 = int(max(0, math.floor(x - r - GLO))), int(min(NC, math.ceil(x + r - GLO)))
        j0, j1 = int(max(0, math.floor(y - r - GLO))), int(min(NC, math.ceil(y + r - GLO)))
        gx = GLO + np.arange(i0, i1) + 0.5; gy = GLO + np.arange(j0, j1) + 0.5
        disc = np.hypot(gx[:, None] - x, gy[None, :] - y) <= r
        return float((U[i0:i1, j0:j1] & disc).sum())

    def _clear_line(self, pub, a, b, lim):
        c = self.c
        L = math.hypot(b[0] - a[0], b[1] - a[1]); n = max(2, int(L / 0.5) + 1)
        nx, ny = ((a[1] - b[1]) / L, (b[0] - a[0]) / L) if L > 1e-6 else (1.0, 0.0)
        H = np.asarray(pub.obst_h); R = np.asarray(pub.ground_range); K = self._K
        for k in range(n):
            px = a[0] + (b[0] - a[0]) * k / (n - 1); py = a[1] + (b[1] - a[1]) * k / (n - 1)
            for s_ in (-c["half"], -0.5 * c["half"], 0.0, 0.5 * c["half"], c["half"]):
                qx, qy = px + s_ * nx, py + s_ * ny
                ix, iy = int(math.floor(qx - GLO)), int(math.floor(qy - GLO))
                if not (0 <= ix < NC and 0 <= iy < NC):
                    return False
                if H[ix, iy] > lim:
                    return False
                ti, tj = int((qx - GLO) // TILE), int((qy - GLO) // TILE)
                if H[ix, iy] <= 0.0 and not np.isfinite(R[ix, iy]) and not (0 <= ti < NT and 0 <= tj < NT and K[ti, tj]):
                    return False
        return True

    def ahead_clear(self, x, y, z, fx, fy):
        pub = self.pub
        if pub is None or self._K is None:
            return False
        d = math.hypot(fx - x, fy - y)
        if d < 1e-3:
            return True
        k = min(1.0, self.c["ahead"] / d)
        return self._clear_line(pub, (x, y), (x + k * (fx - x), y + k * (fy - y)), z - self.c["clear_up"])

    def next(self, pub):
        if self.mode == "back":
            return FALLBACK
        try:
            return self._next(pub)
        except Exception as e:
            self.error = repr(e)[:200]; self.mode = "back"
            self.decisions.append({"t": round(float(pub.t), 2), "rule": 6, "what": "hand-back", "why": "error: " + self.error})
            return FALLBACK

    def _next(self, pub):
        c = self.c
        if self.mode == "new":
            self._prepare(pub)
            if c["stock_first_s"] > 0.0:
                self.mode = "waiting"
            else:
                self._start(pub, "take-over")
        if self.mode == "waiting":
            if pub.t_search < c["stock_first_s"] - 1e-6:
                return STOCK
            self._start(pub, "stock first")
        self.pub = pub
        self.tables(pub)
        self.mark(pub.t, pub.x, pub.y, pub.yaw, pub.z)
        if self.cur is not None and self.cur not in self._wps:
            self.cur = None
        if c["coverage"] and self.cur is not None and pub.t - self._vt >= c["v_check"] - 1e-9:
            self._vt = float(pub.t)
            if self.value(pub, self.cur) < c["v_min"]:
                self._set_done(self.cur); self.stats["wp_skipped"] += 1; self.cur = None
        if self.cur is None:
            p = (float(pub.x), float(pub.y)); best = None
            for w, kind in self._wps.items():
                if self._is_done(w):
                    continue
                wx, wy = tile_centre(w)
                d = math.hypot(wx - p[0], wy - p[1])
                if c["coverage"]:
                    v = self.value(pub, w)
                    if v < c["v_min"]:
                        self._set_done(w); self.stats["wp_skipped"] += 1
                        continue
                    key = (-v / (d + c["v_d0"]),)
                else:
                    if self._seen_around(pub, w):
                        self._set_done(w); self.stats["wp_skipped"] += 1
                        continue
                    rank = (0 if kind == "crossroads" else 1) if c["cross_first"] else 0
                    key = (rank, d)
                if best is None or key < best[3]:
                    best = (d, w, kind, key)
            self._vt = float(pub.t)
            if best is None:
                if float(pub.z) < c["z_high"] - 0.3:
                    self.z_target = c["z_high"]; self.leg_kind = "climb-out"
                    return float(pub.x), float(pub.y)
                self.decisions.append({"t": round(float(pub.t), 2), "rule": 6, "what": "hand-back", "why": "every waypoint done"})
                self.mode = "back"; self.z_target = None
                return FALLBACK
            self.cur, self.cur_kind = best[1], best[2]
            self.order.append([round(float(pub.t), 2), int(best[1][0]), int(best[1][1]), best[2]])
            self.stats["replans"] += 1
            self.decisions.append({"t": round(float(pub.t), 2), "rule": 3, "what": "waypoint", "why": "%s at %s, %.1f m" % (best[2], str(best[1]), best[0])})
        wx, wy = tile_centre(self.cur)
        d = math.hypot(wx - pub.x, wy - pub.y)
        low = (self.cur_kind == "boundary" and d <= c["near"] and self._clear_line(pub, (float(pub.x), float(pub.y)), (wx, wy), c["lim_low"]))
        self.z_target = c["z_low"] if low else c["z_high"]
        if low:
            self.stats["low_s"] = round(self.stats["low_s"] + 0.02, 2)
        self.leg_kind = "tour"
        a = math.atan2(wy - pub.y, wx - pub.x) if d > 1e-3 else float(pub.yaw)
        f = min(max(d, 3.5), c["feed"])
        return float(pub.x + f * math.cos(a)), float(pub.y + f * math.sin(a))

    def _start(self, pub, how):
        self.mode = "road"; self._ready = True
        self.start_info = {"how": how, "t": round(float(pub.t), 2), "x": round(float(pub.x), 2), "y": round(float(pub.y), 2), "z": round(float(pub.z), 2)}
