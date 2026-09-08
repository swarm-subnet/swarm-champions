"""Map-aware pilot layer that post-processes the frozen actor's RC sticks.

The actor was trained while its visual localiser was broken, so it never learned
to use an accurate absolute pose.  It now has one (median 0.24 m) and a perfect
baked map, yet it still flies the raw target bearing into the internal
partition, never decelerates on the terminal approach, and hovers-and-spins
whenever the track goes stale.  This layer supplies the four behaviours the
actor lacks, all as an action-space override:

  * takeoff guard        -- bound the sticks while the drone is still climbing
  * route planner        -- BFS field on the baked grid whenever the direct line
                            to the target is blocked, or the track is stale
  * TTC clearance shield -- cap the speed into whatever the map says is ahead
  * terminal discipline  -- stop yawing and slow the closure in the endgame

Deterministic (no RNG anywhere) and fully re-initialised in reset(), because the
evaluator reuses one agent object for every episode a worker runs.
"""

from __future__ import annotations

import math
import os

import numpy as np

DT = 1.0 / 50.0
SPEED = 3.0
SLEW_STEP = 4.0 * DT
DEAD_ZONE = 0.05


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _i(name: str, default: int) -> int:
    return int(_f(name, default))


class Cfg:
    def __init__(self) -> None:
        self.on = _i("OP_ON", 1)

        self.to_on = _i("OP_TO_ON", 0)
        self.to_t = _f("OP_TO_T", 1.2)
        self.to_z = _f("OP_TO_Z", 1.10)
        self.to_yaw = _f("OP_TO_YAW", 0.35)
        self.to_lat = _f("OP_TO_LAT", 0.35)

        self.sh_on = _i("OP_SH_ON", 1)
        self.sh_ttc = _f("OP_SH_TTC", 0.70)
        self.sh_infl = _f("OP_SH_INFL", 0.30)
        self.sh_margin = _f("OP_SH_MARGIN", 0.12)
        self.sh_look = _f("OP_SH_LOOK", 3.0)
        self.sh_steer = _i("OP_SH_STEER", 1)
        self.sh_fan = _f("OP_SH_FAN", 75.0)
        self.sh_minr = _f("OP_SH_MINR", 0.0)
        # The pilot's own acceptance gate on the visual posterior.  Loosening it
        # only changes what the SHIELD sees -- the actor keeps its own 0.35 gate
        # -- and it is the only way to give the shield reach in the first ~1.6 s,
        # where 34% of the residual failures happen.
        self.sh_spread = _f("OP_SH_SPREAD", 0.35)
        self.sh_zmin = _f("OP_SH_ZMIN", 0.80)

        self.pl_on = _i("OP_PL_ON", 0)
        self.pl_cell = _f("OP_PL_CELL", 0.25)
        self.pl_infl = _f("OP_PL_INFL", 0.38)
        self.pl_los = _f("OP_PL_LOS", 0.32)
        self.pl_block = _f("OP_PL_BLOCK", 0.25)
        self.pl_v = _f("OP_PL_V", 2.7)
        self.pl_minr = _f("OP_PL_MINR", 1.2)
        self.pl_zlo = _f("OP_PL_ZLO", 1.10)
        self.pl_zhi = _f("OP_PL_ZHI", 2.45)
        self.pl_every = _i("OP_PL_EVERY", 4)

        self.se_on = _i("OP_SE_ON", 0)
        self.se_t0 = _f("OP_SE_T0", 2.0)
        self.se_v = _f("OP_SE_V", 2.5)
        self.se_yaw = _f("OP_SE_YAW", 0.0)
        self.se_reach = _f("OP_SE_REACH", 2.0)
        self.se_lost = _f("OP_SE_LOST", 1.2)
        self.pl_stale = _f("OP_PL_STALE", 0.6)
        self.pl_hyst = _i("OP_PL_HYST", 10)
        self.pl_reacq = _f("OP_PL_REACQ", 0.6)

        self.tm_on = _i("OP_TM_ON", 0)
        self.tm_r = _f("OP_TM_R", 1.2)
        self.tm_exit = _f("OP_TM_EXIT", 1.8)
        self.tm_maxage = _f("OP_TM_MAXAGE", 1.2)
        self.tm_yaw = _f("OP_TM_YAW", 0.25)
        self.tm_yawmode = _i("OP_TM_YAWMODE", 1)
        self.tm_kyaw = _f("OP_TM_KYAW", 2.0)
        self.tm_aim = _i("OP_TM_AIM", 1)
        self.tm_match = _i("OP_TM_MATCH", 1)
        self.tm_kc = _f("OP_TM_KC", 2.5)
        self.tm_vcmin = _f("OP_TM_VCMIN", 1.6)
        self.tm_vcmax = _f("OP_TM_VCMAX", 2.6)
        self.tm_vmax = _f("OP_TM_VMAX", 3.0)
        self.tm_lead = _f("OP_TM_LEAD", 0.0)
        self.tm_freeze = _f("OP_TM_FREEZE", 0.30)
        self.tm_vz = _i("OP_TM_VZ", 0)
        self.tm_kz = _f("OP_TM_KZ", 2.0)

        self.ve_on = _i("OP_VE_ON", 0)
        self.ve_max = _f("OP_VE_MAX", 3.5)
        self.ve_t0 = _f("OP_VE_T0", 0.0)

        # ---- terminal ELEVATION guard (vertical stick only) -----------------
        # A level catch cannot register by radius: two Tello hulls (r=0.088)
        # touch at 0.178 m centre distance, outside the 0.15 m kill sphere, so a
        # level intercept depends entirely on a contact manifold surviving a
        # 20 ms control step.  Sitting 0.09-0.12 m ABOVE the target removes the
        # hull block (half-heights sum to 0.042 m) and lets the centre distance
        # fall inside 0.15 m, which registers instantly and with no impulse.
        self.eg_on = _i("OP_EG_ON", 0)
        self.eg_r = _f("OP_EG_R", 1.0)      # arm inside this range
        self.eg_dz = _f("OP_EG_DZ", 0.11)   # desired height above the target
        self.eg_k = _f("OP_EG_K", 8.0)      # proportional gain on the z error
        self.eg_lim = _f("OP_EG_LIM", 0.60)  # max |vertical stick|
        self.eg_age = _f("OP_EG_AGE", 0.4)  # only on a fresh track
        self.eg_up = _i("OP_EG_UP", 1)      # 1 = never command below the actor
        self.eg_zmax = _f("OP_EG_ZMAX", 2.80)
        self.eg_taper = _f("OP_EG_TAPER", 0.0)  # >0: dz ramps in over this range
        self.eg_hold = _f("OP_EG_HOLD", 0.0)    # keep acting for this long after
                                                # the track goes stale

        # ---- pure terminal SPEED cap (direction and yaw untouched) -----------
        self.tc_on = _i("OP_TC_ON", 0)
        self.tc_r = _f("OP_TC_R", 1.0)
        self.tc_v = _f("OP_TC_V", 3.0)
        self.tc_age = _f("OP_TC_AGE", 0.4)

        # ---- STALL breaker ---------------------------------------------------
        # 0.75% of episodes spend almost the whole 60 s with the sticks near
        # centre (the actor hovers and spins whenever the track goes stale) and
        # they carry 11% of all lost score.  This fires ONLY after a long
        # motionless stretch, far more rarely than the (harmful) always-on
        # search sweep.
        self.sb_on = _i("OP_SB_ON", 1)
        self.sb_v = _f("OP_SB_V", 0.30)     # "motionless" speed threshold m/s
        self.sb_t = _f("OP_SB_T", 2.0)      # seconds of stall before firing
        self.sb_minr = _f("OP_SB_MINR", 1.2)  # never inside this target range
        self.sb_age = _f("OP_SB_AGE", 0.6)  # a fresher track than this = leave alone
        self.sb_speed = _f("OP_SB_SPEED", 2.0)
        self.sb_look = _f("OP_SB_LOOK", 3.0)
        self.sb_infl = _f("OP_SB_INFL", 0.30)
        self.sb_rec = _f("OP_SB_REC", 0.6)  # seconds of recovered motion to release
        self.sb_yaw = _f("OP_SB_YAW", 0.0)  # >0: clamp |yaw stick| while active
        self.sb_t0 = _f("OP_SB_T0", 3.0)    # never arm before this time

        self.blend = _f("OP_BLEND", 1.0)
        self.diag_on = _i("OP_DIAG", 0)


# Coverage ring used when the track is stale: main hall first (both drones are
# usually placed there), then the two pockets the actor otherwise never leaves.
_SWEEP = (
    (9.00, 2.60), (14.60, 2.60), (4.20, 2.60), (16.20, 6.30),
    (9.00, 2.60), (1.60, 6.30), (12.00, 2.60), (6.00, 2.60),
)

_NB = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
_BIG = np.int32(1 << 20)


class _Field:
    """8-connected BFS distance field over a coarsened free-space mask."""

    __slots__ = ("free", "cell", "ox", "oy", "nx", "ny", "dist", "goal",
                 "step_i", "step_j")

    def __init__(self, grid, cell: float, infl: float, z_lo: float, z_hi: float):
        clear_cm = grid.clear_cm
        res = float(grid.res)
        oz = float(grid.origin[2])
        k_lo = max(0, int(math.floor((z_lo - oz) / res)))
        k_hi = min(clear_cm.shape[0] - 1, int(math.ceil((z_hi - oz) / res)))
        band = clear_cm[k_lo:k_hi + 1].min(axis=0)
        step = max(1, int(round(cell / res)))
        ny = band.shape[0] // step
        nx = band.shape[1] // step
        blk = band[:ny * step, :nx * step].reshape(ny, step, nx, step)
        self.free = (blk.min(axis=(1, 3)).astype(np.float64) * 0.01) > infl
        self.cell = step * res
        self.ox = float(grid.origin[0])
        self.oy = float(grid.origin[1])
        self.nx, self.ny = nx, ny
        self.dist = None
        self.goal = None
        self.step_i = None
        self.step_j = None

    def cell_of(self, x: float, y: float):
        i = int(math.floor((x - self.ox) / self.cell))
        j = int(math.floor((y - self.oy) / self.cell))
        return (min(max(i, 0), self.nx - 1), min(max(j, 0), self.ny - 1))

    def centre(self, i: int, j: int):
        return (self.ox + (i + 0.5) * self.cell, self.oy + (j + 0.5) * self.cell)

    def nearest_free(self, ij, radius: int = 8):
        i, j = ij
        if self.free[j, i]:
            return (i, j)
        for r in range(1, radius + 1):
            i0, i1 = max(0, i - r), min(self.nx - 1, i + r)
            j0, j1 = max(0, j - r), min(self.ny - 1, j + r)
            sub = self.free[j0:j1 + 1, i0:i1 + 1]
            if not sub.any():
                continue
            jj, ii = np.nonzero(sub)
            d = (ii + i0 - i) ** 2 + (jj + j0 - j) ** 2
            k = int(np.argmin(d))
            return (int(ii[k]) + i0, int(jj[k]) + j0)
        return None

    def build(self, goal_ij) -> bool:
        gi, gj = goal_ij
        if not self.free[gj, gi]:
            return False
        dist = np.full(self.free.shape, _BIG, dtype=np.int32)
        dist[gj, gi] = 0
        cur = np.zeros(self.free.shape, dtype=bool)
        cur[gj, gi] = True
        free = self.free
        d = 0
        for _ in range(self.nx + self.ny + 8):
            nxt = np.zeros_like(cur)
            nxt[1:, :] |= cur[:-1, :]
            nxt[:-1, :] |= cur[1:, :]
            nxt[:, 1:] |= cur[:, :-1]
            nxt[:, :-1] |= cur[:, 1:]
            nxt[1:, 1:] |= cur[:-1, :-1]
            nxt[1:, :-1] |= cur[:-1, 1:]
            nxt[:-1, 1:] |= cur[1:, :-1]
            nxt[:-1, :-1] |= cur[1:, 1:]
            nxt &= free
            nxt &= dist == _BIG
            if not nxt.any():
                break
            d += 1
            dist[nxt] = d
            cur = nxt
        # Precompute the downhill neighbour once so descend() is a plain lookup.
        pad = np.full((self.ny + 2, self.nx + 2), _BIG, dtype=np.int32)
        pad[1:-1, 1:-1] = dist
        stack = np.empty((8, self.ny, self.nx), dtype=np.int32)
        for k, (dj, di) in enumerate(_NB):
            stack[k] = pad[1 + dj:1 + dj + self.ny, 1 + di:1 + di + self.nx]
        best = np.argmin(stack, axis=0).astype(np.int8)
        self.step_i = np.array([d[1] for d in _NB], dtype=np.int8)[best]
        self.step_j = np.array([d[0] for d in _NB], dtype=np.int8)[best]
        self.dist = dist
        self.goal = (gi, gj)
        return True

    def descend(self, start_ij, max_steps: int = 130):
        if self.dist is None:
            return []
        i, j = start_ij
        if self.dist[j, i] >= _BIG:
            return []
        out = []
        for _ in range(max_steps):
            if self.dist[j, i] == 0:
                break
            ii = i + int(self.step_i[j, i])
            jj = j + int(self.step_j[j, i])
            if ii == i and jj == j:
                break
            if not (0 <= ii < self.nx and 0 <= jj < self.ny):
                break
            if self.dist[jj, ii] >= self.dist[j, i]:
                break
            i, j = ii, jj
            out.append((i, j))
        return out


class Pilot:
    def __init__(self, grid) -> None:
        self.cfg = Cfg()
        self.grid = grid
        c = self.cfg
        self.field = (_Field(grid, c.pl_cell, c.pl_infl, c.pl_zlo, c.pl_zhi)
                      if (c.on and (c.pl_on or c.se_on)) else None)
        self._fan = np.radians(np.array(
            [15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90], dtype=np.float64))
        self.reset()

    def reset(self) -> None:
        self._plan_age = 10 ** 6
        self._sweep_i = 0
        self._hold = 0
        self._term = False
        self._u_last = None
        self._eg_t = -1.0
        self._eg_dz = None
        self._stall = 0
        self._sb_active = False
        self._sb_ok = 0
        if self.field is not None:
            self.field.dist = None
            self.field.goal = None
            self.field.step_i = None
            self.field.step_j = None
        self.diag = {}

    @staticmethod
    def _sticks(vx: float, vy: float, yaw: float):
        c, s = math.cos(yaw), math.sin(yaw)
        return (s * vx - c * vy) / SPEED, (c * vx + s * vy) / SPEED

    @staticmethod
    def _vel(lr: float, fb: float, yaw: float):
        c, s = math.cos(yaw), math.sin(yaw)
        return SPEED * (fb * c + lr * s), SPEED * (fb * s - lr * c)

    def _fan_free(self, p3, dirs, look: float, infl: float):
        """Free distance along each of `dirs` (k,2), one batched grid query."""
        n = max(6, int(look / 0.06))
        ts = np.linspace(0.06, look, n)
        k = dirs.shape[0]
        pts = np.empty((k, n, 3))
        pts[:, :, 0] = p3[0] + dirs[:, 0:1] * ts[None, :]
        pts[:, :, 1] = p3[1] + dirs[:, 1:2] * ts[None, :]
        pts[:, :, 2] = p3[2]
        cl = self.grid.clearance(pts.reshape(-1, 3)).reshape(k, n)
        blocked = cl <= infl
        idx = np.argmax(blocked, axis=1)
        any_b = blocked.any(axis=1)
        return np.where(any_b, ts[idx], look)

    def step(self, a, fb):
        c = self.cfg
        if not c.on:
            return a
        a = np.asarray(a, dtype=np.float64).reshape(-1)[:4].copy()
        ego, track = fb.ego, fb.track
        t = float(ego.t)
        z = float(ego.pos[2])

        est = None
        th = 0.0
        vis = fb.visual_last
        if vis is not None and getattr(vis, "posterior", None) is not None:
            po = vis.posterior
            try:
                ok = (po.finite and np.isfinite(po.xy).all()
                      and math.isfinite(float(po.mode_spread_m))
                      and float(po.mode_spread_m) <= c.sh_spread)
            except Exception:
                ok = False
            if ok:
                est = np.asarray(po.xy, dtype=np.float64)
                th = float(po.heading_offset)
        if est is None:
            n_est, _ = fb.loc.stats_at(ego.pos[:2])
            if fb.loc.seeded and fb.loc.converged and fb.loc.theta_conf() > 0.5:
                est = np.asarray(n_est, dtype=np.float64)
                th = float(fb.loc.theta())

        cth, sth = math.cos(th), math.sin(th)

        def to_map(vx, vy):
            return (cth * vx - sth * vy, sth * vx + cth * vy)

        def to_ego(vx, vy):
            return (cth * vx + sth * vy, -sth * vx + cth * vy)

        rng = None
        if track.rel is not None:
            rng = float(np.linalg.norm(track.rel))

        override = None
        mode = "none"
        pose_ok = est is not None and z > c.sh_zmin

        # ---- mode selection ------------------------------------------------
        age = float(track.age)
        have = track.rel is not None

        # terminal latch: once inside the endgame, commit through blind frames
        # (the detector goes dark in the last half metre and the actor's own
        # reaction is to stop and spin, which is how the near-misses happen).
        if c.tm_on and have and rng is not None:
            if not self._term:
                if rng < c.tm_r and age <= 0.5:
                    self._term = True
                    self._u_last = None
            else:
                if rng > c.tm_exit or age > c.tm_maxage:
                    self._term = False
                    self._u_last = None
        else:
            self._term = False

        # ---- route planner / search sweep -----------------------------------
        if pose_ok and self.field is not None and not self._term:
            p3 = np.array([est[0], est[1], z])
            goal_xy = None
            if have and age <= c.pl_stale and c.pl_on and rng > c.pl_minr:
                tm = to_map(float(track.rel[0]), float(track.rel[1]))
                gx, gy = est[0] + tm[0], est[1] + tm[1]
                gz = min(max(z + float(track.rel[2]), c.pl_zlo), c.pl_zhi)
                blocked = self.grid.path_clearance(
                    p3, np.array([gx, gy, gz])) < c.pl_block
                if blocked:
                    self._hold = c.pl_hyst
                elif self._hold > 0:
                    self._hold -= 1
                if blocked or self._hold > 0:
                    goal_xy = (gx, gy)
                    mode = "plan"
            elif have and age <= c.se_lost and c.pl_on and rng > c.pl_reacq:
                tm = to_map(float(track.rel[0]), float(track.rel[1]))
                goal_xy = (est[0] + tm[0], est[1] + tm[1])
                mode = "reacq"
            elif c.se_on and t > c.se_t0 and (not have or age > c.se_lost):
                gx, gy = _SWEEP[self._sweep_i % len(_SWEEP)]
                if math.hypot(est[0] - gx, est[1] - gy) < c.se_reach:
                    self._sweep_i += 1
                    gx, gy = _SWEEP[self._sweep_i % len(_SWEEP)]
                goal_xy = (gx, gy)
                mode = "search"

            if goal_xy is not None:
                gij = self.field.nearest_free(self.field.cell_of(*goal_xy))
                sij = self.field.nearest_free(self.field.cell_of(est[0], est[1]))
                if gij is not None and sij is not None:
                    if gij != self.field.goal or self._plan_age >= c.pl_every:
                        if self.field.build(gij):
                            self._plan_age = 0
                        else:
                            self._plan_age += 1
                    else:
                        self._plan_age += 1
                    cells = self.field.descend(sij)
                    if cells:
                        if len(cells) > 48:
                            k = max(1, len(cells) // 48)
                            cells = cells[::k]
                        pts = np.array([self.field.centre(i, j) for i, j in cells])
                        ends = np.column_stack([pts, np.full(pts.shape[0], z)])
                        starts = np.repeat(p3[None, :], ends.shape[0], axis=0)
                        clr = np.atleast_1d(self.grid.path_clearance(starts, ends))
                        vis_ok = np.nonzero(clr > c.pl_los)[0]
                        k = int(vis_ok[-1]) if vis_ok.size else 0
                        dx = float(pts[k, 0] - est[0])
                        dy = float(pts[k, 1] - est[1])
                        n = math.hypot(dx, dy)
                        if n > 1e-6:
                            v = c.se_v if mode == "search" else c.pl_v
                            override = to_ego(dx / n * v, dy / n * v)
                        else:
                            mode = "none"
                    else:
                        mode = "none"
                else:
                    mode = "none"

        # ---- terminal discipline ---------------------------------------------
        if self._term:
            mode = "term"
            rel = np.asarray(track.rel, dtype=np.float64)
            vt = np.asarray(track.vt, dtype=np.float64)
            aim = rel + vt * c.tm_lead
            n = math.hypot(aim[0], aim[1]) if c.tm_aim else 0.0
            if n > c.tm_freeze:
                ux, uy = aim[0] / n, aim[1] / n
                self._u_last = (ux, uy)
            elif c.tm_aim and self._u_last is not None:
                ux, uy = self._u_last
            elif c.tm_aim and n > 1e-6:
                ux, uy = aim[0] / n, aim[1] / n
            else:
                ux = uy = None
            if ux is not None:
                # Match the target's own velocity and add a bounded closure rate
                # on top: a plain ground-speed cap loses a target that flees at
                # up to 1.95 m/s, and stopping short is how the near-misses die.
                vc = min(max(c.tm_kc * rng, c.tm_vcmin), c.tm_vcmax)
                vx_d, vy_d = ux * vc, uy * vc
                if c.tm_match:
                    vx_d += vt[0]
                    vy_d += vt[1]
                sp = math.hypot(vx_d, vy_d)
                if sp > c.tm_vmax and sp > 1e-6:
                    k = c.tm_vmax / sp
                    vx_d, vy_d = vx_d * k, vy_d * k
                override = (vx_d, vy_d)
            if c.tm_vz:
                a[2] = float(np.clip(
                    (rel[2] + vt[2] * c.tm_lead) * c.tm_kz / SPEED, -1.0, 1.0))
            if c.tm_yawmode:
                az = math.atan2(
                    -math.sin(ego.yaw) * rel[0] + math.cos(ego.yaw) * rel[1],
                    math.cos(ego.yaw) * rel[0] + math.sin(ego.yaw) * rel[1])
                a[3] = float(np.clip(c.tm_kyaw * az / 3.141, -c.tm_yaw, c.tm_yaw))
            else:
                a[3] = float(np.clip(a[3], -c.tm_yaw, c.tm_yaw))

        if override is not None:
            lr, fbs = self._sticks(override[0], override[1], ego.yaw)
            w = c.blend
            a[0] = (1.0 - w) * a[0] + w * float(np.clip(lr, -1.0, 1.0))
            a[1] = (1.0 - w) * a[1] + w * float(np.clip(fbs, -1.0, 1.0))
            if mode == "search" and c.se_yaw > 0 and abs(a[3]) > c.se_yaw:
                a[3] = math.copysign(c.se_yaw, a[3])

        # ---- stall breaker -----------------------------------------------------
        if c.sb_on:
            va = math.hypot(float(ego.vel[0]), float(ego.vel[1]))
            ad4 = a.copy()
            ad4[np.abs(ad4) < DEAD_ZONE] = 0.0
            rc4 = ego.rc + np.clip(ad4 - ego.rc, -SLEW_STEP, SLEW_STEP)
            vcx, vcy = self._vel(rc4[0], rc4[1], ego.yaw)
            spc = math.hypot(vcx, vcy)
            moving = max(va, spc) > c.sb_v
            # "engaged" = the actor still knows where the target is; never
            # interfere then, and never in the endgame.
            engaged = (rng is not None and float(track.age) <= c.sb_age)
            if moving or engaged or t < c.sb_t0:
                self._stall = 0
                self._sb_ok += 1
            else:
                self._stall += 1
                self._sb_ok = 0
            if not self._sb_active:
                if self._stall * DT >= c.sb_t and not engaged:
                    self._sb_active = True
            else:
                if engaged or self._sb_ok * DT >= c.sb_rec:
                    self._sb_active = False
            if self._sb_active and pose_ok:
                p3s = np.array([est[0], est[1], z])
                # body-forward expressed in the map frame
                hx, hy = to_map(math.cos(ego.yaw), math.sin(ego.yaw))
                ang = np.concatenate([[0.0], self._fan])
                ca, sa = np.cos(ang), np.sin(ang)
                dirs = np.column_stack([ca * hx - sa * hy, sa * hx + ca * hy])
                fr = self._fan_free(p3s, dirs, c.sb_look, c.sb_infl)
                k = int(np.argmax(fr))
                if fr[k] > 0.8:
                    ex, ey = to_ego(float(dirs[k, 0]) * c.sb_speed,
                                    float(dirs[k, 1]) * c.sb_speed)
                    lr, fbs = self._sticks(ex, ey, ego.yaw)
                    a[0] = float(np.clip(lr, -1.0, 1.0))
                    a[1] = float(np.clip(fbs, -1.0, 1.0))
                    if c.sb_yaw > 0.0:
                        a[3] = float(np.clip(a[3], -c.sb_yaw, c.sb_yaw))
                    mode = mode + "+sb"

        # ---- TTC clearance shield ---------------------------------------------
        if c.sh_on and pose_ok and (rng is None or rng > c.sh_minr):
            ad2 = a.copy()
            ad2[np.abs(ad2) < DEAD_ZONE] = 0.0
            rc2 = ego.rc + np.clip(ad2 - ego.rc, -SLEW_STEP, SLEW_STEP)
            vx, vy = self._vel(rc2[0], rc2[1], ego.yaw)
            sp = math.hypot(vx, vy)
            if sp > 0.15:
                p3 = np.array([est[0], est[1], z])
                mx, my = to_map(vx / sp, vy / sp)
                look = float(min(c.sh_look, max(1.0, sp * c.sh_ttc * 2.0)))
                need = sp * c.sh_ttc + c.sh_margin
                free0 = float(self._fan_free(
                    p3, np.array([[mx, my]]), look, c.sh_infl)[0])
                if free0 < need:
                    chosen = None
                    if c.sh_steer:
                        ang = self._fan[np.abs(np.degrees(self._fan)) <= c.sh_fan]
                        if ang.size:
                            ca, sa = np.cos(ang), np.sin(ang)
                            dirs = np.column_stack([ca * mx - sa * my,
                                                    sa * mx + ca * my])
                            fr = self._fan_free(p3, dirs, look, c.sh_infl)
                            ok = np.nonzero(fr >= need)[0]
                            if ok.size:
                                chosen = dirs[int(ok[0])]
                    if chosen is not None:
                        ex, ey = to_ego(float(chosen[0]) * sp, float(chosen[1]) * sp)
                    else:
                        allow = max(0.0, free0 - c.sh_margin) / max(c.sh_ttc, 1e-3)
                        k = min(1.0, allow / sp)
                        ex, ey = vx * k, vy * k
                    lr, fbs = self._sticks(ex, ey, ego.yaw)
                    a[0] = float(np.clip(lr, -1.0, 1.0))
                    a[1] = float(np.clip(fbs, -1.0, 1.0))
                    mode = mode + "+sh"

        # ---- terminal elevation guard (vertical stick only) --------------------
        if c.eg_on and rng is not None and track.rel is not None:
            fresh = float(track.age) <= c.eg_age
            if fresh and rng < c.eg_r:
                self._eg_t = t
                self._eg_dz = float(track.rel[2])
            act = fresh and rng < c.eg_r
            if (not act) and c.eg_hold > 0.0 and self._eg_t >= 0.0 \
                    and (t - self._eg_t) <= c.eg_hold and self._eg_dz is not None:
                act = True
            if act:
                relz = float(track.rel[2]) if fresh else float(self._eg_dz)
                dz = c.eg_dz
                if c.eg_taper > 0.0 and rng > c.eg_r:
                    dz = 0.0
                elif c.eg_taper > 0.0:
                    dz = c.eg_dz * min(1.0, max(0.0, (c.eg_r - rng) / c.eg_taper))
                err = relz + dz
                u = float(np.clip(c.eg_k * err / SPEED, -c.eg_lim, c.eg_lim))
                if z > c.eg_zmax:
                    u = min(u, 0.0)
                a[2] = max(a[2], u) if c.eg_up else u
                mode = mode + "+eg"

        # ---- pure terminal speed cap (direction and yaw untouched) -------------
        if c.tc_on and rng is not None and rng < c.tc_r \
                and float(track.age) <= c.tc_age:
            n = math.hypot(a[0], a[1])
            v = SPEED * n
            if v > c.tc_v and n > 1e-9:
                k = c.tc_v / v
                a[0] *= k
                a[1] *= k
                mode = mode + "+tc"

        # ---- velocity-error limiter --------------------------------------------
        # A commanded velocity far from the actual one makes the velocity PID
        # demand a huge tilt; 60 deg truncates the episode.  Bounding the error
        # forbids violent reversals without slowing anything down.
        if c.ve_on and t >= c.ve_t0:
            ad3 = a.copy()
            ad3[np.abs(ad3) < DEAD_ZONE] = 0.0
            rc3 = ego.rc + np.clip(ad3 - ego.rc, -SLEW_STEP, SLEW_STEP)
            vx, vy = self._vel(rc3[0], rc3[1], ego.yaw)
            ax, ay = float(ego.vel[0]), float(ego.vel[1])
            ex, ey = vx - ax, vy - ay
            en = math.hypot(ex, ey)
            if en > c.ve_max:
                k = c.ve_max / en
                lr, fbs = self._sticks(ax + ex * k, ay + ey * k, ego.yaw)
                a[0] = float(np.clip(lr, -1.0, 1.0))
                a[1] = float(np.clip(fbs, -1.0, 1.0))
                mode = mode + "+ve"

        # ---- takeoff guard ------------------------------------------------------
        if c.to_on and (t < c.to_t or z < c.to_z):
            a[3] = float(np.clip(a[3], -c.to_yaw, c.to_yaw))
            a[0] = float(np.clip(a[0], -c.to_lat, c.to_lat))
            a[1] = float(np.clip(a[1], -c.to_lat, c.to_lat))

        if c.diag_on:
            self.diag = {"mode": mode, "rng": rng, "est": None if est is None
                         else (float(est[0]), float(est[1]))}
        return np.clip(a, -1.0, 1.0)
