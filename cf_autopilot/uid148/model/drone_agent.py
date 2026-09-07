"""Swarm cf_autopilot controller (from-scratch v1).

Pipeline per step: depth -> PadNet (ONNX) pad detector + map classifier -> world-frame pad track;
state machine TAKEOFF -> CRUISE -> SEARCH -> APPROACH -> LAND (+RECOVER); depth-based local
planner with a safety tube; acceleration-limited velocity commands.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------- constants
SIM_DT = 1.0 / 50.0
SPEED_LIMIT = 3.0
CAM_FWD_M = 0.13
CAM_UP_M = 0.05
DEPTH_MIN_M = 0.5
DEPTH_MAX_M = 20.0
IMG_W = IMG_H = 128
DEPTH_NEAR_DUMMY = DEPTH_MIN_M
FOV_DEG = 90.0
STRIDE = 4
G = IMG_W // STRIDE

MAP_NAMES = ("city", "open", "mountain", "village", "warehouse", "forest")
_MODELS_DIR = Path(__file__).resolve().parent / "models"

# per-map parameters: cruise height above ground (altitude ray), search height, safety tube radius
MAP_PARAMS = {
    "city":      dict(h_cruise=4.0, h_search=4.0, tube=1.1, tube_min=0.45, v_cruise=3.0, v_search=2.5),
    "open":      dict(h_cruise=3.5, h_search=3.5, tube=1.1, tube_min=0.45, v_cruise=3.0, v_search=2.5),
    "mountain":  dict(h_cruise=5.0, h_search=6.5, tube=1.2, tube_min=0.6, v_cruise=3.0, v_search=2.5),
    "village":   dict(h_cruise=7.5, h_search=6.5, tube=1.1, tube_min=0.45, v_cruise=3.0, v_search=3.0),
    "warehouse": dict(h_cruise=6.5, h_search=6.0, tube=1.0, tube_min=0.35, v_cruise=2.5, v_search=2.0),
    "forest":    dict(h_cruise=4.0, h_search=4.0, tube=0.8, tube_min=0.35, v_cruise=2.5, v_search=2.0),
    "unknown":   dict(h_cruise=4.0, h_search=4.0, tube=1.1, tube_min=0.45, v_cruise=2.5, v_search=2.0),
}


# ----------------------------------------------------------------------------- geometry
def rot_from_rpy(r, p, y):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)


def cam_pose(pos, rpy):
    R = rot_from_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    fwd = R[:, 0]
    up = R[:, 2]
    right = np.cross(fwd, up)
    eye = np.asarray(pos, dtype=np.float64) + fwd * CAM_FWD_M + up * CAM_UP_M
    return eye, fwd, right, up


def pixel_depth_to_world(u, v, z, eye, fwd, right, up):
    t = math.tan(math.radians(FOV_DEG) / 2)
    xr = ((u + 0.5) / IMG_W * 2 - 1) * t
    yu = (1 - (v + 0.5) / IMG_H * 2) * t
    return eye + z * (fwd + xr * right + yu * up)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def unit(v, eps=1e-9):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > eps else np.zeros_like(v)


class DepthGrid:
    def __init__(self, n=32):
        self.n = n
        t = math.tan(math.radians(FOV_DEG) / 2)
        us = (np.arange(n) + 0.5) / n * 2 - 1
        vs = 1 - (np.arange(n) + 0.5) / n * 2
        xr, yu = np.meshgrid(us * t, vs * t)
        self.xr = xr.astype(np.float32)
        self.yu = yu.astype(np.float32)

    def pool_min(self, depth01):
        k = IMG_H // self.n
        return depth01.reshape(self.n, k, self.n, k).min(axis=(1, 3))

    def points_cam(self, pooled):
        z = (DEPTH_MIN_M + pooled.astype(np.float32) * (DEPTH_MAX_M - DEPTH_MIN_M))
        return np.stack([self.xr * z, self.yu * z, z], axis=-1).reshape(-1, 3), z.reshape(-1)


class Planner:
    """Depth-based direction selection with a safety tube. Works in camera frame (right, up, fwd)."""

    def __init__(self, n_pool=32, n_cand=11, half=0.9):
        self.grid = DepthGrid(n_pool)
        c = np.linspace(-half, half, n_cand)
        cy_rows = np.concatenate([c, [1.25, 1.7]])  # extra steep-climb rows (outside the FOV)
        cx, cy = np.meshgrid(c, cy_rows)
        self.cy_raw = cy.ravel().astype(np.float32)
        dirs = np.stack([cx.ravel(), cy.ravel(), np.ones(cx.size)], axis=-1)
        self.cand = (dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)).astype(np.float32)
        self.n_cand = self.cand.shape[0]

    def free_distances(self, depth01, tube_r, lookahead):
        pooled = self.grid.pool_min(depth01)
        pts, z = self.grid.points_cam(pooled)
        valid = z < 19.6
        pts = pts[valid]
        if pts.shape[0] == 0:
            return np.full(self.n_cand, lookahead, dtype=np.float32), None
        proj = self.cand @ pts.T                       # (C, P)
        r2 = np.sum(pts * pts, axis=1)[None, :]        # (1, P)
        lat2 = np.clip(r2 - proj * proj, 0.0, None)
        hit = (proj > 0.0) & (lat2 < tube_r * tube_r)
        pen = np.sqrt(np.clip(tube_r * tube_r - lat2, 0.0, None))
        dist = np.where(hit, np.clip(proj - pen, 0.0, None), np.inf)
        free = dist.min(axis=1)
        free = np.minimum(free, lookahead).astype(np.float32)
        return free, pts


# ----------------------------------------------------------------------------- detector
class PadDetector:
    def __init__(self, path):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    def __call__(self, depth01):
        x = np.ascontiguousarray(depth01.reshape(1, 1, IMG_H, IMG_W), dtype=np.float32)
        heat, off, zed, cls = self.sess.run(None, {self.input_name: x})
        heat = np.nan_to_num(heat, nan=-20.0, posinf=20.0, neginf=-20.0)
        off = np.nan_to_num(off)
        zed = np.nan_to_num(zed, nan=0.0)
        cls = np.nan_to_num(cls)
        hm = 1.0 / (1.0 + np.exp(-heat[0, 0]))
        # nms 3x3
        pad = np.pad(hm, 1, mode="constant")
        mx = np.max(np.stack([pad[i:i + G, j:j + G] for i in range(3) for j in range(3)]), axis=0)
        peaks = hm * (hm >= mx)
        idx = np.argsort(peaks.ravel())[::-1][:3]
        dets = []
        for k in idx:
            s = float(peaks.ravel()[k])
            if s < 0.2:
                break
            iv, iu = divmod(int(k), G)
            u = (iu + float(off[0, 0, iv, iu])) * STRIDE
            v = (iv + float(off[0, 1, iv, iu])) * STRIDE
            z = float(zed[0, 0, iv, iu]) * 20.0
            dets.append((s, u, v, z))
        return dets, cls[0]


# ----------------------------------------------------------------------------- controller
class _MyController:
    def __init__(self, models_dir=None):
        self.models_dir = Path(models_dir) if models_dir else _MODELS_DIR
        det_path = self.models_dir / "padnet.onnx"
        self.detector = PadDetector(det_path) if det_path.exists() else None
        self.planner = Planner()
        self.debug = {}
        self._pos_xy = None
        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(self):
        self.t = 0.0
        self.step = 0
        self.mode = "TAKEOFF"
        self.mode_t0 = 0.0
        self.v_cmd = np.zeros(3)
        self.yaw_cmd = None
        self.start_pos = None
        self.start_alt = None
        self.map_logp = np.zeros(6)
        self.map_n = 0
        self.map_label = "unknown"
        self.map_locked = False
        # pad track
        self.pad_pos = None      # world estimate of pad top centre
        self.pad_vel = np.zeros(2)
        self.pad_hits = 0
        self.pad_last_seen = -1.0
        self.pad_obs = []        # list of (t, x, y, z, dist)
        self.pad_conf = 0.0
        self.pad_conflicts = 0
        self.pad_moving = False
        self.pad_move_votes = 0
        self.pad_miss_t = 0.0
        self.pad_omega = 0.0
        self.pad_acc = np.zeros(2)
        self.pad_slow_t = 0.0
        self.pad_fit_sp = 0.0
        self.pad_fit_z = 99.0
        self.pad_min_z = 99.0
        self.pad_info = 0.0
        self.pad_raw = []
        self.pad_still_t = 0.0
        self.pad_ever_moving = False
        self.pad_close_seen_t = -99.0
        self.pad_hist = []
        self.bad_spots = []
        self.center0 = None
        self._rpy = np.zeros(3)
        self.recover_z = 0.0
        self.land_stuck_t = 0.0
        self.reacq_n = 0
        self.reacq_hold = 0.0
        self.reacq_wp = None
        self.reacq_target = None
        self.reacq_moving = False
        self.reacq_side = 0
        self.reacq_t_enter = 0.0
        self.land_dive = False
        self.touch_t = 0.0
        self.land_pdir = None
        self.land_phase = "track"
        self.land_retry = 0
        self.drop_t0 = 0.0
        self.amb_dbg = None
        self.amb_wait_t = 0.0
        self.search_center_xy = None
        self.drop_z0 = -1.0
        self.dash_v0 = np.zeros(2)
        self.dash_p0 = np.zeros(2)
        self.dash_t1 = 0.0
        self.pv_hist = []
        self.e_hist = []
        self.rest_t = 0.0
        self.touch_hold = -1.0
        self.drop_mis_t = 0.0
        # search
        self.search_wps = None
        self.search_i = 0
        self.search_anchor = None
        self.spin_target = None
        self.spin_done = False
        # landing
        self.land_t0 = None
        self.land_xy = None
        self.land_z = None
        self.recover_n = 0
        self.stuck_t = 0.0
        self.last_pos = None
        self.blocked_steps = 0
        self.debug = {}

    # ------------------------------------------------------------------ helpers
    def _params(self):
        return MAP_PARAMS.get(self.map_label, MAP_PARAMS["unknown"])

    def _update_map(self, cls_logits, pos, alt_ray):
        if self.map_locked:
            return
        z = cls_logits - cls_logits.max()
        p = np.exp(z)
        p /= p.sum()
        self.map_logp += np.log(p + 1e-6)
        self.map_n += 1
        # rules
        if self.step == 1:
            if pos[2] < 0.7:
                self.map_logp[3] += 6.0  # village: ground pad
            if abs(pos[0]) > 76 or abs(pos[1]) > 76:
                self.map_logp[2] += 6.0  # only mountain spans that far
            if abs(pos[1]) > 23.5 or abs(pos[0]) > 38.5:
                self.map_logp[4] -= 6.0  # outside the warehouse box
            if abs(pos[0]) > 42.5 or abs(pos[1]) > 42.5:
                self.map_logp[5] -= 6.0  # outside forest / village box
                self.map_logp[3] -= 6.0
        best = int(np.argmax(self.map_logp))
        self.map_label = MAP_NAMES[best]
        if self.map_n >= 60:
            self.map_locked = True

    def _detect(self, depth, pos, rpy):
        if self.detector is None:
            return [], np.zeros(6)
        dets, cls = self.detector(depth)
        eye, fwd, right, up = cam_pose(pos, rpy)
        out = []
        for s, u, v, zp in dets:
            z = float(zp)  # the network depth is the most accurate estimate (validated offline)
            if z > 19.5 or z < 0.3:
                continue
            w = pixel_depth_to_world(u, v, z, eye, fwd, right, up)
            out.append((s, w, z, u, v))
        return out, cls

    def _update_track(self, dets, pos):
        """Fuse detections into the pad estimate. dets: list of (score, world_pt, z, u, v)."""
        self._pos_xy = np.asarray(pos[:2], dtype=float)
        if (self.mode == "LAND" and self.pad_pos is not None and not self.pad_moving
                and pos[2] - self.pad_pos[2] < 1.5 and self.pad_hits >= 30):
            if dets and max(d[0] for d in dets) >= 0.45:
                self.pad_hits += 1
                self.pad_last_seen = self.t
            return
        if not dets or max(d[0] for d in dets) < 0.45:
            if self.pad_pos is not None and self._pad_in_view(pos, self._rpy):
                self.pad_miss_t += SIM_DT
                if self.pad_miss_t > 1.2 and self.mode != "LAND" and not self._static_established():
                    self._drop_track()
            return
        good = [d for d in dets if d[0] >= 0.45 and np.isfinite(d[1]).all()]
        if not good:
            return
        if self.pad_pos is not None and len(good) > 1:
            pred0, _ = self._motion_at(min(self.t - self.pad_last_seen, 3.0))
            s, w, z, u, v = min(good, key=lambda d: float(np.linalg.norm(d[1][:2] - pred0[:2])))
        else:
            s, w, z, u, v = max(good, key=lambda d: d[0])
        self.pad_miss_t = 0.0
        # reject detections at the start pad or at spots where a landing already failed
        if self.start_pos is not None and np.linalg.norm(w[:2] - self.start_pos[:2]) < 1.5 and abs(w[2] - self.start_pos[2]) < 1.5:
            return
        for bs in self.bad_spots:
            if np.linalg.norm(w[:2] - bs[:2]) < 2.0 and abs(w[2] - bs[2]) < 2.0:
                return
        if self.center0 is not None and np.linalg.norm(w[:2] - self.center0[:2]) > 24.0:
            return  # the pad lies within the search radius (<= 17.6 m) of the noisy centre
        if self.pad_pos is None:
            self.pad_pos = w.copy()
            self.pad_hits = 1
            self.pad_conf = s
            self.pad_obs = [(self.t, w[0], w[1], w[2], z)]
            self.pad_last_seen = self.t
            self.pad_min_z = float(z)
            self.pad_info = 0.0
            self.pad_raw = []
            self.pad_still_t = 0.0
            self._push_raw(w, z)
            return
        dt = self.t - self.pad_last_seen
        pred, _ = self._motion_at(min(dt, 3.0))
        err = float(np.linalg.norm(w[:2] - pred[:2]))
        est_static = self._static_established()
        wide = max(1.5, 0.12 * z) + 1.2 * dt
        gate = (max(0.7, 0.08 * z) if est_static else wide)
        if err <= wide:
            self._push_raw(w, z)
            if self._raw_motion_check(w):
                return
        if err > gate:
            self.pad_conflicts += 1
            need_conf = int(min(40, 6 + self.pad_hits // 4)) if not est_static else 100
            if est_static and self.pad_conflicts >= 40 and z >= 1.0 and err > 1.0:
                # an established static track that is consistently seen elsewhere: it moves
                self.pad_pos = w.copy()
                self.pad_hits = max(self.pad_hits, 30)
                self.pad_obs = [(self.t, w[0], w[1], w[2], z)]
                self.pad_last_seen = self.t
                self.pad_conflicts = 0
                self.pad_info = 0.0
                self.pad_moving = True
                self.pad_ever_moving = True
                self.pad_vel = np.zeros(2)
                return
            if self.pad_conflicts >= need_conf:
                # re-initialise on the new cluster
                self.pad_pos = w.copy()
                self.pad_vel = np.zeros(2)
                self.pad_hits = 1
                self.pad_conf = s
                self.pad_obs = [(self.t, w[0], w[1], w[2], z)]
                self.pad_last_seen = self.t
                self.pad_conflicts = 0
                self.pad_min_z = float(z)
                self.pad_info = 0.0
                self.pad_raw = []
                self.pad_still_t = 0.0
                self.pad_moving = bool(self.pad_ever_moving)
                self.pad_move_votes = 0
                self.pad_omega = 0.0
                self.pad_acc = np.zeros(2)
            return
        self.pad_conflicts = 0
        if self.pad_moving or not est_static:
            alpha = float(np.clip(0.35 * (6.0 / max(z, 2.0)), 0.15, 0.7))
            self.pad_pos = (1 - alpha) * pred + alpha * w
        else:
            # static: information-weighted running mean (noise grows with range)
            sig = 0.05 + 0.012 * float(z)
            wgt = 1.0 / (sig * sig)
            alpha = float(np.clip(wgt / (self.pad_info + wgt), 0.04, 0.7))
            if z < 1.5 and float(np.linalg.norm(w[:2] - self.pad_pos[:2])) > 0.5 and self.pad_hits >= 30:
                alpha = 0.01  # partial view of a pad right below us: barely trust it
            self.pad_pos = (1 - alpha) * self.pad_pos + alpha * w
            self.pad_info = min(self.pad_info + wgt, 4000.0)
        self.pad_hits += 1
        self.pad_conf = 0.8 * self.pad_conf + 0.2 * s
        self.pad_last_seen = self.t
        if z < 6.0:
            self.pad_close_seen_t = self.t
        self.pad_min_z = min(self.pad_min_z, float(z))
        self.pad_obs.append((self.t, w[0], w[1], w[2], z))
        if len(self.pad_obs) > 75:
            self.pad_obs.pop(0)
        pass
        if not self.pad_hist or self.t - self.pad_hist[-1][0] >= 0.2:
            self.pad_hist.append((self.t, float(self.pad_pos[0]), float(self.pad_pos[1]), float(self.pad_pos[2])))
            if len(self.pad_hist) > 150:
                self.pad_hist.pop(0)
        # velocity fit over the last 1.5 s
        win = 1.2 if z < 6.0 else (1.5 if z < 10.0 else 2.0)
        obs = [o for o in self.pad_obs if self.t - o[0] <= win and o[4] < 16.0]
        if len(obs) >= 8 and (obs[-1][0] - obs[0][0]) >= 0.5 * win:
            z_mean = float(np.mean([o[4] for o in obs]))
            a = np.asarray(obs)
            tt = a[:, 0] - a[:, 0].mean()
            sxx = float((tt * tt).sum())
            vx = float((tt * (a[:, 1] - a[:, 1].mean())).sum() / sxx)
            vy = float((tt * (a[:, 2] - a[:, 2].mean())).sum() / sxx)
            v = np.array([vx, vy])
            # fit quality: residual vs explained displacement
            pred_x = a[:, 1].mean() + vx * tt
            pred_y = a[:, 2].mean() + vy * tt
            resid = float(np.sqrt(np.mean((a[:, 1] - pred_x) ** 2 + (a[:, 2] - pred_y) ** 2)))
            sp_lin = float(np.linalg.norm(v))
            span = sp_lin * (a[-1, 0] - a[0, 0])
            # quadratic fit: velocity at the window end (removes the half-window lag) + acceleration
            acc = np.zeros(2)
            if len(a) >= 20 and self.pad_moving:
                try:
                    cx = np.polyfit(tt, a[:, 1], 2)
                    cy = np.polyfit(tt, a[:, 2], 2)
                    acc = np.array([2 * cx[0], 2 * cy[0]])
                    an = float(np.linalg.norm(acc))
                    if an > 1.5:
                        acc *= 1.5 / an
                except Exception:
                    pass
            sp = float(np.linalg.norm(v))
            if sp > 3.0:
                v *= 3.0 / sp
                sp = 3.0
            self.pad_acc = 0.6 * self.pad_acc + 0.4 * acc
            v_new = 0.8 * self.pad_vel + 0.2 * v
            dv_ = v_new - self.pad_vel
            dn_ = float(np.linalg.norm(dv_))
            lim = 2.0 * SIM_DT if self.pad_moving else 9.0
            if dn_ > lim:
                dv_ *= lim / dn_
            self.pad_vel = self.pad_vel + dv_
            if sp < 0.4:
                self.pad_omega *= 0.5
            # turn rate from two half-window velocity fits
            obs2 = [o for o in self.pad_obs if self.t - o[0] <= 2.5 and o[4] < 12.0]
            if len(obs2) >= 24 and sp > 0.3:
                b_ = np.asarray(obs2)
                half = len(b_) // 2
                def _fitv(c):
                    tt2 = c[:, 0] - c[:, 0].mean()
                    sxx2 = float((tt2 * tt2).sum())
                    if sxx2 < 1e-6:
                        return None, 0.0
                    return np.array([float((tt2 * (c[:, 1] - c[:, 1].mean())).sum() / sxx2),
                                     float((tt2 * (c[:, 2] - c[:, 2].mean())).sum() / sxx2)]), float(c[:, 0].mean())
                v1, t1 = _fitv(b_[:half])
                v2, t2 = _fitv(b_[half:])
                if v1 is not None and v2 is not None and np.linalg.norm(v1) > 0.5 and np.linalg.norm(v2) > 0.5 and t2 - t1 > 0.4:
                    om = wrap(math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0])) / (t2 - t1)
                    om = float(np.clip(om, -1.0, 1.0))
                    self.pad_omega = 0.7 * self.pad_omega + 0.3 * om
            if self.map_label in ("warehouse", "forest"):
                need_votes, need_sp, need_span = 999, 9.0, 99.0  # these maps never move the pad
            elif z_mean < 6.0:
                need_votes, need_sp, need_span = 6, 0.45, 0.7
            elif z_mean < 10.0:
                need_votes, need_sp, need_span = 6, 0.5, 0.9
            else:
                need_votes, need_sp, need_span = 8, 0.55, 1.2
            if self.map_label == "open":
                need_votes, need_sp, need_span = 4, min(need_sp, 0.3), min(need_span, 0.45)
            consistent = span > max(need_span, 2.5 * resid)
            if z_mean > 9.0 and not self.pad_moving:
                # range biases fake motion along the line of sight: at range trust only the perpendicular part
                los = unit(self.pad_pos[:2] - self._pos_xy) if self._pos_xy is not None else np.zeros(2)
                v_perp = v - float(v @ los) * los
                sp_perp = float(np.linalg.norm(v_perp))
                span_perp = sp_perp * (a[-1, 0] - a[0, 0])
                consistent = consistent and sp_perp > 0.45 and span_perp > max(0.6 * need_span, 2.0 * resid)
            if sp > need_sp and consistent:
                self.pad_move_votes += 1
                if self.pad_move_votes >= need_votes:
                    if not self.pad_moving:
                        self.pad_pos = np.array([obs[-1][1], obs[-1][2], obs[-1][3]])
                    self.pad_moving = True
                    if z_mean < 9.0 and sp_lin > 0.5:
                        self.pad_ever_moving = True
                    self.pad_slow_t = 0.0
            else:
                self.pad_move_votes = max(0, self.pad_move_votes - 1)
            self.pad_fit_sp = sp_lin
            self.pad_fit_z = z_mean
            if not self.pad_moving and sp < 0.3:
                self.pad_vel *= 0.5
        elif self.t - self.pad_last_seen > 2.0:
            self.pad_vel *= 0.9

    def _motion_at(self, dt):
        """Predicted (position, velocity) of a moving pad dt seconds after the last update."""
        p = self.pad_pos.copy()
        v = self.pad_vel.copy()
        if not self.pad_moving:
            return p, np.zeros(2)
        om = float(self.pad_omega)
        sp = float(np.linalg.norm(v))
        if abs(om) < 0.2 or sp < 0.1:
            p[:2] += v * dt
            return p, v
        th = math.atan2(v[1], v[0])
        th2 = th + om * dt
        p[0] += sp / om * (math.sin(th2) - math.sin(th))
        p[1] += sp / om * (math.cos(th) - math.cos(th2))
        v2 = np.array([sp * math.cos(th2), sp * math.sin(th2)])
        return p, v2

    def _pad_predicted(self):
        if self.pad_pos is None:
            return None
        dt = min(self.t - self.pad_last_seen, 3.0)
        if not self.pad_moving:
            p = self.pad_pos.copy()
            p[:2] += self._pad_slow_vel() * min(dt, 1.5)
            return p
        p, _ = self._motion_at(dt)
        return p

    def _pad_slow_vel(self):
        return np.zeros(2)

    def _pad_vel_now(self):
        if self.pad_pos is None or not self.pad_moving:
            return np.zeros(2)
        dt = min(self.t - self.pad_last_seen, 3.0)
        _, v = self._motion_at(dt)
        return v

    def _track_reliable(self):
        if self.pad_pos is None or self.pad_conf <= 0.6 or (self.t - self.pad_last_seen) >= 2.5:
            return False
        need = 5
        if self.center0 is not None and np.linalg.norm(self.pad_pos[:2] - self.center0[:2]) > 30.0:
            need = 12
        if self.map_label == "forest":
            need = max(need, 15)
            if self.pad_conf <= 0.7:
                return False
        return self.pad_hits >= need

    def _pad_pixel_row(self, pos, rpy):
        p = self._pad_predicted()
        if p is None:
            return None
        eye, fwd, right, up = cam_pose(pos, rpy)
        d = p - eye
        z = float(d @ fwd)
        if z < 0.3:
            return None
        t = math.tan(math.radians(FOV_DEG) / 2)
        yu = float(d @ up) / z / t
        return (1 - yu) / 2 * IMG_H

    def _pad_in_view(self, pos, rpy, margin=8.0):
        p = self._pad_predicted()
        if p is None:
            return False
        eye, fwd, right, up = cam_pose(pos, rpy)
        d = p - eye
        z = float(d @ fwd)
        if z < 1.0 or z > 15.0:
            return False
        t = math.tan(math.radians(FOV_DEG) / 2)
        xr = float(d @ right) / z / t
        yu = float(d @ up) / z / t
        u = (xr + 1) / 2 * IMG_W
        v = (1 - yu) / 2 * IMG_H
        return margin <= u < IMG_W - margin and margin <= v < IMG_H - margin

    def _drop_track(self):
        self.pad_pos = None
        self.pad_vel = np.zeros(2)
        self.pad_hits = 0
        self.pad_conf = 0.0
        self.pad_obs = []
        self.pad_conflicts = 0
        self.pad_moving = False
        self.pad_move_votes = 0
        self.pad_miss_t = 0.0
        self.pad_omega = 0.0
        self.pad_acc = np.zeros(2)
        self.pad_slow_t = 0.0
        self.pad_fit_sp = 0.0
        self.pad_fit_z = 99.0
        self.pad_min_z = 99.0
        self.pad_info = 0.0
        self.pad_raw = []
        self.pad_still_t = 0.0
        pass  # pad_ever_moving is a property of the world, kept across track drops
        self.pad_close_seen_t = -99.0
        self.pad_hist = []
        self.pad_min_z = 99.0
        self.pad_raw = []

    def _push_raw(self, w, z):
        if 1.0 <= z < 8.0 and (not self.pad_raw or self.t - self.pad_raw[-1][0] >= 0.1):
            self.pad_raw.append((self.t, float(w[0]), float(w[1])))
            if len(self.pad_raw) > 80:
                self.pad_raw.pop(0)

    def _raw_motion_check(self, w):
        """Long-baseline displacement of close-range detections. Returns True if the track was re-seeded."""
        if len(self.pad_raw) < 12:
            return False
        r_ = np.asarray(self.pad_raw)
        r_ = r_[self.t - r_[:, 0] <= 8.0]
        if len(r_) < 12 or r_[-1, 0] - r_[0, 0] < 1.5:
            return False
        q = max(3, len(r_) // 4)
        p0 = np.median(r_[:q, 1:3], axis=0)
        p1 = np.median(r_[-q:, 1:3], axis=0)
        d_ = p1 - p0
        disp = float(np.linalg.norm(d_))
        self.pad_still_t = (r_[-1, 0] - r_[0, 0]) if disp < 0.4 else 0.0
        los = unit(self.pad_pos[:2] - self._pos_xy) if self._pos_xy is not None else np.zeros(2)
        perp = float(np.linalg.norm(d_ - float(d_ @ los) * los))
        moved = (perp > 0.35 and disp > 0.5) or disp > 0.9
        if moved and self.map_label not in ("warehouse", "forest"):
            if not self.pad_moving:
                self.pad_pos = w.copy()
                self.pad_info = 0.0
                self.pad_obs = self.pad_obs[-20:]
                self.pad_last_seen = self.t
                self.pad_conflicts = 0
                self.pad_moving = True
                self.pad_ever_moving = True
                self.pad_slow_t = 0.0
                return True
        elif (self.pad_moving and not self.pad_ever_moving and r_[-1, 0] - r_[0, 0] >= 5.0
              and disp < 0.35 and self.pad_fit_sp < 0.2):
            self.pad_moving = False
            self.pad_move_votes = 0
            self.pad_vel = np.zeros(2)
            self.pad_omega = 0.0
            self.pad_acc = np.zeros(2)
        return False

    def _static_established(self):
        """A static track good enough to finish blind (drone position is exact, pad estimate converged)."""
        return (self.pad_pos is not None and not self.pad_moving and not self.pad_ever_moving
                and self.pad_hits >= 30 and self.pad_min_z < 9.0 and self.pad_conf > 0.5
                and (self.pad_still_t >= 3.0 or (self.map_label in ("warehouse", "forest") and self.pad_hits >= 60 and self.pad_conf > 0.7)))

    def _track_solid(self):
        """Enough hits to fly at full speed toward the estimate."""
        return self.pad_pos is not None and self.pad_hits >= 30 and self.pad_conf > 0.5

    def _set_mode(self, m):
        if m != self.mode:
            self.mode = m
            self.mode_t0 = self.t

    # ------------------------------------------------------------------ planner wrapper
    def _plan(self, depth, pos, rpy, v_des, tube, tube_min, lookahead=10.0):
        """Return (v_out world, blocked flag, free distance along chosen direction)."""
        speed = float(np.linalg.norm(v_des))
        if speed < 0.05:
            return v_des, False, lookahead
        d = v_des / speed
        eye, fwd, right, up = cam_pose(pos, rpy)
        M = np.stack([right, up, fwd], axis=0)  # world -> cam rows
        d_cam = M @ d
        # if the desired direction is far outside the FOV, only allow slow motion toward the FOV edge
        cone_cos = math.cos(math.radians(50))
        free, _ = self.planner.free_distances(depth, tube, lookahead)
        if not np.isfinite(free).all():
            free = np.nan_to_num(free, nan=0.0)
        cand = self.planner.cand
        if self.map_label in ("forest", "warehouse"):
            free = free.copy()
            free[self.planner.cy_raw > 1.0] = 0.0  # never pick blind steep climbs under canopies / roofs
        cos_ang = cand @ d_cam
        ang = np.arccos(np.clip(cos_ang, -1, 1))
        # candidate scoring
        L_pref = min(lookahead, max(3.0, speed * speed / (2 * 2.0) + 2.0))
        score = np.minimum(free, L_pref) / L_pref - 0.9 * ang
        # forbid candidates pointing strongly down (elevation < -35 deg) unless desired is down too
        elev = np.arcsin(np.clip(cand[:, 1], -1, 1))
        if d_cam[1] > -0.5:
            score[elev < -0.6] -= 1.0
        best = int(np.argmax(score))
        if d_cam[2] < cone_cos:
            # desired direction not visible: pick the candidate closest to it, slow speed
            best = int(np.argmin(ang))
            fresh = self.mode == "APPROACH" and (self.t - self.pad_last_seen) < 1.0
            out_speed = min(speed, 1.2 if fresh else 0.6)
        else:
            out_speed = speed
        if free[best] < 1.0 and speed > 0.3:
            # try a smaller tube before giving up
            free2, _ = self.planner.free_distances(depth, tube_min, lookahead)
            score2 = np.minimum(free2, L_pref) / L_pref - 0.9 * ang
            if d_cam[1] > -0.5:
                score2[elev < -0.6] -= 1.0
            best2 = int(np.argmax(score2))
            if free2[best2] > free[best] + 0.5:
                best = best2
                free = free2
                out_speed = min(out_speed, 1.2)
        f = float(free[best])
        # braking: speed <= sqrt(2 a (f - margin))
        margin = 0.6
        v_brake = math.sqrt(max(0.0, 2 * 2.2 * (f - margin)))
        out_speed = min(out_speed, max(v_brake, 0.0))
        dir_world = M.T @ cand[best]
        blocked = f < 1.2
        if f < 1.5 and d_cam[2] >= cone_cos:
            # nothing usable ahead: climb (blind) and back off a little
            top = DEPTH_MIN_M + float(depth[0:16, 24:104].min()) * (DEPTH_MAX_M - DEPTH_NEAR_DUMMY)
            climb = self._climb_speed(pos, top)
            if climb > 0.0:
                esc = np.array([0.0, 0.0, climb]) - fwd * 0.3
                return esc, True, f
            best2 = int(np.argmax(free))
            return (M.T @ cand[best2]) * min(0.6, float(free[best2]) * 0.3) - fwd * 0.3, True, f
        return dir_world * out_speed, blocked, f

    def _climb_speed(self, pos, top_band_m):
        m = self.map_label
        if m == "warehouse":
            return 0.6 if (pos[2] < 9.0 and top_band_m > 4.0) else 0.0
        if m == "forest":
            return 0.5 if (pos[2] < 12.0 and top_band_m > 6.0) else 0.0
        return 1.0

    # ------------------------------------------------------------------ main
    def act(self, observation):
        self.step += 1
        self.t = self.step * SIM_DT
        st = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
        depth = np.asarray(observation["depth"], dtype=np.float32).reshape(IMG_H, IMG_W)
        pos = st[0:3].astype(np.float64)
        rpy = st[3:6].astype(np.float64)
        vel = st[6:9].astype(np.float64)
        alt = float(st[137]) * 20.0
        goal_off = st[138:141].astype(np.float64)
        center = pos + goal_off
        if self.start_pos is None:
            self.start_pos = pos.copy()
            self.start_alt = alt
            self.center0 = center.copy()
            self.v_cmd = vel.copy()
        yaw = float(rpy[2])
        self._rpy = rpy
        prm = self._params()

        # perception
        dets, cls = self._detect(depth, pos, rpy)
        self.last_dets = dets
        self._update_map(cls, pos, alt)
        if self.mode != "TAKEOFF" or self.t > 1.0:
            self._update_track(dets, pos)
        if self.pad_moving and self.pad_pos is not None:
            recent = (self.t - self.pad_last_seen) < 0.3
            if recent and self.pad_fit_sp < 0.2 and self.pad_fit_z < 6.0:
                self.pad_slow_t += SIM_DT
            elif recent:
                self.pad_slow_t = 0.0
            pass
        prm = self._params()

        v_des = np.zeros(3)
        yaw_des = yaw
        use_planner = True
        vz_override = None
        tube = prm["tube"]
        tube_min = prm["tube_min"]
        pad = self._pad_predicted()
        h_cruise = prm["h_cruise"]

        # ----------------------------------------------------------- mode logic
        if self.mode == "TAKEOFF":
            yaw_des = math.atan2(goal_off[1], goal_off[0])
            climbed = pos[2] - self.start_pos[2]
            target_h = h_cruise
            if self.map_label in ("forest", "warehouse") and self.start_pos[2] > h_cruise - 0.5:
                target_h = 1.0  # already high: just clear the pad, descend to cruise height afterwards
            # do not climb into a low ceiling: top rows of the depth image
            top = DEPTH_MIN_M + depth[0:12, 32:96].min() * (DEPTH_MAX_M - DEPTH_MIN_M)
            ready = alt >= target_h or climbed >= target_h + 0.5 or (top < 2.5 and climbed > 0.8)
            open_sky = self.map_label not in ("forest", "warehouse")
            v_des = np.array([0.0, 0.0, (2.5 if open_sky else 1.5) if not ready else 0.0])
            use_planner = False
            yaw_err = abs(wrap(yaw_des - yaw))
            early = open_sky and climbed >= 2.0 and yaw_err < 0.6  # keep climbing while already under way
            if (ready and (yaw_err < 0.35 or self.t - self.mode_t0 > 6.0)) or early:
                self._set_mode("CRUISE")
        elif self.mode == "CRUISE":
            to_c = center - pos
            d_h = float(np.linalg.norm(to_c[:2]))
            dir_h = unit(to_c[:2])
            v_h = min(prm["v_cruise"], math.sqrt(2 * 1.8 * max(d_h - 1.0, 0.0)) + 0.4)
            # altitude: terrain following via altitude ray, but ignore the noisy centre z
            vz = float(np.clip(1.2 * (h_cruise - alt), -2.5 if alt > 8.0 else -1.0, 1.5))
            if self.map_label == "mountain" and d_h < 25.0:
                # near the search area: descend toward the (noisy) centre height early, never below 2 m over ground
                z_t = max(float(center[2]) + 3.0, pos[2] - alt + 2.0)
                vz = float(np.clip(1.2 * (z_t - pos[2]), -2.5 if alt > 8.0 else -1.0, 1.5))
            if self.map_label in ("forest", "warehouse") and alt > h_cruise + 1.5:
                v_h = min(v_h, 0.6)  # get down out of the canopy / rack zone before cruising
            v_des = np.array([dir_h[0] * v_h, dir_h[1] * v_h, vz])
            yaw_des = math.atan2(dir_h[1], dir_h[0]) if d_h > 1.0 else yaw
            if self._track_reliable():
                self._set_mode("APPROACH")
            elif d_h < 2.0:
                self._set_mode("SEARCH")
                self.search_wps = None
        elif self.mode == "SEARCH":
            if self._track_reliable():
                self._set_mode("APPROACH")
            else:
                if self.search_wps is None:
                    # spin first, then expanding square path
                    self.spin_start_yaw = yaw
                    self.spin_acc = 0.0
                    self.spin_prev = yaw
                    self.spin_done = False
                    c = center.copy()
                    anchor = getattr(self, "search_anchor", None)
                    if anchor is not None:
                        c[:2] = anchor
                        self.search_anchor = None
                    wps = []
                    n_wp = 4 if self.map_label == "village" else 5
                    for r in (7.0, 14.0, 20.0):
                        for k in range(n_wp):
                            a = self.spin_start_yaw + k * math.pi / 2
                            wps.append(c[:2] + r * np.array([math.cos(a), math.sin(a)]))
                    self.search_wps = wps
                    self.search_i = 0
                    self.search_center_xy = c[:2].copy()
                h_s = prm["h_search"]
                if not self.spin_done and self.map_label not in ("village", "mountain"):
                    h_s = min(h_s, 3.5)
                vz = float(np.clip(1.2 * (h_s - alt), -2.5 if alt > 8.0 else -1.0, 1.5))
                if self.map_label == "mountain":
                    # the pad usually sits in a hollow below the surrounding terrain: aim ~3 m above the
                    # (noisy, +-5 m) search-centre height instead of h_s above the local terrain,
                    # but never closer than 2 m to the ground below
                    z_t = max(float(center[2]) + 1.5, pos[2] - alt + 2.0)
                    vz = float(np.clip(1.2 * (z_t - pos[2]), -2.5 if alt > 8.0 else -1.0, 1.5))
                if not self.spin_done:
                    # rotate in place (yaw target leads by 90 deg), track how far we have turned
                    self.spin_acc += abs(wrap(yaw - self.spin_prev))
                    self.spin_prev = yaw
                    yaw_des = yaw + 1.2
                    v_des = np.array([0.0, 0.0, vz])
                    use_planner = False
                    if self.spin_acc > math.pi * 1.1 or (self.t - self.mode_t0) > 6.0:
                        self.spin_done = True
                else:
                    wp = self.search_wps[self.search_i % len(self.search_wps)]
                    to = wp - pos[:2]
                    d_h = float(np.linalg.norm(to))
                    if d_h < (2.0 if self.map_label == "village" else 1.5):
                        self.search_i += 1
                        wp = self.search_wps[self.search_i % len(self.search_wps)]
                        to = wp - pos[:2]
                        d_h = float(np.linalg.norm(to))
                    dir_h = unit(to)
                    v_h = min(prm["v_search"], math.sqrt(2 * 1.5 * max(d_h - 0.8, 0.0)) + 0.5)
                    v_des = np.array([dir_h[0] * v_h, dir_h[1] * v_h, vz])
                    yaw_des = math.atan2(dir_h[1], dir_h[0])
                    if self.map_label == "village" and self.search_center_xy is not None:
                        # look partly inward while circling: the pad is usually inside the ring (measured +0.02 on village)
                        inward = self.search_center_xy - pos[:2]
                        if float(np.linalg.norm(inward)) > 2.0:
                            off = wrap(math.atan2(inward[1], inward[0]) - yaw_des)
                            yaw_des = yaw_des + float(np.clip(off, -0.7, 0.7))
        elif self.mode == "APPROACH":
            unseen = self.t - self.pad_last_seen
            if pad is None:
                lost = True
            elif self.pad_moving:
                lost = unseen > 2.5
            elif self._static_established():
                lost = unseen > 25.0  # finish on the estimate: the drone position is exact
            else:
                lost = self.pad_miss_t > 2.0 or unseen > 6.0
            if lost:
                if pad is not None and self.reacq_n < 3:
                    self.reacq_n += 1
                    self.reacq_target = pad.copy()
                    self.reacq_moving = bool(self.pad_moving)
                    back = unit(pos[:2] - pad[:2])
                    if float(np.linalg.norm(back)) < 1e-6:
                        back = np.array([math.cos(yaw), math.sin(yaw)]) * -1.0
                    up_ = 1.5 if self.map_label == "forest" else (3.5 if self.map_label in ("village", "city") else (4.0 if self.map_label == "mountain" else 2.5))
                    self.reacq_wp = np.array([pad[0] + back[0] * 5.0, pad[1] + back[1] * 5.0, pad[2] + up_])
                    if not self.pad_moving:
                        self._drop_track()
                    self.reacq_hold = 0.0
                    self.reacq_side = 0
                    self.reacq_t_enter = self.t
                    self._set_mode("REACQUIRE")
                else:
                    self.search_anchor = pad[:2].copy() if pad is not None else None
                    self._drop_track()
                    self._set_mode("SEARCH")
                    self.search_wps = None
            else:
                rel = pad - pos
                d_h = float(np.linalg.norm(rel[:2]))
                if self.map_label == "forest":
                    tube = min(tube, 0.7)
                    tube_min = 0.4
                else:
                    tube = min(tube, 0.6)
                    tube_min = 0.3
                yaw_des = math.atan2(rel[1], rel[0]) if d_h > 0.6 else yaw
                if not self.pad_moving:
                    # straight 3-D line to a point 1.2 m above the pad; speed from a braking profile
                    h = float(pos[2] - pad[2])
                    T = np.array([pad[0], pad[1], pad[2] + 1.2])
                    rel3 = T - pos
                    dist3 = float(np.linalg.norm(rel3))
                    dir3 = unit(rel3)
                    v_mag = min(3.0, math.sqrt(2 * 1.2 * max(dist3 - 0.5, 0.0)) + 0.3)
                    established = self._static_established()
                    if not self._track_solid():
                        v_mag = min(v_mag, 2.0)
                    steep = -dir3[2] > 0.64  # more than ~40 deg down: outside the camera cone
                    if steep:
                        # too high: sink fast (altitude ray guards the column below) while closing slowly
                        v_h = min(1.2, 0.5 * d_h + 0.3)
                        dir_h = unit(rel3[:2])
                        v_des = np.array([dir_h[0] * v_h, dir_h[1] * v_h, 0.0])
                        vz_override = -2.0 if alt > 4.0 else (-1.2 if alt > 2.2 else -0.6)
                        if alt < 1.2 and d_h > 1.0:
                            vz_override = 0.0
                    else:
                        v_des = dir3 * v_mag
                        if v_des[2] < -2.0:
                            v_des *= 2.0 / abs(v_des[2])
                    if not established:
                        # keep the pad inside the image while the estimate is still young
                        prow = self._pad_pixel_row(pos, rpy)
                        if prow is not None and prow > 100.0:
                            v_des[:2] *= 0.7
                    # never sink into terrain away from the pad column
                    if alt < 1.2 and d_h > 1.5 and v_des[2] < 0.0:
                        v_des[2] = 0.0
                    if d_h < 1.0 and pos[2] - pad[2] < 1.8:
                        self._set_mode("LAND")
                        self.land_xy = pad[:2].copy()
                        self.land_z = float(pad[2])
                        self.land_t0 = self.t
                        self.land_stuck_t = 0.0
                        self.touch_t = 0.0
                        self.land_phase = "track"
                        self.land_pdir = unit(rel[:2]) if d_h > 0.2 else None
                else:
                    pv = self._pad_vel_now()
                    sp = float(np.linalg.norm(pv))
                    pdir = unit(pv) if sp > 0.15 else unit(-rel[:2])
                    if d_h > 4.0:
                        lead = float(np.clip(d_h / 2.5, 0.3, 2.0))
                        tgt, _ = self._motion_at(min(self.t - self.pad_last_seen, 3.0) + lead)
                        rel_t = tgt[:2] - pos[:2]
                        dt_h = float(np.linalg.norm(rel_t))
                        v_h = min(2.8, math.sqrt(2 * 1.5 * max(dt_h - 0.5, 0.0)) + 0.6 + sp)
                        dir_h = unit(rel_t)
                        z_t = pad[2] + float(np.clip(0.5 * d_h, 1.2, 4.0))
                        vz = float(np.clip(1.5 * (z_t - pos[2]), -1.2, 1.2))
                        v_des = np.array([dir_h[0] * v_h, dir_h[1] * v_h, vz])
                    else:
                        # pursuit from behind: sit on the pad's velocity line, pad ahead & below
                        h = float(pos[2] - pad[2])
                        prow = self._pad_pixel_row(pos, rpy)
                        below_fov = prow is not None and prow > 124.0
                        if below_fov and d_h < 0.9 and h < 2.2 and self.t - self.pad_last_seen < 1.5:
                            # it slid under us: dive now
                            self._set_mode("LAND")
                            self.land_z = float(pad[2])
                            self.land_t0 = self.t
                            self.land_stuck_t = 0.0
                            self.land_dive = True
                            self.touch_t = 0.0
                            self.land_phase = "track"
                            self.land_retry = 0
                        L_des = max(0.6, 1.7 * h)
                        p_des = pad[:2] - pdir * L_des
                        rel_t = p_des - pos[:2]
                        dt_h = float(np.linalg.norm(rel_t))
                        corr = 2.0 * rel_t
                        cn = float(np.linalg.norm(corr))
                        if cn > 1.6:
                            corr *= 1.6 / cn
                        v_hv = pv + corr
                        h_hold = float(np.clip(0.5 * d_h, 1.0, 3.0))
                        vz = float(np.clip(1.5 * (pad[2] + h_hold - pos[2]), -1.0, 1.0))
                        v_des = np.array([v_hv[0], v_hv[1], vz])
                        use_planner = False
                        vmatch = float(np.linalg.norm(vel[:2] - pv))
                        if (dt_h < 0.6 and vmatch < 0.8 and h < 1.6
                                and self.t - self.pad_last_seen < 0.5 and self.mode != "LAND"):
                            self._set_mode("LAND")
                            self.land_z = float(pad[2])
                            self.land_t0 = self.t
                            self.land_stuck_t = 0.0
                            self.land_dive = False
                            self.touch_t = 0.0
                            self.land_phase = "track"
                            self.land_retry = 0
        elif self.mode == "REACQUIRE":
            # fly to a vantage point behind / above the last estimate and look at it
            if self.reacq_moving and self.pad_pos is not None:
                # look at the orbit centre (mean of the track history) from 7 m back / 3 m up
                hist = [h for h in self.pad_hist if self.t - h[0] <= 25.0] or [(self.t, *self.pad_pos.tolist())]
                cen = np.mean(np.asarray(hist)[:, 1:4], axis=0)
                back = unit(pos[:2] - cen[:2])
                if float(np.linalg.norm(back)) < 1e-6:
                    back = np.array([math.cos(yaw), math.sin(yaw)]) * -1.0
                self.reacq_target = cen.copy()
                self.reacq_wp = np.array([cen[0] + back[0] * 7.0, cen[1] + back[1] * 7.0, cen[2] + 3.0])
                if self.pad_last_seen > self.reacq_t_enter + 0.2 and self.pad_conf > 0.5:
                    self._set_mode("APPROACH")
            to = self.reacq_wp - pos
            d3 = float(np.linalg.norm(to))
            v_h = min(1.5, math.sqrt(2 * 1.2 * max(d3 - 0.3, 0.0)) + 0.3)
            v_des = unit(to) * v_h
            look = self.reacq_target - pos
            yaw_des = math.atan2(look[1], look[0])
            tube = min(tube, 0.6)
            tube_min = 0.3
            if self._track_reliable():
                self._set_mode("APPROACH")
            elif d3 < 0.5 and abs(wrap(yaw_des - yaw)) < 0.3 and self.t - self.mode_t0 > 2.0:
                # arrived and looked: hold 1.5 s then give up to search
                self.reacq_hold += SIM_DT
                if self.reacq_hold > 1.5:
                    if self.reacq_side == 0 and not self.reacq_moving:
                        # second vantage point rotated 120 degrees around the target
                        self.reacq_side = 1
                        self.reacq_hold = 0.0
                        tgt = self.reacq_target
                        v0 = self.reacq_wp[:2] - tgt[:2]
                        ang = math.atan2(v0[1], v0[0]) + 2.0 * math.pi / 3.0
                        self.reacq_wp = np.array([tgt[0] + 5.0 * math.cos(ang), tgt[1] + 5.0 * math.sin(ang), tgt[2] + 2.5])
                        self.mode_t0 = self.t
                    else:
                        self.search_anchor = self.reacq_target[:2].copy()
                        self._drop_track()
                        self._set_mode("SEARCH")
                        self.search_wps = None
            elif self.t - self.mode_t0 > (12.0 if self.reacq_moving else 8.0):
                self.search_anchor = self.reacq_target[:2].copy()
                self._drop_track()
                self._set_mode("SEARCH")
                self.search_wps = None
        elif self.mode == "LAND":
            use_planner = False
            yaw_des = yaw
            if pad is None:
                pad = np.array([self.land_xy[0], self.land_xy[1], self.land_z])
            pv = self._pad_vel_now() if self.pad_moving else self._pad_slow_vel()
            if not self.pad_moving and float(np.linalg.norm((pad - pos)[:2])) > 0.3:
                yaw_des = math.atan2(pad[1] - pos[1], pad[0] - pos[0])
            sp = float(np.linalg.norm(pv))
            h = float(pos[2] - pad[2])
            rel_xy = pad[:2] - pos[:2]
            d_xy = float(np.linalg.norm(rel_xy))
            if sp > 0.15:
                pdir = unit(pv)
            elif d_xy > 0.25:
                pdir = unit(rel_xy)
                self.land_pdir = pdir
            else:
                pdir = self.land_pdir if self.land_pdir is not None else np.array([math.cos(yaw), math.sin(yaw)])
            seen_recent = (self.t - self.pad_last_seen) < 0.8
            unseen = self.t - self.pad_last_seen
            on_pad = alt < 0.10 and abs(vel[2]) < 0.25
            resting = float(np.linalg.norm(vel)) < 0.08 and self.v_cmd[2] < -0.2 and h < 0.5
            if resting:
                self.rest_t += SIM_DT
            else:
                self.rest_t = 0.0
            if on_pad or self.rest_t > 0.4:
                self.touch_t += SIM_DT
                self.touch_hold = self.t + 1.5
            elif self.t > self.touch_hold:
                self.touch_t = 0.0
            abort = False
            if self.touch_t > 0.0:
                # touchdown: hold still and press down gently; success needs 0.5 s of quiet contact
                v_des = np.array([0.0, 0.0, -0.3])
                if self.touch_t > 3.0 and d_xy > 0.15:
                    nudge = unit(rel_xy) * 0.15
                    v_des = np.array([nudge[0], nudge[1], -0.3])
            elif not self.pad_moving:
                # static pad: centre above it (slow drift extrapolated), then a vertical descent
                corr = 1.6 * rel_xy
                cn = float(np.linalg.norm(corr))
                if cn > 0.6:
                    corr *= 0.6 / cn
                v_hv = pv + corr
                if d_xy > 0.35:
                    vz = 0.0 if h < 1.6 else -0.6
                elif h > 0.8:
                    vz = -1.0
                elif h > 0.5:
                    vz = -0.6
                elif d_xy > 0.15:
                    vz = 0.0  # centre precisely before the last half metre: the ground beside the pad is higher than its top
                else:
                    vz = -0.45 if h > 0.25 else -0.35
                v_des = np.array([v_hv[0], v_hv[1], vz])
                if h > 0.5 and abs(vel[2]) < 0.05 and d_xy < 0.35:
                    self.land_stuck_t += SIM_DT
                else:
                    self.land_stuck_t = 0.0
                if self.pad_ever_moving and unseen > 1.0 and h > 0.3:
                    vz = 0.0  # it moved before: do not sink blind
                    v_des = np.array([v_hv[0], v_hv[1], vz])
                    if unseen > 3.0:
                        abort = True
                abort = abort or pos[2] < pad[2] - 0.3 or self.land_stuck_t > 2.5 or self.t - self.land_t0 > 15.0
            else:
                # ---- moving pad: track from behind/above, then a velocity-matched vertical drop ----
                oncoming = sp > 0.3 and float(pv @ rel_xy) < -0.3 * sp * max(d_xy, 1e-3)
                vmatch = float(np.linalg.norm(vel[:2] - pv))
                if self.land_phase == "track":
                    p_des = pad[:2] - pdir * max(0.6, 1.3 * h)
                    rel_t = p_des - pos[:2]
                    vz = float(np.clip(1.5 * (pad[2] + 1.0 - pos[2]), -0.6, 0.6))
                    aligned = (float(np.linalg.norm(rel_t)) < 0.45 and vmatch < 0.7
                               and abs(h - 1.0) < 0.35 and unseen < 0.4)
                    quick = (oncoming or sp < 0.45 or self.land_dive) and d_xy < 1.2 and h < 1.6 and unseen < 0.6
                    if aligned or quick:
                        self.land_phase = "drop"
                        self.drop_t0 = self.t
                    gain, cap = 2.0, 1.5
                    corr = gain * rel_t
                    cn = float(np.linalg.norm(corr))
                    if cn > cap:
                        corr *= cap / cn
                    v_hv = pv + corr
                    v_des = np.array([v_hv[0], v_hv[1], vz])
                else:
                    # drop: get over the predicted pad with matched velocity; descend to a floor just under the top
                    lead = pv * 0.1
                    p_des = pad[:2] + lead
                    rel_t = p_des - pos[:2]
                    since = self.t - self.drop_t0
                    if self.map_label == "village":
                        z_floor = pad[2] - 0.08  # the ground is 0.25 m below the pad top here
                    else:
                        z_floor = pad[2] - (0.15 if since < 2.5 else 0.35)
                    vz = float(np.clip(2.5 * (z_floor - pos[2]), -1.1, 0.0))
                    corr = 2.0 * rel_t
                    cn = float(np.linalg.norm(corr))
                    if cn > 1.6:
                        corr *= 1.6 / cn
                    v_hv = pv + corr
                    v_des = np.array([v_hv[0], v_hv[1], vz])
                    if since > 4.0 or (unseen > 1.5 and d_xy > 0.8):
                        self.land_phase = "track"  # missed: climb back to the tracking geometry and retry
                        self.land_retry += 1
                if d_xy > 0.3:
                    yaw_des = math.atan2(rel_xy[1], rel_xy[0])
                if pos[2] < pad[2] - 0.45 or (unseen > 3.0 and h > 0.7) or self.t - self.land_t0 > 20.0 or self.land_retry > 2:
                    abort = True
            if abort:
                if self.pad_hits < 30 and not self.pad_moving:
                    self.bad_spots.append(np.array([pad[0], pad[1], pad[2]]))
                self._set_mode("RECOVER")
                self.recover_z = max(pad[2] + 2.0, pos[2] + 0.5)
        elif self.mode == "RECOVER":
            use_planner = False
            yaw_des = yaw
            vz = float(np.clip(1.5 * (self.recover_z - pos[2]), -0.5, 1.2))
            v_des = np.array([0.0, 0.0, vz])
            if pad is not None:
                look = pad - pos
                if float(np.linalg.norm(look[:2])) > 0.5:
                    yaw_des = math.atan2(look[1], look[0])
            if (abs(self.recover_z - pos[2]) < 0.3 and self.t - self.mode_t0 > 1.5) or self.t - self.mode_t0 > 4.0:
                self.recover_n += 1
                if self._track_reliable():
                    self._set_mode("APPROACH")
                elif self.pad_moving and self.pad_pos is not None and self.reacq_n < 4:
                    self.reacq_n += 1
                    self.reacq_moving = True
                    self.reacq_target = pad.copy()
                    self.reacq_wp = pad.copy() + np.array([0.0, 0.0, 3.0])
                    self.reacq_hold = 0.0
                    self.reacq_side = 0
                    self.reacq_t_enter = self.t
                    self._set_mode("REACQUIRE")
                else:
                    self.search_anchor = pad[:2].copy() if pad is not None else None
                    self._drop_track()
                    self._set_mode("SEARCH")
                    self.search_wps = None

        # ----------------------------------------------------------- planner + limits
        blocked = False
        free = 10.0
        if use_planner:
            v_des, blocked, free = self._plan(depth, pos, rpy, v_des, tube, tube_min)
        if vz_override is not None:
            v_des = np.array([v_des[0], v_des[1], vz_override])
        # acceleration limit
        if self.mode == "LAND":
            a_max = 3.0 if self.pad_moving else 2.5
        elif self.mode == "APPROACH" and not self.pad_moving:
            a_max = 2.5
        else:
            a_max = 3.2
        dv = v_des - self.v_cmd
        n = float(np.linalg.norm(dv))
        if n > a_max * SIM_DT:
            dv *= (a_max * SIM_DT) / n
        self.v_cmd = self.v_cmd + dv
        # governor: never command a velocity too far from the actual one
        err = self.v_cmd - vel
        en = float(np.linalg.norm(err))
        if en > 2.0:
            self.v_cmd = vel + err * (2.0 / en)
        sp = float(np.linalg.norm(self.v_cmd))
        if sp > SPEED_LIMIT:
            self.v_cmd *= SPEED_LIMIT / sp
            sp = SPEED_LIMIT
        if sp < 1e-3:
            d = np.zeros(3)
            s = 0.0
        else:
            d = self.v_cmd / sp
            s = sp / SPEED_LIMIT
        action = np.array([d[0], d[1], d[2], s, wrap(yaw_des) / math.pi], dtype=np.float32)
        self.debug = {"mode": self.mode, "map": self.map_label, "pad_hits": self.pad_hits,
                      "pv": np.round(self.pad_vel, 2).tolist(), "om": round(float(self.pad_omega), 2),
                      "pe": (np.round(self._pad_predicted()[:2], 1).tolist() if self.pad_pos is not None else None),
                      "pad_conf": round(self.pad_conf, 2), "moving": self.pad_moving, "free": round(free, 1),
                      "blocked": blocked, "n_det": len(dets), "ph": self.land_phase if self.mode == "LAND" else "",
                      "amb": self.amb_dbg if self.mode == "LAND" else None, "vd": np.round(v_des, 2).tolist(),
                      "pz": (round(float(self.pad_pos[2]), 2) if self.pad_pos is not None else None), "evm": self.pad_ever_moving}
        return action


# =============================================================================
# Hybrid router: the reference ("king") controller is embedded below and used on
# the map types where it measured stronger (open, warehouse); everywhere else
# the controller above flies. Both run from step 0 until the map is recognised.
# =============================================================================
import sys as _sys, types as _types, base64 as _b64, zlib as _zlib

_KING_SRC_B64 = (
    "eNq8vVlz4sqyNny/f8WO9715I/a5kMD0aV18FwYjJkObQULoZgcgMxgxrAab4dd/mTWXVCVw73XOiuhY3bakmrJyzif/7+79/M+v"
    "99/H9X73j8Xv/faf//734vP0+fv93//+53p72P8+/XO62+1P0xM8cfwHfeZ0Pax3S/77Xwf83TT9r3+2Tu+/p7P0/b/+Ofo8pO/0"
    "4cP0tErXM/70G/zzH+zvH0cYlf39tP89F7/Y73aX35+703r7/s/p8Z/wI/6b3ef2cMWf7Q7/4D87Xo/4k3/D//8L5/ZO/0n+9l//"
    "nE2P7z+eyE9mP57EcOzT/8a//OMf/35u1Hujf/u/BvXh6N9VeP7/++f/+8c/4b//E+/aX7NhdRGVYjdphtc49BatRnqaN7xrUquu"
    "ZtvKV9LwnenY+5yf9x32vPMeVQ+z7Xw5HbvnWbnttOreOWmkX7Otf3ytVdtJI7xOxu1jPDwvw2Z4njUC/u75/9ChJ9FgP2uka2W4"
    "Pv6sthX//mte8j6Vfzsw1Mf0+XCYRdWv+a6/nJUqn+/NtjvbDQ6zcbCcRN0lTkd55zOBKb4Pq6v5tfrJhp4/7zuDetjoh2E39L1o"
    "5La7A98Lg9qz12pUzq9b/zrf+pWOfzy/fjwfWy/Pn93aedkdPp0jWFHcHFTmjcCb7Z4/4+3laxL1b931Br/ZD+urxShIf0WOs3wb"
    "PsP7Afk5G5r+2p+wX1+7tadzbeNWx743HAT+MHIG7cDt4+9Kyu/ifpgsgo0fDIYwxZfuZ3fUv9TSXnsUevUQp/1SX75en93u8+Gl"
    "H7g+PLsYbLwxfM9nQw+u+NgGV9TpO35vELarkeM1BmnPh13otV6cJQx56Z7lJ8Kwuhg5cSNwwkCu6Nmtbd1F4A6qYT1cDOopfZdO"
    "KeiHg3bf9bojp/fKho7cpD3aDN5C1wuGQe81IFMJztnHI7ftB2HYDjZhPXJW7dDv4pC3Xv8wCus+vrsY1b2Xoev1hkFlEfjtt/7G"
    "rw2DCzy7YdNbdnBqYsPTl8i5vPWdE9t49w1+9jZKxYrd3DQcsunVcJPg6nDoX30nXIzI5k8+YZNz0+mHKzigcAy7OuIbHnqjoB76"
    "A3I+QEIvy064Sbujut+E3zVHjt+N3LgdOGltBBsJtDEchL3WyLnU2dSeamncHtUv1QEMD4dQDTa9Frzj993BW5B6dTioPuxaFaYZ"
    "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"
    "RGazKR3se8PzoYgWJCKQlSR0F5z0FgnaAC1GOXvBSDlZ0udCr7U2QrUQDyBonPTxAUnWeS87GVPteU2pfbkmNpcw71Vbi3kTlW8Z"
    "Vk7sNgmPSTet2Zu1tpQztWipgU7Bz1nTv82Yaos4LYVLR90pZiLKMbrWLLsGdUJzCp27xzWj+HVEm9nRPqns91lJ1tpyW6t4OoLC"
    "87O6Nx1hsKZqSrM44/xhUW+yIYj8t5yffnB5XzqbJpFyas4wF3aaqp6ZQhlWjKu+qtXF5mnCRt/mJZICd03GT3viOSTGEf29DDaR"
    "1eh3lwYP8uktlum4x3OUu/uXFOzrTEDDg12puoKldClYhdGfKVpI7BGFV8lbyXAt65TQyCVg/5TlLE9dkF4C9jaD6iOVpaxBZNgJ"
    "a6l4ZjjOUuDCynJg3n6ArlBXbAwVhw62BpyV4m3L6EHUcMIPoJFiEoEzbRJZAEoYLXkQjRiCjj2XkCfnhiKBE+Q34e8gFjZMU3V7"
    "/bvTwITe9L2BNSWDvbLqacM7xjVWu3EzTiXj9zb3bMlnb4neqaZmOmxTe8YEP9HjoepEjmNMdZTA7CukCydpttjhnZct4Fw8TIdg"
    "z3OM/gIf4VF7O2UvZqHgwUAPvZs1B1FmUu9RLchCl89CniLpobb64NDqK8ZVfwvDlAH/25I5Udpg0mbDx4wNUB/S3bRp6XIgSGoF"
    "bGOFuWS/47G3jYkf5tn6jrVs0J6cmUnQRVaDivDHtERbv+X7LvLc1CqWlGXbRd3HNMqiiTyCOGFB2y8ogpZJuCn23TE5qIszpk09"
    "Iqo5dCix4QbyGBZ8wtw8p4gE1WkSgG+lw3tMSMJ4rueo1HZjIBwS/3qpZ9Cf8iuiSQXF7Ea0bTWc0dZFrx6qB2D0hlggV9DCMz/8"
    "ZBduMZMrR0euvVNVqDM8gbqLTLT8+KfvfVNmxWstnYruL6LfT0onLO++c89FFMlIqtkm1Oa+TEaeDXL4hqrDfM1iYbXnNYiWNUgl"
    "UmEeR611EZcU1gcIQroarEBFgZgmtaVp49N34M0TYK4cbJ8HH4m+Zr1UeWB/edad/zwznpPnYJHwVpEFlC6KoE2YR/cidc8FMt1c"
    "vKFKOx0o8l7biAKSyr7jI4bOCQyoPHxHNqb53bYS1bsANEXCFelK5CCdQGXvATO0SypGhji9v0imDnBA+Ptqap+WNYVmIGFvzZwp"
    "/v0ISl/HdPe3J1B+i5mprLW3kBiG21ag7PSLJBjrZpF+YMMVkl86rnxM0fVToOsp4oP32pqVMBsngXMbfCW4mf4peWQ4BQaAkNZY"
    "6w6e74IlhCZphiV76UWEi80io2aamWbJL8Van/sC1mIaWsjrK6azkqzHerzC7Ls76t/9DlWWthWyHNgxIbmYP+Gu5iSQ2Pttu7Ps"
    "mfN86wFP77miYeLukMLPbp2GSIkBQ7TXD6uLUd2vDYLLqPMd3YtFAOfN9tckvcNk5bOpxCPlPrAjiJJ0Vis8L/G68h5Z4YSbd7z4"
    "vWBHpM0F6j2vt69eZyXafQZX/A7qAAjPPZh7n526Ww18r9sP2v7I7x4e8per8psU01Iy5dwspw7aSYys1vfWCchs6y1gqTLAyRzW"
    "SEc1kmE3KzxNopWJvGKeJ6oNMxJkJBvEDmKZPwAtOWyv5S9hNTtWoidgHL8FZMeGYXDxI8dts6FHqdcPNmF9GPSCYeD9Cq5qxhRi"
    "D1coGW399XR8wfxgbxT4/chth6N6uBgF6a/IrQ5HTrva36TdYTiI+ZR0TaUqyFA4pRGmYWVsbkWSRuj58z4/mL547pB2Ys96I2pQ"
    "rCZbWieExTt5BYpfunAjGx0y7w57Fexi6mzSLlOeikVuQpIil8ra1UU3Qbbehk+1mFTZkKpiFCl5RzQ9N8QeJeDtmGci7jJ9L3Mw"
    "y7eaiAJqBydWTR4z+q9pqefTkrvcMp8+8umYQ210F19zOyaEJjtDzbkUlVe7Cdu0adTd5y+aRor0BjS7GoV31nlyZWNxMjNR6yAY"
    "tPvuoD1yKiPtAtFN1lcxcgkjDtKwOwi93jCoLAK//dbf+LVhcGmH/oZP6WAqG5SbgyVlCyXz+cF7brjLY5UWrMmc5BV6iWAK6RGj"
    "6+zV7Dl1bDUh2TMm3mb29/lZZ1vcRzpON71RbwEK0BEsD+p0iGIeStXJTzoxzXn/lM+Hw6Dvdl3OvS7AasJucJXOSpGNg2RVxSx4"
    "uJ/UkGGuuVLMGtVmuJLICzeyDvIt7r73SxOskkBvIV5gWlR9sDarlZfKYEuL33FaeG8knFMd5lcq+wN/sBjUw2AAzHnktqUukKls"
    "4gk7QtUjn4HZXVIwfklwIdkGZvctP18DyTHntZFBi9xC74opT2JopWAZmSpbuSkhW0tb11dvSfaltDBiQ4d1HyRWT2yKLUvaIARH"
    "4t17/CDzHQlnrV+qzGf7jvdrsEnrg6DyAiwjGPptP3BWb6N631i/p10+RkckKYHXf6WiY6xsTiuUXKJjmy+XVR//BFOfZll/Rw8n"
    "JkAZfSmz8vxx9V9RoOffsFSwCEOuGjPrHjJYeNveBFR5Cf4bFBq1ytA5JG0wVmOUrd+wHjgEbqEfraBPm1i1qUktE570smzBIML7"
    "mwLLYFxL5qewCG+WJljsE4zcdb6QQ+Yg/XGCACkXZy0b8bMIavSOLgKQz5HbC/rhAHg4XLaRFUmbvw7C8YmZbqjW/4ij9hesVP/M"
    "N2CQ9e8OZr92WsdYuikVsI8PqR4ohp8v9/fBaMTz8eGdnb+6E+PbQZ966Ih6rgdSm1hCp9RCDZanbvCgBROm7NKR3wN9UAtVpjUG"
    "TtqD2QSR8VXiIUwRlppn0yXrzRL5fqs5oeFx/bA+Qege0dTv1PTGWgnchGQc3rDTqFIBEddWTNzzVKY4nRDhiBmS9BWeFpVJzBVC"
    "NJN/UiLpF1HXSL7cjaPkgTFey5D6QmF9AIkcWHaGUPOVm0GnpKsTZ27BsqZqmvCUER/NyGHDAD8GrYLe9fyFElYFbLwEGiJmAK10"
    "UL9TwVIjjItu7HikfPPU2QvnBbU47EYOJUVixRhySmEapSPC8xAthpMZCMFBPX0JfQ9+vFqE8PfhNV+rkTXPNbvc5PtmGkx2x3CK"
    "3PC5PqdMexDGbuSE40HYjiMQ+33nNGrV6p65+IJpqBoTlZ4J9WBImI0eghAfxI6q72WYHCldUPitI3P4WcHjxnT2X5Ndt7BPtroL"
    "WXwzdWhmU8/L4YkATbGVmCP6ORs831G4QVx7xJxANsOHtvu7vlji/eW10Fm10T0HDf8Gmuwhbi7xUhGkTubm3bHaAt4xVs5G38gJ"
    "ccf06DAa/qyyAvekDLXSPcWNHvrLb8JlgK5ANC3BGmFDE6Mk84mijZXshTgt1/H4jKvj71NTsNCLmHdoZDRTkoOirnxZ/DmT720S"
    "xTv+b7UukFN4UXqqwaooIq+3D7apVEiCTi7sbco76unbKO12LKXftFA2c05osNYzaj8zGQz1epiuyg+A0AQtEZfvs6HvqPmaSl9Q"
    "LoyGEgJfCO01E2DWogi5UrLnva1UTLt4Je5JEisRXsCc++9sttOEG4eSEqdWFP/0vML24CW4CT/Ln21wZlperR+0haOyngZjzYrc"
    "/LCvvkCEWFy7+vsUlcQQdpEqOxopx73Zweyf4RwrynuEw2GLXtBQDjEozSJxl7t8o+p51kyFJSI0UnKOpPZmvk0395wYRSb7o7Z6"
    "UTaOQdYWNiAu9qOz3BXWdByeU3AVDI+rqMiFkPMss+ohcEkObiUTBozoa/eB2fPpEdZEE94kjf1ekVzysSxOJTf7QCfzZbR+gvmg"
    "WiqMNAMzz5RoEKJ1VBHUM4ByBrJisc3wCSO5SnCZfQ5ERFEjFR2B+eldhlhvwvBhKekcgQ+xMsa+yEMY3xD3/RudbjQo7FVFAQ4W"
    "FTJSchVCU783NzYkgcxzKYe4/YnFdkV1QYrfjKCC+Nqm8Ljz5btU/ABEKgcjgjMrw6ZvcHPy3cTgZ7Wftk008ANMixvgTuUu5OuY"
    "p8kIZfgdSz0J6Kv8FflZMUKfTYUozl3xT2Q6AmHg+2wi6pvy/AeHhMa+R0E9HfaDyyJwwmqw6bWo1tIOQApWQVsJueEDSoxREJYu"
    "qamYisHr7Kdj+D1cOjgMF1b00zik41bHIDj7waAdhqDZjKyYV3RYbxsDA0042cnP/9WyoYDJDrMfGH5FFMdZIziQjtAqGfcfaFZ7"
    "JKY6sJvPDtsBUovbbGPmpIOkZsGiPBS2kGLsScrrHJeCywUyGb0Nd1cpKBhTmOcE3nhVGd8I3OJXvI1JzwBByoyGpCv+sbtavCpB"
    "ZmhvBW61HWz8LpCXD+T1MlTOO3AHvrCvDWQmVLjkCLL4t5mE/F8DB/QuoN4BBh02fj0jfDXGrLaykFVsNmmz7Fzv9MJuisNaoa9V"
    "bztBen78FoXxihiS4kM753HZDOgahUfsNGkLwdhWmBG+6S2SyZx5lbBHvMAxcb8a9TSpwZRpCfHdKlXdsaVVnWt5oI/m96cz3uXZ"
    "tBN5nDupVnLxwWE6lKY3tugOTUFvf81K532xmmjCvkJsarpEmVHJCltB/XvBizGoohcJTPC3vnNagCUyAd7rh5twCNQNl6PqD4J+"
    "YXjlHvizKBu06t2PYZIWKNNGBRl+x32kluhccUZ0Bn7nTvgtu3JJ4U48ds9Js4vYVzeLOcd3gKrzrnebYJbl8PlfWAKqReszRg/c"
    "lBMyZQkOHss0icsqxvMWd1OtZhFKro7Omm1NUtZZi+HwZE5a6PENn7kZ/ZqspDoeBpduP0h8hQ+PeWR2UPd7gUEOG887CweQbzXy"
    "mDJUvVMc+Vj9j1btQtjErTM0QZDbejoU9XHSCqJzdldR/2tUjMbBydRmQm6krQRNqwH7yvbSxqnzs9Y31JnvQtZFVF0VxUkyGjcF"
    "5IaKdLhJu6O63xyE3mgQ+MO+s+KXC7jUS1j3hpEzqI6Y3B2G4ShyDCBUftwe1S/VAXI1qfqhnwTeafssdr0I/XYYuIYgM2lZcjb4"
    "zQ4ZkkIvYMlQRKeXD1sgtxDwwEa63Kug1t++dPVcE2T3ap/jAsHInlVvgCEQQateFNCSGcEg9dPX0cTgPaKPa4WPzMOfGQohHNhO"
    "mQ0j7O0FuoDIqESSAmMlQewj1JvVIUAp7pDzwR4v9BmqFOenkyfB9hGk14ZAO2QupEBhNd1L/lrfZuBi8OgzJm1c2YW8A3ihTlMp"
    "L1KGwk3LfvZRoDmCMoLoQFF5r5jyS1unKtO9o8kesR+kGPEBfdq5tOFnQV5pPi5By3QxE76zft6bv4Wqhv4tkW+WBggk9J0p6Iew"
    "yUqp1WTXFeFXdaOx8xnockIPZ1BLWDIyd70fRTlJWs38Y4fAPBWqS+eo9N1DeNLz+1AvM1F8I8WiormqZLRSXPnTe8j5wdljF5ep"
    "hlUhNOGS3L/b1NmRv8sVohAXoH8pU5AsSQEZy3KmnNeIcCLFEfWQZLtbhsDs6N1Mb/GFzY+Iaa7SQZH/LHtYybhyIBKwSV1B4gD7"
    "ecRdXBUIQALg7pv59rzsXzEdlUFyEUgOWA12/tbudGet/fuQoYWjiHO5B55Zo9Z55AXm0YKujVMOP8G8Q9AxHBbPVwC8UzoqhsdU"
    "PxE+9gmlOF70+MrfYwZKhgJ111u8r8+2OJcJ1tJ8vg8ldhIQK2DQptJvjnGFqrnQyctgXWyzrSSMOnkGxatYreA7ojYctynCGTXC"
    "jOaV72Z1T60waKQGrmRShAs8v2amq6MD5UwAuHueaQU6sCcBLNFg9bK8oEOmplkvfVAZqyOehgHGssioTF/6wdmeAqGt/Okx7NFm"
    "3stM3TrUxOBaii70jDz372a20vDJ81xLDqheTkbsrdt0nHx2dKF4xQaJwBesmy5NgCHd9EtiSzt94F4b+oOQHSRNMgn8S5P0ayS7"
    "KqMApxzrb/hX0s7NEDLBlc6jNpYhSNhErrWq1gl3VlLHNKbYYMi9Oqo9/8XPev0M93dj4UZFh1CdWRycYrXopkXpl12OOGsrwK/y"
    "Cd/4iYJ3C+lArLr4DrLLZzD9DMOavczHrK9Fb9sq4swxadeWzQlegdgXPByLpjfAtRTfChjGjfSEuUpvAYn2af4W9jNjBUQzWc3V"
    "2j3dI2hMvmVZt4ccpjArphLYwzTerU1XYuMcvlsQWRQlyLhi0znuzJhl2rJLquhmFhesxbeilIBnV0l9bMpwZtdePsGPdJ3K53c/"
    "Hx5NvuZZ10W7gssU3sL+Xkm4PpjbhWCRLC/OoPhXAnCfJWpnuhTmh8QsbJbMzYbOEL4J1ceWJy59pjsK7ZP1KodO+mtQ9yKeKGjP"
    "qHxkCBXkt+C53uw1FClQ3ZHTe43cpD3aDN5C11NykF6D2yGxprHk7ri8rxocU9Fz7vGk7ghPdTNtTOp0ckmYJf+GsA6T8cDQTkTz"
    "n+cS+GH4PUaQeNojV5BGxR2+7xRJ0yR+lGzw6ehaPczW9w9E3Ote8t3GpHwVOffs3eIMyla45CqO7NxsXWts9zvBrJuGVyYVFef7"
    "jQ45z0YMhEwqDONQpqYMBOzidSh24UDMeD9D2VRV6I2YE3MECvfdzpL0/Nh0Ph8qPbGsMhvlz7fyvBmKbqQ0swWU755vno8X9WLT"
    "WrXSOGRxAHJ5uFNqpClenJE+Uj70N1wo9SYpq8Z8/ymt9zCoeivM/z22mnCmV0Piz5Zkxm/esWUMpwuQWglt8SdCsvbKJk0UGKUX"
    "GbogtOZrAeZ1TJHBjMVW2lnzR11gD8ciEcGLrwzD8CTPDPJAo52Shg2YxMv4eqZFL1tFJ3u5out5CeL/MyG48XjJYrjb6cZk4M7K"
    "4W6K1WpNVBMpbI+Jrag1PnBGtlQlPry833Togr7mRb1hCBvigHLD6gcN+La4bazkBXsvQ5fq5KPAIxmxkZP37kdmnzltL7T1XEQB"
    "m5QuLmx2ivDZRdwMcYd3oNpl0EJ4eFWQDjGMXCw1WCFGIVocQBsf0+sqt/nYJ+QdmPTb6PkgW2+LH32j5ROo+7QexCYimP5NNjjb"
    "XVrrXmTy+OJlQnR0vsqNParvuG8099vrWfuzkZIkekEFhsZDOtkj5ysxNJpYxrC8kBw24OEI00NZDb28wnmFqvwgnQ2NKYo3raLx"
    "/D3Zrb4/LgHHY2zHmvRFZ0qYZwM9fQ5J3IZb4JImldiChs7+UHDB2Eq1iP99lJjebDoePEXDu/kIBsXKgBqjRRBcUCufFJed2BQM"
    "ibDmdnKjWcNxy/C4Qn7G/xJMk5UlkDym4qaWooJhhEZpeBPYVrbOkeRWZFd4b5fku2LoFZcypk8x/QwFY24zf5pEA4E8zw+ruXEN"
    "ZKatmITP82gReYklpB1hM8XPUh7BlWHat5iyDRsAP52SqQw4+7POIwYxz8ZR9GY2RG72SHp8eNWIUZjjj8dtNl+2/OOigJ0bA476"
    "xqcMF8pMA/buRYQ6mZ72rVXYK87NGy8DEBQmKzNDi+bCFCThQUJgMkPmRrFLQPhIVc+AkzdcFJOc2OKW4pnM7fiXQeUYDkI/xEwO"
    "NvTA91rDcDC6k3FVaDMX+VA0cyKHg2Qhl+/llOVwznIHKE1B7lXQLcJquEkWg3qKm/Wr74SL0Z0hSXtmgzdxUiJ224ZIugzegsGD"
    "9IBjknSBzZBaBtwmPMcUr/Zmw2zgnmGlxYTwi+UhWpRKU0yfkJiELFVVq+sQ7bozblluyeTqNM0ABgWxrOLemgUHIeU1P4+COi7m"
    "ePRzlL6fgpLGK5YKanY75gamim87h28l62wJ2JSwLIreQceGQXOV2FiiRJRCZBWUBK7mzSrmDuUu0BTooMNyRQ26ujgYS5bdHfIp"
    "ri7mgEXF0QPmCpAXLFuJzMvHdCwFEVgwtJIQVamY4sTSVNXG899Ka8wGh/OiJNtcJXehHqgDEzW5VsU1m42l7EAttwP57PnvJnNu"
    "jB7/O2dJKk9ftcDDeam59/VM6q0E59+ohXP2BjoZIKImNjoboPF6ThBnoVlXHSGCDijpnlnPlyVpgifTlVOl2Sxp58khOca37o9e"
    "quMEJ1GP8nSK2JmBRVURpBiCK+j2rV3VaRmU4SbCmsLl8r0SsIQjKTnQ4hyqu13HPXu/tsU72BhR74KCDiwQGaRYp3olvQdqT0+G"
    "Mn/kuQLNnpBcks524UEixyhueWXlzD5Hir4lzTYBLxCp66EWBbR2B5aQ9FjUTsCbZdAo0/2AnrEtq4PHxizRIxUeUyMNgpyNHoR1"
    "9QVRnIYhQjw8/SgsG9LercAFe953LWE5uLhfor0jCY+WYMMPmOcpGo4iLqVhJzo0f/yRaa1467hspEDFilc3+RHYSw3/pkoc17m0"
    "CoKhYGJNogh6ThrZEXycE3wC67eegHo92DBsAUj/PjQVTGbLiGCHdgMKcX6nGEvWaaqbwe4zbdf7zYIrGiG85jZYNK/+ldJv8orF"
    "/7yW6xNrCb5T58OHHq6cuZ3VY1hFlikwNy06Pt5qKqlV/H7YsuSikQbkZ6K91p5/tuq9NwEK2muDMoPJHMNB4Nd5hAeHs+X/ih66"
    "oilHYIRBztFFk/bsFDx8uSei4OX589eLrtrZmtQWoLJmtVTt4nJjSm/luZqkoqnC9XWbYqLfAc7q8L41KsRw34HTlcIN7wwtGp6+"
    "1I+itYGAdXBRg0VzkKBzC0Z6XHYNnkJVZ9OAI3WAd+J+Baq+GZewixHB9y90gIHIuSHvQMe0UAu7mKo0GgbuYuD3DcwPWUG8kkl7"
    "RUm8gytqMOq0O4ZlSW/hF9Z05YZogvZC6+8d4rwSTDU8xIgyUW67wGxvEtlxedCfywlYVEVAk+lzKNSFYVY0YzbFoEGJNRtG7wE2"
    "I8Udyp9zbb/k2Fk0ib/12aVkBjvk3UgfNrjfydbXsuKZIxLFuCX5XnsNRb5hajYRolN4EWq6XvlgrUTWBWa2xssTfXQtz4gCOhAf"
    "L63Dg58HKXVArQMpFSXWbwJWZOhzrXAydva0LBGTDrhGiglZeN+amIdwJCAzvaHuORAw9RzOHrsRvuyLpJPlsiEunqjdMyX5fGvV"
    "2WiektlzB2HAurIhouQ+7buj52N+03ofeFFs7EjX9eRG8x0WTbLYRmd6AjAPokahBYawidQOs/XyghmY2eWJVdNVdq7Pl05//6cb"
    "bq7x3EmrRjUJJZz17Z53VxUDqu2N1gezWljT4SOH1Zu9jro/IstSeHTvG6AjlikYizfUqWe/qwBZ/OnnebqbsFrssh1WGy4CjCz4"
    "g6vAGQ6+WRRLi7AeORxVRquXMJ/0RWCYKH6dlFoUpry6wxAb/Mx6Vy3ynAERBvuk2TrYXfGiuQpImr4dkEah2tLla06pPZ3X8p3B"
    "s8yU74jwKmQ3BozRj+diG4tNzwyFmj8Ekv4IShfoceQd7g/Po6hamGDxXVUQOWfA37Gd9++Wrv7nnVdV1m2OxKv/iNLzAEa2llKI"
    "TDKXOcPPe9EgyQRKhGwij410IPy79kQ0FOkz6z7U2UpmSovZdNbFjbCA2cJzRI2UO3Xem+VzATQXVwvvOCBVmKw8PtbyOyYDd24o"
    "YL+lOGrfSJc5WemwVzZRqvTyWePhGLyNCwQ8iEs/uYlP8lSEMizSVVRyO9n4L2W8Gwtck7mvV/a2SAonUR5SxFxKvrgbNmukKrvA"
    "rM9cN7KHp5vNsttVtZZfQL1lpF5DY7TfpBAOAWv0TTWW9sslkaKNC5p8sibX9Pmf5PNMPXxQOmVbjOWMXk6GlgwNy2cMeNOiEOeM"
    "roGE+EyrJdJ4gTo8tSAUzyNXlGFxVmqFk3DRgsHaaOWCRX+MzJrpSvbNu2tHX81dQDD54fgi4Tv7mJUqHwqw84zm/83yzJKf+3k5"
    "0UrK6sfudSN4ubnfE+3vYpuGdF6J2WSmYW16B+e5VhG2u8OzUjeQJoUp7rvcWfPPdkiBbHB+BLtOvYzWXUuPl186po4SYjvG46f9"
    "vZXSKdXtUyreZO2ZPA9nj7Zn3ZGeJU/6oyruWsPnSUSvOwoeAgSW6E8Ug4xAqdDNureB3ZdDQnwm9md+vI7qmWbFtLO4oiBlTXea"
    "DYfhkYkh2Ss/zfFtf+qGjgmrVBTiZWWCVBUIJnhnbcwB30TlmNpYYpNPT52bnxjss/N8S4RjqVWHJTBYJ4VuZu+14yWyZUrzODSp"
    "Fidlf91RSzTKIT+zmfSFWZhssxlcMmcpV4QypugP2sbnZh3kKNVGogodGG+MsLnCKzsvDSMB3fJRSfn8UOaNYqacYWrSda8XYKoG"
    "FOm/qAJZGAMFOERl8f9z92bNiWPbGuD7+RUV5+neON0nJDycpCP6wWCEAYOTQWLo6KgwYAYjhkqwGX59r7X2LO0t5Mys89BRkVGZ"
    "tqQ9r73G7xtpWyotpcSuBbOBCim5D0ZXpAW805Ig+k7NZUmlKzseZWh+vluEuElOoyc4ZG23NExemjT69EgToVAqA3/rBzec1wFH"
    "KRHUzZ93Zk0qH0M6sGg22YB8tyH4SQi1u+GAiDfEv88TxKEzZbD2LOWmXIagTnIWBf07n1SfXzXWnd5RnE2HMWypUfCNJE2656iL"
    "1UiYWn6HipSo1aLPat874U5vGnd6vJj6xXdZu+cfKC5pbVZhYliaPaioPX0S7/R3kFjxqEyxEft7y4eDkuHO1x2jOSdKudcj8jSw"
    "EcmhwIjFOlPz6JDGgo2qBPFOFjtOCsGanWEq8TbXPPB2XKm94XEMhhKibSWsZG2whfhoHa303OK+nqCbrT+KM4vXHb1PpT7260jY"
    "sButh6KLx2aO+uvEhA0ulVTcmk/uv0hHf8zAz5BOSmKZhd0/8hrawopzbZSROD0KKa4mWAgUooRZqX92ek4mgWnNcyXbts3OZioE"
    "efvZ9nt/JZhMukKoLKEfBgPEF40qoSNsU/oYwfY0FqvAoDY1WB7914TKaWnymQEaGEm88A58zjghDFyS1+uLOBmpC9qzEplTe4W4"
    "ABAPo7RLgxrcxfCcIckY7mjE8Bb6wR5JWkZlUh+te0E0L0XK9R438jd7YMXvK5olpOlOqZfQvExhzTNhxzmoeZkj+coQ5OG6OpIf"
    "YiToM0+OxKzLZzho17oj6z5Yr55hsppdifySzBs6j0khQhxhdUnq2zOB2kcoMVx+ExYtwxinhNGRyLKLpginFnYQUiVo1XtRsRIZ"
    "RVGudQ6Tl6QEqkk0eWBIQ22LQ8N5/hiKgI+VTPeC1cj1LELviPvZtrX0BRDSzLktQN+CBbCp9vm7al93qZtZmsUrHpptdPlhSYdV"
    "8jfP95CzDOHXR2Ifwl4M4Yc+BLNpk4WocNpNyou83dnDLJ0UlkYS7P30OURkN4RI7TPABGlfH+egmpMiiy45MOu2L8tSG/+fPDCS"
    "wPCmBMozEaiw56o+/zQmggZIPWNN6hXtiLXmzSEYL7rYJue5lVF02I9RWaoijHUPLszOGvE1OsFkU/+cLBfr2op1i2sia8yOHz1u"
    "52+FRLeePBG1b6TQGu25w0RRcQO/J7Ks4nlavuuBHndBZ1fvpsSZ6OyxLvXu3WayDkSGxmtQ/Jw8YSZkfM8i8Ah/CHYW42z5nGza"
    "H52b+mJSXXxPPNdEM2Fcvvsexp1ZJ7wrtVfIchE0ew8Wy4OoS4oIufqhMioxXom9moGBehjcdLYs5RzRBB7OjlGIwCJ1rdu/Qysz"
    "xnTXsBBdJgWMhzkrlz2Z9LVIbp+dC6hmgp/1+SwV7i7fB/pkHp2JB+ObEbK9nycXf1xbtdoCGyeMbsFY9Vk2vHqsFntXOBRL8VsV"
    "mn5qzzuDxTuRKjEcDawBux1ebGpjsEM7XSBpbzCbKj6MQmawutCAaqvR4pVMg5I3PmPSR7CqVYovvXjOwi5lmWcOO/523iVWqkUM"
    "JiTb+QiMcobuPikihk0HP3fzCgcMe4tJXWhn1yggWAFx4UrmGi5fC1E8QQ4vuMVgf8j1hjv+Axbk4618hxn3i1oVobxCvEZu4eBJ"
    "2Ntz6T9JkTLr3i4tkz0fwRked3Gim4nb6e44HXQoBeqtEH9MnhA7J/4Yrb99wDAOo95WPIP2+Fo1TSW/+q/GopKcn9tnBjK3G29K"
    "PhyObWOQBPMOfAzZjNQzuPv/U1uzvGNkOKlxuj/8u5DhK285a5sMwHR41tOZPppJAcQIpjM9geEziHYsb5SNUIRppuWFaDoNSqV9"
    "V51r+nyG8HweDWJNYLq2VunwSrN0t6lp3cOqVOrehYQqNa/uazm6WTeB2FUtwgjUBIJc9mzrmPg5r7/n7+IirEefAptBb7pw+8Eg"
    "lWQzybXEkPrSoNFecpI7qrOV79lcswuYibhWOcRv4vCW6zKtcXU3rFUpfXyuWNlryOyKAvFjdNa2Ubn0CbMDM9S+17j1UKguzbWF"
    "2+qGEKIW4z6qjw9LUDfh3fZSL7Qp11s04nKN3LFi8vD4E0wP/L6WMoLMT391BkQeqTYRPz0DFdaVWgIwUlA/8kAF52Yr+TIRqDWW"
    "Zy72PvhjRiiFpb0UV699Ahwz+TEdcQ7+Di9xWLFsraXBVIX5ReONKMBIvBLtTwiJSSSHtriVILSEO0CSIQpqzxsWIeozB7YxFDOP"
    "VLs4jftYSjQrNZwSwLoUE3+3schSO7Koiu5pk1mObzUyycW28/eJA4YQqLTNnE0wDmw/9T2ZR4qfPRxolwfFj2HfjwfdO/q/Y6TJ"
    "tdf/bVWEQY7blWFTDRSUTiwnWL6SowbXSvXo2RRoVXKy1XL+k4/u0qYbUXUmnwNzYYGx7Iu9Gk6caby3oxdZndqpks5c4ecbVIAe"
    "HykTFSOswV6BWfAfNQtKp05iyYOWuuMF0sik8f5avhM/29bWreMU7gmLo3L6VLchO7FdzGCvMWgM5sJuB3o4KUl4mOBGO2LxtK5u"
    "NOBQMgYFrHhj6a/f+514upblwCZVxM+reikdD2avA79DhyZ2GUQNbLn2PKrKvJRhYdEb3tR32JzlDl7DZ0ANOKRGhUpUo8t/j906"
    "mvYZm6m7GIuqEziYCZGiPs0CEbSFrJWrsN7wu8JrnwUgxd8TvhdaDLxeYII30qrhB8zII8UtFRQSu/a+FrBdbiYMwOdicwawCyFZ"
    "mIuPIaI1X7E65Q5v25r90mH5yinBthTvntbkopi8+By7XjffpO0l7a0V/JyVk9twpuVaaxbgiFmUZEMh9PEbzwmHA4IRvD2TwXXv"
    "eTU9w6207SWe6zEL9aPn15Hw9qUd1pudqFjqVZwQx5oZd8HqB+Lbge7ANvFBZoPoIDQ3UoqbjhkQmXasa6geRnAlIcmGvAm3rlIy"
    "oRY81baWtf+t1qes3UsboX+39WkZtVMy/YT1iTP32p/yigohYpgFKqm37Ybo32mBCo3UYYiSBZpa80XxeZluMkOgGuafMCmVzQWW"
    "JdNKXpb8UWmsMJwc82fKgFGWizD75rs8pqMUpMKC3C958eryufywHAqgz43HsuoG7a143YI9LU092Y3rDo1uIbrrDOqwi2vb0VKY"
    "AbrYxxITLhTtpoCYVDCGDuku3KTUCVksSRaEsKGF4+mOIaVLI8iiKGF2jjJ0eDaOe6T6EKVpn3UYaLfj3Y2jW8D2Q8bXC/yf06pz"
    "JTdp8AjtVf+muu10GS7TjcWordTLtrv3ZGwf8fdr3RCXpuoNN98Qkh5MdgokiwltYQCZmfIJTiZpFgRFkYfqNdIk1IYUVNbHkdQC"
    "ChbhGSdc933aYgwzn5ExbJVV7Xxez4rXQt8vLBfpY1RlJZ7i2sBgxPdlyZsOYL2jA4d2OLLUpnWH1dxH3r11suHZUaCUMEmIlkg5"
    "pllY8DVs23iZRCH89dpdhkJAz4vDhrMgpZk3r6Hte6ZyA0YgLyxHhKRmxi4mCrHnMg5QBDJ+2o+UtwmE8oSq19GTxH1oT62tSuaM"
    "a+VF7TmY7JtdLAGeohf/P4nRc5M/SilQoN4f4How6P5gRt6HDK1RIm/Dz2bijxa/JqjLqLgZ9YsHC064DTPBwpig9oupVqSdHPLS"
    "jNzryj0EuSgLyt4pDVvPPQxinTfsWd40QzavoNo/Y0AjEWLjtDK8h/1h/+RfVyPS7iB+SNXNxbw7eDUgEsSwAGc8Qw0A/fw4IsiO"
    "NkVz9O7OMN2pgEbwQZj7ye16HAxkxMema9kO25AqU82D0vc4THn59uPZTJHCbUbiZVCgPFK57fBn0oPESBQYEsxx/oIMN3bGgz3m"
    "D06eVkUbIA3u9MFNZwNiY0moYXLnC+aUAyGJgC4gyHQG3TvME8Sy720C939OKck9PwW1wmAzR3wmTGKNvtcq9cKo246K1Z4/goWI"
    "npKcMDK30EatLF0zS62ARtxS+gwlPApDA8IDZkB4FAzILjXhW+Ut0K0GJm/t5p7hmtUqGkvewPOSzfO7f97IwLJTXRidx+jv2kQH"
    "pBZRW20F+vk3/Heh4eJ5cXxDzNxQQbX87KfYtgLh6ReXIFpAeUaAoqHTQEIgEzi4Hxi6ETdXVER03ONbd0EHYlqdgnJb2z1vOp94"
    "S02XKxfjgfAQkQ5OpuGgBDcb2NJxdneSiECaw6LXtbJaWHKEGRKMzs+mHJlU07OAkW4xY37yFOL/aZgSV2GeYp1Tr1vI7ozf5yEi"
    "Jz3d6Kas02QineR4eX7ClJhmN8OJESDaUxRruruutQr3/FwIaCpb6FV2z1iydgPC90aRZKlD4ryTeaULAxyLGACZDlxhzES8Jx7t"
    "RGUEw1O5mcKWPa20+9ry6unlvTN1YdE+E+Jq6e571DlP+2F6wYLiDWzB/Yhg9UjM2Ca8f8HMivZ9y78uuykX2DqaSurz7LuT+2Yv"
    "SA1BgRtcHQnBKQVFdoioqKbuuYqvjFFrhVYoUsBmn5laClExWjRJKpii6C1zv5+niPzx2PxoWVRD135I1hokyWqfGAIIg0aLGVTa"
    "8miuW/loW8c9l+ezyRWrRNdmTCy7fVo9+PssEnGur732my0SHdXN9PhyFa4+I6/CYHQFXo+VI+jvMTpmhOapH6cWHS+11gbeEYFH"
    "iTJBBpkHFgmuNZtkK0bhc1lPZ7eDlXGmUUnHLNzuscrlr8Enxpv58u+yxVSZv9Uk+zttMdG0wyT7VVvM1OfMWZJ5pL/FwOHbzp8Z"
    "IVY7n4REk3BZCd2HhabecQGK9hOcz9yTO8/0DJfrTyhtFDIngcyweLaYePT6FXwYTWv7snxYuiDtjcUD48fa7YGn8sPZGibWRqzh"
    "uBn4xVoaVSCRySG2KRk55pZUVsfH8/JuagGUU8oqFb0bxqm0rR7NwrqxXChVKN/rapXJKy+FE49VFTK6h6Tg0Xo4iPZw3GuDgrdN"
    "MsFawuP/SU96nYue2r02CxZttm5ssy48LkdqfbxVe98mERoPUg44T4au2ZaKFucVivTampwaWF+9NFhNtIxI6RnErViuS729tjSI"
    "N7SIvd1LbECXf90Vc90H47w4907T3nDd4FWwz8Ng8pWZU07piwuSIznR/XNdegtqsZfyFuR1E4g6zSveAiksf2FttYAEPSdjmvxx"
    "f7+UTolyfXrNKfGT3gjK4lIoMYWfGYVlMdagJi4H3etODoEcYvF1OMw2Q8WXeFYrw4tgy+r5mGyiD5tGykkLLck9RlMyFU7DNdKe"
    "PcjfW6kAzVkU9/U5g/n1bzJ4VT2XmsT/ksGrWE6k3fvfMni1UQu7979l8Fom/L9l8MoEv6+N5HcYvGLUmt373zJ4HXDWv8vgHZI9"
    "1pz3BHaKR1XpYDCdZKrboHZvZMb5GKeOvYH/bZu2mwytYzPsg4i4bOedQrDjTIRMxieieyBqPjHlESxWKmVQBXRg1h0MI5bkMFPj"
    "TMOXzm8ddbnFqFrHxMx7ARwpJrQplaKVLdxG3ZXSjOCMF9ArpvhU2N+T2VICcEqMgIdfPbind0jOwhMD1b8tmuiIGT/3csLBSjgn"
    "sMP5TrSMEic1JssixpsrWA3XsJYPhjNDonu5Fsy8Pt5hi6Ef3CP97LI1PQOmZnkBw/g4hqsAcajF+n1l4RTCAEiv0edkMwV1APN/"
    "eeoSaCz9gu4niacSPV2t5/y5d2t20yxJMod0ZleQ9KUI07zlw6uIFUx6VsIMYJOsoWrr64ocX0YTxzR2BqLPjAp3cGBl/bXCS0gB"
    "tLMQmgPQQP8UEeu490fq5lOgoBLRAyfubR3Bq81t85ww1YOiB589aKgi7Fyf754J7YtdnGaMbD2aURaPGN5mB9qrjHMhfbole5ZP"
    "Zmc2TRIVWgSn0YQYPRMzH12EamJM0X8JemfTqyDP6+BmaFGGR+sB4tWtfQQ64KBk6XrbxJZ1JvsmIz7y8968cRHg+5yP3vykvUqC"
    "DN2SGP0P7XuNZAm5HLXK/zcSMnEmgiJfT0nF/PwiKSjcgpLWH/ZE92iIGpQHMw2PlB8QWWerHZqk/9NM6rQcMuklYkawEj2JKkaN"
    "kgBTm1jTiZFUgwuc0f/7n//433/8Yzn745+v87fN4c/Z9sfb/vDPPzbbwx/LzR9/7s/7f6+304/4bf9/0Uf/fJ39CT/44//+48/D"
    "efe2/3eTftuDv/+P+Y3/1Z//959/zpbx259/4ov8r4nf714nK/gAPfJPNoK309vkfybb9Q6e/58/x/e3/4Y/07fJdgr/fKhWWr0/"
    "g5dOpdv7s3R/+7//5r/558dh9n9+++f//h/moP69O/8TfoTfhN/R9/8wezBdTg5//sn7rY38/zEH9v/iGNhL//jH7Md2/Yf+6z+W"
    "6932x+GPxx/bzVsQL+eLQ3m7OfzYxvHbjz9e93/8GdBz6of/+AcfS1h79G88HAu08D9cQLKyr9mgMPIxUR7M9BkIwsOEaiyx/uLu"
    "k/G0EN1qgz+PWigI1AmcU/+IOaH6dQI7vI6HZ0i0Bcd59BQdx9VQvHvUEu1BLVtqzVERa3kt//3XpFD80P7NYpQPu914QIm4cyqC"
    "A6WXkH764XyIIgK6o73D46ElTIeU2seDHMmC+BhAgeVw0tqImF+7A1cHokK0oStT9H1UT/FkSZ+bddaEJmL+br5tgIVabUdRMwqK"
    "g55fb3aCYiQqESXcDr8Ug/2RVQsjfM6RipOzjFv4dDuqLBgTjSeEZ6j9HLkV6ednUnJXvmCq6QdFxH7vYh1O6FPddEE8gqX2Wo19"
    "EfXpZq99Kse81h67/ViZP58f/ObD7rEd+gE8O+usin34XtDBK/9xRXK87QWtTlQvDbxiVZjscSuAyWhxTfHUPKpPRFFp1vNG1dCL"
    "QjWiB7+8ikadqNjqwed7QfEFn2dRgehOY7EAlQGvg7vb2nwX9MNoFqLXoCJTXdjnKo1yXIrCuBV1o86svYpLIbGCznlzzWN5vmu3"
    "ww6MiLIpSp1KvdRbdWa9qNTj3caRy2c4iWU7pJHXPvTfieo0/ExQFK9fzNeRSagVwgzUO94i7K2iHvvUAxgyYSOsBJUBzMrAOwX4"
    "Dltv0Rw8c4ELM+6Uul70Hb/Xxueiumg6XBUHTVqz9q3tMfo9rfP8aPu9tr18XNOoEjx3wiLcQcVHHFYYM07VjhfUkcJTy4p9wS1l"
    "+SRM6kuIlF9dMWFb+OyBcol6lVOKfBw/2eo9GM+wWavX1XdgJh52kj7dL/Z6lbgC24bRt9FkTXzozjNMdrMbdgL4f7sTRhWxKK32"
    "LtTeex149Xo7avJ9UWHvMvYy/gz97sK3ZygnPEDWtyissMwoOIvdTtSq9byTaAoaZjs0CuolZJMKYc07sB17lWIPTkcP3uPrC9tx"
    "bWZZ0TfYgZSMdWKb2Yjr6DPhUX+cPVcPwiiqQ9OwvRb1KGCjga6lOHO74d0sDOrf26ug3A1P8OyKz8pcwF5eY3YVI091w0UHR2s6"
    "xMNU7WFXibLCbEOMmpoK+h2/WEFijHZ494jigk3UA37CygIMI++DdIKJL/bCShR0ymwbNR/njWRT8GwVZiHg3R0KQSqa6d6eLRMH"
    "QnYEixHhzv+OO9UxQud7xvMgsDtwMoTHVwrImm9b284KDhGWgwSlGrpN21FNPH/M8fwjbsUOP1jNSwiyvNiSEx7DscFtc0LBCD2d"
    "znr+CGT4gkkoeg0hNXaPPc9vgaANO/jzCipR0IxfxMmEpoooF9jOF00d7e+I6wN2dc+PHrsV2OHIJ29eFwfYYrgQrmbhzLYeO+Ed"
    "PL+oMBIVMYttuGZa33uhD7sbbrGg2IR9VIFuPskJ91HAldpw0cFEteGzIVx63V4kdzc2LanuYUJD/k4bhGy5HdbrIDbCPh/py+ND"
    "QxFRj6DZYjkKR1XYkjPcel2/qd1cBZhM26dhRJ2gzX7Orpf31Gfp/MIn4S5uwbUDZ1leHWHy2SCM699Bpncl3A58WruNbJ+GHnfh"
    "jM66oEuEODtwP/dibU0dXQr9UkALpC7WU3O+k9vMHC3JXTqn8HrIpRXcSKi1CDFTxTud3VgdlGyP4ucolOnvS3Vp2rokdzh0CS84"
    "H5SeVSh0K3tTcHd3QHPBsxpWTiDPoxbbYlKDuSTfg2fhjLd489MS6oIyoyo1yUj5A6/VW/qFCAclsR9gH3jByHgG7+tV3ISt+gRb"
    "9qnnBc2BP4JLMy73tKtJXh/yhgJVASQQjKbjoR4F1/yqVYNXcavh9qiADtZGrDIYSYQKFL13xvve8h7fqsiByrcq3nwRqp6GblbT"
    "e9tDDbXtcQUXxUuMI4MReKc6iiDt3u7DrQTndYp3OqoYj92AyfhoBWoeHKZ2CCoDnIBu1OaLMhQ3l63JNkgBEA+E5AKvhxGqC2w9"
    "PdKxV8jqCuJDNfcYxhFpI9BUCGscgZbDT0AbJeEAngu6FdAL4BlxaZoTJD/Rj7UJXRVBS7nTJhMGEcY9Jqs7sD+iirztlnDBpp4/"
    "oS4gnxWCFLZDhDdKIG+w47VXO3QtxK12XHzVJN4jkl7itupE03YPbijUYORMarMhthlX+xyvduEKqUdRS1wj3JIAwRkVYZ2jQVgp"
    "1lG8wPmGw7YI5BCWOIQWdDWGdY+DyC997+JB9FuPQpoxQUiyFdbyFm8oW+9hm8HhW4CmOZKofEI9AKEp4HCPr2IIXvER7nhYW6bt"
    "gASEUxJTW1IjpSZPqMjy0YBm0WrCSOlAtFekaeBoYKJLYZePqnVWo4K7+rEXRD0mG1pNmmSu7cD2ekIJpl++oul0D/Ha+B7hTeNF"
    "ZHXw9Sy8PNhHRAoTyWtxgOaodrThTOM9LtcbFSfEQlLKMBz7qMcPzwkPD3wGZDNKIvOsJp4903ramhDPccGrqf8ggGN5c8GRVzsZ"
    "VACwPsQr+Anhjp1UV2jMnpi1okkv0U396imrm1CXG9gstqdrpJrE0iQVjLQ061emZVSO4VDA50ERChm4CKuPbyE+8OW1P9qNBrWP"
    "100LMVLgTgjaqHC1V1GT7u/yfDksIHtwbV8ry7xvpGUkOBxExu3feehAZPA4WI95WowKIVUmTgvBedRffE4xMkvPI/JudJmWazt0"
    "A4RkhQitNCh3whNdKW/n+s1wUI9fq0V/wr+l0cWgD6S28Vhvfallfu+tTiNpGmifQ6djo0LNgfyug3nfTDUPmucL3HCVDtvpcDLA"
    "HPAW33uVxPUxP6Z6zll6QcuoG0rSnC0EGDUB3GCPqPjgVdE3zMWQD0O+z6waumD3WtMnpuDgo3CYkMQMrg+8XxMT9ykXR1uMWVt0"
    "BdUCuGhALUBlKvHugQjGq8FuvKwLVYGPttYNp3A3Kzvb/SptFZ2QlHebd/fabMVRU10fWVbhkXRpzjTIdK4hWhBg9sdk4OLEt5ko"
    "OTH5Ln/3guecaS8huQY4oIGM5PRHi2n/5IF8no2RHa7s/Svh35ZhVPLgJ0Ai38pmqKRRnhcwgt/I8LXpgYWnB+Eej9/64bZ1LYui"
    "OpqNuwlXnx6VDfapkEv/UkE3IP1RmLIepiiwNIWj5RXCGZSviefdz6afTz6rjVq9or9WmqYrUkpjcl7qf2Bim7b0loBlXST+IHxP"
    "mkwj0ZSdGVQ0wQNNpe2g4CViGXpOaGls6dJJ5pukepZv4vPMTFebHW1mtFGbTdlTD2WaOY9dTqKixPIfDeK48Th0cqhiQJEi9W2D"
    "/AqTrs1PNK9+osU/8fqF5uUC3WgynNbrQ5zTyXua8kOSbEA3n8ulMz0feGmke0WmYn3/DTM2yhIobsQ+h73Z154COK9HbB4+G8/G"
    "kadz4DbKa3Y/izwFEUrrXrYit0h4FNSzQXHY8eqt4aDUFpk3Egeldp9AbsGMuN0U/q+4W3JAs4jIbP9ugdx7BGfPCeWTgUUJrUS1"
    "UzORmpKk9OMxyCWMIJbsgTY6dd40UgMyXswR0Qz1fa/hJLVDSr7oYnlMQZMaQDIgbDG/CAHlem5scCMBDNNdVU6xHmx6XhePyGBS"
    "29SWSIM+Xrc+KSnn4i2ng876uTdHVB+M7CAm8GpctQBKCXwNLOMnHY3jH1UXCDu/H1kJkRCI4Dzm3Ct8hOMarBfcXHBoakuDCuQJ"
    "urAODiyLQjxT156v63A752HfS7xvkEsTvijXLhX0GW/C3y+N3yfo1F17IZEtTWG65zUheskcBK2J4TpAYgzO+lifNvIlby6RchUD"
    "idRkFcmm75A8ZYkYOrBwlEgyXBc/ifxuWZ8K9ywqP1QVWp9aRuPBtokFxCGSCrMZGe44WkiBIfLVP8fECgsTz7Ytx7CsTweWU6Dx"
    "ZI4GTcFVrKc84KepacfncfKddT38ux9IuARnfeVMTWV/3o4ZjM7lOqHXW7LeqXswe/R7RMYGlXEFP/uoxUla7vgDryKx1q5CV/Zn"
    "ybHAl7xgyiCOh5HDDDDg/TxNirYcvFuyyZvS/pVwRzu3GSO9TKqnHUOG6eyGBVqEzGazm5bNwk7/GFGlgpmIy4Tl3nyuGnP44vr0"
    "WrnBlVFrhRbX1nglFeoCXBub9jIZdKZ8wbIVliP1ZzONX/tTGF3Tvcbr+gIJV6aFiNY6z2hdE/7kZVcV2y5FQWJnSsMix58kQ0cY"
    "UA1OFYeANVKaGedwp73KUxR1EWFmy2iFjpR+OuzKRKA13At4g8XpJMCVGHUq8a5b+maDO1RYhInRZ5FqsO/oQvdWi+Qmto7q7dkp"
    "v5F3i4N4zg8m4TCC7tc3lPCpUt2Mq+VKGrLiS0tiJNTmtDBdrF5ArabFzj2VjewQvF1fpIusAzAX7d5VxVK9gys/TClK7Hd0eAhh"
    "UReetdizTDpJQf3ZM52asry5LOL/iJnLcFut3hjyIheY+s/NW858DzSUtdGkkHioHK+wm45znRIlsGZCeqV+ZxmtOud6t5WccFN+"
    "3bxWY9CjSvTpfgEnP2I4KWXv5KrxGa2jBc9f2o2e5hxhtb3V7wHxrYaraZEyjtWEhJoJr4xA50IugIg+MWCFb/Ik8IX5Vnuaf0wK"
    "8T10m4H/Ms/EYmzpbqJEjGTtoPWBdF686Z1rlOZ2ZCMEKbWeBocP1Hh49xKyvTMGubAaRfJc86Rotvvkqw5ycFB2WTImK6AQXUT0"
    "LllUod9m6nv3ljTkNBEKL06uYprhwljzNFJENgOsQXDHS4jxFpMynGkZ7s8rCdXnVwStb8DySOWO9vkITaA4LLKiJPznHq9+6Ae3"
    "omn162QZyOeQ5e0zz9LTwxZeO4+w9MdkMoI1BVlNM8XsaaoB2bT8cXdB1wo8v7NMOONYiqDH3vw7utcYh54bKXfJSgislDFl0RTy"
    "/KBdXTw0LKVk0uZiKn15zSr+mZ+bl25e6itRqp2so+f188bPxIlARcpigciqdLHDM11zmOcNn13i5wkPGjNu5kTpxX5ulHZ/T5V2"
    "g2WzIS8FNj9+Xt5+oJtIJAwIvwjyZkFvjUryd+tncKSqzpq6cGTvbgQ09fYsD2jSyVGNj3KbsVfRPaO4eXYccJmoWUXFsJJgN6w+"
    "a3Lmk/iY1kRIBaE6Hw5mEHjS12ogeOlNfDuPJFqAmPjprVw7YhRDIUmsYlssphkiCOzFO0mIxOr0Lut5uc3otaN87akUI6ct5YRW"
    "mbeBRXyi87BASJxwvSxWz+iaA5WhhW65Hrrvaj7Mmo/uudYFC6FXOIt3+PuXnqJoRQko+eq5M8OoRhDnsw0iIP4AbcWEzDF+Hy3G"
    "y/kuUU3Mc8OhixswEcvm75VXQT229hdvZednjmBlvmPhBGKLgj2OBW9mbY+Q4/oMDVpjZA99tliaYgSJie5fyLTP04VL3i60AtaF"
    "dI6/owtR3i4U8naBN93yPfs6prqwcHRhDrv24RvPK4Ttekf0ns79AQsrD9eRZ9SEO8H82XD0HrbV/Ps7BR5h566ubDHq0r9Yl1Z3"
    "L9rzVtPeNpKHY/NqM4vNW1mOdt44lzZvGaOX1Wk4CdDUzVyQh/rapCG5GfxuZfvdDrm2MJAsuxNwDbWK/LmMb1MIVy7JaF8orGhk"
    "BDUdlU+0vrMx1WZggOkw1V9F6ff2ZLp+GGXv6shcPrwMCXT3Yb8VNyj8At15JAwVoaUwxwMqQ/H4ZemdBubNc9dUytJdE6neqvP5"
    "8/nbvGUxit4eHxg1Ba/FBD1tgdjvKOGmZyb58KaUQeSwSL6wCPmJg6kFh/BWa/6WmofJTDcv8MlQ0BKK3/it93D/1qvA5M+PL5v5"
    "aWAGIHQcMj6ZtABgER4xfApXidhK9HvtFFgon/SFAu3kCV297a3UYKlbdwepIF37THCb9ZnhYK69XyoQQuOgiZDmHrnq5HfghHD9"
    "T8nwjMc1FfBJ3NGoInaoG9R8VEQiO1nJqFWvsuePqgJ14AXl6Km+kMj3h3gC2oir8lSUE0H3jq/VShHEiYfbZQhNDHseSTBy3zz6"
    "grDSGpjg7+OpMGC0ygvoYWk7tLKwg4LDX+HPLhEAzuriYfa0vAmn1vJRxcV0ljWSlsdGHIRfCxJnxMEsnmWavVfFw0ihGDHqpwRW"
    "P9L6YBoEG4m9NgsBKrJCLwIv3igvVL4ZzY3jAoFkEx5/IOaFTjrJgSyM/TEdlFZENi4DVAly8aoKu0osK1H66TknElmZefU/FUfa"
    "okLWCbY5tow4F4KHEAPN85rq460Y/b+69pYoAEXyqrHOChlObDzkdnYSBICkUn3NT0ZDgYtT7+5f2imwhM6ZnYRUjdonLbtWzBIi"
    "6zJUNpLr5vltLEtny7tElqciuXrvftSE242f52fR+3XHayCN55OORVYqMOAKzbhN1uxqu17o+mZdNW4X/My/qHrJsT1ErxPd/atW"
    "nmNeMXYTvrHSo7iP0eoAi+kvxu+qFlNuMyyeowM0G6gST10wwqVHAvSI8CqTd4/KjRKUjnJmdLAKWFebmDnprL9CabUJUmZdCuF4"
    "tglQUcPZMNH50BpF8jP6jt5VdXNJ2fq7hGrr3TraNilhBa2aPLobdedne/Bg6CGNACo/A8Y2d2bsrvNLI/J+WvJJkKGkAAQtZNBC"
    "7+BhlGKeKh0Ifh4JE/pgTZTTsCyoEkxuEBLXB/OhbZ0VYX3A5PxuQalLSLOrzM8uTACLapc4SASvIHAVnAvjEr4xpVaNVQJJTU9/"
    "sm0THk6dMQ8/luFPMWdlN3A1YVlnwkpJP39WeITIAGuW3TNtQ55V/XedZbKu8zBmHMkwG5irIus8Kx/NwLfGvsQ2S05WUEmWZo/7"
    "N/z8oiCFWSKkgKc2KyPVmxPnPPKnWQuTtq+JOANd6WhBTNZEII7bzuI5JHd7tVZ5MGrtZRdBi+kzP+pe11zYz/ypQvrJu21IJHSW"
    "pQfxuTA4QFcXU+xCZ3kXIstvmH6P0hrDQKD+PPwwcUY/JmmD55m/cs9f+awFzdRpYKGZOd7PBNnQSMgFPnoEsbhF0HdnAIKNLERo"
    "HNpKRR/THxqVPawnyyN7SWRoJNVF0dR33pTtWS05RK2DeCWYuj4vJxm7qBI0Ua2Qbt2GRObzF5PH/DHNnrp9CALT5eEN0OB5uJZf"
    "xmIg5HlULl55aQpPL8cJFS7WM9jaYKxiauqLBXGRc24h+qb0+EozH8RKqkkMWD0yP3sKZOgL/AwYORDcPSYw6I4BBrpAQueEkaXy"
    "wzWoLGOCdPTNb9DjOV6e+icImxL94OWHs2mlOBG+6NtCGaawygOhd0yrxU8YBbJx75GeK9GM3kUbBbf08tsd3ETDjdAsR0WINBt3"
    "r/i1bdtQudhlUQYvsvghFwbPeaUThEGnxmsFWsiPrjDq9DJ9rZcEawgXZ2s2LRTP42qbgwZp6IxrhkvqwheWyH4FDc4nC7pYgzdk"
    "MWgHMrLWhcGD5Rn3jElBquDSjGznKoy6H52HUZGHvPHWEqq+/F2BQxYjk40h5UCw6l1Da0RotYJFMDl5nM1Tk2R728QZn0p/oy6/"
    "kSPOldg6s/Ea1pdG1dzytU7uA0x/2IHS1EjMEsKobUYD10I9/CXWOrklNMDPq9tHAkMWd2geEtILnBLMqBuXj8ku+UKeS8LhVD6K"
    "3CJTQvyIxglcq3nmvyUitg2Nk3hWJXyHgfcse6hTw3wtP4UigrbvcOnYfVioug90sexyCkId6tA6KgV72Nw3FJGKm73EBn1YEtvF"
    "goad2lK5/DDPvUrCUYlXhviEYP2lmCb13DR87Gw1CRqZ1DYEXZ1PNs1wCgBSPMbXZKARlJnrx7vZmcGizAQWkpEjzOFvKadF/L2s"
    "IFNFFEA7TNJ1ByOhIioqMSmSY2JwM4qH54WYHaGF+JOguMEIITOQO0FnFUS9oDOLvGK3V4H/RzxKLGdC3lwEoG1C2BniRH9FzgLP"
    "b5DK0w2oFTfNreryEV14s9FNFMPPd1dYDhpaHFt4GaygjuxK2TW6FBE6jbreralAwamonpAU7wd2V+l23Ey4SJEi8G+o8OK9NM3F"
    "p5Vj1CNQtMbV+mJYOOD1YYxciRT986kJFdoFXg0ikUQGhAdkwBoLpHeFz9C+pgGPJUwA6blXj+xF88+4lTR9GyYOs6kRNRdHWUDd"
    "e6SVIcx0+KVIRuxZxdQTKFVVGefqF9jxfxaROhXYJ48CzAryH4I0Oi1QQ0U2IujGJnHe6eeswErc6UQbAzfdxAQLhe7Kw7XSnVPq"
    "00qnTiT5yFgJpbMRuUJXRoUKwu0zZZjEqe+l8P4peVN8cqer6z8BY3uBA7kfDmpzxEPievkWZMPubV3EIry9kmYFOLcgUkBalZXK"
    "zrKqosusW0tN2OBmTlsHJpTSIcSzQiTJxM/yHE1+0NPlgoA+X7mVh+v2yHD4H8zHuuwx/H1eOf7av8OZ28HIqz1/KLuEOj1f+1vc"
    "U7LaBZuJ6cZKKEm8p0c+Uj1ZKEYKis9x4Sh5msa9bVqFuEH6TuQDwaAk//fN5F6tNcLXZan6n1Nf5gzijt9jAtikGryDikDuG2F5"
    "qmwcEk0IXTx7RUc31oFpl6rUw3nqeFvoXcW9TOS0BoOF7ezpdteecAmD4g1qsE1LkrZm+Hoywc+XchYn8rXv+2Pkcuh3/GEhWkge"
    "j4AVVI1vpixD8rI9NKOrhVUHDHCMB9EBVYbRU/SOyUQyy64Tj8lEb30ipv/oacHi2F09wccFP67WHxXmV7/IVYLFdtQXlgtb5ynM"
    "JOOBkZiy7MfwGtZ2+FSV/AlnVijDqAYYQrBfYJ8inMqepkqQFXMg1ldUF8eFk8+RtRFn2JbW6DTpN65PlQ5TLJquYkHNLeafKVng"
    "GALJBn7mJ2A+yPuackEPGMfmDkchFBGW+AexuiK2HWy/KVMhKT1C3un94LZRXbDEXDWxO5WFRduP+WTwWeiqcbgWZ2LjLss1QpuK"
    "BOKou9qJ3c/PJXUVQ+YYp2SFN3Vi6248TWGvMLsaRwfDoi5o19GnzvpbfojRWUHJeY8mTy0/l8iriGlRF+2sHuCs7pKfnJ6zvzU9"
    "s29JrmP1SfgV++Q8EYGv+nALLXizMNIqyx1FFZ+SwviZfkbXTr9DKXJi58PMfYyCIiLxXoQRLUbNP8N3szDb2SfAPMMJFWYeE6gL"
    "ugQV3qgw69h1klAdrPZbytxzmm23Ou63yDUBS+JD6XVitFfqv55WCXNPl8kn8muXPfZ/bedmfxJmAu5jVAffouIGdLKVkpBoEk7P"
    "r/32Fq+it64s/Zb2Vf9uBpJoP4KLTQ+74etwvX/w1ESYTFBq++2EVGPazVvAFCKKfSKahEoa0b9/Z4nufamZyNKM3J7BHV6quE8E"
    "SDPmGKJu3rBk2VmK1GEUbPLgnGKXBnBTjat0h6efjRzPWurFtFwFTD+e+EVY80Pc0NbNsb5GGlyigHqB5aPfu7BXHnK7cYQDEs4f"
    "U+kHTV01gBFMPujn3QU9w1GVUSao5sqlH6N+dERP8OTiU/RfZekEePB2k42O9LVNrM0nUxF8/ORh2vfY75kVckHfJ9pZICBdnAC7"
    "pB/V/P5Ru7lEb3QJxIzam+tOSid8ubLHEVoV0UNWwiMlyaVdjqm6kHJXHFjR0fJMFjREQSpI5Gi8WAIN6RGer0E9oMowYlikmybO"
    "xG5MTFYPZzIF+O/EhMMjulud8sIvv+I/kU0rb1Lldg8qhWSNFDeXetTmRRK8DAgyY9hdYrZAPcQA4z12G+74hFfZzhqrRMoPNUkt"
    "UAmJ/P08oPil5mTUWWoS7lu5ID6F487ID4GiJe3WZV2RGPB6j64eGHLpSOGYykFhDm551jGlsXqIQZlegZJ9EQwZ5vUhz6B4HfUv"
    "USLCdPMFb/ZIBRovmBXroBVB7wR3BVyG/VE8KMBwxIKpvBR3FC51PpW/8/3qs5mOaT01nZvfSvNAL3GMBc9UlSg8D0peb2UQQuL8"
    "o1iiNbd1TXqdifVCMVrEX/NzK7yEPOKl9nTkzAiKbMFca5fOzQ0XAobTpBQy2rQV/wfFt8iz6A6rymqKxFp7uhNKmnfi1UY7rRY+"
    "CxcAUxnofUQUwDpdfC7txlFnXkN/Em62pPw2GMcGJSlQNaYxM/5RTlwxgqUsEUkSa53avf7HL0QG9nYue3PvyG2GW8hP9RTM+4u4"
    "Y5/t8S9/or8HhrIErxE6WhW2ZTVaDQqYqo4OkBNRzOiFNsRKdJgJb730j3T1aO5xbvk52eEuC1NqKLILbO/IrHi+xWyuGcROcfTc"
    "UHyEBuIXYRYwIQRNiOLqdVC/0J7o0u5/xxA97CNpcw2aW9fn5S5m+vXeMG5ULTamN63R15LowjylfPGfC4fGsgQiAwtdiVaVjTba"
    "fzwv4Y4lEYAlJ26HBk+hAPk/AoMomrFUGlWWlPKn30QyB2kdIFU6k1hPLdLLdC1TxT9aCD6yGzFeTZpMyh2Vuh1/14zwkIqgzjWW"
    "qemEw1z1w1HHZ1bxUGQEdnCOYZK4upA8u+wznAn0bLKWqbMt5YZ2GtIixfoqd6cXU+Im14FTYkjDUTKyZ1kwSTLV4ChGgptchWXu"
    "U3Rea9TT6NCoW4uSwtKKMJ90KhUWqOlmDjA0OUJCHNzh2qSGtgiftat5KF1lgp+N2dXZhSvRv3w5LmqbrZyTLdNd0N+ifOXmZGs7"
    "v2tEBhO8PiqvWLpnk6qdc8Tu2JbiWFuPSGtJDEHnYQTtVJoAFOExyI9wK3z102ZQyt19vFr0YBOfQNCxULxg9uR+pIxc6BrCMTQN"
    "MistC8c446aMwIw8oxsoisQ249R9ynVDDEKr63o5n5VxoY50b+9YHSOE6OCmdZL1nTeRSVoKQlcauVz2clOd3TQmtzFYFBf0mTSW"
    "D2dTvbeGyxsGeSmMelzYxWyIrG5MFtoQ89tKEAKjmdZ230aJ/AQwB84rxEZLuuFNfty1Kf9lwgBcA+9kHyW7sE+9npBQ/L1kuozr"
    "PZ7VU1Nhl+SvDI/PUtexpQG7RLwc3RQUCUEJ2t6N5TJ9lzucEqrZxclyvwfJ0RaXyDAIuzlGDQNzh2HyLxNkCq66fmd2H+v2X4WM"
    "3zQFFCpYC1gf/5c4ownKzqXj4C3Ze3TOEZQMFglHdYrl+09eg7DEw7uwTQllQb/nxZWB3xLcLp1VhNjCCTzgh4/WO+GKRz0k5ICf"
    "t6PR927YYRCoq6jLeCBMTGHCHyZUZUuTLNWt2fOKzxJxV8f/rxFws61JwYYQekG1Q1jDfilcTSV9SLN7e7E32Wr2/FbAMEoRXrce"
    "iFEHnI1kpZMrEPB2q+dHL1GF2ElK0apYZ88Weww1eYGkC6BDBTUYeUXgFHM2HK0Lo+9hPIXvdzJQ0+9eeqFO+SOa67WhtwTgzDkc"
    "FDZ80y+vVEJI3kURkVzGblGByShF4Sm0rY25KBXicLF0UcyIq6snZM7B9yQvgPX1NhJlIB9Ah60zAjyXequWQE8/X2netSAXmKmG"
    "yqgUryM1CAKrwwHo1JBEpVspNgVI80sPQymS/ukTbiXiapr2Kn5r0JrBzdYY+OGp1cOtCFuyQkqV/nvETiqA9BT39XiJYqS9rcXG"
    "a4SC3OzWGBpUuXaqzbfvtYrM/QbNBrWYznaERA6PDweQeujWO4JYiseP2/nkZrpDknl0F7DnShfQ0474f3V9gIiozZHxQvJllim3"
    "32s+VuBSvG3UziUN7Le0RDpdNrEIRoLMJrVzC2aChduJ5QoT8T9rT9GHDkA27B/he6O1lOF3MbKxw4W2f+s+/KAKpsc2JhF45WVz"
    "Pn3ESqbQe16Wnobo5EA6MGh6uFlhoPCItOvjm9KFj8ybFr7NQTVYo3ScUEInSMH+CX3qm+f5bhbBAohqlwi5dopt3JldL27Dzg4J"
    "Tzgo+s1e6INM/sRi2PFmtENy8WdKDJsv8RP2dxECGRZouaIoX+29sizHxQi/hbxAPb8USL4PpAnyy3r4ZXBD6w5qQ/HyvIaJXsdn"
    "zMRDk79Rrhufke8j4PN77aMF+6PRZqPTuqYdUh0KVURurbvRvg0j4+CwZiu3sD+WGD41dzoHe/ZU0awMnX+xSfYZdgLgqoCJ3SUn"
    "FIQnjrLrmklVz3VlQvlnaGTISsNGJhMF2MSyg0Mgkxh4XrxPn9BN0DqO+mgm+MeJLF0p7VTt3gWMEVRUDxMs3efsBonkTq5zMchq"
    "7nojPQ7FhjtWwj1JN51bRjifsq+ZzlyfOgxXqYt9hwPzFuyXFn1smYh3pY1k2bwfq4iP+LXz9engwXg15cKLvSTq8hnPv8U6pa67"
    "dbNlBtkdKMMBlphJ5yWzpym0Tk6OREYOdkHFVngOuXJKH6cMnoFe7V92HxR8eKp/jPz9PXoGybMrygSrIonAFgUoeTwIcXx5JxxD"
    "YS6I2aDon+aya7DP/Qu3TEIUMDKd8mpeX6Ij6m7Du/bN8qwuNvQ8BgadSzg5dzDpB1mdOnjYXd/NoG2eVfUarPti/F7bTp46l34B"
    "YxtoqzG0xkFhMX4tRPFEJuiTw/sy1ZL/JBgRj19iqf96stPzSfGgINciP3DHq13EYvlBtJNF0TelGA3i8QCul7LMX9hqQWQWLqet"
    "wntIOQieBtWwY4WugY9GKgM0SU04Yz4xYZtkOIfYU5a2skEjcs/tZvunlxjppVnSusKiQK3lag83nlUaiuoXzXkl8kNfMdHWCArB"
    "30HbIAzKcircsprQrscD2dlNUf73kpDWDwUhouBcg5XSuaNAxHoqicTj4kFFegSfmrC7TCcG2lYg1RbM0PFMEvHz7VE0hbGxQeH0"
    "OS4cFq9ItxyYbcjDNcf8kkYmaJjFBv+OhNJPi3HSdiajWUQLHW5b5dBAAamZ8txiPAtExtLnkGVX7JXEa80mNxWteZfFKb9hmJQG"
    "fTqTrUljVIRNM9w8rOkNd9WsUQ2MZtg9wsuC0QsHJWhr5NQAuT9T1wdMEpsIVaheRz/4TPwOIwUIVoDxyelTPBOYdvBzlZfQP8VT"
    "mFSERWUObuqaZTbvzjJDA6vJ5zvCQ5IePTg8QRH+Xvwx7C/iIYoFqtOCLcdmpaB2d3NnZM0KgQkzBSMvDDm5OEuRWNlr96JEEudo"
    "D9sKc1aoANrxqYbp0iv+hdHBMSOaNjxHFGLHbwRFqaU8LYyaTJRmExgRr712bLFgLLsaeEYV8xh3Oh7AALeZGEZr/zp42IpvK6BI"
    "PaiPBax67//ivcfUxj1L2n/gMZEwQa3dQm7c3aQAWzTGGs7WD7oVtfdrj7d0F0jgqYd1rbzAyUXtY888fUe4SK3rivkIxm4VW7Fh"
    "AiO8jwtgcQxaONmogKFH8WNonmvSQjbRHiaI5yDpo2RZ8eN1BsvBPOl8pov1A5NL9ObhVFxe7dtM4NjBofJgJNLjy/ycARi2cwda"
    "p5k0lGe2hEZqTBpWIJ8WrIzgRNsJ05cRBBC7QuK/cJjyNU9XJF9Ns0iDgppbBLqBuxok16BlbVol74kUCseMaIvXL6g2tJsLd+OA"
    "4aQYE57m8mB/qPo0/bvEuoPqERet202a9jKaQ8m75ijhlkI9DC5JTAqRmdPtfE3Ajh7P1gRffhkx8IMYtSE1asdr4W5gYIzahGvi"
    "QBEwPzWpf0cbGssVV1CoGb1GKLTMprPSYlx1I+8yUzqzp+tp/7RvWEKpCStD2OgIvTWDv3++FoLtqB+s8HZLht3SRVVCs1jAuZzp"
    "monKOUpcKdUW8UcM2T2uTkaXzAmJb5hUsuSlKXUtQwcnVS7SA0URjnT3hhk1ywdTJ6+i1wgW4AZLuyVmKagTiwtypTcSp0Gd621a"
    "NNCfg6kGqt3ev7HMjG+hETLjY9JvLm4udJ/7B5EhpdZ00Jpaz+7SgSXPR0544pI9w9bFgyREg/UZINMFCE/cMhK4tRohsIwROB4i"
    "91pQ/JyskcFCWpvx21ML+e23MqqX0NGeEzMoMzTURCYuviXrltBcKHF/meyqSihgo1YeB0tMm8+oGDU/LP2CmrBBEpdOSyBgOSyn"
    "lVkmntVs2lOhOTT0fAKp1idjWrHnSPJhCQQpPwv/ue09udYshpV+VY9rKRePlrynl46KOFcqqw72zHFiUgydZFoj5v9+GSlZhtS+"
    "UgbMjWWJJrGp/Z4S4O7DpSPzS2siws/jn6X/6BDIvGm9CIYlYV8BDSNI5FXCV1I/MBOCJvdqt2WaBHnxmKsFm+bwSrVqBf4NV8vS"
    "J3kt01yX6SRs1h1K7sZkn6tJnhJXoX58i4re2Idmb9pb6NUeAwiibgM0FwokaHUe87d+cKP+nVEBV21hvPuixA+InvPiLFPdTlSY"
    "TponmGlYrzOuHmXeEPJpTW4eFGcXFrxWY29YmO+4snxB7Ub8TLyHdSQTVgMq38MuY1uaHs6ajBdTlcgFM4C3FamJt6lq4qPBsXey"
    "c7gxLJ0hbrmyf5nEVi42onh7osl5V49kViWrksIqUyEHaoQSukkXJaCRcpo5GfGZJDjzFNNcgl0u2B9g4jmNX0uNMPCuWiawfzwK"
    "2YDy9bZUvhQDv2pvTNjwZkWT9Wah9YM7++6Zr2nT2n38E8ju6y5fB0ZlOnNan9QtplDo++AOkZ+MUWUmlOx0VDeeiroZ9Ysyf0Q6"
    "NBRUgwszRfrPJk8rJ2ZKEhBDCtIcuBj5cTP0GQHttHN+HVBVK3Qtnk2rxRnOkNTDO2fMloZLDkQLzxcVo96U7hA/5dkYXTxuSvCp"
    "krxUU6qBebMZXVJcbNSzdtk8GKr0/0HhmDl2sqq9l4AGqRp8fT/InOEAzLv4NoVVdnlgVWoSWCjjnCsgfs5alVSOSIjCyFsXzKA2"
    "kr5SSXoaSsT1FKh8WAyqDYlHWvyAXkv+hxezyJHzgLBkLZ6lJeUyIjbCdeBY76NzFhWf5rXJvDrq5Kjs6W9qMaRaKHop1osS6rN2"
    "qz5i26xoEA5md0Slqgo2OTBsHtJZdRrGGT9sedLVdWfHj1r2NsMQSdsLWp2oXhp4xWonbgVRUGwlIJZ0wJK/apVWqRdG3XZUjMKg"
    "iAzDvU7UmfUseEhZOMPp0hGaBRtgEdb4SD0+sR0xK48QChKZ1jJn2H112Jp6tpUZ2E/CPr1vCK1AKEiW3orY9AVE/nlCv6tth1gA"
    "D69SpeqgdPneC2LQr89Sca5WLOFTPf+Qa7kmcshwmcYVHWrbhs+MpjTjJ++WrvTzIQHJ2RTpmgUR6GOI6BGoR59LCBj5MVzyf1/s"
    "IbOhsE66D5842qEAR1jyf8/1OnwtQrhMhdiGekCRsRhgSOxey5JcilFzKJakVFvAmstcYfZMa6bMCjULomk5GVy8C6dTtagAaFi0"
    "D5tZoa7O1n1lT3utskQ/TYt9R69Uc7lgXfRlxAebRSEKnzSuA+YRCBGdL52e/DTEES/1dPMXm3N6U9oR/lHiW7bU9LJZ6q1Ef0W/"
    "QAm+x4LcSVbkcMPwcXDWaIc/IdoAojZj9tY2k+hQwXQotUGBAQq1Ie2Qru5ta39r6aIEBS1/swf/q3sTosXBRvXKQnS0T7Dez9he"
    "bdNBiep/knevULw0GJYCi9yJc7lpYUybAF/5ehndmZI8IMDudxSoOhDRK3qSQQ9AfAb83SvKeb5V5TZri8eI6bnxWMGoz7tm3IjD"
    "VZBVjAkRMjXRs1tgU3+0C0UkT0T3O/EOaFjTM6mHCxTteRqo2XDZbQ0LA9Q9yfaLNL8gKr5PnkjtW4yDQxt/NtDq9UTeyss8XfD+"
    "1ILtER/ZFUB+Tnq9ERRnozURLsx4AcXnGMz8BlgZWDCH1Yo4G7U1YweenGuNFDdjoptppC8b2fDM5EMmIkuji8/nhyVnrDlK1R8m"
    "/BlUifFmvjRmah0UBgX/UyUMCHJpirzCNtlNuJ7FRnO3mILuNeTAczChH2/lWnpxyNoozSTRpajx8xdP7RVyQqETEysdVRH0cpFq"
    "rhcGbXolLrZ6XifoBcWXdugHNthy6LaHI7k6hE3JEzNh4VjkMNYzfIVNNIVU0wRo2nPQZOG138rkxKbuUQor+1ZG04h4rUbSAkOm"
    "JfRvZzeMd5Aq7glLD2G3E2sh0cAirOrSFr+2fqLoT59KCEjygUpR5sgKxPIsZmub3NU2uK3Eq3rTjdRuLgX9EPMLi5VOeJr1VqC1"
    "2qDt4+JiGBcxVPf+WmAM0YLhBKlN5Dab2yDlObr93IzYJUcIhwXUQWm8MkbpBdx4o8ShNLunQudrdKWNN2hnn/Y8bNJk4Za7jSEQ"
    "l4sNntnJxX+i/5/nuwQg/ww11LAQEW1nj8Iw9RJXNyVZi4RqiW4GhUAqqxMywTG0FnjwM2/axzQY/nuiI0APElautmCURe/1kQFW"
    "IEf65KlZbBbCQytB5yhBx/h70luIr7vh5kGHLqiukY4VT31YSzYSvPWOb935LoOd8GM0mM6GwoDeyPIiiYxPlG4rE7idudzRf0r5"
    "hOJ1W+XayALUL57vX7aXbOdVwUaExCJ7wknNRx8TCVrq2assGNUUeKBNzx5RdkaT1qlRvSVKMLQ6GAIcahu3zP99XgmASQPEPxED"
    "8zFZgDzET9KrgGEvUPvOL++VS8s3KpfOLb4IOorEoLDHvxeNhdhQfsKZoW8nnz8cml3zedG0eu2ceG0hm3kazV4Rngn/X/BOhpIU"
    "gwpYZqiN6BcjShFzKyai+dIEmFQXC/oRM0wuiWieB4YN/Z1lztUSofIYLMkK6x6lyIXGNgUjmXigqGxQEa9IPFIYGZjpRIZEuhmM"
    "DF3voJ/h5O5cn9KsEpgdpJBqoytXdAe9x/tXuL+HqQWQN9cbYz4o8C6Izxi9l0mZ6+h2Wr7G5TLiz5nEWxSY5t9R0gw/Z7EqM0RL"
    "vL9v+qLY5iHvO3Ag2cIJ55XxqrNGUzChwOEbzUxRgVgqiwKfLWLA4fnmtDh5MaVz91wmA5jN2HhfUKDGYrffIc+iVoZgyNj+pTMt"
    "2xJ05eu3GpmOOQtW4qXU+3eCY7EZM6pdEC0emAI+yG35KVOex2fm/fP0UVshmphEvEssjNp2CTpm49PsAnRz5WojIDAa7qzIYx5i"
    "TFzSytQ/kjv3+eLpXbH6UkRYXV2oWkUrRYxUFh7dgHj1cE4fhUeK+YPxZDNacF+pWdh+46o6Ly5hK56nIlWKF+Dw55fwXdJeG5bk"
    "fnOHYzqbP+pH96PuPJ05J3YrwnBsOjPRHeQSIA2155QHmN1zBrUS/+8RXIt5c2k706BZTrtgU7wu7JM8ZsK6xJE6oQl2K9anuppg"
    "jtr0Hmghl5RP5UZ8roX+c9GspzfbMnldRLTXGJL0h9PI4B4FTaUQe29d58gKRhO+PrJFamTOId3IqD0nQcs6p6t091bWz8LFyGVD"
    "yPKFWZcPKMGEl7hsNe0xW+b9NSgu4OAo2Cz3+h6a+shJ1VCzggSZrplINE25ZgToOJ01C1dn4kLJXoXFbIgeZcmKoGJeWUwp4r7u"
    "LsxzHe+lVZhRcGOzs/xJlZX6E333jTTlmeUhTQsDjMhqsoPJN+3feWjgDAqjPeeBKCqbGYwg7kmA871l7oBSOPA6QVSJHqNKsTvw"
    "TqX2KmqGZ7fNFRe3+GlE7gHL8BMUV+q5xqU3e6tOl5ysejc5L1rtqDTrVuKnblAM2+EhwGJLvamkhtLQhiR9KSGrLNzwbVauMZoZ"
    "+ej0kzJppcxGAzh0JPEyHZ0PB/1oYN4fQHGaqu8oXgDyIFXvMHOGyVy3UAQrsoh53/JShJ8j3APMVssj86C8x3LhtDOTEgbVN4Ru"
    "lvWpx9t56zGRR5ijS2RbdxUsV/LyNSn/rk6S1ZmR7gbyqmKQSh4+hf6n2O1cTmldr9JmYdifki+l+ZDZtGySi5ZZM8AsH0yDK/rj"
    "TVvXSNMewnROoXYwNqB/71lsUztoxAjbAaMI9kxBLgBRwYEpkboBxVo71vFaTJp/HiPy0svftGxRU/SwGVEprLogRAGqRnZMQtHz"
    "VNVqtAGV7w59auxAJnENzW0oth36z3AR5aXZ2tbSKv9ivG5z/0edcLCyd3Naj2cUVHRol4gu1dDkhGS0ORGDxZT7tEZlx2e0dSXs"
    "ScSapuT9EYXH9QC0VJza9iFJX4pNULoS7OFqWROQmFfbOBN1MQ9c7Ogik+2tsB116m2fVUDypgkDoRKDuI+/9+Jm5uew8mm4Dpav"
    "/dNu+rRKfvax591Vu2EnICQCAyrAxJ/VzrUZEDIz3V+x8O1mtAPjiFUsBUJczFOR/eTNZ7kfXHGurK1huUIssS66PihPkWEcLoSU"
    "ZKeGQOo6sqiq3/oB+pnrM0rAcsclqACYnr5maRNa9F4FMrK3Kdzzomntun8712mCYQtlJGMTktPnhGVN70f9Kdhfsbx67LY5F67k"
    "/JTQamC6IUojVZPW1slPrpYJEw7jllnrf29be65uol8cC64+pBsH8eBJrC9J0QXJ9tqPbgQ4kSFuXPLd8nn9u6B777BqSnwrhW8G"
    "5hpOyFNNOo9rcTE1WXKWRHF0d0UJv+qWKy2mT1igdTcjQEGL69cUKbnu2TwX5YgKNx6OTjJEU4YPuC01uJQoC25QaJE2ar+f5a63"
    "rjnm/cMB2g0Le2cXNI5FaGqHRZDTdRxPeUkgVirASJj1kT3qmXOyufKcvc0aifgH+xTmrPRT/FpSTCB5UsazW9d2FPlmKRsrjfdt"
    "ybDcf2GX7xMywsJ/7bKj+EiN3+nm/qEmUyM1/ifT+ySKrMh7rHa4NmGV32Tmsb2jvg2iaNOWQ5ClZJYzS6A09fQOfmoa0g225X5c"
    "DTYj7VCCDrfOss1AWBdkbuFpT4puQm4LlUDqX1RMOfJS1ICJQDKG2XjXhXBdukwA84a6HRXqzEgtKOKEFCT9uS4wcJYzCwy9UViH"
    "mDqPPpmHod8pgUryKFFifASWQtyjSjdshd2w+BKeMz+HhbPLYb++g8n13nrmZwde63tvNap3KhEBWQmwKUSZen5Iy/BrVUtPmAQY"
    "bWEEF+u6fkERxm0nmv4vK8Ko7Ugsu/+uIozDkCgx+RVhSgbsI+9HK/7/tzIMAnL9OqgfKOJDh+60GLIC6Rn+0f0tBDKFED8Migdn"
    "EWRGnWr5LeXA45tSDAeGaq//DvFhU5BNDxK3LqWXnz7baH959yNmtMhRWDXy2VzjKlUVIjjzkUr1WUa7dkOBandOJ4eokbmKLtOS"
    "UmVeXRqwe2nil/Viw+ErwcvTvZYZ3TIP7M6elyKUHLh7p3h1XPRqRse6/8QFWxfBJkTuaVgkEqNkJL08ITgpxbUgbrzU1sS0J2Wd"
    "2N690Rwar9Wij0DryasdJznrBDhcQCmtJKGONK5cH85R27M5DNVQ58vFhWp7MYJR9rthNAgrxbqATIw7s9BDtMZWbeAFTZBqYQQS"
    "CTOlnpWylA7BxPtklD8eM9INuExPYAqkZvHeEt3LjAJQE5hrMKr3KqdSh3AqRVdb3SiMEaew1As69SiIKoTIWLYAoMgrRo9p7izW"
    "pHRUuXqeu8uV0XeU5VFQ7BFgpCf9Zm2/HnUjRGH8DZON8Gp9FE/xj1HGHnGda8dWQUwjUKRkz3+ly2kTgPdciREw2z61A0UTSsAj"
    "BJP4xZ3OfOUfLUs2jstsrxbXow3mnnF0VkYont298jfWvQe7c0MzAZKCToTTEdQRRuIlz7uurSTFjMWXQmrGG2E3+AdUuDR8s8gL"
    "up1w8T3CtfNbCAD6KGjT5Zr6O8QjPajEkLSdpVNuJ/LGZW7D9VEjQtCc0RTsUPs4wGVndU7kOmBMf79Hl8G4L2ngLLY1U3YGpeMY"
    "TTZW+PbOL8nPKUGWxytLnPsilF2wxS6vSNvqiH9IrwKhuQgfOOKZLDkNYHWI+QaGKSB+H2mefi22nfI+wA03aO2QpMUSbOLCDSe1"
    "YM8TTTVnBpzTzcFeQMAy3pSpxSa5XUwco+L7qy9mQXNEqm41EltIYWewYmpTRWCQe4RFCwK1FMbFCm8adm/IqJxMj+5r/3beKNcx"
    "J3Q3KnCYHvSlrYUFMllacwz5zLCJjvfMRkfNp7ZnCOqyio0IZoP0KJ8Z9sXqlWrpO+m1JQN2KD/Hv6OrlObCyKGNdoryT7OfPM71"
    "MZchccs5vbaWxpZlP4/Zc3OHkUvr5n5tSRgovagSgA3eQijUxy63ocB4+t5eBeVueII7e+WKkckua2GXhF8se6KXWLxeumuUV5lh"
    "00TQ+T+1jdBewqUEN3D2Ei0J23lnNnPVEB/MwxAxjwPebtEqbvYqwRNoLU89uFgH/qgeenG5V5GobrrDwbKuXN8yTXPDJncJV5d0"
    "qyp0ZaSRcR8ISl1N2FWahIpusWZkWl1Q6tO4cFpZu5/I3LOEXXA0yc9TIiazEK9tQTPEcqNJwf4dOi5lFxWuwsrMPbKGxQ0LkgBo"
    "LN1Mqxmo9aQnXhi5ruhr9UTQabZZiJ7qviuELtc1sgtjzDWUaRJYHoKWo7pZ+EFz535n3ErJ60iVoKg2ZOKu+bg1NTFzVKOYrhwM"
    "UiWbMmJi6jkJcfzzTQlFWaQ12dLjeHBjN/R5Ic5GOa9gu1Dm44Cqle6dMcobJKskzKsLw1lo6Z4IEpKNKl4XBykBDYcmohLwC1nu"
    "8NSNZcmuW+XPsMUbDSG8YncXJVTLqpFOY0RdzBjBgoEFOhQnnOAMPylTK2OPbbWTUguZ43H6XsejT2R3o/I8M0qb70wbzm5jBtTh"
    "YpRe/WD1luHXpKBSYkZQKXolB4aZTHSlG7LW3reuic5Ck9j9tq14RVNNFGUk/GauEeUhQmogV6bzkKZFj0Ok4DnNEoZfaQJLEsYU"
    "OZQIMlScIQFgiYTaBUSTV32gOh933oJ5z0ubC13w6Cshrsv0Abl+2FizZvHVHkZ3VMljTOK1ODyyot7mrM/CcmT47mQjL2/9Rpep"
    "+VzMsCx4VhruPb+HvNKFQRo7hamRrSGhWlgu4XBpkU53mNhpcbnphEpgddyeG25Lk3j5kt4GhczJnQ7WmmpH8wEWQa5HSypBcHZR"
    "S/gSEC9Y4KFRbx9fu6t5vfuwJZz/MqPhnFRXVDHO/55mfu4HN8JeZjkJZnA58XtCfhLOD+UjNXwgvClt7evHN4v7VndQ2BLy+Xbj"
    "vAB6F7W6j6ZvqcVFLwCY5LCLnyKWovjY/mixLXfJzD3VheYa9bLFYvrUFtsRtrQzt1DV2S0w+d7M2EDg30HdE7kNteXq3pUxbysn"
    "FTs/UaeZkWyfmVyfyKpP3eGTanDDZtBGP8HLR8wRObxBahFAyGI+sRWLWnPFNhl2sd6FnTOjkkTBEQ4MFlbsEVmCE+i4bDN24UZy"
    "ZJbu3UpPYQa4AYNzOMnKl1EB+VwevtGr2RmT94mdfLJVPaHn2WC+0NbjnYGvO5N14RCOwODB7vHqtGQ3/8WKrVwpzWqbVWFSZRUq"
    "MtrcIuAEvuqYYMe57lrOtcPvauYMJ9dLCED9bKNJx6K0qtzAOTIpkgpqaMIDKUNs6leuSWaSSXNYkpe4gp7fdLx7zX/P6wD6fqfU"
    "9SKE6REETDKIvCoOmo87LbOdMUsN/Pr36IJlQrtpeiIRV7TDqF4LopAuxKtBuGGXuBDPa2RYIPiOv6Crd2L9bfY13GCvgx06ppZk"
    "KXCDlkYSFGmyx4gmE6RHwuhmTBhzhnUnTQIkeUDCYwX2Gx+G65gxW/DewtWOMvxA59HSDGZfhEQjZnflILj7sI/sBygXmEKtD0uB"
    "/bLRMbEvGA+IHgbZY7EWc4x203qSdaZVcDmrguIhU6RoKrtFKh3nUSV47oSICMWJ1GLGxdXxAsbLVt7jQfMbLvGjTHs1Me7c/Sxp"
    "ZRfElgnXjCWVHMIcFXykCDCjaxSGzlXo7ECP2zEseYF9823eWjp9b/zeHiGProTLVeCBnNW7drwSdRe5CUlcjaVNipHnUBe62tCE"
    "MvzFETbf27srxRiY0aHwiFHgsvKDSzN7myVzh4zsC0vSUAJkLgnlwxAJDEl4VPSOrDpY/5WpTXauI3RaFSxrSiwRq0nOpgDFB3yO"
    "kSdBM8xr4HbbLsVtxZmjx5YD1+35k+NLnBZHKLhFEJkRMDhuoF+T1ekzHS+mOvDURE9ldABCPtgidkmhK/fJtAr2OUuTYET1Ce+x"
    "HdxAHCTv1S9y6GpVBGdcI+2sC1Y4vDo/BhyrmtBieCGdsj7IyZClkbAb6GDA12K97vRsQDWR7QVb7KYRZSCzLjNkuDFylOOzV0z2"
    "wINDlAWIEy6vCAwwoB/mhwZV7wqZy1zTK+ea1gukEBguO+Jt6pbGWTPTqO7h/G53lA6zfCBEqbfliuJlyW9KQUqfzroQdTuL/Jzo"
    "ZSCIB0T3Qid15UAZlb3KaRZ58UunUhwM/E697R16tu4qLLupS3bDmd8j6bw9/8itt7l08HTThlsOJ2WCiSBV5DaOLlQ4Wy6FGNOi"
    "uKR3whzRXtbks4zKUwxbz7pvpM21IrU/a9LQxG9lHSaeb2bfJx0qkkUgWfxOIqMy+Wgz8LGGese0EtUlUv38ep30MkdXXH51Pfwm"
    "ZbicYCwRw2xYZPaWJrpY72wdXRMhNy24qxH/n7K5VrYCyuvS7AJXCupl+xFPcaQZYGs9w/zgrt+8UjBnkw8K2Dk4TDPkdEZXkJOH"
    "bdHM3S/uddBOElxsJqZwCTPuCEFEfzSJO0xYtUvCTymg5pm12y0nZ6b4PtY80+anJu9nZk3mkZqTZ7v4EO8KfSukfbgQ4K6MlA/x"
    "oPECZH1CPxCWz6DUQhdAI3udH67IcLoeBtli492KrXB9i43FnrGcazWxdWJ/1C5Am+qwY27X1f7qHc8EcK9XiTHx73Xg1evtqLmX"
    "h6vzjPR8mFwN/293wqjyzN2y8koIvOvNZKgGieTBRLqyNev5xaZ3F1ByRWeBio1MR6OuACNKW6ND0MtfGfmdfE5opI+hu6ZHIgEp"
    "lgTmrGDUnaZ7/lhACimyUM6JGGfisAn37NeqEV26ubW2a4qVMJSfYBbeyB3OknT1coN0yfdtVsk3fxYN29aWEkpMKqLHLnKj+6N6"
    "J5q2e3AhiwyNoNjthMFjtxIRPbaWsKuXD8riOOVcfnA7rrQCK1iUDaYuw00m06Qk1cjDv7RH0X/2/kphT5W6mHlhko/MUgXHZ8z2"
    "bval+XN1eS5AA8kdk409a+hYLeSujgVVGJIVMjc7sbufmscs9YHsK8OZKcB/laWJGMDODHYONuMKJh1twHKmK57QoYwuSIT8+KOR"
    "CK9xoPUFXHL3qJHS/W0VJUcMwx0b7Wt7IVhOzgspIRU4P2qTh5pJTYCGClcHKly5NV/PfA+0Ux45InQY1ESR4rnRy73WaY9DGktB"
    "gUu1plflPA8+KQ+SHiYDw2a+tTk5qNdfnFjXzIim5cQ2f9fEuj2F6XP91YkliGr9RmpfVReMnS4dlfzM9YPbRjoQyDCkl6urStBX"
    "YpwqZ3ibfO35TD6w42uCgIPcNNx9lxkXSXfDWEAhw8vEOnT1c1wwXinGOWZGi9JGLnHlOdbbEonVyvZhRkZUKdEyLsrn88Mll32d"
    "xhXUmv/2pe3Wu64Ym7C3jpHpgDUGgh8ctPp5SDU/dJ047m035Itl1Bkqwv11jVSrUrf7R+VFLPPN5H2MW2U7vqkpMLjr55Xw4fl7"
    "mShvWaqC0et0iYFERVfEG2OF6mVArR2asYEAdhoYLrzRXqW6eTlGx0dWl2UkFn42Mfo070PXrEyX6U/pAnX39ZBJlMZmLVZqBq+Q"
    "wzsC+QAKL/PrwaakZMIAhQ52kCCZF1layVQYy8+P1hof+x8X6mKGPob1AHLr2RzeEt8sy+9tXPOY/iaclIJzEa8GhHE4c5ftHmy2"
    "mfWQIi1VzFy5YtRLVANWuZsewZbSQi3EMnvd2KU/lGaV6ZRO37W6/g23GmkyjNNJTmbKuMGqtbYXvOgexybSvD5sczadYzbY+ur0"
    "M9feMRMGZPaU2JU5J/FQq7TqURDPolUcRH7peze8Cwd+67Ed3s26laDZiYujdjSdhaug4qpYtP/5UXusfTTLxNOTuwnM3uLbao05"
    "5XT1ZGfF/8REG1aFSmePwNwzf9/QFkZeHxkKUQ51L795qLarTGtEaOOHUy5BKtUGq1Grn/t/2Ux6seuFHs43/zV9Kp2CnC02ssDn"
    "siM+KagdLCNstvN1T7Ot05A978rSrEYfIm6FMMVEliU1Ty72N81d3lm5CsPXzilS7PYxGcZX/CWjLijZj03BhHPRIdDd0T1ZXHPl"
    "098YdZDhUU5P8DINuXgtsLiRTQv/1w90KP/KeWcYS5KEerRJevltiX0ZbvZ0F/Fg7Ud9wX6QRpZKZM9Ozqv7fHLb2SRFEvJ4lVSc"
    "K70+PdiJj7ld7FfBIZMHM3OHOycUlN1vR7A6PqcBx73hhFrNx4fcs6Y7pa9//kpYJv+Z/oJG+jsEZzLzWl4fFicGGzGF0hppH8q1"
    "WDcDKMmw33JEfNzZzvlt7JyW5n/DvjYwpX/Jc9T4uvwPBCGa02uUKDvSQbgT3Jou9C/X+1ribiaKttYMR1tFlsFh5hnWqSI1VH35"
    "LWnk3grdLLNZxwgctV1G/EtxLMaMl1NDGMigaz1mMoxmJAc4sc8et9ctTX1N7xRroNyGYOAOCMblHbqCMFw25Bh95rZ5TQDLRD0/"
    "5FGOM897HnPvqgM6Kyb2mNuN49iR0nf22vd308CY0CNMJOaEGhNq+c4uhzTLEJpkQ9vc81qNdS+sxN02sleFwVBEhBsOkaOy4pME"
    "wqAstfDK/6Ba++Bg2W5HpN5Oi6KEXjZw5J2JSG4ixpwDV9oizaxOzSOyYmRnXlllrYaFhRF9SbJBzBjd0hJkOILQYPDJlWP8KzL8"
    "Z5Vdh5abaQJ8ad1v97XgkBpZOtnAT0lDlZqemQlrq0S9BuCuJf2aQBjzPLmFrpTj8xXU9Fy48dczKpNIb7q+jdwvqUCyyMCwpF4Y"
    "oxdNv1tqNzKDxw+3podBEE639epFS3maihzIILL8pOXx7AivkHRmRgeekCGeEGf4RYKCDrNRARJVxq4tZi0DdpQJy1FnQ3Kw0acQ"
    "BCRWEsYwy67uJAjIRfMyUzo/hqwpTA1QGpotvaLNiqpvZXhvfDXfgCca6FcLSy4wVAx7QoGBKe0v3gJZb8sTfahW94zhlkZm4MFo"
    "6h59642u52FS0POlwmoByNmZ7UvR67Bcu3qpByLb2dl2tmAlDEusdTKNiRPJwxrt3taYB5qwOK6kRlwPWEj72umc+oWsDbwYxXu2"
    "LiZIsob9+qWR3FYIh5e8/ATfw5k5rb7w3jdOTCwO1ytVLdTg+qe80N2VBN2vhV7b132kDnXwZ7I18sygWOvERLa6Pz+R5gyGzhm8"
    "PuqvxLA/Ww/5F0qlSfz05KbSKy7uKnVNbkggCx2Jj/APjqnw53WbjIDj/oLL18/KURCKtEL6Mi1J/ZNLnr2RJTxdOSv5EwYcdpe7"
    "RovRrburWjI4YxL54QatV45y4DxdTT/jY2KwaLrBcsr2NnuaVS1scaudGLn8z3kYHFjxpDmUyaOPE2nkk5IOjpWrQbGto8To6Yr4"
    "OYmzoaXAZm1RRRela5esqJF9ghEmXc2ctSGFWL6jDKS9lnmVTNq9ho2kF0TaFCE1K/GZ4FQRsI6phOPm41aluiETLOutHWEZjVVN"
    "0F5yu+g5PBcvNdZAUVzWx5NEa+LYc+R43ou7PLvMSAYeUcLtkA9iFOjp0MPzoCArFoeof/fbVybXDSyUjShk/4ZK0tY+RcBQRmKP"
    "VvVwtQharDGeDA1dRoK8YyQXurrS6B2dqn85+xqwyv8yosfwQypqCQjGY7TDwsuwrLJxlqXkqCkYPGYT9Gn5PefkoUu04C4lddrc"
    "ycNVy1G5NDTIxzU+9JwZ02IRVFa8AtynouVVgvInueNFQhBhhpN0Y/Qy+ddbYV6t+Hp/QVrpn1IoXtdw7izOqxQ5kjiXKaP12kxc"
    "sVgt2bMOHDIxMiwhhGvi43o9voEAKN8XoMFCJOVOBHqYt71iqxfGYUSlCqdSb9Wp91bTWc8fBeFqEbTDjrOWL0NVuDbiq5BrZXdg"
    "olZpfe+Ffj0KOvBfsdkO7yrd8O5J8n2MYARBc+Bh6XYriIJi6/rnV7kLqWxXklCGI8dWsuxiK9gYYUx/pcjK2GZmqKWk7XQOZZ2n"
    "sokruzblKRm4cG+zzPXMm/BnmXAyoED19Eyy2jQgYHbO0XUZkHHbWUb9t5xvy96RXgU7UYoEjnJcK0PNH8aI5ovwB3YwHERE/esX"
    "LMGqcmnsjOQe8+SM6b5vDZFgtRtcqbG3Oq8yYx2Lq3EOTXbbkVjJ0Skh+IRI0ZD4UCm66i/NrmZDp9YC6TwFRFOiUJIgu4SPVGZl"
    "sFoOPWHLRPZK3c9MWD7a4OjZH2Ipq8axIO1o9kJdhqOTyQKtJavVcnr993CwPB0sEPcKUoRNRNF7+WGZQGvEKCwbWXScnGtovO4I"
    "3yZH9bi92VtjtHmq2MzcQVT74/GG4h52ypkvajU6ratuaXJ2V7mDN/aiyVy2WHb8hLqt8wLk0Kd+JRyT7bxy8GYSUdph8tRx8h3/"
    "DEuwaPqqUfMTpIZXUqLU4foCyWEGUalG/XXNnDCIDtlEmwfNnuiVeV8nFs71PTNJOwFRroF4amD8Fn+KMt8tJr/98p2IpiUA+zXL"
    "wsWlalUVkkJ31chwxX8lzMYYiuyeg5cMV50YigyxGZNxn9dqzKu35y8HxkjsHqGt0xDzzJL8HUHIK9g4IFbWMIGx2yWjmmxkQhwf"
    "U6SYKvPqujEjtJgv5pBPqgEsxmhdK2+pvP+lNz+2vmbuZejUSVWSuWcL2ZZIDpr1q8ZPTtDnTKDIn1T1Y+ZJpKtFmv2lBewJROP9"
    "qxa06r2oWInyGj4laaCgopyhp+eaXNviabA8V9bwt5p67lHvnjf1eFJYLCZgQTTPqyvcAPopMA7hdiSgXCwBaDFqEUrJOWGGFHOg"
    "T4DRfM8xZ23QW0eNgS6HF+GrowZBWiww7+C88ZXD9WVeJvsfrPEQ1oc+W1JLcU0aud9/uFwxnHO11F4V+52o2O5VgrCzKnbhMLVy"
    "13Ndd8kQ0OfP7IXCgjI5sDhad4BJD5LygyGCV9uLsAyo1qnEvXzeJOUPN5sNluObUZw3rbH8xSqIuGiIH00+rECQvo+4epnU87MF"
    "qXzVuDTLjBaG8cHYzDnHBYsogJjXsGGQjIrHB5EZd7/i8/yqs9Oy1lGlGLW9RaXnT1++5j1UrFXW2crtN1P6drf02PP8VjfqhB10"
    "wVaCF9h+Lx2/+D1aRd1uJSp1fsKxmdF0flJTgtzR19f1nOHMFJHcnGv1O52ZQhnOedGlnJnzn/Ascskp883stw6msVnUu3lYCSoD"
    "AoA+kQs+PF4PuSYXUFyaOchpX35hhCk4JwVQk1PblCqDKvfPiejJgcb0PAdp2rtupnR0YOAiF8+H+rXPXQ6cWUZk52PKeShlqtu1"
    "bMos/+nXEzm/0rRjlIqeE877cdu4Xseb2x+eD5RCK7Rb+0j9OCMs+UGECZ7kI/s65lWuQ8LXFoycUTwBiWbnQvdSpyMBe5susGlN"
    "r2VoTArh/d8TgOATmW8kXxlCMv2J5XZnMI3dJnMXdP56u6J02V6P+LAcYFBkuN6dxMzIzJ61AJx4Xw8sZitKo6fF3dWU5BQDpTCE"
    "vSQK61V1IBbZOFeLJLNmjjFndA2adTYS2hb+77yhxOVrQ5fIluGwM+fzxvLrTTXOlAqL71okID2TFy/FVZ2USI1D6WU4tl65+m/b"
    "bmqtbWrcFzPkVeUT4XDcIJduhLAO1hsuFfv4LYn51pEnT4SyPlK5+qzWA3eoLsmwpLe9y8wVtjuh73NsM8lOsenw4jhjRO9TEJbM"
    "skhLOoWBM2/kTBjIStDLe+aNNGa9QONxS5Dl2dwuOapMQTSs43v4pIcjTiZoXttaYr3zYuNYZbal/IRGZ9+e/CzvrpaSGYnXovbn"
    "9JUCuTyFd+Jw5au/W2K5vw7+eu1QPV+G4t72r9zXjhqNxVt5YSuSY4BE5dtzC7nWxLvrog/Xg/MAiudsZf5sRLamMjLknSXiSdhF"
    "iZWl8sMJMistTlr+ZFPHQLKFBZrgtPJZIm4j9yvyl4F8a2CD8/zZ0bbrA9F7yryM5H17NZle76YhcggFSCbmn/OB89tLvPNcipyx"
    "zlW35boLpCBtX03GzJo8fdZaZlblAb2FgwJyhQQ+vYe5pL1tXlVhuI4XLApbo9R1qbnAFiSOe+jOKGCy/Lrvrb6DEyLVQjnBWQov"
    "vtLc5USPWb72g4usD1iW2lFU70VhHDIQb9k0lmrnBRqSM8C6gvCn2zCIuqEf1cH0L3fDoDfw/FKfIDfZCNnQcvvNrJUL7Ir4gUFn"
    "HNWv4CwImyuRmPV34Cwk63wk2K+iXXSBjLA1zEZrY9w9ds9wDgXJAPJeR4XRoI5QDVYd/Co3boqEwx1sytmkFkj6qnmwY/wDtzfZ"
    "KDG7PEwXnOWdUh+s9K3GyE/78Q3eZCeFURmZ6YoO5tgs2mVXPFRKveN1f/ivZGHkr6CQDHT2sEmhiUlf6wD9XquMMoX77O2lJlmf"
    "Xcnt8ntCKTKGknGHCxND3lw/YWmQaBkKDOo9iBSs4cNsu1jrknNGXMghVlurw0q9kfByeXv8FVyccUGjn7Dev9dr9hzqQ6P77XKt"
    "dNCim41ZJRNeE0Wwo05XPuHEyGhQze7RaQ4I5BALHIfbk4CZsL9ie7G6T4WXwkgNq6dPygdmEDsyVPYFoCHrAjAPMqbE1D6aeTgW"
    "01nOeS7TbJ+q+qY4XLI6EXoHn3/uVT6aD18uhJaGULalwmp5pUNDWpZOcZDYLppYEaFx70seCeVL0cVFIx+lzK/g5yTTGmMYTcI8"
    "S6LkJ7BKI/8LipKpszm2mZ5obz+XKublIFG56qC24yClPPYTqmCLOGPJw9FFYMl0MyyUS2XnYe2X4We3r/X1cMuV7mXhloorRual"
    "II8iwxVsJJKqNdnOc4qyK9fcJ8B0bDjWmo8mOdGXJltHee1PkPRQ43pxyYIrkVysSBsxjNF/1Xivr2W2/0pKe37cQmtm+6+ktF9H"
    "dcvMbP+VlHaZMPBzme2/ktJ+PUMjM7P9V1LapSv+5zLbfyWl3ZDhtuCho9Jp+bV6Pe302FPTczACC7jUcq2RlY0hlWIuqnKVnDi4"
    "1oK9gnCA24tUCQ5bnKEWeFkFVq5znV1gZTNszRIifZ2TeSm8EFom5D8s5dqca4Z5rxmx93ay6YzkbEyDuxla+frcibtZ2RdfitCn"
    "32mj10lQ/rnYpn5P9McKNLpRzBf7r2S+ZkZ1HRkZyQMrDZ8vZWR8KY3CVTegwW2RJFL5KQivlI2T4ju5ku35K4mwqip4zwqhnQft"
    "66b6V/eDBgD7nVFDfC3Zh+HhHDgxdSObZr2F3TuPC6f9aFD7UB4krLuDu5jbx6kQmhN3Nl2/mWhiOigxCIFE6E1baxKIg46gEih8"
    "H6TcdytCi+k1d3nA4xxRYOyiWSJqJnWkw6K37qi9AWRhbcp6vg1X/OJ6SMWZnl7yx2ssyiAgf1eAomg3cn9rSMUdS1GeCw05hG4c"
    "OLvNKwFBI1LH8szy8nMxObHGwycdGo1cTomArMoccS5E7FNMGtn4Zgopm0kd49+4q51+Mld8LLgCWhW8G2iNLtxCNwKvbR+IbjcQ"
    "aq/rhtqTSF9fZZ9j+WVwIDe2qG8i6yoZPt/hTIoJz0Fi9yvyWvOp3ufII3X4OaXgdIB+i98z6tYMX2omsLM/5llTB1yzn0BkXLyV"
    "58hguLctnibDrTjwTk3kijcpy/IIDnS+5c3F9e2qFQchV5WawEvIC5SgIQLlwUvITvzr3E2qYRF3Pk99uTSXRye2kmObZZb76pPt"
    "OICaGW/FLUadXOGb5XI4/U5Pk0wOyedw+p2eJpVll8vh9Ds9TRkixeZw+p2eJsWd+rdiKNg8TQLf7G/GUPh6UdVvw1CweZpUwsDf"
    "iqFg8zRdd179FgwFm6dJaClfwlD4PZ4md7WL1eH0Oz1NGoh3HofT7/Q0yXOdz+H0Oz1NGmhJ0uFk/WTO0eaBS9VYRJOjskV9OCWQ"
    "3PWjauSNb6KzAiRKp0/AM8dpdb53pcTI0aSvdovVmVYZrrKXHV2VyMnREP9K2XjcmBGhQgrJlARlVzr7wz3o3Y1kMlGy4J07EA3M"
    "kyW7Y5PxaSkqkCy+X0FgYIzS7whXw4o8crcAFWTD0LZvl3h6HDU+r4X4CAqx9jjvGuJPJlFGbhabIZxpUA9noLVs3851yYRS23iW"
    "1LjS+7Dvx1KGB8Za1qpwifYjlMmCI3NfC76BPh0dBzp61/I4HxDEKWJa+qh1fo67c5cHwulLcQhRPXxqecYW03yfPs1xdmBhgjVo"
    "tPjv7OILrKUtebBLYfIeiq5DoU9aYraKw0JwGA3qC+POzwayUK90YPtM4+mjLz+NcFuovjtnZV0sYArM6OI/tkM/CFfBrLMq9gde"
    "J+icHXp4teVL/HCeF6wdku99k5/F1XSq20ERMcyYh+HRpySwK5AO6hPR8fsgtZ0cevYULsyDABO9jAsnn55/96yWR0ZWvK6FvlWn"
    "SzibiwmO4rz4Tw3r8vqYhxJ/4KFx3WygFqCdBnsALEzYN2OC2/O/R+Fd2A7jl4EX9HteLEu//Vaps4rCflRstaP6Uze8m/W6x3kv"
    "LEa9uFPCf7ej0fcu1t8G9QCe7YYrhMlsNTth0G2Hd48deBeabPX86CWqBAR6EK2KdazNbkeLWa8C0soLavA8vJdd4wNndVLw9vR/"
    "NoL3MRMpts8/wqd7yGDUDkcBssr2oEsdD0cJ38gYNfS+57eCgV/qduDVMK4H7HNRCXbsS9uLnLPQ9k7fQ/w9kiXB33veoZRdiV6a"
    "jTZtDYJJH0m159cHMPGVgd+pwcT2upVis/Ow/VKWrH3N4cCtgqgXyFoAuY5eUO0gw6vnw2in2A1cvwrMRCkKT6GYrGuz06lE1XYU"
    "NaPAvd5ywr3GF9c73R25aPYtm+wOb1rbJm2YYBh5vdth6xjA66XeCj5J20VN2O9ba20Uo+9hPIUJ7MyiyqLbiVq1nneqpDWT6QJ0"
    "LUbNa4gfusmOSPcIWq735sg5lQoSPoIuF7hpkFwYRcl6ukNz7TtoL5hLNHmqx6M1pbCh03JPGg1z1+Ezx+GZE22ATT0uHBav1QhR"
    "gfB3sMWcCpLlOr8ZDmIRjDBcsMxcD9mliATEvv1S1C5Q9JstoNuohB3w/sdKZaGR9vzZkB2GI+JcjcjhiE6oE6p7LMBQ8Iow2qzR"
    "4Ra0XLKrhs0TqbDsduN+DFZBfTd9WqFJZ01ndOHCJ9RAng3N9Drb89OCRJMQNJums5IZQhwfPosYbdbUfeD8knUFL0Q3edOit5mf"
    "Jz+ZY1cLtsGuw/VueU93xcvXV+7AUgzKUN/fgUV6RpptkeY4qhY9cnLG307Psqv5SDdSnwRNszDKTuLMOtfXZqe9vVJrj24dOGR1"
    "yojFfZDaju8edfG6WZjKQ7yRwSbbGSaKNz1wsNe2EhjD/rTRvUMnFbplt6ZbduX0jcIZ//EK5oKM7gVgMUwXfFLIOzB+zO+eHYI1"
    "Ajv5CJINLJbibsQYJLdCoeYUB/GzCr9KDA3hBxc5voVsBhPyQvjhqdWL4KKF66Oi19w/bM3f4f8P9RCuk6hyy5+ry8CiPRt+gyab"
    "UAVhTfcwM9tpebIXqMqN7ipLPST82cEN68YEBXA1YmzCsECqdu8WJkNEXE3qKHcIlVywE3Q+5gi7J5GY5TZriZwEuRYuf0o+Mg4j"
    "R5jvcvmNrTXssqWJMr33iuNY4v3370AOdKR7/u3/4+1NmxNJlrTR7/0rdPtT1enqOixSddH2nrELCCRAILElS1ubjE2ASJYWSAjm"
    "nf9+3T32yIgEVffcselTVWRkZKy+++OavZt08SLGmI+fqqk6rQjaVry1zuVdBlpegxMqg3q+ERSmK5DPiD8J0dO3UAEgGw3ugdzr"
    "aBNPd4+1BbGWloxVsCq0cwsDm+mgczmt5Mvc2DGdYzKGcRnnpS3WdJDFHOYO43UxM++h1IKu9QPbAGGUXhaPMtyc2zM7hwTsEya8"
    "UgHx4V2rjE5kAQC9q34sVPmptzSDgwQNjzoIoWkRJZBUv+PK1+MOBuzuRIGlbooit4gh02Xj/9YqvCN/hZ9fzwgCwRk+wxl4xtUZ"
    "khdwNoy7WPrn787IBSjuWO1EAYGnOZHJN43uGPhT/d4Ye3N0i4kNbiptTnN2upRnTi9OacGUW44kh58LqFc4WuWeGMejyA2GQHIL"
    "0my3vJfmWbgAc5/lp5DxusGv10fvTG9KGUtm2/oA29lMEw9dNioByO2SMH+APx+xQN6IMWVyTqgcH0yYGOr7qRjc1bBWTMI+Ll5x"
    "0ZG4Vm5iimFFh7EddLPyrABd2FUPl/Je55mh0ejC4yrVbaD67HlZKSqSBNLNot8tucscUALe7CjB+ePrnJbmrKqNo52kchVOw7kk"
    "Yhc29QJZ3LAaO1RZKpU5AA0/+AOwRXCX2R7kdhZndF1Nesu4au/IKDv5qlvc57MwF7QMihHGKSWc/jDPBrBkyrkLjMjpMj+xjz5a"
    "cNlPlYEVFffA2Y6V2+0cgwf76CFeJTZmUpW5l6CwlPJTzTa+j9i1Ky1nkVuzLoRHDOGf/pg0kjOAKThNV1yL4TIIDmVE4TBXzJTY"
    "USRSuk+6kiKYejwC2j9rIVa36BfY0aqLeCPlO+FMtr2lowuK8zCVSQyuk2g3NQody3JRi8rJZGeXYyHGi8cklMy+T2Wnwhe7tr2g"
    "ZlGbGJOzONdyQrBQVvF+2krmiq0w0yKDZYIMVN1uolgNCmE7aDeK3cQ4IBucaaUSxyzK5nOrMYywz2swWcdlSWKFynLyS68aaPBd"
    "q2CcHz9+uPw0C387Z3a1arOdzHeTa58ZwOhTcyya6toMFZb7brKGdrPrZjFTbSXHT/X2e7sZNIIGKDP1xHu51S4/deaLykfSDsUm"
    "xsDy9NN4nzH9d0sWhxNFDKMBYidkNqPAyjiJunS/0xiSLQRHaH4eY8yisYS+WrnqXacGIhEGsCYuocEIUqDti5TFDkomW8QG/qB5"
    "bxxmdFFS9ieOmtdkJ7w5tAkGr5azyetBX4Vt6bptqIVd/Oytyt+kErAaYrNUctHrVt5PAooB3ZCEsqquI/eaKsQWMRjf/EyKSjmn"
    "uikQguZ6hF2JlWFGDK38FPFTsAo4/D6S7APUPJk9nN/LEbOCs5YKEEds08yO2me20xUClozw2M6jtEBZFbi/sR9SXb1u44oh/sye"
    "YM9AqC1lzJkpBCC4GbtBt55BecsYoiK86DXCmn0veqyL6VjkdegvI3ssZDY4JssS1qvv4GwCciqDRorx43BvSxwnS6kF3WPxEoGK"
    "xIaBJIol4zagCkjFBwSW4U19y8tvs5k9Jw60VzJ69h0NV4mHVhVLaBsUqycvj1Ibo583AwClVYEMThgBfxzcFEE5me6qTTdQO4w+"
    "3U33KKaUi/aIGjMTpgFrI3SOqB3DK5lABxolBnJabELYVEDbhJPOggrSQBC3LDfXFca4iH4K3eqdUSQ6Xho0DAWXj868QFSUg44c"
    "yVsrtA6w8tr9J7SfO7jfsZcqbima58a4fG+l276s4wOv4knsR8LMc6/97hhP+bEHR4tIiuPI8LOCaSqi+7/gjDzDUY3GIcFiwypb"
    "BcedC2OtiDEUF0FlBkk8D+J4wpQOKGQXw7vWiXBldXwQZZF102fOf2bvplFrw43ozqfAanrB9qgqxupARHbXyLHEv7dovB67/Bnx"
    "Ya9cemUAOFGSQvYxo6aDNL19f58AewBevhh0EiQSurCxRLQe6OT6RcWCxknbiiyOmftySOrWizE+s73LAZ+nwDE6dipkphGCAp0c"
    "KQa6duw1J4IVEys+lxgY7jFlVDZUuhNxSirQLwe8YozM9qgbNHjCjBHScGPPArTGVV10a1qFPcA1LAmeYbHosU4yOESEPEWPD5aF"
    "sxIdZ/G1mQxd3JvpJmML3YVwouQAywbxm3Ay0GsxfQASVE+E6FDuNNtBt13IlFth46m9CHLCdZ6YFW2nb8VpkDJC3uAsNN7GaJZB"
    "32Yaa+gGS6beZ2NOP+NwFsSxMrV4bZ/WDaglufzlvpRzL5t5VdGzJ5klxaxAu3HnPXG3YuJgKV/AWh7JO1co8y0P9J3LIVYcdjPm"
    "mYGj1iS3N2kKBK1z24BTnaBLARftiXtwhCfHktnQs0M8QIZRiSEaF22ZSVfzqiCakCCBog1XdQQegUUB7mN3D/d3fCzIHB+481PK"
    "9aNy3ASRuzZZSo6gXvrd0l4DrkHdW+EgUWrIJUoZ44PJpwmMCtR45q0li0FaG2KKhng7slbgPYT7H/kc+072UsvxsT9HJ/VazW58"
    "ULOrHjGdN5e2ZncgR9R1VoNs0VbkuRd5R/k++Kvuz923SvLV6NGDvWtmv8N7h2F6tOclw1iE/XOB7rZcOUZutiDRCjMOf5V+4o/V"
    "UDCEdRmuBrf1vUMIkixGnH7TXloTQf6cHyjmLBPoCG8SZGtqukHvmzz1wY5mc7+iI7ZVM8p+k7PBOGNWjdKx52bMg9CxlTBMzBDE"
    "gQPrLgRxbiYvSSeNn81+Y4u752ch+439iTl92WMNM9Zc+48RPDdI74s8OYMJYPKEk2CzB/GQLsMAxDnQBvm9Tg7pk9cbnD165cWR"
    "+sY3AsFBp9VjG1WDjcUsnYQXTfqqiuicdK0UinFkhuP3sSzISMVVTmZA4TNSHKCzYofL0AXD1TFnHw1183brWLyAUb/b8qafmpmf"
    "NGn9Eo4gHjVj4VUMEt1Fx+i2zq7NPZUr5i4xE1nB6UM+d80/3aQYs8x1PfEetBPhQyvM5FuFzHUzgfGDjWK72Ci5AsOYRbAPmicZ"
    "tbWhgdwdvVhYgKlMwX+Jmvh0qxi0mu3MvR2mKOPFDHk6SlZ6yAxTu7DPrYw8hztEJ+T4Vim/JW0jldsF95PUb5A06LigERpPbOl6"
    "n4kzzwspVEbIp9ETTPkh7ywGxXW3LyXncu4nr4GbT7yf/9nGBmZ/BPER2QfelGvHZtYa7eQD/3Q7DJ+CQnjdTFYji3nO4v3IqqlZ"
    "x0oUZ8tkFt+Pe0/o1/L1BVqN3i0ZjHviWSkCwSY6R4RFPSckhr+vMlwdhDSCcyE+adFmshwb3WHeSKe9q8XpYbCyo3SwHTucyBzQ"
    "YEKhbtpM898FngLJbJYH0DrNKInMVrCws/Ft48DzBTCgILLoSgUArYLVQsbQNlRGecYL5unoM0bxfkXZxnwDDGy7Q1IOmWepa//O"
    "oX9bCseaUVrbJ5RIhGWBgkBMHixP9OXEU9iuco1pRdmdTr0U2WoUsRCXvFyZbqMd3jaLdUrL5gt8OYlaBV7iutP6weEbK2RoNi5n"
    "k5xRxGrvBjkQpMR5FBV/Zjn43/FMCPO825YSL2/dbqw8a6H+8RNMSNqRYdjxTRLE20ACidEQExgRXSuZ7hZR6aLqKNiR++bCNpyY"
    "KKws8ZnMLBvLCew0z0xCBySmsSKYxQZye8vCQmRmoigyp1aek4yV0dPbSWknnEz3aNAyz8hdXu3z5DYcgsRKfhN978Ws4b7ps3BF"
    "XZyR2u1MvLJgAs4IBCKroP666wQ7kCQiVkFd6eXHkMyBSjaLfIoIaHTkDtW+EedMpmMlLqhiPTXh3RPOpFpzsXWTjGSuU8y02oWg"
    "2LBlbZs95M+zsamSBM5gTCEWxAVlAtuZDVNmEp1ph5GqPvDwXHKc98abyc+ckavJnIpMN4Ozsh9366+UNMsS8Vw2cjkMyT7EaISy"
    "E76OboNEJ0EZxFueSQycbQ3/bcaO4A9rZleYYLuj2JXrqgj0M/qWqd/M5tE5JN67bMTvcSPWF+5u1UAx8jiiKOuZ1WYPXIyYMWqi"
    "m0qwI3sckC5xue7mV+fPBK1G13XX5ZkNuiDBFjG8NQgbsIHogOrPy+HkNoeXEOV0rg7kJNwWGa6y62b+vYunvVKYboPD/h3/jobK"
    "SuFyWz+8l+6Kow2x/3R9OkmZCyhW7qlZsmOZWO41b+cPYWVhi02/qc2ysTmsxlfiyMWa8mS5qA9/ZtXrZLaRkHTLzdZPFd+GduA3"
    "vzmCmp0Ib4joWJLCmYt9wnponAWR4+M+Embgl1ys3FGcjVpzr+29cFbIY8nadRNzS66nQCNxr8UpJqCC3G9PkdC0Mj+ZJYqmlfZx"
    "V0i76suRv43hNMEWV0twLli0UXI755RszjxA5gm2F720Yie3NAclR5nejZVikTvsU/QNP+IuS/l95ye0cewcyvzEc0A4FsQrTrAd"
    "PDSXgQVnDEf6PqKjihmOYB18daw9zkY2yxHkX5pKd+o/s39y45gcr18qPkzF5WRYYzRC0hoCyFQNmrWBxOce5hGdSeRNWJWTIC5w"
    "up0zpiEBYM2687A3b6DwRBLZfcNBfco+hqA0v1ZurPTxNKwKrsxhIQWkuZNT8ShIjLacgeZYVxnnFtXyDwkDSbT4lXzivcICUCoO"
    "UFDbMyNdMo6VOAsjCT8nSApc2D6zXIRKtacZWoKNAxujgchQb/1lb+4yePDjxVGEwi3ZWtKNNeMFma1Agqs51D0LTkXDlxTmt8Tg"
    "luj7G3rpOVRi8vQwEMwouMTityBM0YpIxed9wbHsjrX6eXHD8XD1JqiFpf4Z2DiLhBPjRBYlvm08tZoutBe5KeGkA1QOSAfqWxTz"
    "n6h1mvzfZtUM89P8VUzzGsO9BGK5Z9WKsq7PHMfA+odoF+mUk/2kfHc1YOKBfwppDF1OoolvLlFiQPPohiHzLwOh5EcERx8pISJD"
    "nWqYZkauuN4qWMFeXrFKCogtz5Ln495XyZLebkTgx7mzkdNI9TegFIWUzZrGnHx2nsRQfCfcPRQVlRO/784N6cNKjikiO3zV8Ejh"
    "Im0mS2ZKbyyLaLT6yB7/0PkQ4crQhcAN7RczywFog7EB9cy9vkZJ1jbNDYMoG3Hl/UjzrAxLexL8GfbxOEpxSTO6uDLRCuSvY7QW"
    "TPJJiBWkNpz3aeOVuBP+g3hINr5ZJ5NEGERMTgY5nMzyJ/Z9jR4gBK9gNL1P0e8n3jGQgVRirBuK2qLpGJjPMaZDYgU8fdAxK0ym"
    "ThmhjxaCkKBmHwQScunaAiMLkUbYiUdx0n8JlZIbwr2tOxMuRETsx/fTXViLnx8hm534lH53fZsyKmaAksGu06VkqYKO4UhEbSkM"
    "O8G442ZkgHKbZj4HCHQxEdkcBUbkLs3XjD1K2udhk5bw98hx2jmGwXL+pER6IvWPLyrP5/pbnxTfUtbC2E/ybJdXILL12LtLdpXa"
    "CyVNxt9xmUCnpXj7TrhMP2KnlU65k9qx6kZoSbBdcrMeyHYjzIRMBbJ6kSutU36yOAeqRbNBeGqULvvd+tq1Ov3uLEEARTK0jefs"
    "raqRIxuJsrvSXsshV8Iga5C5Qmbx9/No5mRMg0bZHYsMdnSdrtEM5LopFpYd90WBJhGkBdbCiCE5siBrL6H1enP3Pgel4Ndnx475"
    "4xRYMmUZI2pfKFw5XpJ5VjD1Z4U7+BmoPWuHU9rqwyObGVCHnbGXo324+IY4jkWJwiriez9cBCsusc57LpgmIxbcDd7plTB4O+4P"
    "uzqH6EbeUfjhiX4nuR87UDZ9r+qLjVmsp2ZpA8TqJf/8p5FRoGTm2MPEyKb31It2oH9vCLgdPgl/LzovpVFgxZ0v7xSIRUGO5faH"
    "GCasYkrudeL9R48YspYBhlA4K1JqKgBB4yK9yODKzCSGhp/gKb0KheTwGTFxMIN8jMT1JpPGIXdMbELf51QVBaDpys817Lq1Sbyv"
    "m14yQ1RKD2smJae4G58zzF63v9LfxeGKQCCvIptDtB8QBkDu5qXePDPc99DEs+yJvf8G9/ttdJgefKYeV1KV/GwDZlncAgc69dkE"
    "LOa8TzScmXSdBfSMfmXNJrx3AnzAvwLwSrq2ZQAWItDHTQPGNztECkFUimM3Dfy9E21rMU1tj5SEugxmfRYw4r1kenvmvw4RuTdW"
    "aVa2lBM1UuMZoKcAqqzR6BpCFL6DYC6JZ/vVdGozui2/9fACdRLW0ZoBo7xct9rFejfxnqsvgmr7sPgICqsKwi/uuXkHFzM5jNlj"
    "0RZmv5F+7ZvinCN6npLNXK/TDJfF+aDzLjCx9iYmVmRFQlh0YaLHgP356CAckRuQ92SOT+XmPRzOZ7V6kHtqFYr5Rvu9VWmelTil"
    "YyahJCIpmKVARY6obVVgs33qrYJlzD3mchgKVAw3WiZGY+2n1NXrxEiK1sy8fBVBtFA4w5oC4yGOIA5wt0l/QwGfqStM5d6dEoYI"
    "VYgt/BLNs1xrlXEpaV4L2XOi9dFqdP0w7DJ67BcXrc92UNBGfbu8kySlHcu1rBnLrCeBHe47csaQTditpcKeVTYPZO/nLFwnmLH6"
    "W7UNEynOHUK4wxujUGJ2rFhG4vwu0uUQuNWVTqmsLCegfqAiLpMsLwwuHb/jawekQzS7KfuXmW6U3LCZBoiwSpexHQZVjLTFKFsC"
    "DC2E10EB40WrRnYTZjvrlmdN+3hoJ3PtYqYZtN+L3UQS8Snr7UUmX2+Xy61C2O7I4CArPVRSOBbA6a4kTcmxRFrEjZBKLoWbUnBA"
    "1bTYA1sph4Sq6Q5jdVEv/k5JRHaYBJpTRulYFARShic5oyLPILJzEThGmRBki4vELtGGKTn8Ow/E3KGxki8ix4i2PQN7FfRn7R8+"
    "d2ZECd+ZiG+BPiXnUq/e5dmoDI+sCuzli7mfWrPFzAjaQytr2UQPyfpouHki6SK8H02M4HX01PJj4z6mQaHZrrUxhKptwB3ndion"
    "196zDLH6ZDloFRBrNLxH5NVWogysP6xi4LUYkuURlKvSPXCoa6WDG/dcqnvmdbf80/pd1s/C1tOWLhEMYTO8CROTFnvH3iCHO9VK"
    "orL2mAIB2N9vqyb1Sgeb/nUbJNBycoiSKDNIp/pNp79Uus5F3KgbWZmoVtAOC7VWTVCvKpCaYquoiqPxY7VH7EPojxmteGroeCli"
    "1xaudGAVLedGQ2d7n2uzIN0AiGWmqYl91gpIxspIzgo+C2oh+US7/bBiKLnRdE+e1ukmjCj4YOUE+QlVsw8jrhW4oDenT3x6rkCC"
    "zKSKzNvQTdk8lmTjs0SYfX3IBS9z2yXV0zxr1DBchzYin7kBibKeKic5xgSTjRxmQ3QTmfsWRlZi1ktSHqvtictn9eEnLY5IaRdd"
    "5pSqFRSK8HrtqQ7DaizCQoNlxbSbwKfbidlDq2DedStGwdDHHA6I8XP5CSSWGTr5++4YBhnEbUBrWTciulHbufKnXQqm+eSIEXbQ"
    "ZzlrQVgb7Ua5nmyUW4mrVhQPPrfzE+haWyJpYxeZa+jiptluFAlDWgfwtutBmGSEJWEZthOpbZIhhJ9q4wJHtA/bdoJRtDMGfxfn"
    "rzbNNWmY7WKYpiRZr41GWQvV57RyAh8x1WglKo7nWJUUeCDZssbLsdZFvFowuikugKotu6nZZpSux8yQm4Lmbk0TI1zP91v+I4YP"
    "FSl9lv3jnzR8qPpcsfYPkysh/k34quvSYsO41VEFd3cxeifJQceUye6hfSVhb0FWjnCuk6eewCLpLLAU4ACGEu4o1BHhArT6x/Gc"
    "yxEOoXUVRfsKRyiHd84O8j2aeUJFVYuN310rxsxr8+xcPQ8YwuNZNfki4ezRdOBIDo/cqwChMKkmbumGL7iqjTx9mOcUkWWcDusG"
    "lFsLjpyfnPW7icQ3HzxmlA9XTAxxWUUyBc+TWjCAQPuS9EAWxRNZjwpwSh41l3n2ppjqG6gQOrpT5MiZbY1+GOo60HbyFtquWiOB"
    "jkGKq7q3LKLZBrfgGIcRyueq7G0VVTGGoqG6+apBGuWD4J6PO23KfJLAsJaZvYLnoVhrc25YbSVqd6DFFEG0uAexotAM4BmIGCJI"
    "25t/ayKI+IahBKL3h3pih/AcwHyvirIuRKJRxja8oINPQKq4Q50Y0DohqIbh+BinyHI3mwmPuh3dzGb66kZTv72XpN5+rzcLRV5u"
    "ol3xYNpFVsU6AxLsWXkBzKOFebT+/RJVMuS+4YZ5KpZ5N0PW+yhW24VZrpvo59rTtaeIAhoyEFaNZ5fnp++E0oboq7e5da9JSZbS"
    "U185iN8Ri0MRZYEMJNGfYnNqvW4yzT8mhxVBm4hWt9MLmDIeWmf7EOPPMOpkW/cdOJ8XVcaSaIAN5fYyPvwKVMWZwwmxx5OLOHSI"
    "P0g0W+JMNh1A7nH+sf16GlPosHKzBZZe2JTQhD7PrvHTuId61LwtkxmLSsYtzDzf4hl4YWnfvWhhxKK3hJAMLffMgqcMv1e9tQO2"
    "mP49q+q1la0+ZOULM6pd51hohDZrpeqIfInp4JilPN52IqwB1WrHoOnzvi7J+izwSH2jl/BqdeMUEwhZmKCLFVsSEtNBWdnmvZ3f"
    "K8VCU0SA7hM2LEvk9OJFyzvs55HP47nJ/UUIvresfiMCS0pYHvL4WF7Z/tOAis76Kx5oK4OVKNcEiahw0qIlR7JnFjCNu5uN4SBM"
    "soTXZs495OLur7tWccwxLStxeKSccmGwJqv1wPAF5V5hrRZK+cXqkNMtz0LdxG7KsphmSRsBFaoHyfVptD+NFY8ojfls/ALCMG00"
    "IWe93XRugRbldgHLgvVv0AAm9Ot2Iw8cCH5O1tsHBhKmzf57qZjDGkw3jZA7GpKNZp0S2xvAyWqtM84CCNcjw+0q2EeMYz++zESw"
    "VbO61GZFibS1drH67YyC4yLJbUzWXPrzr1J+VocVeWos+rwSFa4Koi3iZizOqaeqxTBkES0qTTiWZmVJcYKxcDwmYVATOCYfqd/i"
    "Y5ADHouoRdyqzCZGpZzo2WcKyd4cIQefHyv92rQkZNf+KkaWBhKDPRyJRVHaZyXucnG5ioEAi8tlayWLrWs1HMfSma8tPs1jhzwq"
    "vC8MQrMmCKtxzlJ+I3iGGaaLLb45vABx8pR9XDS1DmmB9lml9vsguqQn9zI+6cIvDpo5t9n16TK93AajgKe0xfsxYAM9mbZ6cEPx"
    "6cdZRV5F7GOkTUT36gyN0oyw8sF/yAVP+C6UlcSsRPkfMWZo/+2kqGAlLzuQNs8BZYeFjsK2NN1Ao3b4E5e5/QYry3Kk9s62FpMs"
    "318GMzSS9LtVaUsT4P0KU5rixOy0QVY4JzMbrRynWWEsnJu1aBBnZZTm+fDQ7fwKFnmD2uZ0GDDzO2mV7YQrhpy/199MNDT9yTI4"
    "DNPVtfW+sWkS0iGmvLbVfee40VcDA0C/9QMK9tF/r5yqu6kFffHXloPujCXR6CsRbr+ZM0iOzwL4vWVKkBYNIPU6zZPL1DsTE5q4"
    "U5fih4dC/eOxaH+B6l9uwTCCYsL1DgXnkxopsXN6XJfLpTSMSntR71aNtyHJ4DY8JuNOo8NVOLnFykX1NVeEBPzDsg+nHziT5GSj"
    "A0eYsBBHFB6pMw+X2UOC40loVO7HEplwKhgkALrN0USEU5JPU1a0EQATpiNRBX1kKNoyPuaEfQbYyvyOSy4RaedWRndIQhqxbZPg"
    "o3f3Kt1u137DlT5sGZ/ogeBS7lRdLafYEgcWDs+CuJxyX7ZwsXHrL/w9sdWK4MVH4MksNgrE83ttIx4CdmnKTe6CFbFHxUwxWATN"
    "Dvo527U7EJwRk9iJ6GkGDOizVKGI2v6MWGIcbYJwuVoqvzzhcLIVFIgDq1aVEFryqJvjOX6L4dSL/89CnNK15LCZfQF6wKaw9+fk"
    "OiKj9PIBApQE5Wdyj7KcHW9I692Bg8BiLTdPPKrExqlbULeYOlB7GnQal3ipGIAgy9fqp6hChj4MC/60hhGXR/aZkk5EtXdqQ+l2"
    "UZ+yMsSfMX0MsQhl8FeBHMrci18bGu/W4zZCkRc0B4pPL4MtFb/i3fDInA9Fx8roG0KNMGbu7EdjH/5gW9I6+L/Pwik8VQeIVkHS"
    "cO14oV2rcNpL7yiKhj4Ra4MocMgKnWBhNAJGb+8uISQ2gI7GHC2Edje1h9aiX24UAqp97Svf7Ap/s8RCjKZBC7705llZpdESj8yq"
    "EKN16BfvmUfp0hRkrEJsUKYHfZUfJVgZOqZokmMbFTTb9WQ1KcLeeJySzzJ8YoHNYT3UF8V8s/1eDoqG3L2JUe8jLER6AeJjubXs"
    "xT20vbIjq6iGy6qM+XrrSBhMN7cfIsq+dMnSCdcCd0G2RpxKZwiMTG6PCYeKPycg+nNNdiP6dlgVXGyfH8MfydvVwqrMds6Y4Q9n"
    "GX8oxEIEfEvORXHfmPa1mJxIhPunouNVXAoFyXui47WSfe487ZM6t8hwXAZYeOcIJ30r0STcielxieuajc040Si1wg0QlRJmGK+i"
    "O6lssfDWbipKbzqtQBjza+brXifmozRItXCK7dRizSyEasOhFy0/MV6yRAoQiuOsP7j4e8pwhFvQTe02hLiJ1eiWuPgBbty23xnP"
    "eunF3GcxxmxHybmwoEYUR9IBICkL2PUQ2YnQREhOj61apop2KIReAVPvAeo1CSL7ZASWyUg1LERSDT+SxXZuqZizhqQw8uzV0jCv"
    "YNG8pAH9Ftai+aq3IwwLC5/gnj1+fBlGcWzZ1vm401hxlxqvakCLl6weLlM1rC5mdv9Wuq5aQfeG5HIYNm3/CaulquEMj9LFAxwt"
    "+enKPJu6e65f1fILRFo8MMJZuIxg/0dN8cYlMm8PRu1tKyf9XD9Ou+/ivH18E4Tr3IRBI8c/7Oc21oe1yiGe1VsXyxWwihYKfQQN"
    "HqzCAp0TWLlp3TqSFrQa+jsQCSh+1FipLpMYYmZTur42V4MqXFB5gwHC4V7XN6WCVsdHSzGS8WY4SjwS+7iLdBhhQSzg05jUXjHx"
    "Vd6pih0Vc1hggbtEpZnLaic8sqrKPPvDi1vLn15c16qecqfGLu7o7MV1rar307GLe/jo4rpW1SWbxS1u88cWN8ax2I0WpA2HzNIQ"
    "wdHArEQH4MWx5GM/qiCDEW/qMcWr/SqivCapkd8AnX35qDn/HMvwxxBCZJUL7ZNOF0yUfRg40Rwt3zbBu4WpcItRG2cJxtE0BOuz"
    "oYCojvfcm+VZLydNnxffhMBWpTz1TxmY7rFBH1RWu2GujmA3zXhJRehcMd75MySVpcDWEbUCDARuUdbA+k1qH3GSB6vX4ZrZXUef"
    "OaGjb83VgN8sEKIPzBqOFuwPC7K3yviyWh7JfrkRjOutRHgfFDPNertRDoLaU+tEVVlpI41F8hnGsZZT4BenJNJ4ZmksMurRCInX"
    "L0qY+r8wrAL2Mjk6YDaj8fzdfs6AI0+hK2/Pnm1KnAfcW6oqeuh1Gptxt7pVZ4Uh5YtnYq+xyd+Rt0zEth2vB+RclbsOG4oAI7JG"
    "9E8PxbVJqraLNqrIhflHhmRzR2W8cjr/nOUFjBIjfsWGaTTlIUVh/iA1sz57pkZqs6GI2mBxrkgZbf1TMaVj4oiqTxOVBUzPgW+I"
    "hMx8i7t8xtFi8NgGhxMnnFWvMJlfENUOHcXOeLexJ5qqltmnXMw6iGHx07+j2u28epvMBYi/MN0zWEs7mSu3F8WqVhUeTflNg6XU"
    "YyMqsQrN8MaFk6Bzn/EaS8UpekB1+1rtQgifedcKsVC0nfp0a32GCoBom8d+V/JpQVJQ717HHS9FzWoI24Ihq9t+k9cG0k6/I/zJ"
    "/d+WaRspDAIrvvaBXVQYbWYQaladnkhsimPDPNBqMfybwsopECB7MorSOGLDVH85IksiK6KkXOfaAnNB5wNW33NmqYdO4vmQwZzk"
    "+owek2Q51yiEQTuJpzWou6rQmV4i3FdPX4nifSPxXsT00EaQE0wTLkXBWc2XKz5WNKTPYohZD7I6kdSvLK+vUB9UmIRhvWUVx2j0"
    "3BgdXGI0rWbxFZzp2Em7ofe6AWkjztCniI00MkMLhCQ86mFRWtQc2cj75Cb3pQ9zr4IG12dWgnZmskX5MiJwlkNy1WBd+/y5BTrC"
    "Ic96c9S/lgvi8Nad8vj4stY97jkZZTfOC+R7BzDJMvkGM3wiKAfgM8Mkx47XHUh1t09UczgwPJ08I7jophXlHVUmU60ezDAFqNMI"
    "yphCxMoGZV01FsUZkO4UgafAQGzi08c3Mlw5ZtbxyE4YecO8gU1XzoDaRD/S148lsdt+bg5a5ctA59gblHkuRIWkzN06Ca8TA9Nx"
    "tm/babKLlE0nUE/TPVaXm1GB/3ixSsLJ6sOFe2jz0x31i4rEeIJClppmKQZ71vTKiU/EZJZbkEsSJ8XAR3PVgHA7EDHHMvFR52Fc"
    "OngUutx8VTqEy7sP4mKh15dnoyePLkRtB5adhepFNBeGJCDoz8LDMquZ9Q96AQcBc+12IkdmAEeN4B1QQOKgoGcNYdBJ4i0ALpdE"
    "YyWDwuXvegQkxCEDEoIXaDbuNt4UalsMQChDU4dFHsW2E8RVnPAgcxbu7A+ENwL958H7AsolYBfUiEvZTwkLXtT4aDpglm41Z7An"
    "niG6aQLzjjYiHEJ/Q4UcUsnHYUq7yUqElASUyrCJCY/Th7qWEEzjNbv6PXfeB8fJMVfFw4tv/bmcPsswC18S6roVyIUZDhzBi4VH"
    "PTD6nrlpBg0nOpQXW4mBfEumidJlCiRJh9o+NnK6/BhnLG5U1l9VMcVRLkY4xlqhQ9PmrRbt3Jh/Z37QRyAT4fO3OQpDrtzW2Yl2"
    "ZanxMstaqqHFMkR/BHxQa7EYtafGImx3E+WgsegXVdlWFnJ6XMeaVOO4mIcS8loB5eFd2zwfIugrDG4ai6DJgm93rXtTdGfeHz0Y"
    "1yMqxkReWasi0SSoRg/npZUjkPi6MzLD243+fnSBG8V6otiX05rGZiwqYHVGNoxFzM8MaQPEjIwQis8kJyReKk0zo4Pw89giyukY"
    "F7VyMgFajKYROA7i67dVlekUA6WXd806WqLVEhVc4qFgD4TS6ZBEzWEHBKm5dfBrmRoqxQNHlpNHDHAJwCafNzOhNCALvmgGNxp2"
    "cwcMbxt2eETWrYVFmsf6uApj1JMP5sQzFTFIGqyptchnSJzmorNyvaeHI1FYY+QxfQP2f08u05+pKDvZ5BxpMyKr3STDHlNsjpgZ"
    "dUYf0rGo72mRlQ4wRTqz4rtcUN7NiJC0G7zikYvf25siPT6tdqbN6uH2y5hK0A7LD0DXm6iHtSjGhMLRlaZJsvWe2cfnPvWdAb0K"
    "hFeWOaG+wz8NnyM1D3TqZitZfmq2kwiD2mm0k0UsPt3yqO7WkHmCdK7YaMM7Qa5FuArGZSXVngitmLWkt1oVYJbE7JmRiZ6rweb+"
    "IrrW+kQ8u0IzKBfr4vewnBOci3669HzGPEJeASmRLNZh1vVEpoWgJE5MO8WK1OVi+hEH4z0ioDqFRzz7ITlOxpNGMih85aLO1yp9"
    "fg+fOIEy3zJE0V/FqcKxFb4PjD/5gH5818y+V9w8XVcDPDHG7IhaUfFnBOWiuvbtRBTHWVDIDj/X+cOIHjcOpdhsBX5ZL36vVdhy"
    "pKbDmUHb8e+RzVQsuDMge+qWTos7B/uvCb81XKzkE0hjIEgDYb4RITToJQSSkuSudFAPBElBLUGWYvR48N3eAId73Y88IGQ9ZMyq"
    "IJqBxkejPUztPFt5WSjJedVAdxrm7FI1m8ntIgLyyj5b3I1AXWCZM+8IakPwqlqBFSsV0C3qg47VSSY2UfWNaSI94HYIeQs613c3"
    "xRPgseGDWPCwGgd5yhTUqZl1Cgz26PBjD2Go4ZA88+jXKk3p301WslcMLT5MIlUzkZ+MMiER9bC4g2fBDrFRmPMi/rP4PYX+FPdZ"
    "xLMj50Gvc6m7zOl3OLU7Dtki/m1XDeXZjlofeZlo42ByQBKSReC9Ofh7K87YzNHxI9oFihS9FNDsG6YOknMD8eXnM1pBdcJxIbel"
    "m+IlaBAZC/NbdtWnOtlluce2n0S5bK6OKCirDBce5Ifm3O5sMzom3t1VTiK4/7pbhoc1416TeFhR7THZhoKHKvP1tDqn+ARj4xgk"
    "EzuikoYbC8L2zEyIXoB89RYJe1thUuQMu/yrMs8eHBmt4j0Rw3bl1TQjexiNjJSojii9XBqLjGBTk5vw+NAyITsU6ZGBY7JwUiex"
    "Hl/nhvewF/RJCoPJRvZOHBusisH3bF9hcQ7hcJnZOQVmwpavayGP5IAU1Kxa3H4TicssZkiRfrf8jZlQPB7BiRRkgMjuNDmckiZx"
    "dYSUQotkFpG39CY7BOY46pZ5mCNbeMewxP6boRZcONY0zTjnorZXUfp9K/yUiGWGsEyxONSRLDauTVgzNXiuF1EAzbjpXAj7adTk"
    "NGeq8+7gFasIK5OdXgP3TczEABxJsQKlBFpjcaVGIWw2OFx1NzHOtfN7BlJ1XdhyrqbzAAx1fpMLXrcWEoFHEBO+xlRAquuhs5Da"
    "EIdyzzDOjN/v58lVr2tCtWGiBnoE4YbwChsaaEnSSxzt1zAmTRDDCPDBTbjgenTkU+wbSpSU/Fq38LJYAlhE0L3ec/VEUGu0r9pm"
    "nIKSuYlfd66uKvzTsYUYNCIt9esaK/5uOIj2OhVCcYBkM3ahtKg6YC/MQWyUfK1YhTlIXyM6kJ8RaJnCGX7mXpljJ81G36Ug+8jJ"
    "pGh4Ru+By3WSSRbWVttrfbDY425tiJAQg05jjBkSSL+dufZepdWKkmbkQ5vFL7ywhkWO7LovBjkijierF3G7SNdKXpekwJlPbTBX"
    "l93M2sCKJxDorH11ZBzzff6waS/KNGMtfDZaOqr8IgDfXfzQm2grXWwuDP/b7Ry4zRaOz6rfLGfiq5c4UkrFM5ft1YBW071vWpIE"
    "ZS/S/Y2IDnokhqcOo1G7MZ89xjubfHUVEQ7zGIf5379BBScglxziCWOWxBAoWh+zk+MymyRImJ4kY+GXYQaMmf3g2EMdfg0dE2ao"
    "FE0h12m236sSPHBc1MIRO8iF0H7WKBRrbU9sYGRP7ZQT7YSfDtytuPM/LqOKjk6C5mfihzevgMTkrip5VwySEef74ZyP+SKKE27c"
    "bTMAmH/aFQesFlLrfmrxdJWv9QYrEx26pQKMVsFROCq14nd6DmUcZpnruCFLCRZhtVUo3mLcUqNdbNYTs6dgIQqxNHItDJdZFAvN"
    "IGgBA976gjmL/XILeHUjEejhqBFbZ1AsB+2ko6jONUVEG5TQOksKQwOrsRsL6MWN5zWwZ3xhh/fPtfGPwKyJTyOHCllk9CnBmLfV"
    "b8DGiXfU7Yuj+gvHLvVjxbPmdkax41MIQMPr1rsTdDDObLhscNiuxmGMMMf5tcv3gYG5FdofUFJ4UxaYGx2OI5hgi9G0BKd41oVk"
    "7SXnqm885laMpngFJrmQ93Ien3Blr1pFDq2OqyX7MzObNjGYZUaBKxCGn3qd8rGbXmvCT5Q5ui4ck3o11zk6eIvVbuK9DI8cCMnb"
    "6fi2nEQMaVR+3V0i+FS/aPeFyrGrvfT4sJHYr5mbELEYzHqr6lrcBn2hO0deJYEUpeIl1ksfJTPfXLVdLIuCwXXO2wRHlgMF7jqz"
    "FTFZTxbdOONk3s6uHLYSmSmhlNyeCI2JvduRyCuWJhK9y1dDA7Y8ehu0IcSTJJH5Yn+aKNF5GS6WiCByPowcXh7EvwIpyJgGbIT0"
    "X2NmsXYc4rJf7D0bd642zLfNMhNl0qWhSswwxRi43BiBoml1PAYNbpbTwQ0EEL9xpytz498b6yyAvpXcjG+iNdR9AQMcfCAaFjPj"
    "Kl9DOBtwf0H5KWPkND9O+9PvsvwwmeNzbhfMQrjppy4d95h7fJGhrmpPEweaa9efdX4O0KtTw7iBi9gJsbaAFX3Lfjdkco6rImm4"
    "LiY4ZXKr2E68WHFKEDbyNM8I4DJ18LPusD4Emxyp+HCXIByTkewmupbVSMu7/26nkkmEfEwZi9Tdgzt+mAlrQIQkVGhoTInh2ksd"
    "RMZcq8gi6xG/EIO16+2903ootA9mRNT31qvem1bAaNIzc7uYmoaL2J6RfDFx2kX/BrHNxgZ9mZGzzHQ+6IxfzeT2LcyO8rTOWPT3"
    "sW0M+fi9dsUe4EImh8uQPPyYegYymDPtkGj+DWIfBglTNvPYvDEmRaBKcKlV105Ezh6L7sCkDfRd51p5zR4egx/u2AttVp7UbzVb"
    "TA1D7qfPik9n63tX0XDeRfGjXfzwORDROOoObh0VSRzRdad92S5gOZTHo/FmHDt6tpFYwEDsRkk9eQZtKwpJG+upUbldq8ZapLja"
    "7XgG/aggBB76bMNZ+4vamRaDbg2UoSChaxk868kerrd4num1p2JnhstsiYh9oKg4NIu/W5tNcS4H4VM2kcgsmY3NhqWPM+2x6mTe"
    "ugCbj9ZSjFsVTStRGOM+ayFyKwx90mL358wQGVN9rOLE3HCkpvjww82AaxuM33QUYXENq/pUkAjvG4VM144LPr+ypB2PcMZQasO7"
    "wC6mNS63Fo2HICmCgTZjV5SWKtHrqC6n7qsRUxzXLrndnVyRMOELfxqnikcRLeWB0XNWitSKNqSqWn4H3BoR5micI1VjkR0nVQny"
    "NAyiuYeispG3KKJdU0DW55KFNCoeBL7YWouxiMxnHTOj3CoPSyNkgFLxu1koyxqCTgG7HnLihamXEZRsFk6gitgqGB+uluCOD3ef"
    "5PQY9rLxNtbw6Dx52CznkxI3LALL4wpjKlXhAlJYhDgiTZXQameyqgjaxOYUukSJD2cYcTbFUTFPxF3CX/ottqqsBA8891TaVU5O"
    "0XOHJwn5uZ4BYdUxfmZp3VShYn2qQk1ckkVc4WKpc/2DF0rz3ncigNHZNYJ93ylT/DDtqEbCiiZs+rfTrXY8wuEq2PW7ZVqRc4bl"
    "Mftqn47ZNzPkocfgTnVMswhv5m2AaCKYf8K5chKFdfc0Rs/tTSbt6MIHhztHmh+pn6plohr6NtvErQij0qohwGJi1YqNu9pBJsky"
    "3d6TMHIWfeW1NMHm7Nf+u42fhzYyx0f7dNYP5BrrExFkSEZnLZzFTXmdLxEcwiNraq5aiM4iiJSZXn4bs0zHLTosnInUAuo+dQWX"
    "6yrZ90G15OyCSDuqgoBRVWRBbMx6qe06uvh67Jn+93hceWdd++JxEMciuOJCC+yuoevCRaIsRao4yqan0osctVJ5KTcR6OmFSmP3"
    "n7o+lwR5baQOv5VMVM9hoYV1NRody+4wk+VBVnNwsRtTzJSlwbbxmYqxUugOKN7MVxSTDZERTwruJUx50FhkWGMyHC0J+YNGH4eU"
    "zGdqhEfEKUedVLiYFCNo274qog75ylV4PhLcUzEWdbGxq9cB/ZBqn6p17ry/WgIkJ5o8BQHlt/zUh8SpsiVuyqGWSFmJQwQ6r8B0"
    "jEiQXQv24upDq9smVXt7Mb2swVF/y4fkaM6YQliJDcVbC22mR2TGwxIcTFT8PXZIMhfAqgyIthHrt7yrnkucwMw/ERk2HT1pGaaF"
    "0ZWYH6ydqM1cbBirZuboI14Ydl0o9xlwVMNg3I3Jafcf/3ScEBRP7U5LLnqUHReQ9Ewkh71028eY8HPEwSai80Y0lm6zfVVoJWVR"
    "S5Qimklfnp51SZ6dZdUbQRuDdxtPzULY9lVNOWGy8x03Ty2X47nn5EwabpTgPr2P32NMPsznaV1UNA2e8+mzsK5Mr3ylHk8Dekrx"
    "4THh8njZR86KpEyObvB4BU5tkmVFKQ+hDsukw4FIZxOhgohiWGG/mFkNU8WtDjyG3USwy56TrPCZI1jTV0cTkzBR1o+p+u3DqlNp"
    "Yaz0+pjCZdSsx89lDXRq6oM7nkl+jXZQZfkVZb96y3CLoconT7t2gZRZ1t8fFixXAfkRY8WJwjn2JZOVqNB8x+NJY+NR3Qu+uY/H"
    "pVTFL/X9K2b0kiRaaRH3GZIOiJij5CpA7Kh6o1jGeZdSVgdmdzOGQXJEkUaEbg9A7q4IFLe619lcsYcpP33iiDHiN0UuduYx+u7Q"
    "ZjA2EVPK8/V2WeZpAsnvtPxJrLaPynJOCWi1k+AVegCg5Nd+UhJfDE+v5RS8wvs2DV8jepAL3sOILXQaoDSvjiPKUhZRQ7IyfoPZ"
    "n11lVp5wV7HZnFPKsCOjozfCusOOjURmKvm1U5mxdWR9BXZ/t7ZudNabircul38vqVAaevWUF8HwddquVY+UwiOwLHHAn07gyyNA"
    "axEOpbRCTFOC3WPaxzWD2xNhjVLWEncvKDR5klT7sKdEKvvuPrRN4ECecV5otK+uG5h1XiwX24nZQ6tQ35wuajltLDII8NfWwAR1"
    "40YuWIwp7Qg+R4CALWd6ittuXg/6t91Eo4zTbC1qIpgTs5bMsCX03jR0VEYJNMVrY8ryq9Ym7NGiNL4toeiRQjPsqKn7yCQrkRXe"
    "kcnlRVFZVTJZdN85rimDmP0eDKvHdcpK+U1Sikoysxelg9jxxIp2DekFHqbLMh9f8uvsfHRbnxue3HTjwKqFqoAt210uCO3kZiwq"
    "x8Kdnv1WWol32vMnc4grIMhJZLg6Ns5tCeOB93fPU+MUj5KZFJz0DQo8vGDt6xiOXT+Ki6HNnLtXGBNNjJZA8YqYkvi+JXDYlDTP"
    "SqdxDGYGEU3CsOoprGG7pO9tjNNJBJI1vdZCYvlAPirm0aB8e5SpSoXaQyus4Ul9aj2fgOfS3r1bAUvKT/eCZEUzm6gWJi0Uuj9h"
    "n2ER58Ob9obLYKa6p1aCo72dMawbrLEdV2DF7vqDKMu5I77bY2nAhpELA0RjXGw4BLJtw58yaS5q/X2v+MuKqNweLeldl+19IN72"
    "Koj7fL0+fhhXnv0XKTujbCnU9V2r+u3jXfvk86uw1y15GbERpD2zoPQaccVzdho+tEjp3jAUCe2oLYJcexpbQohhlOUoTgjhkZAL"
    "tRPAJhAdCDhQvV0X0MQ4k1hQfhNW0Q9gEr1ce5ZLT5BLxAqq+cvLmkkMPUUbJO8mJYhQ/MgztBtHozWlgKTdX0utL867QMt7ndoL"
    "LOjrAAOyb0I0SH3rd+rR2d8EW6B0RyQj5BdhGcx2fWTKE5QRlZQuyJCA0ui9yU4rByf0jqbW1fXYlVlP5PulggNRrgOcas8UMCdU"
    "C02HLkC5qYVM/goQJCoO7BH4OSi9y6ICEItLmAfWdNc0hr3xVhvMAbXKHAmJ1frEuFtjbvDbOilCgvIBN1vUWrWn0Qr+DUeNslG7"
    "fTTZGO0EgxX3WvBZUR0yX844hkPWINhvkLH7MxklTSX/MDY8us8aVg5u4Lo6v9xzaBfpBcC8SuC1cL9TPY4JzyKoiH97MCrN9g9z"
    "x9C8LMR5wp2QtTr/vi2dKNIgGCY7VnBsQQguc4w7TP3PbiPhT1ZTYB9e4hnpvhwOMJATjyhxrK0HcDKiX8ujwgB70WDxV7/DLhpG"
    "VQwOlBibPJE8m3EXkndqpnO54OrO4THrwyWJr2hwetY4DBfJEShh/soX3pmREjOfvmMkdWS1MG972faQIyfmVcxC20ECHJvUEJ78"
    "Kr/zqIWvlevCXpWf8M3ypXRd2Jzu2rPgEUMGB3e+eVeaJpKSpFHrIRbgU2MD0XRxiWUJWkv2WzWZQGgWUo77wXZX5bGGMZFXWnSO"
    "+oSmPyl9LHmy0F08nPX2bxYnjvDjU2dBIynwibCbzCI1awML8Hp3PLPF9xIjn/iwUsMVWLVKLAxHkZAXzUDNrMKaTsW4kdS/WqdA"
    "gi1+3ZVhjaQnYzkJh6uMLziMPnhqX5915EyRw4PgqxJtvAvDV+SX0nVpf8JizIaXd5futjdBlnfE8FJmSfC96iv996MXTZ5wdd8U"
    "Fo4wp2vxw/qq6H4R10l3VPrWb4naa3mZQPhNo6xWib/fL46uF1RXE7MVCcNUGS4dceenCo6LYcTdbczXhXZHEi3VSvnsa7KAhzJU"
    "2nU8TliB3680AzVBlyN84mhV3vTteEQPnKZViEGYaZAdyLIxeiaatoiS5WglZtybEzX5IkqBzPEZv3nBiLTjdteMLTK/8ZRXdwJK"
    "ihPuMRwDE0QIjtXQh+ilrQIByGkYlAJ80jdcg2kmXBkQewW+zuo31PD0itMtUdERswylVJK95aLiajrALWhKgn2wmX2ni8Hji1zd"
    "PxDGUUFFuZ8hqVDOCGKbFTN13FQyYLsykePBXe1u7Co1Rl0mspOSeWA/nmNRhyLBP1j44eQcvuFx+gQwI/bKmTo6g//mMbbxuOI7"
    "/tTvH7y7ljstyq/VBRRQm5qNlKLgUxn4D+5j54oCc2UBLEUsxb5vSzcFPV3o/U7Pag2cR3bb6yqaobQPA+M/bhgcvkPbFHED9tPe"
    "XE+wLWyrh4WU1brexFjdAOnoFqsaZJNVv6jgeEcTISOrVh4K8MCWyTBtE61/pjSk95ghGYvs2etTTTvHwreuB33Vd5TIjnZdSJ52"
    "uxhkntfae5KL1YxfwGor+63LTXm+NvfzxHuVB4uagbsMWXPuKJbDwlQxIDMlgjG1KMzoMIPtt7tWceyKwBPwEYZR2kaP6G1EOfUI"
    "N+MV6LSNGU7yiCHscCTflokrjue5LJAtVvSymFELzo5PJz19qQbe6GhKB2eAFIUDRVXe5g6c9n8oStqqGMsXG2vt7UfL75jS+xpN"
    "CVejFsM9+4hq5wBvjGCarosTsv1yYNQtVPdtLaCbQlgjQ/NhlErEXU2PYniTokvbkcxR3/STH0ETESsIakMVLU0CSUQTpCV4IEPW"
    "vKRU72qr5GzK0TepTc2vCpyNru2uVGUIOGjZjczUhlWjOpnhbJxkSXU4dFkO8Nb8vXtEyMzGQZVZD2H/gMybWKLCEz/pFNNkghWe"
    "+VXuCsErfG3xLIww9IVDd+jP7lIIhiHZh5GUjq+yAB8U456qxAYiI0dZ7MDsI9FnKEhxiSfaH57061I06Ct31euOeUAXhR87P0tR"
    "1J3yseJ4BhKoCJ2gvpCnA/PdoiEaz4p4T1wu4/VS/Ou+2VgYG30qPRWZyg73GTdYWgtp9Mjmn3tYldug3xTGNOszrEkGQGTseW7M"
    "OBQKykyapZnxT9AZouLECy7RjhzsQ2KMouQ5ShWX/VjYPNF17LAFIAaoif2lGGL1uvTNl8Wm4g7MBSt+f496b/jisiJasWi7Ip10"
    "qMw4VBY9P9P2d7Q5y6Kg0FW59bD8hkSUqkTb/ZlRmQdV2+WJf14GYeqvmTAtBELDV4MKDxMLYZhXYS5IUomSIjLcdiKsdRMNA0ML"
    "zggcT4uk0J6JWrcZYy/RceT6ZJ5Ddxj7exVCP8YNYSWeA4bD0iluB92NOOF0f72vtImBAivZD6LxCa99aGdQshSrmcxRdoU7jsRH"
    "/SxEdK7zRkxDzc/O/ixWyqA6brhKTYa87AKeOv/z36Grd+epPqMPmQT98ZmgpZ/NZCtn4rhQuFHP8M6+Yg1HGq98M9tPQQ5jsC3Q"
    "fSR4a5lBgYhqYEsmaRxPkxNSVBdnrmLWVE6bCCWFKnXaxS5WgQ4Kbd1Dpx8x7+rYTJKVlFKfFN8ShkoGOpRw3b/IxePoL5hiOFly"
    "KC1fWwJzZ/zZpoJKDj/3aP1CSDHXLtH+/KHiviumec5ndwygZiEuSyUauHvu52X5CTpK/8BMzpwCepFEhAbOhF+WihvJS2GVnjuc"
    "O1glvBUcls1mHxoU6oIVKQ226pIkmSBLJrliYnAN7AD/tGpvDgNW7g8tiiA8r3EFWLvZTvTZTedAzp+6VAAZYG1/Ln8VouYwvq2u"
    "3ejKxQ0KS41luO23r97GhT5msC0aVKaitO7PyzUaFq+f3V8i/E/9W+l2t5M2Un10s0zJgssb35avnMEg3dqxm8rsOZb822hVf22n"
    "guMoFb4NF/A7y9N3VpuUmqbs4rWfLs9GN7OnYZrKg8C+FTli7jvqWVu6x+ly4m4xPvS6uXXLate6ATWhs39tJcvtbiK4r7fL1UaQ"
    "ybUKXhDvEY42mXkb3TaOd8uQis/icIB2J2E/EwPyVTYIzr7qWQERSsGGFr6iltknpOVaOEIY89Tl2sc0RWmR29LaPj4VLzofdctX"
    "KfM6Oib1jfBBNynYWww5bICeVc10DuXrdjEoTtBoyRKk6vL5vDyOwpebymu/O3vGoNBSIbjspYK9ALrBHM3JTdVvqCzPBljZpFhO"
    "DldUjaTIFFZPTma+fNPrhKgaTMc336cYmdXP5x5ai6sec7vspyqGtPRaKqCi1Fj3m6oqGbsA6V63iqPe415Rd+hMysNosULZEvTs"
    "PFX9nWGZIV8Obmm5C9GrBzclRL4t9xulFaB0w9X0ta/8XL1mbjkA8R3ZyOSWzWCSsu74beK1FPXugSLUB8qWoIW2uNPr6DZIMHCb"
    "3Wa4yiWxRt9w2X+7g2nIIlntb6IpquX95XeWkm/9xvOr2b2dXxG646BztYczcazkw4xFhN/HnZCCOUUbgl47lClAWFCzfGkL+8Y8"
    "6036e620Mgov7AZ0eTBlRJ9NDcgI4gnPQPEpJhGGr48zwxnecjdNt76Ww2s68FJYQbpI90c/8WymgiuNYPqOFoZFHMn5fMip4SnL"
    "MB8l0dbI59OJuYVd9wYzUAuYysC75j6yDTR/5xZH8e6TymJbDjmEiqBKdwy2VH7G3ktkOaWVjPdGSjXniVfYj3zPZZqFCylpeDN3"
    "1++G4g7PW4mr+04+R+kn0idCWYzJDRHE5Ug7Rhh3BqsDK3Q/V+FsRFSNGKYcMNEMrsJrr5PUg7RLC7hM0EVpTnHD81aAM67PmTmW"
    "Lx4cHR4MNm81S5HihXrXp1bAwhnGhfgHV6BGQ8nb4JLMlS4NGsxfwUp4sosAtJxfNqDbK/bcipqnrMX+TUgOZ2A3lH4ADLLQbEcq"
    "ojAHB38H80xUabANqyRYkp47GILILDVeGWHscIC5HMBY3YXKE6zEOhw9Upax3YxH42SYhyjYiiJZaMd21jtVjNPgx5KiTZ1lXCUB"
    "1qmY+LuDczk+Z5zWrri/pIuLY1cb2hes27yiY3ZiKko2c3c7pFOOKLzFXT9IvNKfdfdM7b3X/20LwigBy08HV30xImVPCSed4CyB"
    "GG5EaAm+KCbWSbq50foWEk/dp9rjfYx+9u8LxsVJpy3pu7QglbgczprDnfwtOpN/VkiW7MMtK/9vCsmy3odbVv7fFJKlA8ItK7uE"
    "ZMeRchTZiQybUv8fukpwlmBEPhmbaiVv4DMHkK+HpUWtHrSDSzgmSQZ5qp6XwsSJ2pooDMOnQfZrgBwnFZ+cgL08lm5Afo6RTtor"
    "zOMId/02q7lWuenNG/CMnA23jTeMgsZUE1odDJE69gyppQ9sadjM/SY/HT3Nd/PSD0ub54iZMiD/Y9ImSShLjAEHfVrY0OF90d4n"
    "megMWWJeTee+BfKJdTTT67VIqIuV3+wpIU+X9T5w8aL16slzryQUcptUW+dJqaUlEVCMXUqZIkZwCYz3WaWSAY8uJtALjy5RYCEE"
    "JMNGXrdnJKz5JrGVM2Mudn5Rve+pEr06f2WcirN5WpHkdi4WNo4pGsdVCclMViOSwoQnrDQqRQXenYO5wf4nLBOP5RVQYch621OQ"
    "L0r7cEXS7BEfx2Mdkkk1dptoEC99PtJOBnMGlxiohUct5mLcoDbaAg7XIEtQg0utsyUI0nym9Jkl5t724fRPtFWwuZ/Fr/83LEU+"
    "7icVn3/WUuQKGhNsY3S4ApogPT49kMFGcBLHneQTLArldRGRvC65mai4RHBPx/mr8hADDJYBwmu1BJP0YnCkG4fKbS7hNNn9GBP8"
    "KPeT/PrvM8GPcj+VdX6SCfp1L8bdXKc5jvtZIawGSTilZVBbGMIydOAcyfITMGyijqlBZzzjlRDofgtCqmTv1x6WgjnHGA3d+WXv"
    "q9k4jCc1yveBpCD62f89o/SrQvqybWQ2t0qMVk6tUjvh8rLJC9YSdtNDNBlD22vtyq9gNvtBkV+iTviNpYthWTAiI5KiNRh1erDa"
    "VTGuZZi/emiHjadG+ypXX7wXu4liteXLyTWJ3gDrpC6LOJwnhD/tphtrBluLxDR78KyASMSgoTU7VxjiFiKsgyS83N7qy2zCJPX5"
    "zLH3i603tUAMm9HzjKE6OEiUrl8j0uYyXGCgz0OwndeTuQBIRdgD9o771ZbPS/Nu01kXQg9HZwbpfK7IyA5LS+Hm3+f7qQeWRzLJ"
    "JoIAYi1scWI9ubhxJAhXbtNP8WTLG3YZ7+a5soojTQxu2lO2uHvCEGfiG5xY6nK8wXsLI8ayna/MGFlM9Jr7aclXnWxeful3gj23"
    "fyPIt2Axi1Jeg0y8urxbohWgAHcP7mgTPbWzBM0C7qDj8v1tA7WKsvuQ5vBRlUHK8kllNVboT0du+V8LezU3VM+EeeaJxwaO8zPx"
    "erTwijTjjZ9OaSJOGynKWpn7VjidGmR/nuO2FmW8RgfDqBtgDPlCLupxjR6EyBCemqbVUNJwZQOlpBqvdXjhtAJK+fpAIU7vH9S5"
    "4i6DUFZ+UEsx+hTcTtlIdSIoZ+00uzl471BzOjw0c+LvscOQEqkazSmFxaOpSMuixEO6ra+jQ3cUycqxU33NY4ThjjN4+r0xcjKx"
    "tmLbOEDn3O2t2qn8tcPodvGNeeqvFhz9QbANDOb5BUhNcgT73UkxLA1eZghdZ4TJ0G0599zQ4VTyxdY6vWwVxB7mXTW4OEbOGbAe"
    "LMsF28/4ZasrNAmyXc+wAjBFQN8wY4Xw+ghTfOeQeO+ydnEXqAN7ndQlXHRW9jkqAR/y1ER/mlYK021w2L8jOkg/heAyO2v2V8yU"
    "3o0IUM89FqpuAU/VgDAHh56s9VbfEog3/Cchjr/jpXjqpYo7OKEhsABX1VgelaHl7EVBD/Tz4q7rYwYMcOeAd18lONEZxRbuWoVo"
    "Xhe3jYt9NnwfAdXSK7Ui7hNXhCT3cWNCZXBajCCMJQkYiR4+I1kSKBhLmCHWgDU5nvGOx4gB4eQ2FxISXD73mzncZOYOFSSQuydp"
    "4YKxjusxN+affmqWTtXgEsdtzgqg6RelMWTQxfvp3fzy1cwNgWPGyMtTj5RieewUOD+nPOGOTC53eczh8KGe76eUt9NKOuDz6KSD"
    "jj4GskEIMfLki8pXnQSr9SQdi5uxr1roHWYxpHMgzzuwCaVPq8SwxfXjWtSmUT9VZh2tPLWzEPuQGCYSIl/gG4iKb6DcrGuH/bR6"
    "LGy8wDYOmPpbw1e1dXwiaqzKXx3GHYfZvridK2Q/lG5YFADzb6rgkmho+m++SyKshfdzdeR8ho/Sim1ECbiWuNO9+T4WxDtfpmNC"
    "eZfwGlE3pF7JmM8s6RQn+uk6Dks/ZlvfNFpNGaRtLXD0ldvE1g9ra2xWtC66vZmcIgph+ATDM/eOsxDGwQhupwrE86OcTFmQErrb"
    "G1aB+bPzmA5Y0DmZ4WIV1MkWdEu3LHUUkXXhpDOk3blWhzcCj8mqRmKkbNHKOGSIX2lQE7YsK0JfRP5Oa504gwwRcILLp8m6fBqn"
    "MofhDeFIp0TJXrng4RZDFRP3z+PTZCig5BrXNN4NLLtI9+n758L7R2ZiVT54IscSy1I9KGbMUtbEgsvMNX3WWgIz3dNygmercjDB"
    "iCbCMpBTqG0yf/YIkbfz2ePdc0zWuedY2Dl8Kviao/ox0EgGIlnamhtW37o28I4jSEnto2Ba+TRURQanOI0Kt0b8PxMj7+emUGy0"
    "4axGtFGIu9GmDlkstms9u8WrJiymd63LKazcXuV9DPnx2ZqifziOd7EoM203VcQjmjoBGszy+2zIRHwd9S4MUY8CR1pY4GkY2g67"
    "6K2CFY91AHIxZlWhPQUbFGD7WIqFyyTiT87HnfETdzCaWMOYUuIuYKjQ0oFmw0WCs1JO9pPMjQabkiY0ZRSKLPxah/FKH1l0SGWM"
    "jFZU7YzZRfpAmJBbVUIIAeDS47fR8n3xozNEV80wBTIe1d86WUcgLS3DkX3DCEu0kT1ZIHIxcFzxQwMiexwr2DaZk+sApolf+GBP"
    "ENiIFQ7nQaAxRhEHfBlQNueSkONjFyKrS+cydSxnUUyeNq60D9MBQVF2ur95GqVUQoNwFMaTwzyN8shovR5HqpTYVwpfyhcQKI6h"
    "/lgquYg5w3xd1q5wtl6mK0wyqYpLmXmhe2XXzfx7FySUjdSfUm51PyqhmHqWq3ygoe6dVWp1V40ySnGqd920Hl3nLvWnK7uCkGoL"
    "KZRdLTxNKDQgk/1zSq9K8zd037+r9LqHbSq/GvqTSwc+Q/nVF90TNkGLz4toYZI8u9uuKDtljuXhEurVGFk74qzwBodwz5CMir9U"
    "pUDId4ELZ51ceb+vnge3I03LqL1h8H4/HcCQxSl2VxPGgLHhqoaqvrhcFdtOlkdQR8wwx+y1xUao46X85etd+MOVoXWP0JWCVsP4"
    "YF2Z9eybT7HdOd8T2gysFPJsI3vCtJH+5g5tET4P4RfRF36cMTXQcFg1qprx8Mh0lLSIqt/NmWzih7rVsFDyifeo34OO1ZtNIx5Q"
    "U/HDd5xl29Qis5RayPxfQPkwVAq0lPq0lqfSgcl+t7rFAP1xXkRnZgxDp4JCrZgwqFy8E5qhlkomuymiWR6TKXu8+jv9aSHs+zVO"
    "lUqWqLDC8W+k2D4j9ixdHqBS2fVd5/2tDzzaBnLupdEoXbtC9W5CXYpFbwyZ5gn9FYJ+I8i06olitaFNUYtBGrJiGjxoC5PUCXQK"
    "fq87TLViNYykdlgFh6fX3VYL0j73lYPd1jxu/bfRCgUquqhqOojHkc/uuy7jFUtuJWV3mUlw7/vwHsmKAWKSSxLIIDqUb5jzCLRO"
    "2GcT3gHNP1hp1KigEYekbZlfPLLYyYt37SiGmt+SXxQl1T5C5BZqOVFqpB0060HmobHotxpBwyw1IEC4QUXDJHY4LrVGUM51k0G9"
    "m8gAgakVg6JZENM85ZbakJ89a8kXluMAT/fzBEcnNgJG3UsvMDUYF/wFTvU3wq6TM1kglSI6zi6i2xp1js6VYpeJiN9NZjXo1itn"
    "61laXSauibjMAA4MjY+pb5ajopOqXdIdR1D39GYcZ6b1IX25ldgD5tkKOwteHu3fCofUh7xKthVRjEfmX58sLn6GRUiZgoxjtXN/"
    "OhCYV4i25qni+yGrEH/PtbcSkdmR90HV3RnQ+ga0ygQQ0j2QlzkrNUB0uN1eFLF+R9AuYGmS4CkozLA+CMjO74UYIwadk0GquEUc"
    "jrsVUxWlCjDT982PsCsMFxJvNMcC8bMf1MVNKNR+UejUJScI7AlQUHkpJYYpgni7YVANPNKY0TmG5VPtozK5qc7/om8c/zTsH+4Z"
    "EsRqK8lquzSCXOtUnS42091seJOcDRmYXDjO+y8Y37AXGPpOCkiXp9HvLfupLb2qY4QQ9adNuoKQ2pZdDQ2/my4ocd8j3nNPT1zB"
    "+cgwZXw4jrYPSg/JWx/qQnv3/SPvCmpmdnGI74JIgy/kbezzDNn6u7xctA+nwtc8+JWRM2DSdjczlT7NG0JZzLCyT05QOHIaCA1i"
    "sgxA7qqua+eZbva6SZBq7WoFVsyRO9GwzzE+2lZH011zRf4QEQliR1Seb4jkM2FAnz/ifrEQ8mNnFGI4HIqHGj6OKMPtJjFW+0Ul"
    "rmzrGbZNkXfpJ5gLJzSqzRUVwsD8zjoOd87jsZdHDujlMxohR4dcVUAf11MZtBy8DW/ew9GzCWaDyjLw76fuYerda8OfMZ5hlur9"
    "PHcL2ugrzZpJMlvrCCZAot0MV/VvpQIPYwXJpZMynRnS4A10IyoWMvS9EBe9uADJkpIo+KURksa7GF7FFea0ZMNlpaWKMBxm8Gyw"
    "v69dhZOEDKVmwMPeGvthCveXM0Xt35ZFkJXVJiycDVy0zNHWRPhxDPUAP32WdHKZek8Saacxx700QttuuT28ecr/VUvCZ8IhioNw"
    "FHVnldI+bHdYOL7NvWG0dDfdO3v/dGgHxDmU2glsmogVB1491Gs2CVdYkJB+C7mfzezuwXKtmb4vc2bKXMPUgl4aVzB4HTNc6o2F"
    "M8zZAy2yBnep7SvC35mfMHNy0UJIdfdWSEbGnUbohcdm1kQXbiFFxDm72hvHxLoFUThs9gk7duUo7Dbmvd53UZJcanEo/KLcLbao"
    "ZbJPmpUnl/0wg56hnc3VrFKR5u0oZlKSaZ6s9G18gs9+xcjM1R0qvAg8JFBbtRJj/KICV7SKtWiz7i/7T+g0GpK9RPqu7ZGbJzd/"
    "RgFL0W+TYFaljKeshVaXZrKzNNHhjIt89mSk4N26Cx6KgF7DN44r0U1J1R73U0IvFbbfpPUohlDi/sMs7kxSU8PwZu1myCwa9RsP"
    "k7XvtQUK6rhkQp3nkTaK9Fiwa1wdcBbD3IO8Jvi1xgAL5K1r97cEe1ggMzsoqe/bPpYE4zPsJvoP7UWApcOuW4krUGRAaykEzW7i"
    "PVdfBNU2Bd5rNTWB2ZbInhZgQPex3yxn5KeN0GPmQoU/5Sbk104MLJ04UgpSUVHDijt4d4e3As+MLGqZxP27RkxgmPV2CEenEvlU"
    "MgqfpV9CUAmH3azKboKhU0pDAFNgdrW34S1VrnodKfaBZdxK+e/T0pJ1PTqUpneH7HwEQ5BYRgyIPRyl668wxNdJvuSalWhDIbEN"
    "vuCG9DJnCGBauDKfSabCRz9gJT13Iz2xSvJkGxHM7cUVNbpc7vhIvFlj00uZNXgYhH3xWbsob71VDYiqqNHlvBVPWAp2cEAykn13"
    "Kb16pSoZ3BHr16JjwtvKrGMOZgHn5fIVZ1q5ycxhCAdRzSwOZMzcJ9/rTphr6UTsgOzF/tzzCA8M/EmSoPVMoKF1rRxsTVgVWHHx"
    "YqMQXgeF8KEVVn2GKcN7iysQqZGbPaPCFdAFUSc3ljzE6sw7uJRidhl7Zo4pbWPxzWRezt8py+uLppRBQjGf1ioj8OOHtuyyOA8+"
    "pC9PEcRov9KTa3TPiN00/ljJO6sVJDeKJIqLOs9RDWS58EVWLZh/GushN48nix+dNTwFs0jEmQ0RM2c88JiO7jH6SnKe5zOAflOi"
    "XODVrJeur+GMALMti7p8GxsFUsYq6F3EkoZzebJ5ASlXIEjouSXKxWaTf3clOtfiRwuRy+Jqg07thZtvK4NOkoDqSqgmKJh6RDse"
    "pTA7tao1uUoOu7kryqcWz1dZzI1rthO1p3ayEdQTW/ycUPefRqvaczeVOd6htrEMD1ilrE+zLevvzBELT5afKM0Rca2bLLeAFHQa"
    "7T2bgVasspsuH3uY8nm4Ag2kv0FphlLN8tN5O6xdd5P9YouS7kpYXmZfmgM348NsFcKboDDlw2Ryn/Dkqlm/Au9ODUBTYKmA9XUp"
    "pOFUm+1+uym7HulDpWetdnjPUZ4S5/QnaTh/FWWo50J0xMVaXR/xuUNtJ7O0uFjTqdYsbSr1zROtUKIhqn7DJ1rtfFavaXty3xqF"
    "AD4BQ6LhgrrwbGzadRDUSs2YTZNgRI69S9QCoMcl2XWrF+kaZvWx88D6hGkqQyXtXzapL3SzQGxg+sGFLjULmWpDnomSsXlBuAEi"
    "yo6bVPfO7rrTWSRzHdl14UBdTzdPQaFxDwyzGBQbVeJi7fd23LCli03/RKPQCjPo0W23yKEUVgN1ssUs2o0C8OJE0A4KmFE+K3zk"
    "MuIwBedSoy2pxWq7P5NcTz/6Ga3/Ab9FgmnWWll+hMwZi/0+8+TbK0GuN3lMr6fGN/inxadACCk22j/4KbhoDfibuhFZ/UYUmkGN"
    "0v1xtXCVhBcg9kI0yu1EMd8qtIk8wCZQFF5pXkJ6fj24Ke5GhfKMlx2JUxcr+WUNI6WRJzQV4u7zuBMYOtf9SRVwg5Cof1WWWzMo"
    "W3I3OZynOsLad3N5+CzoXOw9QcPhdafFnxYbLbrEPE3Ycu0Za+/WQkbpRlJqNH5TvJC3eL2OJOPH0aKHslvQGAfJzOUknm+DKl9+"
    "6abXRhizu8oJdDkTixWJeJY6FS+CaeQQJDOY+D4b3zYO3eR3vhr7KQsUXHxzpn4TaBDGk00f8AiFmTnui1smY5Ad9W4DzkU4H3cb"
    "Mp6Qf8LpBcBb0cN3eI0n5cmlTxFoSScZxslgeEQQ1KAy37x45DEimHQs89ONExTc1jTpRCqZbFlAQ8Zfg8PCWFQSFyXQtyU2mkOZ"
    "aYlW/FiqiA7p3VtIr00PHvWbboEWVkXrwp+EZZOR00WyplJc162+y/cZM1oJnfrq0CNQEnMxmZKr69tsFblwbZgA8QYJOZzC1ICC"
    "3RQTBGVN3aDBSSvziGyFXaDNGbCJVPMHP9s/OBinMtkNbsM9hrRpnCgchyx0TX+t3+0jFukGaDNbWGw39/hAIjW+TDQRiepGexlV"
    "59nnyC7OS3saFIkq2iyR7ifR90VFk6i0GPf6aReKaSHAA4adTArTSMUJD4FLdbEcI2W7oLtNJL1ueLltfQF/YaVheobVeBiq2qgM"
    "rF9toHmE90g3hHkWL8F4SZ9klj7ERphn1yLNt7Sko4JMkBTZbpzRIp91zlab1r1Wx6fCNUtZBfRQ4rPFOgCZBCw8VhdbgiC1Q2cz"
    "LVrXiiXjZjr7/uLFNS/p1RBTSLWQGMS40nQrMyg/XUz2O8G3vuXT7t9iNAarhcvr4CovfbSvVM2paa61kmCWhOKKrsJP9b7dF/G0"
    "M6uSjIyloE4qy34piizZx3O8skqDJXjBI5FLid3PxuIikeKrxwI/J6b35kWynRXm0WPHlS4hHjuJjePi1SxrYXCrKpJpmqjrIiYw"
    "WMg9hRpNoXLgNZ+MCA1EERjWcAHrDsp0U2Qlvlz31Fmg1k35mADVOEyAzcQZr4qZzSiJoPwENDUfpjLbyg0zdnTTs1UvZOU9B90q"
    "gtpAu3dY3HFYWjmHH4Fh00BLfLYuND5iedZhXHFpFDUQTKxTn1rBgsygpU2D/1t9GvfocmqrAKT0XPuMWmzhYoyYTxTkq10yLdlG"
    "FhyHvXTx4tpzPRYpIrpCs4VDIXLV75OfLm7398/lMRa+ITKRZP+uOCJz9FlNlsW5/in+LkZTY1Ktog8U17SITXh3w2R9O2em960F"
    "sJQtMNLNjCXVLq4wgtL5nmYtLGbWVIOp6FBeOediM9xHzkPnOX6/B+gXn9Oe74nYhgkqMyf3OnPkP5/o5vI8QotTAdYRt2FmSQLW"
    "9ZwqGaViDcvRGdGn0GEFG/BKzmQgOTgEFLRcyrOIDy/UT5zcZEoop/zEiksyhE+hSx3p+lYMoXMsjkUxHcyS0N7joVG9hEq+8M8O"
    "BJ8xmxn++eYS8ex4BBzqqdshkUOYb5oJvmZcmO9VfEfUqkcRYdjtxfFqK/zNV9uFd1ujmBN7n7jlIW5YUs/iGS4qT0DJ8kdVn2vf"
    "t1DykUejFV8JTbFMlDsf+kfg5yG/qLqt7FSFdwxBHqMNO7/F/EsjQ2moIGyFWjdj4mTAyj+mgzAmWvo4vg3Q0SgVY2vBZQEl2+Qe"
    "sBqK447tWuFDxUDuomEn92Ot8D5VKpns2hUlbcG0zEbL3dOExRguNHLkDId0vjuPGipvGphXN4Ru4l5fk9pP1TLey42wPr8/KKeD"
    "cDh4nJFyk1Tq96m9SkT3nxc8xe6WGeYHcUxBq3cvbC/fHHvtQBEQM0TfFpnXS3N20rVP/oL3mtnYcBUaJc5ytJVQZ4E5I5ICrbH1"
    "fMK9ZixyowhdFFsLh3PRP+xCI6hdN9pXBTF0EY2jZvBdzYCZ4k+FpytrU3ZeTwatJnps59GsVslqLJ+mdmrvzFPr6OJMxnmDbKW0"
    "8SR1CDjrIdUvxuLSPZ/LnDFF9Oiw0QMb0Zgj/B1lO7ZazCJcic8V2CtCehR+TDoG+ZI3nYRy9PJbzO/7rYRElUmdhncoegaCre4F"
    "xk2ShkpuqtOOkG+Pxys0ODJejPd5xGLCpROJvEfRPJCIX9txueK75h6baNdkzPCJE7RJIUuu5H9XaYN7+KSRBOWUyaMLmXwbs9K9"
    "7NP57+fI4NRWWhWiYsG54oMlMlC3Wry418TnpGaUeXrssjr2+350caUoQDm8mnbSs/CRrKzXg24a1NS9/HQT5zxg+81pvRHONF37"
    "HM+vo2ImVWsVKr3U+2yUroJo0UtW20Y55gaQA0wPtCKk3J+S1v4Pm2ufemwlkxjRJfYaMw5PxQ6pcEUQ93JP4+f2u24l6qYKClww"
    "kDP2YjQI13kzHN4/B+P8gi2GexWIdHRqrWKsV+fnnz7/9NP8CXZxOlntHl/n42Q68fPFar27mK8uHreH7dflevwaTra/08cfX+H5"
    "I/x08Z+Lx91hM9l+rdLzFvz9k9nLZ/ONr4+PT/Nw8viIr/K/RlpsBqMFdEKNfmbznbxPRp9G6+UG3vj0OPx2+RX+G09G6zH8M3tT"
    "qLUe26Vr7CL37fLzV/7k59fd06/ff/78xZza183hZ/gJ+4Rn1P+FPYbxfLR7fOSj11bgD3N6f+I8xGs//fT0sl5e6A0u5svN+mV3"
    "cf2yXk2K4Xw62+XXq93LOgwnLxeD7QUftvrxp58eq/fXhbvm43WpAb0/DHazT2KpPn99mWzX4dvk0+evm8ELfOfi3xc/w6cn4fbn"
    "n366LhSz7bvWY+e+cXf92H6A11ebr4OXl8Hh0x+Jr4kvF/Q/ya+JP79cjHHn/gPPn8L1YJdOfYbXH1q3j9VS7bEKbya+Xolfsl36"
    "JQVv/4T/aBSyDRheswU/fhM/9rKdx0a2VYDf0l+Tl8mfmqXq4zU2ScIgrxI/5bPVQiP7WLwPHq8LN/B7Bl/lv94Xi81Ci384mRY/"
    "tx/MJ4mrnwq9Aluhxxq0wSMyOUwex5PdZLRbv3xdr1bABbBRM5+9K7D+6N+tRjZfgR6vYZCPrdtGoXl7f3fNZnqltYAH8L27+2br"
    "sdkqPDRxQomflvAPfPG6hCOh+cG/S3loCO90SrXr+84j/LuC7WG2+tNGod4uNQrXj7elFnV3ZTzGFX9o3OeyudJdqdWjEX03W+AO"
    "3LdK92pr7MfZ4Ia2JFvLF6jRN1jc4j1MpvV4sm0Sd8Ju1SgEhUYT1vGhUGDLlPhuTatYqhWobbfH+vmaunI1ycNqFqLDc7RUbbKN"
    "G3kS3b1m7x5u4TxlG6xNwt+Gvk+tYOnz99VqqdWC7bjL1q5LtZvHe5jobSF7zeeQcDSBzrJ35llMO5q17tv5WzgJNb3przSBaOPr"
    "QjOPlOsWJ807vXQ1xAOSzcEoHx/usi3Y1KpaF34siqWW+xwl9Ba5drFYaMhj+i1hv35daBXy+EuT7rv9uFWqFpoP2dpjkzr/zfh8"
    "odG4byAJKN2LS2W9zlqUqjDIoFDFqVfF1lkt6cw9Vh/Yd5Ip4zmcN/15+msiOg+4bnn2C9xp/AsbcKRhtQCzyd834aywQUeHAjQA"
    "jyUsfV4fdCKtt1QfvMki3TXmnr+vFUuwaWLdzTcL+WxPPoKlCEp3d8DSHm/u4cQFpWYpd2fTq++yURNIcf72MYCPw9JeZ6sPsAPX"
    "pXaTRglLY7c0V6fVaBMVhz1wNYR7AQODYdyV4EiKC2Q3hP1wNPxmt8s32qUm3shWqdW+ZlTgEiiZIFJxrRI/dbKNwu19u2mMjzfu"
    "y8nePxRqcT2l4Ht6mzsg2rLBTTvbUDTA2ywPc8w99sUFzMNVg20qdB7uSzU5jchXrWbaJhHxtR8XsvnbwjV/O/IYWFtNHpnv7se3"
    "2bsiZ8rX/LZe/Avlgc2ctb/P5+/aTeIphep9Q53B5FXCbpFDmgrj4QIBNPrta6RR/g7mbhPvZLQdEJn7duOxUyjd3PKz8j0yJHHJ"
    "2XkH+v9QasD54ssLlL5hLuJXuwmxMNUghctotuBP6YJz6mu1yNZu4O5RA7aKtHwgziS/6zeLvWMO+erK9VzcE3GVqtmHx+5NzhJo"
    "loPN4ygcbLfzp/nk5evzdr36mZreZXMgGUKTTz+P5juSYtebyQr/XK5fV7vBnP7+Ng9DkELxr3uQEmfr1y3942kN4uMOZHLsSzBL"
    "0eV/O9v+j972vnbXY9xVvrWd7D6pkRk9E9W7LuB5sKhXQm/GuFbhuqT4Dj3mvz2iFKLO5pXxDJcUn+ABSLFeQZ6+Sj822rBv2tTE"
    "KomVYdN6aHGGB6RTayyXkrW6hWP62MpWCsDSnc0i/dLXffPSHkZYNSxZE2UB5Fbf5L+a0A2dPiaa8CYFkntSV/AbsJN2rZVFKSG4"
    "L10/NrPFAtwmQ4T6dmW3gl5BGECxVLu4XDKzmuI6393fV7JSSiJdwG7GxLzo15Nf0+6mviFcRofAXogOJGkOBJhJB6m4eeIymSul"
    "HfnXKH1lNfKNjz5qNnWMjLbGbOVfoaSzpW8Aqcj3feuThrWXTX2r89NP48nTBShRT4PXcPd4+7j89Pni1/9Ccrcak/7INP/xDl5g"
    "eh39+52RxOPkZb399OkbqJafXYolNt2wpqCrYTt3m6fHpa6ySs38D6Wj47+SoMfS/4937G9/fjEboI5rtHE0MNu4G6g23ue8Sdxz"
    "bKI91/5qrwJ7wtZi9sG1SJxeisTplUh8fKB/iYH6thakDrguSWr8ojdOextfUeP5qZ75QuGhhOX6ys7kS4q9NFqvRoPdZAX/ffoD"
    "vvsFP/619eeXi8H7fPufBHv7ZbJ7fVlF2r9/udh8wfP4Bfv/gpOE91NfcEyih+Tnn34iLn1xA1yA28Yqv1/Q6Lg4wRsgj68ONg8v"
    "E7QrrV9+/4ldJ7x0j/PVfPf4+Gk7CZ++XJAt53Ez2M1+J9PP59/lJmCDrzC+YThBy1txEG4n5sPV65LJDfh189FwsJ08bkfA1OFZ"
    "bb2y3ty9TCYwkKc1PP3jz+izrfn7/Imsg2qwXyewIrvtJ224anXlT/v5bqa/hFz5088vwEAnq9F6PF9N/yNMdmgbmw1W43Bi9khv"
    "w2BQJvoKCz3+xFp9lq3CyeBlNXmBNtT2j5/5Dz+r4W8GL4MlTok/km0exeDg+c9/+hd3vtp9Yp388bP8/ec/P8esOd7kLbvLcuxb"
    "2YlqCr34Tjr+33C93u5odnLs05fBeI4mR/7s5z//YCbBn/+M2WScwtvnCxDzLt7QzMtf/uNn2ernP08fBHwdf470AKthbpzq4etg"
    "A/s+/mQ8xv/778gvZOoOJ0+7n3/XFxC7+YN+fxzN5uH4BYQ73aIJc4M1++Lu7gWNsK7+6MEPdLjdhHCB5+N3V6f84QruPa7Jx/oE"
    "kjT2d4pP57v5emX2K46Mp2fB432rajy3xvvd1+l+gmu3dfVHJ1s8P2+c/2P88tlHAO1biVdipSsjF0AZ6Dd19j6LZuaF+Kzo8YbR"
    "6MfNy3o44ET5aTIAOjbZ/q4JRJZ8dPF/iar+bpNIfcwu2mjS4neTUIjvOlYNHQGzwWby6dekWqDl4GUKnEisjSIq0CN28EnvYLTe"
    "HD59jtzkLxe0nHie8VIf5xtt+b5YhMQi9yugOgbrYfT4CbmhurMaWaJ1wKMhG7DbabaQF0y2UlfO1RIvhtWUbpLZVj/osrV5+o32"
    "/AzLpvJMm62AfExozn/gavx58f/85+LX5O+RM863ls8KibGcE3vxc+SVt0H4isv7/of28p940PTO/s/F+9ft/Di5mIB4QAd0sIp0"
    "hYcTbvQWnn2ibj//7rzYfENxeNqU4G19nfiv9D3aPt8EJiGOFc8f/ygMlv1TbRJ/9wPDcX1nOznZgXek/BL9Ia/Bnxe//EfsPnsh"
    "cuF+pWu7HLx/4r+oLpGObLkY+76JPh9PVmsUc9lCQKvt6/ITvfT5s05MWLv/A6ri5Ndk6jQp4b+xri7+zd7/7CAEXPdDm9N4stnN"
    "HlPjT/SXGGrHPg9/NekVveaTXmAO0Ap6mC+RCKeJOuMvRMf++DX5J/6sXRXWPfzvH1+/fkXVJNIN3K2U3R4+K0gj/PBF+8LvqT8j"
    "Mn8I5I2aCcdodDnE5aIju/UsTQjS7x80XS72kG88uqZsAFNB54db3Inx/Onp01hqFbzNwd8mwdtsJi8jEPyA4vATpn7AtqgcXsGs"
    "YGYp+PMK/vwN/szAnxn8M8PXg80MBTvFDeg0jr8u56tPnzUWLX+fDNwPtrux+wW4G9HftQn8kfgz9nEy/nEq/nE6/vFl/OOr+Mff"
    "4h//Fv/4e/TxpzFe9MTXZOKzb6VFk9RVXJP/IvPOGU0ymZNfSiSS3jbTd/7k4hc4uN6eoN3Ff5F+/1m1/wQn3fzxM5CrFFxF9j67"
    "9kBW9nBCx+wu00+4fI/4hqaTsN/m0Z8G79pPKOscEijewJ0JofVmMJp8gmsxowvD6Bcwic9//A5kSVGYQ5IzD3j5l4vZxb//fXFp"
    "SlDvrl738b2S4Cd6fsee91bPYhq4AH8cEr8f4GK/J35/T/4ZacIWRKhYbN3xd7G0nx1vzN0vzH3tYS1d7emG6xQFGN4OG6lx+Z/O"
    "4x7CBw3KzdpI40o213oZjBaTGLPKINzMBv9B9/qXi+Fkh39NkuHvP8yaaZtZqL3kyvQvW7mHXmQD/If1nEylnPrtrGeb9dZphXkj"
    "04Y0qKbjDAH0As5xPghB3FMWIbkEwAYnO5q/Pbv/f77/uhmjLY3btbQxLE2pYRmn3qQ/OzUq7cMOMwOb39JWcX5onp65tl5eJ3Gm"
    "LhjCI6qSQhfDIf2ivv4vcUiUvIYjvvhVvujaMdnpL/op/dfFi2sb5V+BxKoT+2/x4c/Ge28rXQQF4jUIp19X65flJ9GNKY1C+/9C"
    "h/vvvsXVZvoJ2sF331bSXvqYDyqDcDlY/e67r/Jifrn4C+5qIvXl4gX+TCfsoxx3zf6Sj/6ynrzIJy/2E7o0n/83b9G3S+ubGC/y"
    "BzdzpxytibM8ImN5GaymeCf+/OFrOBlsQZhdgvChX0j1q3U11QPHuP6RS6q+8MPX9dul46W/s6gfvPOam0ScfO1iF40ozD+SX7nD"
    "6eIPGYz5Z9yM6ub7/GT/Cz/2L+bdonBO1ht7Gt8hTpxbd8TULWEED9HF/6t9VWzYH6gOf5G7Qv/889R2PPDu2LbQO/Cv4tcWUKa6"
    "0RKd5A/kkvpTkLgXo0GFGvxODf590YyM+h1eqyDJ0Y4V++CvF++oYUQG9gBPYODr193k5VPlC/v8739+dp5X1tV/qCvnMVUNko4G"
    "D+LxA1cygZZzBZFph6gwntS6qaVD78aXY3RvamNp3+w3n/4tPkR/Gjo4GXK0Dk09/GUw304uApTRCi8v65dPTz9P3jeT0W7Cv3gB"
    "I2W+IPr0xafbL53PF3Aq8S9f0IU8BRLy39ro/udnTpHld9mg4Rj8n4sUvqr/mqRfYwekxjHfXuzW64vtchCGv7s/atkLuJXDthgg"
    "t+Ta/vJMAwrvWI+7/oUvEZxhPfj6V72R+OJosHx8m4we9+uXkLtT4KfJywB/RS+j9sEv+mOgAPvBy9j7nGxj3qevG/ORc2o0JgxU"
    "krug9w3TY+zXGK9xQX9Rn/O0Trpa86l5XhEWIKcdTNsRtpnw0ieah77gw8F2vjUWezd4mU4cy/UvY9WAfJCzxrusrJvHOYh4a9AE"
    "f0d/Whiz8NzjwNd/97oJJ39oXV+4/871TmtQJjmxHp7jfYB7ab3F7hCSBmDYn8+jDvaoOHmA9zlJcH5D3lO2gs65sEdnzoQ1/hsT"
    "4AOJDF/vWCdpnhMgPyknxv/yq71SPwmzBp39/2jnl73Bpva6eZyC3oxiVyRTRI3ilWgiHi1m0ld9yQaCLnLPDRJGEMy2n/gAvsgv"
    "ee+ZdARY2gZ1+fkzt3N/V0vwBOR5CEo+Di+S1ZLkqS1/xily/tFqfbsN5PrrajnYWPnKGk/kJ6jJF7Exnw0aIz/PG71uhFQweNlO"
    "gJC9wQjfgKNMRUjJ/70ogZSCrkRuatZuPv3w5UI3QQsnz3Y0CAcvojPtJOMv71ITEs+Nxwf3Y9PDIu3HNBRuE9JjC/6/9t7+q5Eb"
    "aRT+3X+Fr+953uOeMQyGYZL1Bc4SYDKcwDAPMMnu5eX0aXADfsbYjtvmY3Ln/u1XVfoqSaXuNpBsdp/NSUK7WyqVSlKpVKqPALRA"
    "DS5hZcXEkx+qFpnsT/P90c8oM9yMp4Ov49EsG3aad/l0NhCf9YKTDXxruZ166Ji+aU0SObq0xUSCiykkzkaz+z38BmnJvn/U78vZ"
    "wmAkwA/6iKlsSAgYbYpBolEzc8J+0xcRg5FY5uOhFvKc/aMp5NZUyBkDOJ/B8z0+88LG+F7dNnrqSQXYiFQKqthTu4G5gtw8xsM6"
    "kLoK0n0c0iy/VctZdHNwO78V7Lg/v8zhNC9hdTTeqDSN2W4FtQFwRyNKq3ZLrsDEXv8llbvBTS5FICTs/aA/00SGMoPRdYp+eHO5"
    "oWuKy5WIxcT/7CoM64j5s1Y6dZgqt/Ni1rzIQWG/pqZ3WMrsKrBIQF8MWlxkj3JVhjVA2S6/gQJX9lt1WWtz5XQQsESBtY68PR3P"
    "R31VGpRZ0Botfs8VR6C0tCaQbOA/hCQvTiBEp+Q1LH8uNe1OLZviK967Fe9lRTpxgDYIs9OUPRFsQ7+71zQwot+on4q1pEQ/Oj+k"
    "vGvnSMPwGcK87dtH522l8Kamkb/alrpm48OmYzvfo1cPa8jKeqz5ig/p9XQgdqdH/KsWqZCV4Fdb8KhHI3SN0ptseEV2ElFSvMVb"
    "y/x6dZr1yVprG8Ik0tZzXc0aA+exPpxHD44c3OxRyk/qTrSYia09Zj8reykA2G649kiPQYFHtwAoD0Z5kQ4HX/K2JlqJkROxqkWG"
    "tNTtxI8kpkMgWhRmKK24ZPraMdCaX/J8Ik7lxSboqxIDoj+YAgRSR6x8abUAvLNtmumg6JVUnZFo2xq+ezAtoifT4h95NFUosOcE"
    "9a1Ej6JKWMUHCCH6pdE/iA9rCx0ZNFLqzPCx01wzQgwFbhh8eMBW5VA91hM4vHJodoaHxWbvnJyZSY1ub5XUmE8qiq/21khxNWhu"
    "nej8icxl7JDepflZrExaZNedyVs5ceXELLKrPL2cPbStJs3XuT3l0B5+rpLcXQ3BfAqu/LCCBIW95tQxX900VGsM9AVAtebAARmR"
    "a8AX5y9qHYFZpKuCZBWYCaWtORY5irHE3SI7dl9EoP75J2EYRMeZ3B3bUbWSAJCnrgko1/Ff6+HeXEgZ0uGhm3HZ5F8H1eaTTfOk"
    "HSykmY8Y1rwPYop6AtGGSquhBB9K4ozkR4GT8fJPHB0GgcTbaoHWdhuAX3IAtNAUQuiw46+l1/n0SkgrBLi7c+GbVx7uUkUNU/o8"
    "yns0YLz4YJhQ2LDZWBVq0KcB3KjpDZV2mmiTgIF7FZBv6WrsJhk2EMwSNf83veXAqlk3nTVSNt/UqnGmHaiEfM6kVUM9Yt+YX+Wi"
    "VJ8U2vSQZXQGtprpr7ch+03XNYIeXHHAY9q8yu05oECg1Yu2RvUOMYUXU5lTfy1Ead4Glq9ulVYcJor/4Jd8JNaFWDbG2DCbXl6O"
    "CW/V9yPshP8r174+ieAtSrkQPL4AOX6oV25a/Kqke7H528vwYPV6C7L56lVz1coY/oQiXm2T6fi/JJIFEtlZ9H+tbmn51DlqWteW"
    "lgur1Ysu+JY7tFiSX/Mtus5tOW/Zt/zVbkt6DEC6oDisUpR1X5CS4diI0uFLUoNQVxQlv5wy3qzDkt47Wf6b1pwOLr9Y8U489FBo"
    "cYStPkRiSsFlal6ANs166628kyVASJw9ptJS2yuy1ml43AvcsDJAiBRcE5O6Ie3DH9LhePwlu8mzPi3xPZRgTyfB/iK6ceZPmnN+"
    "M8Oi3ridx5cPFmcG75xbAVCWDtt5lDWoov7oncsFkV9dAYg7PQbWfMcZGLj/d8dBnwGG+V0mGLI54rv9XZ5md/mwjSzUqLXoGAjA"
    "Pgp2vwOuNnpse41QLW9OVJ9Xw2w2Go/AKiWoQu3//0sIXSCXE/LBGUrBsicscbb6VRUNR+XMFLeejhkcKoaGFyL31UDMYQxsyjQO"
    "r8yjur+GQomFeDkeDgeFQFBQvfgCMpepiia6SfP/a7ZJsxvNdjCer0L6JkwD/YHooZgeajO5mg+HUodiMURd8RV3Fm+4jiwwZi7q"
    "3g4/yUf5bJrZq87i1+ksdEAkB8pa/RKktcQ4czE4Rwq7diRRsbSEPD7Y5iZFM+iCpl5QbYkSIfS5W9Esy7rdWfMwsfxCzIzevs1g"
    "bV2ky4Gc/V+8oroCQ6u8zZdJzpuRtUzmlWHDAXpIJx5yJwI2Ok6uVBVrFOZy2+XY5GjGNxmd5VJloYEVIedhsNjS9GL3qcTosjzY"
    "0m9siyrSL/Jilo6n/Vy51gzzh2LsL51wFi6FSJ15jZ13mLnr7RkVdezwJy7CxKnOBXBm+yPG4pwb1FiP26HswXUziaLiNK0GF98p"
    "GcwV6unkOdOQzh0bFcb8JwozfoyU0oUjaJ7HTpOkrPLTZA+VpJiVNc/N2dJaS8GxgzFVauaTwopKqNOr5X6mRNh6Rh7GL2wxC4+s"
    "ubbb1A3B8U/C+M2AMwpa6GHMqFqUtqsQCwpxBXodxwFEQLiehtJokSpQGYnZtwRMQGHkXiZDK6AlFRXiKtHp5DGdjdPpGC7pwAJO"
    "GRJMBrPLG/PrMbtXz7zMKmhRTHXYjQIhJbh/F2J7wF+y3ESUm9hy2IgtKH/KkmIWFI+2pEDAloMf6nQ1nuF+lj44RiENPzwKmadg"
    "3wn4LhVT720xhS/q5XmUIZs2HyNtQi8B3MSD78WJOVsqZMHLSf02v8bafIQePXotwIvLx7D/JCJNSZtqFpG2/0o7T348lKjcgUtd"
    "52MlqkghP65WV4eAySNzj1PXxK3C9M1gfZvNpgOYOGQNmPalFtX86jq/tBmhYpgRhYrfEDWo7ppgvBGzpcBiqx7cyiC/iTVmctD2"
    "raKokqbUmqvanMuC4jRbi6GySE/LTblig+ZhrIeBL+0aelFEA9wToz/2LS8dDZox6XYWSpld22tmGr5qemGVSWkzp0whEmS5/ObO"
    "NW8UmPud4XGBWHEloyH5THDP4lluBhSmg8LQW7KfYphPsVw76zQvOs0ZCheb6EOvdvzMFSIyTWu53UkxzS1y4RWR272oOLoIbxQy"
    "uW/RVxd268/EEVogBHfIovKGJwNIylwoNN8I2JnCB54vzNqbKQ/dnq6S2S9bGDawZ4GpHXtcFHStAXkS+ylV0ouHOX5TF3zjGdE6"
    "gI2GYJ8IJdDqorhsdMZCTF5t20Y6AMmyEVl2w2MSd6L2BSOsy+P/ZCBOt3xFGlSF47zu8RvaV+72sjMPgmOB4uMvYTiJhxIzVBfq"
    "F4/OD/7nNwGdvyQlinyAJ8fvDRksosQRsoeVnMA8C2lj5Sf7yiEweBJeiiVscP0iTvFgXAOei1/AI0ERBt/Dh3YXTJHddXwnL+Zp"
    "b+6s499hNhjZWPeM+5/j2Galllf28TalMdKOJrDYs+EZBEs7d660qb/e5XQwQe4B7NvG13cUSRRweM/lNw2aRwv1TbOFPv0TtTEP"
    "RioQvWHt2V1ZZfF5cI27ulcPHYcwkMREB4+DLnjx5NoOtDCoLFHmSICibFlXZiwWplJ2lw2GYM2g3d1ceCYYXKQ+6zUJX/v53eAy"
    "j34+5L9A+H/ZeCG2NbmnlhYbjCbzWToS+wsE28Xb41a08Hg+I6UZuBDITZZt0/mRcMUMWHfA3JQGfs3sIp1J/3YYeePs3i7zXPXa"
    "tdNLNe9ORw/Spc1Qsdk8AFXE6Nou2vY1RDvcxJiH7YR4ygoxf3aSQxCfI9BybOM2rPxOR/l9mqmtuhDiay5YyGw2mM37uXq/iSx0"
    "UKTF7Xg8u0GzH7lZr+VL60kQWMuU5FxLp/mdAguM0LTNeZhqI2JbSCsLIHoKISd2URwT4HpBlvSUzH1PkMAyZ721SgdJewei6qx5"
    "O0jUObufsJ3pI9MWhGqPYP8HAQMM2Av1IzGxoZRDLdFZza+ucNhxkM66q73u2nfnhiQYRobsGukX8f9NQxZZG1wJnTK3XabMKimT"
    "KTgAb0nWIPWLCW3DDhQZG+kIjzAgL4NVhc3wrQpwb/W2g0maqdcr6wRMOsofQKzRjvWA2GsJRux2iAril4RVTDQhfIG6QWilo1qj"
    "BFE17hC4LE9W7STP+7oIO+gSgOuPT2p5QiS3KGBSys6XFxOlvGKhNBKBrbCEeAMGtcrrjwgCOkaYgdOIVnh77ix48bvhidOWL/iV"
    "fe7glBdnfggD5jEtyxwG07RM4efCEiTygnvp6szgBVUrhySsISRM3YR/unAKc7wP+J74qhj5+KLIp3eZx/6QW7jsjxRcFhtDu4Vl"
    "WurKkVPSFr/O8/xr3nav93CURN3pLJWtsJJZUEpxMD2mdqCcc7Ujw0uWB7oExfzM06p1cienQkdfxUBaM/Xfmqf1CkiCw40hsj8D"
    "7p0B8p15+r4CHK6Z6ObhtJj4dbOhnOpmM1h6e64O81TGECe6mzSb5pk1vPQxX7KUEGzfPHbjyAvxMZ1m9+laPx6WxG+YdiB0kSfT"
    "8UyJfOeV4W1SGTuDit8o0kgBhpiyEkldJQVIr4bKYVC+f7i+QDCDos1nHvAB3Yjq6Sz7ko+vrqBmDBKTncBdQG2NGAylkHcl5VpS"
    "IsBP1+NsmCo+AMWCs64tKk5xs/TLaHw/krUgiIdYkCCQ4bpmVqUDXg1bMQOZRPDTrn/+rqiy1Xy7shIexvl+BEGyPcmY6Ysn4buR"
    "Zx+EGK7RsU4E8QoG8+w6DIvKlBIzbQW00OklTKKgBh/WElEX2Jj5wXVryeN7XEBQdplp6EoP0uXDarJcQFdt0MAjcDc5zK+zy0dF"
    "SEf/SZeRO1sb/nSEGQfm7m1+6JNaFcQZQp6s+mF5TUw8/aSTYTa7Aq2v2TnUNhSpRxZpGOba5FYJr8ClnYIY/c14hhS+FsYrvhgM"
    "BSvfrEjCUn2LDhtCejfI79Ga6b/NUP0ms+p840gCK4SZvz7Dw4UgztSzXHTvywADEbeciq2GEw+XJXUtqG7NolUqpLMQislgmg1b"
    "tI9CVBOFpkraa6VMTS3DwcU585nZAupyzmquuRjHlAPsU/gyG6WzweWXesWz0eUN8jMe25QnLfOWF5qIJKpme64ngVug3SjnuO6i"
    "9q6PqiZDp8GYfbFcncU9ur1oqxg4vJOaTxLqcFxUCbFwZjyVkRNFV4ZK0xSACfiAyeeU+LuXW/VJDPFfYoswycJegiL/1Lu5TZv2"
    "J97Qx6OrwfQ272M/htkF2Nem4l8z/ZkSRBcCmVrQPPdferTbkZOCwxowldy3MOq8YI8MDdm2uBHSvGw6vhYrv6AmpS5aNTknhjsP"
    "rjcDJgrOywxTFK8Zfife8stefGBnSHB+ie0IvMDlY9ur2ANBSbfZrEzIueR4HOhZcUY10iiaMYSph0Fl4tO6GIRjUA+BqpSqddtn"
    "B7UeCuWZWOsiEBMgtjAFXx1MXsvcoHHZeHx1VeTSvWcdwu9FWnwjk/7ValCCpDOZXTbe5bLg+V8FG769zVCErEp4W4OEzsmexRZS"
    "cDgNc7oVfoCdav6tYzaZDB9RQh9cIjMc3+XT6aDvancLycrvBsVAXia76pohuFqUllDvjaJDi9IOx2ULCcTvSgsSTaxTSLz9okDZ"
    "DbqkkIeUq/Xnp9rGZlPmX98+xnzowF7J1joYNdvEYADSj6qtruXp3yztEEvEuKMzHIldysEw+gFIRT+a/ZSQSKJ3JQRw81nHycRj"
    "mjsnYzTk8PUnM1cG7jWd7PVudtVwTrPjE+k+d2lsFJWRQ0PYYjjXUaDhe9MsTwGeNIKsOqxg9EwsNilUvTbyh8Esnd3APdF42E9q"
    "KChfoM2RON7RRmNKZtt18Llw2/XHz/vMzBGG+wQRoMuY0ALzqYpNlazMWuDISi1ZxwuBkpuAcxlk5iHh+9P5MC/wdvEub7PXA+qO"
    "RdXRQjus0DbLJzrM6u14Coik6sJa+kOxjVINiPps2pP9CLYhuGXh2hB742AGpNYN3K2mKhdcmz/llE1ae7Lx0BZ7Rv+xoih0Qamy"
    "yH1NpBLY1rhNGIckgrc3mrFlQC7hYiiVJ4DgpmXZYmG/VQJxb15rJoYIeVvAIWLsKSwYnWxY1E5KNGmW12HSbV3IphDoTkaWJ/e+"
    "kEVnPUFJg7zdgvTa6/I0BrWZ9WtO8Or+seWO9GN2Hyr0iHEvIzHiXXr42jeVDSGnkPgKzsBMo0uOS4a0XJ0MkuZ/NNuryyvNV/r3"
    "knwIWrKytlW9kvdck28kKOIM4naACvDBcQENWOj3tfXgBEBu3WGUzJAE18FmtYpZdZPNZnJA6VGHGMvdiBVItPrEUEKIjJzxRMAc"
    "yiwxKq0vXCpk99hNa9jj1Q6OMJomKVgeevECyLlHjLMBHoprptktTMwe7vfIRRVibotbmwFdtbW4U24DivE3pHRedJdXGAHO4rcZ"
    "JlapxG+lHn4rT8OPE+soOp7ByNamM5U5xLwaG06FxVEcXBF8oIjPmDHX4EXRjjAYOAa3pSfBm+a7JGJgcDWYahfe9BKvm1Yjd/ra"
    "/IAe08Ixd1Gui97q2/r4rVXhp1W07nq7DGyjrLeVGYaOy9A6lKWWmtaQA4uvKu55eFzmYrOTdstTyVcEDt85hS6mgimSr+vMPZgy"
    "HdSes30VF5d87LyD4BZvYOU1+JNepTY5HBT/4lHdaAY6WMPKnavPLdZgBAuIeXF5g4GUjI7YU3oe723vfNjbTQ+bzBUzglfbxQq3"
    "OqO66K0YBvySjbTNF37ChW0wSDp6aLQBfkqtddfYGq6Q8CISTkwY8OYjnayvQ6w7mB8xRgvNSTR0V05SljwOz4mVNGznVXNVfFxZ"
    "Xk1KBs/rEjFHMua8tSuDQOvxF2j9OV1ee1qX15/ZZQHE8qjfpf+LyLO6DCPIaoG5xlJgIMUXZoSXbPBijhGGs4c0IG9A7ghsychj"
    "oI29vIa0UT1wcURfZh7Ec1YvzESNxYpvy8NatmhrnjJqWcUBv9GxrXuW1q+pIuCphsF8Y67lsXa/cTQLOnZFlR1klT2k06C8YoUD"
    "Rb70rpwqSrAzxCF7iQsSExU4b2I7SPmsiTXsxFojoQzYWU02XohgUzGfTHExpVaSCrBXUyNlwtZX2oeuH5sqjijRhMie+pfBO9sf"
    "09P9nZ9OMB1AEodbzmqtC3x96peWXGpymH7YPnif/n37l/R4e7e0+uum1HhUAZHZFyz5n9j9WpoTSaIqjclC4p99fu2ZjrPiYBTS"
    "1mYzOjEqOP3TJNS4hWNgBB/rjXs42MQEOlHryqS8FzFjy259+WQwapdvvx1UcPGDrY6AUcmgq1LusKeiiuNDfepyMxmMSYf5VV0W"
    "+XsMDm8F26jdvJsKZTGfg1J6ha15tUFRtrKyUO8qLHifN2mfuWCfOT+U7nXBE/BiIm7j95dvF5dt6x14uQPOShJWppF3wgN2o7Yr"
    "C41CVCaymXLaodMiEES28OIE1RPNggY4NVtVNjvS20oztWjDcij72S2YeV0Ox0WeesBMlbic40LtlDAefibzNZL6/K50S4j0eTi4"
    "HcxsZ9EPgnTWrZUstiu6k9rYAn7aP94+wHSuJ5/29naZmU71rpfj0aVYSyO4cj3zSNw8K9G8Vs4afmK6q8xf4Ss9RnXz/ENvXZ7w"
    "DMb8BCI/gb4LG5fUuyV4ys3Dgk6QUU9LujFyGnuCcy2t/feBltZcQa//Hgr9JtHoBwyTs1QgZhERL0ti1FDbxiCorExE4cKudBG6"
    "LVYqLqgtIVSNOn2TeHS11d2mkoyDaH5xqgmjt7PJB1fFaa1djo1UpSy9FWP2posbIKvgdk0Z6ATYsg33nqtwpzwrgkh8d3+uYt90"
    "g+3976jnjqn2n6zWf6pKu2Y3119Wnb/wbvbm7Yv22lGpb0Z06p7RMC70WrIscJ1Xm831iFb8zy3z1jIgiJo2lDTtfhLsDDNJdjte"
    "nbOu+dCtL5YHA7WBHgSO4Q4aTH8y1n9me7w09pD6urlXcfGJAgIU1iEZyjX1bC016f2b7ti2ZiSDwBcphC0kSGmgwzWAH+Wi/q4C"
    "ZbSfE3VjtrjUiA7O+EO5CNuNEtVAfSvNJ1hiPssi86mWmfFDFD8v36pp2fbmJfgdcAT13hNrxRhzdcRd7a0QL2oj0JmwdqXTnxuG"
    "2iSvS96q1WSq6AkVm6Rl1UKH8BD5sgVZNq/7eXGZ4+mBPxEYy3dV3nhGlV9y7BwdHu6fnu7tpgfbH3f3P/6YHv28d/xhb3uXxB0u"
    "19MvvgQdp8IQg/f7H8VhO4h+XP+Wo8roZ7N89NQQqVtP3G1Ku0F2QZf65+U3X/5OtOh9VvkFYnxel0+gqokdZXQvMXurR851xHq2"
    "9qXS2ILefT1VP5NFlTNBiDOie/bNINHbN3L45QWGtJ69bsw+2PBYfdYPG5gI9pZLI7YuGjynvq2uslUr1XgMx+AnqzQX/kW1VwLv"
    "p9kbRNeTisDcsnhWSGKwSUqHJV8TU2cH4SJSyTiF/jyutwkw2qAaW1ItpVB1VQjLcTuZFeW3SgvH1iq/J3vZeyfeWrai+85crGNL"
    "XV2X08BpMco9PmgJueclFvPYqXbDi7H4MIyCz8Zp2LeFNWwyWoKHU49bjzVKLSBL9MouN0OPP6DCE7SF4dhOcyFS5wxyJrwP20zc"
    "ry2wNTaQuR2P2EXVQDYy4i8miVT4dvHduoFzS0yrWYpZCWn0sdTzJYG3cfTrYJtdjO/ySsm5lgO//49J7lbWL9iSXzNyMXhGb/8g"
    "5PP008H26fuj48MnSceBw2PVuG0wuOzunezsfTxNP4CPe8VRARaiR9042rUseErky9IdUhVzpGHp0Oxbx8VaEOIHSrAlE+nZJ7DT"
    "o887H3aPfvlYfQbSagG/X085dtU4+D3nyFXG5P78J63nuJL844+lVWfs35l8/gVcBSFi+0r8vPusG774Ld9aI37IeOI80PVvBRNM"
    "wW1uPhujVPjsaTPN7/JpkacRDTCdNo7afu38+dqFKFIvgYwK51QllGpBuqbMiWGq5e4dmlizwbbJfW5F7yObQ//hMdaCO/+ovj3e"
    "itS8R5WrbAfRyxVJBciUXLzUGGOJZtX2trL8Pdgt1wH1Gi7XROGQGCX8Jn4PD922kvpgNKvmPHWu6gn0y/l0Ctuuz36ecf3vI1x+"
    "8x7USJJnHMPEBBl8HY9maE1lUYr1UqIWHRr2YqliBrDCfLLITvC79oFdpTzGAUjfMqb7u5lkXYzn1nghNrXevJMMl7HxrnEJDq/E"
    "Wl3v0CaTGqBCEzuulwz5qs1bosN81q0QbxZc4GA5Yxznqi7p3PUMt6SrVTfdfi11/b+68PV/2H7cHiDkJGX7Srx361b5tICI49/l"
    "96QEBukq9OahxZs+vyH8QRa73KRe4Uy5o4dsVkQpvziOSXa1rp45NhwDuJDgIa/8GzUP8KwBQHhRMhyO75Uxc8WRPkrgBYlbR0Va"
    "AWHBG9AIhfi+P1k7wRFsoc0RhMN30kwrxOyfhIe9WavVoAyHttSNFOY2RHFU6VTv0C80DKv/qGGIscrk9yLq6tOISuro/mqV8yLE"
    "/pe/Vq5xEPB4v73fZeIevcwtdaRF9t5RTf9pNsIbw3Z04HkTIabgBi8EYyna/bUVvN6m7cdutkMdkQaz5cH993X0f7vr6ErblhpV"
    "Y4tysYtvXMJ0QkfkkrKAUksqbTa7Gb59KmOr0d3KO/WS+/SYOhFDUC7kSVJT8eoeYMSJhglZ9CTFxdP1Ey+t166e3FGSP0UnHOjr"
    "K1V7Z8yhcXFZrBuLehMgtBRxE/JNZAtMGTRD8cwdwKRiwydAZBheT08NS1LCTe82mSm4qO70T6XVnEPGeWDuHPjfSwOyuA6PJjRe"
    "TJXlyu2O61b4mWqk+Ekj14h9fsUeAoIDQBbxI8kiTiScwo+viu4YydOUoMgpisG1p0wkmgtHmyg6W6I0fdK5uKaqqYJ8GOT3H3xA"
    "rXvkYRs+I7ZJ/pGoLCr08w9MjafdU7/MsSpev1RYqq9u9FCgQSf7+Ww8nyqE3FGRORsYfBZPL1fu5+8HwewwGYDkdJBrh7rGck7p"
    "FYlu43lk1sBN2E2wUereXHwZTFL7FWsMXWvKSFIId+JcZrf5NCM7rXohwyZhcq75JCVO6ZCqSxznbsUMHzMWb2VDYr+Lqe8nC3ND"
    "UOv0xUV2lZPMHk6XdCE/C5YYA68+I5fjewFreo2xnA+PPn883QartJ+P9nfTk+33e6d/Tw+3j38U70Irosk0v8oFS+3LAP4q+KkH"
    "5NPx3vu94+O93XTnYG/7eBtypR2yjsrD8fhLdpNn/RAIBGo4ODr6aTtizyQPPYZMXu2dgyNI3FPRGwnD9CkGpl5/JLCSHklgVf1S"
    "zvk23QYF9P7o+Jft412biqTOScwb8d2999ufD06fNeAujKeNtwtjweF2Kz91tDkoTx1sDtZTxlrDCYfaZV+zB+BHQJBUPDP8CHYS"
    "4BmbzJ7CMMBNnyHGKkgGuemyS6Zrd2Kfu97c2T7cO94Wvfk53d37kQGqBDTDaDfJ7lKBA7gLovC8CUqsaOH5ZNPh5EG5+/H0i3Rb"
    "LMbDOSLhCx4em77PHuWpjG6L6WQATprcYIiXnfKFmd5uOr87dVaiqMS87ZQvPVHJ+b1gR6W26QW6S1d03d56S7i6p946jSXqVNJO"
    "WgxuB8NsKtOQMEHu2TMATlW8S+Vua2MUTPi4SYBF15WxYmcP2S4LYVVAiA7dm9oosrAdGglI/fGsjVh3ZNOReAFuzY2Q9/HnMXal"
    "xRDmElPpYDebPgpgx+/j0Kh1cxQ2Lw5D5nDWaVrd6XkizlZwRGhHM9jIKxhWqu1Vao+Gw8EkT7O78aAPK8RJ+abx6PByfOLknbHR"
    "DwajYtBXKr6v+uoFDRGn4iS31ldhFLicTkFV73bBKCgr29hqdldpqAZ5LsC7Kki+p04WnWa9NHzpRSY41V1+6Xri1A6CHBu79CKq"
    "xjVNhilgoVYsZoisJsYI+KvB+g3UqX+xya4jA5k/dp+Blkf2xCpdpHEZnc9hZX6GNxidtDmCx/XzMrUSrFOb71y/CneZbJMPlBYi"
    "eVEuVMw2SU6nst0wSnIf8cX4QYOFL5Z2W6/gJYznuubEZwzpqncNmuCTRC5j06bFj93MoVytuyQMmBJmfQja5jNzMTlT1pkbIr9Y"
    "sHeypZb8i8ZypMLzU4SA4fR2zuc6d3ziM7A480pKdFA6oLrm5oFSjrn6tapChKGKtmOr/HvCpq6AjaWe+kpI2ye5mPL9IyDEttRc"
    "aYzUdgPBO27H49nNpku2pCpPX2XC8urc1lAq5s0ndiyaDPng6BebCfnHz3DGOozl1Luei3md992BoAQq05EudZNI7lIFFlTCJbjt"
    "HOwf/pD+b+YK01Z3RrcKUlJ9WcDfSar2QNyMhLRxwJRFxCKw4D6K/HrjgYl2m5vUoC6XuNtSiYz0JXexhL/FE0eJOzvXVd2YLYW3"
    "MnRpPd5CACZWKOlsgDEbADMIkuYH/l8RtOw0uS9d8YXAgSPNXSrOO9TCanUZoqlhmr03YMMJFyTyo9gnZNMQbG2tal3jSQpu5oUw"
    "dpdPBd396Y1o6eSMMjweYrNpEdMm5BjUdXON3gmpBS44o4waTxmyNAtx8JKctZ8LKYnBTNpOypJp92Hdxy4VB/lhn+K4itZoHnJ2"
    "YmYch1WQF1nWo/w+hcyXmTyvvRIP+Ie0bJNcq6GUdZZo+3xv4jIoghLEvY4tXCyQOPzX1tmydPK20ssMU2naiffGVnNKYh/wCG/y"
    "jbjYN1+r/r6SUIPakkIlXbBNhJK0CyDGddSgbDYJLDgD08qVxzQJCMEoNJ2239ChrnVBJWHRgCxSvRDOxWRZHGpuYX/03A7U+slc"
    "BlTWE1MjJoTiyqtxVYUrsRG5tPIvrGI3I9UXVfwlFVnAgUABQineBPlV3UFxRdv/ERE9TKlSgxrWwlxU1cF6+E4mkGU+ljQuJo6Q"
    "q0Q1MMBVC4y7HFiSqcGuI0E2yAwp5rht1RI8S1giplcEWMtYBGjcXu8k7AyWBRsNT3UNZ2DgAan63QbF30T+kpoOwhJuOs17YEKy"
    "qGzU9mp8v6J3zw6m0hFbI7gpUgiiUFeZO9zoQt997xW6RFvOEBiUuw/KaXj3uty7Va8cjFwAbeV7plQAa+2tV2o6uL4Jgb17xxUL"
    "oP1FY+aMIJBEzFOkH0SMUb0SrzQhxFuJ3cam6ox4o5qAmohTrVGXEOXw0t1gkk/h02Co1Fvp7Rng0wPkOhqPnkJNnLYhYpVHO+fu"
    "qiZg7EwP+yaArrlAsVtPgaroIQkUwg0XOJ7vLyD4IGw1zU2HTGX1nG5jTfdVSV2vd1jZe2eH7Wqaw2mbkAKmn9tWx69ue6y6Fvgg"
    "O7NhQ3LKo52dg88n+0cf0x8OjnZ+Epxyd+/T6QfvYg2Pgh5SWy681z48ctXn3x66K0KjizluYym5At4sE3LBelipYuigGIW7B4yH"
    "5E7drWAUpPn6UneRTcLr+eHe4dHx32XmJaenL9C7yEKvQJBwr7KSCSYQs1BRmYze+bI9iGheJgth+dJTrwTois+2luhuKG+6PQ5R"
    "009vLCBCFzL0VdSHsorukH06mAgzMAYfzWSr1pBpSSGAxoj6eYXVdIvtQEkwoiliEyTbR7juioVI8zqNmp1fu3unR5+P01/29n/8"
    "cJroyAAPbpDe16W1RCXbHc5MSaJUNpYGa3c0ac0a42k+0VTYsj0152jsKJMHh9NIhfnPpNrGqGwOMZTCmhfa4atnIfg1kqgjPHSU"
    "1FxaWe6uk5UERxt628aHqjkzNIW43W8oLTt2lkDobu/bV5ukulHP8s+3SKvSVzkd8Ea8WlOl2YdUMzm0iKqoVJ1yzZQpJBVTcLPi"
    "M2OTnCY4sylhuuzgFl4HSs7BXwpeZ2N6G9jzpO/BaDL3zgEKTlwfgSvKVI4dW1345NcZnKJkNtROsyf+PXcvAjjYMtcC/YKSP1zf"
    "weduWdOiCzOxjRZiZWpZDT91mu1VMUIdyE+pcFpeXq5CRhxw3noTKRsIpvxzNpzne9MpaNZa+cNE0DtXGDfl0aj9ofOLmAvwp9MV"
    "D0KAbn/s7HTk62txvPwt6OC3Fr0JQv8lZsTUkJce3LxjOwem1gHQ7nbOxHIhud/qQoSEFAS50hNlCc3tdYu+X1H0F1AUncNmHELj"
    "rbz8Cu5ov32jKLZwkFpwySPlFv86X1XseQaf9suZAnHurgunERzS5zUiQaC/jZ04TiOSDM9rRcGwLgtMO65FwvPa82Dh7YIz2+z+"
    "ZoGP5zM1lpF2i7wowGp8Oh+1JSugrZLFAzzXnfVBK2KfpL2HFJVhIcyNxxwSBkV6I6RBbehM1hTTUPfcWUiYJ6hRohdyka9almbD"
    "kakBAD2yCl3HxVn2JR9fXbUajFsoepMNhiDVUgNIcLmKlR7mypnWq1Be42Ew8yqsrZdeo1db/ntH9kPf81NCnczSyJf8MU+/ZEMh"
    "BoGWa+fnn/C5nTDFIhGJVXKC+wwVZBfza1X4ejDjm/SLTqbji0zSv14FsM2RJpX1y8tbgFrFrwZDMbJ5f5E6+rZhkRbE2hWCuMlb"
    "5ppExKi6wBBIR76+ao0zugCN8Y1YfPRwd+synBar+fUK+b2OgpZ8XkgpMbCmAE/I2g7PevErJ4SrIegrIkS+Ed9SxR/ghjBSrNSF"
    "mKx2wSDRVTg2qO5N7/N8e4JI+xwZijxPP/kmJrzzPls/CBXEFTKgzBQN6TNJL+ZXVzgVu/TCqTwwEdeadT6tE1eEjyDFj46GbDC1"
    "4D341QODvPwlAdqEPCzuJCQSN4GtAWo4E8IMUP6C9hdzdqGm+ZRfy/Z7kLmndmQFxdfUzRWEbZiOUtHDCFP2EiybOJzVRfk00FGl"
    "6Epj4YuwGgr1GJpxRXpFDU59zrGHSKSK+pEpsGQ/nZl8jzXXZGhzt9KoE3mE6XdlttOSMCARRl4VaKQkUBCDIC4GMdHGU24Gya/T"
    "/L/APrqYTfOMm4qMXTPTybsv9Ufg7sttt35pK/szHITJrcUvZ1LuXqzlMVyXnp2XlNK7U1FVUN9pVxbM7q5tAvVSLJ1EYuWCQTwt"
    "0kJpkBZNe+RZNoWzBj6BjJ0WcxWH62s+HRdtOPcdbn9KD7Z/2Ds4SSp3HwsowgFUgbw/kOfsYXaBHrGcZOaWdI8AgWzAFS+e1RU2"
    "tWKEtNAwbN/cpMLVL9gt7l/msDkejR5agSyKqF+JkTeiQACOFLocjwR7vI3T2hZFkatGOdXd6W2Rx+RlU1aISzjv65RV5keSPS6X"
    "IGCV1ldTIwGX1rjNM4jrUgxGeVXRwa2YFnf5LVzP3FYVdtczMTmSVEdrqomcwDFjo1f28XYwkvTfhEl4/PlgD/NrfDre293fOd0/"
    "+njiliXT3a9x9MP2D/sH+6d/D0yM9ABzN3LuypRprQT/JRcxei3yNd0l2+p4yXzc5anMW8sBkSoteSeECNG7IYwhB33awD4ZImJJ"
    "2uSGatIjHW/DI5VjpFhHtuKrnbCnbEE7G4gxfWQSIJTiHzcpVC9SGRVAbx0lc9jFzDx1ggIUHe83dyFGqQrqU0kYQsv5oL++lk7n"
    "w7xQKUJ9hZ0CEfgxADE+7++K2kgSxeMJ6AlhwTxopZDw5+xkpvQY2V02GEKMVTFXQbThp5a7OZSh++kUbo73dk6Pjhl8FR+G8mX0"
    "ACsPLmIru3fZaK0s6r77Gwc4OCFbmLAiuSr6xFnRej3COTWBiien26f7O+nRx4O/p2Bno0jZiUxmUiW60GKLTdXbOfr4fn93D+xv"
    "jLt7x7fAURtFSZxefmJbP6VYp2lEjT9HL+Hs6MvCIEHzyvZ/HwH+2COA46QAaeXCAVArNS9mg1t5B6v6vIlbfC/0IONzJOtcfRoQ"
    "H8GPXe+minMDSzAqCe5k77X0dKK3yhF8a4I0djGe1YbGTAc9lEBVsMPmVlOtIuRIx3vv9z/K+Bp/+3t6WEKNRhDL7g4uh9XQ02D4"
    "htxoS6VMd7S1Nrnoc+Zv+QRvOOFCpqj5YpyiPdT8uLdOk1HfWbeJdjyWn2xkQwuUFDrY1DFkhvxwvHEkMXcP4TOQZFgUCy8SjkHJ"
    "DaRLdUY4G05usqDvDBbbB58+bEtcGoEnW+lQPL+TOLmiOL3fPmaFPOAyaCOrjOywq2BNp9bna9X5V2bVl/qBlGwbqq1GNKjoE1OU"
    "crrqOFtWWDBOBfEB4loo2XmqVgjn5+ivRDbYL44wcbSl66vjNZJEFgAVH4PdpiSJabDrOKaH0b2nZPyesQGVzLJYPtQF8p+WmAuW"
    "5DeNiR4ws9wMYdGZRUDUy4X69N1X/8N4CZIMtXUhhUaFTm5cRogJxRcqpYIofZVnAlJeqJmnDc7QOafnGahl0+kzzNNgsmowy8Xg"
    "q+C+zVsUrYHXeiKALHk1eNDXIlJfScpXxfYgEM56bsPWckq8KTG5dZEwNc56BI1SE3WlWCZZ342+mTiBAh9BsRved2SlV82T/cN0"
    "9/TcKwbXf/mor1jTXQIRcJp3aHdlkV2ejYdiuNtJEqmNWEgDNTX+kpsVgVdWaGskC0Z90KTOAi0oPNVW+QxT5ZC5lKjIxtNAySbm"
    "la1MGB4sfvNhWR0xOa7nD1dMUQifpck+bRsrbeG5W50pUaoFf4wwykaofyhFSIP/Dwf83s972uEDbChXSkFolb+lhHqSh1wyKy0n"
    "oCPk0RmAlWwgZTcorzdl/fLLEcfBHLZe3QEG4hs0hO92WEjElW+YFUU66D+oscVLu2t069fwE780vG1SCRuLnRlQ534FrSe2tymk"
    "cL3rHgJokVsfi3C9yx/TmzLOFQv04Q7W1mYzrlYJfLoooQTDIjUlwSIVcCjcljhFTLnLbezeygsfs1C3OdV0nU4HOtkaHeeU2k/r"
    "MSfiZZPJ8JGqVyFgw3QAKU4IV4+X4rXGL6WEfZI9UrlNTx07qBr3urF7sd/xclKL/GV56JjpUGIlVpYjmIz+NeZIMIGf/SML/aRD"
    "aPmCLrVE2mwVk8E0G7aY4wxr4sTuOrAnOeZNm00VvCkd5tfZ5WOLS2yCcMv8j8LOxLJ1WFCs+5nXJuuUddZdOANiPNFipCGj+GMG"
    "KfAbfEP7FXZ7hBfShczkGy93Me+LKWNC3ojera+BL/sSgfCmucblH8jckcFAapBs4NfprK3Avmq+W377l7eQwWAVYHe/d9yw61jM"
    "8WH2FuAGnXoj8Ar6w5edTh4hnvp0PGvb1tZEDcwiIcjzNmn+9fkQ3704xO6LQ3z74hC/e3GIqy8Ocf2pEM/DRHABI3RtNQuGFSqf"
    "eqK3W5hZvF5IsxHJMhUzQPXwi4X1suYxxls62gfwKq+DGIVZ7plMClYFV9OgSvcKHdPW8/zuvswG4UB3CPcmRDSAgLa0vku6BYle"
    "6eTnynktbKdZfzAvXM9bP7LO9u7+55P08E/B4i0Ap69ircqeODTo0vdPbEaR3W1Dv3yJBpbYFpZetIkYrZaeTazzGvG6/s0j/ige"
    "Ia3nn80kFlvoC6zp8zj/Ofm0f7x9oEM5nJxuH1vOE0qpo+uhb1xpgp7fDMQ3BXfDBwz6QQ2WH4Hx1RXkpJPRUCSUV5FU22qVXY6L"
    "NmKEcT8MdaNVVPquka2kx42tksQTDnJDspxNJqBoLsNYjZfp6hmMlv0Fs2SRjPD8W0W81/4QKPKfnO59YobWDm9Qcfvjjwd7sp6A"
    "4YfCc07G9oBEeIsgfCRnHwcCOBHgECX0WeCKcg51wnuiEDY520PgX9do5xL+Yj7DwCZLKhK0xwofv8OzNTHqloq7ojM3L6ZCPhIg"
    "sNKkLlCrxaqWGPAFMNz+e/YDuLBBPQ/JbCAMCGscppQ80vQSniLmby9px8haElYb1gU62j+RdZ3VlSnKvOo05RfpJ7Ap6eHQk35n"
    "HWn/8c4GNTWJ/819Ev6hWuDwolMS2Pjy2jkpOKuKYuCZzJsEsz0+i0eUSfARRf0V4SwEtMzmFOs2+Ca4csNkDuxqkAPZXoQMsTKS"
    "qo3sTu+tlFMD3m8cAaNI3++fxm84HI9ajaxLiVmpi4W9vDV3605t6SBJNq569hmcSiLCSrRM1J7h6IfRkMEZK1IXzCcJmX74/P79"
    "3rG8Ao5l7uZYWeTL2VIMthczyUzPYPlVMFPC/ILc8SSHOFfe6fju3s7238v7Xb4U5J5QuhaAs9oLZ8XlwXhmZmNjP3hGbbJO2cW4"
    "edcOTEwUG/c2Ssuz3Q8EB2DTXupQzb07Df6gqnm115jlzGTHVTYv9sKecGX1Te9ASxTfupsW+VV376rXCtnCzPOCG1n4su6+Rn7V"
    "39/IbxqsCjuL5Yv0Ip/JaCTt2DxpbtClsnd8fHQszijihUotQOca6hZCZk5HeMtnzBLi/qFg0D/vHe59PGXitnpVUAhODz+dgKRs"
    "x2XDZfo6aB4UDCAywxPuGbv7x3s78s3xNj4EcOgohQAO97Y/Cgn0ZP/jXiRwLDcaFVzPlQ4DxldXRGQM06rkUBWCtaIcxl/1WHGc"
    "xTu5cGItb3Ip8ep4kcZ2Ahe+O3B4Yjg+5PaDaJN8EiNOnORLsjJj0OMzj0tLjacUOzo+C5daT/0Rhc5Q1ViaVTuJioScCxIVYKPb"
    "m++nlN1OhmiliJaFgYTlixRC1Do7d43HULyRYBKXVeEKRrc/ODBGvVMbxFDvNvfEtbPBLL8FMoNBJDyjTaRsLhDi370lPrrEQyqA"
    "10VtysIwAT+xDWU2uxxijEEtlyT2ZyvnnhmhrhKQ5nT/cO/kk+BPdUijBlzenMkuAecz4mzRaWYPg2KTpniRVXIIt6ioQJU5lkBL"
    "LnQFqRtAUpu1UQ3hxb1GxG3t1avmakKD1mP8ezm+SwZ5/E3jBY/Gjq69mN+2RUUJzI1mCiWlKv0vceq596H1qEYilxkMznrSOBXW"
    "s0M3CTgxUDCkrkDN3+snMq2xwuN10wWpm/Rr1Rk50kA4bq6QxY+b21YwbqX6QI24OzhGhRYdHEdioRKxIQVjw6jyBdkzfCMYs3Q+"
    "wiB7ZgxV+hcSAnWi8ivQw4cRMn7c/tTRtqfA1gyhxcDCRb7jrjAZZpcoUaEZri55JproATewb3pL4t05WxVvWpgRdsCH43qXDQcy"
    "cWoAKhSBUKe9u3/y6WB7xxfvlGAA/HGEJuCDfvKHjJgU2KDnbl/PEIfz5l/dMU1kblwxNIPbuUse2W9VD+Xfv3BhqckZwExlnP8a"
    "ky3UW5C140j/bJ0qlfbTD4EcMcMD4UscBqn6NRupIJ5+8ChGg81kV7wbDIfZdd76toCqmrHKrZ/LxwqubBirjeZapDiNPSUWzBpL"
    "jr4QCHKVvzhVfUt9Ox4dyZuEaNV+ZELiJNczLvUihQCZn/cPDrZ/3NP3Uj/LoAm724efIneLoStHWcDYRrg2vGDidaLNcv5bdQLA"
    "1wv+XhbxP4waTKLAb/ph4K34MZ0Xs8D4kHUf88fkTZ0hcZeaV8E9yZ4efz45dcvD/X6DeINR75m7uNGnn2MXcujW8zpDuGUD5Tbs"
    "e4voylyU/gBl78UbW99VSIsNV2b0lGP1igzsa53DAj+BptcFygEq7Z4qE+pqncqxxHW0dxqU6BetW55OGBL4ZRdF+tXgVzV9ft47"
    "Fqfh7YP0YP9w/9TL0aEJ1vYrbv/NqyhIWAk84bXzJCGm7v4ZBgDY2LT9KV+ypTRxjVIwSYzHYiqMcfwcIp7BjbzUKJ26iII7szg8"
    "no9LlIH7N1Voy3E9ckku1X6S4OUmNg9GIkYbH9+yR/yeZVNwAPD7gicEY+At156dtKR9mhAVcyzmcQVKaLWn8JNaFBebTrRwt07h"
    "SsK5VZ6Q8UQHeJKdJnIDvrECA3pDxAUGmt/jv9OGXMFRDJeSRjqUk/2JeUz5ug61idULvYQOT9Am/j6MoQRHo0+oRZM/A4Mo6cyT"
    "WYbjaidkBMej/4baFBizABsrYlMG8IuFKbrPCqMDxzv9eKwmMnNJS7oefZfUiTzF2gIuHFIhkojdwRB2bhrmgj0MYnQP521SJywW"
    "24tYjKEYclwQoNrIORYCLKX9IEu/7H/cPfolfj0SCxRW8vVsKdrAeUULNMJYydfSFigxzIzu8eYEpZHmEnndWWf2xO6zFou6waIY"
    "DxkWa5SPAIcqE7ylSG0KHnb86pjK/w4hw+p0grXCRZYDtjl+IKdQ1W0ijiWcWaZSdh4mMYt3tAAqJ16F40Fp9Lt2DVrYKcAyB1c3"
    "VWazm5TcDgbmLi88r7TO2R3P9qD/IM2eGj7d9RdoIR/NIT2NMjuOsbnABdidH41weC7AI5iPrIifbrPiC8uWJC3cwuENvQTPBEZs"
    "0I6mOtkx9BTJ5MmB4zkYtg3FIV0angajLzra4KZuagiIYDkHE157MyaXU6XLphGf9upuwMGecTUcj2aD0dxXzFpOc5v3B9kIZcdZ"
    "dvml7cAzF2fmoSx4hVSd4IiGNPwDOMoTuIlLUj3D4Fax69w+Y7dAtBC/E1Ytyfu9lW24ToSSYGXqlu3SBBxYJyYsajlUvLUoi3Px"
    "8FaUe7diIIqJSzTlBSNDaeNvZU9IFjFkamSmqjJlsQVZPhwL68gWboc8gsQqct5vhPzEY+jh8nIYE48yZXTwhy9glqR8iBRyB8aJ"
    "mFlnYzGoOL4OpPlFogJqhEn9xQIJBl0KjV4Zbl03cm8tFNzNQtIpIjpUngE2N6OHgOCai0ybLT8U5n9+3j/e200/7J+eRK7HFpBd"
    "IpXqrKAaVTe4rRtVeKRUO2k8R5JK6kU5pmMXOgrEIkJjwuU8iO7U0WG7xCFGp/HRRulPCQw58wOgBfGOiP1Y6fGJ5XFsHyo5ozpE"
    "c3UTf1bG/AMchwSeZFE8kooDXkm0zpqjk1TEeLNeH6VTkc0WxiuKapi5y+zp/Hjpbal0aDbKRubFcS0Z1dgu8fT2+Mjb8dGucwT+"
    "AwKXGrSfHG40SRq1aAi2sVxwdCZbA8+LaxmKyEhOrW+deo6GizoXRsxPjsTn09SX9rd//pGLCK0qVZSm9raCmtpyhUnv7kTE7DRn"
    "4r/JnfjvV3M61eEbYQ623czw+Efe7svHtd47YmU6L5B5zQwoLgnTZrM1mZEcTLMpNKZvqag2uNSyx1rSiXODNI2aTNF26nuqXVZe"
    "ZRjzE8s5a94gbOJoRrDhVr/OPqUVt2RFzFi1lVQwl3hlaYigZhXz6USMb/oxtnmY5ifjSXslgNqG7hC3Ouwi29RmWVthRF6neq0I"
    "PBOQVZzjVfbgcxZoZwlak2cv9zDeTdjIW2gZYyxq29HwIDSW8NkKb5vSiIYwcKsvrfVqVQ/fjAS+NdzZ2akDhtG0u6BKsaO2R0LC"
    "O9avSHdb8ETwsu1dOdcFND4whLNmAg8FyzqUDwjwELHk1pNKZxI3WVvDvcXQTVqZmWQpJdmYe552OZsUed9kgKTBWpm0t699bxSz"
    "3xpmQ1oF/crr8PVdPoQY+k7LrL/nYDQa32W+pScd9RlMeYODN8WHY7D2c4AYbiUEJ4kXk7opsV/Zrc5XYj24bawtr0S7B9ZPq+I7"
    "8CwfO9SJiIkQXlHZMltec+EEdCafa6bpzDwuIgqXzNPxUqKflprRUGFcws/As8oRKbgagquuR+w8eRcmIhHF8/PWw5YqjuOXMCy9"
    "BqN2lyVYsHbqo4A6DjuyHG94MhblGATSH2vRG5HuuRTcdgsB1mA2hLByLApqADTq3v3UeKosX2W6x+YeL+38bFpSjk8uTmURzb2Q"
    "QSlhLDYryRKQeg3BKxN3QtlO9GqhFfUKZ8puNVdjToaabDxzqsgFHtkanzLTSM7lehOralKtLEM4yzY/e197rokQlXFlGeKIajwq"
    "E9NL+DHwEWCGxdaIbF8daLpiGKSwb9LCW0M5jVannIjuSRyMt6/HoUONGbglnhhJ0PmKUAiLNGZ9Sytbl7PKRnYg42fe6nGzKPiu"
    "BQqxDVGwi9uhB8F+KGVWdEl5AdWLqvKst4d79IUTrT1CR+eBPExnneYFp1h40SnKqPGcATTDgoHZRpeCP4yATZ4xpQTKRgHT7RhD"
    "Jz+FOJ7V68FnigqiVDRCT7puXTjurq+UxBR3w4u4b8+6hCvSNhgkoaEu1xDf99inaJN+axvN7nppvPk+RlnT4GMdXEE3Pu5Tb6m7"
    "QiKogOekJx3DLetDRyZwsResbrsA5tzTVSlzS9/NMpHhhB3DXRfWw+vuuX/KAs501/xrkykaXvSDESvYsHZXVbZhYBI1r/slBWQe"
    "EHDUcwx9eWJHxtcyyPAzkMv1zusTqOb62GkudO81O7Z2bPXo6gJ21pHs5lZzTYy0vJFwYaEHoaAf8wkY7sp676UCFLhIesH8H7N7"
    "vNsQaCjeSd6kXX2JMZ3QhNHynSgIBtL6+xkJOGgKiFaLy2yItDMv32Cwa1OYNrgqyjkIiP1rFcjpvhQUkqdZ97U4MXJQA5jKnNyv"
    "HGIN7MEvuRopGeCzSmSNqysXCRYG1YkW2RVeqovGLYVXzhOcHN9JywPnW1d/8++UFDDeEL4EB0kjQF0CXlllQThEdC+WoNc4zXkn"
    "YXm9NoYkDVeztkGlTAYMqmK4UlqXpNnTkJV+3ZTpOZ5ViP9NNrrOMW/P39K/b/8CAXP2Qo3RKL9PLcV05Fe7GF77AJeAR74tq4/2"
    "5lJtBtyFaeCNDA3vULYC0BaY2ntbRVXT7YoSSwAzkX8WwWWjufTyyLyWyLx2kbHeAjJXRCmMDuKFTgn+pJFz6g+bNUv/njX/7LNm"
    "OM76uB8P1bS5lb/SSTa76TU/ZU6SO/pRoA1f2/RdkECXflzOhYgyK3w7gGk2ELvP+8Ew/zievR/PR/09kD/aV61DqIpgruB1r/kb"
    "BfetRaOuTB99oaMo0vEEY1aMp7PlE/FbHLKOJmhkFGjEVOHl62k2uYHHwe3gayazm+V3mN0MoPwIn4/I1wP4uHx0fJrufdz+4QCy"
    "6B5EQA9GEP14PEmFmJzObuDyBJBbDYpLDQ00tz8S8mA+uswV9oypy2zqDkCHURpJDKDfmwadDqP5Ht+JU+K02Dxr7Xz6vPeQX86h"
    "0if1vnUei9mUP1zmk1lzD/+gJ0IB77hhPp6PIBaOHOEAhdb7bAAC12zchHnZPPr48W+otWnqS97l5v5Vc3aTN7HL4EIhWpoIWola"
    "9wMxKSFB5HQENbKZOEq3QrvQ1pc8nyCMaa7I228uj0ejh2Wo07yCyOcjAQfQcGfcsgsuaV5Nx7fQVcIgRpM5Trrf8EmIvFfj5VF2"
    "m/ea9oU0TrU/0c4WR3gZBFsJo518Iyv91/kALrgs+BZenbdEF1F9Bw+HLVvjdlCgQtRUKJBIbR8SnENmbfnDdflzIXCD+VP+qJdq"
    "OFK6vsFdkQZWsQPZWcfj+Ux3UD5SApI3SEH62yOhgkNpKLrUstkEW1LAHOkWS/tXq3sSUK9J2iBBVgCjFLohutY6hF8txEg/GkSk"
    "MC7G0hkLW/33QlsikjS8lLmocsG8k9DcmZps58v43ivbH8CZnuSWDbQYFiSaea4q1dxgpK1qbAGMny94ZsKltJUtQS5Mt0biW+/q"
    "5Wv5qnqKlDPLRT5ESpGklJJ8wK/lgJzROXaOEzcC5Dasa4c5UpP23jz7u3n+mDs7uvnN7+nuZ72ru2+Dfd39vPDOvveY2ylJN3gX"
    "7r+3eLXFe6Pxz7TJX/mbOvRlMi7e2QmAu6g/9MutP257BUYst1PNXmtseq2wI/yW15MGcvw2x25aPt9UH1DjuyhOQjzK0QYlmwlY"
    "WTFrom0sQmwF/NIOQhXHtCUlpdXepvcJ2WWWjdmaagNXVTUXXFG1PLY2mRGuxt2Q6AJx+1M/tfSMYXwuLJ77OV9f4FjjwPvzMr0/"
    "muu5g/BPzfQYSUxyPXfoeabnTe/YipyALezdwPPHsQtolN0NruXYU/FAvC4TD9zPepW4b4MF4n5+gRXiAvy3XKBWiDcM/3qHfztn"
    "5Yn/Dz//exPvH6YBiJ3niR5ABsdrfXvRM70/AM891r+wvMPjuJjM4/PFStEnqFB+agyK1xV7qKeE4O5tB26HRJOeAN8eTZZHfTTy"
    "7rjncv6T61VTVmY6eQw/J82lreZsPhnmZ2h83CElznseBVz5LXExl9TzQoFJt5C4HTw6f5nKomFxJP4fm801uNOnH6RiAMKWi6/s"
    "XPo5G87Nrm0mDsJoSuVH+0Pnl0436Yj9e9b8LYDuTPTLDEwgiLOSeiFDNHUg0AH8K3qbii/pdT6+9YJzus5OzDcxHB0uxYPUBzDE"
    "VM41pYnr/FwwuUO/FSQfUexUEDLk7T86og+2oKgruMZtNrsUE5jj1WZAfiOtf1NDwSH6rRULBSYH7gIusOwQnsmgU8vLy+d0DGEA"
    "sKg3nqp47zyM3iCGejx59PPzAKzZ9YzCklNhYUiysxdG8UPR9wILGXHwsMy7EL/jRPmaT8dFu/1dp7kG0zy+6qxqq+MCcSXU5enc"
    "k1ccrVhHaz09meK3cNrIva2nxy6UW5Q+sqfJw5RQwynKmIGNlBIDpUrhkDGlDqEl03G3wDcueKLtNuuKWZFk3LqNwpYp14tYiOCE"
    "0u0m9bmZUjQqdtbtdA0388E7zAwNJtETEb0SjZ+JrXQGYf3O3e6m9+PpsC8DjJJi3d7b+Fx3E7xZKOnl+C6A9Lb33UKQfg0gfNfr"
    "duuCQDNTkPJA+TqdMUkKbJ+VURcWTI0zYxJYrRJwG821Zc9eKiQ7tbhTl75eoQ4hWscjoPr9a0Sy+GcUKyiTM/6f2kOJY3acp6jr"
    "KEoCI9xn01tR+GJ+jbqq4fgaI0jGomJ4xakreu1KU7ChwP1hsTp6rdWsIs424nhE1mjNeiRXyEItpcS3jvVqiFFcuUoUgbsHX0F6"
    "cfRVq2xLjv2VL1m7bsiBw7mxI2zUmVVaB9v6F5ZzX07YDG62/LBWMXE0qPkPEEoDHJ4mmho6hOp3EsyiTHvOZ24MtS7MBQFpCegO"
    "Z7W83woE59Hlzb0eajSOvp6P5wUT3FaUmE2zUSFaydtkFnWabbHDgiSQJETsXjwgNXRhPLdRAAKiMfJnCe18KTR+vdEjpKDS3hk1"
    "kjfI0Qix8uUCy0LVsBN+Q4g9tVe1GWQl+KFmzOhC3kI0qHleqKnqNOUscr3xubkkVHnMdr/0DsMjv3MzysTkRoii/EZmOEDXd4Fv"
    "ewkboQ74lw9CDnsU/0HSAN0YSI+i7t7f99KTne2DPWIEN55p52LBocEhSLxho9xSW+ROaYFuVYFVWiCh3hBiY4KdTmMF2a/I7ED7"
    "TO8zMfafT7xvq+cxzUIQkeCJAVO8DAu6B6+aO9uHe8fb6dH79yd7fqLM14CpKfL5U1iKBMImUkrb78RrMdoCkCSL+PEofszBk/4S"
    "gu8rZEpDJZaLbPi3sYi85s3dxgJiG02+ZydxaTiHasHOPGuHy2DTIKeN8sg7pacY22bVIab+QSZ+mKGd8EtsbeIyPz3e3vlJzK1d"
    "UPqYkDecbA8k+5INb7OR9uY1nUkixZ8vYNZxE/SbiiRQfYIw6wgMpIUtQjpBMbEkD44wFtDep2i+aUI+HQ+AF0xUoWi4DOfwH1QT"
    "a77mids5uJckFSgJW0DaFUeXMy4habRwd5HCq5HCKzTRkQkO+vS0qOEMc6htucTTSCw1gG+rdH+BUsZlTQuePRcFQA6hEnfgSzU7"
    "vMAhlSQX4Gd99eZDmAvaKDJLNXkRhc6lOIEWzQPpZb0jpPLpeDjU0YtR04OdS9OYkuc6G4yKXvNH+KMO90TNk88mY9GBXlPenWfD"
    "M6KNCYqj26CgQ9a/yODSHsdDMvyu0u54F0PXqln5V5xFEA9/Img0kKuoR2RM+pkcGNF41kzoNQ+SiyGIl84LqyBT+0iEaJfz6RQS"
    "MgqWFtGPKdTSu5qku8rzvpJ2atRAJZn92OMQo6yTvCa8XbvzujReokC8ob3NrsVkmvfzUHzAAqETKamywQ1Bj3Gv3XTGz/KfVCxV"
    "E54EZ8zyTxBLCCqROOVXuhw3C8i0Se/i0pIF4jXWxfBFhrAWVsLVF7sDI++Qsa7G4DUZRlIxuFoHnqhq6Lo+g5EF7RwXyKfYM2UH"
    "9JNasewiFb3fbP7kVta9V/Utt7BT04dFlnJAROKVJOUQz85vIlhdo5H+eLz96cOnbYxT+HH//d7JKejcpPGOYNSDq7yYLf9XMR61"
    "aNltmZLw4OiXvWPa+Jl0iqL/X9GPrOQewvz86ZMHU/tZef+rgvefn7c/SllNQlNlMCbWq1fNpVVw3JIcP0VrpE/Z5ZdSpi8HZpJh"
    "/PKpQ0v5SptzmSIkhPPspjmeQG5gXfZNkyF+p9mCXOn56HIMe9Bmaz67Wvq+lYAZ0NVNz4sGJodHtAoDtAyqpvbVDZ3ME53CpdRw"
    "SxVbzrWZkta9Qi1juwRWZGifdbL3n5/3Pp7ubx8EEJ5t9PXD9sn+TgC20uCLFBQCSFCwG3YVAxent/ktGCKBjVNwMNBFs35fa8KE"
    "tDC6GlxDttvpY7ul9WPzQsaenN6KiVJgQvTxrWAO6EXUdS1ksEaRXjyCkwVYNX3zvRKo6q1wS4CVk7LJHZnhP2vhq6J13osYwHmt"
    "grFQG+uctcTP1fV3rXMmIY2jMO2xpnCL2NcZK1Q7/xUOYCHWOk9ixwRicaf+diJBDhezuePjPnq0OvPodM5YMcXGTtcd9N16vj2T"
    "KMWMM7x2hxkL+qNszeO+BZlUQMnaAbM8gAO1BS5YvHW+DLkZAhNS+OdGrBm0qpllA1i3ovryJJvOZITr1nKLTfVwgzFDxc4xviha"
    "kVCV0pQddcSgvMKishmZ0j4EjKHDDGixWMfTx7rQVemqBrhYeyw8oJ+AJskDMDllCA7RmSK1GnbGDgMnRqunxkT+YmanHi1l6cjM"
    "+xZRvEM5/hAv7SWlj59NQuJNVgeX84hBYeTo7fOw2XgyTsfTvo3Lg69AfXiZDeWXNqGYfwSUY5cWw/EsTF/ShtdiNYguCaLJ2231"
    "DrWwwEqct/dTMdmn4GQ3EedBmMJ4aRJmHoLiHmeVk+icSSGU6eyU/RyXS+Y5sFkgmfZeiywjFWdK1v+SY4oHB7oDmqcVs5ME2id8"
    "K2C13RlprIKs6VJ39XtBIvwfZ8DkDr6xGSL133Y75RW/JSXI/lVG27rNZzfjvhXDwikkJ0+PXq1BKuXR5UByVPieDvo9NOBNDFsV"
    "rzRHLL4FlWeLVyVfO4ZxYxmez0L5L4NRX7OTFHltyKTlxRLHpQVfBADIF5Ez8VyMEuRMYXgOEk0bGk5K6wjuB4VkcVWXqllkfgtl"
    "Nz0SdEEqAAX6oKsRvaGta0Jol4a+o7TRvOKM3O3kt+KkBKbRMBp6JLBavZbs8NzfgA06kxtwhJnJ8AMXlxuxMvHCBx69Rvn98NHk"
    "+Tg7D0ZYIEVxg91EUD8cKNPTM1H0fLk/KC7FmTRskDiEuFX4sSf46T6IwoGgR4pxqDmjDCMgU4I2/w8OB6mchGbnQL4EbAvgl79W"
    "YzfqeITAE5U94MAFOeq0subl46WQFgN1n+T3ssWKg6/PNt1BxXtpw8lwQ6kKIG5FLVU8lTsO2WrJfmZZoMUTmLI8XAqZKJ/KeMsu"
    "X0NfoVrWAvrSjIAy/oNll5eU6XNZbmYkAcaTcFC+7qU4vGXbFmBIjIaezpDgWqt+Iz5MA3HGILEPWE5th8eKK35Uvb4JPEdEO+Ch"
    "7gTI836pCN6mzB5lx0Ukcsrr4+I1ImHkVVFOcm9epjbwSmVqFySdylHYMXHaBUXH56xNyOKCpG4u1RIrWKlIOxhsLLZGcU8F8n+F"
    "GD8IwhGixfzU/jJhV1zE7ZYvaibn7p01tkOkrynGyHMA+IJflfE6a92SEXNoNCgRDXWaEV0d9wEVbgw8wGIK+kj1xtFXWeUaxJjk"
    "P5TZGWjHxsnyoLgC9VqumhG1hkN/FdA+Ssa8zjHlRpQjlzFkz1oYp4c7UKgwkSCC7VNNp+r0wGrh6CVg1JsIIOwNvXYz4y5Dffh7"
    "n6SO1mXugrXK+yGYgFBlJlZKVTKbnaOPp8dHBwd7x+nB9g97ByfIXVXOAcFX77NpfiOYe64i4JDxPT76fLpHa+kcL6KaTlog63ze"
    "3+2urYQVbiFvj9jNW2QD9FSsr/y4WeYOB9Sr+vYm3NMFVHAFOhR/bd+dgE6bfHit9Op2gjWPdgWa6e7+sZjurUk+VV6YqewaujW2"
    "vFblJ6j9Hp9iLbexkcEVNmZ8eOUVm4MVQet60nfRKnPTBWSk9LTJarTbAA4FcHziFdCJh5ljn+w7p/YaJSg4NSPozgd9MUcAXzlb"
    "CLJJI1qeOcS+xEH2yYdZ70AboOvnhahFRkMaNxdkKSGX04usyOVCiLsJhs07alS4F6po7dnwG8yUgfx66S3s9msr1EPDGM3IiRVl"
    "tQGuuIJmOLvgfLEslvOVzBqG04tn09hIZIK91CR71kRjJluUkG2u480loAps292VlXhqFkkILqVJ+fgGc0iHaJ3P8hROofJI346M"
    "KeabaDtLgH5lJ44YLdlfcgin39Gg5GI6/pKHtzy0pRoFpVOMz93UMUKIDDaRieSt6YmQu9L3B0dHx+kh7H9v1zXJaYHjvffp4ecD"
    "uHtdZT5DZruT7cNPB3sIggZuxg7OxmOQZe6ZWC42hxilVaLtByKt8JnwXHpkt+LIXVDlQNCK5SM3IMKM0U4JsvwEbWsK8YvF0pij"
    "2Stm8gdCkkL3DOkhn5PmmzfN1fPmlkWPUBZHWmeeh+MJuJm6tJXv8ByFbbdICqFg5kV362CNuZDl1KSgw8naq7/38KuxdKpLZJzQ"
    "uFBbvlbkGWYX+dCbefDKitwREdKjWJy9u9uAnfIJp7XCIQ9cq9R7NVgluHKiq0//qrBIZuQaHFo+McGIjVUB8Xlh3fEzzixqyEQ/"
    "lB4NpHKFRlIau0mMJcQelnSA7Itid5rpOxrY7gUgzH5BYvVh4ZZ/tTYf9fPLQZ9kaZPSgqw8vlDppTea8cydjRp7v0yi7l4M2aYh"
    "hTrtUcUsDDNBW1IHC7myrJ2aVpVVQ1R5uqRCZ0ypYCDZstYGx6QDKhwsuvOXHAFsHqiqpR7fux0zcQMuXJBPnB3Mui+bGiEjXmjA"
    "gyNNdBzr0t09MzCEr2JdJdIQT3ozkta/0Gso1kj5FsSwKBwrE++Dsha0ypne5n1kMljOHfta6YKDGgukDmYjiNh9JXpkUdu8QxK1"
    "r8od1d8CVY0a+iYjvyjJJbJLMryW5ddbEF72VQnTrt8zuvb+gA4G/UM5fWMT+6I6gEmaRad+6lWkZjcDGmKm907avYXErgVmQw1C"
    "RYhU+86J0AodbUI6agHRiL8lGToqNisOppJbSoDGWGcULDtuMN3TzM1ijUuglCWXr5Yy6caKa7IqyfsqX8eUtElsJSjlZJwO/gxE"
    "abO6awuIk2RsCD0bC5Bsa7M2c2GAESKWeBvFKMufOhffNZ69e0TEh8jBTCvcGyWFq7fZ8os7v0kgd6tRPl8CaQC5yFYNhluzxbKZ"
    "xrESRafIpVL5IoqTqBbPaCy2/rwrnTqGDNAsf8pXXWL0C4uK4XEYi0iUNZUR/9KqwcEthNdsjguw6E+nxWU6LvTL0fx28mjejyaN"
    "Rnp8siM4x+ednwQo6f2m6izno7vBVEZ+bLdMMbQ4X20liaz6894BYqFDBkQri4JQdWUZWLuqfLD//vRDncpYUFb/zqktmGUV1qoY"
    "4u02ffKpZtMnnxTq66b+7t7JTq36sqCsv2rptvP55EOtrmNBWb37zqleq3lZUNZfseh/gF2hTvNYUGFvqbe7fwLOFKBOiPZbFoGq"
    "LTTRagkWm+7cZLeTHzJMjMjeIxuHmeO8uJznNbxlXmXiv1dfArZ1IVuxTbZ1SZ87iB4AxLaTnEu/ZFliMZtfuknN6QrFBHfzPLKK"
    "J9n0S/rwyC9j/Pg18u1G9ohb/fZWQFJIqdcMkdB+xNP1YTRTMS5IqRZnN7ctQAwuBO+UlnMIxN9IXEUegFKtcXtLfLTIILCSOF47"
    "SkBmGB1VOgRhKD1umM1U8T4bAck24G+uHVOYsy+hS4E9RDCSA0eulpo3WtNZSx72pBA954zoj1M0Jvsw1A7RtrFU0CtYvl2eZnfO"
    "XawyiSBERIGBRFfWibFh463dfWlnJUWzJAFJ7//nL24MSKNDlm4UEhTMb9U8tYefMdOAtyT0x3+5+HWe519pbua+Srs+w/Ts2ibJ"
    "qfjurS1+o/O7Es7rOecKUO96358niYzuaoMKiSPM99JqwndSJWQGqfhGZ3fVmzQnM0ku5gS5QE2e1T86cysOwjCf4BrBnZxWXpf1"
    "RHes/BFVLhB+yiegJSwVxkIMw+q5H6GAYbEqRhDUWD1nSyqG2wIZoPW/Ql4bX4dcZ2ZhPJGvaXYxRpHORUbHdFHI8gtfoafxQ+pq"
    "gJqyUsIi5yTnixCLYhzC9B3kl72Pu60qHHQ5B40N1RgKMZVNQSlx5HWbusmn1sdHD/SSHWi38Kh0XQGsJFBFiEpbkAbzHeeqho3j"
    "nzdQ8BWcvNs3gtFYMSqpNTxMpB2YhK+7yytiZkkrLCtvNgKD2hi9I3CXfLhSDm1Un8n56lKOdNVacuYaPip9qYFWGGoMHyAn2J0N"
    "qRRligQe/nlTMopQwBtFGcANNipu3WPwM8GaFXCWQ/sm91AHawgacHskBGDTRpMZicebNKpO2jG1KKlXzIdWS6tlm7LTvL+XSRDM"
    "7kXtexusAI4D7snejUbjfzaXxD/Nu/Xm9RwCIQyzRyGTN2cDgSm+6TQvxCD1l4pBP29eCzJPxWB1dEiwpcl0/DC4hWAw+luznQ2H"
    "TXFuWLqG6EmCHFdXzYtHoFEmkE+wQe80e03OsreQ3US+hMfwiHsNB9z09Mf06GPTPaNeBycWLIXHHDzjwM/37+tUev9eHY7WdcXD"
    "7b/91LSHabYalIF6thbm861sDkqps9wqrQldrK6pu6jPsT+cpNsff6zqpSxlaSN+72x/qlFLlJLNvcV6u9Dt4x/Fgb2snillG9wF"
    "7D/+uNesqoiloN7asqkJCoJmZU2lHzDDKF59/qQnTWQYsQwO4ve6zu7RLx+r6kAZqGUqfdg+eP9LRSUsQzQv4tX7o59rdEyUgmp/"
    "MfT4dLL/48fKalhKxRbQQ/fpRDYXG7RPJ/Ksj8VPDvZ+qcYOSjmLDsyQo62o71Dhcix4eEvWqTEZZSk193VbOwf7hz9U14NSLo5l"
    "lDh1KSF+7v1t56CsOHyH4hC6o6PUyFBzp85qMaUIhjt1VosuBfXeLZuaP//vak4JpXBq2GqV816VwdXyVtcqJ+SOIaQx54fgKyqM"
    "x24TjU8lc99qrkjNrWGk9o3icfaFZUOk2g59BzFrYEcrV1OZwC5Q9AXUVKbJmJrqOlRSXZeoqK5TXCSMkuoaTKzuUty63VD1+vso"
    "nV2zFUfpRRH50OdrYMTtywx8H7p8S5c1dVd/WtXV9TMUV9d4pzj+onO5Z5Ni0/M8gXcwQTQThqxQ8Epfgkn3DvEmSOUu3rFqKecU"
    "HRgO3TranMDHLQX1zDQbXeftt7wroTiiSIWMtOiDhcvf/F1M8+xL8OWWWAciCBzjxLeEoKUWNCAkpABAaJw1az8sF2LKTJQr/oMM"
    "0zApdEyHTgu1MaZUXW8Ex7SZDrwMTZ2Bb2J4HoCgKsNOczKYXUK4+fE91WQGQ9aXPLTSNVeWYc87ECBWRqsX62otHC9oon+2vLzs"
    "hEeGfz50mr/ARy/DtDTKloiBfL48y0Zt/TzN+oNsVLS1IPNKsKF170x3OR6qDWUKOQXb7V/Esb+byLLi/xiWeskBL2iWiKMjeDi5"
    "sG6+OqA+KCBLTSsSvTKQ+vn1NM+LNhIf4HloQtsfNMrerFpR5ucrHVQUfACUO9C8bOrzJ79CV1WYQojtrqqFNV7LYfeIQhoAAkmw"
    "KB56BRGwAPaLLPjaFoSG3MIquGL/bLrSmwokLld6l93zBhMWAEouF4Ov+cI20UDv10qsQCiAXIJ2oH9ZXl9sMbkGzlV2Rlbxf11f"
    "7X+dljiVWjGkt7j+nI8fj7oLEjx+vQ7k0HlqVs4JrFr7Oq0Kao/NCpnGcKFH9C+WI1iAdkTQyfx66/xa9/SpMkAbqrX/YjXk16wq"
    "qNYM4EnMKYCUmANrJrso2tAjgSk8yvVN9PhaJMI/S6G0VCpHOQXQA9IIpyBkuqIp8UnQAoA+WiTGBLhtjg+R4vAtiV0LaflPtMfu"
    "0oj2RlOrOsy9kam32dTajDK7HU7MrGvzo+sGanFUeFICAgEQ4S2J8MdQ3vfLaLUIvFYjaz9EDd1sd7rREigdBzhzhMfGzbkWBFax"
    "hsALiNHawtyVGkdnbbaBTcJslWtLlUqARclTbZLwoN6GoGABvzHb3GQQM2iV64jXbi4+14oJPU56ulzBGBI2Bo2ohUr5NX4CQTxr"
    "COzIaHvpBa055rOYGWAy7iuwCHjs2Ho85PIeJXylOWAsyF9M+PAVMGZKST1nOKQvr9WZNhjHVCD7prnmpPuoEUyDYMM5cLotcyX+"
    "iDkYHiv7ayUjYzpFeT7eGqovq1VbU39UAl6AciMUCWTkvZLkQSP6q92HcBf9tQRm5nKw2Q+J3JwJyXYVoONlCvxdwQtCQUmmFsjA"
    "IOxZAjf/oykj0jovl8gvH4w4msBhHriOLxNPg/UKy8wqWbBzGsSW/hJOMjl0MEnFU6eplMpJCc+9KFie6+punOY35KHZCO78Nir1"
    "TMyG0JdnM3Madk9s0bOa1oexXE3D9B2J9PsN2p/XTaPM41fpvSIhrnM8FMBDmwcBd6ayFTjSQGnzqYPzMkmifPQeJo9UD241yUa0"
    "ep7EuaoqEXAACyypcMVV6iF22B013sLDzgXe4RyonjMPlK4/SlRmLsSJ2ZcLpq0rLREKmPHUVyBqPKuA6TqH+x87dh7BxyReFxCH"
    "6huwhOP42jUOpUuLOSrE15zgVe2cFoQEUFMEryGCvd92H5lPoKpcMnWTMn0m7ZhoT1IFlBIw4RiRTSAS0Cy2fQMwdSqEz0IYX5GK"
    "viUYnsbT9kvvQjp+/+vptBv/D3jw7q0="
)

_KING_ROUTE_LABELS = ("open", "warehouse")
_KING_MOVING_LABELS = ("city", "mountain")  # both fly; the king takes over once the pad proves to be moving


def _load_king_class():
    src = _zlib.decompress(_b64.b64decode("".join(_KING_SRC_B64))).decode("utf-8")
    mod = _types.ModuleType("_king_agent")
    mod.__file__ = __file__  # its models live in our models/ directory
    _sys.modules["_king_agent"] = mod
    exec(compile(src, "_king_agent", "exec"), mod.__dict__)
    return mod.DroneFlightController


class DroneFlightController:
    def __init__(self, models_dir=None):
        self.mine = _MyController(models_dir)
        self.king = None
        try:
            self.king = _load_king_class()()
        except Exception:
            self.king = None
        self.route = None
        self.step = 0
        self.debug = {}

    def reset(self):
        self.mine.reset()
        if self.king is not None:
            try:
                self.king.reset()
            except Exception:
                pass
        self.route = None
        self.step = 0

    def act(self, observation):
        self.step += 1
        if self.route is None:
            a_mine = self.mine.act(observation)
            a_king = None
            if self.king is not None:
                try:
                    a_king = self.king.act(observation)
                except Exception:
                    self.king = None
            lp = self.mine.map_logp
            top2 = sorted(lp)[-2:]
            decided = (self.step >= 10 and (top2[1] - top2[0]) > 6.0) or self.step >= 60
            if decided or self.king is None:
                label = self.mine.map_label
                if self.king is not None and label in _KING_ROUTE_LABELS:
                    self.route = "king"
                elif self.king is not None and label in _KING_MOVING_LABELS:
                    self.route = "dual"
                else:
                    self.route = "mine"
            self.debug = dict(self.mine.debug)
            self.debug["route"] = self.route
            if self.route == "king" and a_king is not None:
                return a_king
            return a_mine
        if self.route == "king":
            try:
                a = self.king.act(observation)
                self.debug = {"route": "king", "map": self.mine.map_label, "mode": "KING", "pad_hits": 0}
                return a
            except Exception:
                self.route = "mine"
        if self.route == "dual":
            a_mine = self.mine.act(observation)
            try:
                a_king = self.king.act(observation)
            except Exception:
                self.route = "mine"
                a_king = None
            if a_king is not None and (self.mine.pad_ever_moving or self.mine.pad_moving):
                self.route = "king"
                self.debug = {"route": "king", "map": self.mine.map_label, "mode": "KING", "pad_hits": 0}
                return a_king
            self.debug = dict(self.mine.debug)
            self.debug["route"] = self.route
            return a_mine
        a = self.mine.act(observation)
        self.debug = dict(self.mine.debug)
        self.debug["route"] = self.route
        return a
