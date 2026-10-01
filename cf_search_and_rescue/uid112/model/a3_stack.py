"""mtn_a3 MOUNTAIN STACK (Rule A route of drone_agent.py): A3Stack(parent) owns the whole flight when z0 >= 13.

Loads (all from this directory, by file path; nothing from c032 is used on this route):
  a3_core.py   the brain: 1 m depth map + plan_sim belief + time-sliced planner + commit/terminal logic
  flight.py    the M1 flight layer (take-off, terrain rule, 3-D approach/hold), driven through a3_core.SearchFlight
  a3_cfg.json  PDM tables, validity calibration, detector settings, parameter overrides
  models/vdr02_full.onnx          RGB victim detector (M3, from scratch): heat 64^2, off, cls 8, mask 128^2
  models/<vdd>.onnx (optional)    depth victim detector, when a3_cfg.json names it
Any failure while loading raises out of __init__, so the hook keeps the whole flight on c032.
At run time: act() never raises. An exception in a tick returns a hover command; the third one switches the episode
to SAFE MODE (numpy only, no models, no planner): climb to terrain + 6 m, then a blind approach to the heaviest cluster
of the last belief (the clue centre if there is none).
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np

_H = Path(__file__).resolve().parent
_MODS = {}


def _load(name):
    p = _H / name
    key = str(p)
    if key not in _MODS:
        spec = importlib.util.spec_from_file_location("mtn_a3_%s_%d" % (name.replace(".", "_"), abs(hash(key))), key)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _MODS[key] = m
    return _MODS[key]


def _ort_session(path, threads=2):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def _cfg_name():
    """a3lp: KT_A3_VDD selects the detector config: unset / 'vdd01' -> a3_cfg.json; <name> -> a3_cfg_<name>.json."""
    v = os.environ.get("KT_A3_VDD", "lp").strip()
    return "a3_cfg.json" if v in ("", "vdd01") else "a3_cfg_%s.json" % v


class A3Stack:
    def __init__(self, parent=None):
        self.parent = parent
        self.C = _load("a3_core.py")
        self.FL = _load("flight.py")
        self.SF = self.C.make_flight_class(self.FL)
        self.cfg_name = _cfg_name()   # a3lp
        self.cfg = json.loads((_H / self.cfg_name).read_text())
        _thr = os.environ.get("KT_A3_VDD_THR", "0.85").strip()   # a3lp: vdd.thr override
        if _thr and self.cfg.get("vdd"):
            self.cfg["vdd"]["thr"] = float(_thr)
        self.par = dict(self.C.DEF); self.par.update(self.cfg.get("par") or {})
        self.pdm = self.C.PDM(self.cfg["pdm"])
        self.vcal = self.cfg["val_calib"]
        vc = self.cfg["vdr"]
        self.vdr_cfg = vc
        self.vdr = _ort_session(_H / vc["file"])
        o = self.vdr.run(None, {"rgb": np.zeros((256, 256, 3), np.float32)})          # load check + warm-up
        if tuple(o[0].shape) != (64, 64) or tuple(o[3].shape) != (128, 128):
            raise RuntimeError("vdr outputs %s" % [x.shape for x in o])
        self.vdd = None; self.vdd_cfg = self.cfg.get("vdd")
        if self.vdd_cfg:
            self.VP = _load(self.vdd_cfg["post"])
            self.vdd = _ort_session(_H / self.vdd_cfg["file"])
            o = self.vdd.run(None, {"depth": np.ones((256, 256, 1), np.float32)})
            if tuple(o[0].shape) != (64, 64) or tuple(o[2].shape) != (128, 128):
                raise RuntimeError("vdd outputs %s" % [x.shape for x in o])
        fc = self.cfg.get("fov")
        self.FE = _load(fc["module"]) if fc else None
        self.fov_cfg = fc
        self.sensor = None          # HARNESS ONLY: an object with rgb(stack, st, z64) / vdd(stack, st, z64); never set on chain
        self.route = "a3"
        self.reset()

    # ------------------------------------------------------------------------------------------------ episode
    def reset(self):
        self.fl = self.SF(); self.fl.reset()
        vdd_on = (self.vdd is not None) or (self.sensor is not None and getattr(self.sensor, "vdd_on", False))
        self.brain = self.C.Brain(self.par, self.pdm, self.vcal, self.fl, vdd_on)
        self.n_exc = 0; self.safe = None; self.exc_log = []
        self.k = 0
        self.fov = 90.0; self.fov_frames = []; self.fov_job = None; self.fov_log = []; self.fov_done = False
        self.brain.par["tan"] = 1.0
        self.stats = {"vdr_runs": 0, "vdr_fires": 0, "vdr_pad": 0, "vdr_far": 0, "frames_missing": 0, "vdd_runs": 0}

    # ------------------------------------------------------------------------------------------------ perception
    def _vdr_hit(self, rgb, dep, st):
        """vdr02_full on the delivered frame: peak >= thr -> localised hit dict, else None."""
        vc = self.vdr_cfg
        x = np.asarray(rgb, np.float32).reshape(256, 256, 3)
        h, o, c, m = self.vdr.run(None, {"rgb": x})
        self.stats["vdr_runs"] += 1
        i = int(np.argmax(h)); gy, gx = divmod(i, 64)
        pk = float(h[gy, gx])
        if pk < vc["thr"]:
            return None
        self.stats["vdr_fires"] += 1
        cx = 4.0 * (gx + 0.5 + float(o[0, gy, gx])); cy = 4.0 * (gy + 0.5 + float(o[1, gy, gx]))
        plp = float(sum(c[j, gy, gx] for j in vc["lp_ch"]))
        w = int(vc["win"])
        M = np.repeat(np.repeat(m, 2, 0), 2, 1)                   # mask logits at 256^2 (nearest), as ana_vdr.py
        py = min(max(gy * 4 + 2, w), 255 - w); px = min(max(gx * 4 + 2, w), 255 - w)
        L = M[py - w:py + w + 1, px - w:px + w + 1]
        D = dep[py - w:py + w + 1, px - w:px + w + 1]
        pm = L > vc["mask_logit"]
        if not pm.any():
            pm = L >= L.max()
        zc = float(np.percentile(D[pm], vc["pct"]))
        if zc >= vc["zmax"]:
            self.stats["vdr_far"] += 1
            return None
        pos = st[0:3].astype(np.float64); rpy = st[3:6].astype(np.float64)
        cam, fw, up, rt = self.C.cam_of(pos, rpy)
        tan = self.brain.par["tan"]
        u = cx / 128.0 - 1.0; v = 1.0 - cy / 128.0
        P = cam + fw * zc + rt * (u * tan * zc) + up * (v * tan * zc)
        # victim top: back-project the mask pixels near the localised depth (background bleeding dropped)
        W = max(w, min(64, int(round(3.0 * 128.0 / max(zc, 1.0)))))
        y0, y1 = max(0, int(cy) - W), min(256, int(cy) + W + 1); x0, x1 = max(0, int(cx) - W), min(256, int(cx) + W + 1)
        L2 = M[y0:y1, x0:x1]; D2 = dep[y0:y1, x0:x1]
        sel = (L2 > vc["mask_logit"]) & (D2 >= zc - 1.0) & (D2 <= zc + 1.5)
        top = None
        n_ms = int(sel.sum())
        if n_ms >= 3:
            rr, cc = np.nonzero(sel)
            zz = D2[sel].astype(np.float64)
            uu = ((cc + x0) + 0.5) / 128.0 - 1.0; vv = 1.0 - ((rr + y0) + 0.5) / 128.0
            Z = cam[2] + fw[2] * zz + rt[2] * (uu * tan * zz) + up[2] * (vv * tan * zz)
            top = float(np.percentile(Z, 95))
        if top is None or top < P[2]:
            top = float(P[2]) + (0.4 if plp >= 0.5 else 0.9)
        pad = self.brain.pad
        if math.hypot(P[0] - pad[0], P[1] - pad[1]) < self.par["pad_r"] and abs(P[2] - pad[2]) < 3.0:
            self.stats["vdr_pad"] += 1
            return None
        r = float(np.linalg.norm(P - cam))
        return dict(x=float(P[0]), y=float(P[1]), top=top, r=r, lp=bool(plp >= 0.5), peak=pk, zc=zc, n_ms=n_ms)

    def _vdd_fires(self, dep, st):
        """vdd01 (N1) on the delivered depth + N1's numpy post-process (3x3 NMS, <= 4 peaks, mask localisation) at the
        episode FOV estimate. Returns [(x, y, top), ...]."""
        vc = self.vdd_cfg
        h, o, m = self.vdd.run(None, {"depth": np.asarray(dep, np.float32).reshape(256, 256, 1)})
        self.stats["vdd_runs"] += 1
        fs = self.VP.fires(h, m, dep, st[0:3].astype(np.float64), st[3:6].astype(np.float64), vc["thr"], fov_deg=self.fov)
        out = []
        pad = self.brain.pad
        for f in fs:
            if math.hypot(f["x"] - pad[0], f["y"] - pad[1]) < self.par["pad_r"] and abs(f["top"] - pad[2]) < 3.0:
                self.stats["vdd_pad"] = self.stats.get("vdd_pad", 0) + 1
                continue
            out.append((float(f["x"]), float(f["y"]), float(f["top"])))
        self.stats["vdd_fires"] = self.stats.get("vdd_fires", 0) + len(out)
        return out

    def _fov_update(self, dep, st):
        """per-episode camera FOV (N1 fov_est.estimate_joint): depth frames every 5th tick after take-off (fixed
        cadence); at the checkpoints (20/40/80 frames) the joint cost over the lag-10 pairs of the first N frames is
        accumulated ONE PAIR PER TICK; argmin + parabola -> self.fov and the tan every back-projection uses (VDR, VDD,
        map, visibility)."""
        fc = self.fov_cfg; FE = self.FE
        B = self.brain
        if B.stage != "takeoff" and self.k % 5 == 0 and len(self.fov_frames) < fc["checkpoints"][-1] and not self.fov_done:
            self.fov_frames.append((np.array(dep, np.float32), st[0:3].astype(np.float64).copy(), st[3:6].astype(np.float64).copy()))
            N = len(self.fov_frames)
            if N in fc["checkpoints"] and self.fov_job is None:
                fovs = np.arange(fc["lo"], fc["hi"] + 0.05, 0.1)
                self.fov_job = dict(i=0, N=N, lag=fc["lag"], n=0, fovs=fovs, tot=np.zeros(len(fovs)))
                return
        J = self.fov_job
        if J is None:
            return
        while J["i"] < J["N"] - J["lag"] and J["n"] < fc["max_pairs"]:
            i = J["i"]; J["i"] += max(1, J["lag"] // 2)
            A_ = self.fov_frames[i]; B_ = self.fov_frames[i + J["lag"]]
            mv = float(np.linalg.norm(A_[1] - B_[1]))
            dyaw = abs(math.degrees((B_[2][2] - A_[2][2] + math.pi) % (2 * math.pi) - math.pi))
            if mv < fc["min_move"] and dyaw < fc["min_rot"]:
                continue
            c, _ = FE.pair_cost(A_[0], (A_[1], A_[2]), B_[0], (B_[1], B_[2]), J["fovs"])
            if c is not None and np.isfinite(c).all():
                J["tot"] += c; J["n"] += 1
            return                                            # one pair per tick
        if J["n"] > 0:
            tot = J["tot"]; fovs = J["fovs"]; k = int(np.argmin(tot)); f = float(fovs[k])
            if 0 < k < len(fovs) - 1:
                den = tot[k - 1] - 2 * tot[k] + tot[k + 1]
                if den > 0:
                    f = float(fovs[k] + 0.5 * (tot[k - 1] - tot[k + 1]) / den * (fovs[1] - fovs[0]))
            if fc["lo"] <= f <= fc["hi"]:
                self.fov = f; B.par["tan"] = math.tan(math.radians(f) / 2.0)
            self.fov_log.append((self.k, round(f, 3), J["n"]))
        self.fov_job = None
        if J["N"] >= fc["checkpoints"][-1]:
            self.fov_frames = []; self.fov_done = True        # final checkpoint: free the frames

    # ------------------------------------------------------------------------------------------------ act
    def act(self, observation):
        self.k += 1
        try:
            st = np.asarray(observation["state"], np.float64).reshape(-1)
        except Exception:
            return np.zeros(6, np.float32)
        if self.safe is not None:
            return self._safe_act(observation, st)
        try:
            return self._act(observation, st)
        except Exception as e:
            self.n_exc += 1
            if len(self.exc_log) < 5:
                import traceback
                self.exc_log.append((self.k, repr(e), traceback.format_exc(limit=4)[-600:]))
            if self.n_exc >= 3:
                self.safe = {"k0": self.k, "phase": "climb"}
                self.route = "a3safe"
            return self._hover(st)

    def _act(self, obs, st):
        depth = obs["depth"]
        dep = np.asarray(depth, np.float32).reshape(256, 256)
        z64 = dep[::4, ::4].astype(np.float64) * (self.C.DMAX - self.C.DMIN) + self.C.DMIN
        self.fl.observe(st, dep)
        B = self.brain
        if B.pending is not None:
            hit = None; delivered = True
            if self.sensor is not None:
                hit = self.sensor.rgb(self, st, z64)
            else:
                rgb = obs.get("rgb")
                if rgb is None or float(np.max(rgb)) <= 0.0:
                    delivered = False
                else:
                    hit = self._vdr_hit(rgb, dep * (self.C.DMAX - self.C.DMIN) + self.C.DMIN, st)
            if delivered:
                _st = 256 // int(self.par["map_n"])
                B.frame_result(hit, st, z64, lambda: dep[::_st, ::_st].astype(np.float64) * (self.C.DMAX - self.C.DMIN) + self.C.DMIN)
            else:
                self.stats["frames_missing"] += 1; B.pending = None
        if self.FE is not None and self.sensor is None:
            self._fov_update(dep, st)
        vdd_fn = None
        if self.sensor is not None and getattr(self.sensor, "vdd_on", False):
            vdd_fn = lambda: self.sensor.vdd(self, st, z64)            # noqa: E731
        elif self.vdd is not None:
            vdd_fn = lambda: self._vdd_fires(dep, st)                   # noqa: E731
        step = 256 // int(self.par["map_n"])
        zmap_fn = lambda: dep[::step, ::step].astype(np.float64) * (self.C.DMAX - self.C.DMIN) + self.C.DMIN   # noqa: E731
        a, purpose = B.tick(st, z64, vdd_fn, zmap_fn)
        self.route = "a3+" + B.stage
        a = self._tilt_guard(np.asarray(a, np.float32).reshape(6), st)
        if not np.all(np.isfinite(a)):
            raise FloatingPointError("non-finite action")
        return a

    def _tilt_guard(self, a, st):
        """attitude watchdog: above par['tilt_guard'] deg of roll/pitch (normal flight peaks at ~33 deg) the command
        becomes 'keep the current velocity (|vz| <= 1), no yaw error' so the attitude loop regains authority; the
        flight layer's shaper is re-seeded with it. A frame request on that tick is kept."""
        tg = self.par.get("tilt_guard", 0.0)
        if not tg:
            return a
        if math.degrees(max(abs(float(st[3])), abs(float(st[4])))) <= tg:
            return a
        v = np.array([float(st[6]), float(st[7]), float(np.clip(st[8], -1.0, 1.0))])
        self.stats["tilt_guard"] = self.stats.get("tilt_guard", 0) + 1
        try:
            self.fl.sh.cmd = v.copy()
        except Exception:
            pass
        b = self.FL.vel_action(v, float(st[5]), float(a[5]))
        return np.asarray(b, np.float32)

    def _hover(self, st):
        try:
            v = st[6:9]
            c = -np.clip(v, -1.0, 1.0) * 0.5
            n = float(np.linalg.norm(c))
            a = np.zeros(6, np.float32)
            if n > 1e-6:
                a[0:3] = (c / n).astype(np.float32); a[3] = np.float32(min(1.0, n / 3.0))
            a[4] = np.float32(math.atan2(math.sin(st[5]), math.cos(st[5])) / math.pi)
            return a
        except Exception:
            return np.zeros(6, np.float32)

    # ------------------------------------------------------------------------------------------------ safe mode
    def _safe_act(self, obs, st):
        """numpy only (flight.py + the last belief), never raises: climb to terrain + 6 m; cruise at AGL 6 m on
        flight.py's goto rule (grid corridor terrain rule) to the heaviest cluster of the last belief (the clue centre
        if there is none); inside 12 m a blind flight.py approach to its surface + class height."""
        try:
            S = self.safe
            fl = self.fl
            fl.observe(st, obs["depth"])
            if "tgt" not in S:
                tx = None; hz = 1.33
                try:
                    B = self.brain
                    if B.B is not None:
                        cl = B._clusters()
                        if cl:
                            tx, ty = float(cl[0][0]), float(cl[0][1])
                except Exception:
                    tx = None
                if tx is None:
                    c = st[0:2] + st[163:165]
                    tx, ty = float(c[0]), float(c[1])
                S["tgt"] = (tx, ty); S["hz"] = hz
            agl = float(st[162]) * 20.0
            pos = st[0:3]; vel = st[6:9]; yaw_now = float(st[5])
            if S["phase"] == "climb":
                if agl >= 6.0 or (self.k - S["k0"]) * 0.02 > 3.0:
                    S["phase"] = "go"
                else:
                    return self.fl.cmd_takeoff()
            tx, ty = S["tgt"]
            e = np.array([tx - pos[0], ty - pos[1]]); dh = float(np.linalg.norm(e))
            if S["phase"] == "go" and dh > 12.0:
                tgt_hd = math.atan2(e[1], e[0])
                a = fl._goto(pos, vel, yaw_now, fl.ground, np.array([tx, ty]), dh, e, tgt_hd, 6.0)
                a[5] = 0.0
                return np.asarray(a, np.float32)
            if S["phase"] == "go":
                g = fl.grid.disc_max(np.array([tx, ty]), 1.0) if fl.grid is not None else -1e9
                if g < -1e8:
                    g = (fl.ground if fl.ground is not None else float(pos[2]) - 6.0)
                S["vtop"] = float(g) + S["hz"]; S["phase"] = "term"
            a = fl.cmd_approach(tx, ty, S["vtop"])
            a[5] = 0.0
            return self._tilt_guard(np.asarray(a, np.float32), st)
        except Exception:
            return self._hover(st)
