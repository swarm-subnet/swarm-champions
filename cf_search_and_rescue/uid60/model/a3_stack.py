"""mtn_a3 MOUNTAIN STACK (Rule A route of drone_agent.py): A3Stack(parent) owns the whole flight when z0 >= 13.

Loads (all from this directory, by file path; nothing from c032 is used on this route):
  a3_core.py   the brain: 1 m depth map + plan_sim belief + time-sliced planner + commit/terminal logic
  flight.py    the M1 flight layer (take-off, terrain rule, 3-D approach/hold), driven through a3_core.SearchFlight
  a3_cfg.json  PDM tables, validity calibration, detector settings, parameter overrides
  models/vdr02_full.onnx          RGB victim detector (M3, from scratch): heat 64^2, off, cls 8, mask 128^2
  models/<vdd>.onnx (optional)    depth victim detector, when a3_cfg.json names it
  models/nvm_ft5.onnx (FLAG-GATED, KT_A3_NVM=1 only; default '0' = the cand_w1c stack byte for byte): second depth
                                  detector = the Swarm-SAR team's fine-tune (ft5 ep1) of UID 56's new_victim_mountain.onnx,
                                  decoded like UID 56 (see _nvm_flags / nvm_fire). KT_A3_NVM_THR (0.7) fire bar on the raw
                                  sigmoid; KT_A3_NVM_MODE 'add' (its fire joins vdd_lp's fire list for the existing >= 2 of 5
                                  trigger) | 'alone' (replaces vdd_lp). KT_A3_NVM_DC ('0' default = no DC clamp; '1' = UID 56's
                                  mountain DC D1 clamp, see _nvm_dc_flag). Load or run failures fall back to the vdd_lp path.
Any failure while loading raises out of __init__, so the hook keeps the whole flight on c032 (the NVM load never raises).
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


# ---------------------------------------------------------------------------------------------------- NVM (flag-gated)
NVM_FILE = "models/nvm_ft5.onnx"


def _nvm_file():
    """KT_A3_NVM_FILE (w4_mountain): the NVM head file. Unset / '' (default) -> models/nvm_ft5.onnx (cand_w3). Otherwise a plain
    '<name>.onnx' that exists in models/ (e.g. the sar_trn mountain fine-tune); anything else keeps the default. Never raises."""
    try:
        v = os.environ.get("KT_A3_NVM_FILE", "nvm_mft.onnx").strip()
        if v and "/" not in v and "\\" not in v and v.endswith(".onnx") and (_H / "models" / v).is_file():
            return "models/" + v
    except Exception:
        pass
    return NVM_FILE
NVM_DC_MINR, NVM_DC_MINP = 8.0, 0.3          # UID 56 DC D1 (SWSAR_DC_MINR, p >= 0.3), mountain only


def _nvm_flags():
    """KT_A3_NVM ('0' default = off), KT_A3_NVM_THR (0.7), KT_A3_NVM_MODE ('add' | 'alone'). Never raises: a bad value
    turns the NVM off (on=False) and the stack stays on the vdd_lp path."""
    try:
        on = os.environ.get("KT_A3_NVM", "1").strip() == "1"
        thr = float(os.environ.get("KT_A3_NVM_THR", "0.78").strip() or "0.78")
        mode = os.environ.get("KT_A3_NVM_MODE", "add").strip() or "add"
        if mode not in ("add", "alone") or not (0.0 < thr < 1.0):
            return False, 0.7, "add"
        return on, thr, mode
    except Exception:
        return False, 0.7, "add"


def _nvm_dc_flag():
    """KT_A3_NVM_DC: '0' (default) = no DC clamp; '1' = UID 56's mountain DC D1 clamp. Never raises (-> off).
    Default off: on A3 frames the clamp pulls long-range fires onto terrain in front of the victim. A3 dumps (82 flights,
    170 clamp-eligible fires at p >= 0.7): xy error > 2 m 5 unclamped -> 17 clamped (p90 1.17 -> 1.95 m); e1 kdump replay
    (3,499 mountain fires): 2-5 m errors 73 unclamped -> 162 clamped, <= 2 m 3,406 -> 3,316. Review r1, 2026-09-29."""
    try:
        return os.environ.get("KT_A3_NVM_DC", "0").strip() == "1"
    except Exception:
        return False


def _a3t_flags():
    """A3T (tools/patches/a3t.py): KT_A3T ('0' default = off -> None: every path is w4dev's). '1': the far-transit
    vertical law (a3_core SearchFlight.cmd_far) in the brain's transit stage, pad-to-clue distance >= KT_A3T_D m and
    the drone > KT_A3T_R m from the clue. KT_A3T_R (60), KT_A3T_D (85), KT_A3T_V (3.0), KT_A3T_B (3.0), KT_A3T_L (28), KT_A3T_UP (1.0), KT_A3T_DN (1.0),
    KT_A3T_VZ (1).
    Read once per episode (reset). Never raises: a bad value -> None (off)."""
    try:
        if os.environ.get("KT_A3T", "1").strip() != "1":
            return None
        def f(k, d):
            v = os.environ.get(k, "").strip()
            return float(v) if v else float(d)
        c = dict(r=f("KT_A3T_R", 60.0), d=f("KT_A3T_D", 85.0), v=f("KT_A3T_V", 3.0), b=f("KT_A3T_B", 3.0), look=f("KT_A3T_L", 28.0),
                 up=f("KT_A3T_UP", 1.0), dn=f("KT_A3T_DN", 1.0), vz=int(f("KT_A3T_VZ", 1)))
        if not (0.0 <= c["r"] <= 300.0 and 0.0 <= c["d"] <= 300.0 and 1.0 <= c["v"] <= 3.0 and 0.0 <= c["b"] <= 10.0 and 16.0 <= c["look"] <= 30.0
                and 0.1 <= c["up"] <= 2.5 and 0.1 <= c["dn"] <= 2.0 and c["vz"] in (0, 1)):
            return None
        return c
    except Exception:
        return None


def nvm_fire(o, dep01, pos, rpy, tan, thr, cam_of, dc_on=False):
    """Decode of one nvm_ft5 output row, as UID 56 decodes its NEWDET mountain head (cf_swarm_sar uid56
    drone_agent.py: _depth_heads_nd U:1949-1960 with NEWDET_T = 0 -> raw p; update() U:2024-2034 = DC D1 clamp, here only
    with dc_on (KT_A3_NVM_DC=1, default off: see _nvm_dc_flag); measurement_world U:1624-1627):
      p  = sigmoid(o0)                       fire when p >= thr
      u  = o1, v = o2                        image-plane coordinates of the victim AABB centre (u right, v up; the ft5
                                             labels are (c - eye).right / (cz tan(fov/2)), so the ray is scaled by the
                                             EPISODE tan, here A3's FOV estimate; UID 56 uses tan = 1)
      cz = exp(o3)                           range along the camera forward axis to the AABB centre (exp capped at o3 = 6
                                             only against overflow; UID 56 has no cap)
      DC D1 (dc_on only): cz >= 8 and p >= 0.3 and the 5x5 median depth at the (u, v) pixel d_img < 29.5 with
             d_img + 0.6 < cz <= d_img + 4.0  ->  cz = d_img + 0.25
      P  = eye + cz (fwd + u tan right + v tan up), eye = pos + 0.13 fwd + 0.05 up (a3_core.cam_of == UID 56's camera)
    top (victim bbox top; the terminal hovers 3 m above it): see _nvm_top. Returns a dict or None (below thr / not finite)."""
    o = np.asarray(o, np.float64).reshape(-1)
    if o.shape[0] < 5 or not np.all(np.isfinite(o[0:4])):
        return None
    p = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, float(o[0])))))
    if not p >= thr:
        return None
    u, v = float(o[1]), float(o[2])
    cz = math.exp(min(float(o[3]), 6.0))
    d2 = np.asarray(dep01, np.float32).reshape(256, 256)
    px = min(255, max(0, int(round((u + 1.0) * 0.5 * 256 - 0.5))))
    py = min(255, max(0, int(round((1.0 - v) * 0.5 * 256 - 0.5))))
    d_img = float(np.median(d2[max(0, py - 2):py + 3, max(0, px - 2):px + 3])) * 29.5 + 0.5
    dc = False
    if dc_on and cz >= NVM_DC_MINR and p >= NVM_DC_MINP and d_img < 30.0 - 0.5 and (d_img + 0.6) < cz <= (d_img + 4.0):
        cz = d_img + 0.25; dc = True
    cam, fw, up, rt = cam_of(np.asarray(pos, np.float64), np.asarray(rpy, np.float64))
    P = cam + cz * (fw + (u * tan) * rt + (v * tan) * up)
    if not np.all(np.isfinite(P)):
        return None
    top = _nvm_top(P, d2, px, py, cz, d_img, cam, fw, up, rt, tan)
    return dict(x=float(P[0]), y=float(P[1]), top=top, p=p, cz=cz, u=u, v=v, dc=dc)


NVM_TOP_C, NVM_TOP_LO, NVM_TOP_HI, NVM_TOP_BAND, NVM_TOP_W = 0.45, -0.2, 1.4, 0.6, 0.9


def _nvm_top(P, d2, px, py, cz, d_img, cam, fw, up, rt, tan):
    """victim bbox top for an NVM fire: max(centre_z + 0.45, depth-patch top), where the depth-patch top is the 95th pct z
    of the back-projected pixels within +-0.9 m (at range cz) of the predicted pixel whose depth lies within +-0.6 m of
    d_img (the DC reference), used only inside [centre_z - 0.2, centre_z + 1.4] (>= 3 pixels).
    Why not o4: ft5 trains p/u/v/log_cz only and UID 56 takes the height from its champion head (NEWDET_H = 'champ');
    on 1,306 on-target fires in the 82 A3 dump flights o4 has median 0.00 for lying AND standing victims.
    Error vs the true top on those fires (|e| <= 1 m, the hover band's half width): centre+0.45 0.972 (standing 0.947),
    patch-else-const 0.962 (standing 0.871), max(both) 0.974 (standing 0.957, p90 0.81 m) -> max(both)."""
    zc = float(P[2]) + NVM_TOP_C
    try:
        W = int(min(32, max(2, round(NVM_TOP_W * 128.0 / (max(cz, 1.0) * max(tan, 0.3))))))
        y0, y1, x0, x1 = max(0, py - W), min(256, py + W + 1), max(0, px - W), min(256, px + W + 1)
        D = d2[y0:y1, x0:x1].astype(np.float64) * 29.5 + 0.5
        sel = (D >= d_img - NVM_TOP_BAND) & (D <= d_img + NVM_TOP_BAND) & (D < 29.95)
        if int(sel.sum()) < 3:
            return zc
        rr, cc = np.nonzero(sel)
        zz = D[sel]
        uu = ((cc + x0) + 0.5) / 128.0 - 1.0; vv = 1.0 - ((rr + y0) + 0.5) / 128.0
        Z = cam[2] + fw[2] * zz + rt[2] * (uu * tan * zz) + up[2] * (vv * tan * zz)
        tb = float(np.percentile(Z, 95))
        if float(P[2]) + NVM_TOP_LO <= tb <= float(P[2]) + NVM_TOP_HI:
            return max(zc, tb)
    except Exception:
        pass
    return zc


def _pad_flags():
    """a3pad: KT_A3PAD / KT_A3PAD_RV / KT_A3PAD_RD / KT_A3PAD_RN / KT_A3PAD_LOG, read once per episode (A3Stack.reset).
    None when KT_A3PAD and KT_A3PAD_LOG are both unset / '0' (every pad test is then the w4dev one byte for byte).
    Never raises (-> None)."""
    try:
        on = os.environ.get("KT_A3PAD", "1").strip() == "1"
        log = os.environ.get("KT_A3PAD_LOG", "0").strip() == "1"
        if not (on or log):
            return None

        def _r(name, d):
            try:
                v = float(os.environ.get(name, "").strip() or d)
                return v if 0.0 <= v <= 3.0 else d
            except Exception:
                return d
        return dict(on=on, log=log, rv=_r("KT_A3PAD_RV", 1.0), rd=_r("KT_A3PAD_RD", 1.0), rn=_r("KT_A3PAD_RN", 3.0))
    except Exception:
        return None


def _mfar_flags():
    """mfar_r7 (tools/patches/mfar_r7.py): KT_MFAR ('0' default = off -> None: every path is w8dev's). '1': far clues
    (tick-1 pad-to-clue distance >= KT_MFAR_D, default 85 m) get the calibrated spawner prior in Belief.set_val: the in-range
    candidate counts (f1, f2) scaled by KT_MFAR_S1 (0.8) and the fallback 'closest passing candidate' decay by KT_MFAR_S2 (0.6).
    Read once per episode (reset). Never raises: a bad value -> None (off)."""
    try:
        if os.environ.get("KT_MFAR", "1").strip() != "1":
            return None

        def f(k, d):
            v = os.environ.get(k, "").strip()
            return float(v) if v else float(d)
        c = dict(d=f("KT_MFAR_D", 85.0), s1=f("KT_MFAR_S1", 0.8), s2=f("KT_MFAR_S2", 0.6))
        if not (0.0 <= c["d"] <= 300.0 and 0.05 <= c["s1"] <= 1.0 and 0.05 <= c["s2"] <= 1.0):
            return None
        return c
    except Exception:
        return None


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
        if os.environ.get("KT_X_M39", "1").strip() != "1" and vc.get("file") == "models/vdr_m39.onnx":   # w13dev: UID 112 vdr_m39 off
            vc = dict(vc); vc["file"] = "models/vdr02_full.onnx"; self.cfg["vdr"] = vc             # -> UID 86 / cand_w6 vdr02_full
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
        # KT_A3_NVM (flag-gated second depth detector). Off (default): self.nvm stays None and nothing below the flag
        # read runs, so every path is the cand_w1c one. On: any load / warm-up failure leaves self.nvm None (vdd_lp only).
        self.nvm = None; self.nvm_in = None; self.nvm_err = None; self.nvm_dcon = False
        self.nvm_on, self.nvm_thr, self.nvm_mode = _nvm_flags()
        if self.nvm_on:
            self.nvm_dcon = _nvm_dc_flag()
            try:
                self.nvm_file = _nvm_file()
                s = _ort_session(_H / self.nvm_file)
                names = [i.name for i in s.get_inputs()]
                o = s.run(None, {names[0]: np.ones((256, 256, 1), np.float32), names[1]: np.zeros(165, np.float32)})
                if tuple(np.asarray(o[0]).shape) != (1, 5):
                    raise RuntimeError("nvm outputs %s" % [np.asarray(x).shape for x in o])
                if self.nvm_mode == "alone" or self.vdd is not None:
                    self.nvm, self.nvm_in = s, names
            except Exception as e:
                self.nvm = None; self.nvm_err = repr(e)[:200]
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
        if getattr(self, "nvm", None) is not None:
            vdd_on = True
        self.brain = self.C.Brain(self.par, self.pdm, self.vcal, self.fl, vdd_on)
        self.n_exc = 0; self.safe = None; self.exc_log = []
        self.k = 0
        self.fov = 90.0; self.fov_frames = []; self.fov_job = None; self.fov_log = []; self.fov_done = False
        self.brain.par["tan"] = 1.0
        self.stats = {"vdr_runs": 0, "vdr_fires": 0, "vdr_pad": 0, "vdr_far": 0, "frames_missing": 0, "vdd_runs": 0}
        if getattr(self, "nvm_on", False):          # KT_A3_NVM=1 only (flags off: the stats dict is cand_w1c's)
            self.stats.update(nvm_runs=0, nvm_fires=0, nvm_pad=0, nvm_dc=0, nvm_exc=0, nvm_ms=0.0, nvm_ms_max=0.0,
                              nvm_loaded=int(self.nvm is not None), nvm_mode=self.nvm_mode, nvm_thr=self.nvm_thr,
                              nvm_dc_on=int(getattr(self, "nvm_dcon", False)),
                              nvm_ev=[])     # first 16 fires: [tick, x, y, top, p, cz]
            if getattr(self, "nvm_file", NVM_FILE) != NVM_FILE:   # KT_A3_NVM_FILE set (w4_mountain)
                self.stats["nvm_file"] = self.nvm_file
        self.padf = _pad_flags()   # a3pad: None = w4dev pad tests
        self.a3t = _a3t_flags()                    # A3T: None (default) = off, nothing below runs
        if self.a3t is not None:
            self.brain.a3t = self.a3t
            self.stats.update(a3t_n=0, a3t_t1=None, a3t_exc=0, a3t_r=self.a3t["r"], a3t_d=self.a3t["d"])
        self.mfar = _mfar_flags()                  # mfar_r7: None (default) = off, nothing below runs
        if self.mfar is not None:
            self.brain.mfar = self.mfar
            self.stats.update(mfar_on=0, mfar_d=self.mfar["d"], mfar_s1=self.mfar["s1"], mfar_s2=self.mfar["s2"])

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
        if math.hypot(P[0] - pad[0], P[1] - pad[1]) < self.par["pad_r"] and abs(P[2] - pad[2]) < 3.0 and \
                (self.padf is None or self._pad_zone("vdr", float(P[0]), float(P[1]), float(P[2]), pk)):   # a3pad
            self.stats["vdr_pad"] += 1
            return None
        r = float(np.linalg.norm(P - cam))
        return dict(x=float(P[0]), y=float(P[1]), top=top, r=r, lp=bool(plp >= 0.5), peak=pk, zc=zc, n_ms=n_ms)

    def _pad_zone(self, det, x, y, z, peak):
        """a3pad (self.padf set): a fire inside the old pad zone (r < pad_r, |z - pad z| < 3). True = drop it (the w4dev
        behaviour), False = keep it (KT_A3PAD=1 and r >= the detector's radius: rv 'vdr', rd 'vdd', rn 'nvm')."""
        F = self.padf; S = self.stats
        pad = self.brain.pad
        r = math.hypot(x - pad[0], y - pad[1])
        drop = not (F["on"] and r >= F["rv" if det == "vdr" else ("rd" if det == "vdd" else "rn")])
        if not drop:
            S["pad_pass_" + det] = S.get("pad_pass_" + det, 0) + 1
        if F["log"]:
            S["pad_zone"] = S.get("pad_zone", 0) + 1
            ev = S.setdefault("pad_ev", [])
            if len(ev) < 60:
                ev.append([self.k, det, round(float(x - pad[0]), 2), round(float(y - pad[1]), 2), round(float(z - pad[2]), 2),
                           round(float(peak), 3), int(drop)])
        return drop

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
            if math.hypot(f["x"] - pad[0], f["y"] - pad[1]) < self.par["pad_r"] and abs(f["top"] - pad[2]) < 3.0 and \
                    (self.padf is None or self._pad_zone("vdd", f["x"], f["y"], f["top"], f["p"])):   # a3pad
                self.stats["vdd_pad"] = self.stats.get("vdd_pad", 0) + 1
                continue
            out.append((float(f["x"]), float(f["y"]), float(f["top"])))
        self.stats["vdd_fires"] = self.stats.get("vdd_fires", 0) + len(out)
        return out

    def _nvm_fires(self, dep, st):
        """KT_A3_NVM: nvm_ft5 on the delivered depth (state input = zeros with [0:3] pos, [3:6] rpy, as UID 56 and the
        ft5 training feed it) -> [] or [(x, y, top)], with _vdd_fires' pad-proximity rejection."""
        import time as _time
        t0 = _time.perf_counter()
        s = np.zeros(165, np.float32); s[0:3] = st[0:3]; s[3:6] = st[3:6]
        o = self.nvm.run(None, {self.nvm_in[0]: np.asarray(dep, np.float32).reshape(256, 256, 1), self.nvm_in[1]: s})[0]
        ms = 1000.0 * (_time.perf_counter() - t0)
        S = self.stats
        S["nvm_runs"] += 1; S["nvm_ms"] += ms; S["nvm_ms_max"] = max(S["nvm_ms_max"], ms)
        f = nvm_fire(o, dep, st[0:3], st[3:6], float(self.brain.par["tan"]), self.nvm_thr, self.C.cam_of, dc_on=self.nvm_dcon)
        if f is None:
            return []
        pad = self.brain.pad
        if math.hypot(f["x"] - pad[0], f["y"] - pad[1]) < self.par["pad_r"] and abs(f["top"] - pad[2]) < 3.0 and \
                (self.padf is None or self._pad_zone("nvm", f["x"], f["y"], f["top"], f["p"])):   # a3pad
            S["nvm_pad"] += 1
            return []
        S["nvm_fires"] += 1; S["nvm_dc"] += int(f["dc"])
        if len(S["nvm_ev"]) < 16:
            S["nvm_ev"].append([self.k, round(f["x"], 2), round(f["y"], 2), round(f["top"], 2), round(f["p"], 3), round(f["cz"], 2)])
        return [(f["x"], f["y"], f["top"])]

    def _dep_fires(self, dep, st):
        """KT_A3_NVM: the fire list of one depth-detector evaluation (same cadence as VDD: Brain.tick calls it every
        par['vdd_every'] ticks after take-off). 'add': vdd_lp fires + the NVM fire (<= 1), one list, so the existing
        rule (fires from >= 2 of the last 5 evaluations within 2 m) triggers on either detector or on both together.
        'alone': the NVM fire only. Any NVM exception -> the vdd_lp list of the pre-NVM path."""
        alone = self.nvm_mode == "alone"
        base = [] if (alone or self.vdd is None) else self._vdd_fires(dep, st)
        try:
            nf = self._nvm_fires(dep, st)
        except Exception:
            self.stats["nvm_exc"] = self.stats.get("nvm_exc", 0) + 1
            if alone and self.vdd is not None:
                return self._vdd_fires(dep, st)
            return base
        return nf if alone else base + nf

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
        elif self.nvm is not None:                                      # KT_A3_NVM=1 only (None when the flag is off)
            vdd_fn = lambda: self._dep_fires(dep, st)                   # noqa: E731
        elif self.vdd is not None:
            vdd_fn = lambda: self._vdd_fires(dep, st)                   # noqa: E731
        step = 256 // int(self.par["map_n"])
        zmap_fn = lambda: dep[::step, ::step].astype(np.float64) * (self.C.DMAX - self.C.DMIN) + self.C.DMIN   # noqa: E731
        a, purpose = B.tick(st, z64, vdd_fn, zmap_fn)
        self.route = "a3+" + B.stage
        if self.a3t is not None:                   # A3T counters
            self.stats["a3t_n"] = B.a3t_n; self.stats["a3t_t1"] = B.a3t_t1; self.stats["a3t_exc"] = B.fl.a3t_exc
        if self.mfar is not None:                  # mfar_r7 counter
            self.stats["mfar_on"] = int(B.mfar_on)
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
