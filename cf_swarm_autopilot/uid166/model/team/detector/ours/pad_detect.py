from __future__ import annotations
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence
import numpy as np
import torch
from .pad_geom import (DEPTH_MAX_M, DEPTH_MIN_M, FOV_BASE_DEG, GeomChannels,
                      camera_from_pose)
from .pad_model import ModelConfig, PadModel, Z_SCALE
from . import pad_geom as _pg
FAST_CHUNK = 2        # perf "fast": drones per ONNX run


def _maxpool_np(h: np.ndarray, k: int) -> np.ndarray:
    """perf "fast": k x k running max of a 2-D map with -inf padding (= max_pool2d(h, k, 1, k // 2)), separable."""
    r = k // 2
    H, W = h.shape
    p = np.full((H + 2 * r, W), -np.inf, dtype=h.dtype)
    p[r:r + H] = h
    m = p[0:H].copy()
    for s in range(1, k):
        np.maximum(m, p[s:s + H], out=m)
    q = np.full((H, W + 2 * r), -np.inf, dtype=h.dtype)
    q[:, r:r + W] = m
    o = q[:, 0:W].copy()
    for s in range(1, k):
        np.maximum(o, q[:, s:s + W], out=o)
    return o
@dataclass
class PadProposal:
    centre: np.ndarray            
    height: float
    extent: float
    flatness: float
    n_px: int
    rng: float
    score: float
    weak: bool = False            # detv2: weak_thr <= score < the accept threshold (only emitted when weak_thr > 0)
    far: bool = False             # detv2 diag: accepted only through score_thr_far (score < score_thr)
    snap: bool = False            # detv2 diag: the heat peak was re-seated by SNAP (snap_r > 0)
@dataclass
class DetectConfig:
    heat_thr: float = 0.05        
    sources: str = "heat"
    mask_thr: float = 0.5
    min_px: int = 3               
    nms_px: int = 3
    merge_m: float = 1.0          
    max_range_m: float = 28.3     
    min_range_m: float = 0.0
    max_depth_spread_m: float = 1e9
    score_thr: float = 0.95
    # ---- detv2 decode switches (research/detv2/pipeline; all default off = shipped behaviour) ----
    # snap_r > 0: a heat peak whose own pixel has invalid depth (beyond the 20 m cap; the heat target is the projected
    #   pad-top centre, which for grazing views lies just past the slab edge) is kept; depth / features come from the
    #   highest-heat valid pixel within snap_r px, the ray from the peak's sub-pixel position.
    # snap_valid_mask: also re-seat a VALID peak that sits off the mask (mask <= mask_thr) onto the best mask pixel
    #   within snap_r (depth from a surface behind / in front of the pad).
    # score_thr_far >= 0: accept threshold for candidates at 3-D range >= far_rng_m (score_thr below it).
    # weak_thr > 0: candidates with weak_thr <= score < accept threshold are also returned, flagged weak=True.
    snap_r: int = 0
    snap_valid_mask: bool = False
    score_thr_far: float = -1.0
    far_rng_m: float = 20.0
    weak_thr: float = -1.0
class TreeChecker:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.cl, self.cr = z["cl"], z["cr"]
        self.feat, self.thr, self.val = z["feat"], z["thr"], z["val"]
        self.roots = z["roots"]                     
        self.init, self.lr = float(z["init"]), float(z["lr"])
        self.depth = int(z["max_depth"])
        self.names = [str(s) for s in z["feature_names"]]
    def __call__(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        if n == 0:
            return np.zeros(0, np.float32)
        X = np.asarray(X, np.float64)
        node = np.broadcast_to(self.roots, (n, len(self.roots))).copy()
        rows = np.arange(n)[:, None]
        for _ in range(self.depth + 1):
            leaf = self.cl[node] == -1
            if leaf.all():
                break
            go_left = X[rows, self.feat[node]] <= self.thr[node]
            nxt = np.where(go_left, self.cl[node], self.cr[node])
            node = np.where(leaf, node, nxt)
        raw = self.init + self.lr * self.val[node].sum(1)
        return 1.0 / (1.0 + np.exp(-raw))
class PadDetector:
    def __init__(self, ckpt, device: str = "cpu", cfg: Optional[DetectConfig] = None,
                 checker: Optional[str] = None, threads: int = 2,
                 fov_deg: float = FOV_BASE_DEG):
        self.cfg = cfg or DetectConfig()
        if os.environ.get("PAD_SCORE_THR"):        
            self.cfg.score_thr = float(os.environ["PAD_SCORE_THR"])
        self.device = device
        self.fov_deg = float(fov_deg)
        torch.set_num_threads(int(threads))
        ckpt = str(ckpt)
        self.onnx = ckpt.endswith(".onnx")
        if self.onnx:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = int(threads)
            so.inter_op_num_threads = 1
            self.sess = ort.InferenceSession(ckpt, so, providers=["CPUExecutionProvider"])
            meta = self.sess.get_modelmeta().custom_metadata_map
            chans = tuple(meta.get("channels", "").split(","))
        else:
            blob = torch.load(ckpt, map_location=device, weights_only=False)
            chans = tuple(blob.get("channels") or GeomChannels().channels)
            mcfg = ModelConfig(**{**blob.get("cfg", {}), "in_ch": len(chans)})
            self.net = PadModel(mcfg)
            sd = blob.get("ema") or blob["model"]
            sd = {k.replace("module.", ""): v for k, v in sd.items()
                  if not k.startswith("n_averaged")}
            self.net.load_state_dict(sd)
            self.net.eval().to(device)
            with torch.no_grad():
                self.net.fuse_bn()
        self.geom = GeomChannels(channels=chans)
        self.checker = TreeChecker(checker) if checker else None
        self._xfeat = self.needs_xfeat(self.checker)   # detv2 T0-b: False for the shipped 10-feature checker
        self.debug = bool(os.environ.get("PAD_DUMP"))
        self.last_debug = []
    def _forward(self, x: torch.Tensor):
        if self.onnx and _pg.FAST and int(x.shape[0]) > FAST_CHUNK:
            # perf "fast": the net on FAST_CHUNK drones at a time. Each drone's outputs do not depend on the rest of
            # its batch (bit-identical, research/strike_emu/batch_check.py) and small batches stay in cache: an
            # 8-drone batch costs 154 -> 135 ms with 1 thread and 99 -> 81 ms with 2 (chunk_bench.py).
            xn = x.numpy()
            parts = [self.sess.run(None, {"x": xn[i:i + FAST_CHUNK]}) for i in range(0, len(xn), FAST_CHUNK)]
            names = [t.name for t in self.sess.get_outputs()]
            return {n: torch.from_numpy(np.concatenate([p[j] for p in parts])) for j, n in enumerate(names)}
        if self.onnx:
            o = self.sess.run(None, {"x": x.numpy()})
            names = [t.name for t in self.sess.get_outputs()]
            d = {n: torch.from_numpy(v) for n, v in zip(names, o)}
            return d
        with torch.inference_mode():
            return self.net(x)
    def propose_batch(self, poses, rots, depth_batch) -> List[List[PadProposal]]:
        d = np.asarray(depth_batch, dtype=np.float32)
        if d.ndim == 4:
            d = d[..., 0] if d.shape[-1] == 1 else d[:, 0]
        depth = torch.from_numpy(d)
        pos = torch.as_tensor(np.asarray(poses, np.float32))
        R = torch.as_tensor(np.asarray(rots, np.float32))
        cam = camera_from_pose(pos, R)
        dev = "cpu" if self.onnx else self.device
        x = self.geom(depth.to(dev), cam.to(dev), R.to(dev), self.fov_deg)
        out = self._forward(x)
        return [self._decode(i, x, out, depth, cam, R) for i in range(len(pos))]
    def _rays(self, R: torch.Tensor) -> np.ndarray:
        R = R.detach().to("cpu", torch.float32)
        body = self.geom._body_rays(self.fov_deg, R.device, torch.float32)
        return (body @ R.T).numpy()
    def _decode(self, i, x, out, depth, cam, R) -> List[PadProposal]:
        c = self.cfg
        keep = self.candidates(i, x, out, depth, cam, R)
        props: List[PadProposal] = []
        feats = self._features(keep, x[i]) if keep else np.zeros((0, len(self.FEATURES)))
        if self.debug:
            self.last_debug.append((feats.copy(),
                                    np.array([q["p3"] for q in keep], np.float32)
                                    if keep else np.zeros((0, 3), np.float32)))
        scores = (self.checker(feats) if self.checker is not None
                  else np.array([q["prob"] for q in keep]))
        order = np.argsort(-scores) if len(keep) else []
        weak = []
        for j in order:
            q, score = keep[int(j)], float(scores[int(j)])
            thr = (c.score_thr_far if (c.score_thr_far >= 0.0 and q["rng"] >= c.far_rng_m) else c.score_thr)
            if score < thr:
                if c.weak_thr > 0.0 and score >= c.weak_thr:
                    weak.append(PadProposal(centre=q["p3"], height=0.0, extent=q["extent"], flatness=q["flat"],
                                            n_px=int(q["n_px"]), rng=q["rng"], score=score, weak=True))
                continue
            if any(float(np.hypot(q["p3"][0] - p.centre[0], q["p3"][1] - p.centre[1]))
                   < c.merge_m for p in props):
                continue
            props.append(PadProposal(centre=q["p3"], height=0.0, extent=q["extent"],
                                     flatness=q["flat"], n_px=int(q["n_px"]),
                                     rng=q["rng"], score=score,
                                     far=bool(score < c.score_thr), snap=bool(q.get("snap", False))))
        for w in weak:                    # weak ones never shadow an accepted proposal (same 1 m NMS)
            if not any(float(np.hypot(w.centre[0] - p.centre[0], w.centre[1] - p.centre[1])) < c.merge_m
                       for p in props):
                props.append(w)
        return props
    def candidates(self, i, x, out, depth, cam, R) -> List[dict]:
        from scipy import ndimage
        c = self.cfg
        H = W = depth.shape[-1]
        dep = depth[i].detach().cpu().numpy()
        valid = (dep > 0.0) & (dep < 1.0)
        metres = dep * (DEPTH_MAX_M - DEPTH_MIN_M) + DEPTH_MIN_M
        ray = self._rays(R[i])
        cam_np = cam[i].detach().cpu().numpy().astype(np.float32)
        pts = cam_np[None, None, :] + metres[..., None] * ray          
        rng_m = np.linalg.norm(pts - cam_np[None, None, :], axis=-1)
        heat = torch.sigmoid(out["heat"][i, 0].float().detach().cpu())
        mask_p = torch.sigmoid(out["mask"][i, 0].float().detach().cpu()).numpy()
        zres = out["z"][i, 0].float().detach().cpu().numpy() * Z_SCALE
        off = out["offset"][i].float().detach().cpu().numpy()
        k = c.nms_px
        r = int(c.snap_r)
        if _pg.FAST and k % 2 == 1:
            # perf "fast": the same 3x3 peak test as max_pool2d(heat) == heat, as a separable numpy max
            _h = heat.numpy()
            pk = (_maxpool_np(_h, k) == _h) & (_h >= np.float32(c.heat_thr))
            pk = pk if r > 0 else (pk & valid)
        else:
            pk = (torch.nn.functional.max_pool2d(heat[None, None], k, 1, k // 2)[0, 0] == heat)
            pk &= heat >= c.heat_thr
            pk = pk.numpy() if r > 0 else (pk.numpy() & valid)
        cand = []
        heat_np = heat.numpy() if r > 0 else None
        for v, u in zip(*np.nonzero(pk)):
            vi, ui = v, u
            if r > 0:
                v0, v1, u0, u1 = max(0, v - r), min(H, v + r + 1), max(0, u - r), min(W, u + r + 1)
                if not valid[v, u]:
                    sc = np.where(valid[v0:v1, u0:u1], heat_np[v0:v1, u0:u1], -np.inf)
                    a = np.unravel_index(int(np.argmax(sc)), sc.shape)
                    if not np.isfinite(sc[a]):
                        continue                      # no valid pixel near the peak
                    vi, ui = v0 + int(a[0]), u0 + int(a[1])
                elif c.snap_valid_mask and mask_p[v, u] <= c.mask_thr:
                    mw = np.where(valid[v0:v1, u0:u1], mask_p[v0:v1, u0:u1], -1.0)
                    a = np.unravel_index(int(np.argmax(mw)), mw.shape)
                    if mw[a] > c.mask_thr:
                        vi, ui = v0 + int(a[0]), u0 + int(a[1])
            cand.append(dict(u=u + float(off[0, v, u]), v=v + float(off[1, v, u]),
                             ui=ui, vi=vi, prob=float(heat[v, u]), src=1,
                             n_px=int(mask_p[vi, ui] > c.mask_thr)))
            if r > 0 and (vi != v or ui != u):
                cand[-1]["snap"] = True               # detv2 diag (PadProposal.snap)
        m = (mask_p > c.mask_thr) & valid
        lab = None
        if m.any():
            lab, n = ndimage.label(m)
        if lab is not None and "mask" not in c.sources:
            for q in cand:
                j = int(lab[q["vi"], q["ui"]])
                if j:
                    vv, uu = np.nonzero(lab == j)
                    q["pix"] = (vv, uu)
                    q["n_px"] = int(len(vv))
        if lab is not None and "mask" in c.sources:
            for j, sl in enumerate(ndimage.find_objects(lab), start=1):
                if sl is None:
                    continue
                sub = lab[sl] == j
                npx = int(sub.sum())
                if npx < c.min_px:
                    continue
                vv, uu = np.nonzero(sub)
                vv = vv + sl[0].start
                uu = uu + sl[1].start
                w = mask_p[vv, uu]
                cu = float((uu * w).sum() / w.sum())
                cv = float((vv * w).sum() / w.sum())
                ui, vi = int(round(cu)), int(round(cv))
                if not (0 <= ui < W and 0 <= vi < H and valid[vi, ui]):
                    ui, vi = int(uu[0]), int(vv[0])
                cand.append(dict(u=cu, v=cv, ui=ui, vi=vi, prob=float(w.max()),
                                 src=0, n_px=npx, pix=(vv, uu)))
        keep = []
        for q in cand:
            ui, vi = int(q["ui"]), int(q["vi"])
            if not valid[vi, ui]:
                continue
            r = float(rng_m[vi, ui])
            if not (c.min_range_m <= r <= c.max_range_m):
                continue
            t = math.tan(math.radians(self.fov_deg) / 2.0)
            a = (2.0 * q["u"] / W - 1.0) * t
            b = (1.0 - 2.0 * (q["v"] + 1.0) / H) * t
            body = np.array([1.0, -a, b])
            world_ray = (R[i].detach().cpu().numpy().astype(np.float32)
                         @ body.astype(np.float32))
            p3 = cam_np + float(metres[vi, ui]) * world_ray
            p3[2] += float(zres[vi, ui])
            pix = q.get("pix")
            if pix is not None and len(pix[0]) >= 3:
                pv, pu = pix
                dsp = float(np.percentile(rng_m[pv, pu], 90) - np.percentile(rng_m[pv, pu], 10))
                if dsp > c.max_depth_spread_m:      
                    continue
                xy = pts[pv, pu, :2]
                ex = float(np.median(np.linalg.norm(xy - np.median(xy, 0), axis=1)) * 2.0)
                fl = float(np.std(pts[pv, pu, 2]))
            else:
                dsp, ex, fl = 0.0, 1.2, 0.0
            keep.append(dict(q, p3=p3, rng=r, spread=dsp, extent=ex, flat=fl))
            if getattr(self, "_xfeat", False):        # detv2 T0-b extra checker inputs (only for a refit checker)
                keep[-1]["zd"] = float(metres[vi, ui])
                keep[-1]["inval5"] = float(1.0 - valid[max(0, vi - 2):vi + 3, max(0, ui - 2):ui + 3].mean())
        return keep
    FEATURES = ("prob", "n_px", "rng", "spread", "extent", "flat", "src",
                "px_ratio", "height", "normal_z")
    # detv2 T0-b (research/detv2/stage2): a refit checker may name extra features after the 10 above; they are
    # computed only when the loaded checker names them (_xfeat), so the shipped checker's input is unchanged.
    #   border_px: distance of the sub-pixel peak (u, v) to the nearest image border, px (labels.cand_rows)
    #   zd:        planar depth (m) at the candidate's depth pixel (vi, ui)
    #   inval5:    invalid-depth share of the 5x5 window around (vi, ui)
    XFEATURES = ("border_px", "zd", "inval5")
    @classmethod
    def needs_xfeat(cls, checker) -> bool:
        return bool(checker is not None and len(getattr(checker, "names", ())) > len(cls.FEATURES))
    def set_checker(self, path) -> None:
        """detv2 T0-b: give this detector (e.g. a per-slot shallow copy) its own checker npz."""
        self.checker = TreeChecker(path) if path else None
        self._xfeat = self.needs_xfeat(self.checker)
    def _features(self, cands, chan) -> np.ndarray:
        chan = chan.detach().cpu()
        ch = {c: k for k, c in enumerate(self.geom.channels)}
        xf = self.checker.names[len(self.FEATURES):] if getattr(self, "_xfeat", False) else ()
        W = int(chan.shape[-1])
        H = int(chan.shape[-2])
        rows = []
        for q in cands:
            vi, ui = int(q["vi"]), int(q["ui"])
            exp_px = max(1.0, (77.0 / max(q["rng"], 1e-3)) ** 2 * math.pi / 4.0)
            row = [q["prob"], q["n_px"], q["rng"], q["spread"], q["extent"],
                   q["flat"], q["src"], q["n_px"] / exp_px,
                   float(chan[ch["height"], vi, ui]) if "height" in ch else 0.0,
                   float(chan[ch["normal_z"], vi, ui]) if "normal_z" in ch else 0.0]
            for nm in xf:
                if nm == "border_px":
                    row.append(float(min(q["u"], W - 1 - q["u"], q["v"], H - 1 - q["v"])))
                else:
                    row.append(float(q.get(nm, 0.0)))
            rows.append(row)
        return np.asarray(rows, np.float32)
def export_onnx(ckpt: str, out: str, batch: int = 8, channels: Optional[Sequence[str]] = None):
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    chans = tuple(channels or blob.get("channels") or GeomChannels().channels)
    cfg = ModelConfig(**{**blob.get("cfg", {}), "in_ch": len(chans)})
    net = PadModel(cfg)
    sd = blob.get("ema") or blob["model"]
    net.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()
                         if not k.startswith("n_averaged")})
    net.eval()
    with torch.no_grad():
        net.fuse_bn()
    x = torch.rand(batch, len(chans), 128, 128)
    torch.onnx.export(net, x, out, input_names=["x"],
                      output_names=["mask", "heat", "offset", "z"]
                      + (["map"] if cfg.map_head else []),
                      dynamic_axes={"x": {0: "b"}}, opset_version=17, dynamo=False)
    import onnx
    m = onnx.load(out)
    e = m.metadata_props.add()
    e.key, e.value = "channels", ",".join(chans)
    onnx.save(m, out)
    return out
