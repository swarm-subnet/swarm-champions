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
@dataclass
class PadProposal:
    centre: np.ndarray            
    height: float
    extent: float
    flatness: float
    n_px: int
    rng: float
    score: float
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
        self.debug = bool(os.environ.get("PAD_DUMP"))
        self.last_debug = []
    def _forward(self, x: torch.Tensor):
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
        for j in order:
            q, score = keep[int(j)], float(scores[int(j)])
            if score < c.score_thr:
                continue
            if any(float(np.hypot(q["p3"][0] - p.centre[0], q["p3"][1] - p.centre[1]))
                   < c.merge_m for p in props):
                continue
            props.append(PadProposal(centre=q["p3"], height=0.0, extent=q["extent"],
                                     flatness=q["flat"], n_px=int(q["n_px"]),
                                     rng=q["rng"], score=score))
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
        pk = (torch.nn.functional.max_pool2d(heat[None, None], k, 1, k // 2)[0, 0] == heat)
        pk &= heat >= c.heat_thr
        pk = pk.numpy() & valid
        cand = []
        for v, u in zip(*np.nonzero(pk)):
            cand.append(dict(u=u + float(off[0, v, u]), v=v + float(off[1, v, u]),
                             ui=u, vi=v, prob=float(heat[v, u]), src=1,
                             n_px=int(mask_p[v, u] > c.mask_thr)))
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
        return keep
    FEATURES = ("prob", "n_px", "rng", "spread", "extent", "flat", "src",
                "px_ratio", "height", "normal_z")
    def _features(self, cands, chan) -> np.ndarray:
        chan = chan.detach().cpu()
        ch = {c: k for k, c in enumerate(self.geom.channels)}
        rows = []
        for q in cands:
            vi, ui = int(q["vi"]), int(q["ui"])
            exp_px = max(1.0, (77.0 / max(q["rng"], 1e-3)) ** 2 * math.pi / 4.0)
            rows.append([q["prob"], q["n_px"], q["rng"], q["spread"], q["extent"],
                         q["flat"], q["src"], q["n_px"] / exp_px,
                         float(chan[ch["height"], vi, ui]) if "height" in ch else 0.0,
                         float(chan[ch["normal_z"], vi, ui]) if "normal_z" in ch else 0.0])
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
