from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
RES = 128
FOV_BASE_DEG = 90.0          
DEPTH_MIN_M = 0.5            
DEPTH_MAX_M = 20.0           
CAM_FWD = 0.13               
CAM_UP = 0.05                
PAD_DIAM_M = 1.2             
CHANNELS = (
    "depth",        
    "height",       
    "height_fine",  
    "height_slope", 
    "normal_z",     
    "ray_sin",      
    "ray_cos",      
    "z_rel",        
    "log_range",    
    "pad_px",       
    "valid",        
)
def _fmt(v, B, device, dtype):
    if torch.is_tensor(v):
        return v.to(device=device, dtype=dtype).reshape(-1).expand(B)
    return torch.full((B,), float(v), device=device, dtype=dtype)
def rot_from_rpy(rpy: torch.Tensor) -> torch.Tensor:
    rpy = rpy.to(torch.float64)
    r, p_, y = rpy[:, 0], rpy[:, 1], rpy[:, 2]
    cr, sr, cp, sp, cy, sy = r.cos(), r.sin(), p_.cos(), p_.sin(), y.cos(), y.sin()
    R = torch.stack([
        cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
        sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
        -sp,     cp * sr,                cp * cr,
    ], dim=-1)
    return R.reshape(-1, 3, 3).to(torch.float32)
def rot_from_quat(q: torch.Tensor) -> torch.Tensor:
    q = q.to(torch.float64)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
        2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y),
    ], dim=-1)
    return R.reshape(-1, 3, 3).to(torch.float32)
def camera_from_state(state: torch.Tensor, pos_slice=slice(0, 3),
                      rpy_slice=slice(3, 6)) -> Tuple[torch.Tensor, torch.Tensor]:
    pos = state[:, pos_slice].to(torch.float32)
    R = rot_from_rpy(state[:, rpy_slice].to(torch.float32))
    cam = pos + R[:, :, 0] * CAM_FWD + R[:, :, 2] * CAM_UP
    return cam, R
def camera_from_pose(pos: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    return pos + R[:, :, 0] * CAM_FWD + R[:, :, 2] * CAM_UP
def camera_from_dataset(camera_row: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    c = camera_row.to(torch.float32)
    return c[:, 0:3], rot_from_quat(c[:, 3:7]), c[:, 7]
@dataclass
class GeomConfig:
    res: int = RES
    depth_min_m: float = DEPTH_MIN_M
    depth_max_m: float = DEPTH_MAX_M
    cell: float = 0.8            
    grid: int = 128              
    ground_win: int = 7          
    ground_win_fine: int = 3     
    ground_stride: int = 1
    height_lo: float = -2.0
    height_hi: float = 8.0
    z_rel_scale: float = 20.0
    pad_px_scale: float = 20.0
class GeomChannels:
    def __init__(self, cfg: Optional[GeomConfig] = None, channels=CHANNELS):
        self.cfg = cfg or GeomConfig()
        bad = [c for c in channels if c not in CHANNELS]
        if bad:
            raise ValueError(f"unknown channels: {bad}")
        self.channels = tuple(channels)
        self._ray_cache: dict = {}
    def _body_rays(self, fov_deg: float, device, dtype) -> torch.Tensor:
        key = (round(float(fov_deg), 4), str(device), str(dtype))
        if key not in self._ray_cache:
            n = self.cfg.res
            t = math.tan(math.radians(float(fov_deg)) / 2.0)
            u = torch.arange(n, device=device, dtype=dtype)
            a = (2.0 * u / n - 1.0) * t                 
            b = (1.0 - 2.0 * (u + 1.0) / n) * t         
            d = torch.empty(n, n, 3, device=device, dtype=dtype)
            d[..., 0] = 1.0                              
            d[..., 1] = -a[None, :]
            d[..., 2] = b[:, None]
            self._ray_cache[key] = d
        return self._ray_cache[key]
    def _local_ground(self, pts: torch.Tensor, valid: torch.Tensor, cam: torch.Tensor,
                      wins=(7,)) -> Tuple[list, torch.Tensor]:
        cfg = self.cfg
        B, H, W, _ = pts.shape
        G, big, st = cfg.grid, 1.0e6, max(1, int(cfg.ground_stride))
        origin = cam[:, :2] - 0.5 * G * cfg.cell                       
        def cell_index(xy):
            ij = ((xy - origin[:, None, None, :]) / cfg.cell).floor().long()
            ok = (ij[..., 0] >= 0) & (ij[..., 0] < G) & (ij[..., 1] >= 0) & (ij[..., 1] < G)
            return ij.clamp_(0, G - 1), ok
        sub = pts[:, ::st, ::st]
        sij, sok = cell_index(sub[..., :2])
        sok = sok & valid[:, ::st, ::st]
        sflat = (sij[..., 1] * G + sij[..., 0]).reshape(B, -1)
        sz = torch.where(sok, sub[..., 2], torch.full_like(sub[..., 2], big)).reshape(B, -1)
        grid = torch.full((B, G * G), big, device=pts.device, dtype=pts.dtype)
        grid.scatter_reduce_(1, sflat, sz, reduce="amin", include_self=True)
        ij, inside = cell_index(pts[..., :2])
        flat = (ij[..., 1] * G + ij[..., 0]).reshape(B, -1)
        grounds, knowns = [], []
        for k in wins:
            if k < 0:                      
                kk = -k
                pooled = -torch.nn.functional.max_pool2d(-grid.view(B, 1, G, G), kk, stride=1,
                                                         padding=kk // 2)
                P = pooled
                gx = torch.zeros_like(P); gy = torch.zeros_like(P)
                gx[..., 1:-1] = (P[..., 2:] - P[..., :-2]) / (2.0 * cfg.cell)
                gy[:, :, 1:-1, :] = (P[:, :, 2:, :] - P[:, :, :-2, :]) / (2.0 * cfg.cell)
                slope = (gx * gx + gy * gy).clamp_min(0).sqrt()
                corr = (P + slope * (kk * 0.5 * cfg.cell)).view(B, G * G)
                gxf = gx.view(B, G * G).gather(1, flat).view(B, H, W)
                gyf = gy.view(B, G * G).gather(1, flat).view(B, H, W)
                cxy = (ij.to(pts.dtype) + 0.5) * cfg.cell + origin[:, None, None, :]
                g_ = corr.gather(1, flat).view(B, H, W) \
                    + gxf * (pts[..., 0] - cxy[..., 0]) + gyf * (pts[..., 1] - cxy[..., 1])
                grounds.append(g_)
                knowns.append(g_ < big * 0.5)
                continue
            pooled = -torch.nn.functional.max_pool2d(-grid.view(B, 1, G, G), k, stride=1,
                                                     padding=k // 2).view(B, G * G)
            g_ = pooled.gather(1, flat).view(B, H, W)
            grounds.append(g_)
            knowns.append(g_ < big * 0.5)
        base = inside & valid
        return grounds, [kn & base for kn in knowns]
    def __call__(self, depth: torch.Tensor, cam: torch.Tensor, R: torch.Tensor,
                 fov_deg=FOV_BASE_DEG) -> torch.Tensor:
        cfg = self.cfg
        if depth.dim() == 4:                       
            if depth.shape[1] == 1:
                depth = depth[:, 0]
            elif depth.shape[-1] == 1:
                depth = depth[..., 0]
            else:
                raise ValueError(f"depth must be (B,H,W), (B,1,H,W) or (B,H,W,1); got {tuple(depth.shape)}")
        if depth.dim() != 3:
            raise ValueError(f"depth must be (B,H,W); got {tuple(depth.shape)}")
        depth = depth.to(torch.float32)
        B, H, W = depth.shape
        dev, dt = depth.device, depth.dtype
        cam = cam.to(device=dev, dtype=dt).reshape(B, 3)
        R = R.to(device=dev, dtype=dt).reshape(B, 3, 3)
        valid = (depth > 0.0) & (depth < 1.0)      
        metres = depth * (cfg.depth_max_m - cfg.depth_min_m) + cfg.depth_min_m
        fovs = _fmt(fov_deg, B, dev, dt)
        uniq = torch.unique(fovs)
        if uniq.numel() == 1:                      
            body = self._body_rays(float(uniq.item()), dev, dt).expand(B, H, W, 3)
        else:                                      
            body = torch.stack([self._body_rays(float(f), dev, dt) for f in fovs])
        ray = torch.einsum("bij,bhwj->bhwi", R, body)          
        pts = cam[:, None, None, :] + metres[..., None] * ray   
        out, need = {}, set(self.channels)
        if "depth" in need:
            out["depth"] = depth
        if "valid" in need:
            out["valid"] = valid.to(dt)
        if {"ray_sin", "ray_cos"} & need:
            rz = ray[..., 2] / ray.norm(dim=-1).clamp_min(1e-9)  
            out["ray_sin"] = rz
            out["ray_cos"] = (1.0 - rz * rz).clamp_min(0.0).sqrt()
        if "z_rel" in need:
            out["z_rel"] = (pts[..., 2] - cam[:, None, None, 2]) / cfg.z_rel_scale
        if "log_range" in need:
            out["log_range"] = torch.log1p(metres) / math.log1p(cfg.depth_max_m)
        if "pad_px" in need:
            mpp = metres * (2.0 * torch.tan(torch.deg2rad(fovs) / 2.0))[:, None, None] / cfg.res
            out["pad_px"] = (PAD_DIAM_M / mpp.clamp_min(1e-6)) / cfg.pad_px_scale
        if "normal_z" in need:
            rng_m = metres
            dxf = pts[:, :, 2:] - pts[:, :, 1:-1]
            dxb = pts[:, :, 1:-1] - pts[:, :, :-2]
            pick = ((rng_m[:, :, 2:] - rng_m[:, :, 1:-1]).abs()
                    <= (rng_m[:, :, 1:-1] - rng_m[:, :, :-2]).abs())[..., None]
            a = torch.where(pick, dxf, dxb)
            dyf = pts[:, 2:, :] - pts[:, 1:-1, :]
            dyb = pts[:, 1:-1, :] - pts[:, :-2, :]
            picky = ((rng_m[:, 2:, :] - rng_m[:, 1:-1, :]).abs()
                     <= (rng_m[:, 1:-1, :] - rng_m[:, :-2, :]).abs())[..., None]
            b = torch.where(picky, dyf, dyb)
            a, b = a[:, 1:-1, :], b[:, :, 1:-1]          
            nx = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
            ny = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
            nz_ = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
            nz = nz_.abs() / (nx * nx + ny * ny + nz_ * nz_).clamp_min(1e-18).sqrt()
            z = torch.zeros(B, H, W, device=dev, dtype=dt)
            z[:, 1:-1, 1:-1] = nz
            out["normal_z"] = z
        if {"height", "height_fine", "height_slope"} & need:
            wins, names = [], []
            if "height" in need:
                wins.append(cfg.ground_win); names.append("height")
            if "height_fine" in need:
                wins.append(cfg.ground_win_fine); names.append("height_fine")
            if "height_slope" in need:
                wins.append(-cfg.ground_win); names.append("height_slope")
            grounds, knowns = self._local_ground(pts, valid, cam, wins=wins)
            for nm, ground, known in zip(names, grounds, knowns):
                h = (pts[..., 2] - ground).clamp(cfg.height_lo, cfg.height_hi)
                h = (h - cfg.height_lo) / (cfg.height_hi - cfg.height_lo)
                out[nm] = torch.where(known, h, torch.zeros_like(h))
        v = valid.to(dt)
        stack = []
        for c in self.channels:
            x = out[c]
            stack.append(x if c in ("depth", "valid") else x * v)   
        return torch.stack(stack, dim=1).contiguous()
