from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
MAP_CLASSES = ("city", "open", "mountain", "village", "forest")
def dw_sep(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, 1, 1, groups=cin, bias=False),
        nn.BatchNorm2d(cin), nn.SiLU(inplace=True),
        nn.Conv2d(cin, cout, 1, bias=False),
        nn.BatchNorm2d(cout), nn.SiLU(inplace=True),
    )
def conv_bn(cin, cout, k=3, s=1, g=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, k // 2, groups=g, bias=False),
        nn.BatchNorm2d(cout),
        nn.SiLU(inplace=True),
    )
class Bottleneck(nn.Module):
    def __init__(self, c, shortcut=True):
        super().__init__()
        self.cv1, self.cv2, self.add = conv_bn(c, c, 3), conv_bn(c, c, 3), shortcut
    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y
class CSP(nn.Module):
    def __init__(self, cin, cout, n=1, shortcut=True):
        super().__init__()
        c = cout // 2
        self.cv1 = conv_bn(cin, 2 * c, 1)
        self.m = nn.ModuleList(Bottleneck(c, shortcut) for _ in range(n))
        self.cv2 = conv_bn((2 + n) * c, cout, 1)
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))
class SPPF(nn.Module):
    def __init__(self, cin, cout, k=5):
        super().__init__()
        c = cin // 2
        self.cv1, self.cv2 = conv_bn(cin, c, 1), conv_bn(c * 4, cout, 1)
        self.pool = nn.MaxPool2d(k, 1, k // 2)
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        return self.cv2(torch.cat([x, y1, y2, self.pool(y2)], 1))
@dataclass
class ModelConfig:
    in_ch: int = 11
    width: Sequence[int] = (12, 24, 48, 96, 128)    
    depth: Sequence[int] = (1, 1, 2, 1)             
    dec: int = 24                                   
    dec_fine: int = 16                              
    map_head: bool = False                          
    n_map_classes: int = len(MAP_CLASSES)
    mask_prior: float = -4.0                        
    heat_prior: float = -4.6
class PadModel(nn.Module):
    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig()
        w, d = list(cfg.width), list(cfg.depth)
        self.stem = conv_bn(cfg.in_ch, w[0], 3)                       
        self.e1 = nn.Sequential(conv_bn(w[0], w[1], 3, 2), CSP(w[1], w[1], d[0]))   
        self.e2 = nn.Sequential(conv_bn(w[1], w[2], 3, 2), CSP(w[2], w[2], d[1]))   
        self.e3 = nn.Sequential(conv_bn(w[2], w[3], 3, 2), CSP(w[3], w[3], d[2]))   
        self.e4 = nn.Sequential(conv_bn(w[3], w[4], 3, 2), CSP(w[4], w[4], d[3]),
                                SPPF(w[4], w[4]))                                   
        self.lat = nn.ModuleList([nn.Conv2d(w[0], cfg.dec_fine, 1)]
                                 + [nn.Conv2d(c, cfg.dec, 1) for c in w[1:]])
        self.smooth = nn.ModuleList([dw_sep(cfg.dec_fine, cfg.dec_fine)]
                                    + [conv_bn(cfg.dec, cfg.dec, 3) for _ in range(3)])
        self.to_fine = nn.Conv2d(cfg.dec, cfg.dec_fine, 1)   
        self.fuse = dw_sep(cfg.dec_fine + cfg.in_ch, cfg.dec_fine)
        self.mask_head = nn.Conv2d(cfg.dec_fine, 1, 1)
        self.heat_head = nn.Conv2d(cfg.dec_fine, 1, 1)
        self.off_head = nn.Conv2d(cfg.dec_fine, 2, 1)
        self.z_head = nn.Conv2d(cfg.dec_fine, 1, 1)
        self.map_head = (nn.Linear(w[4], cfg.n_map_classes) if cfg.map_head else None)
        nn.init.constant_(self.mask_head.bias, cfg.mask_prior)
        nn.init.constant_(self.heat_head.bias, cfg.heat_prior)
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        f0 = self.stem(x)
        f1 = self.e1(f0)
        f2 = self.e2(f1)
        f3 = self.e3(f2)
        f4 = self.e4(f3)
        u = self.lat[4](f4)
        for k, f in ((3, f3), (2, f2), (1, f1)):
            u = F.interpolate(u, size=f.shape[-2:], mode="bilinear", align_corners=False)
            u = self.smooth[k](u + self.lat[k](f))
        u = self.to_fine(u)                                  
        u = F.interpolate(u, size=f0.shape[-2:], mode="bilinear", align_corners=False)
        u = self.smooth[0](u + self.lat[0](f0))
        u = self.fuse(torch.cat([u, x], 1))
        out = {"mask": self.mask_head(u), "heat": self.heat_head(u),
               "offset": self.off_head(u), "z": self.z_head(u)}
        if self.map_head is not None:
            out["map"] = self.map_head(F.adaptive_avg_pool2d(f4, 1).flatten(1))
        return out
    @torch.no_grad()
    def fuse_bn(self):
        for m in self.modules():
            if not isinstance(m, nn.Sequential):
                continue
            for i in range(len(m) - 1):
                conv, bn = m[i], m[i + 1]
                if not (isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d)):
                    continue
                s = bn.weight / (bn.running_var + bn.eps).sqrt()
                b0 = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
                conv.weight = nn.Parameter(conv.weight * s.reshape(-1, 1, 1, 1))
                conv.bias = nn.Parameter(bn.bias + (b0 - bn.running_mean) * s)
                m[i + 1] = nn.Identity()
        return self
PAD_EXISTS, PAD_KIND, PAD_Z, PAD_U, PAD_V, PAD_ZD, PAD_PXD = 0, 1, 5, 6, 7, 8, 10
PAD_DIAM_M = 1.2
Z_SCALE = 0.5          
@dataclass
class LossConfig:
    min_px: int = 5             
    pos_weight: float = 40.0    
    w_mask: float = 1.0
    w_heat: float = 1.0
    w_off: float = 0.5
    w_z: float = 0.2
    w_map: float = 0.05         
    size_balance: bool = True   
    inst_w_clip: tuple = (0.25, 4.0)
    sigma_min: float = 0.8
    sigma_max: float = 12.0
    win: int = 25               
def make_targets(padmask: torch.Tensor, inst: torch.Tensor, pads: torch.Tensor,
                 point_z: Optional[torch.Tensor] = None,
                 cfg: Optional[LossConfig] = None,
                 fov_deg: float = 90.0) -> Dict[str, torch.Tensor]:
    cfg = cfg or LossConfig()
    B, H, W = padmask.shape
    dev = padmask.device
    t = {}
    mask = (padmask > 0).float()
    t["mask"] = mask[:, None]
    w = torch.ones_like(mask)
    if cfg.size_balance:
        ids = inst.long()
        flat = ids.reshape(B, -1)
        area = torch.zeros(B, 256, device=dev).scatter_add_(
            1, flat, (flat >= 10).float().reshape(B, -1))
        ref = area.sum(1, keepdim=True) / area.gt(0).sum(1, keepdim=True).clamp_min(1)
        scale = (ref / area.clamp_min(1.0)).clamp(*cfg.inst_w_clip)
        w = torch.where(mask > 0, scale.gather(1, flat).reshape(B, H, W), w)
    t["mask_w"] = w[:, None]
    heat = torch.zeros(B, H, W, device=dev)
    off = torch.zeros(B, 2, H, W, device=dev)
    off_m = torch.zeros(B, 1, H, W, device=dev)
    ok = ((pads[..., PAD_EXISTS] > 0) & (pads[..., PAD_PXD] >= cfg.min_px)
          & torch.isfinite(pads[..., PAD_U]) & torch.isfinite(pads[..., PAD_V]))
    bi, pi = torch.nonzero(ok, as_tuple=True)
    if bi.numel():
        u, v = pads[bi, pi, PAD_U], pads[bi, pi, PAD_V]
        zd = pads[bi, pi, PAD_ZD].clamp_min(0.5)
        import math as _math
        px_per_m = W / (2.0 * _math.tan(_math.radians(fov_deg) / 2.0) * zd)
        sigma = (0.5 * PAD_DIAM_M * px_per_m / 3.0).clamp(cfg.sigma_min, cfg.sigma_max)
        ui, vi = u.round().long(), v.round().long()
        inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        bi, ui, vi, u, v, sigma = bi[inb], ui[inb], vi[inb], u[inb], v[inb], sigma[inb]
        if bi.numel():
            k = cfg.win // 2
            d = torch.arange(-k, k + 1, device=dev)
            dy, dx = torch.meshgrid(d, d, indexing="ij")
            gy = vi[:, None, None] + dy[None]
            gx = ui[:, None, None] + dx[None]
            val = torch.exp(-(dx[None] ** 2 + dy[None] ** 2).float()
                            / (2.0 * sigma[:, None, None] ** 2))
            good = (gy >= 0) & (gy < H) & (gx >= 0) & (gx < W)
            idx = (bi[:, None, None] * H * W + gy.clamp(0, H - 1) * W + gx.clamp(0, W - 1))
            heat = heat.reshape(-1).scatter_reduce(
                0, idx[good].reshape(-1), val[good].reshape(-1), reduce="amax"
            ).reshape(B, H, W)
            pk = bi * H * W + vi * W + ui
            heat.reshape(-1)[pk] = 1.0
            off.reshape(B, 2, -1)[bi, 0, vi * W + ui] = (u - ui.float())
            off.reshape(B, 2, -1)[bi, 1, vi * W + ui] = (v - vi.float())
            off_m.reshape(B, -1)[bi, vi * W + ui] = 1.0
    t["heat"], t["offset"], t["offset_mask"] = heat[:, None], off, off_m
    if point_z is not None:
        ids = inst.long()
        zt = torch.zeros(B, 256, device=dev)
        idxs = torch.arange(pads.shape[1], device=dev)
        slot = torch.where(pads[..., PAD_KIND] < 0.5, 10 + idxs[None], 20 + idxs[None] - 8)
        zt.scatter_(1, slot.long().clamp(0, 255), pads[..., PAD_Z].nan_to_num(0.0))
        pad_z = zt.gather(1, ids.reshape(B, -1)).reshape(B, H, W)
        t["z"] = ((pad_z - point_z) / Z_SCALE)[:, None]
        t["z_mask"] = (mask > 0)[:, None].float()
    return t
def focal_heat(pred_logit, target, alpha=2.0, beta=4.0, eps=1e-6):
    p = torch.sigmoid(pred_logit).clamp(eps, 1 - eps)
    pos = (target >= 1.0).float()
    pos_loss = -((1 - p) ** alpha) * torch.log(p) * pos
    neg_loss = -((1 - target) ** beta) * (p ** alpha) * torch.log(1 - p) * (1 - pos)
    n = pos.sum().clamp_min(1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n
class PadLoss(nn.Module):
    def __init__(self, cfg: Optional[LossConfig] = None):
        super().__init__()
        self.cfg = cfg or LossConfig()
    def forward(self, out: Dict[str, torch.Tensor], tgt: Dict[str, torch.Tensor],
                map_label: Optional[torch.Tensor] = None):
        c = self.cfg
        logs = {}
        m_t, m_w = tgt["mask"], tgt.get("mask_w")
        w = torch.ones_like(m_t) if m_w is None else m_w
        bce = F.binary_cross_entropy_with_logits(
            out["mask"], m_t, weight=1.0 + (c.pos_weight * w - 1.0) * m_t)
        p = torch.sigmoid(out["mask"])
        num = 2.0 * (p * m_t).sum((1, 2, 3)) + 1.0
        den = p.sum((1, 2, 3)) + m_t.sum((1, 2, 3)) + 1.0
        dice = 1.0 - (num / den).mean()
        loss = c.w_mask * (bce + dice)
        logs["bce"], logs["dice"] = float(bce.detach()), float(dice.detach())
        hm = focal_heat(out["heat"], tgt["heat"])
        loss = loss + c.w_heat * hm
        logs["heat"] = float(hm.detach())
        om = tgt["offset_mask"]
        n = om.sum().clamp_min(1.0)
        off = (F.l1_loss(out["offset"] * om, tgt["offset"] * om, reduction="sum") / n)
        loss = loss + c.w_off * off
        logs["offset"] = float(off.detach())
        if "z" in tgt:
            zm = tgt["z_mask"]
            z = (F.smooth_l1_loss(out["z"] * zm, tgt["z"] * zm, reduction="sum")
                 / zm.sum().clamp_min(1.0))
            loss = loss + c.w_z * z
            logs["z"] = float(z.detach())
        if map_label is not None and "map" in out:
            mp = F.cross_entropy(out["map"], map_label)
            loss = loss + c.w_map * mp
            logs["map"] = float(mp.detach())
        logs["total"] = float(loss.detach())
        return loss, logs
