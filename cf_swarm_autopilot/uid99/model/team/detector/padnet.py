\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

def _block(cin: int, cout: int, stride: int = 1) -> nn.Sequential:

    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(min(8, cout), cout),
        nn.SiLU(inplace=True),
    )

class PadNet(nn.Module):

    def __init__(self, width: int = 16):
        super().__init__()
        w = width
        self.e1 = nn.Sequential(_block(1, w), _block(w, w))
        self.e2 = nn.Sequential(_block(w, 2 * w, 2), _block(2 * w, 2 * w))
        self.e3 = nn.Sequential(_block(2 * w, 4 * w, 2), _block(4 * w, 4 * w))
        self.e4 = nn.Sequential(_block(4 * w, 8 * w, 2), _block(8 * w, 8 * w))
        self.d3 = _block(8 * w + 4 * w, 4 * w)
        self.d2 = _block(4 * w + 2 * w, 2 * w)
        self.d1 = _block(2 * w + w, w)
        self.head = nn.Conv2d(w, 1, 1)

        nn.init.constant_(self.head.bias, -4.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        u = F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        u = self.d3(torch.cat([u, e3], 1))
        u = F.interpolate(u, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        u = self.d2(torch.cat([u, e2], 1))
        u = F.interpolate(u, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        u = self.d1(torch.cat([u, e1], 1))
        return self.head(u)

def loss_fn(logits: torch.Tensor, target: torch.Tensor, pos_weight: float = 40.0,
            size_balance: bool = False, ref_px: float = 48.0):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\

    if size_balance:
        area = target.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
        scale = (ref_px / area).clamp(0.25, 4.0)

        wmap = 1.0 + (pos_weight * scale - 1.0) * target
        bce = F.binary_cross_entropy_with_logits(logits, target, weight=wmap)
    else:
        bce = F.binary_cross_entropy_with_logits(
            logits, target,
            pos_weight=torch.as_tensor(pos_weight, device=logits.device))
    p = torch.sigmoid(logits)
    num = 2.0 * (p * target).sum(dim=(1, 2, 3)) + 1.0
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1.0
    dice = 1.0 - (num / den).mean()
    return bce + dice, float(bce.detach()), float(dice.detach())

@torch.no_grad()
def recall_by_size(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5):
\
\
\
\
\
\
\
\
\

    pred = torch.sigmoid(logits) > thr
    t = target > 0.5
    area = t.sum(dim=(1, 2, 3))
    hit = (pred & t).sum(dim=(1, 2, 3))
    out = {}
    for name, lo, hi in (("far", 1, 20), ("mid", 20, 60), ("near", 60, 10 ** 9)):
        sel = (area >= lo) & (area < hi)
        n = int(sel.sum())
        out[name] = (float((hit[sel] > 0).float().mean()) if n else float("nan"), n)
    return out

@torch.no_grad()
def pixel_stats(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5):

    pred = (torch.sigmoid(logits) > thr)
    t = target > 0.5
    tp = float((pred & t).sum())
    fp = float((pred & ~t).sum())
    fn = float((~pred & t).sum())
    rec = tp / max(1.0, tp + fn)
    prec = tp / max(1.0, tp + fp)
    return rec, prec, 2 * rec * prec / max(1e-6, rec + prec)
