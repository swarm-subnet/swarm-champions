"""PadNet — depth frame in, per-pixel "this is a landing pad" logits out.

The geometric detector proposes 13.5 regions per episode against 5.4 real pads,
and on one open seed proposed 102 for 7. It fires on cars, kerbs and roofs
because on four of five maps a pad stands only 0.2-0.9 m proud of the ground and
"flat thing slightly above the ground" describes a car roof exactly. Those false
positives are not merely wasted flights: a drone commits, descends onto a car and
is recorded as OBSTACLE_COLLISION, which is where 24 of 43 drone-deaths came from.

So this is a discriminator, and the thing it has to discriminate is appearance in
depth, not geometry — which is what a network is for and a threshold is not.

Deliberately small. The validator gives 500 ms per act() for the whole fleet and
runs on CPU, so the encoder halves resolution three times before doing any real
work and the decoder is plain bilinear upsampling with skips. Inference is
batched over drones (one forward pass for all N) rather than looped.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    # GroupNorm, not BatchNorm: batches at inference are the drone count (2-8)
    # and BatchNorm's running stats would drift with fleet size.
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(min(8, cout), cout),
        nn.SiLU(inplace=True),
    )


class PadNet(nn.Module):
    """(B,1,128,128) depth in [0,1] -> (B,1,128,128) logits."""

    def __init__(self, width: int = 16):
        super().__init__()
        w = width
        self.e1 = nn.Sequential(_block(1, w), _block(w, w))              # 128
        self.e2 = nn.Sequential(_block(w, 2 * w, 2), _block(2 * w, 2 * w))    # 64
        self.e3 = nn.Sequential(_block(2 * w, 4 * w, 2), _block(4 * w, 4 * w))  # 32
        self.e4 = nn.Sequential(_block(4 * w, 8 * w, 2), _block(8 * w, 8 * w))  # 16
        self.d3 = _block(8 * w + 4 * w, 4 * w)
        self.d2 = _block(4 * w + 2 * w, 2 * w)
        self.d1 = _block(2 * w + w, w)
        self.head = nn.Conv2d(w, 1, 1)
        # Start biased hard toward "not a pad": positives are ~0.1% of pixels, and
        # without this the first epochs are spent unlearning a uniform yes.
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
    """BCE (weighted) + soft dice.

    Pads occupy ~0.1% of pixels — a median of 16-23 px in a 16384 px frame — so
    unweighted BCE is minimised by predicting nothing at all. The weight buys
    recall; the dice term is what stops the weight from buying a smear, because
    dice cares about overlap rather than per-pixel counts.

    `size_balance` equalises the gradient a pad contributes regardless of how far
    away it is. A 1.2 m pad covers ~185 px at 5 m and ~14 px at 18 m, so summed
    BCE lets near pads outvote far ones ~13:1 — and the agent measures 75-95%
    detection inside 15 m against 48% beyond it. Dice is already per-image and so
    already scale-free; this fixes the half that is not. Weights clamp to
    [0.25, 4] so one tiny blob cannot dominate a batch, and ref_px is roughly the
    pad area at 10 m.

    Off by default: it changes the OBJECTIVE, so it only takes effect on a
    retrained checkpoint and never on an existing one. Taken from UID182, who
    retrained only their village net with it — unmeasured by us. Note the scale
    is per-IMAGE, not per-pad: a frame holding both a near and a far pad gets one
    weight from their combined area, so the correction is partial whenever a
    frame shows more than one.
    """
    if size_balance:
        area = target.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
        scale = (ref_px / area).clamp(0.25, 4.0)
        # Per-element weights: positives scaled by how small this pad is.
        bce = F.binary_cross_entropy_with_logits(
            logits, target, weight=1.0 + (pos_weight * scale - 1.0) * target)
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
    """Recall split by pad area in pixels, which is a proxy for range.

    PER-FRAME, NOT PER-PAD, despite what UID47's version of this claims: `area`
    is the summed positive area of the whole frame, so a frame holding a near and
    a far pad lands in one bucket on their combined area and counts as "found" if
    either was hit. The buckets are still the right split — most frames show one
    pad — but the far bucket is optimistic wherever they do not.

    Pooled recall hides the failure this is here to find: a model can score 0.75
    overall while detecting almost nothing beyond 15 m, because the near pads
    carry ~13x the pixels. <20 px is roughly 15 m+, 20-60 px is 10-15 m, 60+ px
    is closer. This is the metric that says whether `size_balance` did anything.

    A pad counts as found if ANY predicted pixel lands on it — what the proposer
    needs — not whether the whole silhouette is covered.
    """
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
    """Recall / precision over pixels — the numbers that predict agent behaviour."""
    pred = (torch.sigmoid(logits) > thr)
    t = target > 0.5
    tp = float((pred & t).sum())
    fp = float((pred & ~t).sum())
    fn = float((~pred & t).sum())
    rec = tp / max(1.0, tp + fn)
    prec = tp / max(1.0, tp + fp)
    return rec, prec, 2 * rec * prec / max(1e-6, rec + prec)
