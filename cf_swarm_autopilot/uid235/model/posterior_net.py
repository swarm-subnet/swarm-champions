from __future__ import annotations

"""Deployable conditional remaining-pad density network."""

import numpy as np


class PosteriorScorer:
    MAPS = ("city", "open", "mountain", "village", "forest")

    def __init__(self, checkpoint: str, device: str = "cpu", width: int = 112):
        import torch
        from torch import nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.start = nn.Sequential(nn.Linear(3, width), nn.SiLU(),
                                           nn.Linear(width, width), nn.SiLU())
                self.found = nn.Sequential(nn.Linear(3, width), nn.SiLU(),
                                           nn.Linear(width, width), nn.SiLU())
                self.head = nn.Sequential(
                    nn.Linear(2 * width + 6 + 5, 2 * width), nn.SiLU(),
                    nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, 1))

            @staticmethod
            def pool(enc, rel, mask):
                x = torch.cat((rel, torch.linalg.vector_norm(rel, dim=-1,
                                                              keepdim=True)), -1)
                h = enc(x)
                mf = mask[..., None].to(h.dtype)
                mean = (h * mf).sum(1) / mf.sum(1).clamp_min(1.0)
                return mean * mask.any(1, keepdim=True).to(h.dtype)

            def forward(self, starts_rel, starts_mask, found_rel, found_mask,
                        glob, map_oh):
                hs = self.pool(self.start, starts_rel, starts_mask)
                hf = self.pool(self.found, found_rel, found_mask)
                residual = self.head(torch.cat((hs, hf, glob, map_oh), -1)).squeeze(-1)
                d = torch.linalg.vector_norm(starts_rel, dim=-1) * 120.0
                d = torch.where(starts_mask, d, torch.full_like(d, 1e6))
                lo = torch.tensor([22., 28., 65., 28., 22.], device=d.device)
                hi = torch.tensor([45., 72., 100., 56., 45.], device=d.device)
                lo = (map_oh * lo).sum(-1, keepdim=True)
                hi = (map_oh * hi).sum(-1, keepdim=True)
                mid = 0.5 * (lo + hi)
                sig = torch.maximum(torch.full_like(mid, 6.0), 0.25 * (hi - lo))
                band = (torch.exp(-0.5 * ((d - mid) / sig) ** 2) * starts_mask).sum(-1)
                dc = torch.linalg.vector_norm(glob[:, :2], dim=-1) * 120.0
                base = torch.log((band * torch.exp(-0.5 * (dc / 45.0) ** 2))
                                 .clamp_min(1e-7))
                return base + residual

        self._torch = torch
        self.device = device
        self.net = Net().to(device).eval()
        blob = torch.load(checkpoint, map_location=device, weights_only=True)
        self.net.load_state_dict(blob.get("model", blob), strict=True)

    def score(self, starts, found, clue, xy, map_kind):
        torch = self._torch
        starts = np.asarray(starts, np.float32)
        found = np.asarray(found, np.float32).reshape(-1, 3)
        clue = np.asarray(clue, np.float32)
        xy = np.asarray(xy, np.float32).reshape(-1, 2)
        n, m, b, scale = len(starts), len(found), len(xy), 120.0
        sr = np.zeros((b, 8, 2), np.float32)
        sm = np.zeros((b, 8), bool)
        sr[:, :n] = (starts[None, :, :2] - xy[:, None]) / scale
        sm[:, :n] = True
        fr = np.zeros((b, 8, 2), np.float32)
        fm = np.zeros((b, 8), bool)
        if m:
            fr[:, :m] = (found[None, :, :2] - xy[:, None]) / scale
            fm[:, :m] = True
        rem = max(1, n - m)
        residual = (n * clue[:2] - found[:, :2].sum(0)) / rem
        glob = np.column_stack(((xy - clue[None, :2]) / scale,
                                (residual[None, :] - xy) / scale,
                                np.full(b, n / 8.0, np.float32),
                                np.full(b, m / 8.0, np.float32))).astype(np.float32)
        mo = np.zeros((b, 5), np.float32)
        mo[:, self.MAPS.index(map_kind)] = 1.0
        args = (sr, sm, fr, fm, glob, mo)
        with torch.inference_mode():
            tx = [torch.from_numpy(v).to(self.device) for v in args]
            return self.net(*tx).cpu().numpy()
