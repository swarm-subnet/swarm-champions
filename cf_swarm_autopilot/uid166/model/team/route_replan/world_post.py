"""WPOST-X: posterior over the mountain world half-extent W, from the start pads only.

research/ideas/REPORT.md section 4.1. The generator draws the mountain half-extent per seed in
[90, 120] m and spreads the start pads uniformly inside the +-W box, so the likelihood of the
fleet's starts is (2W)^(-2n) on W + nudge >= max|start coordinate| (pad placement nudges a start
a few metres). Flat prior on a 0.25 m grid. Pure numpy; uses nothing but the observed starts
(and, in alive(), the pads the fleet has found)."""
from __future__ import annotations

import numpy as np


def w_grid(lo: float = 90.0, hi: float = 120.0, dw: float = 0.25) -> np.ndarray:
    return np.arange(float(lo), float(hi) + 1e-4, float(dw))


def posterior(S, lo: float = 90.0, hi: float = 120.0, nudge: float = 4.0, dw: float = 0.25):
    """(WG, p, m): the W grid, the posterior over it (None when no W on the grid reaches the
    starts) and m = max|start coordinate|."""
    S = np.asarray(S, float)
    S = S.reshape(-1, S.shape[-1])[:, :2] if S.size else np.zeros((0, 2))
    n = len(S)
    WG = w_grid(lo, hi, dw)
    m = float(np.max(np.abs(S))) if n else 0.0
    ok = WG + float(nudge) >= m
    if not ok.any() or n == 0:
        return WG, None, m
    lw = np.where(ok, -2.0 * n * np.log(2.0 * WG), -np.inf)
    p = np.exp(lw - lw[ok].max())
    return WG, p / p.sum(), m


def quantile(WG, p, a: float) -> float:
    c = np.cumsum(p)
    return float(WG[min(int(np.searchsorted(c, a)), len(WG) - 1)])


def quantiles3(p, WG=None):
    """The 1/6, 1/2 and 5/6 quantiles (three equal-mass thirds of the posterior)."""
    WG = w_grid() if WG is None else WG
    return [quantile(WG, p, 1.0 / 6.0), quantile(WG, p, 0.5), quantile(WG, p, 5.0 / 6.0)]


def mean(WG, p) -> float:
    return float((np.asarray(WG) * np.asarray(p)).sum())


def components(WG, p, m: float, mix: int = 3, stat: str = "mean", margin: float = 0.0):
    """The W values the field is mixed over. mix >= 2: the quantiles (2j+1)/(2 mix) (mix 3 = the
    1/6, 1/2, 5/6 quantiles); mix 1: one point estimate, `stat` in mean / median / qNN. Each plus
    `margin`. No feasible W on the grid: m for every component (research/ideas/feas/feas_check.py)."""
    k = max(int(mix), 1)
    if p is None:
        base = [float(m)] * k
    elif k == 1:
        s = str(stat or "mean").lower()
        if s == "mean":
            base = [mean(WG, p)]
        elif s == "median":
            base = [quantile(WG, p, 0.5)]
        elif s.startswith("q"):
            base = [quantile(WG, p, float(s[1:]) / 100.0)]
        else:
            base = [mean(WG, p)]
    elif k == 3:
        base = quantiles3(p, WG)
    else:
        base = [quantile(WG, p, (2.0 * j + 1.0) / (2.0 * k)) for j in range(k)]
    return [float(w) + float(margin) for w in base]


def alive(Wq, found, slack: float = 3.9):
    """(kept, n_dropped): drop the components a found pad proves too small (W + slack < the
    pad's max|coordinate|). If every component is dropped, the largest is kept."""
    Wq = [float(w) for w in Wq]
    F = np.asarray(found, float).reshape(-1, 2) if len(found) else np.zeros((0, 2))
    if not len(F) or not Wq:
        return list(Wq), 0
    mf = float(np.max(np.abs(F)))
    kept = [w for w in Wq if w + float(slack) >= mf]
    if not kept:
        kept = [max(Wq)]
    return kept, len(Wq) - len(kept)
