from __future__ import annotations
import numpy as np
SEARCH_R_MIN, SEARCH_R_MAX = 5.0, 30.0   
_SQ2 = np.sqrt(2.0)
BANDS = {"city": (22.0, 45.0), "open": (28.0, 72.0), "mountain": (65.0, 100.0),
         "village": (65.0, 100.0), "forest": (22.0, 45.0)}
WORLD = {"city": 75.0, "open": 60.0, "mountain": 110.0, "village": 40.0, "forest": 42.0}
_A = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)
def _erf(x):
    sgn = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    y = 1.0 - (((((_A[4] * t + _A[3]) * t) + _A[2]) * t + _A[1]) * t + _A[0]) * t * np.exp(-ax * ax)
    return sgn * y
def _ndtr(a):
    return 0.5 * (1.0 + _erf(a / _SQ2))
def _box_gauss(lo, hi, mu, sd):
    return _ndtr((hi - mu) / sd) - _ndtr((lo - mu) / sd)
def annulus_prior(gx, gy, s, r_min, r_max, W):
    X = gx[None, :] - s[0]
    Y = gy[:, None] - s[1]
    r = np.hypot(X, Y)
    rs = np.maximum(r, 1e-9)
    ca, sa = X / rs, Y / rs
    with np.errstate(divide="ignore", invalid="ignore"):
        mx = np.where(np.abs(ca) > 1e-8,
                      np.where(ca > 0, (W - s[0]) / np.where(ca != 0, ca, 1.0),
                               (-W - s[0]) / np.where(ca != 0, ca, 1.0)), np.inf)
        my = np.where(np.abs(sa) > 1e-8,
                      np.where(sa > 0, (W - s[1]) / np.where(sa != 0, sa, 1.0),
                               (-W - s[1]) / np.where(sa != 0, sa, 1.0)), np.inf)
    mx = np.where(mx > 0, mx, np.inf)
    my = np.where(my > 0, my, np.inf)
    rhi = np.minimum(np.minimum(mx, my), r_max)
    ok = (rhi >= r_min) & (r >= r_min) & (r <= rhi * 0.999)
    dens = np.where(ok, 1.0 / (np.maximum(rhi - r_min, 1e-6) * rs), 0.0)
    dens *= ((np.abs(gx)[None, :] <= W) & (np.abs(gy)[:, None] <= W))
    tot = dens.sum()
    return dens / tot if tot > 0 else dens
def load_radial_prior(path):
    import json
    raw = json.load(open(path))
    return {k: (np.asarray(v["edges"], float), np.asarray(v["pdf"], float)) for k, v in raw.items()}
def empirical_prior(gx, gy, s, edges, pdf, W):
    r = np.hypot(gx[None, :] - s[0], gy[:, None] - s[1])
    k = np.clip(np.searchsorted(edges, r, side="right") - 1, 0, len(pdf) - 1)
    dens = np.where(r < edges[-1], pdf[k], 0.0) / np.maximum(r, 0.5)
    dens *= ((np.abs(gx)[None, :] <= W) & (np.abs(gy)[:, None] <= W))
    tot = dens.sum()
    return dens / tot if tot > 0 else dens
def _moments(gx, gy, f):
    px, py = f.sum(0), f.sum(1)
    mx = float((gx * px).sum())
    my = float((gy * py).sum())
    vx = max(float((gx ** 2 * px).sum()) - mx * mx, 1e-6)
    vy = max(float((gy ** 2 * py).sum()) - my * my, 1e-6)
    return mx, my, vx, vy
def pad_field(starts, clue, kind, step=2.0, nR=25, found=(), W=None, radial=None):
    r_min, r_max = BANDS[kind]
    W = float(WORLD[kind] if W is None else W)
    starts = np.asarray(starts, float)[:, :2]
    found = np.asarray(found, float).reshape(-1, 2)
    n_total = len(starts)                      
    gx = np.arange(-W, W + 1e-9, step)
    gy = gx.copy()
    open_idx = [j for j in range(len(starts))]
    if radial is not None and kind in radial:
        edges, pdf = radial[kind]
        F = [empirical_prior(gx, gy, starts[j], edges, pdf, W) for j in open_idx]
    else:
        F = [annulus_prior(gx, gy, starts[j], r_min, r_max, W) for j in open_idx]
    mom = [_moments(gx, gy, f) for f in F]
    Rs = np.linspace(SEARCH_R_MIN, SEARCH_R_MAX, nR)
    XX = np.broadcast_to(gx[None, :], (len(gy), len(gx)))
    YY = np.broadcast_to(gy[:, None], (len(gy), len(gx)))
    fx = float(found[:, 0].sum()) if len(found) else 0.0
    fy = float(found[:, 1].sum()) if len(found) else 0.0
    rho = np.zeros((len(gy), len(gx)))
    for j in range(len(F)):
        mx = sum(mom[k][0] for k in range(len(F)) if k != j) + fx
        my = sum(mom[k][1] for k in range(len(F)) if k != j) + fy
        vx = sum(mom[k][2] for k in range(len(F)) if k != j)
        vy = sum(mom[k][3] for k in range(len(F)) if k != j)
        cx_mu, cx_sd = (XX + mx) / n_total, max(np.sqrt(vx) / n_total, 1e-3)
        cy_mu, cy_sd = (YY + my) / n_total, max(np.sqrt(vy) / n_total, 1e-3)
        acc = np.zeros_like(XX, dtype=float)
        for R in Rs:
            acc += (_box_gauss(clue[0] - R, clue[0] + R, cx_mu, cx_sd)
                    * _box_gauss(clue[1] - R, clue[1] + R, cy_mu, cy_sd)) / (4.0 * R * R)
        rho += F[j] * acc
    tot = rho.sum()
    if tot > 0:
        rho *= len(F) / tot            
    return gx, gy, rho
def unfound_start_weights(ring_at_found):
    from itertools import permutations
    k, n = ring_at_found.shape
    if k == 0:
        return np.ones(n)
    if k > n:
        return np.zeros(n)
    perms = np.asarray(list(permutations(range(n), k)), dtype=np.int64)        
    w = np.prod(ring_at_found[np.arange(k)[None, :], perms], axis=1)            
    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones(len(perms))                                                 
    used = np.zeros((len(perms), n))
    np.put_along_axis(used, perms, 1.0, axis=1)
    return 1.0 - (w[:, None] * used).sum(0) / w.sum()
def pad_field_found(starts, clue, kind, found_xy, step=4.0, nR=25, W=None, radial=None):
    r_min, r_max = BANDS[kind]
    W = float(WORLD[kind] if W is None else W)
    S = np.asarray(starts, float)[:, :2]
    Y = np.asarray(found_xy, float).reshape(-1, 2)
    n, k = len(S), len(Y)
    gx = np.arange(-W, W + 1e-9, step)
    gy = gx.copy()
    if radial is not None and kind in radial:
        edges, pdf = radial[kind]
        F = [empirical_prior(gx, gy, S[j], edges, pdf, W) for j in range(n)]
    else:
        F = [annulus_prior(gx, gy, S[j], r_min, r_max, W) for j in range(n)]
    ix = np.clip(np.rint((Y[:, 0] - gx[0]) / step).astype(int), 0, len(gx) - 1) if k else np.zeros(0, int)
    iy = np.clip(np.rint((Y[:, 1] - gy[0]) / step).astype(int), 0, len(gy) - 1) if k else np.zeros(0, int)
    ring_at = np.array([[F[j][iy[i], ix[i]] for j in range(n)] for i in range(k)]).reshape(k, n)
    u = unfound_start_weights(ring_at)
    mom = [_moments(gx, gy, f) for f in F]
    Rs = np.linspace(SEARCH_R_MIN, SEARCH_R_MAX, nR)
    XX = np.broadcast_to(gx[None, :], (len(gy), len(gx)))
    YY = np.broadcast_to(gy[:, None], (len(gy), len(gx)))
    fx, fy = (float(Y[:, 0].sum()), float(Y[:, 1].sum())) if k else (0.0, 0.0)
    rem = n - k
    rho = np.zeros((len(gy), len(gx)))
    for j in range(n):
        if u[j] <= 1e-9:
            continue
        others = [q for q in range(n) if q != j]
        su = sum(u[q] for q in others)
        c = [(u[q] * (rem - 1) / su) if su > 0 else 0.0 for q in others]
        mx = sum(ci * mom[q][0] for ci, q in zip(c, others)) + fx
        my = sum(ci * mom[q][1] for ci, q in zip(c, others)) + fy
        vx = sum(ci * mom[q][2] + ci * (1 - ci) * mom[q][0] ** 2 for ci, q in zip(c, others))
        vy = sum(ci * mom[q][3] + ci * (1 - ci) * mom[q][1] ** 2 for ci, q in zip(c, others))
        cx_mu, cx_sd = (XX + mx) / n, max(np.sqrt(max(vx, 0.0)) / n, 1e-3)
        cy_mu, cy_sd = (YY + my) / n, max(np.sqrt(max(vy, 0.0)) / n, 1e-3)
        acc = np.zeros_like(XX, dtype=float)
        for R in Rs:
            acc += (_box_gauss(clue[0] - R, clue[0] + R, cx_mu, cx_sd)
                    * _box_gauss(clue[1] - R, clue[1] + R, cy_mu, cy_sd)) / (4.0 * R * R)
        rho += u[j] * F[j] * acc
    for i in range(k):
        rho[iy[i], ix[i]] = 0.0
    tot = rho.sum()
    if tot > 0 and rem > 0:
        rho *= rem / tot
    return gx, gy, rho, u
