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
    out = {}
    for k, v in raw.items():
        entry = (np.asarray(v["edges"], float), np.asarray(v["pdf"], float))
        if v.get("cond"):
            out[k + "_cond"] = [(float(lo), float(hi), np.asarray(pdf, float)) for lo, hi, pdf in v["cond"]]
        out[k] = entry
    return out
def _radial_for_start(radial, kind, s, W):
    """(edges, pdf) for this start: the class conditional on its distance to the nearest wall when the prior has one."""
    edges, pdf = radial[kind]
    cond = radial.get(kind + "_cond")
    if cond:
        edge = float(W) - float(np.max(np.abs(np.asarray(s, float)[:2])))
        for lo, hi, cpdf in cond:
            if lo < edge <= hi:
                return edges, cpdf
    return edges, pdf
def empirical_prior_aniso(gx, gy, s, edges, pdf, W, r_min):
    """Empirical radial pdf with the generator's angular structure: the angle is uniform, so
    every feasible direction carries the same mass however short its box-limited range is, and
    directions whose range cannot reach r_min carry none (the generator resamples them)."""
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
    rhi = np.minimum(np.minimum(mx, my), edges[-1])
    k = np.clip(np.searchsorted(edges, r, side="right") - 1, 0, len(pdf) - 1)
    cdf = np.concatenate([[0.0], np.cumsum(pdf)])
    khi = np.clip(np.searchsorted(edges, rhi, side="right") - 1, 0, len(pdf) - 1)
    kmin = int(np.clip(np.searchsorted(edges, r_min, side="right") - 1, 0, len(pdf) - 1))
    Z = np.maximum(cdf[khi + 1] - cdf[kmin], 1e-9)          # empirical mass reachable along this direction
    ok = (rhi >= r_min) & (r <= rhi) & (r >= 0.5)
    dens = np.where(ok, pdf[k] / Z / np.maximum(r, 0.5), 0.0)
    dens *= ((np.abs(gx)[None, :] <= W) & (np.abs(gy)[:, None] <= W))
    tot = dens.sum()
    return dens / tot if tot > 0 else dens
ANISO = {"on": False}
_PRIOR_GEOM: dict = {}    # per (grid, start, edges) geometry, reused across the bands of a mixture


def _prior_geom(gx, gy, s, edges, npdf, W):
    key = (len(gx), float(gx[0]), float(gx[-1]), len(gy), float(gy[0]), float(gy[-1]),
           float(s[0]), float(s[1]), float(W), int(npdf), len(edges), float(edges[0]), float(edges[-1]))
    hit = _PRIOR_GEOM.get(key)
    if hit is None:
        r = np.hypot(gx[None, :] - s[0], gy[:, None] - s[1])
        k = np.clip(np.searchsorted(edges, r, side="right") - 1, 0, npdf - 1)
        inside = r < edges[-1]
        denom = np.maximum(r, 0.5)
        box = ((np.abs(gx)[None, :] <= W) & (np.abs(gy)[:, None] <= W))
        if len(_PRIOR_GEOM) > 64:
            _PRIOR_GEOM.clear()
        hit = _PRIOR_GEOM[key] = (k, inside, denom, box)
    return hit


def empirical_prior(gx, gy, s, edges, pdf, W):
    if ANISO.get("on"):
        kind = ANISO.get("kind")
        r_min = ANISO.get("r_min", {}).get(kind, 0.0)
        return empirical_prior_aniso(gx, gy, s, edges, pdf, W, float(r_min))
    k, inside, denom, box = _prior_geom(gx, gy, s, edges, len(pdf), W)
    dens = np.where(inside, pdf[k], 0.0) / denom    # same order as the uncached form, to the last bit
    dens *= box
    tot = dens.sum()
    return dens / tot if tot > 0 else dens
def _moments(gx, gy, f):
    px, py = f.sum(0), f.sum(1)
    mx = float((gx * px).sum())
    my = float((gy * py).sum())
    vx = max(float((gx ** 2 * px).sum()) - mx * mx, 1e-6)
    vy = max(float((gy ** 2 * py).sum()) - my * my, 1e-6)
    return mx, my, vx, vy
BAND_MARGIN = 3.0   # pad placement nudges a goal a few metres off its planned radius
# Start-conditional band (default off, {kind: metres}). Village placement samples the 65-100 m ring
# inside the +-W box; when a start's farthest box corner is nearer than that, no ring spot exists and
# the pad falls back to its drawn 28-56 m position (100% of pads below 65 m, 73% at 65-68 m, 14% at
# 68-72 m). For such starts keep the untruncated empirical pdf instead of the band.
BAND_FC_MIN: dict = {}


def _band_radial(radial, kind, s, W, band):
    """The empirical radial pdf for this start, restricted to one distance band (plus margin)."""
    edges, pdf = _radial_for_start(radial, kind, s, W)
    if band is None:
        return edges, pdf
    fc = BAND_FC_MIN.get(kind)
    if fc is not None and W is not None:
        far = float(np.hypot(float(W) + abs(float(s[0])), float(W) + abs(float(s[1]))))
        if far < float(fc):
            return edges, pdf
    lo, hi = band
    centers = 0.5 * (edges[:-1] + edges[1:])[: len(pdf)]
    m = (centers >= lo - BAND_MARGIN) & (centers <= hi + BAND_MARGIN)
    p2 = np.where(m, pdf[: len(centers)], 0.0)
    if p2.sum() <= 0:
        p2 = m.astype(float)
    return edges, p2 / max(p2.sum(), 1e-12)


def _found_likelihood(ring_at):
    """Sum over injective found->start assignments of the product of prior densities."""
    from itertools import permutations
    k, n = ring_at.shape
    if k == 0:
        return 1.0
    if k > n:
        return 0.0
    perms = np.asarray(list(permutations(range(n), k)), dtype=np.int64)
    return float(np.prod(ring_at[np.arange(k)[None, :], perms], axis=1).sum())


def _has_mix(kind):
    try:
        from team.autopilot.geo_posterior import BAND_MIX
        return kind in BAND_MIX
    except Exception:
        return False


def _band_mix(kind):
    """Band mixture for maps listed in geo_posterior.BAND_MIX; the champion's single band otherwise."""
    try:
        from team.autopilot.geo_posterior import BAND_MIX, band_mix
        if kind in BAND_MIX:
            return band_mix(kind)
    except Exception:
        pass
    return ((BANDS[kind][0], BANDS[kind][1], 1.0),)


def pad_field(starts, clue, kind, step=2.0, nR=25, found=(), W=None, radial=None):
    """Expected pad density, as a mixture over the seed template's shared distance bands."""
    if not _has_mix(kind):   # the champion's code path, unchanged
        return _pad_field_band(starts, clue, kind, step, nR, found, W, radial, None, raw=False)
    mix = _band_mix(kind)
    n = len(np.asarray(starts).reshape(-1, np.asarray(starts).shape[-1]))
    parts = []
    for lo, hi, w in mix:
        gx, gy, rho, tot = _pad_field_band(starts, clue, kind, step, nR, found, W, radial, (lo, hi))
        parts.append((w * tot, rho, tot))
    wsum = sum(p[0] for p in parts)
    if wsum <= 0:
        return _pad_field_band(starts, clue, kind, step, nR, found, W, radial, None, raw=False)
    out = np.zeros_like(parts[0][1])
    for wt, rho, tot in parts:
        if tot > 0 and wt > 0:
            out += (wt / wsum) * rho * (n / tot)
    return gx, gy, out


def _pad_field_band(starts, clue, kind, step=2.0, nR=25, found=(), W=None, radial=None, band=None, raw=True):
    r_min, r_max = BANDS[kind] if band is None else band
    W = float(WORLD[kind] if W is None else W)
    starts = np.asarray(starts, float)[:, :2]
    found = np.asarray(found, float).reshape(-1, 2)
    n_total = len(starts)
    gx = np.arange(-W, W + 1e-9, step)
    gy = gx.copy()
    open_idx = [j for j in range(len(starts))]
    if radial is not None and kind in radial:
        F = [empirical_prior(gx, gy, starts[j], *_band_radial(radial, kind, starts[j], W, band), W) for j in open_idx]
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
    if raw:
        return gx, gy, rho, float(tot)
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
    """Density of the pads not yet found, as a band mixture: each band is weighted by its
    prior share, by how well it explains the clue, and by how well it explains the pads
    already found (all pads of a seed share one band)."""
    if not _has_mix(kind):   # the champion's code path, unchanged
        return _pad_field_found_band(starts, clue, kind, found_xy, step, nR, W, radial, None, raw=False)
    mix = _band_mix(kind)
    Y = np.asarray(found_xy, float).reshape(-1, 2)
    rem = len(np.asarray(starts)) - len(Y)
    parts = []
    for lo, hi, w in mix:
        gx, gy, rho, u, tot, flik = _pad_field_found_band(starts, clue, kind, found_xy, step, nR, W, radial, (lo, hi))
        clue_lik = tot / max(float(np.sum(u)), 1e-9)
        parts.append((w * flik * clue_lik, rho, u, tot))
    wsum = sum(p[0] for p in parts)
    if wsum <= 0 or rem <= 0:
        return _pad_field_found_band(starts, clue, kind, found_xy, step, nR, W, radial, None, raw=False)
    out = np.zeros_like(parts[0][1])
    u_mix = np.zeros_like(parts[0][2], dtype=float)
    for wt, rho, u, tot in parts:
        if wt <= 0:
            continue
        s = float(rho.sum())
        if s > 0:
            out += (wt / wsum) * rho * (rem / s)
        u_mix += (wt / wsum) * u
    return gx, gy, out, u_mix


def _pad_field_found_band(starts, clue, kind, found_xy, step=4.0, nR=25, W=None, radial=None, band=None, raw=True):
    r_min, r_max = BANDS[kind] if band is None else band
    W = float(WORLD[kind] if W is None else W)
    S = np.asarray(starts, float)[:, :2]
    Y = np.asarray(found_xy, float).reshape(-1, 2)
    n, k = len(S), len(Y)
    gx = np.arange(-W, W + 1e-9, step)
    gy = gx.copy()
    if radial is not None and kind in radial:
        F = [empirical_prior(gx, gy, S[j], *_band_radial(radial, kind, S[j], W, band), W) for j in range(n)]
    else:
        F = [annulus_prior(gx, gy, S[j], r_min, r_max, W) for j in range(n)]
    ix = np.clip(np.rint((Y[:, 0] - gx[0]) / step).astype(int), 0, len(gx) - 1) if k else np.zeros(0, int)
    iy = np.clip(np.rint((Y[:, 1] - gy[0]) / step).astype(int), 0, len(gy) - 1) if k else np.zeros(0, int)
    ring_at = np.array([[F[j][iy[i], ix[i]] for j in range(n)] for i in range(k)]).reshape(k, n)
    flik = _found_likelihood(ring_at)
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
    tot_clue = float(rho.sum())
    for i in range(k):
        rho[iy[i], ix[i]] = 0.0
    if raw:
        return gx, gy, rho, u, tot_clue, flik
    tot = rho.sum()
    if tot > 0 and rem > 0:
        rho *= rem / tot
    return gx, gy, rho, u
