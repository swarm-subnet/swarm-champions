from __future__ import annotations
import numpy as np
SEARCH_R_MIN, SEARCH_R_MAX = 5.0, 30.0   
_SQ2 = np.sqrt(2.0)
BANDS = {"city": (22.0, 45.0), "open": (28.0, 72.0), "mountain": (65.0, 100.0),
         "village": (28.0, 56.0), "forest": (22.0, 45.0)}
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
# candHt perf key "fast" (drone_agent _TUNED["perf"]): the clue term's _box_gauss for all search radii R in one call
# ((nR, len(mu)) arrays: the same elementwise operations on the same values, so the same numbers) instead of one call
# per R; the per-R outer products are still summed in the same order. Off = the shipped loop.
FAST = False


def _box_gauss_rows(c, Rs, mu, sd):
    """Row r = _box_gauss(c - Rs[r], c + Rs[r], mu, sd)."""
    R = np.asarray(Rs)[:, None]
    return _box_gauss(c - R, c + R, np.asarray(mu)[None, :], sd)
# fov_clue FCL (measurement only, default off): the env draws the camera FOV and the clue's y noise from the
# same RandomState(seed) stream (moving_drone.py:272 2nd draw; swarm_autopilot.py spawn_task_world 2nd draw),
# so clue_y - centroid_y = R * s with s = (FOV - 90) / 2. The agent sets s from its in-flight FOV estimate
# (FOVE); None keeps every code path below bit-identical to c7. sd = error sd of s; w = weight of the FOV
# likelihood in a mixture with the champion's uniform box (1 - w keeps a patched env recoverable).
CLUE_S = {"s": None, "sd": 0.02, "w": 1.0}


def _fcl_yfac(cy, R, cy_mu, cy_sd, dcy, dR):
    """p(clue_y | R, grid row): w * FOV Gaussian (cell-averaged over dcy, the grid row's width in centroid
    units; widened by the R-grid spacing dR, over which the mean moves by dR * s) + (1 - w) * the champion's
    uniform box / 2R."""
    s, sd_s, w = float(CLUE_S["s"]), float(CLUE_S["sd"]), float(CLUE_S["w"])
    sd2 = np.sqrt(cy_sd ** 2 + (R * sd_s) ** 2 + (dR * s) ** 2 / 12.0)
    m = cy - R * s
    half = 0.5 * dcy
    g = (_ndtr((m - cy_mu + half) / sd2) - _ndtr((m - cy_mu - half) / sd2)) / max(dcy, 1e-9)
    if w >= 1.0:
        return g
    return w * g + (1.0 - w) * _box_gauss(cy - R, cy + R, cy_mu, cy_sd) / (2.0 * R)
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
# bughunt VGP (default off: empty dict). Generator-exact village goal prior, conditional on the start.
# platform_placement.build_autopilot_world moves every village goal onto the 65-100 m ring around its
# (adjusted) start, sampled as angle ~ U(0, 2pi), r ~ triangular(65, 100, mode 65) (the preferred
# distance, 28-56 m, is clamped up to 65), rejected outside the +-40 box / on obstacles; when no ring
# spot is found the pad keeps its task_gen draw, 28-56 m from the original start. Measured on 5x1000
# local seeds: P(ring) = 1 - exp(-A / a0) with A the in-box area of the ring (m2), a0 ~ 21 (A < 1 m2:
# 0.3% ring, 1-20: 38%, 20-50: 84%, 50-100: 98%, > 100: 100%). The champion's band prior (57-79 m +-3,
# full empirical pdf only when the far corner is < 68 m) gives no mass to the fallback pads of starts
# with A > ~10 m2 nor to ring pads beyond 82 m: those goals are left unlanded 17% / 13% of the time vs
# 5% for the rest (village, c7 rows A-E). GEN_PRIOR[kind] = (a0, u, amax): u = uniform share of the fallback
# part (relocated starts: the fallback pad sits 28-56 m from the ORIGINAL start, unseen by the agent);
# amax > 0 applies the model only to starts with in-box ring area < amax m2 (others: champion prior).
GEN_PRIOR: dict = {}
_GEN_AREA: dict = {}


def _vgp_ring_area(s, W, lo=65.0, hi=100.0):
    """In-box area (m2) of the lo-hi ring around s, on a fixed 1 m lattice (cached per start)."""
    key = (round(float(s[0]), 3), round(float(s[1]), 3), float(W), lo, hi)
    a = _GEN_AREA.get(key)
    if a is None:
        xs = np.arange(-W + 0.5, W, 1.0)
        d = np.hypot(xs[None, :] - float(s[0]), xs[:, None] - float(s[1]))
        a = float(((d >= lo) & (d <= hi)).sum())
        if len(_GEN_AREA) > 256:
            _GEN_AREA.clear()
        _GEN_AREA[key] = a
    return a


def gen_applies(kind, s, W) -> bool:
    """GEN_PRIOR[kind] = (a0, u, amax): amax > 0 limits the generator prior to starts whose in-box ring
    area is below amax m2 (the others keep the champion prior; VGP-lite)."""
    spec = GEN_PRIOR.get(kind)
    if spec is None:
        return False
    amax = float(spec[2]) if len(spec) > 2 else 0.0
    return amax <= 0.0 or _vgp_ring_area(s, float(W)) < amax


def gen_prior(kind, gx, gy, s, W):
    """Per-cell goal mass (sums to 1) for a start under GEN_PRIOR[kind] (village placement model)."""
    a0, u = GEN_PRIOR[kind][0], GEN_PRIOR[kind][1]
    lo, hi = 65.0, 100.0
    r = np.hypot(gx[None, :] - float(s[0]), gy[:, None] - float(s[1]))
    box = ((np.abs(gx)[None, :] <= W) & (np.abs(gy)[:, None] <= W))
    ring = (r >= lo) & (r <= hi) & box
    f1 = np.where(ring, np.clip((hi - r) / (hi - lo), 0.0, None) / np.maximum(r, 0.5), 0.0)
    t1 = float(f1.sum())
    area = _vgp_ring_area(s, float(W), lo, hi)
    w = (1.0 - float(np.exp(-area / max(float(a0), 1e-6)))) if t1 > 0 else 0.0
    f2 = annulus_prior(gx, gy, s, 28.0, 56.0, W)
    t2 = float(f2.sum())
    nb = float(box.sum())
    ub = box / max(nb, 1.0)
    f2 = ((1.0 - float(u)) * f2 / t2 + float(u) * ub) if t2 > 0 else ub
    out = (1.0 - w) * f2
    if w > 0:
        out = out + w * f1 / t1
    return out
# Soft bands (research/tpl/band_city_forest, default off): {(kind, round(lo,1), round(hi,1)): pdf over the
# radial prior's bins}, the measured start->pad distance pdf of seeds flown in that template band, placement
# relocations included. Empty = today's truncated band. Installed by drone_agent from "band_pdf".
BAND_PDF: dict = {}
BAND_PDF_USED: dict = {}   # diag: {kind: calls} when a soft band pdf was used (cleared per episode by the agent)
# candHv VRAD (research/village_v3/BUILD_V.md; default off: VRAD["on"] False keeps every code path bit-identical).
# UID 149's village prior skips the radial prior (analytic 28-56 m band mixture only). With VRAD on (installed per
# episode by the router from router_attr village_radial) the village field uses an empirical start->pad distance pdf
# measured offline on worlds built by the public 5.1.6.3 generator (floor component inside the pdf file):
#   w >= 1: the empirical pdf alone, one field, never clipped by a band (whole pdf);
#   w <  1: a mixture of the empirical component (weight w) and the BAND_MIX village components (weight 1 - w), each
#           component re-weighted by the clue (pad_field) and by the pads already found (pad_field_found), exactly as
#           the band mixture weights its bands.
# The empirical component's box mask is min(W, WORLD[kind]): the router may lay the grid over +-42 m (world_est), and
# the empirical prior (no ray-exit test, unlike annulus_prior) must leave the cells beyond the +-40 m world empty.
VRAD = {"on": False, "edges": None, "pdf": None, "w": 1.0}
_VRAD_BAND = (-1.0, -1.0)   # band tag of the empirical component in the mixture (never a real distance band)


def _vrad_on(kind):
    return bool(VRAD["on"]) and kind == "village"


def _vrad_mix(kind, n=None):
    """The VRAD mixture: the empirical component, plus the band mixture's components with weight 1 - w."""
    w = float(VRAD["w"])
    if w >= 1.0:
        return ((_VRAD_BAND[0], _VRAD_BAND[1], 1.0),)
    try:
        from team.autopilot.geo_posterior import BAND_MIX, band_mix
        bands = tuple(band_mix(kind, n)) if kind in BAND_MIX else ((BANDS[kind][0], BANDS[kind][1], 1.0),)
    except Exception:
        bands = ((BANDS[kind][0], BANDS[kind][1], 1.0),)
    sw = sum(float(b[2]) for b in bands)
    if w <= 0.0 or sw <= 0.0:
        return ((_VRAD_BAND[0], _VRAD_BAND[1], 1.0),)
    return ((_VRAD_BAND[0], _VRAD_BAND[1], w),) + tuple((lo, hi, (1.0 - w) * float(bw) / sw) for lo, hi, bw in bands)


def _vrad_priors(gx, gy, S, idx, kind, W):
    """Per-start empirical VRAD priors (box min(W, WORLD[kind]))."""
    Wb = min(float(W), float(WORLD[kind]))
    edges, pdf = VRAD["edges"], VRAD["pdf"]
    return [empirical_prior(gx, gy, S[j], edges, pdf, Wb) for j in idx]


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
    if BAND_PDF:
        bp = BAND_PDF.get((kind, round(float(band[0]), 1), round(float(band[1]), 1)))
        if bp is not None and len(bp) == len(pdf):
            BAND_PDF_USED[kind] = BAND_PDF_USED.get(kind, 0) + 1
            return edges, bp
    lo, hi = band
    centers = 0.5 * (edges[:-1] + edges[1:])[: len(pdf)]
    m = (centers >= lo - BAND_MARGIN) & (centers <= hi + BAND_MARGIN)
    p2 = np.where(m, pdf[: len(centers)], 0.0)
    if p2.sum() <= 0:
        p2 = m.astype(float)
    return edges, p2 / max(p2.sum(), 1e-12)


_PERMS: dict = {}


def _perms(n, k):
    """All injective k-of-n index tuples (cached; read-only)."""
    key = (int(n), int(k))
    p = _PERMS.get(key)
    if p is None:
        from itertools import permutations
        p = np.asarray(list(permutations(range(n), k)), dtype=np.int64)
        p.setflags(write=False)
        _PERMS[key] = p
    return p


def _found_likelihood(ring_at):
    """Sum over injective found->start assignments of the product of prior densities."""
    k, n = ring_at.shape
    if k == 0:
        return 1.0
    if k > n:
        return 0.0
    perms = _perms(n, k)
    return float(np.prod(ring_at[np.arange(k)[None, :], perms], axis=1).sum())


def _has_mix(kind):
    if VRAD["on"] and kind == "village":      # candHv VRAD: the mixture path carries the empirical component
        return True
    try:
        from team.autopilot.geo_posterior import BAND_MIX
        return kind in BAND_MIX
    except Exception:
        return False


def _band_mix(kind, n=None):
    """Band mixture for maps listed in geo_posterior.BAND_MIX; the champion's single band otherwise.
    n: fleet size, for the band-by-n weights (geo_posterior.BAND_N; empty table = n-blind)."""
    if VRAD["on"] and kind == "village":      # candHv VRAD
        return _vrad_mix(kind, n)
    try:
        from team.autopilot.geo_posterior import BAND_MIX, band_mix
        if kind in BAND_MIX:
            return band_mix(kind, n)
    except Exception:
        pass
    return ((BANDS[kind][0], BANDS[kind][1], 1.0),)


def pad_field(starts, clue, kind, step=2.0, nR=25, found=(), W=None, radial=None, Wgrid=None):
    """Expected pad density, as a mixture over the seed template's shared distance bands.
    Wgrid (WPOST-X): the grid spans +-Wgrid while the box mask and priors use W; None = W."""
    if not _has_mix(kind):   # the champion's code path, unchanged
        return _pad_field_band(starts, clue, kind, step, nR, found, W, radial, None, raw=False, Wgrid=Wgrid)
    n = len(np.asarray(starts).reshape(-1, np.asarray(starts).shape[-1]))
    mix = _band_mix(kind, n)
    parts = []
    for lo, hi, w in mix:
        gx, gy, rho, tot = _pad_field_band(starts, clue, kind, step, nR, found, W, radial, (lo, hi), Wgrid=Wgrid)
        parts.append((w * tot, rho, tot))
    wsum = sum(p[0] for p in parts)
    if wsum <= 0:
        return _pad_field_band(starts, clue, kind, step, nR, found, W, radial, None, raw=False, Wgrid=Wgrid)
    out = np.zeros_like(parts[0][1])
    for wt, rho, tot in parts:
        if tot > 0 and wt > 0:
            out += (wt / wsum) * rho * (n / tot)
    return gx, gy, out


def _pad_field_band(starts, clue, kind, step=2.0, nR=25, found=(), W=None, radial=None, band=None, raw=True,
                    Wgrid=None):
    r_min, r_max = BANDS[kind] if band is None else band
    W = float(WORLD[kind] if W is None else W)
    Wg = W if Wgrid is None else float(Wgrid)
    starts = np.asarray(starts, float)[:, :2]
    found = np.asarray(found, float).reshape(-1, 2)
    n_total = len(starts)
    gx = np.arange(-Wg, Wg + 1e-9, step)
    gy = gx.copy()
    open_idx = [j for j in range(len(starts))]
    if band == _VRAD_BAND and _vrad_on(kind):      # candHv VRAD empirical component (whole pdf, no band clip)
        F = _vrad_priors(gx, gy, starts, open_idx, kind, W)
    elif radial is not None and kind in radial and kind != "village":
        F = [gen_prior(kind, gx, gy, starts[j], W) if gen_applies(kind, starts[j], W) else
             empirical_prior(gx, gy, starts[j], *_band_radial(radial, kind, starts[j], W, band), W) for j in open_idx]
    else:
        F = [gen_prior(kind, gx, gy, starts[j], W) if gen_applies(kind, starts[j], W) else
             annulus_prior(gx, gy, starts[j], r_min, r_max, W) for j in open_idx]
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
        # the clue term is separable: evaluate it on the 1-D axes (same values, ~50x fewer erf calls)
        cx_mu, cx_sd = (gx + mx) / n_total, max(np.sqrt(vx) / n_total, 1e-3)
        cy_mu, cy_sd = (gy + my) / n_total, max(np.sqrt(vy) / n_total, 1e-3)
        acc = np.zeros_like(XX, dtype=float)
        if CLUE_S.get("s") is None and FAST:
            _by = _box_gauss_rows(clue[1], Rs, cy_mu, cy_sd)
            _bx = _box_gauss_rows(clue[0], Rs, cx_mu, cx_sd)
            for _r, R in enumerate(Rs):
                acc += np.outer(_by[_r], _bx[_r]) / (4.0 * R * R)
        elif CLUE_S.get("s") is None:
            for R in Rs:
                acc += np.outer(_box_gauss(clue[1] - R, clue[1] + R, cy_mu, cy_sd),
                                _box_gauss(clue[0] - R, clue[0] + R, cx_mu, cx_sd)) / (4.0 * R * R)
        else:
            for R in Rs:
                acc += np.outer(_fcl_yfac(clue[1], R, cy_mu, cy_sd, step / n_total, Rs[1] - Rs[0]),
                                _box_gauss(clue[0] - R, clue[0] + R, cx_mu, cx_sd)) / (2.0 * R)
        rho += F[j] * acc
    tot = rho.sum()
    if raw:
        return gx, gy, rho, float(tot)
    if tot > 0:
        rho *= len(F) / tot
    return gx, gy, rho
def unfound_start_weights(ring_at_found):
    k, n = ring_at_found.shape
    if k == 0:
        return np.ones(n)
    if k > n:
        return np.zeros(n)
    perms = _perms(n, k)
    w = np.prod(ring_at_found[np.arange(k)[None, :], perms], axis=1)            
    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones(len(perms))                                                 
    used = np.zeros((len(perms), n))
    np.put_along_axis(used, perms, 1.0, axis=1)
    return 1.0 - (w[:, None] * used).sum(0) / w.sum()
def pad_field_found(starts, clue, kind, found_xy, step=4.0, nR=25, W=None, radial=None, Wgrid=None):
    """Density of the pads not yet found, as a band mixture: each band is weighted by its
    prior share, by how well it explains the clue, and by how well it explains the pads
    already found (all pads of a seed share one band). Wgrid: as in pad_field."""
    if not _has_mix(kind):   # the champion's code path, unchanged
        return _pad_field_found_band(starts, clue, kind, found_xy, step, nR, W, radial, None, raw=False, Wgrid=Wgrid)
    mix = _band_mix(kind, len(np.asarray(starts)))
    Y = np.asarray(found_xy, float).reshape(-1, 2)
    rem = len(np.asarray(starts)) - len(Y)
    parts = []
    for lo, hi, w in mix:
        gx, gy, rho, u, tot, flik = _pad_field_found_band(starts, clue, kind, found_xy, step, nR, W, radial, (lo, hi),
                                                          Wgrid=Wgrid)
        clue_lik = tot / max(float(np.sum(u)), 1e-9)
        parts.append((w * flik * clue_lik, rho, u, tot))
    wsum = sum(p[0] for p in parts)
    if wsum <= 0 or rem <= 0:
        return _pad_field_found_band(starts, clue, kind, found_xy, step, nR, W, radial, None, raw=False, Wgrid=Wgrid)
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


def _pad_field_found_band(starts, clue, kind, found_xy, step=4.0, nR=25, W=None, radial=None, band=None, raw=True,
                          Wgrid=None):
    r_min, r_max = BANDS[kind] if band is None else band
    W = float(WORLD[kind] if W is None else W)
    Wg = W if Wgrid is None else float(Wgrid)
    S = np.asarray(starts, float)[:, :2]
    Y = np.asarray(found_xy, float).reshape(-1, 2)
    n, k = len(S), len(Y)
    gx = np.arange(-Wg, Wg + 1e-9, step)
    gy = gx.copy()
    if band == _VRAD_BAND and _vrad_on(kind):      # candHv VRAD empirical component (whole pdf, no band clip)
        F = _vrad_priors(gx, gy, S, range(n), kind, W)
    elif radial is not None and kind in radial and kind != "village":
        F = [gen_prior(kind, gx, gy, S[j], W) if gen_applies(kind, S[j], W) else
             empirical_prior(gx, gy, S[j], *_band_radial(radial, kind, S[j], W, band), W) for j in range(n)]
    else:
        F = [gen_prior(kind, gx, gy, S[j], W) if gen_applies(kind, S[j], W) else
             annulus_prior(gx, gy, S[j], r_min, r_max, W) for j in range(n)]
    ix = np.clip(np.rint((Y[:, 0] - gx[0]) / step).astype(int), 0, len(gx) - 1) if k else np.zeros(0, int)
    iy = np.clip(np.rint((Y[:, 1] - gy[0]) / step).astype(int), 0, len(gy) - 1) if k else np.zeros(0, int)
    jx, jy = ix, iy
    if k and band == _VRAD_BAND and _vrad_on(kind):
        # candHv VRAD: a found pad estimated just outside the +-40 m world rounds into an empty box-edge cell (grid
        # +-42 m); read its start likelihood from the nearest in-box cell so the empirical component is not zeroed
        inb = np.nonzero(np.abs(gx) <= min(W, float(WORLD[kind])))[0]
        if len(inb):
            jx, jy = np.clip(ix, inb[0], inb[-1]), np.clip(iy, inb[0], inb[-1])
    ring_at = np.array([[F[j][jy[i], jx[i]] for j in range(n)] for i in range(k)]).reshape(k, n)
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
        cx_mu, cx_sd = (gx + mx) / n, max(np.sqrt(max(vx, 0.0)) / n, 1e-3)
        cy_mu, cy_sd = (gy + my) / n, max(np.sqrt(max(vy, 0.0)) / n, 1e-3)
        acc = np.zeros_like(XX, dtype=float)
        if CLUE_S.get("s") is None and FAST:
            _by = _box_gauss_rows(clue[1], Rs, cy_mu, cy_sd)
            _bx = _box_gauss_rows(clue[0], Rs, cx_mu, cx_sd)
            for _r, R in enumerate(Rs):
                acc += np.outer(_by[_r], _bx[_r]) / (4.0 * R * R)
        elif CLUE_S.get("s") is None:
            for R in Rs:
                acc += np.outer(_box_gauss(clue[1] - R, clue[1] + R, cy_mu, cy_sd),
                                _box_gauss(clue[0] - R, clue[0] + R, cx_mu, cx_sd)) / (4.0 * R * R)
        else:
            for R in Rs:
                acc += np.outer(_fcl_yfac(clue[1], R, cy_mu, cy_sd, step / n, Rs[1] - Rs[0]),
                                _box_gauss(clue[0] - R, clue[0] + R, cx_mu, cx_sd)) / (2.0 * R)
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
