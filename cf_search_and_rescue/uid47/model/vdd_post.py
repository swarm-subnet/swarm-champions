"""VDD post-processing shared by the offline eval, the flight eval and (later) the runtime: numpy only.

fires(heat, mask, dep, thr)  -> list of fires (peak cells >= thr after 3x3 NMS), each localised from the mask:
    pixels with mask logit > MLOG inside +-WIN px of the peak, planar depth within [p25 - 0.5, p25 + DBAND] m of the
    25th percentile, back-projected with the pose -> horizontal = median x,y (+ BACK m along the horizontal view ray, the
    body half-thickness), top z = max z over those points + ZBIAS.
Camera: eye = pos + 0.13 fwd + 0.05 up (swarm CAMERA_EYE_*), planar depth, pixel (i row, j col) centre ray
fwd + (128 - (j + .5))/128 tan(fov/2) lft + (128 - (i + .5))/128 tan(fov/2) up. fov: nominal 90 at runtime.
"""
import math
import numpy as np

EYE_F, EYE_U = 0.13, 0.05
DMIN, DSPAN = 0.5, 29.5
RULE = dict(MLOG=2.0, WIN=12, BLO=0.4, BHI=0.6, GROW=4, GSTEP=0.3, BACK=0.15, ZBIAS=0.0, NMS=3, MAXF=4)


def rot(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def backproject(ii, jj, z, pos, R, fov_deg):
    fwd, lft, up = R[:, 0], R[:, 1], R[:, 2]
    cam = np.asarray(pos, float) + EYE_F * fwd + EYE_U * up
    t = math.tan(math.radians(fov_deg) / 2.0)
    a = (128.0 - (jj + 0.5)) / 128.0 * t; b = (128.0 - (ii + 0.5)) / 128.0 * t
    return cam[None] + z[:, None] * (fwd[None] + a[:, None] * lft[None] + b[:, None] * up[None]), cam


def peaks(heat, thr, nms=3, maxf=4):
    """local maxima (nms x nms) >= thr, strongest first: list of (value, gy, gx)."""
    h = np.asarray(heat, np.float32)
    if h.max() < thr:
        return []
    p = np.pad(h, nms // 2, mode='constant', constant_values=-1)
    H, W = h.shape
    mx = np.max(np.stack([p[dy:dy + H, dx:dx + W] for dy in range(nms) for dx in range(nms)]), 0)
    ys, xs = np.where((h >= thr) & (h >= mx))
    o = np.argsort(-h[ys, xs])[:maxf]
    return [(float(h[ys[k], xs[k]]), int(ys[k]), int(xs[k])) for k in o]


def _mode_depth(z, half=0.3):
    """the depth with the most pixels within +-half m (a standing body is a near-constant-depth column; the terrain
    the stride-2 mask bleeds onto is not)."""
    zs = np.sort(z)
    cnt = np.searchsorted(zs, zs + half, 'right') - np.searchsorted(zs, zs - half, 'left')
    return float(zs[int(np.argmax(cnt))])


def _window(mask, gy, gx, W, mlog):
    cy, cx = gy * 4 + 2, gx * 4 + 2
    y0, y1 = max(0, cy - W), min(256, cy + W + 1); x0, x1 = max(0, cx - W), min(256, cx + W + 1)
    m2 = mask[y0 // 2:(y1 + 1) // 2, x0 // 2:(x1 + 1) // 2] > mlog
    mm = np.repeat(np.repeat(m2, 2, 0), 2, 1)[(y0 % 2):(y0 % 2) + (y1 - y0), (x0 % 2):(x0 % 2) + (x1 - x0)]
    return mm, y0, y1, x0, x1


def localise(gy, gx, mask, dep_m, pos, R, fov_deg, rule=RULE):
    """mask: (128,128) logits; dep_m: (256,256) metric planar depth.
    Pixels: mask logit > MLOG (fallback > 0) inside +-WIN px of the peak; body depth d0 = mode of their depths (+-0.3 m);
    kept = depth in [d0 - BLO, d0 + BHI]. x, y = median of the kept pixels back-projected, pushed BACK m further along
    the horizontal view ray (visible front surface -> body centre). top = max z over the kept pixels grown upward
    <= GROW px (depth step <= GSTEP, same band; the stride-2 mask erodes the head) + one pixel of vertical extent
    (z tan(fov/2)/128) + ZBIAS. Rule chosen on held-out fold 0 (loc_study2.py, config G15)."""
    zz_all = None
    mm, y0, y1, x0, x1 = _window(mask, gy, gx, rule['WIN'], rule['MLOG'])
    zz = dep_m[y0:y1, x0:x1]
    sel = mm & (zz < DMIN + DSPAN - 0.05)
    if not sel.any():
        mm, y0, y1, x0, x1 = _window(mask, gy, gx, rule['WIN'], 0.0)
        sel = mm & (zz < DMIN + DSPAN - 0.05)
    t = math.tan(math.radians(fov_deg) / 2.0)
    if not sel.any():
        blk = dep_m[gy * 4:gy * 4 + 4, gx * 4:gx * 4 + 4]
        k = np.unravel_index(np.argmin(blk), blk.shape)
        ii = np.array([gy * 4 + k[0]], float); jj = np.array([gx * 4 + k[1]], float); z = np.array([blk[k]], float)
        gi, gj, gz = ii, jj, z
        n = 0
    else:
        ii, jj = np.where(sel); z = zz[sel].astype(float)
        d0 = _mode_depth(z)
        lo, hi = d0 - rule['BLO'], d0 + rule['BHI']
        keep = (z >= lo) & (z <= hi)
        ii = (ii[keep] + y0).astype(float); jj = (jj[keep] + x0).astype(float); z = z[keep]
        ai, aj, az = [], [], []
        for c in np.unique(jj):
            ci = int(c); r = int(ii[jj == c].min()) - 1; k = 0; dprev = dep_m[r + 1, ci]
            while r >= 0 and k < rule['GROW']:
                d = dep_m[r, ci]
                if lo <= d <= hi and abs(d - dprev) <= rule['GSTEP']:
                    ai.append(r); aj.append(ci); az.append(d); r -= 1; k += 1; dprev = d
                else:
                    break
        if ai:
            gi = np.concatenate([ii, np.asarray(ai, float)]); gj = np.concatenate([jj, np.asarray(aj, float)]); gz = np.concatenate([z, np.asarray(az, float)])
        else:
            gi, gj, gz = ii, jj, z
        n = int(sel.sum())
    P, cam = backproject(ii, jj, z, pos, R, fov_deg)
    x, y = float(np.median(P[:, 0])), float(np.median(P[:, 1]))
    if rule['BACK']:
        dx, dy = x - cam[0], y - cam[1]; h = math.hypot(dx, dy)
        if h > 1e-3:
            x += rule['BACK'] * dx / h; y += rule['BACK'] * dy / h
    Pg, _ = backproject(gi, gj, gz, pos, R, fov_deg)
    kz = int(np.argmax(Pg[:, 2]))
    top = float(Pg[kz, 2] + gz[kz] * t / 128.0) + rule['ZBIAS']
    return dict(x=x, y=y, top=top, zmed=float(np.median(z)), npx=n, cam=cam)


def fires(heat, mask, dep_norm, pos, rpy, thr, fov_deg=90.0, rule=RULE):
    dep_m = np.asarray(dep_norm, np.float32).reshape(256, 256) * DSPAN + DMIN
    R = rot(rpy)
    out = []
    for v, gy, gx in peaks(heat, thr, rule['NMS'], rule['MAXF']):
        L = localise(gy, gx, mask, dep_m, pos, R, fov_deg, rule)
        L.update(p=v, gy=gy, gx=gx); out.append(L)
    return out
