import math
import numpy as np
DMIN, DSPAN = (0.5, 29.5)
EYE_F, EYE_U = (0.13, 0.05)

def rot(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])

def _cam(pos, R):
    return np.asarray(pos, float) + EYE_F * R[:, 0] + EYE_U * R[:, 2]
_GRID = None

def _grid(step):
    ii, jj = np.meshgrid(np.arange(step // 2, 256, step), np.arange(step // 2, 256, step), indexing='ij')
    return (ii.ravel(), jj.ravel())

def pair_cost(dA, poseA, dB, poseB, fovs, step=6, trunc=0.5):
    mA = np.asarray(dA, np.float32) * DSPAN + DMIN
    mB = np.asarray(dB, np.float32) * DSPAN + DMIN
    RA = rot(poseA[1])
    RB = rot(poseB[1])
    cA = _cam(poseA[0], RA)
    cB = _cam(poseB[0], RB)
    ii, jj = _grid(step)
    z = mB[ii, jj].astype(np.float64)
    gy = np.abs(mB[np.minimum(ii + 1, 255), jj] - mB[np.maximum(ii - 1, 0), jj])
    gx = np.abs(mB[ii, np.minimum(jj + 1, 255)] - mB[ii, np.maximum(jj - 1, 0)])
    ok = (z < 28.0) & (z > 1.0) & (gy < 0.08 * z) & (gx < 0.08 * z)
    ii, jj, z = (ii[ok], jj[ok], z[ok])
    if len(z) < 50:
        return (None, 0)
    out = []
    for fov in fovs:
        t = math.tan(math.radians(fov) / 2)
        a = (128.0 - (jj + 0.5)) / 128.0 * t
        b = (128.0 - (ii + 0.5)) / 128.0 * t
        P = cB[None] + z[:, None] * (RB[:, 0][None] + a[:, None] * RB[:, 1][None] + b[:, None] * RB[:, 2][None])
        q = (P - cA[None]) @ RA
        zc = q[:, 0]
        good = zc > 1.0
        u = 128.0 - 128.0 * q[:, 1] / np.maximum(zc, 0.001) / t - 0.5
        v = 128.0 - 128.0 * q[:, 2] / np.maximum(zc, 0.001) / t - 0.5
        good &= (u >= 0) & (u <= 255) & (v >= 0) & (v <= 255)
        if good.sum() < 50:
            out.append(np.nan)
            continue
        ui = np.clip(np.round(u[good]).astype(int), 0, 255)
        vi = np.clip(np.round(v[good]).astype(int), 0, 255)
        dz = mA[vi, ui] - zc[good]
        vis = mA[vi, ui] < 29.4
        r = np.minimum(np.abs(dz[vis]), trunc)
        out.append(float(r.mean()) if len(r) > 50 else np.nan)
    return (np.array(out), int(ok.sum()))

def estimate(frames, fovs=None, min_move=0.8, min_rot_deg=4.0, max_pairs=12, lag=10):
    if fovs is None:
        fovs = np.arange(87.5, 92.55, 0.1)
    est = []
    for i in range(0, len(frames) - lag, max(1, lag // 2)):
        A = frames[i]
        B = frames[i + lag]
        mv = float(np.linalg.norm(np.asarray(A[1]) - np.asarray(B[1])))
        dyaw = abs(math.degrees((B[2][2] - A[2][2] + math.pi) % (2 * math.pi) - math.pi))
        if mv < min_move and dyaw < min_rot_deg:
            continue
        c, n = pair_cost(A[0], (A[1], A[2]), B[0], (B[1], B[2]), fovs)
        if c is None or not np.isfinite(c).any():
            continue
        k = int(np.nanargmin(c))
        f = fovs[k]
        if 0 < k < len(fovs) - 1 and np.isfinite(c[k - 1]) and np.isfinite(c[k + 1]):
            den = c[k - 1] - 2 * c[k] + c[k + 1]
            if den > 0:
                f = fovs[k] + 0.5 * (c[k - 1] - c[k + 1]) / den * (fovs[1] - fovs[0])
        est.append(float(f))
        if len(est) >= max_pairs:
            break
    if not est:
        return (None, 0, [])
    return (float(np.median(est)), len(est), est)

def estimate_joint(frames, fovs=None, min_move=0.8, min_rot_deg=4.0, max_pairs=16, lag=10):
    if fovs is None:
        fovs = np.arange(87.5, 92.55, 0.1)
    tot = np.zeros(len(fovs))
    n = 0
    for i in range(0, len(frames) - lag, max(1, lag // 2)):
        A = frames[i]
        B = frames[i + lag]
        mv = float(np.linalg.norm(np.asarray(A[1]) - np.asarray(B[1])))
        dyaw = abs(math.degrees((B[2][2] - A[2][2] + math.pi) % (2 * math.pi) - math.pi))
        if mv < min_move and dyaw < min_rot_deg:
            continue
        c, _ = pair_cost(A[0], (A[1], A[2]), B[0], (B[1], B[2]), fovs)
        if c is None or not np.isfinite(c).all():
            continue
        tot += c
        n += 1
        if n >= max_pairs:
            break
    if n == 0:
        return (None, 0, 0.0)
    k = int(np.argmin(tot))
    f = float(fovs[k])
    curv = 0.0
    if 0 < k < len(fovs) - 1:
        den = tot[k - 1] - 2 * tot[k] + tot[k + 1]
        if den > 0:
            f = float(fovs[k] + 0.5 * (tot[k - 1] - tot[k + 1]) / den * (fovs[1] - fovs[0]))
            curv = float(den / n)
    return (f, n, curv)
