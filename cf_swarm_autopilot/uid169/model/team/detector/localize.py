from __future__ import annotations
import numpy as np
CAM_FWD = 0.13
CAM_UP = 0.05
FOV_DEG = 90.0          
RES = 256
DEPTH_MIN_M = 0.5       
DEPTH_MAX_M = 30.0      
def rot_from_quat(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])
def rot_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])
def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v
def pixel_to_world_ray(R, u, v, fov_deg=FOV_DEG, res=RES):
    f = _norm(R @ np.array([1.0, 0.0, 0.0]))
    up = R @ np.array([0.0, 0.0, 1.0])
    r = _norm(np.cross(f, up))
    t = np.cross(r, f)
    half = np.tan(np.radians(fov_deg) * 0.5)
    a = (2.0 * (u + 0.5) / res - 1.0) * half           
    b = (1.0 - 2.0 * (v + 0.5) / res) * half           
    return _norm(f + a * r + b * t), f, up
def backproject(pos, R, u, v, D, fov_deg=FOV_DEG, res=RES):
    pos = np.asarray(pos, float)
    ray, f, up = pixel_to_world_ray(R, u, v, fov_deg, res)
    cam = pos + CAM_FWD * f + CAM_UP * up
    return cam + D * ray
def depth_to_meters(n, depth_max: float = DEPTH_MAX_M, depth_min: float = DEPTH_MIN_M):
    return np.asarray(n, float) * (float(depth_max) - float(depth_min)) + float(depth_min)
def range_from_box_depth(depth_img, box, pct=10.0, pad=2):
    d = np.asarray(depth_img)
    if d.ndim == 3:
        d = d[..., 0]
    H, W = d.shape
    u0, v0, u1, v1 = box
    c0 = max(0, int(u0) - pad); c1 = min(W, int(np.ceil(u1)) + pad + 1)
    r0 = max(0, int(v0) - pad); r1 = min(H, int(np.ceil(v1)) + pad + 1)
    patch = d[r0:r1, c0:c1]
    if patch.size == 0:
        return None
    return float(depth_to_meters(np.percentile(patch, pct)))
def range_from_box_size(box, res=RES, fov_deg=FOV_DEG, victim_h=1.2):
    h_px = max(1.0, float(box[3]) - float(box[1]))
    focal = (res * 0.5) / np.tan(np.radians(fov_deg) * 0.5)
    return victim_h * focal / h_px
def robust_range(depth_img, box, res=RES, fov_deg=FOV_DEG, victim_h=1.2, tol=2.2):
    dr = range_from_box_depth(depth_img, box)
    if dr is None:
        return None
    br = range_from_box_size(box, res, fov_deg, victim_h)
    if dr < br / tol or dr > br * tol:
        return None
    return dr
VICTIM_HALF_H = 0.9     
MAX_H_ABOVE_GROUND = 3.5  
def local_ground_z(pos, R, box, depth_img, *, fov_deg=FOV_DEG, res=RES, pad=3):
    d = np.asarray(depth_img)
    if d.ndim == 3:
        d = d[..., 0]
    H, W = d.shape
    u0, v0, u1, v1 = box
    r0 = int(np.clip(v1 + 1, 0, H - 1)); r1 = int(np.clip(v1 + 1 + pad * 2, 0, H))
    c0 = int(np.clip(u0 - pad, 0, W)); c1 = int(np.clip(u1 + pad + 1, 0, W))
    if r1 <= r0 or c1 <= c0:
        return None
    step = max(1, (c1 - c0) // 32)
    cols = np.arange(c0, c1, step)
    rows = np.arange(r0, r1)
    rng = depth_to_meters(d[np.ix_(rows, cols)])
    ok = (rng < DEPTH_MAX_M - 0.25) & (rng > DEPTH_MIN_M + 1e-3)
    if int(ok.sum()) < 4:
        return None
    f = _norm(R @ np.array([1.0, 0.0, 0.0]))
    up = R @ np.array([0.0, 0.0, 1.0])
    right = _norm(np.cross(f, up))
    t = np.cross(right, f)
    half = np.tan(np.radians(fov_deg) * 0.5)
    a = (2.0 * (cols + 0.5) / res - 1.0) * half
    b = (1.0 - 2.0 * (rows + 0.5) / res) * half
    dirs = (f[None, None, :] + a[None, :, None] * right[None, None, :]
            + b[:, None, None] * t[None, None, :])
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    cam_z = float(np.asarray(pos, float)[2] + CAM_FWD * f[2] + CAM_UP * up[2])
    zs = cam_z + rng * dirs[..., 2]
    return float(np.median(zs[ok]))
def localise(pos, R, u, v, depth_img, box, agl, *, fov_deg=FOV_DEG, res=RES,
             altimeter_max=20.0, ground_tol=(0.5, 2.0)):
    rng = range_from_box_depth(depth_img, box)
    ray, f, up = pixel_to_world_ray(R, u, v, fov_deg, res)
    pos = np.asarray(pos, float)
    cam = pos + CAM_FWD * f + CAM_UP * up
    ground_z = (float(pos[2]) - float(agl)
                if (agl is not None and agl < altimeter_max - 1e-3) else None)
    if ground_z is None:
        ground_z = local_ground_z(pos, R, box, depth_img, fov_deg=fov_deg, res=res)
    known_ground = ground_z is not None
    if rng is not None:
        a = cam + rng * ray
        if known_ground and a[2] - ground_z > MAX_H_ABOVE_GROUND:
            return None                                  
        br = range_from_box_size(box, res, fov_deg)
        if rng < br / 2.2 or rng > br * 2.2:
            rng = None                                   
    if known_ground and ray[2] < -1e-3:
        t = (ground_z + VICTIM_HALF_H - cam[2]) / ray[2]
        if 0.0 < t < DEPTH_MAX_M * 1.5 and (rng is None
                                            or ground_tol[0] < t / rng < ground_tol[1]):
            return cam + t * ray
    if rng is None:
        return None
    pt = cam + rng * ray
    if known_ground:
        pt[2] = ground_z + VICTIM_HALF_H     
    return pt
def hover_target(victim, h_confirm=3.5):
    v = np.asarray(victim, float)
    return np.array([v[0], v[1], v[2] + h_confirm])
RANGE_FLOOR = 5.0
def _weighted_median(values, weights):
    idx = np.argsort(values)
    v, w = values[idx], weights[idx]
    cw = np.cumsum(w)
    return float(v[min(len(v) - 1, int(np.searchsorted(cw, cw[-1] * 0.5)))])
class VictimEstimator:
    def __init__(self, max_keep=400, power=1.0):
        self.vs: list = []
        self.ws: list = []
        self.est = None
        self.max_keep = max_keep
        self.power = float(power)
    @property
    def n(self):
        return len(self.vs)
    def update(self, victim_world, conf=1.0, rng=10.0):
        self.vs.append(np.asarray(victim_world, float))
        self.ws.append(float(conf) / max(float(rng), RANGE_FLOOR) ** self.power)
        if len(self.vs) > self.max_keep:
            self.vs.pop(0); self.ws.pop(0)
        V = np.asarray(self.vs); W = np.asarray(self.ws)
        self.est = np.array([_weighted_median(V[:, k], W) for k in range(3)])
        return self.est
class Candidate(VictimEstimator):
    def __init__(self, max_keep=80, power=2.0):
        super().__init__(max_keep=max_keep, power=power)
        self.rgb_confirms = 0
        self.rgb_rejects = 0
    @property
    def colour_score(self) -> int:
        return self.rgb_confirms - self.rgb_rejects
    @property
    def evidence(self) -> float:
        return 10.0 * self.rgb_confirms - 1.0 * self.rgb_rejects + min(self.n, 40) / 10.0
class CandidateTracker:
    def __init__(self, radius=5.0, max_keep=80, power=2.0, max_candidates=24):
        self.radius = float(radius)
        self.max_keep, self.power = max_keep, power
        self.max_candidates = max_candidates
        self.cands: list = []
    def nearest(self, pt, radius=None):
        best, bd = None, (self.radius if radius is None else float(radius))
        for c in self.cands:
            if c.est is None:
                continue
            d = float(np.linalg.norm(np.asarray(pt, float) - c.est))
            if d <= bd:
                best, bd = c, d
        return best
    def add(self, pt, conf=1.0, rng=10.0, from_rgb=False):
        c = self.nearest(pt)
        if c is None:
            c = Candidate(max_keep=self.max_keep, power=self.power)
            self.cands.append(c)
            if len(self.cands) > self.max_candidates:
                self.cands.sort(key=lambda x: (x.colour_score, x.n))
                self.cands.pop(0)
        c.update(pt, conf=conf, rng=rng)
        if from_rgb:
            c.rgb_confirms += 1
        return c
    def reject(self, pt, conf=1.0, rng=10.0, radius=2.0):
        c = self.nearest(pt, radius=radius)
        if c is None:
            c = Candidate(max_keep=self.max_keep, power=self.power)
            self.cands.append(c)
            if len(self.cands) > self.max_candidates:
                self.cands.sort(key=lambda x: (x.colour_score, x.n))
                self.cands.pop(0)
        c.update(pt, conf=conf, rng=rng)
        c.rgb_rejects += 1
        return c
    def best(self):
        live = [c for c in self.cands if c.est is not None]
        if not live:
            return None
        return max(live, key=lambda c: c.evidence)
    def drop(self, cand) -> None:
        if cand in self.cands:
            self.cands.remove(cand)
class ApproachFSM:
    SEARCH, APPROACH, HOVER = "search", "approach", "hover"
    def __init__(self, commit_m=8, commit_n=5, hold_steps=200,
                 confirm_horiz=2.0, blacklist_r=6.0):
        self.state = self.SEARCH
        self.det_hist: list[bool] = []
        self.commit_m, self.commit_n = commit_m, commit_n
        self.hold_steps, self.confirm_horiz, self.blacklist_r = hold_steps, confirm_horiz, blacklist_r
        self.hold = 0
        self.blacklist: list[np.ndarray] = []
    def _blacklisted(self, xy):
        return any(np.linalg.norm(np.asarray(xy) - b) < self.blacklist_r for b in self.blacklist)
    def step(self, detected: bool, victim_est, drone_pos):
        self.det_hist = (self.det_hist + [bool(detected)])[-self.commit_m:]
        if self.state == self.SEARCH:
            if (sum(self.det_hist) >= self.commit_n and victim_est is not None
                    and not self._blacklisted(victim_est[:2])):
                self.state, self.hold = self.APPROACH, 0
        elif self.state == self.APPROACH:
            if victim_est is None or self._blacklisted(victim_est[:2]):
                self.state = self.SEARCH
            else:
                horiz = np.linalg.norm(np.asarray(drone_pos)[:2] - np.asarray(victim_est)[:2])
                if horiz <= self.confirm_horiz:
                    self.state, self.hold = self.HOVER, 0
        elif self.state == self.HOVER:
            self.hold += 1
            if self.hold >= self.hold_steps:            
                if victim_est is not None:
                    self.blacklist.append(np.asarray(victim_est)[:2].copy())
                self.state, self.hold = self.SEARCH, 0
        return self.state
