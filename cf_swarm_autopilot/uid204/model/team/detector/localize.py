"""Victim localisation: back-project a detected victim to world coords, then accumulate.

Deploy-side (numpy only, no sim). The detector gives a centroid pixel (u,v) and the depth
image gives a range D; the drone's own pose comes from the state vector. Camera tilt is
handled exactly by using the drone's rotation matrix — the hard part is per-frame noise,
so `VictimEstimator` averages the (static) victim's world position across frames.

Camera model mirrors MovingDroneAviary exactly: the camera looks down the drone's body-X
(forward) at offset +0.13 fwd, +0.05 up; square image; FOV = 90 deg (the sim jitters it
+/-2 deg per episode and doesn't expose it, so 90 is the best assumption — the residual is
a <2% ray-angle error, worst at the frame edge).
"""
from __future__ import annotations

import numpy as np

CAM_FWD = 0.13
CAM_UP = 0.05
FOV_DEG = 90.0          # sim base; per-episode jitter is +/-2 deg and unobservable
RES = 256
DEPTH_MIN_M = 0.5       # swarm.constants.DEPTH_MIN_M
DEPTH_MAX_M = 30.0      # SAR_DEPTH_MAX_M (the SAR env's _depth_max_m)


def rot_from_quat(q) -> np.ndarray:
    """3x3 rotation matrix from a pybullet quaternion [x, y, z, w]."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def rot_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rz(yaw) @ Ry(pitch) @ Rx(roll) — pybullet euler order."""
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
    """World-space unit ray through pixel (u,v). Returns (ray, forward, up)."""
    f = _norm(R @ np.array([1.0, 0.0, 0.0]))
    up = R @ np.array([0.0, 0.0, 1.0])
    r = _norm(np.cross(f, up))
    t = np.cross(r, f)
    half = np.tan(np.radians(fov_deg) * 0.5)
    a = (2.0 * (u + 0.5) / res - 1.0) * half           # + right
    b = (1.0 - 2.0 * (v + 0.5) / res) * half           # + up (image y is flipped)
    return _norm(f + a * r + b * t), f, up


def backproject(pos, R, u, v, D, fov_deg=FOV_DEG, res=RES):
    """World position of a target seen at pixel (u,v) at range D metres."""
    pos = np.asarray(pos, float)
    ray, f, up = pixel_to_world_ray(R, u, v, fov_deg, res)
    cam = pos + CAM_FWD * f + CAM_UP * up
    return cam + D * ray


def depth_to_meters(n, depth_max: float = DEPTH_MAX_M, depth_min: float = DEPTH_MIN_M):
    """Invert MovingDroneAviary._process_depth (linear in the stored [0,1]).

    The ceiling is per FAMILY, not global: SAR normalises over 30 m
    (SAR_DEPTH_MAX_M) while autopilot and the rest use DEPTH_MAX_M = 20. Passing
    the wrong one does not fail loudly, it just scales every range by 1.5 — so the
    default stays SAR and callers on other families pass their own.
    """
    return np.asarray(n, float) * (float(depth_max) - float(depth_min)) + float(depth_min)


def range_from_box_depth(depth_img, box, pct=10.0, pad=2):
    """Metric range to the victim: a low percentile of the depth over the box (the victim
    is the closest surface; a low percentile rejects background bleeding into a tiny box)."""
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
    """Range implied by how tall the victim appears: a known-height target subtends
    fewer pixels the farther it is. Immune to depth contamination (it's the DETECTED
    box), just coarse (the mannequin's apparent height varies with pose)."""
    h_px = max(1.0, float(box[3]) - float(box[1]))
    focal = (res * 0.5) / np.tan(np.radians(fov_deg) * 0.5)
    return victim_h * focal / h_px


def robust_range(depth_img, box, res=RES, fov_deg=FOV_DEG, victim_h=1.2, tol=2.2):
    """Metric range, or None if unreliable. Uses the depth read (accurate when clean) but
    rejects it when it disagrees with the box-size range by more than `tol`x — that's the
    signature of a fore/background object bleeding into the box (the 2.9 m-vs-21 m bug)."""
    dr = range_from_box_depth(depth_img, box)
    if dr is None:
        return None
    br = range_from_box_size(box, res, fov_deg, victim_h)
    if dr < br / tol or dr > br * tol:
        return None
    return dr


VICTIM_HALF_H = 0.9     # measured: victim centre sits 0.9 m above its support surface
MAX_H_ABOVE_GROUND = 3.5  # a detection higher than this is a lantern head, not a victim


def local_ground_z(pos, R, box, depth_img, *, fov_deg=FOV_DEG, res=RES, pad=3):
    """World Z of the surface under the detection, read from the depth pixels just
    below it. FALLBACK ONLY — used when the altimeter has saturated.

    The idea was that the altimeter answers a different question (height above
    whatever is under the DRONE, i.e. a rooftop or a canopy) while the pixels
    beside the victim are its actual support surface. Measured against ground
    truth on 92 city/village/forest detections it is simply worse — median
    horizontal error 3.49 m vs the altimeter plane's 1.13 m, landing ~2.3 m too
    high. The reason is the box: the size head under-covers a 12 px victim, so the
    "ring below the box" is still the victim's own legs, or the hedge/undergrowth
    it is lying in. A ring is only as good as the box it is drawn around.

    Kept because when the 20 m altimeter saturates (mountain cruise) there is no
    plane at all, and there it measures no worse than the raw depth range.
    """
    d = np.asarray(depth_img)
    if d.ndim == 3:
        d = d[..., 0]
    H, W = d.shape
    u0, v0, u1, v1 = box
    r0 = int(np.clip(v1 + 1, 0, H - 1)); r1 = int(np.clip(v1 + 1 + pad * 2, 0, H))
    c0 = int(np.clip(u0 - pad, 0, W)); c1 = int(np.clip(u1 + pad + 1, 0, W))
    if r1 <= r0 or c1 <= c0:
        return None
    # Subsample wide patches: a close-range box can be 100+ px across and this runs
    # per detection per drone per step, inside a 500 ms act() budget.
    step = max(1, (c1 - c0) // 32)
    cols = np.arange(c0, c1, step)
    rows = np.arange(r0, r1)
    rng = depth_to_meters(d[np.ix_(rows, cols)])
    # A saturated pixel means "nothing within 30 m", not "a surface at 30 m" —
    # counting those as ground drags the plane out to the horizon.
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
    """One detection -> one world point, or None if it fails physical sanity.

    Two ways to turn a pixel into a position, and they fail differently:

      A. walk `depth_range` metres along the ray. The RANGE is the noisy part —
         the centroid is ~10 px out, so at 20 m the box often catches ground in
         front of the victim, the percentile reads short, and the point lands
         short and therefore HIGH along a downward ray.
      B. intersect the ray with the plane the victim must be resting on
         (`local_ground_z` + half a mannequin). Replaces the noisy range with the
         prior "victims lie on a surface", and reads that surface from the depth
         pixels beside the target rather than from the altimeter — the altimeter
         measures rooftops and canopy, which zeroed city and forest outright.

    Measured over 174 real in-flight detections (team/scripts/probe_detector.py):
    B halves the horizontal error at every range (median 0.80 m vs 1.80 m, q75
    1.24 vs 2.69) and removes A's systematic +0.92 m height bias. So B is the
    answer and A is the CHECK — a surface estimate can still be wrong (the ring
    catches a wall, or a slope runs away under the ray), and disagreeing with the
    measured range by more than `ground_tol` is what that looks like.

    A is also the only one of the two that can tell a victim from a streetlight,
    because B projects everything down onto the ground by construction. So the
    height test runs on A, before B is allowed to produce the answer.
    """
    rng = range_from_box_depth(depth_img, box)
    ray, f, up = pixel_to_world_ray(R, u, v, fov_deg, res)
    pos = np.asarray(pos, float)
    cam = pos + CAM_FWD * f + CAM_UP * up

    # The support surface. The drone's own altimeter plane wins on measurement
    # (see local_ground_z for the numbers), so the depth ring is only the fallback
    # for when the altimeter has saturated and there is no plane otherwise.
    ground_z = (float(pos[2]) - float(agl)
                if (agl is not None and agl < altimeter_max - 1e-3) else None)
    if ground_z is None:
        ground_z = local_ground_z(pos, R, box, depth_img, fov_deg=fov_deg, res=res)
    known_ground = ground_z is not None

    if rng is not None:
        a = cam + rng * ray
        if known_ground and a[2] - ground_z > MAX_H_ABOVE_GROUND:
            return None                                  # floating: not a victim
        br = range_from_box_size(box, res, fov_deg)
        if rng < br / 2.2 or rng > br * 2.2:
            rng = None                                   # contaminated depth read

    if known_ground and ray[2] < -1e-3:
        t = (ground_z + VICTIM_HALF_H - cam[2]) / ray[2]
        if 0.0 < t < DEPTH_MAX_M * 1.5 and (rng is None
                                            or ground_tol[0] < t / rng < ground_tol[1]):
            return cam + t * ray
    if rng is None:
        return None
    pt = cam + rng * ray
    if known_ground:
        pt[2] = ground_z + VICTIM_HALF_H     # kill the along-ray height bias
    return pt


def hover_target(victim, h_confirm=3.5):
    """Confirm/hover position: above the victim, in the 2-4 m band (with estimate margin)."""
    v = np.asarray(victim, float)
    return np.array([v[0], v[1], v[2] + h_confirm])


# Floor on the range used for weighting. Without it a single frame claiming to be
# 1 m away carries 25x the weight of a good 5 m one and can hijack the median on
# its own. Below ~5 m the victim is nearly underneath the drone and half out of
# frame, so those reads are not actually more trustworthy — the accuracy gain
# from closing in has already been banked by then.
RANGE_FLOOR = 5.0


def _weighted_median(values, weights):
    idx = np.argsort(values)
    v, w = values[idx], weights[idx]
    cw = np.cumsum(w)
    return float(v[min(len(v) - 1, int(np.searchsorted(cw, cw[-1] * 0.5)))])


class VictimEstimator:
    """World-space estimate of the (static) victim, robust to per-frame noise.

    The victim doesn't move, so we aggregate every back-projection — but with a per-axis
    WEIGHTED MEDIAN, not a mean: a single bad frame (a depth read that caught background,
    given high weight for looking "close") poisons a mean and it can't recover, whereas the
    median just ignores it.

    Weight ~ conf / range**power. A fixed centroid error (~10 px) subtends metres in
    proportion to range, so the error variance goes as range**2 and inverse-variance
    weighting means power=2 — that is what lets the close frames of an approach
    overrule the far sighting that started it. `max_keep` bounds the window for the
    same reason: an approach only improves if old, distant evidence eventually ages out.
    """

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
    """One hypothesis: a place in the world that keeps looking like a victim.

    Carries its own colour ledger, because that is what separates a victim from a
    streetlight — and a REJECTION counts: an RGB frame that came back negative on
    this candidate is evidence against it, which is how an approach gets refuted
    before the hover timeout has to do it.
    """

    def __init__(self, max_keep=80, power=2.0):
        super().__init__(max_keep=max_keep, power=power)
        self.rgb_confirms = 0
        self.rgb_rejects = 0

    @property
    def colour_score(self) -> int:
        return self.rgb_confirms - self.rgb_rejects

    @property
    def evidence(self) -> float:
        """How much this candidate deserves to be the one we fly to.

        Ranked on CONFIRMATIONS first, rejections second, looks only as a
        tie-break, and the weights are not arbitrary — two opposite failures pin
        them down:

          * a candidate the victim had been seen at 241 times lost to a lantern
            glimpsed five times, because one unlucky photograph made its NET
            colour score negative. So looks must count for something.
          * weighting looks at 2x colour then broke it the other way: clutter
            sitting on the sweep path accumulates 40 looks (+4.0) with no colour
            evidence at all, while the victim — genuinely visible for about 6 s of
            a 60 s episode — gets ten looks and one confirmation (+3.0), and the
            lantern wins.

        A confirmation therefore outweighs the entire saturated look budget (10 vs
        4), and a rejection weighs less than the look spread it competes with (1 vs
        up to 4) so it can lower a candidate without single-handedly demoting a
        well-observed one below a barely-seen rival. Both bounds are forced by the
        two failures above; there is very little room between them.
        """
        return 10.0 * self.rgb_confirms - 1.0 * self.rgb_rejects + min(self.n, 40) / 10.0


class CandidateTracker:
    """All the hypotheses at once, clustered by world position.

    A single estimator over every accepted detection assumes there is one thing out
    there. A sweep sees the victim AND several lanterns, and a weighted median over
    that mixture lands BETWEEN the modes — measured: a village seed whose estimate
    settled 37.9 m from anything real, pointing at nothing at all. Clustering keeps
    each object's evidence separate, so the colour ledger can decide between them
    and the winner's median is computed over its own points only.
    """

    def __init__(self, radius=5.0, max_keep=80, power=2.0, max_candidates=24):
        self.radius = float(radius)
        self.max_keep, self.power = max_keep, power
        self.max_candidates = max_candidates
        self.cands: list = []

    def nearest(self, pt, radius=None):
        """The candidate this observation belongs to, or None if it is new."""
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
                # evict the weakest, never the one we are acting on
                self.cands.sort(key=lambda x: (x.colour_score, x.n))
                self.cands.pop(0)
        c.update(pt, conf=conf, rng=rng)
        if from_rgb:
            c.rgb_confirms += 1
        return c

    def reject(self, pt, conf=1.0, rng=10.0, radius=2.0):
        """Colour looked at this place and said no.

        Files a candidate even when there was none, because "we already checked
        that thing and it is not a victim" is worth remembering: it is what stops
        the same lantern from being re-photographed until the 40-frame budget is
        gone. Its colour score is negative, so it can never win `best()`.

        A rejection clusters at a TIGHTER radius than a confirmation, and that
        matters: at the shared 5 m radius a refused lantern standing near the
        victim merged into the victim's own candidate and cancelled its colour
        score, which cost two open seeds that had been clean wins. A "no" is about
        one specific object; a "yes" can be a metre or two off and still be the
        same person.
        """
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
    """SEARCH -> APPROACH -> HOVER -> (timeout) false-positive -> resume.

    Commit to APPROACH only after N-of-M detections (a one-frame lantern glimpse shouldn't
    cost a detour). Declare false-positive after `hold_steps` in the confirm region while the
    episode is still running — a real victim ends the episode at the 2 s dwell, so "still
    flying after ~4 s" means clutter. Blacklist that XY so the sweep doesn't loop back.
    """

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
        """Advance the machine; returns the current state. `victim_est` may be None."""
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
            if self.hold >= self.hold_steps:            # held ~4 s, episode not over -> clutter
                if victim_est is not None:
                    self.blacklist.append(np.asarray(victim_est)[:2].copy())
                self.state, self.hold = self.SEARCH, 0
        return self.state
