\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List, Optional

import math
import numpy as np

AGL_MIN = 1.5
AGL_CLIMB = 0.8
AGL_FAR = 3.0

REFINE_R = 8.0
REFINE_K = 5
REFINE_WIN = 30
REFINE_TIGHT_M = 0.5
REFINE_TIGHT_FRAC = 0.6
REFINE_APPLY_R = 10.0
REFINE_MIN_SHIFT = 0.5
REFINE_MAPS = ("village", "city", "mountain", "open")

from team.autopilot import freedir as FD
from team.detector import pads as PD
from team.detector.localize import depth_to_meters, rot_from_rpy

FIX_SLEW, FIX_FOV, FIX_CLAMP, FIX_NOGROUND, FIX_BELOW = (True, "city", True, True, True)

LANE_INHERIT = True
HURRY_MAPS = ("mountain", "city")
LANE_MODE = "pool"
FOREST_TALL_OVERRIDE = 0.60
LANE_VISITED_M = 10.0
FIX_SLEW_KINDS = ("city", "open", "village")
FIX_BOX_KINDS = ("city", "open", "village")
FREE_DIR_MIN_COS = math.cos(math.radians(50.0))
SLEW_G, SLEW_DZ, SLEW_DXY = 0.2646, 0.5, 0.2
SLEW_DVZ_DOWN_MAX = 0.25
SLEW_TILT_MAX = math.radians(40.0)

def slew_velocity(v_des, v_meas, max_delta_v, tilt_rad=None,
                  tilt_knee=0.30, tilt_stop=0.85, tilt_floor=0.15,
                  ctrl_aware=True):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\

    v_des = np.asarray(v_des, dtype=float)
    if max_delta_v is None or max_delta_v <= 0.0 or v_meas is None:
        return v_des
    if tilt_rad is not None and tilt_rad > tilt_knee:
        span = max(tilt_stop - tilt_knee, 1e-6)
        k = min(1.0, (float(tilt_rad) - tilt_knee) / span)
        max_delta_v = max_delta_v * (1.0 - k * (1.0 - tilt_floor))
    v_meas = np.asarray(v_meas, dtype=float).reshape(3)
    dv = v_des - v_meas
    m = float(np.linalg.norm(dv))
    if m > max_delta_v and m >= 1e-9:
        dv = (max_delta_v / m) * dv

    if not (FIX_SLEW and ctrl_aware):
        return v_meas + dv
    dv[2] = max(float(dv[2]), -SLEW_DVZ_DOWN_MAX)
    tz = SLEW_G + SLEW_DZ * float(dv[2])
    xy_max = math.tan(SLEW_TILT_MAX) * tz / SLEW_DXY
    mxy = float(np.hypot(dv[0], dv[1]))
    if mxy > xy_max and mxy > 1e-9:
        dv[:2] *= xy_max / mxy
    return v_meas + dv

CLIMB, SEARCH, APPROACH, DESCEND, DONE = "CLIMB", "SEARCH", "APPROACH", "DESCEND", "DONE"


POS = slice(0, 3)
RPY = slice(3, 6)
VEL = slice(6, 9)
ALT = 137
GOAL_OFF = slice(138, 141)
MATES = slice(141, 190)

@dataclass
class AutopilotConfig:
    speed_limit: float = 3.0

    max_delta_v: float = 1.0

    tilt_knee: float = 0.30
    tilt_stop: float = 0.85
    tilt_floor: float = 0.15

    cruise_speed: float = 3.0

    cruise_alt: float = 4.5

    cruise_alt_safe: float = 6.0
    climb_speed: float = 1.4

    climb_creep: float = 0.55

    climb_clear_m: float = 0.0

    climb_creep_tilt: float = 0.28

    tilt_fade_lo: float = 0.40
    tilt_fade_hi: float = 0.70

    tilt_guard_rad: float = 0.62
    tilt_guard_brake: float = 1.5

    tilt_guard_hold_sec: float = 0.55
    tilt_guard_speed_cap: float = 1.8

    climb_max_delta_v: float = 0.0

    post_climb_soft_sec: float = 0.0
    post_climb_speed_frac: float = 1.0

    open_boost: bool = False
    cruise_alt_open: float = 3.0
    cruise_speed_open: float = 3.0
    open_clear: float = 18.0

    rewedge: bool = False

    own_annulus: bool = False
    own_bands: tuple = ((22.0, 45.0),)
    own_legs: int = 1
    own_arc: float = 1.05
    own_arc_m: float = 100.0
    own_step_m: float = 12.0

    own_monotone_maps: tuple = ("city",)

    own_near_first: tuple = ()

    own_maps: tuple = ("city", "mountain")

    own_split_tall: float = 0.28

    own_split_kind: str = ""
    search_rings: tuple = (34.0, 20.0, 48.0, 62.0)

    village_rings: tuple = ()

    mountain_rings: tuple = (86.0, 42.0, 118.0, 16.0)
    search_r1: float = 70.0
    search_pitch: float = 14.0
    sweep_step: float = 0.34

    lane_takeover: bool = False

    adaptive_sweep: bool = False
    adaptive_sweep_maps: tuple = ()
    sweep_swath: float = 15.0
    sweep_frac_min: float = 0.10
    sweep_frac_max: float = 0.90
    waypoint_reach: float = 6.0

    avoid_range: float = 7.0
    avoid_climb: float = 1.5
    avoid_brake: float = 0.45

    avoid_range_mtn: float = -1.0
    mtn_yield_r: float = 25.0

    mountain_avoid_climb: float = -1.0
    mountain_avoid_brake: float = -1.0

    mountain_alt: float = -1.0

    mountain_profile_maps: tuple = ("mountain",)
    mountain_steer_clear: float = -1.0
    mountain_boxed_keep: float = -1.0

    mass_planner: bool = False

    mass_plan_maps: tuple = ("city", "forest")
    mass_plan_n: tuple = (2, 3, 4, 5, 6, 7, 8)
    mass_plan_n_by_map: dict = field(default_factory=dict)
    mass_cell_m: float = 8.0
    mass_replan_steps: int = 100
    mass_target_reach_m: float = 7.0
    mass_target_sep_m: float = 14.0
    mass_distance_scale_m: float = 42.0
    mass_observation_discount: float = 0.70
    mass_min_score: float = 0.025
    mass_explained_radius_m: float = 6.0
    mass_explained_factor: float = 0.05

    route_planner: bool = False
    route_plan_maps: tuple = ("city", "mountain")

    route_start_sec: float = 0.0

    steer: bool = False

    steer_maps: tuple = ()

    steer_boxed_only: bool = False
    steer_clear: float = 6.0
    steer_bias: float = 0.8
    steer_slow: float = 0.75

    approach_alt: float = 4.0

    flare_alt: float = 0.50
    sink_speed: float = 2.8
    forest_sink_speed: float = 1.1
    forest_flare_alt: float = 0.8
    descend_speed: float = 0.42
    creep_speed: float = 0.35
    abort_radius: float = 1.2
    forest_abort_radius: float = 0.7
    touch_speed: float = 0.30
    land_radius: float = 0.45

    descend_commit: bool = False
    descend_commit_r: float = 0.45
    descend_commit_above: float = 0.25
    commit_hold_release: bool = False

    sink_center_cap: float = 0.0

    yaw_scan_deg: float = 0.0
    yaw_scan_steps: int = 200
    yaw_scan_maps: tuple = ()
    yaw_scan_min_clear: float = 12.0

    stale_refute_s: float = 0.0
    stale_refute_maps: tuple = ()
    stale_refute_r: float = 25.0
    stale_near_r: float = 6.0

    sep_radius: float = 3.2
    sep_gain: float = 1.6

    same_pad: float = 3.0
    detect_every: int = 3
    min_detect_score: float = 0.05

    pad_z_lo: float = 0.12
    pad_z_hi: float = 1.20
    village_z_lo: float = 0.18

    village_z_hi: float = 0.70

    forest_geometric: bool = False
    forest_band: bool = True
    forest_speed: float = 1.3
    forest_delta_v: float = 0.6
    forest_alt: float = 4.0

    forest_approach_alt: float = 1.8

    forest_ceiling: float = 4.5

    avoid_yield_radius: float = 6.0
    avoid_yield_floor: float = 0.25

    free_dir: bool = False

    free_dir_village: bool = False
    free_dir_mountain: bool = False

    free_lookahead_mountain: float = 20.0
    free_margin_mountain: float = 0.65
    free_grid: int = 24
    free_margin: float = 0.35
    free_want_clear: float = 6.0
    free_lookahead: float = 10.0
    free_slow: float = 0.75

    free_brake: bool = False
    forest_z_lo: float = 1.55
    forest_z_hi: float = 3.35
    forest_clear_m: float = 9.0

    forest_clear_frac: float = 0.18
    forest_min_samples: int = 120

    tall_m: float = 4.0

    tall_near_m: float = 19.0

    city_tall_frac: float = 0.015

    claim_drift: float = 1.5
    min_hits_to_land: int = 2

    min_hits_to_claim: int = 5
    village_min_hits: int = 8

    mountain_detector_swap: bool = False

    avoid_range_village: float = -1.0

    steer_soft_margin: float = -1.0
    steer_preferred: float = 6.0
    avoid_close_r: float = -1.0
    avoid_close_brake: float = 0.20
    open_alt: float = -1.0

    mountain_min_hits: int = -1
    ground_stall_steps: int = 75

    commit_t: float = 0.0

    commit_t_skip: tuple = ()

    agl_band_tol: float = -1.0
    agl_band_r: float = 3.0
    landed_done: bool = False
    landed_r: float = 0.55
    landed_steps: int = 30

    dead_steps: int = 60

    descend_enter_r: float = 1.0
    descend_enter_r_tight: float = 1.0
    descend_enter_wide: tuple = ()

    terrain_ff: float = 0.0
    terrain_ff_maps: tuple = ()
    climb_cap: float = 1.0
    mountain_climb_cap: float = -1.0
    investigate: bool = False
    investigate_hits: int = 2
    investigate_hits_forest: int = 1

    investigate_hits_village: int = 4
    investigate_r: float = 60.0
    investigate_skip: tuple = ("village",)
    investigate_first: tuple = ()
    bad_pad_radius: float = 4.0

@dataclass
class _Drone:
    phase: str = CLIMB
    claim: Optional[int] = None
    theta: float = 0.0
    own_theta: float = 0.0
    radius: float = 0.0
    wedge: int = 0
    nwedge: int = 1
    span: float = 1.0
    bailed: bool = False
    stuck: int = 0
    landed: bool = False
    agl_prev: float = -1.0
    commit_sink: bool = False
    climb_exit_t: float = -1.0
    tilt_soft_until: float = -1.0

    prev_xy: Optional[np.ndarray] = None
    still: int = 0
    flew: bool = False
    own_r: float = 0.0
    own_span: float = 0.0
    mass_target: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    mass_target_valid: bool = False
    mass_target_idx: int = -1
    route_idx: int = 0

@dataclass
class _Pad:
    xyz: np.ndarray
    hits: int = 1
    by: Optional[int] = None
    done: bool = False
    anchor: Optional[np.ndarray] = None
    near: object = None
    near_xy: Optional[np.ndarray] = None
    applied: bool = False
    last_seen: int = -1

class SwarmAutopilotAgent:

    def __init__(self, cfg: Optional[AutopilotConfig] = None,
                 pad_cfg: Optional[PD.PadConfig] = None,
                 detector=None, forest_detector=None, village_detector=None,
                 mountain_detector=None, posterior=None, route_planner=None):
\
\
\
\
\
\
\
\
\
\
\

        self.cfg = cfg or AutopilotConfig()
        self.pad_cfg = pad_cfg or PD.PadConfig()
        self.detector = detector

        self.general_detector = detector
        self.forest_detector = forest_detector
        self.village_detector = village_detector

        self.mountain_detector = mountain_detector
        self.posterior = posterior
        self.route_planner = route_planner
        self.reset()

    def reset(self, *_a, **_k) -> None:
        self.detector = self.general_detector
        self.d: List[_Drone] = []
        self.pads: List[_Pad] = []
        self.bad: List[np.ndarray] = []
        self.clue: Optional[np.ndarray] = None
        self.z_band: Optional[tuple] = None

        self.min_hits = self.cfg.min_hits_to_claim
        self.rings: tuple = self.cfg.search_rings
        self.map_kind = "other"
        self.is_forest = False
        self._band_locked = False
        self._clear_hits = 0
        self._tall_hits = 0
        self._clear_n = 0
        self._launch_z = None
        self._own_start = None
        self._wedge_n = 0
        self._home_of = {}
        self._orphans_taken = set()
        self.wedge_phase = 0.0
        self._wedge_dynamic = self.cfg.rewedge
        self.step_i = -1
        self.n = 0
        self._mass_xy = np.zeros((0, 2), dtype=float)
        self._mass_prior = np.zeros(0, dtype=float)
        self._mass_seen = np.zeros(0, dtype=np.float32)
        self._mass_key_to_idx = {}
        self._route_waypoints = None

    def _assign_wedges(self, state: np.ndarray) -> None:
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\

        idx = [i for i in range(self.n) if self.d[i].phase in (CLIMB, SEARCH)]
        if not idx:
            return
        m = len(idx)
        rel = np.asarray(state[idx, POS], float)[:, :2] - self.clue[None, :2]
        brg = np.arctan2(rel[:, 1], rel[:, 0])

        self.wedge_phase = float(brg[int(np.argmin(brg))] - np.pi / m)
        centres = self.wedge_phase + (np.arange(m) + 0.5) * (2.0 * np.pi / m)

        diff = np.abs(np.angle(np.exp(1j * (brg[:, None] - centres[None, :]))))
        pairs = sorted((float(diff[a, k]), a, k)
                       for a in range(m) for k in range(m))
        taken_a, taken_k = set(), set()
        for _c, a, k in pairs:
            if a in taken_a or k in taken_k:
                continue
            dd = self.d[idx[a]]
            dd.wedge = k
            dd.nwedge = m

            leg = int(dd.theta)
            width = 2.0 * np.pi / m
            lo = self.wedge_phase + width * k
            f = ((float(brg[a]) - lo) % (2.0 * np.pi)) / width
            f = float(np.clip(f, 0.0, 1.0))
            if leg % 2 == 1:
                f = 1.0 - f
            dd.theta = leg + f
            taken_a.add(a)
            taken_k.add(k)
        self._wedge_n = m
        self._home_of = {int(self.d[i].wedge): i for i in range(self.n)}
        self._orphans_taken = set()

    def _vote_forest(self, depth: np.ndarray, live: List[int]) -> None:
\
\
\
\
\
\
\

        cfg = self.cfg
        if self._band_locked or not cfg.forest_band:
            return
        for i in live:
            self._clear_n += 1
            if float(np.min(self._column_clearance(depth, i))) >= cfg.forest_clear_m:
                self._clear_hits += 1

            if self._tall_in_view(depth, i):
                self._tall_hits += 1
        if self._clear_n < cfg.forest_min_samples:
            return
        self._band_locked = True
        if self._clear_hits / float(self._clear_n) < cfg.forest_clear_frac:
            self.is_forest = True
            self.map_kind = "forest"
            self.z_band = (cfg.forest_z_lo, cfg.forest_z_hi)
        else:

            frac = self._tall_hits / float(max(self._clear_n, 1))
            if frac >= FOREST_TALL_OVERRIDE:
                self.is_forest = True
                self.map_kind = "forest"
                self.z_band = (cfg.forest_z_lo, cfg.forest_z_hi)
            else:
                self.map_kind = "city" if frac > cfg.city_tall_frac else "open"

    def _set_z_band(self, state: np.ndarray) -> None:
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\

        cfg = self.cfg
        z0 = np.asarray(state[:, 2], float)

        if bool(0.40 <= float(np.median(z0)) <= 0.70):
            self.z_band = (cfg.village_z_lo, cfg.village_z_hi)
            self.map_kind = "village"
            if cfg.village_rings:
                self.rings = cfg.village_rings
            self.min_hits = cfg.village_min_hits
            if self.village_detector is not None:
                self.detector = self.village_detector
            self._band_locked = True
        elif float(np.median(z0)) > 10.0:
            self.z_band = None
            self.rings = cfg.mountain_rings
            self.map_kind = "mountain"
            if cfg.mountain_min_hits > 0:
                self.min_hits = cfg.mountain_min_hits

            if cfg.mountain_detector_swap and self.mountain_detector is not None:
                self.detector = self.mountain_detector

            self._band_locked = True
        else:

            self.z_band = (cfg.pad_z_lo, cfg.pad_z_hi)

    def _detect(self, state: np.ndarray, depth: np.ndarray, i: int) -> None:
        pos = np.asarray(state[i, POS], float)
        rpy = np.asarray(state[i, RPY], float)
        R = rot_from_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        try:
            props = PD.propose(pos, R, depth[i], self.pad_cfg)
        except Exception:
            return
        for q in props:
            if q.score < self.cfg.min_detect_score:
                continue
            self._merge(np.asarray(q.centre, float))

    _refine_state = None

    @staticmethod
    def _refine_rng(c, poses, batch):
        for r, props in enumerate(batch):
            for q in props:
                if np.allclose(np.asarray(q.centre, float), c):
                    return float(np.linalg.norm(poses[r] - c))
        return None

    def _refine_note(self, pad, xyz, rng):
        if pad.near is None:
            pad.near = deque(maxlen=REFINE_WIN)
            pad.near_xy = None
        if rng is not None and rng < REFINE_R:
            pad.near.append(xyz[:2].copy())
            if len(pad.near) >= REFINE_K:
                arr = np.asarray(pad.near, float)
                med = np.median(arr, axis=0)
                frac = float(np.mean(np.linalg.norm(arr - med, axis=1) < REFINE_TIGHT_M))
                pad.near_xy = med if frac >= REFINE_TIGHT_FRAC else None
        self._refine_apply(pad)

    def _refine_apply(self, pad):
        xy = pad.near_xy
        if xy is None or pad.done or pad.by is None:
            return
        kind = "forest" if getattr(self, "is_forest", False) else str(getattr(self, "map_kind", "other"))
        if kind not in REFINE_MAPS:
            return
        st = self._refine_state
        if st is None or pad.by >= st.shape[0]:
            return
        if float(np.linalg.norm(st[pad.by, :2] - pad.xyz[:2])) > REFINE_APPLY_R:
            return
        if pad.anchor is None:
            pad.anchor = pad.xyz[:2].copy()
        off = xy - pad.anchor
        far = float(np.linalg.norm(off))
        cap = float(self.cfg.claim_drift)
        if far > cap:
            xy = pad.anchor + off * (cap / far)
        if not pad.applied:
            if float(np.linalg.norm(xy - pad.xyz[:2])) < REFINE_MIN_SHIFT:
                return
            pad.applied = True
        pad.xyz = np.array([xy[0], xy[1], pad.xyz[2]], float)

    def _merge(self, xyz: np.ndarray, rng=None) -> None:

        if self.z_band is not None:
            if not (self.z_band[0] <= float(xyz[2]) <= self.z_band[1]):
                return
        for b in self.bad:
            if float(np.linalg.norm(b - xyz[:2])) < self.cfg.bad_pad_radius:
                return
        for pad in self.pads:
            if float(np.linalg.norm(pad.xyz[:2] - xyz[:2])) < self.cfg.same_pad:

                w = 1.0 / (pad.hits + 1.0)
                new = (1.0 - w) * pad.xyz + w * xyz
                if pad.by is not None:

                    if pad.anchor is None:
                        pad.anchor = pad.xyz[:2].copy()
                    off = new[:2] - pad.anchor
                    far = float(np.linalg.norm(off))
                    if far > self.cfg.claim_drift:
                        new[:2] = pad.anchor + off * (self.cfg.claim_drift / far)
                pad.xyz = new
                pad.hits += 1
                pad.last_seen = self.step_i
                self._refine_note(pad, xyz, rng)
                return
        self.pads.append(_Pad(xyz=xyz.copy(), last_seen=self.step_i))

    def _assign(self, state: np.ndarray) -> None:
\

        gate = self.min_hits
        if (self.cfg.commit_t > 0.0
                and self.map_kind not in self.cfg.commit_t_skip
                and self.step_i >= int(self.cfg.commit_t * 50)):
            gate = 1
        free = [k for k, q in enumerate(self.pads)
                if q.by is None and not q.done and q.hits >= gate]
        if not free:
            return
        cand = [i for i in range(self.n)
                if self.d[i].claim is None and self.d[i].phase in (CLIMB, SEARCH)]
        pairs = sorted(
            ((float(np.linalg.norm(np.asarray(state[i, POS], float)[:2]
                                   - self.pads[k].xyz[:2])), i, k)
             for i in cand for k in free

             if (not (FIX_BELOW and self._mass_route_kind() in FIX_BOX_KINDS))
             or float(state[i, 2]) >= float(self.pads[k].xyz[2]) - 2.0))

        spoken = [self.pads[d.claim].xyz[:2] for d in self.d
                  if d.claim is not None and self.pads[d.claim].by is not None]
        taken_i = set()
        for _dist, i, k in pairs:
            if i in taken_i or self.pads[k].by is not None or self.pads[k].done:
                continue
            xy = self.pads[k].xyz[:2]
            if any(float(np.linalg.norm(xy - s)) < self.cfg.same_pad for s in spoken):
                continue
            self.d[i].claim = k
            self.pads[k].by = i
            self.d[i].phase = APPROACH
            taken_i.add(i)
            spoken.append(xy)

    def _endgame_hurry(self, dist: float, above: float) -> bool:
        # A claim-holding drone whose normal-pace completion is projected to
        # miss the horizon scores 0.01 anyway -- terminal risk is free for it.
        if self._mass_route_kind() not in HURRY_MAPS:
            return False
        t_left = 60.0 - self.step_i / 50.0
        need = dist / 1.5 + max(0.0, above) / 0.45 + 3.0
        return t_left < need

    def _pad_xy_dist(self, dd) -> float:
        pad = self.pads[dd.claim]
        return float(np.linalg.norm(np.asarray(pad.xyz, float)[:2] - dd.prev_xy))

    def _refute(self, i: int) -> None:
\
\
\
\
\

        dd = self.d[i]
        if dd.claim is not None:
            pad = self.pads[dd.claim]
            pad.done = True
            pad.by = None
            self.bad.append(pad.xyz[:2].copy())
        dd.claim = None
        dd.stuck = 0

        dd.commit_sink = False
        dd.phase = CLIMB

    ANNULUS = {"city": (22.0, 45.0), "open": (28.0, 72.0), "forest": (22.0, 45.0),
               "village": (28.0, 56.0), "mountain": (65.0, 100.0)}

    OWN_RINGS_N = {
        ("city",     "lo"): (29.0, 37.0, 14.0),
        ("city",     "mid"): (29.0, 14.0, 38.0),
        ("city",     "hi"): (29.0, 14.0, 37.0),
        ("open",     "lo"): (44.0, 29.0, 59.0),
        ("open",     "mid"): (25.0, 40.0, 10.0),
        ("open",     "hi"): (17.0, 32.0, 8.0),
        ("mountain", "lo"): (83.0, 64.0, 41.0),
        ("mountain", "mid"): (72.0, 48.0, 30.0),
        ("mountain", "hi"): (37.0, 20.0, 62.0),
        ("village",  "lo"): (29.0, 11.0, 44.0),
        ("village",  "mid"): (17.0, 32.0, 8.0),
        ("village",  "hi"): (14.0, 28.0, 8.0),
        ("forest",   "lo"): (21.0, 37.0, 8.0),
        ("forest",   "mid"): (20.0, 35.0, 8.0),
        ("forest",   "hi"): (17.0, 29.0, 8.0),
    }

    @staticmethod
    def _n_group(n: int) -> str:
        return "lo" if n <= 3 else ("mid" if n <= 6 else "hi")

    OWN_RINGS = {
        "city":     (29.0, 14.0, 38.0),
        "open":     (26.0, 11.0, 41.0),
        "forest":   (18.0, 33.0, 10.0),
        "village":  (13.0, 28.0, 43.0),
        "mountain": (33.0, 53.0, 71.0),
        "other":    (26.0, 14.0, 40.0),
    }

    WORLD = {"city": 75.0, "open": 60.0, "forest": 42.0,
             "village": 40.0, "mountain": 75.0}

    def _in_bounds(self, tgt: np.ndarray) -> np.ndarray:

        kind = self._mass_route_kind()
        r = self.WORLD.get(kind if (FIX_CLAMP and kind in FIX_BOX_KINDS) else self.map_kind)
        if r is None:
            return tgt
        out = np.asarray(tgt, float).copy()
        out[0] = float(np.clip(out[0], -r, r))
        out[1] = float(np.clip(out[1], -r, r))
        return out

    def _is_open_scene(self) -> bool:
\
\
\
\
\
\
\
\
\
\

        c = self.cfg
        if c.own_split_tall <= 0.0 or self._clear_n <= 0:
            return False
        if self.map_kind not in ("city", "open"):
            return False
        return (self._tall_hits / float(self._clear_n)) <= c.own_split_tall

    def _ring_kind(self) -> str:
\
\
\
\
\
\
\
\
\
\
\

        c = self.cfg
        if self.map_kind != "city" or c.own_split_tall <= 0.0:
            return self.map_kind
        if self._clear_n <= 0:
            return self.map_kind
        if (self._tall_hits / float(self._clear_n)) > c.own_split_tall:
            return "city"
        return c.own_split_kind

    def _own_target(self, i: int) -> Optional[np.ndarray]:
\
\
\
\
\
\
\
\
\

        c = self.cfg
        dd = self.d[i]
        if self._own_start is None:
            return None

        kind = self._ring_kind()
        if kind not in c.own_maps:
            return None
        rings = self.OWN_RINGS_N.get((kind, self._n_group(self.n)))
        if rings is None:
            rings = self.OWN_RINGS.get(kind, self.OWN_RINGS["other"])
        if kind in c.own_near_first:
            rings = tuple(sorted(rings))
        leg = int(dd.own_theta)
        if leg >= len(rings):
            return None
        frac = dd.own_theta - leg
        s = np.asarray(self._own_start[i], float)[:2]
        r = float(rings[leg])
        base = float(np.arctan2(self.clue[1] - s[1], self.clue[0] - s[0]))

        mono = kind in c.own_monotone_maps
        div = 2.0 if mono else 3.0
        half = float(np.clip(c.own_arc_m / (div * max(r, 1.0)), 0.28, np.pi))
        if mono:
            a_off = -half + 2.0 * half * frac
        else:

            a_off = (3.0 * frac * half) if frac <= (1.0 / 3.0) else half * (2.0 - 3.0 * frac)

        dd.own_r, dd.own_span = r, div * half
        if leg % 2:
            a_off = -a_off
        a = base + a_off
        return self._in_bounds(
            np.array([s[0] + r * np.cos(a), s[1] + r * np.sin(a), 0.0]))

    def _take_over_orphan_lanes(self) -> None:
\
\
\
\
\
\
\
\
\

        live = [i for i in range(self.n) if self.d[i].phase in (SEARCH, CLIMB)]
        if not live:
            return
        n = max(1, self.d[live[0]].nwedge)

        orphans = [w for w, j in sorted(self._home_of.items())
                   if w not in self._orphans_taken
                   and self.d[j].phase not in (SEARCH, CLIMB)]
        if not orphans:
            return

        idle = [i for i in live
                if self.d[i].claim is None and not self.d[i].bailed
                and self.d[i].theta >= 1.0]
        for w in orphans:
            if not idle:
                return
            i = min(idle, key=lambda k: min(abs(self.d[k].wedge - w),
                                            n - abs(self.d[k].wedge - w)))
            idle.remove(i)
            self.d[i].wedge = int(w)
            self.d[i].theta = 0.0
            self.d[i].bailed = True
            self._orphans_taken.add(w)

    def _ring_radius(self, leg: int) -> float:
\

        c = self.cfg
        rings = self.rings
        if leg < len(rings):
            return float(rings[leg])

        far = max(rings)
        cap = self.WORLD.get(self.map_kind, c.search_r1)
        return float(min(max(cap, far),
                         far + c.search_pitch * (leg - len(rings) + 1)))

    def _avoid_climb(self) -> float:
        c = self.cfg
        if self.map_kind in c.mountain_profile_maps and c.mountain_avoid_climb > 0.0:
            return float(c.mountain_avoid_climb)
        return float(c.avoid_climb)

    def _avoid_brake(self) -> float:
        c = self.cfg
        if self.map_kind in c.mountain_profile_maps and c.mountain_avoid_brake > 0.0:
            return float(c.mountain_avoid_brake)
        return float(c.avoid_brake)

    def _own_step(self, i: int) -> float:
\
\
\
\
\
\
\
\
\
\
\
\
\
\

        c = self.cfg
        f = float(c.own_step_m / max(c.own_arc_m, 1.0))
        dd = self.d[i]
        if dd.own_r <= 0.0 or dd.own_span <= 0.0:
            return f
        want = 1.5 * c.waypoint_reach

        if 2.0 * dd.own_r * math.sin(min(0.5 * dd.own_span * f, math.pi / 2.0)) >= want:
            return f
        ratio = min(1.0, want / (2.0 * dd.own_r))
        return float(min(1.0, 2.0 * math.asin(ratio) / max(dd.own_span, 1e-6)))

    def _sweep_step(self, i: int) -> float:
\
\
\
\
\

        c = self.cfg
        if not c.adaptive_sweep or (c.adaptive_sweep_maps
                                    and self.map_kind not in c.adaptive_sweep_maps):
            return c.sweep_step
        dd = self.d[i]
        n = max(1, dd.nwedge)
        width = 2.0 * np.pi / n
        r = max(self._ring_radius(int(dd.theta)), 1.0)
        return float(np.clip((c.sweep_swath / r) / width,
                             c.sweep_frac_min, c.sweep_frac_max))

    def _investigate_target(self, state: np.ndarray, i: int):
\
\
\
\
\
\
\
\
\
\
\
\

        c = self.cfg
        pos = np.asarray(state[i, POS], float)[:2]
        free = [j for j in range(self.n)
                if self.d[j].claim is None and self.d[j].phase in (SEARCH, CLIMB)]
        best, bestd = None, c.investigate_r
        thr = (c.investigate_hits_forest if self.map_kind == "forest"
               else c.investigate_hits_village if self.map_kind == "village"
               else c.investigate_hits)
        for q in self.pads:
            if q.by is not None or q.done:
                continue
            if q.hits < thr or q.hits >= self.min_hits:
                continue
            pad = np.asarray(q.xyz, float)[:2]
            dme = float(np.linalg.norm(pad - pos))
            if dme > bestd:
                continue
            if all(float(np.linalg.norm(pad - np.asarray(state[j, POS], float)[:2]))
                   >= dme for j in free):
                best, bestd = pad.copy(), dme
        if best is None:
            return None
        return self._in_bounds(np.array([best[0], best[1], 0.0]))

    def _spiral_target(self, i: int) -> np.ndarray:
\
\
\
\
\
\
\
\
\
\
\

        c = self.cfg
        dd = self.d[i]
        n = max(1, dd.nwedge)
        width = 2.0 * np.pi / n
        lo = self.wedge_phase + width * dd.wedge

        if dd.span > 1.0:
            grow = width * (dd.span - 1.0) * 0.5
            lo -= grow
            width *= dd.span
        leg = int(dd.theta)
        frac = dd.theta - leg
        r = self._ring_radius(leg)

        a = lo + width * (frac if leg % 2 == 0 else (1.0 - frac))

        return self._in_bounds(np.array([self.clue[0] + r * np.cos(a),
                                         self.clue[1] + r * np.sin(a), 0.0]))

    def _route_kind(self) -> str:
        return self._mass_route_kind()

    def _route_enabled(self) -> bool:
        kind = self._route_kind()

        classification_ready = self._band_locked or kind == "mountain"
        intervention_ready = self.step_i * 0.02 >= self.cfg.route_start_sec
        return bool(classification_ready and intervention_ready
                    and self.route_planner is not None
                    and self.cfg.route_planner
                    and kind in self.cfg.route_plan_maps)

    def _init_learned_route(self) -> None:
        if (self._route_waypoints is not None or not self._route_enabled()
                or self._own_start is None or self.clue is None):
            return
        try:
            route = self.route_planner.plan(
                self._own_start, self.clue, self.n, self._route_kind())
            route = np.asarray(route, dtype=float)
            if route.ndim != 3 or route.shape[0] != self.n or route.shape[2] != 2:
                return
            self._route_waypoints = [[np.asarray(w, dtype=float) for w in route[j]]
                                     for j in range(route.shape[0])]
            self._lane_given = [False] * self.n
            self._lane_pool = []
            self._crumbs = []
            for dd in self.d:
                dd.route_idx = 0
        except Exception:

            self._route_waypoints = None

    def _inherit_lanes(self, state: np.ndarray) -> None:
\
\
\
\
\
\

        if (not LANE_INHERIT or self._route_waypoints is None
                or self._mass_route_kind() != "open"):
            return

        if not hasattr(self, "_crumbs"):
            self._crumbs = []
        for j in range(self.n):
            if self.d[j].phase in (SEARCH, APPROACH, CLIMB):
                q = np.asarray(state[j, POS], float)[:2]
                if not self._crumbs or float(np.linalg.norm(q - self._crumbs[-1])) > 3.0:
                    self._crumbs.append(q.copy())
        if len(self._crumbs) > 4000:
            del self._crumbs[:2000]
        given = getattr(self, "_lane_given", None)
        if given is None:
            self._lane_given = given = [False] * self.n
        takers = [j for j in range(self.n)
                  if self.d[j].phase == SEARCH and self.d[j].claim is None]
        for i in range(min(self.n, len(self._route_waypoints))):
            if given[i]:
                continue
            dd = self.d[i]
            frozen = dd.flew and dd.still > 25
            if not (dd.phase in (DESCEND, DONE) or dd.landed or frozen):
                continue
            route = self._route_waypoints[i]
            rest = route[dd.route_idx:]
            given[i] = True
            dd.route_idx = len(route)
            if not rest:
                continue
            if LANE_MODE == "pool":
                if not hasattr(self, "_lane_pool"):
                    self._lane_pool = []
                C = np.asarray(self._crumbs, float) if self._crumbs else np.zeros((0, 2))
                for w in rest:
                    wc = self._in_bounds(np.array([w[0], w[1], 0.0]))[:2]
                    if len(C) and float(np.min(np.linalg.norm(C - wc, axis=1))) < LANE_VISITED_M:
                        continue
                    if all(float(np.linalg.norm(wc - q)) >= 6.0 for q in self._lane_pool):
                        self._lane_pool.append(np.asarray(wc, dtype=float))
                continue
            cands = [j for j in takers if j != i]
            if not cands:
                continue
            pos_i = np.asarray(state[i, POS], float)[:2]
            j = min(cands, key=lambda q: float(np.linalg.norm(
                np.asarray(state[q, POS], float)[:2] - pos_i)))
            dst = self._route_waypoints[j]
            pending = [np.asarray(w, float) for w in dst[self.d[j].route_idx:]]
            for w in rest:
                wc = self._in_bounds(np.array([w[0], w[1], 0.0]))[:2]
                if all(float(np.linalg.norm(wc - q[:2])) >= 6.0 for q in pending):
                    dst.append(np.asarray(wc, dtype=float))
                    pending.append(np.asarray(wc, dtype=float))

    def _orphan_target(self, i: int, pos: np.ndarray):
\

        pool = getattr(self, "_lane_pool", None)
        if not LANE_INHERIT or LANE_MODE != "pool" or not pool:
            return None
        p2 = np.asarray(pos, float)[:2]
        dists = [float(np.linalg.norm(w - p2)) for w in pool]
        j = int(np.argmin(dists))
        if dists[j] <= self.cfg.waypoint_reach:
            pool.pop(j)
            return self._orphan_target(i, pos)
        w = pool[j]
        return np.array([w[0], w[1], 0.0], dtype=float)

    def _learned_route_target(self, i: int, pos: np.ndarray) -> Optional[np.ndarray]:
        self._init_learned_route()
        if self._route_waypoints is None or i >= len(self._route_waypoints):
            return None
        dd = self.d[i]
        route = self._route_waypoints[i]
        while dd.route_idx < len(route):
            target = (self._in_bounds(np.array([route[dd.route_idx][0], route[dd.route_idx][1], 0.0], dtype=float))[:2] if (FIX_CLAMP and self._mass_route_kind() in FIX_BOX_KINDS) else np.asarray(route[dd.route_idx], float))
            if np.linalg.norm(target - pos[:2]) > self.cfg.waypoint_reach:
                return np.array([target[0], target[1], 0.0], dtype=float)
            dd.route_idx += 1
        return None

    def _mass_route_kind(self) -> str:

        if self.map_kind == "city" and self._ring_kind() != "city":
            return "open"
        return self.map_kind

    def _mass_enabled(self) -> bool:
        kind = self._mass_route_kind()
        allowed_n = self.cfg.mass_plan_n_by_map.get(kind, self.cfg.mass_plan_n)
        return bool(self.posterior is not None and self.cfg.mass_planner
                    and kind in self.cfg.mass_plan_maps and self.n in allowed_n)

    def _init_mass_grid(self) -> None:
        if (not self._mass_enabled() or self._own_start is None
                or self.clue is None or self._mass_xy.size):
            return
        kind = self._mass_route_kind()
        cell = max(4.0, float(self.cfg.mass_cell_m))
        bound = {"city": 75.0, "open": 60.0, "mountain": 120.0,
                 "village": 40.0, "forest": 42.0}[kind]
        axis = np.arange(-bound, bound + 0.5 * cell, cell, dtype=float)
        xx, yy = np.meshgrid(axis, axis, indexing="xy")
        xy = np.column_stack((xx.ravel(), yy.ravel()))
        inside = ((np.abs(xy[:, 0]) <= bound) & (np.abs(xy[:, 1]) <= bound))
        xy = xy[inside]
        logits = self.posterior.score(self._own_start, np.zeros((0, 3)),
                                      self.clue, xy, kind)
        prior = np.exp(logits - float(np.max(logits)))
        self._mass_xy = xy
        self._mass_prior = prior.astype(float)
        self._mass_seen = np.zeros(len(xy), dtype=np.float32)
        self._mass_key_to_idx = {
            (int(round(q[0] / cell)), int(round(q[1] / cell))): k
            for k, q in enumerate(xy)
        }

    def _update_mass_coverage(self, state: np.ndarray, depth: np.ndarray) -> None:

        if (not self._mass_enabled() or not len(self._mass_xy)
                or self.step_i % max(1, int(self.cfg.detect_every)) != 0):
            return
        cell = max(4.0, float(self.cfg.mass_cell_m))
        touched = set()
        for i, dd in enumerate(self.d):
            if dd.phase != SEARCH:
                continue
            image = np.asarray(depth[i], dtype=np.float32)
            if image.ndim == 3:
                image = image[..., 0]
            yaw = float(state[i, RPY][2])
            pos = np.asarray(state[i, POS], float)[:2]
            for u in range(8, 121, 14):
                vals = image[72:121:12, max(0, u - 1):min(128, u + 2)]
                forward = float(depth_to_meters(float(np.percentile(vals, 35.0)),
                                                PD.AP_DEPTH_MAX_M, 0.5))
                rel = ((u + 0.5) / 128.0 - 0.5) * np.radians(PD.AP_FOV_DEG)
                ray_len = min(20.0, forward / max(0.35, float(np.cos(rel))))
                direction = np.array([np.cos(yaw + rel), np.sin(yaw + rel)])
                for along in np.arange(4.0, ray_len + 0.1, 0.5 * cell):
                    q = pos + float(along) * direction
                    k = self._mass_key_to_idx.get((int(round(q[0] / cell)),
                                                   int(round(q[1] / cell))))
                    if k is not None:
                        touched.add(int(k))
        if touched:
            idx = np.fromiter(touched, dtype=np.int64)
            self._mass_seen[idx] += 1.0

    def _replan_mass_targets(self, state: np.ndarray) -> None:
        if not self._mass_enabled() or not len(self._mass_xy):
            return
        active = [i for i, d in enumerate(self.d)
                  if d.phase == SEARCH and d.claim is None]
        for i in active:
            dd = self.d[i]
            if (dd.mass_target_valid and
                    np.linalg.norm(dd.mass_target[:2] - state[i, POS][:2])
                    <= self.cfg.mass_target_reach_m):
                dd.mass_target_valid = False
                dd.mass_target_idx = -1
        need = [i for i in active if not self.d[i].mass_target_valid]
        if not need:
            return
        pos = np.asarray([state[i, POS][:2] for i in need], dtype=float)
        remaining_s = max(1.0, 60.0 - self.step_i * 0.02)
        capacity = np.maximum(12.0, remaining_s * self.cfg.cruise_speed /
                              (1.0 + np.linalg.norm(pos - self.clue[:2], axis=1) / 90.0))
        travel = np.linalg.norm(self._mass_xy[:, None, :] - pos[None, :, :], axis=2)
        owner = np.argmin(travel / capacity[None, :], axis=1)
        found = [p.xyz for p in self.pads if p.hits >= self.min_hits]
        logits = self.posterior.score(
            self._own_start, np.asarray(found, dtype=np.float32).reshape(-1, 3),
            self.clue, self._mass_xy, self._mass_route_kind())
        self._mass_prior = np.exp(logits - float(np.max(logits)))
        residual = self._mass_prior / (1.0 + self.cfg.mass_observation_discount *
                                       self._mass_seen)
        for pad in self.pads:
            if pad.done or pad.hits < self.min_hits:
                continue
            near = np.linalg.norm(self._mass_xy - pad.xyz[:2], axis=1)
            residual[near <= self.cfg.mass_explained_radius_m] *=\
                self.cfg.mass_explained_factor
        occupied = [d.mass_target[:2].copy() for d in self.d if d.mass_target_valid]
        chosen = {d.mass_target_idx for d in self.d
                  if d.mass_target_valid and d.mass_target_idx >= 0}
        for a in sorted(range(len(need)), key=lambda j: (-capacity[j], need[j])):
            candidates = np.flatnonzero(owner == a)
            if not len(candidates):
                continue
            score = residual[candidates] / (1.0 + travel[candidates, a] /
                                             self.cfg.mass_distance_scale_m)
            pick = -1
            for k in candidates[np.argsort(-score)]:
                k = int(k)
                if k in chosen or residual[k] < self.cfg.mass_min_score:
                    continue
                if any(np.linalg.norm(self._mass_xy[k] - q) < self.cfg.mass_target_sep_m
                       for q in occupied):
                    continue
                pick = k
                break
            if pick >= 0:
                dd = self.d[need[a]]
                dd.mass_target = np.array([*self._mass_xy[pick], 0.0])
                dd.mass_target_valid = True
                dd.mass_target_idx = pick
                occupied.append(dd.mass_target[:2].copy())
                chosen.add(pick)

    def _mass_target(self, i: int) -> Optional[np.ndarray]:
        if not self._mass_enabled():
            return None
        dd = self.d[i]
        return dd.mass_target if dd.mass_target_valid else None

    def _clearance(self, depth: np.ndarray, i: int) -> float:
\
\
\
\
\

        d = np.asarray(depth[i], dtype=np.float32)
        if d.ndim == 3:
            d = d[..., 0]
        h, w = d.shape

        band = d[int(h * 0.28):int(h * 0.52), int(w * 0.30):int(w * 0.70)]
        if band.size == 0:
            return PD.AP_DEPTH_MAX_M
        return float(np.percentile(band, 3.0)) * (PD.AP_DEPTH_MAX_M - 0.5) + 0.5

    def _column_clearance(self, depth: np.ndarray, i: int) -> np.ndarray:
\
\
\
\

        d = np.asarray(depth[i], dtype=np.float32)
        if d.ndim == 3:
            d = d[..., 0]
        h = d.shape[0]
        band = d[int(h * 0.28):int(h * 0.56), :]
        if band.size == 0:
            return np.full(d.shape[1], PD.AP_DEPTH_MAX_M, dtype=np.float32)
        return band.min(axis=0) * (PD.AP_DEPTH_MAX_M - 0.5) + 0.5

    def _use_free(self) -> bool:
\
\
\
\
\

        cfg = self.cfg
        if not cfg.free_dir:
            return False
        if self.map_kind == "village":
            return cfg.free_dir_village
        if self.map_kind == "mountain":
            return cfg.free_dir_mountain
        return True

    def _cruise(self, clear: bool) -> float:

        cfg = self.cfg
        if self.is_forest:
            return cfg.forest_speed
        return cfg.cruise_speed_open if clear else cfg.cruise_speed

    def _tall_in_view(self, depth: np.ndarray, i: int) -> bool:
\
\
\
\
\
\
\

        d = np.asarray(depth[i], dtype=np.float32)
        if d.ndim == 3:
            d = d[..., 0]
        h = d.shape[0]
        band = d[:int(h * 0.42), :]
        if band.size == 0:
            return False
        near = float(np.percentile(band, 1.0)) * (PD.AP_DEPTH_MAX_M - 0.5) + 0.5
        return bool(near < self.cfg.tall_near_m)

    def _scene_is_open(self, depth: np.ndarray, i: int) -> bool:
\
\
\
\
\
\

        return bool(float(np.min(self._column_clearance(depth, i)))
                    >= self.cfg.open_clear)

    def _steer_around(self, depth: np.ndarray, i: int, v: np.ndarray) -> np.ndarray:
\
\
\
\
\
\
\
\
\
\
\

        c = self.cfg
        want = v[:2]
        speed = float(np.linalg.norm(want))
        if speed < 1e-6:
            return v
        cols = self._column_clearance(depth, i)
        w = cols.shape[0]

        u = 2.0 * (np.arange(w) + 0.5) / w - 1.0
        bearings = np.arctan(u * np.tan(np.radians(PD.AP_FOV_DEG) * 0.5))
        clear_gate = (c.mountain_steer_clear
                      if (self.map_kind in c.mountain_profile_maps and c.mountain_steer_clear > 0.0)
                      else c.steer_clear)

        if c.steer_soft_margin > 0.0:
            passable = cols > c.steer_soft_margin
        else:
            passable = cols > clear_gate
        if passable.all():
            return v
        if passable.any() and c.steer_boxed_only:
            return v
        if not passable.any():

            out = v.copy()
            keep = (c.mountain_boxed_keep
                    if (self.map_kind in c.mountain_profile_maps and c.mountain_boxed_keep > 0.0)
                    else 0.1)
            out[:2] *= keep
            if keep > 0.1:
                half = w // 2
                lft = float(cols[:half].min()); rgt = float(cols[half:].min())

                dirn = _unit(want)
                lat = np.array([-dirn[1], dirn[0]]) * (1.0 if lft > rgt else -1.0)
                out[:2] = out[:2] + lat * speed * keep
            out[2] = max(out[2], self._avoid_climb())
            return out

        heading = float(np.arctan2(want[1], want[0]))
        yaw = self._yaw_now
        rel = np.arctan2(np.sin(heading - yaw), np.cos(heading - yaw))

        if c.steer_soft_margin > 0.0:

            room = np.clip(cols / max(c.steer_preferred, 1e-6), 0.0, 1.0)
            cost = np.abs(bearings - rel) + c.steer_bias * (1.0 - room)
        else:
            cost = np.abs(bearings - rel) + c.steer_bias * (1.0 - np.clip(
                cols / PD.AP_DEPTH_MAX_M, 0.0, 1.0))
        cost[~passable] = np.inf
        best = float(bearings[int(np.argmin(cost))])
        newh = yaw + best
        out = v.copy()
        out[:2] = np.array([np.cos(newh), np.sin(newh)]) * speed * c.steer_slow
        return out

    def _free_dir(self, depth: np.ndarray, i: int, v: np.ndarray,
                  rpy: np.ndarray, pos: np.ndarray) -> np.ndarray:
\
\
\
\
\
\

        c = self.cfg
        speed = float(np.linalg.norm(v))
        if speed < 1e-6:
            return v
        R = rot_from_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        v_body = R.T @ np.asarray(v, float)

        fov_on = FIX_FOV not in ("0", "", "off") and (FIX_FOV != "city" or self._mass_route_kind() == "city")
        if fov_on and float(v_body[0]) < speed * FREE_DIR_MIN_COS:
            return v
        mtn = self.map_kind == "mountain"
        pick = FD.free_direction(
            depth[i], v_body, PD.AP_DEPTH_MAX_M,
            fov_deg=PD.AP_FOV_DEG, grid=c.free_grid,
            radius=0.12,
            margin=c.free_margin_mountain if mtn else c.free_margin,
            want_clear=c.free_want_clear,
            lookahead=c.free_lookahead_mountain if mtn else c.free_lookahead)
        if pick is None:
            return v
        world = R @ np.asarray(pick, float)
        agree = float(np.dot(world, np.asarray(v, float) / speed))
        out = world * speed * (1.0 if agree > 0.98 else c.free_slow)
        if c.free_brake:

            clr = self._clearance(depth, i)
            if clr < c.avoid_range:
                u = float(np.clip((c.avoid_range - clr) / c.avoid_range, 0.0, 1.0))
                out[:2] *= (1.0 - (1.0 - c.avoid_brake) * u)

        if self.is_forest:
            room = c.forest_ceiling - float(pos[2])
            out[2] = min(float(out[2]), max(0.0, room))
        return out

    def _avoid(self, depth: np.ndarray, i: int, v: np.ndarray,
               pos=None, pad_xy=None) -> np.ndarray:
\
\
\
\
\
\
\

        c = self.cfg

        avoid_range, avoid_brake = c.avoid_range, self._avoid_brake()
        if self.map_kind == "mountain" and c.avoid_range_mtn > 0.0:
            avoid_range = c.avoid_range_mtn
            if pos is not None and pad_xy is not None and c.mtn_yield_r > 0.0:
                d = float(np.linalg.norm(
                    np.asarray(pad_xy, float) - np.asarray(pos, float)[:2]))
                if d < c.mtn_yield_r:
                    k = d / max(c.mtn_yield_r, 1e-6)
                    avoid_range = c.avoid_range + (avoid_range - c.avoid_range) * k
                    avoid_brake = c.avoid_brake + (avoid_brake - c.avoid_brake) * k
        if self.map_kind == "village" and c.avoid_range_village > 0.0:
            avoid_range = c.avoid_range_village
        clear = self._clearance(depth, i)
        if clear >= avoid_range:
            return v
        urgency = float(np.clip((avoid_range - clear) / avoid_range, 0.0, 1.0))

        climb_urgency = urgency
        if pos is not None and pad_xy is not None:
            d = float(np.linalg.norm(np.asarray(pad_xy, float) - np.asarray(pos, float)[:2]))
            if d < c.avoid_yield_radius:
                k = d / max(c.avoid_yield_radius, 1e-6)
                climb_urgency *= c.avoid_yield_floor + (1.0 - c.avoid_yield_floor) * k
        out = v.copy()
        if c.avoid_close_r > 0.0 and clear < c.avoid_close_r:
            close_u = float(np.clip((c.avoid_close_r - clear) / c.avoid_close_r,
                                    0.0, 1.0))
            avoid_brake = min(avoid_brake,
                              1.0 - (1.0 - c.avoid_close_brake) * close_u)
        out[:2] *= (1.0 - (1.0 - avoid_brake) * urgency)
        out[2] = max(out[2], self._avoid_climb() * climb_urgency)

        if self.is_forest and pos is not None:
            room = c.forest_ceiling - float(np.asarray(pos, float)[2])
            out[2] = min(out[2], max(0.0, room))
        return out

    def _separate(self, state: np.ndarray, i: int, v: np.ndarray) -> np.ndarray:
\

        slots = np.asarray(state[i, MATES], float).reshape(7, 7)
        push = np.zeros(3)
        for s in slots:
            if s[6] < 0.5:
                continue
            rel = s[:3]
            dist = float(np.linalg.norm(rel[:2]))
            if dist < 1e-3 or dist > self.cfg.sep_radius:
                continue

            push[:2] -= (rel[:2] / dist) * (self.cfg.sep_radius - dist) / self.cfg.sep_radius
        if not push.any():
            return v
        return v + self.cfg.sep_gain * push

    def act(self, observation) -> np.ndarray:
        st = np.asarray(observation["state"], float)
        self._refine_state = st if st.ndim == 2 else st[None, :]
        for pad in self.pads:
            if pad.by is not None:
                self._refine_apply(pad)
        cfg = self.cfg
        state = np.asarray(observation["state"], dtype=np.float32)
        depth = np.asarray(observation["depth"], dtype=np.float32)
        if state.ndim == 1:
            state = state[None, :]
            depth = depth[None, ...]
        n = state.shape[0]
        if n != self.n:
            self.reset()
            self.n = n
            self.d = [_Drone(theta=0.0) for _ in range(n)]
        self.step_i += 1
        if self.clue is None:
            self.clue = (np.asarray(state[0, POS], float)
                         + np.asarray(state[0, GOAL_OFF], float))
            self._set_z_band(state)
            self._assign_wedges(state)

            self._own_start = np.asarray(state[:, POS], float).copy()
            self._init_mass_grid()

            for i in range(n):
                self.bad.append(np.asarray(state[i, POS], float)[:2].copy())
            mine = np.asarray(state[0, POS], float)
            for s in np.asarray(state[0, MATES], float).reshape(7, 7):
                if s[6] >= 0.5:
                    self.bad.append((mine + s[:3])[:2].copy())

        if self.step_i % cfg.detect_every == 0:
            live = [i for i in range(n)
                    if self.d[i].phase in (SEARCH, APPROACH, CLIMB)]
            if live:
                self._vote_forest(depth, live)
                self._init_mass_grid()
                self._init_learned_route()
                self._inherit_lanes(state)
            if self.is_forest and cfg.forest_geometric:
                det = None
            else:
                det = (self.forest_detector
                       if (self.is_forest and self.forest_detector is not None)
                       else self.detector)
            if live and det is not None:

                rots = [rot_from_rpy(*np.asarray(state[i, RPY], float)) for i in live]
                poses = [np.asarray(state[i, POS], float) for i in live]
                try:
                    batch = det.propose_batch(poses, rots, depth[live])
                except Exception:
                    batch = [[] for _ in live]
                for props in batch:
                    for q in props:
                        c = np.asarray(q.centre, float)
                        self._merge(c, rng=self._refine_rng(c, poses, batch))
            else:
                for i in live:
                    self._detect(state, depth, i)
        self._assign(state)
        self._update_mass_coverage(state, depth)
        if (self._mass_enabled()
                and self.step_i % max(1, int(cfg.mass_replan_steps)) == 0):
            self._replan_mass_targets(state)

        if (self.cfg.rewedge
                and sum(1 for d in self.d
                        if d.phase in (CLIMB, SEARCH)) != self._wedge_n):
            self._assign_wedges(state)
        if self.cfg.lane_takeover:
            self._take_over_orphan_lanes()

        act = np.zeros((n, 5), dtype=np.float32)
        for i in range(n):
            act[i] = self._one(state, depth, i)
        return act

    def _one(self, state: np.ndarray, depth: np.ndarray, i: int) -> np.ndarray:
        cfg = self.cfg
        dd = self.d[i]
        pos = np.asarray(state[i, POS], float)
        vel = np.asarray(state[i, VEL], float)
        agl = float(state[i, ALT]) * 20.0

        agl_prev = dd.agl_prev
        dd.agl_prev = agl
        rpy = np.asarray(state[i, RPY], float)
        yaw = float(rpy[2])

        if dd.phase == DONE:
            return _encode(slew_velocity(np.zeros(3), vel, cfg.max_delta_v),
                           cfg.speed_limit, yaw)

        if agl > 1.5:
            dd.flew = True

        descending_low = (dd.phase == DESCEND and agl < 0.5)
        if cfg.dead_steps > 0 and dd.flew and not descending_low:
            if (dd.prev_xy is not None
                    and float(np.linalg.norm(vel)) < 1e-5
                    and float(np.linalg.norm(pos[:2] - dd.prev_xy)) < 1e-4):
                dd.still += 1
            else:
                dd.still = 0
            dd.prev_xy = pos[:2].copy()
            if dd.still >= cfg.dead_steps:

                on_claim = (self._pad_xy_dist(dd) <= cfg.landed_r) if dd.claim is not None else False
                if dd.claim is not None and not on_claim:
                    self.pads[dd.claim].by = None
                    dd.claim = None
                dd.phase = DONE
                return _encode(slew_velocity(np.zeros(3), vel, cfg.max_delta_v),
                               cfg.speed_limit, yaw)

        pad = self.pads[dd.claim] if dd.claim is not None else None
        v_des = np.zeros(3)

        if dd.phase == CLIMB:
            v_des = np.array([0.0, 0.0, cfg.climb_speed])

            tgt = self._learned_route_target(i, pos)
            if tgt is None:
                tgt = self._orphan_target(i, pos)
            if tgt is None:
                tgt = self._spiral_target(i)
            d = tgt[:2] - pos[:2]

            risen = (float(pos[2]) - float(self._own_start[i][2])
                     if self._own_start is not None else 1e9)

            tilt_now = float(max(abs(float(rpy[0])), abs(float(rpy[1]))))
            if cfg.climb_creep_tilt > 0.0 and tilt_now >= cfg.climb_creep_tilt:
                d = np.zeros(2)
            if float(np.linalg.norm(d)) > 1e-6 and risen >= cfg.climb_clear_m:
                v_des[:2] = _unit(d) * cfg.cruise_speed * cfg.climb_creep
                yaw = float(np.arctan2(d[1], d[0]))

            top = (cfg.forest_alt if self.is_forest else cfg.cruise_alt) - 0.6
            if agl >= top:
                dd.phase = SEARCH
                dd.climb_exit_t = float(self.step_i)

        elif dd.phase == SEARCH:
            tgt = self._learned_route_target(i, pos)
            if tgt is None:
                tgt = self._mass_target(i)
            if tgt is None and cfg.own_annulus:
                tgt = self._own_target(i)
                if tgt is not None:
                    d = tgt[:2] - pos[:2]
                    if float(np.linalg.norm(d)) < cfg.waypoint_reach:
                        dd.own_theta += self._own_step(i)
                        tgt = self._own_target(i)

            if (cfg.investigate and self.map_kind in cfg.investigate_first
                    and self.map_kind not in cfg.investigate_skip):
                cand = self._investigate_target(state, i)
                if cand is not None:
                    tgt = cand
            if (tgt is None and cfg.investigate
                    and self.map_kind not in cfg.investigate_skip):
                tgt = self._investigate_target(state, i)
            if tgt is None:
                tgt = self._orphan_target(i, pos)
            if tgt is None:
                tgt = self._spiral_target(i)
                d = tgt[:2] - pos[:2]
                if float(np.linalg.norm(d)) < cfg.waypoint_reach:
                    dd.theta += self._sweep_step(i)
                    tgt = self._spiral_target(i)
            d = tgt[:2] - pos[:2]
            clear = cfg.open_boost and self._scene_is_open(depth, i)
            cruise = self._cruise(clear)

            if (cfg.post_climb_soft_sec > 0.0 and dd.climb_exit_t >= 0.0):
                elapsed = (float(self.step_i) - dd.climb_exit_t) * 0.02
                if elapsed < cfg.post_climb_soft_sec:
                    cruise = min(cruise, cfg.cruise_speed * cfg.post_climb_speed_frac)
            v_des[:2] = _unit(d) * cruise
            if self.is_forest:
                want_alt = cfg.forest_alt
            elif clear:
                want_alt = cfg.cruise_alt_open
            elif cfg.open_alt > 0.0 and self._is_open_scene():

                want_alt = cfg.open_alt
            else:
                want_alt = (cfg.mountain_alt
                            if (self.map_kind == "mountain" and cfg.mountain_alt > 0.0)
                            else (cfg.cruise_alt if self._use_free() else cfg.cruise_alt_safe))
            cap = cfg.climb_cap
            if self.map_kind == "mountain" and cfg.mountain_climb_cap > 0.0:
                cap = cfg.mountain_climb_cap
            ff = 0.0

            if (cfg.terrain_ff > 0.0 and agl_prev >= 0.0
                    and (not cfg.terrain_ff_maps
                         or self.map_kind in cfg.terrain_ff_maps)
                    and agl < 19.9 and agl_prev < 19.9):
                terrain_rate = float(vel[2]) - (agl - agl_prev) / 0.02
                ff = max(0.0, terrain_rate) * cfg.terrain_ff
            v_des[2] = np.clip(ff + (want_alt - agl), -1.0, cap)
            yaw = float(np.arctan2(d[1], d[0]))

            if FIX_NOGROUND and self._mass_route_kind() in FIX_BOX_KINDS and agl >= 19.9:
                v_des[2] = max(float(v_des[2]), 0.0)
                bound = self.WORLD.get(self._mass_route_kind())
                if bound is not None and float(np.max(np.abs(pos[:2]))) > bound:
                    home = -np.asarray(pos[:2], float)
                    v_des[:2] = _unit(home) * self._cruise(False)
                    yaw = float(np.arctan2(home[1], home[0]))

            if (cfg.yaw_scan_deg > 0.0
                    and self.map_kind in cfg.yaw_scan_maps
                    and self._clearance(depth, i) > cfg.yaw_scan_min_clear):
                steps = max(int(cfg.yaw_scan_steps), 1)
                ph = 2.0 * np.pi * float(self.step_i % steps) / steps
                yaw += float(np.radians(cfg.yaw_scan_deg) * np.sin(ph))

        elif dd.phase == APPROACH:
            d = pad.xyz[:2] - pos[:2]
            dist = float(np.linalg.norm(d))
            want_z = pad.xyz[2] + (cfg.forest_approach_alt if self.is_forest
                                   else cfg.approach_alt)
            top = self._cruise(cfg.open_boost and self._scene_is_open(depth, i))
            v_des[:2] = _unit(d) * min(top, max(0.5, dist * 0.8))
            v_des[2] = np.clip(want_z - pos[2], -1.0, 1.0)
            if dist > 1e-6:
                yaw = float(np.arctan2(d[1], d[0]))
            if agl < AGL_MIN and dist > AGL_FAR:
                v_des[2] = max(v_des[2], AGL_CLIMB)

            if (cfg.stale_refute_s > 0.0
                    and self.map_kind in cfg.stale_refute_maps
                    and pad.last_seen >= 0
                    and cfg.stale_near_r < dist < cfg.stale_refute_r
                    and (self.step_i - pad.last_seen) > int(cfg.stale_refute_s * 50)):
                self._refute(i)
                return
            enter_r = (cfg.descend_enter_r
                       if self.map_kind in cfg.descend_enter_wide
                       else cfg.descend_enter_r_tight)
            if self._endgame_hurry(dist, float(pos[2] - pad.xyz[2])):
                v_des[:2] = _unit(d) * top
                enter_r = max(enter_r, 2.5)
                if dist < enter_r and abs(pos[2] - want_z) < 2.5:
                    dd.phase = DESCEND
            if dist < enter_r and abs(pos[2] - want_z) < 1.2:
                dd.phase = DESCEND

        elif dd.phase == DESCEND:
            d = pad.xyz[:2] - pos[:2]
            dist = float(np.linalg.norm(d))

            v_des[:2] = np.clip(d * 1.2, -0.55, 0.55)
            above = float(pos[2] - pad.xyz[2])
            if (cfg.agl_band_tol > 0.0 and dist <= cfg.agl_band_r
                    and agl < 19.9 and abs(agl - above) <= cfg.agl_band_tol):
                above = agl

            if above > (cfg.forest_flare_alt if self.is_forest else cfg.flare_alt):
                v_des[2] = -(cfg.forest_sink_speed if self.is_forest else cfg.sink_speed)
                if self._endgame_hurry(dist, above):
                    v_des[:2] = np.clip(d * 1.2, -0.9, 0.9)
                    v_des[2] = -max(cfg.sink_speed, 1.6)
            else:
                v_des[2] = (-cfg.descend_speed if dist < cfg.land_radius
                            else -cfg.creep_speed)
                if self._endgame_hurry(dist, above):
                    v_des[2] = -max(cfg.descend_speed, 0.7)
                if dist > (cfg.forest_abort_radius if self.is_forest else cfg.abort_radius):
                    v_des[2] = 0.15

            if cfg.sink_center_cap > 0.0 and dist > cfg.land_radius:
                t_center = dist / 0.55
                lim = max(abs(above) / max(t_center, 1e-6) * cfg.sink_center_cap,
                          cfg.descend_speed)
                v_des[2] = max(v_des[2], -lim)
            if cfg.descend_commit:
                if (not dd.commit_sink and dist < cfg.descend_commit_r
                        and above > cfg.descend_commit_above):
                    dd.commit_sink = True
                if dd.commit_sink:

                    v_des[2] = (-(cfg.forest_sink_speed if self.is_forest
                                  else cfg.sink_speed)
                                if above > (cfg.forest_flare_alt if self.is_forest
                                            else cfg.flare_alt)
                                else -cfg.descend_speed)
            late = (cfg.commit_t > 0.0
                    and self.map_kind not in cfg.commit_t_skip
                    and self.step_i >= int(cfg.commit_t * 50))

            if (pad.hits < cfg.min_hits_to_land and pos[2] - pad.xyz[2] < 1.5
                    and not late
                    and not (cfg.commit_hold_release and dd.commit_sink)):
                v_des[2] = max(v_des[2], 0.0)
            if agl < 0.30:

                dd.stuck += 1
                on_pad = (pad is not None
                          and float(np.linalg.norm(
                              pos[:2] - np.asarray(pad.xyz, float)[:2])) <= cfg.landed_r
                          and pad.hits >= cfg.min_hits_to_land)
                if cfg.landed_done and on_pad:

                    pad.done = True
                    dd.landed = True
                    dd.stuck = 0
                elif dd.stuck > cfg.ground_stall_steps:
                    self._refute(i)
            else:
                dd.stuck = 0

        tilt = float(max(abs(float(rpy[0])), abs(float(rpy[1]))))
        if cfg.tilt_guard_rad > 0.0 and tilt >= cfg.tilt_guard_rad:
            dd.tilt_soft_until = float(self.step_i) + (
                max(0.0, float(cfg.tilt_guard_hold_sec)) / 0.02
            )
        in_tilt_hold = (
            dd.tilt_soft_until >= 0.0 and float(self.step_i) <= dd.tilt_soft_until
        )
        if dd.phase != DESCEND and not in_tilt_hold:

            self._yaw_now = yaw
            if self._use_free():
                v_des = self._free_dir(depth, i, v_des, rpy, pos)
            else:
                if cfg.steer or self.is_forest or self.map_kind in cfg.steer_maps:
                    v_des = self._steer_around(depth, i, v_des)
                v_des = self._avoid(depth, i, v_des, pos=pos,
                                    pad_xy=(pad.xyz[:2] if pad is not None else None))
        if not in_tilt_hold:
            v_des = self._separate(state, i, v_des)

        dv = cfg.forest_delta_v if self.is_forest else cfg.max_delta_v
        if dd.phase == CLIMB and cfg.climb_max_delta_v > 0.0:
            dv = min(float(dv), float(cfg.climb_max_delta_v))

        v_des = np.asarray(v_des, float).copy()

        lo = float(cfg.tilt_fade_lo)
        hi = float(cfg.tilt_fade_hi)
        if hi > lo and tilt > lo:
            fade = float(np.clip((hi - tilt) / (hi - lo), 0.0, 1.0))
            v_des[0] *= fade
            v_des[1] *= fade

        if cfg.tilt_guard_rad > 0.0 and (tilt >= cfg.tilt_guard_rad or in_tilt_hold):
            v_des[0] = 0.0
            v_des[1] = 0.0
            v_xy = np.asarray(vel[:2], float)
            spd_xy = float(np.linalg.norm(v_xy))
            if spd_xy > 0.05 and cfg.tilt_guard_brake > 0.0:
                brake = min(spd_xy, float(cfg.tilt_guard_brake))
                v_des[:2] = -_unit(v_xy) * brake
            v_des[2] = float(np.clip(v_des[2], -0.3, 0.5))
            cap = float(cfg.tilt_guard_speed_cap)
            mag = float(np.linalg.norm(v_des))
            if cap > 0.0 and mag > cap:
                v_des *= cap / mag
            dv = min(float(dv), 0.35)
        v_cmd = slew_velocity(v_des, vel, dv, tilt_rad=tilt,
                              tilt_knee=cfg.tilt_knee, tilt_stop=cfg.tilt_stop,
                              tilt_floor=cfg.tilt_floor,
                              ctrl_aware=self._mass_route_kind() in FIX_SLEW_KINDS)
        return _encode(v_cmd, cfg.speed_limit, yaw)

def _unit(v):
    v = np.asarray(v, float)
    m = float(np.linalg.norm(v))
    return v / m if m > 1e-9 else np.zeros_like(v)

def _encode(v_cmd, speed_limit: float, yaw_rad: float) -> np.ndarray:

    v = np.asarray(v_cmd, float)
    if v.shape[0] == 2:
        v = np.array([v[0], v[1], 0.0])
    mag = float(np.linalg.norm(v))
    d = v / mag if mag > 1e-9 else np.zeros(3)
    speed = float(np.clip(mag / max(speed_limit, 1e-6), 0.0, 1.0))
    y = float(np.clip(((yaw_rad + np.pi) % (2 * np.pi) - np.pi) / np.pi, -1.0, 1.0))
    return np.clip(np.array([d[0], d[1], d[2], speed, y], dtype=np.float32),
                   [-1, -1, -1, 0, -1], [1, 1, 1, 1, 1])
