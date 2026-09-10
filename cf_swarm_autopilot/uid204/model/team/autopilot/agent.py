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
FOREST_TALL_OVERRIDE = 0.60      # UID99: tall fraction this high IS forest
LANE_MODE = "pool"
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
    # COVERAGE SKIP, ported from our pre-UID20 agent (UID20 has no equivalent).
    # We shipped it as cov_grid=True, cov_maps=("forest",); improvements.md
    # lists "cov on maps other than forest" as never really tested, because
    # cov_maps gated it to forest alone. Off by default: agent unchanged.
    cov_grid: bool = False
    cov_maps: tuple = ()          # empty = every map; gated on _mass_route_kind
    cov_cell: float = 4.0
    cov_radius: float = 15.0
    cov_skip_max: int = 6
    # Metres to hold back from the world wall when clamping any target.
    bounds_inset: float = 0.0
    # Metres: after clamping, drop a waypoint landing this close to an earlier one.
    bounds_dedup: float = 0.0
    # Maps the inset applies to; empty = every map. Gate it: open wins, city loses.
    inset_maps: tuple = ()
    # Fleet-level waypoint reallocation. Empty = off (the shipped behaviour).
    realloc_maps: tuple = ()
    realloc_obj: str = "latency"      # or "makespan"
    realloc_first: int = 0            # 0 = pool every waypoint; k = first k only
    # DATA GENERATION ONLY: deliberately vary the plan to teach a surrogate.
    route_perturb: str = ""           # jitter|shuffle|swap|arc|radial; "" = off
    route_perturb_m: float = 0.0
    route_perturb_seed: int = 0
    # Enforce this much separation between non-adjacent route legs. 0 = off.
    lane_spread: float = 0.0
    # False = coverage skip applies to the SPIRAL only, never the route.
    cov_route: bool = True

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
    # UID167: cut to the NEXT spiral waypoint from further out. The sweep
    # is a spiral of waypoints and the drone switches at waypoint_reach=6;
    # at 12 on village it skips the last 6 m of every leg, which is ground
    # the camera has already covered. -1 = use waypoint_reach.
    spiral_reach: float = -1.0
    spiral_reach_maps: tuple = ()

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
    replan_sec: tuple = ()
    replan_maps: tuple = ()
    # H3: a claim-holding drone projected to miss the horizon scores 0.01
    # anyway, so terminal risk is free for it. UID99 gates this to mountain and
    # city as a module constant; exposing it as config lets open be MEASURED.
    # Empty tuple = fall back to the module default, preserving the base.
    hurry_maps: tuple = ()
    # VIEW GATE. Existing PadConfig gates ask "does this blob look like a pad".
    # This asks "was it seen from a geometry we can trust". Measured over 5300
    # logged proposals: true detections arrive at 1-3 deg off the nose, phantoms
    # at 10-35 deg; and on open phantoms sit at 20.8 m against 12.1 m for real
    # ones. Gating bearing<=12 deg and range<=18 m cuts the false-positive rate
    # 3-15x while keeping ~70% of true detections:
    #     open 0.40%->0.00%  city 7.39%->2.50%
    #     forest 4.55%->0.30%  village 28.57%->6.77%
    # 0 disables, so the base is untouched.
    # commit_t drops the claim gate all the way to 1. Village needs 8 because
    # 28.6% of its detections are phantoms, so 8->1 is what sends drones
    # descending on empty ground at t=25.6 s. A floor keeps SOME evidence
    # requirement while still relaxing. 0 = the old behaviour (floor of 1).
    # Obstacle avoidance is OFF during DESCEND -- the guard reads
    # "if dd.phase != DESCEND" -- presumably because near the pad the pad and
    # ground themselves read as obstacles and would push the drone off.
    # But that only holds CLOSE IN. Measured on village: commit25 adds +40
    # OBSTACLE_COLLISION on top of the control's 50, and the damage scales with
    # how early drones commit (floor 2/3/4 -> +32/+17/+0) because committing
    # early buys more time in the one phase that does not look where it is
    # going. This keeps avoidance alive while still far out horizontally and
    # restores the old behaviour inside the radius.
    # 0 = disabled, i.e. current all-or-nothing behaviour.
    descend_avoid_r: float = 0.0
    descend_avoid_maps: tuple = ()
    # PER-MAP commit deadlines. commit_t is a single global value and
    # commit_t_skip does NOT fall back to it -- skipping DISABLES the mechanism
    # entirely, stripping the champion's own commit_t=50 off that map. Measured:
    # open wants 25 (+0.0234), village and forest want the base 50 (village
    # c25 -0.0099 and c35 -0.0019 against base50 +0.0099). One global value
    # cannot serve both. Entries here override commit_t for that map_kind;
    # anything unlisted keeps commit_t.
    # rewedge earns on VILLAGE (+0.0060) and is neutral-to-negative
    # elsewhere (open -0.0017, city ~0). Inside the full stack the forest
    # collision cut fell from -11/40 to -3/51, and rewedge is the only
    # component reaching forest that the standalone arm lacked. Empty =
    # applies everywhere, i.e. current behaviour.
    rewedge_maps: tuple = ()
    commit_t_by_map: tuple = ()
    # Per-map claim bar, same (k,v,k,v) layout as commit_t_by_map. Lets us set
    # mh3 on city+forest while open keeps min_hits_to_claim (mh5). Empty =
    # unchanged. village/mountain keep their dedicated fields (they lock in
    # _set_z_band, before this is applied), so naming them here is a no-op.
    min_hits_by_map: tuple = ()
    commit_floor: int = 0
    view_max_bearing: float = 0.0
    view_max_range: float = 0.0
    view_gate_maps: tuple = ()

    route_start_sec: float = 0.0

    steer: bool = False

    steer_maps: tuple = ()

    steer_boxed_only: bool = False
    steer_clear: float = 6.0
    steer_bias: float = 0.8
    steer_slow: float = 0.75

    approach_alt: float = 4.0
    # UID167: APPROACH horizontal speed is min(cruise, max(0.5, dist*gain)).
    # UID99 hardcoded 0.8. 1.2 closes the last 10 m ~1.5x faster; mountain
    # is skipped back to 0.8 because its approaches are long and steep.
    approach_gain: float = 0.8
    approach_gain_skip: tuple = ()

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
    # UID167: point the camera where the drone is ACTUALLY going. In
    # CLIMB/SEARCH the yaw tracks the intended heading, but the avoidance
    # layer then rotates the velocity -- so while dodging, the drone flies
    # sideways and the detector sees ground it is not covering.
    # avoid_only=True fires on the AVOIDANCE ROTATION exceeding dev rad,
    # not on the yaw error itself.
    yaw_track_dev: float = 0.0
    yaw_track_min_spd: float = 0.3
    yaw_track_skip: tuple = ('mountain',)
    steer_true_frame: bool = False
    yaw_track_avoid_only: bool = False

    # SOFT ABORT. _refute() destroys a pad: marks it done AND blacklists its
    # location (self.bad, enforced at bad_pad_radius=4 m in _merge), so an
    # approach abandoned on a REAL pad blinds the whole fleet to that platform
    # -- and with exactly N platforms for N drones that is a guaranteed lost
    # drone. That asymmetry is why stale_refute has never been switched on.
    # A soft abort instead releases the claim and ZEROES the pad's evidence, so
    # it must be re-confirmed before anyone commits again: cheap to undo if the
    # pad was real, still fatal to a phantom that never reappears. Escalates to
    # a hard refute after soft_abort_max attempts so it cannot oscillate.
    # False keeps the destructive behaviour, i.e. the base.
    stale_refute_soft: bool = False
    soft_abort_max: int = 2
    stale_refute_s: float = 0.0
    stale_refute_maps: tuple = ()
    stale_refute_r: float = 25.0
    stale_near_r: float = 6.0

    sep_radius: float = 3.2
    sep_gain: float = 1.6

    # UID167: APPROACH clips vertical speed to +-1.0 m/s while DESCEND
    # sinks at 2.8. On mountain the pad can be far below and the cap binds
    # for the whole leg, so the drone arrives overhead and still has to
    # wait out its own descent.
    approach_vz: float = 1.0
    approach_vz_maps: tuple = ()
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
    city_min_hits: int = -1            # city-only claim bar; -1 => use min_hits_to_claim
    city_min_hits_frac: float = 0.25   # trust "true city" above this tall-fraction
                                       # (city ~0.52, open ~0.14; 0.25 separates 15/16).
                                       # Stricter than city_tall_frac (0.015), which
                                       # mislabels open as "city" -- so this never leaks
                                       # the city claim bar onto open.

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
    cov_last: Optional[np.ndarray] = None
    cov_skipped: int = 0

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
    aborts: int = 0

class SwarmAutopilotAgent:

    def __init__(self, cfg: Optional[AutopilotConfig] = None,
                 pad_cfg: Optional[PD.PadConfig] = None,
                 detector=None, forest_detector=None, village_detector=None,
                 mountain_detector=None, posterior=None, route_planner=None,
                 city_detector=None):
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
        self.city_detector = city_detector

        self.mountain_detector = mountain_detector
        self.posterior = posterior
        self.route_planner = route_planner
        self.reset()

    def reset(self, *_a, **_k) -> None:
        self.detector = self.general_detector
        self.d: List[_Drone] = []
        self.pads: List[_Pad] = []
        self.bad: List[np.ndarray] = []
        # Fleet-shared coverage stamp. Ported back from our pre-UID20 agent,
        # which UID20 does not carry: it is the only mechanism that acts on the
        # re-sweeping we measured (failed drones revisit 12-15% of their OWN
        # ground; the learned route has 0% planned self-overlap, so the waste is
        # entirely in the stock fallback search).
        self._cov: set = set()
        self.src_counts: dict = {}
        self.clue: Optional[np.ndarray] = None
        self.z_band: Optional[tuple] = None

        self.min_hits = self.cfg.min_hits_to_claim
        self.rings: tuple = self.cfg.search_rings
        self.map_kind = "other"
        self._tall_frac = 0.0
        self.is_forest = False
        self.is_city = False
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
        self._replanned = set()

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
                self._tall_frac = frac
        # Per-map claim override (city/forest/open); applied once map_kind is
        # fixed here, so it wins over the base min_hits_to_claim. village/mountain
        # lock earlier in _set_z_band and never reach this, keeping their fields.
        #
        # ORDER MATTERS. city_min_hits used to be applied BEFORE this and was then
        # silently overwritten by any "city" entry here, which made the one field
        # that CAN separate city from open useless. It has to be applied last.
        # map_kind labels every OPEN world "city" (city_tall_frac=0.015 against an
        # open tall-fraction of ~0.1375), so `min_hits_by_map` cannot tell the two
        # apart -- an "open" entry in it is never read on any map. city_min_hits
        # gates on the stricter city_min_hits_frac=0.25, which DOES separate them
        # (open ~0.1375, city ~0.5232), and is the only per-map claim bar that
        # actually distinguishes city from open.
        _mhbm = self._min_hits_for_map()
        if _mhbm is not None:
            self.min_hits = _mhbm
        if (self.map_kind == "city" and cfg.city_min_hits > 0
                and self._tall_frac > cfg.city_min_hits_frac):
            self.min_hits = cfg.city_min_hits

    def _min_hits_for_map(self):
        bym = tuple(getattr(self.cfg, "min_hits_by_map", ()) or ())
        for k in range(0, len(bym) - 1, 2):
            if str(bym[k]) == self.map_kind:
                return int(bym[k + 1])
        return None

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

    def _merge(self, xyz: np.ndarray, rng=None, view=None) -> None:
        if view is not None and self._view_gate_on():
            vr, vb = view
            if self.cfg.view_max_range > 0.0 and vr > self.cfg.view_max_range:
                return
            if (self.cfg.view_max_bearing > 0.0
                    and vb > self.cfg.view_max_bearing):
                return
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
        ct = float(self.cfg.commit_t)
        bym = tuple(getattr(self.cfg, "commit_t_by_map", ()) or ())
        for k in range(0, len(bym) - 1, 2):
            if str(bym[k]) == self.map_kind:
                ct = float(bym[k + 1])
                break
        if (ct > 0.0
                and self.map_kind not in self.cfg.commit_t_skip
                and self.step_i >= int(ct * 50)):
            gate = max(1, int(self.cfg.commit_floor))
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
        """UID99. A claim-holding drone projected to miss the horizon at
        normal pace scores 0.01 anyway, so terminal risk is free for it."""
        maps = tuple(getattr(self.cfg, "hurry_maps", ()) or ()) or HURRY_MAPS
        if self._mass_route_kind() not in maps:
            return False
        t_left = 60.0 - self.step_i / 50.0
        need = dist / 1.5 + max(0.0, above) / 0.45 + 3.0
        return t_left < need

    def _pad_xy_dist(self, dd) -> float:
        pad = self.pads[dd.claim]
        return float(np.linalg.norm(np.asarray(pad.xyz, float)[:2] - dd.prev_xy))

    def _view_gate_on(self) -> bool:
        if self.cfg.view_max_bearing <= 0.0 and self.cfg.view_max_range <= 0.0:
            return False
        m = tuple(getattr(self.cfg, "view_gate_maps", ()) or ())
        return (not m) or (self.map_kind in m)

    def _refute(self, i: int, soft: bool = False) -> None:
\
\
\
\
\

        dd = self.d[i]
        if dd.claim is not None:
            pad = self.pads[dd.claim]
            if soft and pad.aborts < int(self.cfg.soft_abort_max):
                # release, demand fresh evidence, keep the pad alive
                pad.by = None
                pad.hits = 0
                pad.aborts += 1
            else:
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

    # ── coverage skip, ported from team/out/agent_pre_uid20.py.bak ────────
    # GATED ON _mass_route_kind(), not map_kind: map_kind is "city" on ~97% of
    # open seeds, so `cov_maps=("open",)` read off map_kind would be dead code.
    def _cov_on(self) -> bool:
        c = self.cfg
        if not c.cov_grid:
            return False
        return (not c.cov_maps) or (self._mass_route_kind() in c.cov_maps)

    def _cov_key(self, p) -> tuple:
        c = float(self.cfg.cov_cell)
        return (int(math.floor(float(p[0]) / c)),
                int(math.floor(float(p[1]) / c)))

    def _cov_mark(self, i: int, pos) -> None:
        """Stamp the disc this drone could have detected a pad in, just now."""
        if not self._cov_on():
            return
        dd = self.d[i]
        p = np.asarray(pos, float)[:2]
        if (dd.cov_last is not None
                and float(np.linalg.norm(p - dd.cov_last)) < self.cfg.cov_cell):
            return
        dd.cov_last = p.copy()
        c = float(self.cfg.cov_cell)
        rad = int(math.ceil(float(self.cfg.cov_radius) / c))
        cx, cy = self._cov_key(p)
        rr = (float(self.cfg.cov_radius) / c) ** 2
        for dx in range(-rad, rad + 1):
            for dy in range(-rad, rad + 1):
                if dx * dx + dy * dy <= rr:
                    self._cov.add((cx + dx, cy + dy))

    def _cov_seen(self, p) -> bool:
        return self._cov_key(p) in self._cov

    def _cov_skip(self, i: int, tgt, advance, target):
        """Advance past waypoints whose ground is already swept.

        The cap matters: without it a drone whose whole ring is covered would
        spin the counter to the end and leave the search pattern entirely.
        """
        if not self._cov_on() or tgt is None:
            return tgt
        for _ in range(max(int(self.cfg.cov_skip_max), 0)):
            if not self._cov_seen(tgt):
                break
            advance()
            nxt = target()
            if nxt is None:
                return tgt
            tgt = nxt
        return tgt

    def _in_bounds(self, tgt: np.ndarray) -> np.ndarray:

        kind = self._mass_route_kind()
        r = self.WORLD.get(kind if (FIX_CLAMP and kind in FIX_BOX_KINDS) else self.map_kind)
        if r is None:
            return tgt
        # INSET. Clipping to the wall itself parks the drone ON the boundary,
        # where half its 12 m disc falls outside the world and can never hold a
        # pad. Measured on block 162: 9% of the whole fleet's credited sweep is
        # void cells, 37% for the worst drone. Aiming `bounds_inset` metres
        # inside keeps the same ground within sensor range -- a drone at 48 m
        # still reaches the 60 m wall -- while every part of the disc is live.
        # The boundary is pad-rich (42% of pads sit within 12 m of a wall), so
        # the inset must stay well under the detection radius or those pads are
        # only ever seen at the edge of range. 0.0 = off, the shipped behaviour.
        # Gated per map. It is worth +0.0123 +- 0.0066 on open (block 205,
        # failures 45->39) and -0.0116 +- 0.0128 on city (block 216, failures
        # 48->52). City's wall is at 75 m and the map has buildings, so pulling
        # inward there costs more than the void it saves. Empty = every map.
        im = tuple(getattr(self.cfg, "inset_maps", ()) or ())
        if (not im) or (self._mass_route_kind() in im):
            r = max(float(r) - max(float(self.cfg.bounds_inset), 0.0), 1.0)
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
            self._perturb_route()
            self._realloc_route()
            self._spread_route()
            self._dedup_route()
            self._lane_given = [False] * self.n
            self._lane_pool = []
            self._crumbs = []
            for dd in self.d:
                dd.route_idx = 0
        except Exception:

            self._route_waypoints = None

    def _maybe_replan(self, state) -> None:
        """Re-plan the sweep mid-episode from where the drones actually ARE.

        THE FAILURE THIS TARGETS, measured on 110 seeds of fx_open: the drones
        we lose fly 161.3 m while the champion's same drone flies 76.8 m. They
        keep walking a plan computed at t=0 from their START positions, which
        every metre since has invalidated -- 68.6% of the landings we drop are
        drones that came within a median 6.1 m of a pad and were still in
        SEARCH at the horizon.

        Only drones still SEARCHING with nothing claimed are redirected; a
        drone already converging on a pad is never touched. Costs one extra
        planner forward pass per replan time.
        """
        if not self.cfg.replan_sec or self._route_waypoints is None:
            return
        if not self._route_enabled():
            return
        kind = self._route_kind()
        if self.cfg.replan_maps and kind not in self.cfg.replan_maps:
            return
        rs = self.cfg.replan_sec
        # A single time may arrive as a bare scalar (), and the
        # panel parser only builds a tuple when it sees a . Coerce, because
        # tuple(25) raises and tuple("25/x") silently iterates CHARACTERS --
        # which replanned on garbage and cost -0.5352 in a smoke test.
        if isinstance(rs, (int, float)):
            rs = (float(rs),)
        elif isinstance(rs, str):
            rs = tuple(float(x) for x in rs.replace("|", "/").split("/") if x)
        else:
            rs = tuple(float(x) for x in rs)
        t = self.step_i * 0.02
        for k, when in enumerate(rs):
            if k in self._replanned or t < float(when):
                continue
            self._replanned.add(k)
            live = [i for i in range(self.n)
                    if self.d[i].phase == SEARCH and self.d[i].claim is None]
            if not live:
                continue
            try:
                cur = np.asarray(state[:, POS], float).copy()
                route = np.asarray(self.route_planner.plan(
                    cur, self.clue, self.n, kind), dtype=float)
            except Exception:
                continue
            if (route.ndim != 3 or route.shape[0] != self.n
                    or route.shape[2] != 2):
                continue
            for i in live:
                self._route_waypoints[i] = [np.asarray(w, dtype=float)
                                            for w in route[i]]
                self.d[i].route_idx = 0
            self._replan_hits = getattr(self, "_replan_hits", 0) + len(live)

    def _perturb_route(self) -> None:
        """Deliberately vary the plan, to GENERATE TRAINING DATA -- not to fly well.

        Every panel we have ever run varied config with the routes held fixed, so
        the corpus contains almost no examples of "this route change produced
        that outcome change". A surrogate cannot learn a mapping absent from its
        data: fitted on levels it reached 97.4% accuracy predicting whether a
        drone lands and only +0.055 correlation on route deltas, because it
        learned episode difficulty instead.

        The base route matters as much as the perturbation. Jitter alone samples
        only near the net's own manifold; uniform noise samples a space where
        every route is bad. So the modes span both: local jitter for resolution,
        order and allocation changes at the same coverage, and two structured
        non-net plans (arc, radial) for routes the net would never emit.

        Deterministic: the RNG is seeded from the episode geometry plus a salt,
        so a given (seed, arm) is reproducible. Empty mode = off, shipped
        behaviour, base untouched.
        """
        mode = str(getattr(self.cfg, "route_perturb", "") or "")
        if not mode or self._route_waypoints is None or self.clue is None:
            return
        m = float(getattr(self.cfg, "route_perturb_m", 0.0))
        salt = int(getattr(self.cfg, "route_perturb_seed", 0))
        key = int(abs(float(self.clue[0])) * 1000 + abs(float(self.clue[1])) * 17)
        if self._own_start is not None:
            key += int(abs(float(np.asarray(self._own_start, float).sum())) * 7)
        rng = np.random.RandomState((key * 2654435761 + salt) % (2 ** 31 - 1))

        rw = self._route_waypoints
        n = len(rw)
        kind = self._mass_route_kind()
        bound = float(self.WORLD.get(kind, 60.0))
        cl = np.asarray(self.clue, float)[:2]

        if mode == "jitter":
            for i in range(n):
                for k in range(len(rw[i])):
                    rw[i][k] = np.clip(rw[i][k] + rng.randn(2) * m, -bound, bound)
        elif mode == "shuffle":                    # same set, new order per drone
            for i in range(n):
                if len(rw[i]) > 1:
                    rw[i] = [rw[i][j] for j in rng.permutation(len(rw[i]))]
        elif mode == "swap":                       # same fleet set, new allocation
            flat = [w for wl in rw for w in wl]
            if flat:
                idx = rng.permutation(len(flat))
                out, p = [], 0
                for i in range(n):
                    out.append([flat[idx[p + j]] for j in range(len(rw[i]))])
                    p += len(rw[i])
                self._route_waypoints = out
        elif mode == "arc":                        # structured: sweep at radius m
            for i in range(n):
                if self._own_start is None:
                    break
                s = np.asarray(self._own_start[i], float)[:2]
                a0 = float(np.arctan2(s[1] - cl[1], s[0] - cl[0]))
                span = 0.9
                rw[i] = [np.clip(cl + m * np.array(
                    [np.cos(a0 + (k / max(len(rw[i]) - 1, 1) - 0.5) * span),
                     np.sin(a0 + (k / max(len(rw[i]) - 1, 1) - 0.5) * span)]),
                    -bound, bound) for k in range(len(rw[i]))]
        elif mode == "radial":                     # structured: out along the bearing
            for i in range(n):
                if self._own_start is None:
                    break
                s = np.asarray(self._own_start[i], float)[:2]
                d = s - cl
                nrm = float(np.linalg.norm(d))
                u = d / nrm if nrm > 1e-6 else np.array([1.0, 0.0])
                rw[i] = [np.clip(cl + u * (m * (k + 1) / len(rw[i])), -bound, bound)
                         for k in range(len(rw[i]))]

    def _realloc_route(self) -> None:
        """Re-deal the fleet's waypoints to whichever drone is nearest.

        The net emits a route per drone and those routes cross each other, so
        the same ground can be covered with each drone taking what is near it. This
        is a multi-depot mTSP, except the objective is NOT tour length: drones
        abandon the route as soon as they find a pad (median 13% of the way in),
        so what matters is ARRIVAL time at each patch -- the minimum-latency
        (travelling repairman) objective.

        Measured on block 267, 60 episodes, against the net's own allocation:

            first leg   29.6 m -> 17.0 m   -43%   (-4.3 s at cruise, 56/60 eps)
            latency       2023 -> 1580     -22%
            fleet travel   704 ->  611     -13%
            worst tour     157 ->  166     +6%    <- the regression to watch

        The SET of waypoints is preserved exactly, so coverage is unchanged;
        only who visits what, and in what order, moves. That is what separates
        this from the tail-editing post-processes (2-opt, lane spreading,
        dedup), all of which measured null because drones never reach the tail.

        realloc_first > 0 pools only the first k waypoints of each drone and
        leaves the rest in place, which captures the first-leg win while
        touching less of whatever the net intended later.
        """
        c = self.cfg
        maps = tuple(getattr(c, "realloc_maps", ()) or ())
        if not maps or self._route_waypoints is None:
            return
        if self._mass_route_kind() not in maps:
            return
        if self._own_start is None:
            return
        rw = self._route_waypoints
        n = min(len(rw), len(self._own_start))
        if n < 2:
            return
        starts = [np.asarray(self._own_start[i], float)[:2] for i in range(n)]
        k = max(int(getattr(c, "realloc_first", 0)), 0)
        pool, tails = [], []
        for i in range(n):
            wl = [self._in_bounds(np.array([w[0], w[1], 0.0]))[:2] for w in rw[i]]
            if k > 0:
                pool.extend(wl[:k])
                tails.append(wl[k:])
            else:
                pool.extend(wl)
                tails.append([])
        if len(pool) < n:
            return

        latency = str(getattr(c, "realloc_obj", "latency")).lower() != "makespan"

        def cost(start, pts):
            cur, total, lat = np.asarray(start, float), 0.0, 0.0
            for p in pts:
                total += float(np.linalg.norm(p - cur))
                lat += total
                cur = p
            return lat if latency else total

        # ---- assignment ----------------------------------------------------
        load = [[] for _ in range(n)]
        cap = int(math.ceil(len(pool) / float(n)))
        if latency:
            # hardest-to-place first: the point that most prefers one depot
            order = []
            for p in pool:
                d = np.array([float(np.linalg.norm(p - s)) for s in starts])
                srt = np.sort(d)
                order.append((-(srt[1] - srt[0]) if len(srt) > 1 else 0.0, d, p))
            order.sort(key=lambda x: x[0])
            for _, d, p in order:
                for j in np.argsort(d):
                    if len(load[j]) < cap:
                        load[j].append(p)
                        break
        else:
            # greedy min-max: give each point to the drone whose tour grows least
            for p in sorted(pool, key=lambda q: -float(np.linalg.norm(q))):
                best, bcost = -1, None
                for j in range(n):
                    if len(load[j]) >= cap:
                        continue
                    grow = cost(starts[j], load[j] + [p]) - cost(starts[j], load[j])
                    tot = cost(starts[j], load[j] + [p])
                    key = (tot, grow)
                    if bcost is None or key < bcost:
                        best, bcost = j, key
                if best < 0:
                    best = int(np.argmin([len(x) for x in load]))
                load[best].append(p)

        # ---- ordering: nearest neighbour, then 2-opt on the chosen objective -
        for i in range(n):
            pts = load[i]
            if not pts:
                self._route_waypoints[i] = [np.asarray(w, float) for w in tails[i]] \
                    or self._route_waypoints[i]
                continue
            cur, left, seq = starts[i], list(pts), []
            while left:
                j = int(np.argmin([float(np.linalg.norm(p - cur)) for p in left]))
                seq.append(left.pop(j))
                cur = seq[-1]
            if len(seq) > 2:
                best, bc = seq, cost(starts[i], seq)
                for _ in range(24):
                    improved = False
                    for a in range(len(best)):
                        for b in range(a + 1, len(best)):
                            cand = best[:a] + best[a:b + 1][::-1] + best[b + 1:]
                            cc = cost(starts[i], cand)
                            if cc < bc - 1e-9:
                                best, bc, improved = cand, cc, True
                    if not improved:
                        break
                seq = best
            self._route_waypoints[i] = [np.asarray(w, float)
                                        for w in (seq + tails[i])]

    def _spread_route(self) -> None:
        """Push apart route legs that sweep the same corridor twice.

        Two legs running close and parallel cover one lane with two passes. The
        fix is not to reorder (2-opt reordered 74% of first waypoints and pushed
        pad arrival LATER in 5 of 8 measured cases) but to displace the later
        waypoint perpendicular to the lane it duplicates, so the same ground is
        covered by two lanes instead of one.

        Scope, stated honestly: genuine doubling back is 0.0% on leg 1 and only
        appears from leg 2 (15-28%), while drones fly a median 13% of the route.
        So this cannot help the ~86% that land -- it is aimed at the ~14% that
        exhaust the route, fly all of it, and eat the whole tail redundancy.
        0.0 = off, the shipped behaviour.
        """
        d = float(getattr(self.cfg, "lane_spread", 0.0))
        if d <= 0.0 or self._route_waypoints is None:
            return
        for j, wl in enumerate(self._route_waypoints):
            if len(wl) < 4 or self._own_start is None:
                continue
            try:
                base = np.asarray(self._own_start, float)
                p0 = base[j][:2] if base.ndim == 2 else base[:2]
            except Exception:                                # noqa: BLE001
                continue
            pts = [p0] + [self._in_bounds(np.array([w[0], w[1], 0.0]))[:2]
                          for w in wl]
            for m in range(3, len(pts)):
                a, b = pts[m - 1], pts[m]
                mid = 0.5 * (a + b)
                worst, wd = None, d
                for k in range(m - 2):                       # non-adjacent only
                    c, e = pts[k], pts[k + 1]
                    v = e - c
                    L = float(np.linalg.norm(v))
                    if L < 1e-6:
                        continue
                    t = float(np.clip(np.dot(mid - c, v) / (L * L), 0.0, 1.0))
                    gap = float(np.linalg.norm(mid - (c + v * t)))
                    if gap < wd:
                        worst, wd = (c + v * t), gap
                if worst is None:
                    continue
                push = mid - worst
                nrm = float(np.linalg.norm(push))
                if nrm < 1e-6:
                    v = b - a
                    push = np.array([-v[1], v[0]], float)
                    nrm = max(float(np.linalg.norm(push)), 1e-6)
                moved = pts[m] + (push / nrm) * (d - wd)
                pts[m] = self._in_bounds(
                    np.array([moved[0], moved[1], 0.0]))[:2]
            for m in range(1, len(pts)):
                wl[m - 1] = np.asarray(pts[m], float)

    def _dedup_route(self) -> None:
        """Drop waypoints that CLAMPING has piled onto the same spot.

        The net emits 21% of its waypoints outside the world box. Clipping
        projects them all onto the wall, so two waypoints that were far apart
        can land within one sensor swath of each other and the drone flies the
        same ground twice. Measured on block 184: 6.2% of waypoint pairs land
        within 12 m at the wall, and the inset makes it WORSE at 7.8%, because
        clamping to 54 also pulls in everything between 54 and 60.

        This runs once at plan time on the clamped positions, so it is
        deterministic and adds no runtime state. It never drops the last
        waypoint, and never drops so many that a route falls below two.
        0.0 = off, the shipped behaviour.
        """
        d = float(getattr(self.cfg, "bounds_dedup", 0.0))
        if d <= 0.0 or self._route_waypoints is None:
            return
        for j, wl in enumerate(self._route_waypoints):
            if len(wl) < 3:
                continue
            kept, kept_c = [], []
            for k, w in enumerate(wl):
                c = self._in_bounds(np.array([w[0], w[1], 0.0], dtype=float))[:2]
                if (k < len(wl) - 1 and len(wl) - k > 2
                        and any(float(np.linalg.norm(c - p)) <= d for p in kept_c)):
                    continue
                kept.append(w)
                kept_c.append(c)
            if len(kept) >= 2:
                self._route_waypoints[j] = kept

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
        # COVERAGE SKIP ON THE ROUTE ITSELF. Wiring it only into the spiral
        # fallback made it inert here: on open the learned route supplies 89.8%
        # of SEARCH waypoints, so the fallback is almost never reached and the
        # arm came back bit-identical.
        # It is NOT free, despite the argument that swept ground cannot hide an
        # undetected pad. Measured on block 14900000: skipping opens no coverage
        # hole (4 of the 5 pads it cost were never stamped at all, and no arm
        # drone came within 21 m of them) -- it re-deals which drone ends up
        # where. 5 pads lost, 5 gained, -0.805 against +0.767.
        # cov_skip_max is a per-EPISODE budget (dd.cov_skipped). A local counter
        # reset on every call bounded nothing: this runs each step, so a drone
        # could shed its whole route a few waypoints at a time. Measured:
        # divergence ran to 21 m, corr(div, |dScore|) +0.66, all of it variance.
        while dd.route_idx < len(route):
            target = (self._in_bounds(np.array([route[dd.route_idx][0], route[dd.route_idx][1], 0.0], dtype=float))[:2] if (FIX_CLAMP and self._mass_route_kind() in FIX_BOX_KINDS) else np.asarray(route[dd.route_idx], float))
            if np.linalg.norm(target - pos[:2]) > self.cfg.waypoint_reach:
                if (self._cov_on() and self.cfg.cov_route
                        and dd.cov_skipped < int(self.cfg.cov_skip_max)
                        and dd.route_idx < len(route) - 1
                        and self._cov_seen(target)):
                    dd.route_idx += 1
                    dd.cov_skipped += 1
                    continue
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
        yaw = (getattr(self, '_att_yaw_now', None)
               if (c.steer_true_frame
                   and getattr(self, '_att_yaw_now', None) is not None)
               else self._yaw_now)
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
                self._maybe_replan(state)
                self._inherit_lanes(state)
            if self.is_forest and cfg.forest_geometric:
                det = None
            else:
                det = (self.forest_detector
                       if (self.is_forest and self.forest_detector is not None)
                       else self.city_detector
                       if (self.is_city and self.city_detector is not None)
                       else self.detector)
            if live and det is not None:

                rots = [rot_from_rpy(*np.asarray(state[i, RPY], float)) for i in live]
                poses = [np.asarray(state[i, POS], float) for i in live]
                try:
                    batch = det.propose_batch(poses, rots, depth[live])
                except Exception:
                    batch = [[] for _ in live]
                for oi, props in enumerate(batch):
                    opos = np.asarray(poses[oi], float)
                    ofwd = np.asarray(rots[oi], float)[:, 0]
                    for q in props:
                        c = np.asarray(q.centre, float)
                        w = c[:2] - opos[:2]
                        vr = float(np.linalg.norm(w))
                        if vr > 1e-6:
                            wn = w / vr
                            vb = abs(float(np.degrees(np.arctan2(
                                ofwd[0] * wn[1] - ofwd[1] * wn[0],
                                float(ofwd[:2] @ wn)))))
                        else:
                            vb = 0.0
                        self._merge(c, rng=self._refine_rng(c, poses, batch),
                                    view=(vr, vb))
            else:
                for i in live:
                    self._detect(state, depth, i)
        self._assign(state)
        self._update_mass_coverage(state, depth)
        if (self._mass_enabled()
                and self.step_i % max(1, int(cfg.mass_replan_steps)) == 0):
            self._replan_mass_targets(state)

        _rwm = tuple(getattr(self.cfg, "rewedge_maps", ()) or ())
        _rw_on = self.cfg.rewedge and ((not _rwm) or (self.map_kind in _rwm))
        if (_rw_on
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

        # Stamp coverage before the phase dispatch: the camera is on in every
        # phase and a pad seen in transit is just as seen. The SKIP is confined
        # to SEARCH -- marking and skipping are deliberately not the same set.
        self._cov_mark(i, pos)

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
            # WHICH BRANCH ACTUALLY SUPPLIES THE TARGET. Diagnostic only -- a
            # dict of counters, no behaviour change. Added because covspiral
            # measured bit-identical on 60/60 open seeds, and the question
            # "is the spiral even reached" cannot be answered from scores.
            _src = "route"
            tgt = self._learned_route_target(i, pos)
            if tgt is None:
                tgt = self._mass_target(i)
                _src = "mass"
            if tgt is None and cfg.own_annulus:
                _src = "own"
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
                    _src = "investigate"
            if (tgt is None and cfg.investigate
                    and self.map_kind not in cfg.investigate_skip):
                tgt = self._investigate_target(state, i)
                if tgt is not None:
                    _src = "investigate"
            if tgt is None:
                tgt = self._orphan_target(i, pos)
                if tgt is not None:
                    _src = "orphan"
            if tgt is None:
                _src = "spiral"
                tgt = self._spiral_target(i)
                d = tgt[:2] - pos[:2]
                _sr = (cfg.spiral_reach
                       if (cfg.spiral_reach > 0.0
                           and (not cfg.spiral_reach_maps
                                or self.map_kind in cfg.spiral_reach_maps))
                       else cfg.waypoint_reach)
                if float(np.linalg.norm(d)) < _sr:
                    dd.theta += self._sweep_step(i)
                    tgt = self._spiral_target(i)
                    tgt = self._cov_skip(
                        i, tgt,
                        lambda: setattr(dd, "theta",
                                        dd.theta + self._sweep_step(i)),
                        lambda: self._spiral_target(i))
            self.src_counts[_src] = self.src_counts.get(_src, 0) + 1
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
            _ag = (0.8 if self.map_kind in cfg.approach_gain_skip
                   else cfg.approach_gain)
            v_des[:2] = _unit(d) * min(top, max(0.5, dist * _ag))
            _avz = (cfg.approach_vz if self.map_kind in cfg.approach_vz_maps
                    else 1.0)
            v_des[2] = np.clip(want_z - pos[2], -_avz, _avz)
            if dist > 1e-6:
                yaw = float(np.arctan2(d[1], d[0]))
            if agl < AGL_MIN and dist > AGL_FAR:
                v_des[2] = max(v_des[2], AGL_CLIMB)

            if (cfg.stale_refute_s > 0.0
                    and self.map_kind in cfg.stale_refute_maps
                    and pad.last_seen >= 0
                    and cfg.stale_near_r < dist < cfg.stale_refute_r
                    and (self.step_i - pad.last_seen) > int(cfg.stale_refute_s * 50)):
                self._refute(i, soft=bool(cfg.stale_refute_soft))
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
            _ct = float(cfg.commit_t)
            _bym = tuple(getattr(cfg, "commit_t_by_map", ()) or ())
            for _k in range(0, len(_bym) - 1, 2):
                if str(_bym[_k]) == self.map_kind:
                    _ct = float(_bym[_k + 1])
                    break
            late = (_ct > 0.0
                    and self.map_kind not in cfg.commit_t_skip
                    and self.step_i >= int(_ct * 50))

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
        far_pad = False
        if (self.cfg.descend_avoid_r > 0.0 and dd.phase == DESCEND
                and pad is not None):
            dm = tuple(getattr(self.cfg, "descend_avoid_maps", ()) or ())
            if (not dm) or (self.map_kind in dm):
                far_pad = bool(float(np.linalg.norm(
                    np.asarray(pad.xyz, float)[:2] - pos[:2]))
                    > self.cfg.descend_avoid_r)
        _v_pre = np.asarray(v_des, float)[:2].copy()
        if (dd.phase != DESCEND or far_pad) and not in_tilt_hold:

            self._yaw_now = yaw
            self._att_yaw_now = float(rpy[2])
            if self._use_free():
                v_des = self._free_dir(depth, i, v_des, rpy, pos)
            else:
                if cfg.steer or self.is_forest or self.map_kind in cfg.steer_maps:
                    v_des = self._steer_around(depth, i, v_des)
                v_des = self._avoid(depth, i, v_des, pos=pos,
                                    pad_xy=(pad.xyz[:2] if pad is not None else None))
        if not in_tilt_hold:
            v_des = self._separate(state, i, v_des)
        # how far the avoidance + separation layers rotated the velocity away
        # from what the phase asked for -- the yaw_track_avoid_only gate
        _av_rot = 0.0
        _v_post = np.asarray(v_des, float)[:2]
        if (float(np.linalg.norm(_v_pre)) > cfg.yaw_track_min_spd
                and float(np.linalg.norm(_v_post)) > cfg.yaw_track_min_spd):
            _a0 = float(np.arctan2(_v_pre[1], _v_pre[0]))
            _a1 = float(np.arctan2(_v_post[1], _v_post[0]))
            _av_rot = abs((_a1 - _a0 + np.pi) % (2.0 * np.pi) - np.pi)

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
        if (cfg.yaw_track_dev > 0.0 and dd.phase in (CLIMB, SEARCH)
                and self.map_kind not in cfg.yaw_track_skip):
            _vxy = np.asarray(v_cmd[:2], float)
            _sp = float(np.linalg.norm(_vxy))
            if _sp > cfg.yaw_track_min_spd:
                _yt = float(np.arctan2(_vxy[1], _vxy[0]))
                _dth = abs((_yt - yaw + np.pi) % (2.0 * np.pi) - np.pi)
                _gate = (_av_rot > cfg.yaw_track_dev if cfg.yaw_track_avoid_only
                         else _dth > cfg.yaw_track_dev)
                if _gate:
                    yaw = _yt
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
