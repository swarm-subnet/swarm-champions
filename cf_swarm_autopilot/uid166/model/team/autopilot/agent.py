from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque, Counter as _MCCounter
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
from team.autopilot.memgrid import HeightMemory
from team.detector import pads as PD
from team.detector.localize import depth_to_meters, rot_from_rpy, CAM_FWD, CAM_UP
FIX_SLEW, FIX_FOV, FIX_CLAMP, FIX_NOGROUND, FIX_BELOW = (True, "city", True, True, True)
LANE_INHERIT = True
HURRY_MAPS = ("mountain", "city")
# EHM (r9_prior P5): careful-path remaining time (s) to the landing latch in APPROACH, least-squares on c7 base traces
# (fast path, 46 seeds, landed drones): c0 + c1 d + c2 max(ab, 0) + c3 max(-ab, 0) + c4 max(ab - 4 - 0.667 max(d - 2, 0), 0)
EHM_TC = {"city": (0.868, 0.345, 0.115, 0.0, 0.0), "open": (1.275, 0.340, 0.103, 0.0, 0.581),
          "mountain": (0.911, 0.363, 0.080, 0.102, 0.335), "village": (1.673, 0.354, 0.097, 0.0, 0.062),
          "forest": (1.881, 0.413, 0.0, 0.0, 0.0)}
FOREST_TALL_OVERRIDE = 0.60      
LANE_MODE = "pool"
LANE_VISITED_M = 10.0
FIX_SLEW_KINDS = ("city", "open", "village")
FIX_BOX_KINDS = ("city", "open", "village")
FREE_DIR_MIN_COS = math.cos(math.radians(50.0))
SLEW_G, SLEW_DZ, SLEW_DXY = 0.2646, 0.5, 0.2
SLEW_DVZ_DOWN_MAX = 0.25
SLEW_TILT_MAX = math.radians(40.0)
import os as _yt_os
_YT_DIAG = _yt_os.environ.get("SWARM_YT_DIAG") == "1"   # yaw_time dev diagnostics (per-step cap flags); no effect on actions
from time import perf_counter as _om2_clock                 # candHc OM2 diag timers (counters only; no effect on actions)
def slew_velocity(v_des, v_meas, max_delta_v, tilt_rad=None,
                  tilt_knee=0.30, tilt_stop=0.85, tilt_floor=0.15,
                  ctrl_aware=True):
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
# LANDSEG (research/ideas/REPORT.md 4.2): the landing parameters a per-map cfg.landseg entry may override (_lp)
LANDSEG_KEYS = frozenset((
    "approach_alt", "forest_approach_alt", "approach_vz", "apx_glide", "apx_min_above", "apx_glide_vz",
    "approach_gain", "descend_enter_r", "descend_enter_r_tight", "apx_enter_any_alt_r",
    "dsc_gain", "dsc_lateral", "flare_alt", "forest_flare_alt", "sink_speed", "forest_sink_speed",
    "descend_speed", "creep_speed", "land_radius", "sink_center_cap", "land_settle_r",
    "settle_above_dz", "settle_above_vz"))
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
    # Take-off heading (all default off). Drones spawn facing +x and yaw turns only ~40 deg/s,
    # so a drone whose route leaves sideways flies its first seconds with the camera far off track.
    # climb_yaw_hold: in CLIMB keep aiming the camera at the climb target on steps where the tilt
    #   gate stops the creep (the champion then commands the current yaw, which stalls the turn).
    # early_align_deg > 0: until the camera has once pointed within early_align_deg of the
    #   commanded track in SEARCH, cap horizontal speed at early_align_speed (CLIMB and SEARCH).
    # yaw_scan_align_deg > 0: the yaw scan only swings the camera once the actual yaw is within
    #   this of the track, so it never holds the camera off a track it has not yet turned to.
    climb_yaw_hold: bool = False
    climb_yaw_hold_skip: tuple = ("mountain",)
    early_align_deg: float = 0.0
    early_align_speed: float = 0.6
    early_align_maps: tuple = ("city", "forest", "open", "village", "other")
    yaw_scan_align_deg: float = 0.0
    yaw_scan_steps_mtn: int = 0        # >0: yaw scan period (steps) on mountain instead of yaw_scan_steps
    yaw_scan_deg_mtn: float = 0.0      # >0: yaw scan amplitude on mountain instead of yaw_scan_deg
    yaw_scan_align_maps: tuple = ("city",)
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
    cov_grid: bool = False
    cov_maps: tuple = ()          
    cov_cell: float = 4.0
    cov_radius: float = 15.0
    cov_skip_max: int = 6
    bounds_inset: float = 0.0
    bounds_dedup: float = 0.0
    inset_maps: tuple = ()
    realloc_maps: tuple = ()
    realloc_obj: str = "latency"      
    realloc_first: int = 0            
    route_perturb: str = ""           
    route_perturb_m: float = 0.0
    route_perturb_seed: int = 0
    lane_spread: float = 0.0
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
    route_max_n_by_map: tuple = ()          # ((map, max_n), ...): learned route only for fleets up to max_n on that map
    replan_sec: tuple = ()
    replan_maps: tuple = ()
    hurry_maps: tuple = ()
    descend_avoid_r: float = 0.0
    descend_avoid_maps: tuple = ()
    rewedge_maps: tuple = ()
    commit_t_by_map: tuple = ()
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
    yaw_track_dev: float = 0.0
    yaw_track_min_spd: float = 0.3
    yaw_track_skip: tuple = ('mountain',)
    steer_true_frame: bool = False
    yaw_track_avoid_only: bool = False
    stale_refute_soft: bool = False
    soft_abort_max: int = 2
    stale_refute_s: float = 0.0
    stale_refute_maps: tuple = ()
    stale_refute_r: float = 25.0
    stale_near_r: float = 6.0
    sep_radius: float = 3.2
    sep_gain: float = 1.6
    approach_vz: float = 1.0
    approach_vz_maps: tuple = ()
    same_pad: float = 3.0
    # Pad split (default off): two real pads closer than same_pad get merged into one entry, so a
    # seed with a close pair always leaves a drone without a pad. With pad_split_r > 0, on
    # pad_split_maps, close-range evidence (two proposals in one frame >= pad_split_r apart that
    # match the same entry, or a detection >= pad_split_r from a landed drone) splits the entry.
    pad_split_r: float = 0.0
    pad_split_rng: float = 12.0
    pad_split_maps: tuple = ("village", "forest")
    pad_split_min_hits: int = 6
    # Split next to a landed drone (default off): village pads can overlap down to 0.4-1.5 m and
    # detection error under 8 m range is ~0.04 m. With pad_split_landed_r > 0, rule (b) above (a
    # detection well off the drone that already sits on the entry) uses pad_split_landed_r as its
    # distance threshold and pad_split_landed_rng as its maximum observation range, and the entry
    # it creates keeps other claims pad_split_landed_r (not pad_split_r) away.
    pad_split_landed_r: float = 0.0
    pad_split_landed_rng: float = 8.0
    # vil_open PAIR (default off): rule (a) above only fires when the second proposal is >= pad_split_r (1.5 m)
    # from the entry, but an entry fed by both pads of a close pair sits between them (it averages both
    # before the claim), so pairs < 1.5 m are never split (village, fast b238_base: 197 1.42 m, 245 1.47,
    # 402 1.12 + 1.37, 807 1.01 -- one goal lost each; entry 0.4-0.9 m from both pads). With
    # pad_split_pair_r > 0 on pad_split_pair_maps (route kind): two proposals of ONE frame, >= pad_split_pair_r
    # apart (the detector already dedups proposals < 1.0 m apart in a frame; its error under 8 m is ~0.04 m),
    # observed from < pad_split_pair_rng, that both match one entry that is not done split it: the entry keeps
    # the proposal nearer to its claimant (or the first one when unclaimed), the other becomes a split entry
    # (pad_split_min_hits gate, split claim spacing; with spoken_skip_done it is claimable once the first
    # pad is landed, and split entries skip the teammate separation while landing).
    pad_split_pair_r: float = 0.0
    pad_split_pair_rng: float = 8.0
    pad_split_pair_maps: tuple = ("village",)
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
    village_alt: float = -1.0            # >0: search altitude on village (above the 12 m houses to look down into streets)
    city_alt: float = -1.0               # >0: search altitude on city (above building convex hulls)

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
    # Close-range brake for the steering-only avoider (default off): on near_brake_maps, when the
    # forward clearance band reads less than near_brake_range, scale horizontal speed by
    # clearance / near_brake_range (floored at near_brake_min).
    near_brake_maps: tuple = ()
    # free_ray_half_maps: restrict free_direction to headings within free_ray_half_deg of the
    # camera axis (the frame edge is at ~45 deg), default off.
    free_ray_half_maps: tuple = ()
    free_ray_half_deg: float = 30.0
    # NFB (research/beat_h7y/PLAN.md 2.2), default off: on free_berth_maps (free-direction kind, "forest"
    # when is_forest), when the narrow free_direction pick has no near-field tube (first free_berth_look m)
    # clear at radius + free_berth_margin, re-pick the smallest-angle safe ray that has one, inside
    # max(free_berth_cone_deg, the narrow pick's angle); free_berth_keep_speed keeps the narrow pick's
    # free_slow decision.
    free_berth_maps: tuple = ()
    free_berth_margin: float = 0.65
    free_berth_look: float = 3.0
    free_berth_cone_deg: float = 30.0
    free_berth_keep_speed: bool = True
    # free_direction fixes (default off). free_align_maps: pool the whole depth frame into blocks
    # and aim each ray at its own block (the shipped pooling crops to 120 px and re-projects every
    # obstacle 3-6 deg right/below). free_margin_by_map (map, metres, ...): tube margin per map
    # (city buildings collide as convex hulls that stick out ~0.5 m past the rendered mesh).
    # free_ceiling_maps: on forest, rays climbing faster than the room under forest_ceiling allows
    # over free_ceiling_run metres are not flyable (the climb would be clamped away afterwards).
    # free_boxed_brake > 0: when no ray is clear to free_want_clear, cap speed at
    # max(free_boxed_brake_min, free_boxed_brake * clearance of the chosen ray).
    # climb_free_guard: while climbing within 1 m of the pad, never steer down or sideways.
    free_align_maps: tuple = ()
    free_margin_by_map: tuple = ()
    free_ceiling_maps: tuple = ()
    free_ceiling_run: float = 6.0
    free_ceiling_mode: str = "run"     # "clamp": bound = asin(room / speed), the clamp's own limit
    free_ceiling_room_max: float = 99.0  # no ray exclusion while this far below the ceiling
    free_boxed_brake: float = 0.0
    free_margin_route: bool = False    # free_margin_by_map keyed on the route kind (city seeds routed as open get the open margin)
    free_boxed_brake_min: float = 0.5
    climb_free_guard: bool = False
    climb_pass_maps: tuple = ()        # CLIMB: pass an out-of-view (vertical) command through _free_dir unchanged
    # Mountain speed fill (default off): after terrain avoidance, scale the whole 3-D command back
    # up toward the pre-avoidance speed (capped at speed_limit and nf_max_scale), so the climb angle
    # and path stay the same but the drone uses the 3 m/s it has.
    # Terrain-aware touchdown (default off): on touch_off_maps, sample terrain 0.8-2.0 m around the
    # claimed pad from depth during APPROACH, fit a plane at DESCEND entry and aim the touchdown up to
    # touch_off_max downhill of the pad centre (the clearance minimum is set by the terrain uphill of
    # the platform during the last 0.15 s of touchdown). Only on converged pads (hits, views).
    # Descend edge guard (default off): on dsc_edge_maps, when the downward ray reads more than
    # dsc_edge_gap beyond the height above the pad estimate close to touchdown (or the drone tilts
    # with no sink), treat it as over the platform edge: do not sink, hop and re-centre.
    dsc_edge_maps: tuple = ()
    dsc_edge_gap: float = 0.35
    dsc_edge_above: float = 0.25
    dsc_edge_tilt: float = 0.12
    touch_off_maps: tuple = ()
    touch_off_max: float = 0.25
    touch_off_gain: float = 0.6
    touch_off_min_slope: float = 0.25
    touch_off_min_pts: int = 40
    touch_off_min_hits: int = 50
    touch_off_min_views: int = 3
    touch_settle_r: float = 0.2
    nf_maps: tuple = ()
    nf_phases: tuple = ("SEARCH",)
    nf_max_scale: float = 1.4
    # Take-off creep at the full speed norm on tko_maps (default off); tko_tilt replaces
    # climb_creep_tilt there (0 = no tilt gate).
    tko_maps: tuple = ()
    tko_tilt: float = 0.0
    tko_gate_cam_deg: float = 0.0      # > 0: gate the full-speed creep (camera within this of the target)
    tko_gate_clear_m: float = 6.0
    tko_gate_aglrate: float = 0.5
    # yaw_time (dev, all default off). The env clamps the yaw setpoint to +-0.063 rad/step around the
    # current yaw; the attitude PID then turns ~31-42 deg/s (and loses 10-20 deg while the drone
    # accelerates hard), and every drone spawns facing +x. Fast-path traces of the 238 dev base
    # (8 random seeds per map, tools/yan.py, tools/ylegs.py):
    # - village take-off: early_align (0.6 m/s until the camera is within 35 deg of the track) binds
    #   ~1.9 s per drone (speed deficit 1.51 s in CLIMB + 0.42 s in SEARCH); a 150 deg first leg is
    #   released at ~3.3 s having moved ~2 m.
    # - SEARCH leg switches > 30 deg: village 7.5/min (mean 115 deg, 2.0 s until the camera is back
    #   within 15 deg), open 15/min (70 deg, 1.3 s), forest 12.7/min (45 deg, 0.9 s at free_slow on the
    #   +-30 deg edge rays). The new leg is flown with the camera still on the old one.
    # VT (vt_maps, map key "forest" if is_forest else map_kind): while early_align binds, climb at
    #   vt_vz (instead of 1.4 m/s CLIMB / <= 1.0 m/s SEARCH) and release the cap once the drone is
    #   vt_clear_h above its start with AGL >= vt_min_agl (obstacle tops within 10 m of 30 sampled
    #   village starts reach 5.75 m above the start), instead of waiting for the camera.
    # AY (ay_maps, map key "forest" if is_forest else the route kind, so open seeds read "open"): in
    #   SEARCH on a learned-route target within ay_look_m of the switch point (waypoint_reach before the
    #   waypoint), aim the camera at the path point ay_look_m ahead (through the switch point onto the
    #   next leg), at most ay_max_deg (per map: ay_max_by_map) off the current track.
    vt_maps: tuple = ()
    vt_vz: float = 2.4
    vt_clear_h: float = 6.5
    vt_min_agl: float = 4.0
    ay_maps: tuple = ()
    ay_look_m: float = 10.0
    ay_max_deg: float = 90.0
    ay_max_by_map: tuple = ()
    # RT (route_turn_maps, router kinds): turn-aware router chain (team/route_replan/route_core.py
    #   corridor_chain): a leg dth degrees off the arrival heading costs route_turn_k * max(0, dth -
    #   route_turn_free_deg) extra metres (first leg: the camera yaw at the replan; at t = 0 the spawn
    #   yaw 0). Village SEARCH legs turn 115 deg on average at the > 30 deg switches (7.5/min).
    route_turn_maps: tuple = ()
    # FE (fe_maps, map key "forest" if is_forest else map_kind): forest flies only rays within
    #   free_ray_half_deg (30) of the camera; a command further off is mapped to the edge ray and flown
    #   at free_slow (0.75) while the camera turns (forest traces: 2.1 s per drone with the command
    #   30-90 deg off the camera, progress 0.9-1.7 m/s). FE flies such an edge pick at full speed when
    #   its tube is clear for free_want_clear (6 m) and it lies within fe_max_deg of the command.
    fe_maps: tuple = ()
    fe_max_deg: float = 45.0
    # EE (ee_maps, map key "forest" if is_forest else map_kind): while early_align binds and the track
    #   is ee_max_deg or less off the camera, fly along the camera edge ee_edge_deg toward the track (inside
    #   the 45 deg half-FOV) at up to ee_speed when the depth-image tube along that direction is clear for
    #   ee_clear m, instead of the 0.6 m/s cap along the unseen track (village: 1.88 s per drone at the cap;
    #   a 90 deg first leg makes ~1 m in the 1.6 s the camera needs).
    ee_maps: tuple = ()
    ee_edge_deg: float = 35.0
    ee_max_deg: float = 110.0
    ee_speed: float = 1.65
    ee_clear: float = 6.0
    route_turn_k: float = 0.083
    route_turn_free_deg: float = 30.0
    # search_time DCC (dcc_maps, router kinds; default off): directional credit for the FIRST leg of every
    #   router chain. The chain credits a symmetric sweep_r corridor around each leg, but the camera looks
    #   forward (+-45 deg): from the leg start it never sees ground behind it or beside it (a cell at forward
    #   offset x and lateral offset y along the leg is seen only if x >= |y|). Every chain starts where the
    #   drone is (spawn at t = 0, nothing seen yet; replans: the residual already holds only unseen cells),
    #   so the first leg keeps only cells with x >= |y| - dcc_margin; the rest stays in the residual for later
    #   legs. c7 rows: goals < 10 m / 10-20 m from a start are first detected after 5 s in 30% / 22-38% of
    #   the flat-map seeds (city 297 d2: own goal 4 m from the start, found at 15.4 s after 40 m of flight).
    dcc_maps: tuple = ()
    dcc_margin: float = 0.0
    # search_time CWC (cwc_maps, key "forest" if is_forest else map_kind; default off): climb while cruising.
    #   CLIMB creeps at climb_creep * cruise (1.65 m/s) under the tilt gate and exits only at AGL >= cruise_alt -
    #   0.6 (3.9 m). On mountain starts sit on slopes: the creep runs upslope and the ray-below AGL grows slowly
    #   (c7 traces: CLIMB 2.2 s mean per landed drone at 0.9-1.5 m/s horizontal; 15 m of path 1.4-1.6 s later
    #   than an ideal take-off). CWC hands the drone to SEARCH (full cruise, terrain feed-forward, LAV, climb
    #   cap 3 m/s on mountain) once AGL >= cwc_agl and it has risen >= cwc_rise above its start, with the
    #   camera within cwc_cam_deg of the climb target (0 = no camera gate).
    cwc_maps: tuple = ()
    cwc_agl: float = 2.0
    cwc_rise: float = 1.5
    cwc_cam_deg: float = 0.0
    # search_time HOF (hof_maps, route kind via _mass_route_kind(), key "forest" if is_forest; default off): pad
    #   hand-off at claim time. A pad first seen by a busy (APPROACH) drone is claimed by the nearest FREE drone,
    #   however far (c7 dev traces: 22 of 38 landings whose pad another drone saw first; claims from 28-87 m).
    #   When a new claim is made from more than hof_min_d m, an approaching drone (not committed to its sink)
    #   whose own run-in plus the new claimant's run-in shrinks by more than hof_margin m when they swap pads
    #   takes the new pad and hands its own pad to the far drone. The new pad needs >= hof_min_hits hits. The pair
    #   is added to the 2-opt done set (no swap back). assign_2opt covers the same pair only once both pads have
    #   >= 8 hits, which a pad seen from 20 m off the busy drone's track rarely reaches.
    hof_maps: tuple = ()
    hof_min_d: float = 25.0
    hof_margin: float = 5.0
    hof_min_hits: int = 2
    near_brake_range: float = 2.5
    near_brake_min: float = 0.35
    # Stopping-distance guard (default off): on ttc_maps, the command's component along the
    # drone's actual velocity is capped so it can still stop before the first depth point in a
    # ttc_r tube along that velocity (latency ttc_lat, deceleration ttc_decel, margin ttc_margin).
    # ttc_blind_speed >= 0 also caps horizontal speed while the velocity is outside the camera view.
    ttc_maps: tuple = ()
    ttc_decel: float = 4.0
    ttc_lat: float = 0.25
    ttc_margin: float = 0.6
    ttc_r: float = 0.4
    ttc_min_speed: float = 0.8
    ttc_fov_deg: float = 40.0
    ttc_blind_speed: float = -1.0
    ttc_phases: tuple = ("CLIMB", "SEARCH", "APPROACH")
    # Roof guard (default off): on roof_guard_maps, a drone whose downward ray reads less than
    # roof_agl in SEARCH (or in APPROACH farther than roof_pad_r from its pad) is over a roof or
    # crown: climb at up to roof_climb (feed-forward on how fast the surface below rises) and
    # scale horizontal speed down toward roof_brake_min as the gap closes to roof_stop_agl. The
    # champion climbs at most 1 m/s on flat maps, which loses to a pitched roof at 2-3 m/s.
    # Downward-ray landing check (default off). On ray_probe_maps a drone centred over its pad
    # estimate compares the surface its downward ray hits with the pad height and with the ground
    # height sampled around the pad on the way in. If the ray sees ground, the estimate is off the
    # platform (forest estimates carry ~1.1 m errors on some pads; the platform radius is 0.6 m,
    # so the drone descends beside it into the ground): hold ray_probe_alt above the pad and fly
    # a spiral from ray_probe_r0 to ray_probe_r1 around the estimate until the ray finds the
    # platform, then move the estimate there. Pads less than ray_probe_min_h above the ground are
    # left to the old logic (the two surfaces are too close to tell apart).
    ray_probe_maps: tuple = ()
    ray_probe_min_h: float = 0.6
    ray_probe_alt: float = 1.0
    ray_probe_r0: float = 0.7
    ray_probe_r1: float = 2.0
    ray_probe_speed: float = 1.0
    ray_probe_need: int = 4
    ray_probe_tries: int = 2
    roof_guard_maps: tuple = ()
    roof_agl: float = 2.5
    roof_stop_agl: float = 0.5
    roof_climb: float = 2.5
    roof_gain: float = 1.5
    roof_brake_min: float = 0.2
    roof_pad_r: float = 2.5
    forest_z_lo: float = 1.55
    forest_z_hi: float = 3.35
    forest_clear_m: float = 9.0
    forest_clear_frac: float = 0.18
    forest_min_samples: int = 120
    tall_m: float = 4.0
    tall_near_m: float = 19.0
    city_tall_frac: float = 0.015
    # Flat-map classification fixes (all default off).
    # Village start check: village is locked only if median(start z) is in [0.40, 0.70] AND
    # min(start z) >= village_start_zmin AND max(start z) <= village_start_zmax (0.0 / 99.0 = off).
    # Real village starts sit at 0.45-1.32 m; city seeds taken for village have a start above 6 m
    # or below 0.40 m. Otherwise the flat-map path (pad z band + forest/city/open vote) runs.
    village_start_zmin: float = 0.0
    village_start_zmax_n: int = 1      # starts above village_start_zmax needed to veto village (1 = any)
    village_start_zmax: float = 99.0
    # City/open route kind from vertical surfaces: while t <= kind_wall_sec and the forest vote is
    # still collecting, the mean fraction of valid depth pixels whose surface normal lies within
    # 25 deg of horizontal (walls). On kind_wall_maps (map_kind city/open) the route kind is "city"
    # at >= kind_wall_hi, "open" at <= kind_wall_lo, the tall-fraction rule in between; also "city"
    # once the router has fixed the kind to city from geometry. map_kind itself is not changed.
    kind_wall_maps: tuple = ()
    kind_wall_hi: float = 0.003
    kind_wall_lo: float = 0.0015
    kind_wall_sec: float = 1.5
    # Forest veto from pad evidence: when the forest/city/open vote locks on forest but a pad
    # entry already has >= forest_veto_pad_hits hits below forest_veto_pad_z (collected under the
    # flat band before the lock), classify city/open instead (tall-fraction rule). 0.0 = off.
    forest_veto_pad_z: float = 0.0
    forest_veto_pad_hits: int = 6
    # Height-filter rescue (0 = off): proposals the z band rejects are clustered (within
    # same_pad, score >= zband_rescue_score, outside bad_pad_radius). A cluster with
    # >= zband_rescue_hits hits and z std <= zband_rescue_std whose mean z fits the other flat
    # band while the current band is wrong (forest band and pad_z_lo <= z < zband_rescue_flat_z:
    # switch to city/open; city/open band and forest_z_lo <= z <= forest_z_hi: switch to forest)
    # switches the map kind and becomes a pad entry.
    zband_rescue_hits: int = 0
    zband_rescue_std: float = 0.2
    zband_rescue_score: float = 0.9
    zband_rescue_flat_z: float = 1.3
    zband_rescue_max: int = 16
    claim_drift: float = 1.5
    min_hits_to_land: int = 2
    min_hits_to_claim: int = 5
    village_min_hits: int = 8
    city_min_hits: int = -1            
    city_min_hits_frac: float = 0.25   
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
    # --- v2 additions (all default to the shipped behaviour) ---
    search_alt_by_map: tuple = ()      # ("city", 3.5, "open", 3.5, ...) search AGL per map
    # Search height from the clue's z (the mean pad height +-5 m): aim for clue_z + offset, never
    # lower than clue_alt_min_agl or higher than clue_alt_max_agl above the ground under the drone.
    clue_alt_off_by_map: tuple = ()    # ("mountain", 4.0, ...); empty = off
    # Forward clearance along the flight path instead of a fixed image band: depth pixels are
    # projected to 3-D and only points inside a tube of radius corridor_r around the next
    # corridor_look_s seconds of travel count. Falls back to the band when the travel direction
    # is more than corridor_max_off_deg off the camera axis (that space is not observed).
    # Do not fly fast where the camera cannot see: when the drone's actual heading is more than
    # align_gate_deg0 off its horizontal travel direction, point the camera along the travel
    # direction and cut horizontal speed, down to align_gate_min_speed at align_gate_deg1.
    align_gate_maps: tuple = ()        # ("city", "forest", "village", ...); empty = off
    # Forest canopy guard: both sampled canopy crashes were preceded by the downward height
    # reading falling well under the search target (a crown under and ahead of the drone) while it
    # kept flying at ~1.8 m/s. Brake the horizontal command while that holds; the altitude
    # controller already commands the climb.
    canopy_brake_maps: tuple = ()      # ("forest",); empty = off
    canopy_brake_drop: float = 1.5     # metres below the search target that counts as a canopy
    canopy_brake_frac: float = 0.45    # horizontal speed multiplier while it holds
    align_gate_deg0: float = 35.0
    align_gate_deg1: float = 70.0
    align_gate_min_speed: float = 0.6
    # Descend-entry hit gate (default off): on dsc_hits_gate_maps, a claim that reaches DESCEND
    # with fewer than dsc_hits_gate_min hits is dropped (hard refute below dsc_hits_gate_soft_from
    # hits, soft refute from there, so a real pad seen briefly can be claimed again).
    dsc_gate_soft_maps: tuple = ()     # the champion's 8-hit descend gate refutes softly here (pad stays claimable)
    dsc_hits_gate_maps: tuple = ()
    dsc_hits_gate_min: int = 8
    dsc_hits_gate_soft_from: int = 4
    align_gate_by_map: tuple = ()      # (map, deg0, deg1, min_speed, ...): per-map gate, overrides the above
    corridor_avoid_maps: tuple = ()    # ("mountain",); empty = off
    corridor_phases: tuple = ("SEARCH",)
    corridor_r: float = 1.6
    corridor_look_s: float = 3.0
    corridor_min_m: float = 6.0
    corridor_max_off_deg: float = 38.0
    clue_alt_min_agl: float = 3.5
    clue_alt_max_agl: float = 14.0
    clue_alt_mode: str = "clip"        # "below": upper clamp is the map's default height, so the rule only lowers
    clue_alt_max_n: int = 8            # only fleets up to this size
    clue_alt_avoid_range: float = -1.0 # > 0: mountain SEARCH avoid range while the rule holds a drone low
    clear_track_dir: bool = False      # read forward clearance along the travel bearing
    clear_dir_half_deg: float = 20.0   # half-width of that window, degrees
    assign_hungarian: bool = False     # optimal drone->pad assignment instead of greedy
    assign_reopt: bool = False         # let a closer drone take an already-claimed pad
    assign_reopt_gain: float = 8.0     # metres of travel a swap must save to happen
    land_hits_by_map: tuple = ()       # ("city", 8, ...) per-map hits needed to touch down
    land_hits_late: int = 0            # hits still required after the commit deadline
    claim_hits_late: int = 0           # hits still required to claim after that deadline
    clear_band_deg: tuple = ()         # (lo, hi) world elevation window for forward clearance
    steer_band_deg: tuple = ()         # same, for the per-column clearance the steerer reads
    claim_min_views: int = 0           # distinct viewpoints a pad needs before it is claimed
    land_min_views: int = 0            # ... and before a drone descends onto it
    view_sep_m: float = 6.0            # how far apart two viewpoints must be to count twice
    free_fov_speed: float = -1.0       # speed factor when the flight path is outside the FOV
    free_climb: float = 0.0            # climb rate free_dir may add when boxed in
    free_climb_range: float = 7.0      # clearance below which that climb starts
    land_gate_refute_steps: int = 0     # hold the touchdown gate this long, then give up
    assign_mode: str = ""               # "par": assign pads to maximise the fleet's time score
    assign_speed: float = 2.6           # m/s the assigner assumes for the run-in
    climb_speed_by_map: tuple = ()      # ("village", 2.2, ...) take-off climb rate per map
    assign_land_sec: float = 4.0        # seconds it budgets for the descent
    mem_grid: bool = False              # remember obstacle heights between frames
    # ---- DEV ports from our 69 base (all default off = UID 238 behaviour) ----
    same_pad_by_map: tuple = ()         # DEV: ("village", 1.5, ...) per-map pad merge / claim-spacing radius
    spoken_skip_done: int = 0           # DEV: 1 = done (landed) pads do not block claims on their neighbours
    stall_soft: int = 0                 # DEV: 1 = a ground stall releases the claim softly instead of blacklisting the pad
    free_lat_maps: tuple = ()           # DEV: maps where the safe-ray choice prefers a wide lateral pass
    free_lat_target: float = 0.7        # DEV: lateral clearance (m) at which the pass penalty vanishes
    free_lat_w: float = 1.0             # DEV: penalty weight vs the angle term (45 deg = 1)
    # ---- collide FND (forest no-dive), default off ----
    # b238_base forest in-flight collisions: 10 of the 18 forest collision deaths are SEARCH / APPROACH drones that
    # free_direction steered onto a steeply DESCENDING ray while the command itself asked to climb or hold height
    # (collide traces: forest 4, 60, 65, 239, 438 x2, 448, 497, 690, 976; command vz +0.5..+1.0, flown vz -1.0..-1.7).
    # The ceiling rule (free_ceiling_maps) leaves only rays <= ~5 deg up at cruise, so when the level rays are blocked
    # by a trunk the nearest safe ray in angle is a dive under the canopy; the drone sinks to 0.05-0.3 m above a crown,
    # stump or bush (AGL drops 4 -> 0.3) and hits it. Forest 60: the 1.7 m/s dive pitched the drone 41 deg, the tilt
    # hold then kept following the measured (diving) velocity. FND bounds the ray elevation from below:
    #   el_min = max(min(cmd_el, 0) - fnd_tol_deg, -atan2(max(0, agl - fnd_floor), fnd_run))
    # (cmd_el = world elevation of the command after the ceiling clamp; agl = the downward ray): never more than
    # fnd_tol_deg below the command's own slope, and no descending ray at all within fnd_floor of whatever is below.
    fnd_maps: tuple = ()                # map keys ("forest" if is_forest else map_kind); () = off
    fnd_tol_deg: float = 10.0
    fnd_floor: float = 1.5
    fnd_run: float = 4.0
    # ---- collide BNH (convex-hull wedge pad for the CG extrusion), default off ----
    # b238_base city: 14 of the 18 city collision deaths hit building-n (kenney_commercial, 11.6 x 9.1 x 12.4 m).
    # Its collision shape is the convex hull of the mesh: a 2.0-2.5 m podium (local x -5.8..-2.55, y -3.45..2.55)
    # in front of a 12.4 m tower (x -2.55..5.8, y -4.55..3.65). The hull closes the gap between the podium edges
    # and the tower's edges with invisible wedges. 12 of the 14 contacts (collide traces 102, 310 x2, 360, 429 x2,
    # 508 x2, 572, 765, 804, 908) sit 0.5-2.1 m from any visual surface, at local x -5.1..-2.8, z 3.0-5.8, and
    # 0.2-0.9 m OUTSIDE the podium's y extent (y 2.7..3.4 / -3.8..-4.4): beside the podium, where the 238 CG
    # extrusion (podium-roof points copied up to eye height) has no points. BNH: extruded band points with no
    # structure above them (no pooled point higher than band top + cg_roof_above within cg_roof_cell m: a low roof,
    # not the foot of a wall) are tested with the tube radius + cg_roof_pad, which covers the wedges beside the
    # podium; wall points keep the normal tube.
    cg_roof_pad: float = 0.0            # m added to the tube radius of roof-only extruded points; 0 = off
    cg_roof_above: float = 0.4
    cg_roof_cell: float = 0.6
    # ---- collide DZL (CG columns below the eye), default off ----
    # The 238 CG extrusion copies podium / awning points (world z 2.0-3.2) to eye +0 / +0.8 / +1.6 / +2.4 m only.
    # When the rays ahead are blocked, free_direction then picks a ray diving UNDER the columns: 10 of the 12
    # building-n hull-wedge deaths descend at 0.65-1.6 m/s at impact (z 3.0-4.1) while the APPROACH command climbs
    # to pad z + 4. Columns also at cg_dz_low (e.g. -1.6, -0.8) close that gap. Probe on the 9 building-n seeds with
    # the columns extended in every phase (free_extrude_dz [-1.6, -0.8, 0, 0.8, 1.6, 2.4], config only): +0.086,
    # collisions -8, landings +6; but 20 random city seeds -0.009 (search-path butterflies: 849 lost its 43.6 s
    # landing, 943 / 661 late claims later). DZL adds the low columns only in cg_dz_low_phases.
    cg_dz_low: tuple = ()               # extra column offsets (m, relative to the eye) for the CG extrusion; () = off
    cg_dz_low_phases: tuple = ("APPROACH",)
    # ---- collide VRA (roof-aware cruise on village), default off ----
    # b238_base village in-flight collisions: 9 of the 15 village collision deaths are drones flying at roof level
    # (collide traces 146, 218, 259, 264, 551, 555, 585, 723 APPROACH, 847 SEARCH): houses are 5.2-6.2 m tall
    # (roof ridges / eaves), SEARCH cruises at 6.0 m AGL and APPROACH targets pad z + 4.0 = ~4.4 m, so the approach
    # runs through the roof band. _avoid (the village guard) reads only rows 28-52 %% x columns 30-70 %% of the frame
    # and never steers: roof edges just below eye level (551, 555, 723 at z 5.6-6.2) or 30-80 deg off the axis
    # (146, 264) are not seen, and a wall straight ahead (218, 585) is braked only to 45 %% while the 1.5 m/s climb
    # runs the drone into the eave from below. VRA: from the pooled depth cloud, obstacle points in a corridor
    # vra_look m ahead along the horizontal command (half-width vra_half, world z > vra_zmin, APPROACH: > vra_pad_r
    # from the claimed pad) set a height floor = highest point + vra_margin, held vra_hold s then decaying at
    # vra_decay m/s; below the floor the command climbs (<= vra_vz) and the horizontal speed is capped so the climb
    # finishes vra_stop m before the nearest such point (>= vra_min_speed).
    vra_maps: tuple = ()                # map keys ("forest" if is_forest else map_kind); () = off
    vra_phases: tuple = ("APPROACH",)
    vra_look: float = 7.0
    vra_half: float = 1.0
    vra_zmin: float = 1.0
    vra_margin: float = 0.8
    vra_pad_r: float = 2.5
    vra_hold: float = 1.0
    vra_decay: float = 1.0
    vra_vz: float = 1.5
    vra_gain: float = 2.0
    vra_stop: float = 1.2
    vra_min_speed: float = 0.5
    # ---- collide FAF (align first when the command is out of view), default off ----
    # 6 of the 10 forest dive deaths start with the command 50-160 deg off the camera (collide traces 239 -159,
    # 690 +88..103, 4 +49..67, 438 -72 / -58, 497 -40..-50): free_direction can only return an in-view ray, so the
    # out-of-view command becomes a +-30 deg edge ray, often steeply down, flown at 2.1 m/s in a direction unrelated to
    # the target, and the align gate then turns the camera after that ray instead of toward the target. FAF: in
    # faf_phases on faf_maps, a horizontal command more than faf_deg off the camera heading is slowed to faf_speed
    # (the yaw command already points at the target) until the camera has turned to it.
    faf_maps: tuple = ()                # map keys ("forest" if is_forest else map_kind); () = off
    faf_phases: tuple = ("SEARCH", "APPROACH")
    faf_deg: float = 50.0
    faf_speed: float = 0.6
    # ---- vil_open OCCL: occlusion-aware sweep credit in the corridor router (default off) ----
    # Pool: village SEARCH timeouts (54 drones / 1000 seeds on UID 69, stake 0.0063). Replays of the
    # "in the camera wedge <= 15 m for >= 1 s, never registered" goals (fast path, projected goal pixel
    # vs the depth frame): seed 131 goal 0 166/166 in-image frames occluded, seed 274 goal 0 137/137,
    # seed 177 goals 2/5 251/333 and 48/50 -- occluder 3-6 m from the pad, 2-3 m high (walls, sheds,
    # cars, canopy; pads keep only 1.0 m clearance from bodies at pad height). When a pad is visible the
    # detector fires on ~90% of frames (177 goal 1: 190 hits in 240 visible frames). The router credits a
    # cell as swept from the geometric frustum alone, so the occluded pad's cell is written off after the
    # first pass from the wrong side; late in the episode every cell is swept and the fleet circles the
    # prior peak (30 s oscillations over one 10 m strip on seed 131). With occl_sweep_maps a cell only
    # counts as swept once >= occl_min_rays depth rays end on its ground (ReplanRouter._view_depth), so
    # hidden cells keep their mass and a searcher comes back from another side.
    #   occl_sweep_maps     route kinds (router kind) where this applies; () = off (UID 238 behaviour)
    #   occl_sweep_quality  False: keep the shipped credit model (residual = prior * (1 - P_DET * swept)),
    #                       swept = depth ground hits inside the geometric footprint (occlusion-only change);
    #                       True: the router's accumulated per-cell P(seen) from the depth frame (seen_p)
    #   occl_min_rays       depth rays (stride-4 frame) that must end on a cell's ground to credit it
    occl_sweep_maps: tuple = ()
    occl_sweep_quality: bool = False
    occl_min_rays: int = 2
    # flat_search3 DVR (direction-diverse view revisit, router). Traced c7 SEARCH timeouts (list A): half of the
    # village orphan pads (and a quarter of city's) sat inside some drone's frustum within 20 m on 12-205 detector
    # frames, every one of them occluded (house / shed / car between camera and pad), and all those frames came
    # from ONE or two 45-deg bearing sectors (e.g. 175 frames from one side): the fleet kept looking at the hidden
    # pad from the same side; found pads are registered on their first visible sector. The router's swept credit
    # is direction-blind, so a re-pass from the side that already failed is valued like a pass from the open side.
    # With dvr_maps the router records, per cell, the bearing sectors (dvr_nsec) from which a viewer's footprint
    # covered it; while planning a chain, a leg whose camera (looking along the leg) would view an already-covered
    # cell from an already-used sector collects only dvr_w_same of that cell's residual (unswept cells and new
    # sectors unchanged). Nothing is credited that was not credited before; the searcher is steered to view the
    # swept ground from new sides.
    #   dvr_maps          router kinds; () = off (c7)
    #   dvr_w_same        residual weight of a same-sector re-view
    #   dvr_nsec          bearing sectors
    #   dvr_t0            only chains planned at t >= dvr_t0 s
    #   dvr_max_search    only while at most this many drones search
    dvr_maps: tuple = ()
    dvr_w_same: float = 0.25
    dvr_nsec: int = 4
    dvr_t0: float = 0.0
    dvr_max_search: int = 99
    dvr_rmax: float = 0.0      # > 0: only views within this horizontal range (m) mark a cell's sector
    # flat_search3 DVR-C: candidate exemption. Cells within dvr_cand_r m of an unclaimed, not-done registry entry
    # with >= dvr_cand_hits hits keep full value for same-sector re-views (a same-side re-pass is what confirms a
    # sub-gate village detection: A 867's pad seen at 15 s was claimed on a same-side re-pass at 40 s). 0 = off.
    dvr_cand_hits: int = 0
    dvr_cand_r: float = 4.0
    # flat_search3 OBV (occlusion-blocked views, router; needs dvr_maps on the same kind). Per cell the router also
    # records, from each viewer's depth frame (ReplanRouter._view_depth: >= occl_min_rays rays ending on the cell's
    # ground), whether its ground was ever SEEN, and the bearing sectors from which the cell was inside the geometric
    # footprint but its ground was NOT reached (BLOCKED: a house between camera and cell). For a swept cell whose
    # ground was never seen, a chain leg from a blocked sector collects 0 and a leg from any other sector collects
    # obv_boost x its residual (the swept credit 1 - P_DET was never earned there); ground-seen cells keep DVR.
    # Traced c7 village orphans: 16/35 were in a frustum <= 20 m on 1-205 frames, all blocked by houses 0.2-5 m from
    # the pad, from 1-2 sectors only; found pads are registered on their first visible sector.
    #   obv_maps    router kinds; () = off
    #   obv_boost   residual multiplier for a hidden cell viewed from an unblocked sector
    #   obv_block_all  a hidden cell blocked from >= this many sectors is written off (value 0 from every sector)
    obv_maps: tuple = ()
    obv_boost: float = 3.0
    obv_block_all: int = 3
    # flat_search3 BLR (bad-list rehabilitation, agent registry). A hard refute blacklists the pad's xy
    # (bad_pad_radius): every later proposal there is dropped. Traced c7: real village pads hard-refuted at 4 hits
    # kept producing proposals on the blacklisted spot (704: 113 dropped, 956: 47; forest 162: 20), the one phantom
    # hard refute in the traces 0. With blr_maps, proposals within blr_r of a refuted-pad bad entry are counted per
    # entry (distinct detector ticks); at blr_n the entry leaves the bad list and its pad is restored (done False,
    # hits = the count, aborts + 1); once per spot. Keyed "forest" if is_forest else map_kind. () = off.
    blr_maps: tuple = ()
    blr_n: int = 10
    blr_r: float = 1.0
    # flat_search3 FWC (forward-view corridor, router chain planner). The corridor chain credits a leg with every
    # cell within sweep_r (11 m) of the segment, both sides and around its start, but the camera looks along the
    # leg (+-45 deg, axial <= 20 m): cells beside / behind a leg's start are never seen by it. Traced c7 SEARCH
    # timeouts: 8-24% of orphan pads (forest 7/28, village 4/35, city 3/32) lie within 12 m of a start and were
    # never inside any frustum; drones circled some at 0.2-6 m (city A928: take-off loop around its start, the pad
    # 2.7 m from it always 40-180 deg off the camera). With fwc_maps the chain credits a leg only with the cells it
    # sees ahead: cross-track <= sweep_r, along-track >= fwc_k * cross-track, along-track <= leg + fwc_end.
    #   fwc_maps   router kinds; () = off (c7)
    fwc_maps: tuple = ()
    fwc_k: float = 1.0
    fwc_end: float = 11.0
    mem_lookahead: float = 12.0        # how far ahead the memory is consulted, metres
    mem_want_clear: float = 7.0         # remembered clearance treated as "fine"
    mem_brake: float = 0.45             # speed factor when the memory says it is tight
    mem_steer_deg: tuple = (20.0, 40.0, 60.0, 90.0)   # headings tried when blocked

    # ---- FORK: approach / descent profile (all off by default = champion behaviour) ----
    # Enter DESCEND on horizontal proximity alone (the altitude gap no longer gates it);
    # the sink logic in DESCEND handles whatever height the drone arrives at.
    apx_enter_any_alt: bool = False
    apx_max_above: float = 9.0
    # Glide slope: in APPROACH aim for pad_z + clip(dist * apx_glide, apx_min_above,
    # approach_alt) instead of a flat pad_z + approach_alt, so the descent overlaps the
    # transit.  0 = off.
    apx_glide: float = 0.0
    apx_min_above: float = 1.2
    apx_glide_vz: float = 1.6          # vertical speed cap while gliding
    # Within this horizontal radius of the claimed pad the avoidance layer may brake but
    # may not force a climb (its climb override is what parked drones above pads). 0 = off.
    apx_no_climb_r: float = 0.0
    # DESCEND: hover-and-centre before the final sink instead of sinking off-centre.
    dsc_center: bool = False
    dsc_center_r: float = 0.6          # sink only when inside this radius ...
    dsc_center_above: float = 2.0      # ... unless still higher than this above the pad
    dsc_lateral: float = 0.55          # lateral speed clip while descending
    dsc_gain: float = 1.2
    # Mountain world half-size used by _in_bounds. The champion clamps mountain targets to
    # +-75 m, but the mountain box is 250*gs*0.6 = 90..120 m and pads reach |xy| ~100 m
    # (task_gen: TYPE_3_WORLD_RANGE_RATIO). -1 keeps the champion's 75.
    world_mountain: float = -1.0
    # SEARCH altitude hold: the champion clips the descent rate to 1.0 m/s, so over a valley
    # the drone floats 15-20 m above the ground and a pad 12 m away is beyond the 20 m depth
    # range. Per-map cap on the descent rate toward want_alt (m/s); () = every map.
    search_sink_cap: float = 1.0
    search_sink_maps: tuple = ()
    # DESCENT EVIDENCE GATE. Measured on 150 seeds of the champion: pads it descended on
    # that were phantoms had 1-7 detector confirmations (p75 <= 10), real ones 38-99
    # (p10). A real pad keeps re-confirming all the way through the approach; a phantom
    # does not. So refuse to start the descent on a pad with fewer than this many hits and
    # refute it instead (the drone goes back to searching). 0 = off, champion behaviour.
    dsc_min_hits: int = 0
    # ... but not for pads claimed from closer than this (too few detector ticks before the
    # pad drops below the camera), and not after this many seconds (a phantom then costs
    # nothing extra: timeout and collision both score 0.01, while a real pad still lands).
    dsc_gate_min_dist: float = 8.0
    dsc_gate_until_sec: float = 55.0
    # Map kinds (per _mass_route_kind) the gate applies to; () = all. Open is excluded on
    # purpose: it has no phantom descents to prevent, and ~3% of its real pads are
    # confirmed fewer than 8 times before the descent (the pad drops out of view early).
    dsc_gate_maps: tuple = ()
    # Per-map override of dsc_gate_min_dist, (kind, metres, kind, metres, ...). Forest pads
    # are claimed from close range (trees limit visibility) so its gate needs a shorter
    # minimum; forest phantoms carry 3-4 hits against >= 22 for real pads.
    dsc_gate_min_dist_by_map: tuple = ()
    # The any-altitude DESCEND entry only fires this close to the pad (the endgame-hurry
    # radius of 2.5 m is too far to start a 2.8 m/s sink from high above).
    apx_enter_any_alt_r: float = 1.2
    # APPROACH stall watchdog: a drone whose distance to its claimed pad has not improved
    # by apx_stall_gain metres for apx_stall_s seconds is blocked (a wall between it and the
    # pad keeps the avoidance layer braking); release the claim softly and search on.
    apx_stall_s: float = 0.0
    # FOREST OVERHEAD CHECK: canopy crashes cluster 2-5 s after takeoff, 5-9 m from the start,
    # at z 3.7-4.6: the drone climbs to forest_alt into a crown that overhangs it. The crown is
    # visible in the upper rows of the depth frame before the drone is under it, so while the
    # upper band shows something closer than this, do not climb (hold or sink). 0 = off.
    forest_overhead_m: float = 0.0
    forest_overhead_rows: tuple = (0.06, 0.40)   # image row band (fractions from the top)
    forest_overhead_pct: float = 10.0
    apx_stall_gain: float = 0.5
    apx_stall_maps: tuple = ()

    # ---- FORK: HUNT for the last pads. Measured on the champion: 14% of city pads are
    # never within 20 m of any drone's field of view, while its detector misses ~1%.
    # Once at most `hunt_max_unfound` pads are still missing, the searching drones stop
    # walking the generic route/spiral and steer at the cells where the exact generator
    # posterior (team/autopilot/geo_posterior.py), conditioned on the pads already
    # confirmed, still expects a pad and that no camera has swept yet.
    hunt: bool = False
    hunt_max_unfound: int = 2
    hunt_maps: tuple = ()             # per _mass_route_kind; () = every map
    hunt_reach: float = 6.0           # a target counts as visited inside this radius
    hunt_vis_r: float = 14.0          # camera sweep radius credited as coverage
    hunt_dist_scale: float = 30.0     # cell score = residual mass / (1 + dist / scale)
    hunt_sep: float = 12.0            # keep hunters' targets this far apart
    hunt_min_mass: float = 0.02       # below this expected-pad mass a cell is not worth a leg
    hunt_start_sec: float = 0.0
    hunt_max_n: int = 0               # >0: hunt only for fleets up to this size
    hunt_switch_ratio: float = 1.6    # re-target only for a cell this much better
    # mtn_n2 (research/regress/mtn_n2): phantom-safe conditioning for the n=2 mountain survivor hunt.
    hunt_found_min_hits: int = 0      # >0: a claimed, not yet landed pad conditions the hunt only at this many hits
                                      # (mountain claims fire at 2 hits; 3 of 40 first-claim phantoms reach 8)
    hunt_need_found: bool = False     # hunt only while at least one pad is found (never steer on the bare prior)

    # ---- WPOST-X (research/ideas/REPORT.md 4.1): mountain world-size posterior mixture. The half-extent W
    # is drawn per seed in [90, 120] m and the starts are uniform in the box, so the starts give a
    # posterior over W (team/route_replan/world_post.py). Every key default off (bit-identical).
    world_post_router: bool = False   # router field = mean of the pad fields over the W components (n >= min_n)
    world_post_mix: int = 3           # components: 3 = the 1/6, 1/2, 5/6 quantiles; 1 = one point estimate ...
    world_post_stat: str = "mean"     # ... of this statistic (mean / median / qNN)
    world_post_margin: float = 0.0    # metres added to every component
    # candHm WPN (research/champdeep/u149_vs_ours.md s8): extra metres per fleet size, ((n, m), ...). At n >= 6 the
    # posterior is tight and WPX's middle component sits 3-4 m inside the true world edge, so edge pads are found
    # later than with UID 149's soft world; a +2 m shift gained +0.015 per n6-8 seed on two fresh plans (SW1).
    # () = off (bit-identical).
    world_post_margin_by_n: tuple = ()
    world_post_min_n: int = 3         # the router branch needs at least this many drones
    world_post_cem: bool = False      # mountain CEM fleets (search_plan_max_n_by_map, n=2): MAP_PARAMS world
                                      # = Wmax and geo_posterior.WORLD_Q = components (CEM layouts + HUNT8 posterior)
    world_post_clamp_m: float = -1.0  # >= 0: while a branch is active, _in_bounds clamps mountain at Wmax + this
    world_post_lo: float = 90.0
    world_post_hi: float = 120.0
    world_post_nudge: float = 4.0

    # ---- FORK: assignment sanity. The champion's _assign is a greedy nearest-first match
    # with no notion of time: on set B it sends 7% of mountain drones (2-5% elsewhere) on
    # trips they cannot finish (a drone was handed a pad 128 m away with 37 s left), and an
    # approaching drone never switches to a closer pad discovered on the way.
    assign_reach: bool = False        # reachability filter + hopeless-claim release
    assign_swap: bool = False         # switch to a much closer free pad while approaching
    assign_v_eff: float = 2.4         # effective approach speed for the reachability test
    assign_land_sec: float = 4.0      # descend-and-settle allowance
    assign_swap_ratio: float = 0.5    # switch to a free pad this much closer than the claim
    assign_swap_min_dist: float = 15.0

    # ---- FORK: touchdown handling. A drone that touches down on the pad's edge (0.35-0.6 m
    # off centre) keeps being dragged sideways toward the centre while resting on the rim
    # and tips over (TILT, ~0.3% of drones on every map). Instead: inside land_settle_r hold
    # the lateral command at zero and let it settle; outside it hop back up to
    # land_hop_alt above the pad and re-centre in the air.
    land_settle: bool = False
    land_settle_r: float = 0.35
    land_hop_alt: float = 0.7
    land_hop_max: int = 3
    # ---- LANDSEG P4b settle-by-height (research/ideas/REPORT.md 4.2). On raised pads the downward ray reads
    # ~0.6 m after contact, so the land_settle contact test (agl < 0.30) never fires there and the lateral
    # centring push keeps dragging the drone on the pad until it tips (the glide / lower-approach TILT). On
    # settle_above_maps (kind = "forest" when is_forest, else map_kind) contact also counts when the drone is
    # within settle_above_dz of the claimed pad estimate's z and not sinking faster than settle_above_vz
    # (vel z > settle_above_vz). Every other land_settle gate (settle radius, hop, edge) is unchanged. Only the
    # land_settle branch uses it; the ground-stall / landed_done counter keeps agl < 0.30. Empty = off.
    settle_above_maps: tuple = ()
    settle_above_dz: float = 0.12
    settle_above_vz: float = -0.4
    # ---- LANDSEG P4c tilt-rest hop (batch 7 extra; default off). The approach_alt 3.0 failures are not agl-keyed:
    # the drone touches down with ~0.2 rad of braking tilt 0.2-0.3 m off centre, the settle zeroes the lateral
    # command, and the resting tilt then creeps to 0.28-0.30 rad, above the 0.26 rad latch limit, so it never
    # latches (hover to the cap, or 40 s on the pad). On settle_tilt_maps, while the settle holds the drone inside
    # the settle radius (contact by agl or P4b), a tilt >= settle_tilt_rad with |vz| < 0.2 for settle_tilt_s
    # starts the existing hop (climb to land_hop_alt, re-centre, land again; counts against land_hop_max).
    settle_tilt_maps: tuple = ()
    settle_tilt_rad: float = 0.262
    settle_tilt_s: float = 0.3
    # ---- LANDSEG per-map landing overrides {kind: {param: value}} (kind as above), read through _lp() at the
    # APPROACH / DESCEND / settle reads of the parameters in LANDSEG_KEYS. drone_agent.py installs it from the
    # bundle's top-level "landseg" table. Empty = today's behaviour.
    landseg: dict = field(default_factory=dict)

    # ---- FORK: fleet search planner (team/autopilot/search_planner.py). At t=0 the route
    # net's plan is refined by CEM against the exact generator posterior with a fast
    # surrogate of the episode; the optimisation is spread over the first acts (the drones
    # are still climbing) under a per-act time budget.
    search_plan: bool = False
    search_plan_maps: tuple = ("open",)
    search_plan_budget_first: float = 0.1   # seconds of CEM in the act that initialises the route
    search_plan_budget: float = 0.15        # seconds per later act until the iterations are done
    search_plan_layouts: int = 16
    search_plan_pop: int = 32
    search_plan_iters: int = 10
    search_plan_det_range: float = 24.0
    search_plan_blind_r_by_map: tuple = ()  # (map, metres, ...): ground nearer than this is below the camera frame
    search_plan_sector: float = 120.0
    search_plan_min_gain: float = 0.015     # adopt the CEM plan only if the surrogate gain exceeds this
    search_plan_climb_delay: float = 0.5    # surrogate: seconds before the drones start moving
    search_plan_land_delay: float = 3.0     # surrogate: seconds from arrival over a pad to landing
    search_plan_fixed_prefix: int = 2       # keep this many leading waypoints of the warm start
    search_plan_sigma0: float = 6.0         # initial CEM spread (m)
    search_plan_arc_n: int = 0              # fleets up to this size also try posterior-arc sweep plans
    search_plan_holdout: bool = False       # judge the plan's gain on held-out posterior layouts
    search_plan_defer_steps: int = 0        # start the planner this many acts after the route (keeps the first act cheap)
    search_plan_arc_only_maps: tuple = ()   # on these maps only the global (arc) candidates are tried, no CEM
    search_plan_max_n_by_map: tuple = ()    # ((map, max_n), ...): plan only for fleets up to max_n on that map
    search_plan_map_params: tuple = ()      # ((map, det_range, sector, climb_delay, land_delay), ...) per-map surrogate calibration
    # Re-plan the free drones' sweep whenever the set of found pads changes: the posterior of
    # the remaining pads is conditioned on the found ones (ownership enumerated), and a CEM
    # from the drones' CURRENT positions replaces their remaining waypoints when the held-out
    # gain clears search_replan_min_gain.
    search_replan: bool = False
    search_replan_maps: tuple = ()          # () = search_plan_maps
    search_replan_max_n: int = 3            # fleets up to this size (the centroid clue only binds small fleets)
    search_replan_until_sec: float = 45.0
    search_replan_gap_sec: float = 4.0      # minimum sim seconds between two re-plans
    search_replan_iters: int = 8
    search_replan_layouts: int = 32
    search_replan_pop: int = 32
    search_replan_min_gain: float = 0.05
    search_plan_budget_steps: int = 0       # 0 = keep stepping until done; else stop after this many acts

    # ---- FORK: SPIN SEARCH. The only sensor is a forward 90-degree depth camera, but yaw is
    # decoupled from the flight direction (velocity commands are world-frame; yaw is rate
    # limited to pi rad/s). While searching, rotate the camera continuously so the swept
    # sector is the full circle every 2*pi/rate seconds instead of the forward cone. Only
    # after the map is classified (the classifier votes on forward views) and only on maps
    # where flying without a forward view is safe. 0 = off.
    search_spin_rate: float = 0.0           # rad/s of yaw rotation while in SEARCH
    search_spin_maps: tuple = ("open",)
    search_spin_start_sec: float = 3.0

    # ---- FORK: PREDICTIVE TERRAIN FOLLOWING (mountain). The altitude hold reads the ray
    # straight below, so a drone crossing a ridge is still 15-20 m above the valley floor
    # when the far pad comes into the camera's sector, beyond the 20 m depth range: 12.7% of
    # mountain pads on set B were in range and in view yet never detected. The camera sees
    # the valley ahead earlier than the ray below: estimate the ground ahead from the depth
    # frame and hold the search altitude relative to the LOWER of the two.
    terrain_ahead: bool = False
    terrain_ahead_maps: tuple = ("mountain",)
    terrain_ahead_min_m: float = 5.0        # horizontal window ahead used for the ground estimate
    terrain_ahead_max_m: float = 18.0
    terrain_ahead_pct: float = 20.0         # percentile of ground z in the window (low = valley floor)
    terrain_ahead_min_pts: int = 60
    # ---- mtn_speed LAV: look-ahead vertical law (SEARCH, keyed on map_kind; () = off).
    # The 3 m/s cap is a 3-D norm, so every m/s of |vz| costs horizontal search speed (h = 3 / sqrt(1 + slope^2)).
    # Stock mountain SEARCH (c1 traces, 8 random seeds, 70k steps): h 2.52 m/s, |v| 2.88, flown arc / horizontal
    # 1.140, AGL mean 8.3 / p10 6.0 / p90 12.1 (want 6). The vertical is a saw: the AGL P-law sits at the -2.5 sink
    # cap on 27% of steps (drone too high), while _avoid's climb (depth band 3rd pct < 20 m, vz >= 2.5 * urgency)
    # overrides it on 34% of steps (+2.2 m/s on average, 40% of all vz^2); ground ahead does rise (median +7 m
    # within 20 m when it fires), so the climbs are needed, but they start late and hard. Offline (tools/taut.py,
    # tools/mpc.py on the recorded ground under the path): a taut string in AGL [5, 12] with perfect foresight
    # flies arc 1.089 at the same AGL mean 8.3; the greedy funnel below with 10-15 m look-ahead and kp 0.3 flies
    # arc 1.09-1.105 (h +3..+5%) at AGL mean 7.5-7.9, p90 10.6-11.5; the stock P-law without the avoid climb
    # (arc 1.146) goes 6.7 m through the terrain.
    # Law: a fleet-shared max-height grid (lav_cell m) is fed from every flying drone's depth frame (every
    # lav_every steps, stride lav_stride, teammates within lav_mate_r m dropped). In SEARCH the cells along the
    # track (x = 1..lav_look m ahead) give the centre-line top(x); the flight-path slope is the one closest to
    # kp * (want - agl) / h inside [s_min, s_max], s_min = max (top + lav_lo - z) / x (and the max over the lateral
    # +-lav_half_w band + lav_side_lo), s_max = min (top + lav_hi - z) / x (s_min when they cross), clipped to
    # +-lav_smax, vz = slope * |v_xy| (a 5x5 m max over-reads the ground under the path by 2.4 m median on these
    # slopes; the grid cell under the drone by 0.44 m) (fewer than lav_min_cells
    # known cells: stock law). lav_avoid_r > 0: while the LAV drives a SEARCH drone, _avoid uses at most this
    # range (the depth-band climb stays as a close-range safety net).
    # Shipped arm (lav_r): lav_maps ("mountain",), lav_look 20, lav_lo_far 2, lav_kp 0.5, lav_avoid_r 8 (rest default).
    # Traces (8 random seeds): SEARCH h 2.524 -> 2.632 m/s (+4.3%), arc 1.143 -> 1.096, AGL median 7.5 -> 8.0,
    # act p50 +0.6 ms. Farm vs c1 (fast path, mountain): local 201 seeds +0.0190 +- 0.0083 (W/L 135/59, landings
    # +12, t2g -0.75 s; best of 5 variants on the first 100, the other 101 +0.0113 +- 0.0115); B list 100 seeds
    # (out of sample) +0.0111 +- 0.0089 (W/L 63/31, landings +4, goals detected 327 -> 341). SEARCH collisions 0;
    # extra OBSTACLE_COLLISIONs are DESCEND contacts on phantom claims. Other maps: bit-identical (map_kind gate).
    lav_maps: tuple = ()
    lav_look: float = 15.0
    lav_lo: float = 5.0
    lav_hi: float = 12.0
    lav_kp: float = 0.3
    lav_half_w: float = 2.0
    lav_cell: float = 1.0
    lav_stride: int = 4
    lav_every: int = 3
    lav_min_cells: int = 4
    lav_smax: float = 1.2
    lav_avoid_r: float = 0.0
    lav_mate_r: float = 1.5
    lav_side_lo: float = 2.5
    lav_lo_far: float = -1.0           # >= 0: the lo margin falls linearly to this at lav_look m ahead (later, steeper climbs)
    # The champion's controller-aware velocity slew (keeps the commanded thrust vector
    # positive: dv_z >= -0.25 m/s per step, tilt-bounded dv_xy) is only applied on
    # city/open/village (FIX_SLEW_KINDS). Before the map is classified ("other") a drone
    # that spawns above cruise altitude commands a fast descent while accelerating and the
    # DSL PID flips it (TILT at ~1.5 s). Extra map kinds to apply the guard on.
    slew_ctrl_aware_maps: tuple = ()
    slew_pre_cue: float = 0.0            # >0: pre-classification ctrl-aware slew only while the drone has seen no tall structure nearby (upper-half pixels closer than 15 m, running max fraction below this)
    slew_pre_clear: float = 0.0          # >0: the pre-classification ('other') ctrl-aware slew only for drones whose first depth frame has fewer than this fraction of pixels closer than 8 m

    # APXV-R approach check (default off; research/beat_f8kw/village_open + skeptic). Village
    # phantoms (mostly parked-car roofs) gain 0-3 hits in the 2 s before the drone is 6 m from
    # them; real pads gain >= 5. On apx_vfy_maps every claim, and on apx_vfy_late_maps only a claim
    # made after the map's commit_t with fewer than min_hits hits, is checked once the drone is
    # within apx_vfy_r of the pad: it passes when less than apx_vfy_tleft_min s is left, the claim
    # is younger than apx_vfy_min_age_s, the pad has apx_vfy_hits_ok hits, or it gained apx_vfy_gain
    # hits over the last apx_vfy_win_s s. Otherwise hold at that point facing the pad for
    # apx_vfy_hold_s, then fly the chord to a point rotated apx_vfy_orbit_deg about the pad (camera
    # on the chord heading, apx_vfy_orbit_up higher), turn back to the pad (hold timer starts within
    # apx_vfy_yaw_deg of the pad bearing, or after apx_vfy_yaw_wait_s) and hold again; apx_vfy_gain
    # new hits at any time pass, else the pad is hard-refuted. apx_vfy_skip_champ: no check while
    # the champion's descend-entry gate (dsc_min_hits) would refute the claim anyway. Hold and
    # orbit commands go through the avoidance, roof guard and separation like any approach command.
    apx_vfy_maps: tuple = ()
    apx_vfy_late_maps: tuple = ()
    apx_vfy_skip_champ: bool = False
    apx_vfy_r: float = 6.0
    apx_vfy_win_s: float = 2.0
    apx_vfy_gain: int = 5
    apx_vfy_hits_ok: int = 25
    apx_vfy_min_age_s: float = 1.0
    apx_vfy_hold_s: float = 1.5
    apx_vfy_orbit_deg: float = 60.0
    apx_vfy_tleft_min: float = 6.0
    apx_vfy_speed: float = 1.5
    apx_vfy_orbit_up: float = 0.5
    apx_vfy_yaw_deg: float = 20.0
    apx_vfy_yaw_wait_s: float = 2.0
    # SMASK + HOPG (default off). Every start platform is on the reject list with bad_pad_radius
    # (2.0 m), which hides goals 1.1-2.0 m from a start; start-platform detections fall <= 0.91 m
    # from it. start_mask_r > 0: on start_mask_maps the start entries (own and mate starts added
    # on the first step) use start_mask_r; refuted pads keep bad_pad_radius. tko_hop_guard_maps:
    # a drone that has not yet flown (agl never > 1.5 m) with a claim 0.45..tko_hop_r m away climbs
    # straight up first (a sideways hop at 0.5 m flipped the drone, TILT).
    start_mask_r: float = 0.0
    start_mask_maps: tuple = ()
    tko_hop_guard_maps: tuple = ()
    tko_hop_r: float = 3.0
    # WSTOP (default off): on wall_stop_maps, in CLIMB or in the first wall_stop_sec s of SEARCH,
    # a forward clearance under wall_stop_r (the floor is 0.5 m, so the avoidance brake never
    # stops the take-off creep) backs the drone away from the camera heading at wall_stop_back m/s,
    # climbing at least 1 m/s, with the camera held on the wall. Firings count in _wstop_n.
    wall_stop_maps: tuple = ()
    wall_stop_r: float = 1.0
    wall_stop_back: float = 0.5
    wall_stop_sec: float = 3.0
    # 2-opt claim swap (default off; research/beat_f8kw/lateness P1). assign_2opt_m > 0: each step
    # after the assignment, two APPROACH drones (not commit_sink) whose pads have >= assign_2opt_min_hits
    # hits, are not split and both lie >= assign_2opt_min_d away, swap claims when the summed
    # distance falls by more than assign_2opt_m metres after a turn charge of
    # assign_2opt_v * assign_2opt_turn_s * (1 - cos a) / 2 per drone; at most one swap per step and
    # a pair of pads never swaps twice. assign_2opt_maps: () = every map.
    assign_2opt_m: float = 0.0
    assign_2opt_min_hits: int = 8
    assign_2opt_min_d: float = 6.0
    assign_2opt_turn_s: float = 1.2
    assign_2opt_v: float = 2.7
    assign_2opt_maps: tuple = ()
    # ---- batch 2 (research/beat_f8kw city_coll, transfer + skeptics; every key default off) ----
    # CG city collision guard in _free_dir (not in CLIMB), three independent parts keyed on the
    # free-direction map kind ("forest" if is_forest else map_kind).
    # (1) CGE extrusion, free_extrude_maps: pooled depth points at WORLD height free_extrude_band,
    #     within free_extrude_range m horizontally and at most free_extrude_max_drop m below the eye
    #     (skeptic amendment A; <= 0: no limit), are copied as a vertical column (free_extrude_dz m
    #     above the eye, one per free_extrude_vox m cell, the free_extrude_cap nearest cells) into
    #     the tube test of free_direction and of the pass-through guard. City collision hulls are
    #     convex: building-n's 2.5 m awning is a slab 1.7-2.7 m in front of the drawn wall at 4.5 m.
    # (2) pass-through guard, free_pass_guard_maps: on the out-of-view branch (command more than
    #     50 deg off the camera, which flies the raw command at cruise), clamp the command azimuth
    #     to +-free_pass_guard_deg; if the tube along that edge direction is clear to
    #     free_want_clear keep the pass-through, else steer free_direction on the clamped command
    #     at free_slow. Beyond +-free_pass_guard_back_deg the side of the drone's previous guard
    #     step is kept (skeptic amendment B: backward commands do not chatter between sides).
    # (3) saturation escape, free_sat_maps: when the pick's raw tube clearance is <= free_sat_clr
    #     (every ray blocked: free_direction falls back to the ray nearest the command, flown at
    #     full speed when it agrees), fly -m * free_sat_push + (the command's horizontal part minus
    #     its component toward m, <= free_sat_slide m/s), m = 1/d-weighted horizontal direction to
    #     the pooled points within 0.12 + margin + free_sat_band m; vertical = pick z * free_slow.
    free_extrude_maps: tuple = ()
    free_extrude_band: tuple = (2.0, 3.2)
    free_extrude_range: float = 10.0
    free_extrude_dz: tuple = (0.0, 0.8, 1.6, 2.4)
    free_extrude_vox: float = 0.3
    free_extrude_cap: int = 200
    free_extrude_max_drop: float = 4.0
    free_pass_guard_maps: tuple = ()
    free_pass_guard_deg: float = 40.0
    free_pass_guard_back_deg: float = 150.0
    # review add-on (default 180 = spec behaviour): commands more than free_pass_guard_max_deg off
    # the camera pass through unguarded as before. Beyond ~90 deg the clamped +-40 deg edge
    # command points away from the command (local city n=8 seed: 461 of 489 guard steps at >= 75
    # deg, horizontal output opposite to the command in 99% of those at >= 90 deg).
    free_pass_guard_max_deg: float = 180.0
    free_sat_maps: tuple = ()
    free_sat_clr: float = 0.3
    free_sat_band: float = 0.25
    free_sat_push: float = 0.6
    free_sat_slide: float = 1.0
    # P2r forest climb-out (default off). Forest pads are floating 0.2 m slabs; a drone that slid off
    # the edge in DESCEND keeps sinking (sticky commit_sink) and lands on the ground under the slab.
    # dsc_below_maps: a drone dsc_below_z m below the pad top with agl > 0.25 (and, if
    # dsc_below_agl_gap > 0, agl - above > dsc_below_agl_gap: the pad stands that high over the
    # ground below the drone, so a too-high pad-z estimate cannot fire on a drone on the slab) is
    # "below": within 0.95 m of the estimate it steps out at 0.6 m/s holding height, else climbs at
    # 0.8 m/s, until it is 0.3 m above the pad top. dsc_hold_maps: then sink only within dsc_hold_r
    # of the estimate while below dsc_hold_above m over the pad. dsc_hold_after_below (default True,
    # skeptic): the hold only acts after a below event on this descent; False = always-on hold, which
    # delayed every normal forest landing by 0.4-0.7 s in local flights (do not ship).
    dsc_below_maps: tuple = ()
    dsc_below_z: float = 0.15
    dsc_below_agl_gap: float = 0.0
    dsc_hold_maps: tuple = ()
    dsc_hold_r: float = 0.40
    dsc_hold_above: float = 0.9
    dsc_hold_after_below: bool = True
    # P1b village phantom height veto (default off; add-on to APXV-R). Village pads sit at z
    # 0.33-0.43 (estimates up to ~0.57), phantoms (car roofs, house roofs) at 0.5-4 m. At the
    # APPROACH -> DESCEND switch on village, a claim whose estimate z lies strictly inside
    # village_veto_z = (lo, hi) with fewer than village_veto_hits hits and a hit rate since the claim
    # (hits - claim-time hits) / max(0.5 s, claim age) below village_veto_rate is hard-refuted and
    # the drone climbs at 0.5 m/s. Claim-time hits and step: APXV-R's vf_hits0 / vf_step0 (tracked
    # here when APXV-R is off). Events in _vveto_log.
    village_veto_z: tuple = ()
    village_veto_hits: int = 25
    village_veto_rate: float = 3.0
    # ---- batch 3 (research/bughunt/RANKED.md fix specs; every key default off) ----
    # S1 DESCEND gate exemption. The champion's 8-hit descend-entry gate (dsc_min_hits) exempts claims by
    # horizontal claim distance only, so a real pad claimed from steeply above (below the camera frame for
    # the whole approach, few hits) is hard-refuted. On APPROACH detector ticks a claim counts as in view
    # when the pad estimate lies within dsc_gate_inview_deg below the drone (atan2(h, dxy), yaw ignored) and
    # dxy <= 19.5 m. dsc_gate_inview_min > 0: the gate only fires once the claim had >= that many in-view
    # ticks. dsc_gate_steep_deg > 0 (fallback rule): the gate only fires for claims made less than that
    # many degrees below the drone. Counter and claim elevation are reset at every claim / hand-over.
    dsc_gate_inview_min: int = 0
    dsc_gate_inview_deg: float = 44.0
    dsc_gate_steep_deg: float = 0.0
    # S2 endgame hurry at the real sink rate: need = dist/1.5 + max(0, above - flare)/sink
    # + min(above, flare)/descend_speed + 0.5 + hurry_margin (forest flare / sink on forest) instead of
    # dist/1.5 + above/0.45 + 3.0 (the old budget fires ~8 s early and starts fast descents 2.5 m out).
    hurry_real: bool = False
    hurry_margin: float = 1.0
    # S3 open seeds: the FIX_FOV pass-through (fly an out-of-view command unchanged) also applies when
    # map_kind is on yaw_scan_maps (the +-50 deg yaw scan is keyed on map_kind "city", which ~93% of open
    # seeds carry) and the route kind is not city.
    free_fov_open_pass: bool = False
    # S5 separation push: sep_dz_max > 0 ignores teammates more than sep_dz_max m above/below while this
    # drone is in sep_dz_phases (a teammate passing 2 m above pushed a landing drone off its pad).
    sep_dz_max: float = 0.0
    sep_dz_phases: tuple = ("APPROACH", "DESCEND")
    # S6 world box: pad_world_box drops proposals with max(|x|, |y|) > box + pad_world_margin, box =
    # village 40, forest 42, mountain none, anything else 75 (kind = "forest" if is_forest else map_kind).
    pad_world_box: bool = False
    pad_world_margin: float = 2.0
    # S8 map kind from the clue height: mtn_clue_z > 0: after the village test, the seed is mountain iff
    # clue z > mtn_clue_z or (median start z > 10 and clue z > mtn_clue_zmin); otherwise the flat branch
    # (never mountain). Flat clue z <= 8.35 m by construction (mean pad z + U(-5, 5)).
    mtn_clue_z: float = 0.0
    mtn_clue_zmin: float = 7.0
    # S9 open band top: pad_z_hi_open > 0: while the z band is the flat band and the route kind is not
    # city, the band top is max(pad_z_hi, pad_z_hi_open) (open pads reach 1.29 m; detected z runs high on
    # high-FOV seeds).
    pad_z_hi_open: float = 0.0
    # S10 start mask with a height test: start_mask_dz > 0: a start entry only rejects a detection inside
    # its radius (start_mask_r on start_mask_maps, else bad_pad_radius) when |z_det - (start_z - 0.121)|
    # < start_mask_dz; refuted-pad entries stay horizontal-only (goals 4-8.5 m below a raised start).
    start_mask_dz: float = 0.0
    # S11 #16 dead-drone claim release: dead_zero_steps > 0: after it flew, a drone whose linear and
    # angular velocity (state[6:12]) read exactly zero with unchanged xy for that many consecutive steps
    # is frozen (crashed): it releases its claim (unless within landed_r of it) and goes DONE before the
    # assignment, instead of after dead_steps (60). DESCEND below 0.5 m AGL and landed drones are exempt.
    dead_zero_steps: int = 0
    # ---- batch 4 (research/oracle_decomp phantom + seen analysts, bughunt #13; every key default off) ----
    # NE in-view negative evidence. On ne_maps (matched on map_kind; never on forest), after the detector
    # merge of a detector tick, a pad entry with 1 <= hits < ne_max_hits that gained no hit on this tick
    # adds one to its run when some drone that ran the detector has the estimate inside its frustum (planar
    # depth 0.5..range, |a|,|b| <= ne_edge in tan units) and unoccluded (depth at its pixel >= planar depth -
    # ne_occl_tol); a hit resets the run. At ne_ticks: a claimed entry is refuted softly (_refute soft), an
    # unclaimed one drops to 0 hits (not claimable until detected again). Never for an entry whose claimant
    # is in DESCEND or that already passed the descend-entry gates once (ne_safe). range: ne_range_by_map
    # (map, metres, ...), else ne_range. Fires in _ne_log.
    ne_maps: tuple = ()
    ne_ticks: int = 5
    ne_range: float = 14.0
    ne_range_by_map: tuple = ()
    ne_max_hits: int = 30
    ne_occl_tol: float = 1.2
    ne_edge: float = 0.9
    # NE weighted counter (skeptic:phantom amendment; off while ne_range_near <= 0 or ne_far_w == 1): a no-hit
    # in-view tick whose nearest in-view planar depth (over the drones that see the entry) is beyond
    # ne_range_near adds ne_far_w to the run instead of 1.
    ne_range_near: float = 0.0
    ne_far_w: float = 1.0
    # NE straight-line distance band (skeptic:phantom P1 / PLAN.md rank 1; () = off, arm (16.0, 18.0, 0.5)):
    # dist = zg * sqrt(1 + a^2 + b^2) from the camera to the estimate. A view counts only when dist <= band[1] on
    # city / open, or dist <= the map's NE range elsewhere; the tick weight is 1 when the nearest counted dist is
    # <= band[0], else band[2]. When set it replaces ne_range_near / ne_far_w.
    ne_dist_band: tuple = ()
    # ---- audit_cov (DEV; every key default off = c6 behaviour) ----
    # Audit of the worst c6-vs-238 seeds (city/open/village, lists B+C): NE refutes REAL pads. B 148 (city): the
    # approaching drone's real pad (21 hits) got its third NE fire at 11.0 s (fires at 6, 4, 21 hits while in view
    # 11-12 m out) -> soft_abort_max escalated it to a hard refute: pad done + blacklisted, the drone timed out, the
    # goal was never landed. C 228 (village, NE on village is new in c6): a claimed real pad with 23 hits was zeroed
    # at 6.3 s and never re-detected (goal lost; base landed it at 13.1 s). Phantom NE fires in the traces carry
    # 1-5 hits. ne_diag: every NE fire goes to src_counts["ne_ev"] (t, entry, hits, claimant, aborts, x, y, z,
    # nearest counted dist) for offline real/phantom matching. ne_no_hard: an NE refute of a claimed entry never
    # escalates to the hard refute (done + blacklist); it zeroes the hits and releases the claim as the soft path does.
    # ne_max_hits_by_map (map, hits, ...): per-map override of ne_max_hits (entries with >= that many hits are
    # never NE-tested).
    ne_diag: bool = False
    ne_no_hard: bool = False
    ne_max_hits_by_map: tuple = ()
    # ---- regress (DEV, round 9; every key default off = c7 behaviour) ----
    # rg_diag: diagnostics only (no behaviour change): every change of a drone's claim is logged at the end of
    # act() into src_counts["rg_ev"] (t, drone, old entry, new entry, hits, xyz, phase, claim_dist, aborts) and
    # every _refute call into src_counts["rg_ref"] (t, drone, entry, caller, soft, never_hard, hits, aborts).
    rg_diag: bool = False
    # CNE (commit NE shadow). Pool: c7 claims +37 +- 12 more entries at the city late-commit step (t = 25.0 s, gate
    # 2 -> 1) than 238 on the same 1005 seeds (5 lists), and +26 +- 7 more of those drones fail (SEARCH timeouts
    # 217 vs 195). c7's NE distance band (ne_dist_band 16/18/0.5: straight-line cap, half weight at 16-18 m) lets
    # more 1-hit entries survive to the commit than 238's NE (planar 18 m, weight 1). On rg_cne_maps (map_kind) a
    # shadow of 238's NE counter runs on the unclaimed low-hit entries (same frustum / occlusion test, planar range,
    # weight 1, ne_ticks); an entry the shadow would have zeroed (no hit since) is not handed out below the normal
    # claim gate after the commit. Before the commit nothing changes (c7's protection of real pads is kept).
    # rg_cab_max > 0: after the commit, an entry with aborts >= rg_cab_max is not handed out below the normal gate
    # either (ne_no_hard never escalates NE refutes to the hard refute, so a re-detected phantom can be claimed again).
    rg_cne_maps: tuple = ()
    rg_cab_max: int = 0
    # ESK take-off gate. B 725 d6 (city): the drone starts 2.4 m from its goal, claims it at 0.12 s while climbing,
    # enters DESCEND at d 0.71 m / 1.3 m above with 1.2-1.5 m/s lateral; the ESK DESCEND lateral law (gain 2.5, clip
    # 0.9) then the switch to the base law at esk_dsc_above 0.8 m brakes during the 2.8 m/s drop: roll -22 -> -28 deg
    # at contact, rests at 15.4 deg, never latches, the attitude creeps to 60 deg (TILT); the base law landed it level
    # (contact -6.5 deg). esk_dsc_min_claim_d > 0: the ESK DESCEND law only for claims made at least this far away.
    esk_dsc_min_claim_d: float = 0.0
    # Tilt hold. C 938 d1 (city): the drone orbits the pad at ESK height (2.7 m above, roll -32 -> +31 deg, 1.7 m/s
    # tangential), EDR / base entry fires at d 1.04 m, the 2.8 m/s drop starts mid-swing: contact at -11 deg with
    # 0.8 m/s lateral, creeps to 15.2 deg, TILT. dsc_tilt_hold_deg > 0: on ESK maps in DESCEND, while higher than
    # dsc_tilt_hold_above above the pad and tilted more than dsc_tilt_hold_deg (max |roll|, |pitch|), the sink command
    # is held at >= 0 (no sink) and the lateral law is the base law, until level.
    dsc_tilt_hold_deg: float = 0.0
    dsc_tilt_hold_above: float = 0.8
    # CLOSE descend-entry gate for close steep claims (the champion 8-hit gate exempts claims made closer
    # than dsc_gate_min_dist). On close_gate_maps ("forest" if is_forest else map_kind), at DESCEND entry,
    # a claim made closer than close_gate_dist (horizontal) and at least close_gate_elev degrees below the
    # drone, with fewer than close_gate_min_hits hits before dsc_gate_until_sec, is refuted softly and the
    # drone climbs at 0.5 m/s. Events in _close_log.
    close_gate_maps: tuple = ()
    close_gate_min_hits: int = 4
    close_gate_dist: float = 8.0
    close_gate_elev: float = 20.0
    # GLA glimpse look-at. On gla_maps ("forest" if is_forest else map_kind), when a detector proposal of an
    # unclaimed drone in gla_phases leaves a pad entry with 1 <= hits < the claim gate (and hits <= gla_max_hits
    # when > 0) within gla_range m horizontally, that drone re-aims only its camera at the entry for up to
    # gla_s seconds (lowest-hit entry; no re-trigger while one is active): off = wrap(yaw - bearing),
    # keep = acos(min(1, gla_keep / r)), tgt = clip(|off|, keep, gla_maxoff_deg), yaw = bearing + sign(off) * tgt
    # (skipped on a step when r cos(tgt) > 19.75, and on city/forest/open free-dir flight when the yaw is more
    # than gla_travel_deg off the travel heading). Ends at gla_s, when the entry reaches the gate, is claimed,
    # done or zeroed, or the drone claims / leaves gla_phases. Replaces the yaw scan (SEARCH) and the climb
    # yaw (CLIMB); later yaw guards (yaw_track, early_align, align gate) still win. Events in _gla_log.
    gla_maps: tuple = ()
    gla_s: float = 0.5
    gla_max_hits: int = 0
    gla_range: float = 25.8
    gla_phases: tuple = ("SEARCH", "CLIMB")
    gla_keep: float = 18.5
    gla_maxoff_deg: float = 40.0
    gla_travel_deg: float = 60.0
    # GLA on these kinds triggers only before the map's commit time (commit_t_by_map, else commit_t; set even
    # when the map is in commit_t_skip) - skeptic:seen "before the 25 s commit on city and open only".
    gla_precommit_maps: tuple = ("city", "open")
    # CLIMB-claim envelope (bughunt #13, S11). On climb_claim_env_maps ("forest" if is_forest else map_kind),
    # a drone that got its claim while in CLIMB keeps the take-off envelope in APPROACH until its height above
    # ground reaches CLIMB's exit height (forest_alt / cruise_alt - 0.6): vertical command = the CLIMB climb
    # rate, horizontal speed <= cruise_speed * climb_creep and zero while tilt >= climb_creep_tilt, plus the
    # CLIMB-phase guards climb_free_guard, wall_stop (WSTOP) and early_align on their own maps. Events in
    # _cenv_log.
    climb_claim_env_maps: tuple = ()
    # ---- batch 5 (research/regress analyse:regress + skeptic:regress fix specs; every key default off) ----
    # F1 village start box. village_start_box > 0: after the start-height village rules, a seed with any own
    # start beyond village_start_box m (max(|x|, |y|)) is not village (village worlds keep every start within
    # 40.0 m; the city seeds that pass the median-start-z village test start at 42-74 m, and S6's 40 m village
    # box then drops their real pads). Events in _b5_log ("F1").
    village_start_box: float = 0.0
    # F2 (skeptic:regress MK amendment) forest vote needs XGB support. forest_vote_fv_min > 0: when the forest vote
    # locks while the base's XGB ForestVoter has >= 2 samples and its running forest probability is below
    # forest_vote_fv_min, neither forest rule may set forest (not the clear fraction < forest_clear_frac rule, not
    # the tall override, tall fraction >= FOREST_TALL_OVERRIDE); the seed goes to city / open by city_tall_frac.
    # Real forest seeds are forced to forest by the voter at 0.1-0.2 s (p 0.67-1.0, 33 of 33), so the vote never
    # runs there. Events in _b5_log ("F2clear" / "F2tall").
    forest_vote_fv_min: float = 0.0
    # F2b (skeptic:regress MK) fv_force_spread: the XGB ForestVoter may not force forest when _forest_possible() is
    # False (an own start beyond the forest world, FOREST_VETO_SPREAD_M 45 m, the base's vote veto test; or the
    # router's geometric city fix). Implemented in uid167_base._force_forest. Events in _b5_log ("F2b").
    fv_force_spread: bool = False
    # F3 P2r loop breaker. dsc_below_retarget = k > 0 (needs dsc_below_maps): from the k-th P2r below event of a
    # drone on the same claimed entry on, the drone stops sinking on that estimate. Proposals merged into the entry
    # within 3 m of it (close range < REFINE_R preferred, last 64) are split into 2 clusters (2-means on xy); centres
    # >= dsc_below_retarget_sep m apart (each with >= 3 points): the entry moves to the cluster nearest the drone and
    # a split entry is added at the other (as pad_split does). Otherwise the entry moves dsc_below_retarget_off m from
    # its pre-retarget estimate, one bearing per below event, cycling through four bearings starting opposite the
    # slide-off direction (along the cluster axis first when the clusters are >= 0.3 m apart). The moved entry is
    # frozen against detection drift (as the ray probe fixes it); the drone climbs to dsc_hold_above over the pad top
    # before it moves sideways. Events in _b5_log ("F3").
    dsc_below_retarget: int = 0
    dsc_below_retarget_off: float = 1.1
    dsc_below_retarget_sep: float = 1.2
    # F4 S1 exemption needs hits. dsc_gate_inview_hits_min > 0: _s1_gate_ok returns True (the champion's 8-hit
    # descend-entry gate may fire) for a claimed entry with fewer than that many hits, whatever its in-view count.
    dsc_gate_inview_hits_min: int = 0
    # ---- detv2 WCONF (research/detv2/pipeline; default off): weak detector proposals (PadProposal.weak, emitted only
    # when the detector's weak_thr > 0) confirm an EXISTING entry: on wconf_maps ("forest" if is_forest else
    # map_kind), a weak proposal that passes _merge's z band / world box / bad-list tests and lies within wconf_r m (xy)
    # of an unclaimed, not-done entry with 1 <= hits < the claim gate adds one hit, at most wconf_max per entry and never
    # past the claim gate (the descend-entry gates, APXV-R, P1b and landing still need detector hits). It never creates,
    # moves or splits an entry, adds no view and does not reset NE runs. Elsewhere weak proposals are dropped.
    wconf_maps: tuple = ()
    wconf_r: float = 1.5
    wconf_max: int = 3
    # T0 build: wconf_min_hits > 1 - weak hits only go to an entry that already has this many DETECTOR hits (hits minus
    # whits), so a single accepted detection plus weak ones cannot reach the claim gate. 1 = the analyst's rule above.
    wconf_min_hits: int = 1
    # detv2 stage 2 (research/detv2/stage2/TUNE.md; default off): per-map override of wconf_min_hits as a flat
    # (map, n, map, n, ...) tuple on the same "forest" if is_forest else map_kind key; maps not listed use wconf_min_hits.
    wconf_min_hits_by_map: tuple = ()
    # ---- detv2 T0-a close-pair claim fix (research/detv2/PLAN.md rank 1; default off). _assign masks every free
    # entry within same_pad (pad_split_r for a split entry) of a claimed or landed entry, so the second pad of a close
    # pair is never handed out. On t0a_maps ("forest" if is_forest else map_kind) such an entry is handed out anyway
    # when it has >= t0a_min_hits detector hits (weak WCONF hits excluded), has gained hits as its own entry on
    # >= t0a_ticks detector ticks (_Pad.t0a_ticks), and sits >= t0a_sep m (xy) from the anchor of every masking partner
    # (the partner's current estimate while it has no anchor yet). On t0a_maps _merge also feeds a detection to the
    # NEAREST entry within same_pad (as it already does while a split is active), so the second pad's detections stop
    # landing in whichever close entry was created first. t0a_split_maps: maps (_mass_route_kind) where the pad split
    # (pad_split_r rules (a) and (b), split claim spacing and pad_split_min_hits) is allowed in addition to
    # pad_split_maps - on mountain the second pad's detections otherwise merge into the partner's entry.
    # Counters in _t0_n ("a_*"), events in _t0_log (FLY_B5_LOG "t0").
    t0a_maps: tuple = ()
    t0a_min_hits: int = 20
    t0a_sep: float = 1.8
    t0a_ticks: int = 10
    t0a_split_maps: tuple = ()
    # (F5 "NE only before the commit" was rejected by skeptic:regress - the H5 phantom-only excess is a relabel - and
    # is not implemented.)
    # ---- mtn_claims (DEV; every key default off = UID 238 behaviour) ----
    # Pool (b238_base fast path, 201 local mountain seeds, 979 drones): drones that claim a pad and do not land:
    # APPROACH timeout on a real claim 23 / phantom 11, DESCEND timeout real 12 / phantom 6, DESCEND collision on a
    # phantom 16 / real 3 (~0.01 total stake). Keys below are matched on map_kind "mountain" (S8 classification: 0 of
    # the 1000 local seeds misclassified).
    # MZF slab veto. The mountain ground is a noise height field plus hill meshes over a VISUAL-ONLY ground box (top at
    # z 0.05, no collision shape). Where the field dips below 0.05 the depth camera sees a perfectly flat plane at
    # z 0.05 that the detector reads as a pad. 42 of 356 mountain phantom entries (b238_base) sit at z < 3 m, every one
    # where the noise field is below 0.05 (checked with the generator's get_terrain_z); 15 were held by drones that
    # failed: seeds 955 (4 of 6 drones), 821 (4 of 5), 317 (2 of 5), 940 / 886 / 708 / 658 / 145 (1 each) - claims
    # from 139-187 m that time out, descents that end in terrain contact. Real mountain goals: 3 of 4070 (854 distinct
    # seeds in all local rows) below z 2.0 (min 0.2), 8 below 2.5. mtn_zfloor > 0: on mountain a detector proposal
    # with z below it is dropped before the merge (never becomes or feeds an entry).
    mtn_zfloor: float = 0.0
    # MSM start mask without the S10 height test. S10 (start_mask_dz 0.6) lets a detection inside a start's disc
    # through when |z - (start z - 0.121)| >= 0.6. Mountain start platforms are detected 0.76-0.82 m below the drone's
    # start z (|dz + 0.121| = 0.64-0.70): 747 d4 / 347 d5 / 451 d0 claimed a start platform 0.1-0.4 m from its centre
    # and tipped over on it (TILT). No real mountain entry lies within 3 m of a start (0 of 697). On
    # start_mask_nodz_maps the start entries mask horizontally (bad_pad_radius), as without S10.
    start_mask_nodz_maps: tuple = ()
    # FAR long-trip evidence gate. Mountain claims fire at 2 hits and the greedy _assign hands a free entry to the
    # nearest free drone however far it is. Per-step traces of 43 pool seeds (221 claim episodes): claims made from
    # >= 60 m: 29 episodes, 1 landing (a hand-off onto a 75-hit entry), 12 real-pad episodes that timed out or were
    # handed on, 16 phantoms holding 496 drone-seconds (955: three drones fly 30-50 s toward 2-hit slab phantoms
    # 139-157 m away); claims from 40-60 m: 3 of 7 landed (2-hit claims that grew). far_claim_d > 0 (mountain):
    # _assign pairs a drone with an entry more than far_claim_d away (xy) only if the entry has >= far_claim_hits.
    far_claim_d: float = 0.0
    far_claim_hits: int = 8
    # REACH (mountain): mtn_reach_v > 0: _assign skips a pair when dist / mtn_reach_v + mtn_reach_land_s exceeds the
    # time left. Use it as a hard impossibility bound: every one of 279 traced mountain landings took >= d / 3.0 + 2.0 s
    # from the claim (the mean fit 0.334 s/m + 3.4 s is NOT a bound: at v 2.9 / 3.5 s it blocked 135 d1 / 999 d1, two
    # late hurried landings, in the first smoke). 238's global assign_reach (2.4 m/s, also releases claims) is off.
    # 451 d3 / 866 d0 took 128-129 m trips at 48-54 s.
    mtn_reach_v: float = 0.0
    mtn_reach_land_s: float = 2.0
    # STEEP 3-D approach line. APPROACH commands 3 m/s toward the pad and a vertical clip(want_z - z, +-approach_vz 2);
    # the 3 m/s norm then gives ~2.5 m/s horizontal and 1.66 m/s vertical, so a claim made from far above (or below)
    # the pad arrives over it off height. Traced landings (85 seeds, 272 real landings): claims 15-40 m above the pad
    # (14%) reach DESCEND 6.5 m above it (4.0 otherwise) and take 10.75 s claim-to-landing vs 9.6 s; claims below the
    # pad (22%) 10.8 s; the late DESCEND timeouts 48 d1 / 589 d1 / 742 d2 / 485 d4 were claims made 15-20 m above the
    # pad at 49-51 s (48 d1: over the pad at 57.0 s still 7 m up, 0.2 m above it at 59.5 s). steep_maps (map_kind):
    # while |z - aim_z| > steep_ratio * dist and dist > steep_min_d (aim_z = pad z + steep_above), fly the straight line
    # to (pad xy, aim_z) at the cruise speed (vertical part capped at steep_vz); the endgame hurry keeps this direction.
    # v1 (steep_above 4 = approach_alt, climbs too): random 40 +0.011 +- 0.011 (14 W / 7 L), but some take-off and
    # moderate claims landed 0.4-0.5 s LATER: the base reaches the pad up to apx_max_above (7 m) high and then sinks in
    # DESCEND at 2.8 m/s pure vertical, faster than any diagonal. v2: steep_above 6.5 (just under apx_max_above),
    # steep_ratio 0.67 (the base's own slope: 1.66 / 2.5 after the 3 m/s norm), steep_climb False (descents only).
    steep_maps: tuple = ()
    steep_ratio: float = 0.6
    steep_min_d: float = 2.5
    steep_vz: float = 2.5
    steep_above: float = 4.0
    steep_climb: bool = True
    # ---- time_term (DEV; every key default off = UID 238 behaviour) ----
    # ESK early sink. The time term is the largest aggregate loss (0.05-0.08 per landed drone on every map; 60-72%
    # of city/open/village landings are beyond par, by 5.5-7.1 s on average). Per-step traces of the final approach
    # (238 dev bundle, fast path, city/open/village): the whole approach is flown at approach_alt (4 m above the pad)
    # at 2.92-2.95 m/s, the brake starts at d 2.5 m (v = 1.2 d), DESCEND starts at d 1.0 m with ~1.3 m/s of lateral
    # speed and only then does the drone sink: 1.75-1.8 s from DESCEND entry to touchdown (a 0.6 s sink ramp under
    # the ctrl-aware slew, dvz >= -0.25 per step, then 2.77 m/s), then the 0.5 s landing latch (LANDING_STABLE_SEC).
    # The lateral settle after entry takes ~0.6-0.8 s, so ~1 s of that sink is vertical-only time that the last
    # cruise metres could hide. esk_maps ("forest" if is_forest else map_kind, or the route kind): inside esk_r m of
    # the claimed pad the APPROACH height target drops from pad_z + approach_alt to pad_z + esk_above (never raised)
    # with the vertical cap raised to esk_vz; the sink overlaps the cruise (the 3 m/s norm trades a little
    # horizontal speed), DESCEND starts lower and already sinking. The failed glide (apx_glide, -0.044 on 69: TILT
    # in DESCEND, 70 city / 91 open drones - it reached DESCEND 1.2 m above the pad with lateral speed and pitch)
    # is avoided by keeping >= esk_above of pure sink after DESCEND entry; esk_above >= 2.4 also stays above
    # roof_agl 2.5 (city / village roof guard, outside roof_pad_r 2.5) over flat ground. esk_min_hits: only on
    # entries with at least this many hits (0 = any).
    esk_maps: tuple = ()
    esk_r: float = 8.0
    esk_above: float = 2.6
    esk_vz: float = 1.2
    esk_min_hits: int = 0
    esk_kp: float = 1.0                 # P gain of the vertical command toward pad_z + esk_above (1.0 = the base law)
    # ESK v2, from the first smoke (30 seeds city/open/village, fast path): esk_above 2.6 saved 0.40-0.42 s per landing
    # (median, every map) but landing precision fell (distance to the goal centre p50 0.14-0.17 -> 0.22-0.25 m; 2.4 m
    # with a faster sink: 0.27-0.28) and the lost landings were all off-centre touchdowns: TILT / TIMEOUT in DESCEND
    # (the drone rests tilted > 15 deg on the pad edge and never latches, 0.30-0.41 m off) and a village drone that sank
    # 0.43 m off-centre onto the house wall beside the pad (it reached DESCEND 0.73 m above that roof). The lateral
    # settle after DESCEND entry (d 1.0 m at ~1.3 m/s, clip 0.55 m/s, gain 1.2: ~1.6 s to 0.1 m) was hidden under the
    # 1.78 s sink. esk_dsc_gain > 0: DESCEND lateral law clip(d * esk_dsc_gain, +-esk_dsc_lat) on esk maps.
    # esk_agl_tol >= 0: ESK only while the downward ray reads at least (height above the pad - esk_agl_tol), i.e. the
    # drone is over ground at about pad level. esk_min_claim_d: skip claims made closer than this (take-off claims).
    esk_dsc_gain: float = 0.0
    esk_dsc_lat: float = 0.55
    esk_agl_tol: float = -1.0
    esk_min_claim_d: float = 0.0
    # Smoke 2 (esk3 = esk_dsc_gain 2.5 / lat 0.9 at every height): precision p50 0.10-0.11 / p90 0.14-0.16 m (better than
    # base) and village clean, but 3 city/open drones tipped over ON the pad (0.08-0.12 m off): a parked drone whose
    # downward ray reads >= 0.30 m (raised pad, land_settle's xy zeroing needs agl < 0.30) keeps getting the lateral
    # command, 2.5 x the estimate offset (0.1-0.15 m/s instead of 0.05-0.07), the PID winds up against the pad and the
    # pitch creeps 12 -> 60 deg in ~1.5 s (TILT). esk_dsc_above: the boosted law only while higher than this above the
    # pad estimate; below it the base law (dsc_gain, dsc_lateral).
    esk_dsc_above: float = 0.0
    esk_dsc_fade: float = 0.0
    # ---- land_seq (DEV; every key default off = time_term / UID 238 behaviour) ----
    # EDR early drop. Landing-latch traces (land_trace.py, 15 seeds city/open/village n >= 4, 91 landings base /
    # 89 esk5, fast path): the latch starts at first contact (lag 0.00 s: the pad contact kills the 2-2.5 m/s sink
    # in one step, no bounce, no stable-time reset on any landing), the claim -> APPROACH latency is 0 and the
    # approach path is straight (path / D 0.95-0.97). What is left is the order of the last metres: with ESK5 the
    # drone brakes from d 2.5 m on the exponential law v = 1.2 d (0.80 s from d 3 m to DESCEND entry at d 1.0 m with
    # 1.3 m/s left) and only then starts the vertical drop (2.7 m above the pad, 1.30 s to contact: 0.55 s ramp under
    # the -0.25 m/s-per-step slew, then 2.77 m/s). Median seconds above the straight 3-D bound + latch: base 1.82,
    # esk5 1.39-1.51 per landing. EDR starts the drop while the drone is still braking: inside edr_r of a claimed pad
    # (>= edr_min_hits hits, ground under the drone at about pad level) the lateral law becomes the braking profile
    # v = min(edr_vmax, sqrt(2 edr_a d), edr_k d) and a forward simulation of that profile and of the vertical drop
    # (ramp edr_av, sink edr_vs, lateral priority under the 3 m/s norm) switches to DESCEND as soon as contact would
    # come no earlier than edr_margin s after the drone is within edr_dok of the pad. DESCEND keeps the profile law
    # while higher than edr_low above the pad, then the base / ESK law (parked drones never get the stronger law).
    # edr_maps key: "forest" if is_forest else the route kind (like esk_maps).
    # Measured (15-seed traces, on top of ESK5): v1 (a 3, k 2.5, margin 0.25, planner brake 1.5 a) crossed the pad at
    # 1.2 m/s pitched 33 deg and lost 3 of 4 drones of city 617 to TILT; a 2 / k 2 / margin 0.45 / edr_ab 1.0 saves
    # 0.34 s per landing vs ESK5 (0.78 s vs base) but alone loses village drones to tilted contacts (14-18 deg): use it
    # only with CST (below): contact tilt median 1.9, max 4.9 deg, no landing lost to it. edr_k 1.6 / edr_low 0.25 worse.
    edr_maps: tuple = ()
    edr_r: float = 4.0
    edr_a: float = 3.0
    edr_ab: float = 1.0                 # planner brake capacity = edr_ab * edr_a (the tracking lag makes > 1 optimistic)
    edr_k: float = 2.5
    edr_vmax: float = 3.0
    edr_dok: float = 0.12
    edr_margin: float = 0.25
    edr_av: float = 4.5
    edr_vs: float = 2.75
    edr_h0: float = 0.05
    edr_min_hits: int = 20
    edr_agl_tol: float = 0.6
    edr_max_above: float = 5.0
    edr_low: float = 0.8
    edr_prio: bool = True
    # PARK. The downward ray reads 0.5-1.0 m on a drone parked on a raised pad (it misses the thin pad top), so
    # land_settle (agl < 0.30) never zeroes the lateral command: the DESCEND law keeps pushing a parked drone toward
    # the pad estimate (0.05-0.2 m/s). Harmless when the drone touches down level (base contact tilt ~3-4 deg, latched
    # 0.5 s later), fatal after a tilted contact: the push holds the tilt and the attitude integral winds up against
    # the pad (ESK5 traces: city 389 d1 contact tilt 11 deg -> TILT, city 92 d4 17 deg -> TIMEOUT; EDR village 422 /
    # 551 14-18 deg -> TILT). park_maps (key like esk_maps): within park_r of the pad estimate, between -0.1 and
    # park_above above its height and |vz| < park_vz (resting on it), the lateral command is zero. Measured: rescues city
    # 92 d4 (rested at 17.6 deg, latched once the push stopped); the other tilted contacts need CST (a drone resting on
    # the pad keeps its contact tilt whatever the command: tiltdrop.py).
    park_maps: tuple = ()
    park_above: float = 0.2
    park_vz: float = 0.25
    park_r: float = 0.45
    # LVL. Tilted-contact experiment (tiltdrop.py, real env, open 16 pad 0): a drone that touches the pad tilted stays
    # at that tilt (contact friction; the attitude loop cannot level it on the pad): 12 deg at release -> 6 deg on the pad,
    # latched; 15 deg with a 60 deg/s swing -> 14.7 on the pad, latched 0.7 s late; 15 deg / 120 deg/s -> 22 deg on the pad,
    # never latched (TIMEOUT); 20 deg -> TILT. Traces: base passes 1.0 m above the goal at 2.3 deg median tilt (contact
    # max 8.6), ESK5 at 12.1 deg (its failures: contact 11-17.6 deg), EDR at 4.5 but 8-10 deg at 0.6 m. lvl_maps (key
    # like esk_maps): between 0.12 m and lvl_above above the pad estimate with tilt > lvl_deg, the lateral command holds
    # the current velocity (no lateral acceleration) and the sink slows toward lvl_vz by at most lvl_dvz per step (a
    # harder brake saturates the motors and removes attitude authority). Measured on top of ESK5: 0.1-0.2 s slower per
    # landing and still 2 city TILTs (the tilt came in the last 0.3 m, from the land_settle brake): superseded by CST.
    lvl_maps: tuple = ()
    lvl_above: float = 1.2
    lvl_deg: float = 7.0
    lvl_vz: float = 0.3
    lvl_dvz: float = 0.35
    # CST coast touchdown. The contact freezes the attitude (tiltdrop.py: lateral speed at impact is harmless, 0.5 m/s
    # -> 0.2 deg; a lateral brake in the last 0.1 s is not: 0.5 m/s braked to 0 -> 8 deg on the pad). land_settle zeroes
    # the lateral command at agl < 0.30 (~0.1 s before contact at 2.8 m/s): base reaches that point at 0.1-0.17 m/s
    # (contact tilt median 3.8, max 8.6 deg), ESK5 / EDR at 0.3-0.4 m/s (EDR trace open 16 d2: level at 0.3 m, the
    # land_settle brake 0.37 -> 0 m/s, contact at -14.5 deg, TILT). cst_maps (key like esk_maps): within cst_r of the pad
    # estimate, below cst_above above it and sinking faster than cst_vz, the lateral command holds the current velocity.
    # 15-seed traces (city/open/village): ESK5 + PARK + CST contact tilt median 2.6 / max 6.3 deg (ESK5 3.7 / 9.8 plus
    # failures at 11-17.6, base 3.8 / 8.6); every ESK5 DESCEND loss (city 92 d2/d4, 389 d1, random 518 d1) lands; no time
    # cost (dt vs ESK5 0.00). PARK + CST without ESK: identical to base within 0.0001 on 36 random seeds.
    cst_maps: tuple = ()
    cst_above: float = 0.5
    cst_vz: float = 1.0
    cst_r: float = 0.35
    # ---- land_ext (DEV; every key default off = land_seq behaviour) ----
    # FLD forest landing. Traces (land_trace.py, 12 random forest seeds, c1 config, 52 landings): DESCEND starts at
    # d 1.48 m with 2.35 m/s lateral, 1.6 m above the pad; 2.09 s (median) to contact, contact 0.29 m off the goal
    # centre at -0.48 m/s. The DESCEND lateral law clip(1.2 d, 0.55) needs ~1.6 s to centre from 1.5 m, so the forest
    # flare (0.8 m, then a 0.48 m/s creep) waits for it; a lower flare alone lands off-centre and hops (land_seq
    # forest_flare_alt 0.4: -0.024, 0.8 s slower). Many forest pads are floating slabs (the ray under a parked drone
    # reads 2.2 m): an off-centre fast touchdown slides under the slab. FLD centres first and then drops fast: on
    # fld_maps ("forest" if is_forest else the route kind, like esk_maps; needs the same key in edr_maps) the EDR
    # braking profile / forward simulation runs with fld_a / fld_k / fld_margin (> 0 / >= 0 override edr_*), inside
    # the EDR envelope only the EDR planner switches to DESCEND (fld_hold: the forest enter_r 1.5 m would switch
    # first and drop the profile), and in DESCEND the sink above the flare is fld_sink (an EDR drop, or within fld_r
    # of the pad) and the flare is fld_flare while within fld_r of the pad (else the forest flare / creep / abort
    # climb as before). Use with CST + PARK on the same key.
    fld_maps: tuple = ()
    fld_flare: float = 0.15
    fld_sink: float = 2.75
    fld_r: float = 0.3
    fld_a: float = 0.0
    fld_k: float = 0.0
    fld_margin: float = -1.0
    fld_hold: bool = True
    # fld_ca: the champion's controller-aware slew (dv_z >= -0.25 m/s per step) for DESCEND on fld_maps. Forest is not
    # in FIX_SLEW_KINDS / slew_ctrl_aware_maps: its slew lets the command lead the measured sink by ~0.6 m/s, the DSL
    # PID then drives the motors to the minimum and loses attitude authority. With the forest sink 2.0 from 1.6 m this
    # is harmless (pre-contact tilt median 7 deg); an FLD drop at 2.75 m/s without it pitched and rolled to 25 deg
    # (12 dev seeds: pre-contact tilt median 24 deg, 224 d3 contact at 1.8 m/s lateral -> TILT).
    fld_ca: bool = False
    # fld_min_hits: the FLD sink / low flare only on entries with at least this many hits (a 3-hit phantom descent at
    # 2.75 m/s hit the ground: dev 4 d2). fld_vdrop > 0: the EDR drop on fld_maps also needs the horizontal speed at
    # or below fld_vdrop (brake at altitude first). Braking during the drop is weak and tilted: the sink ramp runs the
    # thrust at ~0.5 g, so the EDR drop at d 0.7-1.0 m with 2 m/s left pitched to 24-26 deg, crossed the pad at
    # 1 m/s and came back (dev 808 d4: contact at -17 deg, rested at -23, TILT).
    fld_min_hits: int = 20
    # park_nosep (needs park_maps / cst_maps): no teammate separation push on a step where PARK or CST fired. The S5
    # separation (sep_radius 3.2, gain 1.6) acts in DESCEND: a teammate parked on a pad 2.84 m away pushed a drone
    # resting on its own pad at 0.18-0.22 m/s; a drone resting on the pad cannot level while pushed (land_seq
    # tiltdrop.py), so a 12 deg contact grew to 17 deg and never latched (dev forest 808 d4, TILT in every FLD arm).
    park_nosep: bool = False
    park_nosep_maps: tuple = ()          # () = wherever PARK / CST fire; else only these keys (like esk_maps)
    fld_vdrop: float = 0.0
    # fld_above > 0: inside edr_r of a claimed pad (>= fld_min_hits, ground under the drone at about pad level) the
    # APPROACH height target drops from pad_z + forest_approach_alt (1.8) to pad_z + fld_above, vertical cap fld_avz,
    # so the final drop is shorter (the sink ramp from 1.6 m takes ~0.6 of the ~1.0 s drop).
    # fld_hold_s > 0: the hold ends after this many seconds inside the base entry radius (forest 1.5 m); then the base
    # DESCEND entry rules apply. Random forest screen (flf, 100 seeds): 2 real-pad drones never left APPROACH
    # (162 d0 58 s, 147 d2): the forest avoidance deflected them around an obstacle at the pad, they orbited at
    # 1-2 m/s at 0.4-5 m and the EDR drop (speed gate) never fired; the base enters DESCEND inside 1.5 m.
    fld_hold_s: float = 0.0
    # fld_edr_only: the fast sink / low flare only on EDR drops (False: also on any descent within fld_r of the pad).
    # With fld_hold_s the timed-out hold enters DESCEND by the base rule at 1.5 m / ~2 m/s; crossing fld_r at speed
    # it got the 2.75 m/s sink and 0.15 m flare, touched down tilted and never latched (screen F1 forest 754 d1).
    fld_edr_only: bool = False
    # fld_hold_back > 0: the hold is released for the claim once the drone, having been inside the base entry radius,
    # is fld_hold_back m farther from the pad than its closest approach (the orbit of 162 / 147 / 724 reached 0.3-1.0 m,
    # then 2-5 m). A converging take-off claim (754 d1: climbing from pad level, 1.9 -> 0.3 m in 1.2 s) keeps the hold;
    # the time cap (fld_hold_s, then 3 s) only catches a drone parked at the pad below the drop height.
    fld_hold_back: float = 0.0
    fld_above: float = 0.0
    fld_avz: float = 1.0
    # land_safe TDO touchdown offset (default off). The 0.10 safety term is set at touchdown: on every landed
    # open / mountain / village drone the episode minimum clearance comes in the last 0.6 s. Mountain pads float
    # 0.2 m above the highest terrain within 0.69 m of their centre (platform_placement), so the closest body of a
    # drone resting on the pad is the slope beyond the uphill rim. Probe (tools/geo_probe.py, real envs, 977 mountain
    # goals of c5_all): median terrain slope 0.50 (88% >= 0.25); on slope >= 0.25 the clearance of a drone resting at
    # centre + r * downhill is 0.40 / 0.46 / 0.48 / 0.50 / 0.53 / 0.57 for r 0 / 0.15 / 0.2 / 0.25 / 0.3 / 0.4
    # (safety 0.25 -> 0.35 at r 0.2, 0.41 at r 0.3); the best direction lies 12 deg (median) from the plane's downhill.
    # The pad ESTIMATE is biased downhill of the true centre (landed drones: e_u mean +0.05, 6-14% > 0.2 m on slopes
    # >= 0.25, max 0.45; across sd 0.09), so a fixed offset from the estimate leaves the 0.6 m platform: smoke r 0.3
    # on mountain 232 put 3 of 5 targets 0.51-0.63 m from the centre and one drone sank past the rim (collision).
    # Hence the rim reference: on tdo_maps ("forest" if is_forest else the route kind, like esk_maps) the approaching
    # drone keeps the depth points that fall tdo_ring_lo..tdo_ring_hi from the claimed pad (tdo_samp_lo < d <
    # tdo_samp_hi, every detector tick, <= tdo_frame_pts per frame, last tdo_frames frames). Inside tdo_enter_r (the
    # EDR envelope) it fits a trimmed plane; with >= tdo_min_pts points in >= tdo_min_sect of 8 sectors, >= tdo_min_hits
    # pad hits, slope >= tdo_min_slope and the older / newer halves of the frames agreeing within tdo_max_ang deg the
    # downhill unit u is fixed (frozen at DESCEND entry). tdo_rim: every step within tdo_samp_r of the estimate and
    # > 0.3 m above it the downward ray gives the exact surface height under the drone. The pad top is the HIGHEST
    # window (2 * tdo_ztol) of locally flat samples (neighbour slope <= 0.02) within tdo_zband of the estimate's z and
    # 0.75 m of its xy with >= tdo_plat_n samples spanning >= tdo_plat_ext m and a rim step (a neighbour within 0.12 m
    # >= tdo_rim_step lower). Flat samples at that height are on the platform, samples > tdo_zoff from it are off it
    # (the terrain just outside the downhill rim reads 0.84 m below the top, median; > 0.35 m on 99.8% of goals,
    # tools/rim_probe.py). Prior centres around the estimate (tdo_pr_up uphill .. tdo_pr_dn downhill, +-tdo_pr_lat
    # across; weights: tdo_pr_core N(tdo_pr_mu, tdo_pr_sd) + uniform along u, tdo_pr_pcore N(0, tdo_pr_psd) + uniform
    # across, fitted to the 608 landed estimates) consistent with every on / off sample (0.6 m radius +-0.02) form the
    # posterior; the offset is lam * u with the largest lam <= tdo_r (steps of 0.05) whose target lies beyond 0.6 -
    # tdo_margin of the centre with posterior probability <= tdo_eps (no on-sample: lam 0, the base landing). lam may
    # grow only while > tdo_lock_ab above the pad and is frozen below tdo_frz_ab; the target moves outward at <=
    # tdo_rate m/s and only while > tdo_move_ab above the pad (a late lateral correction in the 2.8 m/s EDR drop
    # freezes a 10-15 deg tilt into the contact), decreases apply at once. tdo_rim False: fixed tdo_r (unsafe).
    # The EDR braking profile, the DESCEND entry and the whole DESCEND (lateral law, EDR, commit, land_settle, PARK,
    # CST, on-pad / claim-distance tests) aim at pad + offset; the offset moves with the pad estimate. Screens and
    # verdict: .work/mech/land_safe/NOTES.md.
    tdo_maps: tuple = ()
    tdo_r: float = 0.25
    tdo_min_slope: float = 0.25
    tdo_min_hits: int = 20
    tdo_min_pts: int = 60
    tdo_min_sect: int = 3
    tdo_max_ang: float = 30.0
    tdo_ring_lo: float = 0.8
    tdo_ring_hi: float = 2.5
    tdo_samp_lo: float = 3.0
    tdo_samp_hi: float = 16.0
    tdo_frame_pts: int = 200
    tdo_frames: int = 40
    tdo_enter_r: float = 4.0
    tdo_rim: bool = True
    tdo_margin: float = 0.15
    tdo_samp_r: float = 1.6
    tdo_zband: float = 0.4
    tdo_rim_step: float = 0.15
    tdo_ztol: float = 0.015
    tdo_zoff: float = 0.06
    tdo_plat_n: int = 5
    tdo_plat_ext: float = 0.1
    tdo_pr_up: float = 0.55
    tdo_pr_dn: float = 0.15
    tdo_pr_lat: float = 0.4
    tdo_pr_core: float = 0.76          # prior of the estimate error along u: core N(mu, sd) + uniform tail
    tdo_pr_mu: float = 0.02
    tdo_pr_sd: float = 0.045
    tdo_pr_pcore: float = 0.8          # across u: core N(0, psd) + uniform tail
    tdo_pr_psd: float = 0.06
    tdo_eps: float = 0.01              # accepted posterior probability of a target beyond 0.6 - tdo_margin
    tdo_lock_ab: float = 1.5
    tdo_rate: float = 0.5
    tdo_move_ab: float = 1.0
    tdo_frz_ab: float = 0.25
    # ---- round-5 MTD: mountain touchdown precision (research/round5/MTD.md; every key default off = candHt) ----
    # Port of the cf_autopilot c29k2 touchdown stack onto TDO (tdo_maps must hold the kind too).
    # mtd_maps ("forest" if is_forest else the route kind, like tdo_maps): every pad entry also keeps an
    # inverse-variance estimate of its centre from the detector fixes merged into it (c29k2 outer.py 3177-3188):
    # fix sigma = mtd_sig0 + mtd_sig_k * range (3-D range from the observing drone; mtd_rng_none without one), weight
    # w = 1 / sigma^2, alpha = w / (info + w) clipped to [mtd_alpha_min, mtd_alpha_max] (the floor only for fixes
    # within mtd_floor_r m: far fixes of other drones never get more than their own weight), info capped at
    # mtd_info_cap; a fix within 1.5 m range more than 0.5 m off an entry with >= 30 hits gets alpha 0.01. An entry
    # without the state (split / injected entries) starts from its current estimate with info = hits / sigma(
    # mtd_init_rng)^2. The last mtd_rec_n fixes (step, xyz, range) are kept for the gate. Bookkeeping only: nothing
    # flies differently unless one of the keys below is on.
    # mtd_est: a CLAIMED entry's estimate is the inverse-variance one (instead of the running mean of all its hits);
    #   the claim_drift anchor cap and REFINE are unchanged.
    # mtd_hm: TDO's touchdown bearing is the height-map pick (c29k2 _s1m_cells / _s1m_quad / _s1m_hm_pick
    #   1403-1461) over TDO's ring points instead of the plane's downhill unit: 0.1 m cells (median height) in the
    #   tdo_ring_lo..tdo_ring_hi annulus, a robust quadratic surface, then for 16 bearings at radius mtd_r_hi (and the
    #   centre) the minimum distance from the resting drone body (r 0.06 m, 0.025 m tall, 0.0125 m over the pad top)
    #   to the surface on rings 0.62-1.61 m; the best bearing wins. No offset when the fitted slope > mtd_hm_smax or the
    #   best bearing's clearance is not at least mtd_hm_gain above the centre's (flat ground: the rim is the risk there).
    #   The plane's slope / stability tests are replaced by that comparison (hits / points / sectors tests stay).
    # mtd_gate: the offset radius comes from the estimate's precision instead of TDO's 1 % posterior (whose 24 %
    #   uniform tail keeps lam at 0 until the down ray has crossed the platform, too late to move the target):
    #   the gate passes when the claimed entry has >= mtd_close_n fixes within mtd_close_r m range in the last
    #   mtd_close_s s, their 80th-percentile distance from their median is <= mtd_agree and the estimate lies within
    #   mtd_agree of that median. Passed: lam is the largest value <= mtd_r_hi (steps of 0.05) whose target stays
    #   within 0.6 - mtd_margin of the centre with probability >= 1 - mtd_eps under a Gaussian prior (sd mtd_pr_sd
    #   along and across) restricted by TDO's on / off-platform ray samples (no sample needed); failed: TDO's own lam.
    #   The target may jump to the new offset while the drone is more than mtd_jump_r m (horizontal) from it and
    #   more than tdo_lock_ab above the pad (early in the drop a new aim point is harmless); later only TDO's
    #   rate-limited moves; decreases apply at once. The decision freezes at DESCEND entry (no fixes after it).
    # mtd_rim: down-ray rim guard in DESCEND (c29k2 _me_rim 3739+, knobs 147-163) while an offset is in use: within
    #   mtd_rg_xy of the target, the surface under the drone (z - ray) lies more than mtd_rg_thr_cal below the
    #   ray-calibrated pad top (TDO rim samples; mtd_rg_thr below the estimate's z without one), the drone sinks, is
    #   not closing on the estimate's centre (< mtd_rg_vin m/s), is still more than 0.08 m above the top and the ray
    #   read > 0.12 m for its last 10 steps, on 2 steps in a row -> the sink stops, the offset steps mtd_rg_step
    #   inward of the drone (at most half the offset; the centre on the 2nd fire), the drone climbs straight up
    #   first while within mtd_rg_margin of the top, then holds height moving to the new target until the ray steps
    #   up onto the top, the target is reached or mtd_rg_hold_s passes.
    # Diag: src_counts mtd_gate_pass / mtd_gate_fail / mtd_hm / mtd_hm_flat / mtd_hm_steep / mtd_jump / mtd_rg_fire /
    #   mtd_rg_step / mtd_rg_arrive / mtd_rg_timeout / mtd_rg_climb; mtd_log [t, drone, claim, gate, lam, hm info].
    mtd_maps: tuple = ()
    mtd_sig0: float = 0.05
    mtd_sig_k: float = 0.012
    mtd_rng_none: float = 16.0
    mtd_init_rng: float = 14.0
    mtd_alpha_min: float = 0.04
    mtd_alpha_max: float = 0.7
    mtd_floor_r: float = 8.0
    mtd_info_cap: float = 4000.0
    mtd_rec_n: int = 32
    mtd_est: bool = False
    mtd_hm: bool = False
    mtd_hm_smax: float = 1.5
    mtd_hm_gain: float = 0.03
    mtd_gate: bool = False
    mtd_r_hi: float = 0.45
    mtd_close_r: float = 7.0
    mtd_close_n: int = 8
    mtd_close_s: float = 3.0
    mtd_agree: float = 0.06
    mtd_margin: float = 0.05
    mtd_pr_sd: float = 0.03
    mtd_eps: float = 0.01
    mtd_jump_r: float = 1.2
    mtd_rim: bool = False
    mtd_rg_xy: float = 0.25
    mtd_rg_thr: float = 0.2
    mtd_rg_thr_cal: float = 0.12
    mtd_rg_vin: float = 0.15
    mtd_rg_margin: float = 0.25
    mtd_rg_step: float = 0.15
    mtd_rg_hold_s: float = 2.0
    # ---- forest_phantom FPN (default off; .work/mech/forest_phantom/NOTES.md) ----
    # In-view no-hit refute of a CLAIMED entry during APPROACH. Forest phantom claims (detections on tree bodies at
    # pad height: 3 hits at the claim, 7-17 at DESCEND entry, the descent ends in the tree's collision hull) gain
    # ~0 hits while the claimant looks straight at them. Traced c6 phantom-collision seeds (A 8 + B 3): the
    # claimant's longest run of in-view detector ticks without a hit within 12 m of the estimate was 45-99 on 12
    # of 13 phantom claims (the 13th was claimed from 2 m and never in view) and <= 4 on 59 real claims (<= 5
    # within 15 m, up to 20 within 22 m). On fpn_maps ("forest" if is_forest else map_kind) an entry with
    # 1 <= hits < fpn_max_hits claimed by a drone in APPROACH adds one to its run on every detector tick on which
    # the claimant (fpn_any_view: any drone that ran the detector) has it inside the camera wedge (|a|,|b| <=
    # fpn_edge in tan units), unoccluded (depth at its pixel >= planar depth - fpn_occl_tol) and within fpn_range
    # (straight line, camera to estimate) while it gained no hit; a hit or a new claimant restarts the run. At
    # fpn_ticks the claim is refuted (_refute, soft unless fpn_hard). fpn_unclaimed: an unclaimed entry seen the
    # same way by any drone drops to 0 hits at fpn_ticks. Diag: src_counts fp_fire / fp_fire_u / fp_log.
    fpn_maps: tuple = ()
    fpn_range: float = 12.0
    fpn_ticks: int = 6
    fpn_edge: float = 0.9
    fpn_occl_tol: float = 1.2
    fpn_max_hits: int = 30
    fpn_hard: bool = False
    fpn_any_view: bool = False
    fpn_unclaimed: bool = False
    # ---- audit_mf (DEV; every key default off = c6 behaviour) ----
    # TRR tilted-rest relevel. c6 fast rows, mountain lists A/B/C: the EDR drop enters DESCEND 3.2-3.5 m out at
    # 2.7-2.9 m/s, brakes during the drop (pitch -0.55 rad) and touches down pitched 0.3-0.4 rad; the contact freezes the
    # attitude (land_seq tiltdrop.py), so the drone rests on the pad beyond LANDING_MAX_TILT_RAD 0.26 and never latches
    # (C232 d4: rests at pitch -0.30 from 11.9 s to the 60 s TIMEOUT, 0.07 m from the goal) or the tilt creeps until
    # TILT (C58 d2: 0.23 -> 0.5 rad in 3 s). On trr_maps ("forest" if is_forest else the route kind, like esk_maps):
    # in DESCEND within trr_r of the pad (or the TDO point), less than trr_above above the pad estimate, nearly still
    # (|vz| < trr_v, |vxy| < trr_v) and tilted > trr_tilt rad for trr_steps consecutive steps, the drone hops: no lateral
    # command and vz +trr_vz for trr_len steps (lift off, the attitude loop levels it in the air), then the normal
    # DESCEND law lands it again from ~0.3 m. At most trr_max hops per claim. It fires only on a drone that cannot latch.
    trr_maps: tuple = ()
    trr_r: float = 0.6
    trr_above: float = 0.6
    trr_v: float = 0.2
    trr_tilt: float = 0.27
    trr_steps: int = 15
    trr_vz: float = 1.0
    trr_len: int = 20
    trr_max: int = 2
    # TWIN claim gate. c6 sets same_pad_by_map forest 1.5 (merge / claim spacing; the 238 default is 3.0) and
    # spoken_skip_done 1, so a forest detection 1.5-3.0 m off a real pad (tree occlusion / partial view; 3-15 hits)
    # becomes its own claimable entry while the real pad is claimed or already landed on. Traces (fast path): C561 d1
    # claimed a 10-hit twin 2.5 m from the pad d3 was landing on -> OBSTACLE_COLLISION; C655 d6 a twin 2.9 m from a
    # claimed pad (14.5 s wasted, refuted at the 8-hit gate), C655 d2 a twin 2.8 m from a landed pad (10 s wasted).
    # On twin_maps ("forest" if is_forest else map_kind): an entry within twin_r of an entry that is claimed by another
    # drone or done is claimable only with >= twin_hits hits (the second pad of a real close pair collects ~100 hits
    # while its neighbour is approached / landed on; twins stay at 3-15).
    twin_maps: tuple = ()
    twin_r: float = 3.0
    twin_hits: int = 20
    # fov_clue FCL - MEASUREMENT ONLY, never in a shipped or composition cfg without the user's explicit decision.
    # The env draws the camera FOV (moving_drone.py:272, RandomState(seed) 2nd draw) and the clue's y noise
    # (swarm_autopilot.py spawn_task_world, a fresh RandomState(seed), 2nd draw) from the same stream:
    # clue_y - centroid_y = R * s, s = (FOV - 90) / 2 exactly. FOVE (bughunt: reproject frame A's depth into
    # frame B over candidate FOVs) estimates the FOV at step fcl_at from frames at fcl_rec; accepted when >=
    # fcl_min_pairs pair estimates agree within fcl_mad_max deg (MAD). Then pad_posterior.CLUE_S /
    # geo_posterior.CLUE_S get s, sd = max(fcl_sd_min, fcl_sd_k * MAD / 2) and w = fcl_w (the FOV likelihood's
    # weight in a mixture with the champion's box), on route kinds in fcl_maps; the router re-plans, the hunt
    # posterior is invalidated. fcl_cem defers the mountain CEM start until the estimate. fcl_scramble (robustness
    # arm): s is replaced by an independent U(-1, 1) draw (what a patched env with independent draws looks like).
    # Diag: src_counts fcl_fov / fcl_mad / fcl_pairs / fcl_ms / fcl_step / fcl_s.
    fcl_maps: tuple = ()
    fcl_w: float = 0.9
    fcl_sd_min: float = 0.025
    fcl_sd_k: float = 1.0
    fcl_mad_max: float = 0.25
    fcl_min_pairs: int = 4
    fcl_rec: tuple = (10, 20, 30, 40)       # bughunt's (1, 11, 21, 31): the step-1 frame has few valid depth pixels
    fcl_at: int = 40
    fcl_bias: float = 0.0                   # deg added to the FOVE median (estimator calibration)
    fcl_max_drones: int = 4
    fcl_cem: bool = False
    fcl_scramble: bool = False
    fcl_geo_floor: float = 0.5
    # ---- phantom_land PVR (default off; .work/mech/phantom_land/NOTES.md) ----
    # Ray platform proof before touchdown. c7 rows (5 lists): 28 mountain drones end on a phantom (2-22 hits: hill /
    # peak tops the detector reads as pads, mostly claimed close or never in view, so the 8-hit descend gate and NE
    # do not fire); the descent ends on the terrain (OBSTACLE_COLLISION). A real mountain pad is a raised platform
    # (top 0.2 m above the highest terrain within 0.69 m), so the exact downward ray crossing its rim shows a flat
    # disc at the estimate's height with a >= 0.15 m step (the TDO rim rule); a hill top is flat but has no step.
    # Traced c7 episodes (mtn_small + phantom_land traces): 0 of 18 phantom descents show the proof, 127 of 132 real
    # landings do (from 0.4-5.4 m above the estimate; the 5 without have >= 75 hits). On pvr_maps ("forest" if
    # is_forest else the route kind): while a drone holds a claim with < pvr_max_hits hits, every step within
    # pvr_samp_r of the estimate and > 0.3 m above it stores (x, y, z - agl); in DESCEND without a proof the sink
    # is limited to sqrt(2 * pvr_acc * (above - pvr_hold_ab)) and held at pvr_hold_ab above the estimate; after
    # pvr_hold_s seconds held there without a proof the claim is refuted (hard unless pvr_soft) and the drone climbs.
    pvr_maps: tuple = ()
    pvr_max_hits: int = 30
    pvr_samp_r: float = 1.6
    pvr_hold_ab: float = 1.2
    pvr_acc: float = 3.0
    pvr_hold_s: float = 1.0
    pvr_soft: bool = False
    pvr_zband: float = 0.8
    pvr_ztol: float = 0.015
    pvr_flat_tol: float = 0.0005
    pvr_plat_n: int = 5
    pvr_plat_ext: float = 0.1
    pvr_rim_step: float = 0.15
    pvr_rad: float = 1.1               # plateau / rim samples within this xy distance of the estimate (pad radius 0.6 +
                                       # estimate error up to ~0.5; 0.75 missed the rim of A648 d4, estimate 0.26 m off)
    # spc (composition onto c14): PVR was measured on c7, which has no EHM (r9_prior). EHM latches endgame claims at
    # 8 hits and enters DESCEND 0.5 m above the pad; PVR would push such a drone back to pvr_hold_ab and refute
    # without a proof, and a refute that late cannot buy another pad. pvr_ehm_skip: no PVR hold / refute on a claim
    # latched by EHM (dd.ehm_key == claim), so each mechanism keeps its own measured domain.
    pvr_ehm_skip: bool = False
    # ---- phantom_land LSD (default off) ----
    # Land-settle deadlock. Forest (dsc_below_maps) traces D34 d2 / B591 d0 / D379 d2 (c7, TIMEOUT with the pad
    # marked done) and D34 d4 (landed 6.9 s after its first rest): the drone rests level on the real pad at agl
    # 0.03 while the estimate sits 0.14-0.23 m above its resting height. When the altitude ray (origin ~0.1 m under
    # the body) reads through the 1 mm pad disc (agl ~2.5-3.0 for a few steps) P2r takes it for a slide off the
    # slab (above < -dsc_below_z) and keeps dd.below while above < 0.3: vz 0 and an outward push, and land_settle
    # zeroes the lateral command at agl < 0.3 -> v_des = 0, the drone floats on the pad with ~0 normal force
    # (contact force <= 0.01 N resets the env latch) until the ground-stall refute hops it off (then F3 retargets
    # the entry around the pad). On lsd_maps ("forest" if is_forest else map_kind): in DESCEND with agl < 0.30,
    # within land_settle_r of the target and less than lsd_ab below the estimate, dd.below is cleared and the drone
    # is pressed down (no lateral command, vz <= -lsd_vz) so the latch can complete. v2: "resting" = within lsd_r of
    # the target, -lsd_ab < above < 0.3 and (agl < 0.30 or |vz| and |vxy| < lsd_v): the ray reads THROUGH the pad the
    # step after contact (D34 d2: agl 0.04 -> 2.99 at 8.08 s), so agl alone misses it; while resting P2r is skipped
    # (no below entry, no F3 retarget of the entry).
    lsd_maps: tuple = ()
    lsd_ab: float = 0.4
    lsd_vz: float = 0.3
    lsd_r: float = 0.45
    lsd_v: float = 0.15
    lsd_mem: int = 100                 # v3: a still drone counts as resting only within lsd_mem steps of an agl < 0.3
                                       # reading at the pad top on this claim (A388 d5 hovered BESIDE the slab at
                                       # above -0.19, still, agl 2.66: v2 pressed it down and lost the landing)
    # ---- phantom_land CPS (default off) ----
    # City close-pair split. City merges detections within same_pad 3.0 m into one entry. c7 rows (5 lists): all 6 city
    # seeds with two goals 2.39-3.00 m apart hold ONE entry for both pads (214-485 hits, 2-4x a single pad), the partner
    # drone times out in SEARCH (the second pad is never an entry) and in 4 of 6 the claimant crashes on the ground
    # between the pads (A691 d0, B493 d1, D112 d3, D424 d5): the refine median of close-range detections of BOTH pads
    # drags the claimed entry toward the other pad and claim_drift (1.5 m) stops it half way (D112: 27.10 -> 25.67 at
    # 10.4 s, goals at 27.15 / 24.21). On cps_maps (route kind): a detection observed from < cps_rng m whose nearest
    # entry lies >= cps_r m away (and that entry has >= cps_min_hits hits) starts its own entry instead of merging;
    # once such an entry exists detections merge into the NEAREST entry (not the first within same_pad), and the
    # refine window of an entry only takes detections within cps_r of it.
    cps_maps: tuple = ()
    cps_r: float = 2.0
    cps_rng: float = 8.0
    cps_min_hits: int = 20
    # ---- r9_prior (round 9; every key default off) ----
    # BNX (P3): n-conditioned distance-band prior. The swarm slot template ties the fleet size to the distance band
    # (n = 2 + slot % 7; for 4 of 7 n per map one sub-band never occurs). On bnx_maps (route kinds with a band
    # mixture: "mountain", "open") geo_posterior.band_mix returns the template's band counts for this n
    # (geo_posterior.BAND_MIX_N_COUNTS) instead of the marginal mixture; every consumer (router pad field, GeoPosterior
    # hunt, CEM search planner) reads band_mix. Set at the first act from the fleet size.
    bnx_maps: tuple = ()
    # EHM (P5): endgame hail-mary landing. A drone with a real claim that cannot finish the careful landing before
    # 60 s scores 0.01 whatever it does, so the careful path's caution buys nothing there. Pool (c7, 5 lists): 147
    # mountain real-claim APPROACH/DESCEND timeouts, ~101 within 1.5 s of landing over all maps; traces show the
    # loss on mountain is the approach geometry (P law to pad + 4 at the 2 m/s vz clip / STEEP to pad + 6.5 = a
    # 0.67 slope: a pad further below arrives 6-7 m under the drone, then a 2.3 s drop).
    # On ehm_maps ("forest" if is_forest else the route kind), once a claim with >= ehm_min_hits hits is in APPROACH
    # and t_left = 60 - t < T_c + ehm_slack (T_c = remaining-time fit of the careful path on landed c7 drones,
    # EHM_TC per map: c0 + c1 d + c2 ab+ + c3 ab- + c4 max(ab - 4 - 0.667 max(d - 2, 0), 0)), the claim latches EHM:
    # straight 3-D line to (pad xy, pad z + ehm_h): lateral speed on the EDR braking profile, vertical sized to
    # reach ehm_h at the lateral arrival (climb at <= 2 m/s when below), scaled to the 3 m/s norm; no descent below
    # AGL ehm_agl while d > 3 m; DESCEND only at d < ehm_enter_r (base hurry/EDR entries are overridden).
    # ehm_dsc_vs > 0: DESCEND sink rate while EHM is latched and the drone is > 1 m above and < 0.8 m from the pad.
    ehm_maps: tuple = ()
    ehm_slack: float = 0.0
    ehm_min_hits: int = 8
    ehm_h: float = 1.0
    ehm_enter_r: float = 0.5
    ehm_agl: float = 2.0
    ehm_dsc_vs: float = 0.0
    ehm_max_d: float = 30.0            # latch only within this horizontal distance (no long low-level lines)
    ehm_enter_v: float = 0.7           # DESCEND entry also needs |v_xy| below this (a braking contact tips)
    # ---- blind_guard (round 8; every key default off) ----
    # OM obstacle memory: on om_maps ("forest" if is_forest else map_kind) every om_every steps each live drone
    # (CLIMB/SEARCH/APPROACH) stores its pooled depth cloud (24 x 24 aligned rays, the free_direction cloud) in WORLD
    # frame, voxel-deduplicated (om_vox) and cut to om_keep_r m horizontally, for om_sec s. Traced c7 city collisions:
    # the wall a drone slides into has usually left the camera (camera 90-130 deg off the motion, or the wall beside
    # the drone after a corner) while it was seen 0.3-5 s earlier within 0.1-0.5 m of the contact point.
    # Users of the memory (only points OUTSIDE the current frustum are used; the frame covers the rest):
    # (1) om_fd: memory points join the free_direction / pass-guard tube tests as extra points (body frame);
    # (2) om_blind (BG): on the out-of-view branch of _free_dir (command > 50 deg off the camera, c7 flies it blind),
    #     the tube along the ACTUAL command is tested against the memory; blocked within om_blind_clear m -> the
    #     horizontal command loses its component toward the remembered points (slide) and is capped to
    #     om_blind_speed x clear / om_blind_clear (>= om_blind_vmin);
    # (3) om_bump_r > 0 (BMP): last stage before the slew, om_bump_phases: remembered (+ with om_frame the current
    #     frame's) points in a slab
    #     around the drone (z -om_bump_zlo..+om_bump_zhi) within om_bump_r m horizontally bound the command's
    #     horizontal component toward each of them by the stopping-distance speed (om_bump_a, om_bump_lat,
    #     om_bump_margin; inside the margin: a push away <= om_bump_push). Tangential motion is untouched.
    # (4) om_hull (HM): memory / frame points of low structures (world z in free_extrude_band, a podium roof / awning)
    #     are copied to the eye height in the BG / BMP tests, in BMP with om_hull_margin m extra margin: city
    #     collision hulls are convex; traced c7 contacts on building-n's hull stand 0.3-0.9 m beside the drawn
    #     podium (x -5.8..-2.55, top 2.5 m) and 0.4-0.9 m off the tower face above the 2.75 m awning (y 4.55).
    om_maps: tuple = ()
    om_sec: float = 3.0
    om_every: int = 2
    om_vox: float = 0.25
    om_keep_r: float = 8.0
    om_fd: bool = False
    om_blind: bool = False
    om_blind_clear: float = 2.5
    om_blind_r: float = 0.6
    om_blind_speed: float = 1.0
    om_blind_vmin: float = 0.3
    om_bump_r: float = 0.0
    om_bump_zlo: float = 0.4
    om_bump_zhi: float = 0.6
    om_bump_push: float = 0.5
    om_bump_a: float = 2.5
    om_bump_lat: float = 0.2
    om_bump_margin: float = 0.4
    om_bump_phases: tuple = ("SEARCH", "APPROACH")
    om_hull: bool = False
    om_hull_margin: float = 0.7
    om_hull_tall: float = 0.0          # > 0: HM2, lift only low points near a surface this far above the band top
    om_hull_tall_r: float = 3.5
    om_hull_push: float = 0.5          # push inside the lifted points' margin (0 = only stop approaching)
    om_hull_stall: float = 0.0         # > 0: HM3 stall escape (s without 1 m of progress while lifted points bind)
    om_hull_esc: float = 4.0
    om_frame: bool = False
    # ---- fsb (round 10, FSB forest / city safety bumper; every key default off = c14) ----
    # Traced c14 forest (24 F seeds): 83 % of the landed-drone safety loss is set by a canopy point 90-135 deg off
    # the camera and ~90 deg off the velocity (the closest point of a straight pass), i.e. a point the frame no
    # longer holds; free_dir's tube is a collision margin (0.47 m) while forest safety saturates at 0.6 m surface
    # clearance. FSB runs BMP (memory points outside the frustum) at the SAFETY scale:
    # om_bump_margin_by_map / om_bump_push_by_map: (map, value, ...) per-map BMP margin / push (key "forest" if
    #   is_forest else map_kind); with fsb_abeam_deg > 0 the per-map margin applies only to points more than
    #   fsb_abeam_deg off the velocity (vel, or the command below fsb_vref_min m/s), the others keep om_bump_margin;
    # om_bump_phases_by_map: (map, "CLIMB,SEARCH,APPROACH", ...) per-map BMP phases;
    # fsb_pad_r > 0: in APPROACH, points within fsb_pad_r m (horizontal) of the claimed pad get no safety margin;
    # fsb_zmin: points below this world z get no safety margin (forest / city ground plane z = 0);
    # fsb_stall > 0: stall guard: < fsb_stall_d m of progress over fsb_stall s while the safety margin binds ->
    #   every point back to om_bump_margin for fsb_esc s.
    om_bump_margin_by_map: tuple = ()
    om_bump_push_by_map: tuple = ()
    om_bump_phases_by_map: tuple = ()
    fsb_abeam_deg: float = 0.0
    fsb_vref_min: float = 0.3
    fsb_pad_r: float = 0.0
    fsb_zmin: float = -99.0
    fsb_stall: float = 0.0
    fsb_stall_d: float = 1.0
    fsb_esc: float = 3.0
    fsb_lat: bool = False              # safety-margin excess removed across the velocity only (no braking)
    fsb_zlo: float = -1.0              # >= 0: slab of the SAFETY points on FSB maps (-fsb_zlo..+fsb_zhi; side
                                       # contacts); om_bump_margin points keep -om_bump_zlo..+om_bump_zhi
    fsb_zhi: float = -1.0
    fsb_only: bool = False             # FSB maps: only safety-margin points bump (no om_bump_margin points; the
                                       # stall escape then switches the bumper off, i.e. back to c14 forest)
    fsb_only_maps: tuple = ()          # per-map fsb_only / fsb_lat (for composing forest and city settings)
    fsb_lat_maps: tuple = ()
    fsb_dz_by_map: tuple = ()          # (map, dz, ...): per-map symmetric slab (overrides fsb_zlo / fsb_zhi)
    # FSB-M (min-aware margin): the safety term is set by the episode MINIMUM clearance, so once a drone has had a
    # close pass, keeping later passes wider than it costs time for nothing. fsb_min_slack >= 0: the drone's own
    # estimate of its running minimum (3-D distance to the nearest memory / frame point, counted after the scoring's
    # take-off grace and committed fsb_min_lag s late so a pass never lowers its own margin) caps the safety margin:
    # margin = min(per-map margin, max(fsb_min_floor, est + fsb_min_slack)).
    # FSB-P (fsb_cpa_maps): predictive pass steering instead of the stopping-distance limit for safety points: in
    # the velocity frame a point x m ahead (0 < x < fsb_cpa_x) at cross offset |y| < margin needs a lateral
    # velocity away of (margin - |y|) * s / x (reach the margin at the closest point of approach, s = commanded
    # along-track speed); points abeam / behind inside the margin need the push; left / right needs bound the
    # command's lateral component (conflict -> the middle), capped at fsb_cpa_vmax; the along-track part is kept.
    fsb_cpa_maps: tuple = ()
    fsb_cpa_x: float = 2.0
    fsb_cpa_vmax: float = 1.0
    fsb_dz3d: bool = False             # safety margin of a point at vertical offset dz = sqrt(margin^2 - dz^2)
                                       # (3-D clearance; with a wider fsb_dz_by_map slab for climbing passes)
    fsb_min_slack: float = -1.0
    fsb_min_floor: float = 0.3
    fsb_min_lag: float = 2.0
    # ---- forest_nav (round 8, default off = c7) ----
    # FSP (fsp_maps, key "forest" if is_forest else map_kind): free_direction's pick is flown at full speed when it
    # deviates <= fsp_deg (3-D) from the command and its own tube is clear for >= fsp_clear m (<= 0: free_want_clear),
    # instead of free_slow (0.75) beyond 11.5 deg. Forest traces (c7): the wide-pass choice (free_lat) turns picks
    # 12-25 deg off a command whose own tube is clear, and the binary agree > 0.98 rule then flies them at 2.1 m/s
    # (class 'dev': ~0.5 s per landing, 1.2 s of SEARCH per drone). Boxed picks (no 6 m ray) keep free_slow.
    # fsp_h: the deviation is measured horizontally (azimuth of pick vs command), see the comment at the rule.
    fsp_maps: tuple = ()
    fsp_deg: float = 30.0
    fsp_clear: float = -1.0
    fsp_h: bool = False
    # CLS (cls_maps): clear-lane speed. When the pick's tube is clear to >= cls_clear m (the lookahead: nothing in
    # the tube), it is flown at full speed (straight, or by FSP) and the command cruises (horizontal >= the map
    # cruise - 0.05), the horizontal speed is raised from forest_speed (2.8) to cls_speed (the env clips the 3-D norm
    # at 3.0). Speed from measured clearance instead of the fixed forest cap.
    cls_maps: tuple = ()
    cls_speed: float = 3.0
    cls_clear: float = 9.9
    # spd_min_hits > 0: FSP / CLS stand down in APPROACH while the claimed entry has fewer hits than this. The DESCEND
    # hit gate (dsc_min_hits 8, forest claims from >= 3 m) assumes the base approach time: an SPD drone reached a real
    # pad claimed 3.8-4 m out with 3-7 hits, was refuted at DESCEND entry and timed out (B 91 d0 / d3, B 458 d5,
    # A 80 d3, A 947 d0: early claims that c7 lands at 5-11 s).
    spd_min_hits: int = 0
    # candF hazard h (research/uid169/PLAN.md 7h; default off): extend UID 169's SPD hit gate (spd_min_hits, forest)
    # to our speed levers. spd_gate_nfb: in APPROACH below spd_min_hits, an NFB berth re-pick no longer inherits the
    # narrow pick's full-speed decision (free_berth_keep_speed); it is sped like any other pick (its own agree).
    # spd_gate_top > 0: in forest APPROACH below spd_min_hits, the approach top speed (forest_speed, 3.4 in our
    # overlay) is held at spd_gate_top (2.8 = UID 169's forest_speed) until the claim has spd_min_hits hits.
    spd_gate_nfb: bool = False
    spd_gate_top: float = 0.0
    # candF hazard i (PLAN 7i; default off): skip the OM obstacle memory (update, bumper, om_fd) once GKC has pinned
    # the seed's route kind to 'open' (terrain ground signature g < -0.05; city/forest/village keep the z=0 plane, so
    # a city seed never pins). OM keys on map_kind 'city', which ~92% of open seeds carry; there it only costs act
    # time (0 bumps measured on open). The literal PLAN rule (_mass_route_kind() == 'open') is not used: _ring_kind
    # can call low-rise city seeds 'open', and OM is the city collision bumper.
    om_skip_route_open: bool = False
    search_alt_gkc_open: float = 0.0     # >0: search AGL on seeds GKC pinned as open (candFd; 0 = off)
    # ---- candFe (research/opencity; every key default off = candFd behaviour) ----
    # TDF soft touchdown. Per-step touchdown log (research/opencity/tdlog.py, NFA, open): the DESCEND sink runs at
    # 2.78 m/s until 0.15 m above the pad estimate, so the drone hits the pad at 2.0-2.5 m/s; under the env's 4 solver
    # iterations it sinks 0.05 m (median) into the 1 mm disc and the pad cylinder, 3-5 steps after contact, and the
    # descent command (slew-limited to v_meas - 0.25) keeps it 0.03 m deep through the 0.5 s latch. The safety minimum
    # to the terrain 0.20 m under the pad top is set on that step (open 0.078 safety loss per seed). tdf_maps (key like
    # cst_maps: "forest" if is_forest else the route kind): in DESCEND within tdf_h of the pad estimate the sink is
    # capped by a braking profile v <= sqrt(tdf_vz^2 + 2 tdf_acc max(0, hc - tdf_h0)), hc = distance to contact from
    # the downward ray (when it reads the pad top: agl < above + tdf_agl_tol) else from the pad estimate; after the
    # ray has read the pad top within tdf_rest_h (contact imminent) and while |vz| < tdf_rest_v, the descent command
    # is capped at tdf_rest_vz (a light press instead of the slew-limited -0.23 m/s) for tdf_rest_win steps. tdf_cst:
    # while the profile brakes within cst_r / cst_above of the pad, the lateral command holds the current velocity (CST
    # needs vz < -cst_vz, which the profile removes in the last ~0.1 m). Skipped under the endgame hurry and on hops.
    tdf_maps: tuple = ()
    tdf_h: float = 1.0
    tdf_acc: float = 7.0
    tdf_h0: float = 0.06
    tdf_vz: float = 0.3
    tdf_agl: bool = True
    tdf_agl_tol: float = 0.12
    tdf_rest_vz: float = 0.0             # > 0: resting press cap (m/s); 0 = the base descent command after contact
    tdf_rest_h: float = 0.03
    tdf_rest_v: float = 0.25
    tdf_rest_win: int = 100
    tdf_cst: bool = True
    tdf_rest_maps: tuple = ()            # () = tdf_maps; else the resting press only on these keys (may be used alone)
    # review fix: HURRY_MAPS is (mountain, city), so the endgame-hurry skip never fires on open / village; TDF also
    # stands down in the last tdf_end_s seconds of the 60 s episode (base touchdown timing; 0 = no such guard)
    tdf_end_s: float = 1.0
    # APX-SCAN approach camera sweep (research/city_nfa_b b15: approach legs sweep fleet-new ground with the camera
    # fixed ahead; a +-40 deg sweep beyond 10 m of the claimed pad gains ~0.40 s per pad on city in the fixed-trajectory
    # model). apx_scan_maps (map_kind keys like yaw_scan_maps; "city" covers ~93% of open seeds): in APPROACH, farther
    # than apx_scan_far (planar) from the claimed pad, with at least apx_scan_min_hits on it (at or above the NE hit
    # ceiling of 12 on city/open, so an edge-of-frame miss can never NE-refute the claim) and camera clearance above
    # apx_scan_min_clear, the yaw toward the pad gets + apx_scan_deg * sin(2 pi step / apx_scan_steps). With 40 deg
    # a pad dead ahead stays inside the 88-92 deg FOV. Not on STEEP legs, APXV holds or inside ESK / EDR range (far),
    # nor in the CLIMB-claim envelope (dd.cenv: still climbing out low near the start; review fix).
    apx_scan_maps: tuple = ()
    apx_scan_deg: float = 40.0
    apx_scan_steps: int = 200
    apx_scan_far: float = 10.0
    apx_scan_min_hits: int = 12
    apx_scan_min_clear: float = 12.0
    # CLX (clx_maps ["forest"]): a RE-CLIMB (after a refute / abort; the first CLIMB has ended once) also ends when the
    # drone is within clx_room of forest_ceiling. CLIMB exits at AGL >= forest_alt - 0.6 = 3.4 m, but the ceiling clamps
    # the climb at z 4.5: over a bush / crown top > 1.1 m the exit can never happen. c7 trace B-list 566 d0: FPN refute
    # at 17.2 s over an obstacle (AGL 1.0 at z 4.4), hovered in CLIMB 36.9 s at 0.1 m/s, real pad claimed at 59.4 s
    # (TIMEOUT; 238 landed it at 28.8 s).
    clx_maps: tuple = ()
    clx_room: float = 0.3
    # HYS (fhy_maps): pass-side hysteresis for free_direction. Traces (c7 forest, 23 seeds): the pick changes side of
    # the command 0.52 times per flight second, a quarter of the runs last <= 2 steps (two passes of equal score
    # alternate; the slewed velocity averages them, i.e. heads at the obstacle between). With a committed side
    # (the side of the last pick more than fhy_dead_deg off the command) rays on the other side cost fhy_w
    # (45 deg of angle = 1); the commitment ends after fhy_release steps with the pick inside the dead zone, or when
    # the command turns by more than fhy_cmd_deg in one step (new leg / claim).
    fhy_maps: tuple = ()
    fhy_w: float = 0.5
    fhy_dead_deg: float = 3.0
    fhy_release: int = 10
    fhy_cmd_deg: float = 20.0
    # SDB (sdb_maps): stopping-distance brake while boxed. When no ray is clear to free_want_clear, free_direction
    # returns the ray with the most clearance and c7 flies it at free_slow x cruise = 2.1 m/s whatever that clearance
    # is. Traces (c7 forest, 23 seeds): 20 episodes with the pick's tube clear < 1 m, 5 of them ended in a crown / bush
    # collision at 3.5 m (438 d1 / d2 / d4, 75 d3): the pick's clearance fell 1.5 -> 0.3 m in 0.5 s at 2.1 m/s and
    # the saturation escape (<= 0.3 m) came too late. SDB caps the speed at sqrt(2 sdb_a (clear - sdb_d0)), at
    # least sdb_vmin, while the pick's clearance is below free_want_clear in SEARCH / APPROACH (the escape still takes
    # over at 0.3 m).
    sdb_maps: tuple = ()
    sdb_a: float = 1.5
    sdb_d0: float = 0.4
    sdb_vmin: float = 0.3
    # bughunt VGP (default off): generator-exact village goal prior for the router (pad_posterior.GEN_PRIOR,
    # see the comment block there). vgp_maps lists router kinds (only "village" is modelled); vgp_a0 is the
    # ring-area scale of P(ring spot found) = 1 - exp(-A / a0); vgp_u the uniform share of the fallback part.
    vgp_maps: tuple = ()
    vgp_a0: float = 21.0
    vgp_u: float = 0.2
    vgp_amax: float = 0.0      # > 0: VGP-lite, only starts with in-box 65-100 m ring area < vgp_amax m2
    # bughunt MZS (default off): mtn_zmax_start > 0 makes any seed with a start above it mountain. Flat-map
    # starts sit on platforms of at most 10 m (START_PLATFORM_MAX_Z) + 0.12 take-off buffer (+ open terrain):
    # max start z city 10.11, open 10.41, forest 10.12, village 1.32 over 5x1000 local seeds, while every
    # mountain seed has a start >= 16.6 m. The shipped S8 clue rule missed B629 (n=2, starts 34.5 / 16.3 m,
    # clue z 6.27 < 7): classified city, both drones timed out (0.010). Suggested value 12.
    mtn_zmax_start: float = 0.0
    # bughunt FOVE (default off): the env draws the camera FOV per seed, 90 + U(-2, 2) deg
    # (MovingDroneAviary: RandomState(map_seed), 2nd draw), while every geometry path here assumes 90
    # (PD.AP_FOV_DEG, PadDetector.fov_deg): final pad-estimate z is biased -0.06..-0.11 m at FOV < 89 and
    # +0.10..+0.15 m at FOV > 91 (c7 rows, 5 lists). With fove_on, depth frames + poses of up to
    # fove_max_drones drones are kept at the steps in fove_rec; at step fove_at the FOV is found by
    # reprojecting frame A's depth into frame B for candidate FOVs (coarse 0.25 deg, fine 0.05 deg,
    # truncated relative depth residual), median over drones and pairs; accepted when at least
    # fove_min_pairs pairs agree within fove_tol deg (MAD), then written to PD.AP_FOV_DEG and every
    # detector's fov_deg. Diag: src_counts fove_deg / fove_pairs / fove_mad.
    fove_on: bool = False
    fove_rec: tuple = (1, 11, 21, 31)
    fove_at: int = 32
    fove_max_drones: int = 4
    fove_min_pairs: int = 4
    fove_tol: float = 0.5
    # ---- gkc (round 10 P2, default off = c14) ----
    # GKC ground-signature kind check. The altitude ray gives the height of the surface under the drone exactly
    # (g = z - ALT x 20): the generator keeps the z = 0 ground plane on city / forest (g = 0.000), moves it to -1000 on
    # open (terrain mesh: continuous heights, negatives) and lays a 0.20 m slab (top 0.10) + 0.12 m tiles on village.
    # Map-kind routing (start z, walls, tall fraction) sends ~3 % of open seeds to the city router and a few village
    # seeds to city. While the band is not locked, every step samples g of each CLIMB / SEARCH / APPROACH drone with
    # AGL in gkc_agl (not within gkc_start_r of a start platform). At the band lock (before the route is planned),
    # map_kind city / open (not forest), >= gkc_min_n samples:
    #  "open" in gkc_maps: frac(|g| < gkc_z0) < gkc_f0_max and frac(g < gkc_neg) > gkc_fneg_min (the city / forest
    #    plane is never below 0) -> route kind open for the episode (map_kind unchanged, as for native open seeds),
    #    the router's evidence kind pinned to "open" (c7: 29 of 45 open misroutes are the router's geometric city fix
    #    from a phantom entry beyond 61 m, mid-flight) and forest ruled out (_forest_possible);
    #  "village" in gkc_maps: no sample at |g| < gkc_z0, and of the samples >= gkc_vil_start_r from every start
    #    platform (platform tops sit at 0.1-1.1 m on every map) >= gkc_min_n with none at |g| < gkc_z0 and
    #    >= gkc_vil_frac on the slab / tile heights (gkc_vil_bins) -> the village branch of _set_z_band.
    # One-directional by design (city seeds routed open score above the city mean). Diag: src_counts gkc = [t, n,
    # n0, nneg, nvil, map_kind, route kind, decision open / pin / village]; gkc_log adds gkc_s (the samples).
    gkc_maps: tuple = ()
    gkc_z0: float = 0.015
    gkc_neg: float = -0.05
    gkc_min_n: int = 60
    gkc_f0_max: float = 0.05
    gkc_fneg_min: float = 0.05
    gkc_vil_frac: float = 0.8
    gkc_vil_bins: tuple = (0.085, 0.215, 0.30, 0.45)
    gkc_agl: tuple = (0.3, 19.0)
    gkc_start_r: float = 0.0
    gkc_vil_start_r: float = 1.5       # village rule: only samples this far (xy) from every start platform, >= gkc_min_n
    gkc_log: bool = False
    gkc_vil_modes: int = 3             # village rule: the commonest gkc_vil_modes 1 cm heights hold >= gkc_vil_frac
    gkc_dry: bool = False              # decide and log only (src_counts gkc decision "dry:<dec>"), never act (base arms)
    # C-L2 gkc_vil_start_guard (default off): the GKC village relabel also needs every own start at village pad
    #   height (fewer than max(1, village_start_zmax_n) starts above village_start_zmax) and within village_start_box
    #   (max |x|, |y|), the rules _set_z_band already applies; rooftop-start city seeds (n=2-3) were relabelled
    #   village and searched the +-42 m village box while their pads sat 43-74 m out (18 of 4,200 C5 city seeds).
    gkc_vil_start_guard: bool = False
    # C-L4 gkc_city_pin (default off): on a route-open seed (map_kind city / open, route kind open, router evidence
    #   kind not fixed, GKC done and not open) with n <= gkc_city_max_n, the altitude-ray ground height g = z - AGL
    #   of every CLIMB / SEARCH / APPROACH drone (AGL 0.3-19 m, >= gkc_city_start_r from every own start) is
    #   sampled until gkc_city_t_max s: any g < gkc_neg (terrain below the plane = open) ends it without a pin;
    #   once the band is locked, t >= gkc_city_t_min and >= gkc_city_cells distinct 1 m cells had |g| <
    #   gkc_city_z0 (the generator keeps the exact z = 0 plane on city), the router evidence kind is pinned to
    #   "city" (as the kind_evidence mid-flight fix) while a SEARCH drone without a claim exists (no stale job).
    #   Diag: _b6_n / src_counts gkc_city = [t, step, cells, neg, route kind before, route kind after].
    gkc_city_pin: bool = False
    gkc_city_max_n: int = 3
    gkc_city_z0: float = 0.003
    gkc_city_cells: int = 10
    gkc_city_start_r: float = 1.5
    gkc_city_t_min: float = 1.9
    gkc_city_t_max: float = 12.0
    # C-RC1 ALIFT (default off = alift_maps empty): approach-stall lift on alift_maps. In APPROACH, once a claim has
    #   made no alift_gain m of progress for alift_stall_s s (alift_retry_s s after reaching a rung), the drone is
    #   farther than alift_min_d m, its lift is below alift_max_above m, the OM2 filter bound its horizontal command
    #   within the last 25 steps and nothing remembered sits above it: the approach height rises by alift_rung m (the
    #   first rung at least to the highest remembered top on the line to the pad + alift_margin when alift_line),
    #   climbing at alift_vz with the horizontal command zeroed until within 0.8 m of the rung. alift_release_m > 0:
    #   progress of that many m below the first-rung distance with a clear line drops the lift. Diag: _b6_n
    #   alift_rung / alift_nobind / alift_ovh / alift_rel, alift_ev = [t, drone, claim, lift, dist] (capped).
    alift_maps: tuple = ()
    alift_stall_s: float = 3.0
    alift_retry_s: float = 1.5
    alift_gain: float = 0.5
    alift_min_d: float = 3.0
    alift_rung: float = 3.0
    alift_max_above: float = 12.0
    alift_vz: float = 2.0
    alift_line: bool = True
    alift_margin: float = 1.3
    alift_release_m: float = 0.0
    # ---- prv (round 10 P7, default off = c14) ----
    # PRV provisional route. The learned route waits for the band lock (median 1.38 s, n = 2: 3.6 s); until then CLIMB
    # creeps / SEARCH flies toward the spiral target and the route's first leg arrives with a turn (median 36 deg, 21 %
    # > 90 deg; early-turn loss ~0.2-0.3 s per drone on city / open / forest). With prv_maps, on detector ticks before
    # the lock the depth frames of the live drones give the ground under the view: world z of every valid pixel within
    # prv_rng m (stride 4); z within prv_z0 of 0 is the city / forest ground plane, z < prv_neg is open terrain (the
    # generator moves the plane to -1000 on open). Once >= prv_min_pts points below 1 m are in: provisional kind
    # "city" if a start lies beyond the open world (61 m, the router's own rule), "open" if >= prv_open_neg of them are
    # below the plane and < prv_open_plane on it, "city" if >= prv_city_plane are on the plane (< prv_city_neg below)
    # and forest is impossible (_forest_possible() False or a start below prv_forest_zmin), else no provisional kind
    # (the base waits for the lock; tried until prv_max_t s). A provisional kind in prv_maps is planned at once with
    # the normal pipeline (_init_learned_route under that kind) and flown; at the lock the route is kept when the
    # locked route kind is the provisional one (the plan depends only on starts, clue, n and kind) and re-planned
    # from scratch otherwise. Diag: src_counts prv = [t plan, kind, t lock, locked kind, kept].
    prv_maps: tuple = ()
    prv_rng: float = 10.0
    prv_z0: float = 0.03
    prv_neg: float = -0.05
    prv_min_pts: int = 400
    prv_open_neg: float = 0.1
    prv_open_plane: float = 1.01      # > 1: off (A smoke: open terrain has 8-10 % of its near points within 3 cm of z = 0)
    prv_city_plane: float = 0.2
    prv_city_neg: float = 0.01
    prv_max_t: float = 1.5
    prv_forest_zmin: float = 1.7
    # ---- candHt act-time guard (research/strike_emu/GUARD.md; default off = candHv) ----
    # act_guard_s > 0: the controller stamps the start of every act() (perf_counter) in _g_t0. On a detector tick
    # the detector batch is cut to the drones that fit before act_guard_s wall seconds, from running estimates of
    # the per-drone detector cost and of the work left after the detector (control loop + router); the other
    # drones are detected on the next steps, oldest detection first (a drone waits at most act_guard_max_age
    # steps). The router defers its replan job stage the same way (act_guard_router). The work is done later,
    # never skipped: per-step compute is amortised, and with time to spare the guard path is the shipped one.
    act_guard_s: float = 0.0
    act_guard_first_s: float = 1.5        # the first act() of an episode (the validator allows 2 s)
    act_guard_max_age: int = 6            # steps a drone's detection may be deferred before it is forced
    act_guard_router: bool = True
    act_guard_det0: float = 0.035         # initial per-drone detector estimate (s)
    act_guard_rest0: float = 0.02         # initial estimate of the work after the detector (s)
    act_guard_job_max_skips: int = 8      # router job stage deferred at most this many steps in a row
    act_guard_chunk: int = 2              # drones per detector call; the time is re-checked between calls
    # ---- candHb descent builder (research/round5/DESCENT.md; every key default off = candHt) ----
    # DSK early sink in the final approach, per map kind ("forest" if is_forest else the route kind). Village traces
    # (C3, research/round5): the approach is flown at approach_alt (4 m above the pad) at 2.95 m/s; the EDR drop
    # starts ~4 m out, but its lateral priority under the 3 m/s norm leaves ~0.3 m/s of sink until the braking
    # profile starts at 2.25 m, so the drone is over the pad (< 0.35 m) still 2.1 m up (median) and sinks the rest
    # pure-vertically (0.8 s to contact). ESK does the early sink on city/open with shared settings; DSK gives other
    # kinds their own. dsk_by_map {kind: {"r": 8, "above": 2.6, "vz": 1.2, "agl_tol": 0.6, "min_hits": 0, "kp": 1,
    # "min_claim_d": 0, "dsc_gain": 0, "dsc_lat": 0.9, "dsc_above": 0.8, "dsc_min_claim_d": 4}}: inside r m of the
    # claimed pad (>= min_hits hits, claim made >= min_claim_d away, and while the downward ray reads at least the
    # height above the pad estimate minus agl_tol: over ground at about pad level, not over a roof / slope above the
    # pad; agl_tol < 0 = no ray test) the APPROACH height target drops to pad z + above (never raised) with the
    # vertical cap raised to vz and P gain kp ("line_d" > 0: gain max(kp, |v_xy| / max(dist - line_d, 0.3)), i.e. a
    # straight glide line that reaches the target line_d m before the pad). dsc_gain > 0: the ESK DESCEND lateral law clip(d * dsc_gain, +-dsc_lat)
    # while higher than dsc_above above the pad (claims made >= dsc_min_claim_d away and latched as below; the EDR
    # profile still wins while it is active). "clear" > 0 (mountain): the sink target is never below the highest known surface under
    # the drone (downward ray) and along the straight path to the pad (fleet LAV max-height grid, +-half_w m
    # corridor, cells within excl_r m of the pad centre skipped) + clear; a target at or above the normal approach
    # height leaves the approach unchanged. "steep": true: while the early sink is active the STEEP 3-D line
    # (steep_maps) aims at that target instead of pad z + steep_above. Only approaches that come down through the
    # target are changed: the claim latches once the drone is more than latch_m (0.3) above the target; a climbing
    # approach (claim made below the pad, take-off claims) keeps the base law - lowering its climb put a mountain drone
    # over the pad 1.9 m up at 2.4 m/s, it touched down sliding and was lost (dev run DM-b, 1609126988:67 d3).
    # The latch belongs to one claim instance (reset in _claim_note on every claim / hand-over, so a re-claim of the
    # same pad index starts unlatched), and the DSK DESCEND lateral law acts only on latched claims: every descent
    # whose approach the early sink could not have lowered keeps the base law. Counters in _b6_n (hb_*).
    dsk_by_map: dict = field(default_factory=dict)
    # ---- candHc OM2 fleet obstacle memory (research/round5/CITY.md; every key default off = candHt) ----
    # Traced hT3/hT4 city crashes: ~40 % hit a building while the camera pointed > 45 deg off the travel direction
    # and ~half while sinking > 0.6 m/s toward a surface below the view; forensic re-flies (contact point vs the
    # crashed drone's depth history) show the struck surface was seen earlier and had left the frustum, or lies
    # on the building's convex collision hull 0.3-0.5 m outside the drawn surface. om2_maps ("forest" if is_forest
    # else map_kind): one world-frame 2.5-D grid for the whole fleet (om2_cell m cells, +-om2_half m; team/autopilot/
    # omgrid.py): highest / lowest surface point seen per cell and the lowest height a depth ray crossed it without a
    # hit. On non-detector steps (om2_upd_skip_det) up to om2_upd_max live drones (stalest first) fold their frame
    # in (om2_hit_grid x om2_hit_grid pooled hits, om2_carve_grid rays carved every om2_carve_step m); points within
    # om2_mask_r of a moving teammate are dropped. Filter (last stage before the slew, om2_phases): cells outside the
    # current camera frustum (om2_view_tan, om2_view_min; om2_inview also keeps in-view cells) whose surface reaches
    # the drone's height (om2_col: surfaces stand on the ground, else the seen [bot, top] interval) bound the
    # horizontal command toward them by the stopping-distance speed (om2_a, om2_lat) to keep om2_r + om2_mh
    # clearance (inside it a push away <= om2_push; tangential motion is kept); with om2_vz, cells below the drone
    # that it is over or closing on within om2_vt s bound the sink rate so it can stop om2_r + om2_mv above them.
    # om2_pad_r: in APPROACH cells within this horizontal distance of the claimed pad are ignored. om2_zrel > -90:
    # hit points lower than the drone's own z + om2_zrel are not stored (forest ground). om2_unk_speed >= 0: the
    # horizontal speed is also capped so the drone can stop before the first path cell no camera has seen
    # (carved) at its height outside the frustum (floor om2_unk_speed); om2_unk_turn then turns the camera to the
    # travel direction while that cap binds. om2_shadow: compute and count only (the command is not changed).
    om2_maps: tuple = ()
    om2_skip_open: bool = True            # off once GKC pinned the seed's route kind to open (as om_skip_route_open)
    om2_cell: float = 0.5
    om2_half: float = 100.0
    om2_hit_grid: int = 32
    om2_carve_grid: int = 16
    om2_carve_step: float = 0.75
    om2_carve_el: tuple = (-25.0, 10.0)   # carve only rays at these world elevations (deg): the band at the drone's height
    om2_carve_max: float = 12.0          # carve rays only this far (m); carving runs only with om2_unk_speed >= 0
    om2_upd_max: int = 2
    om2_upd_skip_det: bool = True
    om2_mask_r: float = 0.6
    om2_col: bool = True
    om2_zrel: float = -99.0
    om2_phases: tuple = ("SEARCH", "APPROACH")
    om2_r: float = 0.12
    om2_mh: float = 0.6
    om2_mv: float = 0.35
    om2_a: float = 3.0
    om2_lat: float = 0.15
    om2_az: float = 2.0
    om2_push: float = 0.4
    om2_look: float = 4.0
    om2_view_tan: float = 0.95
    om2_view_min: float = 0.6
    om2_inview: bool = False
    om2_vz: bool = True
    om2_vt: float = 1.0
    om2_pad_r: float = 1.5
    om2_unk_speed: float = -1.0
    om2_unk_turn: bool = False
    om2_unk_margin: float = 0.3          # a path cell counts as seen free once a carve ray crossed it no higher than z + this
    om2_turn_lead: float = 120.0         # om2_unk_turn: lead of the camera command beyond which the turn direction is held
    # om2_turn_blind_deg > 0 (BLIND turn): whenever the (filtered) horizontal command is faster than om2_turn_blind_v
    # and more than om2_turn_blind_deg off the camera, the camera is commanded toward it through _om2_turn
    # (consistent direction; overrides the yaw scan / GLA for those steps). Traced blind city crashes: the +-50 deg
    # scan swept the target across the 180 deg wrap and the camera reversed for 2-3 s while flying 2.8 m/s.
    om2_turn_blind_deg: float = 0.0
    om2_turn_blind_v: float = 1.0
    om2_shadow: bool = False
    # om2_hull_k > 0 (HULL lift): a low structure cell (om2_hull_foot <= top <= the drone's clearance floor) within
    # om2_hull_r m of a cell whose top reaches the drone's height, over structure cells only, counts as
    # max(top, top' - om2_hull_k * d) and is never exempted as in view (convex collision hulls; see _om2_lift).
    om2_hull_k: float = 0.0
    om2_hull_foot: float = 1.8
    om2_hull_r: float = 5.0
    om2_hull_max_low: int = 40
    om2_hull_max_tall: int = 160
    # per-map variants (map key "forest" if is_forest else map_kind): om2_interval_maps use the seen [bot, top]
    # interval instead of ground-standing columns (forest canopies); om2_zrel_by_map (map, value, ...) overrides
    # om2_zrel; om2_hull_maps limits the hull lift to these maps (empty = every om2 map).
    om2_interval_maps: tuple = ()
    om2_zrel_by_map: tuple = ()
    om2_hull_maps: tuple = ()
    # om2_route_city (round 5 review): on a map key "city" OM2 runs only while _mass_route_kind() is "city" too.
    # Open seeds carry map kind city ~93 % of the time and GKC pins only ~82 % of them; on the rest the memory has
    # no building to guard but the blind turn still overrides the yaw scan.
    om2_route_city: bool = False
    # om2_h3 (H3 hull memory, research/round5/CITY.md): city buildings collide as the convex hull of their mesh.
    # hT3c city crashes (C1 + U149, 81 events): 50 hit hull side fills over the STREET beside building-n's podium
    # (no rendered structure under the contact, 0.3-1.1 m beyond the podium's side edge, z 3.2-5.1 m), 11 the hull
    # above a rendered low roof (the om2_hull_k lift covers those), 20 a rendered surface. H3: every connected
    # component of remembered structure cells (seen top >= om2_h3_foot, 8-connected, at least om2_h3_min_cells
    # cells, highest top >= om2_h3_min_top, at most om2_h3_max_cells cells) within reach + om2_h3_ext m of the
    # drone is replaced by the 3-D convex hull of its cell columns (ground point at the cell centre, seen top at
    # the 4 cell corners with om2_h3_top_corners, else at the centre). Buildings stand on the ground, so that hull
    # lies inside the true collision hull up to the cell quantisation (<= 0.35 m at the top edges) and the pooled
    # ray placement: it asks for (almost) no room the real shape does not take. Top corners matter for steep
    # skirts over narrow low edges (building-n's 0.5 m canopy under the 10.4 m block, skyscraper-c's 1.9 m
    # canopy): with centre points the edge cell is itself a hull vertex and is never raised.
    # Every cell of the filter window inside that hull counts with the hull's height there (te = max(top,
    # envelope)); cells raised more than 0.3 m are never exempted as in view (the camera cannot draw them).
    # Offline (research/round5/CITY.md 3): a drone passing building-n's podium side at 4 m with only its own
    # forward frames is guarded on 10/10 side-fill lines with H3, 4/10 with the lift alone, 0/10 with the plain
    # memory. om2_h3_maps limits H3 to these map keys (empty = every om2 map).
    om2_h3: bool = False
    om2_h3_foot: float = 1.8
    om2_h3_min_top: float = 4.0
    om2_h3_min_cells: int = 6
    om2_h3_max_cells: int = 1200
    om2_h3_ext: float = 12.5
    om2_h3_max_comp: int = 4
    om2_h3_top_corners: bool = True
    om2_h3_maps: tuple = ()

def _mtd_cells(rel, r0, r1, cell):
    """MTD hm (c29k2 _s1m_cells): height-map cells from points relative to the estimate (x, y, z - top):
    (M, 3) [cell centre x, y, median z] over the r0..r1 annulus."""
    d = np.hypot(rel[:, 0], rel[:, 1])
    m = (d >= r0) & (d <= r1) & (np.abs(rel[:, 2]) < 2.5)
    R = rel[m]
    if len(R) == 0:
        return np.zeros((0, 3))
    ij = np.floor(R[:, :2] / cell).astype(np.int64)
    key = (ij[:, 0] + 1000) * 4000 + (ij[:, 1] + 1000)
    o = np.lexsort((R[:, 2], key))
    key, z, ij = key[o], R[o, 2], ij[o]
    _, st_, cnt = np.unique(key, return_index=True, return_counts=True)
    return np.c_[(ij[st_, 0] + 0.5) * cell, (ij[st_, 1] + 0.5) * cell, z[st_ + cnt // 2]]


def _mtd_quad(C, minc):
    """MTD hm (c29k2 _s1m_quad): robust quadratic surface z = a x + b y + c + d x^2 + e xy + f y^2: (coef, keep)."""
    if len(C) < minc:
        return None, None
    x, y = C[:, 0], C[:, 1]
    A = np.c_[x, y, np.ones(len(C)), x * x, x * y, y * y]
    keep = np.ones(len(C), bool)
    coef = None
    for _ in range(3):
        coef = np.linalg.lstsq(A[keep], C[keep, 2], rcond=None)[0]
        r = C[:, 2] - A @ coef
        sd = float(np.sqrt(np.mean(r[keep] ** 2)))
        nk = np.abs(r) <= max(2.5 * sd, 0.05)
        if nk.sum() < minc or np.array_equal(nk, keep):
            break
        keep = nk
    return coef, keep


_MTD_RING = [None]


def _mtd_hm_eval(coef, off_r):
    """MTD hm (c29k2 _s1m_hm_pick): (best bearing index of 16, its clearance, the centre's clearance): minimum
    distance from the resting drone body (r 0.06 m, 0.025 m tall, 0.0125 m over the pad top) to the quadratic surface
    on rings 0.62-1.61 m around the estimate."""
    if _MTD_RING[0] is None:
        rr = np.arange(0.62, 1.61, 0.04)
        aa = np.linspace(0.0, 2.0 * math.pi, 72, endpoint=False)
        _MTD_RING[0] = ((rr[:, None] * np.cos(aa)[None, :]).ravel(), (rr[:, None] * np.sin(aa)[None, :]).ravel())
    gx, gy = _MTD_RING[0]
    gz = coef[0] * gx + coef[1] * gy + coef[2] + coef[3] * gx * gx + coef[4] * gx * gy + coef[5] * gy * gy
    dz = np.maximum(np.abs(gz - 0.0125) - 0.0125, 0.0)

    def clr(qx, qy):
        dxy = np.maximum(np.hypot(gx - qx, gy - qy) - 0.06, 0.0)
        return float(np.sqrt(dxy * dxy + dz * dz).min())
    c0 = clr(0.0, 0.0)
    best_b, best_c = 0, -1.0
    for b in range(16):
        a = 2.0 * math.pi * b / 16.0
        cc = clr(off_r * math.cos(a), off_r * math.sin(a))
        if cc > best_c:
            best_b, best_c = b, cc
    return best_b, best_c, c0


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
    gate_hold: int = 0
    dsc_gated: bool = False
    apx_key: int = -1                 # claim the stall watchdog is tracking
    apx_best: float = 1e9             # closest the drone has got to that pad
    apx_best_step: int = 0
    claim_dist: float = 0.0
    hop: bool = False
    hops: int = 0
    climb_yaw: Optional[float] = None  # last yaw aimed at the climb target (climb_yaw_hold)
    gnd: list = field(default_factory=list)   # ground heights sampled near the claimed pad (ray_probe)
    gnd_key: int = -1
    gnd_g: Optional[float] = None
    probe: Optional[dict] = None
    touch_buf: list = field(default_factory=list)
    touch_key: int = -1
    touch_off: Optional[np.ndarray] = None
    tko_rate: float = 0.0
    clue_low: bool = False             # clue-height rule holds this drone below its default search height
    probe_tries: int = 0
    aligned_once: bool = False         # camera has faced the travel direction once (early_align)
    # APXV-R approach check state (apx_vfy_*): claim tracked, its first step and hits, recent
    # (step, hits) history, state (0 watching, 1 passed, 2 first hold, 3 orbit + second hold), hold
    # start step and hits, hold point, late-claim flag, transit end / second-hold start steps,
    # chord length and heading, and the hold command of this step.
    vf_key: int = -1
    vf_step0: int = 0
    vf_hits0: int = 0
    vf_hist: list = field(default_factory=list)
    vf_state: int = 0
    vf_t: int = 0
    vf_h: int = 0
    vf_pt: Optional[np.ndarray] = None
    vf_late: bool = False
    vf_tr: int = -1
    vf_t2: int = -1
    vf_ch: float = 0.0
    vf_yc: float = 0.0
    vf_cmd: Optional[tuple] = None
    pg_side: float = 0.0               # CG pass-through guard: side (+1/-1) of the last guard step, 0 none
    below: bool = False                # P2r: slid below the pad top in DESCEND (recovery active)
    below_seen: bool = False           # P2r: a below event happened on this descent
    vra_floor: float = -1.0            # collide VRA: current height floor (m, world z), -1 = none
    vra_t: int = -1                    # collide VRA: step of the last floor raise
    gate_inview: int = 0               # S1: APPROACH detector ticks with the claimed pad in view
    claim_elev: float = 0.0            # S1: degrees below the drone of the pad at claim time
    zstill: int = 0                    # S11 dead_zero_steps: consecutive frozen steps
    zprev_xy: Optional[np.ndarray] = None
    note_dist: float = 0.0             # batch 4: horizontal claim distance, set by _claim_note on every claim path
    gl_pad: int = -1                   # GLA: entry the camera is aimed at (-1 none)
    gl_until: int = -1                 # GLA: last step of the look-at
    cenv: bool = False                 # CLIMB-claim envelope active
    bl_key: int = -1                   # F3: claimed entry whose P2r below events are counted
    bl_n: int = 0                      # F3: below events on that entry
    bl_k: int = 0                      # F3: offset bearings tried
    bl_orig: Optional[np.ndarray] = None   # F3: entry xy before the first offset retarget
    bl_brg: Optional[list] = None      # F3: offset bearings (rad), in the order they are tried
    bl_on: bool = False                # F3: retargeted: climb before moving sideways
    sab_key: int = -1                  # LANDSEG P4b diag: claim on which settle-by-height last fired first
    tilt_rest: int = 0                 # LANDSEG P4c: consecutive settle steps resting tilted past settle_tilt_rad
    hop_tilt: bool = False             # LANDSEG P4c: the current hop is a tilt-rest hop
    edr_key: int = -1                  # land_seq EDR: claim whose drop started early (-1 none)
    nosep_step: int = -1               # land_ext park_nosep: step on which PARK / CST fired (no separation push)
    fld_hkey: int = -1                 # land_ext FLD hold: claim the hold counter belongs to
    fhy_side: float = 0.0              # forest_nav HYS: committed pass side (+1 left / -1 right of the command, 0 none)
    fhy_n: int = 0                     # forest_nav HYS: consecutive steps with the pick inside the dead zone
    fld_hn: int = 0                    # land_ext FLD hold: held steps inside the base entry radius
    fld_hmin: float = 99.0             # land_ext FLD hold: closest approach while held inside the base entry radius
    fld_hoff: bool = False             # land_ext FLD hold: released for this claim (fld_hold_back)
    tdo_key: int = -1                  # land_safe TDO: claim the ring samples / offset belong to (-1 none)
    tdo_buf: list = field(default_factory=list)   # TDO: per-frame (k, 3) world points near the claimed pad
    tdo_nb: int = -1                   # TDO: number of frames at the last fit
    tdo_off: Optional[np.ndarray] = None   # TDO: touchdown offset (xy, relative to the pad estimate)
    tdo_frozen: bool = False           # TDO: offset frozen (DESCEND entered)
    tdo_rs: list = field(default_factory=list)    # TDO rim: (x, y, surface z under the drone) samples
    tdo_eff: Optional[np.ndarray] = None   # TDO: offset in use this step (lam * u, or the fixed offset)
    tdo_lam: float = 0.0               # TDO rim: current lam
    tdo_why: str = ""                  # TDO: status of the last fit (ok / flat / stab / pts / sect / hits)
    tdo_g: Optional[np.ndarray] = None # TDO: terrain gradient of the last fit (diag)
    trr_key: int = -1                  # audit_mf TRR: claim the hop counters belong to
    trr_n: int = 0                     # TRR: consecutive tilted-rest steps
    trr_hops: int = 0                  # TRR: hops done for trr_key
    trr_left: int = 0                  # TRR: hop steps left
    lsd_key: int = -1                  # phantom_land LSD: claim of lsd_seen
    lsd_seen: int = -10 ** 9           # LSD: last step with agl < 0.3 at the pad top on lsd_key
    pvr_key: int = -1                  # phantom_land PVR: claim the ray samples belong to
    pvr_rs: list = field(default_factory=list)   # PVR: (x, y, surface z) samples near the claimed entry
    pvr_ok: bool = False               # PVR: raised platform proven for pvr_key
    pvr_n: int = 0                     # PVR: steps held at the hold height without a proof
    pvr_nt: int = -1                   # PVR: number of samples at the last proof test
    tdf_key: int = -1                  # candFe TDF: claim the touchdown state belongs to
    tdf_low: int = -10 ** 9            # candFe TDF: last step the ray read the pad top within tdf_rest_h (or an impact)
    tdf_vzp: float = 0.0               # candFe TDF: measured vz on the previous TDF step
    dsk_key: int = -1                  # candHb DSK: claim whose approach came down through the sink target (latched)
    om2_yaw: Optional[float] = None    # candHc OM2: camera heading requested this step by the unknown-space cap
    om2_turn_sgn: float = 0.0          # candHc OM2: held turn direction of the unknown-space camera turn (0 = none)
    mtd_key: int = -1                  # MTD: claim the gate / rim-guard state belongs to
    mtd_gate: Optional[bool] = None    # MTD: gate verdict (frozen at DESCEND entry; None = not evaluated)
    mtd_gfrz: bool = False             # MTD: gate verdict frozen
    mtd_rg_hold: bool = False          # MTD rim guard: holding after a fire
    mtd_rg_nf: int = 0                 # MTD rim guard: fires on this claim
    mtd_rg_run: int = 0                # MTD rim guard: consecutive signal steps
    mtd_rg_desc: int = 0               # MTD rim guard: consecutive sinking steps
    mtd_rg_agl: list = field(default_factory=list)   # MTD rim guard: last 10 down-ray readings
    mtd_rg_t0: int = 0                 # MTD rim guard: step of the last fire
    mtd_rg_sfire: float = 0.0          # MTD rim guard: surface z under the drone at the fire
    mtd_rg_pfire: Optional[np.ndarray] = None   # MTD rim guard: drone xy at the fire
    mtd_logged: bool = False           # MTD: DESCEND-entry row written for mtd_key
    mtd_hmfit: bool = False            # MTD hm: the TDO offset of mtd_key comes from a height-map fit
    alift_key: int = -1                # C-RC1 ALIFT: claim the lift state belongs to (-1 none)
    alift_best: float = 1e9            # ALIFT: closest approach to that pad so far
    alift_t0: int = 0                  # ALIFT: step of the last progress / rung fired / rung reached
    alift_dz: float = 0.0              # ALIFT: lift above the approach height (m, 0 = none)
    alift_ref: float = 1e9             # ALIFT: smallest distance at which a rung fired (release reference)
    alift_top: int = -1                # ALIFT: step the drone reached the current rung (-1 none)
    alift_last: int = -1000            # ALIFT: last step the ALIFT block ran for this drone (gap > 25 -> reset)

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
    views: list = field(default_factory=list)   # well-separated observer positions
    split: bool = False
    split_r: float = 0.0    # >0: claim-spacing radius of this split entry (pad_split_landed_r)
    probed: bool = False    # xy fixed by the downward-ray probe; detections no longer move it
    cps: bool = False       # phantom_land CPS: entry created by the close-pair split
    ne_run: int = 0         # NE: in-view detector ticks without a hit since the last hit / fire
    ne_safe: bool = False   # NE: a claimant passed the descend-entry gates on this entry; NE never fires on it
    bl_raw: object = None   # F3: recent merged proposals (x, y, z, range) while dsc_below_retarget is on
    whits: int = 0          # detv2 WCONF: hits added by weak proposals (capped at wconf_max, never past the claim gate)
    t0a_ticks: int = 0      # detv2 T0-a: detector ticks on which this entry gained detector hits (counted on t0a_maps)
    fpn_run: int = 0        # FPN: in-view no-hit detector ticks of the current claimant (or of the fleet, unclaimed)
    fpn_by: int = -2        # FPN: whose views the run counts (claimant index, -1 = unclaimed)
    mtd_xyz: Optional[np.ndarray] = None   # MTD: inverse-variance centre estimate (None = no state yet)
    mtd_info: float = 0.0   # MTD: accumulated weight (capped at mtd_info_cap)
    mtd_rec: object = None  # MTD: deque of recent fixes (step, x, y, z, range)
class SwarmAutopilotAgent:
    def __init__(self, cfg: Optional[AutopilotConfig] = None,
                 pad_cfg: Optional[PD.PadConfig] = None,
                 detector=None, forest_detector=None, village_detector=None,
                 mountain_detector=None, posterior=None, route_planner=None,
                 city_detector=None):
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
        if getattr(self, "_fove_deg", None) is not None:
            # bughunt FOVE: back to the assumed 90 deg for the next episode
            PD.AP_FOV_DEG = 90.0
            for nm in ("detector", "general_detector", "village_detector", "mountain_detector",
                       "forest_detector", "city_detector"):
                det = getattr(self, nm, None)
                if det is not None and hasattr(det, "fov_deg"):
                    det.fov_deg = 90.0
        self._fove_deg = None
        self._fove_frames = None
        self._gkc_s = []            # gkc: (step, drone, g, agl, d_start) ground samples before the band lock
        self._gkc_done = False
        self._gkc_route = None
        self._gkcc_cells = set()      # C-L4 gkc_city_pin: 1 m cells with |g| < gkc_city_z0
        self._gkcc_neg = 0            # C-L4: samples below the plane (g < gkc_neg)
        self._gkcc_done = False
        self._prv_g = [0, 0, 0]       # prv: ground points below 1 m / on the plane / below the plane
        self._prv_kind = None
        self._prv_force = None
        self._prv_off = False
        self._prv_relocked = False
        self.d: List[_Drone] = []
        self.pads: List[_Pad] = []
        self.bad: List[np.ndarray] = []
        self._cov: set = set()
        self.src_counts: dict = {}
        self._yt_n = {"ay_steps": 0, "vt_climb": 0, "vt_release": 0, "fe_steps": 0, "ee_steps": 0}   # yaw_time diag counters
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
        self._wall_sum = 0.0        # kind_wall_maps: running mean of the wall fraction
        self._wall_n = 0
        self._wall_frac = None
        self._zb_rej = []           # zband_rescue_hits: clusters of z-band-rejected proposals
        self._zb_switched = False
        self._merge_score = None    # detector score of the proposal being merged, when known
        self._launch_z = None
        self._own_start = None
        self._wedge_n = 0
        self._home_of = {}
        self._orphans_taken = set()
        self.wedge_phase = 0.0
        self._wedge_dynamic = self.cfg.rewedge
        self.step_i = -1
        self._g_t0 = None                 # act guard: start of the running act() (set by the controller)
        self._g_first = True
        self._g_pend = []
        self._g_cyc = None                # act guard: the open detection cycle of a split tick
        self._g_last = {}
        self._g_det = float(getattr(self.cfg, "act_guard_det0", 0.035))
        self._g_rest = float(getattr(self.cfg, "act_guard_rest0", 0.02))
        self._g_mark = None
        self._g_n = {"ticks": 0, "split_ticks": 0, "deferred": 0, "catchup_steps": 0, "forced": 0,
                     "chunk_stops": 0, "router_defer": 0, "max_act_ms": 0.0, "over_limit": 0}
        self.n = 0
        self._mass_xy = np.zeros((0, 2), dtype=float)
        self._mass_prior = np.zeros(0, dtype=float)
        self._mass_seen = np.zeros(0, dtype=np.float32)
        self._mass_key_to_idx = {}
        self._route_waypoints = None
        self._splan_deferred = None
        self._replan = None
        self._replan_free = []
        self._replan_key = None
        self._replan_t = -1e9
        self._replan_applied = 0
        self._replan_pending = None
        self._plan_disabled = False
        self._diag_route_step = None
        self._diag_splan_step = None
        self._diag_err = None
        self._replanned = set()
        self._mem = None
        self._lav_top = None               # mtn_speed LAV: fleet max-height grid (lazy)
        self._lav_live = {}                # mtn_speed LAV: drone -> step the LAV drove its vz
        self._lav_n = {"on": 0, "fb": 0, "inf": 0, "upd": 0}
        self._lav_dbg = {}                 # mtn_speed LAV: drone -> last (smin_c, smin_side, smax, sp, slope, cells)
        self._hunt_post = None
        self._fcl_frames = None             # fov_clue FCL: FOVE frames / estimate / applied state
        self._fcl_est = None
        self._fcl_applied = False
        if getattr(self.cfg, "fcl_maps", ()):
            _fcl_set(None)
        self._hunt_seen = None
        self._hunt_xy = None
        self._hunt_tgt = {}
        self._splan = None
        self._start_clear = {}
        self._tall_cue = {}
        self._splan_pred = None
        self._n_start_bad = 0         # start_mask_r: self.bad[:_n_start_bad] are start platforms
        self._smask_n = 0             # diag: proposals let through only by the smaller start mask
        self._hopg_n = 0              # diag: take-off hop guard steps (tko_hop_guard_maps)
        self._park_n = 0              # diag: PARK steps (land_seq)
        self._lvl_n = 0               # diag: LVL steps (land_seq)
        self._cst_n = 0               # diag: CST steps (land_seq)
        self._vf_log = []             # diag: APXV-R events (t, drone, pad, event, hits)
        self._wstop_n = 0             # diag: WSTOP firing steps (wall_stop_maps)
        self._wstop_drones = set()
        self._2opt_done = set()       # assign_2opt_m: pad pairs already swapped
        self._2opt_log = []           # diag: 2-opt swaps (t, drone i, drone j, pad p, pad q, gain)
        self._cg_n = {"ext_calls": 0, "ext_pts": 0, "pass_kept": 0, "pass_guard": 0, "sat_esc": 0,
                      "pass_back": 0}  # diag: CG
        self._om = {}                 # blind_guard OM: drone -> deque of (step, world pts)
        self._om_cache = {}           # blind_guard OM: drone -> (step, world pts outside the frustum)
        self._om_n = {}               # blind_guard diag counters
        self._om_stall = {}           # blind_guard HM3: drone -> [anchor xy, anchor step, last hull-bind step]
        self._om_esc = {}             # blind_guard HM3: drone -> step until which lifted points are ignored
        self._fsb_stall = {}          # fsb: drone -> [anchor xy, anchor step, last safety-bind step]
        self._fsb_esc = {}            # fsb: drone -> step until which the safety margin is off
        self._fsb_start = {}          # fsb FSB-M: drone -> start xyz (scoring take-off grace)
        self._fsb_est = {}            # fsb FSB-M: drone -> committed running-min estimate (centre-to-point, m)
        self._fsb_pend = {}           # fsb FSB-M: drone -> [(step, est)] not yet committed
        self._om_wc = None            # fsb: (step, drone, current-frame relative points) from _om_near
        self._om2 = None              # candHc OM2: fleet ObstacleGrid (lazy)
        self._om2_last = {}           # candHc OM2: drone -> step its frame was last folded in
        self._om2_n = {}              # candHc OM2: diag counters (src_counts["om2"])
        self._om2_fire = {}           # candHc OM2 diag: drone -> (step, flags, |v_h| in, |v_h| out, vz in, vz out)
        self._om2_hist = {}           # candHc OM2 diag: drone -> [(step, flags)] binding steps (capped)
        self._om2_h3c = {}            # candHc OM2 H3: component key -> hull facets (or False), FIFO-bounded
        self._below_log = []          # diag: P2r below events (t, drone, pad, above, dist, agl)
        self._vveto_log = []          # diag: P1b vetoes (t, drone, pad, z, hits, rate)
        self._sab_log = []            # diag: LANDSEG P4b first settle-by-height contact per drone and claim
        #                               (t, drone, pad, pos z - pad z, agl, dist, vz)
        self._tilt_hop_log = []       # diag: LANDSEG P4c tilt-rest hops (t, drone, pad, tilt, dist, agl, hops)
        # batch 3 (bughunt fix specs)
        self._start_z = []            # S10: z of each start entry, parallel to self.bad[:_n_start_bad]
        self._b3_log = []             # diag: S1 gate exemptions, S2 hurry entries, S8 kind, S11 dead
        self._b3_n = {"s1_exempt": 0, "s1_gated": 0, "s2_apx": 0, "s3_pass": 0,
                      "s5_skip": 0, "s6_drop": 0, "s9_pass": 0, "s10_pass": 0, "s11_dead": 0}
        # batch 4 diagnostics
        self._ne_log = []             # NE fires (t, entry, hits, claimant or None, aborts, xyz, run)
        self._close_log = []          # CLOSE gate refutes (t, drone, entry, hits, claim dist, claim elev)
        self._gla_log = []            # GLA triggers (t, drone, entry, hits, gate, range)
        self._cenv_log = []           # CLIMB-claim envelope (t, drone, entry, "on" / "off", agl)
        self._b4_n = {"ne_fire": 0, "ne_fire_claimed": 0, "close": 0, "gla_trig": 0, "gla_steps": 0,
                      "gla_skip_rng": 0, "gla_skip_trav": 0, "cenv_on": 0, "cenv_steps": 0}
        # batch 5 diagnostics (F1-F5)
        self._b5_log = []             # (t, drone or -1, fix, ...)
        self._hunt_n = {"hunt_steps": 0, "hunt_err": 0, "hunt_first_t": 0}   # hunt engagement diagnostics
        # batch 6: WPOST-X / band-by-n state and diagnostics (research/ideas/REPORT.md 4.1, 4.3)
        self._wpost = None
        try:
            from team.autopilot import geo_posterior as _GP6
            _GP6.world_restore()          # MAP_PARAMS / WORLD_Q back to the defaults every episode
            _GP6.BAND_N_USED.clear()
            _bn_tab = sorted(k for k, v in _GP6.BAND_N.items() if v)
        except Exception:                                        # noqa: BLE001
            _GP6, _bn_tab = None, []
        try:                                                     # soft city/forest bands (diag only)
            from team.route_replan import pad_posterior as _PP6
            _PP6.BAND_PDF_USED.clear()
            _bp_used = _PP6.BAND_PDF_USED
        except Exception:                                        # noqa: BLE001
            _bp_used = {}
        self._b6_n = {"wp_on": 0, "wp_router": 0, "wp_cem": 0, "wp_err": 0, "wp_m": None, "wp_mean": None,
                      "wp_Wq": None, "wp_Wmax": None, "wp_clamp_r": None, "wp_map_world": None, "wp_world_q": None,
                      "wp_grid_W": None, "wp_router_Wq": None, "wp_setups": 0, "wp_comp_ticks": 0,
                      "wp_found_fields": 0, "wp_drop_events": 0, "wp_dropped_max": 0, "wp_alive_last": None,
                      "bn_table": _bn_tab, "bn_used": (_GP6.BAND_N_USED if _GP6 is not None else {}),
                      "bpdf_used": _bp_used}
        self._b5_n = {"f1_vbox": 0, "f2_skip_clear": 0, "f2_skip_tall": 0, "f2b_block": 0, "f3_split": 0,
                      "f3_offset": 0, "f3_below": 0, "f4_gate": 0, "nfb_trig": 0, "nfb_repick": 0}
        # detv2 T0 diagnostics (FLY_B5_LOG "t0"). T0-a: a_exempt claims handed out despite the close-pair mask,
        # a_blk_hits / a_blk_ticks / a_blk_sep masked entries that failed that test (each entry counted once per reason),
        # a_split_* pad splits on t0a_split_maps (rule a / rule b), a_nearest merges the nearest-entry match sent to
        # another entry than the first match. T0-c: c_weak weak proposals seen, c_wconf WCONF hits added,
        # c_far accepted proposals below score_thr (far threshold), c_snap accepted proposals whose peak was snapped.
        self._wconf_n = 0
        self._t0_n = {"a_exempt": 0, "a_blk_hits": 0, "a_blk_ticks": 0, "a_blk_sep": 0, "a_split_a": 0,
                      "a_split_b": 0, "a_nearest": 0, "c_weak": 0, "c_wconf": 0, "c_far": 0, "c_snap": 0}
        self._t0_log = []             # T0-a claims: (t, drone, entry, hits, ticks, [(partner, sep to its anchor)], dist)
        self._t0a_blk = set()         # T0-a diag: (entry, reason) pairs already counted
        self._mc_n = _MCCounter()     # mtn_claims diagnostics (event counts)

    def _assign_wedges(self, state: np.ndarray) -> None:
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
        _fvlow = False
        if float(cfg.forest_vote_fv_min) > 0.0:
            # F2 (MK): no forest from the depth vote while the XGB voter reads clearly not-forest
            _pf = self._fv_forest_prob()
            _fvlow = _pf is not None and _pf < float(cfg.forest_vote_fv_min)
        _clear_forest = self._clear_hits / float(self._clear_n) < cfg.forest_clear_frac
        if _clear_forest and _fvlow:
            _clear_forest = False
            self._b5_n["f2_skip_clear"] += 1
            self._b5_log.append((round(self.step_i * 0.02, 2), -1, "F2clear",
                                 round(self._clear_hits / float(self._clear_n), 3), round(_pf, 3)))
        if _clear_forest:
            self.is_forest = True
            self.map_kind = "forest"
            self.z_band = (cfg.forest_z_lo, cfg.forest_z_hi)
        else:
            frac = self._tall_hits / float(max(self._clear_n, 1))
            _tall = frac >= FOREST_TALL_OVERRIDE
            if _tall and _fvlow:
                _tall = False
                self._b5_n["f2_skip_tall"] += 1
                self._b5_log.append((round(self.step_i * 0.02, 2), -1, "F2tall", round(frac, 3), round(_pf, 3)))
            if _tall:
                self.is_forest = True
                self.map_kind = "forest"
                self.z_band = (cfg.forest_z_lo, cfg.forest_z_hi)
            else:
                self.map_kind = "city" if frac > cfg.city_tall_frac else "open"
                self._tall_frac = frac
        if self.map_kind == "forest" and float(cfg.forest_veto_pad_z) > 0.0:
            # forest veto from pad evidence: a well-confirmed pad below forest height
            if any(p.hits >= int(cfg.forest_veto_pad_hits) and float(p.xyz[2]) < float(cfg.forest_veto_pad_z)
                   for p in self.pads):
                self._set_flat_kind()
                self._pin_kind()
        _mhbm = self._min_hits_for_map()
        if _mhbm is not None:
            self.min_hits = _mhbm
        if (self.map_kind == "city" and cfg.city_min_hits > 0
                and self._tall_frac > cfg.city_min_hits_frac):
            self.min_hits = cfg.city_min_hits
    def _fv_forest_prob(self):
        """F2: the base XGB ForestVoter's running forest probability, or None (no voter / fewer than 2 samples)."""
        fv = getattr(self, "forest_voter", None)
        if fv is None:
            return None
        try:
            cnt = int(fv.count)
            if cnt < 2:
                return None
            labels = list(fv.labels)
            if "forest" not in labels:
                return None
            return float(fv.psum[labels.index("forest")]) / float(cnt)
        except Exception:                                        # noqa: BLE001
            return None
    def _set_flat_kind(self) -> None:
        """City/open by the tall-fraction rule with the flat pad band (forest veto / z-band rescue)."""
        cfg = self.cfg
        frac = self._tall_hits / float(max(self._clear_n, 1))
        self.is_forest = False
        self.map_kind = "city" if frac > cfg.city_tall_frac else "open"
        self._tall_frac = frac
        self.z_band = (cfg.pad_z_lo, cfg.pad_z_hi)
    def _set_forest_kind(self) -> None:
        cfg = self.cfg
        self.is_forest = True
        self.map_kind = "forest"
        self.z_band = (cfg.forest_z_lo, cfg.forest_z_hi)
    def _pin_kind(self) -> None:
        """A kind set from pad evidence (forest veto / z-band rescue) must not be undone by the base's
        map-classifier override (uid167 ForestVoter forces forest whenever map_kind != forest, up to
        tick 90), so that override is closed for the rest of the episode."""
        fv = getattr(self, "forest_voter", None)
        if fv is not None:
            try:
                fv.done = True
            except Exception:                                    # noqa: BLE001
                pass
    def _forest_possible(self) -> bool:
        """False when geometry rules forest out: a start beyond the forest world (the base's spread
        veto, FOREST_VETO_SPREAD_M 45 m = WORLD forest + 3) or the router's geometric city fix."""
        rr = getattr(self, "replan_router", None)
        if rr is not None and getattr(rr, "_kind_fixed", None) == "city":
            return False
        if self._gkc_route is not None:
            return False                                        # gkc: open ground (terrain mesh) is not forest
        if self._own_start is not None and len(self._own_start):
            lim = float(self.WORLD.get("forest", 42.0)) + 3.0
            if float(np.max(np.abs(np.asarray(self._own_start, float)[:, :2]))) > lim:
                return False
        return True
    def _relock_min_hits(self) -> None:
        """Claim gate after a mid-flight kind switch, as the vote would have set it for that kind."""
        cfg = self.cfg
        _mhbm = self._min_hits_for_map()
        self.min_hits = _mhbm if _mhbm is not None else cfg.min_hits_to_claim
        if (self.map_kind == "city" and cfg.city_min_hits > 0
                and self._tall_frac > cfg.city_min_hits_frac):
            self.min_hits = cfg.city_min_hits
    _WALL_SIN = math.sin(math.radians(25.0))
    def _wall_sample(self, state: np.ndarray, depth: np.ndarray, live: List[int]) -> None:
        """kind_wall_maps: per live drone, the fraction of valid depth pixels (stride 2) whose surface
        normal is within 25 deg of horizontal (vertical faces), folded into a running mean."""
        tn = math.tan(math.radians(45.0))
        for i in live:
            d = np.asarray(depth[i], np.float32)
            if d.ndim == 3:
                d = d[..., 0]
            H, W = d.shape
            d = d[::2, ::2]
            rng = d * 19.5 + 0.5
            valid = (d > 0.0) & (d < 0.999)
            a = (2.0 * (np.arange(0, W, 2) + 0.5) / W - 1.0) * tn
            b = (1.0 - 2.0 * (np.arange(0, H, 2) + 0.5) / H) * tn
            body = np.empty(d.shape + (3,), np.float32)
            body[..., 0] = 1.0
            body[..., 1] = -a[None, :]
            body[..., 2] = b[:, None]
            rpy = np.asarray(state[i, RPY], float)
            R = rot_from_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2]))
            cam = np.asarray(state[i, POS], float) + 0.13 * R[:, 0] + 0.05 * R[:, 2]
            P = cam[None, None, :] + rng[..., None] * (body @ R.T)
            dx = P[:, 1:, :] - P[:, :-1, :]
            dy = P[1:, :, :] - P[:-1, :, :]
            nrm = np.cross(dx[:-1, :, :], dy[:, :-1, :])
            ln = np.linalg.norm(nrm, axis=-1)
            ok = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & (ln > 1e-6)
            jump = np.maximum(np.abs(rng[:-1, 1:] - rng[:-1, :-1]), np.abs(rng[1:, :-1] - rng[:-1, :-1]))
            ok &= jump < 0.5
            nok = int(ok.sum())
            if nok <= 50:
                continue
            nz = np.abs(nrm[..., 2]) / np.maximum(ln, 1e-9)
            self._wall_sum += float((ok & (nz < self._WALL_SIN)).sum()) / float(nok)
            self._wall_n += 1
            self._wall_frac = self._wall_sum / float(self._wall_n)
    def _gkc_sample(self, state: np.ndarray) -> None:
        """gkc: height of the surface under each flying drone (altitude ray), before the band lock."""
        cfg = self.cfg
        lo, hi = float(cfg.gkc_agl[0]), float(cfg.gkc_agl[1])
        S = self.bad[:int(getattr(self, "_n_start_bad", 0) or 0)]
        for i in range(min(self.n, len(self.d))):
            if self.d[i].phase not in (CLIMB, SEARCH, APPROACH):
                continue
            agl = float(state[i, ALT]) * 20.0
            if not (lo <= agl <= hi):
                continue
            p = state[i, POS]
            ds = min((float(np.hypot(float(p[0]) - float(b[0]), float(p[1]) - float(b[1]))) for b in S), default=99.0)
            self._gkc_s.append((int(self.step_i), int(i), float(p[2]) - agl, agl, ds))
    def _gkc_vil_ok(self) -> bool:
        """gkc village rule on the samples away from the start platforms (their tops sit at 0.1-1.1 m on every map):
        >= gkc_min_n of them, none on the z = 0 plane, >= gkc_vil_frac on the slab / tile heights."""
        cfg = self.cfg
        g = np.asarray([s[2] for s in self._gkc_s if s[4] >= float(cfg.gkc_vil_start_r)], float)
        if int(g.size) < int(cfg.gkc_min_n) or bool(np.any(np.abs(g) < float(cfg.gkc_z0))):
            return False
        vb = tuple(float(x) for x in (cfg.gkc_vil_bins or ()))
        nv = 0
        for k in range(0, len(vb) - 1, 2):
            nv += int(np.sum((g >= vb[k]) & (g <= vb[k + 1])))
        if nv < float(cfg.gkc_vil_frac) * int(g.size):
            return False
        if int(cfg.gkc_vil_modes) > 0:
            # the slab (top 0.10) and the road meshes are a few exact heights; terrain is continuous (A open 139:
            # 82 % of its off-platform samples at 0.11-0.19 m, but its 3 commonest 1 cm values hold only 46 %)
            _, cnt = np.unique(np.round(g, 2), return_counts=True)
            top = int(np.sum(np.sort(cnt)[::-1][:int(cfg.gkc_vil_modes)]))
            if top < float(cfg.gkc_vil_frac) * int(g.size):
                return False
        return True
    def _gkc_vil_starts_ok(self) -> bool:
        """C-L2: True unless gkc_vil_start_guard is on and the own starts rule village out (height / start box)."""
        cfg = self.cfg
        if not bool(getattr(cfg, "gkc_vil_start_guard", False)):
            return True
        st = self._own_start
        if st is None or not len(st):
            return True
        S = np.asarray(st, float).reshape(-1, 3)
        ok = True
        if float(cfg.village_start_zmax) < 99.0:
            _n_hi = int(np.sum(S[:, 2] > float(cfg.village_start_zmax)))
            if _n_hi >= max(1, int(getattr(cfg, "village_start_zmax_n", 1))):
                ok = False
        if ok and float(cfg.village_start_box) > 0.0:
            if float(np.max(np.abs(S[:, :2]))) > float(cfg.village_start_box):
                ok = False
        if not ok:
            self.src_counts["gkc_vil_block"] = self.src_counts.get("gkc_vil_block", 0) + 1
        return ok
    def _gkc_decide(self) -> None:
        """gkc: one-directional kind check at the band lock (see the AutopilotConfig comment)."""
        cfg = self.cfg
        self._gkc_done = True
        S = [s for s in self._gkc_s if s[4] >= float(cfg.gkc_start_r)]
        g = np.asarray([s[2] for s in S], float)
        n = int(g.size)
        n0 = int(np.sum(np.abs(g) < float(cfg.gkc_z0))) if n else 0
        nneg = int(np.sum(g < float(cfg.gkc_neg))) if n else 0
        vb = tuple(float(x) for x in (cfg.gkc_vil_bins or ()))
        nvil = 0
        for k in range(0, len(vb) - 1, 2):
            nvil += int(np.sum((g >= vb[k]) & (g <= vb[k + 1]))) if n else 0
        mk = str(self.map_kind)
        rk = str(self._mass_route_kind())
        maps = tuple(cfg.gkc_maps or ())
        dec = None
        dry = bool(cfg.gkc_dry)
        if n >= int(cfg.gkc_min_n) and not self.is_forest and mk in ("city", "open"):
            if dry:
                if "open" in maps and n0 < float(cfg.gkc_f0_max) * n and nneg > float(cfg.gkc_fneg_min) * n:
                    dec = "dry:" + ("open" if rk == "city" else "pin")
                elif "village" in maps and n0 == 0 and self._gkc_vil_ok():
                    dec = "dry:village"
            elif ("open" in maps
                    and n0 < float(cfg.gkc_f0_max) * n and nneg > float(cfg.gkc_fneg_min) * n):
                # route kind open for the episode: also pins the router's evidence kind (a phantom entry beyond the
                # open world's 60 m box would otherwise fix "city" mid-flight: 29 of the 45 c7 open misroutes)
                self._gkc_route = "open"
                _rr = getattr(self, "replan_router", None)
                if _rr is not None and hasattr(_rr, "_kind_fixed"):
                    _rr._kind_fixed = "open"
                dec = "open" if rk == "city" else "pin"
            elif "village" in maps and n0 == 0 and self._gkc_vil_ok() and self._gkc_vil_starts_ok():
                self.is_forest = False
                self.z_band = (cfg.village_z_lo, cfg.village_z_hi)
                self.map_kind = "village"
                if cfg.village_rings:
                    self.rings = cfg.village_rings
                self.min_hits = cfg.village_min_hits
                if self.village_detector is not None:
                    self.detector = self.village_detector
                self._pin_kind()
                dec = "village"
        self.src_counts["gkc"] = [round(self.step_i * 0.02, 2), n, n0, nneg, nvil, mk, rk, dec]
        if cfg.gkc_log:
            self.src_counts["gkc_s"] = [[s[0], s[1], round(s[2], 4), round(s[3], 2), round(min(s[4], 99.0), 1)]
                                        for s in self._gkc_s]
        self._gkc_s = []
    def _gkcc_step(self, state: np.ndarray) -> None:
        """C-L4 gkc_city_pin: sample the altitude-ray ground height; pin the router evidence kind to city (see cfg)."""
        c = self.cfg
        t = self.step_i * 0.02
        if self.n > int(c.gkc_city_max_n) or t > float(c.gkc_city_t_max) or self.is_forest:
            self._gkcc_done = True
            return
        S = self._own_start
        if S is None:
            return
        S = np.asarray(S, float).reshape(-1, 3)
        r0 = float(c.gkc_city_start_r)
        for i in range(min(self.n, len(self.d))):
            if self.d[i].phase not in (CLIMB, SEARCH, APPROACH):
                continue
            agl = float(state[i, ALT]) * 20.0
            if not (0.3 <= agl <= 19.0):
                continue
            p = state[i, POS]
            px, py = float(p[0]), float(p[1])
            if len(S) and float(np.min(np.hypot(S[:, 0] - px, S[:, 1] - py))) < r0:
                continue
            g = float(p[2]) - agl
            if g < float(c.gkc_neg):
                self._gkcc_neg += 1
            elif abs(g) < float(c.gkc_city_z0):
                self._gkcc_cells.add((int(round(px)), int(round(py))))
        if self._gkcc_neg > 0:
            self._gkcc_done = True                  # terrain below the plane: open, never pin
            return
        if not self._band_locked or t < float(c.gkc_city_t_min):
            return
        if c.gkc_maps and not self._gkc_done:
            return                                  # GKC decides first
        rr = getattr(self, "replan_router", None)
        if (rr is None or not hasattr(rr, "_kind_fixed") or rr._kind_fixed is not None
                or self._gkc_route is not None or self.map_kind not in ("city", "open")
                or self._mass_route_kind() != "open"):
            self._gkcc_done = True
            return
        if not any(dd.phase == SEARCH and dd.claim is None for dd in self.d[:self.n]):
            return                                  # no searcher: a stale replan job would survive the grid swap
        if len(self._gkcc_cells) >= int(c.gkc_city_cells):
            k0 = str(self._mass_route_kind())
            rr._kind_fixed = "city"
            rr.found_sig = None
            if hasattr(rr, "job"):
                rr.job = None                       # reviewer note 1: no stale W60 job survives the W75 setup
            self._gkcc_done = True
            rec = [round(t, 2), int(self.step_i), len(self._gkcc_cells), int(self._gkcc_neg), k0,
                   str(self._mass_route_kind())]
            self._b6_n["gkc_city"] = rec
            self.src_counts["gkc_city"] = list(rec)
    def _prv_ground(self, state: np.ndarray, depth: np.ndarray, live: List[int]) -> None:
        """prv: count depth points (world z) on / below the z = 0 plane in the live drones' views."""
        cfg = self.cfg
        tn = math.tan(math.radians(45.0))
        z0, zneg, rmax = float(cfg.prv_z0), float(cfg.prv_neg), float(cfg.prv_rng)
        for i in live:
            d = np.asarray(depth[i], np.float32)
            if d.ndim == 3:
                d = d[..., 0]
            H, W = d.shape
            d = d[::4, ::4]
            rng = d * 19.5 + 0.5
            valid = (d > 0.0) & (d < 0.999) & (rng <= rmax)
            if not bool(valid.any()):
                continue
            b = (1.0 - 2.0 * (np.arange(0, H, 4) + 0.5) / H) * tn
            a = (2.0 * (np.arange(0, W, 4) + 0.5) / W - 1.0) * tn
            rpy = np.asarray(state[i, RPY], float)
            R = rot_from_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2]))
            cam = np.asarray(state[i, POS], float) + 0.13 * R[:, 0] + 0.05 * R[:, 2]
            dz = R[2, 0] - R[2, 1] * a[None, :] + R[2, 2] * b[:, None]     # z of the body ray (1, -a, b)
            z = (cam[2] + rng * dz)[valid]
            low = z < 1.0
            self._prv_g[0] += int(low.sum())
            self._prv_g[1] += int((np.abs(z) < z0).sum())
            self._prv_g[2] += int((z < zneg).sum())
    def _prv_pick(self):
        """prv: provisional route kind from the ground signature and the start geometry, or None."""
        cfg = self.cfg
        st = np.asarray(self._own_start, float) if self._own_start is not None else np.zeros((0, 3))
        if st.size and float(np.max(np.abs(st[:, :2]))) > float(self.WORLD.get("open", 60.0)) + 1.0:
            return "city"
        nl, n0, nn = self._prv_g
        if nl < int(cfg.prv_min_pts):
            return None
        if nn >= float(cfg.prv_open_neg) * nl and n0 < float(cfg.prv_open_plane) * nl:
            return "open"
        if n0 >= float(cfg.prv_city_plane) * nl and nn < float(cfg.prv_city_neg) * nl:
            _fz = bool(st.size) and float(np.min(st[:, 2])) < float(cfg.prv_forest_zmin)
            if (not self._forest_possible()) or _fz:
                return "city"
        return None
    def _prv_tick(self, state: np.ndarray, depth: np.ndarray, live: List[int]) -> None:
        """prv: plan the provisional route before the lock; keep or re-plan it at the lock."""
        cfg = self.cfg
        if self._prv_kind is None:
            if self._prv_off:
                return
            if self._band_locked or self._route_waypoints is not None or self.clue is None or self._own_start is None:
                self._prv_off = True
                return
            self._prv_ground(state, depth, live)
            kind = self._prv_pick()
            if kind is None:
                if self.step_i * 0.02 >= float(cfg.prv_max_t):
                    self._prv_off = True
                    self.src_counts["prv"] = [None, None] + [int(x) for x in self._prv_g]
                return
            if kind not in tuple(cfg.prv_maps or ()):
                self._prv_off = True
                self.src_counts["prv"] = [round(self.step_i * 0.02, 2), "skip:" + kind]
                return
            self._prv_force = kind
            try:
                self._init_learned_route()
            finally:
                self._prv_force = None
            if self._route_waypoints is None:
                self._prv_off = True
                return
            self._prv_kind = kind
            self.src_counts["prv"] = [round(self.step_i * 0.02, 2), kind] + [int(x) for x in self._prv_g]
            return
        if self._prv_relocked or not self._band_locked:
            return
        self._prv_relocked = True
        real = str(self._route_kind())
        keep = real == self._prv_kind
        self.src_counts["prv"] = list(self.src_counts.get("prv", [])) + [round(self.step_i * 0.02, 2), real, keep]
        if not keep:
            self._route_waypoints = None             # re-planned under the locked kind by _init_learned_route
            for dd in self.d:
                dd.route_idx = 0
    def _zband_rescue(self, xyz: np.ndarray, obs=None) -> None:
        """zband_rescue_hits: cluster a proposal the z band rejected; a tight, well-hit cluster at the
        other flat band's height switches the map kind (once per episode) and becomes a pad entry."""
        cfg = self.cfg
        sc = self._merge_score
        if sc is not None and float(sc) < float(cfg.zband_rescue_score):
            return
        _smr = self._start_mask_r()
        if float(cfg.start_mask_dz) > 0.0:
            # S10: same start-height test as _merge
            if self._bad_near(xyz[:2], _smr if _smr > 0.0 else float(cfg.bad_pad_radius), z=float(xyz[2])):
                return
        elif _smr > 0.0:
            if self._bad_near(xyz[:2], _smr):
                return
        else:
            for b in self.bad:
                if float(np.linalg.norm(b - xyz[:2])) < cfg.bad_pad_radius:
                    return
        best, bd = None, None
        for c in self._zb_rej:
            dk = float(np.linalg.norm(c["s"][:2] / c["n"] - xyz[:2]))
            if dk < cfg.same_pad and (bd is None or dk < bd):
                best, bd = c, dk
        if best is None:
            if self._zb_rej and len(self._zb_rej) >= max(1, int(cfg.zband_rescue_max)):
                self._zb_rej.pop(min(range(len(self._zb_rej)),
                                     key=lambda j: (self._zb_rej[j]["n"], self._zb_rej[j]["t"])))
            best = {"s": np.zeros(3, float), "zz": 0.0, "n": 0, "t": self.step_i}
            self._zb_rej.append(best)
        best["s"] = best["s"] + np.asarray(xyz, float)
        best["zz"] += float(xyz[2]) ** 2
        best["n"] += 1
        best["t"] = self.step_i
        n = best["n"]
        if n < int(cfg.zband_rescue_hits) or not self._band_locked or getattr(self, "_zb_switched", False):
            return
        mean = best["s"] / n
        mz = float(mean[2])
        if math.sqrt(max(best["zz"] / n - mz * mz, 0.0)) > float(cfg.zband_rescue_std):
            return
        if self.is_forest and float(cfg.pad_z_lo) <= mz < float(cfg.zband_rescue_flat_z):
            self._set_flat_kind()
        elif (not self.is_forest and self.map_kind in ("city", "open")
              and float(cfg.forest_z_lo) <= mz <= float(cfg.forest_z_hi)
              and self._forest_possible()):
            self._set_forest_kind()
        else:
            return
        self._zb_switched = True
        self._relock_min_hits()
        self._pin_kind()
        self._zb_rej = []
        # the cluster becomes a pad entry. Fold it into the nearest entry within same_pad that sits in
        # the new band (the same pad seen under the flat band before the lock) instead of listing the
        # same pad twice. An entry there at the old band's height is another surface (a phantom under
        # the new kind): averaging with it would put the pad between the two heights, so an unclaimed
        # one is re-used for the cluster and a claimed one is left alone.
        zb = self.z_band
        near, nd, stale, sd = None, None, None, None
        for pad in self.pads:
            dk = float(np.linalg.norm(pad.xyz[:2] - mean[:2]))
            if dk >= cfg.same_pad:
                continue
            if zb[0] <= float(pad.xyz[2]) <= zb[1]:
                if nd is None or dk < nd:
                    near, nd = pad, dk
            elif pad.by is None and not pad.done and (sd is None or dk < sd):
                stale, sd = pad, dk
        if near is not None:
            if near.by is None and not near.done and not near.probed:
                near.xyz = (near.xyz * near.hits + mean * n) / float(near.hits + n)
            near.hits += int(n)
            near.last_seen = self.step_i
            self._note_view(near, obs)
            return
        if stale is not None:
            stale.xyz = mean.copy()
            stale.hits = int(n)
            stale.last_seen = self.step_i
            stale.anchor = None
            stale.applied = False
            stale.near = None
            stale.near_xy = None
            stale.probed = False
            stale.views = []
            self._note_view(stale, obs)
            return
        fresh = _Pad(xyz=mean.copy(), hits=int(n), last_seen=self.step_i)
        self._note_view(fresh, obs)
        self.pads.append(fresh)
    def _clue_alt_off_for_map(self):
        """Height above the clue's z to search at on this map, from ``clue_alt_off_by_map``, or None."""
        pairs = tuple(getattr(self.cfg, "clue_alt_off_by_map", ()) or ())
        kind = "forest" if self.is_forest else self.map_kind
        for k in range(0, len(pairs) - 1, 2):
            if str(pairs[k]) == kind:
                return float(pairs[k + 1])
        return None

    def _search_alt_for_map(self):
        """Search AGL for this map from ``search_alt_by_map``, or None to keep the default."""
        pairs = tuple(getattr(self.cfg, "search_alt_by_map", ()) or ())
        kind = "forest" if self.is_forest else self.map_kind
        for k in range(0, len(pairs) - 1, 2):
            if str(pairs[k]) == kind:
                return float(pairs[k + 1])
        return None

    def _land_hits_for_map(self) -> int:
        """Hits a pad needs before this map lets a drone put its wheels on it."""
        pairs = tuple(getattr(self.cfg, "land_hits_by_map", ()) or ())
        kind = "forest" if self.is_forest else self.map_kind
        for k in range(0, len(pairs) - 1, 2):
            if str(pairs[k]) == kind:
                return int(pairs[k + 1])
        return int(self.cfg.min_hits_to_land)

    def _min_hits_for_map(self):
        bym = tuple(getattr(self.cfg, "min_hits_by_map", ()) or ())
        for k in range(0, len(bym) - 1, 2):
            if str(bym[k]) == self.map_kind:
                return int(bym[k + 1])
        return None
    def _set_z_band(self, state: np.ndarray) -> None:
        cfg = self.cfg
        z0 = np.asarray(state[:, 2], float)
        village = bool(0.40 <= float(np.median(z0)) <= 0.70)
        if village and (float(cfg.village_start_zmin) > 0.0 or float(cfg.village_start_zmax) < 99.0):
            # village start check: every start must sit at village pad height
            _n_hi = int(np.sum(z0 > float(cfg.village_start_zmax)))
            village = (float(np.min(z0)) >= float(cfg.village_start_zmin)
                       and _n_hi < max(1, int(getattr(cfg, "village_start_zmax_n", 1))))
        if village and float(cfg.village_start_box) > 0.0:
            # F1: every start of a village world lies within 40 m
            _ext = float(np.max(np.abs(np.asarray(state[:, POS], float)[:, :2])))
            if _ext > float(cfg.village_start_box):
                village = False
                self._b5_n["f1_vbox"] += 1
                self._b5_log.append((0.0, -1, "F1", round(_ext, 1)))
        _mtn = float(np.median(z0)) > 10.0
        if float(cfg.mtn_clue_z) > 0.0 and self.clue is not None:
            # S8: mountain from the clue height (flat clue z <= 8.35 m); never mountain otherwise
            _cz = float(self.clue[2])
            _mtn = _cz > float(cfg.mtn_clue_z) or (float(np.median(z0)) > 10.0 and _cz > float(cfg.mtn_clue_zmin))
            self._b3_log.append((0.0, -1, "S8", "mountain" if (_mtn and not village) else "flat_or_village",
                                 round(_cz, 2), round(float(np.median(z0)), 2)))
        if float(getattr(cfg, "mtn_zmax_start", 0.0)) > 0.0 and float(np.max(z0)) > float(cfg.mtn_zmax_start):
            # bughunt MZS: a start above every flat-map start height (village <= 1.32) can only be mountain
            _mtn = True
            village = False
        if village:
            self.z_band = (cfg.village_z_lo, cfg.village_z_hi)
            self.map_kind = "village"
            if cfg.village_rings:
                self.rings = cfg.village_rings
            self.min_hits = cfg.village_min_hits
            if self.village_detector is not None:
                self.detector = self.village_detector
            self._band_locked = True
        elif _mtn:
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
            if int(self.cfg.zband_rescue_hits) > 0:             # score only feeds the z-band rescue
                self._merge_score = float(q.score)
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
        if (rng is not None and rng < REFINE_R and self.cfg.cps_maps
                and self._mass_route_kind() in tuple(self.cfg.cps_maps)
                and float(np.linalg.norm(np.asarray(xyz, float)[:2] - pad.xyz[:2])) >= float(self.cfg.cps_r)):
            rng = None                  # phantom_land CPS: another pad's detection must not drag this entry
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
        if kind not in REFINE_MAPS and not getattr(pad, "split", False):
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
    def _note_view(self, pad, obs) -> None:
        """Remember an observer position that is new enough to count as a second look.

        A real pad gets seen from wherever the sweep passes; a false one is usually an
        artefact of one vantage point, so parallax separates them where a raw hit count
        cannot.
        """
        if obs is None:
            return
        o = np.asarray(obs, float)[:2]
        sep = float(self.cfg.view_sep_m)
        for v in pad.views:
            if float(np.linalg.norm(v - o)) < sep:
                return
        if len(pad.views) < 8:
            pad.views.append(o.copy())

    def _same_pad_r(self) -> float:
        """DEV: pad merge / claim-spacing radius for the current map (same_pad_by_map, else same_pad)."""
        _sp = float(self.cfg.same_pad)
        _spm = tuple(getattr(self.cfg, "same_pad_by_map", ()) or ())
        for _k in range(0, len(_spm) - 1, 2):
            if str(_spm[_k]) == ("forest" if self.is_forest else self.map_kind):
                _sp = float(_spm[_k + 1])
                break
        return _sp

    def _twin_block(self, k: int) -> bool:
        """audit_mf TWIN: entry k lies within twin_r of an entry claimed by a drone or done and has < twin_hits hits."""
        c = self.cfg
        m = tuple(getattr(c, "twin_maps", ()) or ())
        if not m or ("forest" if self.is_forest else str(self.map_kind)) not in m:
            return False
        P = self.pads[k]
        if int(P.hits) >= int(c.twin_hits):
            return False
        xy = np.asarray(P.xyz, float)[:2]
        for j, q in enumerate(self.pads):
            if j == k or not (q.by is not None or q.done):
                continue
            if float(np.linalg.norm(np.asarray(q.xyz, float)[:2] - xy)) < float(c.twin_r):
                self.src_counts["twin_block"] = self.src_counts.get("twin_block", 0) + 1
                return True
        return False

    def _split_on(self) -> bool:
        if float(self.cfg.pad_split_r) <= 0.0:
            return False
        _k = self._mass_route_kind()
        return (_k in tuple(self.cfg.pad_split_maps or ())
                or _k in tuple(self.cfg.t0a_split_maps or ()))      # detv2 T0-a: split also on these maps

    def _t0a_split_only(self) -> bool:
        """detv2 T0-a (review): the pad split is on here only because of t0a_split_maps (not pad_split_maps)."""
        if float(self.cfg.pad_split_r) <= 0.0:
            return False
        _k = self._mass_route_kind()
        return (_k in tuple(self.cfg.t0a_split_maps or ())
                and _k not in tuple(self.cfg.pad_split_maps or ()))

    def _t0a_on(self) -> bool:
        """detv2 T0-a: close-pair claim exemption and nearest-entry matching on this map."""
        m = tuple(self.cfg.t0a_maps or ())
        return bool(m) and ("forest" if self.is_forest else str(self.map_kind)) in m

    def _t0a_ok(self, k: int, partners, i: int, dist: float) -> bool:
        """detv2 T0-a: may free entry k be handed out although it sits within the claim-spacing radius of the
        claimed / landed entries in `partners` (indices into self.pads)? Counts the failing test in _t0_n and logs
        an exemption in _t0_log."""
        c = self.cfg
        q = self.pads[k]
        n = self._t0_n

        def _blk(reason):                     # count each (entry, reason) once per episode
            if (k, reason) not in self._t0a_blk:
                self._t0a_blk.add((k, reason))
                n[reason] += 1
            return False
        if int(q.hits) - int(getattr(q, "whits", 0)) < int(c.t0a_min_hits):
            return _blk("a_blk_hits")
        if int(getattr(q, "t0a_ticks", 0)) < int(c.t0a_ticks):
            return _blk("a_blk_ticks")
        xy = np.asarray(q.xyz, float)[:2]
        seps = []
        for j in partners:
            P = self.pads[j]
            ref = P.anchor if P.anchor is not None else P.xyz[:2]
            s = float(np.linalg.norm(xy - np.asarray(ref, float)[:2]))
            if s < float(c.t0a_sep):
                return _blk("a_blk_sep")
            seps.append((int(j), round(s, 2)))
        n["a_exempt"] += 1
        if len(self._t0_log) < 60:
            self._t0_log.append((round(self.step_i * 0.02, 2), int(i), int(k), int(q.hits), int(q.t0a_ticks), seps,
                                 round(float(dist), 1)))
        return True

    def _mc_inc(self, key: str) -> None:
        """mtn_claims diagnostics: count an event (also mirrored into src_counts as mc_<key> for the harness rows)."""
        self._mc_n[key] += 1
        self.src_counts["mc_" + key] = self.src_counts.get("mc_" + key, 0) + 1

    def _start_mask_r(self) -> float:
        """start_mask_r on start_mask_maps (SMASK), else 0.0 (every bad entry uses bad_pad_radius)."""
        r = float(self.cfg.start_mask_r)
        if r <= 0.0:
            return 0.0
        kind = "forest" if self.is_forest else str(self.map_kind)
        return r if kind in tuple(self.cfg.start_mask_maps or ()) else 0.0

    def _bad_near(self, xy, r_start: float, count: bool = False, z: Optional[float] = None) -> bool:
        """SMASK reject test: start entries (index < _n_start_bad) within r_start, refuted pads
        within bad_pad_radius. count: tally proposals the plain bad_pad_radius mask would have hidden.
        z (S10, start_mask_dz > 0): a start entry only rejects a detection at its surface height
        (|z - (start z - 0.121)| < start_mask_dz); refuted pads stay horizontal-only."""
        ns = int(getattr(self, "_n_start_bad", 0))
        rb = float(self.cfg.bad_pad_radius)
        _zt = float(self.cfg.start_mask_dz) if z is not None else 0.0
        if _zt > 0.0 and self.cfg.start_mask_nodz_maps and (
                ("forest" if self.is_forest else str(self.map_kind)) in tuple(self.cfg.start_mask_nodz_maps)):
            _zt = 0.0                                    # mtn_claims MSM: horizontal start mask on this map
        _sz = getattr(self, "_start_z", None) or []
        near_start = False
        z_pass = False
        for k, b in enumerate(self.bad):
            dk = float(np.linalg.norm(b - xy))
            if dk < (r_start if k < ns else rb):
                if k < ns and _zt > 0.0 and k < len(_sz) and abs(float(z) - (float(_sz[k]) - 0.121)) >= _zt:
                    z_pass = True                        # S10: inside the disc but not at start height
                    continue
                if (k < ns and _zt == 0.0 and z is not None and float(self.cfg.start_mask_dz) > 0.0 and k < len(_sz)
                        and abs(float(z) - (float(_sz[k]) - 0.121)) >= float(self.cfg.start_mask_dz)):
                    self._mc_inc("msm_mask")             # MSM diag: S10 would have let this one through
                return True
            if k < ns and dk < rb:
                near_start = True
        if z_pass:
            self._b3_n["s10_pass"] += 1
        if count and near_start:
            self._smask_n += 1
        return False

    def _blr_check(self, xyz) -> None:
        """flat_search3 BLR: count proposals on refuted-pad bad entries; rehabilitate at blr_n distinct ticks."""
        c = self.cfg
        if ("forest" if self.is_forest else str(self.map_kind)) not in tuple(c.blr_maps):
            return
        ns = int(getattr(self, "_n_start_bad", 0))
        xy = np.asarray(xyz, float)[:2]
        best, bd = None, float(c.blr_r)
        for k in range(ns, len(self.bad)):
            dk = float(np.linalg.norm(self.bad[k] - xy))
            if dk <= bd:
                best, bd = k, dk
        if best is None:
            return
        b = self.bad[best]
        key = (round(float(b[0]), 2), round(float(b[1]), 2))
        done_keys = self.__dict__.setdefault("_blr_done", set())
        if key in done_keys:
            return
        cnt = self.__dict__.setdefault("_blr_cnt", {})
        ticks = cnt.setdefault(key, set())
        ticks.add(int(self.step_i))
        self.src_counts["blr_drop"] = int(self.src_counts.get("blr_drop", 0)) + 1
        if len(ticks) < int(c.blr_n):
            return
        del self.bad[best]
        done_keys.add(key)
        pad = None
        for q in self.pads:
            if q.done and q.by is None and float(np.linalg.norm(np.asarray(q.xyz, float)[:2] - b)) < 0.5:
                pad = q
                break
        if pad is not None:
            pad.done = False
            pad.hits = max(int(pad.hits), len(ticks))
            pad.aborts += 1
            pad.last_seen = self.step_i
        self.src_counts["blr_rehab"] = int(self.src_counts.get("blr_rehab", 0)) + 1
        _lg = self.src_counts.setdefault("blr_log", [])
        if isinstance(_lg, list) and len(_lg) < 20:
            _lg.append([round(self.step_i * 0.02, 2), round(float(b[0]), 2), round(float(b[1]), 2), len(ticks),
                        None if pad is None else int(pad.hits)])

    def _merge(self, xyz: np.ndarray, rng=None, view=None, obs=None, frame=None) -> None:
        if view is not None and self._view_gate_on():
            vr, vb = view
            if self.cfg.view_max_range > 0.0 and vr > self.cfg.view_max_range:
                return
            if (self.cfg.view_max_bearing > 0.0
                    and vb > self.cfg.view_max_bearing):
                return
        if self.z_band is not None:
            _zhi = self.z_band[1]
            if (float(self.cfg.pad_z_hi_open) > 0.0
                    and tuple(self.z_band) == (self.cfg.pad_z_lo, self.cfg.pad_z_hi)
                    and self._mass_route_kind() != "city"):
                # S9: flat band on a seed not routed as city: open pads reach 1.29 m
                _zhi = max(float(self.cfg.pad_z_hi), float(self.cfg.pad_z_hi_open))
                if self.z_band[1] < float(xyz[2]) <= _zhi and self.z_band[0] <= float(xyz[2]):
                    self._b3_n["s9_pass"] += 1
            if not (self.z_band[0] <= float(xyz[2]) <= _zhi):
                if int(self.cfg.zband_rescue_hits) > 0:
                    self._zband_rescue(np.asarray(xyz, float), obs)
                return
        if (float(self.cfg.mtn_zfloor) > 0.0 and not self.is_forest and str(self.map_kind) == "mountain"
                and float(xyz[2]) < float(self.cfg.mtn_zfloor)):
            # mtn_claims MZF: the flat visual ground box (z 0.05) seen through a terrain dip, not a pad
            self._mc_inc("zfloor_drop")
            return
        if self.cfg.pad_world_box:
            # S6: no pad lies outside the map's world box
            _wk = "forest" if self.is_forest else str(self.map_kind)
            _box = 40.0 if _wk == "village" else 42.0 if _wk == "forest" else None if _wk == "mountain" else 75.0
            if (_box is not None
                    and max(abs(float(xyz[0])), abs(float(xyz[1]))) > _box + float(self.cfg.pad_world_margin)):
                self._b3_n["s6_drop"] += 1
                return
        if self.cfg.blr_maps and len(self.bad) > int(getattr(self, "_n_start_bad", 0)):
            try:
                self._blr_check(xyz)                     # flat_search3 BLR
            except Exception:                            # noqa: BLE001
                pass
        _smr = self._start_mask_r()
        if float(self.cfg.start_mask_dz) > 0.0:
            # S10: start entries also need the detection at the start surface height
            if self._bad_near(xyz[:2], _smr if _smr > 0.0 else float(self.cfg.bad_pad_radius),
                              count=_smr > 0.0, z=float(xyz[2])):
                return
        elif _smr > 0.0:
            # SMASK: start-platform entries use start_mask_r, refuted pads keep bad_pad_radius
            if self._bad_near(xyz[:2], _smr, count=True):
                return
        else:
            for b in self.bad:
                if float(np.linalg.norm(b - xyz[:2])) < self.cfg.bad_pad_radius:
                    return
        cfg = self.cfg
        split = self._split_on() and rng is not None and float(rng) < float(cfg.pad_split_rng)
        # rule (b) (next to a landed drone) has its own radius and range when pad_split_landed_r > 0
        _lr = float(cfg.pad_split_landed_r)
        r_b = _lr if _lr > 0.0 else float(cfg.pad_split_r)
        split_b = ((self._split_on() and rng is not None and float(rng) < float(cfg.pad_split_landed_rng))
                   if _lr > 0.0 else split)
        if split or split_b:
            hit_k, hit_d = None, None
            for k, pad in enumerate(self.pads):
                dk = float(np.linalg.norm(pad.xyz[:2] - xyz[:2]))
                if dk < self._same_pad_r() and (hit_d is None or dk < hit_d):
                    hit_k, hit_d = k, dk
            _ppr = float(cfg.pad_split_pair_r)
            if (_ppr > 0.0 and hit_k is not None and split and hit_d < float(cfg.pad_split_r)
                    and float(rng) < float(cfg.pad_split_pair_rng)
                    and self._mass_route_kind() in tuple(cfg.pad_split_pair_maps or ())):
                P = self.pads[hit_k]
                fx = frame.get(hit_k) if frame is not None else None
                if (fx is not None and not P.done
                        and float(np.linalg.norm(fx - xyz[:2])) >= _ppr):
                    # PAIR: two proposals of one frame >= pad_split_pair_r apart match this entry: two pads
                    keep, other = np.asarray(fx, float)[:2].copy(), np.asarray(xyz, float)[:2].copy()
                    st = self._refine_state
                    if P.by is not None and st is not None and P.by < st.shape[0]:
                        bxy = np.asarray(st[P.by, :2], float)
                        if float(np.linalg.norm(bxy - other)) < float(np.linalg.norm(bxy - keep)):
                            keep, other = other, keep
                    P.xyz = np.array([keep[0], keep[1], P.xyz[2]], float)
                    P.anchor = None if P.by is None else keep.copy()
                    P.applied = False
                    P.near = None
                    P.near_xy = None
                    P.split = True
                    fresh = _Pad(xyz=np.array([other[0], other[1], float(xyz[2])], float),
                                 last_seen=self.step_i, split=True)
                    self._note_view(fresh, obs)
                    self.pads.append(fresh)
                    frame[hit_k] = keep.copy()
                    frame[len(self.pads) - 1] = other.copy()
                    self._b3_n["pair_split"] = self._b3_n.get("pair_split", 0) + 1
                    return
            if hit_k is not None and split and hit_d >= float(cfg.pad_split_r):
                P = self.pads[hit_k]
                fx = frame.get(hit_k) if frame is not None else None
                if fx is not None and float(np.linalg.norm(fx - xyz[:2])) >= float(cfg.pad_split_r):
                    # (a) two proposals in one frame both match this entry: two pads
                    if P.by is None and not P.done:
                        P.xyz = np.array([fx[0], fx[1], P.xyz[2]], float)
                        P.anchor = None
                        P.applied = False
                    P.near = None
                    P.near_xy = None
                    P.split = True
                    fresh = _Pad(xyz=xyz.copy(), last_seen=self.step_i, split=True)
                    self._note_view(fresh, obs)
                    self.pads.append(fresh)
                    frame[len(self.pads) - 1] = xyz[:2].copy()
                    if cfg.t0a_split_maps and self._mass_route_kind() in tuple(cfg.t0a_split_maps):
                        self._t0_n["a_split_a"] += 1        # detv2 T0-a diag: a split on a t0a_split_maps map
                    return
            if hit_k is not None and split_b and hit_d >= r_b:
                P = self.pads[hit_k]
                st = self._refine_state
                if (P.done and P.by is not None and st is not None and P.by < st.shape[0]
                        and float(np.linalg.norm(np.asarray(st[P.by, :2], float) - xyz[:2])) >= r_b):
                    # (b) a detection well off the drone that already sits on this pad
                    P.split = True
                    fresh = _Pad(xyz=xyz.copy(), last_seen=self.step_i, split=True,
                                 split_r=(_lr if _lr > 0.0 else 0.0))
                    self._note_view(fresh, obs)
                    self.pads.append(fresh)
                    if frame is not None:
                        frame[len(self.pads) - 1] = xyz[:2].copy()
                    if cfg.t0a_split_maps and self._mass_route_kind() in tuple(cfg.t0a_split_maps):
                        self._t0_n["a_split_b"] += 1        # detv2 T0-a diag
                    return
        _cps = bool(self.cfg.cps_maps) and self._mass_route_kind() in tuple(self.cfg.cps_maps)
        if _cps and rng is not None and float(rng) < float(self.cfg.cps_rng):
            # phantom_land CPS: a close-range detection far from its nearest entry is another pad
            _nk, _nd = None, None
            for k_pad, pad in enumerate(self.pads):
                dk = float(np.linalg.norm(pad.xyz[:2] - xyz[:2]))
                if dk < self._same_pad_r() and (_nd is None or dk < _nd):
                    _nk, _nd = k_pad, dk
            if (_nk is not None and _nd >= float(self.cfg.cps_r)
                    and int(self.pads[_nk].hits) >= int(self.cfg.cps_min_hits)):
                fresh = _Pad(xyz=xyz.copy(), last_seen=self.step_i)
                fresh.cps = True
                self._note_view(fresh, obs)
                self.pads.append(fresh)
                if frame is not None:
                    frame[len(self.pads) - 1] = xyz[:2].copy()
                self.src_counts["cps_split"] = self.src_counts.get("cps_split", 0) + 1
                _lg = self.src_counts.setdefault("cps_log", [])
                if isinstance(_lg, list) and len(_lg) < 20:
                    _lg.append([round(self.step_i * 0.02, 2), int(_nk), round(float(_nd), 2), int(self.pads[_nk].hits),
                                round(float(xyz[0]), 2), round(float(xyz[1]), 2), round(float(rng), 1)])
                return
        _nsp = split or split_b or any(q.split for q in self.pads)
        # detv2 T0-a: nearest-entry matching on t0a_maps too (close entries each keep their own pad's detections)
        _t0n = (not _nsp) and bool(cfg.t0a_maps) and self._t0a_on()
        if _nsp or _t0n:
            # nearest entry once entries may sit closer than same_pad (first match otherwise, as before)
            order, best = [], None
            first = None
            for k_pad, pad in enumerate(self.pads):
                dk = float(np.linalg.norm(pad.xyz[:2] - xyz[:2]))
                if dk < self._same_pad_r() and (best is None or dk < best):
                    order, best = [k_pad], dk
                if first is None and dk < self._same_pad_r():     # T0-a diag only (candE: per-map radius)
                    first = k_pad
            if _t0n and order and order[0] != first:
                self._t0_n["a_nearest"] += 1
        elif _cps and any(q.cps for q in self.pads):
            # phantom_land CPS: the first match as before, unless a CPS entry is nearer (only the split pair changes)
            _kf, _df, _kc, _dc = None, None, None, None
            for k_pad, pad in enumerate(self.pads):
                dk = float(np.linalg.norm(pad.xyz[:2] - xyz[:2]))
                if dk >= self._same_pad_r():
                    continue
                if _kf is None:
                    _kf, _df = k_pad, dk
                if pad.cps and (_dc is None or dk < _dc):
                    _kc, _dc = k_pad, dk
            if _kc is not None and _kf is not None and _dc < _df:
                order = [_kc]
            elif _kf is not None:
                order = [_kf]
            else:
                order = []
        else:
            order = range(len(self.pads))
        _blr = self._bl_rec_on()                     # F3: keep the entry's recent proposals
        for k_pad in order:
            pad = self.pads[k_pad]
            if float(np.linalg.norm(pad.xyz[:2] - xyz[:2])) < self._same_pad_r():
                if frame is not None:
                    frame.setdefault(k_pad, xyz[:2].copy())
                w = 1.0 / (pad.hits + 1.0)
                if _blr:
                    self._bl_note(pad, xyz, rng)
                if pad.probed:
                    pad.hits += 1
                    pad.last_seen = self.step_i
                    self._note_view(pad, obs)
                    return
                new = (1.0 - w) * pad.xyz + w * xyz
                if cfg.mtd_maps and self._mtd_on():
                    self._mtd_note(pad, xyz, rng)            # MTD: inverse-variance estimate (bookkeeping)
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
                self._note_view(pad, obs)
                self._refine_note(pad, xyz, rng)
                return
        fresh = _Pad(xyz=xyz.copy(), last_seen=self.step_i)
        if cfg.mtd_maps and self._mtd_on():
            self._mtd_note(fresh, xyz, rng, fresh=True)       # MTD: the entry's first fix
        if _blr:
            self._bl_note(fresh, xyz, rng)
        self._note_view(fresh, obs)
        self.pads.append(fresh)
        if frame is not None:
            frame[len(self.pads) - 1] = xyz[:2].copy()
    def _merge_weak(self, xyz: np.ndarray) -> None:
        """detv2 WCONF: a weak proposal adds one hit to the nearest unclaimed low-hit entry within wconf_r (see cfg)."""
        c = self.cfg
        m = tuple(c.wconf_maps or ())
        if not m or ("forest" if self.is_forest else str(self.map_kind)) not in m:
            return
        if self.z_band is not None:
            _zhi = self.z_band[1]
            if (float(c.pad_z_hi_open) > 0.0 and tuple(self.z_band) == (c.pad_z_lo, c.pad_z_hi)
                    and self._mass_route_kind() != "city"):
                _zhi = max(float(c.pad_z_hi), float(c.pad_z_hi_open))
            if not (self.z_band[0] <= float(xyz[2]) <= _zhi):
                return
        if c.pad_world_box:
            _wk = "forest" if self.is_forest else str(self.map_kind)
            _box = 40.0 if _wk == "village" else 42.0 if _wk == "forest" else None if _wk == "mountain" else 75.0
            if _box is not None and max(abs(float(xyz[0])), abs(float(xyz[1]))) > _box + float(c.pad_world_margin):
                return
        _smr = self._start_mask_r()
        if float(c.start_mask_dz) > 0.0:
            if self._bad_near(xyz[:2], _smr if _smr > 0.0 else float(c.bad_pad_radius), z=float(xyz[2])):
                return
        elif _smr > 0.0:
            if self._bad_near(xyz[:2], _smr):
                return
        elif any(float(np.linalg.norm(b - xyz[:2])) < c.bad_pad_radius for b in self.bad):
            return
        best, bd = None, float(c.wconf_r)
        _wmh = int(c.wconf_min_hits)
        _wbm = tuple(getattr(c, "wconf_min_hits_by_map", ()) or ())
        if _wbm:                                    # detv2 stage 2: per-map minimum detector hits
            _wk = "forest" if self.is_forest else str(self.map_kind)
            for _j in range(0, len(_wbm) - 1, 2):
                if str(_wbm[_j]) == _wk:
                    _wmh = int(_wbm[_j + 1])
                    break
        for pad in self.pads:
            if pad.done or pad.by is not None or pad.hits < 1:
                continue
            if _wmh > 1 and int(pad.hits) - int(getattr(pad, "whits", 0)) < _wmh:
                continue
            d = float(np.linalg.norm(pad.xyz[:2] - xyz[:2]))
            if d <= bd:
                best, bd = pad, d
        if best is None:
            return
        n = int(getattr(best, "whits", 0))
        if n >= int(c.wconf_max) or best.hits >= self._claim_gate(best):
            return
        best.whits = n + 1
        best.hits += 1
        best.last_seen = self.step_i
        self._wconf_n = getattr(self, "_wconf_n", 0) + 1
        self._t0_n["c_wconf"] += 1

    def _assign(self, state: np.ndarray) -> None:
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
            gate = max(1, int(self.cfg.commit_floor), int(self.cfg.claim_hits_late))
        need_views = int(self.cfg.claim_min_views)
        free = [k for k, q in enumerate(self.pads)
                if q.by is None and not q.done and q.hits >= gate
                and len(q.views) >= need_views
                and not (q.split and q.hits < int(self.cfg.pad_split_min_hits))]
        if free and gate < int(self.min_hits) and (self.cfg.rg_cne_maps or int(self.cfg.rg_cab_max) > 0):
            free = self._rg_commit_filter(free)
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
        if self.cfg.assign_reach:
            _t_left = 60.0 - self.step_i / 50.0
            _budget = max(0.0, _t_left - float(self.cfg.assign_land_sec)) * float(self.cfg.assign_v_eff)
            pairs = [pr for pr in pairs if pr[0] <= _budget]
        if ((float(self.cfg.far_claim_d) > 0.0 or float(self.cfg.mtn_reach_v) > 0.0)
                and not self.is_forest and str(self.map_kind) == "mountain" and pairs):
            # mtn_claims FAR / REACH: no long trip on thin evidence, no trip that cannot end in time
            _fd, _fh = float(self.cfg.far_claim_d), int(self.cfg.far_claim_hits)
            _rv = float(self.cfg.mtn_reach_v)
            _tl = 60.0 - self.step_i / 50.0
            _keep = []
            for pr in pairs:
                if _fd > 0.0 and pr[0] > _fd and int(self.pads[pr[2]].hits) < _fh:
                    self._mc_inc("far_skip")
                    continue
                if _rv > 0.0 and pr[0] / _rv + float(self.cfg.mtn_reach_land_s) > _tl:
                    self._mc_inc("reach_skip")
                    continue
                _keep.append(pr)
            pairs = _keep

        _spk_done = int(getattr(self.cfg, "spoken_skip_done", 0))   # DEV: a done pad no longer blocks its neighbours
        # candE merge: on a map whose split comes only from t0a_split_maps (mountain) landed partners keep masking,
        # so a split entry next to a landed pad still goes through _t0a_ok (T0 review fix). With t0a_split_maps
        # empty (UID 10) this is spoken_skip_done unchanged.
        _skip_done = bool(_spk_done) and not self._t0a_split_only()
        _spk_d = [d for d in self.d if d.claim is not None and self.pads[d.claim].by is not None
                  and not (_skip_done and self.pads[d.claim].done)]
        spoken = [self.pads[d.claim].xyz[:2] for d in _spk_d]
        # detv2 T0-a: the claimed / landed entry behind each spoken xy (built from the same list as spoken, so the
        # two stay aligned whatever _assign_par / _reopt_claims do below)
        spoken_k = [d.claim for d in _spk_d]
        if str(cfg_assign := getattr(self.cfg, "assign_mode", "")) == "par":
            if self._assign_par(state, cand, free, spoken):
                return
        _ = cfg_assign
        if self.cfg.assign_reopt:
            self._reopt_claims(state, cand)
        taken_i = set()
        _hof_new = [] if self.cfg.hof_maps else None
        # detv2 T0-a: the exemption per (entry, partners)
        _t0a = bool(self.cfg.t0a_maps) and self._t0a_on()
        # T0 review: on a map whose split comes only from t0a_split_maps (mountain), a split entry keeps the full
        # same_pad mask, so it is handed out next to a claimed / landed partner only through the T0-a test (20 hits,
        # 10 ticks, 1.8 m from the anchor), not through the village/forest split route (6 hits, 1.5 m)
        _t0sg = bool(self.cfg.t0a_split_maps) and self._t0a_split_only()
        if _t0a:
            _t0a_memo = {}
        for _dist, i, k in pairs:
            if i in taken_i or self.pads[k].by is not None or self.pads[k].done:
                continue
            xy = self.pads[k].xyz[:2]
            _rad = (self.pads[k].split_r or float(self.cfg.pad_split_r)) if self.pads[k].split else self._same_pad_r()
            if _t0sg:
                _rad = self._same_pad_r()      # == cfg.same_pad on the t0a split-only map (mountain)
            if any(float(np.linalg.norm(xy - s)) < _rad for s in spoken):
                if not _t0a:
                    continue
                _blk = tuple(spoken_k[j] for j, s in enumerate(spoken) if float(np.linalg.norm(xy - s)) < _rad)
                _key = (k, _blk)
                if _key not in _t0a_memo:
                    _t0a_memo[_key] = self._t0a_ok(k, _blk, i, _dist)
                if not _t0a_memo[_key]:
                    continue
            if self.cfg.twin_maps and self._twin_block(k):
                continue
            self.d[i].claim = k
            self.pads[k].by = i
            self.d[i].phase = APPROACH
            self.d[i].dsc_gated = False
            self.d[i].claim_dist = float(_dist)
            self._claim_note(i, k, float(_dist), state)
            taken_i.add(i)
            spoken.append(xy)
            spoken_k.append(k)
            if _hof_new is not None:
                _hof_new.append((i, k, float(_dist)))
        if _hof_new:
            self._hof_apply(state, _hof_new)
        if self.cfg.assign_reach or self.cfg.assign_swap:
            self._assign_review(state, gate)

    def _hof_apply(self, state, new_claims) -> None:
        """search_time HOF: swap a far new claim with a closer approaching drone (see hof_* in the cfg)."""
        c = self.cfg
        _key = "forest" if self.is_forest else str(self._mass_route_kind())
        if _key not in tuple(c.hof_maps):
            return
        for i, k, dist in new_claims:
            if dist <= float(c.hof_min_d):
                continue
            if self.d[i].claim != k or int(self.pads[k].hits) < int(c.hof_min_hits):
                continue
            Q = np.asarray(self.pads[k].xyz, float)[:2]
            xi = np.asarray(state[i, POS], float)[:2]
            best = None
            for j in range(self.n):
                dj = self.d[j]
                if j == i or dj.phase != APPROACH or dj.claim is None or dj.claim == k or dj.commit_sink:
                    continue
                p = dj.claim
                qp = self.pads[p]
                if qp.done or qp.split or frozenset((p, k)) in self._2opt_done:
                    continue
                P = np.asarray(qp.xyz, float)[:2]
                xj = np.asarray(state[j, POS], float)[:2]
                a = float(np.linalg.norm(P - xj))
                b = float(np.linalg.norm(Q - xj))
                e = float(np.linalg.norm(P - xi))
                g = (a + dist) - (b + e)
                if g > float(c.hof_margin) and (best is None or g > best[0]):
                    best = (g, j, p, b, e)
            if best is None:
                continue
            g, j, p, b, e = best
            di, dj = self.d[i], self.d[j]
            di.claim, dj.claim = p, k
            self.pads[p].by, self.pads[k].by = i, j
            di.claim_dist, dj.claim_dist = float(e), float(b)
            self._claim_note(i, p, float(e), state)
            self._claim_note(j, k, float(b), state)
            di.dsc_gated = dj.dsc_gated = False
            di.apx_key = dj.apx_key = -1
            self._2opt_done.add(frozenset((p, k)))
            self.src_counts["hof"] = self.src_counts.get("hof", 0) + 1

    def _rg_commit_filter(self, free):
        """CNE: after the late commit, entries below the normal claim gate that 238's NE would have zeroed (shadow)
        or that were aborted rg_cab_max times are not handed out."""
        c = self.cfg
        _cne = bool(c.rg_cne_maps) and not self.is_forest and str(self.map_kind) in tuple(c.rg_cne_maps)
        _cab = int(c.rg_cab_max)
        keep = []
        for k in free:
            q = self.pads[k]
            if int(q.hits) < int(self.min_hits):
                if _cne and bool(getattr(q, "rg_z", False)):
                    self.src_counts["rg_cne_skip"] = self.src_counts.get("rg_cne_skip", 0) + 1
                    continue
                if _cab > 0 and int(q.aborts) >= _cab:
                    self.src_counts["rg_cab_skip"] = self.src_counts.get("rg_cab_skip", 0) + 1
                    continue
            keep.append(k)
        return keep

    def _assign_review(self, state, gate: int) -> None:
        """Approaching drones: drop a claim that can no longer be reached in time, and
        switch to a free confirmed pad that is much closer than the current one."""
        c = self.cfg
        t_left = 60.0 - self.step_i / 50.0
        v_eff = max(float(c.assign_v_eff), 0.1)
        for i in range(self.n):
            dd = self.d[i]
            if dd.phase != APPROACH or dd.claim is None:
                continue
            pos = np.asarray(state[i, POS], float)[:2]
            cur = self.pads[dd.claim]
            dcur = float(np.linalg.norm(np.asarray(cur.xyz, float)[:2] - pos))
            if c.assign_reach and (dcur / v_eff + float(c.assign_land_sec) > t_left + 3.0) and t_left > 6.0:
                cur.by = None
                dd.claim = None
                dd.dsc_gated = False
                dd.phase = SEARCH
                continue
            if not c.assign_swap or dcur <= float(c.assign_swap_min_dist):
                continue
            best = None
            for k, q in enumerate(self.pads):
                if q.by is not None or q.done or q.hits < gate or k == dd.claim:
                    continue
                if q.split and q.hits < int(c.pad_split_min_hits):
                    continue
                dk = float(np.linalg.norm(np.asarray(q.xyz, float)[:2] - pos))
                if c.twin_maps and dk < float(c.assign_swap_ratio) * dcur and self._twin_block(k):
                    continue
                if dk < float(c.assign_swap_ratio) * dcur and (best is None or dk < best[0]):
                    best = (dk, k)
            if best is not None:
                cur.by = None
                dd.claim = best[1]
                self.pads[best[1]].by = i
                dd.claim_dist = float(best[0])
                dd.dsc_gated = False
                self._claim_note(i, best[1], float(best[0]), state)

    def _claim_note(self, i: int, k: int, dist: float, state) -> None:
        """S1 bookkeeping at every claim / hand-over: restart the in-view counter and store the claim
        elevation (degrees of the pad below the drone at the claim distance)."""
        dd = self.d[i]
        dd.gate_inview = 0
        dd.dsk_key = -1                     # candHb DSK: a new claim instance starts unlatched (read only by DSK code)
        dd.note_dist = float(dist)          # batch 4 CLOSE gate: claim distance on every claim path
        h = float(state[i, 2]) - float(self.pads[k].xyz[2])
        dd.claim_elev = float(math.degrees(math.atan2(h, max(float(dist), 1e-3))))

    def _reopt_claims(self, state: np.ndarray, cand) -> None:
        """Let a searcher take over a pad whose claimant is much farther from it.

        Claims are first-come: the drone that spotted a pad flies to it even when a
        teammate is already almost on top of it. Handing the pad to the nearer drone
        saves the difference in flight time for one landing and returns the far drone
        to searching where it already is. Each pad swaps at most once.
        """
        gain = float(self.cfg.assign_reopt_gain)
        for i in cand:
            if self.d[i].claim is not None:
                continue
            pos = np.asarray(state[i, POS], float)[:2]
            best = None
            for k, pad in enumerate(self.pads):
                j = pad.by
                if j is None or pad.done or pad.aborts > 0 or j == i:
                    continue
                if self.d[j].phase != APPROACH or self.d[j].commit_sink:
                    continue
                dj = float(np.linalg.norm(np.asarray(state[j, POS], float)[:2] - pad.xyz[:2]))
                di = float(np.linalg.norm(pos - pad.xyz[:2]))
                if dj - di >= gain and (best is None or dj - di > best[0]):
                    best = (dj - di, k, j)
            if best is None:
                continue
            _, k, j = best
            pad = self.pads[k]
            pad.aborts += 1                   # marks the pad as already re-assigned once
            self.d[j].claim = None
            self.d[j].commit_sink = False
            self.d[j].phase = SEARCH
            self.d[i].claim = k
            pad.by = i
            self.d[i].phase = APPROACH
            self._claim_note(i, k, float(np.linalg.norm(pos - pad.xyz[:2])), state)

    def _assign_par(self, state: np.ndarray, cand, free, spoken) -> bool:
        """Hand out pads to maximise the fleet's *time* score, not total travel.

        The scorer measures each drone against a par fixed by its own start (the pad
        nearest where it took off), so a second of detour costs a drone with a close
        pad far more than one whose par is already long. Minimising raw distance
        ignores that; this weighs each drone's delay by its own 1/(horizon - par).
        """
        if not cand or not free:
            return False
        try:
            from scipy.optimize import linear_sum_assignment
        except Exception:
            return False
        n = max(int(self.n), 1)
        known = [q.xyz for q in self.pads if q.hits >= 1]
        if not known or self._own_start is None:
            return False
        K = np.asarray(known, float)[:, :2]
        S = np.asarray(self._own_start, float)
        now = self.step_i * 0.02
        v = max(float(self.cfg.assign_speed), 0.1)
        pars = []
        for i in cand:
            d = float(np.min(np.linalg.norm(K - S[i, :2], axis=1))) if i < len(S) else 0.0
            pars.append(1.06 * (d / 3.0) + float(n - 1))
        cost = np.zeros((len(cand), len(free)), dtype=float)
        for a, i in enumerate(cand):
            pos = np.asarray(state[i, POS], float)
            par = pars[a]
            span = max(60.0 - par, 1e-6)
            for b, k in enumerate(free):
                pad = self.pads[k]
                if (FIX_BELOW and self._mass_route_kind() in FIX_BOX_KINDS
                        and float(state[i, 2]) < float(pad.xyz[2]) - 2.0):
                    cost[a, b] = 10.0
                    continue
                t = now + float(np.linalg.norm(pad.xyz[:2] - pos[:2])) / v \
                    + float(self.cfg.assign_land_sec)
                cost[a, b] = -1.0 if t <= par else -float(np.clip(1.0 - (t - par) / span, 0.0, 1.0))
        rows, cols = linear_sum_assignment(cost)
        placed = False
        for a, b in zip(rows, cols):
            if cost[a, b] >= 10.0:
                continue
            i, k = cand[a], free[b]
            if self.pads[k].by is not None or self.pads[k].done:
                continue
            xy = self.pads[k].xyz[:2]
            _rad = (self.pads[k].split_r or float(self.cfg.pad_split_r)) if self.pads[k].split else self._same_pad_r()
            if any(float(np.linalg.norm(xy - sp)) < _rad for sp in spoken):
                continue
            self.d[i].claim = k
            self.pads[k].by = i
            self.d[i].phase = APPROACH
            self._claim_note(i, k, float(np.linalg.norm(np.asarray(state[i, POS], float)[:2] - xy)), state)
            spoken.append(xy)
            placed = True
        return placed

    def _assign_2opt(self, state: np.ndarray) -> None:
        """assign_2opt_m: swap the claims of two approaching drones when that shortens their summed
        run-in by more than assign_2opt_m metres after a turn charge (lateness P1). Runs after
        _assign, which returns early whenever no pad is free."""
        c = self.cfg
        _m = tuple(c.assign_2opt_maps or ())
        if _m and ("forest" if self.is_forest else str(self.map_kind)) not in _m:
            return
        A = []
        for i in range(self.n):
            dd = self.d[i]
            if dd.phase != APPROACH or dd.claim is None or dd.commit_sink:
                continue
            q = self.pads[dd.claim]
            if q.done or q.split or int(q.hits) < int(c.assign_2opt_min_hits):
                continue
            A.append(i)
        if len(A) < 2:
            return
        v, ts, min_d = float(c.assign_2opt_v), float(c.assign_2opt_turn_s), float(c.assign_2opt_min_d)

        def _pen(i, xy, cur, new):
            # turn charge (m): angle between the heading (horizontal velocity above 0.5 m/s, else
            # the direction to the current pad) and the direction to the new pad
            vel = np.asarray(state[i, VEL], float)[:2]
            h = vel if float(np.linalg.norm(vel)) > 0.5 else (cur - xy)
            w = new - xy
            nh, nw = float(np.linalg.norm(h)), float(np.linalg.norm(w))
            if nh < 1e-6 or nw < 1e-6:
                return 0.0
            ca = float(np.clip(float(h @ w) / (nh * nw), -1.0, 1.0))
            return v * ts * (1.0 - ca) * 0.5

        best = None
        for a in range(len(A)):
            for b in range(a + 1, len(A)):
                i, j = A[a], A[b]
                p, q = self.d[i].claim, self.d[j].claim
                if p == q or frozenset((p, q)) in self._2opt_done:
                    continue
                xi = np.asarray(state[i, POS], float)[:2]
                xj = np.asarray(state[j, POS], float)[:2]
                P = np.asarray(self.pads[p].xyz, float)[:2]
                Q = np.asarray(self.pads[q].xyz, float)[:2]
                dip, djq = float(np.linalg.norm(P - xi)), float(np.linalg.norm(Q - xj))
                if min(dip, djq) < min_d:
                    continue
                diq, djp = float(np.linalg.norm(Q - xi)), float(np.linalg.norm(P - xj))
                g = dip + djq - diq - djp - _pen(i, xi, P, Q) - _pen(j, xj, Q, P)
                if g > float(c.assign_2opt_m) and (best is None or g > best[0]):
                    best = (g, i, j, p, q, diq, djp)
        if best is None:
            return
        g, i, j, p, q, diq, djp = best
        di, dj = self.d[i], self.d[j]
        di.claim, dj.claim = q, p
        self.pads[q].by, self.pads[p].by = i, j
        di.claim_dist, dj.claim_dist = float(diq), float(djp)
        self._claim_note(i, q, float(diq), state)
        self._claim_note(j, p, float(djp), state)
        di.dsc_gated = dj.dsc_gated = False
        di.apx_key = dj.apx_key = -1
        self._2opt_done.add(frozenset((p, q)))
        self._2opt_log.append((round(self.step_i * 0.02, 2), i, j, int(p), int(q), round(float(g), 2)))

    def _lsv(self, name: str):
        """LANDSEG: the per-map override of landing parameter ``name`` (cfg.landseg[kind][name]), else None."""
        t = self.cfg.landseg
        if t:
            m = t.get("forest" if self.is_forest else str(self.map_kind))
            if m:
                return m.get(name)
        return None

    def _lp(self, name: str):
        """LANDSEG: landing parameter ``name`` for the current map kind (the cfg.landseg override, else cfg)."""
        t = self.cfg.landseg
        if t:
            m = t.get("forest" if self.is_forest else str(self.map_kind))
            if m:
                v = m.get(name)
                if v is not None:
                    return v
        return getattr(self.cfg, name)

    def _endgame_hurry(self, dist: float, above: float) -> bool:
        maps = tuple(getattr(self.cfg, "hurry_maps", ()) or ()) or HURRY_MAPS
        if self._mass_route_kind() not in maps:
            return False
        t_left = 60.0 - self.step_i / 50.0
        if self.cfg.hurry_real:
            # S2: budget the descent at the real sink rate (sink above the flare height, then descend_speed)
            c = self.cfg
            _lp = self._lp
            fl = float(_lp("forest_flare_alt") if self.is_forest else _lp("flare_alt"))
            sk = float(_lp("forest_sink_speed") if self.is_forest else _lp("sink_speed"))
            ab = max(0.0, float(above))
            need = (dist / 1.5 + max(0.0, ab - fl) / max(sk, 0.1) + min(ab, fl) / max(float(_lp("descend_speed")), 0.1)
                    + 0.5 + float(c.hurry_margin))
            return t_left < need
        need = dist / 1.5 + max(0.0, above) / 0.45 + 3.0
        return t_left < need
    def _pad_xy_dist(self, dd) -> float:
        pad = self.pads[dd.claim]
        if self.cfg.tdo_maps and self._tdo_active(dd):
            return self._tdo_pad_dist(dd, pad, dd.prev_xy)            # TDO: the nearer of centre / offset point
        return float(np.linalg.norm(np.asarray(pad.xyz, float)[:2] - dd.prev_xy))
    def _view_gate_on(self) -> bool:
        if self.cfg.view_max_bearing <= 0.0 and self.cfg.view_max_range <= 0.0:
            return False
        m = tuple(getattr(self.cfg, "view_gate_maps", ()) or ())
        return (not m) or (self.map_kind in m)
    def _pvr_on(self) -> bool:
        """PVR (phantom_land): this seed's kind ("forest" if is_forest else the route kind) is in pvr_maps."""
        m = tuple(self.cfg.pvr_maps or ())
        return bool(m) and ("forest" if self.is_forest else str(self._mass_route_kind())) in m

    def _pvr_note(self, dd, pad, pos, agl) -> None:
        """PVR: reset on a new claim; store the surface point under the drone near the claimed entry."""
        k = int(dd.claim) if dd.claim is not None else -1
        if dd.pvr_key != k:
            dd.pvr_key, dd.pvr_rs, dd.pvr_ok, dd.pvr_n, dd.pvr_nt = k, [], False, 0, -1
        if k < 0 or dd.pvr_ok or not (0.05 < agl < 19.9) or float(pos[2]) - float(pad.xyz[2]) < 0.3:
            return
        pxy = np.asarray(pad.xyz, float)[:2]
        if float(np.hypot(float(pos[0]) - pxy[0], float(pos[1]) - pxy[1])) > float(self.cfg.pvr_samp_r):
            return
        if dd.pvr_rs:
            lx, ly, _lz = dd.pvr_rs[-1]
            if math.hypot(float(pos[0]) - lx, float(pos[1]) - ly) < 0.01:
                return
        dd.pvr_rs.append((float(pos[0]), float(pos[1]), float(pos[2]) - float(agl)))
        if len(dd.pvr_rs) > 400:
            del dd.pvr_rs[0]

    def _pvr_proof(self, S, pad) -> bool:
        """PVR: a locally flat window (2 * pvr_ztol) of >= pvr_plat_n samples spanning >= pvr_plat_ext m within
        pvr_zband of the estimate's z and pvr_rad m of its xy, with a sequence neighbour within 0.12 m that reads
        >= pvr_rim_step lower (a raised platform edge; the TDO rim rule with its own parameters)."""
        c = self.cfg
        if len(S) < int(c.pvr_plat_n):
            return False
        pc = np.asarray(pad.xyz, float)
        fl = np.zeros(len(S), bool)
        dxy = np.hypot(np.diff(S[:, 0]), np.diff(S[:, 1]))
        pr = (np.abs(np.diff(S[:, 2])) <= 0.02 * dxy + float(c.pvr_flat_tol)) & (dxy <= 0.15)
        fl[:-1] |= pr
        fl[1:] |= pr
        m = (fl & (np.abs(S[:, 2] - pc[2]) <= float(c.pvr_zband))
             & (np.hypot(S[:, 0] - pc[0], S[:, 1] - pc[1]) <= float(c.pvr_rad)))
        idx = np.nonzero(m)[0]
        if len(idx) < int(c.pvr_plat_n):
            return False
        order = idx[np.argsort(-S[idx, 2])]
        z = S[order, 2]
        w = 2.0 * float(c.pvr_ztol)
        step = float(c.pvr_rim_step)
        tried = set()
        for a in range(len(order)):
            zt = float(z[a])
            key = round(zt, 3)
            if key in tried:
                continue
            tried.add(key)
            win = order[(z <= zt + 1e-9) & (z >= zt - w)]
            if len(win) < int(c.pvr_plat_n):
                continue
            W = S[win]
            if float(np.hypot(W[:, 0].max() - W[:, 0].min(), W[:, 1].max() - W[:, 1].min())) < float(c.pvr_plat_ext):
                continue
            for k in win:
                for j in (k - 1, k + 1):
                    if (0 <= j < len(S) and S[j, 2] <= S[k, 2] - step
                            and math.hypot(S[j, 0] - S[k, 0], S[j, 1] - S[k, 1]) <= 0.12):
                        return True
        return False

    def _pvr_descend(self, i, dd, pad, pos, agl, dist, v_des):
        """PVR in DESCEND: limit / hold the sink until the platform is proven; refute after pvr_hold_s held.
        Returns the (possibly changed) v_des, or None when the claim was refuted."""
        c = self.cfg
        self._pvr_note(dd, pad, pos, agl)
        if dd.pvr_ok or int(pad.hits) >= int(c.pvr_max_hits):
            return v_des
        if len(dd.pvr_rs) != dd.pvr_nt:
            dd.pvr_nt = len(dd.pvr_rs)
            try:
                dd.pvr_ok = self._pvr_proof(np.asarray(dd.pvr_rs, float).reshape(-1, 3), pad)
            except Exception:                            # noqa: BLE001
                dd.pvr_ok = False
            if dd.pvr_ok:
                self.src_counts["pvr_ok"] = self.src_counts.get("pvr_ok", 0) + 1
                return v_des
        ab = float(pos[2]) - float(pad.xyz[2])
        hab = float(c.pvr_hold_ab)
        lim = math.sqrt(max(0.0, 2.0 * float(c.pvr_acc) * (ab - hab)))
        v = np.asarray(v_des, float).copy()
        if float(v[2]) < -lim:
            v[2] = -lim
            self.src_counts["pvr_lim"] = self.src_counts.get("pvr_lim", 0) + 1
        if ab < hab:
            v[2] = max(float(v[2]), min(1.0, 1.5 * (hab - ab)))
        if ab < hab + 0.3:
            dd.pvr_n += 1
            if dd.pvr_n > int(float(c.pvr_hold_s) * 50.0):
                _lg = self.src_counts.setdefault("pvr_log", [])
                if isinstance(_lg, list) and len(_lg) < 40:
                    _lg.append([round(self.step_i * 0.02, 2), i, int(dd.claim), int(pad.hits), round(ab, 2),
                                round(float(dist), 2), len(dd.pvr_rs)])
                self.src_counts["pvr_fire"] = self.src_counts.get("pvr_fire", 0) + 1
                self._refute(i, soft=bool(c.pvr_soft))
                dd.pvr_key = -1
                return None
        return v

    def _refute(self, i: int, soft: bool = False, never_hard: bool = False) -> None:
        dd = self.d[i]
        if bool(getattr(self.cfg, "rg_diag", False)):
            import sys as _sys
            _rl = self.src_counts.setdefault("rg_ref", [])
            if len(_rl) < 300:
                _k = dd.claim
                _p = self.pads[_k] if _k is not None and _k < len(self.pads) else None
                _rl.append([round(self.step_i * 0.02, 2), int(i), _k, _sys._getframe(1).f_code.co_name,
                            bool(soft), bool(never_hard), None if _p is None else int(_p.hits),
                            None if _p is None else int(_p.aborts)])
        if dd.claim is not None:
            pad = self.pads[dd.claim]
            if soft and (never_hard or pad.aborts < int(self.cfg.soft_abort_max)):
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

    def _esk_on(self) -> bool:
        """ESK (time_term): this seed's kind ("forest" if is_forest else the route kind) is in esk_maps."""
        m = tuple(self.cfg.esk_maps or ())
        return bool(m) and ("forest" if self.is_forest else str(self._mass_route_kind())) in m

    def _hb_inc(self, key: str, n: int = 1) -> None:
        """candHb diagnostics: counters in _b6_n (FLY_B5_LOG "b6"), keys hb_*."""
        self._b6_n["hb_" + key] = self._b6_n.get("hb_" + key, 0) + n

    def _dsk_cfg(self):
        """candHb DSK: this seed's dsk_by_map entry ("forest" if is_forest else the route kind), or None."""
        t = self.cfg.dsk_by_map
        if not t:
            return None
        m = t.get("forest" if self.is_forest else str(self._mass_route_kind()))
        return m if m else None

    def _dsk_floor(self, pos, pad_xy, agl: float, dk: dict) -> float:
        """candHb DSK "clear": the lowest height the early sink may aim at = the highest known surface under the drone
        (downward ray) and along the straight path to the pad (fleet LAV max-height grid, lav_cell cells in a
        +-half_w corridor; cells within excl_r of the pad centre, i.e. the pad and its rim, skipped) + clear.
        -1e9 when nothing is known (never lowers the target on its own)."""
        top = -1e9
        if 0.05 < float(agl) < 19.9:
            top = float(pos[2]) - float(agl)
        g = self._lav_top
        if g is not None:
            dx = float(pad_xy[0]) - float(pos[0])
            dy = float(pad_xy[1]) - float(pos[1])
            L = math.hypot(dx, dy)
            ex = float(dk.get("excl_r", 1.2))
            if L > ex + 1e-6:
                ux, uy = dx / L, dy / L
                cell = float(self.cfg.lav_cell)
                H = self._LAV_HALF
                hw = float(dk.get("half_w", 1.0))
                xs = np.arange(0.0, L - ex + 1e-6, 0.5 * cell)
                ws = np.array([-hw, 0.0, hw], float)
                px = float(pos[0]) + ux * xs[:, None] - uy * ws[None, :]
                py = float(pos[1]) + uy * xs[:, None] + ux * ws[None, :]
                ix = np.floor((px + H) / cell).astype(np.int64)
                iy = np.floor((py + H) / cell).astype(np.int64)
                nn = g.shape[0]
                ok = (ix >= 0) & (ix < nn) & (iy >= 0) & (iy < nn)
                if ok.any():
                    v = g[ix[ok], iy[ok]]
                    v = v[v > -1e8]
                    if v.size:
                        top = max(top, float(v.max()))
        return (top + float(dk.get("clear", 0.0))) if top > -1e8 else -1e9

    def _tdf_key_in(self, maps) -> bool:
        """candFe TDF: this seed's kind ("forest" if is_forest else the route kind) is in maps."""
        m = tuple(maps or ())
        return bool(m) and ("forest" if self.is_forest else str(self._mass_route_kind())) in m

    def _tdf_step(self, i, dd, pad, pos, vel, agl, dist, v_des) -> None:
        """candFe TDF soft touchdown (DESCEND, after PARK / CST): cap the sink by a braking profile on the distance
        to contact and, once contact is imminent / made, replace the slew-limited descent press by a light one.
        Only lowers |v_des[2]| (and, with tdf_cst, holds the lateral velocity while braking); never raises a climb."""
        c = self.cfg
        _prof = self._tdf_key_in(c.tdf_maps)
        _rest = self._tdf_key_in(c.tdf_rest_maps if c.tdf_rest_maps else c.tdf_maps) and float(c.tdf_rest_vz) > 0.0
        if not (_prof or _rest):
            return
        if dd.tdf_key != int(dd.claim):
            dd.tdf_key, dd.tdf_low, dd.tdf_vzp = int(dd.claim), -10 ** 9, 0.0
        above = float(pos[2]) - float(pad.xyz[2])
        _vzp, dd.tdf_vzp = float(dd.tdf_vzp), float(vel[2])
        if float(vel[2]) > 0.3:
            dd.tdf_low = -10 ** 9           # review fix: climbing (settle hop / TRR hop): an earlier contact window ends
        if (above > float(c.tdf_h) or self._endgame_hurry(dist, above)
                or (float(c.tdf_end_s) > 0.0 and 60.0 - self.step_i / 50.0 < float(c.tdf_end_s))):
            return
        hc = above - 0.023                  # contact: drone base (0.0125 below its centre) on the disc (pad top + 0.0105)
        if bool(c.tdf_agl) and 0.0 < float(agl) < 19.9 and float(agl) < above + float(c.tdf_agl_tol):
            hc = float(agl) - 0.0125        # the downward ray reads the pad top (terrain lies >= 0.2 m lower)
            if hc < float(c.tdf_rest_h):
                dd.tdf_low = int(self.step_i)
        if _vzp < -0.3 and float(vel[2]) > -0.15 and hc < 0.15:
            dd.tdf_low = int(self.step_i)   # impact: the sink stopped within 0.15 m of contact (the ray reads through)
        if _prof:
            v_allow = math.sqrt(float(c.tdf_vz) ** 2 + 2.0 * float(c.tdf_acc) * max(0.0, hc - float(c.tdf_h0)))
            if float(v_des[2]) < -v_allow:
                v_des[2] = -v_allow
                self._yt_n["tdf_cap"] = self._yt_n.get("tdf_cap", 0) + 1
                if (bool(c.tdf_cst) and self._cst_on() and 0.0 < above < float(c.cst_above)
                        and dist <= float(c.cst_r) and float(vel[2]) < -float(c.tdf_rest_v)):
                    v_des[0], v_des[1] = float(vel[0]), float(vel[1])     # CST coast while the profile brakes
                    self._yt_n["tdf_cst"] = self._yt_n.get("tdf_cst", 0) + 1
                    if bool(c.park_nosep):
                        dd.nosep_step = int(self.step_i)
        if (_rest and int(self.step_i) - int(dd.tdf_low) <= int(c.tdf_rest_win)
                and abs(float(vel[2])) < float(c.tdf_rest_v) and float(v_des[2]) < -float(c.tdf_rest_vz)):
            v_des[2] = -float(c.tdf_rest_vz)
            self._yt_n["tdf_rest"] = self._yt_n.get("tdf_rest", 0) + 1

    def _cst_on(self) -> bool:
        """CST (land_seq): this seed's kind ("forest" if is_forest else the route kind) is in cst_maps."""
        m = tuple(self.cfg.cst_maps or ())
        return bool(m) and ("forest" if self.is_forest else str(self._mass_route_kind())) in m

    def _lvl_on(self) -> bool:
        """LVL (land_seq): this seed's kind ("forest" if is_forest else the route kind) is in lvl_maps."""
        m = tuple(self.cfg.lvl_maps or ())
        return bool(m) and ("forest" if self.is_forest else str(self._mass_route_kind())) in m

    def _park_on(self) -> bool:
        """PARK (land_seq): this seed's kind ("forest" if is_forest else the route kind) is in park_maps."""
        m = tuple(self.cfg.park_maps or ())
        return bool(m) and ("forest" if self.is_forest else str(self._mass_route_kind())) in m

    def _edr_on(self) -> bool:
        """EDR (land_seq): this seed's kind ("forest" if is_forest else the route kind) is in edr_maps."""
        m = tuple(self.cfg.edr_maps or ())
        return bool(m) and ("forest" if self.is_forest else str(self._mass_route_kind())) in m

    def _fld_on(self) -> bool:
        """FLD (land_ext): this seed's kind ("forest" if is_forest else the route kind) is in fld_maps."""
        m = tuple(self.cfg.fld_maps or ())
        return bool(m) and ("forest" if self.is_forest else str(self._mass_route_kind())) in m

    def _le_inc(self, key: str) -> None:
        """land_ext diagnostics: count an event in src_counts as le_<key> (harness rows)."""
        self.src_counts["le_" + key] = self.src_counts.get("le_" + key, 0) + 1

    def _edr_ak(self):
        """EDR (edr_a, edr_k, edr_margin), with the FLD overrides on fld_maps."""
        c = self.cfg
        a, k, m = float(c.edr_a), float(c.edr_k), float(c.edr_margin)
        if c.fld_maps and self._fld_on():
            if float(c.fld_a) > 0.0:
                a = float(c.fld_a)
            if float(c.fld_k) > 0.0:
                k = float(c.fld_k)
            if float(c.fld_margin) >= 0.0:
                m = float(c.fld_margin)
        return a, k, m

    def _edr_vlat(self, dist: float) -> float:
        """EDR lateral braking profile: min(edr_vmax, sqrt(2 edr_a d), edr_k d)."""
        c = self.cfg
        d = max(0.0, float(dist))
        _a, _k, _m = self._edr_ak()
        return min(float(c.edr_vmax), math.sqrt(2.0 * _a * d), _k * d)

    def _edr_times(self, dist: float, v_al: float, h: float, vz: float):
        """EDR forward simulation (dt 0.04 s, <= 4 s): the lateral speed follows the braking profile (brakes at
        up to edr_ab * edr_a, speeds up at 4 m/s^2) and a drop started now ramps its sink rate at edr_av up to edr_vs
        (edr_prio: capped by the 3 m/s norm left over by the lateral speed). Returns (t_lat, t_contact): seconds
        until the drone is within edr_dok of the pad and until it has sunk h metres (99 = not within 4 s)."""
        c = self.cfg
        dt = 0.04
        d = float(dist)
        v = max(0.0, float(v_al))
        w = max(0.0, -float(vz))
        hh = float(h)
        a_b = float(c.edr_ab) * self._edr_ak()[0]
        av = float(c.edr_av)
        vs = float(c.edr_vs)
        if c.fld_maps and self._fld_on():
            vs = float(c.fld_sink)          # FLD: the drop sinks at fld_sink
        dok = float(c.edr_dok)
        prio = bool(c.edr_prio)
        t = 0.0
        t_lat = 99.0
        for _ in range(100):
            if t_lat > 90.0 and d <= dok:
                t_lat = t
            if hh <= 0.0:
                return t_lat, t
            vp = self._edr_vlat(d)
            v = max(vp, v - a_b * dt) if v > vp else min(vp, v + 4.0 * dt)
            wm = min(vs, math.sqrt(max(0.0, 9.0 - v * v))) if prio else vs
            w = min(wm, w + av * dt)
            d = max(0.0, d - v * dt)
            hh -= w * dt
            t += dt
        return t_lat, 99.0

    def _ehm_kind(self) -> str:
        return "forest" if self.is_forest else str(self._mass_route_kind())

    def _ehm_on(self) -> bool:
        m = tuple(self.cfg.ehm_maps or ())
        return bool(m) and self._ehm_kind() in m

    def _ehm_tc(self, dist: float, ab: float) -> float:
        c0, c1, c2, c3, c4 = EHM_TC.get(self._ehm_kind(), EHM_TC["city"])
        harr = max(0.0, ab - 4.0 - 0.667 * max(dist - 2.0, 0.0))
        return c0 + c1 * dist + c2 * max(ab, 0.0) + c3 * max(-ab, 0.0) + c4 * harr

    def _ehm_tlat(self, dist: float, top: float) -> float:
        """Time to fly from dist to ~0.8 ehm_enter_r on the EDR lateral profile min(top, sqrt(2 a d), k d)."""
        a, k, _m = self._edr_ak()
        a = max(float(a), 0.1)
        k = max(float(k), 0.1)
        top = max(float(top), 0.3)
        d_end = max(0.05, 0.8 * float(self.cfg.ehm_enter_r))
        d = max(float(dist), d_end)
        d_b = top * top / (2.0 * a)            # sqrt-profile meets the cruise cap
        d_k = 2.0 * a / (k * k)                # sqrt-profile meets the linear tail
        t = 0.0
        if d > d_b:
            t += (d - d_b) / top
            d = d_b
        if d > d_k:
            t += math.sqrt(2.0 * d / a) - math.sqrt(2.0 * max(d_k, d_end) / a)
            d = max(d_k, d_end)
        if d > d_end:
            t += math.log(d / d_end) / k
        return t

    def _ehm_apx(self, i, dd, pad, pos, vel, agl, d, dist, top):
        """EHM in APPROACH: None (not latched) or the hail-mary velocity; sets dd.phase."""
        c = self.cfg
        key = int(dd.claim)
        ab = float(pos[2] - pad.xyz[2])
        if getattr(dd, "ehm_key", -1) != key:
            if int(pad.hits) < int(c.ehm_min_hits) or dist > float(c.ehm_max_d):
                return None
            t_left = 60.0 - self.step_i / 50.0
            tc = self._ehm_tc(dist, ab)
            if t_left >= tc + float(c.ehm_slack):
                return None
            dd.ehm_key = key
            self.src_counts["ehm_fire"] = self.src_counts.get("ehm_fire", 0) + 1
            _lg = self.src_counts.setdefault("ehm_log", [])
            if isinstance(_lg, list) and len(_lg) < 64:
                _lg.append([round(self.step_i * 0.02, 2), i, key, int(pad.hits), round(dist, 2), round(ab, 2),
                            round(tc, 2), round(t_left, 2)])
        self.src_counts["ehm_steps"] = self.src_counts.get("ehm_steps", 0) + 1
        h_t = float(c.ehm_h)
        u = _unit(d)
        vl = min(float(top), self._edr_vlat(dist))
        _val = float(np.dot(np.asarray(vel[:2], float), u))
        t_lat = self._edr_times(dist, _val, 99.0, float(vel[2]))[0]     # EDR lateral plan to edr_dok (<= 4 s)
        if t_lat > 90.0:
            t_lat = self._ehm_tlat(dist, top) + 0.5
        if ab > h_t:
            vz = -min((ab - h_t) / max(t_lat, 0.25), 2.8)
        else:
            vz = min(2.0, 1.5 * (h_t - ab) + 0.3)
        if vz < 0.0 and dist > 3.0 and agl < float(c.ehm_agl):
            vz = 0.0
        if agl < AGL_MIN and dist > AGL_FAR:
            vz = max(vz, AGL_CLIMB)
        v = np.array([u[0] * vl, u[1] * vl, vz], dtype=float)
        mag = float(np.linalg.norm(v))
        if mag > float(top) and mag > 1e-9:
            v *= float(top) / mag
        if dist < float(c.ehm_enter_r) and float(np.hypot(float(vel[0]), float(vel[1]))) < float(c.ehm_enter_v):
            dd.phase = DESCEND
            dd.edr_key = key
        else:
            dd.phase = APPROACH
        return v

    def _vf_on(self) -> bool:
        c = self.cfg
        return bool(c.apx_vfy_maps or c.apx_vfy_late_maps)

    def _vf_champ_gate(self, dd, pad) -> bool:
        """True when the champion's descend-entry gate (dsc_min_hits) would refute this claim now
        (same test as the DESCEND branch: hits, claim distance, time, route kind)."""
        c = self.cfg
        if int(c.dsc_min_hits) <= 0 or dd.dsc_gated:
            return False
        kind = self._mass_route_kind()
        _gm = tuple(getattr(c, "dsc_gate_maps", ()) or ())
        _gmd = float(c.dsc_gate_min_dist)
        _bym = tuple(getattr(c, "dsc_gate_min_dist_by_map", ()) or ())
        for _k in range(0, len(_bym) - 1, 2):
            if str(_bym[_k]) == kind:
                _gmd = float(_bym[_k + 1])
                break
        return (int(pad.hits) < int(c.dsc_min_hits)
                and float(dd.claim_dist) >= _gmd
                and self.step_i * 0.02 < float(c.dsc_gate_until_sec)
                and ((not _gm) or (kind in _gm))
                and self._s1_gate_ok(dd))

    def _s1_gate_ok(self, dd) -> bool:
        """S1: False when the champion's descend-entry gate must not fire on this claim (fewer than
        dsc_gate_inview_min in-view ticks, or claimed dsc_gate_steep_deg or more below the drone)."""
        c = self.cfg
        if int(c.dsc_gate_inview_hits_min) > 0 and dd.claim is not None and 0 <= int(dd.claim) < len(self.pads):
            # F4: an entry with fewer than dsc_gate_inview_hits_min hits gets no S1 exemption
            if int(self.pads[int(dd.claim)].hits) < int(c.dsc_gate_inview_hits_min):
                return True
        if int(c.dsc_gate_inview_min) > 0 and int(dd.gate_inview) < int(c.dsc_gate_inview_min):
            return False
        if float(c.dsc_gate_steep_deg) > 0.0 and float(dd.claim_elev) >= float(c.dsc_gate_steep_deg):
            return False
        return True

    def _apx_vfy(self, i: int, dd, pad, pos, rpy, apx_alt: float) -> bool:
        """APXV-R approach check for drone i in APPROACH. Returns True on a hold / orbit step with
        the command left in dd.vf_cmd = (v_des, yaw); on a failed check the pad is hard-refuted
        (dd.claim is None afterwards). False: fly the normal approach this step."""
        c = self.cfg
        key = dd.claim if dd.claim is not None else -1
        if dd.vf_key != key:
            dd.vf_key, dd.vf_step0, dd.vf_hits0, dd.vf_hist = key, self.step_i, int(pad.hits), []
            dd.vf_state, dd.vf_t, dd.vf_h, dd.vf_pt = 0, 0, 0, None
            dd.vf_tr, dd.vf_t2, dd.vf_cmd = -1, -1, None
            ct = float(c.commit_t)
            _bym = tuple(getattr(c, "commit_t_by_map", ()) or ())
            for _k in range(0, len(_bym) - 1, 2):
                if str(_bym[_k]) == self.map_kind:
                    ct = float(_bym[_k + 1])
                    break
            dd.vf_late = bool(ct > 0.0 and self.map_kind not in c.commit_t_skip
                              and self.step_i >= int(ct * 50) and int(pad.hits) < int(self.min_hits))
        if dd.vf_state == 1:
            return False
        kind = "forest" if self.is_forest else str(self.map_kind)
        if not (kind in tuple(c.apx_vfy_maps or ())
                or (dd.vf_late and kind in tuple(c.apx_vfy_late_maps or ()))):
            return False
        pxy = np.asarray(pad.xyz, float)[:2]
        d = pxy - pos[:2]
        dist = float(np.linalg.norm(d))
        t = self.step_i * 0.02
        if dd.vf_state == 0:
            W = int(round(float(c.apx_vfy_win_s) * 50))
            h = dd.vf_hist
            h.append((self.step_i, int(pad.hits)))
            while len(h) >= 2 and h[1][0] <= self.step_i - W:
                h.pop(0)
            if dist >= float(c.apx_vfy_r):
                return False
            if c.apx_vfy_skip_champ and self._vf_champ_gate(dd, pad):
                return False                  # refuted at DESCEND entry anyway; re-tested every step
            h_ago = h[0][1] if h[0][0] <= self.step_i - W else dd.vf_hits0
            if ((60.0 - self.step_i / 50.0) < float(c.apx_vfy_tleft_min)
                    or (self.step_i - dd.vf_step0) * 0.02 < float(c.apx_vfy_min_age_s)
                    or int(pad.hits) >= int(c.apx_vfy_hits_ok)
                    or int(pad.hits) - int(h_ago) >= int(c.apx_vfy_gain)):
                dd.vf_state = 1
                return False
            dd.vf_state, dd.vf_t, dd.vf_h, dd.vf_pt = 2, self.step_i, int(pad.hits), pos[:2].copy()
            self._vf_log.append((round(t, 2), i, int(dd.claim), "HOLD", int(pad.hits)))
        if int(pad.hits) - dd.vf_h >= int(c.apx_vfy_gain):
            dd.vf_state = 1
            self._vf_log.append((round(t, 2), i, int(dd.claim), "PASS", int(pad.hits)))
            return False
        if dd.vf_state == 2 and (self.step_i - dd.vf_t) * 0.02 >= float(c.apx_vfy_hold_s):
            # second viewpoint: the hold point rotated about the pad
            a = math.radians(float(c.apx_vfy_orbit_deg))
            r = dd.vf_pt - pxy
            new = pxy + np.array([r[0] * math.cos(a) - r[1] * math.sin(a),
                                  r[0] * math.sin(a) + r[1] * math.cos(a)])
            ch = new - pos[:2]
            dd.vf_ch = float(np.linalg.norm(ch))
            dd.vf_yc = float(np.arctan2(ch[1], ch[0])) if dd.vf_ch > 1e-6 else float(rpy[2])
            dd.vf_pt, dd.vf_state, dd.vf_t, dd.vf_tr, dd.vf_t2 = new, 3, self.step_i, -1, -1
            self._vf_log.append((round(t, 2), i, int(dd.claim), "ORBIT", int(pad.hits)))
        spd = max(float(c.apx_vfy_speed), 0.1)
        tp = dd.vf_pt - pos[:2]
        dn = float(np.linalg.norm(tp))
        brg = float(np.arctan2(d[1], d[0])) if dist > 1e-6 else float(rpy[2])
        transit = False
        if dd.vf_state == 3:
            if dd.vf_tr < 0 and (dn < 0.5 or (self.step_i - dd.vf_t) * 0.02 >= dd.vf_ch / spd + 1.0):
                dd.vf_tr = self.step_i
            transit = dd.vf_tr < 0
            if not transit and dd.vf_t2 < 0:
                off = abs((float(rpy[2]) - brg + np.pi) % (2.0 * np.pi) - np.pi)
                if (off <= math.radians(float(c.apx_vfy_yaw_deg))
                        or (self.step_i - dd.vf_tr) * 0.02 >= float(c.apx_vfy_yaw_wait_s)):
                    dd.vf_t2 = self.step_i
            if dd.vf_t2 >= 0 and (self.step_i - dd.vf_t2) * 0.02 >= float(c.apx_vfy_hold_s):
                self._vf_log.append((round(t, 2), i, int(dd.claim), "REFUTE", int(pad.hits)))
                self._refute(i)
                dd.vf_cmd = None
                return True
        v = np.zeros(3)
        if dn > 1e-6:
            v[:2] = tp / dn * min(spd, 1.2 * dn)
        zt = float(pad.xyz[2]) + float(apx_alt) + (float(c.apx_vfy_orbit_up) if transit else 0.0)
        v[2] = float(np.clip(zt - float(pos[2]), -1.0, 1.0))
        dd.vf_cmd = (v, dd.vf_yc if transit else brg)
        return True
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
        if self.map_kind == "mountain" and float(getattr(self.cfg, "world_mountain", -1.0)) > 0.0:
            r = float(self.cfg.world_mountain)
        _wp = getattr(self, "_wpost", None)
        if (_wp is not None and self.map_kind == "mountain" and _wp.get("clamp_r") is not None):
            r = float(_wp["clamp_r"])            # WPOST-X: Wmax + world_post_clamp_m
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
        c = self.cfg
        if c.own_split_tall <= 0.0 or self._clear_n <= 0:
            return False
        if self.map_kind not in ("city", "open"):
            return False
        return (self._tall_hits / float(self._clear_n)) <= c.own_split_tall
    def _ring_kind(self) -> str:
        c = self.cfg
        if self.map_kind != "city" or c.own_split_tall <= 0.0:
            return self.map_kind
        if self._clear_n <= 0:
            return self.map_kind
        if (self._tall_hits / float(self._clear_n)) > c.own_split_tall:
            return "city"
        return c.own_split_kind
    def _own_target(self, i: int) -> Optional[np.ndarray]:
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
        classification_ready = self._band_locked or kind == "mountain" or self._prv_force is not None
        intervention_ready = self.step_i * 0.02 >= self.cfg.route_start_sec
        for mp in tuple(self.cfg.route_max_n_by_map or ()):
            if mp and mp[0] == kind and self.n > int(mp[1]):
                return False
        return bool(classification_ready and intervention_ready
                    and self.route_planner is not None
                    and self.cfg.route_planner
                    and kind in self.cfg.route_plan_maps)

    def _splan_gain(self) -> float:
        """Surrogate gain of the planner's best plan over the champion route; on held-out
        layouts when configured (the CEM's own layouts overstate it)."""
        if self._splan is None:
            return 0.0
        if self.cfg.search_plan_holdout:
            h0, h1 = self._splan.holdout_scores
            if h0 is None:
                return 0.0
            return float(h1 - h0)
        s0, s1 = self._splan.scores
        return float(s1 - s0)

    def _splan_params(self):
        det, sec, climb, land, spd = (float(self.cfg.search_plan_det_range), float(self.cfg.search_plan_sector),
                                      float(self.cfg.search_plan_climb_delay), float(self.cfg.search_plan_land_delay), 3.0)
        for mp in tuple(self.cfg.search_plan_map_params or ()):
            if mp and mp[0] == self._route_kind():
                det, sec, climb, land = float(mp[1]), float(mp[2]), float(mp[3]), float(mp[4])
                if len(mp) > 5:
                    spd = float(mp[5])
        return det, sec, climb, land, spd

    def _splan_blind_r(self) -> float:
        pairs = tuple(getattr(self.cfg, "search_plan_blind_r_by_map", ()) or ())
        for k in range(0, len(pairs) - 1, 2):
            if str(pairs[k]) == self._route_kind():
                return float(pairs[k + 1])
        return 0.0

    def _replan_tick(self, state) -> None:
        """Conditioned re-planning of the free drones' sweep (see search_replan)."""
        cfg = self.cfg
        if getattr(self, "_plan_disabled", False):
            return
        if not cfg.search_replan or self._route_waypoints is None or self.clue is None or self._own_start is None:
            return
        if self.n > int(cfg.search_replan_max_n):
            return
        kind = self._route_kind()
        maps = tuple(cfg.search_replan_maps or ()) or tuple(cfg.search_plan_maps or ())
        if kind not in maps:
            return
        t = self.step_i * 0.02
        # a pending (phased) start: one cheap phase per act until the planner is ready
        if getattr(self, "_replan_pending", None) is not None:
            try:
                ready = self._replan_pending.start_conditioned_phase()
                if ready is None:
                    self._replan_pending = None
                elif ready:
                    self._replan = self._replan_pending
                    self._replan_pending = None
            except Exception:                                    # noqa: BLE001
                self._replan_pending = None
            return
        # advance a running re-plan; apply once it is done
        if self._replan is not None:
            try:
                self._replan.step(float(cfg.search_plan_budget), chunk=2)
                if self._replan.done:
                    h0, h1 = self._replan.holdout_scores
                    self._splan_pred = (h0, h1)
                    if h0 is not None and (h1 - h0) >= float(cfg.search_replan_min_gain):
                        best = np.asarray(self._replan.best(), dtype=float)
                        for jj, j in enumerate(self._replan_free):
                            dd = self.d[j]
                            if dd.claim is None and not dd.landed and dd.phase in (SEARCH, CLIMB):
                                self._route_waypoints[j] = [np.asarray(w, dtype=float) for w in best[jj]]
                                dd.route_idx = 0
                        self._replan_applied = getattr(self, "_replan_applied", 0) + 1
                    self._replan = None
            except Exception:                                    # noqa: BLE001
                self._replan = None
            return
        if t > float(cfg.search_replan_until_sec) or (t - self._replan_t) < float(cfg.search_replan_gap_sec):
            return
        found = self._hunt_found()
        if not found:
            return
        key = tuple(np.round(np.asarray(found, float).reshape(-1), 0).tolist())
        if key == self._replan_key:
            return
        free = [i for i, dd in enumerate(self.d)
                if dd.claim is None and not dd.landed and dd.phase in (SEARCH, CLIMB)]
        if not free or len(found) >= self.n:
            return
        try:
            from team.autopilot.search_planner import SearchPlanner
            det, sec, climb, land, spd = self._splan_params()
            K = 5
            init = np.zeros((len(free), K, 2), np.float32)
            cur = np.asarray(state[free, POS], float)[:, :2]
            for jj, j in enumerate(free):
                rest = [np.asarray(w, float)[:2] for w in self._route_waypoints[j][self.d[j].route_idx:]]
                if not rest:
                    rest = [cur[jj]]
                rest = (rest + [rest[-1]] * K)[:K]
                init[jj] = np.asarray(rest, np.float32)
            pl = SearchPlanner(K=K, layouts=int(cfg.search_replan_layouts), pop=int(cfg.search_replan_pop),
                               iters=int(cfg.search_replan_iters), fixed_prefix=0,
                               sigma0=float(cfg.search_plan_sigma0), det_range=det, sector_deg=sec,
                               climb_delay=climb, land_delay=land, speed=spd, blind_r=self._splan_blind_r())
            ok = pl.start_conditioned_light(cur, np.asarray(self._own_start, float)[free, :2],
                                            np.asarray(self._own_start, float)[:, :2], np.asarray(self.clue, float)[:2],
                                            kind, np.asarray(found, float)[:, :2], init_plan=init, t0=t)
            self._replan_key = key
            self._replan_t = t
            self._replan_free = list(free)
            if ok is None:
                pass
            elif ok:
                self._replan = pl
            else:
                self._replan_pending = pl
        except Exception:                                        # noqa: BLE001
            self._replan = None

    def _start_splan(self, route):
        """Create the search planner around the champion route; returns the route to fly now."""
        if getattr(self, "_plan_disabled", False):
            return route
        try:
            from team.autopilot.search_planner import SearchPlanner
            det, sec, climb, land, spd = self._splan_params()
            iters = 0 if self._route_kind() in tuple(self.cfg.search_plan_arc_only_maps or ()) else int(self.cfg.search_plan_iters)
            self._splan = SearchPlanner(K=int(route.shape[1]), layouts=int(self.cfg.search_plan_layouts),
                                        pop=int(self.cfg.search_plan_pop), iters=iters,
                                        det_range=det, sector_deg=sec, climb_delay=climb, land_delay=land, speed=spd,
                                        blind_r=self._splan_blind_r(),
                                        fixed_prefix=int(self.cfg.search_plan_fixed_prefix),
                                        sigma0=float(self.cfg.search_plan_sigma0),
                                        arc_n=int(self.cfg.search_plan_arc_n))
            self._splan.start(np.asarray(self._own_start, float)[:, :2], np.asarray(self.clue, float)[:2],
                              self._route_kind(), init_plan=route)
            self._diag_splan_step = int(self.step_i)
            self._splan.step(float(self.cfg.search_plan_budget_first), chunk=2)
            self._splan_pred = (self._splan.holdout_scores if self.cfg.search_plan_holdout else self._splan.scores)
            if self._splan_gain() >= float(self.cfg.search_plan_min_gain):
                return np.asarray(self._splan.best(), dtype=float)
        except Exception:                                    # noqa: BLE001
            self._splan = None
            try:
                import traceback as _tb
                self._diag_err = _tb.format_exc()[-800:]
            except Exception:  # noqa: BLE001
                pass
        return route

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
            plan_ok = self.cfg.search_plan and self._route_kind() in tuple(self.cfg.search_plan_maps or ())
            for mp in tuple(self.cfg.search_plan_max_n_by_map or ()):
                if mp and mp[0] == self._route_kind() and self.n > int(mp[1]):
                    plan_ok = False
            if plan_ok:
                if int(self.cfg.search_plan_defer_steps) > 0:
                    self._splan_deferred = (np.asarray(route, dtype=float).copy(), self.step_i + int(self.cfg.search_plan_defer_steps))
                else:
                    route = self._start_splan(route)
            self._diag_route_step = int(self.step_i)
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
            try:
                import traceback as _tb
                self._diag_err = 'route: ' + _tb.format_exc()[-800:]
            except Exception:  # noqa: BLE001
                pass

    def _maybe_replan(self, state) -> None:
        if not self.cfg.replan_sec or self._route_waypoints is None:
            return
        if not self._route_enabled():
            return
        kind = self._route_kind()
        if self.cfg.replan_maps and kind not in self.cfg.replan_maps:
            return
        rs = self.cfg.replan_sec
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
        elif mode == "shuffle":                    
            for i in range(n):
                if len(rw[i]) > 1:
                    rw[i] = [rw[i][j] for j in rng.permutation(len(rw[i]))]
        elif mode == "swap":                       
            flat = [w for wl in rw for w in wl]
            if flat:
                idx = rng.permutation(len(flat))
                out, p = [], 0
                for i in range(n):
                    out.append([flat[idx[p + j]] for j in range(len(rw[i]))])
                    p += len(rw[i])
                self._route_waypoints = out
        elif mode == "arc":                        
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
        elif mode == "radial":                     
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
        load = [[] for _ in range(n)]
        cap = int(math.ceil(len(pool) / float(n)))
        if latency:
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
        d = float(getattr(self.cfg, "lane_spread", 0.0))
        if d <= 0.0 or self._route_waypoints is None:
            return
        for j, wl in enumerate(self._route_waypoints):
            if len(wl) < 4 or self._own_start is None:
                continue
            try:
                base = np.asarray(self._own_start, float)
                p0 = base[j][:2] if base.ndim == 2 else base[:2]
            except Exception:                                
                continue
            pts = [p0] + [self._in_bounds(np.array([w[0], w[1], 0.0]))[:2]
                          for w in wl]
            for m in range(3, len(pts)):
                a, b = pts[m - 1], pts[m]
                mid = 0.5 * (a + b)
                worst, wd = None, d
                for k in range(m - 2):                       
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
            if self._prv_kind is not None and not self._band_locked:
                self._prv_force = self._prv_kind               # prv: clamp as the provisional kind
                try:
                    target = (self._in_bounds(np.array([route[dd.route_idx][0], route[dd.route_idx][1], 0.0], dtype=float))[:2] if (FIX_CLAMP and self._mass_route_kind() in FIX_BOX_KINDS) else np.asarray(route[dd.route_idx], float))
                finally:
                    self._prv_force = None
            else:
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

    # --------------------------------------------------------------- WPOST-X --
    def _wpost_cem_n(self) -> bool:
        """This fleet flies the mountain CEM search planner (search_plan_max_n_by_map, i.e. n=2)."""
        c = self.cfg
        if not c.search_plan or "mountain" not in tuple(c.search_plan_maps or ()):
            return False
        for mp in tuple(c.search_plan_max_n_by_map or ()):
            if mp and mp[0] == "mountain" and self.n > int(mp[1]):
                return False
        return True

    def _wpost_init(self) -> None:
        """First step, mountain only: the world-size posterior from the starts (world_post.py) and
        its W components; the n=2 CEM branch also installs them in geo_posterior."""
        c = self.cfg
        if not (c.world_post_router or c.world_post_cem) or self.map_kind != "mountain" or self._own_start is None:
            return
        b6 = self._b6_n
        try:
            from team.route_replan import world_post as WP
            from team.autopilot import geo_posterior as GP
            S = np.asarray(self._own_start, float)[:, :2]
            WG, p, m = WP.posterior(S, float(c.world_post_lo), float(c.world_post_hi), float(c.world_post_nudge))
            margin = float(c.world_post_margin) + sum(float(v) for n_, v in tuple(c.world_post_margin_by_n or ())
                                                      if int(n_) == int(self.n))
            Wq = WP.components(WG, p, m, int(c.world_post_mix), str(c.world_post_stat), margin)
            mean = WP.mean(WG, p) if p is not None else m
            router = bool(c.world_post_router and self.n >= int(c.world_post_min_n))
            cem = bool(c.world_post_cem and self._wpost_cem_n())
            Wmax = float(max(Wq))
            clamp = (Wmax + float(c.world_post_clamp_m)) if (float(c.world_post_clamp_m) >= 0.0 and (router or cem)) else None
            self._wpost = {"mean": float(mean), "Wq": [float(w) for w in Wq], "Wmax": Wmax, "m": float(m),
                           "router": router, "cem": cem, "clamp_r": clamp}
            if cem:
                GP.MAP_PARAMS["mountain"] = dict(GP.MAP_PARAMS["mountain"], world=Wmax)
                GP.WORLD_Q = tuple(float(w) for w in Wq)
            if margin != float(c.world_post_margin):
                b6["wp_margin_n"] = round(margin, 2)
            b6.update({"wp_on": 1, "wp_router": int(router), "wp_cem": int(cem), "wp_m": round(float(m), 2),
                       "wp_mean": round(float(mean), 2), "wp_Wq": [round(float(w), 2) for w in Wq],
                       "wp_Wmax": round(Wmax, 2), "wp_clamp_r": None if clamp is None else round(clamp, 2),
                       "wp_map_world": float(GP.MAP_PARAMS["mountain"]["world"]),
                       "wp_world_q": None if GP.WORLD_Q is None else list(GP.WORLD_Q)})
        except Exception:                                        # noqa: BLE001
            self._wpost = None
            b6["wp_err"] += 1

    # ------------------------------------------------------------------ hunt --
    def _hunt_found(self):
        """Pads the posterior is conditioned on: claimed or landed on, or confirmed at least
        8 times (the descent evidence level). Refuted pads (done without an owner) are
        phantoms and are excluded; a 3-hit sighting is too often a phantom to count."""
        out = []
        for q in self.pads:
            if q.done and q.by is None:
                continue
            if (q.by is not None and not q.done and int(self.cfg.hunt_found_min_hits) > 0
                    and q.hits < int(self.cfg.hunt_found_min_hits)):
                continue                    # mtn_n2: a young claim (2-hit mountain claims are 9% phantoms) does not condition
            if q.by is not None or q.hits >= 8:
                out.append(np.asarray(q.xyz, float)[:2])
        return out

    def _hunt_unassigned(self) -> int:
        """Drones still looking for a pad of their own."""
        return sum(1 for dd in self.d
                   if dd.claim is None and not dd.landed and dd.phase in (SEARCH, CLIMB))

    def _hunt_kind(self):
        from team.autopilot.geo_posterior import MAP_PARAMS
        kind = self._mass_route_kind()
        return kind if kind in MAP_PARAMS else None

    def _hunt_active(self) -> bool:
        c = self.cfg
        if not c.hunt or self._hunt_post is None or self.clue is None:
            return False
        if self.step_i * 0.02 < float(c.hunt_start_sec):
            return False
        if int(c.hunt_max_n) > 0 and self.n > int(c.hunt_max_n):
            return False
        kind = self._hunt_kind()
        if kind is None:
            return False
        hm = tuple(getattr(c, "hunt_maps", ()) or ())
        if hm and kind not in hm:
            return False
        # engage once only a few drones are still unassigned: the pads they need are the
        # ones the generic sweep has failed to reach, and the posterior is now sharp
        if bool(c.hunt_need_found) and not self._hunt_found():
            return False
        return 1 <= self._hunt_unassigned() <= int(c.hunt_max_unfound)

    def _hunt_grid(self):
        """(xy (M,2), residual expected-pad mass (M,)) for the current fleet knowledge."""
        post = self._hunt_post
        kind = self._hunt_kind()
        post._ensure(kind)
        found = self._hunt_found()
        post._apply_found(np.asarray(found, np.float32).reshape(-1, 2))
        xs, ys, grid = post.expected_count_grid()
        if self._hunt_xy is None or self._hunt_xy.shape[0] != grid.size:
            X, Y = np.meshgrid(xs, ys)
            self._hunt_xy = np.column_stack((X.ravel(), Y.ravel()))
            self._hunt_seen = np.zeros(grid.size, dtype=bool)
        res = grid.reshape(-1).copy()
        res[self._hunt_seen] = 0.0
        for f in found:
            res[np.hypot(self._hunt_xy[:, 0] - f[0], self._hunt_xy[:, 1] - f[1]) <= 6.0] = 0.0
        return self._hunt_xy, res

    def _hunt_mark(self, state) -> None:
        """Credit the camera sweep of every flying drone: cells within hunt_vis_r inside
        the +-47 deg field of view around the drone's yaw."""
        if self._hunt_xy is None:
            if self._hunt_kind() is None:
                return
            self._hunt_grid()
        xy = self._hunt_xy
        for i in range(self.n):
            if self.d[i].phase not in (SEARCH, APPROACH, CLIMB):
                continue
            p = np.asarray(state[i, POS], float)[:2]
            yaw = float(state[i, RPY][2])
            rel = xy - p[None, :]
            rng = np.hypot(rel[:, 0], rel[:, 1])
            brg = np.abs((np.arctan2(rel[:, 1], rel[:, 0]) - yaw + np.pi) % (2 * np.pi) - np.pi)
            self._hunt_seen |= (rng <= self.cfg.hunt_vis_r) & (brg <= np.radians(47.0))

    def _hunt_target(self, i: int, pos):
        c = self.cfg
        xy, res = self._hunt_grid()
        p = np.asarray(pos, float)[:2]
        cur = self._hunt_tgt.get(i)
        if cur is not None:
            if float(np.linalg.norm(cur - p)) < c.hunt_reach:
                self._hunt_seen |= np.hypot(xy[:, 0] - cur[0], xy[:, 1] - cur[1]) <= c.hunt_reach
                res[np.hypot(xy[:, 0] - cur[0], xy[:, 1] - cur[1]) <= c.hunt_reach] = 0.0
                cur = None
                self._hunt_tgt.pop(i, None)
        others = [t for j, t in self._hunt_tgt.items() if j != i and t is not None]
        dist = np.hypot(xy[:, 0] - p[0], xy[:, 1] - p[1])
        score = res / (1.0 + dist / max(c.hunt_dist_scale, 1e-6))
        for t in others:
            score[np.hypot(xy[:, 0] - t[0], xy[:, 1] - t[1]) <= c.hunt_sep] = 0.0
        k = int(np.argmax(score))
        if res[k] < c.hunt_min_mass:
            return None
        if cur is not None:
            kc = int(np.argmin(np.hypot(xy[:, 0] - cur[0], xy[:, 1] - cur[1])))
            if res[kc] >= c.hunt_min_mass and score[k] < c.hunt_switch_ratio * score[kc]:
                k = kc
        tgt = np.array([xy[k, 0], xy[k, 1]], float)
        self._hunt_tgt[i] = tgt
        return self._in_bounds(np.array([tgt[0], tgt[1], 0.0]))

    def _ay_key(self) -> str:
        return "forest" if self.is_forest else str(self._mass_route_kind())

    def _ay_yaw(self, i: int, pos, tgt, base_yaw: float):
        """yaw_time AY: camera yaw toward the path point ay_look_m ahead (current leg up to its switch
        point, then the next leg), clipped to ay_max_deg off base_yaw; None when not near a switch."""
        c = self.cfg
        rw = self._route_waypoints
        if rw is None or i >= len(rw):
            return None
        route = rw[i]
        k = int(self.d[i].route_idx)
        if k + 1 >= len(route):
            return None
        nxt = np.array([float(route[k + 1][0]), float(route[k + 1][1]), 0.0])
        if FIX_CLAMP and self._mass_route_kind() in FIX_BOX_KINDS:
            nxt = self._in_bounds(nxt)
        nxt = np.asarray(nxt, float)[:2]
        cur = np.asarray(tgt, float)[:2]
        p = np.asarray(pos, float)[:2]
        v = cur - p
        dc = float(np.linalg.norm(v))
        look = float(c.ay_look_m)
        r = dc - float(c.waypoint_reach)
        if dc < 1e-6 or r >= look:
            return None
        r = max(r, 0.0)
        psw = p + v / dc * r
        w = nxt - psw
        dw = float(np.linalg.norm(w))
        if dw < 1e-6:
            return None
        la = psw + w / dw * (look - r)
        ya = float(np.arctan2(la[1] - p[1], la[0] - p[0]))
        mx = float(c.ay_max_deg)
        _bm = tuple(c.ay_max_by_map or ())
        for _k in range(0, len(_bm) - 1, 2):
            if str(_bm[_k]) == self._ay_key():
                mx = float(_bm[_k + 1])
                break
        off = (ya - base_yaw + np.pi) % (2.0 * np.pi) - np.pi
        mxr = math.radians(mx)
        off = float(np.clip(off, -mxr, mxr))
        return float(base_yaw + off)

    def _mass_route_kind(self) -> str:
        if self._prv_force is not None:
            return self._prv_force                              # prv: only inside the provisional plan call
        if self._gkc_route is not None and self.map_kind in ("city", "open"):
            return self._gkc_route                              # gkc: the ground signature says open
        _wm = self.cfg.kind_wall_maps
        if _wm and self.map_kind in ("city", "open") and self.map_kind in tuple(_wm):
            # kind_wall_maps: the router's geometric city fix, then the vertical-surface fraction
            _rr = getattr(self, "replan_router", None)
            if _rr is not None and getattr(_rr, "_kind_fixed", None) == "city":
                return "city"
            _wf = self._wall_frac
            if _wf is not None:
                if _wf >= float(self.cfg.kind_wall_hi):
                    return "city"
                if _wf <= float(self.cfg.kind_wall_lo):
                    return "open"
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
    def _pitch_rows(self, h: int, band) -> tuple:
        """Image rows covering a world-elevation window, corrected for the drone's pitch.

        The fixed 28-52%% band sits almost entirely above the horizon, so a rooftop at
        the drone's own altitude reaches the very edge of it -- and a nose-down cruise
        attitude pushes it out altogether. Anchoring the window to world elevation puts
        whatever is level with the drone back in the middle of what avoidance reads.
        """
        lo_deg, hi_deg = float(band[0]), float(band[1])
        pitch = float(getattr(self, "_pitch_now", 0.0) or 0.0)   # + is nose-up
        fov = math.radians(PD.AP_FOV_DEG)
        def _row(e_deg):
            e = math.radians(e_deg)
            return int(round(h * (0.5 - (e - pitch) / fov)))
        top = _row(hi_deg)
        bot = _row(lo_deg)
        top = min(max(top, 0), h - 2)
        bot = min(max(bot, top + 2), h)
        return top, bot

    def _ground_ahead(self, depth: np.ndarray, state: np.ndarray, i: int):
        """World z of the ground 5-18 m ahead (low percentile), or None if the frame carries
        too little valid terrain there (e.g. looking at the sky over a valley)."""
        from team.detector.geom import _cloud
        cfg = self.cfg
        pos = np.asarray(state[i, POS], float)
        rpy = np.asarray(state[i, RPY], float)
        R = rot_from_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        d = np.asarray(depth[i], np.float32)
        if d.ndim == 3:
            d = d[..., 0]
        pts, valid, _ = _cloud(pos, R, d[::2, ::2], PD._geom_cfg(self.pad_cfg), PD.AP_FOV_DEG, PD.AP_RES // 2)
        if pts is None:
            return None
        p = pts[valid]
        if p.shape[0] < cfg.terrain_ahead_min_pts:
            return None
        rel = p[:, :2] - pos[:2]
        horiz = np.hypot(rel[:, 0], rel[:, 1])
        fwd = np.array([np.cos(float(rpy[2])), np.sin(float(rpy[2]))])
        ahead = (rel @ fwd) / np.maximum(horiz, 1e-6)
        sel = (horiz >= cfg.terrain_ahead_min_m) & (horiz <= cfg.terrain_ahead_max_m) & (ahead >= 0.5)
        if int(sel.sum()) < cfg.terrain_ahead_min_pts:
            return None
        return float(np.percentile(p[sel, 2], cfg.terrain_ahead_pct))

    _LAV_HALF = 160.0

    def _lav_on(self) -> bool:
        m = tuple(self.cfg.lav_maps or ())
        return bool(m) and ("forest" if self.is_forest else str(self.map_kind)) in m

    def _lav_update(self, i: int, state: np.ndarray, depth: np.ndarray) -> None:
        """mtn_speed LAV: fold drone i's depth frame (subsampled) into the fleet max-height grid."""
        from team.detector.geom import _cloud
        c = self.cfg
        cell = float(c.lav_cell)
        H = self._LAV_HALF
        if self._lav_top is None:
            nn = int(2.0 * H / cell) + 1
            self._lav_top = np.full((nn, nn), -1e9, dtype=np.float32)
        pos = np.asarray(state[i, POS], float)
        rpy = np.asarray(state[i, RPY], float)
        R = rot_from_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        d = np.asarray(depth[i], np.float32)
        if d.ndim == 3:
            d = d[..., 0]
        st = max(1, int(c.lav_stride))
        sub = d[st // 2::st, st // 2::st]
        pts, valid, _ = _cloud(pos, R, sub, PD._geom_cfg(self.pad_cfg), PD.AP_FOV_DEG, int(sub.shape[0]))
        if pts is None:
            return
        p = pts[valid].astype(np.float64)
        if p.shape[0] == 0:
            return
        if float(c.lav_mate_r) > 0.0 and self.n > 1:
            slots = np.asarray(state[i, MATES], float).reshape(7, 7)
            keep = np.ones(p.shape[0], bool)
            for s in slots:
                if s[6] < 0.5:
                    continue
                mw = pos + s[:3]
                keep &= np.sum((p - mw[None, :]) ** 2, axis=1) > float(c.lav_mate_r) ** 2
            p = p[keep]
        ix = np.floor((p[:, 0] + H) / cell).astype(np.int64)
        iy = np.floor((p[:, 1] + H) / cell).astype(np.int64)
        nn = self._lav_top.shape[0]
        ok = (ix >= 0) & (ix < nn) & (iy >= 0) & (iy < nn)
        if ok.any():
            np.maximum.at(self._lav_top, (ix[ok], iy[ok]), p[ok, 2].astype(np.float32))
            self._lav_n["upd"] += 1

    def _lav_vz(self, i: int, pos, agl: float, want: float, v_des) -> Optional[float]:
        """mtn_speed LAV: vertical command from the funnel over the known ground ahead, or None (stock law)."""
        c = self.cfg
        g = self._lav_top
        if g is None or not (0.05 < agl < 19.9):
            return None
        h = float(np.hypot(float(v_des[0]), float(v_des[1])))
        if h < 0.5:
            return None
        u0, u1 = float(v_des[0]) / h, float(v_des[1]) / h
        cell = float(c.lav_cell)
        H = self._LAV_HALF
        xs = np.arange(1.0, float(c.lav_look) + 1e-6, cell)
        ws = np.arange(-float(c.lav_half_w), float(c.lav_half_w) + 1e-6, cell)
        px = float(pos[0]) + u0 * xs[:, None] - u1 * ws[None, :]
        py = float(pos[1]) + u1 * xs[:, None] + u0 * ws[None, :]
        ix = np.floor((px + H) / cell).astype(np.int64)
        iy = np.floor((py + H) / cell).astype(np.int64)
        nn = g.shape[0]
        ok = (ix >= 0) & (ix < nn) & (iy >= 0) & (iy < nn)
        top = np.full(ix.shape, -1e9, dtype=np.float64)
        top[ok] = g[ix[ok], iy[ok]]
        mid = int(np.argmin(np.abs(ws)))
        tx = top[:, mid]                      # centre line: lo / hi corridor
        seen = tx > -1e8
        if int(seen.sum()) < int(c.lav_min_cells):
            self._lav_n["fb"] += 1
            return None
        x = xs[seen]
        t = tx[seen]
        z = float(pos[2])
        _lo = float(c.lav_lo)
        if float(c.lav_lo_far) >= 0.0:
            _lo = _lo + (float(c.lav_lo_far) - _lo) * (x / max(float(c.lav_look), 1e-6))
        smin_c = float(np.max((t + _lo - z) / x))
        smin = smin_c
        ts = top.max(axis=1)                  # the +-lav_half_w band: side floor lav_side_lo
        sseen = ts > -1e8
        smin_s = -9.0
        if float(c.lav_side_lo) > 0.0 and sseen.any():
            smin_s = float(np.max((ts[sseen] + float(c.lav_side_lo) - z) / xs[sseen]))
            smin = max(smin, smin_s)
        smx = float(np.min((t + float(c.lav_hi) - z) / x))
        sp = float(c.lav_kp) * (float(want) - float(agl)) / h
        if smin > smx:
            sl = smin
            self._lav_n["inf"] += 1
        else:
            sl = min(max(sp, smin), smx)
        sl = min(max(sl, -float(c.lav_smax)), float(c.lav_smax))
        self._lav_n["on"] += 1
        self._lav_dbg[i] = (smin_c, smin_s, smx, sp, sl, float(seen.sum()))
        return sl * h

    def _clearance(self, depth: np.ndarray, i: int) -> float:
        d = np.asarray(depth[i], dtype=np.float32)
        if d.ndim == 3:
            d = d[..., 0]
        h, w = d.shape
        lo, hi = 0.30, 0.70
        if self.cfg.clear_track_dir:
            # The camera is not always pointed where the drone is going (yaw scan,
            # avoidance turns), so read the clearance from the columns the flight
            # path actually crosses rather than from the middle of the frame.
            rel = float(getattr(self, "_clear_rel", 0.0) or 0.0)
            half = math.radians(float(self.cfg.clear_dir_half_deg))
            t = math.tan(math.radians(PD.AP_FOV_DEG) * 0.5)
            def _u(a):
                return 0.5 * (math.tan(max(-1.45, min(1.45, a))) / t + 1.0)
            lo, hi = sorted((_u(rel - half), _u(rel + half)))
            lo = min(max(lo, 0.0), 1.0)
            hi = min(max(hi, 0.0), 1.0)
            if hi - lo < 0.12:
                mid = 0.5 * (lo + hi)
                lo, hi = max(0.0, mid - 0.06), min(1.0, mid + 0.06)
        r0, r1 = ((int(h * 0.28), int(h * 0.52)) if not self.cfg.clear_band_deg
                  else self._pitch_rows(h, self.cfg.clear_band_deg))
        band = d[r0:r1, int(w * lo):max(int(w * hi), int(w * lo) + 1)]
        if band.size == 0:
            return PD.AP_DEPTH_MAX_M
        return float(np.percentile(band, 3.0)) * (PD.AP_DEPTH_MAX_M - 0.5) + 0.5
    def _column_clearance(self, depth: np.ndarray, i: int,
                          steer: bool = False) -> np.ndarray:
        d = np.asarray(depth[i], dtype=np.float32)
        if d.ndim == 3:
            d = d[..., 0]
        h = d.shape[0]
        r0, r1 = ((int(h * 0.28), int(h * 0.56))
                  if not (steer and self.cfg.steer_band_deg)
                  else self._pitch_rows(h, self.cfg.steer_band_deg))
        band = d[r0:r1, :]
        if band.size == 0:
            return np.full(d.shape[1], PD.AP_DEPTH_MAX_M, dtype=np.float32)
        return band.min(axis=0) * (PD.AP_DEPTH_MAX_M - 0.5) + 0.5
    def _update_memory(self, live, poses, rots, depth) -> None:
        """Fold this frame into each live drone's height memory."""
        if not self.cfg.mem_grid:
            return
        if self._mem is None:
            self._mem = HeightMemory(self.n)
        for oi, i in enumerate(live):
            try:
                self._mem.update(i, poses[oi], rots[oi], depth[i],
                                 PD.AP_DEPTH_MAX_M, PD.AP_FOV_DEG)
            except Exception:
                return

    def _mem_clear(self, i: int, pos, direction) -> float:
        if self._mem is None:
            return float(self.cfg.mem_lookahead)
        try:
            return self._mem.blocked_range(i, pos, direction, float(self.cfg.mem_lookahead))
        except Exception:
            return float(self.cfg.mem_lookahead)

    def _mem_steer(self, i: int, pos, v: np.ndarray) -> np.ndarray:
        """Steer on remembered obstacles when the camera is not covering the track.

        Unlike the depth frame the memory is not limited to the field of view, so a
        heading well off the camera axis -- even behind it -- can be judged here.
        """
        want = np.asarray(v, float)[:2]
        speed = float(np.linalg.norm(want))
        if speed < 1e-6:
            return v
        base = want / speed
        clear = self._mem_clear(i, pos, base)
        want_clear = float(self.cfg.mem_want_clear)
        if clear >= want_clear:
            return v
        best_dir, best_clear = base, clear
        ang0 = float(np.arctan2(base[1], base[0]))
        for deg in tuple(self.cfg.mem_steer_deg or ()):
            for sgn in (1.0, -1.0):
                a = ang0 + sgn * np.radians(float(deg))
                cand = np.array([np.cos(a), np.sin(a)])
                c = self._mem_clear(i, pos, cand)
                if c > best_clear + 1e-6:
                    best_dir, best_clear = cand, c
            if best_clear >= want_clear:
                break
        out = np.asarray(v, float).copy()
        scale = float(np.clip(best_clear / max(want_clear, 1e-6), 0.0, 1.0))
        slow = self.cfg.mem_brake + (1.0 - self.cfg.mem_brake) * scale
        out[:2] = best_dir * speed * slow
        return out

    def _use_free(self) -> bool:
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
        return bool(float(np.min(self._column_clearance(depth, i)))
                    >= self.cfg.open_clear)
    def _steer_around(self, depth: np.ndarray, i: int, v: np.ndarray) -> np.ndarray:
        c = self.cfg
        want = v[:2]
        speed = float(np.linalg.norm(want))
        if speed < 1e-6:
            return v
        cols = self._column_clearance(depth, i, steer=True)
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
    # ---- blind_guard OM (obstacle memory) ----
    def _om_on(self) -> bool:
        c = self.cfg
        if c.om_skip_route_open and self._om_skip_open():
            return False                                    # candF hazard i: GKC-pinned open seed
        return bool(c.om_maps) and ("forest" if self.is_forest else str(self.map_kind)) in tuple(c.om_maps)

    def _om_skip_open(self) -> bool:
        """candF hazard i: the seed is GKC-pinned open (not forest), so OM has nothing to guard."""
        return (not self.is_forest) and getattr(self, "_gkc_route", None) == "open"

    def _spd_gate_ok(self, i: int) -> bool:
        """UID 169's SPD predicate (_free_dir's _spd_ok) for drone i: False in APPROACH while the claim has fewer
        than spd_min_hits hits."""
        c = self.cfg
        if int(c.spd_min_hits) > 0 and self.d[i].phase == APPROACH and self.d[i].claim is not None:
            try:
                return int(self.pads[self.d[i].claim].hits) >= int(c.spd_min_hits)
            except Exception:                                    # noqa: BLE001
                return True
        return True

    def _om_cnt(self, k: str, v: int = 1) -> None:
        self._om_n[k] = self._om_n.get(k, 0) + v
        self.src_counts["om"] = self._om_n

    def _om_update(self, state, depth) -> None:
        """Fold every live drone's pooled depth cloud (world frame) into its voxel memory (newest point per voxel)."""
        c = self.cfg
        if self.step_i % max(1, int(c.om_every)) != 0:
            return
        hor = float(c.om_sec) / 0.02
        vox = max(float(c.om_vox), 0.05)
        if float(c.fsb_min_slack) >= 0.0:
            for _i in range(len(self.d)):
                if _i not in self._fsb_start:
                    self._fsb_start[_i] = np.asarray(state[_i, POS], float).copy()
        kr = float(c.om_keep_r)
        for i in range(len(self.d)):
            if self.d[i].phase not in (CLIMB, SEARCH, APPROACH):
                continue
            d = np.asarray(depth[i], np.float32)
            if d.ndim == 3:
                d = d[..., 0]
            dirs, eu = FD._cloud(d, int(c.free_grid), PD.AP_FOV_DEG, PD.AP_DEPTH_MAX_M, 0.5, True)
            valid = eu * dirs[:, 0] < PD.AP_DEPTH_MAX_M - 0.3
            R = rot_from_rpy(float(state[i, 3]), float(state[i, 4]), float(state[i, 5]))
            pos = np.asarray(state[i, POS], float)
            W = (R @ (dirs[valid] * eu[valid][:, None]).T).T
            W = W[np.hypot(W[:, 0], W[:, 1]) <= kr] + pos[None, :]
            old = self._om.get(i)
            if old is not None:
                P0, S0 = old
                ok = (float(self.step_i) - S0) <= hor
                P = np.vstack([W, P0[ok]]) if len(W) else P0[ok]
                S = np.concatenate([np.full(len(W), self.step_i, np.int64), S0[ok]])
            else:
                P = W
                S = np.full(len(W), self.step_i, np.int64)
            if len(P) == 0:
                self._om[i] = (np.zeros((0, 3), np.float64), np.zeros(0, np.int64))
                continue
            k = np.floor(P / vox).astype(np.int64) + (1 << 20)
            key = (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]
            _, idx = np.unique(key, return_index=True)
            self._om[i] = (P[idx], S[idx])

    def _om_pts(self, i: int, pos, R):
        """Remembered points (relative world, body) of drone i outside its current camera frustum, or (None, None)."""
        e = self._om_cache.get(i)
        if e is not None and e[0] == self.step_i:
            return e[1], e[2]
        c = self.cfg
        out = (None, None)
        m = self._om.get(i)
        if m is not None and len(m[0]):
            P, S = m
            rel = P - np.asarray(pos, float)[None, :]
            keep = ((np.hypot(rel[:, 0], rel[:, 1]) <= float(c.om_keep_r))
                    & ((float(self.step_i) - S) * 0.02 <= float(c.om_sec)))
            rel = rel[keep]
            if len(rel):
                B = rel @ np.asarray(R, float)
                # in the frustum AND beyond the depth floor (0.5 m planar; nearer surfaces read 0.5 m) = the frame has it
                inf = (B[:, 0] > 0.6) & (np.abs(B[:, 1]) <= B[:, 0]) & (np.abs(B[:, 2]) <= B[:, 0])
                rel, B = rel[~inf], B[~inf]
                if len(rel):
                    out = (rel, B)
        self._om_cache[i] = (self.step_i, out[0], out[1])
        return out

    def _om_near(self, i: int, pos, R, depth, rng: float, zlo: float, zhi: float, want_extra: bool = False):
        """Relative world points (memory outside the frustum + the current frame) within rng m horizontally and
        -zlo..+zhi m of the drone's height; om_hull adds eye-height copies of low-structure points."""
        c = self.cfg
        parts = []
        rel, _B = self._om_pts(i, pos, R)
        if rel is not None:
            parts.append(rel)
        d = np.asarray(depth[i], np.float32)
        if d.ndim == 3:
            d = d[..., 0]
        dirs, eu = FD._cloud(d, int(c.free_grid), PD.AP_FOV_DEG, PD.AP_DEPTH_MAX_M, 0.5, True)
        valid = eu * dirs[:, 0] < PD.AP_DEPTH_MAX_M - 0.3
        Wc = (np.asarray(R, float) @ (dirs[valid] * eu[valid][:, None]).T).T
        Wc = Wc[np.hypot(Wc[:, 0], Wc[:, 1]) <= rng + 1.5]
        if float(c.fsb_min_slack) >= 0.0:
            self._om_wc = (self.step_i, i, Wc)
        if c.om_frame or not want_extra:
            parts.append(Wc)                   # the current frame's own points (BG always; BMP only with om_frame)
        P = np.vstack(parts) if parts else np.zeros((0, 3))
        h = np.hypot(P[:, 0], P[:, 1])
        P = P[h <= rng + 1.5]
        X = np.zeros(len(P))
        if c.om_hull:
            Q = np.vstack([P, Wc]) if (want_extra and not c.om_frame) else P
            zw = float(pos[2]) + Q[:, 2]
            lo, hi = float(c.free_extrude_band[0]), float(c.free_extrude_band[1])
            low = (zw >= lo) & (zw <= hi) & (Q[:, 2] < -zlo)
            if float(c.om_hull_tall) > 0.0 and np.any(low):
                # HM2: only low points with a tall surface (world z > band top + om_hull_tall) within
                # om_hull_tall_r m: a podium / awning attached to a tower is where the convex hull bulges
                T = Q[zw > hi + float(c.om_hull_tall)]
                if len(T) == 0:
                    low[:] = False
                else:
                    li = np.flatnonzero(low)
                    dxy = np.hypot(Q[li, 0][:, None] - T[None, :, 0], Q[li, 1][:, None] - T[None, :, 1])
                    low[li[dxy.min(axis=1) > float(c.om_hull_tall_r)]] = False
            if np.any(low):
                L = Q[low].copy()
                L[:, 2] = 0.0
                P = np.vstack([P, L])
                X = np.concatenate([X, np.full(len(L), float(c.om_hull_margin))])
        if not len(P):
            return (P, X) if want_extra else P
        h = np.hypot(P[:, 0], P[:, 1])
        k = (h <= rng + X) & (P[:, 2] >= -zlo) & (P[:, 2] <= zhi)
        return (P[k], X[k]) if want_extra else P[k]

    def _om_blind(self, i: int, pos, R, v, depth):
        """BG: a blind (out-of-view) command tested against the memory; None = not blocked (c7 behaviour)."""
        c = self.cfg
        v = np.asarray(v, float)
        vh = v[:2].copy()
        sh = float(np.hypot(vh[0], vh[1]))
        if sh < 0.2:
            return None
        rr = float(c.om_blind_r)
        clr_lim = float(c.om_blind_clear)
        P = self._om_near(i, pos, R, depth, clr_lim + rr + 0.5, float(c.om_bump_zlo), float(c.om_bump_zhi))
        if not len(P):
            return None

        def tube(u):
            pr = P[:, 0] * u[0] + P[:, 1] * u[1]
            lat = np.abs(-P[:, 0] * u[1] + P[:, 1] * u[0])
            hit = (pr > 0.0) & (lat < rr)
            if not np.any(hit):
                return 99.0, hit
            return float(np.maximum(pr[hit] - np.sqrt(rr * rr - lat[hit] ** 2), 0.0).min()), hit

        u = vh / sh
        clr, hit = tube(u)
        if clr >= clr_lim:
            self._om_cnt("bg_clear")
            return None
        Hh = P[hit, :2]
        dh = np.maximum(np.hypot(Hh[:, 0], Hh[:, 1]), 0.05)
        m = (Hh / dh[:, None] / dh[:, None]).sum(axis=0)
        mn = float(np.hypot(m[0], m[1]))
        out = v.copy()
        if mn > 1e-6:
            m = m / mn
            t = float(vh @ m)
            if t > 0.0:
                vh = vh - t * m
        s2 = float(np.hypot(vh[0], vh[1]))
        clr2 = clr
        if s2 > 0.05:
            clr2, _h2 = tube(vh / s2)
        cap = float(c.om_blind_vmin) + (max(float(c.om_blind_speed), float(c.om_blind_vmin)) - float(c.om_blind_vmin)) \
            * float(np.clip(clr2 / max(clr_lim, 1e-3), 0.0, 1.0))
        if s2 > cap:
            vh = vh * (cap / s2)
        out[:2] = vh
        self._om_cnt("bg_block")
        return out

    def _fsb_val(self, key: str):
        """fsb: the value for this map ("forest" if is_forest else map_kind) of a (map, value, ...) key, or None."""
        vals = tuple(getattr(self.cfg, key, ()) or ())
        if not vals:
            return None
        mk = "forest" if self.is_forest else str(self.map_kind)
        for k in range(0, len(vals) - 1, 2):
            if str(vals[k]) == mk:
                return vals[k + 1]
        return None

    def _om_phases(self) -> tuple:
        """BMP phases: om_bump_phases_by_map entry for this map (comma string or list), else om_bump_phases."""
        ph = self._fsb_val("om_bump_phases_by_map") if self.cfg.om_bump_phases_by_map else None
        if ph is None:
            return tuple(self.cfg.om_bump_phases or ())
        if isinstance(ph, (list, tuple)):
            return tuple(str(s) for s in ph)
        return tuple(s.strip() for s in str(ph).split(",") if s.strip())

    def _fsb_min_margin(self, i: int, pos, R, sm: float) -> float:
        """FSB-M: update drone i's running-minimum clearance estimate (memory outside the frustum + current frame,
        3-D centre-to-point, after the take-off grace, committed fsb_min_lag s late) and return the capped margin."""
        c = self.cfg
        st = self._fsb_start.get(i)
        pos = np.asarray(pos, float)
        if st is not None and float(np.linalg.norm(pos - st)) >= (0.6 if self.is_forest else 1.0):
            parts = []
            rel, _B = self._om_pts(i, pos, R)
            if rel is not None:
                parts.append(rel)
            wc = self._om_wc
            if wc is not None and wc[0] == self.step_i and wc[1] == i:
                parts.append(wc[2])
            if parts:
                Q = np.vstack(parts)
                keep = (float(pos[2]) + Q[:, 2]) >= float(c.fsb_zmin)       # the ground plane is not scored
                dd = self.d[i]
                if float(c.fsb_pad_r) > 0.0 and dd.phase == APPROACH and dd.claim is not None:
                    _pxy = np.asarray(self.pads[dd.claim].xyz, float)[:2] - pos[:2]
                    keep &= np.hypot(Q[:, 0] - _pxy[0], Q[:, 1] - _pxy[1]) > float(c.fsb_pad_r)
                Q = Q[keep]
                if len(Q):
                    e = float(np.sqrt((Q * Q).sum(axis=1)).min())
                    if e < sm + 0.5:
                        self._fsb_pend.setdefault(i, []).append((self.step_i, e))
        pend = self._fsb_pend.get(i)
        if pend:
            lag = int(float(c.fsb_min_lag) / 0.02)
            k = 0
            while k < len(pend) and self.step_i - pend[k][0] >= lag:
                self._fsb_est[i] = min(self._fsb_est.get(i, 99.0), pend[k][1])
                k += 1
            if k:
                del pend[:k]
        est = self._fsb_est.get(i)
        if est is None:
            return sm
        m = min(sm, max(float(c.fsb_min_floor), est + float(c.fsb_min_slack)))
        if m < sm:
            self._om_cnt("fsb_mcap")
        return m

    def _om_bump(self, i: int, pos, rpy, v, depth, vel=None):
        """BMP: stopping-distance constraint of the horizontal command against remembered (outside the frustum) and
        seen points in a slab around the drone within om_bump_r m: the command's component toward point k is
        limited to v_ok(d_k) = a (-lat + sqrt(lat^2 + 2 (d_k - margin) / a)) (a = om_bump_a, lat = om_bump_lat,
        margin = om_bump_margin); inside the margin it becomes a push away of up to om_bump_push. The tangential
        part is kept (no time cost unless the drone heads into something it cannot stop for).
        FSB (om_bump_margin_by_map has this map): the per-map (safety) margin for points more than fsb_abeam_deg
        off the velocity, pad / ground filters and the stall guard (see the fsb block of AutopilotConfig)."""
        c = self.cfg
        R = rot_from_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        rb = float(c.om_bump_r)
        _sm = self._fsb_val("om_bump_margin_by_map") if c.om_bump_margin_by_map else None
        if _sm is not None:
            rb = max(rb, float(_sm) + 0.5)
        _zlo, _zhi = float(c.om_bump_zlo), float(c.om_bump_zhi)
        _fzlo, _fzhi = _zlo, _zhi
        _only = _latm = _cpa = False
        if _sm is not None:
            _mk = "forest" if self.is_forest else str(self.map_kind)
            _only = bool(c.fsb_only) or _mk in tuple(c.fsb_only_maps or ())
            _latm = bool(c.fsb_lat) or _mk in tuple(c.fsb_lat_maps or ())
            _cpa = _mk in tuple(c.fsb_cpa_maps or ())
            if float(c.fsb_zlo) >= 0.0:
                _fzlo, _fzhi = float(c.fsb_zlo), float(c.fsb_zhi)
            _dzm = self._fsb_val("fsb_dz_by_map") if c.fsb_dz_by_map else None
            if _dzm is not None:
                _fzlo = _fzhi = float(_dzm)
            _zlo, _zhi = max(_zlo, _fzlo), max(_zhi, _fzhi)
        if _sm is not None and _only and self._fsb_esc.get(i, -1) >= self.step_i:
            return v                                   # FSB-only map in the stall escape: no bumper (c14)
        P, X = self._om_near(i, pos, R, depth, rb, _zlo, _zhi, want_extra=True)
        _sme = None if _sm is None else float(_sm)
        if _sm is not None and float(c.fsb_min_slack) >= 0.0:
            _sme = self._fsb_min_margin(i, pos, R, float(_sm))
        _stall = float(c.om_hull_stall) > 0.0 and c.om_hull
        if _stall and len(P) and self._om_esc.get(i, -1) >= self.step_i:
            P, X = P[X <= 0.0], X[X <= 0.0]            # HM3 escape: lifted points ignored for a while
        _F = None
        if _sm is not None and len(P):
            # safety-eligible points: above the ground floor, not beside the claimed pad in APPROACH
            _F = (float(pos[2]) + P[:, 2]) >= float(c.fsb_zmin)
            dd = self.d[i]
            if float(c.fsb_pad_r) > 0.0 and dd.phase == APPROACH and dd.claim is not None:
                _pxy = np.asarray(self.pads[dd.claim].xyz, float)[:2] - np.asarray(pos, float)[:2]
                _F &= np.hypot(P[:, 0] - _pxy[0], P[:, 1] - _pxy[1]) > float(c.fsb_pad_r)
        if not len(P):
            return v
        H = P[:, :2]
        dh = np.hypot(H[:, 0], H[:, 1])
        ok = dh > 0.02
        H, dh, X = H[ok], dh[ok], X[ok]
        if not len(dh):
            return v
        U = H / dh[:, None]
        Z = P[ok, 2] if _sm is not None else None
        if _F is not None:
            _F = _F[ok]
        a = max(float(c.om_bump_a), 0.3)
        lat = max(float(c.om_bump_lat), 0.0)
        mg = float(c.om_bump_margin) + X
        SP = None
        ref = None
        if _sm is not None and self._fsb_esc.get(i, -1) < self.step_i:
            SP = np.ones(len(dh), bool)
            if float(c.fsb_abeam_deg) > 0.0 or _latm or _cpa:
                for _rv in (vel, v):
                    if _rv is None:
                        continue
                    _r2 = np.asarray(_rv, float)[:2]
                    _rn = float(np.hypot(_r2[0], _r2[1]))
                    if _rn >= float(c.fsb_vref_min):
                        ref = _r2 / _rn
                        break
                if float(c.fsb_abeam_deg) > 0.0:
                    if ref is None:
                        SP[:] = False
                    else:
                        SP = (U @ ref) < math.cos(math.radians(float(c.fsb_abeam_deg)))
            SP = SP & _F & (Z >= -_fzlo) & (Z <= _fzhi)
            _bz = (Z >= -float(c.om_bump_zlo)) & (Z <= float(c.om_bump_zhi))
            if not np.all(SP | _bz):                    # points only the safety slab admitted and not safety points
                _k2 = SP | _bz
                U, dh, X, mg, SP, Z = U[_k2], dh[_k2], X[_k2], mg[_k2], SP[_k2], Z[_k2]
                if not len(dh):
                    return v
            if c.fsb_dz3d:
                _msp = np.sqrt(np.maximum(_sme * _sme - Z * Z, 0.0)) + X
            else:
                _msp = _sme + X
            mg = np.where(SP, np.maximum(mg, _msp), mg)
            if _only:
                U, dh, X, mg, Z, SP = U[SP], dh[SP], X[SP], mg[SP], Z[SP], SP[SP]
                if not len(dh):
                    return v
        elif _sm is not None:                           # stall escape (not FSB-only): base slab, base margin
            _bz = (Z >= -float(c.om_bump_zlo)) & (Z <= float(c.om_bump_zhi))
            U, dh, X, mg = U[_bz], dh[_bz], X[_bz], mg[_bz]
            if not len(dh):
                return v
        room = np.maximum(dh - mg, 0.0)
        vok = a * (-lat + np.sqrt(lat * lat + 2.0 * room / a))
        inside = dh < mg
        _pv = self._fsb_val("om_bump_push_by_map") if c.om_bump_push_by_map else None
        _push = np.where(X > 0.0, float(c.om_hull_push), float(c.om_bump_push) if _pv is None else float(_pv))
        vok = np.where(inside, -_push * (mg - dh) / np.maximum(mg, 1e-3), vok)
        out = np.asarray(v, float).copy()
        vh = out[:2].copy()
        vh0 = float(np.hypot(vh[0], vh[1]))
        fired = False
        hull_bind = False
        safe_bind = False
        if _cpa and SP is not None and ref is not None and np.any(SP):
            # FSB-P: lateral bounds from the closest point of approach of every safety point
            nL = np.array([-ref[1], ref[0]])
            Hs = U[SP] * dh[SP][:, None]
            xs = Hs @ ref
            ys = Hs @ nL
            ms_ = mg[SP]
            s_ref = max(float(vh @ ref), 0.0)
            vlat = float(vh @ nL)
            ahead = (xs > 0.05) & (xs < float(c.fsb_cpa_x)) & (np.abs(ys) < ms_)
            side = (~ahead) & (xs <= 0.05) & (dh[SP] < ms_)
            need = np.zeros(len(xs))
            need[ahead] = (ms_[ahead] - np.abs(ys[ahead])) * s_ref / xs[ahead]
            need[side] = _push[SP][side] * (ms_[side] - dh[SP][side]) / np.maximum(ms_[side], 1e-3)
            act = ahead | side
            if np.any(act):
                rgt = act & (ys <= 0.0)                 # point on the right -> v_lat >= need
                lft = act & (ys > 0.0)                  # point on the left  -> v_lat <= -need
                lo = float(need[rgt].max()) if np.any(rgt) else -1e9
                hi = -float(need[lft].max()) if np.any(lft) else 1e9
                new = min(max(vlat, lo), hi) if lo <= hi else 0.5 * (lo + hi)
                cap = max(float(c.fsb_cpa_vmax), abs(vlat))
                new = float(np.clip(new, -cap, cap))
                if abs(new - vlat) > 1e-3:
                    vh = vh + (new - vlat) * nL
                    fired = True
                    safe_bind = True
            kb = ~SP                                    # the stopping-distance loop keeps the other points
            U, vok, X, SP = U[kb], vok[kb], X[kb], SP[kb]
        for _k in range(4) if len(U) else ():
            ex = U @ vh - vok
            j = int(np.argmax(ex))
            if float(ex[j]) <= 1e-3:
                break
            _sj = SP is not None and bool(SP[j])
            if _sj and _latm and ref is not None:
                # FSB lateral: remove the excess with a velocity change across ref (away from point j) only
                nrm = np.array([-ref[1], ref[0]])
                if float(U[j] @ nrm) > 0.0:
                    nrm = -nrm
                sn = -float(U[j] @ nrm)
                if sn > 0.3:
                    vh = vh + (float(ex[j]) / sn) * nrm
                else:
                    vh = vh - float(ex[j]) * U[j]
            else:
                vh = vh - float(ex[j]) * U[j]
            fired = True
            hull_bind = hull_bind or bool(X[j] > 0.0)
            if _sj:
                safe_bind = True
        if _sm is not None and float(c.fsb_stall) > 0.0:
            # FSB stall guard: < fsb_stall_d m of progress over fsb_stall s while the safety margin bound the
            # command -> safety margin off (om_bump_margin for every point) for fsb_esc s
            st = self._fsb_stall.get(i)
            p2 = np.asarray(pos, float)[:2]
            if st is None or float(np.hypot(*(p2 - st[0]))) > float(c.fsb_stall_d):
                st = [p2.copy(), self.step_i, -1]
                self._fsb_stall[i] = st
            if safe_bind:
                st[2] = self.step_i
            if (st[2] >= st[1] and (self.step_i - st[1]) * 0.02 >= float(c.fsb_stall)):
                self._fsb_esc[i] = self.step_i + int(float(c.fsb_esc) / 0.02)
                self._fsb_stall[i] = [p2.copy(), self.step_i, -1]
                self._om_cnt("fsb_esc")
        if safe_bind:
            self._om_cnt("fsb_bind")
            self._om_cnt("fsb_" + str(self.d[i].phase)[:3])
            self._om_cnt("fsb_dv_mm", int(round(1000.0 * max(0.0, vh0 - float(np.hypot(vh[0], vh[1]))))))
        if _stall:
            # HM3: the drone made < 1 m of progress over om_hull_stall s while lifted points bound its command
            # -> ignore them for om_hull_esc s (base behaviour) instead of hovering in front of the podium
            st = self._om_stall.get(i)
            p2 = np.asarray(pos, float)[:2]
            if st is None or float(np.hypot(*(p2 - st[0]))) > 1.0:
                st = [p2.copy(), self.step_i, -1]
                self._om_stall[i] = st
            if hull_bind:
                st[2] = self.step_i
            if (st[2] >= st[1] and (self.step_i - st[1]) * 0.02 >= float(c.om_hull_stall)):
                self._om_esc[i] = self.step_i + int(float(c.om_hull_esc) / 0.02)
                self._om_stall[i] = [p2.copy(), self.step_i, -1]
                self._om_cnt("hull_esc")
        if not fired:
            return v
        out[:2] = vh
        self._om_cnt("bump")
        return out

    # ---- candHc OM2 fleet obstacle memory (AutopilotConfig om2_*; team/autopilot/omgrid.py) ----
    def _om2_on(self) -> bool:
        c = self.cfg
        if c.om2_skip_open and self._om_skip_open():
            return False                                    # GKC-pinned open seed: nothing to guard
        mk = "forest" if self.is_forest else str(self.map_kind)
        if c.om2_route_city and mk == "city" and str(self._mass_route_kind()) != "city":
            return False                                    # routed as open: no building to guard
        return mk in tuple(c.om2_maps)

    def _om2_mk(self) -> str:
        return "forest" if self.is_forest else str(self.map_kind)

    def _om2_zrel(self) -> float:
        vals = tuple(self.cfg.om2_zrel_by_map or ())
        mk = self._om2_mk()
        for k in range(0, len(vals) - 1, 2):
            if str(vals[k]) == mk:
                return float(vals[k + 1])
        return float(self.cfg.om2_zrel)

    def _om2_cnt(self, k: str, v: int = 1) -> None:
        self._om2_n[k] = self._om2_n.get(k, 0) + v
        self.src_counts["om2"] = self._om2_n

    def _om2_update(self, state, depth) -> None:
        """Fold the frames of up to om2_upd_max live drones (stalest first) into the fleet grid, on non-detector
        steps. Points within om2_mask_r of a moving teammate are dropped (no ghost obstacles)."""
        c = self.cfg
        if self._om2 is None:
            from team.autopilot.omgrid import ObstacleGrid
            self._om2 = ObstacleGrid(half=float(c.om2_half), cell=float(c.om2_cell))
        if c.om2_upd_skip_det and self.step_i % max(1, int(c.detect_every)) == 0:
            return
        n = len(self.d)
        live = [k for k in range(n) if self.d[k].phase in (CLIMB, SEARCH, APPROACH, DESCEND)]
        if not live:
            return
        live.sort(key=lambda k: (self._om2_last.get(k, -10 ** 9), k))
        movers = [k for k in range(n) if self.d[k].phase != DONE]
        zr = self._om2_zrel()
        for i in live[:max(1, int(c.om2_upd_max))]:
            pos = np.asarray(state[i, POS], float)
            R = rot_from_rpy(float(state[i, 3]), float(state[i, 4]), float(state[i, 5]))
            mask = [np.asarray(state[k, POS], float) for k in movers if k != i]
            nh, nc = self._om2.add_frame(pos, R, depth[i], PD.AP_DEPTH_MAX_M, PD.AP_FOV_DEG,
                                         hit_grid=int(c.om2_hit_grid),
                                         carve_grid=int(c.om2_carve_grid) if float(c.om2_unk_speed) >= 0.0 else 0,
                                         carve_step=float(c.om2_carve_step), carve_max=float(c.om2_carve_max),
                                         carve_el=tuple(c.om2_carve_el), mask=mask,
                                         mask_r=float(c.om2_mask_r),
                                         zmin=(float(pos[2]) + zr) if zr > -90.0 else None)
            self._om2_last[i] = self.step_i
            self._om2_cnt("upd")
            self._om2_cnt("hit_pts", nh)
            self._om2_cnt("carve_pts", nc)

    def _om2_lift(self, G, tp, cx, cy, ok, zc, foot, kh, te, p, reach) -> np.ndarray:
        """OM2 hull lift: city collision shapes are convex hulls of each mesh, so above an awning / porch /
        podium roof (a low structure cell, foot <= top <= zc) next to a taller part of the same building the
        hull fills the air the camera shows as free (building-n: the hull climbs ~2.5 m per m from the 2.5 m
        awning edge to the 12.4 m block). A low cell gets te = max(top, top' - kh * d) over the cells within
        om2_hull_r m whose top reaches the drone's height, when every cell on the segment between them is
        structure too (top >= foot: no street gap). Writes te in place; returns the lifted mask."""
        c = self.cfg
        low = ok & (tp >= foot) & (tp <= zc)
        lifted = np.zeros(len(tp), bool)
        if not np.any(low):
            return lifted
        li = np.flatnonzero(low)
        dl = np.hypot(cx[li] - p[0], cy[li] - p[1])
        li = li[np.argsort(dl)[:max(1, int(c.om2_hull_max_low))]]
        rl = float(c.om2_hull_r)
        cell = G.cell
        I0, I1, J0, J1 = G.window(float(p[0]), float(p[1]), reach + rl)
        T = G.top[I0:I1, J0:J1]
        ti, tj = np.nonzero(T > zc)
        if not len(ti):
            return lifted
        tcx = (I0 + ti + 0.5) * cell - G.half
        tcy = (J0 + tj + 0.5) * cell - G.half
        tt = T[ti, tj].astype(float)
        lcx, lcy = cx[li], cy[li]
        mx, my = float(lcx.mean()), float(lcy.mean())
        near = np.argsort(np.hypot(tcx - mx, tcy - my))[:max(1, int(c.om2_hull_max_tall))]
        tcx, tcy, tt = tcx[near], tcy[near], tt[near]
        dx = tcx[None, :] - lcx[:, None]
        dy = tcy[None, :] - lcy[:, None]
        dist = np.hypot(dx, dy)
        lv = np.where(dist <= rl, tt[None, :] - kh * dist, -1e9)
        pi, pj = np.nonzero(lv > zc)
        if not len(pi):
            return lifted
        ns = max(1, int(rl / cell))
        fr = (np.arange(ns, dtype=float) + 1.0) / (ns + 1.0)
        sx = lcx[pi][:, None] + dx[pi, pj][:, None] * fr[None, :]
        sy = lcy[pi][:, None] + dy[pi, pj][:, None] * fr[None, :]
        gi, gj, _okg = G._ij(sx, sy)
        gi = np.clip(gi, 0, G.size - 1)
        gj = np.clip(gj, 0, G.size - 1)
        conn = (G.top[gi, gj] >= foot).all(axis=1)
        if not np.any(conn):
            return lifted
        best = np.full(len(li), -1e9)
        np.maximum.at(best, pi[conn], lv[pi[conn], pj[conn]])
        up = best > te[li]
        te[li[up]] = best[up]
        lifted[li[up]] = best[up] > tp[li[up]] + 0.3
        return lifted

    def _om2_h3_fit(self, cx, cy, tops, half_cell: float = 0.0):
        """H3: upward and side facets of the 3-D convex hull of the cell columns (ground point at the cell centre,
        top point(s) at the seen top: at the 4 cell corners when half_cell > 0, so a steep skirt over a narrow low
        edge - a 0.5 m canopy under a 10-18 m wall - is not flattened onto the edge cell's centre); False when the
        points are degenerate (one row of cells) or qhull fails."""
        try:
            from scipy.spatial import ConvexHull
            if half_cell > 0.0:
                h = float(half_cell)
                tp = [np.stack([cx + sx, cy + sy, tops], 1) for sx in (-h, h) for sy in (-h, h)]
            else:
                tp = [np.stack([cx, cy, tops], 1)]
            P = np.concatenate(tp + [np.stack([cx, cy, np.zeros(len(cx))], 1)], 0)
            eq = ConvexHull(P).equations                        # rows (nx, ny, nz, d): n . x + d <= 0 inside
        except Exception:                                        # noqa: BLE001
            return False
        up = eq[:, 2] > 1e-6
        if not np.any(up):
            return False
        return (eq[up].copy(), eq.copy())

    def _om2_h3_env(self, G, px: float, py: float, reach: float, win):
        """H3 envelope over the filter window win = (i0, i1, j0, j1): the height of the convex hull of each
        remembered building (om2_h3_* keys) above every window cell inside it, -1e9 elsewhere; None when no
        building hull reaches the window. Hulls are cached per component (bbox, cells, coarse top sum)."""
        c = self.cfg
        i0, i1, j0, j1 = win
        I0, I1, J0, J1 = G.window(px, py, reach + float(c.om2_h3_ext))
        T = G.top[I0:I1, J0:J1]
        S = T >= float(c.om2_h3_foot)
        if not np.any(S):
            return None
        from scipy import ndimage
        lab, nl = ndimage.label(S, structure=np.ones((3, 3), bool))
        if nl == 0:
            return None
        env = None
        cell, half = G.cell, G.half
        used = 0
        objs = ndimage.find_objects(lab)
        # components nearest the drone first (bbox centre distance); at most om2_h3_max_comp per call
        order = []
        for k, sl in enumerate(objs, start=1):
            if sl is None:
                continue
            a0, a1 = sl[0].start + I0, sl[0].stop + I0
            b0, b1 = sl[1].start + J0, sl[1].stop + J0
            if a1 <= i0 or a0 >= i1 or b1 <= j0 or b0 >= j1:
                continue                                        # the hull lies inside the component's bbox
            dc = math.hypot((0.5 * (a0 + a1)) * cell - half - px, (0.5 * (b0 + b1)) * cell - half - py)
            order.append((dc, k, sl, a0, a1, b0, b1))
        order.sort(key=lambda r: r[0])
        for _dc, k, sl, a0, a1, b0, b1 in order:
            if used >= max(1, int(c.om2_h3_max_comp)):
                break
            m = lab[sl] == k
            ncell = int(m.sum())
            if ncell < int(c.om2_h3_min_cells) or ncell > int(c.om2_h3_max_cells):
                continue
            tops = T[sl][m].astype(float)
            if float(tops.max()) < float(c.om2_h3_min_top):
                continue
            used += 1
            key = (a0, a1, b0, b1, ncell, int(float(tops.sum()) / 4.0))
            hull = self._om2_h3c.get(key)
            if hull is None:
                mi, mj = np.nonzero(m)
                hcx = (a0 + mi + 0.5) * cell - half
                hcy = (b0 + mj + 0.5) * cell - half
                hull = self._om2_h3_fit(hcx, hcy, tops,
                                        0.5 * cell if bool(c.om2_h3_top_corners) else 0.0)
                self._om2_cnt("h3_fit")
                if len(self._om2_h3c) >= 64:
                    self._om2_h3c.pop(next(iter(self._om2_h3c)))
                self._om2_h3c[key] = hull
            if hull is False:
                self._om2_cnt("h3_degen")
                continue
            eu, ea = hull
            w0, w1 = max(a0, i0), min(a1, i1)
            v0, v1 = max(b0, j0), min(b1, j1)
            if w1 <= w0 or v1 <= v0:
                continue
            gx = (np.arange(w0, w1) + 0.5) * cell - half
            gy = (np.arange(v0, v1) + 0.5) * cell - half
            X, Y = np.meshgrid(gx, gy, indexing="ij")
            xy = np.stack([X.ravel(), Y.ravel()], 1)
            inside = ((xy @ ea[:, :2].T) + ea[:, 3][None, :] <= 1e-6).all(axis=1)   # the hull's z = 0 slice
            if not np.any(inside):
                continue
            h = np.min(-((xy[inside] @ eu[:, :2].T) + eu[:, 3][None, :]) / eu[:, 2][None, :], axis=1)
            if env is None:
                env = np.full((i1 - i0, j1 - j0), -1e9)
            sub = env[w0 - i0:w1 - i0, v0 - j0:v1 - j0].reshape(-1)
            sub[inside] = np.maximum(sub[inside], h)
            env[w0 - i0:w1 - i0, v0 - j0:v1 - j0] = sub.reshape(w1 - w0, v1 - v0)
        return env

    def _om2_turn(self, dd, yaw_now: float, head: float) -> float:
        """Camera heading toward `head` that the env turns to consistently. The env clamps the yaw setpoint to
        +-pi*dt around the current yaw along the SHORTEST way, so a target near 180 deg away (a +-50 deg yaw scan
        sweeping across it) flips the turn direction step to step and the camera stalls facing away from the
        travel (traced city crashes: 2-3 s of blind flight at 2.8 m/s). Beyond om2_turn_lead deg the command
        leads the current yaw by om2_turn_lead in one direction, held until the rest of the turn is short."""
        lead = math.radians(float(self.cfg.om2_turn_lead))
        dlt = (head - yaw_now + math.pi) % (2.0 * math.pi) - math.pi
        if abs(dlt) <= lead:
            dd.om2_turn_sgn = 0.0
            return head
        sgn = float(dd.om2_turn_sgn) if dd.om2_turn_sgn != 0.0 else (1.0 if dlt > 0.0 else -1.0)
        dd.om2_turn_sgn = sgn
        self._om2_cnt("turn_lead")
        return yaw_now + sgn * lead

    def _alift_tops(self, G, x: float, y: float, reach: float):
        """C-RC1 ALIFT: (window, remembered tops over it, max-combined with the H3 hull envelope as _om2_filter does)."""
        c = self.cfg
        win = G.window(float(x), float(y), float(reach))
        i0, i1, j0, j1 = win
        T = G.top[i0:i1, j0:j1].astype(float)
        if c.om2_h3 and (not c.om2_h3_maps or self._om2_mk() in tuple(c.om2_h3_maps)) and T.size:
            try:
                env = self._om2_h3_env(G, float(x), float(y), float(reach), win)
            except Exception:                                    # noqa: BLE001
                env = None
            if env is not None:
                T = np.maximum(T, env)
        return win, T

    def _alift_overhead(self, pos) -> bool:
        """C-RC1 ALIFT: a remembered surface (or H3 hull) within 0.3 m reaches above the drone: no climb."""
        G = self._om2
        if G is None or not self._om2_on():
            return False
        z = float(pos[2])
        _w, T = self._alift_tops(G, float(pos[0]), float(pos[1]), 0.3)
        hit = bool(T.size) and bool(np.any(T > z + 0.3))
        if hit:
            self._b6_n["alift_ovh"] = self._b6_n.get("alift_ovh", 0) + 1
        return hit

    def _alift_line_top(self, pos, pad, base_z: float):
        """C-RC1 ALIFT: highest remembered top (H3 hull included) within 0.75 m of the drone -> pad segment (samples
        every 0.5 m, none within om2_pad_r of the pad); None when the grid is off or it is not above base_z - 0.47."""
        G = self._om2
        if G is None or not self._om2_on():
            return None
        c = self.cfg
        p0 = np.asarray(pos, float)[:2]
        p1 = np.asarray(pad.xyz, float)[:2]
        seg = p1 - p0
        L = float(np.hypot(seg[0], seg[1]))
        mid = 0.5 * (p0 + p1)
        (i0, i1, j0, j1), T = self._alift_tops(G, float(mid[0]), float(mid[1]), 0.5 * L + 1.0)
        if not T.size:
            return None
        s = np.arange(0.0, L + 1e-9, 0.5)
        u = seg / L if L > 1e-9 else np.zeros(2)
        sx, sy = p0[0] + u[0] * s, p0[1] + u[1] * s
        keep = np.hypot(sx - p1[0], sy - p1[1]) > float(c.om2_pad_r)
        if not np.any(keep):
            return None
        sx, sy = sx[keep], sy[keep]
        cx = (i0 + np.arange(i1 - i0) + 0.5) * G.cell - G.half
        cy = (j0 + np.arange(j1 - j0) + 0.5) * G.cell - G.half
        ci, cj = np.nonzero(T > -1e8)
        if not len(ci):
            return None
        tv = T[ci, cj]
        near = np.hypot(cx[ci][None, :] - sx[:, None], cy[cj][None, :] - sy[:, None]) <= 0.75
        if not np.any(near):
            return None
        m = float(np.max(np.where(near, tv[None, :], -1e9)))
        return m if m > float(base_z) - 0.47 else None

    def _om2_filter(self, i: int, dd, pos, rpy, v, pad):
        """OM2 filter (last stage before the slew): stopping-distance bounds of the command against remembered
        surfaces outside the camera frustum (horizontal: toward cells reaching the drone's height, tangential
        motion kept; vertical: the sink rate over / toward lower surfaces), then the optional unknown-space cap."""
        from team.autopilot.omgrid import EYE_FWD_M, EYE_UP_M, stop_speed
        c = self.cfg
        G = self._om2
        if G is None:
            return v
        v = np.asarray(v, float)
        vh = v[:2].copy()
        sp = float(np.hypot(vh[0], vh[1]))
        vz = float(v[2])
        sink = bool(c.om2_vz) and vz < -0.05
        unk = float(c.om2_unk_speed) >= 0.0 and sp > float(c.om2_unk_speed) + 1e-6
        if sp < 0.05 and not sink:
            return v
        self._om2_cnt("calls")
        R = rot_from_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        p = np.asarray(pos, float)
        eye = p + R[:, 0] * EYE_FWD_M + R[:, 2] * EYE_UP_M
        rb, mh, mv = float(c.om2_r), float(c.om2_mh), float(c.om2_mv)
        a, lat = max(float(c.om2_a), 0.3), max(float(c.om2_lat), 0.0)
        rh = rb + mh
        z = float(p[2])
        zc = z - rb - mv                    # a surface above this height is in the drone's way horizontally
        cell = G.cell
        vt, vmin = float(c.om2_view_tan), float(c.om2_view_min)
        reach = rh + 0.5 + sp * lat + sp * sp / (2.0 * a)
        if sink:
            reach = max(reach, rh + 0.5 + sp * float(c.om2_vt))
        reach = min(reach, max(float(c.om2_look), rh + 0.5))
        i0, i1, j0, j1 = G.window(float(p[0]), float(p[1]), reach)
        vh2 = vh.copy()
        vz2 = vz
        flags = 0
        if i1 > i0 and j1 > j0:
            top = G.top[i0:i1, j0:j1]
            dz_look = (abs(vz) * float(c.om2_vt) + 0.5) if sink else 0.0
            _mk = self._om2_mk()
            colm = _mk not in tuple(c.om2_interval_maps or ())
            kh = float(c.om2_hull_k) if (not c.om2_hull_maps or _mk in tuple(c.om2_hull_maps)) else 0.0
            foot = float(c.om2_hull_foot)
            if kh > 0.0:
                sel = (top > (zc - dz_look)) | ((top >= foot) & (top <= zc))
            else:
                sel = top > (zc - dz_look)
            h3 = None
            if c.om2_h3 and (not c.om2_h3_maps or _mk in tuple(c.om2_h3_maps)):
                try:
                    h3 = self._om2_h3_env(G, float(p[0]), float(p[1]), reach, (i0, i1, j0, j1))
                except Exception:                                # noqa: BLE001
                    self._om2_cnt("err_h3")
                    h3 = None
                if h3 is not None:
                    sel = sel | (h3 > (zc - dz_look))           # hull cells (street side fills included)
            ii, jj = np.nonzero(sel)
            if len(ii):
                tp = top[ii, jj].astype(float)
                if c.om2_col and colm:
                    bt = np.full(len(tp), -1e9)
                else:
                    bt = G.bot[i0:i1, j0:j1][ii, jj].astype(float)
                cx = (i0 + ii + 0.5) * cell - G.half
                cy = (j0 + jj + 0.5) * cell - G.half
                hx, hy = cx - p[0], cy - p[1]
                dc = np.hypot(hx, hy)
                ok = (dc <= reach + 0.5 * cell) & (dc > 1e-6)
                dn = np.maximum(dc - 0.5 * cell, 0.0)          # to the near side of the cell
                if pad is not None and dd.phase == APPROACH and float(c.om2_pad_r) > 0.0:
                    pxy = np.asarray(pad.xyz, float)[:2]
                    ok &= np.hypot(cx - pxy[0], cy - pxy[1]) > float(c.om2_pad_r)
                te = tp.copy()
                lifted = np.zeros(len(tp), bool)
                if kh > 0.0:
                    lifted = self._om2_lift(G, tp, cx, cy, ok, zc, foot, kh, te, p, reach)
                l3 = None
                if h3 is not None:
                    hv = h3[ii, jj]
                    te = np.maximum(te, hv)
                    l3 = hv > tp + 0.3                          # H3 hull the camera cannot draw
                hz = (te > zc) & (bt < z + rb + mv)
                vzc = (te <= zc) & (te > zc - dz_look)
                zq = np.where(hz, np.minimum(np.maximum(z, bt), tp), tp)
                B = (np.stack([cx, cy, zq], axis=1) - eye[None, :]) @ R
                inv = (B[:, 0] > vmin) & (np.abs(B[:, 1]) <= vt * B[:, 0]) & (np.abs(B[:, 2]) <= vt * B[:, 0])
                inv &= ~lifted                                  # a hull above a low roof is never drawn
                if l3 is not None:
                    inv &= ~l3                                  # nor an H3 hull side fill
                if not c.om2_inview:
                    ok &= ~inv
                if np.any(lifted & ok):
                    self._om2_cnt("cells_lift", int((lifted & ok).sum()))
                if l3 is not None and np.any(l3 & ok):
                    self._om2_cnt("cells_h3", int((l3 & ok).sum()))
                U = np.stack([hx, hy], axis=1) / np.maximum(dc, 1e-6)[:, None]
                H = ok & hz
                if np.any(H):
                    Uh = U[H]
                    room = dn[H] - rh
                    vok = np.where(room > 0.0, stop_speed(room, a, lat),
                                   -float(c.om2_push) * np.minimum(1.0, -room / max(rh, 1e-3)))
                    l3h = l3[H] if l3 is not None else None
                    for _k in range(6):
                        ex = Uh @ vh2 - vok
                        j = int(np.argmax(ex))
                        if float(ex[j]) <= 1e-3:
                            break
                        vh2 = vh2 - float(ex[j]) * Uh[j]
                        flags |= 1
                        if l3h is not None and bool(l3h[j]):
                            flags |= 8                          # diag: an H3 hull cell bound the command
                    self._om2_cnt("cells_h", int(H.sum()))
                if sink:
                    V = ok & vzc
                    if np.any(V):
                        roomz = zc - te[V]
                        dh = dn[V] - rh
                        cs = U[V] @ vh2
                        over = dh <= 0.0
                        t_hit = np.where(over, 0.0, dh / np.maximum(cs, 1e-3))
                        act = over | ((cs > 0.05) & (t_hit <= float(c.om2_vt)))
                        if np.any(act):
                            lim = -float(stop_speed(roomz[act], max(float(c.om2_az), 0.3), lat).min())
                            if vz2 < lim:
                                vz2 = lim
                                flags |= 2
        if unk:
            spn = float(np.hypot(vh2[0], vh2[1]))
            if spn > float(c.om2_unk_speed) + 1e-6:
                u = vh2 / spn
                D = rh + spn * lat + spn * spn / (2.0 * a) + 0.25
                s = np.arange(0.25, D + 1e-6, 0.25)
                px, py = p[0] + u[0] * s, p[1] + u[1] * s
                gi, gj, okg = G._ij(px, py)
                gi, gj = np.clip(gi, 0, G.size - 1), np.clip(gj, 0, G.size - 1)
                known = (G.free[gi, gj] <= z + float(c.om2_unk_margin)) | (G.top[gi, gj] > zc) | ~okg
                Bs = (np.stack([px, py, np.full(len(s), z)], axis=1) - eye[None, :]) @ R
                invs = (Bs[:, 0] > vmin) & (np.abs(Bs[:, 1]) <= vt * Bs[:, 0]) & (np.abs(Bs[:, 2]) <= vt * Bs[:, 0])
                unkn = ~known & ~invs
                if np.any(unkn):
                    s_u = float(s[int(np.argmax(unkn))])
                    cap = max(float(c.om2_unk_speed), float(stop_speed(s_u - rb - 0.1, a, lat)))
                    if spn > cap:
                        vh2 = vh2 * (cap / spn)
                        flags |= 4
                        if c.om2_unk_turn and not c.om2_shadow:
                            dd.om2_yaw = self._om2_turn(dd, float(rpy[2]), float(math.atan2(u[1], u[0])))
        if (float(c.om2_turn_blind_deg) > 0.0 and not c.om2_shadow and dd.om2_yaw is None
                and float(np.hypot(vh2[0], vh2[1])) > float(c.om2_turn_blind_v)):
            # BLIND turn: the command flies far outside the camera view -> turn the camera to it the consistent way
            # (no yaw-scan / wrap stall); nothing else changes
            head = float(math.atan2(vh2[1], vh2[0]))
            off = abs((head - float(rpy[2]) + math.pi) % (2.0 * math.pi) - math.pi)
            if off > math.radians(float(c.om2_turn_blind_deg)):
                dd.om2_yaw = self._om2_turn(dd, float(rpy[2]), head)
                self._om2_cnt("turn_blind")
            elif dd.om2_turn_sgn != 0.0 and off < math.radians(float(c.om2_turn_lead)):
                dd.om2_turn_sgn = 0.0
        if not flags:
            return v
        sp2 = float(np.hypot(vh2[0], vh2[1]))
        self._om2_fire[i] = (self.step_i, flags, round(sp, 3), round(sp2, 3), round(vz, 3), round(vz2, 3))
        _hl = self._om2_hist.setdefault(i, [])
        if len(_hl) < 3000:
            _hl.append((self.step_i, flags))                     # diag: binding steps per drone
        if flags & 1:
            self._om2_cnt("h_bind")
        if flags & 2:
            self._om2_cnt("v_bind")
        if flags & 4:
            self._om2_cnt("unk_bind")
        if flags & 8:
            self._om2_cnt("h3_bind")
        self._om2_cnt("bind_" + str(dd.phase)[:3])
        self._om2_cnt("dv_mm", int(round(1000.0 * (max(0.0, sp - sp2) + max(0.0, vz2 - vz)))))
        if c.om2_shadow:
            self._om2_cnt("shadow")
            return v
        out = v.copy()
        out[:2] = vh2
        out[2] = vz2
        return out

    def _free_margin(self, _fk: str) -> float:
        """free_direction tube margin for this map (free_margin / mountain / free_margin_by_map)."""
        c = self.cfg
        _margin = c.free_margin_mountain if self.map_kind == "mountain" else c.free_margin
        _mbm = tuple(getattr(c, "free_margin_by_map", ()) or ())
        _mk = (("forest" if self.is_forest else str(self._mass_route_kind()))
               if getattr(c, "free_margin_route", False) else _fk)
        for _k2 in range(0, len(_mbm) - 1, 2):
            if str(_mbm[_k2]) == _mk:
                _margin = float(_mbm[_k2 + 1])
        return _margin

    def _free_dir(self, depth: np.ndarray, i: int, v: np.ndarray,
                  rpy: np.ndarray, pos: np.ndarray) -> np.ndarray:
        c = self.cfg
        speed = float(np.linalg.norm(v))
        if speed < 1e-6:
            return v
        R = rot_from_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        _fk = "forest" if self.is_forest else str(self.map_kind)
        if (c.climb_free_guard and self._refine_state is not None
                and (self.d[i].phase == CLIMB or (self.d[i].cenv and self.d[i].phase == APPROACH))):
            try:
                _agl = float(self._refine_state[i, ALT]) * 20.0
            except Exception:
                _agl = 9.0
            if _agl < 1.0:
                out = np.asarray(v, float).copy()
                out[2] = max(float(out[2]), 0.0)
                return out
        _el_max = None
        if c.free_ceiling_maps and _fk in tuple(c.free_ceiling_maps) and self.is_forest:
            _room = max(0.0, float(c.forest_ceiling) - float(pos[2]))
            v = np.asarray(v, float).copy()
            v[2] = min(float(v[2]), _room)
            speed = float(np.linalg.norm(v))
            if speed < 1e-6:
                return v
            if str(getattr(c, "free_ceiling_mode", "run")) == "clamp":
                # the steepest ray the vz <= room clamp can still fly at this speed
                _el_max = math.asin(min(1.0, _room / max(speed, 0.1)))
            else:
                _el_max = math.atan2(_room, max(float(c.free_ceiling_run), 0.1))
            if _room >= float(getattr(c, "free_ceiling_room_max", 99.0)):
                _el_max = None
        _el_min = None
        if c.fnd_maps and _fk in tuple(c.fnd_maps) and self.d[i].phase in (CLIMB, SEARCH, APPROACH):
            # collide FND: no free_direction dive below the command's own slope / onto what is below
            _vn = float(np.linalg.norm(v))
            _cel = math.asin(max(-1.0, min(1.0, float(v[2]) / max(_vn, 1e-6))))
            try:
                _fagl = float(self._refine_state[i, ALT]) * 20.0
            except Exception:                                    # noqa: BLE001
                _fagl = 20.0
            _el_min = max(min(_cel, 0.0) - math.radians(float(c.fnd_tol_deg)),
                          -math.atan2(max(0.0, _fagl - float(c.fnd_floor)), max(float(c.fnd_run), 0.1)))
            self._fnd_n = getattr(self, "_fnd_n", 0) + 1
        v_body = R.T @ np.asarray(v, float)
        if (c.climb_pass_maps and self.d[i].phase == CLIMB and _fk in tuple(c.climb_pass_maps)
                and float(v_body[0]) < speed * FREE_DIR_MIN_COS):
            # a mostly vertical take-off command lies outside the camera view: free_direction
            # would remap it onto the frame's top edge at free_slow (climb ~0.8 instead of 1.4 m/s)
            return v
        # CG (batch 2, city collision guard): parts on for this map kind, never in CLIMB
        _cg_live = self.d[i].phase != CLIMB
        _cg_ext = _cg_live and bool(c.free_extrude_maps) and _fk in tuple(c.free_extrude_maps)
        _cg_pg = _cg_live and bool(c.free_pass_guard_maps) and _fk in tuple(c.free_pass_guard_maps)
        _cg_sat = _cg_live and bool(c.free_sat_maps) and _fk in tuple(c.free_sat_maps)
        _ex = None
        _exw = None
        _guard = False
        _xdz = tuple(c.free_extrude_dz)
        if c.cg_dz_low and str(self.d[i].phase) in tuple(c.cg_dz_low_phases or ()):
            _xdz = tuple(float(x) for x in c.cg_dz_low) + _xdz          # collide DZL
        if _cg_ext and float(c.cg_roof_pad) > 0.0:
            # collide BNH: the roof-only subset of the extruded points gets a wider tube
            _ex, _exw = FD.extrude_pts2(depth[i], R, float(pos[2]) + 0.05,
                                        band=tuple(c.free_extrude_band), rng=float(c.free_extrude_range),
                                        dz=_xdz, vox=float(c.free_extrude_vox),
                                        cap=int(c.free_extrude_cap), grid=c.free_grid, fov_deg=PD.AP_FOV_DEG,
                                        depth_max=PD.AP_DEPTH_MAX_M,
                                        aligned=bool(c.free_align_maps) and _fk in tuple(c.free_align_maps),
                                        max_drop=float(c.free_extrude_max_drop),
                                        above=float(c.cg_roof_above), cell=float(c.cg_roof_cell))
            self._cg_n["ext_calls"] += 1
            if _ex is not None:
                self._cg_n["ext_pts"] += int(len(_ex))
            if _exw is not None:
                self._cg_n["bnh_pts"] = self._cg_n.get("bnh_pts", 0) + int(len(_exw))
        elif _cg_ext:
            _ex = FD.extrude_pts(depth[i], R, float(pos[2]) + 0.05,
                                 band=tuple(c.free_extrude_band), rng=float(c.free_extrude_range),
                                 dz=_xdz, vox=float(c.free_extrude_vox),
                                 cap=int(c.free_extrude_cap), grid=c.free_grid, fov_deg=PD.AP_FOV_DEG,
                                 depth_max=PD.AP_DEPTH_MAX_M,
                                 aligned=bool(c.free_align_maps) and _fk in tuple(c.free_align_maps),
                                 max_drop=float(c.free_extrude_max_drop))
            self._cg_n["ext_calls"] += 1
            if _ex is not None:
                self._cg_n["ext_pts"] += int(len(_ex))
        _om = bool(c.om_maps) and self._om_on() and self.d[i].phase != CLIMB
        if _om and c.om_fd:
            _omr, _omb = self._om_pts(i, pos, R)                  # blind_guard OM (1): memory in the tube tests
            if _omb is not None:
                _omb = _omb[np.linalg.norm(_omb, axis=1) <= (c.free_lookahead + 1.5)]
                if len(_omb):
                    _ex = _omb.astype(np.float32) if _ex is None else np.vstack([_ex, _omb.astype(np.float32)])
                    self._om_cnt("fd_pts", int(len(_omb)))
        fov_on = FIX_FOV not in ("0", "", "off") and (FIX_FOV != "city" or self._mass_route_kind() == "city")
        if (not fov_on and c.free_fov_open_pass and self.map_kind in tuple(c.yaw_scan_maps or ())
                and self._mass_route_kind() != "city"):
            # S3: the yaw scan (keyed on map_kind) swings the camera off the track on seeds routed as
            # open: pass the out-of-view command through like on city (open maps have no obstacles)
            fov_on = True
            if float(v_body[0]) < speed * FREE_DIR_MIN_COS:
                self._b3_n["s3_pass"] += 1
        if fov_on and float(v_body[0]) < speed * FREE_DIR_MIN_COS:
            # Flying somewhere the camera is not looking (a yaw scan swings 50 deg off
            # the track): there is no image evidence to steer on, so the only safe
            # correction left is to go slower rather than to carry on blind at cruise.
            if c.mem_grid:
                return self._mem_steer(i, pos, v)
            if c.free_fov_speed >= 0.0:
                out = np.asarray(v, float).copy()
                out[:2] *= float(c.free_fov_speed)
                return out
            if _om and c.om_blind:
                _bg = self._om_blind(i, pos, R, v, depth)          # blind_guard BG
                if _bg is not None:
                    return _bg
            if not _cg_pg:
                return v
            # CG pass-through guard: test the tube along the frame-edge direction nearest the
            # command; clear -> pass through as before, blocked -> steer on the clamped command
            dd = self.d[i]
            _az = math.atan2(float(v_body[1]), float(v_body[0]))
            if abs(_az) > math.radians(float(c.free_pass_guard_max_deg)):
                self._cg_n["pass_back"] += 1
                return v
            if abs(_az) > math.radians(float(c.free_pass_guard_back_deg)) and dd.pg_side != 0.0:
                _side = float(dd.pg_side)
            else:
                _side = math.copysign(1.0, _az)
            dd.pg_side = _side
            _ea = math.radians(float(c.free_pass_guard_deg)) * _side
            _h = float(np.hypot(v_body[0], v_body[1]))
            _want = np.array([_h * math.cos(_ea), _h * math.sin(_ea), float(v_body[2])])
            _ce = FD.ray_clearance(depth[i], _want, PD.AP_DEPTH_MAX_M, fov_deg=PD.AP_FOV_DEG,
                                   grid=c.free_grid, r_eff=0.12 + self._free_margin(_fk),
                                   lookahead=(c.free_lookahead_mountain if self.map_kind == "mountain"
                                              else c.free_lookahead),
                                   aligned=bool(c.free_align_maps) and _fk in tuple(c.free_align_maps),
                                   extra_pts=_ex, extra_wide=_exw, wide_add=float(c.cg_roof_pad))
            if _ce >= float(c.free_want_clear):
                self._cg_n["pass_kept"] += 1
                return v
            self._cg_n["pass_guard"] += 1
            v_body = _want
            _guard = True
        mtn = self.map_kind == "mountain"
        _margin = self._free_margin(_fk)
        _hys = bool(c.fhy_maps) and _fk in tuple(c.fhy_maps) and not _guard
        _hkw = {}
        if _hys:
            _dh = self.d[i]
            _cmd_az = float(math.atan2(float(v[1]), float(v[0])))
            _pa = getattr(_dh, "_fhy_cmd", None)
            if _pa is not None and abs((_cmd_az - _pa + math.pi) % (2.0 * math.pi) - math.pi) > math.radians(
                    float(c.fhy_cmd_deg)):
                _dh.fhy_side, _dh.fhy_n = 0.0, 0
            _dh._fhy_cmd = _cmd_az
            if _dh.fhy_side != 0.0:
                _hkw = dict(side=float(_dh.fhy_side), side_w=float(c.fhy_w), side_dead_deg=float(c.fhy_dead_deg))
        # NFB: berth arguments only on free_berth_maps (keyed by the free-direction kind)
        _nfb = bool(c.free_berth_maps) and _fk in tuple(c.free_berth_maps)
        _berth = (dict(berth_margin=float(c.free_berth_margin), berth_look=float(c.free_berth_look),
                       berth_cone_deg=float(c.free_berth_cone_deg), berth_keep=bool(c.free_berth_keep_speed))
                  if _nfb else {})
        _nfb0 = (FD.NFB_N[0], FD.NFB_N[1]) if _nfb else None
        pick = FD.free_direction(
            depth[i], v_body, PD.AP_DEPTH_MAX_M,
            fov_deg=PD.AP_FOV_DEG, grid=c.free_grid,
            radius=0.12,
            margin=_margin,
            want_clear=c.free_want_clear,
            lookahead=c.free_lookahead_mountain if mtn else c.free_lookahead,
            half_deg=(float(c.free_ray_half_deg) if c.free_ray_half_maps and
                      _fk in tuple(c.free_ray_half_maps)
                      else 0.0),
            aligned=bool(c.free_align_maps) and _fk in tuple(c.free_align_maps),
            el_max_world=_el_max, R=R if (_el_max is not None or _el_min is not None) else None,
            el_min_world=_el_min,
            extra_pts=_ex, extra_wide=_exw, wide_add=float(c.cg_roof_pad),
            lat_target=(float(c.free_lat_target) if c.free_lat_maps and _fk in tuple(c.free_lat_maps) else 0.0),
            lat_w=float(c.free_lat_w), **_hkw, **_berth)
        if _nfb0 is not None:
            self._b5_n["nfb_trig"] += FD.NFB_N[0] - _nfb0[0]
            self._b5_n["nfb_repick"] += FD.NFB_N[1] - _nfb0[1]
        if pick is None:
            return v
        if _hys:
            _rel = float(math.atan2(float(pick[1]), float(pick[0])) - math.atan2(float(v_body[1]), float(v_body[0])))
            _rel = (_rel + math.pi) % (2.0 * math.pi) - math.pi
            _dh = self.d[i]
            if abs(_rel) > math.radians(float(c.fhy_dead_deg)):
                if _dh.fhy_side != 0.0 and math.copysign(1.0, _rel) != _dh.fhy_side:
                    self.src_counts["fn_hys_flip"] = self.src_counts.get("fn_hys_flip", 0) + 1
                _dh.fhy_side, _dh.fhy_n = math.copysign(1.0, _rel), 0
            elif _dh.fhy_side != 0.0:
                _dh.fhy_n += 1
                if _dh.fhy_n >= int(c.fhy_release):
                    _dh.fhy_side, _dh.fhy_n = 0.0, 0
        if _cg_sat and FD.LAST_CLEAR[0] is not None and float(FD.LAST_CLEAR[0]) <= float(c.free_sat_clr):
            # CG saturation escape: every ray is blocked inside the tube; back off the nearest
            # surface and slide along it instead of flying the fallback ray into it
            _m = FD.near_dir(depth[i], 0.12 + _margin + float(c.free_sat_band), grid=c.free_grid,
                             fov_deg=PD.AP_FOV_DEG, depth_max=PD.AP_DEPTH_MAX_M,
                             aligned=bool(c.free_align_maps) and _fk in tuple(c.free_align_maps))
            if _m is not None:
                self._cg_n["sat_esc"] += 1
                _vb = R.T @ np.asarray(v, float)
                _wh = np.array([float(_vb[0]), float(_vb[1]), 0.0])
                _tw = float(_wh @ _m)
                if _tw > 0.0:
                    _wh = _wh - _tw * _m
                _sl = float(np.linalg.norm(_wh))
                if _sl > float(c.free_sat_slide):
                    _wh = _wh * (float(c.free_sat_slide) / _sl)
                _esc = -_m * float(c.free_sat_push) + _wh
                _esc[2] = 0.0
                out = R @ _esc
                out[2] = float((R @ np.asarray(pick, float))[2]) * speed * float(c.free_slow)
                return out
        world = R @ np.asarray(pick, float)
        _nfb_keep = FD.LAST_NARROW[0] is not None
        if _nfb_keep and c.spd_gate_nfb and not self._spd_gate_ok(i):
            _nfb_keep = False                   # candF hazard h: no inherited full speed below the SPD hit gate
            self.src_counts["hz_nfb_gate"] = self.src_counts.get("hz_nfb_gate", 0) + 1
        if _nfb_keep:
            # NFB free_berth_keep_speed: the speed decision of the narrow pick the berth re-pick replaced
            agree = float(np.dot(R @ np.asarray(FD.LAST_NARROW[0], float), np.asarray(v, float) / speed))
        else:
            agree = float(np.dot(world, np.asarray(v, float) / speed))
        _fsl = float(c.free_slow)
        if (c.fe_maps and _fk in tuple(c.fe_maps) and not _guard and agree <= 0.98
                and c.free_ray_half_maps and _fk in tuple(c.free_ray_half_maps)
                and FD.LAST_CLEAR[0] is not None and float(FD.LAST_CLEAR[0]) >= float(c.free_want_clear)
                and abs(math.degrees(math.atan2(float(v_body[1]), float(v_body[0])))) > float(c.free_ray_half_deg)
                and agree >= math.cos(math.radians(float(c.fe_max_deg)))):
            _fsl = 1.0                      # yaw_time FE: clear edge ray while the camera turns
            self._yt_n["fe_steps"] += 1
            if _YT_DIAG:
                self.d[i].yt |= 262144
        _spd_ok = True
        if int(c.spd_min_hits) > 0 and self.d[i].phase == APPROACH and self.d[i].claim is not None:
            try:
                _spd_ok = int(self.pads[self.d[i].claim].hits) >= int(c.spd_min_hits)
            except Exception:                                    # noqa: BLE001
                _spd_ok = True
        if (c.fsp_maps and _fk in tuple(c.fsp_maps) and not _guard and agree <= 0.98 and _spd_ok
                and FD.LAST_CLEAR[0] is not None
                and float(FD.LAST_CLEAR[0]) >= (float(c.fsp_clear) if float(c.fsp_clear) > 0.0
                                                 else float(c.free_want_clear))):
            if bool(c.fsp_h):
                # horizontal deviation only: the command's climb (vz <= room) is steeper than the ceiling rule lets
                # any ray climb (room over free_ceiling_run), so a 3-D agree counts an unflyable climb as a detour
                _hw = float(math.atan2(float(v[1]), float(v[0])))
                _hp = float(math.atan2(float(world[1]), float(world[0])))
                _ok = (float(np.hypot(float(world[0]), float(world[1]))) > 0.5
                       and abs((_hp - _hw + math.pi) % (2.0 * math.pi) - math.pi) <= math.radians(float(c.fsp_deg)))
            else:
                _ok = agree >= math.cos(math.radians(float(c.fsp_deg)))
            if _ok:
                _fsl = 1.0                  # forest_nav FSP: safe pick close to the command at full speed
                self.src_counts["fn_fsp"] = self.src_counts.get("fn_fsp", 0) + 1
        out = world * speed * (1.0 if (agree > 0.98 and not _guard) else _fsl)
        if (c.cls_maps and _fk in tuple(c.cls_maps) and not _guard and _spd_ok
                and (agree > 0.98 or _fsl >= 1.0)
                and FD.LAST_CLEAR[0] is not None and float(FD.LAST_CLEAR[0]) >= float(c.cls_clear)):
            _hin = float(np.hypot(float(v[0]), float(v[1])))
            _cr = float(self._cruise(False))
            if _cr > 0.0 and _hin >= _cr - 0.05 and float(c.cls_speed) > _cr:
                out = out * (float(c.cls_speed) / _cr)     # forest_nav CLS: clear lane
                self.src_counts["fn_cls"] = self.src_counts.get("fn_cls", 0) + 1
        if (c.sdb_maps and _fk in tuple(c.sdb_maps) and self.d[i].phase in (SEARCH, APPROACH)
                and FD.LAST_CLEAR[0] is not None and float(FD.LAST_CLEAR[0]) < float(c.free_want_clear)):
            _vc = max(float(c.sdb_vmin),
                      math.sqrt(2.0 * float(c.sdb_a) * max(0.0, float(FD.LAST_CLEAR[0]) - float(c.sdb_d0))))
            _m = float(np.linalg.norm(out))
            if _m > _vc:
                out = out * (_vc / _m)      # forest_nav SDB: stop within the pick's clearance
                self.src_counts["fn_sdb"] = self.src_counts.get("fn_sdb", 0) + 1
        if float(c.free_boxed_brake) > 0.0:
            _clr = FD.LAST_CLEAR[0]
            if _clr is not None and _clr < float(c.free_want_clear):
                _cap = max(float(c.free_boxed_brake_min), float(c.free_boxed_brake) * float(_clr))
                _m = float(np.linalg.norm(out))
                if _m > _cap:
                    out = out * (_cap / _m)
        if c.free_brake or c.free_climb > 0.0:
            clr = self._clearance(depth, i)
            if c.free_brake and clr < c.avoid_range:
                u = float(np.clip((c.avoid_range - clr) / c.avoid_range, 0.0, 1.0))
                out[:2] *= (1.0 - (1.0 - c.avoid_brake) * u)
            # free_direction only ever steers, so a drone that finds no clear heading
            # keeps flying at whatever is in front of it. Climbing is the escape the
            # camera cannot see -- the same response _avoid gives on the other maps.
            if c.free_climb > 0.0 and clr < c.free_climb_range:
                u = float(np.clip((c.free_climb_range - clr) / max(c.free_climb_range, 1e-6),
                                  0.0, 1.0))
                out[2] = max(float(out[2]), c.free_climb * u)
        if c.near_brake_maps and ("forest" if self.is_forest else str(self.map_kind)) in tuple(c.near_brake_maps):
            clr = self._clearance(depth, i)
            if clr < float(c.near_brake_range):
                out[:2] *= max(float(c.near_brake_min), clr / max(float(c.near_brake_range), 1e-6))
        if self.is_forest:
            room = c.forest_ceiling - float(pos[2])
            out[2] = min(float(out[2]), max(0.0, room))
        return out
    def _vra(self, depth, i, dd, pad, pos, rpy, v_des):
        """collide VRA: roof-aware height floor along the horizontal command (see the cfg block)."""
        c = self.cfg
        v = np.asarray(v_des, float).copy()
        sp = float(np.hypot(v[0], v[1]))
        fl = float(dd.vra_floor)
        if fl > -1.0 and (self.step_i - int(dd.vra_t)) * 0.02 > float(c.vra_hold):
            fl -= float(c.vra_decay) * 0.02
        near = 99.0
        _look = float(c.vra_look)
        if dd.phase == APPROACH and pad is not None:
            # only what lies between the drone and the claimed pad matters: it descends at the pad
            _look = min(_look, float(np.linalg.norm(np.asarray(pad.xyz, float)[:2] - np.asarray(pos, float)[:2]))
                        - float(c.vra_pad_r))
        if sp > 0.3 and _look > 0.2:
            u = v[:2] / sp
            d = np.asarray(depth[i], np.float32)
            if d.ndim == 3:
                d = d[..., 0]
            dirs, eu = FD._cloud(d, int(c.free_grid), PD.AP_FOV_DEG, PD.AP_DEPTH_MAX_M, 0.5, True)
            ok = eu * dirs[:, 0] < PD.AP_DEPTH_MAX_M - 0.6
            if ok.any():
                R = rot_from_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2]))
                P = (R @ (dirs[ok] * eu[ok, None]).T).T
                along = P[:, :2] @ u
                lat = np.abs(P[:, 0] * u[1] - P[:, 1] * u[0])
                zw = float(pos[2]) + P[:, 2]
                m = (along > 0.2) & (along < _look) & (lat < float(c.vra_half)) & (zw > float(c.vra_zmin))
                if dd.phase == APPROACH and pad is not None:
                    pxy = np.asarray(pad.xyz, float)[:2]
                    m &= np.hypot(P[:, 0] + float(pos[0]) - pxy[0], P[:, 1] + float(pos[1]) - pxy[1]) > float(c.vra_pad_r)
                if m.any():
                    top = float(zw[m].max()) + float(c.vra_margin)
                    if top >= fl:
                        fl = top
                        dd.vra_t = self.step_i
                    hit = m & (zw > float(pos[2]) - float(c.vra_margin))
                    if hit.any():
                        near = float(along[hit].min())
        dd.vra_floor = fl if fl > float(c.vra_zmin) else -1.0
        if dd.vra_floor > float(pos[2]):
            dz = dd.vra_floor - float(pos[2])
            v[2] = max(float(v[2]), min(float(c.vra_vz), float(c.vra_gain) * dz))
            if near < 98.0 and sp > 1e-6:
                t_up = dz / max(float(c.vra_vz), 0.1)
                cap = max(float(c.vra_min_speed), (near - float(c.vra_stop)) / max(t_up, 0.1))
                if sp > cap:
                    v[:2] *= cap / sp
            self._vra_n = getattr(self, "_vra_n", 0) + 1
        return v

    def _touch_on(self) -> bool:
        return bool(self.cfg.touch_off_maps) and (
            ("forest" if self.is_forest else str(self.map_kind)) in tuple(self.cfg.touch_off_maps))

    def _touch_sample(self, depth, state, i, dd, pad) -> None:
        if dd.touch_key != dd.claim:
            dd.touch_key = dd.claim if dd.claim is not None else -1
            dd.touch_buf = []
            dd.touch_off = None
        if pad is None or len(dd.touch_buf) >= 3000:
            return
        from team.detector.geom import _cloud
        pos = np.asarray(state[i, POS], float)
        rpy = np.asarray(state[i, RPY], float)
        R = rot_from_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        d = np.asarray(depth[i], np.float32)
        if d.ndim == 3:
            d = d[..., 0]
        pts, valid, _ = _cloud(pos, R, d[::2, ::2], PD._geom_cfg(self.pad_cfg), PD.AP_FOV_DEG, PD.AP_RES // 2)
        if pts is None:
            return
        p = np.asarray(pts[valid], float)
        if p.shape[0] == 0:
            return
        pc = np.asarray(pad.xyz, float)
        dx = p[:, 0] - pc[0]
        dy = p[:, 1] - pc[1]
        r = np.hypot(dx, dy)
        sel = (r >= 0.8) & (r <= 2.0) & (np.abs(p[:, 2] - pc[2]) < 3.0)
        if sel.any():
            dd.touch_buf.extend(np.column_stack([dx[sel], dy[sel], p[sel, 2]]).tolist())

    def _touch_offset(self, dd, pad):
        c = self.cfg
        B = np.asarray(dd.touch_buf, float).reshape(-1, 3)
        if (len(B) < int(c.touch_off_min_pts) or int(pad.hits) < int(c.touch_off_min_hits)
                or len(getattr(pad, "views", []) or []) < int(c.touch_off_min_views)):
            return np.zeros(2)
        az = ((np.arctan2(B[:, 1], B[:, 0]) + np.pi) / (2.0 * np.pi) * 8.0).astype(int) % 8
        if len(set(az.tolist())) < 3:
            return np.zeros(2)
        A = np.column_stack([np.ones(len(B)), B[:, 0], B[:, 1]])
        try:
            coef, *_ = np.linalg.lstsq(A, B[:, 2], rcond=None)
        except Exception:
            return np.zeros(2)
        g = np.array([coef[1], coef[2]], float)
        slope = float(np.linalg.norm(g))
        if slope < float(c.touch_off_min_slope):
            return np.zeros(2)
        return -g / slope * min(float(c.touch_off_max), float(c.touch_off_gain) * slope)

    # ---- land_safe TDO touchdown offset (all default off; see the tdo_* config block) ----
    def _tdo_on(self) -> bool:
        """TDO: this seed's kind ("forest" if is_forest else the route kind) is in tdo_maps."""
        m = tuple(self.cfg.tdo_maps or ())
        return bool(m) and ("forest" if self.is_forest else str(self._mass_route_kind())) in m

    def _tdo_inc(self, key: str, n: int = 1) -> None:
        self.src_counts["tdo_" + key] = self.src_counts.get("tdo_" + key, 0) + n

    def _tdo_sync(self, dd) -> None:
        """Reset the TDO state when the drone's claim changed."""
        k = int(dd.claim) if dd.claim is not None else -1
        if dd.tdo_key != k:
            dd.tdo_key = k
            dd.tdo_buf = []
            dd.tdo_nb = -1
            dd.tdo_off = None
            dd.tdo_frozen = False
            dd.tdo_rs = []
            dd.tdo_eff = None
            dd.tdo_lam = 0.0

    def _tdo_sample(self, depth, state, i, dd, pad) -> None:
        """Keep this frame's depth points near the ring around the claimed pad (world xyz, float32)."""
        c = self.cfg
        from team.detector.geom import _cloud
        pos = np.asarray(state[i, POS], float)
        rpy = np.asarray(state[i, RPY], float)
        R = rot_from_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        d = np.asarray(depth[i], np.float32)
        if d.ndim == 3:
            d = d[..., 0]
        pts, valid, _ = _cloud(pos, R, d, PD._geom_cfg(self.pad_cfg), PD.AP_FOV_DEG, int(d.shape[0]))
        if pts is None:
            return
        P = np.asarray(pts[valid], np.float32)
        if P.shape[0] == 0:
            return
        pc = np.asarray(pad.xyz, float)
        r = np.hypot(P[:, 0] - pc[0], P[:, 1] - pc[1])
        sel = ((r >= float(c.tdo_ring_lo) - 0.4) & (r <= float(c.tdo_ring_hi) + 0.4)
               & (np.abs(P[:, 2] - pc[2]) < 3.0))
        Q = P[sel]
        if Q.shape[0] == 0:
            return
        k = int(c.tdo_frame_pts)
        if k > 0 and Q.shape[0] > k:
            Q = Q[:: int(math.ceil(Q.shape[0] / k))]
        dd.tdo_buf.append(Q)
        if len(dd.tdo_buf) > int(c.tdo_frames):
            del dd.tdo_buf[0]

    @staticmethod
    def _tdo_plane(X, Z):
        """Trimmed least-squares plane z = gx x + gy y + h: (gx, gy) or None."""
        if len(Z) < 8:
            return None
        A = np.column_stack([X[:, 0], X[:, 1], np.ones(len(Z))])
        m = np.ones(len(Z), bool)
        coef = None
        for _it in range(3):
            coef, *_ = np.linalg.lstsq(A[m], Z[m], rcond=None)
            res = np.abs(Z - A @ coef)
            mad = float(np.median(res[m])) * 1.4826
            m2 = res <= max(0.08, 2.5 * mad)
            if int(m2.sum()) < max(8, len(Z) // 2):
                break
            m = m2
        coef, *_ = np.linalg.lstsq(A[m], Z[m], rcond=None)
        return np.array([coef[0], coef[1]], float)

    def _tdo_fit(self, dd, pad):
        """TDO offset from the ring samples: (offset xy, status). The offset is zero unless status == 'ok'."""
        c = self.cfg
        zero = np.zeros(2)
        if int(pad.hits) < int(c.tdo_min_hits):
            return zero, "hits"
        pc = np.asarray(pad.xyz, float)
        X, Z, K = [], [], []
        for k, a in enumerate(dd.tdo_buf):
            rel = np.asarray(a[:, :2], float) - pc[:2]
            r = np.hypot(rel[:, 0], rel[:, 1])
            m = (r >= float(c.tdo_ring_lo)) & (r <= float(c.tdo_ring_hi))
            if m.any():
                X.append(rel[m])
                Z.append(np.asarray(a[m, 2], float) - pc[2])
                K.append(np.full(int(m.sum()), k))
        if not X:
            return zero, "pts"
        X, Z, K = np.concatenate(X), np.concatenate(Z), np.concatenate(K)
        if len(Z) < int(c.tdo_min_pts):
            return zero, "pts"
        az = ((np.arctan2(X[:, 1], X[:, 0]) + np.pi) / (2.0 * np.pi) * 8.0).astype(int) % 8
        if int((np.bincount(az, minlength=8) >= 5).sum()) < int(c.tdo_min_sect):
            return zero, "sect"
        if bool(c.mtd_hm) and c.mtd_maps and dd.phase == DESCEND and self._mtd_on():
            dd.mtd_hmfit = True
            return self._mtd_hm_fit(X, Z)              # MTD hm: height-map bearing instead of the plane's
        g = self._tdo_plane(X, Z)
        if g is None:
            return zero, "pts"
        self._tdo_g = g
        s = float(np.linalg.norm(g))
        if s < float(c.tdo_min_slope):
            return zero, "flat"
        if float(c.tdo_max_ang) < 180.0:
            ks = sorted(set(K.tolist()))
            if len(ks) < 2:
                return zero, "stab"
            half = ks[len(ks) // 2]
            ga = self._tdo_plane(X[K < half], Z[K < half])
            gb = self._tdo_plane(X[K >= half], Z[K >= half])
            if ga is None or gb is None:
                return zero, "stab"
            na, nb = float(np.linalg.norm(ga)), float(np.linalg.norm(gb))
            if na < 1e-6 or nb < 1e-6:
                return zero, "stab"
            cosang = float(np.dot(ga, gb) / (na * nb))
            if cosang < math.cos(math.radians(float(c.tdo_max_ang))):
                return zero, "stab"
        return -g / s * float(c.tdo_r), "ok"

    def _tdo_update(self, dd, pad, freeze: bool) -> None:
        """Refit the offset when new frames arrived (not once frozen); freeze=True freezes it (DESCEND entry)."""
        if dd.tdo_frozen:
            return
        _hmf = False
        if freeze and bool(self.cfg.mtd_hm) and self.cfg.mtd_maps and self._mtd_on():
            self._mtd_sync(dd)
            _hmf = not dd.mtd_hmfit                     # MTD hm: the freeze fit is the height-map pick
        if dd.tdo_off is None or dd.tdo_nb != len(dd.tdo_buf) or _hmf:
            self._tdo_g = None
            if self.cfg.mtd_maps:
                self._mtd_hm_info = None
            try:
                off, why = self._tdo_fit(dd, pad)
            except Exception:                            # noqa: BLE001
                off, why = np.zeros(2), "err"
            dd.tdo_off, dd.tdo_nb = off, len(dd.tdo_buf)
            if self.cfg.mtd_maps:
                dd._mtd_hi = self._mtd_hm_info          # MTD diag: this drone's last height-map pick
            dd.tdo_why = why
            dd.tdo_g = None if self._tdo_g is None else np.asarray(self._tdo_g, float).copy()
        if freeze:
            dd.tdo_frozen = True
            self._tdo_inc("frz_" + str(getattr(dd, "tdo_why", "none")))
            _lg = self.src_counts.setdefault("tdo_log", [])
            if isinstance(_lg, list) and len(_lg) < 64:
                _o = dd.tdo_off if dd.tdo_off is not None else np.zeros(2)
                _lg.append([round(self.step_i * 0.02, 2), int(dd.tdo_key), round(float(pad.xyz[0]), 3),
                            round(float(pad.xyz[1]), 3), round(float(_o[0]), 3), round(float(_o[1]), 3),
                            str(dd.tdo_why), int(sum(len(a) for a in dd.tdo_buf)), len(dd.tdo_buf),
                            None if dd.tdo_g is None else [round(float(dd.tdo_g[0]), 3), round(float(dd.tdo_g[1]), 3)]])

    def _tdo_dir(self, dd) -> bool:
        """A downhill direction (nominal offset) exists for the current claim."""
        return (dd.tdo_off is not None and dd.claim is not None and dd.tdo_key == int(dd.claim)
                and float(dd.tdo_off[0] ** 2 + dd.tdo_off[1] ** 2) > 1e-12)

    def _tdo_active(self, dd) -> bool:
        """The offset in use for the current claim is nonzero."""
        return (self._tdo_dir(dd) and dd.tdo_eff is not None
                and float(dd.tdo_eff[0] ** 2 + dd.tdo_eff[1] ** 2) > 1e-12)

    def _tdo_pad_dist(self, dd, pad, xy) -> float:
        """Horizontal distance from xy to the claimed pad: the nearer of the estimate and the TDO offset point."""
        pxy = np.asarray(pad.xyz, float)[:2]
        dc = float(np.linalg.norm(pxy - np.asarray(xy, float)[:2]))
        if self._tdo_active(dd):
            dc = min(dc, float(np.linalg.norm(pxy + dd.tdo_eff - np.asarray(xy, float)[:2])))
        return dc

    def _tdo_ray_note(self, dd, pad, pos, agl) -> None:
        """TDO rim: keep (x, y, surface z) under the drone near the claimed pad (new xy only)."""
        if not (0.05 < agl < 19.9) or float(pos[2]) - float(pad.xyz[2]) < 0.3:
            return                  # the altitude ray starts ~0.1 m under the body: unreliable at touchdown
        pxy = np.asarray(pad.xyz, float)[:2]
        if float(np.hypot(float(pos[0]) - pxy[0], float(pos[1]) - pxy[1])) > float(self.cfg.tdo_samp_r):
            return
        if dd.tdo_rs:
            lx, ly, _lz = dd.tdo_rs[-1]
            if math.hypot(float(pos[0]) - lx, float(pos[1]) - ly) < 0.01:
                return
        dd.tdo_rs.append((float(pos[0]), float(pos[1]), float(pos[2]) - float(agl)))
        if len(dd.tdo_rs) > 600:
            del dd.tdo_rs[0]

    @staticmethod
    def _tdo_flat(S):
        """Per sample: a sequence neighbour within 0.15 m reads the same height to within a 0.02 slope + 0.5 mm (the
        pad top is exactly flat; the rim step and sloped terrain are not)."""
        fl = np.zeros(len(S), bool)
        if len(S) > 1:
            dxy = np.hypot(np.diff(S[:, 0]), np.diff(S[:, 1]))
            pr = (np.abs(np.diff(S[:, 2])) <= 0.02 * dxy + 0.0005) & (dxy <= 0.15)
            fl[:-1] |= pr
            fl[1:] |= pr
        return fl

    def _tdo_ztop(self, S, pad):
        """Pad-top height from the ray samples (time order), or None. Candidate windows of 2 * tdo_ztol in height
        (the flat disc reads 0.011 m above the platform annulus) among the locally flat samples within tdo_zband of
        the estimate's z and 0.75 m of its xy; the HIGHEST window with >= tdo_plat_n samples spanning >= tdo_plat_ext m
        that also shows a rim step (a sequence neighbour within 0.12 m reading >= tdo_rim_step lower) wins. The step
        proves a raised platform edge: the flattened terrain just outside a rim (B 950 d4: 20 samples at top - 0.37 m,
        0.002-0.003 m apart, no step) passed the old densest-window test and sank a drone off the platform."""
        c = self.cfg
        pc = np.asarray(pad.xyz, float)
        fl = self._tdo_flat(S)
        m = (fl & (np.abs(S[:, 2] - pc[2]) <= float(c.tdo_zband))
             & (np.hypot(S[:, 0] - pc[0], S[:, 1] - pc[1]) <= 0.75))
        idx = np.nonzero(m)[0]
        if len(idx) < int(c.tdo_plat_n):
            return None
        order = idx[np.argsort(-S[idx, 2])]            # highest first
        z = S[order, 2]
        w = 2.0 * float(c.tdo_ztol)
        step = float(c.tdo_rim_step)
        tried = set()
        for a in range(len(order)):
            zt_hi = float(z[a])
            key = round(zt_hi, 3)
            if key in tried:
                continue
            tried.add(key)
            win = order[(z <= zt_hi + 1e-9) & (z >= zt_hi - w)]
            if len(win) < int(c.tdo_plat_n):
                continue
            W = S[win]
            if float(np.hypot(W[:, 0].max() - W[:, 0].min(), W[:, 1].max() - W[:, 1].min())) < float(c.tdo_plat_ext):
                continue
            ok = False
            for k in win:
                for j in (k - 1, k + 1):
                    if (0 <= j < len(S) and S[j, 2] <= S[k, 2] - step
                            and math.hypot(S[j, 0] - S[k, 0], S[j, 1] - S[k, 1]) <= 0.12):
                        ok = True
                        break
                if ok:
                    break
            if ok:
                return float(np.median(W[:, 2]))
        return None

    def _tdo_lambda(self, dd, pad, u) -> float:
        """TDO rim: the largest lam in [0, tdo_r] whose target pad + lam * u is within 0.6 - tdo_margin of every
        prior centre consistent with the on / off-platform ray samples (0 if none, or no sample is informative)."""
        c = self.cfg
        pc = np.asarray(pad.xyz, float)
        S = np.asarray(dd.tdo_rs, float).reshape(-1, 3)
        if len(S) == 0:
            return 0.0
        zt = self._tdo_ztop(S, pad)
        if zt is None:
            on = np.zeros(len(S), bool)
            off = np.abs(S[:, 2] - pc[2]) > float(c.tdo_zband)
        else:
            dz = np.abs(S[:, 2] - zt)
            # on the platform: at the plateau height AND locally flat (sloped terrain crossing the pad-top
            # height 1.0-1.2 m uphill is not)
            on = (dz <= float(c.tdo_ztol)) & self._tdo_flat(S)
            off = dz > float(c.tdo_zoff)
        if not on.any():
            return 0.0
        g = getattr(self, "_tdo_grid", None)
        key = (float(c.tdo_pr_up), float(c.tdo_pr_dn), float(c.tdo_pr_lat), float(c.tdo_pr_core), float(c.tdo_pr_mu),
               float(c.tdo_pr_sd), float(c.tdo_pr_pcore), float(c.tdo_pr_psd))
        if g is None or g[0] != key:
            A, B = np.meshgrid(np.arange(-float(c.tdo_pr_dn), float(c.tdo_pr_up) + 1e-9, 0.03),
                               np.arange(-float(c.tdo_pr_lat), float(c.tdo_pr_lat) + 1e-9, 0.03))
            A, B = A.ravel(), B.ravel()
            pa = (float(c.tdo_pr_core) * np.exp(-0.5 * ((A - float(c.tdo_pr_mu)) / float(c.tdo_pr_sd)) ** 2)
                  / (float(c.tdo_pr_sd) * math.sqrt(2 * math.pi))
                  + (1.0 - float(c.tdo_pr_core)) / (float(c.tdo_pr_up) + float(c.tdo_pr_dn)))
            pb = (float(c.tdo_pr_pcore) * np.exp(-0.5 * (B / float(c.tdo_pr_psd)) ** 2)
                  / (float(c.tdo_pr_psd) * math.sqrt(2 * math.pi))
                  + (1.0 - float(c.tdo_pr_pcore)) / (2.0 * float(c.tdo_pr_lat)))
            g = (key, A, B, pa * pb)
            self._tdo_grid = g
        A, B, Wp = g[1], g[2], g[3]
        perp = np.array([-u[1], u[0]])
        C = pc[:2][None, :] - A[:, None] * u[None, :] + B[:, None] * perp[None, :]
        feas = np.ones(len(C), bool)
        P = S[on, :2]
        D = np.hypot(C[:, None, 0] - P[None, :, 0], C[:, None, 1] - P[None, :, 1])
        feas &= (D <= 0.62).all(axis=1)
        if off.any():
            P = S[off, :2]
            D = np.hypot(C[:, None, 0] - P[None, :, 0], C[:, None, 1] - P[None, :, 1])
            feas &= (D >= 0.58).all(axis=1)
        if not feas.any():
            self._tdo_inc("incons")
            return 0.0
        Cf, Wf = C[feas], Wp[feas]
        Wf = Wf / float(Wf.sum())
        lim = 0.6 - float(c.tdo_margin)
        lam = float(c.tdo_r)
        while lam > 1e-9:
            T = pc[:2] + lam * u
            if float(Wf[np.hypot(Cf[:, 0] - T[0], Cf[:, 1] - T[1]) > lim].sum()) <= float(c.tdo_eps):
                return lam
            lam = round(lam - 0.05, 6)
        return 0.0

    def _tdo_step(self, dd, pad, pos, agl) -> None:
        """Set dd.tdo_eff for this step (fixed offset, or the rim-referenced lam * u)."""
        _mtd = bool(self.cfg.mtd_maps) and self._mtd_on()
        if _mtd:
            self._mtd_sync(dd)
            if dd.mtd_rg_pfire is not None and dd.mtd_rg_nf > 0:
                self._tdo_ray_note(dd, pad, pos, agl)
                return                                   # MTD rim: the guard owns the target after a fire
        if not self._tdo_dir(dd):
            dd.tdo_eff = None
            return
        if not bool(self.cfg.tdo_rim):
            dd.tdo_eff = dd.tdo_off
            return
        self._tdo_ray_note(dd, pad, pos, agl)
        ab = float(pos[2]) - float(pad.xyz[2])
        if ab < float(self.cfg.tdo_frz_ab) and dd.tdo_eff is not None:
            return
        u = dd.tdo_off / float(np.linalg.norm(dd.tdo_off))
        try:
            lam = self._tdo_lambda(dd, pad, u)
        except Exception:                                # noqa: BLE001
            lam = 0.0
        _gpass = False
        if _mtd and bool(self.cfg.mtd_gate) and float(self.cfg.mtd_r_hi) > 0.0 and dd.phase == DESCEND:
            _gpass = self._mtd_gate_now(dd, pad)          # MTD gate: precise estimate -> Gaussian-prior lam
            if _gpass:
                try:
                    lam = self._mtd_lambda(dd, pad, u)
                except Exception:                        # noqa: BLE001
                    lam = 0.0
        if lam > dd.tdo_lam and ab <= float(self.cfg.tdo_lock_ab):
            lam = dd.tdo_lam
        if lam != dd.tdo_lam:
            self._tdo_inc("lam_up" if lam > dd.tdo_lam else "lam_dn")
        dd.tdo_lam = lam
        tgt = u * lam
        rate = float(self.cfg.tdo_rate)
        cur = dd.tdo_eff if dd.tdo_eff is not None else np.zeros(2)
        if (_gpass and ab > float(self.cfg.tdo_lock_ab)
                and float(np.linalg.norm(np.asarray(pad.xyz, float)[:2] + tgt - np.asarray(pos, float)[:2]))
                > float(self.cfg.mtd_jump_r)):
            if float(np.linalg.norm(tgt - cur)) > 1e-9:
                self.src_counts["mtd_jump"] = self.src_counts.get("mtd_jump", 0) + 1
            dd.tdo_eff = tgt                             # MTD: early in the drop a new aim point is harmless
        elif rate <= 0.0 or float(np.linalg.norm(tgt)) < float(np.linalg.norm(cur)) - 1e-9:
            dd.tdo_eff = tgt                             # no rate limit, or a safety decrease: at once
        elif ab > float(self.cfg.tdo_move_ab):
            # outward moves of the target at <= tdo_rate m/s and only while > tdo_move_ab above the pad: a late
            # lateral correction in the 2.8 m/s EDR drop freezes a 10-15 deg tilt into the contact (B 426 d0 TILT)
            dv = tgt - cur
            nv = float(np.linalg.norm(dv))
            mx = rate * 0.02
            dd.tdo_eff = tgt if nv <= mx else cur + dv * (mx / nv)

    # ---- round-5 MTD mountain touchdown precision (all default off; see the mtd_* config block) ----
    def _mtd_on(self) -> bool:
        """MTD: this seed's kind ("forest" if is_forest else the route kind) is in mtd_maps."""
        m = tuple(self.cfg.mtd_maps or ())
        return bool(m) and ("forest" if self.is_forest else str(self._mass_route_kind())) in m

    def _mtd_note(self, pad, xyz, rng, fresh: bool = False) -> None:
        """MTD: fold one detector fix into the entry's inverse-variance centre estimate (c29k2 outer.py 3181-3187)."""
        c = self.cfg
        r = float(rng) if rng is not None else float(c.mtd_rng_none)
        sig = float(c.mtd_sig0) + float(c.mtd_sig_k) * r
        wgt = 1.0 / max(sig * sig, 1e-6)
        x = np.asarray(xyz, float)[:3]
        if pad.mtd_rec is None:
            pad.mtd_rec = deque(maxlen=max(1, int(c.mtd_rec_n)))
        pad.mtd_rec.append((int(self.step_i), float(x[0]), float(x[1]), float(x[2]), r))
        if fresh:
            pad.mtd_xyz = x.copy()
            pad.mtd_info = min(wgt, float(c.mtd_info_cap))
            return
        if pad.mtd_xyz is None:
            # an entry that never had the state (split / injected): start from its estimate, weighted by its hits
            s0 = float(c.mtd_sig0) + float(c.mtd_sig_k) * float(c.mtd_init_rng)
            pad.mtd_xyz = np.asarray(pad.xyz, float)[:3].copy()
            pad.mtd_info = min(float(c.mtd_info_cap), max(1, int(pad.hits)) / max(s0 * s0, 1e-6))
        a = wgt / (float(pad.mtd_info) + wgt)
        lo = float(c.mtd_alpha_min) if r <= float(c.mtd_floor_r) else 0.0
        a = min(max(a, lo), float(c.mtd_alpha_max))
        if (r < 1.5 and int(pad.hits) >= 30
                and float(math.hypot(x[0] - pad.mtd_xyz[0], x[1] - pad.mtd_xyz[1])) > 0.5):
            a = 0.01                                     # a close outlier on an established entry
        pad.mtd_xyz = (1.0 - a) * pad.mtd_xyz + a * x
        pad.mtd_info = min(float(pad.mtd_info) + wgt, float(c.mtd_info_cap))

    def _mtd_use_est(self, pad) -> None:
        """MTD est: the claimed entry's estimate becomes its inverse-variance estimate (claim_drift anchor cap kept)."""
        if pad.mtd_xyz is None:
            return
        new = np.asarray(pad.mtd_xyz, float)[:3].copy()
        if pad.anchor is not None:
            off = new[:2] - pad.anchor
            far = float(np.linalg.norm(off))
            if far > float(self.cfg.claim_drift):
                new[:2] = pad.anchor + off * (float(self.cfg.claim_drift) / far)
        pad.xyz = new

    def _mtd_sync(self, dd) -> None:
        """MTD: reset the drone's gate / rim-guard state when its claim changed."""
        k = int(dd.claim) if dd.claim is not None else -1
        if dd.mtd_key != k:
            dd.mtd_key = k
            dd.mtd_gate, dd.mtd_gfrz = None, False
            dd.mtd_rg_hold, dd.mtd_rg_nf, dd.mtd_rg_run, dd.mtd_rg_desc = False, 0, 0, 0
            dd.mtd_rg_agl, dd.mtd_rg_t0, dd.mtd_rg_sfire, dd.mtd_rg_pfire = [], 0, 0.0, None
            dd.mtd_logged = False
            dd.mtd_hmfit = False

    def _mtd_ztop(self, dd, pad):
        """MTD: TDO's ray-calibrated pad-top height for the claim, cached until a new ray sample arrives."""
        rs = dd.tdo_rs
        if not rs:
            return None
        key = (len(rs), rs[-1], round(float(pad.xyz[0]), 4), round(float(pad.xyz[1]), 4), round(float(pad.xyz[2]), 4))
        zc = getattr(dd, "_mtd_zc", None)
        if zc is not None and zc[0] == key:
            return zc[1]
        zt = self._tdo_ztop(np.asarray(rs, float).reshape(-1, 3), pad)
        dd._mtd_zc = (key, zt)
        return zt

    def _mtd_gate_eval(self, pad) -> bool:
        """MTD gate: the last mtd_close_n close fixes agree and the estimate sits at their median."""
        c = self.cfg
        rec = pad.mtd_rec
        if not rec:
            return False
        s_min = int(self.step_i) - int(round(float(c.mtd_close_s) * 50.0))
        F = [q for q in rec if q[4] <= float(c.mtd_close_r) and q[0] >= s_min]
        n = max(1, int(c.mtd_close_n))
        if len(F) < n:
            return False
        A = np.asarray(F[-n:], float)[:, 1:3]
        med = np.median(A, axis=0)
        d = np.hypot(A[:, 0] - med[0], A[:, 1] - med[1])
        if float(np.percentile(d, 80)) > float(c.mtd_agree):
            return False
        return float(math.hypot(float(pad.xyz[0]) - med[0], float(pad.xyz[1]) - med[1])) <= float(c.mtd_agree)

    def _mtd_gate_now(self, dd, pad) -> bool:
        """MTD gate verdict for the drone's claim: evaluated while approaching, frozen at DESCEND entry."""
        if dd.mtd_gfrz:
            return bool(dd.mtd_gate)
        g = self._mtd_gate_eval(pad)
        dd.mtd_gate = g
        if dd.phase == DESCEND:
            dd.mtd_gfrz = True
            self.src_counts["mtd_gate_pass" if g else "mtd_gate_fail"] = (
                self.src_counts.get("mtd_gate_pass" if g else "mtd_gate_fail", 0) + 1)
        return g

    def _mtd_lambda(self, dd, pad, u) -> float:
        """MTD gate: the largest lam <= mtd_r_hi (steps of 0.05) whose target pad + lam * u stays within
        0.6 - mtd_margin of the true centre with probability >= 1 - mtd_eps, under a Gaussian prior on the estimate
        error (sd mtd_pr_sd per axis) restricted by TDO's on / off-platform ray samples (none needed)."""
        c = self.cfg
        pc = np.asarray(pad.xyz, float)
        sd = max(1e-3, float(c.mtd_pr_sd))
        g = getattr(self, "_mtd_grid", None)
        if g is None or g[0] != sd:
            st = sd / 3.0
            ax = np.arange(-4.0 * sd, 4.0 * sd + 1e-9, st)
            A, B = np.meshgrid(ax, ax)
            A, B = A.ravel(), B.ravel()
            W = np.exp(-0.5 * (A * A + B * B) / (sd * sd))
            g = (sd, A, B, W / float(W.sum()))
            self._mtd_grid = g
        A, B, Wp = g[1], g[2], g[3]
        perp = np.array([-u[1], u[0]])
        C = pc[:2][None, :] + A[:, None] * u[None, :] + B[:, None] * perp[None, :]
        feas = np.ones(len(C), bool)
        S = np.asarray(dd.tdo_rs, float).reshape(-1, 3)
        if len(S):
            zt = self._mtd_ztop(dd, pad)
            if zt is None:
                on = np.zeros(len(S), bool)
                off = np.abs(S[:, 2] - pc[2]) > float(c.tdo_zband)
            else:
                dz = np.abs(S[:, 2] - zt)
                on = (dz <= float(c.tdo_ztol)) & self._tdo_flat(S)
                off = dz > float(c.tdo_zoff)
            if on.any():
                P = S[on, :2]
                feas &= (np.hypot(C[:, None, 0] - P[None, :, 0], C[:, None, 1] - P[None, :, 1]) <= 0.62).all(axis=1)
            if off.any():
                P = S[off, :2]
                feas &= (np.hypot(C[:, None, 0] - P[None, :, 0], C[:, None, 1] - P[None, :, 1]) >= 0.58).all(axis=1)
        if not feas.any():
            self.src_counts["mtd_incons"] = self.src_counts.get("mtd_incons", 0) + 1
            return 0.0
        Cf, Wf = C[feas], Wp[feas]
        Wf = Wf / float(Wf.sum())
        lim = 0.6 - float(c.mtd_margin)
        lam = round(float(c.mtd_r_hi) / 0.05) * 0.05
        while lam > 1e-9:
            T = pc[:2] + lam * np.asarray(u, float)
            if float(Wf[np.hypot(Cf[:, 0] - T[0], Cf[:, 1] - T[1]) > lim].sum()) <= float(c.mtd_eps):
                return float(round(lam, 6))
            lam = round(lam - 0.05, 6)
        return 0.0

    def _mtd_hm_fit(self, X, Z):
        """MTD hm: touchdown bearing from the height map of TDO's ring points (c29k2 _s1m_hm_pick): (offset, status).
        X: (M, 2) ring points relative to the estimate, Z: (M,) heights relative to the estimate's z."""
        c = self.cfg
        zero = np.zeros(2)
        C = _mtd_cells(np.column_stack([X, Z]), float(c.tdo_ring_lo), float(c.tdo_ring_hi), 0.1)
        if len(C) < 12:
            self._mtd_hm_info = ("few", len(C))
            return zero, "hmfew"
        coef, keep = _mtd_quad(C, 12)
        if coef is None:
            self._mtd_hm_info = ("cells", len(C))
            return zero, "hmcells"
        x, y = C[keep, 0], C[keep, 1]
        cp = np.linalg.lstsq(np.c_[x, y, np.ones(len(x))], C[keep, 2], rcond=None)[0]
        slope = float(math.hypot(cp[0], cp[1]))
        if slope > float(c.mtd_hm_smax):
            self._mtd_hm_info = ("steep", round(slope, 2))
            return zero, "hmsteep"
        r_off = float(c.mtd_r_hi) if float(c.mtd_r_hi) > 0.0 else float(c.tdo_r)
        best_b, best_c, c0 = _mtd_hm_eval(coef, r_off)
        self._mtd_hm_info = (round(best_b * 22.5, 1), round(best_c, 3), round(c0, 3), int(len(C)), round(slope, 2))
        if best_c < c0 + float(c.mtd_hm_gain):
            return zero, "hmflat"
        a = 2.0 * math.pi * best_b / 16.0
        return np.array([math.cos(a), math.sin(a)]) * float(c.tdo_r), "ok"

    def _mtd_rim_step(self, i, dd, pad, pos, vel, agl, v_des):
        """MTD rim guard (c29k2 _me_rim): the down ray says the target is beside the platform -> stop the sink and
        step the target inward (centre on the 2nd fire). Returns the velocity command."""
        c = self.cfg
        self._mtd_sync(dd)
        pc = np.asarray(pad.xyz, float)
        eff = np.asarray(dd.tdo_eff, float) if dd.tdo_eff is not None else np.zeros(2)
        zt = self._mtd_ztop(dd, pad)
        if zt is not None:
            ref, thr = float(zt), float(c.mtd_rg_thr_cal)
        else:
            ref, thr = float(pc[2]), float(c.mtd_rg_thr)
        s_ = float(pos[2]) - float(agl)
        ex = ref - s_
        dd.mtd_rg_desc = dd.mtd_rg_desc + 1 if float(vel[2]) < -0.2 else 0
        dd.mtd_rg_agl = (list(dd.mtd_rg_agl) + [float(agl)])[-10:]
        tgt = pc[:2] + eff
        d_xy = float(np.linalg.norm(tgt - np.asarray(pos, float)[:2]))
        cv = pc[:2] - np.asarray(pos, float)[:2]
        v_in = float(vel[0] * cv[0] + vel[1] * cv[1]) / max(float(np.linalg.norm(cv)), 1e-6)
        sig = (float(agl) < 19.5 and ex > thr and d_xy < float(c.mtd_rg_xy) and v_in < float(c.mtd_rg_vin)
               and dd.mtd_rg_desc >= 5 and float(pos[2]) - ref > 0.08 and len(dd.mtd_rg_agl) >= 10
               and min(dd.mtd_rg_agl) > 0.12)
        dd.mtd_rg_run = dd.mtd_rg_run + 1 if sig else 0
        if not dd.mtd_rg_hold and dd.mtd_rg_run >= 2 and dd.mtd_rg_nf < 2:
            dd.mtd_rg_nf += 1
            dd.mtd_rg_hold = True
            dd.mtd_rg_t0 = int(self.step_i)
            dd.mtd_rg_sfire = s_
            dd.mtd_rg_pfire = np.asarray(pos, float)[:2].copy()
            cur = np.asarray(pos, float)[:2] - pc[:2]
            r_ = float(np.linalg.norm(cur))
            new_r = min(r_ - float(c.mtd_rg_step), 0.5 * float(np.linalg.norm(eff)))
            if dd.mtd_rg_nf < 2 and r_ > 1e-3 and new_r > 0.05:
                dd.tdo_eff = cur * (new_r / r_)
            else:
                dd.tdo_eff = np.zeros(2)
            dd.tdo_lam = float(np.linalg.norm(dd.tdo_eff))
            self.src_counts["mtd_rg_fire"] = self.src_counts.get("mtd_rg_fire", 0) + 1
            _lg = self.src_counts.setdefault("mtd_rg_log", [])
            if isinstance(_lg, list) and len(_lg) < 32:
                _lg.append([round(self.step_i * 0.02, 2), int(i), int(dd.claim), round(ex, 3),
                            round(float(pos[2]) - ref, 3), int(zt is not None), round(d_xy, 3), round(r_, 3)])
        if dd.mtd_rg_hold:
            eff = np.asarray(dd.tdo_eff, float) if dd.tdo_eff is not None else np.zeros(2)
            rel = pc[:2] + eff - np.asarray(pos, float)[:2]
            arrived = float(np.linalg.norm(rel)) < 0.1
            stepped = s_ >= dd.mtd_rg_sfire + 0.1
            if stepped or arrived or int(self.step_i) - dd.mtd_rg_t0 > int(round(float(c.mtd_rg_hold_s) * 50.0)):
                dd.mtd_rg_hold = False
                dd.mtd_rg_run = 0
                key = "mtd_rg_step" if stepped else ("mtd_rg_arrive" if arrived else "mtd_rg_timeout")
                self.src_counts[key] = self.src_counts.get(key, 0) + 1
                return v_des
            if float(pos[2]) - ref < float(c.mtd_rg_margin):
                self.src_counts["mtd_rg_climb"] = self.src_counts.get("mtd_rg_climb", 0) + 1
                return np.array([0.0, 0.0, 0.6])        # straight up first: sideways at this height meets the rim
            corr = 1.6 * rel
            cn = float(np.linalg.norm(corr))
            if cn > 0.6:
                corr *= 0.6 / cn
            return np.array([float(corr[0]), float(corr[1]), 0.0])
        return v_des

    def _probe_on(self) -> bool:
        return bool(self.cfg.ray_probe_maps) and (
            ("forest" if self.is_forest else str(self.map_kind)) in tuple(self.cfg.ray_probe_maps))

    def _probe_sample(self, dd, pad, pos, agl) -> None:
        if dd.gnd_key != dd.claim:
            dd.gnd_key = dd.claim if dd.claim is not None else -1
            dd.gnd = []
            dd.gnd_g = None
            dd.probe = None
            dd.probe_tries = 0
        if pad is None or not (0.05 < agl < 19.9):
            return
        pxy = np.asarray(pad.xyz, float)[:2]
        dxy = float(np.linalg.norm(pxy - pos[:2]))
        if dxy < 8.0 and len(dd.gnd) < 400:
            # (x, y, surface height under the drone)
            dd.gnd.append((float(pos[0]), float(pos[1]), float(pos[2]) - float(agl)))
        if dd.gnd_g is None and len(dd.gnd) >= 8:
            S = np.asarray(dd.gnd, float)
            ring = S[np.hypot(S[:, 0] - pxy[0], S[:, 1] - pxy[1]) > 1.3]
            if len(ring) >= 8:
                dd.gnd_g = float(np.percentile(ring[:, 2], 20.0))

    def _probe_fix(self, pad, xy) -> None:
        pad.xyz = np.array([float(xy[0]), float(xy[1]), pad.xyz[2]], float)
        pad.anchor = np.asarray(xy, float)[:2].copy()
        pad.probed = True

    # ---- F3 P2r loop breaker (dsc_below_retarget) ----
    def _bl_rec_on(self) -> bool:
        c = self.cfg
        return (int(c.dsc_below_retarget) > 0 and bool(c.dsc_below_maps)
                and ("forest" if self.is_forest else str(self.map_kind)) in tuple(c.dsc_below_maps))

    @staticmethod
    def _bl_note(pad, xyz, rng) -> None:
        if pad.bl_raw is None:
            pad.bl_raw = deque(maxlen=64)
        pad.bl_raw.append((float(xyz[0]), float(xyz[1]), float(xyz[2]), 99.0 if rng is None else float(rng)))

    def _bl_clusters(self, pad):
        """2-means (xy) of the proposals merged into this entry within 3 m of it: ((c, z, n), (c, z, n)) or None."""
        exy = np.asarray(pad.xyz, float)[:2]
        P = np.zeros((0, 4), float)
        if pad.bl_raw:
            P = np.asarray(list(pad.bl_raw), float).reshape(-1, 4)
            P = P[np.hypot(P[:, 0] - exy[0], P[:, 1] - exy[1]) <= 3.0]
            _cl = P[P[:, 3] < REFINE_R]
            if len(_cl) >= 6:
                P = _cl
        if len(P) < 6 and pad.near is not None and len(pad.near) >= 6:
            Q = np.asarray(list(pad.near), float).reshape(-1, 2)
            Q = Q[np.hypot(Q[:, 0] - exy[0], Q[:, 1] - exy[1]) <= 3.0]
            P = np.column_stack([Q, np.full(len(Q), float(pad.xyz[2])), np.zeros(len(Q))])
        if len(P) < 6:
            return None
        xy = P[:, :2]
        m = xy.mean(axis=0)
        try:
            ax = np.linalg.svd(xy - m, full_matrices=False)[2][0]
        except Exception:                                        # noqa: BLE001
            return None
        pr = (xy - m) @ ax
        ca, cb = xy[int(np.argmin(pr))].copy(), xy[int(np.argmax(pr))].copy()
        lab = np.zeros(len(xy), dtype=bool)
        for _it in range(10):
            lab = (np.hypot(xy[:, 0] - cb[0], xy[:, 1] - cb[1])
                   < np.hypot(xy[:, 0] - ca[0], xy[:, 1] - ca[1]))
            if lab.all() or (~lab).all():
                return None
            na, nb = xy[~lab].mean(axis=0), xy[lab].mean(axis=0)
            if np.allclose(na, ca) and np.allclose(nb, cb):
                break
            ca, cb = na, nb
        return ((ca, float(np.median(P[~lab, 2])), int((~lab).sum())),
                (cb, float(np.median(P[lab, 2])), int(lab.sum())))

    def _bl_event(self, i: int, dd, pad, pos, rpy) -> None:
        """F3: one P2r below event of drone i on its claimed entry; retarget from the k-th event on."""
        c = self.cfg
        key = int(dd.claim)
        if dd.bl_key != key:
            dd.bl_key, dd.bl_n, dd.bl_k, dd.bl_orig, dd.bl_brg, dd.bl_on = key, 0, 0, None, None, False
        dd.bl_n += 1
        self._b5_n["f3_below"] += 1
        if dd.bl_n < int(c.dsc_below_retarget):
            return
        t = round(self.step_i * 0.02, 2)
        exy = np.asarray(pad.xyz, float)[:2].copy()
        cl = None
        if dd.bl_brg is None:
            cl = self._bl_clusters(pad)
            if cl is not None:
                (ca, za, na), (cb, zb, nb) = cl
                sep = float(np.linalg.norm(ca - cb))
                nmin = max(3, int(0.2 * (na + nb)))
                if sep >= float(c.dsc_below_retarget_sep) and na >= nmin and nb >= nmin:
                    # two pads merged into one entry: hold over the nearer one, list the other as a split entry
                    near_a = float(np.linalg.norm(ca - pos[:2])) <= float(np.linalg.norm(cb - pos[:2]))
                    (cn, zn, _nn), (cf, zf, nf) = (cl if near_a else (cl[1], cl[0]))
                    self._probe_fix(pad, cn)
                    pad.xyz = np.array([float(cn[0]), float(cn[1]), float(zn)], float)
                    pad.split = True
                    pad.near = None
                    pad.near_xy = None
                    fresh = _Pad(xyz=np.array([float(cf[0]), float(cf[1]), float(zf)], float), hits=int(nf),
                                 last_seen=self.step_i, split=True)
                    self.pads.append(fresh)
                    dd.bl_brg = []
                    dd.bl_on = True
                    self._b5_n["f3_split"] += 1
                    self._b5_log.append((t, i, "F3split", key, [round(float(x), 2) for x in cn],
                                         [round(float(x), 2) for x in cf], round(sep, 2), int(_nn), int(nf)))
                    return
        if not dd.bl_brg:
            # offset mode: four bearings around the estimate, starting opposite the slide-off direction
            dd.bl_orig = exy
            so = np.asarray(pos[:2], float) - exy
            if float(np.linalg.norm(so)) >= 0.1:
                b0 = float(np.arctan2(-so[1], -so[0]))
            else:
                b0 = float(rpy[2])
            order = (0.0, 0.5 * np.pi, -0.5 * np.pi, np.pi)
            if cl is not None:
                (ca, _za, na), (cb, _zb, nb) = cl
                if float(np.linalg.norm(ca - cb)) >= 0.3:
                    big = ca if na >= nb else cb
                    w = big - exy
                    if float(np.linalg.norm(w)) > 1e-3:
                        b0 = float(np.arctan2(w[1], w[0]))      # along the cluster axis first
                        order = (0.0, np.pi, 0.5 * np.pi, -0.5 * np.pi)
            dd.bl_brg = [b0 + o for o in order]
            dd.bl_k = 0
        b = float(dd.bl_brg[dd.bl_k % len(dd.bl_brg)])
        dd.bl_k += 1
        tgt = dd.bl_orig + float(c.dsc_below_retarget_off) * np.array([np.cos(b), np.sin(b)])
        self._probe_fix(pad, tgt)
        pad.near = None
        pad.near_xy = None
        dd.bl_on = True
        self._b5_n["f3_offset"] += 1
        self._b5_log.append((t, i, "F3off", key, int(dd.bl_k), round(float(np.degrees(b)), 1),
                             [round(float(x), 2) for x in tgt]))

    def _ray_probe(self, i, dd, pad, pos, agl, dist):
        """Velocity command while probing for the platform, or None to descend as usual."""
        c = self.cfg
        g = dd.gnd_g
        if g is None:
            return None
        pz = float(pad.xyz[2])
        if pz - g < float(c.ray_probe_min_h):
            return None
        surf = float(pos[2]) - float(agl) if 0.05 < agl < 19.9 else -1e9
        on = abs(surf - pz) < abs(surf - g)
        above = float(pos[2]) - pz
        pxy = np.asarray(pad.xyz, float)[:2]
        if dd.probe is None:
            centred = dist < c.land_radius and above > 0.3
            sunk = above < -0.3
            if on or not (centred or sunk):
                return None
            if dd.probe_tries >= int(c.ray_probe_tries):
                self._refute(i)
                return np.array([0.0, 0.0, 0.6])
            dd.probe_tries += 1
            # the ray may already have crossed the platform on the way in: use those samples
            S = np.asarray(dd.gnd, float).reshape(-1, 3)
            near = S[np.hypot(S[:, 0] - pxy[0], S[:, 1] - pxy[1]) < 2.5] if len(S) else S
            if dd.probe_tries == 1 and len(near):
                hit = near[(np.abs(near[:, 2] - pz) < np.abs(near[:, 2] - g)) & (np.abs(near[:, 2] - pz) < 0.5)]
                if len(hit) >= 3:
                    new = hit[:, :2].mean(axis=0)
                    if float(np.linalg.norm(new - pxy)) > 0.3:
                        self._probe_fix(pad, new)
                        return None
            off = pos[:2] - pxy
            th = float(np.arctan2(off[1], off[0])) if float(np.linalg.norm(off)) > 0.05 else 0.0
            dd.probe = {"c": pxy.copy(), "th0": th, "th": th, "r": float(c.ray_probe_r0), "on": []}
        pr = dd.probe
        at_alt = abs(above - float(c.ray_probe_alt)) < 0.6
        if on and at_alt:
            pr["on"].append(pos[:2].copy())
        if pr["on"] and (not on or len(pr["on"]) >= 70):
            P = np.asarray(pr["on"], float)
            pr["on"] = []
            if len(P) >= 3:
                # chord of the platform disc (radius 0.6): the centre sits on its perpendicular
                # bisector, on the side away from the spiral centre (the inner side was swept already)
                M = P.mean(axis=0)
                ch = P[-1] - P[0]
                L = float(np.linalg.norm(ch))
                out = M - pr["c"]
                if L > 0.05:
                    u = np.array([-ch[1], ch[0]]) / L
                    if float(u @ out) < 0.0:
                        u = -u
                    M = M + math.sqrt(max(0.0, 0.36 - 0.25 * L * L)) * u
                elif float(np.linalg.norm(out)) > 1e-6:
                    M = M + 0.3 * out / float(np.linalg.norm(out))
                self._probe_fix(pad, M)
                dd.probe = None
                return None
        r1, r0 = float(c.ray_probe_r1), float(c.ray_probe_r0)
        pr["th"] += float(c.ray_probe_speed) / max(pr["r"], 0.3) * 0.02
        pr["r"] = r0 + (r1 - r0) * (pr["th"] - pr["th0"]) / (4.0 * np.pi)
        if pr["r"] > r1:
            dd.probe = None
            self._refute(i)
            return np.array([0.0, 0.0, 0.6])
        tgt = pr["c"] + pr["r"] * np.array([np.cos(pr["th"]), np.sin(pr["th"])])
        v = np.zeros(3)
        v[:2] = np.clip((tgt - pos[:2]) * 2.5, -1.2, 1.2)
        v[2] = float(np.clip(pz + float(c.ray_probe_alt) - float(pos[2]), -0.8, 0.8))
        return v

    def _ttc_guard(self, depth: np.ndarray, i: int, v: np.ndarray, rpy, vel) -> np.ndarray:
        c = self.cfg
        vel = np.asarray(vel, float)
        sp = float(np.linalg.norm(vel))
        if sp < float(c.ttc_min_speed):
            return v
        R = rot_from_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        u = vel / sp
        ub = R.T @ u
        out = np.asarray(v, float).copy()
        if float(ub[0]) < math.cos(math.radians(float(c.ttc_fov_deg))):
            if float(c.ttc_blind_speed) >= 0.0:
                h = float(np.hypot(out[0], out[1]))
                if h > float(c.ttc_blind_speed):
                    out[:2] *= float(c.ttc_blind_speed) / h
            return out
        _fk = "forest" if self.is_forest else str(self.map_kind)
        dist = FD.ray_clearance(depth[i], ub, PD.AP_DEPTH_MAX_M, fov_deg=PD.AP_FOV_DEG,
                                grid=c.free_grid, r_eff=float(c.ttc_r), lookahead=12.0,
                                aligned=bool(c.free_align_maps) and _fk in tuple(c.free_align_maps))
        a, lat, mg = max(float(c.ttc_decel), 0.5), max(float(c.ttc_lat), 0.0), float(c.ttc_margin)
        if dist >= sp * lat + sp * sp / (2.0 * a) + mg:
            return v
        room = max(dist - mg, 0.0)
        v_ok = a * (-lat + math.sqrt(lat * lat + 2.0 * room / a))
        along = float(out @ u)
        if along > v_ok:
            out = out - (along - v_ok) * u
        return out

    def _avoid(self, depth: np.ndarray, i: int, v: np.ndarray,
               pos=None, pad_xy=None) -> np.ndarray:
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
            if (c.clue_alt_avoid_range > 0.0 and self.d[i].phase == SEARCH
                    and getattr(self.d[i], "clue_low", False)):
                avoid_range = min(avoid_range, float(c.clue_alt_avoid_range))
        if self.map_kind == "village" and c.avoid_range_village > 0.0:
            avoid_range = c.avoid_range_village
        if (float(c.lav_avoid_r) > 0.0 and self._lav_live.get(i, -1) == self.step_i
                and self.d[i].phase == SEARCH):
            avoid_range = min(avoid_range, float(c.lav_avoid_r))   # mtn_speed LAV: close-range safety net only
        clear = None
        if (c.corridor_avoid_maps and self.map_kind in c.corridor_avoid_maps
                and str(getattr(self.d[i], "phase", "")) in tuple(c.corridor_phases or ())):
            try:
                clear = self._corridor_clearance(depth, i, v)
            except Exception:                                    # noqa: BLE001
                clear = None
        if clear is None:
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
    def _corridor_clearance(self, depth: np.ndarray, i: int, v) -> Optional[float]:
        """Distance along the commanded velocity to the first depth point inside the flight tube,
        or None when the travel direction is outside the observed part of the frame."""
        from team.detector.geom import _cloud
        c = self.cfg
        st = self._refine_state
        pos = np.asarray(st[i, POS], float)
        rpy = np.asarray(st[i, RPY], float)
        R = rot_from_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        vv = np.asarray(v, float)[:3]
        sp = float(np.linalg.norm(vv))
        if sp < 0.3:
            return None
        u = vv / sp
        fwd = np.asarray(R[:, 0], float)
        if float(np.degrees(np.arccos(np.clip(float(u @ fwd), -1.0, 1.0)))) > c.corridor_max_off_deg:
            return None
        d = np.asarray(depth[i], np.float32)
        if d.ndim == 3:
            d = d[..., 0]
        pts, valid, _ = _cloud(pos, R, d[::2, ::2], PD._geom_cfg(self.pad_cfg), PD.AP_FOV_DEG, PD.AP_RES // 2)
        if pts is None:
            return PD.AP_DEPTH_MAX_M
        rel = pts[valid].astype(np.float64) - pos[None, :]
        along = rel @ u
        L = max(float(c.corridor_min_m), sp * float(c.corridor_look_s))
        perp = np.linalg.norm(rel - along[:, None] * u[None, :], axis=1)
        hit = (along > 0.3) & (along < L) & (perp < float(c.corridor_r))
        return float(along[hit].min()) if hit.any() else PD.AP_DEPTH_MAX_M
    def _separate(self, state: np.ndarray, i: int, v: np.ndarray) -> np.ndarray:
        slots = np.asarray(state[i, MATES], float).reshape(7, 7)
        push = np.zeros(3)
        _dzm = float(self.cfg.sep_dz_max)
        _dz_on = _dzm > 0.0 and str(self.d[i].phase) in tuple(self.cfg.sep_dz_phases or ())
        for s in slots:
            if s[6] < 0.5:
                continue
            if _dz_on and abs(float(s[2])) > _dzm:
                # S5: a teammate far above / below does not push a landing drone sideways
                if float(np.linalg.norm(s[:2])) <= self.cfg.sep_radius:
                    self._b3_n["s5_skip"] += 1
                continue
            rel = s[:3]
            dist = float(np.linalg.norm(rel[:2]))
            if dist < 1e-3 or dist > self.cfg.sep_radius:
                continue
            push[:2] -= (rel[:2] / dist) * (self.cfg.sep_radius - dist) / self.cfg.sep_radius
        if not push.any():
            return v
        return v + self.cfg.sep_gain * push
    def _dead_zero(self, state: np.ndarray) -> None:
        """S11 dead_zero_steps: a frozen (crashed) drone reads exactly zero linear and angular velocity
        with a fixed position. After dead_zero_steps such steps it releases its claim (unless within
        landed_r of it) and goes DONE before the assignment, so it neither holds nor takes a pad.
        DESCEND below 0.5 m AGL and landed drones keep the existing landing handling."""
        c = self.cfg
        for i in range(self.n):
            dd = self.d[i]
            if dd.phase == DONE or dd.landed or not dd.flew:
                continue
            if dd.phase == DESCEND and float(state[i, ALT]) * 20.0 < 0.5:
                dd.zstill, dd.zprev_xy = 0, None
                continue
            xy = np.asarray(state[i, POS], float)[:2]
            frozen = (dd.zprev_xy is not None and not np.any(np.asarray(state[i, 6:12], float))
                      and float(np.linalg.norm(xy - dd.zprev_xy)) < 1e-4)
            dd.zstill = dd.zstill + 1 if frozen else 0
            dd.zprev_xy = xy.copy()
            if dd.zstill < int(c.dead_zero_steps):
                continue
            kept = None
            if dd.claim is not None:
                _dz = float(np.linalg.norm(np.asarray(self.pads[dd.claim].xyz, float)[:2] - xy))
                if c.tdo_maps and self._tdo_active(dd):
                    _dz = self._tdo_pad_dist(dd, self.pads[dd.claim], xy)   # TDO: the nearer of centre / offset
                if _dz <= float(c.landed_r):
                    kept = int(dd.claim)
                else:
                    self.pads[dd.claim].by = None
                    dd.claim = None
            self._b3_n["s11_dead"] += 1
            self._b3_log.append((round(self.step_i * 0.02, 2), i, "S11", str(dd.phase), kept))
            dd.phase = DONE

    # ---- batch 4 (NE, CLOSE, GLA, CLIMB-claim envelope); all default off ----
    def _b4_kind(self) -> str:
        return "forest" if self.is_forest else str(self.map_kind)

    def _commit_time(self) -> float:
        """The map's late-commit deadline in seconds (commit_t_by_map, else commit_t; 0 = none)."""
        c = self.cfg
        ct = float(c.commit_t)
        bym = tuple(getattr(c, "commit_t_by_map", ()) or ())
        for k in range(0, len(bym) - 1, 2):
            if str(bym[k]) == self.map_kind:
                ct = float(bym[k + 1])
                break
        return ct

    def _claim_gate(self, pad) -> int:
        """Hits _assign needs right now to hand this entry out (claim gate, late-commit gate, split gate)."""
        c = self.cfg
        gate = self.min_hits
        ct = self._commit_time()
        if ct > 0.0 and self.map_kind not in c.commit_t_skip and self.step_i >= int(ct * 50):
            gate = max(1, int(c.commit_floor), int(c.claim_hits_late))
        if pad.split:
            gate = max(int(gate), int(c.pad_split_min_hits))
        return int(gate)

    def _ne_on(self) -> bool:
        m = tuple(self.cfg.ne_maps or ())
        return bool(m) and not self.is_forest and str(self.map_kind) in m

    def _fpn_on(self) -> bool:
        """FPN (forest_phantom): this seed's kind ("forest" if is_forest else map_kind) is in fpn_maps."""
        m = tuple(self.cfg.fpn_maps or ())
        return bool(m) and ("forest" if self.is_forest else str(self.map_kind)) in m

    def _fp_inc(self, key: str, n: int = 1) -> None:
        """forest_phantom diagnostics: count an event in src_counts as fp_<key> (harness rows)."""
        self.src_counts["fp_" + key] = self.src_counts.get("fp_" + key, 0) + n

    def _fpn_tick(self, live, poses, rots, depth, h0) -> None:
        """FPN: after the detector merge of this tick, count in-view no-hit ticks per claimed (APPROACH) entry
        and refute the claim at fpn_ticks (fpn_unclaimed: unclaimed entries drop to 0 hits). h0: hits before."""
        c = self.cfg
        cand = []
        for k, p in enumerate(self.pads):
            if k >= len(h0) or int(p.hits) - int(getattr(p, "whits", 0)) > int(h0[k]):
                p.fpn_run = 0                 # new this tick or gained a detector hit: the run restarts
                continue
            if p.done or int(p.hits) < 1 or int(p.hits) >= int(c.fpn_max_hits):
                continue
            if p.by is not None:
                j = int(p.by)
                if j >= len(self.d) or self.d[j].claim != k or self.d[j].phase != APPROACH:
                    continue
                own = j
            elif bool(c.fpn_unclaimed):
                own = -1
            else:
                continue
            if p.fpn_by != own:
                p.fpn_by, p.fpn_run = own, 0
            cand.append(k)
        if not cand:
            return
        X = np.asarray([np.asarray(self.pads[k].xyz, float) for k in cand], float).reshape(-1, 3)
        dimg = np.asarray(depth, np.float32)
        H, W = int(dimg.shape[1]), int(dimg.shape[2])
        edge, occ, rng = float(c.fpn_edge), float(c.fpn_occl_tol), float(c.fpn_range)
        dscale = float(PD.AP_DEPTH_MAX_M) - 0.5
        seen = np.zeros(len(cand), dtype=bool)
        for oi, i in enumerate(live):
            # observers: the claimant only (fpn_any_view: every drone that ran the detector); unclaimed: all
            allow = np.array([bool(c.fpn_any_view) or self.pads[k].fpn_by in (-1, int(i)) for k in cand])
            ok0 = allow & ~seen
            if not ok0.any():
                continue
            R = np.asarray(rots[oi], float)
            f = R[:, 0]
            up = R[:, 2]
            r = np.cross(f, up)
            nr = float(np.linalg.norm(r))
            if nr < 1e-9:
                continue
            r = r / nr
            tv = np.cross(r, f)
            cam = np.asarray(poses[oi], float) + CAM_FWD * f + CAM_UP * up
            D = X - cam[None, :]
            zg = D @ f
            ok = ok0 & (zg > 0.5)
            if not ok.any():
                continue
            zs = np.where(ok, zg, 1.0)
            a = (D @ r) / zs
            b = (D @ tv) / zs
            ok &= (np.abs(a) <= edge) & (np.abs(b) <= edge) & (zs * np.sqrt(1.0 + a * a + b * b) <= rng)
            if not ok.any():
                continue
            u = np.clip(np.floor((a + 1.0) * 0.5 * W), 0, W - 1).astype(int)
            v = np.clip(np.floor((1.0 - b) * 0.5 * H), 0, H - 1).astype(int)
            img = dimg[i]
            if img.ndim == 3:
                img = img[..., 0]
            dm = img[v, u].astype(float) * dscale + 0.5
            seen |= ok & (dm >= zg - occ)
        for jj, k in enumerate(cand):
            if not seen[jj]:
                continue
            p = self.pads[k]
            p.fpn_run += 1
            if p.fpn_run < int(c.fpn_ticks):
                continue
            by = p.by
            p.fpn_run = 0
            lg = self.src_counts.setdefault("fp_log", [])
            if len(lg) < 60:
                lg.append([round(self.step_i * 0.02, 2), -1 if by is None else int(by), int(k), int(p.hits),
                           [round(float(x), 2) for x in p.xyz]])
            if by is not None:
                self._fp_inc("fire")
                self._refute(int(by), soft=not bool(c.fpn_hard))
            else:
                self._fp_inc("fire_u")
                p.hits = 0

    def _ne_tick(self, live, poses, rots, depth, h0) -> None:
        """NE: after the detector merge of this tick, count in-view no-hit ticks per low-hit entry and refute
        (claimed: soft _refute; unclaimed: hits = 0) at ne_ticks. h0: entry hits before the merge."""
        c = self.cfg
        kind = str(self.map_kind)
        rng = float(c.ne_range)
        pr = tuple(c.ne_range_by_map or ())
        for k in range(0, len(pr) - 1, 2):
            if str(pr[k]) == kind:
                rng = float(pr[k + 1])
                break
        need = max(1, int(c.ne_ticks))
        _nmx = int(c.ne_max_hits)
        _nmb = tuple(getattr(c, "ne_max_hits_by_map", ()) or ())
        for _k in range(0, len(_nmb) - 1, 2):
            if str(_nmb[_k]) == kind:          # audit_cov: per-map NE hit ceiling
                _nmx = int(_nmb[_k + 1])
                break
        _rgc = bool(c.rg_cne_maps) and kind in tuple(c.rg_cne_maps)
        cand = []
        for k, p in enumerate(self.pads):
            if k >= len(h0) or int(p.hits) - int(getattr(p, "whits", 0)) > int(h0[k]):
                p.ne_run = 0                  # new this tick or gained a hit: the run restarts
                if _rgc:
                    p.rg_run = 0              # CNE shadow restarts too
                    p.rg_z = False
                continue
            if p.done or p.ne_safe or int(p.hits) < 1 or int(p.hits) >= _nmx:
                continue
            if p.by is not None:
                j = int(p.by)
                if (j >= len(self.d) or self.d[j].claim != k or self.d[j].phase == DESCEND
                        or self.d[j].landed or self.d[j].phase == DONE):
                    continue                  # being landed on (or inconsistent): never refute
            cand.append(k)
        if not cand:
            return
        X = np.asarray([np.asarray(self.pads[k].xyz, float) for k in cand], float).reshape(-1, 3)
        dimg = np.asarray(depth, np.float32)
        H, W = int(dimg.shape[1]), int(dimg.shape[2])
        edge, occ = float(c.ne_edge), float(c.ne_occl_tol)
        dscale = float(PD.AP_DEPTH_MAX_M) - 0.5
        band = tuple(c.ne_dist_band or ())
        use_band = len(band) >= 3               # straight-line distance band (replaces the planar weighting)
        if use_band:
            if kind in ("city", "open"):
                rng = float(band[1])
            near, wfar = float(band[0]), float(band[2])
            weighted = True
        else:
            near, wfar = float(c.ne_range_near), float(c.ne_far_w)
            weighted = near > 0.0 and wfar != 1.0
        seen = np.zeros(len(cand), dtype=bool)
        zmin = np.full(len(cand), np.inf) if weighted else None
        seen238 = np.zeros(len(cand), dtype=bool) if _rgc else None
        for oi, i in enumerate(live):
            R = np.asarray(rots[oi], float)
            f = R[:, 0]
            up = R[:, 2]
            r = np.cross(f, up)
            nr = float(np.linalg.norm(r))
            if nr < 1e-9:
                continue
            r = r / nr
            tv = np.cross(r, f)
            cam = np.asarray(poses[oi], float) + CAM_FWD * f + CAM_UP * up
            D = X - cam[None, :]
            zg = D @ f
            ok = (zg > 0.5) & (zg <= rng)
            if not weighted:
                ok &= ~seen                   # already counted this tick (weighted: keep the nearest view)
            if not ok.any():
                continue
            zs = np.where(ok, zg, 1.0)
            a = (D @ r) / zs
            b = (D @ tv) / zs
            ok &= (np.abs(a) <= edge) & (np.abs(b) <= edge)
            ok238 = ok.copy() if _rgc else None
            dist = None
            if use_band:
                dist = zg * np.sqrt(1.0 + a * a + b * b)       # camera to the estimate, straight line
                ok &= dist <= rng
            if _rgc and ok238.any():
                # CNE shadow: 238's NE view test (planar range, no straight-line cap, weight 1)
                _u2 = np.clip(np.floor((a + 1.0) * 0.5 * W), 0, W - 1).astype(int)
                _v2 = np.clip(np.floor((1.0 - b) * 0.5 * H), 0, H - 1).astype(int)
                _im2 = dimg[i]
                if _im2.ndim == 3:
                    _im2 = _im2[..., 0]
                _dm2 = _im2[_v2, _u2].astype(float) * dscale + 0.5
                seen238 |= ok238 & (_dm2 >= zg - occ)
            if not ok.any():
                continue
            u = np.clip(np.floor((a + 1.0) * 0.5 * W), 0, W - 1).astype(int)
            v = np.clip(np.floor((1.0 - b) * 0.5 * H), 0, H - 1).astype(int)
            img = dimg[i]
            if img.ndim == 3:
                img = img[..., 0]
            dm = img[v, u].astype(float) * dscale + 0.5
            vis = ok & (dm >= zg - occ)
            seen |= vis
            if weighted:
                zmin = np.where(vis, np.minimum(zmin, dist if use_band else zg), zmin)
        if _rgc:
            for jj, k in enumerate(cand):
                p = self.pads[k]
                if seen238[jj] and p.by is None:
                    p.rg_run = int(getattr(p, "rg_run", 0)) + 1
                    if p.rg_run >= need:
                        p.rg_run = 0
                        if not bool(getattr(p, "rg_z", False)):
                            p.rg_z = True
                            self.src_counts["rg_shadow_z"] = self.src_counts.get("rg_shadow_z", 0) + 1
        for jj, k in enumerate(cand):
            if not seen[jj]:
                continue
            p = self.pads[k]
            p.ne_run += (wfar if (weighted and float(zmin[jj]) > near) else 1)
            if p.ne_run < need - 1e-9:
                continue
            run, by = (round(float(p.ne_run), 2) if weighted else int(p.ne_run)), p.by
            p.ne_run = 0
            self._ne_log.append((round(self.step_i * 0.02, 2), int(k), int(p.hits), by, int(p.aborts),
                                 [round(float(x), 2) for x in p.xyz], run))
            self._b4_n["ne_fire"] += 1
            if bool(getattr(c, "ne_diag", False)):
                self.src_counts.setdefault("ne_ev", []).append(
                    [round(self.step_i * 0.02, 2), int(k), int(p.hits), by, int(p.aborts),
                     round(float(p.xyz[0]), 2), round(float(p.xyz[1]), 2), round(float(p.xyz[2]), 2),
                     round(float(zmin[jj]), 2) if weighted else None])
            if by is not None:
                self._b4_n["ne_fire_claimed"] += 1
                if bool(getattr(c, "ne_no_hard", False)):
                    # audit_cov: never escalate an NE refute to done + blacklist
                    self._refute(int(by), soft=True, never_hard=True)
                else:
                    self._refute(int(by), soft=True)
            else:
                p.hits = 0

    def _gla_on(self) -> bool:
        m = tuple(self.cfg.gla_maps or ())
        return bool(m) and self._b4_kind() in m

    def _gla_trigger(self, i: int, frame, opos) -> None:
        """GLA trigger for observer drone i after its proposals of this tick (frame: entry -> xy)."""
        c = self.cfg
        dd = self.d[i]
        if (not self._gla_on() or dd.claim is not None
                or str(dd.phase) not in tuple(c.gla_phases or ())):
            return
        if dd.gl_pad >= 0 and self.step_i <= dd.gl_until:
            return
        _pcm = tuple(c.gla_precommit_maps or ())
        if _pcm and self._b4_kind() in _pcm:
            _ct = self._commit_time()
            if _ct > 0.0 and self.step_i >= int(_ct * 50):
                return                        # city / open: glances only before the late commit
        best = None
        for k in frame:
            if not (0 <= int(k) < len(self.pads)):
                continue
            p = self.pads[k]
            if p.by is not None or p.done:
                continue
            h = int(p.hits)
            gate = self._claim_gate(p)
            if not (1 <= h < gate):
                continue
            if int(c.gla_max_hits) > 0 and h > int(c.gla_max_hits):
                continue
            rr = float(np.hypot(float(p.xyz[0]) - float(opos[0]), float(p.xyz[1]) - float(opos[1])))
            if rr > float(c.gla_range):
                continue
            if best is None or h < best[1]:
                best = (int(k), h, gate, rr)
        if best is None:
            return
        k, h, gate, rr = best
        dd.gl_pad = k
        dd.gl_until = self.step_i + max(1, int(round(float(c.gla_s) * 50)))
        self._b4_n["gla_trig"] += 1
        self._gla_log.append((round(self.step_i * 0.02, 2), int(i), k, h, int(gate), round(rr, 1)))

    def _gla_yaw(self, i: int, dd, pos, rpy, travel_yaw):
        """GLA camera yaw for drone i this step, or None (no look-at active / skipped this step)."""
        c = self.cfg
        k = int(dd.gl_pad)
        if k < 0:
            return None
        ok = (self.step_i <= dd.gl_until and k < len(self.pads) and dd.claim is None
              and str(dd.phase) in tuple(c.gla_phases or ()) and self._gla_on())
        if ok:
            p = self.pads[k]
            ok = p.by is None and not p.done and 1 <= int(p.hits) < self._claim_gate(p)
        if not ok:
            dd.gl_pad = -1
            return None
        p = self.pads[k]
        dx = float(p.xyz[0]) - float(pos[0])
        dy = float(p.xyz[1]) - float(pos[1])
        rr = math.hypot(dx, dy)
        brg = math.atan2(dy, dx)
        keep = math.acos(min(1.0, float(c.gla_keep) / max(rr, 1e-6)))
        off = (float(rpy[2]) - brg + math.pi) % (2.0 * math.pi) - math.pi
        tgt = min(max(abs(off), keep), math.radians(float(c.gla_maxoff_deg)))
        if rr * math.cos(tgt) > 19.75:
            self._b4_n["gla_skip_rng"] += 1          # no in-view yaw brings it inside the depth range
            return None
        yc = brg + (1.0 if off >= 0.0 else -1.0) * tgt
        if travel_yaw is not None and self._use_free() and self._b4_kind() in ("city", "forest", "open"):
            dtr = abs((yc - float(travel_yaw) + math.pi) % (2.0 * math.pi) - math.pi)
            if dtr > math.radians(float(c.gla_travel_deg)):
                self._b4_n["gla_skip_trav"] += 1
                return None
        self._b4_n["gla_steps"] += 1
        return float(yc)

    def _cenv_on(self) -> bool:
        m = tuple(self.cfg.climb_claim_env_maps or ())
        return bool(m) and self._b4_kind() in m

    def _climb_rate(self) -> float:
        """CLIMB's vertical command (climb_speed, per map from climb_speed_by_map)."""
        cfg = self.cfg
        _cs = float(cfg.climb_speed)
        _csm = tuple(getattr(cfg, "climb_speed_by_map", ()) or ())
        for _k in range(0, len(_csm) - 1, 2):
            if str(_csm[_k]) == self.map_kind:
                _cs = float(_csm[_k + 1])
                break
        return _cs

    def _cenv_apply(self, i: int, dd, v_des, agl: float, rpy):
        """CLIMB-claim envelope on this APPROACH command (dd.cenv set); ends at CLIMB's exit height."""
        c = self.cfg
        top = (c.forest_alt if self.is_forest else c.cruise_alt) - 0.6
        if agl >= top:
            dd.cenv = False
            self._cenv_log.append((round(self.step_i * 0.02, 2), int(i), dd.claim, "off", round(float(agl), 2)))
            return v_des
        v = np.asarray(v_des, float).copy()
        v[2] = self._climb_rate()
        tilt = float(max(abs(float(rpy[0])), abs(float(rpy[1]))))
        if float(c.climb_creep_tilt) > 0.0 and tilt >= float(c.climb_creep_tilt):
            v[:2] = 0.0
        else:
            cap = float(c.cruise_speed) * float(c.climb_creep)
            s = float(np.hypot(v[0], v[1]))
            if s > cap and s > 1e-9:
                v[:2] *= cap / s
        self._b4_n["cenv_steps"] += 1
        return v

    def _fove_tick(self, state, depth) -> None:
        """bughunt FOVE: record frames at cfg.fove_rec, estimate the camera FOV at cfg.fove_at."""
        cfg = self.cfg
        fr = getattr(self, "_fove_frames", None)
        if fr is None:
            fr = self._fove_frames = {}
        if self.step_i in tuple(int(x) for x in cfg.fove_rec):
            for i in range(min(int(state.shape[0]), int(cfg.fove_max_drones))):
                R = rot_from_rpy(*np.asarray(state[i, RPY], float))
                pos = np.asarray(state[i, POS], float)
                cam = pos + R[:, 0] * CAM_FWD + R[:, 2] * CAM_UP
                fr.setdefault(i, []).append((cam, R, np.asarray(depth[i], np.float64).reshape(depth.shape[1], depth.shape[2]).copy()))
        if self.step_i == int(cfg.fove_at):
            import time as _ft
            _t0 = _ft.perf_counter()
            try:
                ests = _fove_estimate([fr[i] for i in sorted(fr)])
            except Exception:                                    # noqa: BLE001
                ests = []
            self.src_counts["fove_ms"] = round((_ft.perf_counter() - _t0) * 1000.0, 1)
            self._fove_frames = {}
            self._fove_deg = 90.0
            sc = self.src_counts
            sc["fove_pairs"] = len(ests)
            if len(ests) >= int(cfg.fove_min_pairs):
                med = float(np.median(ests))
                mad = float(np.median(np.abs(np.asarray(ests) - med)))
                sc["fove_mad"] = round(mad, 3)
                sc["fove_raw"] = round(med, 3)
                if mad <= float(cfg.fove_tol) and 87.5 <= med <= 92.5:
                    self._fove_deg = med
                    PD.AP_FOV_DEG = med
                    for nm in ("detector", "general_detector", "village_detector", "mountain_detector",
                               "forest_detector", "city_detector"):
                        det = getattr(self, nm, None)
                        if det is not None and hasattr(det, "fov_deg"):
                            det.fov_deg = med
            sc["fove_deg"] = round(float(self._fove_deg), 3)

    def _fcl_tick(self, state, depth) -> None:
        """fov_clue FCL: FOVE frames at cfg.fcl_rec, the estimate at cfg.fcl_at, then apply it on fcl_maps kinds."""
        return   # clean build: FCL (FOV-coupled clue) removed
        cfg = self.cfg
        if self._fcl_est is None:
            fr = self._fcl_frames
            if fr is None:
                fr = self._fcl_frames = {}
            if self.step_i in tuple(int(x) for x in cfg.fcl_rec):
                for i in range(min(int(state.shape[0]), int(cfg.fcl_max_drones))):
                    R = rot_from_rpy(*np.asarray(state[i, RPY], float))
                    pos = np.asarray(state[i, POS], float)
                    cam = pos + R[:, 0] * CAM_FWD + R[:, 2] * CAM_UP
                    fr.setdefault(i, []).append(
                        (cam, R, np.asarray(depth[i], np.float64).reshape(depth.shape[1], depth.shape[2]).copy()))
            if self.step_i >= int(cfg.fcl_at):
                import time as _ft
                _t0 = _ft.perf_counter()
                try:
                    ests = _fove_estimate([fr[i] for i in sorted(fr)])
                except Exception:                                    # noqa: BLE001
                    ests = []
                sc = self.src_counts
                sc["fcl_ms"] = round((_ft.perf_counter() - _t0) * 1000.0, 1)
                sc["fcl_pairs"] = len(ests)
                self._fcl_frames = None
                self._fcl_est = (None, None)
                if len(ests) >= int(cfg.fcl_min_pairs):
                    med = float(np.median(ests))
                    mad = float(np.median(np.abs(np.asarray(ests) - med)))
                    med += float(cfg.fcl_bias)
                    sc["fcl_fov"] = round(med, 3)
                    sc["fcl_mad"] = round(mad, 3)
                    if mad <= float(cfg.fcl_mad_max) and 87.9 <= med <= 92.1:
                        self._fcl_est = (med, mad)
            if self._fcl_est is None:
                return
        med, mad = self._fcl_est
        if med is None:
            return
        want = False
        try:
            kind = self._route_kind()
            want = kind in tuple(cfg.fcl_maps) and (self._band_locked or kind == "mountain")
        except Exception:                                            # noqa: BLE001
            want = False
        if want == self._fcl_applied:
            return
        if want:
            s = float(np.clip((med - 90.0) / 2.0, -1.0, 1.0))
            if cfg.fcl_scramble:
                c = np.asarray(self.clue if self.clue is not None else state[0, POS], float)
                key = int(abs(float(c[0])) * 1000.0 + abs(float(c[1])) * 17.0) % (2 ** 31 - 1)
                s = float(np.random.RandomState(key).uniform(-1.0, 1.0))
            sd = max(float(cfg.fcl_sd_min), float(cfg.fcl_sd_k) * mad / 2.0)
            _fcl_set(s, sd, float(cfg.fcl_w), float(cfg.fcl_geo_floor))
            self.src_counts["fcl_s"] = round(s, 4)
            self.src_counts["fcl_step"] = int(self.step_i)
        else:
            _fcl_set(None)
            self.src_counts["fcl_off_step"] = int(self.step_i)
        self._fcl_applied = want
        hp = getattr(self, "_hunt_post", None)
        if hp is not None:
            hp._found_key = None
            hp._grid = None
        rr = getattr(self, "replan_router", None)
        if rr is not None and getattr(rr, "ready", False):
            try:
                from team.route_replan import pad_posterior as _pp
                if getattr(rr, "_geo", None) is not None:
                    pass
                elif getattr(rr, "_mix_w", None) is not None:
                    rr.gx, rr.gy, rr.rho0 = rr._mix_field(())
                else:
                    rr.gx, rr.gy, rr.rho0 = _pp.pad_field(rr.S, rr.clue, rr.kind, step=rr.cfg.step,
                                                          radial=rr.radial, W=rr.W)
                rr._ring_cache = {}
                rr.found_sig = None
            except Exception:                                        # noqa: BLE001
                pass

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
            _g_t0 = getattr(self, "_g_t0", None)
            self.reset()
            self._g_t0 = _g_t0              # act guard: keep this act()'s start stamp through the reset
            self.n = n
            self.d = [_Drone(theta=0.0) for _ in range(n)]
        self.step_i += 1
        if cfg.fcl_maps:
            self._fcl_tick(state, depth)
        if cfg.fove_on and getattr(self, "_fove_deg", None) is None:
            self._fove_tick(state, depth)
        if self.clue is None:
            self.clue = (np.asarray(state[0, POS], float)
                         + np.asarray(state[0, GOAL_OFF], float))
            if tuple(getattr(cfg, "bnx_maps", ()) or ()):
                try:
                    import team.autopilot.geo_posterior as _bnx_gp     # BNX (r9_prior P3)
                    _bnx_gp.N_FLEET = int(n)
                    _bnx_gp.BNX_MAPS = tuple(cfg.bnx_maps)
                except Exception:                                # noqa: BLE001
                    pass
            if cfg.hunt:
                try:
                    from team.autopilot.geo_posterior import GeoPosterior
                    self._hunt_post = GeoPosterior(K=4000)
                    self._hunt_post.observe_reset(observation)
                except Exception:                                # noqa: BLE001
                    self._hunt_post = None
            self._set_z_band(state)
            self._assign_wedges(state)
            self._own_start = np.asarray(state[:, POS], float).copy()
            self._wpost_init()
            self._init_mass_grid()
            for i in range(n):
                self.bad.append(np.asarray(state[i, POS], float)[:2].copy())
                self._start_z.append(float(state[i, 2]))
            mine = np.asarray(state[0, POS], float)
            for s in np.asarray(state[0, MATES], float).reshape(7, 7):
                if s[6] >= 0.5:
                    self.bad.append((mine + s[:3])[:2].copy())
                    self._start_z.append(float((mine + s[:3])[2]))
            self._n_start_bad = len(self.bad)      # start_mask_r: the entries so far are start platforms

        if (getattr(self, "_splan_deferred", None) is not None and self.step_i >= self._splan_deferred[1]
                and self.step_i % cfg.detect_every != 0
                and not False):   # clean build: FCL CEM start delay removed
            route0, self._splan_deferred = self._splan_deferred[0], None
            new_route = self._start_splan(route0)
            if self._splan is not None and self._route_waypoints is not None and new_route is not route0:
                for j in range(min(self.n, new_route.shape[0])):
                    if self.d[j].route_idx == 0 and self.d[j].phase in (CLIMB, SEARCH):
                        self._route_waypoints[j] = [np.asarray(w, dtype=float) for w in new_route[j]]
        if cfg.search_replan and self.step_i % cfg.detect_every != 0 and (self._splan is None or self._splan.done):
            self._replan_tick(state)
        if (self._splan is not None and not self._splan.done
                and self.step_i % cfg.detect_every != 0):     # never on a detector tick
            try:
                self._splan.step(float(cfg.search_plan_budget), chunk=2)
                new_route = np.asarray(self._splan.best(), dtype=float)
                self._splan_pred = (self._splan.holdout_scores if self.cfg.search_plan_holdout else self._splan.scores)
                if (self._route_waypoints is not None
                        and self._splan_gain() >= float(cfg.search_plan_min_gain)):
                    for j in range(min(self.n, new_route.shape[0])):
                        if self.d[j].route_idx == 0 and self.d[j].phase in (CLIMB, SEARCH):
                            self._route_waypoints[j] = [np.asarray(w, dtype=float) for w in new_route[j]]
            except Exception:                                        # noqa: BLE001
                self._splan = None
        if (cfg.hunt and self._hunt_post is not None and self.step_i % cfg.detect_every == 0
                and (not cfg.hunt_need_found or ((not tuple(cfg.hunt_maps or ()) or self._hunt_kind() in tuple(cfg.hunt_maps))
                                                 and (int(cfg.hunt_max_n) <= 0 or self.n <= int(cfg.hunt_max_n))))):
            try:
                self._hunt_mark(state)
            except Exception:                                    # noqa: BLE001
                pass
        if cfg.gkc_maps and not self._gkc_done and not self._band_locked:
            try:
                self._gkc_sample(state)                         # gkc (round 10 P2)
            except Exception:                                    # noqa: BLE001
                pass
        if cfg.gkc_city_pin and not self._gkcc_done:
            try:
                self._gkcc_step(state)                          # C-L4 gkc_city_pin
            except Exception:                                    # noqa: BLE001
                self._gkcc_done = True
        if self.step_i % cfg.detect_every == 0:
            live = [i for i in range(n)
                    if self.d[i].phase in (SEARCH, APPROACH, CLIMB)]
            if live:
                if (cfg.kind_wall_maps and not self._band_locked
                        and self.step_i * 0.02 <= float(cfg.kind_wall_sec)):
                    try:
                        self._wall_sample(state, depth, live)
                    except Exception:                            # noqa: BLE001
                        pass
                self._vote_forest(depth, live)
                if cfg.gkc_maps and self._band_locked and not self._gkc_done:
                    try:
                        self._gkc_decide()                      # gkc: before the route is planned
                    except Exception:                            # noqa: BLE001
                        self._gkc_done = True
                if cfg.prv_maps:
                    try:
                        self._prv_tick(state, depth, live)      # prv: provisional route before the lock
                    except Exception:                            # noqa: BLE001
                        self._prv_force = None
                        self._prv_off = True
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
            if live and det is not None and float(cfg.act_guard_s) > 0.0:
                # act guard: this tick detects the drones that fit in the step, the rest on the next steps
                self._g_n["ticks"] += 1
                self._det_close(depth)                  # a cycle still open from the last tick ends here
                run = self._guard_pick(live, forced=())
                self._g_pend = [i for i in live if i not in run]
                if self._g_pend:
                    self._g_n["split_ticks"] += 1
                    self._g_n["deferred"] += len(self._g_pend)
                self._g_cyc = self._det_block(state, depth, run, det)
                if not self._g_pend:
                    self._det_close(depth)              # not split: NE / FPN / T0-a right after the merge, as shipped
            elif live and det is not None:
                rots = [rot_from_rpy(*np.asarray(state[i, RPY], float)) for i in live]
                poses = [np.asarray(state[i, POS], float) for i in live]
                self._update_memory(live, poses, rots, depth)
                try:
                    batch = det.propose_batch(poses, rots, depth[live])
                except Exception:
                    batch = [[] for _ in live]
                _ne = self._ne_on()                       # batch 4 NE: entry hits before the merge
                _fpn = bool(cfg.fpn_maps) and self._fpn_on()
                _h0 = ([int(p.hits) - int(getattr(p, "whits", 0)) for p in self.pads]   # WCONF hits never reset NE / FPN
                       if (_ne or _fpn) else None)
                _gla = self._gla_on()
                _t0t = bool(cfg.t0a_maps) and self._t0a_on()   # detv2 T0-a: entry detector hits before the merge
                _h1 = [int(p.hits) - int(getattr(p, "whits", 0)) for p in self.pads] if _t0t else None
                _t0n = self._t0_n
                for oi, props in enumerate(batch):
                    frame = {}
                    opos = np.asarray(poses[oi], float)
                    ofwd = np.asarray(rots[oi], float)[:, 0]
                    for q in props:
                        c = np.asarray(q.centre, float)
                        if getattr(q, "weak", False):          # detv2 WCONF: weak proposals only confirm
                            _t0n["c_weak"] += 1
                            self._merge_weak(c)
                            continue
                        if getattr(q, "far", False):           # detv2 T0-c diag
                            _t0n["c_far"] += 1
                        if getattr(q, "snap", False):
                            _t0n["c_snap"] += 1
                        w = c[:2] - opos[:2]
                        vr = float(np.linalg.norm(w))
                        if vr > 1e-6:
                            wn = w / vr
                            vb = abs(float(np.degrees(np.arctan2(
                                ofwd[0] * wn[1] - ofwd[1] * wn[0],
                                float(ofwd[:2] @ wn)))))
                        else:
                            vb = 0.0
                        if int(cfg.zband_rescue_hits) > 0:      # score only feeds the z-band rescue
                            _qs = getattr(q, "score", None)
                            self._merge_score = None if _qs is None else float(_qs)
                        self._merge(c, rng=self._refine_rng(c, poses, batch),
                                    view=(vr, vb), obs=opos, frame=frame)
                    if _gla and frame:
                        self._gla_trigger(live[oi], frame, opos)
                if _t0t:
                    # detv2 T0-a: a tick on which an entry gained detector hits as its own entry
                    for _k1, _p1 in enumerate(self.pads):
                        if _k1 >= len(_h1) or int(_p1.hits) - int(getattr(_p1, "whits", 0)) > _h1[_k1]:
                            _p1.t0a_ticks += 1
                if _ne:
                    self._ne_tick(live, poses, rots, depth, _h0)
                if _fpn:
                    self._fpn_tick(live, poses, rots, depth, _h0)
            else:
                for i in live:
                    self._detect(state, depth, i)
                if cfg.mem_grid:
                    rots = [rot_from_rpy(*np.asarray(state[i, RPY], float)) for i in live]
                    poses = [np.asarray(state[i, POS], float) for i in live]
                    self._update_memory(live, poses, rots, depth)
        elif float(cfg.act_guard_s) > 0.0 and self._g_pend:
            # act guard catch-up: the drones deferred from the last detector tick, on their current frame
            pend = [i for i in self._g_pend if i < n and self.d[i].phase in (SEARCH, APPROACH, CLIMB)]
            self._g_pend = []
            det = (None if (self.is_forest and cfg.forest_geometric) else
                   (self.forest_detector if (self.is_forest and self.forest_detector is not None)
                    else self.city_detector if (self.is_city and self.city_detector is not None)
                    else self.detector))
            if pend and det is not None:
                self._g_n["catchup_steps"] += 1
                age = int(cfg.act_guard_max_age)
                forced = [i for i in pend if self.step_i - int(self._g_last.get(i, -10 ** 6)) >= age]
                run = self._guard_pick(pend, forced=forced)
                self._g_pend = [i for i in pend if i not in run]
                self._g_cyc = self._det_block(state, depth, run, det, cycle=self._g_cyc)
            if not self._g_pend:
                self._det_close(depth)                  # the cycle's last drones are in: NE / FPN / T0-a once
        if float(cfg.act_guard_s) > 0.0:
            import time as _gtime
            self._g_mark = _gtime.perf_counter()
        if int(cfg.dead_zero_steps) > 0:
            self._dead_zero(state)
        _cenv = self._cenv_on()
        _ph0 = [d.phase for d in self.d] if _cenv else None
        self._assign(state)
        if float(cfg.assign_2opt_m) > 0.0:
            self._assign_2opt(state)
        if _cenv:
            # batch 4 CLIMB-claim envelope: drones that got their claim while in CLIMB
            for i in range(n):
                dd = self.d[i]
                if _ph0[i] == CLIMB and dd.phase == APPROACH and dd.claim is not None and not dd.cenv:
                    dd.cenv = True
                    self._b4_n["cenv_on"] += 1
                    self._cenv_log.append((round(self.step_i * 0.02, 2), i, int(dd.claim), "on",
                                           round(float(state[i, ALT]) * 20.0, 2)))
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
        if cfg.om_maps and self._om_on():
            try:
                self._om_update(state, depth)                 # blind_guard OM
            except Exception:                                    # noqa: BLE001
                self._om_cnt("err_upd")
        elif cfg.om_skip_route_open and cfg.om_maps and self._om_skip_open():
            self.src_counts["hz_om_skip"] = self.src_counts.get("hz_om_skip", 0) + 1   # candF hazard i
        if cfg.om2_maps and self._om2_on():
            _t2 = _om2_clock()
            try:
                self._om2_update(state, depth)                # candHc OM2 fleet obstacle memory
            except Exception:                                    # noqa: BLE001
                self._om2_cnt("err_upd")
            self._om2_cnt("us_upd", int(1e6 * (_om2_clock() - _t2)))
        act = np.zeros((n, 5), dtype=np.float32)
        for i in range(n):
            act[i] = self._one(state, depth, i)
        if bool(getattr(cfg, "rg_diag", False)):
            self._rg_diag_tick()
        return act

    def _guard_limit(self) -> float:
        cfg = self.cfg
        return float(cfg.act_guard_first_s) if self._g_first else float(cfg.act_guard_s)

    def _guard_pick(self, live, forced=()) -> list:
        """act guard: the drones of `live` whose detection fits in this act() (all of them with time to spare)."""
        import time as _gtime
        t0 = self._g_t0
        if t0 is None or not live:
            return list(live)
        lim = self._guard_limit()
        el = _gtime.perf_counter() - t0
        k = int((lim - el - self._g_rest) / max(self._g_det, 1e-4))
        if k >= len(live):
            return list(live)
        k = max(k, 1 if el < lim else 0)
        forced = [i for i in live if i in forced]
        if forced:
            self._g_n["forced"] += len(forced)
        rest = sorted((i for i in live if i not in forced), key=lambda i: (int(self._g_last.get(i, -10 ** 6)), i))
        run = forced + rest[:max(0, k - len(forced))]
        return sorted(run)

    def _det_block(self, state, depth, live, det, cycle=None):
        """act guard: the shipped detector block of act() for `live` (propose, merge), in calls of act_guard_chunk
        drones with the time re-checked before each further call: drones that no longer fit go back to _g_pend.
        Proposals do not depend on the other drones of a call (batch_check.py) and the merges run in the shipped
        order (drone by drone, _refine_rng on the calls so far, which hold the proposal's own drone), so a block
        that runs every drone is the shipped block. The per-tick counters (T0-a ticks, NE, FPN) belong to the
        detection cycle and run once in _det_close, after all of the cycle's merges. Returns the cycle."""
        import time as _gtime
        cfg = self.cfg
        if not live:
            return cycle
        _ts = _gtime.perf_counter()
        rots = [rot_from_rpy(*np.asarray(state[i, RPY], float)) for i in live]
        poses = [np.asarray(state[i, POS], float) for i in live]
        self._update_memory(live, poses, rots, depth)
        if cycle is None:
            _ne = self._ne_on()                       # batch 4 NE: entry hits before the merge
            _fpn = bool(cfg.fpn_maps) and self._fpn_on()
            _h0 = ([int(p.hits) - int(getattr(p, "whits", 0)) for p in self.pads]   # WCONF hits never reset NE / FPN
                   if (_ne or _fpn) else None)
            _t0t = bool(cfg.t0a_maps) and self._t0a_on()   # detv2 T0-a: entry detector hits before the merge
            _h1 = [int(p.hits) - int(getattr(p, "whits", 0)) for p in self.pads] if _t0t else None
            cycle = {"ne": _ne, "fpn": _fpn, "h0": _h0, "t0t": _t0t, "h1": _h1,
                     "live": [], "poses": [], "rots": [], "rows": {}, "split": False}
        else:
            cycle["split"] = True
        _gla = self._gla_on()
        _t0n = self._t0_n
        C = max(1, int(getattr(cfg, "act_guard_chunk", 2)))
        batch = []
        done = 0
        while done < len(live):
            m = min(C, len(live) - done)
            if done > 0 and self._g_t0 is not None:
                el = _gtime.perf_counter() - self._g_t0
                if el + m * self._g_det + self._g_rest > self._guard_limit():
                    self._g_n["chunk_stops"] += 1
                    break
            sub = live[done:done + m]
            try:
                props_sub = det.propose_batch(poses[done:done + m], rots[done:done + m], depth[sub])
            except Exception:
                props_sub = [[] for _ in sub]
            batch += props_sub
            for oi in range(done, done + m):
                props = batch[oi]
                frame = {}
                opos = np.asarray(poses[oi], float)
                ofwd = np.asarray(rots[oi], float)[:, 0]
                for q in props:
                    c = np.asarray(q.centre, float)
                    if getattr(q, "weak", False):          # detv2 WCONF: weak proposals only confirm
                        _t0n["c_weak"] += 1
                        self._merge_weak(c)
                        continue
                    if getattr(q, "far", False):           # detv2 T0-c diag
                        _t0n["c_far"] += 1
                    if getattr(q, "snap", False):
                        _t0n["c_snap"] += 1
                    w = c[:2] - opos[:2]
                    vr = float(np.linalg.norm(w))
                    if vr > 1e-6:
                        wn = w / vr
                        vb = abs(float(np.degrees(np.arctan2(
                            ofwd[0] * wn[1] - ofwd[1] * wn[0],
                            float(ofwd[:2] @ wn)))))
                    else:
                        vb = 0.0
                    if int(cfg.zband_rescue_hits) > 0:      # score only feeds the z-band rescue
                        _qs = getattr(q, "score", None)
                        self._merge_score = None if _qs is None else float(_qs)
                    self._merge(c, rng=self._refine_rng(c, poses[:len(batch)], batch),
                                view=(vr, vb), obs=opos, frame=frame)
                if _gla and frame:
                    self._gla_trigger(live[oi], frame, opos)
            done += m
        ran = live[:done]
        if done < len(live):                            # out of time: the rest go first on the next step
            self._g_pend = list(live[done:]) + [i for i in self._g_pend if i not in live[done:]]
            self._g_n["deferred"] += len(live) - done
        cycle["live"] += ran
        cycle["poses"] += poses[:done]
        cycle["rots"] += rots[:done]
        if self._g_pend or cycle["split"]:              # frames the cycle's NE / FPN will need later
            for i in ran:
                cycle["rows"][i] = np.array(depth[i], copy=True)
        for i in ran:
            self._g_last[i] = self.step_i
        # rise fast, decay slowly. The first act() counts too: its warm-up makes it a pessimistic first reading, and
        # on a contended host it is the only reading before the second tick (with det0 alone that tick overshoots).
        if done:
            per = (_gtime.perf_counter() - _ts) / done
            a = 0.5 if per > self._g_det else 0.1
            self._g_det += a * (per - self._g_det)
        return cycle

    def _det_close(self, depth) -> None:
        """act guard: T0-a / NE / FPN of the open detection cycle, once, over every drone it detected (the shipped
        per-tick semantics: a pad counts one NE tick per cycle, with the hits it had before the cycle's first merge)."""
        cyc, self._g_cyc = self._g_cyc, None
        if cyc is None or not cyc["live"]:
            return
        if cyc["t0t"]:
            _h1 = cyc["h1"]
            # detv2 T0-a: a tick on which an entry gained detector hits as its own entry
            for _k1, _p1 in enumerate(self.pads):
                if _k1 >= len(_h1) or int(_p1.hits) - int(getattr(_p1, "whits", 0)) > _h1[_k1]:
                    _p1.t0a_ticks += 1
        if not (cyc["ne"] or cyc["fpn"]):
            return
        live, poses, rots = cyc["live"], cyc["poses"], cyc["rots"]
        dep = depth
        if cyc["split"] or cyc["rows"]:
            order = sorted(range(len(live)), key=lambda j: live[j])
            live = [live[j] for j in order]
            poses = [poses[j] for j in order]
            rots = [rots[j] for j in order]
            dep = np.array(depth, copy=True)
            for i, row in cyc["rows"].items():
                if i < len(dep):
                    dep[i] = row
        if cyc["ne"]:
            self._ne_tick(live, poses, rots, dep, cyc["h0"])
        if cyc["fpn"]:
            self._fpn_tick(live, poses, rots, dep, cyc["h0"])

    def _guard_end(self, t_end: float) -> None:
        """act guard: called by the controller when act() returns; updates the after-detector estimate."""
        t0, mk = self._g_t0, self._g_mark
        if t0 is not None:
            tot = t_end - t0
            if tot > self._guard_limit():
                self._g_n["over_limit"] += 1
            self._g_n["max_act_ms"] = max(float(self._g_n["max_act_ms"]), round(tot * 1e3, 1))
        if mk is not None and not self._g_first:
            r = max(0.0, t_end - mk)
            a = 0.5 if r > self._g_rest else 0.1
            self._g_rest += a * (r - self._g_rest)
        self._g_first = False
        self._g_mark = None

    def _rg_diag_tick(self) -> None:
        """regress rg_diag: log every claim change since the previous act() (diagnostics only)."""
        prev = getattr(self, "_rg_prev", None)
        if prev is None or len(prev) != self.n:
            prev = [None] * self.n
        ev = self.src_counts.setdefault("rg_ev", [])
        for i, dd in enumerate(self.d):
            if dd.claim != prev[i] and len(ev) < 400:
                k = dd.claim
                p = self.pads[k] if k is not None and k < len(self.pads) else None
                ev.append([round(self.step_i * 0.02, 2), int(i), prev[i], k,
                           None if p is None else int(p.hits),
                           None if p is None else [round(float(x), 2) for x in p.xyz],
                           str(dd.phase), round(float(getattr(dd, "claim_dist", 0.0) or 0.0), 1),
                           None if p is None else int(p.aborts)])
        self._rg_prev = [dd.claim for dd in self.d]
    def _one(self, state: np.ndarray, depth: np.ndarray, i: int) -> np.ndarray:
        cfg = self.cfg
        dd = self.d[i]
        pos = np.asarray(state[i, POS], float)
        vel = np.asarray(state[i, VEL], float)
        agl = float(state[i, ALT]) * 20.0
        self._cov_mark(i, pos)
        agl_prev = dd.agl_prev
        dd.agl_prev = agl
        rpy = np.asarray(state[i, RPY], float)
        yaw = float(rpy[2])
        if _YT_DIAG:
            dd.yt = 0
            dd.yt_h = None
        if dd.cenv and (dd.phase != APPROACH or dd.claim is None):
            dd.cenv = False                 # batch 4 envelope: only while approaching the CLIMB-time claim
        if dd.phase == DONE:
            return _encode(slew_velocity(np.zeros(3), vel, cfg.max_delta_v),
                           cfg.speed_limit, yaw)
        if agl > 1.5:
            dd.flew = True
        if (cfg.lav_maps and dd.flew and (self.step_i + i) % max(1, int(cfg.lav_every)) == 0
                and self._lav_on()):
            try:
                self._lav_update(i, state, depth)       # mtn_speed LAV: fleet max-height grid
            except Exception:                            # noqa: BLE001
                pass
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
        if pad is None and dd.vf_key != -1:
            dd.vf_key = -1                  # APXV-R: a later claim of the same pad is checked afresh
        if (int(cfg.dsc_gate_inview_min) > 0 and dd.phase == APPROACH and pad is not None
                and self.step_i % max(1, int(cfg.detect_every)) == 0):
            # S1: count detector ticks on which the claimed pad can be in the camera frame (yaw ignored)
            _dxy = float(np.linalg.norm(np.asarray(pad.xyz, float)[:2] - pos[:2]))
            _h = float(pos[2]) - float(pad.xyz[2])
            if (_dxy > 1e-3 and math.degrees(math.atan2(_h, _dxy)) <= float(cfg.dsc_gate_inview_deg)
                    and _dxy <= 19.5):
                dd.gate_inview += 1
        if (cfg.tko_hop_guard_maps and pad is not None and dd.phase in (APPROACH, DESCEND) and not dd.flew
                and ("forest" if self.is_forest else str(self.map_kind)) in tuple(cfg.tko_hop_guard_maps)
                and 0.45 < float(np.linalg.norm(np.asarray(pad.xyz, float)[:2] - pos[:2])) < float(cfg.tko_hop_r)):
            # HOPG: claimed a pad next to the start before flying; climb straight up first
            self._hopg_n += 1
            if _YT_DIAG:
                dd.yt |= 8192
            return _encode(slew_velocity(np.array([0.0, 0.0, 1.0]), vel, cfg.max_delta_v),
                           cfg.speed_limit, yaw)
        v_des = np.zeros(3)
        if dd.phase == CLIMB:
            _cs = float(cfg.climb_speed)
            _csm = tuple(getattr(cfg, "climb_speed_by_map", ()) or ())
            for _k in range(0, len(_csm) - 1, 2):
                if str(_csm[_k]) == self.map_kind:
                    _cs = float(_csm[_k + 1])
                    break
            v_des = np.array([0.0, 0.0, _cs])
            tgt = self._learned_route_target(i, pos)
            if tgt is None:
                tgt = self._orphan_target(i, pos)
            if tgt is None:
                tgt = self._spiral_target(i)
            d = tgt[:2] - pos[:2]
            _gtrav = (float(np.arctan2(d[1], d[0])) if float(np.linalg.norm(d)) > 1e-6 else None)
            risen = (float(pos[2]) - float(self._own_start[i][2])
                     if self._own_start is not None else 1e9)
            tilt_now = float(max(abs(float(rpy[0])), abs(float(rpy[1]))))
            _cyh = cfg.climb_yaw_hold and self.map_kind not in tuple(cfg.climb_yaw_hold_skip or ())
            if _cyh and float(np.linalg.norm(d)) > 1e-6:
                dd.climb_yaw = float(np.arctan2(d[1], d[0]))
            _tko = bool(cfg.tko_maps) and self.map_kind in tuple(cfg.tko_maps)
            if _tko and float(cfg.tko_gate_cam_deg) > 0.0:
                # full-speed creep only while the camera faces the target, the path ahead is clear
                # and the drone is still gaining height over the ground
                if agl_prev >= 0.0:
                    dd.tko_rate = 0.8 * dd.tko_rate + 0.2 * (agl - agl_prev) / 0.02
                _off = abs((yaw - float(np.arctan2(d[1], d[0])) + np.pi) % (2.0 * np.pi) - np.pi) \
                    if float(np.linalg.norm(d)) > 1e-6 else np.pi
                _tko = (float(np.linalg.norm(d)) > 1e-6
                        and _off <= math.radians(float(cfg.tko_gate_cam_deg))
                        and self._clearance(depth, i) >= float(cfg.tko_gate_clear_m)
                        and dd.tko_rate >= float(cfg.tko_gate_aglrate))
            _ctilt = (float(cfg.tko_tilt) if _tko else float(cfg.climb_creep_tilt))
            if _ctilt > 0.0 and tilt_now >= _ctilt:
                d = np.zeros(2)
                if _YT_DIAG:
                    dd.yt |= 2
            if _YT_DIAG and float(np.linalg.norm(d)) > 1e-6 and risen >= cfg.climb_clear_m:
                dd.yt |= 1
            if float(np.linalg.norm(d)) > 1e-6 and risen >= cfg.climb_clear_m:
                _creep = cfg.cruise_speed * cfg.climb_creep
                if _tko:
                    _creep = max(_creep, math.sqrt(max(0.0, cfg.speed_limit ** 2 - _cs ** 2)) * 0.98)
                v_des[:2] = _unit(d) * _creep
                yaw = float(np.arctan2(d[1], d[0]))
            elif _cyh and dd.climb_yaw is not None:
                yaw = float(dd.climb_yaw)
            if dd.gl_pad >= 0:
                _gy = self._gla_yaw(i, dd, pos, rpy, _gtrav)     # batch 4 GLA look-at replaces the climb yaw
                if _gy is not None:
                    yaw = _gy
                    if _YT_DIAG:
                        dd.yt |= 4096
            top = (cfg.forest_alt if self.is_forest else cfg.cruise_alt) - 0.6
            _clx = False
            if (agl < top and cfg.clx_maps and self.is_forest and "forest" in tuple(cfg.clx_maps)
                    and dd.climb_exit_t >= 0.0
                    and float(pos[2]) >= float(cfg.forest_ceiling) - float(cfg.clx_room)):
                _clx = True                 # forest_nav CLX: re-CLIMB at the forest ceiling over an obstacle
                self.src_counts["fn_clx"] = self.src_counts.get("fn_clx", 0) + 1
            if agl >= top or _clx:
                dd.phase = SEARCH
                dd.climb_exit_t = float(self.step_i)
            elif (cfg.cwc_maps and ("forest" if self.is_forest else str(self.map_kind)) in tuple(cfg.cwc_maps)
                  and agl >= float(cfg.cwc_agl) and risen >= float(cfg.cwc_rise)
                  and (float(cfg.cwc_cam_deg) <= 0.0 or _gtrav is None
                       or abs((float(rpy[2]) - _gtrav + np.pi) % (2.0 * np.pi) - np.pi)
                       <= math.radians(float(cfg.cwc_cam_deg)))):
                dd.phase = SEARCH                      # search_time CWC: climb while cruising
                dd.climb_exit_t = float(self.step_i)
                self.src_counts["cwc_exit"] = self.src_counts.get("cwc_exit", 0) + 1
        elif dd.phase == SEARCH:
            _src = "route"
            tgt = None
            if self._hunt_active():
                try:
                    tgt = self._hunt_target(i, pos)
                except Exception:                                # noqa: BLE001
                    tgt = None
                    self._hunt_n["hunt_err"] += 1
                if tgt is not None:
                    _src = "hunt"
                    self._hunt_n["hunt_steps"] += 1
                    if not self._hunt_n["hunt_first_t"]:
                        self._hunt_n["hunt_first_t"] = self.step_i
            if tgt is None:
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
            elif self.map_kind == "village" and cfg.village_alt > 0.0:
                want_alt = cfg.village_alt
            elif self.map_kind == "city" and cfg.city_alt > 0.0:
                want_alt = cfg.city_alt
            elif clear:
                want_alt = cfg.cruise_alt_open
            elif cfg.open_alt > 0.0 and self._is_open_scene():
                want_alt = cfg.open_alt
            else:
                want_alt = (cfg.mountain_alt
                            if (self.map_kind == "mountain" and cfg.mountain_alt > 0.0)
                            else (cfg.cruise_alt if self._use_free() else cfg.cruise_alt_safe))
            _sab = self._search_alt_for_map()
            if _sab is not None:
                want_alt = _sab
            if float(getattr(cfg, "search_alt_gkc_open", 0.0)) > 0.0 and self._om_skip_open():
                want_alt = float(cfg.search_alt_gkc_open)       # candFd: GKC-pinned open seeds search higher
            _cao = self._clue_alt_off_for_map()
            dd.clue_low = False
            if _cao is not None and self.clue is not None and self.n <= int(cfg.clue_alt_max_n):
                _ground_z = float(pos[2]) - float(agl)     # an upper bound while the ray saturates at 20 m
                _hi = (want_alt if str(cfg.clue_alt_mode) == "below" else float(cfg.clue_alt_max_agl))
                _wa = float(np.clip(float(self.clue[2]) + _cao - _ground_z,
                                    cfg.clue_alt_min_agl, max(float(cfg.clue_alt_min_agl), _hi)))
                dd.clue_low = _wa < want_alt - 0.5
                want_alt = _wa
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
            _sink = 1.0
            if (cfg.search_sink_cap > 1.0
                    and ((not cfg.search_sink_maps) or (self.map_kind in cfg.search_sink_maps))):
                _sink = float(cfg.search_sink_cap)
            _dz = want_alt - agl
            if (cfg.terrain_ahead and self.map_kind in tuple(cfg.terrain_ahead_maps or ())
                    and agl < 19.9):
                if self.step_i % cfg.detect_every == 0 or not hasattr(dd, "_gz_ahead"):
                    try:
                        dd._gz_ahead = self._ground_ahead(depth, state, i)
                    except Exception:                            # noqa: BLE001
                        dd._gz_ahead = None
                gz = getattr(dd, "_gz_ahead", None)
                if gz is not None:
                    dz_ahead = (gz + want_alt) - float(pos[2])
                    # descend toward the valley floor ahead as soon as it is in view; climbing
                    # for rising terrain ahead is left to the ray-below feed-forward
                    if dz_ahead < _dz:
                        _dz = 0.5 * _dz + 0.5 * dz_ahead
            v_des[2] = np.clip(ff + _dz, -_sink, cap)
            if cfg.lav_maps and self._lav_on():
                try:
                    _lz = self._lav_vz(i, pos, agl, want_alt, v_des)     # mtn_speed LAV
                except Exception:                        # noqa: BLE001
                    _lz = None
                if _lz is not None:
                    v_des[2] = float(np.clip(_lz, -_sink, cap))
                    self._lav_live[i] = self.step_i
                for _lk, _lv in self._lav_n.items():
                    self.src_counts["lav_" + _lk] = _lv
            if (cfg.canopy_brake_maps and self.map_kind in cfg.canopy_brake_maps
                    and 0.3 < agl < want_alt - cfg.canopy_brake_drop):
                v_des[:2] *= float(cfg.canopy_brake_frac)
            yaw = float(np.arctan2(d[1], d[0]))
            if cfg.ay_maps and _src == "route" and self._ay_key() in tuple(cfg.ay_maps):
                _ya = self._ay_yaw(i, pos, tgt, yaw)     # yaw_time AY: look along the path ahead
                if _ya is not None:
                    yaw = _ya
                    self._yt_n["ay_steps"] += 1
            if FIX_NOGROUND and self._mass_route_kind() in FIX_BOX_KINDS and agl >= 19.9:
                v_des[2] = max(float(v_des[2]), 0.0)
                bound = self.WORLD.get(self._mass_route_kind())
                if bound is not None and float(np.max(np.abs(pos[:2]))) > bound:
                    home = -np.asarray(pos[:2], float)
                    v_des[:2] = _unit(home) * self._cruise(False)
                    yaw = float(np.arctan2(home[1], home[0]))
            _gy = None
            if dd.gl_pad >= 0:
                _gy = self._gla_yaw(i, dd, pos, rpy, yaw)     # batch 4 GLA look-at replaces the yaw scan
                if _gy is not None:
                    yaw = _gy
                    if _YT_DIAG:
                        dd.yt |= 4096
            _scan_ok = True
            if cfg.yaw_scan_align_deg > 0.0 and self.map_kind in tuple(cfg.yaw_scan_align_maps or ()):
                _off = abs((yaw - float(rpy[2]) + np.pi) % (2.0 * np.pi) - np.pi)
                _scan_ok = _off <= math.radians(float(cfg.yaw_scan_align_deg))
            if (_gy is None and cfg.yaw_scan_deg > 0.0 and _scan_ok
                    and self.map_kind in cfg.yaw_scan_maps
                    and self._clearance(depth, i) > cfg.yaw_scan_min_clear):
                steps = max(int(cfg.yaw_scan_steps), 1)
                _amp = float(cfg.yaw_scan_deg)
                if self.map_kind == "mountain":
                    if int(cfg.yaw_scan_steps_mtn) > 0:
                        steps = int(cfg.yaw_scan_steps_mtn)
                    if float(cfg.yaw_scan_deg_mtn) > 0.0:
                        _amp = float(cfg.yaw_scan_deg_mtn)
                ph = 2.0 * np.pi * float(self.step_i % steps) / steps
                yaw += float(np.radians(_amp) * np.sin(ph))
        elif (dd.phase == APPROACH and self._vf_on()
              and self._apx_vfy(i, dd, pad, pos, rpy,
                                cfg.forest_approach_alt if self.is_forest else cfg.approach_alt)):
            # APXV-R hold / orbit step: replaces the approach body (no DESCEND transition); the
            # command still passes through the avoidance, roof guard and separation below
            if dd.claim is None:
                return _encode(slew_velocity(np.array([0.0, 0.0, 0.5]), vel, cfg.max_delta_v),
                               cfg.speed_limit, yaw)
            v_des, yaw = np.asarray(dd.vf_cmd[0], float).copy(), float(dd.vf_cmd[1])
            if _YT_DIAG:
                dd.yt |= 65536
            if dd.cenv:
                v_des = self._cenv_apply(i, dd, v_des, agl, rpy)    # batch 4 CLIMB-claim envelope
            self._apx_vz_cmd = float(v_des[2])
        elif dd.phase == APPROACH:
            d = pad.xyz[:2] - pos[:2]
            dist = float(np.linalg.norm(d))
            dd.below = False                # P2r: a new descent starts without a below event
            dd.below_seen = False
            if cfg.village_veto_z and not self._vf_on() and dd.vf_key != dd.claim:
                # P1b claim-time hits / step (APXV-R's _apx_vfy keeps them when it is on)
                dd.vf_key, dd.vf_step0, dd.vf_hits0 = int(dd.claim), self.step_i, int(pad.hits)
            if self._probe_on():
                self._probe_sample(dd, pad, pos, agl)
            if cfg.pvr_maps and self._pvr_on():
                self._pvr_note(dd, pad, pos, agl)            # phantom_land PVR: ray samples near the entry
            if self._touch_on() and 3.0 < dist < 16.0 and self.step_i % max(1, int(cfg.detect_every)) == 0:
                try:
                    self._touch_sample(depth, state, i, dd, pad)
                except Exception:
                    pass
            if cfg.tdo_maps and self._tdo_on():
                # land_safe TDO: collect ring points while approaching; inside tdo_enter_r aim at the offset point
                self._tdo_sync(dd)
                if (float(cfg.tdo_samp_lo) < dist < float(cfg.tdo_samp_hi)
                        and self.step_i % max(1, int(cfg.detect_every)) == 0):
                    try:
                        self._tdo_sample(depth, state, i, dd, pad)
                    except Exception:                    # noqa: BLE001
                        pass
                if dist < float(cfg.tdo_enter_r):
                    self._tdo_update(dd, pad, freeze=False)
                    self._tdo_step(dd, pad, pos, agl)
                    if self._tdo_active(dd):
                        d = (np.asarray(pad.xyz, float)[:2] + dd.tdo_eff) - pos[:2]
                        dist = float(np.linalg.norm(d))
            _lp = self._lp                  # LANDSEG per-map landing parameters (cfg when the table is empty)
            _apx_alt = (_lp("forest_approach_alt") if self.is_forest
                        else _lp("approach_alt"))
            want_z = pad.xyz[2] + _apx_alt
            _ls_avz = self._lsv("approach_vz")
            if _ls_avz is not None:         # a per-map approach_vz applies whatever approach_vz_maps says
                _avz = float(_ls_avz)
            else:
                _avz = (cfg.approach_vz if self.map_kind in cfg.approach_vz_maps
                        else 1.0)
            _glide = _lp("apx_glide")
            if _glide > 0.0:
                want_z = pad.xyz[2] + float(np.clip(dist * _glide,
                                                    _lp("apx_min_above"), _apx_alt))
                _avz = max(_avz, _lp("apx_glide_vz"))
            _ekz = 1.0
            if (self._esk_on() and dist < float(cfg.esk_r) and int(pad.hits) >= int(cfg.esk_min_hits)
                    and float(dd.claim_dist) >= float(cfg.esk_min_claim_d)
                    and (float(cfg.esk_agl_tol) < 0.0
                         or agl >= float(pos[2] - pad.xyz[2]) - float(cfg.esk_agl_tol))):
                # ESK: sink toward pad_z + esk_above during the last esk_r metres of the approach
                # (esk_agl_tol >= 0: only over ground at about pad level, not over a roof / car / wall)
                want_z = min(want_z, float(pad.xyz[2]) + float(cfg.esk_above))
                _avz = max(_avz, float(cfg.esk_vz))
                _ekz = float(cfg.esk_kp)
            _dsk = self._dsk_cfg() if cfg.dsk_by_map else None
            _dsk_act = False
            _dsk_tz = None
            if (_dsk is not None and dist < float(_dsk.get("r", 8.0))
                    and int(pad.hits) >= int(_dsk.get("min_hits", 0))
                    and float(dd.claim_dist) >= float(_dsk.get("min_claim_d", 0.0))
                    and (float(_dsk.get("agl_tol", 0.6)) < 0.0
                         or agl >= float(pos[2] - pad.xyz[2]) - float(_dsk.get("agl_tol", 0.6)))):
                # candHb DSK: early sink toward pad_z + above in the last r metres of the approach (see cfg)
                _dsk_tz = float(pad.xyz[2]) + float(_dsk.get("above", 2.6))
                if float(_dsk.get("clear", 0.0)) > 0.0:
                    # terrain floor: never below the highest known surface on the way to the pad + clear
                    _dfl = self._dsk_floor(pos, np.asarray(pad.xyz, float)[:2], agl, _dsk)
                    _dsk_tz = max(_dsk_tz, _dfl) if _dfl > -1e8 else float(want_z)   # nothing known: no sink
                if dd.dsk_key != int(dd.claim) and float(pos[2]) > _dsk_tz + float(_dsk.get("latch_m", 0.3)):
                    dd.dsk_key = int(dd.claim)      # the approach comes down through the target: latch this claim
                if dd.dsk_key != int(dd.claim):
                    self._hb_inc("dsk_below_skip")  # a climbing approach (below the target) keeps the base law
                elif _dsk_tz < float(want_z):
                    want_z = _dsk_tz
                    _avz = max(_avz, float(_dsk.get("vz", 1.2)))
                    _ekz = float(_dsk.get("kp", 1.0))
                    if float(_dsk.get("line_d", 0.0)) > 0.0:
                        # glide line: the gain that reaches the target line_d m before the pad at the current ground
                        # speed (a straight 3-D line, the cheapest sink under the 3 m/s norm); kp is the floor
                        _vh = max(1.0, float(np.hypot(float(vel[0]), float(vel[1]))))
                        _ekz = max(_ekz, _vh / max(dist - float(_dsk["line_d"]), 0.3))
                    _dsk_act = True
                    self._hb_inc("dsk_steps")
                else:
                    self._hb_inc("dsk_floor_block")
            if (float(cfg.fld_above) > 0.0 and cfg.fld_maps and self._fld_on() and dist < float(cfg.edr_r)
                    and int(pad.hits) >= int(cfg.fld_min_hits)
                    and agl >= float(pos[2] - pad.xyz[2]) - float(cfg.edr_agl_tol)):
                # FLD: sink toward pad_z + fld_above inside the EDR envelope (shorter final drop)
                want_z = min(want_z, float(pad.xyz[2]) + float(cfg.fld_above))
                _avz = max(_avz, float(cfg.fld_avz))
            _alift_climb = False
            if cfg.alift_maps and ("forest" if self.is_forest else str(self.map_kind)) in tuple(cfg.alift_maps):
                # C-RC1 ALIFT: om2-bound approach-stall lift (see cfg); applied after ESK / DSK / FLD as a max
                _al_base = float(pad.xyz[2]) + float(_apx_alt)
                _al_gap = self.step_i - dd.alift_last > 25    # reviewer note 3: left APPROACH -> fresh state
                dd.alift_last = self.step_i
                if dd.alift_key != dd.claim or _al_gap:
                    dd.alift_key, dd.alift_best, dd.alift_t0 = int(dd.claim), dist, self.step_i
                    dd.alift_dz, dd.alift_ref, dd.alift_top = 0.0, 1e9, -1
                elif dist < dd.alift_best - float(cfg.alift_gain):
                    dd.alift_best, dd.alift_t0 = dist, self.step_i
                    if (float(cfg.alift_release_m) > 0.0 and dd.alift_dz > 0.0
                            and dist < dd.alift_ref - float(cfg.alift_release_m)
                            and self._alift_line_top(pos, pad, _al_base) is None):
                        dd.alift_dz, dd.alift_ref, dd.alift_top = 0.0, 1e9, -1
                        self._b6_n["alift_rel"] = self._b6_n.get("alift_rel", 0) + 1
                _al_climbing = dd.alift_dz > 0.0 and float(pos[2]) < _al_base + dd.alift_dz - 0.8
                if dd.alift_dz > 0.0 and not _al_climbing and dd.alift_top < dd.alift_t0:
                    dd.alift_t0 = dd.alift_top = self.step_i     # retry_s counts from reaching the rung
                _al_wait = float(cfg.alift_stall_s) if dd.alift_dz <= 0.0 else float(cfg.alift_retry_s)
                if (not _al_climbing and dist > float(cfg.alift_min_d)
                        and dd.alift_dz < float(cfg.alift_max_above)
                        and self.step_i - dd.alift_t0 > _al_wait * 50):
                    _al_f = self._om2_fire.get(i)
                    _al_bound = (_al_f is not None and int(_al_f[0]) >= self.step_i - 25
                                 and (int(_al_f[1]) & 1) != 0)
                    if not _al_bound:
                        self._b6_n["alift_nobind"] = self._b6_n.get("alift_nobind", 0) + 1
                    elif not self._alift_overhead(pos):
                        _al_new = dd.alift_dz + float(cfg.alift_rung)
                        if dd.alift_dz <= 0.0 and cfg.alift_line:
                            _al_lt = self._alift_line_top(pos, pad, _al_base)
                            if _al_lt is not None:
                                _al_new = max(_al_new, _al_lt + float(cfg.alift_margin) - _al_base)
                        dd.alift_dz = min(float(cfg.alift_max_above), _al_new)
                        dd.alift_ref = min(dd.alift_ref, dist)
                        dd.alift_t0 = self.step_i
                        self._b6_n["alift_rung"] = self._b6_n.get("alift_rung", 0) + 1
                        _al_ev = self._b6_n.setdefault("alift_ev", [])
                        if len(_al_ev) < 40:
                            _al_ev.append([round(self.step_i * 0.02, 2), int(i), int(dd.claim),
                                           round(float(dd.alift_dz), 2), round(dist, 2)])
                        _al_climbing = float(pos[2]) < _al_base + dd.alift_dz - 0.8
                if dd.alift_dz > 0.0 and _al_base + dd.alift_dz > want_z:
                    want_z = _al_base + dd.alift_dz
                    _avz = max(_avz, float(cfg.alift_vz))
                    _ekz = max(_ekz, 1.0)
                    _alift_climb = _al_climbing
            if cfg.apx_stall_s > 0.0 and ((not cfg.apx_stall_maps) or self.map_kind in cfg.apx_stall_maps):
                if dd.apx_key != dd.claim:
                    dd.apx_key, dd.apx_best, dd.apx_best_step = dd.claim, dist, self.step_i
                elif dist < dd.apx_best - float(cfg.apx_stall_gain):
                    dd.apx_best, dd.apx_best_step = dist, self.step_i
                elif self.step_i - dd.apx_best_step > int(float(cfg.apx_stall_s) * 50):
                    self._refute(i, soft=True)
                    dd.apx_key = -1
                    return _encode(slew_velocity(np.array([0.0, 0.0, 0.5]), vel, cfg.max_delta_v),
                                   cfg.speed_limit, yaw)
            top = self._cruise(cfg.open_boost and self._scene_is_open(depth, i))
            if (float(cfg.spd_gate_top) > 0.0 and self.is_forest and top > float(cfg.spd_gate_top)
                    and not self._spd_gate_ok(i)):
                top = float(cfg.spd_gate_top)   # candF hazard h: forest_speed held below the SPD hit gate
                self.src_counts["hz_top_cap"] = self.src_counts.get("hz_top_cap", 0) + 1
            _ag = self._lsv("approach_gain")    # a per-map approach_gain applies whatever approach_gain_skip says
            if _ag is None:
                _ag = (0.8 if self.map_kind in cfg.approach_gain_skip
                       else cfg.approach_gain)
            v_des[:2] = _unit(d) * min(top, max(0.5, dist * _ag))
            v_des[2] = np.clip(_ekz * (want_z - pos[2]), -_avz, _avz)
            self._apx_vz_cmd = float(v_des[2])
            if dist > 1e-6:
                yaw = float(np.arctan2(d[1], d[0]))
            _stp = False
            if (cfg.steep_maps and not self.is_forest and str(self.map_kind) in tuple(cfg.steep_maps)
                    and dist > float(cfg.steep_min_d)):
                _sab = float(cfg.steep_above)
                if _dsk_act and bool(_dsk.get("steep", False)):
                    _sab = float(_dsk_tz) - float(pad.xyz[2])     # candHb DSK: the STEEP line aims at the DSK point
                _sh = float(pad.xyz[2]) + _sab - float(pos[2])
                if abs(_sh) > float(cfg.steep_ratio) * dist and (_sh < 0.0 or bool(cfg.steep_climb)):
                    # mtn_claims STEEP: straight 3-D line to the approach point above the pad
                    _sL = math.hypot(dist, _sh)
                    v_des[:2] = _unit(d) * (float(top) * dist / _sL)
                    v_des[2] = float(np.clip(float(top) * _sh / _sL, -float(cfg.steep_vz), float(cfg.steep_vz)))
                    self._apx_vz_cmd = float(v_des[2])
                    _stp = True
                    self._mc_inc("steep_steps")
            if (cfg.apx_scan_maps and not _stp and not dd.cenv and self.map_kind in tuple(cfg.apx_scan_maps)
                    and dist > float(cfg.apx_scan_far) and int(pad.hits) >= int(cfg.apx_scan_min_hits)
                    and self._clearance(depth, i) > float(cfg.apx_scan_min_clear)):
                # candFe APX-SCAN: sweep the camera across the approach corridor (the claimed pad stays in the FOV)
                _ast = max(int(cfg.apx_scan_steps), 1)
                yaw += float(np.radians(float(cfg.apx_scan_deg))
                             * np.sin(2.0 * np.pi * float(self.step_i % _ast) / _ast))
                self._yt_n["apx_scan"] = self._yt_n.get("apx_scan", 0) + 1
            if agl < AGL_MIN and dist > AGL_FAR:
                v_des[2] = max(v_des[2], AGL_CLIMB)
            if (cfg.stale_refute_s > 0.0
                    and self.map_kind in cfg.stale_refute_maps
                    and pad.last_seen >= 0
                    and cfg.stale_near_r < dist < cfg.stale_refute_r
                    and (self.step_i - pad.last_seen) > int(cfg.stale_refute_s * 50)):
                self._refute(i, soft=bool(cfg.stale_refute_soft))
                return
            _eab = float(pos[2] - pad.xyz[2])
            _fld_hold = False
            if (self._edr_on() and dist < float(cfg.edr_r) and int(pad.hits) >= int(cfg.edr_min_hits)
                    and 0.0 < _eab <= float(cfg.edr_max_above)
                    and (float(cfg.edr_agl_tol) < 0.0 or agl >= _eab - float(cfg.edr_agl_tol))
                    and not self._endgame_hurry(dist, _eab)):
                # EDR: braking profile inside edr_r; start the drop once contact would follow the lateral arrival
                v_des[:2] = _unit(d) * min(top, self._edr_vlat(dist))
                _val = float(np.dot(np.asarray(vel[:2], float), _unit(d)))
                _tl, _tc = self._edr_times(dist, _val, _eab - float(cfg.edr_h0), float(vel[2]))
                _fvd = (float(cfg.fld_vdrop) > 0.0 and bool(cfg.fld_maps) and self._fld_on()
                        and float(np.hypot(float(vel[0]), float(vel[1]))) > float(cfg.fld_vdrop))
                if _tc >= _tl + self._edr_ak()[2] and not _fvd:
                    dd.edr_key = int(dd.claim)
                    dd.phase = DESCEND
                    if cfg.fld_maps and self._fld_on():
                        self._le_inc("fld_drop")
                elif cfg.fld_maps and bool(cfg.fld_hold) and self._fld_on():
                    # FLD: the EDR planner alone decides the DESCEND entry here, for at most fld_hold_s seconds
                    # inside the base entry radius (then the base rules: an avoidance-driven orbit never slows)
                    if dd.fld_hkey != int(dd.claim):
                        dd.fld_hkey, dd.fld_hn, dd.fld_hmin, dd.fld_hoff = int(dd.claim), 0, 99.0, False
                    _er0 = (_lp("descend_enter_r") if self.map_kind in cfg.descend_enter_wide
                            else _lp("descend_enter_r_tight"))
                    if dist < float(_er0):
                        dd.fld_hn += 1
                        dd.fld_hmin = min(dd.fld_hmin, dist)
                    if (not dd.fld_hoff and float(cfg.fld_hold_back) > 0.0
                            and dist > dd.fld_hmin + float(cfg.fld_hold_back)):
                        dd.fld_hoff = True       # pushed away from the pad (avoidance orbit): base rules
                        self._le_inc("fld_hold_back")
                    if dd.fld_hoff:
                        pass
                    elif float(cfg.fld_hold_s) <= 0.0 or dd.fld_hn <= int(float(cfg.fld_hold_s) * 50.0):
                        _fld_hold = True
                    elif dd.fld_hn == int(float(cfg.fld_hold_s) * 50.0) + 1:
                        self._le_inc("fld_hold_out")
            enter_r = (_lp("descend_enter_r")
                       if self.map_kind in cfg.descend_enter_wide
                       else _lp("descend_enter_r_tight"))
            if _fld_hold:
                enter_r = 0.0
            if self._endgame_hurry(dist, float(pos[2] - pad.xyz[2])):
                if cfg.hurry_real:
                    self._b3_n["s2_apx"] += 1
                    if not any(e[1] == i and e[2] == "S2" and e[3] == int(dd.claim) for e in self._b3_log):
                        self._b3_log.append((round(self.step_i * 0.02, 2), i, "S2", int(dd.claim),
                                             round(dist, 2), round(float(pos[2] - pad.xyz[2]), 2)))
                if not _stp:
                    v_des[:2] = _unit(d) * top
                enter_r = max(enter_r, 2.5)
                if dist < enter_r and abs(pos[2] - want_z) < 2.5:
                    dd.phase = DESCEND
            if dist < enter_r and abs(pos[2] - want_z) < 1.2:
                dd.phase = DESCEND
            if (cfg.apx_enter_any_alt and dist < min(enter_r, float(_lp("apx_enter_any_alt_r")))
                    and (pos[2] - pad.xyz[2]) < cfg.apx_max_above):
                dd.phase = DESCEND
            _vvz = tuple(cfg.village_veto_z or ())
            if (dd.phase == DESCEND and len(_vvz) == 2 and self.map_kind == "village"
                    and not self.is_forest and dd.vf_key == dd.claim):
                # P1b: a village estimate well above pad height, few hits, slow hit growth: phantom
                _vrate = (int(pad.hits) - int(dd.vf_hits0)) / max(0.5, (self.step_i - int(dd.vf_step0)) * 0.02)
                if (float(_vvz[0]) < float(pad.xyz[2]) < float(_vvz[1])
                        and int(pad.hits) < int(cfg.village_veto_hits)
                        and _vrate < float(cfg.village_veto_rate)):
                    self._vveto_log.append((round(self.step_i * 0.02, 2), i, int(dd.claim),
                                            round(float(pad.xyz[2]), 2), int(pad.hits), round(_vrate, 2)))
                    self._refute(i)
                    return _encode(slew_velocity(np.array([0.0, 0.0, 0.5]), vel, cfg.max_delta_v),
                                   cfg.speed_limit, yaw)
            if dd.cenv:
                # batch 4 CLIMB-claim envelope: climb and creep like CLIMB until CLIMB's exit height
                v_des = self._cenv_apply(i, dd, v_des, agl, rpy)
                self._apx_vz_cmd = float(v_des[2])
            if cfg.ehm_maps and dd.claim is not None and self._ehm_on():
                _ehv = self._ehm_apx(i, dd, pad, pos, vel, agl, d, dist, top)      # EHM (r9_prior P5)
                if _ehv is not None:
                    v_des = _ehv
                    self._apx_vz_cmd = float(v_des[2])
            if _alift_climb and dd.phase == APPROACH:
                # C-RC1 ALIFT: a pure climb to the rung (EDR / STEEP / hurry may have set v_des[:2] after the trigger)
                v_des = np.asarray(v_des, float).copy()
                v_des[:2] = 0.0
                self._apx_vz_cmd = float(v_des[2])

        elif dd.phase == DESCEND:
            if cfg.mtd_est and cfg.mtd_maps and dd.claim is not None and self._mtd_on():
                self._mtd_use_est(pad)      # MTD est: from the DESCEND switch the claim is the inverse-variance estimate
            d = pad.xyz[:2] - pos[:2]
            dist = float(np.linalg.norm(d))
            if self._probe_on():
                self._probe_sample(dd, pad, pos, agl)
                _pv = self._ray_probe(i, dd, pad, pos, agl, dist)
                if _pv is not None:
                    return _encode(slew_velocity(_pv, vel, cfg.max_delta_v), cfg.speed_limit, yaw)
                if dd.claim is None:
                    return _encode(slew_velocity(np.array([0.0, 0.0, 0.5]), vel, cfg.max_delta_v),
                                   cfg.speed_limit, yaw)
                pad = self.pads[dd.claim]
                d = pad.xyz[:2] - pos[:2]
                dist = float(np.linalg.norm(d))
            if cfg.dsc_min_hits > 0 and not dd.dsc_gated:
                dd.dsc_gated = True
                _gm = tuple(getattr(cfg, "dsc_gate_maps", ()) or ())
                _gmd = float(cfg.dsc_gate_min_dist)
                _bym = tuple(getattr(cfg, "dsc_gate_min_dist_by_map", ()) or ())
                for _k in range(0, len(_bym) - 1, 2):
                    if str(_bym[_k]) == self._mass_route_kind():
                        _gmd = float(_bym[_k + 1])
                        break
                _gate = (int(pad.hits) < int(cfg.dsc_min_hits)
                         and float(dd.claim_dist) >= _gmd
                         and self.step_i * 0.02 < float(cfg.dsc_gate_until_sec)
                         and ((not _gm) or (self._mass_route_kind() in _gm)))
                if (_gate and int(cfg.dsc_gate_inview_hits_min) > 0
                        and int(pad.hits) < int(cfg.dsc_gate_inview_hits_min)
                        and int(cfg.dsc_gate_inview_min) > 0
                        and int(dd.gate_inview) < int(cfg.dsc_gate_inview_min)):
                    # F4 diag: S1 would have exempted this claim; too few hits, the champion gate fires
                    self._b5_n["f4_gate"] += 1
                    self._b5_log.append((round(self.step_i * 0.02, 2), i, "F4", int(dd.claim), int(pad.hits),
                                         int(dd.gate_inview)))
                if _gate and not self._s1_gate_ok(dd):
                    # S1: never in view (or claimed steeply from above): few hits prove nothing
                    _gate = False
                    self._b3_n["s1_exempt"] += 1
                    self._b3_log.append((round(self.step_i * 0.02, 2), i, "S1", int(dd.claim), int(pad.hits),
                                         int(dd.gate_inview), round(float(dd.claim_elev), 1),
                                         round(float(dd.claim_dist), 1)))
                elif _gate and (int(cfg.dsc_gate_inview_min) > 0 or float(cfg.dsc_gate_steep_deg) > 0.0):
                    self._b3_n["s1_gated"] += 1
                if _gate:
                    # unconfirmed through a whole approach: a phantom, not a pad
                    _soft = ("forest" if self.is_forest else str(self.map_kind)) in tuple(
                        getattr(cfg, "dsc_gate_soft_maps", ()) or ())
                    self._refute(i, soft=_soft)
                    dd.dsc_gated = False
                    return _encode(slew_velocity(np.array([0.0, 0.0, 0.5]), vel, cfg.max_delta_v),
                                   cfg.speed_limit, yaw)
                _cgm = tuple(cfg.close_gate_maps or ())
                if (_cgm and ("forest" if self.is_forest else str(self.map_kind)) in _cgm
                        and float(dd.note_dist) < float(cfg.close_gate_dist)
                        and float(dd.claim_elev) >= float(cfg.close_gate_elev)
                        and int(pad.hits) < int(cfg.close_gate_min_hits)
                        and self.step_i * 0.02 < float(cfg.dsc_gate_until_sec)):
                    # batch 4 CLOSE: a close, steep claim reaching DESCEND with few hits (the champion gate
                    # exempts it by claim distance): phantom-like, refute softly and climb away
                    self._b4_n["close"] += 1
                    self._close_log.append((round(self.step_i * 0.02, 2), i, int(dd.claim), int(pad.hits),
                                            round(float(dd.note_dist), 2), round(float(dd.claim_elev), 1)))
                    self._refute(i, soft=True)
                    dd.dsc_gated = False
                    return _encode(slew_velocity(np.array([0.0, 0.0, 0.5]), vel, cfg.max_delta_v),
                                   cfg.speed_limit, yaw)
                _hg = tuple(getattr(cfg, "dsc_hits_gate_maps", ()) or ())
                if (_hg and ("forest" if self.is_forest else str(self.map_kind)) in _hg
                        and int(pad.hits) < int(cfg.dsc_hits_gate_min)):
                    # real pads reach descend entry with ~60-100 hits (p10 >= 26 on every map);
                    # phantoms with 1-10. Few hits here: drop the claim and keep searching.
                    self._refute(i, soft=int(pad.hits) >= int(cfg.dsc_hits_gate_soft_from))
                    dd.dsc_gated = False
                    return _encode(slew_velocity(np.array([0.0, 0.0, 0.5]), vel, cfg.max_delta_v),
                                   cfg.speed_limit, yaw)
            if cfg.ne_maps and dd.claim is not None and not pad.ne_safe:
                pad.ne_safe = True          # batch 4 NE: this entry passed the descend-entry gates; never refute it

            _lp = self._lp                  # LANDSEG per-map landing parameters (cfg when the table is empty)
            _lsr = _lp("land_settle_r")
            if self._touch_on():
                if dd.touch_key != dd.claim:
                    dd.touch_key = dd.claim if dd.claim is not None else -1
                    dd.touch_buf = []
                    dd.touch_off = None
                if dd.touch_off is None:
                    try:
                        dd.touch_off = self._touch_offset(dd, pad)
                    except Exception:
                        dd.touch_off = np.zeros(2)
                if float(np.linalg.norm(dd.touch_off)) > 1e-6:
                    d = (np.asarray(pad.xyz, float)[:2] + dd.touch_off) - pos[:2]
                    dist = float(np.linalg.norm(d))
                    _lsr = float(cfg.touch_settle_r)
            if cfg.tdo_maps and self._tdo_on():
                # land_safe TDO: freeze the offset at DESCEND entry; the whole descent aims at the offset point
                self._tdo_sync(dd)
                self._tdo_update(dd, pad, freeze=True)
                self._tdo_step(dd, pad, pos, agl)
                if cfg.mtd_maps and self._mtd_on():
                    self._mtd_sync(dd)
                    if not dd.mtd_logged:
                        dd.mtd_logged = True
                        _ml = self.src_counts.setdefault("mtd_log", [])
                        if isinstance(_ml, list) and len(_ml) < 64:
                            _mi = getattr(dd, "_mtd_hi", None)
                            _ml.append([round(self.step_i * 0.02, 2), int(i), int(dd.claim), dd.mtd_gate,
                                        round(float(dd.tdo_lam), 3), str(getattr(dd, "tdo_why", "")),
                                        None if _mi is None else list(_mi),
                                        None if pad.mtd_xyz is None else [round(float(v), 3) for v in pad.mtd_xyz],
                                        [round(float(v), 3) for v in pad.xyz]])
                            if str(getattr(dd, "tdo_why", "")) in ("hmflat", "hmsteep", "hmfew", "hmcells"):
                                self.src_counts["mtd_" + str(dd.tdo_why)] = self.src_counts.get("mtd_" + str(dd.tdo_why), 0) + 1
                            elif bool(cfg.mtd_hm) and str(getattr(dd, "tdo_why", "")) == "ok":
                                self.src_counts["mtd_hm"] = self.src_counts.get("mtd_hm", 0) + 1
                if self._tdo_active(dd):
                    d = (np.asarray(pad.xyz, float)[:2] + dd.tdo_eff) - pos[:2]
                    dist = float(np.linalg.norm(d))
            _dgain, _dlat = _lp("dsc_gain"), _lp("dsc_lateral")
            _flare = _lp("forest_flare_alt") if self.is_forest else _lp("flare_alt")
            _sink = _lp("forest_sink_speed") if self.is_forest else _lp("sink_speed")
            _dspd, _lrad = _lp("descend_speed"), _lp("land_radius")
            _dsk_law = False
            _dsk_d = self._dsk_cfg() if cfg.dsk_by_map else None
            if (float(cfg.esk_dsc_gain) > 0.0 and self._esk_on()
                    and float(pos[2] - pad.xyz[2]) > float(cfg.esk_dsc_above)
                    and float(dd.claim_dist) >= float(getattr(cfg, "esk_dsc_min_claim_d", 0.0))):
                # ESK: faster lateral centring in DESCEND (the shorter sink leaves less time to settle); only
                # above esk_dsc_above so the touchdown and the parked drone keep the base law; esk_dsc_fade > 0
                # blends the gain / clip linearly back to the base law over that many metres above esk_dsc_above
                _ef = 1.0
                if float(cfg.esk_dsc_fade) > 0.0:
                    _ef = float(np.clip((float(pos[2] - pad.xyz[2]) - float(cfg.esk_dsc_above))
                                        / float(cfg.esk_dsc_fade), 0.0, 1.0))
                _eg = float(_dgain) + _ef * (float(cfg.esk_dsc_gain) - float(_dgain))
                _el = float(_dlat) + _ef * (float(cfg.esk_dsc_lat) - float(_dlat))
                v_des[:2] = np.clip(d * _eg, -_el, _el)
            elif (_dsk_d is not None and float(_dsk_d.get("dsc_gain", 0.0)) > 0.0
                  and dd.claim is not None and dd.dsk_key == int(dd.claim)
                  and float(pos[2] - pad.xyz[2]) > float(_dsk_d.get("dsc_above", 0.8))
                  and float(dd.claim_dist) >= float(_dsk_d.get("dsc_min_claim_d", 4.0))):
                # candHb DSK: the ESK DESCEND lateral law on this kind (the lower entry leaves less time to centre),
                # only for claims the early sink latched; every other descent keeps the base law
                _dg = float(_dsk_d["dsc_gain"])
                _dl = float(_dsk_d.get("dsc_lat", 0.9))
                v_des[:2] = np.clip(d * _dg, -_dl, _dl)
                _dsk_law = True
            else:
                v_des[:2] = np.clip(d * _dgain, -_dlat, _dlat)
            _edr_d = (dd.edr_key >= 0 and dd.claim is not None and dd.edr_key == int(dd.claim) and self._edr_on()
                      and float(pos[2] - pad.xyz[2]) > float(cfg.edr_low))
            if _dsk_law and not _edr_d:
                self._hb_inc("dsk_dsc_steps")          # candHb diag: the DSK lateral law in effect (EDR not overriding)
            if _edr_d:
                # EDR: keep the braking profile while higher than edr_low above the pad
                v_des[:2] = _unit(d) * self._edr_vlat(dist)
            above = float(pos[2] - pad.xyz[2])
            if (cfg.agl_band_tol > 0.0 and dist <= cfg.agl_band_r
                    and agl < 19.9 and abs(agl - above) <= cfg.agl_band_tol):
                above = agl
            _fl = _flare
            _sk = _sink
            if (cfg.fld_maps and self._fld_on() and dd.claim is not None
                    and int(pad.hits) >= int(cfg.fld_min_hits)):
                # FLD: fast sink on an EDR drop or once centred; the low flare only once centred
                _fdr = dd.edr_key >= 0 and dd.edr_key == int(dd.claim)
                _fok = _fdr or not bool(cfg.fld_edr_only)
                if dist <= float(cfg.fld_r) and _fok:
                    _fl = min(float(_fl), float(cfg.fld_flare))
                if _fdr or (dist <= float(cfg.fld_r) and _fok):
                    _sk = float(cfg.fld_sink)
                if above > float(cfg.fld_flare) and dist <= float(cfg.fld_r) and _fok:
                    self._le_inc("fld_low")
            if above > _fl:
                v_des[2] = -_sk
                if (cfg.dsc_center and dist > cfg.dsc_center_r
                        and above <= cfg.dsc_center_above):
                    v_des[2] = 0.0          # hold height, centre first
                if self._endgame_hurry(dist, above):
                    v_des[:2] = np.clip(d * 1.2, -0.9, 0.9)
                    v_des[2] = -max(_lp("sink_speed"), 1.6)
                elif _edr_d and bool(cfg.edr_prio) and float(v_des[2]) < 0.0:
                    # EDR: the lateral profile keeps priority under the 3 m/s norm
                    _vl2 = float(v_des[0] ** 2 + v_des[1] ** 2)
                    v_des[2] = -min(-float(v_des[2]), math.sqrt(max(0.09, 9.0 - _vl2)))
            else:
                v_des[2] = (-_dspd if dist < _lrad
                            else -_lp("creep_speed"))
                if self._endgame_hurry(dist, above):
                    v_des[2] = -max(_dspd, 0.7)
                if dist > (cfg.forest_abort_radius if self.is_forest else cfg.abort_radius):
                    v_des[2] = 0.15
            if (float(cfg.ehm_dsc_vs) > 0.0 and cfg.ehm_maps and dd.claim is not None
                    and getattr(dd, "ehm_key", -1) == int(dd.claim) and above > 1.0 and dist < 0.8
                    and self._ehm_on()):
                v_des[2] = min(float(v_des[2]), -float(cfg.ehm_dsc_vs))       # EHM (r9_prior P5)
            _scc = _lp("sink_center_cap")
            if _scc > 0.0 and dist > _lrad:
                t_center = dist / 0.55
                lim = max(abs(above) / max(t_center, 1e-6) * _scc,
                          _dspd)
                v_des[2] = max(v_des[2], -lim)
            if cfg.descend_commit:
                if (not dd.commit_sink and dist < cfg.descend_commit_r
                        and above > cfg.descend_commit_above):
                    dd.commit_sink = True
                if dd.commit_sink:
                    v_des[2] = (-_sk if above > _fl else -_dspd)
            if (float(getattr(cfg, "dsc_tilt_hold_deg", 0.0)) > 0.0 and self._esk_on() and dd.claim is not None
                    and float(pos[2] - pad.xyz[2]) > float(cfg.dsc_tilt_hold_above)
                    and math.degrees(max(abs(float(rpy[0])), abs(float(rpy[1])))) > float(cfg.dsc_tilt_hold_deg)
                    and not self._endgame_hurry(dist, float(pos[2] - pad.xyz[2]))):
                # audit_cov tilt hold: no sink while swinging this high above the pad; base lateral law
                v_des[:2] = np.clip(d * _dgain, -_dlat, _dlat)
                v_des[2] = max(float(v_des[2]), 0.0)
                self.src_counts["ac_tilt_hold"] = self.src_counts.get("ac_tilt_hold", 0) + 1
            if self._lvl_on() and not dd.below and not dd.hop and dd.claim is not None:
                _lab = float(pos[2] - pad.xyz[2])
                _ltd = math.degrees(max(abs(float(rpy[0])), abs(float(rpy[1]))))
                if 0.12 < _lab < float(cfg.lvl_above) and _ltd > float(cfg.lvl_deg):
                    # LVL: tilted this close to the pad: no lateral acceleration and a slower sink until level
                    # (a contact at > ~12 deg stays tilted on the pad and never latches)
                    v_des[:2] = np.asarray(vel[:2], float)
                    v_des[2] = max(float(v_des[2]), min(-float(cfg.lvl_vz), float(vel[2]) + float(cfg.lvl_dvz)))
                    self._lvl_n += 1
            _lsd_rest = False
            if (cfg.lsd_maps and ("forest" if self.is_forest else str(self.map_kind)) in tuple(cfg.lsd_maps)
                    and not dd.hop and dist <= float(cfg.lsd_r)
                    and -float(cfg.lsd_ab) < float(pos[2] - pad.xyz[2]) < 0.3):
                if dd.lsd_key != int(dd.claim):
                    dd.lsd_key, dd.lsd_seen = int(dd.claim), -10 ** 9
                if agl < 0.30:
                    dd.lsd_seen = int(self.step_i)      # the ray touched a surface at the pad top on this claim
                if (agl < 0.30 or (abs(float(vel[2])) < float(cfg.lsd_v)
                                   and float(np.hypot(float(vel[0]), float(vel[1]))) < float(cfg.lsd_v)
                                   and int(self.step_i) - int(dd.lsd_seen) <= int(cfg.lsd_mem))):
                    _lsd_rest = True        # phantom_land LSD: resting on the pad (the ray may read through it)
            if (cfg.dsc_below_maps or cfg.dsc_hold_maps) and not _lsd_rest:
                # P2r: slid off a floating slab -> step out / climb back above the pad top, then
                # (dsc_hold_maps) sink only once centred again; overrides the sticky commit_sink
                _dk = "forest" if self.is_forest else str(self.map_kind)
                _ab = float(pos[2] - pad.xyz[2])
                if cfg.dsc_below_maps and _dk in tuple(cfg.dsc_below_maps):
                    _gap = float(cfg.dsc_below_agl_gap)
                    _enter = (_ab < -float(cfg.dsc_below_z) and agl > 0.25
                              and (_gap <= 0.0 or (agl - _ab) > _gap))
                    if _enter or (dd.below and _ab < 0.3):
                        if not dd.below:
                            self._below_log.append((round(self.step_i * 0.02, 2), i, int(dd.claim),
                                                    round(_ab, 2), round(dist, 2), round(agl, 2)))
                            if int(cfg.dsc_below_retarget) > 0:
                                self._bl_event(i, dd, pad, pos, rpy)     # F3 (may move the entry)
                        dd.below = True
                        dd.below_seen = True
                        _out = pos[:2] - np.asarray(pad.xyz, float)[:2]
                        _no = float(np.linalg.norm(_out))
                        if _no < 0.95:
                            _o = (_out / _no if _no > 1e-3
                                  else -np.array([np.cos(float(rpy[2])), np.sin(float(rpy[2]))]))
                            v_des[:2] = _o * 0.6
                            v_des[2] = 0.0
                        else:
                            v_des[:2] = 0.0
                            v_des[2] = 0.8
                    else:
                        dd.below = False
                if (cfg.dsc_hold_maps and _dk in tuple(cfg.dsc_hold_maps) and not dd.below
                        and _ab <= float(cfg.dsc_hold_above) and dist > float(cfg.dsc_hold_r)
                        and (not bool(cfg.dsc_hold_after_below) or dd.below_seen)):
                    v_des[2] = max(float(v_des[2]), 0.0)
                if dd.bl_on:
                    # F3: retargeted entry: climb to dsc_hold_above over the pad top, then move over it
                    if dd.bl_key != dd.claim:
                        dd.bl_on = False
                    elif not dd.below:
                        _bxy = np.asarray(pad.xyz, float)[:2] - pos[:2]
                        if float(np.linalg.norm(_bxy)) > float(cfg.dsc_hold_r):
                            if float(pos[2] - pad.xyz[2]) < float(cfg.dsc_hold_above):
                                v_des[:2] = 0.0
                                v_des[2] = 0.6
                            else:
                                v_des[2] = max(float(v_des[2]), 0.0)
                        else:
                            dd.bl_on = False
            _ct = float(cfg.commit_t)
            _bym = tuple(getattr(cfg, "commit_t_by_map", ()) or ())
            for _k in range(0, len(_bym) - 1, 2):
                if str(_bym[_k]) == self.map_kind:
                    _ct = float(_bym[_k + 1])
                    break
            late = (_ct > 0.0
                    and self.map_kind not in cfg.commit_t_skip
                    and self.step_i >= int(_ct * 50))
            # Phantom detections are what the drone descends into: real pads collect
            # ~100 hits, phantoms 1-3, so the touchdown gate is the cheapest place to
            # tell them apart. Past the commit deadline the gate relaxes to
            # ``land_hits_late`` instead of vanishing.
            _need = self._land_hits_for_map()
            _views = int(cfg.land_min_views)
            if late:
                _need = min(_need, int(cfg.land_hits_late))
                _views = min(_views, 1)
            _blocked = ((pad.hits < _need or len(pad.views) < _views)
                        and pos[2] - pad.xyz[2] < 1.5
                        and not (cfg.commit_hold_release and dd.commit_sink))
            if _blocked:
                v_des[2] = max(v_des[2], 0.0)
                dd.gate_hold += 1
                # Refusing to touch down is only half a save: hovering over a pad the
                # gate will never pass burns the rest of the episode for the same 0.01
                # a crash would have scored. Give up on it and go back to searching.
                if (cfg.land_gate_refute_steps > 0
                        and dd.gate_hold >= int(cfg.land_gate_refute_steps)):
                    self._refute(i)
                    dd.gate_hold = 0
                    v_des = np.array([0.0, 0.0, cfg.climb_speed], dtype=float)
            else:
                dd.gate_hold = 0
            _edge = False
            if cfg.dsc_edge_maps and ("forest" if self.is_forest else str(self.map_kind)) in tuple(cfg.dsc_edge_maps):
                # near touchdown the downward ray reading more than the height above the pad means
                # the drone is over the platform edge (it would sink beside it or tip off it)
                _ae = float(pos[2] - pad.xyz[2])
                _miss = (0.05 < agl < 19.9) and agl > _ae + float(cfg.dsc_edge_gap)
                _tl = float(max(abs(float(rpy[0])), abs(float(rpy[1]))))
                _edge = _ae < float(cfg.dsc_edge_above) and (
                    _miss or (_tl > float(cfg.dsc_edge_tilt) and abs(float(vel[2])) < 0.2))
                if _ae < 0.6 and _miss and dist > _lrad:
                    v_des[2] = max(float(v_des[2]), 0.0)
            if cfg.land_settle:
                _p4b = False
                if (cfg.settle_above_maps and agl >= 0.30
                        and ("forest" if self.is_forest else str(self.map_kind)) in tuple(cfg.settle_above_maps)):
                    # LANDSEG P4b: contact by height over the claimed pad estimate (raised pads read agl ~0.6)
                    _p4b = (float(pos[2] - pad.xyz[2]) <= float(_lp("settle_above_dz"))
                            and float(vel[2]) > float(_lp("settle_above_vz")))
                    _ck = int(dd.claim) if dd.claim is not None else -1
                    if _p4b and not dd.hop and dd.sab_key != _ck:
                        dd.sab_key = _ck
                        self._sab_log.append((round(self.step_i * 0.02, 2), i, _ck,
                                              round(float(pos[2] - pad.xyz[2]), 3), round(float(agl), 2),
                                              round(float(dist), 2), round(float(vel[2]), 2)))
                _tr_seen = False
                if dd.hop:
                    # climbing back to land_hop_alt above the pad, centring on the way
                    v_des[:2] = np.clip(d * _dgain, -_dlat, _dlat)
                    v_des[2] = 0.6
                    if above >= cfg.land_hop_alt or (agl >= cfg.land_hop_alt and not dd.hop_tilt):
                        dd.hop = False
                        dd.hop_tilt = False
                elif agl < 0.30 or _edge or _p4b:
                    if dist <= _lsr and not _edge:
                        v_des[0] = 0.0
                        v_des[1] = 0.0
                        if (cfg.settle_tilt_maps and ("forest" if self.is_forest else str(self.map_kind))
                                in tuple(cfg.settle_tilt_maps)):
                            # LANDSEG P4c: resting past the latch tilt limit -> hop and land again
                            _tlr = float(max(abs(float(rpy[0])), abs(float(rpy[1]))))
                            if _tlr >= float(cfg.settle_tilt_rad) and abs(float(vel[2])) < 0.2:
                                dd.tilt_rest += 1
                                _tr_seen = True
                            if (dd.tilt_rest >= max(1, int(round(float(cfg.settle_tilt_s) * 50)))
                                    and dd.hops < int(cfg.land_hop_max)):
                                self._tilt_hop_log.append((round(self.step_i * 0.02, 2), i,
                                                           int(dd.claim) if dd.claim is not None else -1,
                                                           round(_tlr, 3), round(float(dist), 2),
                                                           round(float(agl), 2), int(dd.hops)))
                                dd.tilt_rest = 0
                                dd.hop = True
                                dd.hop_tilt = True      # ends by height over the pad (raised pads read agl >= 0.6)
                                dd.hops += 1
                                v_des[:2] = 0.0
                                v_des[2] = 0.6
                    elif (dist <= (cfg.forest_abort_radius if self.is_forest else cfg.abort_radius) or _edge) \
                            and dd.hops < int(cfg.land_hop_max):
                        dd.hop = True
                        dd.hops += 1
                        v_des[:2] = 0.0
                        v_des[2] = 0.6
                    elif _edge:
                        v_des[:2] = 0.0
                        v_des[2] = max(float(v_des[2]), 0.0)
                if cfg.settle_tilt_maps and not _tr_seen:
                    dd.tilt_rest = 0            # P4c counts consecutive tilted resting steps only
            if _lsd_rest and not dd.hop:
                # phantom_land LSD: resting on the pad -> no below state, press down so the latch completes
                if dd.below or float(v_des[2]) > -float(cfg.lsd_vz):
                    self.src_counts["lsd_press"] = self.src_counts.get("lsd_press", 0) + 1
                dd.below = False
                v_des[0] = 0.0
                v_des[1] = 0.0
                v_des[2] = min(float(v_des[2]), -float(cfg.lsd_vz))
            if self._park_on() and not dd.hop and not dd.below:
                _pab = float(pos[2] - pad.xyz[2])
                if (-0.1 < _pab < float(cfg.park_above) and abs(float(vel[2])) < float(cfg.park_vz)
                        and dist <= float(cfg.park_r)):
                    # PARK: resting on the pad (height of the pad estimate, no vertical motion): no lateral
                    # command, so the attitude loop can level the drone instead of pushing it over the pad
                    v_des[0] = 0.0
                    v_des[1] = 0.0
                    self._park_n += 1
                    if bool(cfg.park_nosep):
                        dd.nosep_step = int(self.step_i)
            if self._cst_on() and not dd.hop and not dd.below and dd.claim is not None:
                _cab = float(pos[2] - pad.xyz[2])
                if (0.0 < _cab < float(cfg.cst_above) and float(vel[2]) < -float(cfg.cst_vz)
                        and dist <= float(cfg.cst_r)):
                    # CST: the last ~0.15 s before contact: no lateral acceleration (hold the current lateral
                    # velocity); a brake here is frozen into the contact tilt
                    v_des[:2] = np.asarray(vel[:2], float)
                    self._cst_n += 1
                    if bool(cfg.park_nosep):
                        dd.nosep_step = int(self.step_i)
            if (cfg.tdf_maps or cfg.tdf_rest_maps) and dd.claim is not None and not dd.hop and not dd.below:
                self._tdf_step(i, dd, pad, pos, vel, agl, dist, v_des)     # candFe TDF soft touchdown
            if (cfg.mtd_rim and cfg.mtd_maps and dd.claim is not None and not dd.hop and self._mtd_on()
                    and cfg.tdo_maps and self._tdo_on() and (self._tdo_active(dd) or dd.mtd_rg_hold)):
                v_des = self._mtd_rim_step(i, dd, pad, pos, vel, agl, v_des)   # MTD rim guard
            if cfg.trr_maps and dd.claim is not None and (
                    ("forest" if self.is_forest else str(self._mass_route_kind())) in tuple(cfg.trr_maps)):
                # audit_mf TRR: resting on the pad tilted beyond the latch limit -> hop and land again
                if dd.trr_key != int(dd.claim):
                    dd.trr_key, dd.trr_n, dd.trr_hops, dd.trr_left = int(dd.claim), 0, 0, 0
                if dd.trr_left > 0:
                    dd.trr_left -= 1
                    v_des[:2] = 0.0
                    v_des[2] = float(cfg.trr_vz)
                else:
                    _tt = float(max(abs(float(rpy[0])), abs(float(rpy[1]))))
                    _tab = float(pos[2] - pad.xyz[2])
                    if (dist <= float(cfg.trr_r) and _tab < float(cfg.trr_above)
                            and abs(float(vel[2])) < float(cfg.trr_v)
                            and float(np.hypot(float(vel[0]), float(vel[1]))) < float(cfg.trr_v)
                            and _tt > float(cfg.trr_tilt)):
                        dd.trr_n += 1
                    else:
                        dd.trr_n = 0
                    if dd.trr_n >= int(cfg.trr_steps) and dd.trr_hops < int(cfg.trr_max):
                        dd.trr_hops += 1
                        dd.trr_n = 0
                        dd.trr_left = int(cfg.trr_len)
                        dd.commit_sink = False
                        v_des[:2] = 0.0
                        v_des[2] = float(cfg.trr_vz)
                        self.src_counts["trr_hop"] = self.src_counts.get("trr_hop", 0) + 1
                        _tl = self.src_counts.setdefault("trr_log", [])
                        if isinstance(_tl, list) and len(_tl) < 32:
                            _tl.append([round(self.step_i * 0.02, 2), i, int(dd.claim), round(_tt, 3),
                                        round(_tab, 2), round(dist, 2)])
            if (cfg.pvr_maps and dd.claim is not None and self._pvr_on()
                    and not (cfg.pvr_ehm_skip and getattr(dd, "ehm_key", -1) == int(dd.claim))):
                _pv = self._pvr_descend(i, dd, pad, pos, agl, dist, v_des)   # phantom_land PVR
                if _pv is None:
                    return _encode(slew_velocity(np.array([0.0, 0.0, 0.8]), vel, cfg.max_delta_v),
                                   cfg.speed_limit, yaw)
                v_des = _pv
            if agl < 0.30:
                dd.stuck += 1
                on_pad = (pad is not None
                          and float(np.linalg.norm(
                              pos[:2] - np.asarray(pad.xyz, float)[:2])) <= cfg.landed_r
                          and pad.hits >= cfg.min_hits_to_land)
                if (cfg.tdo_maps and not on_pad and pad is not None and pad.hits >= cfg.min_hits_to_land
                        and self._tdo_active(dd)):
                    on_pad = self._tdo_pad_dist(dd, pad, pos) <= cfg.landed_r     # TDO: at the offset point
                if cfg.landed_done and on_pad:
                    if cfg.tdo_maps and not dd.landed and self._tdo_dir(dd):
                        _lg = self.src_counts.setdefault("tdo_land", [])
                        if isinstance(_lg, list) and len(_lg) < 64:
                            _S = np.asarray(dd.tdo_rs, float).reshape(-1, 3)
                            _zt = self._tdo_ztop(_S, pad) if len(_S) else None
                            _lg.append([round(self.step_i * 0.02, 2), i, int(dd.claim), round(float(dd.tdo_lam), 3),
                                        round(float(pad.xyz[0]), 3), round(float(pad.xyz[1]), 3),
                                        round(float(pos[0]), 3), round(float(pos[1]), 3), len(_S),
                                        None if _zt is None else round(float(_zt - pad.xyz[2]), 3)])
                    pad.done = True
                    dd.landed = True
                    dd.stuck = 0
                elif dd.stuck > cfg.ground_stall_steps:
                    self._refute(i, soft=bool(getattr(cfg, "stall_soft", 0)))   # DEV stall_soft
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
        if (cfg.vra_maps and str(dd.phase) in tuple(cfg.vra_phases or ())
                and ("forest" if self.is_forest else str(self.map_kind)) in tuple(cfg.vra_maps)):
            try:
                v_des = self._vra(depth, i, dd, pad, pos, rpy, v_des)     # collide VRA
            except Exception:                                    # noqa: BLE001
                pass
        if (cfg.faf_maps and str(dd.phase) in tuple(cfg.faf_phases or ())
                and ("forest" if self.is_forest else str(self.map_kind)) in tuple(cfg.faf_maps)):
            _fsp = float(np.hypot(float(v_des[0]), float(v_des[1])))
            if _fsp > float(cfg.faf_speed):
                _foff = abs((math.atan2(float(v_des[1]), float(v_des[0])) - float(rpy[2]) + math.pi)
                            % (2.0 * math.pi) - math.pi)
                if _foff > math.radians(float(cfg.faf_deg)):
                    v_des = np.asarray(v_des, float).copy()
                    v_des[:2] *= float(cfg.faf_speed) / _fsp       # collide FAF
                    self._faf_n = getattr(self, "_faf_n", 0) + 1
        _v_pre = np.asarray(v_des, float)[:2].copy()
        _wst_yaw = None                     # WSTOP: camera heading held while backing off a wall
        if (dd.phase != DESCEND or far_pad) and not in_tilt_hold:
            self._yaw_now = yaw
            self._att_yaw_now = float(rpy[2])
            self._pitch_now = -float(rpy[1])   # env rpy pitch is + nose-down
            # bearing of the intended travel relative to where the camera is pointing
            if float(np.hypot(_v_pre[0], _v_pre[1])) > 1e-6:
                self._clear_rel = float(np.arctan2(_v_pre[1], _v_pre[0]) - float(rpy[2]))
                self._clear_rel = float(np.arctan2(np.sin(self._clear_rel),
                                                   np.cos(self._clear_rel)))
            else:
                self._clear_rel = 0.0
            _v_pre3 = float(np.linalg.norm(np.asarray(v_des, float)))
            if _YT_DIAG:
                _yt_h0 = float(np.hypot(float(v_des[0]), float(v_des[1])))
                if _v_pre3 > 1e-6:
                    _yt_vb = rot_from_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2])).T @ np.asarray(v_des, float)
                    if float(_yt_vb[0]) < _v_pre3 * FREE_DIR_MIN_COS:
                        dd.yt |= 1024
            if self._use_free():
                v_des = self._free_dir(depth, i, v_des, rpy, pos)
                if _YT_DIAG and float(np.hypot(float(v_des[0]), float(v_des[1]))) < 0.9 * _yt_h0:
                    dd.yt |= 512
            else:
                if cfg.steer or self.is_forest or self.map_kind in cfg.steer_maps:
                    v_des = self._steer_around(depth, i, v_des)
                v_des = self._avoid(depth, i, v_des, pos=pos,
                                    pad_xy=(pad.xyz[:2] if pad is not None else None))
                if _YT_DIAG and float(np.hypot(float(v_des[0]), float(v_des[1]))) < 0.9 * _yt_h0:
                    dd.yt |= 16384
                if (cfg.wall_stop_maps
                        and ("forest" if self.is_forest else str(self.map_kind)) in tuple(cfg.wall_stop_maps)
                        and (dd.phase == CLIMB or (dd.cenv and dd.phase == APPROACH)
                             or (dd.phase == SEARCH and dd.climb_exit_t >= 0.0
                                 and (float(self.step_i) - dd.climb_exit_t) * 0.02 < float(cfg.wall_stop_sec)))
                        and self._clearance(depth, i) < float(cfg.wall_stop_r)):
                    # WSTOP: a wall inside wall_stop_r during take-off: back away from it and climb
                    _wst_yaw = float(rpy[2])
                    v_des = np.asarray(v_des, float).copy()
                    v_des[:2] = -np.array([math.cos(_wst_yaw), math.sin(_wst_yaw)]) * float(cfg.wall_stop_back)
                    v_des[2] = max(float(v_des[2]), 1.0)
                    self._wstop_n += 1
                    self._wstop_drones.add(i)
                    if _YT_DIAG:
                        dd.yt |= 256
            if (cfg.nf_maps and str(self.map_kind) in tuple(cfg.nf_maps)
                    and dd.phase in tuple(cfg.nf_phases or ())):
                _vo = np.asarray(v_des, float)
                _no = float(np.linalg.norm(_vo))
                _want = min(_v_pre3, float(cfg.speed_limit))
                if 0.3 < _no < _want:
                    v_des = _vo * min(float(cfg.nf_max_scale), _want / _no)
            if (cfg.ttc_maps and dd.phase in tuple(cfg.ttc_phases or ())
                    and ("forest" if self.is_forest else str(self.map_kind)) in tuple(cfg.ttc_maps)):
                v_des = self._ttc_guard(depth, i, v_des, rpy, state[i, 6:9])
            if (cfg.roof_guard_maps and dd.phase in (SEARCH, APPROACH) and 0.05 < agl < float(cfg.roof_agl)
                    and ("forest" if self.is_forest else str(self.map_kind)) in tuple(cfg.roof_guard_maps)):
                _far = True
                if dd.phase == APPROACH and pad is not None:
                    _far = float(np.linalg.norm(np.asarray(pad.xyz, float)[:2] - pos[:2])) > float(cfg.roof_pad_r)
                if _far:
                    _rise = 0.0
                    if 0.0 < agl_prev < 19.9:
                        _rise = max(0.0, float(vel[2]) - (agl - agl_prev) / 0.02)
                    _up = min(float(cfg.roof_climb), float(cfg.roof_gain) * (float(cfg.roof_agl) - agl) + _rise)
                    v_des = np.asarray(v_des, float).copy()
                    v_des[2] = max(float(v_des[2]), _up)
                    if self.is_forest:
                        v_des[2] = min(float(v_des[2]), max(0.0, cfg.forest_ceiling - float(pos[2])))
                    _k = float(np.clip((agl - float(cfg.roof_stop_agl))
                                       / max(float(cfg.roof_agl) - float(cfg.roof_stop_agl), 1e-6),
                                       float(cfg.roof_brake_min), 1.0))
                    v_des[:2] *= _k
                    if _YT_DIAG and _k < 0.999:
                        dd.yt |= 32
        if (self.is_forest and cfg.forest_overhead_m > 0.0 and dd.phase in (CLIMB, SEARCH)
                and not in_tilt_hold and float(v_des[2]) > 0.0):
            try:
                _d = np.asarray(depth[i], np.float32)
                if _d.ndim == 3:
                    _d = _d[..., 0]
                _h, _w = _d.shape
                _band = _d[int(_h * cfg.forest_overhead_rows[0]):int(_h * cfg.forest_overhead_rows[1]),
                           int(_w * 0.30):int(_w * 0.70)]
                _m = float(np.percentile(_band, cfg.forest_overhead_pct)) * 19.5 + 0.5
                if _m < float(cfg.forest_overhead_m):
                    v_des = np.asarray(v_des, float).copy()
                    v_des[2] = 0.0
            except Exception:
                pass
        if (cfg.apx_no_climb_r > 0.0 and dd.phase == APPROACH and pad is not None
                and not in_tilt_hold):
            _dpad = float(np.linalg.norm(np.asarray(pad.xyz, float)[:2] - pos[:2]))
            if _dpad < cfg.apx_no_climb_r:
                v_des = np.asarray(v_des, float).copy()
                v_des[2] = min(float(v_des[2]), float(getattr(self, "_apx_vz_cmd", v_des[2])))
        _split_pad = (pad is not None and getattr(pad, "split", False)
                      and dd.phase in (APPROACH, DESCEND))
        _nosep = (bool(cfg.park_nosep) and dd.nosep_step == int(self.step_i)
                  and (not cfg.park_nosep_maps
                       or ("forest" if self.is_forest else str(self._mass_route_kind())) in tuple(cfg.park_nosep_maps)))
        if _nosep:
            self._le_inc("nosep")
        if not in_tilt_hold and not _split_pad and not _nosep:
            if _YT_DIAG:
                _yt_hs = float(np.hypot(float(v_des[0]), float(v_des[1])))
            v_des = self._separate(state, i, v_des)
            if _YT_DIAG and float(np.hypot(float(v_des[0]), float(v_des[1]))) < 0.9 * _yt_hs - 0.05:
                dd.yt |= 2048
        _av_rot = 0.0
        _v_post = np.asarray(v_des, float)[:2]
        if (float(np.linalg.norm(_v_pre)) > cfg.yaw_track_min_spd
                and float(np.linalg.norm(_v_post)) > cfg.yaw_track_min_spd):
            _a0 = float(np.arctan2(_v_pre[1], _v_pre[0]))
            _a1 = float(np.arctan2(_v_post[1], _v_post[0]))
            _av_rot = abs((_a1 - _a0 + np.pi) % (2.0 * np.pi) - np.pi)
        if (float(cfg.om_bump_r) > 0.0 and cfg.om_maps and not in_tilt_hold
                and str(dd.phase) in self._om_phases() and self._om_on()):
            try:
                v_des = self._om_bump(i, pos, rpy, v_des, depth, vel=vel)   # blind_guard BMP (+ fsb)
            except Exception:                                        # noqa: BLE001
                self._om_cnt("err_bump")
        if (cfg.om2_maps and not in_tilt_hold and str(dd.phase) in tuple(cfg.om2_phases or ())
                and self._om2_on()):
            _t2 = _om2_clock()
            try:
                v_des = self._om2_filter(i, dd, pos, rpy, v_des, pad)       # candHc OM2 obstacle-memory filter
            except Exception:                                        # noqa: BLE001
                self._om2_cnt("err_flt")
            self._om2_cnt("us_flt", int(1e6 * (_om2_clock() - _t2)))
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
            if _YT_DIAG:
                dd.yt |= 64
        if cfg.tilt_guard_rad > 0.0 and (tilt >= cfg.tilt_guard_rad or in_tilt_hold):
            if _YT_DIAG:
                dd.yt |= 128
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
        _mrk = self._mass_route_kind()
        _ca_extra = tuple(getattr(cfg, "slew_ctrl_aware_maps", ()) or ())
        _pre_ok = True
        if float(getattr(cfg, "slew_pre_clear", 0.0) or 0.0) > 0.0 and self.map_kind == "other":
            _sc = getattr(self, "_start_clear", None)
            if _sc is None:
                _sc = {}; self._start_clear = _sc
            if i not in _sc:
                try:
                    _sc[i] = bool(float((np.asarray(depth[i], np.float32) < 0.385).mean()) < float(cfg.slew_pre_clear))
                except Exception:
                    _sc[i] = True
            _pre_ok = bool(_sc[i])
        if float(getattr(cfg, "slew_pre_cue", 0.0) or 0.0) > 0.0 and self.map_kind == "other":
            _cu = getattr(self, "_tall_cue", None)
            if _cu is None:
                _cu = {}; self._tall_cue = _cu
            try:
                _f = float((np.asarray(depth[i], np.float32)[:64] < 0.744).mean())
            except Exception:
                _f = 0.0
            _cu[i] = max(_cu.get(i, 0.0), _f)
            if _cu[i] >= float(cfg.slew_pre_cue):
                _pre_ok = False
        v_cmd = slew_velocity(v_des, vel, dv, tilt_rad=tilt,
                              tilt_knee=cfg.tilt_knee, tilt_stop=cfg.tilt_stop,
                              tilt_floor=cfg.tilt_floor,
                              ctrl_aware=(_mrk in FIX_SLEW_KINDS) or (_mrk in _ca_extra)
                              or (self.map_kind in _ca_extra and _pre_ok)
                              or (bool(cfg.fld_maps) and bool(cfg.fld_ca) and dd.phase == DESCEND
                                  and self._fld_on()))
        if _YT_DIAG:
            _yt_hd = float(np.hypot(float(v_des[0]), float(v_des[1])))
            if float(np.hypot(float(v_cmd[0]), float(v_cmd[1]))) < _yt_hd - 0.2:
                dd.yt |= 32768
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
                    if _YT_DIAG:
                        dd.yt |= 131072
        if (cfg.early_align_deg > 0.0 and not dd.aligned_once
                and (dd.phase in (CLIMB, SEARCH) or (dd.cenv and dd.phase == APPROACH))
                and ("forest" if self.is_forest else str(self.map_kind)) in tuple(cfg.early_align_maps or ())):
            _vxy = np.asarray(v_cmd[:2], float)
            _sp = float(np.linalg.norm(_vxy))
            if _sp > 0.3:
                _head = float(np.arctan2(_vxy[1], _vxy[0]))
                if (cfg.ee_maps and ("forest" if self.is_forest else str(self.map_kind)) in tuple(cfg.ee_maps)
                        and float(np.hypot(_v_pre[0], _v_pre[1])) > 0.3):
                    _head = float(np.arctan2(_v_pre[1], _v_pre[0]))   # EE: track heading, not the edge-dragged command
                _off = abs(math.degrees((_head - float(rpy[2]) + np.pi) % (2.0 * np.pi) - np.pi))
                _vt_rel = False
                if (_off > float(cfg.early_align_deg) and cfg.vt_maps
                        and ("forest" if self.is_forest else str(self.map_kind)) in tuple(cfg.vt_maps)
                        and self._own_start is not None):
                    # yaw_time VT: climb while turning; above the village obstacle tops release the cap
                    _risen = float(pos[2]) - float(self._own_start[i][2])
                    if _risen >= float(cfg.vt_clear_h) and agl >= float(cfg.vt_min_agl):
                        dd.aligned_once = True
                        _vt_rel = True
                        self._yt_n["vt_release"] += 1
                    elif float(v_cmd[2]) < float(cfg.vt_vz):
                        v_cmd = np.asarray(v_cmd, float).copy()
                        v_cmd[2] = float(cfg.vt_vz)
                        self._yt_n["vt_climb"] += 1
                if _vt_rel:
                    pass
                elif _off <= float(cfg.early_align_deg):
                    if dd.phase == SEARCH:
                        dd.aligned_once = True
                else:
                    _ee_ok = False
                    if (cfg.ee_maps and ("forest" if self.is_forest else str(self.map_kind)) in tuple(cfg.ee_maps)
                            and _off <= float(cfg.ee_max_deg)):
                        # yaw_time EE: fly the camera edge toward the track when its tube is clear
                        _sgn = math.copysign(1.0, (_head - float(rpy[2]) + np.pi) % (2.0 * np.pi) - np.pi)
                        _ea = float(rpy[2]) + _sgn * math.radians(float(cfg.ee_edge_deg))
                        _dw = np.array([math.cos(_ea), math.sin(_ea), 0.0])
                        try:
                            _Rm = rot_from_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2]))
                            _clr = FD.ray_clearance(depth[i], _Rm.T @ _dw, PD.AP_DEPTH_MAX_M, fov_deg=PD.AP_FOV_DEG,
                                                    grid=cfg.free_grid, r_eff=0.12 + float(cfg.free_margin),
                                                    lookahead=float(cfg.free_lookahead), aligned=True)
                        except Exception:                        # noqa: BLE001
                            _clr = 0.0
                        if _clr >= float(cfg.ee_clear):
                            _ee_ok = True
                            v_cmd = np.asarray(v_cmd, float).copy()
                            v_cmd[:2] = _dw[:2] * min(_sp, float(cfg.ee_speed))
                            self._yt_n["ee_steps"] += 1
                            if _YT_DIAG:
                                dd.yt |= 524288
                    if not _ee_ok and _sp > float(cfg.early_align_speed):
                        v_cmd = np.asarray(v_cmd, float).copy()
                        v_cmd[:2] = _vxy * (float(cfg.early_align_speed) / _sp)
                        if _YT_DIAG:
                            dd.yt |= 8
                    yaw = _head
        _ag = None
        if cfg.align_gate_maps and self.map_kind in cfg.align_gate_maps:
            _ag = (float(cfg.align_gate_deg0), float(cfg.align_gate_deg1), float(cfg.align_gate_min_speed))
        _agm = tuple(getattr(cfg, "align_gate_by_map", ()) or ())
        for _k4 in range(0, len(_agm) - 3, 4):
            if str(_agm[_k4]) == self.map_kind:
                _ag = (float(_agm[_k4 + 1]), float(_agm[_k4 + 2]), float(_agm[_k4 + 3]))
        if _ag is not None and dd.phase in (CLIMB, SEARCH, APPROACH):
            _d0, _d1, _vmin = _ag
            _vxy = np.asarray(v_cmd[:2], float)
            _sp = float(np.linalg.norm(_vxy))
            if _sp > _vmin:
                _head = float(np.arctan2(_vxy[1], _vxy[0]))
                _off = abs(math.degrees((_head - float(rpy[2]) + np.pi) % (2.0 * np.pi) - np.pi))
                if _off > _d0:
                    _k = float(np.clip((_d1 - _off) / max(_d1 - _d0, 1e-6), 0.0, 1.0))
                    _cap = _vmin + (_sp - _vmin) * _k
                    v_cmd = np.asarray(v_cmd, float).copy()
                    v_cmd[:2] = _vxy * (_cap / _sp)
                    yaw = _head
                    if _YT_DIAG and _k < 0.999:
                        dd.yt |= 16
        if (cfg.search_spin_rate > 0.0 and dd.phase == SEARCH
                and self._mass_route_kind() in tuple(cfg.search_spin_maps or ())
                and self.step_i * 0.02 >= float(cfg.search_spin_start_sec)
                and (self._band_locked or self.step_i * 0.02 >= 4.0)):
            # phase-offset the drones so the fleet's cameras do not all point the same way
            yaw = float(((self.step_i * 0.02 * cfg.search_spin_rate + i * 0.9) + np.pi) % (2.0 * np.pi) - np.pi)
        if dd.om2_yaw is not None:
            yaw = float(dd.om2_yaw)         # candHc OM2: look along the travel the unknown-space cap is holding back
            dd.om2_yaw = None
        if _wst_yaw is not None:
            yaw = _wst_yaw                  # WSTOP: keep the camera on the wall (early_align / yaw_track would turn it)
        if _YT_DIAG:
            try:
                dd.yt_h = (float(np.hypot(float(_v_pre[0]), float(_v_pre[1]))), _yt_hd,
                           float(np.hypot(float(v_cmd[0]), float(v_cmd[1]))), float(yaw))
            except Exception:
                pass
        return _encode(v_cmd, cfg.speed_limit, yaw)
_FOVE_RES = 128
_FOVE_A = 2.0 * np.arange(_FOVE_RES) / _FOVE_RES - 1.0              # TinyRenderer column sampling
_FOVE_B = 1.0 - 2.0 * (np.arange(_FOVE_RES) + 1.0) / _FOVE_RES      # TinyRenderer row sampling


def _fove_resid(f1, f2, fovs, stride):
    """bughunt FOVE: truncated relative depth residual of frame f1 reprojected into frame f2 per candidate FOV."""
    (c1, R1, d1), (c2, R2, d2) = f1, f2
    dn1 = d1[::stride, ::stride]
    ok1 = (dn1 > 0.001) & (dn1 < 0.999)
    D1 = dn1 * 19.5 + 0.5
    Aa, Bb = np.meshgrid(_FOVE_A[::stride], _FOVE_B[::stride])
    Aa, Bb, D1 = Aa[ok1], Bb[ok1], D1[ok1]
    ok2 = (d2 > 0.001) & (d2 < 0.999)
    D2 = d2 * 19.5 + 0.5
    out = np.full(len(fovs), np.inf)
    if D1.size < 100:
        return out
    for k, fv in enumerate(fovs):
        t = math.tan(math.radians(float(fv)) * 0.5)
        ray = np.stack([np.ones_like(Aa), -Aa * t, Bb * t], -1)
        P = c1 + D1[:, None] * (ray @ R1.T)
        q = (P - c2) @ R2
        x = q[:, 0]
        xs = np.maximum(x, 1e-6)
        i2 = np.rint((-q[:, 1] / (t * xs) + 1.0) * _FOVE_RES / 2.0).astype(int)
        j2 = np.rint((1.0 - q[:, 2] / (t * xs)) * _FOVE_RES / 2.0 - 1.0).astype(int)
        v = (x > 0.6) & (i2 >= 0) & (i2 < _FOVE_RES) & (j2 >= 0) & (j2 < _FOVE_RES)
        if int(v.sum()) < 100:
            continue
        jj, ii, xv = j2[v], i2[v], x[v]
        g = ok2[jj, ii]
        if int(g.sum()) < 100:
            continue
        r = np.abs(D2[jj[g], ii[g]] - xv[g]) / np.maximum(xv[g], 1.0)
        out[k] = float(np.mean(np.minimum(r, 0.05)))
    return out


def _fove_estimate(frames_by_drone):
    """bughunt FOVE: per-pair best FOV (coarse then fine) for every drone's frame pairs -> list of estimates."""
    ests = []
    coarse = np.arange(87.5, 92.5001, 0.25)
    for fr in frames_by_drone:
        if len(fr) < 3:
            continue
        pairs = [(0, len(fr) - 2), (1, len(fr) - 1), (0, len(fr) - 1)]
        for a, b in pairs:
            if a >= b:
                continue
            rc = _fove_resid(fr[a], fr[b], coarse, 4)
            if not np.isfinite(rc).any():
                continue
            c0 = float(coarse[int(np.argmin(rc))])
            fine = np.arange(c0 - 0.3, c0 + 0.3001, 0.05)
            rf = _fove_resid(fr[a], fr[b], fine, 2)
            if not np.isfinite(rf).any():
                continue
            ests.append(float(fine[int(np.argmin(rf))]))
    return ests


def _fcl_set(s, sd=0.02, w=1.0, floor=0.5) -> None:
    """fov_clue FCL: set (s float) or clear (s None) the FOV-coupled clue term in both posteriors."""
    return   # clean build: FCL (FOV-coupled clue) removed; CLUE_S keeps its default s=None
    try:
        from team.route_replan import pad_posterior as _pp
        _pp.CLUE_S.update(s=s, sd=sd, w=w)
    except Exception:                                                # noqa: BLE001
        pass
    try:
        from team.autopilot import geo_posterior as _gp
        _gp.CLUE_S.update(s=s, sd=sd, w=w, floor=floor)
    except Exception:                                                # noqa: BLE001
        pass


_FOVE_RES = 128
_FOVE_A = 2.0 * np.arange(_FOVE_RES) / _FOVE_RES - 1.0              # TinyRenderer column sampling
_FOVE_B = 1.0 - 2.0 * (np.arange(_FOVE_RES) + 1.0) / _FOVE_RES      # TinyRenderer row sampling


def _fove_resid(f1, f2, fovs, stride):
    """bughunt FOVE: truncated relative depth residual of frame f1 reprojected into frame f2 per candidate FOV."""
    (c1, R1, d1), (c2, R2, d2) = f1, f2
    dn1 = d1[::stride, ::stride]
    ok1 = (dn1 > 0.001) & (dn1 < 0.999)
    D1 = dn1 * 19.5 + 0.5
    Aa, Bb = np.meshgrid(_FOVE_A[::stride], _FOVE_B[::stride])
    Aa, Bb, D1 = Aa[ok1], Bb[ok1], D1[ok1]
    ok2 = (d2 > 0.001) & (d2 < 0.999)
    D2 = d2 * 19.5 + 0.5
    out = np.full(len(fovs), np.inf)
    if D1.size < 100:
        return out
    for k, fv in enumerate(fovs):
        t = math.tan(math.radians(float(fv)) * 0.5)
        ray = np.stack([np.ones_like(Aa), -Aa * t, Bb * t], -1)
        P = c1 + D1[:, None] * (ray @ R1.T)
        q = (P - c2) @ R2
        x = q[:, 0]
        xs = np.maximum(x, 1e-6)
        i2 = np.rint((-q[:, 1] / (t * xs) + 1.0) * _FOVE_RES / 2.0).astype(int)
        j2 = np.rint((1.0 - q[:, 2] / (t * xs)) * _FOVE_RES / 2.0 - 1.0).astype(int)
        v = (x > 0.6) & (i2 >= 0) & (i2 < _FOVE_RES) & (j2 >= 0) & (j2 < _FOVE_RES)
        if int(v.sum()) < 100:
            continue
        jj, ii, xv = j2[v], i2[v], x[v]
        g = ok2[jj, ii]
        if int(g.sum()) < 100:
            continue
        r = np.abs(D2[jj[g], ii[g]] - xv[g]) / np.maximum(xv[g], 1.0)
        out[k] = float(np.mean(np.minimum(r, 0.05)))
    return out


def _fove_estimate(frames_by_drone):
    """bughunt FOVE: per-pair best FOV (coarse then fine) for every drone's frame pairs -> list of estimates."""
    ests = []
    coarse = np.arange(87.5, 92.5001, 0.25)
    for fr in frames_by_drone:
        if len(fr) < 3:
            continue
        pairs = [(0, len(fr) - 2), (1, len(fr) - 1), (0, len(fr) - 1)]
        for a, b in pairs:
            if a >= b:
                continue
            rc = _fove_resid(fr[a], fr[b], coarse, 4)
            if not np.isfinite(rc).any():
                continue
            c0 = float(coarse[int(np.argmin(rc))])
            fine = np.arange(c0 - 0.3, c0 + 0.3001, 0.05)
            rf = _fove_resid(fr[a], fr[b], fine, 2)
            if not np.isfinite(rf).any():
                continue
            ests.append(float(fine[int(np.argmin(rf))]))
    return ests


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
