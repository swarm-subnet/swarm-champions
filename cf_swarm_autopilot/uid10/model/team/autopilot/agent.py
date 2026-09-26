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
    edr_key: int = -1                  # land_seq EDR: claim whose drop started early (-1 none)
    nosep_step: int = -1               # land_ext park_nosep: step on which PARK / CST fired (no separation push)
    fld_hkey: int = -1                 # land_ext FLD hold: claim the hold counter belongs to
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
    ne_run: int = 0         # NE: in-view detector ticks without a hit since the last hit / fire
    ne_safe: bool = False   # NE: a claimant passed the descend-entry gates on this entry; NE never fires on it
    bl_raw: object = None   # F3: recent merged proposals (x, y, z, range) while dsc_below_retarget is on
    fpn_run: int = 0        # FPN: in-view no-hit detector ticks of the current claimant (or of the fleet, unclaimed)
    fpn_by: int = -2        # FPN: whose views the run counts (claimant index, -1 = unclaimed)
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
        self._below_log = []          # diag: P2r below events (t, drone, pad, above, dist, agl)
        self._vveto_log = []          # diag: P1b vetoes (t, drone, pad, z, hits, rate)
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
        self._hunt_n = {"hunt_steps": 0, "hunt_err": 0, "hunt_first_t": 0}   # research counter (candH7a only)
        self._b5_n = {"f1_vbox": 0, "f2_skip_clear": 0, "f2_skip_tall": 0, "f2b_block": 0, "f3_split": 0,
                      "f3_offset": 0, "f3_below": 0, "f4_gate": 0}
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
        return (float(self.cfg.pad_split_r) > 0.0
                and self._mass_route_kind() in tuple(self.cfg.pad_split_maps or ()))

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
                    return
        if split or split_b or any(q.split for q in self.pads):
            # nearest entry once entries may sit closer than same_pad (first match otherwise, as before)
            order, best = [], None
            for k_pad, pad in enumerate(self.pads):
                dk = float(np.linalg.norm(pad.xyz[:2] - xyz[:2]))
                if dk < self._same_pad_r() and (best is None or dk < best):
                    order, best = [k_pad], dk
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
        if _blr:
            self._bl_note(fresh, xyz, rng)
        self._note_view(fresh, obs)
        self.pads.append(fresh)
        if frame is not None:
            frame[len(self.pads) - 1] = xyz[:2].copy()
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
        spoken = [self.pads[d.claim].xyz[:2] for d in self.d
                  if d.claim is not None and self.pads[d.claim].by is not None
                  and not (_spk_done and self.pads[d.claim].done)]
        if str(cfg_assign := getattr(self.cfg, "assign_mode", "")) == "par":
            if self._assign_par(state, cand, free, spoken):
                return
        _ = cfg_assign
        if self.cfg.assign_reopt:
            self._reopt_claims(state, cand)
        taken_i = set()
        for _dist, i, k in pairs:
            if i in taken_i or self.pads[k].by is not None or self.pads[k].done:
                continue
            xy = self.pads[k].xyz[:2]
            _rad = (self.pads[k].split_r or float(self.cfg.pad_split_r)) if self.pads[k].split else self._same_pad_r()
            if any(float(np.linalg.norm(xy - s)) < _rad for s in spoken):
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
        if self.cfg.assign_reach or self.cfg.assign_swap:
            self._assign_review(state, gate)

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

    def _endgame_hurry(self, dist: float, above: float) -> bool:
        maps = tuple(getattr(self.cfg, "hurry_maps", ()) or ()) or HURRY_MAPS
        if self._mass_route_kind() not in maps:
            return False
        t_left = 60.0 - self.step_i / 50.0
        if self.cfg.hurry_real:
            # S2: budget the descent at the real sink rate (sink above the flare height, then descend_speed)
            c = self.cfg
            fl = float(c.forest_flare_alt if self.is_forest else c.flare_alt)
            sk = float(c.forest_sink_speed if self.is_forest else c.sink_speed)
            ab = max(0.0, float(above))
            need = (dist / 1.5 + max(0.0, ab - fl) / max(sk, 0.1) + min(ab, fl) / max(float(c.descend_speed), 0.1)
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
    def _refute(self, i: int, soft: bool = False, never_hard: bool = False) -> None:
        dd = self.d[i]
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
        classification_ready = self._band_locked or kind == "mountain"
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
            lat_w=float(c.free_lat_w))
        if pick is None:
            return v
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
        out = world * speed * (1.0 if (agree > 0.98 and not _guard) else _fsl)
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
        if dd.tdo_off is None or dd.tdo_nb != len(dd.tdo_buf):
            self._tdo_g = None
            try:
                off, why = self._tdo_fit(dd, pad)
            except Exception:                            # noqa: BLE001
                off, why = np.zeros(2), "err"
            dd.tdo_off, dd.tdo_nb = off, len(dd.tdo_buf)
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
        if lam > dd.tdo_lam and ab <= float(self.cfg.tdo_lock_ab):
            lam = dd.tdo_lam
        if lam != dd.tdo_lam:
            self._tdo_inc("lam_up" if lam > dd.tdo_lam else "lam_dn")
        dd.tdo_lam = lam
        tgt = u * lam
        rate = float(self.cfg.tdo_rate)
        cur = dd.tdo_eff if dd.tdo_eff is not None else np.zeros(2)
        if rate <= 0.0 or float(np.linalg.norm(tgt)) < float(np.linalg.norm(cur)) - 1e-9:
            dd.tdo_eff = tgt                             # no rate limit, or a safety decrease: at once
        elif ab > float(self.cfg.tdo_move_ab):
            # outward moves of the target at <= tdo_rate m/s and only while > tdo_move_ab above the pad: a late
            # lateral correction in the 2.8 m/s EDR drop freezes a 10-15 deg tilt into the contact (B 426 d0 TILT)
            dv = tgt - cur
            nv = float(np.linalg.norm(dv))
            mx = rate * 0.02
            dd.tdo_eff = tgt if nv <= mx else cur + dv * (mx / nv)

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
            if k >= len(h0) or int(p.hits) > int(h0[k]):
                p.fpn_run = 0                 # new this tick or gained a hit: the run restarts
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
        cand = []
        for k, p in enumerate(self.pads):
            if k >= len(h0) or int(p.hits) > int(h0[k]):
                p.ne_run = 0                  # new this tick or gained a hit: the run restarts
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
            dist = None
            if use_band:
                dist = zg * np.sqrt(1.0 + a * a + b * b)       # camera to the estimate, straight line
                ok &= dist <= rng
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
                and self.step_i % cfg.detect_every != 0):
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
                self._update_memory(live, poses, rots, depth)
                try:
                    batch = det.propose_batch(poses, rots, depth[live])
                except Exception:
                    batch = [[] for _ in live]
                _ne = self._ne_on()                       # batch 4 NE: entry hits before the merge
                _fpn = bool(cfg.fpn_maps) and self._fpn_on()
                _h0 = [int(p.hits) for p in self.pads] if (_ne or _fpn) else None
                _gla = self._gla_on()
                for oi, props in enumerate(batch):
                    frame = {}
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
                        if int(cfg.zband_rescue_hits) > 0:      # score only feeds the z-band rescue
                            _qs = getattr(q, "score", None)
                            self._merge_score = None if _qs is None else float(_qs)
                        self._merge(c, rng=self._refine_rng(c, poses, batch),
                                    view=(vr, vb), obs=opos, frame=frame)
                    if _gla and frame:
                        self._gla_trigger(live[oi], frame, opos)
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
            if agl >= top:
                dd.phase = SEARCH
                dd.climb_exit_t = float(self.step_i)
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
            _apx_alt = (cfg.forest_approach_alt if self.is_forest
                        else cfg.approach_alt)
            want_z = pad.xyz[2] + _apx_alt
            _avz = (cfg.approach_vz if self.map_kind in cfg.approach_vz_maps
                    else 1.0)
            if cfg.apx_glide > 0.0:
                want_z = pad.xyz[2] + float(np.clip(dist * cfg.apx_glide,
                                                    cfg.apx_min_above, _apx_alt))
                _avz = max(_avz, cfg.apx_glide_vz)
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
            if (float(cfg.fld_above) > 0.0 and cfg.fld_maps and self._fld_on() and dist < float(cfg.edr_r)
                    and int(pad.hits) >= int(cfg.fld_min_hits)
                    and agl >= float(pos[2] - pad.xyz[2]) - float(cfg.edr_agl_tol)):
                # FLD: sink toward pad_z + fld_above inside the EDR envelope (shorter final drop)
                want_z = min(want_z, float(pad.xyz[2]) + float(cfg.fld_above))
                _avz = max(_avz, float(cfg.fld_avz))
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
                _sh = float(pad.xyz[2]) + float(cfg.steep_above) - float(pos[2])
                if abs(_sh) > float(cfg.steep_ratio) * dist and (_sh < 0.0 or bool(cfg.steep_climb)):
                    # mtn_claims STEEP: straight 3-D line to the approach point above the pad
                    _sL = math.hypot(dist, _sh)
                    v_des[:2] = _unit(d) * (float(top) * dist / _sL)
                    v_des[2] = float(np.clip(float(top) * _sh / _sL, -float(cfg.steep_vz), float(cfg.steep_vz)))
                    self._apx_vz_cmd = float(v_des[2])
                    _stp = True
                    self._mc_inc("steep_steps")
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
                    _er0 = (cfg.descend_enter_r if self.map_kind in cfg.descend_enter_wide
                            else cfg.descend_enter_r_tight)
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
            enter_r = (cfg.descend_enter_r
                       if self.map_kind in cfg.descend_enter_wide
                       else cfg.descend_enter_r_tight)
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
            if (cfg.apx_enter_any_alt and dist < min(enter_r, float(cfg.apx_enter_any_alt_r))
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

        elif dd.phase == DESCEND:
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

            _lsr = cfg.land_settle_r
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
                if self._tdo_active(dd):
                    d = (np.asarray(pad.xyz, float)[:2] + dd.tdo_eff) - pos[:2]
                    dist = float(np.linalg.norm(d))
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
                _eg = float(cfg.dsc_gain) + _ef * (float(cfg.esk_dsc_gain) - float(cfg.dsc_gain))
                _el = float(cfg.dsc_lateral) + _ef * (float(cfg.esk_dsc_lat) - float(cfg.dsc_lateral))
                v_des[:2] = np.clip(d * _eg, -_el, _el)
            else:
                v_des[:2] = np.clip(d * cfg.dsc_gain, -cfg.dsc_lateral, cfg.dsc_lateral)
            _edr_d = (dd.edr_key >= 0 and dd.claim is not None and dd.edr_key == int(dd.claim) and self._edr_on()
                      and float(pos[2] - pad.xyz[2]) > float(cfg.edr_low))
            if _edr_d:
                # EDR: keep the braking profile while higher than edr_low above the pad
                v_des[:2] = _unit(d) * self._edr_vlat(dist)
            above = float(pos[2] - pad.xyz[2])
            if (cfg.agl_band_tol > 0.0 and dist <= cfg.agl_band_r
                    and agl < 19.9 and abs(agl - above) <= cfg.agl_band_tol):
                above = agl
            _fl = (cfg.forest_flare_alt if self.is_forest else cfg.flare_alt)
            _sk = (cfg.forest_sink_speed if self.is_forest else cfg.sink_speed)
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
                    v_des[2] = -max(cfg.sink_speed, 1.6)
                elif _edr_d and bool(cfg.edr_prio) and float(v_des[2]) < 0.0:
                    # EDR: the lateral profile keeps priority under the 3 m/s norm
                    _vl2 = float(v_des[0] ** 2 + v_des[1] ** 2)
                    v_des[2] = -min(-float(v_des[2]), math.sqrt(max(0.09, 9.0 - _vl2)))
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
                    v_des[2] = (-_sk if above > _fl else -cfg.descend_speed)
            if (float(getattr(cfg, "dsc_tilt_hold_deg", 0.0)) > 0.0 and self._esk_on() and dd.claim is not None
                    and float(pos[2] - pad.xyz[2]) > float(cfg.dsc_tilt_hold_above)
                    and math.degrees(max(abs(float(rpy[0])), abs(float(rpy[1])))) > float(cfg.dsc_tilt_hold_deg)
                    and not self._endgame_hurry(dist, float(pos[2] - pad.xyz[2]))):
                # audit_cov tilt hold: no sink while swinging this high above the pad; base lateral law
                v_des[:2] = np.clip(d * cfg.dsc_gain, -cfg.dsc_lateral, cfg.dsc_lateral)
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
            if cfg.dsc_below_maps or cfg.dsc_hold_maps:
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
                if _ae < 0.6 and _miss and dist > cfg.land_radius:
                    v_des[2] = max(float(v_des[2]), 0.0)
            if cfg.land_settle:
                if dd.hop:
                    # climbing back to land_hop_alt above the pad, centring on the way
                    v_des[:2] = np.clip(d * cfg.dsc_gain, -cfg.dsc_lateral, cfg.dsc_lateral)
                    v_des[2] = 0.6
                    if above >= cfg.land_hop_alt or agl >= cfg.land_hop_alt:
                        dd.hop = False
                elif agl < 0.30 or _edge:
                    if dist <= _lsr and not _edge:
                        v_des[0] = 0.0
                        v_des[1] = 0.0
                    elif (dist <= (cfg.forest_abort_radius if self.is_forest else cfg.abort_radius) or _edge) \
                            and dd.hops < int(cfg.land_hop_max):
                        dd.hop = True
                        dd.hops += 1
                        v_des[:2] = 0.0
                        v_des[2] = 0.6
                    elif _edge:
                        v_des[:2] = 0.0
                        v_des[2] = max(float(v_des[2]), 0.0)
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
        if _wst_yaw is not None:
            yaw = _wst_yaw                  # WSTOP: keep the camera on the wall (early_align / yaw_track would turn it)
        if _YT_DIAG:
            try:
                dd.yt_h = (float(np.hypot(float(_v_pre[0]), float(_v_pre[1]))), _yt_hd,
                           float(np.hypot(float(v_cmd[0]), float(v_cmd[1]))), float(yaw))
            except Exception:
                pass
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
