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
from team.autopilot.memgrid import HeightMemory
from team.detector import pads as PD
from team.detector.localize import depth_to_meters, rot_from_rpy
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
    mem_lookahead: float = 12.0         # how far ahead the memory is consulted, metres
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
    # The champion's controller-aware velocity slew (keeps the commanded thrust vector
    # positive: dv_z >= -0.25 m/s per step, tilt-bounded dv_xy) is only applied on
    # city/open/village (FIX_SLEW_KINDS). Before the map is classified ("other") a drone
    # that spawns above cruise altitude commands a fast descent while accelerating and the
    # DSL PID flips it (TILT at ~1.5 s). Extra map kinds to apply the guard on.
    slew_ctrl_aware_maps: tuple = ()
    slew_pre_cue: float = 0.0            # >0: pre-classification ctrl-aware slew only while the drone has seen no tall structure nearby (upper-half pixels closer than 15 m, running max fraction below this)
    slew_pre_clear: float = 0.0          # >0: the pre-classification ('other') ctrl-aware slew only for drones whose first depth frame has fewer than this fraction of pixels closer than 8 m

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
        self._hunt_post = None
        self._hunt_seen = None
        self._hunt_xy = None
        self._hunt_tgt = {}
        self._splan = None
        self._start_clear = {}
        self._tall_cue = {}
        self._splan_pred = None

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
        if village:
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

    def _split_on(self) -> bool:
        return (float(self.cfg.pad_split_r) > 0.0
                and self._mass_route_kind() in tuple(self.cfg.pad_split_maps or ()))

    def _merge(self, xyz: np.ndarray, rng=None, view=None, obs=None, frame=None) -> None:
        if view is not None and self._view_gate_on():
            vr, vb = view
            if self.cfg.view_max_range > 0.0 and vr > self.cfg.view_max_range:
                return
            if (self.cfg.view_max_bearing > 0.0
                    and vb > self.cfg.view_max_bearing):
                return
        if self.z_band is not None:
            if not (self.z_band[0] <= float(xyz[2]) <= self.z_band[1]):
                if int(self.cfg.zband_rescue_hits) > 0:
                    self._zband_rescue(np.asarray(xyz, float), obs)
                return
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
                if dk < cfg.same_pad and (hit_d is None or dk < hit_d):
                    hit_k, hit_d = k, dk
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
                if dk < self.cfg.same_pad and (best is None or dk < best):
                    order, best = [k_pad], dk
        else:
            order = range(len(self.pads))
        for k_pad in order:
            pad = self.pads[k_pad]
            if float(np.linalg.norm(pad.xyz[:2] - xyz[:2])) < self.cfg.same_pad:
                if frame is not None:
                    frame.setdefault(k_pad, xyz[:2].copy())
                w = 1.0 / (pad.hits + 1.0)
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

        spoken = [self.pads[d.claim].xyz[:2] for d in self.d
                  if d.claim is not None and self.pads[d.claim].by is not None]
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
            _rad = (self.pads[k].split_r or float(self.cfg.pad_split_r)) if self.pads[k].split else self.cfg.same_pad
            if any(float(np.linalg.norm(xy - s)) < _rad for s in spoken):
                continue
            self.d[i].claim = k
            self.pads[k].by = i
            self.d[i].phase = APPROACH
            self.d[i].dsc_gated = False
            self.d[i].claim_dist = float(_dist)
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
                if dk < float(c.assign_swap_ratio) * dcur and (best is None or dk < best[0]):
                    best = (dk, k)
            if best is not None:
                cur.by = None
                dd.claim = best[1]
                self.pads[best[1]].by = i
                dd.claim_dist = float(best[0])
                dd.dsc_gated = False
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
            _rad = (self.pads[k].split_r or float(self.cfg.pad_split_r)) if self.pads[k].split else self.cfg.same_pad
            if any(float(np.linalg.norm(xy - sp)) < _rad for sp in spoken):
                continue
            self.d[i].claim = k
            self.pads[k].by = i
            self.d[i].phase = APPROACH
            spoken.append(xy)
            placed = True
        return placed

    def _endgame_hurry(self, dist: float, above: float) -> bool:
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
        dd = self.d[i]
        if dd.claim is not None:
            pad = self.pads[dd.claim]
            if soft and pad.aborts < int(self.cfg.soft_abort_max):
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
    def _free_dir(self, depth: np.ndarray, i: int, v: np.ndarray,
                  rpy: np.ndarray, pos: np.ndarray) -> np.ndarray:
        c = self.cfg
        speed = float(np.linalg.norm(v))
        if speed < 1e-6:
            return v
        R = rot_from_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        _fk = "forest" if self.is_forest else str(self.map_kind)
        if c.climb_free_guard and self.d[i].phase == CLIMB and self._refine_state is not None:
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
        v_body = R.T @ np.asarray(v, float)
        if (c.climb_pass_maps and self.d[i].phase == CLIMB and _fk in tuple(c.climb_pass_maps)
                and float(v_body[0]) < speed * FREE_DIR_MIN_COS):
            # a mostly vertical take-off command lies outside the camera view: free_direction
            # would remap it onto the frame's top edge at free_slow (climb ~0.8 instead of 1.4 m/s)
            return v
        fov_on = FIX_FOV not in ("0", "", "off") and (FIX_FOV != "city" or self._mass_route_kind() == "city")
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
            return v
        mtn = self.map_kind == "mountain"
        _margin = c.free_margin_mountain if mtn else c.free_margin
        _mbm = tuple(getattr(c, "free_margin_by_map", ()) or ())
        _mk = (("forest" if self.is_forest else str(self._mass_route_kind()))
               if getattr(c, "free_margin_route", False) else _fk)
        for _k2 in range(0, len(_mbm) - 1, 2):
            if str(_mbm[_k2]) == _mk:
                _margin = float(_mbm[_k2 + 1])
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
            el_max_world=_el_max, R=R if _el_max is not None else None)
        if pick is None:
            return v
        world = R @ np.asarray(pick, float)
        agree = float(np.dot(world, np.asarray(v, float) / speed))
        out = world * speed * (1.0 if agree > 0.98 else c.free_slow)
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
            mine = np.asarray(state[0, POS], float)
            for s in np.asarray(state[0, MATES], float).reshape(7, 7):
                if s[6] >= 0.5:
                    self.bad.append((mine + s[:3])[:2].copy())

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
        if cfg.hunt and self._hunt_post is not None and self.step_i % cfg.detect_every == 0:
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
            else:
                for i in live:
                    self._detect(state, depth, i)
                if cfg.mem_grid:
                    rots = [rot_from_rpy(*np.asarray(state[i, RPY], float)) for i in live]
                    poses = [np.asarray(state[i, POS], float) for i in live]
                    self._update_memory(live, poses, rots, depth)
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
            if float(np.linalg.norm(d)) > 1e-6 and risen >= cfg.climb_clear_m:
                _creep = cfg.cruise_speed * cfg.climb_creep
                if _tko:
                    _creep = max(_creep, math.sqrt(max(0.0, cfg.speed_limit ** 2 - _cs ** 2)) * 0.98)
                v_des[:2] = _unit(d) * _creep
                yaw = float(np.arctan2(d[1], d[0]))
            elif _cyh and dd.climb_yaw is not None:
                yaw = float(dd.climb_yaw)
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
                if tgt is not None:
                    _src = "hunt"
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
            if (cfg.canopy_brake_maps and self.map_kind in cfg.canopy_brake_maps
                    and 0.3 < agl < want_alt - cfg.canopy_brake_drop):
                v_des[:2] *= float(cfg.canopy_brake_frac)
            yaw = float(np.arctan2(d[1], d[0]))
            if FIX_NOGROUND and self._mass_route_kind() in FIX_BOX_KINDS and agl >= 19.9:
                v_des[2] = max(float(v_des[2]), 0.0)
                bound = self.WORLD.get(self._mass_route_kind())
                if bound is not None and float(np.max(np.abs(pos[:2]))) > bound:
                    home = -np.asarray(pos[:2], float)
                    v_des[:2] = _unit(home) * self._cruise(False)
                    yaw = float(np.arctan2(home[1], home[0]))
            _scan_ok = True
            if cfg.yaw_scan_align_deg > 0.0 and self.map_kind in tuple(cfg.yaw_scan_align_maps or ()):
                _off = abs((yaw - float(rpy[2]) + np.pi) % (2.0 * np.pi) - np.pi)
                _scan_ok = _off <= math.radians(float(cfg.yaw_scan_align_deg))
            if (cfg.yaw_scan_deg > 0.0 and _scan_ok
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
        elif dd.phase == APPROACH:
            d = pad.xyz[:2] - pos[:2]
            dist = float(np.linalg.norm(d))
            if self._probe_on():
                self._probe_sample(dd, pad, pos, agl)
            if self._touch_on() and 3.0 < dist < 16.0 and self.step_i % max(1, int(cfg.detect_every)) == 0:
                try:
                    self._touch_sample(depth, state, i, dd, pad)
                except Exception:
                    pass
            _apx_alt = (cfg.forest_approach_alt if self.is_forest
                        else cfg.approach_alt)
            want_z = pad.xyz[2] + _apx_alt
            _avz = (cfg.approach_vz if self.map_kind in cfg.approach_vz_maps
                    else 1.0)
            if cfg.apx_glide > 0.0:
                want_z = pad.xyz[2] + float(np.clip(dist * cfg.apx_glide,
                                                    cfg.apx_min_above, _apx_alt))
                _avz = max(_avz, cfg.apx_glide_vz)
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
            v_des[2] = np.clip(want_z - pos[2], -_avz, _avz)
            self._apx_vz_cmd = float(v_des[2])
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
            if (cfg.apx_enter_any_alt and dist < min(enter_r, float(cfg.apx_enter_any_alt_r))
                    and (pos[2] - pad.xyz[2]) < cfg.apx_max_above):
                dd.phase = DESCEND

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
                if (int(pad.hits) < int(cfg.dsc_min_hits)
                        and float(dd.claim_dist) >= _gmd
                        and self.step_i * 0.02 < float(cfg.dsc_gate_until_sec)
                        and ((not _gm) or (self._mass_route_kind() in _gm))):
                    # unconfirmed through a whole approach: a phantom, not a pad
                    _soft = ("forest" if self.is_forest else str(self.map_kind)) in tuple(
                        getattr(cfg, "dsc_gate_soft_maps", ()) or ())
                    self._refute(i, soft=_soft)
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
            v_des[:2] = np.clip(d * cfg.dsc_gain, -cfg.dsc_lateral, cfg.dsc_lateral)
            above = float(pos[2] - pad.xyz[2])
            if (cfg.agl_band_tol > 0.0 and dist <= cfg.agl_band_r
                    and agl < 19.9 and abs(agl - above) <= cfg.agl_band_tol):
                above = agl
            if above > (cfg.forest_flare_alt if self.is_forest else cfg.flare_alt):
                v_des[2] = -(cfg.forest_sink_speed if self.is_forest else cfg.sink_speed)
                if (cfg.dsc_center and dist > cfg.dsc_center_r
                        and above <= cfg.dsc_center_above):
                    v_des[2] = 0.0          # hold height, centre first
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
            self._pitch_now = -float(rpy[1])   # env rpy pitch is + nose-down
            # bearing of the intended travel relative to where the camera is pointing
            if float(np.hypot(_v_pre[0], _v_pre[1])) > 1e-6:
                self._clear_rel = float(np.arctan2(_v_pre[1], _v_pre[0]) - float(rpy[2]))
                self._clear_rel = float(np.arctan2(np.sin(self._clear_rel),
                                                   np.cos(self._clear_rel)))
            else:
                self._clear_rel = 0.0
            _v_pre3 = float(np.linalg.norm(np.asarray(v_des, float)))
            if self._use_free():
                v_des = self._free_dir(depth, i, v_des, rpy, pos)
            else:
                if cfg.steer or self.is_forest or self.map_kind in cfg.steer_maps:
                    v_des = self._steer_around(depth, i, v_des)
                v_des = self._avoid(depth, i, v_des, pos=pos,
                                    pad_xy=(pad.xyz[:2] if pad is not None else None))
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
        if not in_tilt_hold and not _split_pad:
            v_des = self._separate(state, i, v_des)
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
                              or (self.map_kind in _ca_extra and _pre_ok))
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
        if (cfg.early_align_deg > 0.0 and not dd.aligned_once and dd.phase in (CLIMB, SEARCH)
                and ("forest" if self.is_forest else str(self.map_kind)) in tuple(cfg.early_align_maps or ())):
            _vxy = np.asarray(v_cmd[:2], float)
            _sp = float(np.linalg.norm(_vxy))
            if _sp > 0.3:
                _head = float(np.arctan2(_vxy[1], _vxy[0]))
                _off = abs(math.degrees((_head - float(rpy[2]) + np.pi) % (2.0 * np.pi) - np.pi))
                if _off <= float(cfg.early_align_deg):
                    if dd.phase == SEARCH:
                        dd.aligned_once = True
                else:
                    if _sp > float(cfg.early_align_speed):
                        v_cmd = np.asarray(v_cmd, float).copy()
                        v_cmd[:2] = _vxy * (float(cfg.early_align_speed) / _sp)
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
        if (cfg.search_spin_rate > 0.0 and dd.phase == SEARCH
                and self._mass_route_kind() in tuple(cfg.search_spin_maps or ())
                and self.step_i * 0.02 >= float(cfg.search_spin_start_sec)
                and (self._band_locked or self.step_i * 0.02 >= 4.0)):
            # phase-offset the drones so the fleet's cameras do not all point the same way
            yaw = float(((self.step_i * 0.02 * cfg.search_spin_rate + i * 0.9) + np.pi) % (2.0 * np.pi) - np.pi)
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
