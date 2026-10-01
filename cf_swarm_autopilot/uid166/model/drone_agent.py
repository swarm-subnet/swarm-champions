from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent, _HERE.parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.environ.setdefault("SWARM_BASE", "uid167")
ACTION_DIM = 5

# Tuned settings baked into this bundle (the installer fills this in). SWARM_DEV_CFG
# still layers overrides on top so the same bundle can be swept during development.
_TUNED = json.loads(r"""{"cfg": {"min_hits_by_map": ["city", 2], "detect_every": 2}, "router_attr": {"sweep_quality": true, "sweep_quality_maps": ["open", "city"], "view_rel_deg": 45.0, "view_min_h_frac": 1.0, "sweep_phases": ["SEARCH", "CLIMB", "APPROACH", "DESCEND"], "kind_evidence": true, "sweep_r_by_map": {"open": 20.0, "village": 12.45}, "world_core_mix": true, "world_soft_hi": 120.0, "world_soft_lo": 90.0, "world_soft_maps": ["mountain"], "world_soft_margin": 4.0, "world_est_maps": ["village"], "world_est_lo": 42.0, "world_est_hi": 42.0, "village_radial": {"file": "pad_radial_prior_v563m20.json", "w": 1.0}, "p_det_by_map": {"village": 0.95}}, "post": {"pad_split_r": 1.5, "pad_split_rng": 12.0, "pad_split_maps": ["village", "forest", "city"], "pad_split_min_hits": 6, "climb_yaw_hold": true, "early_align_deg": 35, "early_align_maps": ["village"], "roof_guard_maps": ["city", "village"], "free_ray_half_maps": ["forest"], "free_ray_half_deg": 30.0, "align_gate_maps": ["forest"], "align_gate_deg0": 30.0, "align_gate_deg1": 60.0, "align_gate_min_speed": 0.8, "free_align_maps": ["city", "forest", "open", "other"], "climb_free_guard": true, "free_margin_by_map": ["city", 0.75], "free_ceiling_maps": ["forest"], "approach_gain_skip": [], "nf_maps": ["mountain"], "pad_z_lo": -0.1, "village_z_lo": 0.0, "kind_wall_maps": ["city", "open"], "apx_vfy_maps": ["village"], "apx_vfy_late_maps": ["open"], "apx_vfy_skip_champ": true, "dsc_hits_gate_maps": ["village"], "dsc_hits_gate_min": 5, "dsc_hits_gate_soft_from": 99, "start_mask_r": 1.1, "start_mask_maps": ["village"], "tko_hop_guard_maps": ["village"], "wall_stop_maps": ["village"], "assign_2opt_m": 3.0, "free_extrude_maps": ["city"], "free_pass_guard_maps": ["city"], "free_sat_maps": ["city", "forest"], "dsc_below_maps": ["forest"], "dsc_below_z": 0.15, "dsc_below_agl_gap": 1.0, "dsc_hold_maps": ["forest"], "dsc_hold_r": 0.4, "dsc_hold_above": 0.9, "dsc_hold_after_below": true, "village_veto_z": [0.6, 7.0], "village_veto_hits": 25, "village_veto_rate": 3.0, "dsc_gate_inview_min": 5, "hurry_real": true, "sep_dz_max": 1.5, "pad_world_box": true, "village_start_zmin": 0.43, "village_start_zmax": 1.5, "village_start_zmax_n": 2, "mtn_clue_z": 10.0, "pad_z_hi_open": 1.5, "start_mask_dz": 0.6, "dead_zero_steps": 3, "free_fov_open_pass": true, "free_pass_guard_max_deg": 90.0, "ne_maps": ["city", "open", "village", "mountain"], "ne_ticks": 5, "ne_range": 14.0, "ne_range_by_map": ["city", 18, "open", 18, "mountain", 14], "ne_max_hits": 30, "ne_occl_tol": 1.2, "ne_edge": 0.9, "close_gate_maps": ["city", "open"], "close_gate_min_hits": 4, "close_gate_dist": 8.0, "close_gate_elev": 20.0, "gla_maps": ["city", "open", "village"], "gla_s": 0.9, "gla_max_hits": 0, "gla_range": 25.8, "gla_phases": ["SEARCH", "CLIMB"], "gla_keep": 18.5, "gla_maxoff_deg": 40.0, "gla_travel_deg": 60.0, "village_start_box": 42.0, "forest_vote_fv_min": 0.5, "fv_force_spread": true, "dsc_below_retarget": 2, "dsc_gate_inview_hits_min": 3, "hunt": true, "hunt_maps": ["mountain"], "hunt_max_n": 2, "hunt_max_unfound": 1, "hunt_min_mass": 0.003, "hunt_found_min_hits": 8, "hunt_need_found": true, "same_pad_by_map": ["village", 1.5, "forest", 1.5], "spoken_skip_done": 1, "pad_split_pair_r": 1, "free_lat_maps": ["forest"], "free_lat_target": 0.7, "free_lat_w": 3.0, "esk_maps": ["city", "open"], "esk_r": 8, "esk_above": 2.6, "esk_vz": 1.2, "esk_dsc_gain": 2.5, "esk_dsc_lat": 0.9, "esk_dsc_above": 0.8, "esk_agl_tol": 0.6, "ne_dist_band": [16.0, 18.0, 0.5], "mtn_zfloor": 2, "start_mask_nodz_maps": ["mountain"], "steep_maps": ["mountain"], "steep_ratio": 0.67, "steep_min_d": 2.5, "steep_vz": 2.5, "steep_above": 6.5, "steep_climb": false, "park_maps": ["city", "open", "forest", "mountain", "village"], "cst_maps": ["city", "open", "forest", "mountain", "village"], "edr_maps": ["city", "forest", "mountain", "village"], "edr_a": 2, "edr_ab": 1, "edr_k": 2, "edr_margin": 0.45, "vra_maps": ["village"], "lav_maps": ["mountain"], "lav_avoid_r": 8, "lav_kp": 0.5, "lav_look": 20, "lav_lo_far": 2, "fld_maps": ["forest"], "fld_flare": 0.15, "fld_sink": 2, "fld_r": 0.3, "fld_ca": true, "fld_vdrop": 1, "fld_hold_s": 3, "fld_hold_back": 0.5, "fld_edr_only": true, "park_nosep": true, "lav_hi": 10.0, "lav_lo": 4.0, "tdo_maps": ["mountain"], "tdo_r": 0.3, "ne_no_hard": true, "ne_max_hits_by_map": ["city", 12, "open", 12, "village", 12], "esk_dsc_min_claim_d": 4, "trr_maps": ["mountain"], "twin_maps": ["forest"], "twin_r": 3, "twin_hits": 20, "fpn_maps": ["forest"], "fpn_range": 8, "fpn_ticks": 6, "mtn_zmax_start": 12.0, "dvr_maps": ["village"], "dvr_t0": 15.0, "dvr_rmax": 12.0, "blr_maps": ["village", "city", "forest", "mountain"], "fsp_maps": ["forest"], "fsp_deg": 30.0, "cls_maps": ["forest"], "spd_min_hits": 8, "sdb_maps": ["forest"], "clx_maps": ["forest"], "om_maps": ["city", "forest"], "om_sec": 5.0, "om_bump_r": 2.5, "om_bump_a": 4.0, "om_bump_lat": 0.1, "om_bump_margin": 0.3, "ehm_maps": ["mountain"], "ehm_h": 0.5, "ehm_slack": 1.0, "rg_cne_maps": ["city"], "gkc_maps": ["open", "village"], "gkc_fneg_min": 0.01, "lsd_maps": ["forest"], "hof_maps": ["open"], "fsb_abeam_deg": 45.0, "fsb_pad_r": 1.5, "fsb_stall": 1.5, "fsb_stall_d": 1.0, "fsb_esc": 3.0, "fsb_zmin": 0.3, "om_bump_margin_by_map": ["forest", 0.7, "city", 0.9], "om_bump_push_by_map": ["forest", 0.6, "city", 0.5], "om_bump_phases_by_map": ["forest", "CLIMB,SEARCH"], "fsb_dz_by_map": ["forest", 0.25, "city", 0.25], "fsb_only_maps": ["forest"], "fsb_lat_maps": ["forest", "city"], "fcl_maps": [], "fcl_w": 0.0, "fcl_bias": 0.03, "fcl_cem": false, "climb_pass_maps": ["forest"], "climb_speed_by_map": ["village", 2.4], "free_berth_cone_deg": 30.0, "free_berth_keep_speed": true, "free_berth_look": 3.0, "free_berth_maps": ["forest"], "free_berth_margin": 0.65, "mountain_detector_swap": true, "t0a_maps": ["village", "forest", "mountain"], "t0a_min_hits": 20, "t0a_sep": 1.8, "t0a_split_maps": ["mountain"], "t0a_ticks": 10, "world_post_clamp_m": 4.0, "world_post_mix": 3, "world_post_router": true, "spd_gate_nfb": true, "om_skip_route_open": true, "nf_phases": ["SEARCH", "APPROACH"], "tdf_acc": 7.0, "tdf_h0": 0.06, "tdf_vz": 0.3, "tdf_rest_vz": 0.05, "tdf_agl_tol": 0.17, "tdf_maps": ["mountain"], "dsk_by_map": {"village": {"r": 8.0, "above": 2.6, "vz": 1.2, "kp": 1.0, "agl_tol": 0.6, "latch_m": 0.3, "dsc_gain": 2.5, "dsc_lat": 0.9, "dsc_above": 0.8, "dsc_min_claim_d": 4.0}}, "om2_h3": true, "om2_upd_max": 1, "om2_hull_k": 2.0, "om2_route_city": true, "om2_hull_max_tall": 96, "om2_maps": ["city"], "om2_hull_max_low": 24, "om2_turn_blind_deg": 80.0, "mtd_gate": true, "mtd_est": true, "mtd_hm": true, "mtd_rim": true, "mtd_maps": ["mountain"], "world_post_margin_by_n": [[6, 2.0], [7, 2.0], [8, 2.0]], "dvr_w_same": 0.12, "search_alt_by_map": ["village", 6.4], "cwc_maps": ["mountain"], "cwc_cam_deg": 45.0, "gkc_vil_start_guard": true, "hurry_maps": ["mountain"], "gkc_city_pin": true, "alift_maps": ["city"], "alift_release_m": 0.0, "pad_split_pair_maps": ["village", "city"]}, "band_fc_min": null, "vpost": {"msp_max_n": 2, "search_plan_max_n_by_map": [["mountain", 2]]}, "band_n": {"mountain": {"2": [0.226, 0.258, 0.258, 0.258], "3": [0.305, 0.0, 0.347, 0.347], "4": [0.369, 0.21, 0.21, 0.21], "5": [0.305, 0.347, 0.347, 0.0], "6": [0.305, 0.347, 0.347, 0.0], "7": [0.226, 0.258, 0.258, 0.258], "8": [0.305, 0.347, 0.0, 0.347]}, "open": {"2": [0.305, 0.0, 0.347, 0.347]}, "city": {"2": [0.375, 0.225, 0.2, 0.2], "3": [0.292, 0.375, 0.333, 0.0], "4": [0.304, 0.348, 0.348, 0.0], "5": [0.242, 0.242, 0.242, 0.273], "6": [0.292, 0.333, 0.0, 0.375], "7": [0.242, 0.242, 0.273, 0.242], "8": [0.292, 0.0, 0.375, 0.333]}}, "landseg": {"forest": {"descend_speed": 0.54}}, "ours": {"slot_cfg": {"village_detector": {"checker": "team/out/ours/checker_s2rf1_village.npz", "score_thr": 0.85}, "forest_detector": {"checker": "team/out/ours/checker_s2rf2_forest.npz", "score_thr": 0.9, "snap_r": 1}, "mountain_detector": {"checker": "team/out/ours/checker_s2rf1_mountain.npz", "score_thr": 0.9}}}, "perf": {"fast": true}, "band_mix": {"city": [[22.0, 40.0, 0.2935], [22.0, 29.7, 0.2488], [29.7, 37.3, 0.2488], [37.3, 45.0, 0.209]]}, "band_pdf": {"file": "pad_band_pdf_cf.json", "maps": ["city"]}}""")


def _dev_cfg() -> dict:
    raw = os.environ.get("SWARM_DEV_CFG", "").strip()
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _TUNED.items()}
    if not raw:
        return out
    over = json.loads(raw) if raw.startswith("{") else json.loads(Path(raw).read_text())
    for k, v in over.items():
        if isinstance(v, dict):
            out.setdefault(k, {}).update(v)
        else:
            out[k] = v
    return out


_DEV = _dev_cfg()
# candHt perf keys (research/strike_emu/GUARD.md), all absent = candHv:
#   "fast": exact speedups (bit-identical actions): detector geometry channels without repeated work, numpy peak test,
#           scipy.ndimage imported at load instead of on the first detector tick; router pad fields with the clue term
#           evaluated for all search radii at once (pad_posterior.FAST).
#   "guard_s": act-time guard limit in wall seconds (0 = off); "guard_first_s", "guard_max_age", "guard_router",
#           "guard_det0", "guard_rest0", "guard_job_max_skips" map to the agent's act_guard_* settings.
_PERF = dict(_DEV.get("perf") or {})
_PERF_MAP = (("guard_first_s", "act_guard_first_s"), ("guard_max_age", "act_guard_max_age"),
             ("guard_router", "act_guard_router"), ("guard_det0", "act_guard_det0"),
             ("guard_rest0", "act_guard_rest0"), ("guard_job_max_skips", "act_guard_job_max_skips"))


def _tuplify(v):
    return tuple(v) if isinstance(v, list) else v


class DroneFlightController:
    def __init__(self):
        import torch
        torch.set_num_threads(int(os.environ.get("SWARM_TORCH_THREADS", "1")))
        if _PERF.get("fast"):
            import team.detector.ours.pad_geom as _pg
            import team.route_replan.pad_posterior as _ppf
            _pg.FAST = True
            _ppf.FAST = True
            try:
                from scipy import ndimage as _ndi  # noqa: F401  (else imported inside the first detector tick)
            except Exception:                      # noqa: BLE001
                pass
        from team.deploy_autopilot.base import build_agent
        kw = dict(
            general_ckpt="team/out/padgeneral_hn2_net",
            village_ckpt="team/out/padvillage_hn_net",
            village_thr=0.65,
            village_min_hits=4,
            min_hits_to_claim=3,
            min_hits_by_map=("city", 2),
            commit_t_by_map=("open", 25.0, "city", 25.0),
            device=os.environ.get("SWARM_DEVICE", "cpu"),
        )
        self._agent = _with_overrides(build_agent, **kw)
        _install_ours(self._agent, combine=_DEV.get("combine") or {})
        _learned = getattr(self._agent, "route_planner", None)
        from team.route_replan.adapter import install as _install_replan
        _radial = _DEV.get("radial")
        _router = _install_replan(
            self._agent,
            radial_path=(str(_HERE / "team" / "route_replan" / _radial) if _radial else None))
        self._router = _router
        _post = _variant_post()
        try:
            import team.route_replan.pad_posterior as _ppm
            for _k, _v in (_DEV.get("band_fc_min") or {}).items():
                _ppm.BAND_FC_MIN[str(_k)] = float(_v)
        except Exception:
            pass
        try:  # band-by-n (research/ideas/REPORT.md 4.3): {kind: {n: [w_screen, w_b1, w_b2, w_b3]}}
            import team.autopilot.geo_posterior as _gpm
            for _k, _t in (_DEV.get("band_n") or {}).items():
                _gpm.BAND_N[str(_k)] = {int(_n): tuple(float(_x) for _x in _w) for _n, _w in (_t or {}).items()}
        except Exception:
            pass
        try:  # soft city/forest bands (research/tpl/band_city_forest): "band_mix" {kind: [[lo, hi, w], ...]},
            # "band_pdf" {"file": name in team/route_replan, "maps": [kinds whose pdfs to load]}. A kind gets a
            # BAND_MIX entry only when every band's pdf loaded: a hard (truncated) band on city/forest is harmful.
            import team.autopilot.geo_posterior as _gpm
            import team.route_replan.pad_posterior as _ppm
            _bp = _DEV.get("band_pdf") or {}
            _tab = json.loads((_HERE / "team" / "route_replan" / str(_bp["file"])).read_text()) if _bp.get("file") else {}
            _bpm = set(str(_x) for _x in (_bp.get("maps") or _tab.keys()))
            for _k, _m in (_DEV.get("band_mix") or {}).items():
                _k = str(_k)
                _rows = {(round(float(a), 1), round(float(b), 1)): p
                         for a, b, p in ((_tab.get(_k) or {}).get("bands", []) if _k in _bpm else [])}
                _bands = tuple((float(a), float(b), float(w)) for a, b, w in _m)
                if not _bands:
                    continue
                if _k in ("city", "forest") and not all((round(a, 1), round(b, 1)) in _rows for a, b, _ in _bands):
                    continue                      # no pdf -> keep today's single field
                for (a, b), p in _rows.items():
                    _ppm.BAND_PDF[(_k, a, b)] = np.asarray(p, float)
                _gpm.BAND_MIX[_k] = _bands
        except Exception:
            pass
        for _k, _v in (_DEV.get("vpost") or {}).items():
            _post[_k] = _tuplify(_v)
        for _k, _v in (_post.get("router_pdet") or {}).items():
            try:
                import team.route_replan.adapter as _ad
                _ad.P_DET[_k] = float(_v)
            except Exception:
                pass
        if _post.get("router_sweep_r"):
            try:
                _router.cfg.sweep_r = float(_post["router_sweep_r"])
            except Exception:
                pass
        if _post.get("msp_maps"):
            _install_dispatch(self._agent, _learned, _router, tuple(_post["msp_maps"]),
                              int(_post.get("msp_max_n", 99)), _post.get("router_sweep_by_map"))
        for k, v in _post.items():
            if k not in ("msp_maps", "msp_max_n", "router_sweep_by_map", "router_pdet", "router_sweep_r"):
                setattr(self._agent.cfg, k, v)
        for k, v in (_DEV.get("router") or {}).items():
            setattr(_router.cfg, k, v)
        for k, v in (_DEV.get("router_attr") or {}).items():
            setattr(_router, k, _tuplify(v) if k.endswith("_maps") or k == "sweep_phases" else v)
        for k, v in (_DEV.get("post") or {}).items():
            setattr(self._agent.cfg, k, _tuplify(v))
        _ls = _DEV.get("landseg") or {}
        if _ls:  # LANDSEG (research/ideas/REPORT.md 4.2): per-map landing overrides {kind: {param: value}}
            try:
                from team.autopilot.agent import LANDSEG_KEYS as _lsk
                self._agent.cfg.landseg = {str(_k): {str(_p): float(_v) for _p, _v in (_t or {}).items() if _p in _lsk}
                                           for _k, _t in _ls.items()}
            except Exception:
                pass
        _gs = float(_PERF.get("guard_s", 0.0) or 0.0)
        self._guard = _gs > 0.0
        if self._guard:
            _c = self._agent.cfg
            _c.act_guard_s = _gs
            for _src, _dst in _PERF_MAP:
                if _src in _PERF:
                    setattr(_c, _dst, type(getattr(_c, _dst))(_PERF[_src]))
        self._agent.reset()
        self._last_state = None
        self._last_depth = None
        self._last_action = None

    def reset(self) -> None:
        self._agent.reset()
        self._last_state = None
        self._last_depth = None
        self._last_action = None
        self._kind_applied = False
        prof = _variant_profiles()
        if prof:
            for k, v in _variant_overrides().items():
                setattr(self._agent.cfg, k, v)
        self._kind_profiles = prof

    def _apply_kind_profile(self):
        prof = getattr(self, "_kind_profiles", None)
        if not prof or getattr(self, "_kind_applied", False):
            return
        ag = self._agent
        try:
            locked = bool(getattr(ag, "_band_locked", False)) or int(getattr(ag, "step_i", 0)) >= 75
            if not locked:
                return
            kind = ag._route_kind()
        except Exception:
            return
        self._kind_applied = True
        p = prof.get(kind)
        if p:
            for k, v in p.items():
                setattr(ag.cfg, k, v)

    def act(self, observation) -> np.ndarray:
        _gt0 = time.perf_counter() if getattr(self, "_guard", False) else None
        state = np.asarray(observation["state"])
        depth = np.asarray(observation["depth"])
        n = int(state.shape[0]) if state.ndim == 2 else 1
        # The validator can retry the exact observation after a slow act().
        # The policy is stateful, so the retry must return the previous action
        # without advancing its internal step, route or detector hit counts.
        if (self._last_state is not None
                and np.array_equal(state, self._last_state)
                and np.array_equal(depth, self._last_depth)):
            return self._last_action.copy()
        try:
            self._apply_kind_profile()
            if _gt0 is not None:
                self._agent._g_t0 = _gt0           # act guard: the agent budgets its optional work from here
            action = self._agent.act(observation)
            action = np.asarray(action, dtype=np.float32).reshape(n, ACTION_DIM)
            if not np.all(np.isfinite(action)):
                action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
            action = np.clip(action, -1.0, 1.0)
        except Exception:
            action = np.zeros((n, ACTION_DIM), dtype=np.float32)
        if _gt0 is not None:
            try:
                self._agent._guard_end(time.perf_counter())
            except Exception:                      # noqa: BLE001
                pass
        self._last_state = state.copy()
        self._last_depth = depth.copy()
        self._last_action = action.copy()
        return action


_OURS = {"onnx": "team/out/ours/best.onnx", "pt": "team/out/ours/best.pt",
         "checker": "team/out/ours/checker.npz", "score_thr": 0.9, "threads": 2,
         "forest_heat_thr": -1.0, "slot_thr": {}, "sources": "", "heat_thr": -1.0,
         "merge_m": -1.0}
_OURS.update(_DEV.get("ours") or {})
_OVERRIDES = {'commit_t_skip': ('forest', 'village', 'mountain'),
              'mass_planner': False,
              'route_plan_maps': ('mountain', 'city', 'open', 'village', 'forest')}
_OVERRIDES.update({k: _tuplify(v) for k, v in (_DEV.get("base") or {}).items()})

# The champion's (spilot_UID39) "ship" settings: robustness set + village z band.
_ROBUST = dict(apx_enter_any_alt=True, apx_enter_any_alt_r=1.2, apx_max_above=7.0, apx_no_climb_r=1.5,
               assign_swap=True, dsc_gate_maps=('city', 'village', 'forest', 'mountain'),
               dsc_gate_min_dist_by_map=('forest', 3.0), dsc_min_hits=8, land_settle=True,
               search_sink_cap=2.5, search_sink_maps=('mountain',), slew_ctrl_aware_maps=('mountain', 'other'),
               world_mountain=105.0, hunt=False, search_plan=False)
VARIANTS = {"theirs": {}, "merge": dict(_ROBUST)}
VARIANTS["ship"] = dict(VARIANTS["merge"], village_z_lo=0.15, village_z_hi=7.0, bad_pad_radius=2.0)
# post-install settings: per-map dispatch between the learned route + CEM search planner
# (msp_maps, fleets up to msp_max_n) and the corridor router (everything else).
_MSP = dict(msp_maps=("mountain",), search_plan=True, search_plan_maps=("mountain",),
            search_plan_arc_n=2, search_plan_budget=0.05, search_plan_budget_first=0.05,
            search_plan_climb_delay=8.0, search_plan_defer_steps=5, search_plan_det_range=16.0,
            search_plan_fixed_prefix=0, search_plan_holdout=True, search_plan_iters=16,
            search_plan_land_delay=6.0, search_plan_layouts=48, search_plan_min_gain=0.05,
            search_plan_pop=48, search_plan_sector=95.0, search_plan_sigma0=12.0,
            search_plan_map_params=(('open', 20.0, 120.0, 3.0, 5.0), ('forest', 12.0, 95.0, 0.5, 3.0)))
POST = {"ship": dict(_MSP, msp_max_n=3, search_plan_max_n_by_map=(("mountain", 3),))}
PROFILES = {}


def _variant_post():
    name = os.environ.get("SWARM_FORK_VARIANT", "ship")
    return dict(POST.get(name, POST.get("ship", {})))


class _Dispatch:
    def __init__(self, learned, router, maps, max_n=99, sweep_by_map=None):
        self.learned, self.router, self.maps, self.max_n = learned, router, maps, int(max_n)
        self.sweep_by_map = dict(sweep_by_map or {})
        self.mine_active = False

    def plan(self, starts, clue, n, map_kind):
        if map_kind in self.sweep_by_map:
            try:
                from dataclasses import replace as _replace
                self.router.cfg = _replace(self.router.cfg, sweep_r=float(self.sweep_by_map[map_kind]))
            except Exception:
                pass
        self.mine_active = bool(self.learned is not None and map_kind in self.maps and int(n) <= self.max_n)
        if self.mine_active:
            return self.learned.plan(starts, clue, n, map_kind)
        return self.router.plan(starts, clue, n, map_kind)

    def __getattr__(self, name):
        return getattr(self.router, name)


def _install_dispatch(agent, learned, router, maps, max_n=99, sweep_by_map=None):
    disp = _Dispatch(learned, router, maps, max_n, sweep_by_map)
    agent.route_planner = disp
    _obs = router.observe

    def observe(ag, obs):
        try:
            if disp.mine_active or (ag._route_kind() in maps and int(getattr(ag, "n", 99)) <= disp.max_n):
                return
        except Exception:
            pass
        return _obs(ag, obs)
    router.observe = observe
    _reset = router.reset

    def reset(*a, **k):
        disp.mine_active = False
        return _reset(*a, **k)
    router.reset = reset


def _variant_profiles():
    name = os.environ.get("SWARM_FORK_VARIANT", "ship")
    return dict(PROFILES.get(name, PROFILES.get("ship", {})))


def _variant_overrides():
    name = os.environ.get("SWARM_FORK_VARIANT", "ship")
    return dict(VARIANTS.get(name, VARIANTS["ship"]))


def _with_overrides(fn, **kw):
    kw.update(_OVERRIDES)
    kw.update(_variant_overrides())
    for k, v in (_DEV.get("cfg") or {}).items():     # development overrides win
        kw[k] = _tuplify(v)
    return fn(**kw)


def _load_ours():
    from team.detector.ours.pad_detect import PadDetector
    here = os.path.dirname(os.path.abspath(__file__))
    ck = os.path.join(here, _OURS["onnx"])
    chk = os.path.join(here, _OURS["checker"]) if _OURS["checker"] else None
    threads = int(os.environ.get("SWARM_TORCH_THREADS", _OURS["threads"]))
    try:
        det = PadDetector(ck, device="cpu", checker=chk, threads=threads)
    except Exception:
        det = PadDetector(os.path.join(here, _OURS["pt"]), device="cpu", checker=chk,
                          threads=threads)
    det.cfg.score_thr = float(_OURS["score_thr"])
    if _OURS.get("sources"):
        det.cfg.sources = str(_OURS["sources"])
    if float(_OURS.get("heat_thr", -1.0)) >= 0.0:
        det.cfg.heat_thr = float(_OURS["heat_thr"])
    if float(_OURS.get("merge_m", -1.0)) >= 0.0:
        det.cfg.merge_m = float(_OURS["merge_m"])
    # detv2 decode switches (research/detv2/pipeline), all absent = shipped decode
    for _k, _t in (("snap_r", int), ("snap_valid_mask", bool), ("score_thr_far", float), ("far_rng_m", float),
                   ("weak_thr", float)):
        if _k in _OURS:
            setattr(det.cfg, _k, _t(_OURS[_k]))
    return det


def _install_ours(agent, combine=None):
    prior = {slot: getattr(agent, slot, None)
             for slot in ("general_detector", "detector", "forest_detector",
                          "village_detector", "mountain_detector", "city_detector")}
    det = _load_ours()
    forest = det
    if float(_OURS.get("forest_heat_thr", -1.0)) >= 0.0:
        from team.detector.ours.pad_detect import PadDetector
        here = os.path.dirname(os.path.abspath(__file__))
        threads = int(os.environ.get("SWARM_TORCH_THREADS", _OURS["threads"]))
        forest = PadDetector(os.path.join(here, _OURS["onnx"]), device="cpu", threads=threads)
        forest.cfg.score_thr = float(_OURS["forest_heat_thr"])
    per_slot = dict(forest_detector=forest)
    for slot, thr in (_OURS.get("slot_thr") or {}).items():
        d2 = _load_ours()
        d2.cfg.score_thr = float(thr)
        per_slot[slot] = d2
    # detv2 T0-c: per-slot DetectConfig overrides {slot: {field: value}} (e.g. a far threshold on mountain_detector
    # only; mountain uses its slot only with post.mountain_detector_swap). Absent = every slot shares the shipped
    # detector. A slot without its own net is a shallow copy of the shared detector with its own DetectConfig (same
    # ONNX session, checker and geometry: no extra load time or memory). A slot may also name its own net / checker
    # ("onnx", "checker": a retrained per-map head). "general_detector" and "detector" are separate keys; the agent's
    # reset() puts general_detector back into detector.
    for slot, over in (_OURS.get("slot_cfg") or {}).items():
        over = dict(over or {})
        # detv2 T0-b (research/detv2/stage2): "checker" without "onnx" = this slot keeps the shared net and gets its
        # own refit checker npz (path relative to this folder; it may name the extra features border_px, zd, inval5)
        _slot_chk = over.pop("checker", None) if not over.get("onnx") else None
        if over.get("onnx"):
            from team.detector.ours.pad_detect import PadDetector
            here = os.path.dirname(os.path.abspath(__file__))
            threads = int(os.environ.get("SWARM_TORCH_THREADS", _OURS["threads"]))
            chk = over.get("checker", _OURS["checker"])
            d2 = PadDetector(os.path.join(here, over.pop("onnx")), device="cpu",
                             checker=(os.path.join(here, chk) if chk else None), threads=threads)
            over.pop("checker", None)
            d2.cfg = det.cfg.__class__(**vars(det.cfg))     # inherit the global switches, then the slot's own
        elif slot in per_slot and per_slot[slot] is not det:
            d2 = per_slot[slot]
        else:
            import copy as _copy
            d2 = _copy.copy(det)
            d2.cfg = det.cfg.__class__(**vars(det.cfg))
        for k, v in over.items():
            setattr(d2.cfg, k, type(getattr(d2.cfg, k))(v))
        if _slot_chk:
            d2.set_checker(os.path.join(os.path.dirname(os.path.abspath(__file__)), _slot_chk))
        per_slot[slot] = d2
    from team.deploy_autopilot.uid167_base import CombinedDetector
    for slot in ('general_detector', 'detector', 'forest_detector', 'village_detector',
                 'mountain_detector', 'city_detector'):
        head = per_slot.get(slot, det)
        mode = (combine or {}).get(slot) or (combine or {}).get("all")
        if mode and prior.get(slot) is not None:
            head = CombinedDetector(head, prior[slot], mode=str(mode))
        setattr(agent, slot, head)
    agent.reset()
    return agent
