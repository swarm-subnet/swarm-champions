from __future__ import annotations
import json
import os
import sys
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
_TUNED = json.loads(r"""{"cfg": {"min_hits_by_map": ["city", 2]}, "router_attr": {"sweep_quality": true, "sweep_quality_maps": ["open", "city"], "view_rel_deg": 45.0, "view_min_h_frac": 1.0, "sweep_phases": ["SEARCH", "CLIMB", "APPROACH", "DESCEND"], "kind_evidence": true}, "post": {"pad_split_r": 1.5, "pad_split_rng": 12.0, "pad_split_maps": ["village", "forest"], "pad_split_min_hits": 6, "climb_yaw_hold": true, "early_align_deg": 35, "early_align_maps": ["village"], "roof_guard_maps": ["city", "village"], "free_ray_half_maps": ["forest"], "free_ray_half_deg": 30.0, "align_gate_maps": ["forest"], "align_gate_deg0": 30.0, "align_gate_deg1": 60.0, "align_gate_min_speed": 0.8, "free_align_maps": ["city", "forest", "open", "other"], "climb_free_guard": true, "free_margin_by_map": ["city", 0.75], "free_ceiling_maps": ["forest"], "approach_gain_skip": [], "nf_maps": ["mountain"], "pad_z_lo": -0.1, "village_z_lo": 0.0, "kind_wall_maps": ["city", "open"], "apx_vfy_maps": ["village"], "apx_vfy_late_maps": ["open"], "apx_vfy_skip_champ": true, "dsc_hits_gate_maps": ["village"], "dsc_hits_gate_min": 5, "dsc_hits_gate_soft_from": 99, "start_mask_r": 1.1, "start_mask_maps": ["village"], "tko_hop_guard_maps": ["village"], "wall_stop_maps": ["village"], "assign_2opt_m": 3.0, "free_extrude_maps": ["city"], "free_pass_guard_maps": ["city"], "free_sat_maps": ["city"], "dsc_below_maps": ["forest"], "dsc_below_z": 0.15, "dsc_below_agl_gap": 1.0, "dsc_hold_maps": ["forest"], "dsc_hold_r": 0.4, "dsc_hold_above": 0.9, "dsc_hold_after_below": true, "village_veto_z": [0.6, 7.0], "village_veto_hits": 25, "village_veto_rate": 3.0, "dsc_gate_inview_min": 5, "hurry_real": true, "sep_dz_max": 1.5, "pad_world_box": true, "village_start_zmin": 0.43, "village_start_zmax": 1.5, "village_start_zmax_n": 2, "mtn_clue_z": 10.0, "pad_z_hi_open": 1.5, "start_mask_dz": 0.6, "dead_zero_steps": 3, "free_fov_open_pass": true, "free_pass_guard_max_deg": 90.0, "ne_maps": ["city", "open"], "ne_ticks": 5, "ne_range": 14.0, "ne_range_by_map": ["city", 18.0, "open", 18.0], "ne_max_hits": 30, "ne_occl_tol": 1.2, "ne_edge": 0.9, "close_gate_maps": ["city", "open"], "close_gate_min_hits": 4, "close_gate_dist": 8.0, "close_gate_elev": 20.0, "gla_maps": ["city", "open"], "gla_s": 0.9, "gla_max_hits": 0, "gla_range": 25.8, "gla_phases": ["SEARCH", "CLIMB"], "gla_keep": 18.5, "gla_maxoff_deg": 40.0, "gla_travel_deg": 60.0, "village_start_box": 42.0, "forest_vote_fv_min": 0.5, "fv_force_spread": true, "dsc_below_retarget": 2, "dsc_gate_inview_hits_min": 3, "hunt": true, "hunt_maps": ["mountain"], "hunt_max_n": 2, "hunt_max_unfound": 1, "hunt_min_mass": 0.003, "hunt_found_min_hits": 8, "hunt_need_found": true}, "band_fc_min": {"village": 68.0}, "vpost": {"msp_max_n": 2, "search_plan_max_n_by_map": [["mountain", 2]]}}""")


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


def _tuplify(v):
    return tuple(v) if isinstance(v, list) else v


class DroneFlightController:
    def __init__(self):
        import torch
        torch.set_num_threads(int(os.environ.get("SWARM_TORCH_THREADS", "1")))
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
        self._agent.reset()

    def reset(self) -> None:
        self._agent.reset()
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
        state = np.asarray(observation["state"])
        n = int(state.shape[0]) if state.ndim == 2 else 1
        try:
            self._apply_kind_profile()
            action = self._agent.act(observation)
            action = np.asarray(action, dtype=np.float32).reshape(n, ACTION_DIM)
            if not np.all(np.isfinite(action)):
                action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
            return np.clip(action, -1.0, 1.0)
        except Exception:
            return np.zeros((n, ACTION_DIM), dtype=np.float32)


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
