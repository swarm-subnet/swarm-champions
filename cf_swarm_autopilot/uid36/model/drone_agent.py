from __future__ import annotations
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
class DroneFlightController:
    def __init__(self):
        import torch
        torch.set_num_threads(int(os.environ.get("SWARM_TORCH_THREADS", "1")))
        from team.deploy_autopilot.base import build_agent
        self._agent = _with_overrides(build_agent, 
            general_ckpt="team/out/padgeneral_hn2_net",
            village_ckpt="team/out/padvillage_hn_net",
            village_thr=0.65,
            village_min_hits=4,
            min_hits_to_claim=3,
            min_hits_by_map=("city", 2),
            commit_t_by_map=("open", 25.0, "city", 25.0),
            device=os.environ.get("SWARM_DEVICE", "cpu"),
        )
        _install_ours(self._agent)
        _learned = getattr(self._agent, "route_planner", None)
        from team.route_replan.adapter import install as _install_replan
        _router = _install_replan(self._agent)
        _post = _variant_post()
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
            _install_dispatch(self._agent, _learned, _router, tuple(_post["msp_maps"]), int(_post.get("msp_max_n", 99)),
                              _post.get("router_sweep_by_map"))
            for k, v in _post.items():
                if k not in ("msp_maps", "msp_max_n", "router_sweep_by_map", "router_pdet", "router_sweep_r"):
                    setattr(self._agent.cfg, k, v)
        self._agent.reset()
    def reset(self) -> None:
        self._agent.reset()
        self._kind_applied = False
        prof = _variant_profiles()
        if prof:
            # restore the variant's base values first (profiles only revert what a kind wants reverted)
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
_OURS = {"onnx": "team/out/ours/best.onnx", "pt": "team/out/ours/best.pt", "checker": "team/out/ours/checker.npz", "score_thr": 0.9, "threads": 2, "forest_heat_thr": -1.0, "slot_thr": {}}
_OVERRIDES = {'commit_t_skip': ('forest', 'village', 'mountain'),
              'mass_planner': False,
              'route_plan_maps': ('mountain', 'city', 'open', 'village', 'forest')}
# Fork variants (local A/B via SWARM_FORK_VARIANT; the validator sets nothing so "ship" flies).
_ROBUST = dict(apx_enter_any_alt=True, apx_enter_any_alt_r=1.2, apx_max_above=7.0, apx_no_climb_r=1.5,
               assign_swap=True, dsc_gate_maps=('city', 'village', 'forest', 'mountain'),
               dsc_gate_min_dist_by_map=('forest', 3.0), dsc_min_hits=8, land_settle=True,
               search_sink_cap=2.5, search_sink_maps=('mountain',), slew_ctrl_aware_maps=('mountain', 'other'),
               world_mountain=105.0, hunt=False, search_plan=False)
VARIANTS = {
    "theirs": {},
    "merge": dict(_ROBUST),
}
VARIANTS["merge_c8"] = dict(_ROBUST, city_alt=8.0)
VARIANTS["merge_c11"] = dict(_ROBUST, city_alt=11.0)
# ablations of the robustness set
VARIANTS["merge_nog"] = {k: v for k, v in _ROBUST.items() if k not in ("dsc_min_hits", "dsc_gate_maps", "dsc_gate_min_dist_by_map")}
VARIANTS["merge_noapx"] = {k: v for k, v in _ROBUST.items() if not k.startswith("apx_")}
VARIANTS["merge_noswap"] = {k: v for k, v in _ROBUST.items() if k != "assign_swap"}
VARIANTS["merge_msp"] = dict(_ROBUST)
VARIANTS["merge_pm"] = dict(_ROBUST)
VARIANTS["merge_mtnonly"] = {k: v for k, v in _ROBUST.items() if k in ("search_sink_cap", "search_sink_maps", "slew_ctrl_aware_maps", "world_mountain", "hunt", "search_plan")}
VARIANTS["merge_landonly"] = {k: v for k, v in _ROBUST.items() if k not in ("search_sink_cap", "search_sink_maps", "slew_ctrl_aware_maps", "world_mountain")}
# merge2: the controller-aware slew only on mountain (with 'other' it also changed every map's takeoff: city -0.02)
_ROBUST2 = dict(_ROBUST, slew_ctrl_aware_maps=("mountain",))
VARIANTS["merge2"] = dict(_ROBUST2)
VARIANTS["merge2_msp"] = dict(_ROBUST2)
VARIANTS["merge2_fgate"] = dict(_ROBUST2, route_max_n_by_map=(("forest", 5),))
VARIANTS["merge2_mspo"] = dict(_ROBUST2)
VARIANTS["merge2_msp3"] = dict(_ROBUST2)
VARIANTS["merge_msp3"] = dict(_ROBUST)
VARIANTS["merge_mspf3"] = dict(_ROBUST)
VARIANTS["merge_pc"] = dict(_ROBUST, slew_pre_clear=0.40)
VARIANTS["merge_pdet6"] = dict(_ROBUST)
VARIANTS["merge_sw8"] = dict(_ROBUST)
VARIANTS["merge_cue"] = dict(_ROBUST, slew_pre_cue=0.03)
VARIANTS["merge_ct"] = dict(_ROBUST, commit_t_skip=("forest", "village"))
VARIANTS["merge_pdetv"] = dict(_ROBUST)
VARIANTS["merge_sw8p6"] = dict(_ROBUST)
VARIANTS["ship2"] = dict(_ROBUST)
VARIANTS["ship"] = dict(VARIANTS["merge"], village_z_lo=0.15, village_z_hi=7.0, bad_pad_radius=2.0)
# post-install settings (applied after the replan router is installed): per-map dispatch between the
# learned route + my CEM search planner (msp_maps) and their corridor router (all other maps).
_MSP = dict(msp_maps=("mountain",), search_plan=True, search_plan_maps=("mountain",),
            search_plan_arc_n=2, search_plan_budget=0.05, search_plan_budget_first=0.05,
            search_plan_climb_delay=8.0, search_plan_defer_steps=5, search_plan_det_range=16.0,
            search_plan_fixed_prefix=0, search_plan_holdout=True, search_plan_iters=16,
            search_plan_land_delay=6.0, search_plan_layouts=48, search_plan_min_gain=0.05,
            search_plan_pop=48, search_plan_sector=95.0, search_plan_sigma0=12.0,
            search_plan_map_params=(('open', 20.0, 120.0, 3.0, 5.0), ('forest', 12.0, 95.0, 0.5, 3.0)))
POST = {"merge_msp": dict(_MSP)}
POST["merge2_msp"] = dict(_MSP)
POST["merge2_mspo"] = dict(_MSP, msp_maps=("mountain", "open"), search_plan_maps=("mountain", "open"))
POST["merge2_msp3"] = dict(_MSP, msp_max_n=3, search_plan_max_n_by_map=(("mountain", 3),))
POST["merge_msp3"] = dict(_MSP, msp_max_n=3, search_plan_max_n_by_map=(("mountain", 3),))
POST["ship"] = dict(POST["merge_msp3"])
POST["ship2"] = dict(_MSP, msp_max_n=3, search_plan_max_n_by_map=(("mountain", 3),), router_sweep_by_map={"mountain": 8.0, "city": 8.0, "forest": 8.0, "open": 11.0, "village": 11.0})
POST["merge_sw8p6"] = dict(_MSP, msp_max_n=3, search_plan_max_n_by_map=(("mountain", 3),), router_sweep_r=8.0, router_pdet={"mountain": 0.6})
POST["merge_ct"] = dict(_MSP, msp_max_n=3, search_plan_max_n_by_map=(("mountain", 3),))
POST["merge_pdetv"] = dict(_MSP, msp_max_n=3, search_plan_max_n_by_map=(("mountain", 3),), router_pdet={"village": 0.6})
POST["merge_pdet6"] = dict(_MSP, msp_max_n=3, search_plan_max_n_by_map=(("mountain", 3),), router_pdet={"mountain": 0.6})
POST["merge_sw8"] = dict(_MSP, msp_max_n=3, search_plan_max_n_by_map=(("mountain", 3),), router_sweep_r=8.0)
POST["merge_mspf3"] = dict(_MSP, msp_maps=("mountain", "forest"), search_plan_maps=("mountain", "forest"), msp_max_n=3, search_plan_max_n_by_map=(("mountain", 3), ("forest", 3)))
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
# per-kind profiles: fields set once the map kind is locked (values = their pure config for that kind)
_PURE = dict(apx_enter_any_alt=False, apx_no_climb_r=0.0, assign_swap=False, dsc_min_hits=0,
             dsc_gate_maps=(), land_settle=False, slew_ctrl_aware_maps=())
PROFILES = {"merge_pm": {"city": dict(_PURE)}}
def _variant_profiles():
    name = os.environ.get("SWARM_FORK_VARIANT", "ship")
    return dict(PROFILES.get(name, PROFILES.get("ship", {})))
def _variant_overrides():
    name = os.environ.get("SWARM_FORK_VARIANT", "ship")
    return dict(VARIANTS.get(name, VARIANTS["ship"]))
def _with_overrides(fn, **kw):
    kw.update(_OVERRIDES)
    kw.update(_variant_overrides())
    return fn(**kw)
def _load_ours():
    import os
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
    return det
def _install_ours(agent):
    det = _load_ours()
    forest = det
    if float(_OURS.get("forest_heat_thr", -1.0)) >= 0.0:
        import os
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
    for slot in ('general_detector', 'detector', 'forest_detector', 'village_detector', 'mountain_detector', 'city_detector'):
        setattr(agent, slot, per_slot.get(slot, det))
    agent.reset()          
    return agent
