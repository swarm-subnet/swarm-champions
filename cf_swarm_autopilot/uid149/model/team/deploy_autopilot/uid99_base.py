from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
UID20_CFG = dict(
    creep_speed=0.48,
    descend_speed=0.48,
    flare_alt=0.15,
    free_dir=True,
    forest_speed=2.8,
    climb_clear_m=0.0,
    climb_creep_tilt=0.28,
    tilt_fade_lo=0.4,
    tilt_fade_hi=0.7,
    tilt_guard_rad=0.62,
    tilt_guard_brake=1.5,
    tilt_guard_hold_sec=0.55,
    tilt_guard_speed_cap=1.8,
    climb_max_delta_v=0.0,
    post_climb_soft_sec=0.0,
    post_climb_speed_frac=1.0,
    own_annulus=True,
    own_arc_m=100.0,
    own_step_m=12.0,
    mountain_avoid_climb=2.5,
    own_maps=('city',),
    own_monotone_maps=('city',),
    own_near_first=(),
    own_split_tall=0.28,
    own_split_kind='',
    commit_t=50.0,
    landed_done=True,
    investigate=True,
    forest_sink_speed=2.0,
    descend_enter_r=1.5,
    descend_enter_wide=('forest',),
    investigate_hits=1,
    investigate_skip=('village', 'mountain'),
    descend_commit=True,
    avoid_range_mtn=20.0,
    yaw_scan_deg=50.0,
    mountain_min_hits=3,
    mountain_detector_swap=True,
    yaw_scan_maps=('mountain', 'city'),
)
ROUTE_CFG = dict(route_planner=True,
                 route_plan_maps=('mountain', 'city', 'open'),
                 route_start_sec=0.0)
MTN_CFG = dict(terrain_ff=1.0,
               terrain_ff_maps=("mountain",),
               mountain_climb_cap=3.0)
MASS_CFG = dict(mass_planner=True,
                mass_plan_maps=('village', 'forest'),
                mass_plan_n=(2, 3, 4, 5, 6, 7, 8),
                mass_plan_n_by_map={})
WEIGHTS = {
    "general": "team/out/padnet6_sb",
    "forest": "team/out/padforest_net",
    "village": "team/out/padvillage_u134",
    "mountain": "team/out/padmtn_net",   
    "route": "team/out/u235_route",           
    "posterior": "team/out/uid134_goal_posterior",
}
VILLAGE_THR = 0.80
MOUNTAIN_THR = 0.80
MOUNTAIN_SHAPE = dict(net_max_aspect=1.6, net_min_short=0.45)
FOREST_SHAPE = dict(net_max_aspect=2.6, net_min_short=0.20, min_pixels=3)
FOREST_VETO_SPREAD_M = 45.0     
def _ck(rel: str) -> str:
    p = ROOT / rel
    return str(p / "best.pt" if p.is_dir() else p)
def config(**overrides):
    from team.autopilot.agent import AutopilotConfig
    cfg = AutopilotConfig()
    for k, v in {**UID20_CFG, **ROUTE_CFG, **MASS_CFG, **MTN_CFG}.items():
        if not hasattr(cfg, k):
            raise AttributeError(
                "UID20 base sets %r which this AutopilotConfig lacks -- the "
                "agent and the base have drifted apart" % k)
        setattr(cfg, k, v)
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise AttributeError("unknown config field %r" % k)
        setattr(cfg, k, v)
    return cfg
def _install_forest_veto(agent):
    inner = agent._vote_forest
    def vetoed(depth, live, state=None, _inner=inner, _a=agent):
        was_locked = bool(getattr(_a, "_band_locked", False))
        try:
            _inner(depth, live, state)
        except TypeError:          
            _inner(depth, live)
        if was_locked or not getattr(_a, "_band_locked", False):
            return
        if getattr(_a, "map_kind", None) != "forest":
            return
        starts = getattr(_a, "_own_start", None)
        if starts is None or not len(starts):
            return
        try:
            spread = float(np.max(np.abs(np.asarray(starts, float)[:, :2])))
        except Exception:                                    
            return
        if not np.isfinite(spread) or spread <= FOREST_VETO_SPREAD_M:
            return
        cfg = _a.cfg
        _a.is_forest = False
        frac = _a._tall_hits / float(max(_a._clear_n, 1))
        _a.map_kind = "city" if frac > cfg.city_tall_frac else "open"
        _a.z_band = (cfg.pad_z_lo, cfg.pad_z_hi)
    agent._vote_forest = vetoed
def build_agent(route_ckpt: str = "", posterior: bool = True,
                forest_veto: bool = True, device: str = "cpu",
                general_thr: float = 0.5, general_ckpt: str = "",
                forest_ckpt: str = "",
                village_ckpt: str = "",
                village_thr: float = -1.0, **overrides):
    import torch
    torch.set_num_threads(int(os.environ.get("SWARM_TORCH_THREADS", "1")))
    from team.autopilot.agent import SwarmAutopilotAgent
    from team.detector.pads import PadConfig, PadNetDetector
    cfg = config(**overrides)
    gen = _ck(general_ckpt or WEIGHTS["general"])
    det = PadNetDetector(gen, device=device, thr=float(general_thr))
    fdet = PadNetDetector(_ck(forest_ckpt or WEIGHTS["forest"]), device=device,
                          cfg=PadConfig(**FOREST_SHAPE))
    vdet = PadNetDetector(_ck(village_ckpt or WEIGHTS["village"]), device=device,
                          thr=(VILLAGE_THR if village_thr < 0 else float(village_thr)))
    mdet = PadNetDetector(_ck(WEIGHTS["mountain"]), device=device)
    rp = None
    if cfg.route_planner:
        from team.autopilot.route_planner import RoutePlanner
        rp = RoutePlanner(_ck(route_ckpt or WEIGHTS["route"]), device=device)
    agent = SwarmAutopilotAgent(cfg=cfg, detector=det, forest_detector=fdet,
                                village_detector=vdet, mountain_detector=mdet,
                                route_planner=rp)
    if posterior and cfg.mass_planner:
        from team.autopilot.goal_posterior_runtime import ActorVisibleGoalPosterior
        agent.posterior = ActorVisibleGoalPosterior(
            Path(_ck(WEIGHTS["posterior"])), device=device)
        _wrap_posterior_feed(agent)
    if forest_veto:
        _install_forest_veto(agent)
    return agent
def _wrap_posterior_feed(agent):
    inner_act, inner_reset = agent.act, agent.reset
    post = agent.posterior
    def act(observation, _a=inner_act, _p=post):
        _p.observe_reset(observation)
        state = np.asarray(observation["state"])
        n = int(state.shape[0]) if state.ndim == 2 else 1
        agent._in_act = True
        try:
            action = np.asarray(_a(observation), dtype=np.float32).reshape(n, 5)
            if not np.all(np.isfinite(action)):
                action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
            return np.clip(action, -1.0, 1.0)
        except Exception:                                    
            return np.zeros((n, 5), dtype=np.float32)
        finally:
            agent._in_act = False
    def reset(_r=inner_reset, _p=post):
        if not getattr(agent, "_in_act", False):
            _p.reset()
        _r()
    agent.act = act
    agent.reset = reset
def assert_is_base(cfg) -> None:
    bad = [(k, getattr(cfg, k, "<absent>"), v)
           for k, v in {**UID20_CFG, **ROUTE_CFG, **MTN_CFG}.items()
           if getattr(cfg, k, None) != v]
    if bad:
        raise AssertionError(
            "config is not the UID20 base; %d field(s) differ, e.g. %s"
            % (len(bad), bad[:3]))
