from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from team.autopilot.agent import AutopilotConfig, SwarmAutopilotAgent
from team.detector.pads import PadConfig, PadNetDetector
from route_net import RoutePlanner
from goal_posterior_runtime import ActorVisibleGoalPosterior

SHIPPED_CFG = dict(
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
    mountain_min_hits=2,
    mountain_detector_swap=True,
    yaw_scan_maps=('mountain', 'city'),
)

WEIGHTS = {

    "general": ("team/out/padnet6_sb/best.pt", "team/out/padnet6_sb"),
    "forest": ("team/out/padforest_net/best.pt", "team/out/padforest_net"),

    "village": ("team/out/padvillage_net/best.pt", "team/out/padvillage_net"),

}

ACTION_DIM = 5
POSTERIOR_MAPS = ("village", "forest")

def _find(rels):
\
\
\
\
\

    if isinstance(rels, str):
        rels = (rels,)
    for base in (_HERE, _HERE.parent, _HERE.parent.parent, Path.cwd()):
        for rel in rels:
            for cand in (base / rel, base / Path(rel).name):
                if cand.exists() and cand.is_file():
                    return str(cand)
    return None

class DroneFlightController:
    def __init__(self):
        import torch

        torch.set_num_threads(int(os.environ.get("SWARM_TORCH_THREADS", "1")))
        device = os.environ.get("SWARM_DEVICE", "cpu")

        det = fdet = None
        gen = _find(WEIGHTS["general"])
        if gen is not None:
            det = PadNetDetector(gen, device=device)
        fp = _find(WEIGHTS["forest"])
        if fp is not None:
            fdet = PadNetDetector(
                fp, device=device,
                cfg=PadConfig(net_max_aspect=1.6, net_min_short=0.45),
            )
        vp = _find(WEIGHTS["village"])
        vdet = PadNetDetector(vp, device=device, thr=0.80) if vp is not None else None

        mdet = PadNetDetector(
            gen, device=device, thr=0.80,
            cfg=PadConfig(net_max_aspect=1.6, net_min_short=0.45),
        ) if gen is not None else vdet
        agent = SwarmAutopilotAgent(cfg=AutopilotConfig(**SHIPPED_CFG),
                                    detector=det, forest_detector=fdet,
                                    village_detector=vdet,
                                    mountain_detector=mdet)
        self._agent = agent

                                                                              
                                                                                           
                                                                                             
                                                                                        
                                                                                        
                                                                                    
        _vote_inner = agent._vote_forest

        def _vote_forest_vetoed(depth, live, _inner=_vote_inner, _a=agent):
            was_locked = bool(getattr(_a, "_band_locked", False))
            _inner(depth, live)
            if was_locked or not getattr(_a, "_band_locked", False):
                return
            if getattr(_a, "map_kind", None) != "forest":
                return
            starts = getattr(_a, "_own_start", None)
            if starts is None or not len(starts):
                return
            try:
                spread = float(np.max(np.abs(np.asarray(starts, dtype=float)[:, :2])))
                if not np.isfinite(spread) or spread <= 45.0:
                    return
            except Exception:
                return
            cfg = _a.cfg
            _a.is_forest = False
            frac = _a._tall_hits / float(max(_a._clear_n, 1))
            _a.map_kind = "city" if frac > cfg.city_tall_frac else "open"
            _a.z_band = (cfg.pad_z_lo, cfg.pad_z_hi)

        agent._vote_forest = _vote_forest_vetoed

        self._route = RoutePlanner(str(_HERE / "team" / "out" / "uid134_route_policy" / "best.pt"), device="cpu")
        agent.route_planner = self._route
        agent.cfg.route_planner = True
        agent.cfg.route_plan_maps = ('mountain', 'city', 'open')
        agent.cfg.route_start_sec = 0.0

        self._posterior = ActorVisibleGoalPosterior(_HERE / "team" / "out" / "uid134_goal_posterior" / "best.pt", device="cpu")
        agent.posterior = self._posterior
        agent.cfg.mass_planner = True
        agent.cfg.mass_plan_maps = POSTERIOR_MAPS
        agent.cfg.mass_plan_n = (2, 3, 4, 5, 6, 7, 8)
        agent.cfg.mass_plan_n_by_map = {}

    def reset(self) -> None:
        self._posterior.reset()
        self._agent.reset()

    def act(self, observation) -> np.ndarray:
        self._posterior.observe_reset(observation)
        state = np.asarray(observation["state"])
        n = int(state.shape[0]) if state.ndim == 2 else 1
        try:
            action = self._agent.act(observation)
            action = np.asarray(action, dtype=np.float32).reshape(n, ACTION_DIM)
            if not np.all(np.isfinite(action)):
                action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
            return np.clip(action, -1.0, 1.0)
        except Exception:
            return np.zeros((n, ACTION_DIM), dtype=np.float32)
