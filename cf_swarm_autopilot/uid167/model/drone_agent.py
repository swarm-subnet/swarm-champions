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
    commit_t=40.0,
    landed_done=True,
    investigate=True,
    forest_sink_speed=2.0,
    descend_enter_r=1.5,
    descend_enter_wide=('forest',),
    investigate_hits=1,
    spiral_reach=12.0,
    spiral_reach_maps=('village',),
    investigate_skip=('village', 'mountain'),
    descend_commit=True,
    avoid_range_mtn=20.0,
    yaw_scan_deg=50.0,
    yaw_track_dev=0.52,
    yaw_track_avoid_only=True,
    mountain_min_hits=2,
    mountain_detector_swap=True,
    yaw_scan_maps=('mountain', 'city'),
    approach_vz=2.0,
    approach_vz_maps=('mountain',),
    approach_gain=1.2,
    approach_gain_skip=('mountain',),
)

WEIGHTS = {

    "general": ("team/out/padnet6_sb/best.pt", "team/out/padnet6_sb"),
    "forest": ("team/out/padforest_net/best.pt", "team/out/padforest_net"),

    "village": ("team/out/padvillage_net/best.pt", "team/out/padvillage_net"),

    "mountain": ("team/out/padmtn_net/best.pt", "team/out/padmtn_net"),

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
                cfg=PadConfig(net_max_aspect=2.6, net_min_short=0.20, min_pixels=3),
            )
        vp = _find(WEIGHTS["village"])
        vdet = PadNetDetector(vp, device=device, thr=0.80) if vp is not None else None

        mp = _find(WEIGHTS["mountain"])
        mdet = PadNetDetector(mp, device=device) if mp is not None else vdet
        agent = SwarmAutopilotAgent(cfg=AutopilotConfig(**SHIPPED_CFG),
                                    detector=det, forest_detector=fdet,
                                    village_detector=vdet,
                                    mountain_detector=mdet)
        agent.cfg.terrain_ff = 1.0
        agent.cfg.terrain_ff_maps = ("mountain",)
        agent.cfg.mountain_climb_cap = 3.0
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


# ---------------------------------------------------------------------------
# Forest-classifier override grafted onto rotor-nav-suite's controller.
# The base class below is rotor's DroneFlightController, so the five rotor
# levers (gated commit_t, approach_vz, approach_gain, spiral_reach, yaw_track)
# are all still live; this only adds the map-classifier forest override on top.
# map_classifier.json IS included in this artifact -- without it the predictor
# loads disabled and the whole override silently does nothing.
# ---------------------------------------------------------------------------
# drone_agent_20 (SHIPPABLE) = UID99 + forest-only trained-classifier override.
# Self-contained: the classifier code is vendored as _apclf.py next to this file and its
# weights as map_classifier.json in the artifact, so nothing outside the zip is referenced.
import sys as _s, os as _os
from pathlib import Path as _P
_HERE = _P(__file__).resolve().parent
if str(_HERE) not in _s.path:
    _s.path.insert(0, str(_HERE))
import numpy as _np

try:
    import _apclf as _C
    _PRED = _C._XGBMapPredictor(_HERE / "map_classifier.json")
    _LAB, _SDIM, _DT = _C.MAP_LABELS, int(_C.m_STATE_DIM), float(_C.SIM_DT)
    _FEATS = _C._map_depth_feature_values
except Exception:                      # never let the classifier break the controller
    _PRED = None

def _force_forest(agent):
    cfg = agent.cfg
    agent.is_forest = True
    agent.map_kind = "forest"
    agent.z_band = (cfg.forest_z_lo, cfg.forest_z_hi)
    agent._band_locked = True

class _ForestVoter:
    """Accumulate the classifier's verdict from ONE drone per cycle (the map is a property of
    the seed, not the drone) and force 'forest' only when it is confident and the agent disagrees."""
    def __init__(self, min_count=2, min_prob=0.60, every=5, max_tick=90):
        self.min_count=min_count; self.min_prob=min_prob; self.every=every; self.max_tick=max_tick
        self.reset()
    def reset(self):
        self.psum = _np.zeros(len(_LAB)) if _PRED is not None else None
        self.count=0; self.tick=-1; self.done=False; self.rr=0
    def step(self, obs, agent):
        if _PRED is None or not _PRED.enabled: return
        self.tick += 1
        if self.done or self.tick > self.max_tick or self.tick % self.every: return
        st = _np.asarray(obs["state"], _np.float32).reshape(-1, 190)
        dp = _np.asarray(obs["depth"], _np.float32)
        n = st.shape[0]; i = self.rr % n; self.rr += 1
        f = _np.asarray([float(self.tick), float(self.tick)*_DT] + st[i, :_SDIM].tolist()
                        + list(_FEATS(dp[i])), dtype=_np.float32)
        p = _PRED.predict_proba(f)
        if p is None: return
        self.psum += p; self.count += 1
        if self.count < self.min_count: return
        avg = self.psum / self.count; k = int(_np.argmax(avg))
        if _LAB[k] == "forest" and float(avg[k]) >= self.min_prob \
           and getattr(agent, "map_kind", None) != "forest":
            _force_forest(agent); self.done = True

_B20 = DroneFlightController
class _Cand20Ship(_B20):
    def __init__(self):
        super().__init__()
        self._fv = _ForestVoter()
    def reset(self):
        self._fv.reset()
        return super().reset()
    def act(self, observation):
        out = super().act(observation)
        try:
            self._fv.step(observation, self._agent)
        except Exception:
            pass
        return out
DroneFlightController = _Cand20Ship
