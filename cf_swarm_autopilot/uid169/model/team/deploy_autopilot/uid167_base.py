from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from team.deploy_autopilot.uid99_base import (
    FOREST_SHAPE, MASS_CFG, MTN_CFG, ROUTE_CFG, UID20_CFG, VILLAGE_THR,
    WEIGHTS, _ck, _install_forest_veto, _wrap_posterior_feed)
UID167_DELTA = dict(
    commit_t=40.0,                       
    commit_t_skip=('forest', 'village'),  
    mountain_min_hits=2,                 
    spiral_reach=12.0,
    spiral_reach_maps=('village',),
    yaw_track_dev=0.52,
    yaw_track_avoid_only=True,
    approach_vz=2.0,
    approach_vz_maps=('mountain',),
    approach_gain=1.2,
    approach_gain_skip=('mountain',),
)
UID167_CFG = {**UID20_CFG, **UID167_DELTA}
CLF_JSON = ROOT / "team" / "out" / "map_classifier.json"
CLF_MIN_COUNT = 2       
CLF_MIN_PROB = 0.60     
CLF_CITY_PROB = 0.70    
CLF_EVERY = 5           
CLF_MAX_TICK = 90       
def _force_forest(agent):
    cfg = agent.cfg
    if float(getattr(cfg, "mtn_clue_z", 0.0) or 0.0) > 0.0 and getattr(agent, "map_kind", None) == "mountain":
        return      # S8 (optional guard): a seed called mountain from its clue height is never forced to forest
    if bool(getattr(cfg, "fv_force_spread", False)) and not agent._forest_possible():
        # F2b (skeptic:regress MK): a start beyond the forest world (FOREST_VETO_SPREAD_M, the base's vote veto
        # test) or the router's geometric city fix rules forest out, for the XGB force as for the vote
        agent._b5_n["f2b_block"] += 1
        st = getattr(agent, "_own_start", None)
        spread = (float(np.max(np.abs(np.asarray(st, float)[:, :2]))) if st is not None and len(st) else None)
        agent._b5_log.append((round(agent.step_i * 0.02, 2), -1, "F2b", spread and round(spread, 1)))
        return
    agent.is_forest = True
    agent.map_kind = "forest"
    agent.z_band = (cfg.forest_z_lo, cfg.forest_z_hi)
    agent._band_locked = True
    _mhbm = agent._min_hits_for_map()
    if _mhbm is not None:
        agent.min_hits = _mhbm
class ForestVoter:
    def __init__(self, min_count=CLF_MIN_COUNT, min_prob=CLF_MIN_PROB,
                 every=CLF_EVERY, max_tick=CLF_MAX_TICK, path=CLF_JSON,
                 city_prob=None):
        from team.autopilot import map_clf as C
        self._C = C
        self.pred = C._XGBMapPredictor(Path(path))
        self.labels = C.MAP_LABELS
        self.sdim = int(C.m_STATE_DIM)
        self.dt = float(C.SIM_DT)
        self.feats = C._map_depth_feature_values
        self.min_count = int(min_count)
        self.min_prob = float(min_prob)
        self.every = int(every)
        self.max_tick = int(max_tick)
        self.city_prob = None if city_prob is None else float(city_prob)
        self.reset()
    @property
    def enabled(self) -> bool:
        return bool(getattr(self.pred, "enabled", False))
    def reset(self):
        self.psum = np.zeros(len(self.labels))
        self.count = 0
        self.tick = -1
        self.done = False
        self.rr = 0
        self.fired = False
    def step(self, obs, agent):
        if not self.enabled:
            return
        self.tick += 1
        if self.done or self.tick > self.max_tick or self.tick % self.every:
            return
        st = np.asarray(obs["state"], np.float32).reshape(-1, 190)
        dp = np.asarray(obs["depth"], np.float32)
        n = st.shape[0]
        i = self.rr % n
        self.rr += 1
        f = np.asarray([float(self.tick), float(self.tick) * self.dt]
                       + st[i, :self.sdim].tolist() + list(self.feats(dp[i])),
                       dtype=np.float32)
        p = self.pred.predict_proba(f)
        if p is None:
            return
        self.psum += p
        self.count += 1
        if self.count < self.min_count:
            return
        avg = self.psum / self.count
        k = int(np.argmax(avg))
        if (self.labels[k] == "forest" and float(avg[k]) >= self.min_prob
                and getattr(agent, "map_kind", None) != "forest"):
            _force_forest(agent)
            self.done = True
            self.fired = True
        elif (self.city_prob is not None and self.labels[k] == "city"
                and float(avg[k]) >= self.city_prob
                and getattr(agent, "map_kind", None) not in ("village", "mountain")):
            agent.is_city = True
            self.done = True
            self.fired = True
def _install_forest_clf(agent, **kw):
    voter = ForestVoter(**kw)
    agent.forest_voter = voter
    if not voter.enabled:
        raise RuntimeError(
            "map_classifier.json missing or malformed at %s -- UID167's forest "
            "override would load DISABLED and silently do nothing" % CLF_JSON)
    inner_act, inner_reset = agent.act, agent.reset
    def act(observation, _a=inner_act, _v=voter):
        out = _a(observation)
        try:
            _v.step(observation, agent)
        except Exception:                                    
            pass
        return out
    def reset(_r=inner_reset, _v=voter):
        if not getattr(agent, "_in_act", False):
            _v.reset()
        _r()
    agent.act = act
    agent.reset = reset
def config(**overrides):
    from team.autopilot.agent import AutopilotConfig
    cfg = AutopilotConfig()
    for k, v in {**UID167_CFG, **ROUTE_CFG, **MASS_CFG, **MTN_CFG}.items():
        if not hasattr(cfg, k):
            raise AttributeError(
                "UID167 base sets %r which this AutopilotConfig lacks -- the "
                "agent and the base have drifted apart" % k)
        setattr(cfg, k, v)
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise AttributeError("unknown config field %r" % k)
        setattr(cfg, k, v)
    return cfg
class AgreementDetector:
    MATCH_M = 2.0
    def __init__(self, primary, veto):
        self.primary = primary
        self.veto = veto
    def propose_batch(self, poses, rots, depth_batch):
        pa = self.primary.propose_batch(poses, rots, depth_batch)
        pb = self.veto.propose_batch(poses, rots, depth_batch)
        out = []
        for fa, fb in zip(pa, pb):
            vc = [q.centre for q in fb]
            keep = [p for p in fa
                    if any(float(np.hypot(p.centre[0] - c[0], p.centre[1] - c[1]))
                           <= self.MATCH_M for c in vc)]
            out.append(keep)
        return out
class CombinedDetector:
    MATCH_M = 2.0
    VOTE_THR = 0.70
    def __init__(self, ours, champ, mode="intersect"):
        self.ours = ours
        self.champ = champ
        self.mode = str(mode)
    def _near(self, p, centres):
        return any(float(np.hypot(p.centre[0] - c[0], p.centre[1] - c[1]))
                   <= self.MATCH_M for c in centres)
    def propose_batch(self, poses, rots, depth_batch):
        A = self.ours.propose_batch(poses, rots, depth_batch)
        B = self.champ.propose_batch(poses, rots, depth_batch)
        out = []
        for fa, fb in zip(A, B):
            ca = [q.centre for q in fa]
            cb = [q.centre for q in fb]
            if self.mode == "intersect":
                keep = [p for p in fa if self._near(p, cb)]
            elif self.mode == "union":
                keep = list(fa) + [p for p in fb if not self._near(p, ca)]
            elif self.mode == "vote":
                keep = list(fb) + [p for p in fa
                                   if not self._near(p, cb)
                                   and float(getattr(p, "score", 0.0)) >= self.VOTE_THR]
            else:
                keep = list(fa)
            out.append(keep)
        return out
def build_agent(route_ckpt: str = "", posterior: bool = True,
                forest_veto: bool = True, forest_clf: bool = True,
                device: str = "cpu",
                general_thr: float = 0.5, general_ckpt: str = "",
                forest_ckpt: str = "", village_ckpt: str = "",
                mountain_ckpt: str = "", village_thr: float = -1.0,
                city_ckpt: str = "", city_thr: float = 0.5,
                forest_agree: bool = False,
                general_combine: str = "",
                village_combine: str = "",
                posterior_kind: str = "learned",
                geo_K: int = 24000,
                mountain_thr: float = -1.0,
                **overrides):
    import torch
    torch.set_num_threads(int(os.environ.get("SWARM_TORCH_THREADS", "1")))
    from team.autopilot.agent import SwarmAutopilotAgent
    from team.detector.pads import PadConfig, PadNetDetector
    cfg = config(**overrides)
    det = PadNetDetector(_ck(general_ckpt or WEIGHTS["general"]), device=device,
                         thr=float(general_thr))
    if general_combine:
        champ_head = PadNetDetector(_ck(WEIGHTS["general"]), device=device,
                                    thr=float(general_thr))
        det = CombinedDetector(det, champ_head, mode=general_combine)
    fdet = PadNetDetector(_ck(forest_ckpt or WEIGHTS["forest"]), device=device,
                          cfg=PadConfig(**FOREST_SHAPE))
    vdet = PadNetDetector(_ck(village_ckpt or WEIGHTS["village"]), device=device,
                          thr=(VILLAGE_THR if village_thr < 0
                               else float(village_thr)))
    if village_combine:
        champ_v = PadNetDetector(_ck(WEIGHTS["village"]), device=device,
                                 thr=(VILLAGE_THR if village_thr < 0
                                      else float(village_thr)))
        vdet = CombinedDetector(vdet, champ_v, mode=village_combine)
    mdet = PadNetDetector(_ck(mountain_ckpt or WEIGHTS["mountain"]),
                          device=device,
                          **({"thr": float(mountain_thr)} if mountain_thr > 0 else {}))
    # City specialist: dispatched by the map classifier at p >= CLF_CITY_PROB.
    # None unless a ckpt is given, so the pooled general head still serves city
    # (and open) exactly as before when this is empty.
    cdet = (PadNetDetector(_ck(city_ckpt), device=device, thr=float(city_thr))
            if city_ckpt else None)
    if forest_agree:
        veto_head = PadNetDetector(_ck(WEIGHTS["forest"]), device=device,
                                   cfg=PadConfig(**FOREST_SHAPE))
        fdet = AgreementDetector(fdet, veto_head)
    rp = None
    if cfg.route_planner:
        from team.autopilot.route_planner import RoutePlanner
        rp = RoutePlanner(_ck(route_ckpt or WEIGHTS["route"]), device=device)
    agent = SwarmAutopilotAgent(cfg=cfg, detector=det, forest_detector=fdet,
                                village_detector=vdet, mountain_detector=mdet,
                                city_detector=cdet, route_planner=rp)
    if posterior and cfg.mass_planner:
        if posterior_kind == "geo":
            # FORK: exact Monte-Carlo posterior from the generator's geometry, conditioned on
            # the pads already confirmed (team/autopilot/geo_posterior.py).
            from team.autopilot.geo_posterior import GeoPosterior
            agent.posterior = GeoPosterior(K=int(geo_K))
        else:
            from team.autopilot.goal_posterior_runtime import ActorVisibleGoalPosterior
            agent.posterior = ActorVisibleGoalPosterior(
                Path(_ck(WEIGHTS["posterior"])), device=device)
        _wrap_posterior_feed(agent)
    if forest_veto:
        _install_forest_veto(agent)
    if forest_clf:
        _install_forest_clf(agent, **({"city_prob": CLF_CITY_PROB}
                                      if cdet is not None else {}))
    return agent
def assert_is_base(cfg) -> None:
    bad = [(k, getattr(cfg, k, "<absent>"), v)
           for k, v in {**UID167_CFG, **ROUTE_CFG, **MTN_CFG}.items()
           if getattr(cfg, k, None) != v]
    if bad:
        raise AssertionError(
            "config is not the UID167 base; %d field(s) differ, e.g. %s"
            % (len(bad), bad[:3]))
