"""THE BASE WE NOW DEVELOP ON: UID167, constructed in ONE place.

Adopted 2026-09-05. UID167 (`sus210/swarm-auto-v1`) crowned at 0.6248, bar
0.6392. Their delta over UID99 is **entirely code and config** -- all six
checkpoints are byte-identical to UID99's, so they trained nothing:

  city 0.6908 -> 0.7072   open 0.7114 -> 0.7267   mountain 0.3648 -> 0.3896
  village 0.6249 -> 0.6186   forest 0.6624 -> 0.6848      equal-weight +0.0145

Two things carry that:

1. **Five motion/timing levers** they credit to "rotor-nav-suite" -- gated
   commit_t, approach_vz, approach_gain, spiral_reach, yaw_track. Each is a
   handful of lines in `agent.py` and a config field; none needs a weight.
2. **A trained XGBoost map classifier**, vendored from the cf_autopilot
   champion UID14, used ONLY to force `map_kind="forest"` when the agent's own
   depth vote disagrees. That is the forest +0.0224 and it is the same class of
   bug we found in `city_tall_frac` -- they fixed theirs with a learned model
   where we only measured ours.

WHAT IS STILL OURS. Their bundle has no hard-negative detector heads, no
`rewedge`, no per-map commit table. Our +0.0104 was measured against UID99 and
has to be re-measured here, but nothing in it is duplicated by them.

    from team.deploy_autopilot.uid167_base import build_agent, assert_is_base
    agent = build_agent()                       # the base, exactly
    agent = build_agent(commit_t=25.0)          # one field changed, nothing else
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from team.deploy_autopilot.uid99_base import (  # noqa: E402
    FOREST_SHAPE, MASS_CFG, MTN_CFG, ROUTE_CFG, UID20_CFG, VILLAGE_THR,
    WEIGHTS, _ck, _install_forest_veto, _wrap_posterior_feed)

# ---------------------------------------------------------------------------
# UID167's SHIPPED_CFG, parsed out of their drone_agent.py. Only the fields
# that DIFFER from UID99 are listed; the rest is inherited verbatim.
#
# commit_t_skip is here because UID167 moved the AutopilotConfig DEFAULT to
# ("forest","village") rather than setting it in SHIPPED_CFG. We left our
# dataclass default at () so that uid99_base keeps reproducing UID99 exactly,
# and pin the value here instead. Same effective config, no blast radius.
# ---------------------------------------------------------------------------
UID167_DELTA = dict(
    commit_t=40.0,                       # was 50.0
    commit_t_skip=('forest', 'village'),  # was () -- DISABLES the deadline there
    mountain_min_hits=2,                 # was 3
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

# ---------------------------------------------------------------------------
# The forest classifier.
# ---------------------------------------------------------------------------
CLF_JSON = ROOT / "team" / "out" / "map_classifier.json"
CLF_MIN_COUNT = 2       # votes before it may act
CLF_MIN_PROB = 0.60     # mean probability of "forest" required
CLF_CITY_PROB = 0.70    # mean probability of "city" required to dispatch a city head
                        # (only active when build_agent is given a city_ckpt)
CLF_EVERY = 5           # vote every 5th act()
CLF_MAX_TICK = 90       # give up after 1.8 s of sim


def _force_forest(agent):
    cfg = agent.cfg
    agent.is_forest = True
    agent.map_kind = "forest"
    agent.z_band = (cfg.forest_z_lo, cfg.forest_z_hi)
    agent._band_locked = True
    # The per-map claim override keys on map_kind and is normally applied at the
    # END of _vote_forest -- which this bypasses, because locking the band makes
    # _vote_forest return early forever. Without this line the "forest" entry in
    # min_hits_by_map is UNREADABLE: the classifier fires around step 10 (
    # CLF_MIN_COUNT=2 x CLF_EVERY=5) while the geometric vote needs ~step 72, so
    # it wins on essentially every forest seed. Caught 2026-09-10 when three
    # forest arms came back bit-identical.
    # No-op unless min_hits_by_map names "forest", so the champion base and every
    # config shipped to date are untouched.
    _mhbm = agent._min_hits_for_map()
    if _mhbm is not None:
        agent.min_hits = _mhbm


class ForestVoter:
    """UID167's classifier override, reimplemented field-for-field.

    The map is a property of the SEED, not of a drone, so one drone per cycle
    votes (round-robin) and the probabilities are averaged. It only ever forces
    forest -- it never argues a forest classification back down, which is what
    keeps it from fighting the geometric veto.
    """

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
            # Dispatch the city-specialist head. Only reached when build_agent was
            # given a city_ckpt (city_prob stays None otherwise), so the base and
            # every prior panel are byte-identical.
            agent.is_city = True
            self.done = True
            self.fired = True


def _install_forest_clf(agent, **kw):
    """Wrap act/reset the way their _Cand20Ship does.

    ORDER MATTERS: their controller delegates to the inner act FIRST and votes
    afterwards, so the classifier can never change the action of the step it
    votes on -- only the next one. The wrapper goes on the OUTSIDE of the
    posterior wrapper for the same reason.
    """
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
        except Exception:                                    # noqa: BLE001
            pass
        return out

    def reset(_r=inner_reset, _v=voter):
        # Same re-entrancy rule as the posterior: the agent calls self.reset()
        # from inside its own act() when n changes, and self.reset IS this
        # wrapper. Only the harness's reset may clear the vote.
        if not getattr(agent, "_in_act", False):
            _v.reset()
        _r()

    agent.act = act
    agent.reset = reset


def config(**overrides):
    """UID167's AutopilotConfig, plus route/posterior wiring, plus overrides."""
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
    """Two-head agreement veto (Path B, forest). Keep a proposal only when the
    veto head ALSO proposes a pad within MATCH_M metres; the primary head's
    proposal (position + score) is passed through unchanged, the veto head only
    gates it. Measured (probe_agreement.py): the champion forest head confirms
    98% of our head's real pads and rejects 43% of its phantoms, so intersecting
    keeps the recall and drops the uncorrelated phantoms. Only the primary head
    drives; this wraps its `propose_batch`, so the agent is otherwise untouched.
    """

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
    """Ensemble OUR general head with the CHAMPION's, per drone, to bank the
    'best of both worlds' across seeds where each head wins. Only the primary
    (`ours`) head's config drives the agent; this wraps `propose_batch`.

      intersect : keep OURS' proposals confirmed by CHAMP within MATCH_M
                  -> precision: drops the uncorrelated phantoms one head invents.
      union     : proposals from BOTH heads, deduped within MATCH_M
                  -> recall: finds platforms either head sees (faster/less coverage-gap).
      vote      : CHAMP's proposals (trusted) + OURS' unique finds where CHAMP is
                  silent AND ours.score >= VOTE_THR
                  -> guarded recall: champ's precision base, our high-confidence extras.
    """

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
                **overrides):
    """The UID167 agent, complete. Overrides touch the config and NOTHING else.

    The detector hooks are ours, not theirs: UID167 ships UID99's four heads
    unchanged, so `general_ckpt` / `village_ckpt` / `forest_ckpt` are how our
    hard-negative retrains reach a panel arm. Empty keeps the base exactly.
    """
    import torch
    torch.set_num_threads(int(os.environ.get("SWARM_TORCH_THREADS", "1")))

    from team.autopilot.agent import SwarmAutopilotAgent
    from team.detector.pads import PadConfig, PadNetDetector

    cfg = config(**overrides)

    det = PadNetDetector(_ck(general_ckpt or WEIGHTS["general"]), device=device,
                         thr=float(general_thr))
    # Ensemble OUR general head with the CHAMPION's (padnet6_sb) to capture the
    # per-seed 'best of both': our head wins some seeds, champ's others. Modes:
    #   intersect -> keep OURS only where CHAMP agrees (precision; drop phantoms)
    #   union     -> proposals from BOTH heads (recall; find more platforms fast)
    #   vote      -> CHAMP's proposals + OURS' unique high-score finds (guarded recall)
    # No-op unless general_combine is set AND general_ckpt differs from the champ.
    if general_combine:
        champ_head = PadNetDetector(_ck(WEIGHTS["general"]), device=device,
                                    thr=float(general_thr))
        det = CombinedDetector(det, champ_head, mode=general_combine)
    fdet = PadNetDetector(_ck(forest_ckpt or WEIGHTS["forest"]), device=device,
                          cfg=PadConfig(**FOREST_SHAPE))
    vdet = PadNetDetector(_ck(village_ckpt or WEIGHTS["village"]), device=device,
                          thr=(VILLAGE_THR if village_thr < 0
                               else float(village_thr)))
    # Same best-of-both ensemble on the VILLAGE head: our padvillage_hn vs the
    # champion's padvillage_u134 differ per seed. No-op unless village_combine set.
    if village_combine:
        champ_v = PadNetDetector(_ck(WEIGHTS["village"]), device=device,
                                 thr=(VILLAGE_THR if village_thr < 0
                                      else float(village_thr)))
        vdet = CombinedDetector(vdet, champ_v, mode=village_combine)
    mdet = PadNetDetector(_ck(mountain_ckpt or WEIGHTS["mountain"]),
                          device=device)
    # City specialist: dispatched by the map classifier at p >= CLF_CITY_PROB.
    # None unless a ckpt is given, so the pooled general head still serves city
    # (and open) exactly as before when this is empty.
    cdet = (PadNetDetector(_ck(city_ckpt), device=device, thr=float(city_thr))
            if city_ckpt else None)
    # Two-head agreement veto on forest (Path B): our forest head keeps a
    # detection only where the CHAMPION forest head agrees. No-op unless
    # forest_agree, so the base and every prior panel stay byte-identical.
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
    """Fail loudly if a probe is about to fly something other than UID167."""
    bad = [(k, getattr(cfg, k, "<absent>"), v)
           for k, v in {**UID167_CFG, **ROUTE_CFG, **MTN_CFG}.items()
           if getattr(cfg, k, None) != v]
    if bad:
        raise AssertionError(
            "config is not the UID167 base; %d field(s) differ, e.g. %s"
            % (len(bad), bad[:3]))
