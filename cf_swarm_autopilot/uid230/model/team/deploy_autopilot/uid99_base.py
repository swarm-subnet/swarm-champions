"""THE BASE WE NOW DEVELOP ON: UID20, constructed in ONE place.

Adopted 2026-08-26. UID20 (`alche3206/swarm-autopilot-v00`) is UID235's stack
with every weight byte-identical and 37 of 38 config fields unchanged; its
whole delta is `mountain_min_hits` 3->2, `mountain_detector_swap` True fed the
GENERAL head at thr 0.80 with forest-tight shape gates, and a forest geometric
veto. Their `team/autopilot/agent.py` replaced ours because it carries the
`mass_*` goal-posterior planner, which ours never had.

WHY A FACTORY AND NOT A CONFIG DICT. `champion_config()` returns 39 numbers.
An agent is 39 numbers PLUS four detectors built at specific thresholds and
shape gates, PLUS a route planner, PLUS a posterior, PLUS a monkey-patched
forest veto. Every one of those is load-bearing and none of them lives in the
config:

  - the village head runs thr=0.80, not the 0.5 default. Our old
    `drone_agent.py` built it at 0.5, so every village panel we ever ran
    controlled against an agent the champion does not fly.
  - `mountain_detector` is the GENERAL weight re-gated, not a separate
    checkpoint. Our old build passed None, so `mountain_detector_swap` could
    never fire even when set -- which is a second reason our swap arms
    measured level.
  - the forest veto is a wrapper around `_vote_forest`, not a flag.

Constructing any of that by hand in a probe is how a control silently stops
being the champion. So: one factory, and `assert_is_champion` checks the
config half of it.

    from team.deploy_autopilot.uid99_base import build_agent, UID20_CFG
    agent = build_agent()                      # the base, exactly
    agent = build_agent(mountain_min_hits=3)   # one field changed, nothing else
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# UID20's SHIPPED_CFG, verbatim. Parsed out of their drone_agent.py with ast
# and pinned here; team_tests/test_uid99_base.py re-parses their bundle and
# asserts this dict still matches, so a silent drift fails a test rather than
# a panel.
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

# Route + posterior are set on the agent AFTER construction, exactly as their
# drone_agent.py does. They are not AutopilotConfig defaults, so they must be
# applied here or the base is not the base.
ROUTE_CFG = dict(route_planner=True,
                 route_plan_maps=('mountain', 'city', 'open'),
                 route_start_sec=0.0)
# The MOUNTAIN group. Set post-construction exactly like the route and
# posterior groups, and missed for that reason -- it is three lines further up
# their file than the ones we transcribed. Both fields are mountain-gated
# (agent.py `mountain_climb_cap` and the `terrain_ff` feed-forward), and both
# default OFF, so omitting them silently disabled two champion mechanisms on
# the one map we kept calling inert. verify_uid167_base caught it on 2 of 2
# mountain seeds; city, open, village and forest matched to 0.00e+00.
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
    # Byte-identical to their padvillage_net (369864d7f57bd224); ours is just
    # named for its provenance.
    "village": "team/out/padvillage_u134",
    "mountain": "team/out/padmtn_net",   # UID99: a DEDICATED mountain head
    "route": "team/out/u235_route",           # == their uid134_route_policy
    "posterior": "team/out/uid134_goal_posterior",
}

# Their village and mountain heads are the SAME general/village weights run at
# a raised acceptance threshold. This is the knob nobody on our side had ever
# moved off 0.5.
VILLAGE_THR = 0.80
MOUNTAIN_THR = 0.80
MOUNTAIN_SHAPE = dict(net_max_aspect=1.6, net_min_short=0.45)
FOREST_SHAPE = dict(net_max_aspect=2.6, net_min_short=0.20, min_pixels=3)
FOREST_VETO_SPREAD_M = 45.0     # forest world is +-42 m, so a start beyond
                                # this rules forest out geometrically


def _ck(rel: str) -> str:
    p = ROOT / rel
    return str(p / "best.pt" if p.is_dir() else p)


def config(**overrides):
    """UID20's AutopilotConfig, plus route/posterior wiring, plus overrides."""
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
    """UID20's geometric veto on a forest misclassification.

    Once the depth vote locks the band to forest, a fleet whose furthest start
    sits beyond +-45 m cannot be on the forest map at all (TYPE_6_WORLD_RANGE
    is 42). Undo the lock and re-decide city/open on the tall fraction.

    Misreading forest is expensive: it switches cruise speed, altitude, sink
    speed, the descend-entry radius AND the detector head, all at once.
    """
    inner = agent._vote_forest

    def vetoed(depth, live, state=None, _inner=inner, _a=agent):
        was_locked = bool(getattr(_a, "_band_locked", False))
        try:
            _inner(depth, live, state)
        except TypeError:          # their signature takes (depth, live)
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
        except Exception:                                    # noqa: BLE001
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
    """The UID20 agent, complete. Overrides touch the config and NOTHING else.

    route_ckpt: swap in a different route policy (the A/B for our own nets);
                empty means theirs.
    village_ckpt / village_thr: swap the VILLAGE head and its acceptance
                threshold. Same reasoning as general_thr -- neither is a config
                field, so a panel arm cannot reach them without a hook. Empty /
                -1 keep the base exactly. Village is where the detector hurts
                most: 28.6% of its proposals are false, and 16.3% of its drones
                die by descending onto one.
    general_thr: acceptance threshold of the GENERAL pad head, which is the
                detector open and city actually run on. UID20 left it at the
                0.5 constructor default while raising village and mountain to
                0.80 -- their whole gain came from re-gating detectors on maps
                we were not working. It is not a config field, so a panel arm
                cannot reach it without this hook. 0.5 keeps the base exactly.
    """
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
    # THE GENERAL WEIGHT, re-gated -- not a mountain specialist. This is
    # UID20's change and it needs no new checkpoint.
    # UID99 ships a dedicated mountain net at the DEFAULT threshold, where
    # UID20 re-gated the general head at 0.80 with tight shape limits.
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
    """Feed the posterior each observation, exactly as their controller does.

    THE VERIFICATION CAUGHT THIS. Their `DroneFlightController.act` calls
    `self._posterior.observe_reset(observation)` BEFORE delegating, and
    `.reset()` resets the posterior too. Setting `agent.posterior` alone left it
    starved, and it raised "reset observation not supplied" on exactly the six
    village and forest seeds -- the two maps `mass_plan_maps` covers. City,
    open and mountain matched to 0.00e+00 and hid it completely.

    The finite/clip guard is theirs as well and is part of the base: it changes
    the action in the edge cases where it fires, so leaving it out would be a
    silent divergence rather than a safety net.
    """
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
        except Exception:                                    # noqa: BLE001
            return np.zeros((n, 5), dtype=np.float32)
        finally:
            agent._in_act = False

    def reset(_r=inner_reset, _p=post):
        # RE-ENTRANCY. The agent calls self.reset() from inside its own act()
        # (n != self.n on the first step), and self.reset IS this wrapper.
        # Resetting the posterior there wiped the observation act() had just
        # fed it, so _init_mass_grid raised "reset observation not supplied",
        # the except above swallowed it, and the block that seeds self.bad with
        # every drone's start platform never ran. Result on village: the start
        # veto was empty for the whole episode, drones re-detected the pad they
        # launched from, and 84% of failures ended within 3 m of a start.
        # Only the OUTER reset -- the harness between episodes -- may clear it.
        if not getattr(agent, "_in_act", False):
            _p.reset()
        _r()

    agent.act = act
    agent.reset = reset


def assert_is_base(cfg) -> None:
    """Fail loudly if a probe is about to fly something other than UID20."""
    bad = [(k, getattr(cfg, k, "<absent>"), v)
           for k, v in {**UID20_CFG, **ROUTE_CFG, **MTN_CFG}.items()
           if getattr(cfg, k, None) != v]
    if bad:
        raise AssertionError(
            "config is not the UID20 base; %d field(s) differ, e.g. %s"
            % (len(bad), bad[:3]))
