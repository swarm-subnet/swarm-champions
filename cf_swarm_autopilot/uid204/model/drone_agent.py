# Swarm Subnet 124 — cf_swarm_autopilot submission agent.
#
# Ships our current stack through the SAME entry our panels use
# (team.deploy_autopilot.base.build_agent -> uid167_base), so the flown agent is
# bit-identical to what we benchmarked. Do NOT hand-construct the agent here: the
# old UID134 wrapper this replaces did, and drifted from what we actually tested.
#
# THE CONFIG (submission 3, 2026-09-10). Every number is a paired panel against
# the UID167 champion, full config, one field varied per arm:
#   general_ckpt      = padgeneral_hn2_net       (open + city detector, ours)
#   village_ckpt      = padvillage_hn_net        (village detector, ours)
#   village_thr       = 0.65    +0.0024 +/- 0.0017  n=150   (was 0.70)
#   village_min_hits  = 4       +0.0438 vs the mh8 default on its own block
#   min_hits_to_claim = 3       base bar: pre-classification window + FOREST
#   min_hits_by_map   = ('city', 2)   open AND city -> 2
#                                     city +0.0210 +/- 0.0076 (z=2.8, n=70,
#                                     three blocks); open +0.0041 +/- 0.0039
#   open follows to 2                 its bar is flat 2..4; measured +0.0041
#                             NB map_kind labels open "city", so this ONE entry
#                             serves both maps; an 'open' key is never read.
#   commit_t_by_map   = open/25, city/25   city 20 measured +0.0047 +/- 0.0038 but
#                             open reads the CITY key too and open@20 is untested,
#                             so both stay at 25 -- the value every panel measured.
#   everything else   = UID167 champion defaults (route u235, posterior, map
#                       classifier, forest/mountain heads, MTN_CFG). The forest
#                       VETO is ON: build_agent's forest_veto defaults True and we
#                       do not pass it -- same as every panel, so they agree. (An
#                       earlier header here claimed "no forest veto"; that was wrong.)
#   Forest is untouched (falls through to mh3) and mountain is byte-identical to
#   the champion, verified 30/30.
#
# Measured EW over the five maps, ship vs champion, 150 seeds/map on fresh blocks:
#   village +0.0251  open +0.0147  city +0.0127  forest +0.0209 (validator-real)
#   mountain 0. UID108 scored 0.6353 = EW +0.0104 on chain; this -> ~+0.0154.
#
# Checkpoints resolve through uid167_base._ck, anchored to the BUNDLE ROOT, so the
# six WEIGHTS dirs (padgeneral_hn2_net, padvillage_hn_net, padforest_net,
# padmtn_net, u235_route, uid134_goal_posterior) and map_classifier.json must be
# packaged at <bundle>/team/out/.
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

# cf_swarm_autopilot action = [dir_x, dir_y, dir_z, speed, yaw] (5-dim). NOT 6 —
# the 6th field (rgb_request) is the Search-and-Rescue contract. The agent's act()
# already returns (n, 5); this must match or verify fails invalid_action_shape.
ACTION_DIM = 5


class DroneFlightController:
    def __init__(self):
        import torch
        # One thread: the validator runs several agents on one box; these nets are
        # small enough that contention costs more than parallelism buys.
        torch.set_num_threads(int(os.environ.get("SWARM_TORCH_THREADS", "1")))

        from team.deploy_autopilot.base import build_agent

        self._agent = build_agent(
            general_ckpt="team/out/padgeneral_hn2_net",
            village_ckpt="team/out/padvillage_hn_net",
            # 0.65 beats 0.70 on village: +0.0024 +/- 0.0017 (z=1.4, 150 paired
            # seeds, block 64000000). vt75 +0.0017, vmh5 -0.0002. Adopted on
            # positive expected value -- it is ~1.4 sigma, not confirmed.
            village_thr=0.65,
            village_min_hits=4,
            # Base bar for the pre-classification window and for forest (whose
            # +0.0209 came from mh3-from-reset). The claim: "open's commits happen
            # post-classification, so a post-classification mh5 restores the win
            # without touching any other map" was WRONG and is retracted -- the
            # post-classification override keys on map_kind, which calls open
            # "city". See the block below.
            min_hits_to_claim=3,
            # CLAIM BAR, corrected 2026-09-09.
            #
            # `map_kind` labels every OPEN world "city" (city_tall_frac=0.015 vs an
            # open tall-fraction of ~0.1375; true city is ~0.5232), so the "open"
            # key in min_hits_by_map is NEVER READ -- on the open map the agent
            # looks up "city". The previous ("open",5,"city",5) worked only by
            # coincidence: both values were 5, so open got 5 via the city key.
            # Measured directly: an open panel varying only the "open" key came
            # back bit-identical across 5/6/7.
            #
            # BOTH maps measured 4, so the shared key is not a compromise -- it is
            # the optimum for each, and no gate is needed at all (nothing to fail):
            #   city  4 vs 5  +0.0084 +/- 0.0035  z=2.4  n=100  (67 landers faster,
            #                 5 rescued, 1 broken; mh3 ties at +0.0086 but breaks 3)
            #   open  4 vs 5  +0.0037 +/- 0.0029  P(>0)=90%  n=80  W50/L21
            #                 (82 landers faster, 24 slower, 0 rescued, 1 broken)
            #   open  6 vs 5  -0.0011  W30/L40 -- the bar wants to go DOWN, not up.
            # Both gains are TIME-term: the claim is what starts the descent
            # profile (`investigate` only steers laterally in SEARCH), so a lower
            # bar starts the sink earlier on drones that were landing anyway.
            #
            # forest is untouched: map_kind="forest" finds no entry here and keeps
            # min_hits_to_claim=3, which is where its +0.0209 came from.
            # village/mountain lock in _set_z_band before this and keep their own
            # fields (village_min_hits=4, mountain_min_hits=2).
            # SUBMISSION 4: claim bar 2 on BOTH city and open (the shared key).
            # city 2 vs 4: +0.0210 +/- 0.0076 (z=2.8) pooled over 70 paired seeds
            #   on three blocks: 72M +0.0233(20), 73M +0.0271(30), 74M +0.0096(20).
            #   Corroborated independently: the speculative-approach panel gave
            #   spec1-stale = +0.0253 for the same early-claim effect. It is a
            #   TIME win -- 207 drones landed a mean 0.53 s sooner at claim=1.
            # open 2 vs 4: +0.0041 +/- 0.0039, W24/L5, 29/30 seeds moved (so the
            #   arm is live, not the bit-identical trap). Open's bar is flat from
            #   2 to 4 -- 5->4 +0.0037, 4->3 -0.0018, 4->2 +0.0041, all inside
            #   each other's noise -- so the shared key costs nothing here.
            # city_min_hits (the tall-fraction gate) was the alternative and is
            # NOT used: it fires on only 70% of city seeds (6 of 20 sit at or
            # below city_min_hits_frac=0.25), delivering ~+0.0133 EW against
            # +0.0154 for the shared key. The backlog's "15/16 separation" figure
            # is older and measured something else; 20 seeds contradict it.
            # forest keeps min_hits_to_claim=3; village/mountain lock earlier.
            min_hits_by_map=("city", 2),
            # 25 on both. NOT 20, and the reason is the mislabel again:
            # commit_t_by_map keys on map_kind, which calls open "city", so OPEN
            # READS THE CITY ENTRY. Setting city=20 silently sets open=20 too.
            #   city 20 vs 25  +0.0047 +/- 0.0038 (n=150) -- real but ~1.2 sigma
            #   open 20        NEVER MEASURED. The oct20 arm (open/20, city/25)
            #                  was a no-op: +0.0001 +/- 0.0001, i.e. bit-identical,
            #                  because the "open" key is never read.
            # An unconfirmed +0.0047 on city is not worth an untested change to
            # open in a submission. Both entries stay at 25, which is the value
            # every ov_* panel measured. Keep BOTH keys: dropping a map from
            # commit_t_by_map DISABLES the deadline there rather than falling
            # back to commit_t.
            commit_t_by_map=("open", 25.0, "city", 25.0),
            device=os.environ.get("SWARM_DEVICE", "cpu"),
        )
        self._agent.reset()

    def reset(self) -> None:
        self._agent.reset()

    def act(self, observation) -> np.ndarray:
        state = np.asarray(observation["state"])
        n = int(state.shape[0]) if state.ndim == 2 else 1
        try:
            action = self._agent.act(observation)
            action = np.asarray(action, dtype=np.float32).reshape(n, ACTION_DIM)
            if not np.all(np.isfinite(action)):
                action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
            return np.clip(action, -1.0, 1.0)
        except Exception:  # noqa: BLE001
            # A raised exception zeroes the whole episode; a hover is merely bad.
            return np.zeros((n, ACTION_DIM), dtype=np.float32)
