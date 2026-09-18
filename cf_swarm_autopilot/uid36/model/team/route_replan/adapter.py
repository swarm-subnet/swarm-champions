from __future__ import annotations
import math
import os
from dataclasses import replace
import numpy as np
from . import pad_posterior as pp
from .route_core import RouteConfig, corridor_chain, power_diagram
MAPS = ("mountain", "city", "open", "village", "forest")
P_DET = {"city": 0.934, "open": 0.913, "mountain": 0.938, "village": 0.819, "forest": 0.877}
PAD_DISC_K = 7510.0 / 1.4
V_CRUISE = 2.9
REPLAN_M = 20.0
SEARCH_PHASES = ("SEARCH", "CLIMB")
CLAIM_PHASES = ("APPROACH", "DESCEND")
class ReplanRouter:
    def __init__(self, radial=None, cfg: RouteConfig | None = None, replan_m: float = REPLAN_M):
        self.radial = radial
        self.cfg = cfg or RouteConfig()
        self.replan_m = float(replan_m)
        self.stats = {"plans": 0, "replans": 0, "errors": 0, "landed": 0, "crashed": 0}
        self.reset()
    def reset(self):
        self.ready = False
        self.kind = None
        self.tick = 0
        self.found_sig = None
        self.landed = []
        self._ring_cache = {}
        self.job = None           
    def _setup(self, starts, clue, n, kind):
        self.n = int(n)
        self.S = np.asarray(starts, float)[:n, :2].copy()
        self.clue = np.asarray(clue, float).copy()
        self.kind = kind if kind in pp.WORLD else "city"
        self.gx, self.gy, self.rho0 = pp.pad_field(self.S, self.clue, self.kind,
                                                   step=self.cfg.step, radial=self.radial)
        self.XX, self.YY = np.meshgrid(self.gx, self.gy)
        self.swept = np.zeros(self.rho0.shape, bool)
        self.flew = np.zeros(n, bool)
        self.frozen = np.zeros(n, bool)
        self.stable = np.zeros(n, int)
        self.travel = np.zeros(n)
        self.moved = np.zeros(n)
        self.last = self.S.copy()
        self._ring_cache = {}
        self.ready = True
    def plan(self, starts, clue, n, map_kind):
        self._setup(starts, clue, n, map_kind)
        routes = self._routes(list(range(self.n)), self.S, t_now=0.0, found=[])
        routes = [r if r else [self.S[i] + np.array([10.0, 0.0])] for i, r in enumerate(routes)]
        K = max(len(r) for r in routes)
        out = np.zeros((self.n, K, 2))
        for i, r in enumerate(routes):
            for j in range(K):
                out[i, j] = r[min(j, len(r) - 1)]
        self.stats["plans"] += 1
        return out
    def _residual(self, found):
        key = tuple((round(float(x), 1), round(float(y), 1)) for x, y in found)
        if key not in self._ring_cache:
            self._ring_cache.clear()
            if found:
                self._ring_cache[key] = pp.pad_field_found(
                    self.S, self.clue, self.kind, np.asarray(found, float),
                    step=self.cfg.step, radial=self.radial)[2]
            else:
                self._ring_cache[key] = self.rho0
        r = self._ring_cache[key] * (1.0 - P_DET[self.kind] * self.swept)
        rem = self.n - len(found)
        tot = float(r.sum())
        if rem <= 0 or tot <= 0:
            return None
        return r * (rem / tot)
    def _chain_for(self, i, k, r, owner, P, t_now):
        budget = min(self.cfg.budget_m - self.travel[i], (60.0 - t_now) * V_CRUISE - 20.0)
        budget = max(15.0, budget)
        return corridor_chain(self.gx, self.gy, r, owner, k, P[k], replace(self.cfg, budget_m=budget))
    def _routes(self, active, pos, t_now, found):
        r = self._residual(found)
        out = [[] for _ in range(self.n)]
        if r is None or not active:
            return out
        P = np.asarray(pos, float)[: len(active)] if len(pos) == len(active) else pos[active]
        owner = power_diagram(self.gx, self.gy, r, P, self.cfg, target=float(r.sum()) / len(active))
        for k, i in enumerate(active):
            budget = min(self.cfg.budget_m - self.travel[i], (60.0 - t_now) * V_CRUISE - 20.0)
            budget = max(15.0, budget)
            out[i] = corridor_chain(self.gx, self.gy, r, owner, k, P[k], replace(self.cfg, budget_m=budget))
        return out
    def _view(self, pos, yaw, agl):
        alt = max(float(agl), 1.0)
        dx, dy = self.XX - pos[0], self.YY - pos[1]
        h = np.hypot(dx, dy)
        rr = np.hypot(h, alt)
        rel = (np.arctan2(dy, dx) - yaw + np.pi) % (2 * np.pi) - np.pi
        return ((h >= alt) & (rr <= 20.0) & (np.abs(rel) <= np.radians(45.0))
                & (PAD_DISC_K * (alt / rr) / (rr * rr) >= 5.0))
    def observe(self, agent, obs):
        if not self.ready or getattr(agent, "_route_waypoints", None) is None:
            return
        st = np.asarray(obs["state"], float).reshape(-1, 190)
        if len(st) != self.n:
            return
        self.tick += 1
        t_now = float(getattr(agent, "step_i", self.tick)) * 0.02
        kind = agent._route_kind()
        if kind != self.kind and kind in pp.WORLD:          
            self._setup(self.S, self.clue, self.n, kind)
        pads = [np.asarray(q.xyz, float) for q in getattr(agent, "pads", [])]
        froze = False
        for i in range(self.n):
            if self.frozen[i]:
                continue
            vel, ang, p = st[i, 6:9], st[i, 9:12], st[i, 0:2]
            if np.any(vel != 0.0) or np.any(ang != 0.0):
                self.flew[i] = True
                ok = (abs(vel[2]) <= 0.5 and math.hypot(vel[0], vel[1]) <= 0.6
                      and abs(st[i, 3]) <= 0.26 and abs(st[i, 4]) <= 0.26)
                self.stable[i] = self.stable[i] + 1 if ok else 0
                step = float(np.hypot(*(p - self.last[i])))
                self.travel[i] += step
                self.moved[i] += step
                self.last[i] = p
            elif self.flew[i]:
                self.frozen[i] = True
                froze = True
                upright = abs(st[i, 3]) <= 1.0 and abs(st[i, 4]) <= 1.0
                near = min((math.hypot(q[0] - p[0], q[1] - p[1]) for q in pads), default=1e9)
                if upright and self.stable[i] >= 20 and near <= 1.0:
                    self.landed.append(p.copy())
                    self.stats["landed"] += 1
                else:
                    self.stats["crashed"] += 1
        d = agent.d
        searching = [i for i in range(self.n) if not self.frozen[i] and i < len(d)
                     and getattr(d[i], "phase", None) in SEARCH_PHASES and getattr(d[i], "claim", None) is None]
        if self.tick % 3 == 0:
            for i in searching:
                self.swept |= self._view(st[i, 0:2], float(st[i, 5]), float(st[i, 137]) * 20.0)
        found = [q for q in self.landed]
        for i in range(min(self.n, len(d))):
            c = getattr(d[i], "claim", None)
            if (c is not None and 0 <= c < len(pads) and not self.frozen[i]
                    and getattr(d[i], "phase", None) in CLAIM_PHASES):
                found.append(pads[c][:2])
        dedup = []
        for q in found:
            if all(math.hypot(q[0] - e[0], q[1] - e[1]) > 1.5 for e in dedup):
                dedup.append(np.asarray(q, float))
        sig = tuple(sorted((round(float(q[0]), 0), round(float(q[1]), 0)) for q in dedup))
        moved = max((self.moved[i] for i in searching), default=0.0)
        if searching and (froze or sig != self.found_sig or moved >= self.replan_m):
            self.job = {"stage": 0, "searching": list(searching), "found": dedup, "t_now": t_now,
                        "pos": st[searching, 0:2].copy()}
            self.moved[:] = 0.0
            self.found_sig = sig
            self.stats["replans"] += 1
            return
        self._advance_job(agent)
    def _advance_job(self, agent):
        job = self.job
        if job is None:
            return
        if job["stage"] == 0:                       
            job["r"] = self._residual(job["found"])
            job["stage"] = 1 if job["r"] is not None else 99
        elif job["stage"] == 1:                     
            r = job["r"]
            job["owner"] = power_diagram(self.gx, self.gy, r, job["pos"], self.cfg,
                                         target=float(r.sum()) / len(job["searching"]))
            job["k"] = 0
            job["stage"] = 2
        elif job["stage"] == 2:                     
            k = job["k"]
            i = job["searching"][k]
            d = agent.d
            if not self.frozen[i] and i < len(d) and getattr(d[i], "claim", None) is None:
                chain = self._chain_for(i, k, job["r"], job["owner"], job["pos"], job["t_now"])
                if chain:
                    agent._route_waypoints[i] = [np.asarray(w, float) for w in chain]
                    d[i].route_idx = 0
            job["k"] += 1
            if job["k"] >= len(job["searching"]):
                job["stage"] = 99
        if job["stage"] == 99:
            self.job = None
def install(agent, radial_path=None):
    import team.autopilot.agent as agent_module
    agent_module.LANE_INHERIT = False       
    here = os.path.dirname(os.path.abspath(__file__))
    radial = pp.load_radial_prior(radial_path or os.path.join(here, "pad_radial_prior.json"))
    router = ReplanRouter(radial=radial)
    agent.route_planner = router
    agent.cfg.route_planner = True
    agent.cfg.route_plan_maps = MAPS
    agent.cfg.mass_planner = False
    orig_act, orig_reset = agent.act, agent.reset
    import time as _time
    router.timing = {"agent_ms": [], "router_ms": []}
    def act(observation):
        t0 = _time.perf_counter()
        out = orig_act(observation)
        t1 = _time.perf_counter()
        try:
            router.observe(agent, observation)
        except Exception:  
            router.stats["errors"] += 1
        t2 = _time.perf_counter()
        tm = router.timing
        if len(tm["router_ms"]) < 5000:
            tm["agent_ms"].append((t1 - t0) * 1e3)
            tm["router_ms"].append((t2 - t1) * 1e3)
        return out
    def reset(*a, **k):
        router.reset()
        router.timing = {"agent_ms": [], "router_ms": []}
        return orig_reset(*a, **k)
    agent.act = act
    agent.reset = reset
    agent.replan_router = router
    return router
