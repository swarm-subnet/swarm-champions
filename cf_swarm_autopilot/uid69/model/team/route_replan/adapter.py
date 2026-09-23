from __future__ import annotations
import math
import os
from dataclasses import replace
import numpy as np
from . import pad_posterior as pp
from .route_core import RouteConfig, corridor_chain, lawnmower_chain, power_diagram
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
        # Footprint the coverage model credits a drone with. The defaults reproduce
        # the shipped radial-range + pixel-area test; measured values are set from
        # the bundle (the depth buffer clips on the camera *axis*, so the real
        # footprint is a 20 m slab, not a 20 m sphere).
        self.view_axial = 0.0        # >0: use axial (camera-forward) range instead
        self.view_rel_deg = 45.0
        self.view_min_h_frac = 1.0   # ground points nearer than frac*alt are below the FOV
        self.view_px_min = 5.0       # 0 disables the pixel-area test
        self.sweep_r_by_map = {}
        # Mountain's world half-extent is drawn per seed (90-120 m) while the
        # posterior assumes a fixed 110, so the search spreads over ground the map
        # does not have. The starts are uniform in the box, so their extreme is an
        # estimator of it.
        self.p_det_by_map = {}       # per-sweep detection credit, overriding P_DET
        self.route_mode_by_map = {}  # "lawn" to sweep a cell serpentine instead of greedily
        # Which drones' views are written into the swept map. The shipped router only
        # credits *searching* drones, so ground a drone crosses on its way to a claimed
        # pad is forgotten and can be swept again later by somebody else.
        self.sweep_phases = tuple(SEARCH_PHASES)
        # Sweep *quality* memory. The boolean swept map says a cell was looked at, not
        # how well: a pad glimpsed at the far edge of the frustum is missed ~1 time in 6,
        # one 8 m away almost never. Keeping P(cell effectively seen) per cell lets the
        # residual credit fresh ground strongly and still send a drone back, late, to
        # the cells that were only ever seen badly. Bins are axial-range edges (m) and
        # the measured single-view recall inside each.
        self.sweep_quality = False
        self.sweep_quality_maps = ()     # if set, quality memory only on these map kinds
        self.sweep_phases_by_map = {}    # per-map override of sweep_phases
        self.quality_bins = (10.0, 14.0, 17.0, 20.0)
        self.quality_p = (0.98, 0.95, 0.91, 0.82)
        self.quality_scale_by_map = {}
        # Depth-ray sweep: instead of crediting the whole geometric frustum, credit only the
        # cells whose ground the depth rays actually reached (a house or tree in front of
        # the cell leaves it uncredited, so the router can send somebody back there).
        self.sweep_depth_maps = ()
        self.depth_stride = 4          # ray subsampling of the 128x128 frame
        self.depth_low_m = 1.5         # a hit counts as ground if within this of the ground under the drone
        self.depth_min_hits = 2        # rays per cell needed to credit it
        # Maps where any terrain hit credits its cell (no "near the ground under the drone" test):
        # on mountain the ground under the drone says nothing about the ground 15 m away, and the
        # point of the depth sweep there is occlusion -- a valley behind a ridge stays unswept.
        self.depth_any_maps = ()
        # Maps where the depth credit is also limited to the geometric footprint (frustum, 20 m
        # range, pixel-size test): the depth sweep then only removes what terrain hides.
        self.depth_geom_maps = ()
        # Suspect-candidate boost: a pad candidate the detector raised a few times but
        # nobody claimed (too few hits for the claim gate) is strong evidence of a pad the
        # fleet is about to forget. Add P(real | hits) of expected-pad mass around it so a
        # searcher routes back past it and the hits accumulate to a claim.
        self.cand_maps = ()
        self.cand_p0 = 0.15          # P(real) at one hit ...
        self.cand_p1 = 0.10          # ... plus this per extra hit ...
        self.cand_pmax = 0.7         # ... capped here
        self.cand_r = 6.0            # radius (m) of cells the mass is spread over
        self.cand_min_hits = 1
        self.cand_max_age_s = 1e9    # ignore candidates last seen longer ago than this
        self._cands = []
        # Sequential fleet planning: instead of partitioning the residual by a power diagram and
        # planning every drone inside its own cell, plan the drones one after another on the
        # whole residual, each removing what its chain sweeps before the next one plans.
        self.seq_plan = False
        # Restrict sequential planning to these map kinds; () keeps seq_plan as given everywhere.
        self.seq_plan_maps = ()
        self.seq_order = "near"      # "near": drones closest to the mass go first
        # Exact Monte-Carlo generator posterior (team/autopilot/geo_posterior.GeoPosterior) in
        # place of the moment-matched analytic field, for the initial plan and the found-
        # conditioned residual alike. Per map kind; () = off everywhere.
        self.geo_post_maps = ()
        self.geo_K = 4000
        self._geo = None
        # Evidence-based city/open kind for the router. The agent tells city from open by the
        # early tall-structure fraction, which overlaps badly (measured: 35% of city episodes are
        # routed with the open prior, whose 60 m grid gives city pads beyond |xy| 60 zero mass).
        # Geometry is unambiguous: a start or found pad beyond the open world (|xy| > 60) means
        # city; a found pad farther than the city band (45 m) from every start means open.
        self.kind_evidence = False
        self.kind_open_world = 60.0
        self.kind_city_band = 45.0
        self._kind_fixed = None
        # Mixture prior for ambiguous city/open maps: the field is w*city + (1-w)*open on the
        # city grid (W 75), w from the agent's tall-structure fraction, until geometry fixes the
        # kind (kind_evidence). Robust to both misroutes at the price of some diluted mass.
        # Running tall-structure fraction over the episode (measured on seeds 800-839: after 20 s
        # open maps sit at <= 0.075 and city maps at >= 0.05; after 30 s open <= 0.051, city >= 0.057),
        # a much better city/open cue than the agent's first-seconds vote. Staged rules below.
        # Anisotropic empirical prior (per-direction renormalisation, see pad_posterior); per map.
        self.aniso_maps = ()
        self.aniso_r_min = {"city": 22.0, "open": 28.0, "forest": 22.0, "mountain": 65.0, "village": 65.0}
        self.kind_tall_evidence = False
        self.tall_rules = ((10.0, 0.15, 0.055), (20.0, 0.10, 0.045), (30.0, 0.056, 0.052))   # (t_sec, city_if_ge, open_if_le)
        self._tall_n = 0
        self._tall_sum = 0.0
        self.kind_mix = False
        self.mix_w0 = 0.3            # P(city) at tall_frac 0 ...
        self.mix_slope = 1.2         # ... rising with tall_frac ...
        self.mix_wmax = 0.9          # ... capped here
        self._agent = None
        self._mix_w = None
        self.seen_p = None
        self.cov_log = []           # (t, swept cells, n searching, n frozen) for diagnostics
        self.world_est_maps = ()
        self.world_est_lo = 85.0
        self.world_est_hi = 125.0
        self.W = None
        self.stats = {"plans": 0, "replans": 0, "errors": 0, "landed": 0, "crashed": 0}
        self.reset()
    def reset(self):
        self.ready = False
        self.kind = None
        self.tick = 0
        self.found_sig = None
        self.landed = []
        self._ring_cache = {}
        self._cands = []
        self._kind_fixed = None
        self._tall_n = 0
        self._tall_sum = 0.0
        self.job = None           
    def _evidence_kind(self, kind, found=()):
        """City/open kind from geometry when the classifier's kind is one of the two."""
        if not self.kind_evidence or kind not in ("city", "open"):
            return kind
        if self._kind_fixed is not None:
            return self._kind_fixed
        S = np.asarray(self.S, float)
        lim = float(self.kind_open_world) + 1.0
        if S.size and float(np.max(np.abs(S))) > lim:
            self._kind_fixed = "city"
            return "city"
        for q in found:
            q = np.asarray(q, float)[:2]
            if float(np.max(np.abs(q))) > lim:
                self._kind_fixed = "city"
                return "city"
            if S.size and float(np.min(np.hypot(S[:, 0] - q[0], S[:, 1] - q[1]))) > float(self.kind_city_band) + 3.0:
                self._kind_fixed = "open"
                return "open"
        if self.kind_tall_evidence and self._tall_n >= 20:
            t_now = float(getattr(self._agent, "step_i", 0)) * 0.02 if self._agent is not None else 0.0
            frac = self._tall_sum / float(self._tall_n)
            for (t_rule, city_ge, open_le) in self.tall_rules:
                if t_now >= t_rule:
                    if city_ge is not None and frac >= city_ge:
                        self._kind_fixed = "city"
                        return "city"
                    if open_le is not None and frac <= open_le:
                        self._kind_fixed = "open"
                        return "open"
        return kind

    def _setup(self, starts, clue, n, kind):
        self.n = int(n)
        self.S = np.asarray(starts, float)[:n, :2].copy()
        self.clue = np.asarray(clue, float).copy()
        kind = self._evidence_kind(kind)
        self.kind = kind if kind in pp.WORLD else "city"
        self.W = self._world_half_extent()
        self._aniso()
        self._mix_w = None
        if self.kind_mix and self.kind in ("city", "open") and self._kind_fixed is None:
            tf = getattr(self._agent, "_tall_frac", None) if self._agent is not None else None
            tf = float(tf) if tf is not None else (0.5 if self.kind == "city" else 0.1)
            self._mix_w = float(np.clip(self.mix_w0 + self.mix_slope * tf, self.mix_w0, self.mix_wmax))
            self.kind = "city"
            self.W = pp.WORLD["city"]
            self.gx, self.gy, self.rho0 = self._mix_field(())
        else:
            self.gx, self.gy, self.rho0 = pp.pad_field(self.S, self.clue, self.kind,
                                                       step=self.cfg.step, radial=self.radial,
                                                       W=self.W)
        self.XX, self.YY = np.meshgrid(self.gx, self.gy)
        self._geo = None
        if self.kind in tuple(self.geo_post_maps or ()):
            try:
                from team.autopilot.geo_posterior import GeoPosterior
                g = GeoPosterior(K=int(self.geo_K), cell=float(self.cfg.step))
                g._starts = np.column_stack([self.S, np.zeros(len(self.S))]).astype(np.float64)
                g._clue = np.array([self.clue[0], self.clue[1], 0.0], np.float64)
                g._ensure(self.kind)
                self._geo = g
                self.rho0 = self._geo_field(np.zeros((0, 2), np.float32))
            except Exception:
                self._geo = None
        self.swept = np.zeros(self.rho0.shape, bool)
        self.seen_p = np.zeros(self.rho0.shape, float)
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
    def _world_half_extent(self):
        """Half-extent to lay the posterior grid over: the map's fixed value, or an
        estimate from how far apart the seed spread the start pads."""
        if self.kind not in tuple(self.world_est_maps or ()):
            return None
        m = float(np.max(np.abs(self.S))) if len(self.S) else 0.0
        k = (2.0 * self.n + 1.0) / max(2.0 * self.n, 1.0)
        return float(np.clip(m * k, self.world_est_lo, self.world_est_hi))

    def _mix_field(self, found):
        """w*city + (1-w)*open field on the city grid; the open part is masked to its world."""
        W = pp.WORLD["city"]
        Wo = float(self.kind_open_world)
        found = np.asarray(found, float).reshape(-1, 2)
        if len(found):
            gx, gy, rc, _ = pp.pad_field_found(self.S, self.clue, "city", found, step=self.cfg.step, radial=self.radial, W=W)
            _, _, ro, _ = pp.pad_field_found(self.S, self.clue, "open", found, step=self.cfg.step, radial=self.radial, W=W)
        else:
            gx, gy, rc = pp.pad_field(self.S, self.clue, "city", step=self.cfg.step, radial=self.radial, W=W)
            _, _, ro = pp.pad_field(self.S, self.clue, "open", step=self.cfg.step, radial=self.radial, W=W)
        XX, YY = np.meshgrid(gx, gy)
        ro = ro * ((np.abs(XX) <= Wo) & (np.abs(YY) <= Wo))
        rem = max(self.n - len(found), 0)
        for r in (rc, ro):
            t = float(r.sum())
            if t > 0:
                r *= rem / t
        w = float(self._mix_w if self._mix_w is not None else 0.5)
        return gx, gy, w * rc + (1.0 - w) * ro

    def _geo_field(self, found_xy):
        """Expected unfound pads per router cell from the exact posterior (found pads fixed)."""
        g = self._geo
        g._apply_found(np.asarray(found_xy, np.float32).reshape(-1, 2))
        grid = g._grid
        nb = grid.shape[0]
        ix = np.clip(((self.XX + g._bound) / g.cell).astype(int), 0, nb - 1)
        iy = np.clip(((self.YY + g._bound) / g.cell).astype(int), 0, nb - 1)
        rho = grid[iy, ix].astype(float)
        rem = self.n - len(found_xy)
        for q in np.asarray(found_xy, float).reshape(-1, 2):
            rho[np.hypot(self.XX - q[0], self.YY - q[1]) < 1.5 * self.cfg.step] = 0.0
        tot = float(rho.sum())
        if tot > 0 and rem > 0:
            rho *= rem / tot
        return rho

    def _seq_on(self) -> bool:
        """Whether this episode's map uses sequential fleet planning."""
        maps = tuple(getattr(self, "seq_plan_maps", ()) or ())
        return (self.kind in maps) if maps else bool(self.seq_plan)

    def _aniso(self):
        pp.ANISO["on"] = bool(self.kind in tuple(self.aniso_maps or ()))
        pp.ANISO["kind"] = self.kind
        pp.ANISO["r_min"] = dict(self.aniso_r_min)

    def _residual(self, found):
        self._aniso()
        key = tuple((round(float(x), 1), round(float(y), 1)) for x, y in found)
        if key not in self._ring_cache:
            self._ring_cache.clear()
            if self._geo is not None:
                self._ring_cache[key] = self._geo_field(np.asarray(found, float).reshape(-1, 2))
            elif self._mix_w is not None:
                self._ring_cache[key] = self._mix_field(found)[2]
            elif found:
                self._ring_cache[key] = pp.pad_field_found(
                    self.S, self.clue, self.kind, np.asarray(found, float),
                    step=self.cfg.step, radial=self.radial, W=self.W)[2]
            else:
                self._ring_cache[key] = self.rho0
        if (self._quality_on() or self._depth_on()) and self.seen_p is not None:
            r = self._ring_cache[key] * (1.0 - self.seen_p)
        else:
            p_det = float(self.p_det_by_map.get(self.kind, P_DET[self.kind]))
            r = self._ring_cache[key] * (1.0 - p_det * self.swept)
        rem = self.n - len(found)
        tot = float(r.sum())
        if rem <= 0 or tot <= 0:
            return None
        r = r * (rem / tot)
        if self.kind in tuple(self.cand_maps or ()) and self._cands:
            r = r.copy()
            for (cx, cy, hits) in self._cands:
                p = min(self.cand_pmax, self.cand_p0 + self.cand_p1 * (float(hits) - 1.0))
                m = np.hypot(self.XX - cx, self.YY - cy) <= self.cand_r
                k = int(m.sum())
                if k:
                    r[m] += p / k
        return r
    def _chain_fn(self):
        return (lawnmower_chain
                if str(self.route_mode_by_map.get(self.kind, "")) == "lawn"
                else corridor_chain)

    def _sweep_cfg(self):
        """Route config for the current map, with its own sweep radius when given."""
        r = self.sweep_r_by_map.get(self.kind)
        return self.cfg if r is None else replace(self.cfg, sweep_r=float(r))

    def _chain_for(self, i, k, r, owner, P, t_now):
        budget = min(self.cfg.budget_m - self.travel[i], (60.0 - t_now) * V_CRUISE - 20.0)
        budget = max(15.0, budget)
        return self._chain_fn()(self.gx, self.gy, r, owner, k, P[k],
                                replace(self._sweep_cfg(), budget_m=budget, t_now_s=float(t_now)))
    def _routes_seq(self, active, P, r, t_now):
        """Sequential greedy fleet plan (see seq_plan)."""
        out = [[] for _ in range(self.n)]
        r = np.asarray(r, float).copy()
        XX, YY = self.XX, self.YY
        mass_dist = []
        for k, i in enumerate(active):
            w = r / max(float(r.sum()), 1e-9)
            mx, my = float((XX * w).sum()), float((YY * w).sum())
            mass_dist.append(math.hypot(P[k][0] - mx, P[k][1] - my))
        order = np.argsort(mass_dist) if self.seq_order == "near" else np.arange(len(active))
        everyone = np.zeros(r.shape, int)
        for k in order:
            i = active[k]
            budget = min(self.cfg.budget_m - self.travel[i], (60.0 - t_now) * V_CRUISE - 20.0)
            budget = max(15.0, budget)
            chain = self._chain_fn()(self.gx, self.gy, r, everyone, 0, P[k],
                                     replace(self._sweep_cfg(), budget_m=budget, t_now_s=float(t_now)))
            out[i] = chain
            cur = np.asarray(P[k], float)
            sr = float(self._sweep_cfg().sweep_r)
            for w in chain:
                w = np.asarray(w, float)
                ab = w - cur
                L2 = max(float(ab @ ab), 1e-9)
                t = np.clip(((XX - cur[0]) * ab[0] + (YY - cur[1]) * ab[1]) / L2, 0.0, 1.0)
                dist = np.hypot(XX - cur[0] - t * ab[0], YY - cur[1] - t * ab[1])
                r[dist <= sr] = 0.0
                cur = w
        return out

    def _routes(self, active, pos, t_now, found):
        r = self._residual(found)
        out = [[] for _ in range(self.n)]
        if r is None or not active:
            return out
        P = np.asarray(pos, float)[: len(active)] if len(pos) == len(active) else pos[active]
        if self._seq_on():
            return self._routes_seq(active, P, r, t_now)
        owner = power_diagram(self.gx, self.gy, r, P, self.cfg, target=float(r.sum()) / len(active))
        for k, i in enumerate(active):
            budget = min(self.cfg.budget_m - self.travel[i], (60.0 - t_now) * V_CRUISE - 20.0)
            budget = max(15.0, budget)
            out[i] = self._chain_fn()(self.gx, self.gy, r, owner, k, P[k],
                                      replace(self._sweep_cfg(), budget_m=budget, t_now_s=float(t_now)))
        return out
    def _view(self, pos, yaw, agl):
        alt = max(float(agl), 1.0)
        dx, dy = self.XX - pos[0], self.YY - pos[1]
        h = np.hypot(dx, dy)
        rr = np.hypot(h, alt)
        rel = (np.arctan2(dy, dx) - yaw + np.pi) % (2 * np.pi) - np.pi
        ok = (h >= self.view_min_h_frac * alt) & (np.abs(rel) <= np.radians(self.view_rel_deg))
        if self.view_axial > 0.0:
            ok &= (h * np.cos(rel) <= self.view_axial)
        else:
            ok &= (rr <= 20.0)
        if self.view_px_min > 0.0:
            ok &= (PAD_DISC_K * (alt / rr) / (rr * rr) >= self.view_px_min)
        return ok
    def _quality_on(self) -> bool:
        if not self.sweep_quality:
            return False
        maps = tuple(self.sweep_quality_maps or ())
        return (not maps) or (self.kind in maps)

    def _phases_now(self):
        return tuple(self.sweep_phases_by_map.get(self.kind, self.sweep_phases))

    def _view_p(self, pos, yaw, agl):
        """Per-cell probability this view would have caught a pad there."""
        alt = max(float(agl), 1.0)
        dx, dy = self.XX - pos[0], self.YY - pos[1]
        h = np.hypot(dx, dy)
        rel = (np.arctan2(dy, dx) - yaw + np.pi) % (2 * np.pi) - np.pi
        axial = h * np.cos(rel)
        inside = ((h >= self.view_min_h_frac * alt) & (np.abs(rel) <= np.radians(self.view_rel_deg))
                  & (axial <= self.quality_bins[-1]) & (axial > 0.0))
        pv = np.zeros(h.shape, float)
        lo = 0.0
        for hi, pk in zip(self.quality_bins, self.quality_p):
            pv[inside & (axial >= lo) & (axial < hi)] = pk
            lo = hi
        return pv * float(self.quality_scale_by_map.get(self.kind, 1.0))

    def _depth_on(self) -> bool:
        return self.kind in tuple(self.sweep_depth_maps or ())

    def _view_depth(self, pos, rpy, depth, agl):
        """Per-cell P(seen) from the depth frame: cells with >= depth_min_hits rays ending
        on low (ground-level) points, weighted by the range-recall bins of _view_p."""
        from team.detector.geom import _rays
        from team.detector.localize import CAM_FWD, CAM_UP, rot_from_rpy
        d = np.asarray(depth, np.float32)
        if d.ndim == 3:
            d = d[..., 0]
        st = int(self.depth_stride)
        d = d[::st, ::st]
        rng = d * 19.5 + 0.5
        valid = (rng < 19.75) & (rng > 0.6)
        pv = np.zeros(self.XX.shape, float)
        if not valid.any():
            return pv
        R = rot_from_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        body = _rays(128, 90.0, st)
        dirs = body @ R.T
        cam = np.asarray(pos, float) + CAM_FWD * R[:, 0] + CAM_UP * R[:, 2]
        euclid = rng / np.maximum(body[..., 0], 1e-6)
        pts = cam[None, None, :] + euclid[..., None] * dirs
        ground_z = float(pos[2]) - max(float(agl), 0.0)
        if self.kind in tuple(self.depth_any_maps or ()):
            low = valid
        else:
            low = valid & (pts[..., 2] <= ground_z + float(self.depth_low_m))
        if not low.any():
            return pv
        x, y = pts[..., 0][low], pts[..., 1][low]
        axial = rng[low]
        step = float(self.gx[1] - self.gx[0]) if len(self.gx) > 1 else 4.0
        ix = np.rint((x - self.gx[0]) / step).astype(int)
        iy = np.rint((y - self.gy[0]) / step).astype(int)
        ok = (ix >= 0) & (ix < len(self.gx)) & (iy >= 0) & (iy < len(self.gy))
        if not ok.any():
            return pv
        flat = iy[ok] * len(self.gx) + ix[ok]
        cnt = np.bincount(flat, minlength=pv.size)
        amin = np.full(pv.size, np.inf)
        np.minimum.at(amin, flat, axial[ok])
        cells = np.flatnonzero(cnt >= int(self.depth_min_hits))
        if cells.size == 0:
            return pv
        a = amin[cells]
        q = np.zeros(cells.size)
        lo = 0.0
        for hi, pk in zip(self.quality_bins, self.quality_p):
            q[(a >= lo) & (a < hi)] = pk
            lo = hi
        q[a >= lo] = self.quality_p[-1]
        pv.reshape(-1)[cells] = q * float(self.quality_scale_by_map.get(self.kind, 1.0))
        return pv

    def observe(self, agent, obs):
        if not self.ready or getattr(agent, "_route_waypoints", None) is None:
            return
        st = np.asarray(obs["state"], float).reshape(-1, 190)
        if len(st) != self.n:
            return
        self.tick += 1
        t_now = float(getattr(agent, "step_i", self.tick)) * 0.02
        kind = agent._route_kind()
        if kind in pp.WORLD:
            ek = self._evidence_kind(kind)
            target = ("city" if (self.kind_mix and ek in ("city", "open") and self._kind_fixed is None) else ek)
            if target != self.kind or (self.kind_mix and ek in ("city", "open") and self._kind_fixed is None and self._mix_w is None):
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
            viewers = [i for i in range(self.n) if not self.frozen[i] and i < len(d)
                       and getattr(d[i], "phase", None) in self._phases_now()]
            if self.kind_tall_evidence and self.kind in ("city", "open") and self._kind_fixed is None and obs.get("depth") is not None:
                try:
                    dep = np.asarray(obs["depth"])
                    for i in range(self.n):
                        if not self.frozen[i] and i < len(d) and getattr(d[i], "phase", None) in ("SEARCH", "CLIMB", "APPROACH"):
                            self._tall_sum += float(agent._tall_in_view(dep, i))
                            self._tall_n += 1
                except Exception:
                    pass
            for i in viewers:
                if self._depth_on() and obs.get("depth") is not None:
                    try:
                        pv = self._view_depth(st[i, 0:3], st[i, 3:6], np.asarray(obs["depth"])[i],
                                              float(st[i, 137]) * 20.0)
                        if self.kind in tuple(self.depth_geom_maps or ()):
                            pv = pv * self._view(st[i, 0:2], float(st[i, 5]), float(st[i, 137]) * 20.0)
                    except Exception:
                        pv = self._view_p(st[i, 0:2], float(st[i, 5]), float(st[i, 137]) * 20.0)
                    self.seen_p = 1.0 - (1.0 - self.seen_p) * (1.0 - pv)
                    self.swept |= pv > 0.0
                elif self._quality_on():
                    pv = self._view_p(st[i, 0:2], float(st[i, 5]), float(st[i, 137]) * 20.0)
                    self.seen_p = 1.0 - (1.0 - self.seen_p) * (1.0 - pv)
                    self.swept |= pv > 0.0
                else:
                    self.swept |= self._view(st[i, 0:2], float(st[i, 5]), float(st[i, 137]) * 20.0)
            if self.tick % 30 == 0:
                self.cov_log.append((t_now, int(self.swept.sum()), len(searching),
                                     int(self.frozen.sum())))
        found = [q for q in self.landed]
        for i in range(min(self.n, len(d))):
            c = getattr(d[i], "claim", None)
            if (c is not None and 0 <= c < len(pads) and not self.frozen[i]
                    and getattr(d[i], "phase", None) in CLAIM_PHASES):
                found.append(pads[c][:2])
        if self.kind in tuple(self.cand_maps or ()):
            cands = []
            for q in getattr(agent, "pads", []):
                if getattr(q, "by", None) is not None or getattr(q, "done", False):
                    continue
                h = int(getattr(q, "hits", 0))
                if h < int(self.cand_min_hits):
                    continue
                ls = int(getattr(q, "last_seen", -1))
                if ls >= 0 and (t_now - ls * 0.02) > float(self.cand_max_age_s):
                    continue
                xy = np.asarray(q.xyz, float)[:2]
                if any(math.hypot(xy[0] - e[0], xy[1] - e[1]) < 3.0 for e in found):
                    continue
                cands.append((float(xy[0]), float(xy[1]), h))
            self._cands = cands
        dedup = []
        for q in found:
            if all(math.hypot(q[0] - e[0], q[1] - e[1]) > 1.5 for e in dedup):
                dedup.append(np.asarray(q, float))
        if self.kind_evidence and self.kind in ("city", "open") and self._kind_fixed is None:
            ek = self._evidence_kind(self.kind, dedup)
            if ek != self.kind:
                self._setup(self.S, self.clue, self.n, ek)
                self.found_sig = None
        sig = tuple(sorted((round(float(q[0]), 0), round(float(q[1]), 0)) for q in dedup))
        if self.kind in tuple(self.cand_maps or ()):
            sig = (sig, tuple(sorted((round(c[0], 0), round(c[1], 0)) for c in self._cands)))
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
            if self._seq_on():
                routes = self._routes_seq(job["searching"], job["pos"], r, job["t_now"])
                d = agent.d
                for i in job["searching"]:
                    if not self.frozen[i] and i < len(d) and getattr(d[i], "claim", None) is None and routes[i]:
                        agent._route_waypoints[i] = [np.asarray(w, float) for w in routes[i]]
                        d[i].route_idx = 0
                job["stage"] = 99
            else:
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
    router._agent = agent
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
