from __future__ import annotations
import math
import os
from dataclasses import replace
import numpy as np
from . import pad_posterior as pp
from . import world_post as wpost
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
        self.world_soft_maps = ()
        self.world_soft_lo = 90.0
        self.world_soft_hi = 120.0
        self.world_soft_margin = 4.0
        self.world_core_mix = False
        # candHv VRAD (default None = off): empirical 5.1.6.3 village start->pad distance prior for the village field
        # (pad_posterior.VRAD). {"file": radial json in team/route_replan (its "village" entry is used), "w": weight
        # of the empirical component (1.0 = the pdf alone; < 1 = mixture with the BAND_MIX village bands)}.
        self.village_radial = None
        self._vrad_cache = None
        self.W = None
        # WPOST-X (research/ideas/REPORT.md 4.1): mountain W components from the agent's world-size
        # posterior (agent._wpost, set only with world_post_router); None = off, the shipped field.
        self._Wq = None
        self.wpost_slack = 3.9       # alive(): a found pad beyond W + this kills component W
        # Compute a component only on ticks where the agent did not run its detector (the CEM's rule):
        # a component field stacked on a detector tick is the act-time maximum (impl6 shadow test:
        # 380 vs 242 ms on the same flight). False = one component on every tick (literal REPORT 4.1).
        self.wpost_skip_det = True
        # coverage2 OEND (open endgame residual sweep; research/coverage2/README.md). Traced open timeouts: once most pads
        # are claimed, the last searcher's chain budget min(150 - travel, (60 - t) * 2.9 - 20) no longer reaches the
        # unexplored side of the map, so its chains shrink to one leg over swept ground and then come back EMPTY, and the
        # drone spirals round the clue over swept cells until the timeout (seed 2777874131:16: 28 new cells in the last
        # 28 s, the missed pad 4 m from the wall never in any footprint). While the endgame gate holds (router kind in
        # oend_maps, t >= oend_t0, at least oend_found_frac * n pads landed or claimed):
        #   - the chain budget is (oend_t_end - t) * V_CRUISE + oend_see_m (no whole-flight travel cap): a cell only
        #     has to come within ~oend_see_m ahead of the camera before the last useful find time;
        #   - an empty chain is replaced by one leg toward the best far pocket of residual mass inside the drone's own
        #     partition (box-filtered mass within oend_far_r / (distance + oend_far_d0)), at least oend_sep from the
        #     pockets teammates took in the same replan (oend_far);
        #   - a searcher whose route is used up triggers a replan at once instead of spiralling (oend_replan_done);
        #   - with oend_rsv the searchers plan one after another, best placed first, each discounting the ground the
        #     earlier ones' chains will view (matched sighting cone, see crsv_*);
        #   - oend_fwc: endgame chains are credited with the cells a leg really views (route_core FWC: cross-track <=
        #     oend_fwc_r, along-track >= oend_fwc_k * cross-track and <= leg + oend_fwc_end) instead of the symmetric
        #     sweep_r capsule, legs shorter than oend_min_leg excluded. The capsule (20 m on open) credits cells beside
        #     a leg's start and end that the forward camera never views; they stay unviewed, so every later replan
        #     sends a searcher back past them (kinematic proxy: the fleet's never-viewed share of the grid only falls
        #     0.46 -> 0.40 from 25 s to 55 s, i.e. endgame legs re-sweep viewed ground);
        #   - oend_keep > 0 (hysteresis): a searcher keeps the rest of its current route unless the new chain's value
        #     (residual mass inside the matched footprint per metre) beats it by the factor oend_keep. Replans come
        #     every 20 m moved and the greedy chain restarts from scratch each time, so a lone searcher flip-flops
        #     between pockets (proxy seed 2777874131: a chain east toward the missed pad at 15 s, replaced by one
        #     west at 20 s; the drone never left the west half);
        #   - oend_seen_w < 1 scales the endgame residual of every already-viewed cell: the swept credit assumes one
        #     view finds a pad 82-98% of the time, so a viewed cell near the clue keeps 2-18% of a prior that can be
        #     20x the edge's and the greedy chain keeps re-sweeping it (kinematic proxy: the last searcher circles the
        #     clue-implied spot); open ground is flat and unoccluded, one view there is close to conclusive;
        #   - oend_floor > 0 mixes a uniform floor over never-viewed cells into the endgame residual (template-robust
        #     weight on the generator tail at the world edge, where the never-found open pads sit: sw7o OSR20, 190
        #     never-claimed pads, median 4 m from the wall, 47% outside the prior's 90% HPD region).
        self.oend_maps = ()
        self.oend_t0 = 25.0
        self.oend_found_frac = 0.5
        self.oend_t_end = 54.0
        self.oend_see_m = 15.0
        self.oend_min_budget = 15.0
        self.oend_far = True
        self.oend_far_r = 20.0
        self.oend_far_d0 = 10.0
        self.oend_sep = 25.0
        self.oend_replan_done = True
        self.oend_gap_ticks = 25
        self.oend_rsv = False
        self.oend_fwc = False
        self.oend_fwc_r = 20.0
        self.oend_fwc_k = 1.0
        self.oend_fwc_end = 12.0
        self.oend_min_leg = 12.0
        self.oend_keep = 0.0
        self.oend_seen_w = 1.0
        self.oend_floor = 0.0
        # coverage2 CRSV (city overlap-aware sweep planning). City analysis (research/city_nfa_b): 81-86% of the time a
        # pad waits to be seen no detector-live drone is in range, while cross-drone overlap is large (a searching
        # drone's new ground is fleet-new 78% of the time at n=2 but 39% at n=8; approach legs view as much fleet-new
        # band ground as search legs, 29% of it already seen). The shipped plan credits each chain with a symmetric
        # capsule inside its own power cell and never looks at what teammates are about to view. With crsv_maps the
        # router keeps, per replan, a reservation grid of the cells teammates will view soon - the straight approach
        # leg of every APPROACH drone to its claimed pad (crsv_app) and the chains already planned in this replan -
        # using a footprint matched to the real sighting geometry: a cell is viewed by a leg a->b if its cross-track
        # offset y <= crsv_r, its along-track s >= crsv_k * y (the +-45 deg frustum) and s <= L + crsv_end (20 m axial
        # depth range beyond the leg end). Each chain is planned on residual * (1 - crsv_w * reserved), so searchers
        # stop planning legs over ground a teammate is already committed to view soon: with crsv_rsv_m > 0 only the
        # first crsv_rsv_m metres of a teammate's chain are reserved (chains are replanned every 20 m, so the far part
        # of a plan is no commitment, and a cell a teammate reaches 40 s later is not covered for the fleet) (the
        # chain's own credit is unchanged:
        # crediting it with the 20 m cone makes the mass-per-metre greedy chain pick 4 m hops, below waypoint_reach).
        self.crsv_maps = ()
        # reservation weight: the offline planning proxy (research/coverage2/a2_crsv_scan.out, 240 city fleets) puts
        # the early-coverage optimum at 0.5-0.7; at 1.0 reserved cells drop out of the chain's node set and the plan
        # covers less prior mass early (AUC -0.009)
        self.crsv_w = 0.5
        self.crsv_app = True
        self.crsv_app_w = 1.0
        self.crsv_app_end = 8.0
        self.crsv_r = 20.0
        self.crsv_k = 1.0
        self.crsv_end = 20.0
        # dynamic kinematic proxy (research/coverage2/sim_city_150_*): whole-chain reservation -0.005 to -0.009 per
        # city seed, first 40 m only -0.003 (se 0.005): the far part of a teammate's plan is no commitment
        self.crsv_rsv_m = 40.0
        self.crsv_order = "index"     # "index" (searching order) or "gain" (best-placed searcher plans first)
        self._oend_armed = False
        self._oend_last_tick = -10 ** 9
        self.stats = {"plans": 0, "replans": 0, "errors": 0, "landed": 0, "crashed": 0}
        self.reset()
    def reset(self):
        self.ready = False
        self._Wq = None
        self.stats.pop("w_post", None)
        self.kind = None
        self.tick = 0
        self.found_sig = None
        self.landed = []
        self._ring_cache = {}
        self._cands = []
        self._kind_fixed = None
        self._tall_n = 0
        self._tall_sum = 0.0
        self._oend_armed = False
        self._oend_last_tick = -10 ** 9
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
        # bughunt VGP: generator-exact village prior when the agent cfg lists the map (default off)
        pp.GEN_PRIOR.clear()
        _c = getattr(self._agent, "cfg", None) if self._agent is not None else None
        for _k in tuple(getattr(_c, "vgp_maps", ()) or ()):
            pp.GEN_PRIOR[str(_k)] = (float(getattr(_c, "vgp_a0", 21.0)), float(getattr(_c, "vgp_u", 0.2)),
                                     float(getattr(_c, "vgp_amax", 0.0)))
        self._vrad_install()
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
        elif self._Wq is not None:
            self.gx, self.gy, self.rho0 = self._wpost_field(())
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
        # candH merge: the UID 149 soft world weight only where WPOST-X is not active (self._Wq None; see
        # _world_half_extent). Our keys off -> self._Wq is always None -> UID 149's line.
        self._world_soft = (self._soft_world_weight()
                            if (self._Wq is None and self.kind in tuple(self.world_soft_maps or ())) else None)
        if self._world_soft is not None:
            self.rho0 *= self._world_soft
            mass = float(self.rho0.sum())
            if mass > 0:
                self.rho0 *= self.n / mass
            self.rho0 = self._blend_world_core(self.rho0, self.n)
        self.swept = np.zeros(self.rho0.shape, bool)
        self.seen_p = np.zeros(self.rho0.shape, float)
        self.dvr_mask = np.zeros(self.rho0.shape, np.int64)      # flat_search3 DVR sector bits per cell
        self.obv_seen = np.zeros(self.rho0.shape, bool)          # flat_search3 OBV: ground reached by a depth ray
        self.obv_block = np.zeros(self.rho0.shape, np.int64)     # flat_search3 OBV: sectors the ground was blocked from
        self.flew = np.zeros(n, bool)
        self.frozen = np.zeros(n, bool)
        self.stable = np.zeros(n, int)
        self.travel = np.zeros(n)
        self.moved = np.zeros(n)
        self.last = self.S.copy()
        self._ring_cache = {}
        self.ready = True
    def _vrad_install(self):
        """candHv VRAD: put the village_radial prior into pad_posterior.VRAD (off when village_radial is unset or its
        file does not load; then the village field is UID 149's band mixture, bit-identical)."""
        spec = self.village_radial
        if not spec:
            pp.VRAD["on"] = False
            return
        try:
            spec = dict(spec) if isinstance(spec, dict) else {"file": str(spec)}
            fn = str(spec["file"])
            if self._vrad_cache is None or self._vrad_cache[0] != fn:
                path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fn)
                edges, pdf = pp.load_radial_prior(path)["village"]
                if len(edges) != len(pdf) + 1 or not np.all(np.isfinite(pdf)) or float(np.sum(pdf)) <= 0:
                    raise ValueError("bad village pdf")
                self._vrad_cache = (fn, edges, pdf / float(np.sum(pdf)))
            pp.VRAD["edges"], pp.VRAD["pdf"] = self._vrad_cache[1], self._vrad_cache[2]
            pp.VRAD["w"] = float(spec.get("w", 1.0))
            pp.VRAD["on"] = True
        except Exception:
            pp.VRAD["on"] = False

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
        self._Wq = None
        wq = self._wpost_components()
        if wq is not None:                  # WPOST-X: the grid spans the largest component
            self._Wq = tuple(float(w) for w in wq)
            W = float(max(self._Wq))
            prev = self.stats.get("w_post") or {}
            self.stats["w_post"] = {"W_grid": W, "Wq": list(self._Wq), "setups": int(prev.get("setups", 0)) + 1,
                                    "comp_ticks": int(prev.get("comp_ticks", 0)),
                                    "found_fields": int(prev.get("found_fields", 0)),
                                    "drop_events": int(prev.get("drop_events", 0)),
                                    "dropped_max": int(prev.get("dropped_max", 0)), "alive_last": list(self._Wq)}
            self._wpost_diag()
            return W
        # UID 149 soft world size (candH merge): the grid spans the largest legal half-width plus the
        # placement margin. WPOST-X above takes precedence where it is active (router branch: mountain,
        # world_post_router, n >= world_post_min_n); _setup then leaves the soft weight off, so the two
        # world-size models never stack. With world_post_router off this is UID 149's code.
        if self.kind in tuple(self.world_soft_maps or ()):
            return float(self.world_soft_hi + self.world_soft_margin)
        if self.kind not in tuple(self.world_est_maps or ()):
            return None
        m = float(np.max(np.abs(self.S))) if len(self.S) else 0.0
        k = (2.0 * self.n + 1.0) / max(2.0 * self.n, 1.0)
        return float(np.clip(m * k, self.world_est_lo, self.world_est_hi))

    # ---------------------------------------------------------------- WPOST-X --
    def _wpost_components(self):
        """The agent's W components for this episode, when the router branch is on (mountain only)."""
        ag = self._agent
        if ag is None or self.kind != "mountain":
            return None
        wp = getattr(ag, "_wpost", None)
        if not wp or not wp.get("router"):
            return None
        wq = wp.get("Wq")
        return list(wq) if wq else None

    def _wpost_diag(self):
        st = self.stats.get("w_post")
        b6 = getattr(self._agent, "_b6_n", None) if self._agent is not None else None
        if st is None or b6 is None:
            return
        b6["wp_grid_W"] = st["W_grid"]
        b6["wp_router_Wq"] = list(st["Wq"])
        b6["wp_setups"] = st["setups"]
        b6["wp_comp_ticks"] = st["comp_ticks"]
        b6["wp_found_fields"] = st["found_fields"]
        b6["wp_drop_events"] = st["drop_events"]
        b6["wp_dropped_max"] = st["dropped_max"]
        b6["wp_alive_last"] = list(st["alive_last"])

    def _wpost_alive(self, found):
        kept, dropped = wpost.alive(self._Wq, found, self.wpost_slack)
        st = self.stats.get("w_post")
        if st is not None and len(found):
            st["found_fields"] += 1
            st["alive_last"] = list(kept)
            if dropped:
                st["drop_events"] += 1
                st["dropped_max"] = max(st["dropped_max"], int(dropped))
            self._wpost_diag()
        return kept

    def _wpost_one(self, Wc, found):
        """One component's field on the common +-self.W grid (found pads conditioned when given)."""
        if len(found):
            return pp.pad_field_found(self.S, self.clue, self.kind, np.asarray(found, float),
                                      step=self.cfg.step, radial=self.radial, W=float(Wc), Wgrid=self.W)
        return pp.pad_field(self.S, self.clue, self.kind, step=self.cfg.step, radial=self.radial,
                            W=float(Wc), Wgrid=self.W)

    def _wpost_field(self, found):
        """(gx, gy, mean over the live components of their fields), all components at once."""
        comps = self._wpost_alive(found) if len(found) else list(self._Wq)
        gx = gy = acc = None
        for Wc in comps:
            out = self._wpost_one(Wc, found)
            gx, gy = out[0], out[1]
            acc = out[2].copy() if acc is None else acc + out[2]
        return gx, gy, acc / float(len(comps))

    def _wpost_stage(self, job, agent=None) -> bool:
        """Job stage 0 under WPOST-X: one component's found-conditioned field per tick, so a
        tick costs what today's single field does. True while components remain; once the
        last one is in, the mean goes into the ring cache and _residual picks it up."""
        found = job["found"]
        if (self._Wq is None or not len(found) or self._geo is not None or self._mix_w is not None):
            return False
        key = self._found_key(found)
        if key in self._ring_cache:
            return False
        tag = (self.W, self._Wq, key)
        if job.get("wp_tag") != tag:
            job["wp_tag"] = tag
            job["wp_alive"] = self._wpost_alive(found)
            job["wp_i"] = 0
            job["wp_acc"] = None
        alive = job["wp_alive"]
        i = int(job["wp_i"])
        if i < len(alive):
            if self._wpost_det_tick(agent):
                return True                         # wait for a tick without the detector
            self._aniso()
            f = self._wpost_one(alive[i], found)[2]
            job["wp_acc"] = f.copy() if job["wp_acc"] is None else job["wp_acc"] + f
            job["wp_i"] = i + 1
            st = self.stats.get("w_post")
            if st is not None:
                st["comp_ticks"] += 1
                self._wpost_diag()
            if job["wp_i"] < len(alive):
                return True
        self._ring_cache.clear()
        self._ring_cache[key] = job["wp_acc"] / float(len(alive))
        return False

    def _wpost_det_tick(self, agent) -> bool:
        """This act ran the agent's detector (step_i % detect_every == 0) and a component is still due."""
        if not self.wpost_skip_det or agent is None:
            return False
        try:
            de = max(1, int(agent.cfg.detect_every))
            return int(agent.step_i) % de == 0
        except Exception:
            return False

    @staticmethod
    def _found_key(found):
        return tuple((round(float(x), 1), round(float(y), 1)) for x, y in found)

    def _soft_world_weight(self):
        """P(world half-width includes cell | uniformly drawn start coordinates).

        With 2n observed start coordinates and a uniform prior on W, the
        likelihood is proportional to W**(-2n) for W >= max(abs(starts)).
        Keep the full legal support; a point estimate would remove real pads.
        """
        lo = max(float(self.world_soft_lo), float(np.max(np.abs(self.S))))
        hi = float(self.world_soft_hi)
        if lo >= hi:
            return np.ones(self.XX.shape, float)
        # The platform placer may move a drawn goal by up to four metres.
        q = np.maximum(np.abs(self.XX), np.abs(self.YY)) - float(self.world_soft_margin)
        a = np.maximum(q, lo)
        power = 1.0 - 2.0 * float(self.n)
        numerator = np.maximum(np.power(a, power) - hi ** power, 0.0)
        denominator = lo ** power - hi ** power
        return np.clip(numerator / denominator, 0.0, 1.0)

    def _blend_world_core(self, field, remaining):
        """Reserve most search mass in the champion's box while retaining legal edge support.

        The wide-map weight is the posterior probability W > 110 m from the
        observed starts, floored for possible pads outside the old box. A
        bounded mixture limits the route changes from broadening the grid.
        """
        if not self.world_core_mix or self._world_soft is None:
            return field
        lo = max(float(self.world_soft_lo), float(np.max(np.abs(self.S))))
        hi = float(self.world_soft_hi)
        edge = 110.0
        if lo >= edge:
            probability = 1.0
        elif lo >= hi:
            probability = 0.0
        else:
            power = 1.0 - 2.0 * float(self.n)
            probability = (edge ** power - hi ** power) / (lo ** power - hi ** power)
        wide_share = float(np.clip(probability, 0.25, 0.85))
        core = field.copy()
        core[(np.abs(self.XX) > edge) | (np.abs(self.YY) > edge)] = 0.0
        core_mass = float(core.sum())
        if core_mass <= 0:
            return field
        core *= float(remaining) / core_mass
        return wide_share * field + (1.0 - wide_share) * core

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
            elif found and self._Wq is not None:
                self._ring_cache[key] = self._wpost_field(found)[2]
            elif found:
                self._ring_cache[key] = pp.pad_field_found(
                    self.S, self.clue, self.kind, np.asarray(found, float),
                    step=self.cfg.step, radial=self.radial, W=self.W)[2]
            else:
                self._ring_cache[key] = self.rho0
            if found and self._world_soft is not None:
                self._ring_cache[key] *= self._world_soft
                mass = float(self._ring_cache[key].sum())
                if mass > 0:
                    self._ring_cache[key] *= max(self.n - len(found), 0) / mass
                self._ring_cache[key] = self._blend_world_core(
                    self._ring_cache[key], max(self.n - len(found), 0))
        if (self._quality_on() or self._depth_on()) and self.seen_p is not None and not self._occl_bool():
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

    def _turn_kw(self, hdg):
        """yaw_time RT: RouteConfig overrides for the turn-aware chain (agent cfg route_turn_*)."""
        c = getattr(self._agent, "cfg", None) if self._agent is not None else None
        if c is None or not tuple(getattr(c, "route_turn_maps", ()) or ()):
            return {}
        if self.kind not in tuple(c.route_turn_maps):
            return {}
        return dict(turn_k=float(c.route_turn_k), turn_free_deg=float(c.route_turn_free_deg),
                    heading=(None if hdg is None else float(hdg)))

    def _chain_for(self, i, k, r, owner, P, t_now, budget=None, extra=None):
        if budget is None:
            budget = min(self.cfg.budget_m - self.travel[i], (60.0 - t_now) * V_CRUISE - 20.0)
            budget = max(15.0, budget)
        _h = getattr(self, "_job_hdg", None)
        _tkw = self._turn_kw(None if _h is None or k >= len(_h) else _h[k])
        _tkw.update(self._dvr_kw(t_now, len(P)))
        _tkw.update(self._fwc_kw())
        if extra:
            _tkw.update(extra)                  # coverage2: endgame / reservation chain overrides
        return self._chain_fn()(self.gx, self.gy, r, owner, k, P[k],
                                replace(self._sweep_cfg(), budget_m=budget, t_now_s=float(t_now), **_tkw))

    # ------------------------------------------------------------- coverage2 --
    def _cv2_count(self, key, inc=1):
        """Diagnostic counter in the agent's _b6_n (dumped by tools/fly.py FLY_B5_LOG as b6)."""
        try:
            b6 = getattr(self._agent, "_b6_n", None)
            if b6 is not None:
                b6[key] = b6.get(key, 0) + inc
        except Exception:                                   # noqa: BLE001
            pass

    def _crsv_on(self) -> bool:
        return bool(self.crsv_maps) and self.kind in tuple(self.crsv_maps)

    def _oend_gate(self, t_now, n_found, n_search) -> bool:
        """OEND endgame gate: router kind listed, late enough, most pads landed or claimed, somebody still searching."""
        if not self.oend_maps or self.kind not in tuple(self.oend_maps):
            return False
        if n_search <= 0 or float(t_now) < float(self.oend_t0):
            return False
        return int(n_found) >= int(math.ceil(float(self.oend_found_frac) * float(self.n) - 1e-9))

    def _rsv_cone(self, a, pts):
        """CRSV reservation footprint of a planned chain: the matched cone of its first crsv_rsv_m metres (all when 0)."""
        m = float(self.crsv_rsv_m)
        if m > 0.0:
            out, cur, left = [], np.asarray(a, float)[:2], m
            for w in pts:
                w = np.asarray(w, float)[:2]
                L = float(np.hypot(*(w - cur)))
                if L >= left:
                    out.append(cur + (w - cur) * (left / max(L, 1e-9)))
                    break
                out.append(w)
                left -= L
                cur = w
            pts = out
        return self._cone(a, pts, float(self.crsv_r), float(self.crsv_k), float(self.crsv_end)).astype(float)

    def _current_rest(self, agent, i):
        """Remaining waypoints of drone i's current route ([] when used up)."""
        rw = getattr(agent, "_route_waypoints", None)
        if rw is None or i >= len(rw):
            return []
        return [np.asarray(w, float)[:2] for w in rw[i][int(getattr(agent.d[i], "route_idx", 0)):]]

    def _path_value(self, a, pts, r):
        """Residual mass inside the matched footprint of the polyline a -> pts, per metre (+1) of it."""
        if not pts:
            return 0.0
        m = self._cone(a, pts, float(self.oend_fwc_r), float(self.oend_fwc_k), float(self.oend_fwc_end))
        L, cur = 0.0, np.asarray(a, float)[:2]
        for w in pts:
            w = np.asarray(w, float)[:2]
            L += float(np.hypot(*(w - cur)))
            cur = w
        return float(r[m].sum()) / (L + 1.0)

    def _oend_fwc_kw(self):
        """OEND oend_fwc: RouteConfig overrides crediting endgame legs with the cells they view ({} when off)."""
        if not self.oend_fwc:
            return {}
        return dict(fwc_k=float(self.oend_fwc_k), fwc_end=float(self.oend_fwc_end), sweep_r=float(self.oend_fwc_r),
                    min_leg=float(self.oend_min_leg))

    def _oend_budget(self, t_now) -> float:
        return max(float(self.oend_min_budget),
                   (float(self.oend_t_end) - float(t_now)) * V_CRUISE + float(self.oend_see_m))

    def _unseen(self):
        """Cells no viewer has covered yet (quality / depth memory, else the boolean swept map)."""
        if (self._quality_on() or self._depth_on()) and self.seen_p is not None and not self._occl_bool():
            return self.seen_p <= 0.0
        return ~self.swept

    def _oend_floor_r(self, r):
        """OEND tail floor: (1 - f) * residual + f * (same mass spread uniformly over never-viewed cells)."""
        f = float(self.oend_floor)
        if f <= 0.0:
            return r
        un = self._unseen()
        cnt = int(un.sum())
        tot = float(r.sum())
        if cnt <= 0 or tot <= 0.0:
            return r
        return (1.0 - f) * r + f * tot * un.astype(float) / float(cnt)

    def _cone(self, a, pts, axial, k, end):
        """Matched sighting footprint (bool grid): cells viewed from the polyline a -> pts[0] -> ... when the camera
        looks along each leg: cross-track y <= axial, along-track s >= k * y and s <= L + end."""
        out = np.zeros(self.XX.shape, bool)
        cur = np.asarray(a, float)[:2]
        for w in pts:
            b = np.asarray(w, float)[:2]
            v = b - cur
            L = float(np.hypot(v[0], v[1]))
            if L > 1e-3:
                u = v / L
                rx, ry = self.XX - cur[0], self.YY - cur[1]
                s = rx * u[0] + ry * u[1]
                y = np.abs(rx * u[1] - ry * u[0])
                out |= (y <= axial) & (s >= k * y) & (s <= L + end)
            cur = b
        return out

    def _rsv_app(self, agent, st_pos):
        """CRSV: reservation grid of the cells APPROACH drones will view on their straight leg to the claimed pad."""
        rsv = np.zeros(self.XX.shape, float)
        if not (self._crsv_on() and self.crsv_app) or agent is None:
            return rsv
        try:
            pads = getattr(agent, "pads", [])
            for i in range(min(self.n, len(agent.d))):
                dd = agent.d[i]
                c = getattr(dd, "claim", None)
                if (self.frozen[i] or getattr(dd, "phase", None) != "APPROACH" or c is None
                        or not (0 <= c < len(pads)) or st_pos is None):
                    continue
                q = np.asarray(pads[c].xyz, float)[:2]
                m = self._cone(st_pos[i], [q], float(self.crsv_r), float(self.crsv_k), float(self.crsv_app_end))
                if m.any():
                    rsv = np.maximum(rsv, m * float(self.crsv_app_w))
                    self._cv2_count("crsv_app_legs")
        except Exception:                                   # noqa: BLE001
            self._cv2_count("crsv_err")
        return rsv

    def _rsv_order(self, r, owner, P, mode):
        """Planning order of the searchers: "index" keeps it; "gain" puts the best-placed searcher first (own-cell
        residual mass over distance to that mass)."""
        m = len(P)
        if mode != "gain" or m <= 1:
            return list(range(m))
        val = []
        for k in range(m):
            w = r * (owner == k)
            tot = float(w.sum())
            if tot <= 0.0:
                val.append(0.0)
                continue
            cx, cy = float((self.XX * w).sum()) / tot, float((self.YY * w).sum()) / tot
            val.append(tot / (math.hypot(P[k][0] - cx, P[k][1] - cy) + 10.0))
        return [int(k) for k in np.argsort(-np.asarray(val), kind="stable")]

    def _box_mass(self, w, rad_m):
        """Residual mass inside a (2 rad + 1)-cell square around every cell (summed-area table)."""
        step = float(self.gx[1] - self.gx[0]) if len(self.gx) > 1 else 4.0
        h = max(1, int(round(float(rad_m) / step)))
        S = np.zeros((w.shape[0] + 1, w.shape[1] + 1))
        S[1:, 1:] = np.cumsum(np.cumsum(w, 0), 1)
        ny, nx = w.shape
        iy0 = np.clip(np.arange(ny) - h, 0, ny)
        iy1 = np.clip(np.arange(ny) + h + 1, 0, ny)
        ix0 = np.clip(np.arange(nx) - h, 0, nx)
        ix1 = np.clip(np.arange(nx) + h + 1, 0, nx)
        return (S[iy1][:, ix1] - S[iy0][:, ix1] - S[iy1][:, ix0] + S[iy0][:, ix0])

    def _oend_far_target(self, k, r, owner, pos, taken):
        """OEND: one far leg toward the best pocket of residual mass in the drone's own cell (else anywhere), away
        from pockets teammates took in this replan. None when there is no mass."""
        mine = r * (owner == k)
        w = mine if float(mine.sum()) > 1e-9 else r
        if float(w.sum()) <= 1e-9:
            return None
        M = self._box_mass(w, float(self.oend_far_r))
        d = np.hypot(self.XX - float(pos[0]), self.YY - float(pos[1]))
        val = np.where(w > 0.0, M, 0.0) / (d + float(self.oend_far_d0))
        for t in taken:
            val[np.hypot(self.XX - t[0], self.YY - t[1]) < float(self.oend_sep)] = 0.0
        j = int(np.argmax(val))
        if float(val.reshape(-1)[j]) <= 0.0:
            return None
        return np.array([float(self.XX.reshape(-1)[j]), float(self.YY.reshape(-1)[j])])

    def _fwc_kw(self):
        """flat_search3 FWC: RouteConfig overrides for a forward-view corridor ({} when off)."""
        cfg = getattr(getattr(self, "_agent", None), "cfg", None)
        if cfg is None or self.kind not in tuple(getattr(cfg, "fwc_maps", ()) or ()):
            return {}
        return dict(fwc_k=float(cfg.fwc_k), fwc_end=float(cfg.fwc_end))

    def _dvr_on(self) -> bool:
        """flat_search3 DVR: agent cfg dvr_maps names this router kind."""
        cfg = getattr(getattr(self, "_agent", None), "cfg", None)
        return bool(cfg is not None and self.kind in tuple(getattr(cfg, "dvr_maps", ()) or ()))

    def _dvr_kw(self, t_now, n_search):
        """RouteConfig overrides for a DVR chain ({} when off / outside its window)."""
        if not self._dvr_on() or getattr(self, "dvr_mask", None) is None:
            return {}
        c = self._agent.cfg
        if float(t_now) < float(c.dvr_t0) or int(n_search) > int(c.dvr_max_search):
            return {}
        try:
            _sc = self._agent.src_counts
            _sc["dvr_chains"] = int(_sc.get("dvr_chains", 0)) + 1
        except Exception:                                   # noqa: BLE001
            pass
        ns = max(1, int(c.dvr_nsec))
        _ch = int(getattr(c, "dvr_cand_hits", 0) or 0)
        _obv = self.kind in tuple(getattr(c, "obv_maps", ()) or ())
        if _ch <= 0 and not _obv:
            return dict(dvr_sec=self.dvr_mask, dvr_w=float(c.dvr_w_same), dvr_nsec=ns)
        # per-sector weights (ns, ny, nx): DVR, then candidate exemption, then OBV for hidden cells
        bits = (self.dvr_mask[None, :, :] >> np.arange(ns, dtype=np.int64)[:, None, None]) & 1
        W = np.where(bits == 1, float(c.dvr_w_same), 1.0)
        if _ch > 0:
            _r = float(c.dvr_cand_r)
            for q in getattr(self._agent, "pads", []):
                if getattr(q, "by", None) is not None or getattr(q, "done", False) or int(q.hits) < _ch:
                    continue
                m = np.hypot(self.XX - float(q.xyz[0]), self.YY - float(q.xyz[1])) <= _r
                if m.any():
                    W[:, m] = 1.0
        if _obv:
            hid = self.swept & (~self.obv_seen) & (self.obv_block != 0)
            if hid.any():
                bb = (self.obv_block[None, :, :] >> np.arange(ns, dtype=np.int64)[:, None, None]) & 1
                Wh = np.where(bb == 1, 0.0, float(c.obv_boost))
                nb = bb.sum(0)
                Wh[:, nb >= int(c.obv_block_all)] = 0.0
                W = np.where(hid[None, :, :], Wh, W)
                try:
                    self._agent.src_counts["obv_hidden"] = max(int(self._agent.src_counts.get("obv_hidden", 0)), int(hid.sum()))
                except Exception:                           # noqa: BLE001
                    pass
        return dict(dvr_sec=self.dvr_mask, dvr_w=float(c.dvr_w_same), dvr_nsec=ns, dvr_wsec=W)

    def _obv_mark(self, pos, rpy, depth, agl, fp):
        """flat_search3 OBV: ground-seen cells (any range) and blocked sectors (footprint, not seen, <= dvr_rmax)."""
        pv = self._view_depth(pos, rpy, depth, agl)
        seen = (pv > 0.0) & fp
        self.obv_seen |= seen
        blk = fp & ~(pv > 0.0)
        _rm = float(getattr(self._agent.cfg, "dvr_rmax", 0.0) or 0.0)
        if _rm > 0.0:
            blk &= np.hypot(self.XX - float(pos[0]), self.YY - float(pos[1])) <= _rm
        if not blk.any():
            return
        ns = max(1, int(self._agent.cfg.dvr_nsec))
        wid = 2.0 * np.pi / ns
        b = np.arctan2(float(pos[1]) - self.YY[blk], float(pos[0]) - self.XX[blk])
        s = (np.floor(((b % (2.0 * np.pi)) + 0.5 * wid) / wid).astype(np.int64)) % ns
        self.obv_block[blk] |= (np.int64(1) << s)

    def _dvr_mark(self, pos, fp):
        """Set the bearing-sector bit (cell -> viewer) on the cells of a viewer's footprint."""
        m = np.asarray(fp, bool) if np.asarray(fp).dtype == bool else (np.asarray(fp) > 0.0)
        if not m.any():
            return
        ns = max(1, int(self._agent.cfg.dvr_nsec))
        wid = 2.0 * np.pi / ns
        _rm = float(getattr(self._agent.cfg, "dvr_rmax", 0.0) or 0.0)
        if _rm > 0.0:
            # only good (near) looks mark a sector: far-edge views keep the same-sector re-view at full value
            m = m & (np.hypot(self.XX - float(pos[0]), self.YY - float(pos[1])) <= _rm)
            if not m.any():
                return
        b = np.arctan2(float(pos[1]) - self.YY[m], float(pos[0]) - self.XX[m])
        s = (np.floor(((b % (2.0 * np.pi)) + 0.5 * wid) / wid).astype(np.int64)) % ns
        self.dvr_mask[m] |= (np.int64(1) << s)
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
        if self._crsv_on():
            return self._routes_crsv(active, P, r, owner, t_now)
        for k, i in enumerate(active):
            budget = min(self.cfg.budget_m - self.travel[i], (60.0 - t_now) * V_CRUISE - 20.0)
            budget = max(15.0, budget)
            _kw0 = self._turn_kw(0.0 if t_now <= 0.0 else None)
            _kw0.update(self._fwc_kw())
            out[i] = self._chain_fn()(self.gx, self.gy, r, owner, k, P[k],
                                      replace(self._sweep_cfg(), budget_m=budget, t_now_s=float(t_now),
                                              **_kw0))
        return out

    def _routes_crsv(self, active, P, r, owner, t_now):
        """CRSV initial plan: the partition as shipped, chains planned one after another on the residual discounted
        by the matched footprint of the chains already planned (crsv_order decides who goes first)."""
        out = [[] for _ in range(self.n)]
        rsv = np.zeros(r.shape, float)
        w = float(self.crsv_w)
        for k in self._rsv_order(r, owner, P, self.crsv_order):
            i = active[k]
            budget = min(self.cfg.budget_m - self.travel[i], (60.0 - t_now) * V_CRUISE - 20.0)
            budget = max(15.0, budget)
            _kw0 = self._turn_kw(0.0 if t_now <= 0.0 else None)
            _kw0.update(self._fwc_kw())
            rk = r * np.clip(1.0 - w * rsv, 0.0, 1.0) if rsv.any() else r
            out[i] = self._chain_fn()(self.gx, self.gy, rk, owner, k, P[k],
                                      replace(self._sweep_cfg(), budget_m=budget, t_now_s=float(t_now),
                                              **_kw0))
            if out[i]:
                rsv = np.maximum(rsv, self._rsv_cone(P[k], out[i]))
            self._cv2_count("crsv_plan_chains")
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
        return self.kind in tuple(self.sweep_depth_maps or ()) or self._occl_on()

    def _occl_on(self) -> bool:
        """vil_open OCCL: agent cfg occl_sweep_maps names this router kind (see AutopilotConfig)."""
        cfg = getattr(getattr(self, "_agent", None), "cfg", None)
        return bool(cfg is not None and self.kind in tuple(getattr(cfg, "occl_sweep_maps", ()) or ()))

    def _occl_bool(self) -> bool:
        """OCCL with the shipped boolean credit model (residual = prior * (1 - P_DET * swept))."""
        return self._occl_on() and not bool(getattr(self._agent.cfg, "occl_sweep_quality", False))

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
        _mh = int(getattr(self._agent.cfg, "occl_min_rays", 2)) if self._occl_on() else int(self.depth_min_hits)
        cells = np.flatnonzero(cnt >= _mh)
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
                        if self.kind in tuple(self.depth_geom_maps or ()) or self._occl_bool():
                            pv = pv * self._view(st[i, 0:2], float(st[i, 5]), float(st[i, 137]) * 20.0)
                    except Exception:
                        if self._occl_bool():
                            pv = self._view(st[i, 0:2], float(st[i, 5]), float(st[i, 137]) * 20.0).astype(float)
                        else:
                            pv = self._view_p(st[i, 0:2], float(st[i, 5]), float(st[i, 137]) * 20.0)
                    self.seen_p = 1.0 - (1.0 - self.seen_p) * (1.0 - pv)
                    self.swept |= pv > 0.0
                    if self._dvr_on():
                        self._dvr_mark(st[i, 0:2], pv > 0.0)
                elif self._quality_on():
                    pv = self._view_p(st[i, 0:2], float(st[i, 5]), float(st[i, 137]) * 20.0)
                    self.seen_p = 1.0 - (1.0 - self.seen_p) * (1.0 - pv)
                    self.swept |= pv > 0.0
                    if self._dvr_on():
                        self._dvr_mark(st[i, 0:2], pv > 0.0)
                else:
                    _fp = self._view(st[i, 0:2], float(st[i, 5]), float(st[i, 137]) * 20.0)
                    self.swept |= _fp
                    if self._dvr_on():
                        self._dvr_mark(st[i, 0:2], _fp)
                        if (obs.get("depth") is not None
                                and self.kind in tuple(getattr(self._agent.cfg, "obv_maps", ()) or ())):
                            try:
                                self._obv_mark(st[i, 0:3], st[i, 3:6], np.asarray(obs["depth"])[i],
                                               float(st[i, 137]) * 20.0, _fp)
                            except Exception:                   # noqa: BLE001
                                pass
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
        _oend_now = bool(self.oend_maps) and self._oend_gate(t_now, len(dedup), len(searching))
        _oend_go = False
        if _oend_now and not (froze or sig != self.found_sig or moved >= self.replan_m):
            # OEND: replan at the gate onset, and whenever a searcher has used up its route (instead of the spiral)
            if not self._oend_armed:
                _oend_go = True
                self._cv2_count("oend_onset")
            elif (self.oend_replan_done and self.job is None
                  and (self.tick - self._oend_last_tick) >= int(self.oend_gap_ticks)):
                rw = getattr(agent, "_route_waypoints", None)
                for i in searching:
                    if rw is None or i >= len(rw) or int(getattr(d[i], "route_idx", 0)) >= len(rw[i]):
                        _oend_go = True
                        self._cv2_count("oend_done_replans")
                        break
        if _oend_now:
            self._oend_armed = True
        if searching and (froze or sig != self.found_sig or moved >= self.replan_m or _oend_go):
            self.job = {"stage": 0, "searching": list(searching), "found": dedup, "t_now": t_now,
                        "pos": st[searching, 0:2].copy(), "hdg": st[searching, 5].copy()}
            if _oend_now:
                self.job["oend"] = True
                self._oend_last_tick = self.tick
            if self._crsv_on():
                self.job["all_pos"] = st[:, 0:2].copy()
            self.moved[:] = 0.0
            self.found_sig = sig
            self.stats["replans"] += 1
            return
        _g = self._guard_hold(agent)
        if _g is True:
            return                                  # act guard: this job stage runs on a later step
        self._advance_job(agent)
        if _g is not None:
            import time as _gtime
            dt = _gtime.perf_counter() - _g
            a = 0.5 if dt > self._g_job else 0.1
            self._g_job += a * (dt - self._g_job)

    def _guard_hold(self, agent):
        """candHt act guard (agent cfg act_guard_s > 0, act_guard_router): None = guard off or no job; True = the
        job stage would push this act() past the limit, run it on a later step (at most act_guard_job_max_skips
        steps in a row); otherwise the perf_counter start time of the stage, to time it."""
        cfg = getattr(agent, "cfg", None)
        lim = float(getattr(cfg, "act_guard_s", 0.0) or 0.0) if cfg is not None else 0.0
        t0 = getattr(agent, "_g_t0", None)
        if lim <= 0.0 or self.job is None or t0 is None or not bool(getattr(cfg, "act_guard_router", True)):
            return None
        import time as _gtime
        if not hasattr(self, "_g_job"):
            self._g_job, self._g_skips = 0.01, 0
        if getattr(agent, "_g_first", False):
            lim = float(getattr(cfg, "act_guard_first_s", lim))
        now = _gtime.perf_counter()
        if now - t0 + self._g_job > lim and self._g_skips < int(getattr(cfg, "act_guard_job_max_skips", 8)):
            self._g_skips += 1
            try:
                agent._g_n["router_defer"] += 1
            except Exception:                        # noqa: BLE001
                pass
            return True
        self._g_skips = 0
        return now
    def _advance_job(self, agent):
        job = self.job
        if job is None:
            return
        if job["stage"] == 0:                       
            if self._Wq is not None and self._wpost_stage(job, agent):
                return                              # WPOST-X: more components to compute
            job["r"] = self._residual(job["found"])
            job["stage"] = 1 if job["r"] is not None else 99
        elif job["stage"] == 1 and (job.get("oend") or self._crsv_on()) and not self._seq_on():
            self._cv2_stage1(job, agent)            # coverage2 OEND / CRSV
        elif job["stage"] == 2 and "cv2" in job:
            self._cv2_stage2(job, agent)
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
                self._job_hdg = job.get("hdg")
                chain = self._chain_for(i, k, job["r"], job["owner"], job["pos"], job["t_now"])
                self._job_hdg = None
                if chain:
                    agent._route_waypoints[i] = [np.asarray(w, float) for w in chain]
                    d[i].route_idx = 0
            job["k"] += 1
            if job["k"] >= len(job["searching"]):
                job["stage"] = 99
        if job["stage"] == 99:
            self.job = None

    def _cv2_stale(self, job) -> bool:
        """coverage2 review fix: the router kind (and with it the grid, open 31x31 <-> city 38x38) changed under this
        job, so its arrays no longer match self.XX and every later stage would raise on each tick until another trigger
        replaced the job. Drop it and force a replan on the new grid at the next tick."""
        if np.shape(job.get("r")) == self.XX.shape:
            return False
        job["stage"] = 99
        self.found_sig = None
        self._cv2_count("cv2_stale_jobs")
        return True

    def _cv2_stage1(self, job, agent):
        """coverage2 stage 1: endgame floor (OEND), approach reservation (CRSV), partition, planning order."""
        if self._cv2_stale(job):
            return
        r = job["r"]
        oend = bool(job.get("oend"))
        crsv = self._crsv_on()
        if oend and float(self.oend_seen_w) < 1.0:
            r = np.where(self._unseen(), r, r * float(self.oend_seen_w))
            job["r"] = r
            self._cv2_count("oend_seen_fields")
        if oend and float(self.oend_floor) > 0.0:
            r = self._oend_floor_r(r)
            job["r"] = r
            self._cv2_count("oend_floor_fields")
        rsv = self._rsv_app(agent, job.get("all_pos")) if crsv else np.zeros(r.shape, float)
        w = float(self.crsv_w) if crsv else 1.0
        r_eff = r * np.clip(1.0 - w * rsv, 0.0, 1.0) if rsv.any() else r
        if float(r_eff.sum()) <= 1e-12:
            r_eff = r
        job["owner"] = power_diagram(self.gx, self.gy, r_eff, job["pos"], self.cfg,
                                     target=float(r_eff.sum()) / len(job["searching"]))
        mode = "gain" if (oend and self.oend_rsv) else (str(self.crsv_order) if crsv else "index")
        job["order"] = self._rsv_order(r_eff, job["owner"], job["pos"], mode)
        job["rsv"] = rsv
        job["rsv_w"] = w
        job["use_rsv"] = bool(crsv or (oend and self.oend_rsv))
        job["taken"] = []
        job["cv2"] = True
        job["k"] = 0
        job["stage"] = 2

    def _cv2_stage2(self, job, agent):
        """coverage2 stage 2: one searcher per tick, in job["order"], on the reserved residual; OEND budget and far leg."""
        if self._cv2_stale(job):
            return
        k = job["order"][job["k"]]
        i = job["searching"][k]
        d = agent.d
        oend = bool(job.get("oend"))
        if not self.frozen[i] and i < len(d) and getattr(d[i], "claim", None) is None:
            r = job["r"]
            rsv = job["rsv"]
            rk = r * np.clip(1.0 - job["rsv_w"] * rsv, 0.0, 1.0) if (job["use_rsv"] and rsv.any()) else r
            self._job_hdg = job.get("hdg")
            chain = self._chain_for(i, k, rk, job["owner"], job["pos"], job["t_now"],
                                    budget=(self._oend_budget(job["t_now"]) if oend else None),
                                    extra=(self._oend_fwc_kw() if oend else None))
            self._job_hdg = None
            if oend and float(self.oend_keep) > 0.0:
                cur = self._current_rest(agent, i)
                if cur and self._path_value(job["pos"][k], cur, rk) * float(self.oend_keep) > (
                        self._path_value(job["pos"][k], chain, rk) if chain else 0.0):
                    chain = None                     # keep the current route (hysteresis)
                    self._cv2_count("oend_keep")
                    if job["use_rsv"]:
                        job["rsv"] = np.maximum(rsv, self._rsv_cone(job["pos"][k], cur))
            if oend and chain is not None:
                self._cv2_count("oend_chains")
                if not chain and self.oend_far:
                    tgt = self._oend_far_target(k, rk, job["owner"], job["pos"][k], job["taken"])
                    if tgt is not None:
                        chain = [tgt]
                        job["taken"].append(tgt)
                        self._cv2_count("oend_far_legs")
            elif self._crsv_on():
                self._cv2_count("crsv_chains")
            if chain:
                agent._route_waypoints[i] = [np.asarray(w, float) for w in chain]
                d[i].route_idx = 0
                if job["use_rsv"]:
                    job["rsv"] = np.maximum(rsv, self._rsv_cone(job["pos"][k], chain))
        job["k"] += 1
        if job["k"] >= len(job["searching"]):
            job["stage"] = 99
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
            router._cv2_count("router_errors")      # coverage2 diag: surfaces swallowed router faults in the b6 log
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
