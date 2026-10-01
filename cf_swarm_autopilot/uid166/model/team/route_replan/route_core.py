from __future__ import annotations
from dataclasses import dataclass
import numpy as np
@dataclass
class RouteConfig:
    step: float = 4.0          
    sweep_r: float = 11.0      
    budget_m: float = 150.0
    n_nodes: int = 250
    power_iters: int = 40
    power_lr: float = 6.0
    # Time discount of swept mass: a pad found late is worth less (the time term decays
    # to 0 at the 60 s horizon), so a leg's gain is scaled by 1 - t_disc * t_arrive / 60.
    # 0 = off (pure mass per metre). t_now_s / v_cruise convert metres into arrival time.
    t_disc: float = 0.0
    t_now_s: float = 0.0
    v_cruise: float = 2.9
    # Two-leg lookahead: instead of the leg with the best mass per metre, take the first leg
    # of the best two-leg plan (mass of both legs over their total length), evaluated for
    # the lookahead_k best single legs. Cuts the greedy zig-zag between mass pockets. 0 = off.
    lookahead_k: int = 0
    # yaw_time RT: turn-aware greedy chain (0 = off). A leg whose direction is dth degrees off the
    # heading the drone arrives with (camera yaw for the first leg, previous leg after that) costs
    # turn_k * max(0, dth - turn_free_deg) extra metres in the gain denominator (the camera turns
    # ~35 deg/s, so at 2.9 m/s the new leg is flown ~0.083 m/deg blind; a reversal also brakes).
    turn_k: float = 0.0
    turn_free_deg: float = 30.0
    heading: object = None
    # flat_search3 DVR: per-cell bitmask of bearing sectors already viewed from (grid-shaped, None = off),
    # the residual weight of a same-sector re-view, and the sector count (see AutopilotConfig dvr_*).
    dvr_sec: object = None
    dvr_w: float = 1.0
    dvr_nsec: int = 4
    dvr_wsec: object = None      # (nsec, ny, nx) per-sector residual weights; overrides dvr_sec / dvr_w when set
    # flat_search3 FWC: forward-view corridor (0 = off: the symmetric sweep_r capsule around each leg). The camera
    # looks along the leg, so a leg only sees cells AHEAD of it within the half-FOV: cross-track y <= sweep_r,
    # along-track s >= fwc_k * y (45 deg for k = 1) and s <= leg length + fwc_end. Cells beside / behind a leg's
    # start (e.g. the start pad's neighbourhood under a take-off loop) are no longer credited to it.
    fwc_k: float = 0.0
    fwc_end: float = 11.0
    # coverage2 (OEND / CRSV): legs shorter than min_leg are not eligible (0 = off). With the forward-view corridor a
    # 4 m hop is credited with a 20 m cone ahead and wins the mass-per-metre ratio, but the agent treats a waypoint
    # within waypoint_reach (6 m) as reached and skips it.
    min_leg: float = 0.0
def power_diagram(gx, gy, rho, starts, cfg, target=1.0):
    n = len(starts)
    XX = np.broadcast_to(gx[None, :], rho.shape)
    YY = np.broadcast_to(gy[:, None], rho.shape)
    tau = np.stack([np.hypot(XX - s[0], YY - s[1]) for s in starts])
    w = np.zeros(n)
    owner = np.zeros(rho.shape, int)
    for _ in range(cfg.power_iters):
        owner = np.argmin(tau - w[:, None, None], axis=0)
        mass = np.array([rho[owner == i].sum() for i in range(n)])
        w += cfg.power_lr * (target - mass)
        w -= w.mean()
    return owner
def seg_dist(cells, a, b):
    cells = cells.astype(np.float32, copy=False)
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(1), 1e-9)
    ap = cells[None, :, :] - a[None, None, :]
    t = np.clip((ap * ab[:, None, :]).sum(-1) / L2[:, None], 0.0, 1.0)
    dx = ap[..., 0] - t * ab[:, None, 0]
    dy = ap[..., 1] - t * ab[:, None, 1]
    return np.sqrt(dx * dx + dy * dy)
def _fwc_win(cells, a, nodes, r, k, end):
    """flat_search3 FWC coverage (nodes x cells, bool): cells a leg a -> node sees ahead of it."""
    a = np.asarray(a, np.float32)
    ab = np.asarray(nodes, np.float32) - a[None, :]
    L = np.sqrt((ab * ab).sum(1))
    u = ab / np.maximum(L, 1e-6)[:, None]
    rel = cells.astype(np.float32, copy=False) - a[None, :]
    s = u[:, 0:1] * rel[None, :, 0] + u[:, 1:2] * rel[None, :, 1]
    y = np.abs(u[:, 0:1] * rel[None, :, 1] - u[:, 1:2] * rel[None, :, 0])
    win = (y <= r) & (s >= k * y) & (s <= L[:, None] + end)
    win &= (L > 1e-3)[:, None]
    return win


def corridor_chain(gx, gy, rho, owner, i, start_xy, cfg):
    mine = owner == i
    if not mine.any():
        return []
    XX, YY = np.meshgrid(gx, gy)
    flat = np.column_stack([XX.ravel(), YY.ravel()]).astype(np.float32)
    res_full = (rho * mine).ravel().astype(np.float32)
    live = np.flatnonzero(res_full > 0)
    if live.size == 0:
        return []
    cells, res = flat[live], res_full[live]
    k = min(cfg.n_nodes, live.size)
    sel = np.random.default_rng(0).choice(live.size, size=k, replace=False, p=res / res.sum())
    nodes = cells[sel]
    cur = np.asarray(start_xy, np.float32)
    spent, chain = 0.0, []
    _tk = float(getattr(cfg, "turn_k", 0.0) or 0.0)
    _hdg = getattr(cfg, "heading", None) if _tk > 0.0 else None
    _dvm = getattr(cfg, "dvr_sec", None)
    msk = None
    if _dvm is not None:
        # flat_search3 DVR: sector bitmask per live cell (aligned with cells / res)
        msk = np.asarray(_dvm).reshape(-1)[live].astype(np.int64)
        _dns = max(1, int(getattr(cfg, "dvr_nsec", 4)))
        _dw = float(getattr(cfg, "dvr_w", 1.0))
        _dwid = 2.0 * np.pi / _dns
        _dws = getattr(cfg, "dvr_wsec", None)
        wsec = None if _dws is None else np.asarray(_dws, np.float64).reshape(_dns, -1)[:, live]
    _fk = float(getattr(cfg, "fwc_k", 0.0) or 0.0)
    while spent < cfg.budget_m and res.size and res.sum() > 1e-9:
        d = seg_dist(cells, cur, nodes)
        if _fk > 0.0:
            _win = _fwc_win(cells, cur, nodes, float(cfg.sweep_r), _fk, float(cfg.fwc_end))
        else:
            _win = None
        if msk is None and _win is None:
            corridor = ((d <= cfg.sweep_r) * res[None, :]).sum(1)
        elif msk is None:
            corridor = (_win * res[None, :]).sum(1)
        else:
            # leg j views its corridor from behind: bearing cell -> camera = leg direction + pi
            _vb = np.arctan2(nodes[:, 1] - cur[1], nodes[:, 0] - cur[0]) + np.pi
            _sj = (np.floor(((_vb % (2.0 * np.pi)) + 0.5 * _dwid) / _dwid).astype(np.int64)) % _dns
            if _win is None:
                _win = (d <= cfg.sweep_r)
            corridor = np.zeros(nodes.shape[0], np.float64)
            for _s in np.unique(_sj):
                if wsec is not None:
                    _rs = res * wsec[int(_s)]
                else:
                    _rs = np.where(((msk >> int(_s)) & 1) == 1, res * _dw, res)
                _jj = np.flatnonzero(_sj == _s)
                corridor[_jj] = (_win[_jj] * _rs[None, :]).sum(1)
        leg = np.linalg.norm(nodes - cur, axis=1)
        ok = (spent + leg) <= cfg.budget_m
        _ml = float(getattr(cfg, "min_leg", 0.0) or 0.0)
        if _ml > 0.0:
            ok &= leg >= _ml
        if not ok.any():
            break
        if cfg.t_disc > 0.0:
            t_arr = cfg.t_now_s + (spent + leg) / max(cfg.v_cruise, 0.1)
            corridor = corridor * np.clip(1.0 - cfg.t_disc * t_arr / 60.0, 0.05, 1.0)
        _den = leg + 1.0
        if _hdg is not None:
            _ang = np.degrees(np.abs((np.arctan2(nodes[:, 1] - cur[1], nodes[:, 0] - cur[0]) - float(_hdg)
                                      + np.pi) % (2.0 * np.pi) - np.pi))
            _den = _den + _tk * np.maximum(0.0, _ang - float(cfg.turn_free_deg))
        gain = np.where(ok, corridor / _den, -1.0)
        j = int(np.argmax(gain))
        if gain[j] <= 0:
            break
        if cfg.lookahead_k > 0 and nodes.shape[0] > 2:
            top = np.argsort(-gain)[: int(cfg.lookahead_k)]
            best_j, best_val = j, -1.0
            for jj in top:
                jj = int(jj)
                if gain[jj] <= 0:
                    continue
                keep2 = d[jj] > cfg.sweep_r
                cells2, res2 = cells[keep2], res[keep2]
                if res2.size == 0:
                    val = gain[jj]
                else:
                    d2 = seg_dist(cells2, nodes[jj], nodes)
                    corr2 = ((d2 <= cfg.sweep_r) * res2[None, :]).sum(1)
                    leg2 = np.linalg.norm(nodes - nodes[jj], axis=1)
                    ok2 = (spent + leg[jj] + leg2) <= cfg.budget_m
                    ok2[jj] = False
                    if cfg.t_disc > 0.0:
                        t2 = cfg.t_now_s + (spent + leg[jj] + leg2) / max(cfg.v_cruise, 0.1)
                        corr2 = corr2 * np.clip(1.0 - cfg.t_disc * t2 / 60.0, 0.05, 1.0)
                    two = np.where(ok2, (corridor[jj] + corr2) / (leg[jj] + leg2 + 1.0), -1.0)
                    val = max(float(gain[jj]), float(two.max()))
                if val > best_val:
                    best_val, best_j = val, jj
            j = best_j
        chain.append(nodes[j].astype(float).copy())
        spent += float(leg[j])
        if _tk > 0.0 and float(leg[j]) > 1e-6:
            _hdg = float(np.arctan2(float(nodes[j][1] - cur[1]), float(nodes[j][0] - cur[0])))
        cur = nodes[j].copy()
        keep = (~_win[j]) if _win is not None else (d[j] > cfg.sweep_r)
        cells, res = cells[keep], res[keep]
        if msk is not None:
            msk = msk[keep]
            if wsec is not None:
                wsec = wsec[:, keep]
        if res.size == 0:
            break
        alive = np.isin(nodes[:, 0] + 1j * nodes[:, 1], cells[:, 0] + 1j * cells[:, 1])
        nodes = nodes[alive]
        if nodes.shape[0] == 0:
            break
    return chain


def lawnmower_chain(gx, gy, rho, owner, i, start_xy, cfg):
    """Serpentine coverage of a drone's cell, as an alternative to the greedy chain.

    ``corridor_chain`` is myopic: it hops to whichever node offers the most swept mass
    per metre, which zig-zags and can leave slivers between legs. A boustrophedon over
    the cell's principal axis, with legs one full swath apart, covers the same cell with
    no gaps and no backtracking -- better when the posterior over a cell is smooth,
    which is what the clue plus an annulus prior tends to produce.
    """
    mine = owner == i
    if not mine.any():
        return []
    XX, YY = np.meshgrid(gx, gy)
    w = rho * mine
    if float(w.sum()) <= 0.0:
        return []
    pts = np.column_stack([XX[mine], YY[mine]]).astype(np.float64)
    ws = w[mine].astype(np.float64)
    tot = ws.sum()
    mu = (pts * ws[:, None]).sum(0) / tot
    d = pts - mu
    cov = (d * ws[:, None]).T @ d / tot
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, int(np.argmax(evals))]
    minor = np.array([-major[1], major[0]])
    along = d @ major
    across = d @ minor
    lo, hi = float(across.min()), float(across.max())
    span = max(hi - lo, 1e-6)
    step = max(2.0 * float(cfg.sweep_r), 1e-6)
    n_lines = max(1, int(round(span / step)))
    half = span / (2.0 * n_lines)
    lines = []
    for k in range(n_lines):
        c = lo + (2 * k + 1) * half
        sel = np.abs(across - c) <= half
        if not sel.any():
            continue
        # Trim each leg to where the strip's mass actually is: with a budget of
        # ~150 m a leg drawn across the cell's full extent spends most of it over
        # ground the posterior has already written off.
        a_s, w_s = along[sel], ws[sel]
        order = np.argsort(a_s)
        a_s, w_s = a_s[order], w_s[order]
        cum = np.cumsum(w_s)
        strip = float(cum[-1])
        if strip <= 0.0:
            continue
        a_lo = float(a_s[int(np.searchsorted(cum, 0.05 * strip))])
        a_hi = float(a_s[min(int(np.searchsorted(cum, 0.95 * strip)), len(a_s) - 1)])
        if a_hi - a_lo < 1e-6:
            continue
        lines.append((strip, mu + major * a_lo + minor * c, mu + major * a_hi + minor * c))
    if not lines:
        return []
    peak = max(l[0] for l in lines)
    lines = [l for l in lines if l[0] >= 0.10 * peak]
    cur = np.asarray(start_xy, dtype=np.float64)
    chain, spent = [], 0.0
    while lines and spent < cfg.budget_m:
        best = None
        for idx, (mass, p0, p1) in enumerate(lines):
            leg = float(np.linalg.norm(p1 - p0))
            for e0, e1 in ((p0, p1), (p1, p0)):
                hop = float(np.linalg.norm(e0 - cur))
                # mass per metre, counting the dead-heading hop: picks up the
                # richest strip the budget can still reach rather than the nearest
                gain = mass / (hop + leg + 1.0)
                if best is None or gain > best[0]:
                    best = (gain, idx, e0, e1, hop, leg)
        _g, idx, e0, e1, hop, leg = best
        if chain and spent + hop + leg > cfg.budget_m:
            break
        chain.append(np.asarray(e0, float))
        chain.append(np.asarray(e1, float))
        spent += hop + leg
        cur = e1
        lines.pop(idx)
    return chain
