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
    while spent < cfg.budget_m and res.size and res.sum() > 1e-9:
        d = seg_dist(cells, cur, nodes)
        corridor = ((d <= cfg.sweep_r) * res[None, :]).sum(1)
        leg = np.linalg.norm(nodes - cur, axis=1)
        ok = (spent + leg) <= cfg.budget_m
        if not ok.any():
            break
        gain = np.where(ok, corridor / (leg + 1.0), -1.0)
        j = int(np.argmax(gain))
        if gain[j] <= 0:
            break
        chain.append(nodes[j].astype(float).copy())
        spent += float(leg[j])
        cur = nodes[j].copy()
        keep = d[j] > cfg.sweep_r
        cells, res = cells[keep], res[keep]
        if res.size == 0:
            break
        alive = np.isin(nodes[:, 0] + 1j * nodes[:, 1], cells[:, 0] + 1j * cells[:, 1])
        nodes = nodes[alive]
        if nodes.shape[0] == 0:
            break
    return chain
