import math
import numpy as np


def make_prior(clue, pad):
    axis = np.arange(-48., 50., 2.)
    x, y = np.meshgrid(axis, axis)
    points = np.column_stack((x.ravel(), y.ravel()))
    keep = ((np.linalg.norm(points - clue, axis=1) <= 33.) &
            (np.linalg.norm(points - pad, axis=1) <= 83.))
    points = points[keep]
    sub = np.arange(-.875, 1., .25)
    dx, dy = np.meshgrid(sub, sub)
    samples = points[:, None, :] + np.column_stack((dx.ravel(), dy.ravel()))
    valid = ((np.abs(samples).max(axis=2) <= 25.) &
             (np.linalg.norm(samples - clue, axis=2) <= 30.))
    weights = valid.mean(axis=1) / (50. * 50.)
    return points, weights


def cells(plan, xy):
    ij = np.rint((xy + 48.) / 2.).astype(np.int64)
    valid = ((ij >= 0) & (ij < 49)).all(axis=-1)
    clipped = np.clip(ij, 0, 48)
    return np.where(valid, plan.gidx[clipped[..., 1], clipped[..., 0]], -1)


def blocked(plan, xy, z=None):
    idx = cells(plan, xy)
    occupied = (idx >= 0) & plan.occ[np.maximum(idx, 0)]
    if z is not None and hasattr(plan, 'rim_height'):
        height = plan.rim_height[np.maximum(idx, 0)]
        occupied &= (height <= 0.) | (height + .35 >= z)
    return occupied


def choose(plan, pos, yaw, tick):
    if (plan.wp is not None and tick - plan.t_wp < 125 and
            np.linalg.norm(plan.wp - pos[:2]) > 4.):
        return plan.wp
    weights = plan.mass.copy()
    weights[plan.occ | (plan.w0 == 0.)] = 0.
    if weights.sum() < .05 * plan.w0.sum():
        plan.mass = .5 * plan.w0.copy()
        plan.mass[plan.occ] = 0.
        weights = plan.mass.copy()
        plan.resets += 1
    headings = yaw + np.arange(24) * (2. * math.pi / 24.)
    directions = np.column_stack((np.cos(headings), np.sin(headings)))
    lengths = np.repeat(np.array([8., 14., 22.]), len(headings))
    directions = np.tile(directions, (3, 1))
    endpoints = pos[:2] + lengths[:, None] * directions
    allowed = ((np.linalg.norm(endpoints - plan.rim_clue, axis=1) <= 33.) &
               (np.abs(endpoints).max(axis=1) <= 32.) & ~blocked(plan, endpoints))
    turn = np.abs((np.arctan2(directions[:, 1], directions[:, 0]) - yaw +
                   math.pi) % (2. * math.pi) - math.pi) / .7
    travel = lengths / 2.8
    duration = turn + travel + 1.5
    allowed &= duration <= max(4., 58. - tick * .02 - 6.)
    best = np.zeros((len(endpoints), len(plan.sup)))
    min_range = max(5.5, (float(pos[2]) - .8) * 1.05)
    detour = np.zeros(len(endpoints))
    for fraction in (0., .25, .5, .75, 1.):
        eye = pos[:2] + fraction * lengths[:, None] * directions
        rays = plan.sup[None, :, :] - eye[:, None, :]
        distance = np.linalg.norm(rays, axis=2)
        forward = np.sum(rays * directions[:, None, :], axis=2)
        visible = ((distance >= min_range) & (distance <= 24.) &
                   (forward >= math.cos(math.radians(38.)) * distance))
        for along in (.2, .4, .6, .8):
            ray_z = float(pos[2]) + along * (.8 - float(pos[2]))
            visible &= ~blocked(plan, eye[:, None, :] + along * rays, ray_z)
        quality = np.clip((26. - distance) / 12., 0., 1.)
        first_view = (turn + fraction * travel) / duration
        best = np.maximum(best, visible * quality * (1. - .35 * first_view[:, None]))
        if fraction > 0.:
            detour += blocked(plan, eye, float(pos[2]) - 2.) * 1.5
    score = (best @ weights) / (duration + detour)
    score[~allowed] = -1.
    if plan.blk is not None and tick <= plan.blk[1]:
        angle = np.arctan2(directions[:, 1], directions[:, 0])
        excluded = np.abs((angle - plan.blk[0] + math.pi) % (2 * math.pi) - math.pi) < math.pi / 3
        if np.any((~excluded) & (score > 0.)):
            score[excluded] = -1.
    choice = int(np.argmax(score))
    if score[choice] <= 0.:
        return None
    plan.wp = endpoints[choice].copy()
    plan.t_wp = tick
    plan.n_plan += 1
    plan.n_rim = getattr(plan, 'n_rim', 0) + 1
    plan.rim_rate = float(score[choice])
    return plan.wp
