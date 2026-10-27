import heapq
import math
import os
import numpy as np


def _knob(name, default):
    try:
        return float(os.environ.get(name, {'KTF_CORNER': '1', 'KTF_HANDOFF': '4', 'KTF_HOLD': '125', 'KTF_HOVER': '3.5', 'KTF_REHOLD': '125', 'KTF_REHOLD_UP': '3.9', 'KTF_VMAX': '2.8'}.get(name, default)))
    except ValueError:
        return float(default)


KTF_HOLD = int(_knob('KTF_HOLD', 0))
KTF_HOVER = _knob('KTF_HOVER', 0)
KTF_NEAR = _knob('KTF_NEAR', 0)
KTF_VMAX = _knob('KTF_VMAX', 2.4)
KTF_CORNER = int(_knob('KTF_CORNER', 0))
KTF_HOVER_LO = _knob('KTF_HOVER_LO', 0)
KTF_HOVER_R = _knob('KTF_HOVER_R', .8)
KTF_HANDOFF = _knob('KTF_HANDOFF', 0)
KTF_CRUISE = _knob('KTF_CRUISE', 0)
KTF_CRUISE_R = _knob('KTF_CRUISE_R', 5.)
KTF_CRUISE_UP = _knob('KTF_CRUISE_UP', 1.5)
KTF_REHOLD = int(_knob('KTF_REHOLD', 0))
KTF_REHOLD_UP = _knob('KTF_REHOLD_UP', 3.7)
KTF_BANDSTEP = _knob('KTF_BANDSTEP', 0)


class Approach:
    cell = .65

    def __init__(self):
        self.target = None
        self.source = None
        self.active = False
        self.log = {}
        self.stats = dict(plans=0, no_path=0, commands=0, protected=0, releases=0, escape_starts=0)
        self.points = np.empty((0, 3))
        self.stamps = np.empty(0, int)
        self.free = np.empty((0, 3))
        self.path = None
        self.goal = None
        self.last_plan = -10000
        self.target_tick = 0
        self.progress_tick = -10000
        self.best = float('inf')
        self.direct_best = float('inf')
        self.direct_tick = 0
        self.previous = None
        self.yaw = None
        self.suspend = 0
        self.lease_target = None
        self.progress_pos = None
        self.hold = 0
        self.hold_ban = None
        self.handoff = None
        self.z_act = None
        self.rehold = None
        self.rehold_n = 0
        self.rehold_up = KTF_REHOLD_UP
        self.band_steps = 0

    def observe(self, tick, depth, pos, rpy, axes, omb):
        if tick % 10:
            return
        dm = np.asarray(depth, float).reshape(256, 256)*29.5+.5
        tiles = dm.reshape(32, 8, 32, 8).transpose(0, 2, 1, 3).reshape(32, 32, 64)
        j = tiles.argmin(2)
        by, bx = np.indices((32, 32))
        py, px = by*8+j//8, bx*8+j%8
        z = np.take_along_axis(tiles, j[:, :, None], 2)[:, :, 0].ravel()
        fwd, up, right = axes(rpy)
        camera = pos + .13*fwd + .05*up
        rays = fwd + ((px.ravel()+.5)/128-1)[:, None]*right + (1-(py.ravel()+.5)/128)[:, None]*up
        good = np.isfinite(z) & (z < 24.) & (z > .55)
        new = camera + rays[good]*z[good, None]
        if omb is not None and len(omb):
            new = np.concatenate((new, omb))
        old = (tick-self.stamps < 900) & (np.linalg.norm(self.points-pos, axis=1) < 32)
        points = np.concatenate((new, self.points[old]))
        stamps = np.concatenate((np.full(len(new), tick), self.stamps[old]))
        if len(points):
            _, first = np.unique(np.floor(points/.4).astype(int), axis=0, return_index=True)
            self.points, self.stamps = points[first], stamps[first]
        distances = np.minimum(z, 22.)[:, None]*np.linspace(.08, .95, 14)[None, :]
        self.free = (camera+rays[:, None, :]*distances[:, :, None]).reshape(-1, 3)

    def clear(self, a, b, radius=.8):
        if not len(self.points):
            return True
        direction = b-a
        length2 = float(direction@direction)
        t = np.clip((self.points-a)@direction/max(length2, 1e-8), 0, 1)
        distance2 = np.sum((self.points-(a+t[:, None]*direction))**2, axis=1)
        return bool(np.all(distance2 >= radius*radius))

    def plan(self, pos, target, tick):
        self.stats['plans'] += 1
        goal = target + [0, 0, 3.2]
        low = np.minimum(pos, goal)-[5.2, 5.2, 3.25]
        high = np.maximum(pos, goal)+[5.2, 5.2, 3.25]
        low[2] = max(.9, target[2]+1.4, low[2])
        low[2] = min(low[2], pos[2]-.65)
        origin = np.floor(low/self.cell)*self.cell
        shape = np.minimum(np.ceil((high-origin)/self.cell).astype(int)+1, [64, 64, 26])
        if np.any(shape < 3):
            return None
        shape = tuple(int(x) for x in shape)

        def indices(points):
            ij = np.rint((points-origin)/self.cell).astype(int)
            keep = np.all((ij >= 0) & (ij < shape), axis=1)
            return ij[keep]

        clearance = np.full(shape, 2.5, np.float32)
        occupied = indices(self.points)
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                for dz in range(-3, 4):
                    distance = self.cell*math.sqrt(dx*dx+dy*dy+dz*dz)
                    if distance > 2.2:
                        continue
                    ij = occupied+[dx, dy, dz]
                    keep = np.all((ij >= 0) & (ij < shape), axis=1)
                    ij = ij[keep]
                    if len(ij):
                        np.minimum.at(clearance, tuple(ij.T), distance)
        blocked = clearance < .9
        seen = np.zeros(shape, bool)
        ij = indices(self.free)
        seen[tuple(ij.T)] = True
        costs = 1.0 + .65*(~seen) + 1.8*np.maximum(0, (1.7-clearance)/1.7)**2
        if KTF_CRUISE > 0:
            zc = origin[2]+np.arange(shape[2])*self.cell
            xc = origin[0]+np.arange(shape[0])*self.cell
            yc = origin[1]+np.arange(shape[1])*self.cell
            far = np.hypot(xc[:, None]-goal[0], yc[None, :]-goal[1]) > KTF_CRUISE_R
            hover_z = float(target[2])+max(KTF_HOVER, 3.2)
            z0 = self.z_act if self.z_act is not None else float(pos[2])
            cruise = min(max(z0, hover_z), hover_z+KTF_CRUISE_UP)
            low = np.maximum(0., cruise-zc)
            costs = costs+KTF_CRUISE*far[:, :, None]*low[None, None, :]
        start = tuple(np.clip(np.rint((pos-origin)/self.cell).astype(int), 0, np.array(shape)-1))
        ij = np.indices(shape).reshape(3, -1).T
        world = origin+ij*self.cell
        goal_mask = ((np.linalg.norm(world[:, :2]-target[:2], axis=1) <= .85) &
                     (world[:, 2] >= target[2]+2.8) & (world[:, 2] <= target[2]+3.65))
        goals = {tuple(q) for q in ij[goal_mask] if not blocked[tuple(q)]}
        if not goals:
            return None

        def heuristic(q):
            return max(0., float(np.linalg.norm(origin+np.asarray(q)*self.cell-goal))-1.1)

        queue = [(heuristic(start), 0., start)]
        distance = {start: 0.}
        previous = {}
        if blocked[start]:
            queue = []
            current_clearance = float(np.min(np.linalg.norm(self.points-pos, axis=1)))
            escape_radius = min(.8, max(.35, current_clearance-.025))
            delta = world-pos
            lengths = np.linalg.norm(delta, axis=1)
            choices = np.flatnonzero((lengths <= 3.) & (~blocked.ravel()))
            rank = lengths[choices]+np.linalg.norm(world[choices]-goal, axis=1)
            connected = 0
            for j in choices[np.argsort(rank)[:96]]:
                anchor = world[j]
                if not self.clear(pos, anchor, escape_radius):
                    continue
                key = tuple(ij[j])
                g = float(lengths[j])*(1.+.3*float(costs[key]))
                distance[key] = g
                previous[key] = start
                heapq.heappush(queue, (g+1.15*heuristic(key), g, key))
                connected += 1
                if connected >= 12:
                    break
            if not queue:
                return None
            self.stats['escape_starts'] += 1
        end = None
        expanded = 0
        limit = min(24000, int(np.prod(shape)))
        while queue and expanded < limit:
            _, g, q = heapq.heappop(queue)
            if g != distance.get(q):
                continue
            expanded += 1
            if q in goals:
                end = q
                break
            for axis in range(3):
                for sign in (-1, 1):
                    v = list(q)
                    v[axis] += sign
                    if not 0 <= v[axis] < shape[axis]:
                        continue
                    v = tuple(v)
                    if blocked[v]:
                        continue
                    ng = g+self.cell*float(costs[v])*(1.08 if axis == 2 else 1.)
                    if ng < distance.get(v, float('inf')):
                        distance[v] = ng
                        previous[v] = q
                        heapq.heappush(queue, (ng+1.15*heuristic(v), ng, v))
        if end is None:
            return None
        if end == start:
            self.goal = self._hover(origin+np.asarray(end)*self.cell, target)
            return np.asarray([self.goal.copy()])
        route = [origin+np.asarray(end)*self.cell]
        q = end
        while q != start:
            q = previous[q]
            route.append(origin+np.asarray(q)*self.cell)
        route.reverse()
        route[0] = pos.copy()
        smooth = [route[0]]
        i = 0
        while i < len(route)-1:
            j = len(route)-1
            while j > i+1 and not (self.clear(route[i], route[j], .85) and self._keeps_height(route, i, j)):
                j -= 1
            smooth.append(route[j])
            i = j
        smooth[-1] = self._hover(smooth[-1], target)
        self.goal = smooth[-1].copy()
        return np.asarray(smooth[1:] if len(smooth) > 1 else [self.goal.copy()])

    @staticmethod
    def _keeps_height(route, i, j):
        if KTF_CRUISE <= 0 or j <= i+1:
            return True
        pts = np.asarray(route[i:j+1], float)
        s = np.concatenate(([0.], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
        if s[-1] <= 1e-9:
            return True
        line = pts[0, 2]+(pts[-1, 2]-pts[0, 2])*s/s[-1]
        return bool(np.all(line >= pts[:, 2]-.3))

    def _hover(self, cell, target):
        if KTF_HOVER <= 0:
            return cell
        cell = np.asarray(cell, float)
        levels = [self.rehold_up if self._reheld() else KTF_HOVER]
        radii = [.8]
        if KTF_HOVER_LO > 0:
            levels = [KTF_HOVER-.1*i for i in range(int(round((KTF_HOVER-KTF_HOVER_LO)/.1))+1)]
            radii = [.8, KTF_HOVER_R]
        for radius in radii:
            for up in levels:
                goal = cell.copy()
                goal[2] = float(target[2])+up
                if abs(goal[2]-cell[2]) <= 1e-6 or self.clear(cell, goal, radius):
                    return goal
        return cell

    def _reheld(self):
        return bool(self.rehold is not None and self.target is not None
                    and np.linalg.norm(self.target[:2]-self.rehold[:2]) < 2.0)

    def protect(self, tick, q):
        return bool(self.active and self.target is not None and q is not None
                    and tick-self.progress_tick <= 150 and self.best > .8
                    and np.linalg.norm(np.asarray(q)[:2]-self.target[:2]) < 1.5)

    def command(self, tick, st, q, source, incoming):
        pos, velocity = st[:3], st[6:9]
        if q is None or not np.all(np.isfinite(q)):
            self.active = False
            self.path = None
            return None
        q = np.asarray(q, float)
        if self.target is None or np.linalg.norm(q-self.target) > 2.0:
            self.target = q.copy()
            self.target_tick = tick
            self.path = None
            self.active = False
            self.best = self.direct_best = float('inf')
            self.direct_tick = tick
            self.progress_tick = -10000
            self.progress_pos = pos.copy()
            self.last_plan = -10000
            self.hold = 0
            self.rehold_n = 0
        elif np.linalg.norm(q-self.target) > .3:
            self.target = q.copy()
            self.last_plan = -10000
        self.source = source
        hover = self.target+[0, 0, 3.2]
        direct = float(np.linalg.norm(hover-pos))
        if direct < self.direct_best-.4:
            self.direct_best = direct
            self.direct_tick = tick
        if np.max(np.abs(st[3:5])) > math.radians(25):
            self.suspend = tick+50
        if (tick < self.suspend or (direct > 24 and self.path is None)
                or tick-self.target_tick < 20):
            self.active = False
            return None
        if not self.active:
            force = False
            if (KTF_REHOLD and self.handoff is not None and self.rehold is None
                    and np.linalg.norm(self.target[:2]-self.handoff[:2]) < 2.0):
                near = (float(np.linalg.norm(pos[:2]-self.target[:2])) < 1.5
                        and float(np.linalg.norm(velocity)) < .5)
                self.rehold_n = self.rehold_n+1 if near else 0
                if self.rehold_n > KTF_REHOLD:
                    self.rehold = self.target.copy()
                    self.stats['rehold'] = self.stats.get('rehold', 0)+1
                    force = True
            if self.hold_ban is not None and np.linalg.norm(self.target[:2]-self.hold_ban[:2]) < 2.0:
                return None
            obstructed = ((KTF_NEAR <= 0 or direct <= KTF_NEAR)
                          and not (self.handoff is not None
                                   and np.linalg.norm(self.target[:2]-self.handoff[:2]) < 2.0)
                          and not self.clear(pos, hover, .9))
            stalled = tick-self.direct_tick >= 90 and direct > 2.5
            if not (obstructed or stalled or force):
                return None
            self.previous = incoming.copy()
            self.yaw = float(st[5])
            self.best = float('inf')
            self.z_act = float(pos[2])
        if self.path is None and tick-self.last_plan < 40:
            self.active = False
            return None
        if tick-self.last_plan >= 40 or self.path is None:
            self.last_plan = tick
            path = self.plan(pos, self.target, tick)
            if path is None:
                self.stats['no_path'] += 1
                self.active = False
                self.path = None
                return None
            self.path = path
        self.active = True
        while len(self.path) > 1 and np.linalg.norm(self.path[0]-pos) < .6:
            self.path = self.path[1:]
        remaining = float(np.linalg.norm(self.path[0]-pos)+np.linalg.norm(np.diff(self.path, axis=0), axis=1).sum())
        if remaining < self.best-.25:
            self.best = remaining
            if self.progress_pos is not None and np.linalg.norm(pos-self.progress_pos) >= .25:
                self.progress_tick = tick
                self.progress_pos = pos.copy()
            elif self.progress_tick == -10000:
                self.progress_tick = tick
        if (KTF_HANDOFF > 0 and not self._reheld() and remaining < KTF_HANDOFF
                and self.clear(pos, self.path[-1], .9)):
            self.handoff = self.target.copy()
            self.active = False
            self.path = None
            self.stats['handoff'] = self.stats.get('handoff', 0)+1
            return None
        if tick-self.progress_tick > 250:
            self.active = False
            self.path = None
            self.suspend = tick+150
            self.stats['releases'] += 1
            return None
        offset = self.path[0]-pos
        distance = float(np.linalg.norm(offset))
        final = len(self.path) == 1
        if KTF_HOLD:
            if final and distance < .35 and float(np.linalg.norm(velocity)) < .5:
                self.hold += 1
            else:
                self.hold = 0
            if self.hold > KTF_HOLD and KTF_BANDSTEP > 0 and self._reheld() and self.band_steps < 1:
                self.band_steps += 1
                self.rehold_up += KTF_BANDSTEP
                self.hold = 0
                self.last_plan = -10000
                self.stats['band_step'] = self.stats.get('band_step', 0)+1
            if self.hold > KTF_HOLD:
                self.hold_ban = self.target.copy()
                self.hold = 0
                self.active = False
                self.path = None
                self.stats['hold_yield'] = self.stats.get('hold_yield', 0)+1
                return None
        speed = min(KTF_VMAX, math.sqrt(2*1.2*max(0., remaining-.15)))
        if final:
            speed = min(speed, 1.25*distance)
        elif KTF_CORNER:
            leg = self.path[1]-self.path[0]
            cosine = float(offset@leg)/max(distance*float(np.linalg.norm(leg)), 1e-8)
            corner = max(.7, KTF_VMAX*max(0., cosine))
            speed = min(speed, math.sqrt(corner*corner+2*1.2*distance))
        else:
            speed = min(speed, max(.55, 1.5*distance))
        desired = offset/max(distance, 1e-8)*speed
        desired[2] = np.clip(desired[2], -1.4, 1.4)
        previous = self.previous if self.previous is not None else velocity
        change = desired-previous
        desired = previous+change*min(1., .045/max(float(np.linalg.norm(change)), 1e-8))
        head = math.atan2(offset[1], offset[0]) if np.linalg.norm(offset[:2]) > .5 else self.yaw
        turn = (head-self.yaw+math.pi)%(2*math.pi)-math.pi
        self.yaw += float(np.clip(turn, -.018, .018))
        self.previous = desired.copy()
        self.stats['commands'] += 1
        self.log = dict(active=True, target=self.target.tolist(), source=source,
                        goal=self.goal.tolist(), waypoint=self.path[0].tolist(),
                        remaining=remaining, progress_tick=self.progress_tick,
                        route=self.path.tolist(), point_count=len(self.points))
        return desired, (self.yaw+math.pi)%(2*math.pi)-math.pi
