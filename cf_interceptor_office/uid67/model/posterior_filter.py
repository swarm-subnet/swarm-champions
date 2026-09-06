"""Multimodal visual/map particle filter for office self-localization."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PosteriorState:
    xy: np.ndarray
    heading_offset: float
    global_spread_m: float
    mode_weight: float
    mode_spread_m: float
    effective_particles: float
    finite: bool


@dataclass(frozen=True)
class DirectionalCollisionRisk:
    danger_probability: float
    clearance_p01_m: float
    clearance_p05_m: float
    expected_clearance_m: float
    finite: bool


class VisualMotionParticleFilter:
    """Fuse absolute visual map-cell likelihoods with relative ego-motion.

    Each particle carries `(world_x, world_y, takeoff_heading_offset)`. Relative
    odometry is rotated by that particle's heading hypothesis, matching v12's
    existing map-localizer state rather than assuming a known global yaw.
    """

    def __init__(
        self,
        office_map,
        *,
        particles: int = 8192,
        seed: int = 0,
        visual_outlier: float = 0.08,
        visual_power: float = 0.65,
        alive_floor: float = 0.02,
        hull_m: float = 0.16,
    ):
        self.om = office_map
        self.n = int(particles)
        self.rng = np.random.default_rng(seed)
        self.visual_outlier = float(visual_outlier)
        self.visual_power = float(visual_power)
        self.alive_floor = float(alive_floor)
        self.hull_m = float(hull_m)
        self.reset()

    def reset(self) -> None:
        margin = 0.25
        lo = self.om.origin[:2] + margin
        hi = self.om.origin[:2] + np.array(
            [self.om.shape[2], self.om.shape[1]], dtype=np.float64
        ) * self.om.res - margin
        keep = []
        while sum(len(x) for x in keep) < self.n:
            xy = self.rng.uniform(lo, hi, (self.n, 2))
            probe = np.column_stack([xy, np.full(self.n, 1.5)])
            keep.append(xy[np.asarray(self.om.clearance(probe)) > self.hull_m])
        self.xy = np.concatenate(keep, axis=0)[: self.n]
        self.theta = self.rng.uniform(-math.pi, math.pi, self.n)
        self.w = np.full(self.n, 1.0 / self.n)
        self.last_dr = np.zeros(2, dtype=np.float64)
        self.visual_seeded = False

    def _seed_from_visual(
        self,
        probability: np.ndarray,
        z_m: float,
        yaw_probability: np.ndarray | None = None,
        relative_yaw_rad: float | None = None,
    ) -> None:
        """Place most particles in visual modes without deleting global support."""
        flat = np.asarray(probability, dtype=np.float64).reshape(-1)
        flat = np.maximum(flat, 0.0)
        total = float(flat.sum())
        if total <= 0.0 or not math.isfinite(total):
            return
        flat /= total
        proposal = 0.95 * flat + 0.05 / len(flat)
        cells = self.rng.choice(len(flat), size=self.n, replace=True, p=proposal)
        xb = cells % 36
        yb = cells // 36
        self.xy[:, 0] = (xb + self.rng.random(self.n)) * (18.0 / 36.0)
        self.xy[:, 1] = (yb + self.rng.random(self.n)) * (7.6 / 16.0)
        if yaw_probability is not None and relative_yaw_rad is not None:
            yaw_p = np.maximum(np.asarray(yaw_probability, dtype=np.float64).reshape(24), 0.0)
            yaw_p /= max(float(yaw_p.sum()), 1e-12)
            yaw_proposal = 0.90 * yaw_p + 0.10 / 24.0
            yaw_bins = self.rng.choice(24, size=self.n, replace=True, p=yaw_proposal)
            global_yaw = (yaw_bins + self.rng.random(self.n)) * (2.0 * math.pi / 24.0)
            self.theta = global_yaw - float(relative_yaw_rad)
        else:
            self.theta = self.rng.uniform(-math.pi, math.pi, self.n)
        probe = np.column_stack([self.xy, np.full(self.n, float(z_m))])
        free = np.asarray(self.om.clearance(probe), dtype=np.float64) > self.hull_m
        # Sampling already represents the proposal distribution, so particles
        # begin equally weighted before map/free-space evidence is applied.
        self.w = np.ones(self.n, dtype=np.float64)
        self.w *= np.where(free, 1.0, self.alive_floor)
        self._normalize()
        self.visual_seeded = True

    def _normalize(self) -> bool:
        total = float(np.sum(self.w))
        if not math.isfinite(total) or total <= 0.0 or not np.isfinite(self.w).all():
            return False
        self.w /= total
        return True

    def _resample(self) -> None:
        positions = (self.rng.random() + np.arange(self.n)) / self.n
        indices = np.clip(np.searchsorted(np.cumsum(self.w), positions), 0, self.n - 1)
        self.xy = self.xy[indices] + self.rng.normal(0.0, 0.035, (self.n, 2))
        self.theta = self.theta[indices] + self.rng.normal(0.0, 0.012, self.n)
        self.w.fill(1.0 / self.n)

    def step(
        self,
        dr_xy: np.ndarray,
        z_m: float,
        visual_xy_probability: np.ndarray,
        *,
        tof_m: float | None = None,
        relative_yaw_rad: float | None = None,
        visual_yaw_probability: np.ndarray | None = None,
    ) -> PosteriorState:
        dr = np.asarray(dr_xy, dtype=np.float64)
        probability = np.asarray(visual_xy_probability, dtype=np.float64).reshape(16, 36)
        if dr.shape != (2,) or not np.isfinite(dr).all() or not np.isfinite(probability).all():
            return self.state(finite=False)
        if not self.visual_seeded:
            self._seed_from_visual(
                probability, z_m, visual_yaw_probability, relative_yaw_rad
            )
        delta = dr - self.last_dr
        self.last_dr = dr.copy()
        c, s = np.cos(self.theta), np.sin(self.theta)
        moved = float(np.linalg.norm(delta))
        self.xy[:, 0] += c * delta[0] - s * delta[1]
        self.xy[:, 1] += s * delta[0] + c * delta[1]
        if moved > 0.0:
            self.xy += self.rng.normal(0.0, 0.006 + 0.012 * moved, self.xy.shape)

        probe = np.column_stack([self.xy, np.full(self.n, float(z_m))])
        clearance = np.asarray(self.om.clearance(probe), dtype=np.float64)
        self.w *= np.where(clearance > self.hull_m, 1.0, self.alive_floor)
        if tof_m is not None and math.isfinite(float(tof_m)) and 0.0 < float(tof_m) < 50.0:
            residual = np.abs(np.asarray(self.om.predict_tof(probe)) - float(tof_m))
            self.w *= 0.35 + 0.65 * np.exp(-0.5 * (residual / 0.13) ** 2)

        xb = np.floor(self.xy[:, 0] / 18.0 * 36).astype(np.int64)
        yb = np.floor(self.xy[:, 1] / 7.6 * 16).astype(np.int64)
        inside = (xb >= 0) & (xb < 36) & (yb >= 0) & (yb < 16)
        visual = np.full(self.n, self.visual_outlier / 576.0, dtype=np.float64)
        visual[inside] += (1.0 - self.visual_outlier) * probability[yb[inside], xb[inside]]
        self.w *= np.power(np.maximum(visual, 1e-12), self.visual_power)
        if visual_yaw_probability is not None and relative_yaw_rad is not None:
            yaw_p = np.maximum(
                np.asarray(visual_yaw_probability, dtype=np.float64).reshape(24), 0.0
            )
            yaw_p /= max(float(yaw_p.sum()), 1e-12)
            global_yaw = (self.theta + float(relative_yaw_rad)) % (2.0 * math.pi)
            yaw_bin = np.floor(global_yaw / (2.0 * math.pi) * 24).astype(np.int64)
            yaw_like = 0.20 / 24.0 + 0.80 * yaw_p[yaw_bin]
            self.w *= np.power(np.maximum(yaw_like, 1e-12), 0.55)
        if not self._normalize():
            self.reset()
            return self.state(finite=False)
        ess = 1.0 / float(np.sum(self.w * self.w))
        # Preserve alternatives longer than v12's geometric-only filter; visual
        # likelihood can be sharply multimodal under repeated office layouts.
        if ess < 0.25 * self.n:
            self._resample()
        return self.state()

    def state(self, *, finite: bool = True) -> PosteriorState:
        mean = np.average(self.xy, axis=0, weights=self.w)
        global_spread = float(np.sqrt(np.average(np.sum((self.xy - mean) ** 2, axis=1), weights=self.w)))
        # Search high-weight particles for the dominant 0.5 m mode.
        best_mask = np.ones(self.n, dtype=bool)
        best_weight = -1.0
        for index in np.argsort(-self.w)[:32]:
            mask = np.linalg.norm(self.xy - self.xy[index], axis=1) <= 0.50
            weight = float(self.w[mask].sum())
            if weight > best_weight:
                best_weight, best_mask = weight, mask
        mode_xy = np.average(self.xy[best_mask], axis=0, weights=self.w[best_mask])
        mode_spread = float(np.sqrt(np.average(
            np.sum((self.xy[best_mask] - mode_xy) ** 2, axis=1), weights=self.w[best_mask]
        )))
        mode_theta = float(math.atan2(
            np.sum(self.w[best_mask] * np.sin(self.theta[best_mask])),
            np.sum(self.w[best_mask] * np.cos(self.theta[best_mask])),
        ))
        ess = 1.0 / float(np.sum(self.w * self.w))
        okay = bool(finite and np.isfinite(mode_xy).all() and math.isfinite(mode_spread))
        return PosteriorState(
            xy=np.asarray(mode_xy, dtype=np.float64),
            heading_offset=mode_theta,
            global_spread_m=global_spread,
            mode_weight=max(0.0, best_weight),
            mode_spread_m=mode_spread,
            effective_particles=ess,
            finite=okay,
        )


class GridHeadingFilter:
    """Exact short-horizon Bayes filter over 576 XY cells and heading bins."""

    def __init__(self, office_map, *, heading_bins: int = 72,
                 visual_outlier: float = 0.08, yaw_outlier: float = 0.20):
        self.om = office_map
        self.heading_bins = int(heading_bins)
        self.visual_outlier = float(visual_outlier)
        self.yaw_outlier = float(yaw_outlier)
        xs = (np.arange(36) + 0.5) * (18.0 / 36.0)
        ys = (np.arange(16) + 0.5) * (7.6 / 16.0)
        xx, yy = np.meshgrid(xs, ys)
        self.cell_xy = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
        self.theta = (np.arange(self.heading_bins) + 0.5) * (
            2.0 * math.pi / self.heading_bins
        ) - math.pi
        self.belief = None
        self.last_dr = np.zeros(2, dtype=np.float64)

    def _observation(self, probability, yaw_probability, relative_yaw, z_m, tof_m):
        visual = self.visual_outlier / 576.0 + (
            1.0 - self.visual_outlier
        ) * np.asarray(probability, dtype=np.float64).reshape(576)
        visual = np.power(np.maximum(visual, 1e-12), 0.70)
        global_yaw = (self.theta + float(relative_yaw)) % (2.0 * math.pi)
        yaw_bin = np.floor(global_yaw / (2.0 * math.pi) * 24).astype(np.int64)
        yaw_p = np.asarray(yaw_probability, dtype=np.float64).reshape(24)
        yaw_p = yaw_p / max(float(yaw_p.sum()), 1e-12)
        yaw_like = self.yaw_outlier / 24.0 + (1.0 - self.yaw_outlier) * yaw_p[yaw_bin]
        yaw_like = np.power(np.maximum(yaw_like, 1e-12), 0.55)
        probe = np.column_stack([self.cell_xy, np.full(576, float(z_m))])
        map_like = np.where(np.asarray(self.om.clearance(probe)) > 0.16, 1.0, 0.02)
        if tof_m is not None and 0.0 < float(tof_m) < 50.0:
            residual = np.abs(np.asarray(self.om.predict_tof(probe)) - float(tof_m))
            map_like *= 0.35 + 0.65 * np.exp(-0.5 * (residual / 0.13) ** 2)
        return visual[:, None] * yaw_like[None, :] * map_like[:, None]

    def step(self, dr_xy, z_m, visual_xy_probability, *, relative_yaw_rad,
             visual_yaw_probability, tof_m=None) -> PosteriorState:
        dr = np.asarray(dr_xy, dtype=np.float64)
        observation = self._observation(
            visual_xy_probability, visual_yaw_probability, relative_yaw_rad, z_m, tof_m
        )
        if self.belief is None:
            self.belief = observation.copy()
        else:
            delta = dr - self.last_dr
            propagated = np.zeros_like(self.belief)
            for t, theta in enumerate(self.theta):
                c, s = math.cos(theta), math.sin(theta)
                shifted = self.cell_xy + np.array([
                    c * delta[0] - s * delta[1],
                    s * delta[0] + c * delta[1],
                ])
                xb = np.floor(shifted[:, 0] / 18.0 * 36).astype(np.int64)
                yb = np.floor(shifted[:, 1] / 7.6 * 16).astype(np.int64)
                inside = (xb >= 0) & (xb < 36) & (yb >= 0) & (yb < 16)
                destination = yb[inside] * 36 + xb[inside]
                propagated[:, t] = np.bincount(
                    destination, weights=self.belief[inside, t], minlength=576
                )
            # Small diffusion prevents cell quantization from deleting support.
            grid = propagated.reshape(16, 36, self.heading_bins)
            diffuse = (0.80 * grid + 0.05 * np.roll(grid, 1, axis=0)
                       + 0.05 * np.roll(grid, -1, axis=0)
                       + 0.05 * np.roll(grid, 1, axis=1)
                       + 0.05 * np.roll(grid, -1, axis=1))
            self.belief = diffuse.reshape(576, self.heading_bins) * observation
        self.last_dr = dr.copy()
        total = float(self.belief.sum())
        if total <= 0.0 or not math.isfinite(total):
            self.belief = observation.copy()
            total = float(self.belief.sum())
        self.belief /= max(total, 1e-300)
        return self.state()

    def state(self) -> PosteriorState:
        position = self.belief.sum(axis=1)
        best = int(np.argmax(position))
        distance = np.linalg.norm(self.cell_xy - self.cell_xy[best], axis=1)
        mask = distance <= 0.75
        mode_weight = float(position[mask].sum())
        xy = np.average(self.cell_xy[mask], axis=0, weights=position[mask])
        spread = float(np.sqrt(np.average(
            np.sum((self.cell_xy[mask] - xy) ** 2, axis=1), weights=position[mask]
        )))
        global_mean = np.average(self.cell_xy, axis=0, weights=position)
        global_spread = float(np.sqrt(np.average(
            np.sum((self.cell_xy - global_mean) ** 2, axis=1), weights=position
        )))
        heading_weight = self.belief[mask].sum(axis=0)
        heading = float(math.atan2(np.sum(heading_weight * np.sin(self.theta)),
                                   np.sum(heading_weight * np.cos(self.theta))))
        ess = 1.0 / float(np.sum(self.belief * self.belief))
        return PosteriorState(xy=xy, heading_offset=heading,
                              global_spread_m=global_spread,
                              mode_weight=mode_weight, mode_spread_m=spread,
                              effective_particles=ess, finite=np.isfinite(xy).all())


class OriginHeadingViterbiFilter:
    """Max-product path scoring over fixed origin-cell and heading hypotheses."""

    def __init__(self, office_map, *, heading_bins: int = 72,
                 visual_outlier: float = 0.08, yaw_outlier: float = 0.20,
                 offset_weight: float = 0.0, offset_sigma: float = 0.55,
                 offset_outlier: float = 0.20):
        self.om = office_map
        self.heading_bins = int(heading_bins)
        self.visual_outlier = float(visual_outlier)
        self.yaw_outlier = float(yaw_outlier)
        self.offset_weight = float(offset_weight)
        self.offset_sigma = float(offset_sigma)
        self.offset_outlier = float(offset_outlier)
        xs = (np.arange(36) + 0.5) * (18.0 / 36.0)
        ys = (np.arange(16) + 0.5) * (7.6 / 16.0)
        xx, yy = np.meshgrid(xs, ys)
        self.origin_xy = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
        self.theta = (np.arange(self.heading_bins) + 0.5) * (
            2.0 * math.pi / self.heading_bins
        ) - math.pi
        self.c = np.cos(self.theta)[None, :]
        self.s = np.sin(self.theta)[None, :]
        self.log_score = np.zeros((576, self.heading_bins), dtype=np.float64)
        self.steps = 0
        self.current_xy = None

    def step(self, dr_xy, z_m, visual_xy_probability, *, relative_yaw_rad,
             visual_yaw_probability, tof_m=None,
             visual_xy_offset=None) -> PosteriorState:
        dr = np.asarray(dr_xy, dtype=np.float64)
        offset = None
        if visual_xy_offset is not None:
            candidate_offset = np.asarray(visual_xy_offset, dtype=np.float64).reshape(-1)
            if candidate_offset.shape == (2,) and np.isfinite(candidate_offset).all():
                offset = np.clip(candidate_offset, -1.0, 1.0)
        # At the first trajectory frame, dead reckoning is zero and the pose is
        # exactly the unknown origin. Move every candidate from its cell center
        # to the network's shared within-cell coordinate. This preserves all 576
        # global modes while removing the otherwise unavoidable 0.35 m corner
        # quantization error.
        if self.steps == 0 and offset is not None and self.offset_weight > 0.0:
            self.origin_xy = self.origin_xy + offset * np.array(
                [0.5 * (18.0 / 36.0), 0.5 * (7.6 / 16.0)], dtype=np.float64
            )
        current_x = self.origin_xy[:, 0:1] + self.c * dr[0] - self.s * dr[1]
        current_y = self.origin_xy[:, 1:2] + self.s * dr[0] + self.c * dr[1]
        xb = np.floor(current_x / 18.0 * 36).astype(np.int64)
        yb = np.floor(current_y / 7.6 * 16).astype(np.int64)
        inside = (xb >= 0) & (xb < 36) & (yb >= 0) & (yb < 16)
        probability = np.asarray(visual_xy_probability, dtype=np.float64).reshape(576)
        visual = np.full_like(current_x, self.visual_outlier / 576.0)
        cell = np.clip(yb * 36 + xb, 0, 575)
        visual += (1.0 - self.visual_outlier) * probability[cell]
        visual[~inside] = self.visual_outlier / 576.0
        yaw_p = np.asarray(visual_yaw_probability, dtype=np.float64).reshape(24)
        yaw_p /= max(float(yaw_p.sum()), 1e-12)
        global_yaw = (self.theta + float(relative_yaw_rad)) % (2.0 * math.pi)
        yaw_bin = np.floor(global_yaw / (2.0 * math.pi) * 24).astype(np.int64)
        yaw_like = self.yaw_outlier / 24.0 + (1.0 - self.yaw_outlier) * yaw_p[yaw_bin]
        self.log_score += 0.70 * np.log(np.maximum(visual, 1e-12))
        self.log_score += 0.55 * np.log(np.maximum(yaw_like[None, :], 1e-12))
        if offset is not None and self.offset_weight > 0.0:
            cell_x, cell_y = 18.0 / 36.0, 7.6 / 16.0
            candidate_offset_x = (current_x - (xb + 0.5) * cell_x) / (0.5 * cell_x)
            candidate_offset_y = (current_y - (yb + 0.5) * cell_y) / (0.5 * cell_y)
            residual2 = ((candidate_offset_x - offset[0]) ** 2
                         + (candidate_offset_y - offset[1]) ** 2)
            offset_like = self.offset_outlier + (1.0 - self.offset_outlier) * np.exp(
                -0.5 * residual2 / max(self.offset_sigma ** 2, 1e-9)
            )
            offset_like[~inside] = self.offset_outlier
            self.log_score += self.offset_weight * np.log(np.maximum(offset_like, 1e-12))
        flat_xy = np.column_stack([current_x.reshape(-1), current_y.reshape(-1)])
        probe = np.column_stack([flat_xy, np.full(len(flat_xy), float(z_m))])
        clearance = np.asarray(self.om.clearance(probe)).reshape(576, self.heading_bins)
        self.log_score += np.where(clearance > 0.16, 0.0, math.log(0.02))
        if tof_m is not None and 0.0 < float(tof_m) < 50.0:
            residual = np.abs(np.asarray(self.om.predict_tof(probe)).reshape(
                576, self.heading_bins
            ) - float(tof_m))
            self.log_score += np.log(0.35 + 0.65 * np.exp(-0.5 * (residual / 0.13) ** 2))
        self.log_score -= float(np.max(self.log_score))
        self.current_xy = np.stack([current_x, current_y], axis=2)
        self.steps += 1
        return self.state()

    def state(self) -> PosteriorState:
        weight = np.exp(np.clip(self.log_score, -80.0, 0.0))
        weight /= max(float(weight.sum()), 1e-300)
        best_flat = int(np.argmax(weight))
        best_origin, best_theta = np.unravel_index(best_flat, weight.shape)
        best_xy = self.current_xy[best_origin, best_theta]
        positions = self.current_xy.reshape(-1, 2)
        flat_weight = weight.reshape(-1)
        distance = np.linalg.norm(positions - best_xy, axis=1)
        mask = distance <= 0.75
        mode_weight = float(flat_weight[mask].sum())
        xy = np.average(positions[mask], axis=0, weights=flat_weight[mask])
        mode_spread = float(np.sqrt(np.average(
            np.sum((positions[mask] - xy) ** 2, axis=1), weights=flat_weight[mask]
        )))
        global_mean = np.average(positions, axis=0, weights=flat_weight)
        global_spread = float(np.sqrt(np.average(
            np.sum((positions - global_mean) ** 2, axis=1), weights=flat_weight
        )))
        ess = 1.0 / float(np.sum(flat_weight * flat_weight))
        return PosteriorState(
            xy=xy, heading_offset=float(self.theta[best_theta]),
            global_spread_m=global_spread, mode_weight=mode_weight,
            mode_spread_m=mode_spread, effective_particles=ess,
            finite=np.isfinite(xy).all(),
        )

    @staticmethod
    def _weighted_quantile(values, weights, quantile: float) -> float:
        order = np.argsort(values)
        sorted_values = np.asarray(values)[order]
        cumulative = np.cumsum(np.asarray(weights)[order])
        index = int(np.searchsorted(cumulative, float(quantile), side="left"))
        return float(sorted_values[min(index, len(sorted_values) - 1)])

    def directional_collision_risk(
        self,
        local_velocity_xyz,
        z_m: float,
        *,
        horizon_s: float = 0.60,
        safety_margin_m: float = 0.50,
        samples: int = 5,
    ) -> DirectionalCollisionRisk:
        """Evaluate a command against every live pose/heading hypothesis.

        This does not choose a global pose. It rotates the takeoff-frame command
        under each heading hypothesis, probes the fixed map along each resulting
        segment, and reports posterior mass that enters the requested margin.
        """
        velocity = np.asarray(local_velocity_xyz, dtype=np.float64).reshape(-1)
        if (self.steps < 1 or self.current_xy is None or velocity.shape != (3,)
                or not np.isfinite(velocity).all() or not math.isfinite(float(z_m))):
            return DirectionalCollisionRisk(math.nan, math.nan, math.nan, math.nan, False)
        weight = np.exp(np.clip(self.log_score, -80.0, 0.0))
        total = float(weight.sum())
        if total <= 0.0 or not math.isfinite(total):
            return DirectionalCollisionRisk(math.nan, math.nan, math.nan, math.nan, False)
        weight = (weight / total).reshape(-1)
        vx = self.c * velocity[0] - self.s * velocity[1]
        vy = self.s * velocity[0] + self.c * velocity[1]
        minimum = np.full((576, self.heading_bins), np.inf, dtype=np.float64)
        for t in np.linspace(0.0, max(0.0, float(horizon_s)), max(2, int(samples))):
            x = self.current_xy[:, :, 0] + vx * t
            y = self.current_xy[:, :, 1] + vy * t
            z = np.full_like(x, float(z_m) + velocity[2] * t)
            probe = np.column_stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)])
            clearance = np.asarray(self.om.clearance(probe), dtype=np.float64).reshape(
                576, self.heading_bins
            )
            minimum = np.minimum(minimum, clearance)
        flat = minimum.reshape(-1)
        finite = bool(np.isfinite(flat).all())
        if not finite:
            return DirectionalCollisionRisk(math.nan, math.nan, math.nan, math.nan, False)
        return DirectionalCollisionRisk(
            danger_probability=float(weight[flat <= float(safety_margin_m)].sum()),
            clearance_p01_m=self._weighted_quantile(flat, weight, 0.01),
            clearance_p05_m=self._weighted_quantile(flat, weight, 0.05),
            expected_clearance_m=float(np.sum(weight * flat)),
            finite=True,
        )
