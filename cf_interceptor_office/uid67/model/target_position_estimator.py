"""Standalone, causal target-position estimator for Interceptor Office.

This module is intentionally not wired into ``drone_agent.py``.  It consumes
only the controller-visible state vector and returns a point, a calibrated 3-D
error radius, and two reliability flags.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


DT = 0.02
FOCAL_NOMINAL_PX = 128.0
IMAGE_SIZE_PX = 256.0
TARGET_WIDTH_M = 0.18
TARGET_HEIGHT_M = 0.08
CAMERA_FORWARD_M = 0.13
CAMERA_UP_M = 0.05
MAX_TARGET_SPEED_MPS = 2.20

# 99.95% calibration quantile on the first 40 complete seeds.  On the 60
# untouched validation seeds, this radius covered 99.9031% of 52,633 frames.
CALIBRATED_RADIUS_MULTIPLIER = 1.2650482776227983


@dataclass(frozen=True)
class TargetPositionEstimate:
    position_takeoff_m: np.ndarray | None
    radius_3d_m: float
    track_valid: bool
    reliable_for_pursuit: bool
    exact_7_5cm: bool
    detector_age_s: float
    selected_box_slot: int | None
    mode_probability: tuple[float, float]


def _to_takeoff(forward: float, right: float, up: float, yaw: float) -> np.ndarray:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return np.array(
        [forward * cosine + right * sine, forward * sine - right * cosine, up],
        dtype=np.float64,
    )


def _remove_camera_tilt(vector, pitch: float, roll: float) -> tuple[float, float, float]:
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    x, y, z = vector
    return (
        cp * x + sp * sr * y + sp * cr * z,
        cr * y - sr * z,
        -sp * x + cp * sr * y + cp * cr * z,
    )


def _ego_velocity(state: np.ndarray) -> np.ndarray:
    yaw = math.atan2(float(state[2]), float(state[3]))
    return _to_takeoff(float(state[4]), float(state[5]), -float(state[6]), yaw)


def _measurement_covariance(
    position: np.ndarray, size_disagreement: float, scale: float = 1.0
) -> np.ndarray:
    distance = max(float(np.linalg.norm(position)), 0.25)
    direction = position / distance
    radial_sigma = scale * distance * (0.065 + 0.035 * min(size_disagreement, 1.5))
    transverse_sigma = scale * (0.055 + 0.006 * distance)
    return (
        transverse_sigma**2 * np.eye(3)
        + (radial_sigma**2 - transverse_sigma**2) * np.outer(direction, direction)
        + np.eye(3) * 1e-5
    )


def _clip_target_velocity(state: np.ndarray) -> None:
    speed = float(np.linalg.norm(state[3:6]))
    if speed > MAX_TARGET_SPEED_MPS:
        state[3:6] *= MAX_TARGET_SPEED_MPS / speed


class _VisibleMeasurementDecoder:
    def __init__(self) -> None:
        self.time = 0.0
        self.attitude = deque([(0.0, 0.0, 0.0, 0.0)], maxlen=48)
        self.last_capture_time = -99.0

    def _attitude_at(self, when: float) -> tuple[float, float, float]:
        value = min(self.attitude, key=lambda item: abs(item[0] - when))
        return value[1], value[2], value[3]

    def step(self, state: np.ndarray):
        self.time += DT
        pitch, roll = float(state[0]), float(state[1])
        yaw = math.atan2(float(state[2]), float(state[3]))
        self.attitude.append((self.time - float(state[13]), pitch, roll, yaw))
        age = max(float(state[16]), 0.0)
        capture_time = self.time - age
        fresh = capture_time > self.last_capture_time + 0.5 * DT
        if fresh:
            self.last_capture_time = capture_time

        measurements = []
        if fresh:
            capture_pitch, capture_roll, capture_yaw = self._attitude_at(capture_time)
            count = min(int(round(float(state[15]))), 2)
            for slot in range(count):
                cx, cy, width, height, confidence = map(
                    float, state[17 + 5 * slot : 22 + 5 * slot]
                )
                if width <= 1e-6 or height <= 1e-6:
                    continue
                depth_from_width = FOCAL_NOMINAL_PX * TARGET_WIDTH_M / (
                    width * IMAGE_SIZE_PX
                )
                depth_from_height = FOCAL_NOMINAL_PX * TARGET_HEIGHT_M / (
                    height * IMAGE_SIZE_PX
                )
                depth = math.sqrt(max(depth_from_width * depth_from_height, 1e-8))
                image_x = (cx - 0.5) * IMAGE_SIZE_PX / FOCAL_NOMINAL_PX
                image_y = (cy - 0.5) * IMAGE_SIZE_PX / FOCAL_NOMINAL_PX
                forward, left, up = _remove_camera_tilt(
                    (
                        depth + CAMERA_FORWARD_M,
                        -image_x * depth,
                        -image_y * depth + CAMERA_UP_M,
                    ),
                    capture_pitch,
                    capture_roll,
                )
                position = _to_takeoff(forward, -left, up, capture_yaw)
                measurements.append(
                    {
                        "slot": slot,
                        "capture_time": capture_time,
                        "age": age,
                        "position_at_capture": position,
                        "confidence": confidence,
                        "range": float(np.linalg.norm(position)),
                        "size_disagreement": abs(
                            math.log(max(depth_from_width, 1e-8) / max(depth_from_height, 1e-8))
                        ),
                    }
                )
        return _ego_velocity(state), measurements, fresh


class _CartesianKalmanPDA:
    def __init__(self, process_acceleration: float) -> None:
        self.decoder = _VisibleMeasurementDecoder()
        self.process_acceleration = process_acceleration
        self.state: np.ndarray | None = None
        self.covariance: np.ndarray | None = None
        self.age = 99.0
        self.hits = 0
        self.valid = False

    def _predict(self, ego_velocity: np.ndarray) -> None:
        if self.state is None:
            return
        transition = np.eye(6)
        transition[:3, 3:6] = np.eye(3) * DT
        self.state = transition @ self.state
        self.state[:3] -= ego_velocity * DT
        acceleration = self.process_acceleration
        process = np.zeros((6, 6))
        process[:3, :3] = np.eye(3) * acceleration * DT**3 / 3.0
        process[:3, 3:6] = process[3:6, :3] = (
            np.eye(3) * acceleration * DT**2 / 2.0
        )
        process[3:6, 3:6] = np.eye(3) * acceleration * DT
        self.covariance = transition @ self.covariance @ transition.T + process

    def step(self, visible_state: np.ndarray):
        ego_velocity, measurements, fresh = self.decoder.step(visible_state)
        self._predict(ego_velocity)
        self.age += DT
        selected_slot = None
        if fresh and measurements:
            if self.state is None:
                measurement = max(
                    measurements,
                    key=lambda item: item["confidence"] - 0.12 * item["size_disagreement"],
                )
                position = measurement["position_at_capture"] - ego_velocity * measurement["age"]
                self.state = np.r_[position, np.zeros(3)]
                noise = _measurement_covariance(position, measurement["size_disagreement"], 1.3)
                self.covariance = np.zeros((6, 6))
                self.covariance[:3, :3] = noise * 2.0
                self.covariance[3:6, 3:6] = np.eye(3) * 1.5**2
                self.hits = 1
                self.age = 0.0
                selected_slot = measurement["slot"]
            else:
                observation = np.zeros((3, 6))
                observation[:, :3] = np.eye(3)
                candidates = []
                for measurement in measurements:
                    position = measurement["position_at_capture"] + (
                        self.state[3:6] - ego_velocity
                    ) * measurement["age"]
                    noise = _measurement_covariance(position, measurement["size_disagreement"])
                    residual = position - self.state[:3]
                    innovation = observation @ self.covariance @ observation.T + noise
                    mahalanobis = float(residual @ np.linalg.solve(innovation, residual))
                    gate = max(1.0, 0.24 * float(np.linalg.norm(position)))
                    if float(np.linalg.norm(residual)) <= gate:
                        cost = mahalanobis - 0.35 * math.log(max(measurement["confidence"], 0.1))
                        candidates.append((cost, measurement, noise, residual, innovation))
                if candidates:
                    _, winner, noise, residual, innovation = min(candidates, key=lambda item: item[0])
                    gain = self.covariance @ observation.T @ np.linalg.inv(innovation)
                    self.state += gain @ residual
                    stable = np.eye(6) - gain @ observation
                    self.covariance = (
                        stable @ self.covariance @ stable.T + gain @ noise @ gain.T
                    )
                    self.covariance = 0.5 * (self.covariance + self.covariance.T)
                    _clip_target_velocity(self.state)
                    selected_slot = winner["slot"]
                    self.age = 0.0
                    self.hits += 1
                    self.valid = self.hits >= 2
        if self.age > 0.75:
            self.valid = False
            if self.age > 1.6:
                self.state = None
                self.covariance = None
                self.hits = 0
        radius = (
            3.37
            * math.sqrt(max(float(np.linalg.eigvalsh(self.covariance[:3, :3])[-1]), 0.0))
            if self.covariance is not None
            else math.inf
        )
        position = self.state[:3].copy() if self.state is not None and self.valid else None
        return position, selected_slot, self.age, radius


class ReliableTargetPositionEstimator:
    """Maneuver-aware dual-Kalman tracker with empirical reliability flags."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.filters = (_CartesianKalmanPDA(1.0), _CartesianKalmanPDA(8.0))
        self.mode_probability = np.array([0.78, 0.22], dtype=np.float64)

    def step(self, state: Sequence[float] | np.ndarray) -> TargetPositionEstimate:
        visible_state = np.asarray(state, dtype=np.float64)
        if visible_state.ndim != 1 or visible_state.size < 27:
            raise ValueError("state must be a one-dimensional controller state vector")

        values = [tracker.step(visible_state) for tracker in self.filters]
        self.mode_probability = self.mode_probability @ np.array(
            [[0.965, 0.035], [0.12, 0.88]], dtype=np.float64
        )
        if all(tracker.state is not None for tracker in self.filters):
            disagreement = float(
                np.linalg.norm(self.filters[0].state[:3] - self.filters[1].state[:3])
            )
            maneuver_evidence = min(0.95, disagreement / 0.65)
            likelihood = np.array(
                [max(0.05, 1.0 - maneuver_evidence), 0.15 + maneuver_evidence]
            )
            self.mode_probability *= likelihood
            self.mode_probability /= self.mode_probability.sum()

        available = [index for index, value in enumerate(values) if value[0] is not None]
        if not available:
            return TargetPositionEstimate(
                None, math.inf, False, False, False, 99.0, None,
                tuple(map(float, self.mode_probability)),
            )
        weights = self.mode_probability[available].copy()
        weights /= weights.sum()
        position = sum(weights[j] * values[index][0] for j, index in enumerate(available))
        raw_radius = max(values[index][3] for index in available)
        radius = CALIBRATED_RADIUS_MULTIPLIER * raw_radius
        strongest = available[int(np.argmax(weights))]
        selected_slot = values[strongest][1]
        age = min(values[index][2] for index in available)
        pursuit_limit = max(0.35, 0.12 * float(np.linalg.norm(position)))
        return TargetPositionEstimate(
            position_takeoff_m=np.asarray(position, dtype=np.float64),
            radius_3d_m=float(radius),
            track_valid=True,
            reliable_for_pursuit=bool(radius <= pursuit_limit),
            exact_7_5cm=bool(radius <= 0.075),
            detector_age_s=float(age),
            selected_box_slot=None if selected_slot is None else int(selected_slot),
            mode_probability=tuple(map(float, self.mode_probability)),
        )


__all__ = [
    "CALIBRATED_RADIUS_MULTIPLIER",
    "ReliableTargetPositionEstimator",
    "TargetPositionEstimate",
]
