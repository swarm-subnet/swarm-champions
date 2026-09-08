"""CPU-only ONNX pose proposal and conservative certification gate.

This module intentionally has no controller dependency.  It can be copied into
an agent for shadow logging first; callers must explicitly consume
``CertifiedPose.certified`` before changing a command.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import onnxruntime as ort

from posterior_filter import (
    DirectionalCollisionRisk,
    OriginHeadingViterbiFilter,
    PosteriorState,
)


XYZ_SCALE = np.array([18.0, 7.6, 3.0], dtype=np.float32)
BIN_COUNTS = (36, 16, 6, 24)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class PoseProposal:
    xyz: np.ndarray
    yaw: float
    sigma_xy_m: float
    direct_joint_disagreement_m: float
    joint_confidence: float
    fresh: bool
    top_xy_cells: np.ndarray | None = None
    top_xy_probabilities: np.ndarray | None = None
    # Multi-hypothesis output. ``candidate_xy_m`` contains a dense subcell
    # lattice for every retained map cell, not a falsely precise single pose.
    # Its associated radius is the worst-case distance to one lattice point.
    candidate_xy_m: np.ndarray | None = None
    candidate_cell_radius_m: float = 0.0
    candidate_probability_mass: float = 0.0
    xy_probability: np.ndarray | None = None
    yaw_probability: np.ndarray | None = None


@dataclass(frozen=True)
class CertifiedPose:
    proposal: PoseProposal
    certified: bool
    reason: str
    count: int


@dataclass(frozen=True)
class TemporalShadowResult:
    """Read-only localization result; callers must not treat it as authority."""

    proposal: PoseProposal
    posterior: PosteriorState | None
    risk: float
    certified: bool
    reason: str
    count: int


class VisualPoseRuntime:
    def __init__(
        self,
        model_path: str | Path,
        *,
        input_mode: str = "mixed",
        threads: int = 2,
        posterior_temperature: float = 1.4,
        # 256 cells is the smallest budget that clears 99.9% within 7.5 cm on
        # the untouched test split with the dense subcell refinement.
        candidate_top_k: int = 256,
        candidate_subcell_grid: int = 8,
    ):
        if input_mode not in ("rgb", "mixed", "grayscale"):
            raise ValueError("input_mode must be rgb, mixed, or grayscale")
        self.input_mode = input_mode
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(threads))
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        self.image_size = int(self.session.get_inputs()[0].shape[-1])
        self.posterior_temperature = float(posterior_temperature)
        self.candidate_top_k = max(1, int(candidate_top_k))
        self.candidate_subcell_grid = max(2, int(candidate_subcell_grid))
        if not math.isfinite(self.posterior_temperature) or self.posterior_temperature <= 0.0:
            raise ValueError("posterior_temperature must be positive and finite")
        self._last_signature: tuple[float, float, float, float] | None = None
        self._last_proposal: PoseProposal | None = None

    @staticmethod
    def _signature(rgb: np.ndarray) -> tuple[float, float, float, float]:
        image = np.asarray(rgb, dtype=np.float32)
        if image.shape != (256, 256, 3):
            raise ValueError(f"expected RGB (256,256,3), got {image.shape}")
        sample = image[::32, ::32]
        return (
            float(sample.mean()),
            float(sample.std()),
            float(sample[::2, ::2].mean()),
            float(sample[-1, -1, 0]),
        )

    def reset(self) -> None:
        self._last_signature = None
        self._last_proposal = None

    def _preprocess(self, rgb: np.ndarray) -> np.ndarray:
        image = np.asarray(rgb, dtype=np.float32)
        if image.shape != (256, 256, 3):
            raise ValueError(f"expected RGB (256,256,3), got {image.shape}")
        image = np.clip(image, 0.0, 1.0)
        if self.input_mode == "grayscale":
            gray = image @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
            image = np.repeat(gray[..., None], 3, axis=2)
        # Pillow is part of the evaluator image; import lazily so model startup
        # remains cheap when an agent is running in a non-vision mode.
        from PIL import Image

        image8 = np.asarray(Image.fromarray(np.rint(image * 255).astype(np.uint8)).resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        ), dtype=np.float32) / 255.0
        image8 = (image8 - IMAGENET_MEAN[None, None, :]) / IMAGENET_STD[None, None, :]
        return np.transpose(image8, (2, 0, 1))[None].astype(np.float32, copy=False)

    def predict(self, rgb: np.ndarray) -> PoseProposal:
        # Held frames are intentionally not counted as independent evidence.
        # Check this before resize/normalization/ONNX so a held 50 Hz frame is
        # effectively free between the camera's fresh 25 Hz captures.
        signature = self._signature(rgb)
        fresh = self._last_signature != signature
        if not fresh and self._last_proposal is not None:
            return replace(self._last_proposal, fresh=False)
        self._last_signature = signature
        image = self._preprocess(rgb)
        output = self.session.run(None, {"image": image})
        direct_norm, yaw_vec, log_std_norm, _clearance, _stratum, xy_offset, xy_logits = output[:7]
        direct_xyz = np.asarray(direct_norm[0], dtype=np.float32) * XYZ_SCALE
        yaw_vec = np.asarray(yaw_vec[0], dtype=np.float32)
        yaw = float(math.atan2(float(yaw_vec[0]), float(yaw_vec[1])))
        sigma_xyz = np.exp(np.asarray(log_std_norm[0], dtype=np.float32)) * XYZ_SCALE
        sigma_xy = float(np.linalg.norm(sigma_xyz[:2]))
        xy_scaled = np.asarray(xy_logits[0], dtype=np.float64) / self.posterior_temperature
        xy_prob = np.exp(xy_scaled - np.max(xy_scaled))
        xy_prob /= max(float(xy_prob.sum()), 1e-9)
        yaw_logits = np.asarray(output[10][0], dtype=np.float64) / self.posterior_temperature
        yaw_prob = np.exp(yaw_logits - np.max(yaw_logits))
        yaw_prob /= max(float(yaw_prob.sum()), 1e-9)
        joint = int(np.argmax(xy_prob))
        top_count = min(self.candidate_top_k, len(xy_prob))
        top_cells = np.argpartition(xy_prob, -top_count)[-top_count:]
        top_cells = top_cells[np.argsort(xy_prob[top_cells])[::-1]]
        top_x = top_cells % BIN_COUNTS[0]
        top_y = top_cells // BIN_COUNTS[0]
        cell = np.array([18.0 / BIN_COUNTS[0], 7.6 / BIN_COUNTS[1]], dtype=np.float32)
        g = self.candidate_subcell_grid
        centers = (np.arange(g, dtype=np.float32) + 0.5) / float(g)
        grid_x, grid_y = np.meshgrid(centers, centers, indexing="xy")
        subcell_offsets = np.column_stack([grid_x.reshape(-1), grid_y.reshape(-1)])
        candidate_bins = np.column_stack([top_x, top_y]).astype(np.float32)
        candidate_xy = (
            candidate_bins[:, None, :] + subcell_offsets[None, :, :]
        ).reshape(-1, 2) * cell
        candidate_radius = float(np.linalg.norm(0.5 * cell / float(g)))
        xbin, ybin = joint % BIN_COUNTS[0], joint // BIN_COUNTS[0]
        joint_xy = (np.array([xbin, ybin], dtype=np.float32) + 0.5) * cell
        joint_xy += np.asarray(xy_offset[0], dtype=np.float32) * (0.5 * cell)
        disagreement = float(np.linalg.norm(direct_xyz[:2] - joint_xy))
        proposal = PoseProposal(
            xyz=direct_xyz,
            yaw=yaw,
            sigma_xy_m=sigma_xy,
            direct_joint_disagreement_m=disagreement,
            joint_confidence=float(xy_prob[joint]),
            fresh=fresh,
            top_xy_cells=top_cells.astype(np.int16, copy=False),
            top_xy_probabilities=xy_prob[top_cells].astype(np.float32, copy=False),
            candidate_xy_m=candidate_xy,
            candidate_cell_radius_m=candidate_radius,
            candidate_probability_mass=float(xy_prob[top_cells].sum()),
            xy_probability=xy_prob.astype(np.float32, copy=False),
            yaw_probability=yaw_prob.astype(np.float32, copy=False),
        )
        self._last_proposal = proposal
        return proposal


class TemporalVisualShadow:
    """Fuse visual retrieval with motion and map evidence without control authority.

    The threshold is frozen from the validation split.  Requiring two fresh,
    consecutive passes is an additional live guard; this class only reports a
    candidate pose and never modifies an action.
    """

    FROZEN_RISK_THRESHOLD = 0.12741169488448215

    def __init__(
        self,
        model_path: str | Path,
        office_map,
        *,
        input_mode: str = "mixed",
        threads: int = 2,
        risk_threshold: float = FROZEN_RISK_THRESHOLD,
        consecutive: int = 2,
    ):
        self.runtime = VisualPoseRuntime(
            model_path,
            input_mode=input_mode,
            threads=threads,
            posterior_temperature=1.4,
        )
        self.office_map = office_map
        self.risk_threshold = float(risk_threshold)
        self.consecutive_required = max(1, int(consecutive))
        self.filter = OriginHeadingViterbiFilter(office_map)
        self._count = 0
        self._last: TemporalShadowResult | None = None

    def reset(self) -> None:
        self.runtime.reset()
        self.filter = OriginHeadingViterbiFilter(self.office_map)
        self._count = 0
        self._last = None

    def update(
        self,
        rgb: np.ndarray,
        dead_reckoned_xy: np.ndarray,
        relative_yaw_rad: float,
        z_m: float,
        tof_m: float | None,
    ) -> TemporalShadowResult:
        proposal = self.runtime.predict(rgb)
        if not proposal.fresh:
            posterior = self._last.posterior if self._last is not None else None
            risk = self._last.risk if self._last is not None else math.inf
            return TemporalShadowResult(
                proposal, posterior, risk, False, "held_frame", self._count
            )
        if proposal.xy_probability is None or proposal.yaw_probability is None:
            self._count = 0
            result = TemporalShadowResult(
                proposal, None, math.inf, False, "missing_posterior", 0
            )
            self._last = result
            return result
        dr_xy = np.asarray(dead_reckoned_xy, dtype=np.float64)
        inputs_finite = (
            dr_xy.shape == (2,)
            and np.isfinite(dr_xy).all()
            and math.isfinite(float(relative_yaw_rad))
            and math.isfinite(float(z_m))
        )
        if not inputs_finite:
            self._count = 0
            result = TemporalShadowResult(
                proposal, None, math.inf, False, "nonfinite_input", 0
            )
            self._last = result
            return result
        posterior = self.filter.step(
            dr_xy,
            float(z_m),
            proposal.xy_probability,
            tof_m=tof_m,
            relative_yaw_rad=float(relative_yaw_rad),
            visual_yaw_probability=proposal.yaw_probability,
        )
        disagreement = float(np.linalg.norm(proposal.xyz[:2] - posterior.xy))
        risk = (
            posterior.mode_spread_m
            + 2.0 * (1.0 - posterior.mode_weight)
            + 0.05 * posterior.global_spread_m
            + 2.0 * disagreement
        )
        eligible = bool(
            posterior.finite
            and math.isfinite(risk)
            and risk <= self.risk_threshold
        )
        self._count = self._count + 1 if eligible else 0
        certified = eligible and self._count >= self.consecutive_required
        reason = (
            "certified" if certified else
            "warming_up" if eligible else
            "risk"
        )
        result = TemporalShadowResult(
            proposal, posterior, float(risk), certified, reason, self._count
        )
        self._last = result
        return result

    def directional_collision_risk(
        self,
        local_velocity_xyz: np.ndarray,
        z_m: float,
        *,
        horizon_s: float = 0.60,
        safety_margin_m: float = 0.50,
    ) -> DirectionalCollisionRisk:
        return self.filter.directional_collision_risk(
            local_velocity_xyz,
            z_m,
            horizon_s=horizon_s,
            safety_margin_m=safety_margin_m,
        )


class PoseCertificationGate:
    """Temporal, uncertainty-aware abstention for safe shadow integration."""

    def __init__(
        self,
        *,
        max_sigma_xy_m: float = 0.20,
        max_decoder_disagreement_m: float = 0.20,
        min_joint_confidence: float = 0.02,
        consecutive: int = 3,
        max_jump_m: float = 0.35,
        max_yaw_jump_rad: float = math.radians(25.0),
    ):
        self.max_sigma_xy_m = float(max_sigma_xy_m)
        self.max_decoder_disagreement_m = float(max_decoder_disagreement_m)
        self.min_joint_confidence = float(min_joint_confidence)
        self.consecutive_required = int(consecutive)
        self.max_jump_m = float(max_jump_m)
        self.max_yaw_jump_rad = float(max_yaw_jump_rad)
        self._history: deque[PoseProposal] = deque(maxlen=max(2, consecutive))

    def reset(self) -> None:
        self._history.clear()

    def update(self, proposal: PoseProposal) -> CertifiedPose:
        if not proposal.fresh:
            return CertifiedPose(proposal, False, "held_frame", len(self._history))
        if not np.isfinite(proposal.xyz).all() or not math.isfinite(proposal.yaw):
            self.reset()
            return CertifiedPose(proposal, False, "nonfinite", 0)
        if proposal.sigma_xy_m > self.max_sigma_xy_m:
            self.reset()
            return CertifiedPose(proposal, False, "uncertainty", 0)
        if proposal.direct_joint_disagreement_m > self.max_decoder_disagreement_m:
            self.reset()
            return CertifiedPose(proposal, False, "decoder_disagreement", 0)
        if proposal.joint_confidence < self.min_joint_confidence:
            self.reset()
            return CertifiedPose(proposal, False, "low_map_confidence", 0)
        if self._history:
            previous = self._history[-1]
            if float(np.linalg.norm(proposal.xyz[:2] - previous.xyz[:2])) > self.max_jump_m:
                self.reset()
                return CertifiedPose(proposal, False, "temporal_jump", 0)
            yaw_jump = math.remainder(proposal.yaw - previous.yaw, 2.0 * math.pi)
            if abs(yaw_jump) > self.max_yaw_jump_rad:
                self.reset()
                return CertifiedPose(proposal, False, "yaw_jump", 0)
        self._history.append(proposal)
        count = len(self._history)
        return CertifiedPose(
            proposal,
            count >= self.consecutive_required,
            "certified" if count >= self.consecutive_required else "warming_up",
            count,
        )
