from __future__ import annotations

"""Runtime-only adapter for the map-ID-free UID134 goal posterior."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

GRID_SIZE = 45
GRID_HALF = 110.0

class GoalPosterior(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.depth = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(48, 64), nn.SiLU(),
        )
        self.state = nn.Sequential(nn.Linear(state_dim, 96), nn.SiLU(), nn.LayerNorm(96))
        layer = nn.TransformerEncoderLayer(
            d_model=160, nhead=5, dim_feedforward=320, dropout=0.08,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.team = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(161, 384), nn.SiLU(), nn.LayerNorm(384),
            nn.Dropout(0.08), nn.Linear(384, GRID_SIZE * GRID_SIZE),
        )

    def forward(self, state, depth_u8, mask):
        batch, drones = state.shape[:2]
        vision = self.depth(depth_u8.float().reshape(batch * drones, 1, 32, 32) / 255.0)
        vision = vision.reshape(batch, drones, -1)
        token = torch.cat((self.state(state), vision), dim=-1)
        token = self.team(token, src_key_padding_mask=~mask)
        pooled = (token * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return self.head(torch.cat((pooled, mask.float().sum(1, keepdim=True) / 8.0), dim=-1))

class ActorVisibleGoalPosterior:
\
\
\
\
\

    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        self.device = torch.device(device)
        artifact = torch.load(checkpoint, map_location=self.device, weights_only=False)
        if artifact.get("schema") != "uid134.actor_visible_goal_posterior.v1":
            raise RuntimeError("wrong goal posterior checkpoint")
        if not artifact.get("admitted_for_planner_experiment", False):
            raise RuntimeError("posterior did not pass sealed admission gate")
        self.columns = np.asarray(artifact["state_columns"], np.int64)
        self.mean = artifact["state_mean"].float().to(self.device)
        self.std = artifact["state_std"].float().to(self.device)
        self.grid_size = int(artifact["grid_size"])
        self.grid_half = float(artifact["grid_half_extent_m"])
        self.model = GoalPosterior(len(self.columns)).to(self.device)
        self.model.load_state_dict(artifact["model_state"], strict=True)
        self.model.eval().requires_grad_(False)
        self.reset()

    def reset(self):
        self._state = None
        self._depth = None
        self._logits = None

    def observe_reset(self, observation) -> None:
        if self._state is not None:
            return
        state = np.asarray(observation["state"], np.float32).reshape(-1, 190)
        depth = np.asarray(observation["depth"], np.float32)
        if depth.ndim == 4:
            depth = depth[..., 0]
        n = len(state)
        state_pad = np.zeros((8, len(self.columns)), np.float32)
        state_pad[:n] = np.take(state, self.columns, axis=1)
        depth32 = depth.reshape(n, 32, 4, 32, 4).mean((2, 4))
        depth_pad = np.zeros((8, 32, 32), np.uint8)
        depth_pad[:n] = np.rint(np.clip(depth32, 0.0, 1.0) * 255).astype(np.uint8)
        self._state = state_pad
        self._depth = depth_pad
        self._n = n

    def _predict(self):
        if self._logits is not None:
            return
        if self._state is None:
            raise RuntimeError("reset observation not supplied")
        state = torch.from_numpy(self._state).to(self.device)
        state = (state - self.mean) / self.std
        depth = torch.from_numpy(self._depth).to(self.device)
        mask = torch.arange(8, device=self.device) < self._n
        with torch.inference_mode():
            value = self.model(state[None], depth[None], mask[None])[0]
        self._logits = value.float().cpu().numpy().reshape(self.grid_size, self.grid_size)

    def score(self, starts, found, clue, xy, map_kind=None):
        del starts, found, map_kind
        self._predict()
        candidates = np.asarray(xy, np.float32).reshape(-1, 2)
        relative = candidates - np.asarray(clue, np.float32)[:2]
        unit = (relative + self.grid_half) / (2.0 * self.grid_half) * (self.grid_size - 1)
        x = unit[:, 0]; y = unit[:, 1]
        outside = (x < 0) | (x > self.grid_size - 1) | (y < 0) | (y > self.grid_size - 1)
        x = np.clip(x, 0, self.grid_size - 1); y = np.clip(y, 0, self.grid_size - 1)
        x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
        x1 = np.minimum(x0 + 1, self.grid_size - 1)
        y1 = np.minimum(y0 + 1, self.grid_size - 1)
        wx = x - x0; wy = y - y0
        grid = self._logits
        value = ((1 - wx) * (1 - wy) * grid[y0, x0]
                 + wx * (1 - wy) * grid[y0, x1]
                 + (1 - wx) * wy * grid[y1, x0]
                 + wx * wy * grid[y1, x1])
        value[outside] = -40.0
        return value.astype(np.float32)
