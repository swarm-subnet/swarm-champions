"""Map classifier extracted verbatim from the cf_autopilot champion (UID14).
Trained XGBoost over [tick, tick*dt] + state[:141] + 68 depth statistics.
Extracted so a swarm submission can vendor it without the 462 KB host file."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

SIM_DT = 1 / 50

m_STATE_DIM = 141

MAP_LABELS = ('city', 'open', 'mountain', 'village', 'warehouse', 'forest')

class _XGBMapPredictor:

    def __init__(self, model_path: Path):
        self.enabled = False
        self.num_class = 0
        self.base_score = None
        self.tree_info = []
        self.trees = []
        if not model_path.exists():
            return
        with model_path.open('r', encoding='utf-8') as handle:
            model = json.load(handle)
        learner = model['learner']
        params = learner['learner_model_param']
        self.num_class = int(params['num_class'])
        self.base_score = np.asarray(json.loads(params['base_score']), dtype=np.float32)
        booster = learner['gradient_booster']['model']
        self.tree_info = [int(v) for v in booster['tree_info']]
        self.trees = []
        for tree in booster['trees']:
            self.trees.append({'left': np.asarray(tree['left_children'], dtype=np.int32), 'right': np.asarray(tree['right_children'], dtype=np.int32), 'split_idx': np.asarray(tree['split_indices'], dtype=np.int32), 'split_cond': np.asarray(tree['split_conditions'], dtype=np.float32), 'default_left': np.asarray(tree['default_left'], dtype=np.int8), 'weights': np.asarray(tree['base_weights'], dtype=np.float32)})
        self.enabled = self.num_class == len(MAP_LABELS) and len(self.trees) == len(self.tree_info)

    def predict_proba(self, features: np.ndarray) -> np.ndarray | None:
        if not self.enabled:
            return None
        x = np.asarray(features, dtype=np.float32).reshape(-1)
        margins = self.base_score.astype(np.float32).copy()
        for tree, class_idx in zip(self.trees, self.tree_info):
            node = 0
            left = tree['left']
            right = tree['right']
            split_idx = tree['split_idx']
            split_cond = tree['split_cond']
            default_left = tree['default_left']
            weights = tree['weights']
            while left[node] != -1:
                feature_idx = int(split_idx[node])
                value = x[feature_idx] if feature_idx < x.size else np.nan
                if np.isnan(value):
                    node = int(left[node] if default_left[node] else right[node])
                elif float(value) < float(split_cond[node]):
                    node = int(left[node])
                else:
                    node = int(right[node])
            margins[class_idx] += weights[node]
        margins -= np.max(margins)
        probs = np.exp(margins)
        denom = float(np.sum(probs))
        if denom <= 1e-12:
            return None
        return (probs / denom).astype(np.float32)

def _map_depth_2d(depth: np.ndarray) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        arr = np.reshape(arr, arr.shape[:2])
    return np.clip(arr, 0.0, 1.0)

def _map_depth_feature_values(depth: np.ndarray) -> list[float]:
    d = _map_depth_2d(depth)
    gx = np.abs(np.diff(d, axis=1))
    gy = np.abs(np.diff(d, axis=0))
    percentiles = np.percentile(d, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    values = [float(d.min()), float(d.mean()), float(d.std()), float(d.max()), float(percentiles[0]), float(percentiles[1]), float(percentiles[2]), float(percentiles[3]), float(percentiles[4]), float(percentiles[5]), float(percentiles[6]), float(percentiles[7]), float(percentiles[8]), float((d <= 0.1).mean()), float((d <= 0.25).mean()), float((d >= 0.95).mean()), float((d >= 0.999).mean()), float((d <= 0.001).mean()), float(gx.mean() + gy.mean()), float(((gx > 0.05).mean() + (gy > 0.05).mean()) / 2.0)]
    h, w = d.shape
    tile_means = []
    tile_mins = []
    tile_maxs = []
    for y0 in np.linspace(0, h, 5, dtype=int)[:-1]:
        y1 = int(y0 + h // 4)
        for x0 in np.linspace(0, w, 5, dtype=int)[:-1]:
            x1 = int(x0 + w // 4)
            tile = d[y0:y1, x0:x1]
            tile_means.append(float(tile.mean()))
            tile_mins.append(float(tile.min()))
            tile_maxs.append(float(tile.max()))
    values.extend(tile_means)
    values.extend(tile_mins)
    values.extend(tile_maxs)
    return values
