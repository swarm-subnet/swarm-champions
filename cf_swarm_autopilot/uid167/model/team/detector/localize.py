\
\
\
\
\
\
\
\
\
\
\

from __future__ import annotations

import numpy as np

CAM_FWD = 0.13

CAM_UP = 0.05

FOV_DEG = 90.0

RES = 256

DEPTH_MIN_M = 0.5

DEPTH_MAX_M = 30.0

def rot_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])

def depth_to_meters(n, depth_max: float = DEPTH_MAX_M, depth_min: float = DEPTH_MIN_M):
\
\
\
\
\
\

    return np.asarray(n, float) * (float(depth_max) - float(depth_min)) + float(depth_min)

