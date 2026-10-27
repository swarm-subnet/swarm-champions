import copy
import importlib.util
from pathlib import Path

import numpy as np

_DIR = Path(__file__).resolve().parent


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, str(_DIR / fn))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_A = _load("swsar_agent_a", "agent_a.py")
_B = _load("swsar_agent_b", "agent_b.py")
_PICK_TICK = 12


def _states(obs):
    st = np.asarray(obs["state"], np.float32)
    return st.reshape(1, -1) if st.ndim == 1 else st.reshape(-1, st.shape[-1])


def _depth_label(obs, n):
    dp = np.asarray(obs["depth"], np.float32).reshape(n, _B.DEPTH_RES, _B.DEPTH_RES, 1)
    f = np.mean([np.asarray(_B._map_depth_feature_values(dp[i]), np.float64) for i in range(n)], axis=0)
    return _B._MAPSW_MAPS[int(np.argmax(((f - _B._MAPSW_MU) / _B._MAPSW_SD) @ _B._MAPSW_W + _B._MAPSW_B))]


def _wants_b(label, n, z0):
    if n in (5, 6):
        return label == "forest"
    if n == 2:
        return label != "forest" and float(np.max(z0)) >= 12.0 and (label == "mountain" or float(np.min(z0)) >= 4.5)
    return False


class DroneFlightController:
    def __init__(self):
        self._a = _A.DroneFlightController()
        self._b = _B.DroneFlightController()
        self._pick = None
        self._t = 0
        self._same = True
        self._z0 = None

    def reset(self):
        self._pick = None
        self._t = 0
        self._same = True
        self._z0 = None
        r = self._a.reset()
        self._b.reset()
        return r

    def act(self, observation):
        if self._pick is not None:
            return self._pick.act(observation)
        st = _states(observation)
        n = st.shape[0]
        if self._z0 is None:
            self._z0 = st[:, 2].astype(np.float64).copy()
        if self._t >= _PICK_TICK:
            wb = self._same and _wants_b(_depth_label(observation, n), n, self._z0)
            self._pick = self._b if wb else self._a
            return self._pick.act(observation)
        ob = copy.deepcopy(observation)
        a = self._a.act(observation)
        b = self._b.act(ob)
        if self._same and not np.array_equal(np.asarray(a, np.float32), np.asarray(b, np.float32)):
            self._same = False
        self._t += 1
        return a
