from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from uid134_base_agent import DroneFlightController as UID134Controller
from route_net import RoutePlanner
from goal_posterior_runtime import ActorVisibleGoalPosterior


class DroneFlightController:
    """Mountain-v12 plus only independently selected posterior map lanes."""

    def __init__(self):
        selection_path = _HERE / "goal_posterior_selection.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("schema") != "uid134.goalposterior.map_selection.v1":
            raise RuntimeError("invalid goal-posterior map selection")
        maps = tuple(str(name) for name in selection.get("maps", ()))
        allowed = {"city", "open", "village", "forest"}
        if not maps or len(set(maps)) != len(maps) or not set(maps) <= allowed:
            raise RuntimeError("empty, duplicate, or unknown selected map")

        self._base = UID134Controller()
        agent = self._base._agent
        route_ckpt = _HERE / "team" / "out" / "uid134_route_policy" / "best.pt"
        self._route = RoutePlanner(str(route_ckpt), device="cpu")
        agent.route_planner = self._route
        agent.cfg.route_planner = True
        agent.cfg.route_plan_maps = ("mountain",)
        agent.cfg.route_start_sec = 0.0

        posterior_ckpt = _HERE / "team" / "out" / "uid134_goal_posterior" / "best.pt"
        self._posterior = ActorVisibleGoalPosterior(posterior_ckpt, device="cpu")
        agent.posterior = self._posterior
        agent.cfg.mass_planner = True
        agent.cfg.mass_plan_maps = maps
        agent.cfg.mass_plan_n = (2, 3, 4, 5, 6, 7, 8)
        agent.cfg.mass_plan_n_by_map = {}

    def reset(self) -> None:
        self._posterior.reset()
        self._base.reset()

    def act(self, observation) -> np.ndarray:
        self._posterior.observe_reset(observation)
        return self._base.act(observation)
