from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent, _HERE.parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.environ.setdefault("SWARM_BASE", "uid167")
ACTION_DIM = 5
class DroneFlightController:
    def __init__(self):
        import torch
        torch.set_num_threads(int(os.environ.get("SWARM_TORCH_THREADS", "1")))
        from team.deploy_autopilot.base import build_agent
        self._agent = _with_overrides(build_agent, 
            general_ckpt="team/out/padgeneral_hn2_net",
            village_ckpt="team/out/padvillage_hn_net",
            village_thr=0.65,
            village_min_hits=4,
            min_hits_to_claim=3,
            min_hits_by_map=("city", 2),
            commit_t_by_map=("open", 25.0, "city", 25.0),
            device=os.environ.get("SWARM_DEVICE", "cpu"),
        )
        _install_ours(self._agent)
        from team.route_replan.adapter import install as _install_replan
        _install_replan(self._agent)
        self._agent.reset()
    def reset(self) -> None:
        self._agent.reset()
    def act(self, observation) -> np.ndarray:
        state = np.asarray(observation["state"])
        n = int(state.shape[0]) if state.ndim == 2 else 1
        try:
            action = self._agent.act(observation)
            action = np.asarray(action, dtype=np.float32).reshape(n, ACTION_DIM)
            if not np.all(np.isfinite(action)):
                action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
            return np.clip(action, -1.0, 1.0)
        except Exception:  
            return np.zeros((n, ACTION_DIM), dtype=np.float32)
_OURS = {"onnx": "team/out/ours/best.onnx", "pt": "team/out/ours/best.pt", "checker": "team/out/ours/checker.npz", "score_thr": 0.9, "threads": 2, "forest_heat_thr": -1.0, "slot_thr": {}}
_OVERRIDES = {'commit_t_skip': ('forest', 'village', 'mountain'),
              'mass_planner': False,
              'route_plan_maps': ('mountain', 'city', 'open', 'village', 'forest')}
def _with_overrides(fn, **kw):
    kw.update(_OVERRIDES)
    return fn(**kw)
def _load_ours():
    import os
    from team.detector.ours.pad_detect import PadDetector
    here = os.path.dirname(os.path.abspath(__file__))
    ck = os.path.join(here, _OURS["onnx"])
    chk = os.path.join(here, _OURS["checker"]) if _OURS["checker"] else None
    threads = int(os.environ.get("SWARM_TORCH_THREADS", _OURS["threads"]))
    try:
        det = PadDetector(ck, device="cpu", checker=chk, threads=threads)
    except Exception:  
        det = PadDetector(os.path.join(here, _OURS["pt"]), device="cpu", checker=chk,
                          threads=threads)
    det.cfg.score_thr = float(_OURS["score_thr"])
    return det
def _install_ours(agent):
    det = _load_ours()
    forest = det
    if float(_OURS.get("forest_heat_thr", -1.0)) >= 0.0:
        import os
        from team.detector.ours.pad_detect import PadDetector
        here = os.path.dirname(os.path.abspath(__file__))
        threads = int(os.environ.get("SWARM_TORCH_THREADS", _OURS["threads"]))
        forest = PadDetector(os.path.join(here, _OURS["onnx"]), device="cpu", threads=threads)
        forest.cfg.score_thr = float(_OURS["forest_heat_thr"])
    per_slot = dict(forest_detector=forest)
    for slot, thr in (_OURS.get("slot_thr") or {}).items():
        d2 = _load_ours()
        d2.cfg.score_thr = float(thr)
        per_slot[slot] = d2
    for slot in ('general_detector', 'detector', 'forest_detector', 'village_detector', 'mountain_detector', 'city_detector'):
        setattr(agent, slot, per_slot.get(slot, det))
    agent.reset()          
    return agent
