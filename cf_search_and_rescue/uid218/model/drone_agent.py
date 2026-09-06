from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "policy.onnx"


class DroneFlightController:
    def __init__(
        self,
        *,
        model_path: Optional[Path] = None,
        providers: Optional[list[str]] = None,
    ):
        model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            str(model_path), options, providers=providers or ["CPUExecutionProvider"]
        )
        self._memory_size = next(
            int(i.shape[0]) for i in self.session.get_inputs() if i.name == "memory_tensor"
        )
        self.reset()

    def act(self, observation) -> np.ndarray:
        action, memory = self.session.run(
            ["action", "memory_tensor_out"],
            {
                "depth": np.asarray(observation["depth"], dtype=np.float32).reshape(256, 256, 1),
                "state": np.asarray(observation["state"], dtype=np.float32).reshape(-1),
                "memory_tensor": self.memory_tensor,
            },
        )
        self.memory_tensor = np.asarray(memory, dtype=np.float32).reshape(self._memory_size)
        return np.asarray(action, dtype=np.float32).reshape(-1)

    def reset(self) -> None:
        self.memory_tensor = np.zeros(self._memory_size, dtype=np.float32)
