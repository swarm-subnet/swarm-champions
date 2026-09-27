from __future__ import annotations
from pathlib import Path
import numpy as np
MAP_IDS = (1, 2, 3, 4, 6)
NAMES = {"city": 1, "open": 2, "mountain": 3, "village": 4, "forest": 6}
BOUNDS = {1: 75.0, 2: 60.0, 3: 120.0, 4: 40.0, 6: 42.0}
MAX_N = 8
SCHEMA = "uid134.team_route_policy.v1"
class RoutePlanner:
    def __init__(self, checkpoint: str, device: str = "cpu"):
        import torch
        from torch import nn
        p = Path(checkpoint)
        if p.is_dir():                       
            p = p / "best.pt"
        art = torch.load(str(p), map_location=device, weights_only=True)
        if art.get("schema") != SCHEMA:
            raise RuntimeError("route checkpoint schema %r, expected %r"
                               % (art.get("schema"), SCHEMA))
        if art.get("runtime_uses_privileged_inputs") is not False:
            raise RuntimeError("route checkpoint does not declare "
                               "runtime_uses_privileged_inputs=False")
        cfg = art["model_config"]
        hidden, heads = int(cfg["hidden"]), int(cfg["heads"])
        waypoints = int(cfg["waypoints"])
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.waypoints = waypoints
                self.start_encoder = nn.Sequential(
                    nn.Linear(10, hidden), nn.SiLU(), nn.LayerNorm(hidden),
                    nn.Linear(hidden, hidden), nn.SiLU(),
                )
                layer = nn.TransformerEncoderLayer(
                    hidden, heads, dim_feedforward=hidden * 3, dropout=0.05,
                    activation="gelu", batch_first=True, norm_first=True,
                )
                self.team = nn.TransformerEncoder(layer, num_layers=2)
                self.step = nn.Embedding(waypoints, hidden // 2)
                self.decoder = nn.Sequential(
                    nn.Linear(hidden * 2 + hidden // 2, hidden * 2), nn.SiLU(),
                    nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden),
                    nn.SiLU(), nn.Linear(hidden, 2),
                )
            def forward(self, starts, clue, n, map_index, bound):
                batch, drones, _ = starts.shape
                mask = torch.arange(drones, device=starts.device)[None] < n[:, None]
                map_oh = nn.functional.one_hot(map_index, 5).to(starts.dtype)
                rel = (starts[..., :2] - clue[:, None, :2]) / bound[:, None, None]
                features = torch.cat((
                    rel, starts[..., 2:3] / 80.0,
                    clue[:, None, 2:3].expand(-1, drones, -1) / 80.0,
                    (n.to(starts.dtype) / 8.0)[:, None, None].expand(-1, drones, -1),
                    map_oh[:, None].expand(-1, drones, -1),
                ), dim=-1)
                encoded = self.team(self.start_encoder(features),
                                    src_key_padding_mask=~mask)
                pooled = ((encoded * mask[..., None]).sum(1)
                          / mask.sum(1, keepdim=True))
                step = self.step.weight[None, None].expand(batch, drones, -1, -1)
                decoded = self.decoder(torch.cat((
                    encoded[:, :, None].expand(-1, -1, waypoints, -1),
                    pooled[:, None, None].expand(-1, drones, waypoints, -1), step,
                ), dim=-1))
                return (clue[:, None, None, :2]
                        + torch.tanh(decoded) * bound[:, None, None, None])
        self.torch = torch
        self.device = torch.device(device)
        self.net = Net().to(self.device).eval()
        self.net.load_state_dict(art["model"], strict=True)
        self.meta = {k: v for k, v in art.items() if k != "model"}
    def plan(self, starts, clue, n: int, map_kind: str):
        torch = self.torch
        kind = NAMES[map_kind]
        padded = np.zeros((1, MAX_N, 3), np.float32)
        s = np.asarray(starts, np.float32)
        m = min(int(n), MAX_N, len(s))
        padded[0, :m] = s[:m]
        c = np.asarray(clue, np.float32).reshape(1, 3).copy()
        padded[..., 2] = 0.0                 
        c[..., 2] = 0.0
        with torch.inference_mode():
            route = self.net(
                torch.from_numpy(padded).to(self.device),
                torch.from_numpy(c).to(self.device),
                torch.tensor([m], dtype=torch.long, device=self.device),
                torch.tensor([MAP_IDS.index(kind)], dtype=torch.long,
                             device=self.device),
                torch.tensor([BOUNDS[kind]], dtype=torch.float32,
                             device=self.device),
            )
        return route[0, :m].cpu().numpy()
