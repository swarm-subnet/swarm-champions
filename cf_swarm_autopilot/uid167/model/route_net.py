from __future__ import annotations

import numpy as np

class RoutePlanner:
    MAPS = (1, 2, 3, 4, 6)
    NAMES = {"city": 1, "open": 2, "mountain": 3,
             "village": 4, "forest": 6}
    BOUNDS = {1: 75.0, 2: 60.0, 3: 120.0, 4: 40.0, 6: 42.0}

    def __init__(self, checkpoint: str, device: str = "cpu"):
        import torch
        from torch import nn

        artifact = torch.load(checkpoint, map_location=device, weights_only=True)
        if (artifact.get("schema") != "uid134.team_route_policy.v1"
                or artifact.get("runtime_uses_privileged_inputs") is not False):
            raise RuntimeError("invalid route policy artifact")
        config = artifact["model_config"]
        hidden = int(config["hidden"]); heads = int(config["heads"])
        waypoints = int(config["waypoints"])

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
                    nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.SiLU(),
                    nn.Linear(hidden, 2),
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
                return clue[:, None, None, :2] + torch.tanh(decoded) * bound[:, None, None, None]

        self.torch = torch
        self.device = torch.device(device)
        self.net = Net().to(self.device).eval()
        self.net.load_state_dict(artifact["model"], strict=True)

    def plan(self, starts, clue, n: int, map_kind: str):
        torch = self.torch
        kind = self.NAMES[map_kind]
        padded = np.zeros((1, 8, 3), np.float32)
        padded[0, :n] = np.asarray(starts, np.float32)[:n]
        clue = np.asarray(clue, np.float32).reshape(1, 3)

        padded[..., 2] = 0.0
        clue[..., 2] = 0.0
        with torch.inference_mode():
            route = self.net(
                torch.from_numpy(padded).to(self.device),
                torch.from_numpy(clue).to(self.device),
                torch.tensor([n], dtype=torch.long, device=self.device),
                torch.tensor([self.MAPS.index(kind)], dtype=torch.long,
                             device=self.device),
                torch.tensor([self.BOUNDS[kind]], dtype=torch.float32,
                             device=self.device),
            )
        return route[0, :n].cpu().numpy()
