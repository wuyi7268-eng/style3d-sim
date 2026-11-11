"""Read-only adapter for the pinned upstream RSL-RL 5 Swing actor."""

import hashlib
import urllib.request
from pathlib import Path

import torch
from torch import nn

REVISION = "c5fcc50219fef01ac9931d0079c583ccbb29b689"
SHA256 = "49fb3599be519cb604ed65e1ba495fb9f5746e5bee1d4cb95107dcd165f47f9c"
CHECKPOINT = Path(__file__).with_name("assets") / "swing" / "alpha050.pt"
URL = (f"https://raw.githubusercontent.com/Vottivott/microduck-playground/{REVISION}/"
       "experiments/swing/checkpoints/alpha050.pt")


def pretrained_checkpoint():
    if not CHECKPOINT.exists():
        data = urllib.request.urlopen(URL, timeout=30).read()
        if hashlib.sha256(data).hexdigest() != SHA256:
            raise ValueError("Downloaded Swing checkpoint does not match the pinned upstream release")
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        CHECKPOINT.write_bytes(data)
    if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != SHA256:
        raise ValueError(f"Unexpected upstream checkpoint contents: {CHECKPOINT}")
    return CHECKPOINT


class PretrainedSwingPolicy(nn.Module):
    def __init__(self, path, device="cpu"):
        super().__init__()
        state = torch.load(path, map_location="cpu", weights_only=True)["actor_state_dict"]
        self.register_buffer("mean", state["obs_normalizer._mean"])
        self.register_buffer("std", state["obs_normalizer._std"])
        if self.mean.shape != (1, 61) or self.std.shape != (1, 61) or (self.std < 0).any():
            raise ValueError("Unexpected Swing observation normalizer")
        dims = (61, 512, 256, 128, 14)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ELU())
        self.mlp = nn.Sequential(*layers)
        weights = {key.removeprefix("mlp."): value for key, value in state.items() if key.startswith("mlp.")}
        self.mlp.load_state_dict(weights)
        if not all(torch.isfinite(value).all() for value in self.state_dict().values()):
            raise ValueError("Non-finite Swing checkpoint")
        self.to(device).eval().requires_grad_(False)

    def forward(self, observations):
        # RSL-RL 5.0.1 uses std + 0.01, not sqrt(var + epsilon). Never update release statistics.
        return self.mlp((observations - self.mean) / (self.std + 0.01)).clamp(-1, 1)
