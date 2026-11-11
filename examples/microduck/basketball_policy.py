"""Verified release ONNX inference with explicit, per-world LSTM memory."""

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import torch

REPOSITORY = "HannesVonEssen/microduck-basketball"
ONNX_SHA256 = "e105148b160b3a86170621648215be38dd5018e955930ce182394f4513c7569b"
CACHE = Path(__file__).resolve().parents[2] / "logs/pretrained/microduck-basketball/policy.onnx"


def initialize_actor_from_onnx(runner, path):
    """Warm-start the actor from b11; the local critic and optimizer remain fresh."""
    import onnx
    from onnx import numpy_helper

    path = pretrained_policy(path)
    graph = onnx.load(str(path)).graph
    tensors = {item.name: torch.from_numpy(numpy_helper.to_array(item).copy()) for item in graph.initializer}
    lstm = next(node for node in graph.node if node.op_type == "LSTM")
    divide = next(node for node in graph.node if node.op_type == "Div")
    policy = runner.alg.policy
    # ONNX orders LSTM gates i/o/f/c, while PyTorch expects i/f/g/o.
    def reorder(value):
        gates = value.chunk(4, dim=0)
        return torch.cat([gates[i] for i in (0, 2, 3, 1)])

    bias_ih, bias_hh = tensors[lstm.input[3]][0].chunk(2)
    policy.memory_a.rnn.load_state_dict({
        "weight_ih_l0": reorder(tensors[lstm.input[1]][0]),
        "weight_hh_l0": reorder(tensors[lstm.input[2]][0]),
        "bias_ih_l0": reorder(bias_ih), "bias_hh_l0": reorder(bias_hh),
    })
    policy.actor.load_state_dict({key.removeprefix("mlp."): value for key, value in tensors.items()
                                  if key.startswith("mlp.")})
    normalizer = runner.obs_normalizer
    std = tensors[divide.input[1]] - normalizer.eps
    if std.shape != (1, 61) or (std < 0).any():
        raise ValueError("Unexpected b11 observation normalization")
    normalizer.load_state_dict({"_mean": tensors["obs_normalizer._mean"], "_std": std,
                                "_var": std.square(), "count": torch.tensor(0, dtype=torch.long)})
    # Keep the normalization used by the release, including after runner.train_mode().
    normalizer.until = 0
    return {"kind": "b11_onnx_actor_warm_start", "path": str(path.resolve()), "sha256": ONNX_SHA256,
            "critic": "fresh", "optimizer": "fresh", "actor_normalization": "frozen"}


def pretrained_policy(path=None):
    path = Path(path) if path is not None else CACHE
    if not path.is_file():
        if path != CACHE:
            raise FileNotFoundError(path)
        url = f"https://huggingface.co/{REPOSITORY}/resolve/main/policy.onnx"
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read()
        except OSError as exc:
            raise RuntimeError(f"Cannot download {url}. Download policy.onnx and pass --policy-file PATH.") from exc
        if hashlib.sha256(data).hexdigest() != ONNX_SHA256:
            raise ValueError("Basketball ONNX checksum differs from the published b11 release")
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(data)
        os.replace(temporary, path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != ONNX_SHA256:
        raise ValueError("Expected the published b11 basketball policy.onnx checksum")
    return path


class BasketballOnnxPolicy:
    def __init__(self, path, count=1, device="cpu"):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        inputs = {x.name: x.shape for x in self.session.get_inputs()}
        if inputs != {"obs": [1, 61], "h_in": [1, 1, 256], "c_in": [1, 1, 256]}:
            raise ValueError(f"Unexpected basketball recurrent ONNX inputs: {inputs}")
        if {x.name for x in self.session.get_outputs()} != {"actions", "h_out", "c_out"}:
            raise ValueError("Unexpected basketball recurrent ONNX outputs")
        self.device, self.count = device, count
        self.h = np.zeros((count, 1, 1, 256), dtype=np.float32)
        self.c = np.zeros_like(self.h)

    def reset(self, ids=None):
        ids = slice(None) if ids is None else np.asarray(torch.as_tensor(ids).cpu())
        self.h[ids] = 0
        self.c[ids] = 0

    def __call__(self, observations):
        if observations.shape != (self.count, 61) or not torch.isfinite(observations).all():
            raise ValueError("Basketball ONNX requires finite [num_envs, 61] observations")
        observations = observations.detach().cpu().numpy().astype(np.float32)
        actions = []
        # The released graph has batch size one. Keep independent memory for every world.
        for i in range(self.count):
            action, h, c = self.session.run(["actions", "h_out", "c_out"],
                                           {"obs": observations[i:i + 1], "h_in": self.h[i], "c_in": self.c[i]})
            if not all(np.isfinite(value).all() for value in (action, h, c)):
                raise FloatingPointError("Non-finite basketball policy output")
            self.h[i], self.c[i] = h, c
            actions.append(action)
        return torch.as_tensor(np.concatenate(actions), device=self.device)
