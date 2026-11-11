"""Edge stopping must use the saved pad dimensions and both horizontal axes."""

from types import SimpleNamespace

import pytest
import torch

from demos.microduck_soft_pad.recording import PlaybackRecording


@pytest.mark.parametrize("position,expected", [
    ((0.599, 0.399), False), ((0.6, 0), True), ((-0.6, 0), True),
    ((0, 0.4), True), ((0, -0.4), True),
])
def test_edge_uses_saved_rectangular_pad(position, expected):
    recording = PlaybackRecording.__new__(PlaybackRecording)
    recording.env = SimpleNamespace(terrain=SimpleNamespace(
        origins=torch.zeros(1, 3), config=SimpleNamespace(size=(1.2, 0.8, 0.12)),
    ))
    recording.timeline = [{"root_pose": [*position, 0.25]}]
    assert recording.reached_pad_edge() is expected


@pytest.mark.parametrize("position,expected", [((0.599, 0.399), False), ((0.6, 0), True), ((0, -0.4), True)])
def test_cloth_edge_uses_saved_dimensions(position, expected):
    recording = PlaybackRecording.__new__(PlaybackRecording)
    recording.env = SimpleNamespace(terrain=SimpleNamespace(
        kind="cloth", origins=torch.zeros(1, 3), cfg={"config": {"cloth_length": 1.2, "cloth_width": 0.8}},
    ))
    recording.timeline = [{"root_pose": [*position, 1.0]}]
    assert recording.reached_pad_edge() is expected
