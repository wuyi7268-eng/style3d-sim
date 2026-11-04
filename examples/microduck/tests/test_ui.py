"""Workbench contracts and lifecycle checks without a native solver or GPU."""

import json
import re
import sys
import threading
import time
import zipfile
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from examples.microduck.ui import make_server
from examples.microduck.ui_backend import (
    Workbench,
    catalog,
    completed_checkpoints,
    playback_plan,
    training_plan,
)


def make_checkpoint(tmp_path, iteration=9, **updates):
    directory = tmp_path / "microduck_mujoco" / "2026_test"
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {
        "task": "velocity",
        "solver": "mujoco",
        "observation_contract": "official_asymmetric_pd_v2",
        "recipe": "official_adapted_v1",
        "ppo": {"algorithm": {}},
        "num_envs": 4,
        **updates,
    }
    (directory / "config.json").write_text(json.dumps(metadata))
    checkpoint = directory / f"model_{iteration}.pt"
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("test", "checkpoint placeholder")
    return checkpoint


def wait_until(predicate):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    pytest.fail("child process did not reach the expected state")


def test_environment_defaults_remain_in_env_py(tmp_path):
    from examples.microduck.common import setup
    from examples.microduck.env import ENVIRONMENT_CONFIGS

    argv, config = training_plan({"solver": "mujoco", "num_envs": "", "pad_probability": ""}, tmp_path)
    args, _, _ = setup("train", argv[4:])
    assert args.num_envs == ENVIRONMENT_CONFIGS["official"].num_envs
    assert args.pad_probability == ENVIRONMENT_CONFIGS["official"].pad_probability
    assert args.viewer == "null"
    assert config == {"ppo": {}}


def test_ppo_overrides_are_separate_from_environment(tmp_path):
    argv, config = training_plan(
        {
            "num_envs": "8",
            "pad_probability": "0.3",
            "learning_rate": "0.0001",
            "save_interval": "10",
            "num_steps_per_env": "24",
        },
        tmp_path,
    )
    assert config == {"ppo": {"save_interval": 10, "num_steps_per_env": 24, "algorithm": {"learning_rate": 0.0001}}}
    assert argv[argv.index("--num-envs") + 1] == "8"
    assert argv[argv.index("--pad-probability") + 1] == "0.3"


@pytest.mark.parametrize(
    "data",
    [
        {"iterations": "nan"},
        {"iterations": "1.2"},
        {"num_envs": 0},
        {"num_envs": True},
        {"run_name": "../overwrite"},
        {"device": "cuda:0;echo"},
        {"solver": "fake"},
        {"environment_profile": "mixed_soft_pad", "solver": "mujoco"},
        {"solver": "synreal", "device": "cpu"},
        {"use_checkpoint_environment": True},
    ],
)
def test_reject_invalid_training_before_launch(tmp_path, data):
    with pytest.raises(ValueError):
        training_plan(data, tmp_path)


def test_resume_keeps_saved_ppo_and_rejects_other_task(tmp_path):
    checkpoint = make_checkpoint(tmp_path)
    data = {"mode": "resume", "solver": "mujoco", "checkpoint": str(checkpoint), "learning_rate": 0.4}
    argv, config = training_plan(data, tmp_path)
    assert "--resume" in argv and config == {"ppo": {}}
    with pytest.raises(ValueError, match="任务和后端"):
        training_plan({**data, "task": "standup"}, tmp_path)


def test_playback_selects_metadata_and_mismatch_is_opt_in(tmp_path):
    checkpoint = make_checkpoint(tmp_path, task="standup", self_collisions=True)
    argv = playback_plan({"checkpoint": str(checkpoint), "device": "cpu"})
    assert argv[argv.index("--solver") + 1] == "mujoco"
    assert "--task" not in argv
    assert "--self-collisions" in argv
    assert "--allow-recipe-mismatch" not in argv
    assert "--use-checkpoint-environment" not in argv
    argv = playback_plan({"checkpoint": str(checkpoint), "allow_recipe_mismatch": True})
    assert "--allow-recipe-mismatch" in argv


def test_catalog_skips_partial_saves_and_sorts_numerically(tmp_path):
    older = make_checkpoint(tmp_path, 9)
    latest = make_checkpoint(tmp_path, 100)
    (latest.parent / "model_101.pt").write_bytes(b"PK\x03\x04partially written")
    assert completed_checkpoints(latest.parent) == [latest, older]
    assert catalog(tmp_path)[0]["checkpoints"] == [str(latest), str(older)]


def test_preview_requires_checkpoint_not_official_walk(tmp_path):
    workbench = Workbench(tmp_path)
    with pytest.raises(ValueError, match="指定要预览"):
        workbench.play({}, preview=True)
    with pytest.raises(ValueError, match="选择一个模型"):
        playback_plan({})


def test_parallel_trainings_keep_logs_previews_and_stop_independent(tmp_path):
    workbench = Workbench(tmp_path)

    def command(marker):
        return [sys.executable, "-u", "-c", f"import time; print('Learning iteration {marker}/10', flush=True); time.sleep(60)"]

    try:
        first = workbench._launch("train", command(2), settings={"run_name": "same_name"})["id"]
        second = workbench._launch("train", command(7), settings={"run_name": "same_name"})["id"]
        assert first != second
        preview = workbench._launch("preview", command(1), parent_id=first)["id"]
        other_preview = workbench._launch("preview", command(3), parent_id=second)["id"]
        wait_until(lambda: workbench.snapshot()[second]["metrics"].get("iteration") == 7)
        wait_until(lambda: workbench.snapshot()[first]["metrics"].get("iteration") == 2)
        with pytest.raises(ValueError, match="仍在运行"):
            workbench._launch("preview", command(1), parent_id=first)
        selected = workbench.snapshot(log_id=first)
        assert "iteration 2/10" in selected[first]["log"]
        assert selected[second]["log"] is None
        assert selected[first]["log_path"] != selected[second]["log_path"]
        workbench.stop(preview)
        wait_until(lambda: workbench.snapshot()[preview]["state"] == "stopped")
        assert workbench.snapshot()[other_preview]["state"] == "running"
        workbench.stop(first)
        wait_until(lambda: workbench.snapshot()[first]["state"] == "stopped")
        assert workbench.snapshot()[second]["state"] == "running"
        workbench._launch("preview", [sys.executable, "-c", "print('reopened')"], parent_id=first)
        wait_until(lambda: workbench.snapshot()[preview]["state"] == "completed")
        with pytest.raises(ValueError, match="有效的进程 ID"):
            workbench.stop("train")
        assert workbench.snapshot()[second]["state"] == "running"
    finally:
        workbench.close()
    assert all(job.process.poll() is not None for job in workbench.jobs.values())


def test_preview_uses_explicit_training_and_its_device(tmp_path, monkeypatch):
    workbench = Workbench(tmp_path, child_env={"DISPLAY": ":0"})
    checkpoint = make_checkpoint(tmp_path)
    try:
        first = workbench._launch("train", [sys.executable, "-c", "pass", "--device", "cpu"])["id"]
        second = workbench._launch("train", [sys.executable, "-c", "pass", "--device", "cuda:1"])["id"]
        workbench.jobs[first].run_dir = str(checkpoint.parent)
        other = make_checkpoint(tmp_path / "other")
        workbench.jobs[second].run_dir = str(other.parent)
        monkeypatch.setattr(workbench, "_launch", lambda kind, argv, **kwargs: (kind, argv, kwargs))
        kind, argv, kwargs = workbench.play({"train_id": second, "device": "cpu"}, preview=True)
        assert kind == "preview" and kwargs["parent_id"] == second
        assert argv[argv.index("--device") + 1] == "cuda:1"
        assert argv[argv.index("--checkpoint") + 1] == str(other)
        assert "--use-checkpoint-environment" in argv
        with pytest.raises(ValueError, match="指定要预览"):
            workbench.play({"train_id": "unknown"}, preview=True)
    finally:
        workbench.close()


def test_launch_failure_is_visible(tmp_path):
    workbench = Workbench(tmp_path)
    try:
        job_id = workbench._launch("train", [sys.executable, "-c", "raise RuntimeError('test failure')"])["id"]
        wait_until(lambda: workbench.snapshot()[job_id]["state"] == "failed")
        workbench.jobs[job_id].reader.join(timeout=2)
        result = workbench.snapshot()[job_id]
        assert result["exit_code"] != 0
        assert "test failure" in result["log"]
    finally:
        workbench.close()


def test_missing_synreal_login_fails_before_spawn(tmp_path):
    workbench = Workbench(tmp_path, child_env={})
    with pytest.raises(ValueError, match="--login"):
        workbench.train({})
    assert not workbench.jobs


def test_cpu_world_limit_is_reported_before_launch(tmp_path):
    with pytest.raises(ValueError, match="仅支持 1 个环境"):
        training_plan({"solver": "mujoco", "device": "cpu", "num_envs": 2}, tmp_path)
    checkpoint = make_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="仅支持 1 个环境"):
        playback_plan({"checkpoint": str(checkpoint), "device": "cpu", "num_envs": 2})


def test_http_requires_local_host_and_session_token(tmp_path):
    workbench = Workbench(tmp_path, child_env={})
    server = make_server(workbench, 0, {"tasks": ["velocity"]})
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(url) as response:
            meta = re.search(r'<meta[^>]*name="workbench-token"[^>]*>', response.read().decode())[0]
            token = re.search(r'content="([^"]+)"', meta)[1]
        with pytest.raises(HTTPError) as error:
            urlopen(url + "/api/status")
        assert error.value.code == 403
        request = Request(url + "/api/status", headers={"X-Workbench-Token": token})
        with urlopen(request) as response:
            assert json.load(response) == {}
        request = Request(url + "/", headers={"Host": "untrusted.example"})
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 403
        request = Request(url + "/api/train", data=b'{"iterations": 0}', headers={"X-Workbench-Token": token})
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 400
        assert not workbench.jobs
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
        workbench.close()
