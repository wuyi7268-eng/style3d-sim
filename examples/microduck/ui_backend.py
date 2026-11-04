"""Local process supervision for the MicroDuck workbench; never imports a solver on import."""

from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
import zipfile
from collections import deque
from pathlib import Path
from uuid import uuid4

from .task_registry import TASKS

ROOT = Path(__file__).resolve().parents[2]
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
CHECKPOINT_NAME = re.compile(r"model_(\d+)\.pt$")


def number(data, key, default, minimum, maximum, *, integer=False):
    value = data.get(key, default)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} 必须是数字") from None
    if isinstance(value, bool) or not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(f"{key} 必须在 {minimum} 到 {maximum} 之间")
    if integer and not parsed.is_integer():
        raise ValueError(f"{key} 必须是整数")
    return int(parsed) if integer else parsed


def choice(data, key, default, values):
    value = data.get(key, default)
    if value not in values:
        raise ValueError(f"无效的 {key}: {value}")
    return value


def flag(data, key, default=False):
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"{key} 必须为布尔值")
    return value


def device(data):
    value = data.get("device", "cuda:0")
    if not isinstance(value, str) or not re.fullmatch(r"cpu|cuda(?::\d+)?", value):
        raise ValueError("设备必须为 cpu、cuda 或 cuda:N")
    return value


def checkpoint_info(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("请先选择一个模型权重（.pt）")
    path = Path(value).expanduser()
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if path.suffix != ".pt" or not path.is_file():
        raise ValueError("模型文件不存在，或不是 .pt 权重")
    try:
        metadata = json.loads(path.with_name("config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ValueError("权重旁缺少有效的 config.json，请保留完整训练目录") from None
    if not isinstance(metadata, dict) or metadata.get("observation_contract") != "official_asymmetric_pd_v2":
        raise ValueError("这个权重不属于当前 MicroDuck 观测契约")
    if metadata.get("task") not in TASKS or metadata.get("solver") not in ("mujoco", "synreal"):
        raise ValueError("权重的任务或后端不受支持")
    return path, metadata


def completed_checkpoints(directory):
    """A new torch.save file is visible before its ZIP directory is committed."""
    candidates = []
    for path in Path(directory).glob("model_*.pt"):
        match = CHECKPOINT_NAME.fullmatch(path.name)
        try:
            if match and path.stat().st_size and zipfile.is_zipfile(path):
                candidates.append((int(match[1]), path))
        except OSError:
            continue
    return [path for _, path in sorted(candidates, reverse=True)]


def catalog(log_root):
    runs = []
    for config in Path(log_root).glob("microduck_*/*/config.json"):
        try:
            metadata = json.loads(config.read_text(encoding="utf-8"))
            if metadata.get("observation_contract") != "official_asymmetric_pd_v2":
                continue
            checkpoints = completed_checkpoints(config.parent)
            runs.append(
                {
                    "path": str(config.parent),
                    "name": config.parent.name,
                    "task": metadata.get("task"),
                    "solver": metadata.get("solver"),
                    "terrain": metadata.get("terrain", {}).get("kind", "default"),
                    "num_envs": metadata.get("num_envs"),
                    "modified": config.stat().st_mtime,
                    "checkpoints": [str(path) for path in checkpoints],
                }
            )
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return sorted(runs, key=lambda run: run["name"], reverse=True)


def training_plan(data, log_root):
    task = choice(data, "task", "velocity", TASKS)
    solver = choice(data, "solver", "synreal", ("synreal", "mujoco"))
    from .env import ENVIRONMENT_CONFIGS

    profile = choice(data, "environment_profile", "official", tuple(ENVIRONMENT_CONFIGS))
    mode = choice(data, "mode", "new", ("new", "resume", "transfer"))
    run_name = data.get("run_name", "workbench")
    if not isinstance(run_name, str) or not re.fullmatch(r"[\w.-]{1,80}", run_name) or run_name in (".", ".."):
        raise ValueError("实验名称限 1–80 个字母、数字、中文、下划线、点或连字符")
    if profile != "official" and solver != "synreal":
        raise ValueError("非平地场景需要 SynReal 求解器")
    target_device = device(data)
    if solver == "synreal" and target_device == "cpu":
        raise ValueError("SynReal 需要 CUDA 设备")
    if target_device == "cpu" and number(data, "num_envs", 1024, 1, 65536, integer=True) != 1:
        raise ValueError("当前 MuJoCo CPU 路径仅支持 1 个环境；多环境训练请选择 CUDA")
    argv = [
        sys.executable,
        "-u",
        "-m",
        "examples.microduck.train",
        "--viewer",
        "null",
        "--solver",
        solver,
        "--task",
        task,
        "--device",
        target_device,
        "--environment-profile",
        profile,
        "--run-name",
        run_name,
        "--log-root",
        str(log_root),
    ]
    for key, default, low, high in (("iterations", 1000, 1, 10000000), ("seed", 42, 0, 2147483647)):
        argv += ["--" + key.replace("_", "-"), str(number(data, key, default, low, high, integer=True))]
    # Blank overrides deliberately defer to the single source of truth in env.py.
    if data.get("num_envs") not in (None, ""):
        argv += ["--num-envs", str(number(data, "num_envs", 1024, 1, 65536, integer=True))]
    if data.get("pad_probability") not in (None, ""):
        argv += ["--pad-probability", str(number(data, "pad_probability", 0.5, 0, 1))]
    friction = choice(data, "friction", "default", ("default", "on", "off"))
    if friction != "default":
        argv += ["--randomize-foot-friction" if friction == "on" else "--no-randomize-foot-friction"]
    if flag(data, "self_collisions"):
        argv += ["--self-collisions"]
    if mode != "new":
        path, metadata = checkpoint_info(data.get("checkpoint"))
        if mode == "resume" and (metadata["task"], metadata["solver"]) != (task, solver):
            raise ValueError("续训必须使用权重原有的任务和后端")
        if mode == "transfer" and ((solver, task) != ("synreal", "velocity") or flag(data, "self_collisions")):
            raise ValueError("迁移目前仅支持关闭自碰撞的 SynReal / velocity")
        argv += ["--resume" if mode == "resume" else "--init-from", str(path)]
    if flag(data, "use_checkpoint_environment"):
        if mode != "resume":
            raise ValueError("恢复历史环境只适用于续训")
        argv += ["--use-checkpoint-environment"]
    ppo = {}
    # Resume must retain its saved optimizer, normalizers and PPO settings.
    if mode != "resume":
        for key, bounds in {"num_steps_per_env": (24, 1, 10000), "save_interval": (100, 1, 1000000)}.items():
            if data.get(key) not in (None, ""):
                ppo[key] = number(data, key, *bounds, integer=True)
        algorithm = {}
        for key, bounds in {
            "learning_rate": (0.001, 1e-9, 1),
            "entropy_coef": (0.01, 0, 1),
            "num_learning_epochs": (5, 1, 100),
            "num_mini_batches": (4, 1, 1024),
        }.items():
            if data.get(key) not in (None, ""):
                algorithm[key] = number(data, key, *bounds, integer=key.startswith("num_"))
        if data.get("schedule") not in (None, "", "default"):
            algorithm["schedule"] = choice(data, "schedule", "adaptive", ("fixed", "adaptive"))
        if algorithm:
            ppo["algorithm"] = algorithm
    return argv, {"ppo": ppo}


def playback_plan(data):
    path, metadata = checkpoint_info(data.get("checkpoint"))
    target_device = device(data)
    if metadata["solver"] == "synreal" and target_device == "cpu":
        raise ValueError("SynReal 需要 CUDA 设备")
    if target_device == "cpu" and number(data, "num_envs", 1, 1, 128, integer=True) != 1:
        raise ValueError("当前 MuJoCo CPU 路径仅支持 1 个环境")
    argv = [
        sys.executable,
        "-u",
        "-m",
        "examples.microduck.play",
        "--checkpoint",
        str(path),
        "--solver",
        metadata["solver"],
        "--device",
        target_device,
        "--viewer",
        "gl",
        "--keep-open",
        "--num-envs",
        str(number(data, "num_envs", 1, 1, 128, integer=True)),
        "--num-frames",
        str(number(data, "num_frames", 10000000, 1, 10000000, integer=True)),
    ]
    argv += ["--command"] + [str(number(data, key, 0, -5, 5)) for key in ("vx", "vy", "yaw")]
    if metadata.get("self_collisions"):
        argv += ["--self-collisions"]
    for key in ("allow_recipe_mismatch", "use_checkpoint_environment", "diagnostics"):
        if flag(data, key):
            argv += ["--" + key.replace("_", "-")]
    return argv


class Job:
    """Own a child process and drain stdout continuously, even without an open browser."""

    def __init__(self, kind, argv, directory, child_env, *, job_id, name, parent_id=None, settings=None):
        self.kind, self.argv, self.directory = kind, argv, directory
        self.id, self.name, self.parent_id = job_id, name, parent_id
        self.settings = dict(settings or {})
        self.lock = threading.RLock()
        self.lines = deque(maxlen=1200)
        self.metrics = {}
        self.run_dir = None
        self.started = time.time()
        self.finished = None
        self.stopping = False
        self.stop_requested = False
        self.log_file = directory / "output.log"
        self.process = subprocess.Popen(
            argv,
            cwd=ROOT,
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=os.name != "nt",
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        with self.log_file.open("w", encoding="utf-8") as output, self.process.stdout:
            for raw in self.process.stdout:
                line = ANSI.sub("", raw).rstrip()
                output.write(line + "\n")
                output.flush()
                with self.lock:
                    self.lines.append(line)
                    if line.startswith("Training directory: "):
                        self.run_dir = line.removeprefix("Training directory: ").strip()
                    match = re.search(r"Learning iteration\s+(\d+)\s*/\s*(\d+)", line)
                    if match:
                        self.metrics.update(iteration=int(match[1]), total=int(match[2]))
                    for name, pattern in {
                        "fps": r"Computation:\s*([\d.]+) steps/s",
                        "reward": r"Mean reward:\s*([-+\d.eE]+)",
                        "episode_length": r"Mean episode length:\s*([\d.]+)",
                    }.items():
                        match = re.search(pattern, line)
                        if match:
                            self.metrics[name] = float(match[1])
        self.process.wait()
        with self.lock:
            self.finished = time.time()

    def snapshot(self, *, include_log=True):
        with self.lock:
            code = self.process.poll()
            state = (
                ("stopping" if self.stopping else "running")
                if code is None
                else ("stopped" if self.stop_requested else "completed" if code == 0 else "failed")
            )
            return {
                "id": self.id,
                "name": self.name,
                "parent_id": self.parent_id,
                "settings": self.settings,
                "kind": self.kind,
                "state": state,
                "pid": self.process.pid,
                "exit_code": code,
                "started": self.started,
                "elapsed": (self.finished or time.time()) - self.started,
                "run_dir": self.run_dir,
                "command": self.argv,
                "metrics": dict(self.metrics),
                "log": "\n".join(self.lines) if include_log else None,
                "log_path": str(self.log_file),
            }

    def stop(self):
        with self.lock:
            if self.process.poll() is not None or self.stopping:
                return
            self.stopping = self.stop_requested = True
        threading.Thread(target=self._stop, daemon=True).start()

    def _stop(self):
        # Signal only this owned process group; never find or kill unrelated training jobs.
        try:
            if os.name == "nt":
                self.process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(self.process.pid, signal.SIGINT)
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        except ProcessLookupError:
            pass


class Workbench:
    def __init__(self, log_root=ROOT / "logs/rsl_rl", child_env=None):
        self.log_root = Path(log_root).resolve()
        self.jobs = {}
        self.lock = threading.RLock()
        self.child_env = dict(os.environ if child_env is None else child_env)
        self.closing = False

    def _launch(self, kind, argv, config=None, *, parent_id=None, settings=None):
        if self.closing:
            raise ValueError("控制台正在关闭")
        # Every training run has an immutable ID, even when experiment names are identical.
        # Preview slots belong to that ID so stopping or reopening one cannot target another run.
        job_id = uuid4().hex if kind == "train" else f"preview:{parent_id}" if kind == "preview" else kind
        existing = self.jobs.get(job_id)
        if existing and existing.process.poll() is None:
            raise ValueError("该进程仍在运行，请先关闭或等待结束")
        if (
            "--solver" in argv
            and argv[argv.index("--solver") + 1] == "synreal"
            and not all(self.child_env.get(key) for key in ("STYLE3D_ACCOUNT", "STYLE3D_PASSWORD"))
        ):
            raise ValueError("请在终端用 --login 启动控制台并隐藏输入 SynReal 登录信息，再开始运行")
        directory = self.log_root / ".workbench" / (time.strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
        directory.mkdir(parents=True)
        if config is not None:
            config_path = directory / "ppo.json"
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            argv = [*argv, "--training-config", str(config_path)]
        (directory / "command.json").write_text(json.dumps(argv, ensure_ascii=False, indent=2), encoding="utf-8")
        name = (settings or {}).get("run_name", "workbench") if kind == "train" else kind
        if parent_id:
            name = self.jobs[parent_id].name
        self.jobs[job_id] = Job(
            kind, argv, directory, self.child_env, job_id=job_id, name=name, parent_id=parent_id, settings=settings
        )
        return self.jobs[job_id].snapshot()

    def train(self, data):
        with self.lock:
            argv, config = training_plan(data, self.log_root)
            # Keep only form fields used by the training command; never persist arbitrary request data.
            keys = ("run_name", "solver", "task", "device", "environment_profile", "num_envs", "iterations", "mode")
            settings = {
                key: argv[argv.index("--" + key.replace("_", "-")) + 1] for key in keys if "--" + key.replace("_", "-") in argv
            }
            settings["mode"] = data.get("mode", "new")
            return self._launch("train", argv, config, settings=settings)

    def play(self, data, *, preview=False):
        with self.lock:
            parent_id = None
            if preview:
                parent_id = data.get("train_id")
                training = self.jobs.get(parent_id)
                if training is None or training.kind != "train":
                    raise ValueError("请指定要预览的训练实验")
                run_dir = training.snapshot(include_log=False)["run_dir"]
                if not run_dir:
                    raise ValueError("训练尚未建立运行目录，请等待初始化完成")
                checkpoints = completed_checkpoints(run_dir)
                if not checkpoints:
                    raise ValueError("还没有完整保存的权重，请等到首次保存后再查看")
                data = {**data, "checkpoint": str(checkpoints[0])}
                # Follow the selected run's device, not the new-experiment form or a stale browser selection.
                data["device"] = training.argv[training.argv.index("--device") + 1]
                # Preview restores the actual environment selected for this run, including CLI overrides.
                data["use_checkpoint_environment"] = True
            argv = playback_plan(data)
            if os.name != "nt" and not (self.child_env.get("DISPLAY") or self.child_env.get("WAYLAND_DISPLAY")):
                raise ValueError("未检测到桌面显示服务；3D 窗口需要在有图形桌面的机器上运行控制台")
            return self._launch("preview" if preview else "evaluation", argv, parent_id=parent_id)

    def tensorboard(self):
        with self.lock:
            return self._launch(
                "tensorboard",
                [
                    sys.executable,
                    "-m",
                    "tensorboard.main",
                    "--logdir",
                    str(self.log_root),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "6006",
                ],
            )

    def stop(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise ValueError("请指定有效的进程 ID；未停止任何实验")
            self.jobs[job_id].stop()

    def snapshot(self, *, log_id=None):
        with self.lock:
            # HTTP polling requests just one log; adding experiments must not multiply log traffic.
            return {job_id: job.snapshot(include_log=log_id is None or log_id == job_id) for job_id, job in self.jobs.items()}

    def close(self):
        with self.lock:
            self.closing = True
            jobs = list(self.jobs.values())
        for job in jobs:
            job.stop()
        for job in jobs:
            try:
                job.process.wait(timeout=22)
                job.reader.join(timeout=2)
            except subprocess.TimeoutExpired:
                job.process.kill()
                job.process.wait()
