"""PPO defaults, official task overrides and symmetry for the shared RSL-RL 2 runner."""

import copy

import torch


class InitialObservationEnv:
    """Normalize learn()'s initial observation, omitted by the pinned RSL-RL 2.3 runner.

    The runner normalizes every step() result itself. Only its entry observation
    needs this adapter, particularly after loading non-identity checkpoint stats.
    """

    def __init__(self, env, runner):
        self.env, self.runner = env, runner

    def __getattr__(self, name):
        return getattr(self.env, name)

    def get_observations(self):
        observation, extras = self.env.get_observations()
        with torch.no_grad():
            observation = self.runner.obs_normalizer(observation)
            extras = {**extras, "observations": dict(extras["observations"])}
            extras["observations"]["critic"] = self.runner.privileged_obs_normalizer(extras["observations"]["critic"])
        return observation, extras


def create_runner(env, config, log_dir=None, device="cpu", *, training=True):
    """Use the pinned runner unchanged, registering only the optional policy class."""
    from rsl_rl.runners import on_policy_runner

    from .cloth_policy import ClothHistoryActorCritic, ClothNormalization
    from .observations import HISTORY_SIZE

    cloth = config["policy"]["class_name"] == "ClothHistoryActorCritic"
    if cloth:
        on_policy_runner.ClothHistoryActorCritic = ClothHistoryActorCritic
    history = config["policy"]["class_name"] == "HistoryActorCritic"
    if history:
        from .policy import HistoryActorCritic, HistoryNormalization

        # RSL-RL 2 resolves class names in this module's namespace.
        on_policy_runner.HistoryActorCritic = HistoryActorCritic
    bc = config.get("teacher_bc") if training else None
    runner_class = on_policy_runner.OnPolicyRunner
    if bc:
        from .distillation import DualTeacherPPO, TeacherObservationEnv, TeacherRunner

        if not history or not config["empirical_normalization"]:
            raise ValueError("Teacher BC requires a normalized history student")
        env = TeacherObservationEnv(env)
        runner_class = TeacherRunner
    runner = runner_class(env, copy.deepcopy(config), log_dir, device)
    if cloth and runner.empirical_normalization:
        runner.obs_normalizer = ClothNormalization(runner.alg.policy.actor_dim).to(device)
        runner.privileged_obs_normalizer = ClothNormalization(runner.alg.policy.critic_dim, runner.obs_normalizer).to(device)
    if history and runner.empirical_normalization:
        runner.obs_normalizer = HistoryNormalization(61).to(device)
        runner.privileged_obs_normalizer = HistoryNormalization(
            env.num_privileged_obs - HISTORY_SIZE, runner.obs_normalizer
        ).to(device)
    if bc:
        options = copy.deepcopy(config["algorithm"])
        if options.pop("class_name") != "PPO":
            raise ValueError("Teacher BC requires PPO")
        algorithm = DualTeacherPPO(runner.alg.policy, device=device, multi_gpu_cfg=runner.multi_gpu_cfg, **options)
        algorithm.storage = runner.alg.storage
        algorithm.attach(bc, env, runner.obs_normalizer)
        runner.alg = algorithm
    # Every variant needs the same normalized entry observation, including flat-ground resume.
    runner.env = InitialObservationEnv(env, runner)
    return runner


def ppo_config(num_envs, policy_variant="standard"):
    if policy_variant not in ("standard", "history", "cloth_history"):
        raise ValueError(f"Unknown policy variant: {policy_variant}")
    return {
        "num_steps_per_env": 24,
        "save_interval": 500,
        "empirical_normalization": True,
        "logger": "tensorboard",
        "policy": {
            "class_name": {"standard": "ActorCritic", "history": "HistoryActorCritic",
                           "cloth_history": "ClothHistoryActorCritic"}[policy_variant],
            "init_noise_std": 1.0,
            "noise_std_type": "scalar",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": min(4, num_envs),
            "learning_rate": 0.001,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
    }


def apply_official_ppo(config, spec, num_envs):
    cfg = copy.deepcopy(config)
    official = spec["rl"]
    for key in ("num_steps_per_env", "save_interval"):
        cfg[key] = official[key]
    algorithm = official["algorithm"]
    for key in cfg["algorithm"]:
        if key != "class_name" and key in algorithm:
            cfg["algorithm"][key] = copy.deepcopy(algorithm[key])
    cfg["algorithm"]["num_mini_batches"] = min(num_envs, algorithm["num_mini_batches"])
    cfg["algorithm"]["normalize_advantage_per_mini_batch"] = algorithm["normalize_advantage_per_mini_batch"]
    if algorithm.get("symmetry_cfg"):
        symmetry = copy.deepcopy(algorithm["symmetry_cfg"])
        symmetry["data_augmentation_func"] = "examples.microduck.ppo:microduck_symmetry"
        cfg["algorithm"]["symmetry_cfg"] = symmetry
    return cfg


def microduck_symmetry(env, obs=None, actions=None, obs_type="policy"):
    # Tables follow the official 61-D sagittal reflection, adapted from TensorDict to RSL 2 tensors.
    perm = [9, 10, 11, 12, 13, 5, 6, 7, 8, 0, 1, 2, 3, 4]
    sign = [-1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1]
    obs_out = act_out = None
    if obs is not None:
        if obs_type == "critic":
            obs_out = torch.cat((obs, obs))
        else:
            indices = list(range(6)) + [start + i for start in (6, 20, 34) for i in perm] + list(range(48, 61))
            signs = [-1, 1, -1, 1, -1, 1] + sign * 3 + [1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1]
            mirrored = obs[:, indices] * obs.new_tensor(signs)
            if obs.shape[1] in (1231, 1294):
                from .cloth_policy import LENGTH, layout

                actor_dim, _, frame_dim, _ = layout(1 if obs.shape[1] == 1231 else 2)
                leg_perm = list(range(5, 10)) + list(range(5))
                version = getattr(env, "cfg", {}).get("history", {}).get("version", 3)
                # Current magnitudes swap legs without changing sign; old load proxies are signed.
                feedback_sign = 1 if obs.shape[1] == 1294 and version >= 3 else -1
                feedback = feedback_sign * obs[:, 61:71][:, leg_perm]
                past = obs[:, actor_dim:].reshape(-1, LENGTH, frame_dim)
                proprioception = past[:, :, indices[:48]] * obs.new_tensor(signs[:48])
                history = torch.cat((proprioception, feedback_sign * past[:, :, 48:58][:, :, leg_perm]), -1)
                if frame_dim == 61:
                    # Acceleration is a polar vector; unlike angular velocity, only Y changes sign.
                    acceleration_signs = obs.new_tensor([1, -1, 1])
                    feedback = torch.cat((feedback, obs[:, 71:74] * acceleration_signs), -1)
                    history = torch.cat((history, past[:, :, 58:61] * acceleration_signs), -1)
                mirrored = torch.cat((mirrored, feedback, history.flatten(1)), -1)
            elif obs.shape[1] != 61:
                from .observations import HISTORY_DIM, HISTORY_LENGTH, HISTORY_SIZE

                if obs.shape[1] != 61 + HISTORY_SIZE:
                    raise ValueError("Unexpected actor observation width for symmetry")
                history = obs[:, 61:].reshape(-1, HISTORY_LENGTH, HISTORY_DIM)
                history = history[:, :, indices[:HISTORY_DIM]] * obs.new_tensor(signs[:HISTORY_DIM])
                mirrored = torch.cat((mirrored, history.flatten(1)), -1)
            obs_out = torch.cat((obs, mirrored))
    if actions is not None:
        act_out = torch.cat((actions, actions[:, perm] * actions.new_tensor(sign)))
    return obs_out, act_out
