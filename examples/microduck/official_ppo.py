"""Map official PPO hyperparameters onto the shared RSL-RL 2 runner (no teachers)."""

import copy

import torch


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
        symmetry["data_augmentation_func"] = "examples.microduck.official_ppo:microduck_symmetry"
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
            obs_out = torch.cat((obs, obs[:, indices] * obs.new_tensor(signs)))
    if actions is not None:
        act_out = torch.cat((actions, actions[:, perm] * actions.new_tensor(sign)))
    return obs_out, act_out
