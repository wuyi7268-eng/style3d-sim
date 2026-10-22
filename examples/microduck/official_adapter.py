"""Small, explicit mjlab state facade for reward formulas; no physics emulation."""

from types import SimpleNamespace

import torch

from examples.locomotion.math_utils import quat_rotate_inverse


class SceneEntityCfg(SimpleNamespace):
    def __init__(self, name, **kwargs):
        defaults = {"name": name, "body_ids": [0], "site_ids": [0], "joint_ids": slice(None)}
        if "site_names" in kwargs:
            defaults["site_ids"] = [("mouth_tip", "left_foot", "right_foot").index(n) for n in kwargs["site_names"]]
        super().__init__(**(defaults | kwargs))


def quat_apply_inverse(quat, vector):
    # mjlab uses wxyz; Newton and the shared math helpers use xyzw.
    return quat_rotate_inverse(torch.cat((quat[..., 1:], quat[..., :1]), -1), vector.expand_as(quat[..., :3]))


def wrap_to_pi(value):
    return (value + torch.pi) % (2 * torch.pi) - torch.pi


def link_origin_velocity(com_velocity, angular_velocity, quat_xyzw, local_com):
    """Newton stores COM velocity; mjlab rewards read link-origin velocity."""
    from .tasks import rotate

    offset = rotate(quat_xyzw, local_com.expand_as(com_velocity))
    return com_velocity - torch.cross(angular_velocity, offset, dim=-1)
