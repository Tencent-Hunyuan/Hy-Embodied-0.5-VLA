# coding=utf-8
# Copyright (C) 2026 Tencent.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RoboDojo entry point for Hy-VLA evaluation.

This module mirrors ``robotwin_eval.deploy_policy`` but accepts RoboDojo-style
observations:

* ``vision/{cam_head,cam_left_wrist,cam_right_wrist}/color``
* ``state/{left,right}_ee_pose`` as xyz + quat_wxyz
* ``state/{left,right}_ee_joint_state`` as gripper scalars
* optional ``instruction`` / ``prompt``
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .policy_wrapper import HyVLARoboDojoPolicyWrapper, build_policy

CAM_HEAD = ["cam_head", "head_camera", "cam_high", "top_camera"]
CAM_LEFT = ["cam_left_wrist", "left_camera", "left_wrist"]
CAM_RIGHT = ["cam_right_wrist", "right_camera", "right_wrist"]


def _extract_image(obs: dict[str, Any], candidates: list[str]) -> np.ndarray:
    vision = obs.get("vision", {})
    for name in candidates:
        if name not in vision:
            continue
        entry = vision[name]
        if isinstance(entry, dict):
            for key in ("color", "rgb", "colors"):
                if key in entry:
                    return _to_hwc_uint8_rgb(entry[key])
        else:
            return _to_hwc_uint8_rgb(entry)
    raise KeyError(f"No image for any of {candidates}; have {list(vision.keys())}")


def _to_hwc_uint8_rgb(img: Any) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got shape {img.shape}")
    if img.shape[-1] == 4:
        img = img[..., :3]
    elif img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3, 4):
        img = np.transpose(img, (1, 2, 0))
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    if np.issubdtype(img.dtype, np.floating):
        img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return img


def _to_chw_float(img: np.ndarray) -> np.ndarray:
    return img.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0


def _pad_state(state: np.ndarray, max_state_dim: int = 32) -> np.ndarray:
    if state.shape[-1] == max_state_dim:
        return state
    shape = list(state.shape)
    cur = shape[-1]
    shape[-1] = max_state_dim
    out = np.zeros(shape, dtype=state.dtype)
    out[..., :cur] = state
    return out


def encode_obs(observation: dict[str, Any], instruction: str | None = None) -> dict[str, Any]:
    """Pack a RoboDojo observation into the Hy-VLA batch format."""
    head = _extract_image(observation, CAM_HEAD)
    left = _extract_image(observation, CAM_LEFT)
    right = _extract_image(observation, CAM_RIGHT)

    state = observation.get("state", {})
    left_pose = np.asarray(state["left_ee_pose"], dtype=np.float32)
    right_pose = np.asarray(state["right_ee_pose"], dtype=np.float32)
    left_gripper = float(np.asarray(state["left_ee_joint_state"]).reshape(-1)[0])
    right_gripper = float(np.asarray(state["right_ee_joint_state"]).reshape(-1)[0])
    state16 = np.concatenate(
        [
            left_pose[:3],
            left_pose[3:7],
            [left_gripper],
            right_pose[:3],
            right_pose[3:7],
            [right_gripper],
        ]
    ).astype(np.float32)

    task = instruction or observation.get("instruction") or observation.get("prompt") or ""
    return {
        "observation.images.top_head": _to_chw_float(head),
        "observation.images.hand_left": _to_chw_float(left),
        "observation.images.hand_right": _to_chw_float(right),
        "observation.state": _pad_state(state16[np.newaxis, :], max_state_dim=32),
        "task": [task],
        "raw_images.top_head": head,
        "raw_images.hand_left": left,
        "raw_images.hand_right": right,
    }


def get_model(usr_args: dict[str, Any]) -> HyVLARoboDojoPolicyWrapper:
    """Factory called once per evaluation run."""
    return build_policy(usr_args)


def eval(TASK_ENV, model: HyVLARoboDojoPolicyWrapper, observation: dict[str, Any]) -> None:  # noqa: A001
    """Per-step closed-loop hook."""
    instruction = None
    get_instruction = getattr(TASK_ENV, "get_instruction", None)
    if callable(get_instruction):
        instruction = get_instruction()
    batch = encode_obs(observation, instruction)
    actions = model.get_action(batch)
    take_action = getattr(TASK_ENV, "take_action")
    is_episode_end = getattr(TASK_ENV, "is_episode_end", None)
    for action in actions:
        try:
            take_action(action, action_type="ee")
        except TypeError:
            take_action(action)
        if callable(is_episode_end) and is_episode_end():
            break


def reset_model(model: HyVLARoboDojoPolicyWrapper) -> str:
    """Per-episode reset hook."""
    return model.reset()


__all__ = ["encode_obs", "get_model", "eval", "reset_model"]
