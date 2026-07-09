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

"""Hy-VLA policy wrapper for RoboDojo evaluation.

This wrapper intentionally mirrors the XPolicyLab
``Hy_Embodied_05_VLA/model.py`` inference behavior while remaining independent
of XPolicyLab. It returns RoboDojo action dictionaries in executable chunks,
keeps MEM frame history per ``env_idx`` for parallel rollouts, and uses the
same UMI-coordinate preprocessing defaults as RoboDojo training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from hy_vla import HyVLA, HyVLAConfig
from hy_vla.utils.transform_utils import (
    convert_PosQuat2PosRotationMatrix_batch,
    convert_frame_robo_to_umi,
    convert_frame_umi_to_robo,
)
from robotwin_eval.policy_wrapper import _blend_dual_arm_pose_quat
from robotwin_eval.transforms import (
    get_norm_data,
    pos_rotation_matrix_to_pos_quat,
    relative_to_dual_arm_poses,
)

L_POS, L_QUAT, L_GRIP = slice(0, 3), slice(3, 7), 7
R_POS, R_QUAT, R_GRIP = slice(8, 11), slice(11, 15), 15


def _pad_state(state: np.ndarray, max_state_dim: int = 32) -> np.ndarray:
    if state.shape[-1] == max_state_dim:
        return state
    shape = list(state.shape)
    cur = shape[-1]
    shape[-1] = max_state_dim
    out = np.zeros(shape, dtype=state.dtype)
    out[..., :cur] = state
    return out


def _convert_pose_robo_dojo(
    eepose16_wxyz: np.ndarray,
    qpos_mean: np.ndarray,
    qpos_std: np.ndarray,
    *,
    umi_coord_frame: bool,
    umi_gripper_space: bool,
) -> np.ndarray:
    """Encode a 16-d RoboDojo EE state into normalized Hy-VLA state space."""
    e = eepose16_wxyz.copy()
    e[3:7] = eepose16_wxyz[[4, 5, 6, 3]]
    e[11:15] = eepose16_wxyz[[12, 13, 14, 11]]
    if umi_coord_frame:
        e = convert_frame_robo_to_umi(
            e[None, :], convert_gripper=umi_gripper_space,
        )[0]
    ee_prop = convert_PosQuat2PosRotationMatrix_batch(e[None, :], quat_order="xyzw")[0]
    return ((ee_prop - qpos_mean) / (qpos_std + 1e-8))[None, ...]


def action16_to_robodojo_dict(action_wxyz: np.ndarray) -> dict[str, np.ndarray]:
    """Convert one 16-d dual-arm EE action to RoboDojo's action dict."""
    row = np.asarray(action_wxyz, dtype=np.float32)
    if row.shape != (16,):
        raise ValueError(f"Expected a 16-d action vector, got shape {row.shape}")
    return {
        "left_ee_pose": np.concatenate([row[L_POS], row[L_QUAT]]).astype(np.float32),
        "right_ee_pose": np.concatenate([row[R_POS], row[R_QUAT]]).astype(np.float32),
        "left_ee_joint_state": np.array([row[L_GRIP]], dtype=np.float32),
        "right_ee_joint_state": np.array([row[R_GRIP]], dtype=np.float32),
    }


def action_chunk_to_robodojo_dicts(chunk16_wxyz: np.ndarray) -> list[dict[str, np.ndarray]]:
    """Convert a ``(T, 16)`` action chunk to RoboDojo action dictionaries."""
    return [action16_to_robodojo_dict(row) for row in np.asarray(chunk16_wxyz)]


class HyVLARoboDojoPolicyWrapper:
    """RoboDojo-facing wrapper around ``HyVLA``.

    Public methods intentionally match XPolicyLab adapter conventions:
    ``update_obs``, ``update_obs_batch``, ``get_action``,
    ``get_action_batch``, and ``reset``.
    """

    def __init__(
        self,
        ckpt_path: str,
        norm_path: str,
        *,
        blend_mode: str = "rel_only",
        with_absolute: bool = False,
        exc_action_size: int = 10,
        exc_action_interval: int = 1,
        img_history_size: int = 6,
        img_history_interval: int = 20,
        weight_dtype: torch.dtype = torch.bfloat16,
        vlm_model_path: str | None = None,
        umi_coord_frame: bool = True,
        umi_gripper_space: bool = False,
    ) -> None:
        if umi_gripper_space and not umi_coord_frame:
            raise ValueError("umi_gripper_space=true requires umi_coord_frame=true")
        if blend_mode not in ("rel_abs", "rel_only", "abs_only"):
            raise ValueError(
                f"blend_mode must be one of rel_abs|rel_only|abs_only, got {blend_mode!r}"
            )

        self.weight_dtype = weight_dtype
        self.blend_mode = blend_mode
        self._with_abs = bool(with_absolute)
        self.exc_action_size = int(exc_action_size)
        self.exc_action_interval = int(exc_action_interval)
        self.img_history_size = int(img_history_size)
        self.img_history_interval = int(img_history_interval)
        self.umi_coord_frame = bool(umi_coord_frame)
        self.umi_gripper_space = bool(umi_gripper_space)

        assert self.exc_action_interval >= 1, "exc_action_interval must be >= 1"
        assert (
            self.img_history_interval % self.exc_action_interval == 0
        ), (
            f"img_history_interval ({self.img_history_interval}) must be divisible "
            f"by exc_action_interval ({self.exc_action_interval})"
        )

        self.config = HyVLAConfig.from_pretrained(ckpt_path)
        self.policy = HyVLA.from_pretrained(
            ckpt_path, config=self.config, vlm_model_path=vlm_model_path,
        )
        self.policy.enable_video_encoder_if_needed()
        self.policy.cuda().eval()
        self.policy = self.policy.to(self.weight_dtype)

        self.norm_data = get_norm_data(norm_path)
        has_abs = (
            self.norm_data.get("act_mean_abs") is not None
            and self.norm_data.get("act_std_abs") is not None
        )
        if self._with_abs and not has_abs:
            raise ValueError(f"with_absolute=true requires abs stats in {norm_path!r}")
        if not has_abs and blend_mode != "rel_only":
            raise ValueError(f"blend_mode={blend_mode!r} requires abs stats in {norm_path!r}")

        n_act = int(self.config.n_action_steps)
        effective_chunk = n_act // 2 if self._with_abs else n_act
        for key in ("act_mean", "act_std", "act_mean_abs", "act_std_abs"):
            val = self.norm_data.get(key)
            if val is not None and val.shape[0] != effective_chunk:
                assert effective_chunk <= val.shape[0]
                self.norm_data[key] = val[:effective_chunk].copy()

        self.use_video_encoder = bool(self.config.use_video_encoder)
        self._obs_by_env: dict[int, dict[str, Any]] = {}
        self._top_imgs_by_env: dict[int, list[np.ndarray]] = {}
        self._left_imgs_by_env: dict[int, list[np.ndarray]] = {}
        self._right_imgs_by_env: dict[int, list[np.ndarray]] = {}
        self._latest_env_idx_list: list[int] = [0]

    def update_obs(self, obs: dict[str, Any]) -> None:
        if "env_idx" not in obs:
            obs = {**obs, "env_idx": 0}
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list: list[dict[str, Any]]) -> None:
        from .deploy_policy import encode_obs

        self._latest_env_idx_list = [int(obs.get("env_idx", i)) for i, obs in enumerate(obs_list)]
        for env_idx, obs in zip(self._latest_env_idx_list, obs_list):
            batch = encode_obs(obs)
            self._obs_by_env[env_idx] = batch
            if self.use_video_encoder:
                self._top_imgs_by_env.setdefault(env_idx, []).append(batch["raw_images.top_head"])
                self._left_imgs_by_env.setdefault(env_idx, []).append(batch["raw_images.hand_left"])
                self._right_imgs_by_env.setdefault(env_idx, []).append(batch["raw_images.hand_right"])

    def get_action(self, batch: dict[str, Any] | None = None, **kwargs: Any) -> list[dict[str, np.ndarray]]:
        """Return one executable RoboDojo action chunk."""
        if batch is not None:
            self._store_batch(0, batch)
            env_idx = 0
        else:
            if not self._obs_by_env:
                raise AssertionError("call update_obs first")
            env_idx = self._latest_env_idx_list[0]
        return action_chunk_to_robodojo_dicts(self._infer_chunk_wxyz(env_idx))

    def get_action_dict(self, batch: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
        """Compatibility helper for one-step runners."""
        return self.get_action(batch)[0]

    def get_action_batch(
        self, env_idx_list: list[int] | np.ndarray | int | None = None, **kwargs: Any
    ) -> list[list[dict[str, np.ndarray]]]:
        if not self._obs_by_env:
            raise AssertionError("call update_obs_batch first")
        if env_idx_list is None:
            env_idx_list = kwargs.get("obs")
        if env_idx_list is None:
            env_idx_list = self._latest_env_idx_list
        elif isinstance(env_idx_list, np.ndarray):
            env_idx_list = env_idx_list.reshape(-1).tolist()
        elif isinstance(env_idx_list, (int, np.integer)):
            env_idx_list = [int(env_idx_list)]
        else:
            env_idx_list = list(env_idx_list)
        return [
            action_chunk_to_robodojo_dicts(self._infer_chunk_wxyz(int(env_idx)))
            for env_idx in env_idx_list
        ]

    def _store_batch(self, env_idx: int, batch: dict[str, Any]) -> None:
        self._latest_env_idx_list = [env_idx]
        self._obs_by_env[env_idx] = batch
        if self.use_video_encoder:
            self._top_imgs_by_env.setdefault(env_idx, []).append(batch["raw_images.top_head"])
            self._left_imgs_by_env.setdefault(env_idx, []).append(batch["raw_images.hand_left"])
            self._right_imgs_by_env.setdefault(env_idx, []).append(batch["raw_images.hand_right"])

    @torch.no_grad()
    def _infer_chunk_wxyz(self, env_idx: int) -> np.ndarray:
        batch = self._obs_by_env[env_idx]

        initial_wxyz = batch["observation.state"][0, :16].copy()
        initial_xyzw = initial_wxyz.copy()
        initial_xyzw[3:7] = initial_wxyz[[4, 5, 6, 3]]
        initial_xyzw[11:15] = initial_wxyz[[12, 13, 14, 11]]
        if self.umi_coord_frame:
            initial_xyzw = convert_frame_robo_to_umi(
                initial_xyzw[None, :], convert_gripper=self.umi_gripper_space,
            )[0]

        net_batch = dict(batch)
        net_batch["observation.state"] = _convert_pose_robo_dojo(
            batch["observation.state"][0],
            self.norm_data["qpos_mean"],
            self.norm_data["qpos_std"],
            umi_coord_frame=self.umi_coord_frame,
            umi_gripper_space=self.umi_gripper_space,
        )

        if self.use_video_encoder:
            self._inject_history_stacks(net_batch, env_idx)

        feed = {}
        for key, value in net_batch.items():
            if key.startswith("raw_images.") or key == "task":
                continue
            if isinstance(value, np.ndarray):
                feed[key] = torch.from_numpy(value).to(self.weight_dtype).cuda()
            elif isinstance(value, torch.Tensor):
                feed[key] = value.to(self.weight_dtype).cuda()
            else:
                feed[key] = value
        feed["task"] = net_batch["task"]

        self.policy.reset()
        action0 = self.policy.select_action(feed)
        actions = [action0]
        for _ in range(len(self.policy._action_queue)):
            actions.append(self.policy._action_queue.popleft())
        actions = torch.cat(actions, dim=0).float().cpu().numpy()

        actions_xyzw = self._decode_actions(actions, initial_xyzw)
        if self.umi_coord_frame:
            actions_xyzw = convert_frame_umi_to_robo(
                actions_xyzw, convert_gripper=self.umi_gripper_space,
            )

        actions_wxyz = actions_xyzw.copy()
        actions_wxyz[:, 3:7] = actions_xyzw[:, [6, 3, 4, 5]]
        actions_wxyz[:, 11:15] = actions_xyzw[:, [14, 11, 12, 13]]

        if self.exc_action_interval > 1:
            needed = self.exc_action_size * self.exc_action_interval
            return actions_wxyz[1 : needed + 1 : self.exc_action_interval]
        return actions_wxyz[1 : self.exc_action_size + 1]

    def _decode_actions(self, actions: np.ndarray, initial_xyzw: np.ndarray) -> np.ndarray:
        if not self._with_abs:
            rel = actions * self.norm_data["act_std"] + self.norm_data["act_mean"]
            return relative_to_dual_arm_poses(rel, initial_xyzw)

        half = actions.shape[0] // 2 if actions.shape[0] % 2 == 0 else actions.shape[0]
        if self.blend_mode == "rel_only":
            rel = actions[:half, :20] * self.norm_data["act_std"] + self.norm_data["act_mean"]
            return relative_to_dual_arm_poses(rel, initial_xyzw)

        assert actions.shape[0] % 2 == 0, "rel_abs/abs need even token count"
        if self.blend_mode == "abs_only":
            abs_actions = (
                actions[half:, :20] * self.norm_data["act_std_abs"]
                + self.norm_data["act_mean_abs"]
            )
            out = np.zeros((abs_actions.shape[0], 16), dtype=abs_actions.dtype)
            for i in range(abs_actions.shape[0]):
                out[i] = np.concatenate(
                    [
                        pos_rotation_matrix_to_pos_quat(abs_actions[i, :10]),
                        pos_rotation_matrix_to_pos_quat(abs_actions[i, 10:20]),
                    ]
                )
            return out

        rel = actions[:half, :20] * self.norm_data["act_std"] + self.norm_data["act_mean"]
        rel_pose = relative_to_dual_arm_poses(rel, initial_xyzw)
        abs_actions = (
            actions[half:, :20] * self.norm_data["act_std_abs"]
            + self.norm_data["act_mean_abs"]
        )
        abs_pose = np.zeros((abs_actions.shape[0], 16), dtype=abs_actions.dtype)
        for i in range(abs_actions.shape[0]):
            abs_pose[i] = np.concatenate(
                [
                    pos_rotation_matrix_to_pos_quat(abs_actions[i, :10]),
                    pos_rotation_matrix_to_pos_quat(abs_actions[i, 10:20]),
                ]
            )
        return _blend_dual_arm_pose_quat(rel_pose, abs_pose)

    @staticmethod
    def _eval_history_indices(step_id: int, history_size: int, interval: int) -> list[int]:
        out = [max(step_id - (history_size - 1 - k) * interval, 0) for k in range(history_size)]
        out[-1] = step_id
        return out

    def _inject_history_stacks(self, batch: dict[str, Any], env_idx: int) -> None:
        history_size = self.img_history_size
        interval = max(1, self.img_history_interval // self.exc_action_interval)
        top_buf = self._top_imgs_by_env[env_idx]
        left_buf = self._left_imgs_by_env[env_idx]
        right_buf = self._right_imgs_by_env[env_idx]
        step_id = len(top_buf) - 1
        idx_list = self._eval_history_indices(step_id, history_size, interval)
        valid = [
            (step_id - (history_size - 1 - k) * interval) >= 0
            for k in range(history_size)
        ]

        def _stack(buf: list[np.ndarray]) -> torch.Tensor:
            frames = [buf[i] for i in idx_list]
            arr = torch.from_numpy(np.stack(frames, 0)).permute(0, 3, 1, 2).float() / 255.0
            for k, ok in enumerate(valid):
                if not ok:
                    arr[k].zero_()
            return arr.unsqueeze(0)

        batch["observation.images.top_head"] = _stack(top_buf)
        batch["observation.images.hand_left"] = _stack(left_buf)
        batch["observation.images.hand_right"] = _stack(right_buf)

    def reset(self) -> str:
        self.policy.reset()
        self._obs_by_env.clear()
        self._top_imgs_by_env.clear()
        self._left_imgs_by_env.clear()
        self._right_imgs_by_env.clear()
        self._latest_env_idx_list = [0]
        return "Hy-VLA RoboDojo wrapper reset"


def build_policy(usr_args: dict[str, Any]) -> HyVLARoboDojoPolicyWrapper:
    """Build a RoboDojo wrapper from a ``deploy_policy.yml``-style dict."""
    ckpt_path = usr_args["ckpt_path"]
    norm_path = usr_args.get("norm_path")
    if not norm_path and Path(ckpt_path).is_dir():
        cand = Path(ckpt_path) / "norm_stats.pkl"
        norm_path = str(cand) if cand.is_file() else None
    if not norm_path:
        raise ValueError(
            "norm_path is required (no norm_stats.pkl found next to the ckpt either)"
        )

    return HyVLARoboDojoPolicyWrapper(
        ckpt_path=ckpt_path,
        norm_path=norm_path,
        blend_mode=usr_args.get("blend_mode", "rel_only"),
        with_absolute=bool(usr_args.get("with_absolute", False)),
        exc_action_size=int(usr_args.get("exc_action_size", 10)),
        exc_action_interval=int(usr_args.get("exc_action_interval", 1)),
        img_history_size=int(usr_args.get("img_history_size", 6)),
        img_history_interval=int(usr_args.get("img_history_interval", 20)),
        vlm_model_path=usr_args.get("vlm_model_path"),
        umi_coord_frame=bool(usr_args.get("umi_coord_frame", True)),
        umi_gripper_space=bool(usr_args.get("umi_gripper_space", False)),
    )


__all__ = [
    "HyVLARoboDojoPolicyWrapper",
    "action16_to_robodojo_dict",
    "action_chunk_to_robodojo_dicts",
    "build_policy",
]
