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

"""Hy-VLA policy wrapper for RoboTwin evaluation.

Two execution modes are supported, chosen automatically from the value of
``HyVLAConfig.use_video_encoder`` written into the checkpoint's
``config.json``:

* **Single-frame** (``use_video_encoder=False``).
  One RGB frame per camera, fed to the ViT directly. Used by the
  pre-train checkpoint released alongside the paper.

* **MEM video-encoder** (``use_video_encoder=True``).
  ``K = img_history_size`` past RGB frames per camera (slot ``K-1`` is
  the current frame, earlier slots are clipped-to-zero at episode
  starts to match the training-time mask). Used by the post-train
  checkpoint.

Action decoding mirrors the training-time ``act_type``:

* **rel_only** (default, also the only legal mode for non-relabs ckpts).
  The full ``chunk`` 20-d output is treated as RT-relative and decoded
  into 16-d dual-arm PosQuat via ``relative_to_dual_arm_poses``.
* **rel_abs** / **abs_only** (relabs ckpts only, requires the loaded
  ``norm_stats.pkl`` to carry ``action_mean_abs`` / ``action_std_abs``).
  The network emits ``2 * chunk`` tokens; the first half is rel, the
  second half is abs (PosRotMat). ``rel_abs`` blends the two via
  slerp(0.5) on the quaternion and arithmetic mean on position/gripper;
  ``abs_only`` discards the rel half.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R, Slerp

from hy_vla import HyVLAConfig, HyVLA
from hy_vla.utils.transform_utils import (
    convert_frame_robo_to_umi,
    convert_frame_umi_to_robo,
)

from .transforms import (
    get_norm_data,
    pos_quat_to_pos_rotation_matrix,
    pos_rotation_matrix_to_pos_quat,
    relative_to_dual_arm_poses,
)


# ---------------------------------------------------------------------------
# Quaternion blending helpers (only used by blend_mode == "rel_abs")
# ---------------------------------------------------------------------------
def _slerp_quat_xyzw_half(q1_xyzw: np.ndarray, q2_xyzw: np.ndarray) -> np.ndarray:
    """Per-frame slerp(0.5) between two ``(N, 4)`` xyzw quaternion batches."""
    out = np.empty_like(q1_xyzw)
    dots = np.einsum("ij,ij->i", q1_xyzw, q2_xyzw)
    flip_mask = dots < 0.0
    q2_aligned = q2_xyzw.copy()
    q2_aligned[flip_mask] = -q2_aligned[flip_mask]
    for i in range(q1_xyzw.shape[0]):
        rots = R.from_quat(np.stack([q1_xyzw[i], q2_aligned[i]], axis=0))
        slerp = Slerp([0.0, 1.0], rots)
        out[i] = slerp([0.5]).as_quat()[0]
    return out


def _blend_dual_arm_pose_quat(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """1:1 blend two ``(chunk, 16)`` dual-arm PosQuat (xyzw) command chunks."""
    assert p1.shape == p2.shape and p1.shape[-1] == 16, (
        f"shape mismatch in dual-arm pose blend: {p1.shape} vs {p2.shape}"
    )
    out = np.empty_like(p1)
    out[:, 0:3] = 0.5 * (p1[:, 0:3] + p2[:, 0:3])
    out[:, 3:7] = _slerp_quat_xyzw_half(p1[:, 3:7], p2[:, 3:7])
    out[:, 7:8] = 0.5 * (p1[:, 7:8] + p2[:, 7:8])
    out[:, 8:11] = 0.5 * (p1[:, 8:11] + p2[:, 8:11])
    out[:, 11:15] = _slerp_quat_xyzw_half(p1[:, 11:15], p2[:, 11:15])
    out[:, 15:16] = 0.5 * (p1[:, 15:16] + p2[:, 15:16])
    return out


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
class HyVLAPolicyWrapper:
    """RoboTwin-facing wrapper around ``HyVLA``.

    Public surface (called by ``deploy_policy.py`` / RoboTwin's eval loop):

    * ``__init__(...)`` -- loads the ckpt and the unified norm pickle.
    * ``reset()`` -- clears action cache and per-episode buffers.
    * ``get_action(batch)`` -- consumes the dict produced by
      ``encode_obs`` and returns a single 16-d dual-arm PosQuat
      (RoboTwin wxyz layout) ready for ``TASK_ENV.take_action(...)``.
    """

    def __init__(
        self,
        ckpt_path: str,
        norm_path: str,
        *,
        blend_mode: str = "rel_only",
        exc_action_size: int = 20,
        img_history_size: int = 1,
        img_history_interval: int = 1,
        weight_dtype: torch.dtype = torch.bfloat16,
        vlm_model_path: str | None = None,
        umi_coord_frame: bool = False,
        umi_gripper_space: bool = False,
    ) -> None:
        # All architectural switches (chunk_size, use_video_encoder,
        # spacetime_layer_stride, past_drop_layer,
        # visual_segment_isolation) live in the ckpt's config.json and
        # are picked up automatically by ``HyVLAConfig.from_pretrained``.
        self.weight_dtype = weight_dtype
        self.config = HyVLAConfig.from_pretrained(ckpt_path)
        self.policy = HyVLA.from_pretrained(
            ckpt_path, config=self.config, vlm_model_path=vlm_model_path,
        )
        self.policy.enable_video_encoder_if_needed()
        self.policy.cuda()
        self.policy.eval()
        self.policy = self.policy.to(self.weight_dtype)

        # Normalization stats (schema: see ``transforms.get_norm_data``).
        self.norm_data = get_norm_data(norm_path)
        self._has_abs_stats = (
            self.norm_data.get("act_mean_abs") is not None
            and self.norm_data.get("act_std_abs") is not None
        )
        if self._has_abs_stats:
            assert self.norm_data["act_mean_abs"].shape == self.norm_data["act_mean"].shape, (
                f"abs act_mean shape {self.norm_data['act_mean_abs'].shape} must match "
                f"rel act_mean shape {self.norm_data['act_mean'].shape}"
            )

        if blend_mode not in ("rel_abs", "rel_only", "abs_only"):
            raise ValueError(
                f"blend_mode must be one of rel_abs|rel_only|abs_only, got {blend_mode!r}"
            )
        if not self._has_abs_stats and blend_mode != "rel_only":
            raise ValueError(
                f"blend_mode={blend_mode!r} requires the loaded norm pkl to "
                f"carry 'action_mean_abs'/'action_std_abs' (relabs-trained "
                f"ckpt); the pkl at {norm_path!r} does not. Use "
                f"blend_mode=rel_only or pass a relabs norm pkl."
            )
        self.blend_mode = blend_mode

        # Per-episode state.
        self.exc_action_size = int(exc_action_size)
        self.action_cache: deque[np.ndarray] = deque()

        # MEM video-encoder cadence.
        self.use_video_encoder = bool(self.config.use_video_encoder)
        self.img_history_size = int(img_history_size)
        self.img_history_interval = int(img_history_interval)
        self._top_imgs: list[np.ndarray] = []
        self._left_imgs: list[np.ndarray] = []
        self._right_imgs: list[np.ndarray] = []

        # UMI coordinate frame (must match training config).
        self.umi_coord_frame = bool(umi_coord_frame)
        self.umi_gripper_space = bool(umi_gripper_space)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> str:
        self.policy.reset()
        self.action_cache.clear()
        self._top_imgs.clear()
        self._left_imgs.clear()
        self._right_imgs.clear()
        return "Hy-VLA wrapper reset"

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def get_action(self, batch: dict[str, Any]) -> np.ndarray:
        """Return one 16-d action (RoboTwin wxyz layout) per call.

        The wrapper amortizes a single network call over
        ``exc_action_size`` consecutive eval steps via ``action_cache``,
        which keeps the overall RoboTwin step cadence equal to one model
        forward per ``exc_action_size`` env steps.
        """
        # Always grow the per-camera frame buffers on every call so that
        # the K-frame stack stays time-aligned even while we are serving
        # cached actions.
        if self.use_video_encoder:
            self._append_history_frames(batch)

        if len(self.action_cache) > 0:
            return self.action_cache.popleft()

        initial_ee_pose_wxyz = batch["observation.state"][0, :16].copy()
        # wxyz → xyzw
        initial_ee_pose_xyzw = initial_ee_pose_wxyz.copy()
        initial_ee_pose_xyzw[3:7] = initial_ee_pose_wxyz[[4, 5, 6, 3]]
        initial_ee_pose_xyzw[11:15] = initial_ee_pose_wxyz[[12, 13, 14, 11]]
        # RoboTwin → UMI coordinate frame (for RT-relative decode),
        # only when norm_stats.pkl was computed in UMI frame.
        if self.umi_coord_frame:
            initial_ee_pose_umi = convert_frame_robo_to_umi(
                initial_ee_pose_xyzw[None, :],
                convert_gripper=self.umi_gripper_space,
            )[0]
        else:
            initial_ee_pose_umi = initial_ee_pose_xyzw.copy()

        # State: wxyz → xyzw → (optional) UMI → PosRotMat → normalize
        state = batch["observation.state"][0].copy()
        state[3:7] = state[[4, 5, 6, 3]]
        state[11:15] = state[[12, 13, 14, 11]]
        if self.umi_coord_frame:
            state = convert_frame_robo_to_umi(
                state[None, :], convert_gripper=self.umi_gripper_space,
            )[0]
        ee_prop = np.concatenate([
            pos_quat_to_pos_rotation_matrix(state[:3], state[3:7], state[7]),
            pos_quat_to_pos_rotation_matrix(state[8:11], state[11:15], state[15]),
        ])
        batch["observation.state"] = ((ee_prop - self.norm_data["qpos_mean"]) / self.norm_data["qpos_std"])[None, ...]

        if self.use_video_encoder:
            self._inject_history_stacks(batch)

        for k, v in batch.items():
            if isinstance(v, np.ndarray) and not k.startswith("raw_images.") and k != "task":
                batch[k] = torch.from_numpy(v).to(self.weight_dtype).cuda()
            elif isinstance(v, torch.Tensor):
                batch[k] = v.to(self.weight_dtype).cuda()

        self.policy.reset()
        action0 = self.policy.select_action(batch)
        actions = [action0]
        for _ in range(len(self.policy._action_queue)):
            actions.append(self.policy._action_queue.popleft())
        actions = torch.cat(actions, dim=0).cpu().numpy()

        actions_xyzw = self._decode_actions(actions, initial_ee_pose_umi)

        # UMI → RoboTwin coordinate frame, then xyzw → wxyz
        if self.umi_coord_frame:
            actions_xyzw = convert_frame_umi_to_robo(
                actions_xyzw, convert_gripper=self.umi_gripper_space,
            )
        actions_wxyz = actions_xyzw.copy()
        actions_wxyz[:, 3:7] = actions_xyzw[:, [6, 3, 4, 5]]
        actions_wxyz[:, 11:15] = actions_xyzw[:, [14, 11, 12, 13]]

        # Cache the rest of the chunk for subsequent calls.
        for action in actions_wxyz[1 : self.exc_action_size]:
            self.action_cache.append(action)
        return actions_wxyz[0]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _decode_actions(self, actions: np.ndarray, initial_ee_pose_xyzw: np.ndarray) -> np.ndarray:
        """Apply rel/abs/blend decoding to ``(T, 20)`` raw network output."""
        if not self._has_abs_stats:
            actions = actions * self.norm_data["act_std"] + self.norm_data["act_mean"]
            return relative_to_dual_arm_poses(actions, initial_ee_pose_xyzw)

        # Relabs ckpt: T == 2 * chunk; first half is RT-rel, second is abs.
        assert actions.shape[0] % 2 == 0, (
            f"with_absolute path expects an even number of action tokens, got {actions.shape[0]}"
        )
        half = actions.shape[0] // 2

        actions_p1 = None
        actions_p2 = None
        if self.blend_mode in ("rel_abs", "rel_only"):
            rel = (
                actions[:half, :20] * self.norm_data["act_std"] + self.norm_data["act_mean"]
            )
            actions_p1 = relative_to_dual_arm_poses(rel, initial_ee_pose_xyzw)

        if self.blend_mode in ("rel_abs", "abs_only"):
            abs_ = (
                actions[half:, :20] * self.norm_data["act_std_abs"]
                + self.norm_data["act_mean_abs"]
            )
            n_chunk = abs_.shape[0]
            actions_p2 = np.zeros((n_chunk, 16), dtype=abs_.dtype)
            for i in range(n_chunk):
                left = pos_rotation_matrix_to_pos_quat(abs_[i, :10])
                right = pos_rotation_matrix_to_pos_quat(abs_[i, 10:20])
                actions_p2[i] = np.concatenate([left, right])

        if self.blend_mode == "rel_abs":
            return _blend_dual_arm_pose_quat(actions_p1, actions_p2)
        if self.blend_mode == "rel_only":
            return actions_p1
        # abs_only
        return actions_p2

    # --- MEM video-encoder helpers -------------------------------------
    def _append_history_frames(self, batch: dict[str, Any]) -> None:
        self._top_imgs.append(batch["raw_images.top_head"])
        self._left_imgs.append(batch["raw_images.hand_left"])
        self._right_imgs.append(batch["raw_images.hand_right"])

    @staticmethod
    def _eval_history_indices(step_id: int, history_size: int, interval: int) -> list[int]:
        """Equally-spaced past-frame indices on the per-camera buffer.

        Slot ``history_size - 1`` is the current frame; earlier slots are
        ``step_id - (K-1-k) * S`` clipped to 0 (matching the training-time
        ``get_history_indices(random_sample=False)``).
        """
        assert history_size >= 1 and interval >= 1
        out = []
        for k in range(history_size):
            end = step_id - (history_size - 1 - k) * interval
            out.append(max(end, 0))
        out[-1] = step_id
        return out

    def _inject_history_stacks(self, batch: dict[str, Any]) -> None:
        """Build the ``(1, K, C, H, W)`` visual stacks for the MEM ViT."""
        K = self.img_history_size
        S = self.img_history_interval
        step_id = len(self._top_imgs) - 1
        idx_list = self._eval_history_indices(step_id, K, S)
        # Episode-start slots whose un-clipped index is <0 must be zeroed
        # to match the training-time padding (the dataset replaces those
        # frames with all-zero pixels before the ViT sees them).
        valid = [(step_id - (K - 1 - k) * S) >= 0 for k in range(K)]

        def _stack(buf: list[np.ndarray]) -> torch.Tensor:
            frames = [buf[i] for i in idx_list]
            arr = np.stack(frames, axis=0)
            arr = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
            for k, ok in enumerate(valid):
                if not ok:
                    arr[k].zero_()
            return arr.unsqueeze(0)  # (1, K, C, H, W)

        batch["observation.images.top_head"] = _stack(self._top_imgs)
        batch["observation.images.hand_left"] = _stack(self._left_imgs)
        batch["observation.images.hand_right"] = _stack(self._right_imgs)


# ---------------------------------------------------------------------------
# Factory: resolve norm pickle paths from a ckpt directory if not given,
# then instantiate the wrapper.
# ---------------------------------------------------------------------------
def build_policy(usr_args: dict[str, Any]) -> HyVLAPolicyWrapper:
    """Build a ``HyVLAPolicyWrapper`` from a ``deploy_policy.yml``-style dict.

    The single norm pickle defaults to ``<ckpt_path>/norm_stats.pkl``
    (the layout used by the released HuggingFace repos); an explicit
    override via ``norm_path`` always takes precedence. The same yml
    works for both pre-train (rel-only) and post-train (rel+abs)
    checkpoints because the abs half is opt-in inside the pkl.
    """
    ckpt_path = usr_args["ckpt_path"]

    norm_path = usr_args.get("norm_path")
    if not norm_path and Path(ckpt_path).is_dir():
        cand = Path(ckpt_path) / "norm_stats.pkl"
        norm_path = str(cand) if cand.is_file() else None
    if not norm_path:
        raise ValueError(
            "norm_path is required (no norm_stats.pkl found next to the ckpt either)"
        )

    return HyVLAPolicyWrapper(
        ckpt_path=ckpt_path,
        norm_path=norm_path,
        blend_mode=usr_args.get("blend_mode", "rel_only"),
        exc_action_size=int(usr_args.get("exc_action_size", 20)),
        img_history_size=int(usr_args.get("img_history_size", 1)),
        img_history_interval=int(usr_args.get("img_history_interval", 1)),
        vlm_model_path=usr_args.get("vlm_model_path"),
        umi_coord_frame=bool(usr_args.get("umi_coord_frame", False)),
        umi_gripper_space=bool(usr_args.get("umi_gripper_space", False)),
    )


__all__ = ["HyVLAPolicyWrapper", "build_policy"]
