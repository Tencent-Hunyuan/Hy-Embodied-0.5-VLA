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

"""HDF5-backed RoboTwin dataset loader for Hy-VLA training.

The dataset reads RoboTwin-format episode HDF5 files listed in a CSV
index (``assets/dataset_index.csv``) and emits per-step samples consumed
by ``hy_vla.data.vla_dataset.VLADataset``.
"""

import os
import csv
import json
import pickle
import random
import h5py
import cv2
import numpy as np
from hy_vla.utils.transform_utils import (
    convert_PosQuat2PosRotationMatrix_batch,
    dual_arm_poses_to_relative,
    convert_frame_robo_to_umi,
)


def _load_dataset_csv(path: str, hdf5_dir: str) -> list[dict]:
    """Load the open-source minimal CSV dataset index.

    Schema (5 columns, see ``assets/dataset_index.csv``):

        episode_dir          relative path of the per-episode directory
        hdf5_name            file name of the hdf5 inside that directory
        instruction_name     file name of the instructions json next to the hdf5
        num_frames           int, raw frame count of the hdf5
        is_dirty             0/1, whether the episode is filtered out by
                             ``dataset.filter_dirty=True``

    File names in the CSV are relative to ``hdf5_dir`` and re-glued to
    absolute paths here.
    """
    eps: list[dict] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "episode_dir", "hdf5_name", "instruction_name",
            "num_frames", "is_dirty",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"dataset CSV {path} is missing required columns: "
                f"{sorted(missing)}"
            )
        for row in reader:
            ep_dir_abs = os.path.join(hdf5_dir, row["episode_dir"])
            eps.append({
                "episode_dir": row["episode_dir"],
                "hdf5_path": os.path.join(ep_dir_abs, row["hdf5_name"]),
                "instruction_path": os.path.join(
                    ep_dir_abs, row["instruction_name"]
                ),
                "num_frames": int(row["num_frames"]),
                "is_dirty": bool(int(row["is_dirty"])),
            })
    return eps


def pad_vector(vector, new_dim):
    """Can be (sequence_length x features_dimension)"""
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = np.zeros(shape)
    new_vector[..., :current_dim] = vector
    return new_vector


def get_history_indices(step_id, history_size, interval, random_sample=True):
    """Compute history frame indices for the K-frame image stack.

    Returns K indices (history + current). Slot K-1 is forced to
    ``step_id``; for slot k, the target end index is
    ``end = step_id - (K - 1 - k) * interval``.

      * ``random_sample=True``  (train): uniformly sample in
        ``[end - interval + 1, end]``, clipped to ``>= 0``.
      * ``random_sample=False`` (eval): take ``max(end, 0)``.

    Out-of-range indices collapse to 0 (the downstream code treats the
    duplicated frame 0 as padding).

    Note: ``history_size`` here is the TOTAL count (history + current),
    matching how the dataset pipeline consumes ``img_history_size``
    end-to-end. The upstream video-encoder repo defines it as
    history-only; the two conventions differ by 1 in the yaml.
    """
    assert history_size >= 1
    assert interval >= 1
    indices = []
    for k in range(history_size):
        end = step_id - (history_size - 1 - k) * interval
        if random_sample:
            start = max(end - interval + 1, 0)
            end_clamped = max(end, 0)
            if end_clamped < start:
                idx = start
            else:
                idx = random.randint(start, end_clamped)
        else:
            idx = max(end, 0)
        indices.append(idx)
    indices[-1] = step_id
    return indices


class RoboTwinVLADataset:
    """
    This class is used to sample episodes from the embododiment dataset
    stored in HDF5.
    """

    def __init__(self, config) -> None:
        # The episode list comes from a CSV (``assets/dataset_index.csv``);
        # ``HDF5_DIR`` is just the prefix used to re-build absolute paths.
        HDF5_DIR = config.dataset.hdf5_dir
        self.HDF5_DIR = HDF5_DIR

        # ``dataset.dataset_index_csv`` overrides; default is
        # ``<repo_root>/assets/dataset_index.csv`` (this file lives at
        # ``<repo_root>/hy_vla/data/robotwin_dataset.py``).
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        default_csv_path = os.path.join(repo_root, "assets", "dataset_index.csv")
        dataset_index_csv = getattr(
            config.dataset, "dataset_index_csv", default_csv_path
        )
        assert os.path.isfile(dataset_index_csv), (
            f"dataset_index CSV not found: {dataset_index_csv}. "
            f"Set `dataset.dataset_index_csv` or place the CSV at "
            f"`assets/dataset_index.csv` under the repo root."
        )
        print(f"[robotwin_dataset] dataset_index_csv: {dataset_index_csv}")
        all_episodes = _load_dataset_csv(dataset_index_csv, HDF5_DIR)

        # Group by subset (first segment of ``episode_dir``) just to keep
        # enumeration order stable; rows with ``is_dirty=True`` are dropped
        # iff ``dataset.filter_dirty`` is set (default False).
        self.filter_dirty = bool(getattr(config.dataset, "filter_dirty", False))
        eps_by_subset: dict[str, list] = {}
        n_total = 0
        n_dirty_dropped = 0
        for ep in all_episodes:
            n_total += 1
            if self.filter_dirty and ep["is_dirty"]:
                n_dirty_dropped += 1
                continue
            subset_name = ep["episode_dir"]
            eps_by_subset.setdefault(subset_name, []).append(ep)
        print(
            f"[robotwin_dataset] filter_dirty: {self.filter_dirty} -- "
            f"{n_total - n_dirty_dropped}/{n_total} episodes kept "
            f"({n_dirty_dropped} dirty dropped)"
        )

        # When ``use_video_encoder=True`` we emit a per-camera K-frame
        # stack (K = ``img_history_size``); otherwise only the current
        # frame is delivered.
        self.use_video_encoder = bool(getattr(config.dataset, "use_video_encoder", False))
        print(f"[robotwin_dataset] use_video_encoder: {self.use_video_encoder}")

        self.action_type = config.dataset.act_type
        print(f"[robotwin_dataset] action_type: {self.action_type}")

        # Optional: convert EEF poses to UMI coordinate frame.
        self.umi_coord_frame = bool(getattr(config.dataset, "umi_coord_frame", False))
        if self.umi_coord_frame:
            print(
                "[robotwin_dataset] umi_coord_frame=True: "
                "applying world rotation W + local column permutation P "
                "(W @ R_rd @ P, pos → W @ pos). "
                "WARNING: normalization stats must be regenerated "
                "with the same coord frame!"
            )
        self.umi_gripper_space = bool(getattr(config.dataset, "umi_gripper_space", False))
        if self.umi_coord_frame and self.umi_gripper_space:
            print(
                "[robotwin_dataset] umi_gripper_space=True: "
                "gripper values will also be mapped RoboTwin [1,0] → UMI [0,90]."
            )

        self.downsample_rate = config.dataset.downsample_rate
        print(f"[robotwin_dataset] downsample_rate: {self.downsample_rate}")

        # Norm-stats pickle schema (produced by
        # ``utils/hdf5_normalization_process_relabs_chunk_ee.py``)::
        #
        #   {
        #     "qpos_mean":         (20,),
        #     "qpos_std":          (20,),
        #     "action_mean":       (chunk, 20),    # rel half (always)
        #     "action_std":        (chunk, 20),
        #     "action_mean_abs":   (chunk, 20),    # required iff with_absolute
        #     "action_std_abs":    (chunk, 20),
        #   }
        def _load_mean_std(path, chunk_slice=None, with_abs=False):
            if not os.path.exists(path):
                raise ValueError(f"File does not exist: {path}")
            with open(path, "rb") as fp:
                info = pickle.load(fp)
            qm = np.array(info["qpos_mean"], dtype=np.float32)
            qs = np.array(info["qpos_std"], dtype=np.float32)
            am = np.array(info["action_mean"], dtype=np.float32)
            as_ = np.array(info["action_std"], dtype=np.float32)
            am_absolute = None
            as_absolute = None
            if with_abs:
                if "action_mean_abs" not in info or "action_std_abs" not in info:
                    raise KeyError(
                        f"act_type contains 'with_absolute' but the norm pkl at "
                        f"{path} does not carry 'action_mean_abs' / "
                        f"'action_std_abs'. Re-generate it with "
                        f"utils/hdf5_normalization_process_relabs_chunk_ee.py."
                    )
                am_absolute = np.array(info["action_mean_abs"], dtype=np.float32)
                as_absolute = np.array(info["action_std_abs"], dtype=np.float32)
                if am_absolute.shape != am.shape:
                    raise ValueError(
                        f"action_mean_abs shape {am_absolute.shape} must match "
                        f"action_mean shape {am.shape} in {path}"
                    )
            if chunk_slice is not None:
                n = int(chunk_slice)
                assert n > 0 and n <= am.shape[0], (
                        f"chunk_slice={n} out of range [1, {am.shape[0]}] for {path}"
                )
                print(f"[robotwin_dataset] mean_std slice: {am.shape[0]} -> {n} ({path})")
                am = am[:n]
                as_ = as_[:n]
                if am_absolute is not None:
                    am_absolute = am_absolute[:n]
                    as_absolute = as_absolute[:n]
            return qm, qs, am, as_, am_absolute, as_absolute

        if not hasattr(config.dataset, "mean_std_path"):
            raise ValueError(
                "dataset.mean_std_path is required: the dataset always "
                "normalizes states/actions with the loaded pkl."
            )
        mean_std_path = config.dataset.mean_std_path
        print(f"[robotwin_dataset] mean_std_path: {mean_std_path}")
        with_abs = "with_absolute" in self.action_type
        (
            self.qpos_mean,
            self.qpos_std,
            self.act_mean,
            self.act_std,
            _act_mean_abs,
            _act_std_abs,
        ) = _load_mean_std(
            mean_std_path,
            chunk_slice=getattr(config.dataset, "mean_std_chunk_slice", None),
            with_abs=with_abs,
        )

        # For ``_with_absolute``: cat along axis 0 so ``self.act_mean``
        # becomes (2*chunk, 20), aligning row-wise with the doubled-time
        # actions tensor produced by ``parse_hdf5_file``
        # (rows [0..chunk-1] = RT_relative, rows [chunk..2*chunk-1] =
        # absolute PosRotMat over the SAME chunk frames).
        if with_abs:
            self.act_mean = np.concatenate([self.act_mean, _act_mean_abs], axis=0)
            self.act_std = np.concatenate([self.act_std, _act_std_abs], axis=0)
            print(
                f"[robotwin_dataset] with_absolute: act_mean/std cat along time axis -> "
                f"{self.act_mean.shape}"
            )

        self.CHUNK_SIZE = config["dataset"]["action_chunk_size"]

        # Force K=1 on the single-frame pathway; the video-encoder path
        # honours the YAML.
        raw_img_history_size = int(config["dataset"]["img_history_size"])
        if self.use_video_encoder:
            self.IMG_HISORY_SIZE = raw_img_history_size
        else:
            if raw_img_history_size != 1:
                print(
                    f"[WARN] use_video_encoder=False but img_history_size="
                    f"{raw_img_history_size}; forcing img_history_size=1 "
                    f"for the single-frame pathway."
                )
            self.IMG_HISORY_SIZE = 1

        # ``img_history_interval`` is in ACTION steps (post-downsample);
        # the effective raw-frame stride is interval * downsample_rate.
        # K==1 makes ``get_history_indices`` return ``[step_id]`` regardless
        # of these values, so no extra branching is needed below.
        self.IMG_HISTORY_INTERVAL = int(
            config["dataset"].get("img_history_interval", 1)
        )
        assert self.IMG_HISTORY_INTERVAL >= 1, "img_history_interval must be >= 1"
        self.IMG_HISTORY_RANDOM_SAMPLE = bool(
            config["dataset"].get("img_history_random_sample", True)
        )
        print(
            f"[robotwin_dataset] img_history_size: {self.IMG_HISORY_SIZE}, "
            f"img_history_interval: {self.IMG_HISTORY_INTERVAL}, "
            f"img_history_random_sample: {self.IMG_HISTORY_RANDOM_SAMPLE}"
        )
        self.STATE_DIM = config["dataset"]["state_dim"]

        # Flat episode pool: subset-grouped (sorted by subset name), CSV
        # order preserved within each subset.
        self.episodes = []
        for subset_name in sorted(eps_by_subset.keys()):
            self.episodes.extend(eps_by_subset[subset_name])

        print(f"[robotwin_dataset] num episodes: {len(self.episodes)}")

        # Deterministic mode: expose a finite ordered list of
        # (episode_index, raw_step) pairs as the dataset index space
        # (matches the support of the random sampler exactly:
        # ``np.random.randint(0, num_steps)``). Cross-rank disjointness is
        # delegated to a ``DistributedSampler`` on top.
        self.deterministic = bool(getattr(config.dataset, "deterministic", False))
        self.deterministic_index = None
        if self.deterministic:
            pairs: list[tuple[int, int]] = []
            for ep_idx, ep in enumerate(self.episodes):
                n_raw = int(ep["num_frames"])
                for t in range(n_raw):
                    pairs.append((ep_idx, t))

            self.deterministic_index = pairs
            print(
                f"[robotwin_dataset] deterministic=True: enumerated "
                f"{len(self.deterministic_index)} (episode, raw_step) pairs "
                f"across {len(self.episodes)} episodes"
            )
            assert len(self.deterministic_index) > 0, (
                "deterministic mode produced an empty index; check "
                "num_frames in dataset_index.csv"
            )

    def __len__(self):
        if self.deterministic and self.deterministic_index is not None:
            return len(self.deterministic_index)
        return len(self.episodes)

    def get_item(self, index: int = None):
        """Get a training sample.

        Args:
            index (int, optional): the dataset-side index. Semantics depend on
                ``self.deterministic``:

                * deterministic=False (default): ``index`` is treated as a
                  *flat episode index* into ``self.episodes``; the step within
                  the episode is sampled randomly inside ``parse_hdf5_file``.
                  ``index=None`` triggers fully random (episode, step)
                  sampling.
                * deterministic=True: ``index`` is a *global step index*
                  into ``self.deterministic_index``; it uniquely picks
                  both the episode AND the raw frame, AND is forwarded
                  to ``parse_hdf5_file`` as ``forced_step_id`` so the
                  step is bit-deterministic. ``index=None`` is rejected
                  in this mode (the caller must drive enumeration).

        Returns:
           sample (dict): a dictionary containing the training sample.
        """
        if self.deterministic:
            assert index is not None, (
                "deterministic=True requires an explicit dataset index; "
                "the random ``index=None`` path is disabled."
            )
            N = len(self.deterministic_index)
            i = int(index) % N
            attempts = 0
            while True:
                ep_idx, raw_step = self.deterministic_index[i]
                ep = self.episodes[ep_idx]
                # ``instruction_offset = i`` makes every global pair pick
                # its own instruction bucket, maximizing coverage.
                valid, sample = self.parse_hdf5_file(
                    ep,
                    forced_step_id=int(raw_step),
                    instruction_offset=int(i),
                )
                if valid:
                    return sample
                # Skip to next pair on degenerate episodes.
                i = (i + 1) % N
                attempts += 1
                assert attempts < N, (
                    "deterministic mode: every (episode, step) pair was "
                    "rejected by parse_hdf5_file -- check data integrity"
                )

        # Random path: uniform over the flat episode list.
        while True:
            if index is None:
                ep = self.episodes[np.random.randint(0, len(self.episodes))]
            else:
                ep = self.episodes[index]
            valid, sample = self.parse_hdf5_file(ep)
            if valid:
                return sample

    def parse_hdf5_file(self, ep, forced_step_id=None, instruction_offset=0):
        """[Modify] Parse a hdf5 file to generate a training sample at
            a random timestep, OR at a caller-specified raw frame.

        Args:
            ep (dict): an episode descriptor from the dataset CSV
                (``assets/dataset_index.csv``). Must carry at least
                ``hdf5_path`` (absolute), ``instruction_path`` (absolute)
                and ``num_frames``.
            forced_step_id (int, optional): when not None, bypass the
                in-episode random step sampling and use this raw hdf5
                frame index as the current step (``0 <= t < num_frames``).
                The instruction is ALSO derived deterministically (hash
                of ``(hdf5_path, forced_step_id)`` mod len(``seen``)).
            instruction_offset (int, optional): only meaningful with
                ``forced_step_id``; added MOD len(``seen``) to the
                hash-derived bucket so different global pairs pick
                different instructions. Defaults to 0.

        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "meta": {
                        "dataset_name": str,    # the name of your dataset.
                        "#steps": int,          # the number of steps in the episode,
                                                # also the total timesteps.
                        "instruction": str      # the language instruction for this episode.
                    },
                    "step_id": int,             # the index of the sampled step,
                                                # also the timestep t.
                    "state": ndarray,           # state[t], (1, STATE_DIM).
                    "actions": ndarray,         # action[t:t+CHUNK_SIZE], (CHUNK_SIZE, STATE_DIM).
                    "state_indicator", ndarray, # indicates the validness of each dim, (STATE_DIM,).
                    "cam_high": ndarray,        # external camera image, (IMG_HISORY_SIZE, H, W, 3)
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_high_mask": ndarray,   # indicates the validness of each timestep, (IMG_HISORY_SIZE,) boolean array.
                                                # For the first IMAGE_HISTORY_SIZE-1 timesteps, the mask should be False.
                    "cam_left_wrist": ndarray,  # left wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_left_wrist_mask": ndarray,
                    "cam_right_wrist": ndarray, # right wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                                                # If only one wrist, make it right wrist, plz.
                    "cam_right_wrist_mask": ndarray
                } or None if the episode is invalid.
        """
        file_path = ep["hdf5_path"]
        with h5py.File(file_path, "r") as f:
            if "relative_chunk_ee" not in self.action_type:
                raise ValueError(
                    f"Unsupported action_type='{self.action_type}'. "
                    f"Expected 'relative_chunk_ee_RT' or "
                    f"'relative_chunk_ee_RT_with_absolute'."
                )

            qpos = f["observations"]["eepos"][:]

            # robotwin quaternion convention: wxyz -> xyzw.
            qpos_converted = qpos.copy()
            qpos_converted[:, 3:7] = qpos[:, [4, 5, 6, 3]]
            qpos_converted[:, 11:15] = qpos[:, [12, 13, 14, 11]]
            qpos = qpos_converted

            # Optional: convert to UMI coordinate frame.
            if self.umi_coord_frame:
                qpos = convert_frame_robo_to_umi(qpos, convert_gripper=self.umi_gripper_space)

            target_qpos = qpos.copy()
            qpos = convert_PosQuat2PosRotationMatrix_batch(qpos)

            num_steps = qpos.shape[0]

            first_idx = 0
            if forced_step_id is not None:
                t = int(forced_step_id)
                if t < first_idx or t >= num_steps:
                    return False, None
                step_id = t
            else:
                step_id = np.random.randint(first_idx, num_steps)
            c_id = step_id
            M = num_steps

            img_root = f["observations"]["images"]

            instructions_path = ep["instruction_path"]
            with open(instructions_path, "r") as f_instr:
                instruction_dict = json.load(f_instr)
            instructions_names = instruction_dict["seen"]
            if forced_step_id is not None:
                # Deterministic instruction: content-stable FNV-1a hash
                # of (hdf5_path, raw_step) -> bucket in ``seen``.
                # Python's ``hash`` is salted across processes, so we
                # roll our own.
                if len(instructions_names) == 1:
                    instruction = instructions_names[0]
                else:
                    h = 1469598103934665603  # FNV-1a 64-bit offset basis
                    for b in file_path.encode("utf-8"):
                        h ^= b
                        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
                    s = int(forced_step_id) & 0xFFFFFFFFFFFFFFFF
                    for shift in (0, 8, 16, 24, 32, 40, 48, 56):
                        h ^= (s >> shift) & 0xFF
                        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
                    bucket = (h + int(instruction_offset)) % len(instructions_names)
                    instruction = instructions_names[bucket]
            else:
                instruction = np.random.choice(instructions_names)

            meta = {"#steps": num_steps, "step_id": step_id, "instruction": instruction}

            state = qpos[c_id : c_id + 1]

            sample_ds = self.downsample_rate

            # Action chunk: slot k -> raw index c_id + k*sample_ds,
            # clipped to M-1 ("hold-last" padding).
            chunk_offsets = np.arange(self.CHUNK_SIZE, dtype=np.int64) * sample_ds
            chunk_compressed = np.minimum(c_id + chunk_offsets, M - 1)
            actions = target_qpos[chunk_compressed]

            # For ``_with_absolute``: also compute the absolute PosRotMat
            # target on the SAME chunk frames and cat along axis 0, so
            # the final actions tensor is (2*chunk, 20):
            # rows [0..chunk-1] = RT_relative, rows [chunk..2*chunk-1] =
            # absolute PosRotMat. ``self.act_mean / self.act_std`` was
            # already cat'd to (2*chunk, 20) in __init__.
            if "with_absolute" in self.action_type:
                actions_abs = convert_PosQuat2PosRotationMatrix_batch(
                    target_qpos[chunk_compressed], quat_order="xyzw"
                )
            else:
                actions_abs = None

            actions = dual_arm_poses_to_relative(actions)

            if actions_abs is not None:
                # Cat BEFORE normalize so act_mean/std aligns row-wise.
                actions = np.concatenate([actions, actions_abs], axis=0)

            state = (state - self.qpos_mean) / self.qpos_std
            actions = (actions - self.act_mean) / self.act_std

            state = pad_vector(state, self.STATE_DIM)
            state_indicator = pad_vector(np.ones(qpos.shape[-1]), self.STATE_DIM)
            actions = pad_vector(actions, self.STATE_DIM)

            def parse_img(key):
                if key not in img_root:
                    return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0), dtype=np.uint8)

                dataset = img_root[key]
                if len(dataset) == 0:
                    return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0), dtype=np.uint8)

                # K=1 degenerates to [c_id]; otherwise step in raw frames
                # so history spacing aligns with the action timeline.
                raw_interval = self.IMG_HISTORY_INTERVAL * sample_ds
                planned_indices = np.asarray(
                    get_history_indices(
                        c_id,
                        self.IMG_HISORY_SIZE,
                        raw_interval,
                        random_sample=self.IMG_HISTORY_RANDOM_SAMPLE,
                    ),
                    dtype=int,
                )
                planned_indices = np.clip(planned_indices, 0, len(dataset) - 1)

                unique_sorted = np.unique(planned_indices)
                raw_batch = dataset[unique_sorted]
                index_to_pos = {int(idx): pos for pos, idx in enumerate(unique_sorted)}

                imgs = []
                for idx in planned_indices:
                    raw = raw_batch[index_to_pos[int(idx)]]
                    if isinstance(raw, np.ndarray) and raw.ndim == 3:
                        imgs.append(raw)
                    else:
                        try:
                            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                            if img is None:
                                continue
                            imgs.append(img)
                        except Exception:
                            continue

                if len(imgs) == 0:
                    return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0), dtype=np.uint8)

                imgs = np.stack(imgs)
                if imgs.shape[0] < self.IMG_HISORY_SIZE:
                    # Left-pad with the first image.
                    imgs = np.concatenate(
                        [np.tile(imgs[:1], (self.IMG_HISORY_SIZE - imgs.shape[0], 1, 1, 1)), imgs], axis=0
                    )
                return imgs

            cam_high = parse_img("cam_high")

            # Slot k is valid iff its un-clipped end index
            # (``c_id - (K-1-k) * S``) is >= ``first_idx``; otherwise it
            # was collapsed to frame 0 by ``get_history_indices`` and
            # downstream code should skip it. K==1 -> always valid.
            K = self.IMG_HISORY_SIZE
            S = self.IMG_HISTORY_INTERVAL * sample_ds
            cam_high_mask = np.array(
                [(c_id - (K - 1 - k) * S) >= first_idx for k in range(K)],
                dtype=bool,
            )
            cam_left_wrist = parse_img("cam_left_wrist")
            cam_left_wrist_mask = cam_high_mask.copy()
            cam_right_wrist = parse_img("cam_right_wrist")
            cam_right_wrist_mask = cam_high_mask.copy()

            sample: dict[str, np.ndarray] = {
                "meta": meta,
                "state": state,
                "actions": actions,
                "state_indicator": state_indicator,
                "cam_high": cam_high,  # (IMG_HISORY_SIZE,h,w,3)
                "cam_high_mask": cam_high_mask,
                "cam_left_wrist": cam_left_wrist,
                "cam_left_wrist_mask": cam_left_wrist_mask,
                "cam_right_wrist": cam_right_wrist,
                "cam_right_wrist_mask": cam_right_wrist_mask,
            }
            return True, sample
