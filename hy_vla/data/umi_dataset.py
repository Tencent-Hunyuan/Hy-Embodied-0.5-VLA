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

"""Lance-backed dataset for Hy-VLA pretraining.

``LanceTableReader`` -- low-level table reader for visualisation scripts.
``LanceVLADataset`` -- training wrapper compatible with ``VLADataset``.

Supports local directories and HuggingFace Hub repos (multi-table)."""

from __future__ import annotations

import json
import os
import pickle
import random
import traceback
from bisect import bisect_right
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import lance

try:
    from huggingface_hub import hf_hub_download, HfApi, get_token as _get_token, snapshot_download
    import lancedb as ldb
    _HAS_HF = True
except ImportError:
    _HAS_HF = False

from PIL import Image

from hy_vla.utils.transform_utils import (
    dual_arm_poses_to_relative,
    convert_PosQuat2PosRotationMatrix_batch,
)


def _lc(key: str) -> str:
    """Lance-safe column name: ``.``  -> ``_``."""
    return key.replace(".", "_")


def pad_vector(vector, new_dim):
    """Pad last dim to ``new_dim`` with zeros."""
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    shape[-1] = new_dim
    new_vector = np.zeros(shape, dtype=vector.dtype)
    new_vector[..., : vector.shape[-1]] = vector
    return new_vector


def _get_arr(frame, key, default=None):
    """Extract float32 array from a frame dict."""
    val = frame.get(key)
    if val is None:
        return default if default is not None else np.zeros(0, dtype=np.float32)
    if isinstance(val, np.ndarray):
        return val.astype(np.float32)
    if isinstance(val, (list, tuple)):
        return np.array(val, dtype=np.float32)
    return np.array([val], dtype=np.float32)


class LanceTableReader:
    """Read a single Lance table (local or HF Hub)."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        repo_id: str | None = None,
        revision: str | None = None,
        table_name: Optional[str] = None,
        cache_dir: str | Path | None = None,
    ):
        if root is None and repo_id is None:
            raise TypeError("LanceTableReader requires `root` or `repo_id`.")

        if repo_id is not None:
            if not _HAS_HF:
                raise ImportError("huggingface_hub + lancedb required for HF Hub mode.")
            self.table_name = table_name
            if self.table_name is None:
                self.table_name = repo_id.split("/")[-1]
            patterns = [f"{self.table_name}/meta/*"]
            self.root = Path(snapshot_download(
                repo_id, repo_type="dataset", revision=revision,
                allow_patterns=patterns, cache_dir=cache_dir,
            )) / self.table_name
            suffix = f"@{revision}" if revision else ""
            uri = f"hf://datasets/{repo_id}/{self.table_name}{suffix}"
            opts: Dict[str, str] = {}
            if _HAS_HF:
                try:
                    t = _get_token()
                    if t:
                        opts.update(token=t)
                except Exception:
                    pass
            self._uri = uri
            self._storage_opts = opts
            self._ds = None  # lazy init: created in each worker after fork
        else:
            assert root is not None
            if table_name:
                self.table_name = table_name
            else:
                candidates = []
                for p in sorted(Path(root).iterdir()):
                    if p.is_dir() and (p / f"{p.name}.lance").is_dir():
                        candidates.append(p)
                if not candidates:
                    candidates = sorted(Path(root).glob("*.lance"))
                if not candidates:
                    raise FileNotFoundError(f"No lance table found under {root}")
                if len(candidates) > 1:
                    raise ValueError(
                        f"Multiple lance tables under {root}: "
                        f"{[c.name for c in candidates]}. Pass `table_name=` explicitly."
                    )
                self.table_name = candidates[0].stem if candidates[0].suffix == ".lance" else candidates[0].name
            self.root = Path(root) / self.table_name
            self._lance_path = str(self.root / f"{self.table_name}.lance")
            self._ds = None  # lazy init: created in each worker after fork

        self.meta = self._load_meta()
        self.num_frames = self.meta.get("total_frames", 0)
        self.num_episodes = self.meta.get("total_episodes", 0)
        self.fps = self.meta.get("fps", 30)
        self.features = self.meta.get("features", {})
        self.image_keys = self.meta.get("image_keys", [])

        self._col_map = {_lc(k): k for k in self.features}
        for k in ("episode_index", "frame_index", "index", "timestamp", "task", "task_index"):
            self._col_map[k] = k

        eps = self.meta.get("episodes", [])
        if eps:
            self._ep_starts = np.array([e["dataset_from_index"] for e in eps], dtype=np.int64)
            self._ep_ends = np.array([e["dataset_to_index"] for e in eps], dtype=np.int64)
            self._ep_ids = np.array([e["episode_index"] for e in eps], dtype=np.int64)
        else:
            self._ep_starts = np.array([])
            self._ep_ends = np.array([])
            self._ep_ids = np.array([])

        # Count rows safely: open, count, del — avoid live handle across fork.
        if self.num_frames == 0:
            self.num_frames = self._safe_count_rows()

    def _load_meta(self) -> dict:
        meta_root = self.root / f"{self.table_name}_meta"
        if not (meta_root / "info.json").exists():
            meta_root = self.root / "meta"
        info = {}
        if (meta_root / "info.json").exists():
            info = json.loads((meta_root / "info.json").read_text())
        stats = {}
        if (meta_root / "stats.json").exists():
            stats = json.loads((meta_root / "stats.json").read_text())
        episodes: list = []
        ed = meta_root / "episodes"
        if ed.exists():
            for p in sorted(ed.glob("**/*.parquet")):
                episodes.extend(pq.read_table(p).to_pylist())
        image_keys = [k for k, v in info.get("features", {}).items()
                      if v.get("dtype") in ("video", "image")]
        tasks = []
        if (meta_root / "tasks.parquet").exists():
            tasks = pq.read_table(meta_root / "tasks.parquet").to_pylist()
        return {"info": info, "stats": stats, "episodes": episodes,
                "image_keys": image_keys, "tasks": tasks,
                "total_frames": info.get("total_frames", 0),
                "total_episodes": info.get("total_episodes", len(episodes)),
                "fps": info.get("fps", 30), "features": info.get("features", {})}

    def _open_dataset(self):
        """Create a fresh lance handle. Never cache from main process."""
        if hasattr(self, '_uri'):
            # HF Hub mode
            if not _HAS_HF:
                raise ImportError("huggingface_hub + lancedb required for HF Hub mode.")
            return ldb.connect(
                self._uri,
                storage_options=self._storage_opts,
            ).open_table(self.table_name).to_lance()
        # Local directory mode
        return lance.LanceDataset(self._lance_path)

    def _safe_count_rows(self) -> int:
        """Count rows then close handle to avoid fork-safety warnings."""
        ds = self._open_dataset()
        n = ds.count_rows()
        del ds
        return n

    def _ensure_connection(self):
        """Lazily create a process-local lance connection (fork-safe)."""
        if self._ds is None:
            self._ds = self._open_dataset()

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> dict:
        return self.get_frame(idx)

    def _take_and_build_rows(self, indices, columns=None):
        """Take rows from lance and parse into dicts."""
        self._ensure_connection()
        if not indices:
            return []
        col_map = self._col_map
        ik = set(self.image_keys)
        indices_arr = pa.array(indices, type=pa.int64())
        if columns is not None:
            result = self._ds.take(indices_arr, columns=[_lc(c) for c in columns])
        else:
            result = self._ds.take(indices_arr)

        cols = {}
        for f in result.schema:
            v = result.column(f.name)
            if pa.types.is_binary(v.type) or pa.types.is_large_binary(v.type):
                cols[f.name] = v.to_pylist()
            elif pa.types.is_fixed_size_list(v.type):
                cols[f.name] = v.combine_chunks().flatten().to_numpy().reshape(
                    len(v), v.type.list_size)
            else:
                cols[f.name] = v.to_numpy(zero_copy_only=False)

        rows = []
        for i in range(len(indices)):
            row = {}
            for ln, vals in cols.items():
                key = col_map.get(ln, ln)
                val = vals[i]
                if isinstance(val, bytes) and key in ik:
                    try:
                        row[key] = np.array(Image.open(BytesIO(val)))
                    except Exception:
                        shape = self.meta.get("features", {}).get(key, {}).get("shape", [240, 424, 3])
                        row[key] = np.zeros(tuple(shape), dtype=np.uint8)
                else:
                    row[key] = val
            rows.append(row)
        return rows

    def get_frame(self, idx, columns=None):
        rows = self._take_and_build_rows([idx], columns=columns)
        return rows[0] if rows else {}

    def get_frames(self, indices, columns=None):
        return self._take_and_build_rows(indices, columns=columns)

    def episode_for_index(self, idx: int) -> int:
        if len(self._ep_ids) == 0:
            return 0
        pos = np.searchsorted(self._ep_ends, idx, "right")
        return int(self._ep_ids[min(pos, len(self._ep_ids) - 1)])

    def get_episode(self, ep_idx: int,
                    columns: Optional[List[str]] = None) -> List[dict]:
        if len(self._ep_ids) == 0:
            return [self.get_frame(ep_idx, columns=columns)]
        mask = self._ep_ids == ep_idx
        if not mask.any():
            raise IndexError(f"Episode {ep_idx} not found")
        return self.get_frames(
            list(range(int(self._ep_starts[mask][0]),
                        int(self._ep_ends[mask][0]))),
            columns=columns)

    def __repr__(self) -> str:
        return (f"LanceTableReader({self.table_name!r}, "
                f"episodes={self.num_episodes}, frames={self.num_frames})")


def _list_lance_tables(repo_id=None, local_root=None, revision=None):
    if repo_id is not None:
        if not _HAS_HF:
            raise ImportError("huggingface_hub required to list tables on HF Hub")
        try:
            manifest_path = hf_hub_download(
                repo_id, "tables.json", repo_type="dataset", revision=revision)
            with open(manifest_path) as f:
                manifest = json.load(f)
            tables = [t["table_name"] for t in manifest.get("tables", [])]
            if tables:
                return tables
        except Exception:
            pass
        tables = []
        for s in HfApi().dataset_info(repo_id, revision=revision).siblings:
            parts = s.rfilename.split("/")
            if len(parts) >= 2 and parts[1].endswith(".lance"):
                tn = parts[0]
                if tn not in tables:
                    tables.append(tn)
            elif len(parts) >= 1 and parts[0].endswith(".lance"):
                tn = parts[0].replace(".lance", "")
                if tn not in tables:
                    tables.append(tn)
        return sorted(tables)
    else:
        tables = []
        for p in sorted(Path(local_root).iterdir()):
            if p.is_dir() and (p / f"{p.name}.lance").is_dir():
                tables.append(p.name)
        if tables:
            return tables
        return sorted(p.stem for p in Path(local_root).glob("*.lance") if p.is_dir())


def _load_mean_std(path, chunk_slice=None, with_abs=False):
    if not os.path.exists(path):
        raise ValueError(f"File does not exist: {path}")
    with open(path, "rb") as fp:
        info = pickle.load(fp)
    qm = np.array(info["qpos_mean"], dtype=np.float32)
    qs = np.array(info["qpos_std"], dtype=np.float32)
    am = np.array(info["action_mean"], dtype=np.float32)
    as_ = np.array(info["action_std"], dtype=np.float32)
    am_absolute, as_absolute = None, None
    if with_abs:
        am_absolute = np.array(info["action_mean_abs"], dtype=np.float32)
        as_absolute = np.array(info["action_std_abs"], dtype=np.float32)
    if chunk_slice is not None:
        n = int(chunk_slice)
        am = am[:n]
        as_ = as_[:n]
        if am_absolute is not None:
            am_absolute = am_absolute[:n]
            as_absolute = as_absolute[:n]
    return qm, qs, am, as_, am_absolute, as_absolute


class LanceVLADataset:
    """Lance-backed dataset consumed transparently by ``VLADataset``."""

    def __init__(self, config) -> None:
        lance_source = getattr(config.dataset, "lance_source", None)
        if lance_source is None:
            raise ValueError(
                "dataset.lance_source must be set for Lance backend.")

        lance_revision = getattr(config.dataset, "lance_revision", None)
        lance_tables_str = getattr(config.dataset, "lance_tables", "all")
        lance_cache = getattr(config.dataset, "lance_cache_dir", None)

        # auto-detect: local dir or HF Hub repo id
        if os.path.isdir(lance_source):
            self._repo_id = None
            self._local_root = Path(lance_source)
        else:
            self._repo_id = lance_source
            self._local_root = None

        self._revision = lance_revision
        self._cache_dir = lance_cache

        all_tables = _list_lance_tables(
            self._repo_id,
            str(self._local_root) if self._local_root else None,
            self._revision)
        if not all_tables:
            raise FileNotFoundError(
                f"No *.lance tables found in {lance_source}")

        if lance_tables_str == "all":
            self._table_names = all_tables
        else:
            requested = [t.strip() for t in lance_tables_str.split(",") if t.strip()]
            self._table_names = [t for t in requested if t in all_tables]
            missing = set(requested) - set(self._table_names)
            if missing:
                raise ValueError(
                    f"Requested tables not found: {sorted(missing)}. "
                    f"Available: {all_tables}")

        print(f"[umi_dataset] tables: {self._table_names}")

        self._readers: Dict[str, LanceTableReader] = {}
        self._offsets: List[int] = [0]
        for tn in self._table_names:
            if self._repo_id:
                r = LanceTableReader(
                    repo_id=self._repo_id, revision=self._revision,
                    table_name=tn, cache_dir=self._cache_dir)
            else:
                r = LanceTableReader(root=self._local_root, table_name=tn)
            self._readers[tn] = r
            self._offsets.append(self._offsets[-1] + r.num_frames)

        self.total_frames = self._offsets[-1]
        self.total_episodes = sum(r.num_episodes for r in self._readers.values())
        print(f"[umi_dataset] total frames: {self.total_frames}, "
              f"total episodes: {self.total_episodes}")

        self.action_type = config.dataset.act_type
        self.downsample_rate = config.dataset.downsample_rate
        self.CHUNK_SIZE = config["dataset"]["action_chunk_size"]
        self.STATE_DIM = config["dataset"]["state_dim"]
        self.use_video_encoder = bool(getattr(config.dataset, "use_video_encoder", False))

        raw_img_history_size = int(config["dataset"]["img_history_size"])
        if self.use_video_encoder:
            self.IMG_HISTORY_SIZE = raw_img_history_size
        else:
            if raw_img_history_size != 1:
                print(f"[umi_dataset] WARN: use_video_encoder=False but "
                      f"img_history_size={raw_img_history_size}; forcing to 1")
            self.IMG_HISTORY_SIZE = 1

        self.IMG_HISTORY_INTERVAL = int(
            config["dataset"].get("img_history_interval", 1))
        self.IMG_HISTORY_RANDOM_SAMPLE = bool(
            config["dataset"].get("img_history_random_sample", False))

        if not hasattr(config.dataset, "mean_std_path"):
            raise ValueError("dataset.mean_std_path is required.")
        mean_std_path = config.dataset.mean_std_path
        print(f"[umi_dataset] mean_std_path: {mean_std_path}")
        with_abs = "with_absolute" in self.action_type
        (self.qpos_mean, self.qpos_std, self.act_mean, self.act_std,
         _act_mean_abs, _act_std_abs) = _load_mean_std(
            mean_std_path,
            chunk_slice=getattr(config.dataset, "mean_std_chunk_slice", None),
            with_abs=with_abs)
        if with_abs:
            self.act_mean = np.concatenate([self.act_mean, _act_mean_abs], axis=0)
            self.act_std = np.concatenate([self.act_std, _act_std_abs], axis=0)

        self.deterministic = bool(getattr(config.dataset, "deterministic", False))
        self.deterministic_index = None
        if self.deterministic:
            pairs: List[Tuple[int, int, int]] = []
            for reader_idx, tn in enumerate(self._table_names):
                r = self._readers[tn]
                for ep_idx in range(r.num_episodes):
                    start = int(r._ep_starts[ep_idx])
                    end = int(r._ep_ends[ep_idx])
                    num_frames = end - start
                    for t in range(num_frames):
                        pairs.append((reader_idx, ep_idx, t))
            self.deterministic_index = pairs
            print(f"[umi_dataset] deterministic: {len(pairs)} triples enumerated")

    def _locate_global(self, global_idx: int) -> Tuple[int, str, int]:
        t = bisect_right(self._offsets, global_idx) - 1
        tn = self._table_names[t]
        local = global_idx - self._offsets[t]
        return t, tn, local

    def __len__(self):
        if self.deterministic and self.deterministic_index is not None:
            return len(self.deterministic_index)
        return self.total_frames

    def get_item(self, index: int = None):
        if self.deterministic:
            assert index is not None
            N = len(self.deterministic_index)
            i = int(index) % N
            attempts = 0
            while True:
                table_idx, ep_idx, step = self.deterministic_index[i]
                valid, sample = self._get_item_from(table_idx, ep_idx, step)
                if valid:
                    return sample
                i = (i + 1) % N
                attempts += 1
                if attempts >= N:
                    raise RuntimeError("All samples rejected in deterministic mode")
        else:
            while True:
                if index is None:
                    gi = np.random.randint(0, self.total_frames)
                else:
                    gi = int(index) % self.total_frames
                table_idx, tn, local_idx = self._locate_global(gi)
                r = self._readers[tn]
                ep = r.episode_for_index(local_idx)
                ep_start = (int(r._ep_starts[r._ep_ids == ep][0])
                            if len(r._ep_ids) > 0 else 0)
                step = local_idx - ep_start
                valid, sample = self._get_item_from(table_idx, ep, step)
                if valid:
                    return sample

    def _get_item_from(self, table_idx: int, ep_idx: int, step: int):
        try:
            return self._get_item_from_impl(table_idx, ep_idx, step)
        except Exception as e:
            print(f"[umi_dataset] WARNING: _get_item_from failed "
                  f"(table={self._table_names[table_idx]}, ep={ep_idx}, step={step}): {e}")
            traceback.print_exc()
            return False, None

    def _build_umi_action_chunk(self, ep_frames, c_id, sample_ds, num_steps):
        """Build UMI action chunk: future state skeleton + gripper → RT-relative."""
        default_state = np.zeros(self.STATE_DIM, dtype=np.float32)
        default_gripper = np.zeros(2, dtype=np.float32)

        future_states, future_grippers = [], []
        for k in range(self.CHUNK_SIZE):
            idx = min(c_id + k * sample_ds, num_steps - 1)
            future_states.append(_get_arr(ep_frames[idx], "observation.state", default_state))
            future_grippers.append(_get_arr(ep_frames[idx], "action", default_gripper))

        actions_16d = np.stack(future_states, axis=0).copy()
        grippers = np.stack(future_grippers, axis=0)
        actions_16d[:, 7] = grippers[:, 0]   # left gripper
        actions_16d[:, 15] = grippers[:, 1]  # right gripper

        if "with_absolute" in self.action_type:
            actions_abs = convert_PosQuat2PosRotationMatrix_batch(actions_16d.copy(), quat_order="xyzw")
        else:
            actions_abs = None

        actions = dual_arm_poses_to_relative(actions_16d)

        if actions_abs is not None:
            actions = np.concatenate([actions, actions_abs], axis=0)
        return actions

    def _parse_camera_images(self, ep_frames, planned_indices, num_steps, K):
        """Parse camera images into (K, H, W, 3) arrays."""
        cam_map = {
            "cam_high": "observation.images.cam_high",
            "cam_left_wrist": "observation.images.cam_left_wrist",
            "cam_right_wrist": "observation.images.cam_right_wrist",
        }

        def _parse(key_lance):
            imgs = []
            for pi in planned_indices:
                pi_c = max(0, min(pi, num_steps - 1))
                img = ep_frames[pi_c].get(key_lance)
                if img is not None and isinstance(img, np.ndarray) and img.ndim == 3:
                    imgs.append(img)
                else:
                    imgs.append(np.zeros((0, 0, 0), dtype=np.uint8))
            if all(i.size == 0 for i in imgs):
                return np.zeros((K, 0, 0, 0), dtype=np.uint8)
            shapes = {i.shape for i in imgs if i.size > 0}
            if not shapes:
                return np.zeros((K, 0, 0, 0), dtype=np.uint8)
            target_shape = next(iter(shapes))
            result = []
            for i in imgs:
                result.append(i if i.shape == target_shape
                              else np.zeros(target_shape, dtype=np.uint8))
            return np.stack(result, axis=0)

        return (_parse(cam_map["cam_high"]),
                _parse(cam_map["cam_left_wrist"]),
                _parse(cam_map["cam_right_wrist"]))

    def _resolve_instruction(self, r, ep_frames, c_id):
        """Resolve task instruction string from frame."""
        task_map = {int(t["task_index"]): t["task"] for t in r.meta.get("tasks", [])}
        task_val = ep_frames[c_id].get("task_index", ep_frames[c_id].get("task", ""))
        if isinstance(task_val, (int, np.integer)):
            return task_map.get(int(task_val), "")
        if isinstance(task_val, str):
            return task_val
        return str(task_val) if task_val is not None else ""

    def _get_item_from_impl(self, table_idx: int, ep_idx: int, step: int):
        tn = self._table_names[table_idx]
        r = self._readers[tn]

        mask = r._ep_ids == ep_idx
        if not mask.any():
            return False, None
        ep_start = int(r._ep_starts[mask][0])
        ep_end = int(r._ep_ends[mask][0])
        num_steps = ep_end - ep_start
        if step < 0 or step >= num_steps:
            return False, None

        c_id = step
        sample_ds = self.downsample_rate

        # collect needed frame indices
        needed = {c_id}
        for k in range(self.CHUNK_SIZE):
            needed.add(min(c_id + k * sample_ds, num_steps - 1))

        raw_interval = self.IMG_HISTORY_INTERVAL * sample_ds
        K = self.IMG_HISTORY_SIZE
        planned_indices = []
        for k in range(K):
            end = c_id - (K - 1 - k) * raw_interval
            planned_indices.append(max(end, 0))
        planned_indices[-1] = c_id
        needed.update(planned_indices)

        sorted_relative = sorted(needed)
        global_indices = [ep_start + x for x in sorted_relative]
        frames = r.get_frames(global_indices)

        ep_frames = [{} for _ in range(num_steps)]
        for idx, frame in zip(sorted_relative, frames):
            ep_frames[idx] = frame

        # build action chunk
        try:
            actions = self._build_umi_action_chunk(ep_frames, c_id, sample_ds, num_steps)
        except Exception:
            return False, None

        state = _get_arr(ep_frames[c_id], "observation.state",
                         np.zeros(self.STATE_DIM, dtype=np.float32)).reshape(1, -1)

        # Reject frames with invalid (near-zero-norm) quaternions.
        left_quat = state[0, 3:7]
        right_quat = state[0, 11:15]
        if np.linalg.norm(left_quat) < 1e-8 or np.linalg.norm(right_quat) < 1e-8:
            return False, None

        state = convert_PosQuat2PosRotationMatrix_batch(state)
        state = (state - self.qpos_mean) / np.maximum(self.qpos_std, 1e-8)
        actions = (actions - self.act_mean) / np.maximum(self.act_std, 1e-8)

        state = pad_vector(state, self.STATE_DIM)
        state_indicator = pad_vector(
            np.ones(state.shape[-1], dtype=np.float32).reshape(1, -1), self.STATE_DIM)
        actions = pad_vector(actions, self.STATE_DIM)

        cam_high, cam_left, cam_right = self._parse_camera_images(
            ep_frames, planned_indices, num_steps, K)

        mask = np.array(
            [(c_id - (K - 1 - k) * raw_interval) >= 0 for k in range(K)],
            dtype=bool)

        instruction = self._resolve_instruction(r, ep_frames, c_id)

        meta = {"#steps": num_steps, "step_id": step, "instruction": instruction}

        return True, {
            "meta": meta,
            "state": state,
            "actions": actions,
            "state_indicator": state_indicator,
            "cam_high": cam_high,
            "cam_high_mask": mask,
            "cam_left_wrist": cam_left,
            "cam_left_wrist_mask": mask.copy(),
            "cam_right_wrist": cam_right,
            "cam_right_wrist_mask": mask.copy(),
        }
