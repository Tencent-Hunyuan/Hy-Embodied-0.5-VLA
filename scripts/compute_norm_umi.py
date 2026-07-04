#!/usr/bin/env python
# Copyright (C) 2026 Tencent.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Compute Hy-VLA action normalization pickle from a Lance (LeRobot-format) dataset.

Single-pass Welford accumulation over all episodes in all tables.
Output layout (consumed by ``hy_vla.data.umi_dataset`` and ``robotwin_eval.policy_wrapper``)::

    {
        "qpos_mean":         (20,)            # per-frame proprio (PosRotMat)
        "qpos_std":          (20,)
        "action_mean":       (chunk, 20)      # rel half (RT-relative)
        "action_std":        (chunk, 20)
        "action_mean_abs":   (chunk, 20)      # abs half (PosRotMat)
        "action_std_abs":    (chunk, 20)
        "first_frame":       None
    }

Inputs:
    --lance-source   HF Hub repo id or local directory (default:
                     tencent/Hy-Embodied-0.5-VLA-Data)
    --tables         "all" or comma-separated table names (default: all)
    --downsample-rate  temporal downsample for action timeline (default: 3)
    --chunk-size      sliding-window length / action chunk (default: 50)
    --output          destination pkl path
    --max-episodes    cap on total episodes (for quick testing, default: unlimited)
    --seed            random seed for episode shuffling (default: 42)

Usage
-----
python scripts/compute_norm_umi.py \\
        --lance-source  /path/to/lance_dataset \\
        --downsample-rate 3 \\
        --chunk-size  50 \\
        --output  /path/to/norm_stats_lance.pkl

    # HF Hub, single table, quick test
python scripts/compute_norm_umi.py \\
        --tables table_001 \\
        --max-episodes 100 \\
        --output /path/to/norm_stats.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hy_vla.data.umi_dataset import LanceTableReader  # noqa: E402
from hy_vla.utils.transform_utils import (  # noqa: E402
    convert_PosQuat2PosRotationMatrix_batch,
    dual_arm_poses_to_relative,
)


# ---------------------------------------------------------------------------
# online (Welford) stats – identical to the HDF5 version
# ---------------------------------------------------------------------------
def _update_welford(x: np.ndarray, count: int, mean, M2):
    for row in x:
        if count == 0:
            mean = row.copy()
            M2 = np.zeros_like(row)
        count += 1
        delta = row - mean
        mean += delta / count
        delta2 = row - mean
        M2 += delta * delta2
    return count, mean, M2


def _finalize(count, mean, M2, kind: str, std_eps: float):
    if mean is None or count <= 1:
        raise RuntimeError(f"{kind} accumulator is empty (count={count}).")
    std = np.sqrt(M2 / (count - 1))
    zero_idx = np.where(std < std_eps)
    if len(zero_idx[0]) > 0:
        print(
            f"[warn] {len(zero_idx[0])} {kind} dimensions have zero std, "
            f"set to 1."
        )
        std[zero_idx] = 1
    return mean, std


# ---------------------------------------------------------------------------
# per-episode action chunk builder (mirrors LanceVLADataset._build_umi_action_chunk)
# ---------------------------------------------------------------------------
def _sanitize_quat(arr_16d: np.ndarray):
    """Replace zero-norm quaternions with identity [0, 0, 0, 1] in-place.

    ``arr_16d``: (N, 16) with quaternions at columns 3:7 (left) and 11:15 (right).
    """
    for cols in (slice(3, 7), slice(11, 15)):
        norms = np.linalg.norm(arr_16d[:, cols], axis=1)
        zero_mask = norms < 1e-8
        if zero_mask.any():
            arr_16d[zero_mask, cols] = [0, 0, 0, 1]


def _build_action_chunk(ep_frames, c_id: int, sample_ds: int, num_steps: int,
                        chunk_size: int):
    """Build a (chunk_size, 16) PosQuat action chunk starting at c_id."""
    future_states, future_grippers = [], []
    for k in range(chunk_size):
        idx = min(c_id + k * sample_ds, num_steps - 1)
        state = ep_frames[idx].get("observation.state")
        if state is None:
            state = np.zeros(16, dtype=np.float32)
        future_states.append(np.asarray(state, dtype=np.float32))
        action = ep_frames[idx].get("action")
        if action is None:
            action = np.zeros(2, dtype=np.float32)
        future_grippers.append(np.asarray(action, dtype=np.float32))

    actions_16d = np.stack(future_states, axis=0).copy()
    grippers = np.stack(future_grippers, axis=0)
    actions_16d[:, 7] = grippers[:, 0]   # left gripper
    actions_16d[:, 15] = grippers[:, 1]  # right gripper
    return actions_16d


# ---------------------------------------------------------------------------
# main accumulation
# ---------------------------------------------------------------------------
def compute(
    lance_source: str,
    tables: list[str] | None,
    output_path: str,
    downsample_rate: int,
    chunk_size: int,
    max_episodes: int | None,
    seed: int,
) -> None:
    print(f"[config] lance_source      = {lance_source}")
    print(f"[config] tables            = {tables or 'all'}")
    print(f"[config] downsample_rate   = {downsample_rate}")
    print(f"[config] chunk_size        = {chunk_size}")
    print(f"[config] max_episodes      = {max_episodes or 'unlimited'}")
    print(f"[config] seed              = {seed}")
    print(f"[config] output            = {output_path}")

    # --- open readers ---
    if os.path.isdir(lance_source):
        if tables is None:
            tables = []
            for p in sorted(Path(lance_source).iterdir()):
                if p.is_dir() and (p / f"{p.name}.lance").is_dir():
                    tables.append(p.name)
            if not tables:
                tables = sorted(
                    p.stem for p in Path(lance_source).glob("*.lance") if p.is_dir()
                )
        readers = {
            tn: LanceTableReader(root=lance_source, table_name=tn)
            for tn in tables
        }
    else:
        if tables is None:
            from hy_vla.data.umi_dataset import _list_lance_tables
            tables = _list_lance_tables(repo_id=lance_source)
        readers = {
            tn: LanceTableReader(repo_id=lance_source, table_name=tn)
            for tn in tables
        }

    total_frames = sum(r.num_frames for r in readers.values())
    total_episodes = sum(r.num_episodes for r in readers.values())
    print(f"[load] {len(readers)} table(s), {total_episodes} episodes, "
          f"{total_frames:,} frames")

    # --- enumerate all (table, ep_idx) pairs ---
    all_items: list[tuple[str, int]] = []
    for tn, r in readers.items():
        eps = r.meta.get("episodes", [])
        if eps:
            all_items.extend((tn, int(e["episode_index"])) for e in eps)
        else:
            all_items.extend((tn, i) for i in range(r.num_episodes))

    # shuffle for unbiased estimate when using --max-episodes
    rng = np.random.RandomState(seed)
    rng.shuffle(all_items)
    if max_episodes is not None:
        all_items = all_items[:max_episodes]
    print(f"[load] accumulating {len(all_items):,} episodes")

    # --- Welford accumulators ---
    count_qpos = 0
    mean_qpos = None
    M2_qpos = None

    count_rel = 0
    mean_rel = None
    M2_rel = None

    count_abs = 0
    mean_abs = None
    M2_abs = None

    skipped_too_short = 0

    for tn, ep_idx in tqdm(all_items, desc="episodes"):
        r = readers[tn]
        ep_frames = r.get_episode(ep_idx)
        num_steps = len(ep_frames)
        if num_steps < 1:
            skipped_too_short += 1
            continue

        # --- collect full-episode observation.state (16-d) as float32 ---
        qpos_raw = np.array(
            [np.asarray(ep_frames[i].get("observation.state",
                        np.zeros(16, dtype=np.float32)), dtype=np.float32)
             for i in range(num_steps)],
            dtype=np.float32,
        )

        # temporal downsample
        qpos_raw = qpos_raw[::downsample_rate]
        if qpos_raw.shape[0] < 1:
            skipped_too_short += 1
            continue

        # qpos for proprio accumulator: 16-d PosQuat → 20-d PosRotMat
        _sanitize_quat(qpos_raw)
        qpos_20d = convert_PosQuat2PosRotationMatrix_batch(qpos_raw, quat_order="xyzw")

        # action chunk sliding window
        actions_16d = qpos_raw.copy()
        repeated = np.tile(actions_16d[-1:, :], (chunk_size, 1))
        actions_16d_padded = np.concatenate([actions_16d, repeated], axis=0)
        action_chunks = np.lib.stride_tricks.sliding_window_view(
            actions_16d_padded, window_shape=(chunk_size,), axis=0
        ).copy()
        action_chunks = np.transpose(action_chunks, (0, 2, 1))  # (M, chunk, 16)

        # --- relative branch (RT-relative) ---
        rel = np.zeros((action_chunks.shape[0], chunk_size, 20), dtype=np.float32)
        for n in range(rel.shape[0]):
            chunk = action_chunks[n].copy()
            _sanitize_quat(chunk)
            rel[n] = dual_arm_poses_to_relative(chunk)
        rel_2d = rel.reshape(rel.shape[0], -1)
        count_rel, mean_rel, M2_rel = _update_welford(
            rel_2d, count_rel, mean_rel, M2_rel
        )

        # --- absolute branch (PosRotMat) ---
        abs_ = np.zeros_like(rel)
        for n in range(abs_.shape[0]):
            chunk = action_chunks[n].copy()
            _sanitize_quat(chunk)
            abs_[n] = convert_PosQuat2PosRotationMatrix_batch(
                chunk, quat_order="xyzw"
            )
        abs_2d = abs_.reshape(abs_.shape[0], -1)
        count_abs, mean_abs, M2_abs = _update_welford(
            abs_2d, count_abs, mean_abs, M2_abs
        )

        # --- proprio (single-frame 20-d PosRotMat) ---
        count_qpos, mean_qpos, M2_qpos = _update_welford(
            qpos_20d, count_qpos, mean_qpos, M2_qpos
        )

    if skipped_too_short:
        print(f"[warn] skipped {skipped_too_short} too-short episodes.")

    # --- finalize ---
    mean_qpos, std_qpos = _finalize(count_qpos, mean_qpos, M2_qpos, "qpos", 1e-4)
    mean_rel_f, std_rel_f = _finalize(count_rel, mean_rel, M2_rel, "rel-action", 1e-5)
    mean_abs_f, std_abs_f = _finalize(count_abs, mean_abs, M2_abs, "abs-action", 1e-5)

    mean_rel_f = mean_rel_f.reshape(chunk_size, -1)
    std_rel_f = std_rel_f.reshape(chunk_size, -1)
    mean_abs_f = mean_abs_f.reshape(chunk_size, -1)
    std_abs_f = std_abs_f.reshape(chunk_size, -1)

    print(f"[stat] qpos:       {mean_qpos.shape}")
    print(f"[stat] rel action: {mean_rel_f.shape}")
    print(f"[stat] abs action: {mean_abs_f.shape}")

    payload = {
        "qpos_mean": mean_qpos,
        "qpos_std": std_qpos,
        "action_mean": mean_rel_f,
        "action_std": std_rel_f,
        "action_mean_abs": mean_abs_f,
        "action_std_abs": std_abs_f,
        "first_frame": None,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(payload, f)
    print(f"[save] wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--lance-source",
        default="tencent/Hy-Embodied-0.5-VLA-Data",
        help="HF Hub repo id or local directory path "
             "(default: tencent/Hy-Embodied-0.5-VLA-Data).",
    )
    parser.add_argument(
        "--tables", default=None,
        help="Comma-separated table names, or 'all' (default: all).",
    )
    parser.add_argument("--output", required=True,
                        help="Destination pkl path (e.g. <ckpt_dir>/norm_stats.pkl).")
    parser.add_argument("--downsample-rate", type=int, default=3,
                        help="Temporal downsample rate (default: 3).")
    parser.add_argument("--chunk-size", type=int, default=50,
                        help="Sliding-window length / action chunk (default: 50).")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Cap on total episodes (for quick testing).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for episode shuffling (default: 42).")
    args = parser.parse_args()

    tables = None
    if args.tables and args.tables.lower() != "all":
        tables = [t.strip() for t in args.tables.split(",") if t.strip()]

    if args.downsample_rate < 1:
        sys.exit("--downsample-rate must be >= 1")
    if args.chunk_size < 1:
        sys.exit("--chunk-size must be >= 1")

    compute(
        lance_source=args.lance_source,
        tables=tables,
        output_path=args.output,
        downsample_rate=args.downsample_rate,
        chunk_size=args.chunk_size,
        max_episodes=args.max_episodes,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
