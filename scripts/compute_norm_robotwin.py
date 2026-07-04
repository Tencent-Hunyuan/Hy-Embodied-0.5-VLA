#!/usr/bin/env python
# Copyright (C) 2026 Tencent.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Compute the unified Hy-VLA action normalization pickle.

The rel (RT) and abs (PosRotMat) Welford streams are accumulated in a
SINGLE pass over the dataset and dumped into ONE pkl with the layout
consumed by ``hy_vla.data.robotwin_dataset`` and
``robotwin_eval.policy_wrapper``::

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
    --csv         dataset_index.csv (bundled in this repo)
    --hdf5-dir    root that prefixes every ``episode_dir`` in the CSV
                  (= ``dataset.hdf5_dir`` in training config)
    --downsample-rate     temporal downsample (action timeline)
    --chunk-size          sliding-window length (action chunk)
    --output      destination pkl path
    --skip-dirty  drop episodes flagged ``is_dirty=1`` (default: keep them,
                  matches the released checkpoint training behaviour)
    --umi-coord-frame   apply UMI coordinate transform to state before computing
                  stats (see ``transform_utils.convert_frame_robo_to_umi``)
    --umi-gripper-space  also convert gripper values from RoboTwin convention
                  (0-1 norm) to UMI convention (0-90 mm). Requires --umi-coord-frame.

This script intentionally drops every knob the open-source pipeline
does not use:
* only RT-relative is supported;
* no ``use_kept_frames`` (the open-source dataset code does not consume
  ``kept_indices``; we always walk the raw timeline);
* no per-task tag string in the output path -- ``--output`` is the
  exact destination.

Usage
-----
python scripts/compute_norm_robotwin.py \
        --csv         dataset_index.csv \
        --hdf5-dir    /path/to/robotwin \
        --downsample-rate 3 \
        --chunk-size  20 \
        --output      /path/to/Hy-VLA-RoboTwin/norm_stats.pkl

python scripts/compute_norm_robotwin.py \
        --csv         dataset_index.csv \
        --hdf5-dir    /path/to/robotwin \
        --umi-coord-frame \
        --downsample-rate 3 \
        --chunk-size  20 \
        --output      /path/to/Hy-VLA-RoboTwin/norm_stats_umi.pkl

python scripts/compute_norm_robotwin.py \
        --csv         dataset_index.csv \
        --hdf5-dir    /path/to/robotwin \
        --umi-coord-frame \
        --umi-gripper-space \
        --downsample-rate 3 \
        --chunk-size  20 \
        --output      /path/to/Hy-VLA-RoboTwin/norm_stats_umi_with_gripper.pkl
"""

import argparse
import csv
import os
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

# Make the in-repo package importable when running from a fresh clone.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hy_vla.utils.transform_utils import (  # noqa: E402
    convert_PosQuat2PosRotationMatrix_batch,
    dual_arm_poses_to_relative,
    convert_frame_robo_to_umi,
)


# ---------------------------------------------------------------------------
# online (Welford) stats
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
# CSV -> episode list (mirror of hy_vla.data.robotwin_dataset._load_dataset_csv)
# ---------------------------------------------------------------------------
def _load_episodes(csv_path: str, hdf5_dir: str, skip_dirty: bool) -> list[dict]:
    eps: list[dict] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "episode_dir", "hdf5_name", "instruction_name",
            "num_frames", "is_dirty",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"dataset CSV {csv_path} is missing required columns: "
                f"{sorted(missing)}"
            )
        for row in reader:
            is_dirty = bool(int(row["is_dirty"]))
            if skip_dirty and is_dirty:
                continue
            ep_dir_abs = os.path.join(hdf5_dir, row["episode_dir"])
            eps.append({
                "episode_dir": row["episode_dir"],
                "hdf5_path": os.path.join(ep_dir_abs, row["hdf5_name"]),
                "num_frames": int(row["num_frames"]),
                "is_dirty": is_dirty,
            })
    return eps


# ---------------------------------------------------------------------------
# main accumulation
# ---------------------------------------------------------------------------
def compute(
    csv_path: str,
    hdf5_dir: str,
    output_path: str,
    downsample_rate: int,
    chunk_size: int,
    skip_dirty: bool,
    umi_coord_frame: bool,
    umi_gripper_space: bool = False,
) -> None:
    print(f"[config] csv               = {csv_path}")
    print(f"[config] hdf5_dir          = {hdf5_dir}")
    print(f"[config] downsample_rate   = {downsample_rate}")
    print(f"[config] chunk_size        = {chunk_size}")
    print(f"[config] skip_dirty        = {skip_dirty}")
    print(f"[config] umi_coord_frame   = {umi_coord_frame}")
    print(f"[config] umi_gripper_space = {umi_gripper_space}")
    print(f"[config] output            = {output_path}")

    eps = _load_episodes(csv_path, hdf5_dir, skip_dirty=skip_dirty)
    print(f"[load] {len(eps)} episodes")

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

    for ep in tqdm(eps, desc="episodes"):
        source_file = ep["hdf5_path"]
        if not os.path.isfile(source_file):
            print(f"[warn] missing hdf5: {source_file}")
            continue

        with h5py.File(source_file, "r") as f:
            qpos = f["observations/eepos"][:].astype(np.float32)

        # robotwin wxyz -> xyzw
        qpos_converted = qpos.copy()
        qpos_converted[:, 3:7] = qpos[:, [4, 5, 6, 3]]
        qpos_converted[:, 11:15] = qpos[:, [12, 13, 14, 11]]
        qpos = qpos_converted

        # Optional: UMI coordinate transform.
        if umi_coord_frame:
            qpos = convert_frame_robo_to_umi(qpos, convert_gripper=umi_gripper_space)

        # temporal downsample (e.g. 50Hz -> ~16.6Hz at ds=3)
        qpos = qpos[::downsample_rate]
        if qpos.shape[0] < 1:
            skipped_too_short += 1
            continue

        # ``actions`` keeps the raw PosQuat+gripper layout (16-d) -- both
        # branches consume the same expansion below.
        actions = qpos.copy()

        # qpos for the proprio accumulator is always 20-d PosRotMat.
        qpos = convert_PosQuat2PosRotationMatrix_batch(qpos)

        # sliding-window padding: replicate the last frame so that every
        # step has a full ``chunk_size`` lookahead.
        repeated_rows = np.tile(actions[-1:, :], (chunk_size, 1))
        actions = np.concatenate([actions, repeated_rows], axis=0)

        # (T, 16) -> (M, chunk, 16) with M = T (after pad).
        action_expanded = np.lib.stride_tricks.sliding_window_view(
            actions, window_shape=(chunk_size,), axis=0
        ).copy()
        assert action_expanded.shape[-1] == chunk_size
        action_expanded = np.transpose(action_expanded, (0, 2, 1))

        # ---- relative branch: dual_arm_poses_to_relative ----
        rel = np.zeros(
            (action_expanded.shape[0], action_expanded.shape[1], 20),
            dtype=np.float32,
        )
        for n in range(rel.shape[0]):
            rel[n] = dual_arm_poses_to_relative(action_expanded[n])
        rel_2d = rel.reshape(rel.shape[0], -1)
        count_rel, mean_rel, M2_rel = _update_welford(
            rel_2d, count_rel, mean_rel, M2_rel
        )

        # ---- absolute branch: PosQuat -> PosRotMat ----
        abs_ = np.zeros_like(rel)
        for n in range(abs_.shape[0]):
            abs_[n] = convert_PosQuat2PosRotationMatrix_batch(action_expanded[n])
        abs_2d = abs_.reshape(abs_.shape[0], -1)
        count_abs, mean_abs, M2_abs = _update_welford(
            abs_2d, count_abs, mean_abs, M2_abs
        )

        # ---- proprio (single-frame 20-d PosRotMat) ----
        count_qpos, mean_qpos, M2_qpos = _update_welford(
            qpos, count_qpos, mean_qpos, M2_qpos
        )

    if skipped_too_short:
        print(f"[warn] skipped {skipped_too_short} too-short episodes.")

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
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv", required=True,
                        help="dataset_index.csv (bundled in this repo).")
    parser.add_argument("--hdf5-dir", required=True,
                        help="Root that prefixes every episode_dir in the CSV.")
    parser.add_argument("--output", required=True,
                        help="Destination pkl path (e.g. <ckpt_dir>/norm_stats.pkl).")
    parser.add_argument("--downsample-rate", type=int, default=3,
                        help="Temporal downsample rate (default: 3, matches the released ckpt).")
    parser.add_argument("--chunk-size", type=int, default=20,
                        help="Sliding-window length / action chunk (default: 20, matches the released ckpt).")
    parser.add_argument("--skip-dirty", action="store_true",
                        help="Drop episodes with is_dirty=1. Default: keep them.")
    parser.add_argument("--umi-coord-frame", action="store_true",
                        help="Apply convert_frame_robo_to_umi to state "
                             "before computing stats (UMI coordinate frame).")
    parser.add_argument("--umi-gripper-space", action="store_true",
                        help="Also convert gripper values from RoboTwin convention "
                             "(0-1 norm) to UMI convention (0-90 mm). "
                             "Requires --umi-coord-frame to be set.")
    args = parser.parse_args()

    if args.umi_gripper_space and not args.umi_coord_frame:
        sys.exit("--umi-gripper-space requires --umi-coord-frame to be set")

    if not os.path.isfile(args.csv):
        sys.exit(f"--csv not found: {args.csv}")
    if not os.path.isdir(args.hdf5_dir):
        sys.exit(f"--hdf5-dir not found: {args.hdf5_dir}")
    if args.downsample_rate < 1:
        sys.exit("--downsample-rate must be >= 1")
    if args.chunk_size < 1:
        sys.exit("--chunk-size must be >= 1")

    compute(
        csv_path=args.csv,
        hdf5_dir=args.hdf5_dir,
        output_path=args.output,
        downsample_rate=args.downsample_rate,
        chunk_size=args.chunk_size,
        skip_dirty=args.skip_dirty,
        umi_coord_frame=args.umi_coord_frame,
        umi_gripper_space=args.umi_gripper_space,
    )


if __name__ == "__main__":
    main()
