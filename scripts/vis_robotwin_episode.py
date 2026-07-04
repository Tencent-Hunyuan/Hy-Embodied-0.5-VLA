#!/usr/bin/env python3
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
"""
Per-episode video visualisation for RoboTwin HDF5 datasets.

3-column layout: camera stack | 2D position plots | 3D trajectory.

Instruction banner: if multiple unique instructions exist across the task's
episodes, they are cycled frame-by-frame (carousel); otherwise the single
instruction is shown statically.

Usage:
  # Visualize a specific episode by its global index in the CSV:
  python scripts/vis_robotwin_episode.py \\
      /path/to/robotwin_hdf5 \\
      -e 0

  # Limit to a specific task and set output fps:
  python scripts/vis_robotwin_episode.py \\
      /path/to/robotwin_hdf5 \\
      -t adjust_bottle -e 5 --fps 15

  # Use a custom CSV index:
  python scripts/vis_robotwin_episode.py \\
      /path/to/robotwin_hdf5 \\
      --csv /path/to/dataset_index.csv -e 0

  # UMI coordinate frame:
  python scripts/vis_robotwin_episode.py \\
      /path/to/robotwin_hdf5 \\
      -t adjust_bottle -e 5 --umi-coord-frame
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import h5py
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# UMI coordinate-frame transforms
from hy_vla.utils.transform_utils import (
    dual_arm_poses_to_relative,
    convert_frame_robo_to_umi,
)

warnings.filterwarnings("ignore", message="not fork-safe")

_CJK_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

# Camera keys in the RoboTwin HDF5 and their display names.
# RoboTwin stores images under ``observations/images/``.
CAM_KEYS: list[tuple[str, str, str]] = [
    ("cam_left_wrist",  "observations/images/cam_left_wrist",  "cam_left_wrist"),
    ("cam_high",        "observations/images/cam_high",        "cam_high"),
    ("cam_right_wrist", "observations/images/cam_right_wrist", "cam_right_wrist"),
]

# State layout (16-dim): left [x,y,z,qx,qy,qz,qw,grip] | right [x,y,z,qx,qy,qz,qw,grip]
LEFT_X,  LEFT_Y,  LEFT_Z  = 0, 1, 2
LEFT_G  = 7
RIGHT_X, RIGHT_Y, RIGHT_Z = 8, 9, 10
RIGHT_G = 15

POS_INDEX = {
    "xl": LEFT_X,   "yl": LEFT_Y,   "zl": LEFT_Z,   "gl": LEFT_G,
    "xr": RIGHT_X,  "yr": RIGHT_Y,  "zr": RIGHT_Z,  "gr": RIGHT_G,
}

# RT-relative action layout (20-dim):
#   [left_dxyz(3), left_relRot6d(6), left_gripper(1),
#    right_dxyz(3), right_relRot6d(6), right_gripper(1)]
# We only plot position deltas + grippers (skip the 6D rotation).
RT_ACTION_INDEX = {
    "xl": 0,   "yl": 1,   "zl": 2,   "gl": 9,
    "xr": 10,  "yr": 11,  "zr": 12,  "gr": 19,
}

PLOT_ORDER = ["xl", "yl", "zl", "gl", "gr", "xr", "yr", "zr"]
COLORS = {
    "xl": "tab:red",   "yl": "tab:green",  "zl": "tab:blue",   "gl": "tab:orange",
    "gr": "tab:purple", "xr": "tab:red",   "yr": "tab:green",  "zr": "tab:blue",
}


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def _load_csv_episodes(csv_path: str, hdf5_dir: str) -> list[dict]:
    """Load episode list from the RoboTwin CSV dataset index.

    CSV schema (5 columns):
        episode_dir          relative path of the per-episode directory
        hdf5_name            file name of the hdf5 inside that directory
        instruction_name     file name of the instructions json
        num_frames           int, raw frame count
        is_dirty             0/1 filter flag

    File names in the CSV are relative to ``hdf5_dir``; we re-glue to
    absolute paths here.
    """
    episodes: list[dict] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "episode_dir", "hdf5_name", "instruction_name",
            "num_frames", "is_dirty",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"CSV {csv_path} is missing columns: {sorted(missing)}"
            )
        for i, row in enumerate(reader):
            ep_dir_abs = os.path.join(hdf5_dir, row["episode_dir"])
            episodes.append({
                "global_index": i,
                "episode_dir": row["episode_dir"],
                "task_name": row["episode_dir"].split("/")[0],
                "hdf5_path": os.path.join(ep_dir_abs, row["hdf5_name"]),
                "instruction_path": os.path.join(
                    ep_dir_abs, row["instruction_name"]
                ),
                "num_frames": int(row["num_frames"]),
                "is_dirty": bool(int(row["is_dirty"])),
            })
    return episodes


def _decode_jpeg(raw):
    """Decode a JPEG byte string into an RGB uint8 image."""
    if raw is None:
        return None
    if isinstance(raw, np.ndarray) and raw.ndim == 3 and raw.shape[-1] == 3 \
            and raw.dtype == np.uint8:
        return raw
    if hasattr(raw, "tobytes"):
        raw = raw.tobytes()
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _collect_task_instructions(
    episodes: list[dict], task_name: str
) -> list[str]:
    """Collect all unique instructions for a task by scanning its episodes."""
    seen = set()
    instructions = []
    for ep in episodes:
        if ep["task_name"] != task_name:
            continue
        try:
            with open(ep["instruction_path"], "r") as f:
                instr_dict = json.load(f)
            for text in instr_dict.get("seen", []):
                if text not in seen:
                    seen.add(text)
                    instructions.append(text)
        except Exception:
            pass
    return instructions


def load_episode_data(hdf5_path: str, instruction_path: str, *,
                     umi_coord_frame: bool = False,
                     umi_gripper_space: bool = False) -> dict:
    """Load state, action, images and instruction from one RoboTwin episode.

    State layout (16-dim, xyzw quaternion convention):
        [left_x, left_y, left_z, left_qx, left_qy, left_qz, left_qw,
         left_gripper,
         right_x, right_y, right_z, right_qx, right_qy, right_qz, right_qw,
         right_gripper]

    Action layout (20-dim, RT-relative in wrist frame):
        [left_dxyz(3), left_relRot6d(6), left_gripper(1),
         right_dxyz(3), right_relRot6d(6), right_gripper(1)]

    Note: HDF5 stores quaternions in wxyz; we convert to xyzw (scipy convention).

    When *umi_coord_frame* is True, the full UMI coordinate transform is applied
    at load-time so that downstream consumers see pre-transformed data:

    * World frame:  (right,fwd,up) → (fwd,left,up)  via W
    * Local frame:  col permutation  via P = [[0,0,1],[1,0,0],[0,1,0]]

    Positions:  p_umi = W @ p_rd       (swaps & negates x-y)
    Rotation:   R_umi = W @ R_rd @ P   (applied to quaternions)
    """
    with h5py.File(hdf5_path, "r") as f:
        # --- State ---
        # ``observations/eepos``: (T, 16), wxyz quaternion order in HDF5.
        qpos = f["observations"]["eepos"][:].astype(np.float32)

        # Convert quaternion from wxyz to xyzw.
        # Left quat: columns 3,4,5,6 (w,x,y,z) -> (x,y,z,w)
        qpos[:, [3, 4, 5, 6]] = qpos[:, [4, 5, 6, 3]]
        # Right quat: columns 11,12,13,14 (w,x,y,z) -> (x,y,z,w)
        qpos[:, [11, 12, 13, 14]] = qpos[:, [12, 13, 14, 11]]

        state = qpos  # (T, 16)

        # --- Action: compute RT-relative from consecutive states ---
        T = state.shape[0]
        action = np.zeros((T, 20), dtype=np.float32)
        if T >= 2:
            for t in range(T - 1):
                chunk = state[t:t+2].copy()  # (2, 16)
                rel = dual_arm_poses_to_relative(chunk)  # (2, 20)
                action[t] = rel[1].astype(np.float32)
            # Last frame: no next frame, leave as zeros.

        # --- Full UMI coordinate pre-transform ---
        if umi_coord_frame:
            # State: full transform (positions + quaternions).
            state = convert_frame_robo_to_umi(state, convert_gripper=umi_gripper_space)

            # Action (20-dim RT-relative): apply W to delta positions only.
            # Layout: ldx=0, ldy=1, ldz=2; rdx=10, rdy=11, rdz=12
            ld = action[:, 0:3].copy()
            action[:, 0:3] = np.column_stack([ld[:, 1], -ld[:, 0], ld[:, 2]])
            rd = action[:, 10:13].copy()
            action[:, 10:13] = np.column_stack([rd[:, 1], -rd[:, 0], rd[:, 2]])

        # --- Images ---
        images: dict[str, list] = {}
        vis_grp = f["observations"]["images"]
        for name, key, _ in CAM_KEYS:
            cam_key = key.split("/")[-1]  # "cam_left_wrist" etc.
            img_list = []
            if cam_key in vis_grp:
                dataset = vis_grp[cam_key]
                for i in range(len(dataset)):
                    raw = dataset[i]
                    img = _decode_jpeg(raw)
                    if img is not None:
                        img_list.append(img)
                    else:
                        img_list.append(
                            np.zeros((240, 424, 3), dtype=np.uint8)
                        )
            else:
                img_list = [
                    np.zeros((240, 424, 3), dtype=np.uint8)
                ] * T
            images[name] = img_list

        # --- Instruction ---
        with open(instruction_path, "r") as f_instr:
            instruction_dict = json.load(f_instr)
        instructions = instruction_dict.get("seen", [])
        instruction = instructions[0] if instructions else ""

        # --- Frequency (default 25 Hz if not stored) ---
        freq = 25
        if "additional_info" in f and "frequency" in f["additional_info"]:
            freq = int(f["additional_info"]["frequency"][()])

    return {
        "state": state,
        "action": action,
        "images": images,
        "instruction": instruction,
        "fps": float(freq),
        "T": T,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Middle column: 2D position plots
# ═══════════════════════════════════════════════════════════════════════════

def build_static_figure(state: np.ndarray, action: np.ndarray | None,
                        fig_w_in: float, fig_h_in: float, dpi: int):
    T = state.shape[0]
    t_arr = np.arange(T)
    if action is not None and action.shape[0] > 0:
        t_act = np.arange(min(action.shape[0], T))
    else:
        t_act = None
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)

    left_x, right_x = 0.10, 0.99
    top_pad, bot_pad = 0.012, 0.012
    inner_pad = 0.010
    n = len(PLOT_ORDER)
    usable = max(1.0 - top_pad - bot_pad - (n - 1) * inner_pad, 1e-3)
    h_each = usable / n

    axes: list = []
    for i, key in enumerate(PLOT_ORDER):
        y_top = 1.0 - top_pad - i * (h_each + inner_pad)
        y_bot = y_top - h_each
        ax = fig.add_axes([left_x, y_bot, right_x - left_x, h_each])

        idx = POS_INDEX.get(key)
        combined: list[np.ndarray] = []
        if idx is not None and idx < state.shape[1]:
            ax.plot(t_arr, state[:, idx], color=COLORS[key], lw=1.0,
                    label=f"s:{key}")
            combined.append(state[:, idx])
        else:
            combined.append(np.array([0.0]))
            ax.plot([], [], color=COLORS.get(key, "gray"), lw=1.0,
                    label=f"s:{key}")

        # Action overlay: dashed black (RT-relative format, 20-dim).
        act_idx = RT_ACTION_INDEX.get(key)
        if t_act is not None and act_idx is not None and act_idx < action.shape[1]:
            ax.plot(t_act, action[:len(t_act), act_idx],
                    color="black", lw=0.9, ls="--", alpha=0.55,
                    label=f"a:{key}")
            combined.append(action[:len(t_act), act_idx])

        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="upper right", fontsize=6, framealpha=0.55,
                  handlelength=2.0, borderpad=0.2, labelspacing=0.15)
        ax.set_xlim(0, max(T - 1, 1))

        all_vals = np.concatenate(combined)
        lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
        pad = max((hi - lo) * 0.10, 1e-3)
        ax.set_ylim(lo - pad, hi + pad)

        if i < n - 1:
            ax.set_xticklabels([])
        axes.append(ax)

    fig.canvas.draw()
    return fig, axes


def compute_cursor_bboxes(fig, axes: list, T: int,
                          target_w: int, target_h: int) -> list[dict]:
    renderer = fig.canvas.get_renderer()
    fig_w_px = renderer.width
    fig_h_px = renderer.height
    scale_x = target_w / fig_w_px
    scale_y = target_h / fig_h_px
    t_idx = np.arange(T)
    entries: list[dict] = []

    for ax in axes:
        (x0_lim, x1_lim) = ax.get_xlim()
        (x0_px, _) = ax.transData.transform((x0_lim, 0))
        (x1_px, _) = ax.transData.transform((x1_lim, 0))
        bbox = ax.get_window_extent()
        y0_img = int(round((fig_h_px - bbox.y1) * scale_y))
        y1_img = int(round((fig_h_px - bbox.y0) * scale_y))
        denom = max(x1_lim - x0_lim, 1e-9)
        x_pix = x0_px + (t_idx - x0_lim) * (x1_px - x0_px) / denom
        entries.append({
            "x_pix": np.round(x_pix * scale_x).astype(np.int32),
            "y0": max(y0_img, 0),
            "y1": max(y1_img, 0),
        })
    return entries


def rasterize_figure(fig) -> np.ndarray:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()


# ═══════════════════════════════════════════════════════════════════════════
# Right column: 3D trajectory
# ═══════════════════════════════════════════════════════════════════════════

def build_3d_axes(state: np.ndarray, fig_w_in: float, fig_h_in: float,
                  dpi: int, *, umi_coord_frame: bool = False):
    pos_l = state[:, [LEFT_X, LEFT_Y, LEFT_Z]].astype(np.float64)
    pos_r = state[:, [RIGHT_X, RIGHT_Y, RIGHT_Z]].astype(np.float64)
    all_pts = np.concatenate([pos_l, pos_r], axis=0)
    mid = all_pts.mean(axis=0)
    half = max(np.ptp(all_pts, axis=0).max(), 1e-3) * 0.52

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.view_init(elev=30, azim=225)
    if umi_coord_frame:
        ax.set_xlabel("X \u2192 fwd", fontsize=7)
        ax.set_ylabel("Y \u2190 left", fontsize=7)
        ax.set_zlabel("Z \u2191 up", fontsize=7)
    else:
        ax.set_xlabel("X \u2192", fontsize=7)
        ax.set_ylabel("Y (into screen)", fontsize=7)
        ax.set_zlabel("Z \u2191", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title("Trajectory", fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.08)
    return fig, ax


def render_3d_panel(state: np.ndarray, size: int, dpi: int = 100, *, umi_coord_frame: bool = False):
    T = state.shape[0]
    pos_l = state[:, [LEFT_X, LEFT_Y, LEFT_Z]]
    pos_r = state[:, [RIGHT_X, RIGHT_Y, RIGHT_Z]]
    quat_l = state[:, 3:7]      # xyzw
    quat_r = state[:, 11:15]

    t_arr = np.arange(T)
    inch = size / dpi

    # --- Static background: full trajectory ---
    fig_bg, ax_bg = build_3d_axes(state, inch, inch, dpi, umi_coord_frame=umi_coord_frame)
    ax_bg.scatter(pos_l[:, 0], pos_l[:, 1], pos_l[:, 2],
                  c=t_arr, cmap="Oranges", s=3, alpha=0.7, label="Left")
    ax_bg.scatter(pos_r[:, 0], pos_r[:, 1], pos_r[:, 2],
                  c=t_arr, cmap="Blues",   s=3, alpha=0.7, label="Right")
    kw = {"s": 50, "edgecolors": "black", "linewidths": 0.6}
    ax_bg.scatter(*pos_l[0],  c="darkorange", marker="o", **kw)
    ax_bg.scatter(*pos_r[0],  c="darkblue",   marker="o", **kw)
    ax_bg.scatter(*pos_l[-1], c="red",        marker="s", **kw)
    ax_bg.scatter(*pos_r[-1], c="navy",       marker="s", **kw)
    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="darkorange", markersize=7, label="Left"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="darkblue",   markersize=7, label="Right"),
    ]
    ax_bg.legend(handles=legend_handles, fontsize=6, loc="lower center",
                 ncol=2, bbox_to_anchor=(0.5, -0.06), frameon=False)

    canvas_bg = FigureCanvasAgg(fig_bg)
    canvas_bg.draw()
    static_bg = np.asarray(canvas_bg.buffer_rgba())[..., :3].copy()
    if static_bg.shape[:2] != (size, size):
        static_bg = cv2.resize(static_bg, (size, size),
                               interpolation=cv2.INTER_AREA)
    plt.close(fig_bg)

    # --- Per-frame overlay: current-position marker + EE triads ---
    elev, azim = ax_bg.elev, ax_bg.azim
    all_pts = np.concatenate([pos_l, pos_r], axis=0)
    triad_len = max(np.ptp(all_pts, axis=0).max(), 1e-3) * 0.16

    markers_rgba: list[np.ndarray] = []
    fig_mk, ax_mk = build_3d_axes(state, inch, inch, dpi, umi_coord_frame=umi_coord_frame)
    ax_mk.view_init(elev=elev, azim=azim)
    fig_mk.patch.set_alpha(0.0)
    ax_mk.patch.set_alpha(0.0)
    ax_mk.set_facecolor("none")
    for axis_name in ("xaxis", "yaxis", "zaxis"):
        getattr(ax_mk, axis_name).set_visible(False)
    ax_mk.set_xlabel(""); ax_mk.set_ylabel(""); ax_mk.set_zlabel("")
    ax_mk.set_title("")
    ax_mk.grid(False)

    sc_l = ax_mk.scatter([], [], [], c="orange", s=80,
                          edgecolors="white", linewidths=1.5, zorder=10)
    sc_r = ax_mk.scatter([], [], [], c="dodgerblue", s=80,
                          edgecolors="white", linewidths=1.5, zorder=10)

    def _plot_triad(ax, origin, rot, length, colors):
        """Draw 3-colour arrow-headed triad at *origin*.

        Returns a list of artists (quivers) for later cleanup.
        """
        ox, oy, oz = origin
        artists = []
        for col_idx, clr in enumerate(colors):
            d = rot[:, col_idx] * length
            q = ax.quiver(ox, oy, oz, d[0], d[1], d[2],
                          color=clr, arrow_length_ratio=0.22, lw=2.0,
                          capstyle="butt", zorder=9)
            artists.append(q)
        return artists

    TRIAD_L_COLORS = ("#e74c3c", "#2ecc71", "#3498db")
    TRIAD_R_COLORS = ("#c0392b", "#27ae60", "#2980b9")

    canvas_mk = FigureCanvasAgg(fig_mk)

    for t in range(T):
        p_l = state[t, [LEFT_X, LEFT_Y, LEFT_Z]].astype(np.float64)
        p_r = state[t, [RIGHT_X, RIGHT_Y, RIGHT_Z]].astype(np.float64)
        rot_l = R.from_quat(quat_l[t]).as_matrix()
        rot_r = R.from_quat(quat_r[t]).as_matrix()

        sc_l._offsets3d = (
            [float(p_l[0])], [float(p_l[1])], [float(p_l[2])]
        )
        sc_r._offsets3d = (
            [float(p_r[0])], [float(p_r[1])], [float(p_r[2])]
        )

        for line in list(ax_mk.lines):
            line.remove()
        for col in list(ax_mk.collections):
            if col not in (sc_l, sc_r):
                col.remove()

        _plot_triad(ax_mk, p_l, rot_l, triad_len, TRIAD_L_COLORS)
        _plot_triad(ax_mk, p_r, rot_r, triad_len, TRIAD_R_COLORS)

        canvas_mk.draw()
        rgba = np.asarray(canvas_mk.buffer_rgba()).copy()
        if rgba.shape[:2] != (size, size):
            rgba = cv2.resize(rgba, (size, size),
                              interpolation=cv2.INTER_AREA)
        markers_rgba.append(rgba)

    plt.close(fig_mk)
    return static_bg, markers_rgba


# ═══════════════════════════════════════════════════════════════════════════
# Per-frame rendering helpers
# ═══════════════════════════════════════════════════════════════════════════

def _resize_cam(img: np.ndarray, target_w: int,
                *, target_h: int | None = None) -> np.ndarray:
    h, w = img.shape[:2]
    new_h = int(round(h * target_w / w)) if target_h is None else target_h
    return cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_AREA)


def _render_text_line(text: str, width: int, height: int, fontsize: int = 13,
                      color=(80, 80, 80),
                      bg_color=(245, 245, 245)) -> np.ndarray:
    canvas = np.full((height, width, 3), bg_color, dtype=np.uint8)
    if any(ord(ch) > 127 for ch in text):
        try:
            font = ImageFont.truetype(_CJK_FONT_PATH, fontsize)
        except Exception:
            font = ImageFont.load_default()
        img_pil = Image.new("RGB", (width, height), bg_color)
        draw = ImageDraw.Draw(img_pil)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = max((width - tw) // 2, 0)
        y = max((height - th) // 2, 0)
        draw.text((x, y), text, fill=color, font=font)
        return np.array(img_pil)
    else:
        scale = 0.40
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        x = max((width - tw) // 2, 0)
        y = height // 2 + 5
        cv2.putText(canvas, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        return canvas


def _draw_title_banner(width: int, text: str, *,
                       banner_h: int = 44,
                       bg=(255, 255, 255),
                       fg=(0, 0, 0)) -> np.ndarray:
    banner = np.full((banner_h, width, 3), bg, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    while scale > 0.3:
        (tw, _), _ = cv2.getTextSize(text, font, scale, 1)
        if tw <= width - 20:
            break
        scale -= 0.05
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    org_x = max((width - tw) // 2, 10)
    org_y = (banner_h + th) // 2
    cv2.putText(banner, text, (org_x, org_y), font, scale, fg, 1, cv2.LINE_AA)
    return banner


# ═══════════════════════════════════════════════════════════════════════════
# Main video renderer
# ═══════════════════════════════════════════════════════════════════════════

def render_video(
    hdf5_path: str,
    instruction_path: str,
    output_path: str,
    *,
    task_name: str = "",
    episode_id: int = 0,
    all_instructions: Optional[list[str]] = None,
    fps: float = 30.0,
    left_width: int = 640,
    step: int = 1,
    no_3d: bool = False,
    umi_coord_frame: bool = False,
    umi_gripper_space: bool = False,
) -> None:
    print(f"Loading episode {hdf5_path} ...")
    data = load_episode_data(hdf5_path, instruction_path,
                             umi_coord_frame=umi_coord_frame,
                             umi_gripper_space=umi_gripper_space)
    state = data["state"]
    images = data["images"]
    T = data["T"]
    instruction = data["instruction"]
    source_fps = data["fps"]
    print(f"  frames={T}, state={list(state.shape)}, "
          f"action={list(data['action'].shape)}")
    print(f"  instruction: {instruction[:80] if instruction else '(empty)'}")
    print(f"  source fps: {source_fps}")

    # Determine camera heights.
    cam_heights: dict[str, int] = {}
    for name, _, _ in CAM_KEYS:
        cam_heights[name] = _resize_cam(images[name][0], left_width).shape[0]
    panel_h = sum(cam_heights.values())

    dpi = 100
    middle_w = 450
    traj_w = 800

    # Middle column: 2D position plots.
    print("Building 2D position plots ...")
    fig2d, axes = build_static_figure(
        state, data["action"], middle_w / dpi, panel_h / dpi, dpi
    )
    static_mid = rasterize_figure(fig2d)
    static_mid = cv2.resize(static_mid, (middle_w, panel_h),
                            interpolation=cv2.INTER_AREA)
    cursors = compute_cursor_bboxes(fig2d, axes, T, middle_w, panel_h)
    plt.close(fig2d)

    # Right column: 3D trajectory.
    if not no_3d:
        print("Building 3D trajectory ...")
        # Render 3D in UMI frame when umi_coord_frame is active.
        _3d_frame = umi_coord_frame
        static_3d, markers_rgba = render_3d_panel(state, traj_w, dpi, umi_coord_frame=_3d_frame)
        if traj_w < panel_h:
            pad_top = (panel_h - traj_w) // 2
            pad_bot = panel_h - traj_w - pad_top
            static_3d = np.concatenate([
                np.full((pad_top, traj_w, 3), 255, dtype=np.uint8),
                static_3d,
                np.full((pad_bot, traj_w, 3), 255, dtype=np.uint8),
            ], axis=0)
            pad_top_rgba = np.zeros((pad_top, traj_w, 4), dtype=np.uint8)
            pad_bot_rgba = np.zeros((pad_bot, traj_w, 4), dtype=np.uint8)
            markers_rgba = [
                np.concatenate([pad_top_rgba, mk, pad_bot_rgba], axis=0)
                for mk in markers_rgba
            ]
    else:
        static_3d = None
        markers_rgba = []

    # Pre-resize cameras.
    print("Resizing camera frames ...")
    resized_cams: dict[str, list[np.ndarray]] = {}
    for name, _, _ in CAM_KEYS:
        h_target = cam_heights[name]
        resized_cams[name] = [
            _resize_cam(img, left_width, target_h=h_target)
            for img in images[name]
        ]

    # Prepare instruction carousel.
    if all_instructions and len(all_instructions) > 1:
        instr_carousel = all_instructions
        use_carousel = True
        print(f"  instruction carousel: {len(instr_carousel)} unique "
              f"instructions")
    else:
        instr_carousel = [instruction] if instruction else [""]
        use_carousel = False

    # Video writer.
    title_h = 44
    instr_h = 28 if instruction else 0
    right_w = 0 if no_3d else traj_w
    canvas_w = left_width + middle_w + right_w
    canvas_h = panel_h + title_h + instr_h
    canvas_w -= canvas_w % 2
    canvas_h -= canvas_h % 2

    writer = imageio.get_writer(
        output_path, fps=fps, codec="libx264", format="FFMPEG"
    )
    cursor_color = (40, 40, 40)

    try:
        for t in tqdm(range(T), desc=f"Ep{episode_id}", unit="frame"):
            if step > 1 and t % step != 0:
                continue

            # Left: 3-cam stack.
            left_col = np.concatenate(
                [resized_cams[name][t] for name, _, _ in CAM_KEYS], axis=0)

            # Middle: 2D plots + cursor.
            mid_col = static_mid.copy()
            mid_col = cv2.cvtColor(mid_col, cv2.COLOR_RGB2BGR)
            for c in cursors:
                x = int(c["x_pix"][t])
                cv2.line(mid_col, (x, c["y0"]), (x, c["y1"]),
                         cursor_color, 1, cv2.LINE_AA)
            mid_col = cv2.cvtColor(mid_col, cv2.COLOR_BGR2RGB)

            # Right: 3D + marker overlay.
            if not no_3d:
                mk_rgba = markers_rgba[t].astype(np.float32) / 255.0
                mk_rgb, mk_a = mk_rgba[..., :3], mk_rgba[..., 3:4]
                right_col = (
                    static_3d.astype(np.float32) * (1.0 - mk_a)
                    + (mk_rgb * 255.0) * mk_a
                )
                right_col = np.clip(right_col, 0, 255).astype(np.uint8)
                body = np.concatenate([left_col, mid_col, right_col], axis=1)
            else:
                body = np.concatenate([left_col, mid_col], axis=1)

            # Title bar.
            title_bar = _draw_title_banner(
                canvas_w,
                f"{task_name}  |  Episode {episode_id}  |  "
                f"frame {t}/{T - 1}  ({t / source_fps:.2f}s)",
            )

            bars = [title_bar]

            # Instruction banner.
            if instr_h > 0:
                cur_instr = (
                    instr_carousel[t % len(instr_carousel)]
                    if use_carousel else instruction
                )
                instr_bar = _render_text_line(
                    cur_instr, canvas_w, instr_h, fontsize=11,
                    color=(30, 30, 120), bg_color=(240, 240, 255)
                )
                bars.append(instr_bar)

            canvas = np.concatenate([*bars, body], axis=0)
            if canvas.shape[0] % 2:
                canvas = canvas[:-1, :, :]
            if canvas.shape[1] % 2:
                canvas = canvas[:, :-1, :]
            writer.append_data(canvas)
    finally:
        writer.close()

    print(f"Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Visualize a single episode from a RoboTwin HDF5 dataset "
                    "as mp4.")
    parser.add_argument(
        "hdf5_dir", type=str,
        help="Path to the RoboTwin HDF5 root directory (the one containing "
             "task subdirectories like 'adjust_bottle/...').")
    parser.add_argument(
        "-t", "--task", type=str, default=None,
        help="Optional task name filter (e.g. 'adjust_bottle'). "
             "If not provided, all tasks are included.")
    parser.add_argument(
        "-e", "--episode", type=int, default=0,
        help="Episode index (global index in the CSV, default: 0)")
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to dataset_index.csv. Default: auto-detect "
             "in hdf5_dir, then fall back to repo's assets/dataset_index.csv")
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output mp4 path (default: assets/{task}_episode_{idx:03d}.mp4)")
    parser.add_argument(
        "--fps", type=float, default=30.0,
        help="Output video frame rate (default: 30)")
    parser.add_argument(
        "--left-width", type=int, default=640,
        help="Camera panel pixel width (default: 640)")
    parser.add_argument(
        "--step", type=int, default=1,
        help="Write every Nth frame (default: 1)")
    parser.add_argument(
        "--no-3d", action="store_true", default=False,
        help="Disable 3D trajectory panel")
    parser.add_argument("--umi-coord-frame", action="store_true",
                        help="Convert EE poses to UMI coordinate frame "
                             "(fwd=+x, left=+y, up=+z) for rendering.")
    parser.add_argument("--umi-gripper-space", action="store_true",
                        help="Also convert gripper values from RoboTwin convention "
                             "(0-1 norm) to UMI convention (0-90 mm) "
                             "when --umi-coord-frame is used.")
    args = parser.parse_args()

    hdf5_dir = os.path.abspath(args.hdf5_dir)

    # Resolve CSV path: try {hdf5_dir}/dataset_index.csv first,
    # then fall back to repo's assets/dataset_index.csv.
    if args.csv:
        csv_path = args.csv
    else:
        csv_path = os.path.join(hdf5_dir, "dataset_index.csv")
        if not os.path.isfile(csv_path):
            csv_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "assets", "dataset_index.csv"
            )
    if not os.path.isfile(csv_path):
        raise SystemExit(f"CSV index not found: {csv_path}")
    print(f"CSV index: {csv_path}")

    # Load all episodes.
    all_episodes = _load_csv_episodes(csv_path, hdf5_dir)
    print(f"Loaded {len(all_episodes)} episodes from CSV")

    # Filter by task (prefix match, e.g. "adjust_bottle" matches
    # "adjust_bottle-aloha-agilex_clean_50-50").
    if args.task:
        episodes = [ep for ep in all_episodes
                    if ep["task_name"].startswith(args.task)]
        if not episodes:
            tasks = sorted(set(ep["task_name"] for ep in all_episodes))
            raise SystemExit(
                f"No episodes matching task prefix '{args.task}'. "
                f"Available tasks: {tasks[:20]}"
                f"{'...' if len(tasks) > 20 else ''}"
            )
    else:
        episodes = all_episodes

    # Find requested episode by global index.
    target = None
    for ep in episodes:
        if ep["global_index"] == args.episode:
            target = ep
            break

    if target is None:
        if args.task:
            indices = sorted(set(ep["global_index"] for ep in episodes))
            raise SystemExit(
                f"Episode {args.episode} not found for task '{args.task}'. "
                f"Available global indices ({len(indices)} total): "
                f"{indices[:20]}{'...' if len(indices) > 20 else ''}"
            )
        else:
            raise SystemExit(
                f"Episode {args.episode} not found in CSV "
                f"(total: {len(episodes)}). "
                f"Use -t to filter by task first."
            )

    task_name = target["task_name"]
    hdf5_path = target["hdf5_path"]
    instruction_path = target["instruction_path"]

    if not os.path.isfile(hdf5_path):
        raise SystemExit(f"HDF5 file not found: {hdf5_path}")
    if not os.path.isfile(instruction_path):
        raise SystemExit(f"Instruction file not found: {instruction_path}")

    # Collect task instructions for carousel.
    all_instructions = _collect_task_instructions(all_episodes, task_name)
    if all_instructions:
        print(f"Task '{task_name}': {len(all_instructions)} unique "
              f"instruction(s)")

    if args.output:
        output = args.output
    else:
        out_dir = Path("assets")
        out_dir.mkdir(parents=True, exist_ok=True)
        output = str(
            out_dir / f"{task_name}_episode_{args.episode:03d}.mp4"
        )

    render_video(
        hdf5_path,
        instruction_path,
        output,
        task_name=task_name,
        episode_id=args.episode,
        all_instructions=all_instructions,
        fps=args.fps,
        left_width=args.left_width,
        step=args.step,
        no_3d=args.no_3d,
        umi_coord_frame=args.umi_coord_frame,
        umi_gripper_space=args.umi_gripper_space,
    )


if __name__ == "__main__":
    main()
