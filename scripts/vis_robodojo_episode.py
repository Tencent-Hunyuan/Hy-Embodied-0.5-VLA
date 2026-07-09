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
Per-episode video visualisation for RoboDojo HDF5 datasets.

3-column layout: camera stack | 2D position plots | 3D trajectory.

Instruction banner: if multiple unique instructions exist across the task's
episodes, they are cycled frame-by-frame (carousel) similar to the RoboTwin
``vis_video.py`` behaviour; otherwise the single instruction is shown
statically.

Usage:
  python scripts/vis_robodojo_episode.py \
      /path/to/RoboDojo_hdf5/arrange_largest_number \
      -e 0

  python scripts/vis_robodojo_episode.py \
      /path/to/RoboDojo_hdf5 \
      -t arrange_largest_number -e 5 --fps 15
"""

from __future__ import annotations

import argparse
import glob
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
from hy_vla.utils.transform_utils import convert_frame_robo_to_umi

warnings.filterwarnings("ignore", message="not fork-safe")

_CJK_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

# Camera keys in the HDF5 and their display names.
CAM_KEYS: list[tuple[str, str, str]] = [
    ("cam_left_wrist", "vision/cam_left_wrist", "cam_left_wrist"),
    ("cam_head",       "vision/cam_head",        "cam_head"),
    ("cam_right_wrist", "vision/cam_right_wrist", "cam_right_wrist"),
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
# Action layout (16-dim): same as state — delta EE poses:
#   [ldx,ldy,ldz, ldqw,ldqx,ldqy,ldqz, lgrip, rdx,rdy,rdz, rdqw,rdqx,rdqy,rdqz, rgrip]
ACTION_INDEX = POS_INDEX  # 1:1 dimension correspondence with state
PLOT_ORDER = ["xl", "yl", "zl", "gl", "gr", "xr", "yr", "zr"]
COLORS = {
    "xl": "tab:red",   "yl": "tab:green",  "zl": "tab:blue",   "gl": "tab:orange",
    "gr": "tab:purple", "xr": "tab:red",   "yr": "tab:green",  "zr": "tab:blue",
}


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

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
    # NOTE: RoboDojo HDF5 JPEGs decode to RGB directly.
    return img


def _scan_episodes(task_dir: str, task_filter: Optional[str] = None) -> list[dict]:
    """Scan for episode HDF5 files in a task directory or root directory.

    If ``task_dir`` points directly to a task subdirectory (contains
    ``arx_x5/data/``), scan only that task. Otherwise treat it as a
    root directory and scan all tasks.
    """
    episodes = []

    def _scan_one(task_path, task_name):
        for robot_dir in sorted(os.listdir(task_path)):
            robot_path = os.path.join(task_path, robot_dir)
            if not os.path.isdir(robot_path):
                continue
            data_dir = os.path.join(robot_path, "data")
            if not os.path.isdir(data_dir):
                continue
            for h5_path in sorted(glob.glob(os.path.join(data_dir, "episode_*.hdf5"))):
                basename = os.path.basename(h5_path)
                ep_id_str = basename.replace("episode_", "").replace(".hdf5", "")
                try:
                    ep_id = int(ep_id_str)
                except ValueError:
                    continue
                episodes.append({
                    "task_name": task_name,
                    "robot_type": robot_dir,
                    "hdf5_path": h5_path,
                    "episode_id": ep_id,
                })

    # Check if task_dir is itself a task directory.
    if task_filter is not None:
        task_subdir = os.path.join(task_dir, task_filter)
        if os.path.isdir(task_subdir):
            _scan_one(task_subdir, task_filter)
    elif any(
        os.path.isdir(os.path.join(task_dir, d, "arx_x5", "data"))
        for d in os.listdir(task_dir)
    ):
        # This looks like a root directory with multiple tasks.
        for d in sorted(os.listdir(task_dir)):
            dp = os.path.join(task_dir, d)
            if os.path.isdir(dp):
                _scan_one(dp, d)
    else:
        # Treat as a single task directory.
        task_name = os.path.basename(os.path.abspath(task_dir))
        _scan_one(task_dir, task_name)

    return episodes


def _collect_task_instructions(task_dir: str, task_name: str) -> list[str]:
    """Collect all unique instructions for a task by scanning all its episodes."""
    seen = set()
    instructions = []
    eps = _scan_episodes(task_dir, task_filter=task_name)
    for ep in eps:
        try:
            with h5py.File(ep["hdf5_path"], "r") as f:
                raw = f["instruction"][()]
                if isinstance(raw, bytes):
                    text = raw.decode("utf-8")
                else:
                    text = str(raw)
                if text not in seen:
                    seen.add(text)
                    instructions.append(text)
        except Exception:
            pass
    return instructions


def load_episode_data(hdf5_path: str, *, umi_coord_frame: bool = False,
                       umi_gripper_space: bool = False) -> dict:
    """Load state, action, images and instruction from one RoboDojo episode.

    When *umi_coord_frame* is True, the full UMI coordinate transform is applied
    at load-time so that downstream consumers see pre-transformed data:

    * World frame:  (right,fwd,up) → (fwd,left,up)  via W
    * Local frame:  col permutation  via P = [[0,0,1],[1,0,0],[0,1,0]]

    Positions:  p_umi = W @ p_rd       (swaps & negates x-y)
    Rotation:   R_umi = W @ R_rd @ P   (applied to quaternions)
    """
    with h5py.File(hdf5_path, "r") as f:
        state_grp = f["state"]
        left_ee = state_grp["left_ee_poses"][:]    # (T, 7)
        right_ee = state_grp["right_ee_poses"][:]   # (T, 7)
        left_grip = state_grp["left_ee_joint_states"][:]    # (T, 1)
        right_grip = state_grp["right_ee_joint_states"][:]   # (T, 1)

        # Build state keeping raw wxyz quaternion order from HDF5.
        # Layout: [lx,ly,lz, qw,qx,qy,qz, lgrip, rx,ry,rz, qw,qx,qy,qz, rgrip]
        state = np.concatenate([
            left_ee, left_grip, right_ee, right_grip,
        ], axis=1).astype(np.float32)  # (T, 16)

        # HDF5 stores quaternions in wxyz; convert to xyzw (scipy convention).
        state[:, [3, 4, 5, 6]] = state[:, [4, 5, 6, 3]]
        state[:, [11, 12, 13, 14]] = state[:, [12, 13, 14, 11]]

        # Delta EE poses: (T, 7) = [dx, dy, dz, dqw, dqx, dqy, dqz] (wxyz).
        # Build a 16-dim action that matches the state layout dimension-by-dimension.
        left_delta = state_grp["left_delta_ee_poses"][:].astype(np.float32)    # (T, 7)
        right_delta = state_grp["right_delta_ee_poses"][:].astype(np.float32)   # (T, 7)

        # Action: joint-level actions for gripper reference.
        act_grp = f["action"]
        la_grip = act_grp["left_ee_joint_states"][:].astype(np.float32)     # (T, 1)
        ra_grip = act_grp["right_ee_joint_states"][:].astype(np.float32)    # (T, 1)

        # 16-dim delta action: [ldx,ldy,ldz, ldqw,ldqx,ldqy,ldqz, lgrip,
        #                       rdx,rdy,rdz, rdqw,rdqx,rdqy,rdqz, rgrip]
        action = np.concatenate([
            left_delta, la_grip, right_delta, ra_grip,
        ], axis=1).astype(np.float32)  # (T, 16)

        # Convert action quaternions to xyzw as well.
        action[:, [3, 4, 5, 6]] = action[:, [4, 5, 6, 3]]
        action[:, [11, 12, 13, 14]] = action[:, [12, 13, 14, 11]]

        # --- Full UMI coordinate pre-transform ---
        # Apply world-frame W and local-frame P transforms once at load time.
        # After this, all downstream consumers see pre-transformed UMI data.
        if umi_coord_frame:
            state = convert_frame_robo_to_umi(state, convert_gripper=umi_gripper_space)
            action = convert_frame_robo_to_umi(action, convert_gripper=umi_gripper_space)  # robodojo action is 16-dim, same layout

        # Images: JPEG-decoded.
        images: dict[str, list] = {}
        vis_grp = f["vision"]
        for name, key, _ in CAM_KEYS:
            cam_key = key.split("/")[-1]  # "cam_left_wrist" etc.
            img_list = []
            if cam_key in vis_grp and "colors" in vis_grp[cam_key]:
                colors = vis_grp[cam_key]["colors"]
                for i in range(len(colors)):
                    img = _decode_jpeg(colors[i])
                    if img is not None:
                        img_list.append(img)
                    else:
                        img_list.append(np.zeros((240, 424, 3), dtype=np.uint8))
            else:
                img_list = [np.zeros((240, 424, 3), dtype=np.uint8)] * state.shape[0]
            images[name] = img_list

        # Instruction.
        raw = f["instruction"][()]
        if isinstance(raw, bytes):
            instruction = raw.decode("utf-8")
        else:
            instruction = str(raw)

        # Frequency.
        freq = 25
        if "additional_info" in f and "frequency" in f["additional_info"]:
            freq = int(f["additional_info"]["frequency"][()])

    return {
        "state": state,
        "action": action,
        "images": images,
        "instruction": instruction,
        "fps": float(freq),
        "T": state.shape[0],
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
            ax.plot(t_arr, state[:, idx], color=COLORS[key], lw=1.0, label=f"s:{key}")
            combined.append(state[:, idx])
        else:
            combined.append(np.array([0.0]))
            ax.plot([], [], color=COLORS.get(key, "gray"), lw=1.0, label=f"s:{key}")

        # Action overlay: dashed black, semi-transparent (matching vis_video.py style).
        act_idx = ACTION_INDEX.get(key)
        if t_act is not None and act_idx is not None and act_idx < action.shape[1]:
            ax.plot(t_act, action[:len(t_act), act_idx],
                    color="black", lw=0.9, ls="--", alpha=0.55, label=f"a:{key}")
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


def compute_cursor_bboxes(fig, axes: list, T: int, target_w: int, target_h: int) -> list[dict]:
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

def build_3d_axes(state: np.ndarray, fig_w_in: float, fig_h_in: float, dpi: int,
                  *, umi_coord_frame: bool = False):
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
        # UMI world: X=fwd, Y=left, Z=up
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
    # Extract quaternions: state is xyzw (scipy convention).
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
        Line2D([0], [0], marker="o", color="w", markerfacecolor="darkorange",
               markersize=7, label="Left"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="darkblue",
               markersize=7, label="Right"),
    ]
    ax_bg.legend(handles=legend_handles, fontsize=6, loc="lower center",
                 ncol=2, bbox_to_anchor=(0.5, -0.06), frameon=False)

    canvas_bg = FigureCanvasAgg(fig_bg)
    canvas_bg.draw()
    static_bg = np.asarray(canvas_bg.buffer_rgba())[..., :3].copy()
    if static_bg.shape[:2] != (size, size):
        static_bg = cv2.resize(static_bg, (size, size), interpolation=cv2.INTER_AREA)
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

        sc_l._offsets3d = ([float(p_l[0])], [float(p_l[1])], [float(p_l[2])])
        sc_r._offsets3d = ([float(p_r[0])], [float(p_r[1])], [float(p_r[2])])

        # Remove all old triad artists (lines + quiver collections).
        for line in list(ax_mk.lines):
            line.remove()
        for col in list(ax_mk.collections):
            if col not in (sc_l, sc_r):  # preserve current-position markers
                col.remove()

        _plot_triad(ax_mk, p_l, rot_l, triad_len, TRIAD_L_COLORS)
        _plot_triad(ax_mk, p_r, rot_r, triad_len, TRIAD_R_COLORS)

        canvas_mk.draw()
        rgba = np.asarray(canvas_mk.buffer_rgba()).copy()
        if rgba.shape[:2] != (size, size):
            rgba = cv2.resize(rgba, (size, size), interpolation=cv2.INTER_AREA)
        markers_rgba.append(rgba)

    plt.close(fig_mk)
    return static_bg, markers_rgba


# ═══════════════════════════════════════════════════════════════════════════
# Per-frame rendering helpers
# ═══════════════════════════════════════════════════════════════════════════

def _resize_cam(img: np.ndarray, target_w: int, *, target_h: int | None = None) -> np.ndarray:
    h, w = img.shape[:2]
    new_h = int(round(h * target_w / w)) if target_h is None else target_h
    return cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_AREA)


def _render_text_line(text: str, width: int, height: int, fontsize: int = 13,
                      color=(80, 80, 80), bg_color=(245, 245, 245)) -> np.ndarray:
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
    data = load_episode_data(hdf5_path, umi_coord_frame=umi_coord_frame,
                             umi_gripper_space=umi_gripper_space)
    state = data["state"]
    images = data["images"]
    T = data["T"]
    instruction = data["instruction"]
    source_fps = data["fps"]
    print(f"  frames={T}, state={list(state.shape)}, action={list(data['action'].shape)}")
    print(f"  instruction: {instruction[:80]}...")
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
    fig2d, axes = build_static_figure(state, data["action"], middle_w / dpi, panel_h / dpi, dpi)
    static_mid = rasterize_figure(fig2d)
    static_mid = cv2.resize(static_mid, (middle_w, panel_h), interpolation=cv2.INTER_AREA)
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
            _resize_cam(img, left_width, target_h=h_target) for img in images[name]
        ]

    # Prepare instruction carousel.
    if all_instructions and len(all_instructions) > 1:
        instr_carousel = all_instructions
        use_carousel = True
        print(f"  instruction carousel: {len(instr_carousel)} unique instructions")
    else:
        instr_carousel = [instruction]
        use_carousel = False

    # Video writer.
    title_h = 44
    instr_h = 28 if instruction else 0
    right_w = 0 if no_3d else traj_w
    canvas_w = left_width + middle_w + right_w
    canvas_h = panel_h + title_h + instr_h
    canvas_w -= canvas_w % 2
    canvas_h -= canvas_h % 2

    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", format="FFMPEG")
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
            # Convert mid_col back to RGB (matplotlib canvas was BGR for cv2.draw).
            mid_col = cv2.cvtColor(mid_col, cv2.COLOR_BGR2RGB)

            # Right: 3D + marker overlay.
            if not no_3d:
                mk_rgba = markers_rgba[t].astype(np.float32) / 255.0
                mk_rgb, mk_a = mk_rgba[..., :3], mk_rgba[..., 3:4]
                right_col = (static_3d.astype(np.float32) * (1.0 - mk_a)
                             + (mk_rgb * 255.0) * mk_a)
                right_col = np.clip(right_col, 0, 255).astype(np.uint8)
                body = np.concatenate([left_col, mid_col, right_col], axis=1)
            else:
                body = np.concatenate([left_col, mid_col], axis=1)

            # Title bar.
            title_bar = _draw_title_banner(
                canvas_w,
                f"{task_name}  |  Episode {episode_id}  |  frame {t}/{T - 1}  ({t / source_fps:.2f}s)",
            )

            bars = [title_bar]

            # Instruction banner (static or carousel).
            if instr_h > 0:
                cur_instr = instr_carousel[t % len(instr_carousel)] if use_carousel else instruction
                instr_bar = _render_text_line(cur_instr, canvas_w, instr_h, fontsize=11,
                                              color=(30, 30, 120), bg_color=(240, 240, 255))
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
        description="Visualize a single episode from a RoboDojo HDF5 dataset as mp4")
    parser.add_argument(
        "data_dir", type=str,
        help="Path to either a task subdirectory (e.g. .../arrange_largest_number) "
             "or the root RoboDojo directory containing multiple tasks.")
    parser.add_argument(
        "-t", "--task", type=str, default=None,
        help="Task name (required if data_dir is the root directory with multiple tasks)")
    parser.add_argument(
        "-e", "--episode", type=int, default=0,
        help="Episode index within the task (default: 0)")
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
                        help="Also convert gripper values from RoboDojo convention "
                             "(0-1 norm) to UMI convention (0-90 mm) "
                             "when --umi-coord-frame is used.")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)

    # Scan episodes.
    episodes = _scan_episodes(data_dir, task_filter=args.task)

    if not episodes:
        task_msg = f" for task '{args.task}'" if args.task else ""
        raise SystemExit(f"No episodes found in {data_dir}{task_msg}")

    # Filter by task if needed.
    if args.task is not None:
        episodes = [ep for ep in episodes if ep["task_name"] == args.task]
        if not episodes:
            raise SystemExit(f"No episodes for task '{args.task}' in {data_dir}")

    # Find the requested episode.
    target = None
    for ep in episodes:
        if ep["episode_id"] == args.episode:
            target = ep
            break

    if target is None:
        task_name = args.task or episodes[0]["task_name"]
        ep_ids = sorted(set(ep["episode_id"] for ep in episodes))
        raise SystemExit(
            f"Episode {args.episode} not found for task '{task_name}'. "
            f"Available episodes: {ep_ids[:20]}{'...' if len(ep_ids) > 20 else ''}")

    task_name = target["task_name"]
    hdf5_path = target["hdf5_path"]

    # Collect task instructions for carousel.
    all_instructions = None
    if args.task is not None:
        all_instructions = _collect_task_instructions(data_dir, args.task)
    else:
        # data_dir IS a task directory; scan it.
        task_dir_name = os.path.basename(data_dir)
        # Check if data_dir is a root or a leaf task directory.
        if os.path.isdir(os.path.join(data_dir, "arx_x5", "data")):
            all_instructions = _collect_task_instructions(
                os.path.dirname(data_dir) if "/" in data_dir else ".",
                task_dir_name,
            )
        else:
            all_instructions = [target.get("instruction", "")]

    if args.output:
        output = args.output
    else:
        out_dir = Path("assets")
        out_dir.mkdir(parents=True, exist_ok=True)
        output = str(out_dir / f"{task_name}_episode_{args.episode:03d}.mp4")

    render_video(
        hdf5_path,
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
