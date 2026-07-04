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
Per-episode video visualisation for Lance (LeRobot-format) datasets.

3-column layout: camera stack | 2D position plots | 3D trajectory.

Usage:
  python scripts/vis_umi_episode.py                         # HF Hub, table auto-detect
  python scripts/vis_umi_episode.py -t table_000 -e 666     # specific table & episode
  python scripts/vis_umi_episode.py /path/to/lance_dir      # local Lance root
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.lines import Line2D
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from hy_vla.data.umi_dataset import LanceTableReader

warnings.filterwarnings("ignore", message="lance is not fork-safe")
warnings.filterwarnings("ignore", message="lancedb fork support is experimental")

# 16-dim state layout: left arm [0-7], right arm [8-15]
EEPOS_INDEX = {
    "xl": 0, "yl": 1, "zl": 2, "gl": 7,
    "xr": 8, "yr": 9, "zr": 10, "gr": 15,
}
PLOT_ORDER = ["xl", "yl", "zl", "gl", "gr", "xr", "yr", "zr"]
COLORS = {
    "xl": "tab:red", "yl": "tab:green", "zl": "tab:blue", "gl": "tab:orange",
    "gr": "tab:purple", "xr": "tab:red", "yr": "tab:green", "zr": "tab:blue",
}

_CJK_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

CAM_KEYS: list[tuple[str, str]] = [
    ("cam_left_wrist", "observation.images.cam_left_wrist"),
    ("cam_high",        "observation.images.cam_high"),
    ("cam_right_wrist", "observation.images.cam_right_wrist"),
]

# ════════════
# Data loading
# ═══════════════════

def _get_arr(frame: dict, key: str, default_shape=(16,)):
    val = frame.get(key)
    if val is None:
        return np.zeros(default_shape, dtype=np.float32)
    if isinstance(val, np.ndarray):
        return val.astype(np.float32)
    if isinstance(val, (list, tuple)):
        return np.array(val, dtype=np.float32)
    return np.array([val], dtype=np.float32)


def load_episode(ds: LanceTableReader, ep_idx: int) -> dict:
    frames = ds.get_episode(ep_idx)
    T = len(frames)

    state = np.array([_get_arr(f, "observation.state") for f in frames], dtype=np.float32)
    action = np.array([_get_arr(f, "action") for f in frames], dtype=np.float32)

    images: dict[str, list[np.ndarray]] = {}
    for name, key in CAM_KEYS:
        img_list = []
        for f in frames:
            img = f.get(key)
            if img is not None and isinstance(img, np.ndarray) and img.ndim == 3:
                img_list.append(img)
            else:
                img_list.append(np.zeros((240, 424, 3), dtype=np.uint8))
        images[name] = img_list

    task_map = {int(t["task_index"]): t["task"] for t in ds.meta.get("tasks", [])}
    task_texts: list[str] = []
    for f in frames:
        tv = f.get("task_index", f.get("task", ""))
        if isinstance(tv, (int, np.integer)):
            task_texts.append(task_map.get(int(tv), ""))
        else:
            task_texts.append(str(tv) if tv else "")

    return {"state": state, "action": action, "images": images,
            "T": T, "task_texts": task_texts}


# ══════════
# Middle column: 2D position-vs-time subplots (rendered once)
# ════════════════════

def build_static_figure(state: np.ndarray, fig_w_in: float, fig_h_in: float, dpi: int):
    T = state.shape[0]
    t_arr = np.arange(T)
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

        idx = EEPOS_INDEX.get(key)
        if idx is not None and idx < state.shape[1]:
            ax.plot(t_arr, state[:, idx], color=COLORS[key], lw=1.0, label=f"s:{key}")
            combined = state[:, idx]
        else:
            combined = np.array([0.0])
            ax.plot([], [], color=COLORS.get(key, "gray"), lw=1.0, label=f"s:{key}")

        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="upper right", fontsize=6, framealpha=0.55,
                  handlelength=2.0, borderpad=0.2, labelspacing=0.15)
        ax.set_xlim(0, max(T - 1, 1))

        lo, hi = float(np.min(combined)), float(np.max(combined))
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


def rastarize_figure(fig) -> np.ndarray:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()


# ══════════════════
# Right column: 3D end-effector trajectory
# ═══════════════════════════════════════════════

def build_3d_axes(state: np.ndarray, fig_w_in: float, fig_h_in: float, dpi: int):
    n_dims = state.shape[1]
    # fallback: use first 6 dims for left/right xyz if < 16
    lx, ly, lz = 0, 1, 2
    rx, ry, rz = 8, 9, 10
    if n_dims < 16:
        lx, ly, lz = 0, 1, 2
        rx, ry, rz = 3, 4, 5 if n_dims >= 6 else lx, ly, lz

    pos_l = state[:, [lx, ly, lz]]
    pos_r = state[:, [rx, ry, rz]]
    all_pts = np.concatenate([pos_l, pos_r], axis=0)
    mid = all_pts.mean(axis=0)
    half = max(np.ptp(all_pts, axis=0).max(), 1e-3) * 0.52

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.set_xlabel("X (into screen)", fontsize=7)
    ax.set_ylabel("Y ←", fontsize=7)
    ax.set_zlabel("Z ", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title("Trajectory", fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.08)
    ax.view_init(elev=30, azim=225)
    return fig, ax


def render_3d_panel(state: np.ndarray, size: int, dpi: int = 100):
    T = state.shape[0]
    n_dims = state.shape[1]
    lx, ly, lz = 0, 1, 2
    rx, ry, rz = 8, 9, 10
    l_q0, l_q1, l_q2, l_q3 = 3, 4, 5, 6
    r_q0, r_q1, r_q2, r_q3 = 11, 12, 13, 14
    has_quat = n_dims >= 16

    pos_l = state[:, [lx, ly, lz]]
    pos_r = state[:, [rx, ry, rz]]
    t_arr = np.arange(T)
    inch = size / dpi

    # ── static background: full trajectory ──
    fig_bg, ax_bg = build_3d_axes(state, inch, inch, dpi)
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

    # ── per-frame marker overlays (transparent) ──
    elev, azim = ax_bg.elev, ax_bg.azim

    markers_rgba: list[np.ndarray] = []
    fig_mk, ax_mk = build_3d_axes(state, inch, inch, dpi)
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
        """Draw 3-colour arrow-headed triad at *origin* using quiver."""
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

    # Extract quaternions once (xyzw).
    if has_quat:
        quat_l = state[:, l_q0:l_q3 + 1]
        quat_r = state[:, r_q0:r_q3 + 1]

    canvas_mk = FigureCanvasAgg(fig_mk)
    all_pts = np.concatenate([pos_l, pos_r], 0)
    triad_len = max(np.ptp(all_pts, axis=0).max(), 1e-3) * 0.16

    for t in range(T):
        sc_l._offsets3d = ([float(state[t, lx])],
                           [float(state[t, ly])],
                           [float(state[t, lz])])
        sc_r._offsets3d = ([float(state[t, rx])],
                           [float(state[t, ry])],
                           [float(state[t, rz])])

        if has_quat:
            # Remove old triad artists from previous frame.
            for line in list(ax_mk.lines):
                line.remove()
            for col in list(ax_mk.collections):
                if col not in (sc_l, sc_r):
                    col.remove()

            p_l = state[t, [lx, ly, lz]].astype(np.float64)
            p_r = state[t, [rx, ry, rz]].astype(np.float64)
            rot_l = R.from_quat(quat_l[t]).as_matrix()
            rot_r = R.from_quat(quat_r[t]).as_matrix()
            _plot_triad(ax_mk, p_l, rot_l, triad_len, TRIAD_L_COLORS)
            _plot_triad(ax_mk, p_r, rot_r, triad_len, TRIAD_R_COLORS)

        canvas_mk.draw()
        rgba = np.asarray(canvas_mk.buffer_rgba()).copy()
        if rgba.shape[:2] != (size, size):
            rgba = cv2.resize(rgba, (size, size), interpolation=cv2.INTER_AREA)
        markers_rgba.append(rgba)

    plt.close(fig_mk)
    return static_bg, markers_rgba


# ═════════════════════════════════════════
# Per-frame rendering
# ═════════════════════════════════════════

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
        cv2.putText(canvas, text, (width // 2 - 100, height // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)
        return canvas


def render_video(
    ds: LanceTableReader,
    ep_idx: int,
    output_path: str,
    *,
    fps: float = 30.0,
    left_width: int = 640,
    step: int = 1,
    no_3d: bool = False,
) -> None:
    print(f"Loading episode {ep_idx} …")
    data = load_episode(ds, ep_idx)
    state = data["state"]
    images = data["images"]
    T = data["T"]
    task_texts = data.get("task_texts", [])
    print(f"  frames={T}, state={list(state.shape)}, action={list(data['action'].shape)}")
    first_task = next((t for t in task_texts if t), "")
    if first_task:
        print(f"  task: {first_task}")

    source_fps = float(ds.fps)

    # ── Determine camera heights ──
    cam_heights: dict[str, int] = {}
    for name, _ in CAM_KEYS:
        cam_heights[name] = _resize_cam(images[name][0], left_width).shape[0]
    panel_h = sum(cam_heights.values())

    dpi = 100
    middle_w = 450
    traj_w = 800

    # ── Middle column: 2D position plots ──
    print("Building 2D position plots …")
    fig2d, axes = build_static_figure(state, middle_w / dpi, panel_h / dpi, dpi)
    static_mid = rastarize_figure(fig2d)
    static_mid = cv2.resize(static_mid, (middle_w, panel_h), interpolation=cv2.INTER_AREA)
    cursors = compute_cursor_bboxes(fig2d, axes, T, middle_w, panel_h)
    plt.close(fig2d)

    # ── Right column: 3D trajectory ──
    if not no_3d:
        print("Building 3D trajectory …")
        static_3d, markers_rgba = render_3d_panel(state, traj_w, dpi)
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

    # ── Pre-resize cameras ──
    print("Resizing camera frames …")
    resized_cams: dict[str, list[np.ndarray]] = {}
    for name, _ in CAM_KEYS:
        h_target = cam_heights[name]
        resized_cams[name] = [
            _resize_cam(img, left_width, target_h=h_target) for img in images[name]
        ]

    # ── Video writer ──
    title_h = 36
    task_h = 28 if any(task_texts) else 0
    right_w = 0 if no_3d else traj_w
    canvas_w = left_width + middle_w + right_w
    canvas_h = panel_h + title_h + task_h
    canvas_w -= canvas_w % 2
    canvas_h -= canvas_h % 2

    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", format="FFMPEG")
    cursor_color = (40, 40, 40)
    font = cv2.FONT_HERSHEY_SIMPLEX

    try:
        for t in tqdm(range(T), desc=f"Episode {ep_idx}", unit="frame"):
            if step > 1 and t % step != 0:
                continue

            # Left: 3-cam stack
            left_col = np.concatenate(
                [resized_cams[name][t] for name, _ in CAM_KEYS], axis=0)

            # Middle: 2D plots + cursor
            mid_col = static_mid.copy()
            mid_col = cv2.cvtColor(mid_col, cv2.COLOR_RGB2BGR)
            for c in cursors:
                x = int(c["x_pix"][t])
                cv2.line(mid_col, (x, c["y0"]), (x, c["y1"]),
                         cursor_color, 1, cv2.LINE_AA)
            mid_col = cv2.cvtColor(mid_col, cv2.COLOR_BGR2RGB)

            # Right: 3D + marker overlay
            if not no_3d:
                mk_rgba = markers_rgba[t].astype(np.float32) / 255.0
                mk_rgb, mk_a = mk_rgba[..., :3], mk_rgba[..., 3:4]
                right_col = (static_3d.astype(np.float32) * (1.0 - mk_a)
                             + (mk_rgb * 255.0) * mk_a)
                right_col = np.clip(right_col, 0, 255).astype(np.uint8)
                body = np.concatenate([left_col, mid_col, right_col], axis=1)
            else:
                body = np.concatenate([left_col, mid_col], axis=1)

            # Title bar
            title_bar = np.full((title_h, canvas_w, 3), (255, 255, 255), dtype=np.uint8)
            text = f"Episode {ep_idx}  |  frame {t}/{T - 1}  ({t / source_fps:.2f}s)"
            (tw, _), _ = cv2.getTextSize(text, font, 0.45, 1)
            cv2.putText(title_bar, text, ((canvas_w - tw) // 2, title_h - 10),
                        font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            bars = [title_bar]
            cur_task = task_texts[t] if t < len(task_texts) else ""
            if cur_task and task_h > 0:
                task_bar = _render_text_line(cur_task, canvas_w, task_h, fontsize=13)
            else:
                task_bar = np.full((task_h, canvas_w, 3), (245, 245, 245), dtype=np.uint8)
            bars.append(task_bar)

            canvas = np.concatenate([*bars, body], axis=0)
            if canvas.shape[0] % 2:
                canvas = canvas[:-1, :, :]
            if canvas.shape[1] % 2:
                canvas = canvas[:, :-1, :]
            writer.append_data(canvas)
    finally:
        writer.close()

    print(f"Saved: {output_path}")


# ════════════════════
# CLI
# ════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Visualize a single episode from a Lance dataset as mp4")
    parser.add_argument("lance_source", type=str, nargs="?",
                        default="tencent/Hy-Embodied-0.5-VLA-Data",
                        help="Lance data source: HF Hub repo id (default) or local directory path")
    parser.add_argument("-t", "--table", type=str, default="table_000",
                        help="Table name (required if multiple tables exist)")
    parser.add_argument("-e", "--episode", type=int, default=666,
                        help="Episode index (default: 666)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output mp4 path (default: assets/{table}_episode_{idx:03d}.mp4)")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Output video frame rate (default: 30)")
    parser.add_argument("--left-width", type=int, default=640,
                        help="Camera panel pixel width (default: 640)")
    parser.add_argument("--step", type=int, default=3,
                        help="Write every Nth frame (default: 3)")
    parser.add_argument("--no-3d", action="store_true", default=False,
                        help="Disable 3D trajectory panel (2-col layout only)")
    args = parser.parse_args()

    if os.path.isdir(args.lance_source):
        ds = LanceTableReader(root=args.lance_source, table_name=args.table)
    else:
        ds = LanceTableReader(repo_id=args.lance_source, table_name=args.table)
    print(ds)

    if args.episode < 0 or args.episode >= ds.num_episodes:
        raise SystemExit(
            f"Episode {args.episode} out of range [0, {ds.num_episodes - 1}]")

    if args.output:
        output = args.output
    else:
        out_dir = Path("assets")
        out_dir.mkdir(parents=True, exist_ok=True)
        table_prefix = ds.table_name or "lance"
        output = str(out_dir / f"{table_prefix}_episode_{args.episode:03d}.mp4")
    render_video(ds, args.episode, output,
                 fps=args.fps, left_width=args.left_width,
                 step=args.step, no_3d=args.no_3d)


if __name__ == "__main__":
    main()
