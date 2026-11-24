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

"""Hy-VLA quickstart smoke test.

Loads the released ``tencent/Hy-Embodied-0.5-VLA-RoboTwin`` checkpoint,
runs one forward pass on dummy inputs, and prints the resulting action
tensor shape. Use this to verify that the install + ckpt download path works
before plugging Hy-VLA into a real RoboTwin / hardware loop.

Usage::

    python scripts/quick_start.py                 # CUDA, else Intel XPU, else CPU
    python scripts/quick_start.py --device xpu    # or pick the device
"""

import argparse

import torch
from huggingface_hub import snapshot_download
from hy_vla import HyVLA, HyVLAConfig
from hy_vla.utils.device import get_default_device

parser = argparse.ArgumentParser(description="Hy-VLA quickstart smoke test.")
parser.add_argument(
    "--device", default=None,
    help="cuda, xpu or cpu (default: the first one available, in that order)",
)
args = parser.parse_args()
device = torch.device(args.device) if args.device else get_default_device()

ckpt = snapshot_download("tencent/Hy-Embodied-0.5-VLA-RoboTwin")

config = HyVLAConfig.from_pretrained(ckpt)
policy = HyVLA.from_pretrained(ckpt, config=config)
policy.enable_video_encoder_if_needed()
policy = policy.to(device=device, dtype=torch.bfloat16).eval()

# (B, K, C, H, W); K=6 history slots, slot K-1 is current
img = torch.zeros(1, 6, 3, 224, 224, device=device, dtype=torch.bfloat16)
# normalized dual-arm EEF: [xyz(3) + rot6d(6) + gripper(1)]*2
state = torch.zeros((1, config.max_state_dim), device=device, dtype=torch.bfloat16)
batch = {
    "observation.images.top_head":   img,
    "observation.images.hand_left":  img,
    "observation.images.hand_right": img,
    "observation.state": state,
    "task": ["pick up the bottle"],
}

# (B, H*2, D) normalized actions in delta EEF and EEF: 
# - delta EEF: [xyz(3) + rot6d(6) + gripper(1)]*2
# - EEF: [xyz(3) + rot6d(6) + gripper(1)]*2
with torch.no_grad():
    actions = policy.forward_evaluate(batch)["pred"]
    actions = actions[..., : config.action_feature.shape[0]]
print(actions.shape)