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

"""Vision-language-action dataset wrapper for Hy-VLA supervised training.

Wraps :class:`hy_vla.data.robotwin_dataset.RoboTwinVLADataset` with image
augmentation, history-stack assembly, and the collator that produces
the batch dict consumed by :class:`hy_vla.modeling_hy_vla.HyVLA.forward`.
"""

import traceback
import math
import random
from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from hy_vla.utils.image_corrupt import image_corrupt


class VLADataset(Dataset):
    """A vision-languange-action Dataset for supervised training.
    This dataset will load data from the buffer directory.
    """

    def __init__(self, config):
        super(VLADataset, self).__init__()

        self.num_cameras = config.dataset.num_cameras
        # ``num_input_images`` is the camera count; the K-frame stack is
        # wired separately via ``use_video_encoder`` / ``img_history_size``.
        self.num_input_images = self.num_cameras
        self.img_history_size = config.dataset.img_history_size
        self.cond_mask_prob = config.dataset.cond_mask_prob
        self.cam_ext_mask_prob = config.dataset.cam_ext_mask_prob
        self.use_video_encoder = getattr(config.dataset, "use_video_encoder", False)
        # When True, ``__getitem__`` forwards the dataloader-provided
        # ``index`` straight through to ``hdf5_dataset.get_item`` so each
        # call resolves to a fixed (episode, raw_step, instruction) triple.
        self.deterministic = bool(getattr(config.dataset, "deterministic", False))

        # Auto-detect backend via unified ``source`` key.
        #   source="robotwin"  → RoboTwinVLADataset  (RoboTwin)
        #   source="umi"       → LanceVLADataset  (UMI / Hy-Embodied)
        dataset_source = getattr(config.dataset, "source", "robotwin")
        if dataset_source == "umi":
            from .umi_dataset import LanceVLADataset
            self.hdf5_dataset = LanceVLADataset(config)
        else:
            from .robotwin_dataset import RoboTwinVLADataset
            self.hdf5_dataset = RoboTwinVLADataset(config)

        self.image_size = config.dataset.image_size
        self.auto_adjust_image_brightness = config.dataset.auto_adjust_image_brightness
        self.image_aug = config.dataset.image_aug
        self.camera_randomcrop_aug = config.dataset.camera_randomcrop_aug

    @staticmethod
    def pairwise(iterable):
        a = iter(iterable)
        return zip(a, a)

    def __len__(self) -> int:
        return len(self.hdf5_dataset)

    def __getitem__(self, index):
        # Retry on per-sample failures so a single bad hdf5 entry does
        # not bring the whole dataloader worker down.
        while True:
            data_dict = None
            try:
                if self.deterministic:
                    res = self.hdf5_dataset.get_item(index=index)
                else:
                    res = self.hdf5_dataset.get_item()
                content = res['meta']
                states = res['state']
                actions = res['actions']
                state_elem_mask = res['state_indicator']
                image_metas = [
                    res['cam_high'], res['cam_high_mask'],
                    res['cam_left_wrist'], res['cam_left_wrist_mask'],
                    res['cam_right_wrist'], res['cam_right_wrist_mask'],
                ]

                data_dict = {}
                data_dict["states"] = states
                data_dict["actions"] = actions
                data_dict["state_elem_mask"] = state_elem_mask \
                    if random.random() > self.cond_mask_prob else np.zeros_like(state_elem_mask)

                # ---- RGB ----
                image_metas = list(self.pairwise(image_metas))
                mask_probs = [self.cond_mask_prob] * self.num_input_images
                if self.cam_ext_mask_prob >= 0.0:
                    mask_probs[0] = self.cam_ext_mask_prob
                rearranged_images = []
                for i in range(self.img_history_size):
                    for j in range(self.num_input_images):
                        images, image_mask = image_metas[j]
                        image, valid = images[i], image_mask[i]
                        if valid and (math.prod(image.shape) > 0) and (random.random() > mask_probs[j]):
                            rearranged_images.append((image, True))
                        else:
                            rearranged_images.append((np.zeros(image.shape, dtype=image.dtype), False))
                preprocessed_images = []
                for i, (image, valid) in enumerate(rearranged_images):
                    image = Image.fromarray(image)
                    if self.image_size is not None:
                        image = transforms.Resize(self.image_size)(image)  # keeps ratio

                    if valid and self.auto_adjust_image_brightness:
                        pixel_values = list(image.getdata())
                        average_brightness = sum(sum(pixel) for pixel in pixel_values) / (len(pixel_values) * 255.0 * 3)
                        if average_brightness <= 0.15:
                            image = transforms.ColorJitter(brightness=(1.75, 1.75))(image)

                    # Apply image augmentation to ~50% of valid images.
                    if valid and self.image_aug and (random.random() > 0.5):
                        aug_type = random.choice(["corrput_only", "color_only", "both"])
                        if aug_type != "corrput_only":
                            image = transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.05)(image)
                        if aug_type != "color_only":
                            image = image_corrupt(image)

                        # Head-camera-only random crop+resize.
                        if self.camera_randomcrop_aug:
                            if random.random() > 0.5:
                                if i == 0:  # head camera
                                    width, height = image.size
                                    transform = transforms.Compose([
                                        transforms.RandomCrop((int(height * 0.95), int(width * 0.95))),
                                        transforms.Resize((height, width)),
                                    ])
                                    image = transform(image)

                    preprocessed_images.append(np.array(image))

                data_dict["images"] = [torch.from_numpy(img).permute(2, 0, 1) for img in preprocessed_images]
                data_dict["instructions"] = content["instruction"] if random.random() > self.cond_mask_prob else ""
                if self.use_video_encoder:
                    K = self.img_history_size
                    imgs = data_dict["images"]
                    # Per-camera K-frame stack: list of ``num_cameras``
                    # tensors, each (K, 3, H, W).
                    data_dict["images_history"] = [
                        torch.stack([imgs[k * self.num_input_images + j] for k in range(K)], dim=0)
                        for j in range(self.num_cameras)
                    ]

                for k, v in data_dict.items():
                    if isinstance(v, np.ndarray):
                        data_dict[k] = torch.from_numpy(v).float()

                return data_dict
            except (KeyboardInterrupt, SystemExit, MemoryError):
                # NEVER swallow these: KeyboardInterrupt/SystemExit must
                # propagate for clean shutdown; MemoryError means the
                # worker is unrecoverable and should die loudly so
                # DataLoader can replace it instead of looping forever.
                raise
            except Exception as e:
                # Surface index + hdf5 path so per-sample failures are
                # diagnosable. ``data_dict`` is set lazily after the hdf5
                # read, so for hdf5-side failures we look the path up
                # via the underlying dataset's deterministic index.
                _hdf5_path = None
                try:
                    if (
                        getattr(self, "hdf5_dataset", None) is not None
                        and getattr(self.hdf5_dataset, "deterministic", False)
                        and getattr(self.hdf5_dataset, "deterministic_index", None) is not None
                    ):
                        _idx_pairs = self.hdf5_dataset.deterministic_index
                        _N = len(_idx_pairs)
                        if _N > 0:
                            _ep_idx, _raw_step = _idx_pairs[int(index) % _N]
                            _hdf5_path = self.hdf5_dataset.episodes[_ep_idx].get("hdf5_path")
                except Exception:
                    _hdf5_path = None
                if data_dict is not None:
                    print(
                        f"[dataset][__getitem__ ERR] index={index}, "
                        f"hdf5_path={_hdf5_path}, "
                        f"dataset_name={data_dict.get('dataset_name')}, err={e!r}"
                    )
                else:
                    print(
                        f"[dataset][__getitem__ ERR] index={index}, "
                        f"hdf5_path={_hdf5_path}, err={e!r}"
                    )
                traceback.print_exc()
                # Advance the index. In deterministic mode this can step
                # OUT of the per-rank slice, but that's safer than
                # raising (which would crash the worker and surface as a
                # spurious StopIteration upstream).
                index = (index + 1) % len(self)


class VLADataCollator(object):
    """Collate examples for supervised training."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = {
            "states": [],
            "actions": [],
            "state_elem_mask": [],
            "images": [],
            "instructions": [],
        }

        for instance in instances:
            keys_to_check = ['states', 'actions', 'state_elem_mask']
            for key in keys_to_check:
                if isinstance(instance[key], torch.Tensor):
                    item = instance[key]
                else:
                    item = torch.from_numpy(instance[key])
                batch[key].append(item)

            batch["images"].append(instance["images"])
            batch["instructions"].append(instance["instructions"])

        keys_to_stack = ['states', 'actions', 'state_elem_mask']
        for key in keys_to_stack:
            batch[key] = torch.stack(batch[key], dim=0)

        # If the dataset emitted a per-camera K-frame history stack, the
        # three main camera keys become 5D ``(B, K, C, H, W)`` so the
        # MEM video-encoder branch of ``prepare_images`` sees a real
        # time-series. Otherwise they stay 4D ``(B, C, H, W)``.
        has_history = "images_history" in instances[0]
        if has_history:
            top_stack = torch.stack([inst["images_history"][0] for inst in instances], dim=0) / 255.
            left_stack = torch.stack([inst["images_history"][1] for inst in instances], dim=0) / 255.
            right_stack = torch.stack([inst["images_history"][2] for inst in instances], dim=0) / 255.
        else:
            top_stack = torch.stack([v[0] for v in batch["images"]], dim=0) / 255.
            left_stack = torch.stack([v[1] for v in batch["images"]], dim=0) / 255.
            right_stack = torch.stack([v[2] for v in batch["images"]], dim=0) / 255.

        batch_format = {
            # (B,C,H,W) when has_history=False, else (B,K,C,H,W); 0-1 range.
            "observation.images.top_head": top_stack,
            "observation.images.hand_left": left_stack,
            "observation.images.hand_right": right_stack,
            "observation.state": batch["states"][:, 0, :],   # (B, D)
            "action": batch["actions"][:, :, :],             # (B, Chunk, D)
            "task": batch["instructions"],                   # list of len B
        }

        return batch_format
