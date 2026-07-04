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

"""Hy-VLA training entry point.

Driven by Hydra (config tree under ``hy_vla/config``); launches under
HuggingFace ``accelerate`` with a DeepSpeed ZeRO-2 plugin. See
``hy_vla/config/base.yaml`` and the README for the full launch recipe.
"""
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import copy
import gc
import logging
import math
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration, set_seed
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from transformers import logging as hf_logging

try:
    # accelerate >= 0.30 moved DataLoader-related kwargs onto a dedicated
    # DataLoaderConfiguration; accelerate 1.x removes the legacy kwarg
    # entirely. Feature-detect once and pick the supported call shape.
    from accelerate import DataLoaderConfiguration  # type: ignore
    _HAS_DATALOADER_CFG = True
except ImportError:  # pragma: no cover -- accelerate < 0.30
    DataLoaderConfiguration = None  # type: ignore
    _HAS_DATALOADER_CFG = False

from hy_vla.data.vla_dataset import VLADataCollator, VLADataset
from hy_vla.modeling_hy_vla import HyVLA, HyVLAConfig, _load_vlm_autoconfig
from hy_vla.utils.logger import initialize_wandb, save_wandb

from hy_vla.hunyuan_vl_mot import HunYuanVLMoTForConditionalGeneration

hf_logging.set_verbosity_error()

script_logger = logging.getLogger("train")


@hydra.main(
    version_base=None,
    config_path="./config",
    config_name="base",
)
def train(cfg: DictConfig) -> None:
    # for better performance, but reduce reproducibility
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    torch.backends.cuda.matmul.allow_tf32 = True

    accelerator_project_config = ProjectConfiguration(total_limit=cfg.checkpoints_total_limit)
    _ds_cfg = cfg.deepspeed
    if isinstance(_ds_cfg, str) and not os.path.isabs(_ds_cfg):
        _ds_cfg = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), _ds_cfg)
        )
    _accel_kwargs = dict(
        mixed_precision=cfg.training.mixed_precision,
        gradient_accumulation_steps=cfg.training.grad_accumulation_steps,
        deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=_ds_cfg),
        project_config=accelerator_project_config,
    )
    if _HAS_DATALOADER_CFG:
        _accel_kwargs["dataloader_config"] = DataLoaderConfiguration(
            dispatch_batches=False,
        )
    else:
        _accel_kwargs["dispatch_batches"] = False
    accelerator = Accelerator(**_accel_kwargs)
    if not cfg.debug:
        if accelerator.is_main_process:
            if cfg.resume_ckpt:
                cfg_prev = OmegaConf.load(Path(cfg.resume_ckpt).parent / "base.yaml")
                cfg = OmegaConf.merge(cfg_prev, {"resume_ckpt": cfg.resume_ckpt})
                cfg.ckpt_save_dir = cfg.resume_ckpt
                cfg = initialize_wandb(cfg)  # load wandb.run.id
            else:
                cfg = initialize_wandb(cfg)
                wandb.config.update(OmegaConf.to_container(cfg))
                os.makedirs(cfg.ckpt_save_dir, exist_ok=True)
                save_wandb(wandb, cfg.ckpt_save_dir)
                OmegaConf.save(cfg, Path(cfg.ckpt_save_dir) / "base.yaml")

    # Sync RNG across ranks BEFORE ``accelerator.prepare`` so DataLoader's
    # RandomSampler is identical on every rank.
    set_seed(int(cfg.seed))

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    elif accelerator.mixed_precision == "no":
        weight_dtype = torch.float32

    ############# Config Model ############
    accelerator.print("Initialize Model")
    
    # Resolve the model-side chunk length. ``with_absolute`` action types
    # double the action tensor along the chunk-time axis (rel + abs over
    # the same N future frames -> 2*N rows), so the model must produce
    # 2*N tokens per call.
    _chunk_mult = 2 if "with_absolute" in cfg.dataset.act_type else 1
    _model_chunk = int(cfg.dataset.action_chunk_size) * _chunk_mult
    if _chunk_mult != 1:
        script_logger.info(
            "[model] with_absolute: model chunk_size = %d * %d = %d",
            cfg.dataset.action_chunk_size, _chunk_mult, _model_chunk,
        )

    hy_config = HyVLAConfig(
        vlm_model_path=cfg.model.vlm_model_path,
        n_action_steps=_model_chunk,
        chunk_size=_model_chunk,
        optimizer_lr=cfg.training.optimizer_lr,
        optimizer_betas=tuple(cfg.training.optimizer_betas),
        optimizer_eps=cfg.training.optimizer_eps,
        optimizer_weight_decay=cfg.training.optimizer_weight_decay,
        scheduler_warmup_steps=cfg.training.scheduler_warmup_steps,
        scheduler_decay_steps=cfg.training.scheduler_decay_steps,
        scheduler_decay_lr=cfg.training.scheduler_decay_lr,
        resize_imgs_with_padding=(cfg.dataset.image_size, cfg.dataset.image_size),
        use_video_encoder=getattr(cfg.dataset, 'use_video_encoder', False),
        visual_segment_isolation=bool(cfg.model.visual_segment_isolation),
    )
    script_logger.info(
        "[model] image_features=%s n_action_steps=%d",
        list(hy_config.image_features.keys()), hy_config.n_action_steps,
    )

    if cfg.resume_ckpt:
        ckpt_model_path = os.path.join(cfg.resume_ckpt, "model")
        accelerator.print(f"Resuming from checkpoint {cfg.resume_ckpt}")
        accelerator.print(f"Loading model weights from: {ckpt_model_path}")
        policy = HyVLA.from_pretrained(ckpt_model_path, local_files_only=True)
    else:
        # Pretrain loading branches, controlled by ``cfg.model.pretrain_source``:
        #   * ``vla``     : load a full VLA pretrain via
        #                   ``HyVLA.from_pretrained``; reads ``vla_model_path``.
        #   * ``vlm``     : load only the upstream Hy-Embodied VLM backbone;
        #                   action expert randomly initialized; reads
        #                   ``vlm_model_path``.
        #   * ``scratch`` : both VLM backbone and action expert randomly
        #                   initialized.
        pretrain_source = getattr(cfg.model, 'pretrain_source', 'vla')
        if pretrain_source == 'vla':
            vla_path = cfg.model.vla_model_path
            if not vla_path:
                raise ValueError(
                    "pretrain_source='vla' requires model.vla_model_path "
                    "to be set (HF repo id or local directory)."
                )
            policy = HyVLA.from_pretrained(vla_path, config=hy_config, strict=False)
            accelerator.print(f"[pretrain=vla] load full VLA model from: {vla_path}")
        elif pretrain_source == 'vlm':
            policy = HyVLA(hy_config)
            vlm_path = cfg.model.vlm_model_path
            vlm_inner_config = _load_vlm_autoconfig(vlm_path)
            local_only = bool(os.path.isdir(vlm_path))
            policy.model.dual_tower.vlm = HunYuanVLMoTForConditionalGeneration.from_pretrained(
                vlm_path, config=vlm_inner_config, local_files_only=local_only,
            )
            # Replacing the inner VLM module discards the SpaceTime wrappers
            # installed by ``HyVLA.__init__``; re-arm them so the MEM
            # video-encoder path survives the swap.
            policy.enable_video_encoder_if_needed()
            # Re-apply ViT unfreeze (params + forward patch) lost by the vlm swap.
            policy.model.dual_tower.set_requires_grad()
            accelerator.print(
                f"[pretrain=vlm] load hy-vlm backbone from: {vlm_path}; "
                f"action expert randomly initialized"
            )
        elif pretrain_source == 'scratch':
            policy = HyVLA(hy_config)
            accelerator.print(
                "[pretrain=scratch] both VLM backbone and action expert "
                "randomly initialized"
            )
        else:
            raise ValueError(
                f"Unknown model.pretrain_source: {pretrain_source!r}, "
                f"expected one of 'scratch' / 'vlm' / 'vla'"
            )

    del policy.model.dual_tower.expert.model.embed_tokens
    del policy.model.dual_tower.expert.lm_head
    gc.collect()

    policy.to(weight_dtype)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    ############# Config Optimizer ############
    accelerator.print("Initialize Optimizer")
    opt_preset = hy_config.get_optimizer_preset()
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=opt_preset.lr,
        betas=tuple(opt_preset.betas),
        eps=opt_preset.eps,
        weight_decay=opt_preset.weight_decay,
    )

    hy_config.scheduler_warmup_steps *= accelerator.num_processes
    hy_config.scheduler_decay_steps *= accelerator.num_processes
    sch_preset = hy_config.get_scheduler_preset()
    # Cosine ramp-up to ``peak_lr`` over ``num_warmup_steps``, then
    # cosine decay to ``decay_lr`` over ``num_decay_steps``, then a
    # constant ``decay_lr`` floor for the rest of training. Implemented
    # via LambdaLR so we avoid pulling a third-party scheduler lib.
    _peak_lr = float(sch_preset.peak_lr)
    _decay_lr = float(sch_preset.decay_lr)
    _warmup = max(1, int(sch_preset.num_warmup_steps))
    _decay = max(1, int(sch_preset.num_decay_steps))
    _floor_ratio = _decay_lr / _peak_lr if _peak_lr > 0 else 0.0

    def _lr_lambda(step: int) -> float:
        if step < _warmup:
            return float(step) / float(_warmup)
        if step < _warmup + _decay:
            progress = (step - _warmup) / float(_decay)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return _floor_ratio + (1.0 - _floor_ratio) * cosine
        return _floor_ratio

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    ############# Config Dataset and Dataloader ############
    accelerator.print("Initialize Dataset and Dataloader")
    # The dataset is built with ``deterministic=True`` so ``__getitem__(idx)``
    # resolves to a fixed sample. Combined with accelerate's default
    # RandomSampler + per-rank sharding this gives an unbiased,
    # no-repeat / no-miss epoch with a fresh shuffle every iter() rebuild.
    train_dataset = VLADataset(config=cfg)
    data_collator = VLADataCollator()
    train_dataloader = hydra.utils.instantiate(
        cfg.dataloader, dataset=train_dataset, collate_fn=data_collator
    )

    ############# Preapare `accelerator` ############
    # Prepare everything with our `accelerator`.
    # will cast the parameters of model into mix-precision type in deepspeed mode
    policy, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, train_dataloader, lr_scheduler
    )

    if cfg.debug and accelerator.is_main_process:
        # Enumerate frozen / trainable layers so the user can spot-check
        # requires_grad after ``accelerator.prepare``.
        frozen_layers = [
            n for n, p in policy.named_parameters() if not p.requires_grad
        ]
        print(f"\n[debug] frozen layers ({len(frozen_layers)}):")
        for i, name in enumerate(frozen_layers):
            print(f"  {i+1}. {name}")
        print(
            f"[debug] total={num_total_params:,} "
            f"trainable={num_learnable_params:,} "
            f"frozen={num_total_params - num_learnable_params:,} "
            f"({num_learnable_params / num_total_params * 100:.2f}% trainable)\n"
        )

    if cfg.resume_ckpt:
        accelerator.print(f"Resuming optimizer and scheduler from checkpoint {cfg.resume_ckpt}")
        accelerator.load_state(os.path.join(cfg.resume_ckpt, "state", "training_state.pth"))
        accelerator.wait_for_everyone()

    total_batch_size = cfg.training.batch_size * accelerator.num_processes * cfg.training.grad_accumulation_steps

    accelerator.print("***** Running training *****")
    accelerator.print(f"  Total parameters: {num_total_params} M")
    accelerator.print(f"  Trainable parameters: {num_learnable_params} M")
    accelerator.print(f"  Num examples = {len(train_dataset)}")
    accelerator.print(f"  Instantaneous batch size per device = {cfg.training.batch_size}")
    accelerator.print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    accelerator.print(f"  Gradient Accumulation steps = {accelerator.gradient_accumulation_steps}")
    accelerator.print(f"  Total optimization steps = {cfg.training.max_training_steps}")

    train_loss = []
    val_loss = []
    angle_dist = []
    # Iter-based loop: step counter, checkpoint naming and wandb x-axis
    # are iter-based; ``infinite_loader`` keeps the dataloader epoch-based.
    # When ``cfg.dataset.deterministic=False`` the dataset draws samples
    # uniformly at random and a "cycle" is NOT a true epoch, so the wandb
    # metric is renamed to ``train/cycle`` in that case.
    is_deterministic = bool(getattr(cfg.dataset, "deterministic", False))
    epoch_metric_key = "epoch" if is_deterministic else "cycle"

    current_step = int(cfg.resume_ckpt.split("/")[-1]) + 1 if cfg.resume_ckpt else 0
    steps_per_epoch = len(train_dataloader)
    epoch_holder = [current_step // steps_per_epoch]

    if accelerator.is_main_process:
        script_logger.info(
            "[loader] dataset_len=%d world=%d per_device_bs=%d ga=%d "
            "steps_per_epoch=%d start_step=%d start_epoch=%d "
            "deterministic=%s metric=train/%s",
            len(train_dataset),
            accelerator.num_processes,
            int(cfg.training.batch_size),
            accelerator.gradient_accumulation_steps,
            steps_per_epoch,
            current_step,
            epoch_holder[0],
            is_deterministic,
            epoch_metric_key,
        )
        if not is_deterministic:
            script_logger.warning(
                "[loader] dataset.deterministic=False: a `cycle` is NOT a "
                "true epoch (samples may repeat / be missed)."
            )

    def infinite_loader(loader, start_step, spe, eh):
        epoch = start_step // spe
        offset = start_step % spe
        eh[0] = epoch
        # Resumed partial epoch: only emit the remaining batches of the
        # current epoch from a fresh iterator (new shuffle).
        if offset != 0:
            remaining = spe - offset
            count = 0
            for batch in loader:
                yield batch
                count += 1
                if count >= remaining:
                    break
            epoch += 1
            eh[0] = epoch
        # Steady state: full-epoch passes.
        while True:
            if accelerator.is_main_process:
                script_logger.info("[loader] start epoch=%d", epoch)
            for batch in loader:
                yield batch
            epoch += 1
            eh[0] = epoch

    train_loader_iter = infinite_loader(
        train_dataloader, current_step, steps_per_epoch, epoch_holder,
    )

    _val_base_iter = iter(train_dataloader)
    def val_batch_iterator():
        nonlocal _val_base_iter
        try:
            return next(_val_base_iter)
        except StopIteration:
            _val_base_iter = iter(train_dataloader)
            return next(_val_base_iter)

    progress_bar = tqdm(
        range(current_step, cfg.training.max_training_steps), ncols=100, disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Training")
    progress_bar.update(current_step)

    for n_iter in range(current_step, cfg.training.max_training_steps):
        policy.train()
        with accelerator.accumulate(policy):
            batch = next(train_loader_iter)

            for key in batch.keys():
                if "image" in key:
                    batch[key] = batch[key].to(weight_dtype)
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(weight_dtype)

            output_dict = policy.forward(batch)
            loss = output_dict["loss"]
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                params_to_clip = policy.parameters()
                accelerator.clip_grad_norm_(params_to_clip, cfg.training.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_loss.append(loss.item())
            if (
                (n_iter + 1) % (cfg.training.train_frequency * cfg.training.grad_accumulation_steps) == 0
                and accelerator.is_main_process
                and not cfg.debug
            ):
                epoch_f = (n_iter + 1) / steps_per_epoch
                wandb.log({
                    "train/loss": np.mean(train_loss),
                    "train/iter": n_iter,
                    f"train/{epoch_metric_key}": epoch_holder[0],
                    f"train/{epoch_metric_key}_f": epoch_f,
                    "train/lr": optimizer.param_groups[0]["lr"],
                })
                train_loss = []

            progress_bar.update(1)
            progress_bar.set_postfix({"loss": np.mean(train_loss) if len(train_loss) > 0 else 0})

        # validation
        if (n_iter + 1) % (cfg.training.eval_frequency * cfg.training.grad_accumulation_steps) == 0:
            accelerator.wait_for_everyone()
            accelerator.print(f"\n Evaluate at n_iter {n_iter}.")
            eval_model = accelerator.unwrap_model(policy)
            eval_model.eval()

            per_chunk_dists = []
            per_dim_dists = []

            for i in tqdm(
                range(cfg.training.max_evaluation_steps),
                desc="Validation",
                ncols=100,
                disable=not accelerator.is_local_main_process,
            ):
                batch = val_batch_iterator()

                with torch.no_grad():
                    for key in batch.keys():
                        if "image" in key:
                            batch[key] = batch[key].to(weight_dtype)
                        if isinstance(batch[key], torch.Tensor):
                            batch[key] = batch[key].to(weight_dtype)

                    output_dict = eval_model(batch)
                    flow_loss = output_dict["loss"]

                    output = eval_model.forward_evaluate(batch)
                    gt_actions, pred_actions = output["gt"], output["pred"]

                    all_predictions, all_targets, all_loss = accelerator.gather_for_metrics(
                        (pred_actions, gt_actions, flow_loss)
                    )
                    l1_error = (all_predictions - all_targets).abs()
                    per_chunk = l1_error.mean(dim=(0, 2))
                    per_dim = l1_error.mean(dim=(0, 1))
                    per_chunk_dists.append(per_chunk.cpu())
                    per_dim_dists.append(per_dim.cpu())
                    dist = l1_error.mean()
                    all_loss = all_loss.mean()

                val_loss.append(all_loss.item())
                angle_dist.append(dist.item())
            eval_model.train()
            if accelerator.is_main_process:
                if not cfg.debug:
                    # Average per-chunk / per-dim dists across eval steps.
                    avg_per_chunk = torch.stack(per_chunk_dists).mean(dim=0)  # (n_chunks,)
                    avg_per_dim = torch.stack(per_dim_dists).mean(dim=0)      # (n_dims,)

                    val_epoch_f = (n_iter + 1) / steps_per_epoch
                    log_dict = {
                        "val/loss": np.mean(val_loss),
                        "val/angle dist": np.mean(angle_dist),
                        "val/iter": n_iter,
                        f"val/{epoch_metric_key}": epoch_holder[0],
                        f"val/{epoch_metric_key}_f": val_epoch_f,
                    }
                    
                    n_chunks = len(avg_per_chunk)
                    n_dims = len(avg_per_dim)
                    fig, (ax1, ax2) = plt.subplots(
                        1, 2, figsize=(max(14, n_chunks * 0.3 + n_dims * 0.15), 5),
                    )
                    # Per-chunk subplot.
                    ax1.bar(range(n_chunks), avg_per_chunk.detach().cpu().numpy(),
                            color='steelblue', alpha=0.8)
                    ax1.set_xlabel('Chunk index')
                    ax1.set_ylabel('Mean L1 distance')
                    ax1.set_title(f'Per-chunk distance (iter {n_iter})')
                    ax1.grid(axis='y', alpha=0.3)
                    # Per-dim subplot.
                    ax2.bar(range(n_dims), avg_per_dim.detach().cpu().numpy(),
                            color='coral', alpha=0.8)
                    ax2.set_xlabel('Dimension index')
                    ax2.set_ylabel('Mean L1 distance')
                    ax2.set_title(f'Per-dim distance (iter {n_iter})')
                    ax2.grid(axis='y', alpha=0.3)
                    fig.tight_layout()
                    plot_file = os.path.join(
                        cfg.ckpt_save_dir, f"val_dists_iter_{n_iter:06d}.png",
                    )
                    fig.savefig(plot_file, dpi=50)
                    plt.close(fig)
                    log_dict["val/distance_breakdown"] = wandb.Image(plot_file)

                    wandb.log(log_dict)
                    val_loss = []
                    angle_dist = []

            torch.cuda.empty_cache()
            accelerator.wait_for_everyone()

        # save checkpoint
        if (n_iter + 1) % (cfg.training.ckpt_frequency * cfg.training.grad_accumulation_steps) == 0 and not cfg.debug:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                ckpt_model = accelerator.unwrap_model(policy)
                # require copy or clone to avoid shared memory of tensors. https://github.com/huggingface/safetensors/issues/202
                model_to_save = copy.deepcopy(ckpt_model)
                model_to_save.save_pretrained(os.path.join(cfg.ckpt_save_dir, f"{n_iter}", "model"))
                del model_to_save

            # also save the state
            if cfg.save_training_state:
                # if you use deepspeed, recommend to save state via accelerator.save_state
                # and no need to use accelerator.is_main_process before saving state.
                # ref to https://github.com/huggingface/diffusers/issues/2606
                accelerator.save_state(os.path.join(cfg.ckpt_save_dir, f"{n_iter}", "state", "training_state.pth"))
                accelerator.print(f"Saved state checkpoint at epoch {n_iter}.")

            torch.cuda.empty_cache()
    accelerator.end_training()


if __name__ == "__main__":
    train()
