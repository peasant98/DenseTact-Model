import os
import os.path as osp
import cv2
import numpy as np
import argparse

import copy
import math
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from omegaconf import ListConfig

from dinov2.loss.dino_loss import DINOLoss
from dinov2.loss.ibot_patch_loss import iBOTPatchLoss
from dinov2.loss.koleo_loss import KoLeoLoss
from dinov2.utils.ema import update_moving_average
from dinov2.utils.logging import get_pylogger, img_logger
from dinov2.utils import patchify_image, patches_to_image

from xformers.ops import fmha


from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models
from torchvision import transforms
from process_data import FullDataset
from lightning.pytorch.loggers import TensorBoardLogger

import lightning as L
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor

from models import pretrain_dict
from util.loss_util import ssim
from util.scheduler_util import LinearWarmupCosineAnnealingLR

from configs import get_cfg_defaults

from tactile_ssl.loss.dino_loss import DINOLoss
from tactile_ssl.loss.ibot_patch_loss import iBOTPatchLoss
from tactile_ssl.loss.koleo_loss import KoLeoLoss


class LightningDTDinoV2Model(L.LightningModule):
    def __init__(self, 
                encoder: nn.Module,
                dino_head: partial,
                optim_cfg: partial,
                lr_scheduler_cfg: Optional[partial],
                wd_scheduler_cfg: Optional[partial],
                online_probes: Optional[List[nn.Module]] = None,
                online_probes_lrs: List[float] = [],
                local_mask_scale: Tuple[float, float] = (0.2, 0.8),
                global_mask_scale: Tuple[float, float] = (0.2, 0.8),
                num_global_masks: int = 1,
                num_local_masks: int = 4,
                min_keep_num_sensors: int = 4,
                allow_mask_overlap: bool = False,
                moving_average_decay: Union[float, Tuple[float, ...]] = 0.99,
                teacher_temp: Union[float, Tuple[float, ...]] = (0.04, 0.07),
                teacher_warmup_epochs: int = 10,
                use_momentum=True,
                centering: Literal['centering', 'sinkhorn_knopp'] = 'centering',
                ibot_separate_head: bool = False,
                koleo_weight: float= 0.1,
                log_freq_reconstruction: int = 1000,
        ):
        """ MAE Model for training on the DT dataset """
        super(LightningDTDinoV2Model, self).__init__()

        self.optim_partial = optim_cfg
        self.lr_scheduler_partial = lr_scheduler_cfg
        self.wd_scheduler_partial = wd_scheduler_cfg
        self.use_momentum = use_momentum
        self.global_mask_scale = global_mask_scale
        self.local_mask_scale = local_mask_scale
        self.num_global_masks = num_global_masks
        self.num_local_masks = num_local_masks
        self.min_keep = min_keep_num_sensors
        self.centering = centering
        self.allow_mask_overlap = allow_mask_overlap
        self.log_freq_img = log_freq_reconstruction
        self.ibot_separate_head = ibot_separate_head
        self.koleo_weight = koleo_weight

        self.generator = torch.Generator()
        self.step = -1


        # Encoders
        dino_head = partial(dino_head, in_dim=encoder.embed_dim)
        if self.ibot_separate_head:
            ibot_head = partial(dino_head, in_dim=encoder.embed_dim)

        
        self.student_encoder_dict, self.teacher_encoder_dict = dict(), dict()
        self.student_encoder_dict["backbone"] = encoder
        self.student_encoder_dict["dino_head"] = dino_head()
        if self.ibot_separate_head: 
            self.student_encoder_dict['ibot_head'] = ibot_head()
        self.student_encoder = nn.ModuleDict(self.student_encoder_dict)

        self.teacher_encoder_dict["backbone"] = copy.deepcopy(encoder)
        self.teacher_encoder_dict["dino_head"] = dino_head()
        if self.ibot_separate_head: 
            self.teacher_encoder_dict['ibot_head'] = ibot_head()
        self.teacher_encoder = nn.ModuleDict(self.teacher_encoder_dict)
        self.teacher_encoder.requires_grad_(False)

        out_dim = self.student_encoder_dict["dino_head"].last_layer.out_features
        self.dino_loss = DINOLoss(out_dim=out_dim)
        self.ibot_patch_loss = iBOTPatchLoss(patch_out_dim=out_dim)
        self.koleo_loss = KoLeoLoss()

        self.patch_size = encoder.patch_size
        self.img_size = encoder.img_size
        self.in_chans = encoder.in_chans

        self.momentum_scheduler = None
        if not isinstance(moving_average_decay, float):
            assert isinstance(moving_average_decay, list) or isinstance(
                moving_average_decay, ListConfig
            )
            assert len(moving_average_decay) == 2
            moving_average_decay = tuple(moving_average_decay)
        self.moving_average_decay = moving_average_decay


        self.teacher_temp_scheduler = None
        if not isinstance(teacher_temp, float):
            assert isinstance(teacher_temp, list) or isinstance(
                teacher_temp, ListConfig
            )
            assert len(teacher_temp) == 2
            teacher_temp = tuple(teacher_temp)
        self.teacher_temp = teacher_temp
        self.teacher_warmup_epochs = teacher_warmup_epochs
        self.val_reconstruction_error = []
    


    def log_on_batch_end(
        self, outputs, stage: Literal["train", "val"] = "train", trainer_instance=None
    ):
        if trainer_instance is not None:
            step = (
                trainer_instance.global_step
                if stage == "train"
                else trainer_instance.global_val_step
            )
            for k, v in outputs.items(): 
                trainer_instance.wandb.log(
                    {f"{stage}/{k}": v, f"global_{stage}_step": step}
            )
            if "ppl_loss" in outputs.keys():
                ppl_loss = outputs["ppl_loss"]
                trainer_instance.wandb.log(
                    {f"{stage}/ppl_loss": ppl_loss, f"global_{stage}_step": step}
                )

            trainer_instance.wandb.log(
                {
                    f"{stage}/teacher_temperature": self.current_teacher_temp,
                    f"global_{stage}_step": step,
                }
            )
    

    def on_train_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        assert self.teacher_encoder is not None, "target encoder has not been created"
        self.current_teacher_temp = (
            next(self.teacher_temp_scheduler)
            if self.teacher_temp_scheduler is not None
            else self.teacher_temp
        )
        if self.use_momentum:
            moving_average_decay = (
                next(self.momentum_scheduler)
                if self.momentum_scheduler is not None
                else self.moving_average_decay
            )
            with torch.no_grad():
                update_moving_average(
                    self.teacher_encoder,
                    self.student_encoder,
                    moving_average_decay,
                )
        self.log_on_batch_end(outputs, stage="train", trainer_instance=trainer_instance)


    def on_validation_batch_end(
        self, outputs: Dict, batch: Dict, batch_idx: int, trainer_instance=None
    ):
        self.log_on_batch_end(outputs, stage="val", trainer_instance=trainer_instance)
        # Plot online probe predictions
        step = trainer_instance.global_val_step
        if trainer_instance is not None and step is not None:
            trainer_instance.wandb.log(
                {
                    f"val/loss": outputs["loss"],
                    f"global_val_step": step,
                }
            )
            if (step % self.log_freq_img == 0) and "pred_img" in outputs.keys():
                Xpred = outputs["pred_img"]
                Xorg = outputs["gt_img"] if "gt_img" in outputs.keys() else None
                if Xorg is not None:
                    self.val_reconstruction_error.append(
                        torch.mean((Xpred - Xorg) ** 2, dim=[1, 2, 3])
                    )
                img_logger(
                    wandb=trainer_instance.wandb,
                    global_step=step,
                    predictions=Xpred,
                    X=Xorg,
                    label="val",
                )


    def on_validation_epoch_end(self, trainer_instance=None):
        if len(self.val_reconstruction_error) > 0:
            reconstruction_error = torch.cat(self.val_reconstruction_error, dim=0)
            root_mean_square_error = torch.sqrt(torch.mean(reconstruction_error, dim=0))
            print(f"RMSE: {root_mean_square_error}")
            trainer_instance.wandb.log({"val/rmse": root_mean_square_error})
            self.val_reconstruction_error = []


    def _sample_block_size(self, height, width, scale):
        _rand = torch.rand(1, generator=self.generator).item()
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(height * width * mask_scale)
        aspect_ratio = 1.0
        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h > height:
            h -= 1
        while w > width:
            w -= 1

        return (h, w)
    

    def _sample_block_mask(self, height, width, b_size, acceptable_regions=None):
        h, w = b_size

        def constrain_mask(mask, tries=0):
            """Helper to restrict given mask to a set of acceptable regions"""
            N = max(int(len(acceptable_regions) - tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]

        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # -- Sample block top-left corner
            top = torch.randint(0, height - h + 1, (1,), generator=self.generator)
            left = torch.randint(0, width - w + 1, (1,), generator=self.generator)
            mask = torch.zeros((height, width), dtype=torch.int32)
            mask_complement = torch.ones_like(mask)
            mask[top : top + h, left : left + w] = 1
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    log.warning(
                        f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"'
                    )
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones((height, width), dtype=torch.int32)
        mask_complement[top : top + h, left : left + w] = 0
        # --
        return mask, mask_complement


    def sample_masks(self, x):
        batch_size, _, image_height, image_width = x.shape
        height, width = (
            image_height // self.patch_size,
            image_width // self.patch_size,
        )

        local_maskblock_sizes = self._sample_block_size(
            height, width, self.local_mask_scale
        )
        global_maskblock_sizes = self._sample_block_size(
            height, width, self.global_mask_scale
        )

        collated_local_masks, collated_global_masks = [], []
        min_keep_local_patches, min_keep_global_patches = (
            height * width,
            height * width,
        )
        for _ in range(batch_size):
            masks_local, masks_complement = [], []
            for _ in range(self.num_local_masks):
                mask, mask_complement = self._sample_block_mask(
                    height, width, local_maskblock_sizes
                )
                masks_local.append(mask)
                masks_complement.append(mask_complement)
                min_keep_local_patches = min(min_keep_local_patches, len(mask))
            collated_local_masks.append(masks_local)

            acceptable_regions = masks_complement

            if self.allow_mask_overlap:
                acceptable_regions = None

            masks_encoder = []
            for _ in range(self.num_global_masks):
                mask, _ = self._sample_block_mask(
                    height, width, global_maskblock_sizes, acceptable_regions
                )
                masks_encoder.append(mask)
                min_keep_global_patches = min(min_keep_global_patches, len(mask))
            collated_global_masks.append(masks_encoder)

        collated_global_masks = [
            [cm[:min_keep_global_patches] for cm in masks]
            for masks in collated_global_masks
        ]
        collated_local_masks = [
            [cm[:min_keep_local_patches] for cm in masks]
            for masks in collated_local_masks
        ]
        local_masks = data.default_collate(collated_local_masks)
        global_masks = data.default_collate(collated_global_masks)

        for i in range(len(global_masks)):
            global_masks[i] = global_masks[i].to(x.device)
        for i in range(len(local_masks)):
            local_masks[i] = local_masks[i].to(x.device)

        return global_masks, local_masks


    def forward(
        self,
        x: torch.Tensor,
        global_masks: List[torch.Tensor],
        local_masks: List[torch.Tensor],
    ):
        assert (
            global_masks is not None and local_masks is not None
        ), "Masks are required for DINOModule during training"

        # TODO: @Akash Sharma - Raise to make sure context encoder implements taking masks as an argument

        student_global_dict = self.student_encoder_dict["backbone"].forward_features(
            x, global_masks
        )
        assert (
            "x_norm_regtokens" in student_global_dict.keys()
        ), "Dino requires backbone to contain 1 register token"

        student_global_cls_tokens = student_global_dict["x_norm_regtokens"]
        student_global_cls_tokens = einops.rearrange(student_global_cls_tokens, "(p b) 1 c -> b p c", p=len(global_masks))
        
        student_local_dict = self.student_encoder_dict["backbone"].forward_features(
            x, local_masks
        )
        student_local_cls_tokens = student_local_dict["x_norm_regtokens"]
        student_local_cls_tokens = einops.rearrange(
            student_local_cls_tokens, "(p b) 1 c -> b p c", p=len(local_masks)
        )
        # NOTE: we are not masking patch tokens randomly as done in iBOT and dinov2
        student_global_patch_tokens = student_global_dict["x_norm_patchtokens"]


        _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list([student_global_cls_tokens, student_local_cls_tokens])
        after_head_list = _attn_bias.split(self.student_encoder_dict['dino_head'](cat_inputs))
        student_global_cls_tokens_after_head, student_local_cls_tokens_after_head = after_head_list[0].squeeze(0), after_head_list[1].squeeze(0)
        student_cls_tokens_after_head = torch.cat([student_global_cls_tokens_after_head, student_local_cls_tokens_after_head], dim=1)
        student_cls_tokens_after_head = einops.rearrange(student_cls_tokens_after_head, 'b p c -> p b 1 c')

        if self.ibot_separate_head: 
            student_patch_tokens_after_head = self.student_encoder_dict['ibot_head'](student_global_patch_tokens)
        else: 
            student_patch_tokens_after_head = self.student_encoder_dict['dino_head'](student_global_patch_tokens)


        with torch.no_grad():
            teacher_global_dict = self.teacher_encoder_dict[
                "backbone"
            ].forward_features(x, global_masks)
            teacher_global_cls_tokens = teacher_global_dict["x_norm_regtokens"]

            teacher_global_cls_tokens = teacher_global_cls_tokens.chunk(self.num_global_masks)

            # watch out: these are chunked and cat'd in reverse so A is matched to B in the global crops dino loss
            teacher_global_cls_tokens = torch.cat((teacher_global_cls_tokens[1], teacher_global_cls_tokens[0]))
            teacher_global_patch_tokens = teacher_global_dict["x_norm_patchtokens"]

            teacher_cls_tokens_after_head = self.teacher_encoder_dict["dino_head"](
                teacher_global_cls_tokens
            )
            if self.ibot_separate_head: 
                teacher_patch_tokens_after_head = self.teacher_encoder_dict['ibot_head'](teacher_global_patch_tokens)
            else: 
                teacher_patch_tokens_after_head = self.teacher_encoder_dict["dino_head"](
                    teacher_global_patch_tokens
                )
            if self.centering == 'centering':
                teacher_dino_softmaxed_centered_list = (
                    self.dino_loss.softmax_center_teacher(
                        teacher_cls_tokens_after_head,
                        teacher_temp=self.current_teacher_temp,
                    ).view(
                        self.num_global_masks, -1, *teacher_cls_tokens_after_head.shape[1:]
                    )
                )
                teacher_ibot_softmaxed_centered = (
                    self.ibot_patch_loss.softmax_center_teacher(
                        teacher_patch_tokens_after_head.unsqueeze(0),
                        teacher_temp=self.current_teacher_temp,
                    )
                )
                teacher_ibot_softmaxed_centered = teacher_ibot_softmaxed_centered.squeeze()
                self.dino_loss.update_center(teacher_cls_tokens_after_head)
                self.ibot_patch_loss.update_center(teacher_patch_tokens_after_head)
                teacher_ibot_softmaxed_centered = einops.rearrange(teacher_ibot_softmaxed_centered, '(p b) k c -> p b k c', p=len(global_masks), b=x.shape[0])
            elif self.centering == 'sinkhorn_knopp': 
                n_masked_patches_tokens = teacher_patch_tokens_after_head.shape[1]
                teacher_patch_tokens_after_head = einops.rearrange(teacher_patch_tokens_after_head, 'b k c -> (b k) c')
                teacher_dino_softmaxed_centered_list = (
                    self.dino_loss.sinkhorn_knopp_teacher(
                        teacher_cls_tokens_after_head.squeeze(),
                        teacher_temp=self.current_teacher_temp
                    ).view(self.num_global_masks, -1, *teacher_cls_tokens_after_head.shape[1:])
                )
                teacher_ibot_softmaxed_centered = (
                    self.ibot_patch_loss.sinkhorn_knopp_teacher(
                        teacher_patch_tokens_after_head,
                        teacher_temp= self.current_teacher_temp,
                        n_masked_patches_tensor=torch.tensor(n_masked_patches_tokens, dtype=int, device=teacher_patch_tokens_after_head.device)
                    )
                )
                teacher_ibot_softmaxed_centered = teacher_ibot_softmaxed_centered.squeeze()
                teacher_ibot_softmaxed_centered = einops.rearrange(teacher_ibot_softmaxed_centered, '(p b k) c -> p b k c', p=len(global_masks), b=x.shape[0])
            else: 
                raise NotImplementedError

        student_patch_tokens_after_head = einops.rearrange(student_patch_tokens_after_head, '(p b) k c -> p b k c', p=len(global_masks))

        n_local_crops_loss_terms = max(self.num_local_masks * self.num_global_masks, 1)
        n_global_crops_loss_terms = (self.num_global_masks - 1) * self.num_global_masks

        dino_loss = self.dino_loss(
            list(student_cls_tokens_after_head),
            list(teacher_dino_softmaxed_centered_list),
        ) / (n_local_crops_loss_terms + n_global_crops_loss_terms)

        # student_global_cls_tokens b x p x c
        koleo_loss = self.koleo_weight * sum(
            self.koleo_loss(p.squeeze()) for p in student_global_cls_tokens.chunk(2, dim=1)
        )  # we don't apply koleo loss between cls tokens of a same image

        ibot_loss_scale = 1.0 / self.num_global_masks
        patch_loss = ibot_loss_scale * self.ibot_patch_loss(
            list(student_patch_tokens_after_head), list(teacher_ibot_softmaxed_centered)
        )
        return dino_loss, patch_loss, koleo_loss



    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:

        self.step = self.step + 1
        self.generator.manual_seed(self.step)

        x = batch["image"]
        global_masks, local_masks = self.sample_masks(x)
        dino_loss, ibot_loss, koleo_loss = self.forward(x, global_masks, local_masks)
        loss = dino_loss + ibot_loss + koleo_loss
        output = {
            "dino_loss": dino_loss.item(),
            "ibot_loss": ibot_loss.item(),
            "koleo_loss": koleo_loss.item()
        }

        output["loss"] = loss
        return output
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        return self.training_step(batch, batch_idx)
    
    def configure_optimizers(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, num_iterations_per_epoch, num_epochs
    ) -> Tuple[torch.optim.Optimizer, Optional[Dict], Optional[Dict]]:
        param_dict = {
            pn: p
            for pn, p in self.named_parameters()
            if not pn.startswith("online_probes")
        }
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params},
            {"params": nodecay_params, "WD_exclude": True, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        for probe, lr in zip(self.online_probes, self.online_probes_lrs):
            trainable_probe_params = {
                pn: p for pn, p in probe.named_parameters() if p.requires_grad
            }
            optim_groups.append({"params": trainable_probe_params.values(), "lr": lr})

        log.info(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        log.info(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )

        optimizer = self.optim_partial(optim_groups)
        if self.lr_scheduler_partial is None:
            return optimizer, None, None

        lr_scheduler = self.lr_scheduler_partial(
            optimizer=optimizer,
            T_max=int(num_epochs * num_iterations_per_epoch),
            steps_per_epoch=num_iterations_per_epoch,
        )
        if isinstance(self.moving_average_decay, tuple):
            self.momentum_scheduler = (
                self.moving_average_decay[0]
                + i
                * (self.moving_average_decay[1] - self.moving_average_decay[0])
                / (num_epochs * num_iterations_per_epoch)
                for i in range(int(num_epochs * num_iterations_per_epoch) + 1)
            )
        self.current_teacher_temp = self.teacher_temp
        if isinstance(self.teacher_temp, tuple):
            self.teacher_temp_scheduler = self.teacher_temp_schedule(
                num_epochs, num_iterations_per_epoch
            )

            self.current_teacher_temp = self.teacher_temp[0]

        if self.wd_scheduler_partial is None:
            return (
                optimizer,
                {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "monitor": None,
                },
                None,
            )

        wd_scheduler = self.wd_scheduler_partial(
            optimizer,
            T_max=int(num_epochs * num_iterations_per_epoch),
        )
        return (
            optimizer,
            {"scheduler": lr_scheduler, "interval": "step", "monitor": None},
            {"wd_scheduler": wd_scheduler, "interval": "step", "frequency": 1},
        )

    def teacher_temp_schedule(self, num_epochs, num_iterations_per_epoch):
        assert isinstance(
            self.teacher_temp, tuple
        ), "Teacher temp must be a tuple if this function is called"
        for i in range(int(num_epochs * num_iterations_per_epoch) + 1):
            teacher_temp = None
            if i > (self.teacher_warmup_epochs * num_iterations_per_epoch):
                teacher_temp = self.teacher_temp[1]
            else:
                teacher_temp = self.teacher_temp[0] + i * (
                    self.teacher_temp[1] - self.teacher_temp[0]
                ) / (self.teacher_warmup_epochs * num_iterations_per_epoch)
            yield teacher_temp


if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('--dataset_ratio', type=float, default=1.0)
    arg.add_argument('--dataset_dir', type=str, default="/arm/u/maestro/Desktop/DenseTact-Model/es4t/es4t/dataset_local/")
    arg.add_argument('--epochs', type=int, default=200)
    arg.add_argument('--config', type=str, default="configs/QHiera_disp.yaml")
    arg.add_argument('--gpus', type=int, default=4)
    
    arg.add_argument('--model', type=str, default="mae_vit_base_patch16", help="Model Architecture, choose either hiera or vit")
    arg.add_argument('--batch_size', type=int, default=64)
    arg.add_argument('--num_workers', type=int, default=24)
    arg.add_argument('--mask_ratio', type=float, default=0.75)
    arg.add_argument('--exp_name', type=str, default="DT_Ultra_vit_mae")
    arg.add_argument('--ckpt_path', type=str, default=None)
    arg.add_argument('--real_world', action='store_true')
    
    opt = arg.parse_args()
    
    cfg = get_cfg_defaults()
    cfg.merge_from_file(opt.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256), antialias=True),
    ])
    

    extra_samples_dirs = ['/arm/u/maestro/Desktop/DenseTact-Model/es1t/dataset_local/', 
                          '/arm/u/maestro/Desktop/DenseTact-Model/es2t/es2t/dataset_local/',
                          '/arm/u/maestro/Desktop/DenseTact-Model/es3t/es3t/dataset_local/']
    dataset = FullDataset(cfg, transform=transform, samples_dir=opt.dataset_dir, 
                          extra_samples_dirs=extra_samples_dirs,
                          is_real_world=opt.real_world, is_mae=True)
    print("Dataset total samples: {}".format(len(dataset)))

    full_dataset_length = len(dataset)
    
    # take only 10 percent of dataset for train and test
    dataset_length = int(opt.dataset_ratio * full_dataset_length)
    train_size = int(0.95 * dataset_length)
    test_size = dataset_length - train_size
    
    train_dataset, test_dataset, _ = random_split(dataset, [train_size, test_size, full_dataset_length - dataset_length])
    print(f"Train dataset length: {len(train_dataset)}, Test dataset length: {len(test_dataset)}")
    
    X, y = dataset[1000]
    deform_color = X[:3, :, :].detach().cpu().numpy()
    undeform_color = X[3:6, :, :].detach().cpu().numpy()
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(deform_color.transpose(1, 2, 0))
    ax[0].set_title("Deformed Image")
    ax[1].imshow(undeform_color.transpose(1, 2, 0))
    ax[1].set_title("Undeformed Image")
    plt.savefig("sample_image.png")
    plt.close()
    
    dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=12)
    
    calibration_model = LightningDTModel(
        model_name=opt.model, 
        num_epochs=opt.epochs,
        total_steps=opt.epochs * len(train_dataset),
        mask_ratio=opt.mask_ratio,
        learning_rate=8e-4
    )

    logger = TensorBoardLogger(osp.join(opt.exp_name, 'tb_logs/'), name="lightning_logs")
    
    # create callback to save checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor='train/loss_epoch',
        dirpath=osp.join(opt.exp_name, 'checkpoints/'),
        filename='dt_model-{epoch:02d}-{train_loss:.2f}',
        save_top_k=1,
        save_last=True,
        mode='min',
    )

    # log learning rate
    lr_monitor = LearningRateMonitor(logging_interval='step')

    strategy = "ddp" if opt.gpus > 1 else "auto"

    # add callbacks
    callbacks = [checkpoint_callback, lr_monitor]
    trainer = L.Trainer(max_epochs=opt.epochs, callbacks=callbacks, logger=logger,
                        accelerator="gpu", devices=opt.gpus, strategy=strategy)
    
    trainer.fit(model=calibration_model, train_dataloaders=dataloader, ckpt_path=opt.ckpt_path)