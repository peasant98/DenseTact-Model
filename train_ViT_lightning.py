import os
import os.path as osp
import cv2
import torch.distributed as dist
import shutil
import numpy as np
import argparse
from typing import Dict
from weakref import proxy
from copy import deepcopy
import matplotlib as mpl
import wandb
import datetime

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models
from torchvision import transforms
from process_data import FullDataset, FEAT_CHANNEL
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

import lightning as L
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor

from configs import get_cfg_defaults

from models import build_model, replace_LoRA, MonkeyPatchLoRALinear, HieraDPT
from util.loss_util import ssim
from util.scheduler_util import LinearWarmupCosineAnnealingLR

def apply_colormap(depth_map, cmap='viridis'):  
    depth_map_normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map) + 0.00001)  # Normalize to [0, 1]
    colormap = plt.get_cmap(cmap)
    depth_map_colored = colormap(depth_map_normalized)[:, :, :, :3]  # Get RGB values, ignore alpha channel
    depth_map_colored = (depth_map_colored * 255).astype(np.uint8)  # Convert to 8-bit image
    return depth_map_colored

class LightningDTModel(L.LightningModule):
    OUTPUT_ORDER = ["depth", "stress_x", "stress_y", "stress_z"]
    
    def __init__(self, cfg):
        """ MAE Model for training on the DT dataset """
        super(LightningDTModel, self).__init__()
        self.cfg = cfg

        self.model = build_model(cfg)

        if len(self.cfg.model.pretrained_model) > 0:
            print("Load pretrained model")
            self.model.load_from_pretrained_model(self.cfg.model.pretrained_model)   
            self.model.freeze_encoder()

            # use LoRA finetune
            if cfg.model.LoRA:
                self.model.replace_LoRA(self.cfg.model.LoRA_rank, self.cfg.model.LoRA_scale)

        # build loss
        if cfg.model.loss == "L1":
            self.criterion = nn.L1Loss()
        elif cfg.model.loss == "L2":
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError("Loss not implemented {}".format(cfg.model.loss))
        
        # check channels
        total_output_channels = sum(cfg.model.out_chans)
        dataset_output_type = cfg.dataset.output_type
        expected_output_channels = sum([FEAT_CHANNEL[t] for t in dataset_output_type])
        assert total_output_channels == expected_output_channels, \
                    f"Output channels mismatch {total_output_channels} != {expected_output_channels}"

        
        self.output_names = []
        if cfg.dataset.contiguous_on_direction:
            # depth always comes first
            if "depth" in dataset_output_type:
                self.output_names.append("depth")

            # add the other channels
            for d in ["x", "y", "z"]:
                for t in dataset_output_type:
                    if t == "depth":
                        continue
                    else:
                        # extend the three channels
                        self.output_names.append(f"{t}_{d}")
        
        else:
            for t in dataset_output_type:
                if t == "depth":
                    self.output_names.append("depth")
                else:
                    for d in ["x", "y", "z"]:
                        self.output_names.append(f"{t}_{d}")

        # MSE for validation
        self.mse = nn.MSELoss()
        
        self.num_epochs = cfg.epochs
        self.total_steps = cfg.total_steps
        self.validation_stats = {}        
            
    def training_step(self, batch, batch_idx):
        # X - (N, C1, H, W); Y - (N, C2, H, W)
        X, Y = batch 
        
        # # force Negative samples to be 0
        # Y = torch.where(torch.abs(Y) < self.cfg.metric.PN_thresh, torch.zeros_like(Y), Y)
        
        N, C, H, W = X.shape 
        
        total_loss = 0. 
        pred = self.model(X) 
        predict_dict = self._process_prediction(pred)
        label_dict = self._process_prediction(Y)

        for name in self.cfg.dataset.output_type:
            if name == 'depth':
                loss = self.criterion(predict_dict[name], label_dict[name] * self.cfg.scale)

            else:
                channels = [f"{name}_{d}" for d in ["x", "y", "z"]]
                pred_vec = torch.stack([predict_dict[c] for c in channels], dim=1)
                label_vec = torch.stack([label_dict[c] for c in channels], dim=1)

                loss = self.criterion(pred_vec, label_vec * self.cfg.scale)

            if 'disp' in name:
                weight = self.cfg.loss.disp_weight
            elif 'stress' in name:
                weight = self.cfg.loss.stress_weight
            elif 'depth' in name:
                weight = self.cfg.loss.depth_weight

            total_loss += weight * loss
            self.log(f'train/{name}/loss', weight * loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        # outputs = torch.cat(outputs, dim=1)
        self.log('train/loss', total_loss , on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return {"loss": total_loss}

    def reset_validation_stats(self):
        """ resetting stats for val/test """
        self.validation_stats = {
            "mse": [],
        }
        
        for name in self.cfg.dataset.output_type:
            self.validation_stats[name] = dict(NegativeNum=0, 
                                                FNNum=0, PosNum=0, num_images = 0, mse = 0,
                                                error_cureve=np.zeros(11), abs_error_curve=np.zeros(11))
    
    def on_validation_epoch_start(self):
        self.reset_validation_stats()

    @torch.no_grad()
    def compute_metric(self, prediction:torch.Tensor, 
                       label:torch.Tensor,
                       PN_thresh:float=1e-6,
                       TF_rel_error_rate:float=0.1,
                       TF_abs_error_thresh:float=1e-3):
        """ Compute the prediction metric 
        
        Args:
            prediction: (N, C, H, W)
            label: (N, C, H, W)
            
            PN_thresh: threshold for negative samples
            TF_rel_error_rate: threshold for relative error
            TF_abs_error_thresh: threshold for absolute error

        Returns:
            dict: dictionary of metrics
                NegativeNum: Number of negative samples
                FNNum: Number of false negative samples
                PosNum: Number of positive samples
                error_curve: Error curve (11)
                    [<0.1, <0.2, ..., <0.9, <1.0, >1.0]
                abs_error_curve: Absolute error curve
                    [<0.1mm, <0.2mm, ..., <0.9mm, <1.0mm, >1.0mm]
        """
        prediction_flatten = prediction.flatten()
        label_flatten = label.flatten()

        num_bins = self.cfg.metric.num_bins
        interval = TF_rel_error_rate / num_bins
        threshold_ratio = [k * interval for k in range(1, num_bins + 1)]
        # normally set TF_abs_error_thresh to 1e-3 (=1mm)
        abs_interval = TF_abs_error_thresh / num_bins
        threshold_abs = [k * abs_interval for k in range(1, num_bins + 1)]

        # Here, negative samples are the pixels with value than 1e-6
        # positive samples are the pixels with value greater than 1e-6
        eps = PN_thresh

        # compute the negative pixels
        Neg_mask = torch.abs(label_flatten) < eps
        Pos_mask = torch.abs(label_flatten) >= eps

        # compute the false negative
        NegativeNum = torch.sum(Neg_mask).item()
        FNNum = torch.sum(Neg_mask * (torch.abs(prediction_flatten) >= eps)).item()

        # compute the error curve
        gt_pos_items = label_flatten[Pos_mask]
        pred_pos_items = prediction_flatten[Pos_mask]

        PosNum = torch.sum(Pos_mask).item()
        error_curve = []
        abs_error_curve = []
        if PosNum > 0:
            # compute the relative error
            rel_error = torch.abs(gt_pos_items - pred_pos_items) / torch.abs(gt_pos_items)

            # compute the threshold ratio
            for ratio in threshold_ratio:
                error_curve.append(torch.sum(rel_error < ratio).item())

            error_curve.append(torch.sum(rel_error > TF_rel_error_rate).item())
            error_curve = np.array(error_curve)

            # compute abs error
            abs_error = torch.abs(gt_pos_items - pred_pos_items)

            for thresh in threshold_abs:
                abs_error_curve.append(torch.sum(abs_error < thresh).item())

            abs_error_curve.append(torch.sum(abs_error > TF_abs_error_thresh).item())
            abs_error_curve = np.array(abs_error_curve)
        
        return dict(NegativeNum=NegativeNum, FNNum=FNNum, num_images=prediction.shape[0], mse = self.mse(prediction, label).item(),
                        PosNum=PosNum, error_cureve=error_curve, abs_error_curve=abs_error_curve)
    
    @torch.no_grad()
    def compute_metric_vector(self, prediction:torch.Tensor, 
                       label:torch.Tensor,
                       PN_thresh:float=1e-6,
                       TF_rel_error_rate:float=0.1,
                       TF_abs_error_thresh:float=1e-3):
        """ Compute the prediction metric 
        
        Args:
            prediction: (N, 3, H, W)
            label: (N, 3, H, W)
            
            PN_thresh: threshold for negative samples
            TF_rel_error_rate: threshold for relative error
            TF_abs_error_thresh: threshold for absolute error

        Returns:
            dict: dictionary of metrics
                NegativeNum: Number of negative samples
                FNNum: Number of false negative samples
                PosNum: Number of positive samples
                error_curve: Error curve (11)
                    [<0.1, <0.2, ..., <0.9, <1.0, >1.0]
                abs_error_curve: Absolute error curve
                    [<0.1mm, <0.2mm, ..., <0.9mm, <1.0mm, >1.0mm]
        """
        prediction = prediction.permute(0, 2, 3, 1).reshape(-1, 3) # (N, 3)
        label = label.permute(0, 2, 3, 1).reshape(-1, 3) # (N, 3)

        num_bins = self.cfg.metric.num_bins
        interval = TF_rel_error_rate / num_bins
        threshold_ratio = [k * interval for k in range(1, num_bins + 1)]
        # normally set TF_abs_error_thresh to 1e-3 (=1mm)
        abs_interval = TF_abs_error_thresh / num_bins
        threshold_abs = [k * abs_interval for k in range(1, num_bins + 1)]

        # Here, negative samples are the pixels with value than 1e-6
        # positive samples are the pixels with value greater than 1e-6
        eps = PN_thresh

        # compute the negative pixels
        prediction_length = torch.norm(prediction, dim=1)
        label_length = torch.norm(label, dim=1)

        Neg_mask = label_length < eps
        Pos_mask = label_length >= eps

        # compute the false negative
        NegativeNum = torch.sum(Neg_mask).item()
        FNNum = torch.sum(Neg_mask * (prediction_length >= eps)).item()

        # compute the error curve
        gt_pos_items = prediction[Pos_mask]
        gt_pos_length = label_length[Pos_mask]

        pred_pos_items = label[Pos_mask]
        error_term = torch.norm(gt_pos_items - pred_pos_items, dim=1)

        PosNum = torch.sum(Pos_mask).item()
        error_curve = []
        abs_error_curve = []
        if PosNum > 0:
            # compute the relative error
            rel_error = error_term / gt_pos_length

            # compute the threshold ratio
            for ratio in threshold_ratio:
                error_curve.append(torch.sum(rel_error < ratio).item())

            error_curve.append(torch.sum(rel_error > TF_rel_error_rate).item())
            error_curve = np.array(error_curve)

            # compute abs error
            abs_error = error_term

            for thresh in threshold_abs:
                abs_error_curve.append(torch.sum(abs_error < thresh).item())

            abs_error_curve.append(torch.sum(abs_error > TF_abs_error_thresh).item())
            abs_error_curve = np.array(abs_error_curve)
        
        return dict(NegativeNum=NegativeNum, FNNum=FNNum, num_images=prediction.shape[0], mse = self.mse(prediction, label).item(),
                        PosNum=PosNum, error_cureve=error_curve, abs_error_curve=abs_error_curve)
    
    def update_metric(self, cumulative_metric_dict:dict, current_step_dict:dict):
        for key in ['NegativeNum', 'FNNum', 'PosNum','error_cureve', 'abs_error_curve']:
            cumulative_metric_dict[key] += current_step_dict[key] 
        
        # update mse
        fraction = cumulative_metric_dict["num_images"] / (cumulative_metric_dict["num_images"] + current_step_dict["num_images"])
        cumulative_metric_dict["mse"] = fraction * cumulative_metric_dict["mse"] + (1 - fraction) * current_step_dict["mse"]
        cumulative_metric_dict["num_images"] += current_step_dict["num_images"]
            
    def _process_prediction(self, prediction:torch.Tensor) -> dict:
        """ Process the prediction and label for computing the metric """
        output = dict()
        for idx in range(len(self.output_names)):
            output[self.output_names[idx]] = prediction[:, idx, :, :]
        
        return output

    def _run_val_test_step(self, batch, batch_idx, name="val"):
        # X - (N, C1, H, W); Y - (N, C2, H, W)
        X, Y = batch 
        N, C, H, W = X.shape 
        
        pred = self.model(X) / self.cfg.scale
        mse = self.mse(pred, Y)
        
        # visualize first y and pred
        if batch_idx % 10 == 0 and name == "test":
            for i in range(N):
                gt_depth = Y[i].detach().clone()
                gt_depth = gt_depth.cpu().numpy()
                
                pred_depth = pred[i].detach().clone()
                pred_depth = pred_depth.cpu().numpy()
             
                # plot both
                fig, axes = plt.subplots(2, len(self.output_names), figsize=(20, 10))

                pred_ax = axes[0]
                for name, ax, p in zip(self.output_names, pred_ax, pred_depth):
                    ax.imshow(p)
                    ax.set_title(name)
                
                gt_ax = axes[1]
                for name, ax, g in zip(self.output_names, gt_ax, gt_depth):
                    ax.imshow(g)
                    ax.set_title(name)

                fig.savefig(osp.join(self.logger.save_dir, f"{name}_prediction_{batch_idx}_{i}.png"))
                plt.close(fig)

        # outputs = torch.cat(outputs, dim=1)
        self.log(f'{name}/mse', mse, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.validation_stats["mse"].append(mse.item())

        predict_dict, label_dict = self._process_prediction(pred), self._process_prediction(Y)

        # compute force metric
        for name in self.cfg.dataset.output_type:
            if name == 'depth':
                metric_dict = self.compute_metric(predict_dict[name], label_dict[name],
                                                  PN_thresh=self.cfg.metric.PN_thresh,
                                                  TF_rel_error_rate=self.cfg.metric.TF_rel_error_rate,
                                                  TF_abs_error_thresh=self.cfg.metric.TF_abs_error_thresh)
            else:
                channels = [f"{name}_{d}" for d in ["x", "y", "z"]]
                pred_vec = torch.stack([predict_dict[c] for c in channels], dim=1)
                label_vec = torch.stack([label_dict[c] for c in channels], dim=1)

                # compute vector metric
                metric_dict = self.compute_metric_vector(pred_vec, label_vec,
                                                        PN_thresh=self.cfg.metric.PN_thresh,
                                                        TF_rel_error_rate=self.cfg.metric.TF_rel_error_rate,
                                                        TF_abs_error_thresh=self.cfg.metric.TF_abs_error_thresh)
                
            self.update_metric(self.validation_stats[name], metric_dict)

    def validation_step(self, batch, batch_idx):
        self._run_val_test_step(batch, batch_idx, name="val")
    
    def summarize_metric(self, name="val"):
        avg_mse = np.array(self.validation_stats["mse"])
        avg_mse = np.mean(avg_mse)
        self.log(f'{name}/avg_mse', avg_mse, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        psnr = 10 * np.log10(1.0 / avg_mse)
        self.log(f'{name}_psnr', psnr, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        if dist.is_initialized():
            # summarize the results from multiple GPUs
            output = [None for _ in range(dist.get_world_size())]

            # gather the results from all GPUs
            dist.all_gather_object(output, self.validation_stats)

            if self.global_rank == 0:
                # sum to the validation stats
                for stats in output[1:]:
                    
                    for key, item in stats.items():
                        if isinstance(item, dict):
                            self.update_metric(self.validation_stats[key], item)
                        else:
                            self.validation_stats[key] += stats[key]
        
        # compute the metric in the rank = 0
        depth_AUC_abs = 0.
        if self.global_rank == 0:
            for item in self.cfg.dataset.output_type:
                # log mse
                mse = self.validation_stats[item]["mse"]
                self.log(f'{name}/{item}/mse', mse, on_step=False, on_epoch=True, prog_bar=True, logger=True)

                # compute FNR
                FNR = self.validation_stats[item]["FNNum"] / self.validation_stats[item]["NegativeNum"]

                # here FPR is defined as the rel error > 100 %
                FPR_rel = self.validation_stats[item]["error_cureve"][-1] / self.validation_stats[item]["PosNum"]

                # here FPR is defined as the abs error > TF_abs_error_thresh
                FPR_abs = self.validation_stats[item]["abs_error_curve"][-1] / self.validation_stats[item]["PosNum"]

                # log
                self.log(f'{name}/{item}/FNR', FNR, on_step=False, on_epoch=True, prog_bar=False, logger=True)
                self.log(f'{name}/{item}/FPR_rel', FPR_rel, on_step=False, on_epoch=True, prog_bar=False, logger=True)
                self.log(f'{name}/{item}/FPR_abs', FPR_abs, on_step=False, on_epoch=True, prog_bar=False, logger=True)

                # draw the curve
                error_curve = self.validation_stats[item]["error_cureve"][:-1] / self.validation_stats[item]["PosNum"]
                # compute the Area under curve
                AUC_rel = np.mean(error_curve)
                self.log(f'{name}/{item}/AUC_rel', AUC_rel, on_step=False, on_epoch=True, prog_bar=False, logger=True)
                
                abs_error_curve = self.validation_stats[item]["abs_error_curve"][:-1] / self.validation_stats[item]["PosNum"]
                AUC_abs = np.mean(abs_error_curve)
                self.log(f'{name}/{item}/AUC_abs', AUC_abs, on_step=False, on_epoch=True, prog_bar=False, logger=True)
                
                # select depth as final metric
                if item == "depth":
                    depth_AUC_abs = AUC_abs
                
                # for ckpt name
                # self.log('AUC', AUC_abs * 100, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                
                # disable plotting when training in DDP
                if not dist.is_initialized():

                    num_bins = self.cfg.metric.num_bins
                    interval = self.cfg.metric.TF_rel_error_rate / num_bins
                    threshold_ratio = [k * interval for k in range(1, num_bins + 1)]

                    fig = plt.figure()
                    ax = fig.add_subplot(111)

                    # set x-y limit
                    ax.set_xlim([0, threshold_ratio[-1]])
                    ax.set_ylim([0, 1])
                    # draw the plot of the error curve
                    ax.plot(threshold_ratio, error_curve, label="Error Curve")
                    ax.set_xlabel("Threshold")
                    ax.set_ylabel("Error Rate")
                    if isinstance(self.logger, TensorBoardLogger):
                        self.logger.experiment.add_figure(f"{name}/{item}/Rel Error Curve", fig, self.global_step)
                    # else:
                    #     fig.savefig(osp.join(self.logger.save_dir, f"{name}_{item}_error_curve_{self.global_step}.png"))
                    # fig.clf()

                    # draw the plot of the abs error curve
                    abs_interval = self.cfg.metric.TF_abs_error_thresh / num_bins
                    threshold_abs = [k * abs_interval for k in range(1, num_bins + 1)]
                    fig = plt.figure()
                    ax = fig.add_subplot(111)

                    # set x-y limit
                    ax.set_xlim([0, threshold_abs[-1]])
                    ax.set_ylim([0, 1])
                    # draw the plot of the error curve
                    ax.plot(threshold_abs, abs_error_curve, label="Abs Error Curve")
                    ax.set_xlabel("Threshold")
                    ax.set_ylabel("Error Rate")
                    if isinstance(self.logger, TensorBoardLogger):
                        self.logger.experiment.add_figure(f"{name}/{item}/Abs Error Curve", fig, self.global_step)
                    # else:
                    #     fig.savefig(osp.join(self.logger.save_dir, f"{name}_{item}_abs_error_curve_{self.global_step}.png"))

        sync_data = depth_AUC_abs * 100 if self.global_rank == 0 else 0
        if dist.is_initialized():
            # boardcast to all GPUS
            output = [None for _ in range(dist.get_world_size())]

            # gather the results from all GPUs
            dist.all_gather_object(output, sync_data)
            
            # debug purpose
            print(f"collected AUC {output} rank {self.global_rank}") 

            AUC = sum(output)
            self.log('AUC', AUC, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        else:
            # normal case
            self.log('AUC', sync_data, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_end(self):
        self.summarize_metric(name="val")

    def on_test_epoch_start(self):
        self.reset_validation_stats()

    def test_step(self, batch, batch_idx):
        self._run_val_test_step(batch, batch_idx, name="test")
    
    def on_test_epoch_end(self):
        stats = self.summarize_metric(name="test")
        if stats is not None:
            printable = {key: value for key, value in stats.items() if key != "fig"}

            from pprint import pprint
            pprint(printable)

            stats["fig"].savefig(osp.join(self.logger.save_dir, "test_error_curve.png"))
            
    def configure_optimizers(self):
        
        if isinstance(self.model, HieraDPT):
            optimizer = self.model.configure_optimizers(self.cfg)
        else:
            if self.cfg.optimizer.name == "Adam":
                optimizer = optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=self.cfg.optimizer.lr)
            elif self.cfg.optimizer.name == "AdamW":
                optimizer = optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.cfg.optimizer.lr)

        opt_config = dict(optimizer = optimizer)
        
        T_max = self.total_steps // (2 * self.cfg.scheduler.cycle_k + 1)
        if cfg.scheduler.name == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=cfg.optimizer.eta_min)
            opt_config["lr_scheduler"] = dict(scheduler=scheduler, interval="step")
        elif cfg.scheduler.name == "linear_cosine":
            scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_steps=cfg.scheduler.warmup, T_max=T_max, eta_min=cfg.optimizer.eta_min)
            opt_config["lr_scheduler"] = dict(scheduler=scheduler, interval="step")
        elif cfg.scheduler.name == "none":
            print("no scheduler")
            pass

        return opt_config


if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('--gpus', type=int, default=4)
    arg.add_argument('--exp_name', type=str, default="exp/base")
    arg.add_argument('--ckpt_path', type=str, default=None)
    arg.add_argument('--config', type=str, default="configs/dt_vit.yaml")
    arg.add_argument('--logger', type=str, default="wandb")
    arg.add_argument('--dataset_dir', type=str, default="../Documents/Dataset/sim_dataset")
    arg.add_argument('--eval', action='store_true')
    arg.add_argument('--finetune', action='store_true')
    arg.add_argument('--real_world', action='store_true')
    opt = arg.parse_args()

    # load config
    cfg = get_cfg_defaults()
    cfg.merge_from_file(opt.config)
    # copy config file
    os.makedirs(opt.exp_name, exist_ok=True)

    if opt.config != osp.join(opt.exp_name, 'config.yaml'):
        shutil.copy(opt.config, osp.join(opt.exp_name, 'config.yaml'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((cfg.model.img_size, cfg.model.img_size), antialias=True),
    ])
    
    dataset = FullDataset(cfg, transform=transform, 
                          samples_dir=opt.dataset_dir, is_real_world=opt.real_world)
    
    print("Dataset total samples: {}".format(len(dataset)))
    full_dataset_length = len(dataset)

    dataset_length = int(cfg.dataset_ratio * full_dataset_length)
    train_size = int(0.7 * dataset_length)
    test_size = dataset_length - train_size
    cfg.total_steps = train_size * cfg.epochs // (cfg.batch_size * opt.gpus)
    
    train_dataset, test_dataset, _ = random_split(dataset, 
                                                  [train_size, test_size, full_dataset_length - dataset_length],
                                                  generator=torch.Generator().manual_seed(cfg.seed))
    print(f"Train dataset length: {len(train_dataset)}, Test dataset length: {len(test_dataset)}")
    
    dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=12)

    calibration_model = LightningDTModel(cfg)

    # get date
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    name = "{}-{}".format(opt.exp_name.split("/")[-1], date)
   
    if opt.logger == "tensorboard":
        logger = TensorBoardLogger(osp.join(opt.exp_name, 'tb_logs/'), name="lightning_logs")
    if opt.logger == "wandb":
        logger = WandbLogger(project="DenseTact", name=name, config=cfg)
    
    # create callback to save checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor='val_psnr',
        dirpath=osp.join(opt.exp_name, 'checkpoints/'),
        filename='dt_model-{epoch:02d}-{val_psnr:.2f}',
        save_top_k=3,
        verbose=True,
        save_last=True,
        mode='max',
    )

    # log learning rate
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # add callbacks
    strategy = "ddp_find_unused_parameters_true" if opt.gpus > 1 else "auto"
    callbacks = [checkpoint_callback, lr_monitor]
    trainer = L.Trainer(max_epochs=cfg.epochs, callbacks=callbacks, logger=logger,
                        accelerator="gpu", devices=opt.gpus, strategy=strategy,
                        gradient_clip_val=cfg.gradient_clip_val, gradient_clip_algorithm=cfg.gradient_clip_algorithm)
    
    # only load states for finetuning
    if opt.finetune:
        model_state = torch.load(opt.ckpt_path)
        calibration_model.model.decoders[0].head.conv1 = nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        calibration_model.load_state_dict(model_state["state_dict"])
        calibration_model.model.decoders[0].head.conv1 = nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        opt.ckpt_path = None
        # have model decoder match number of output channels
        
    if opt.eval:
        trainer.test(model=calibration_model, dataloaders=test_dataloader, ckpt_path=opt.ckpt_path)
    else:
        trainer.fit(model=calibration_model, train_dataloaders=dataloader, 
                    ckpt_path=opt.ckpt_path, val_dataloaders=test_dataloader)
        
        trainer.test(model=calibration_model, dataloaders=test_dataloader)