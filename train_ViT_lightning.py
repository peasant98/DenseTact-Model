import os
import os.path as osp
import cv2
import torch.distributed as dist
import shutil
import numpy as np
import argparse
from copy import deepcopy

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models
from torchvision import transforms
from process_data import FullDataset
from lightning.pytorch.loggers import TensorBoardLogger

import lightning as L
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor

from configs import get_cfg_defaults

from models.dt_vit import dt_vit_base_patch16 
from models.dpt import DPTV2Net
from util.loss_util import ssim

class LightningDTModel(L.LightningModule):
    def __init__(self, cfg):
        """ MAE Model for training on the DT dataset """
        super(LightningDTModel, self).__init__()
        self.cfg = cfg
        
        # build model
        if cfg.model.name == "ViT":
            self.model = dt_vit_base_patch16(img_size=cfg.model.img_size, 
                            in_chans=cfg.model.in_chans, out_chans=cfg.model.out_chans)
        elif cfg.model.name == "DPT":
            self.model = DPTV2Net(img_size=cfg.model.img_size, 
                            in_chans=cfg.model.in_chans, out_dims=cfg.model.out_chans)
        else:
            raise NotImplementedError("Model not implemented {}".format(cfg.model.name))

        # build loss
        if cfg.model.loss == "L1":
            self.criterion = nn.L1Loss()
        elif cfg.model.loss == "L2":
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError("Loss not implemented {}".format(cfg.model.loss))
        
        # MSE for validation
        self.mse = nn.MSELoss()
        
        self.num_epochs = cfg.epochs
        self.total_steps = cfg.total_steps
        self.validation_stats = {}

    def on_train_start(self):
        # only load pre-trained model at the first epoch
        if self.cfg.model.pretrained_model is not None and self.trainer.current_epoch == 0:
            print("Load pretrained model")
            self.model.load_from_pretrained_model(self.cfg.model.pretrained_model)

    def on_train_epoch_start(self):
        if self.cfg.model.pretrained_model is not None:
            print("Freezing encoder when loading from pretrained model")
            if self.trainer.current_epoch < self.cfg.finetune_ratio * self.num_epochs:
                self.model.freeze_encoder()
            else:
                self.model.unfreeze_encoder()
            
    def training_step(self, batch, batch_idx):
        # X - (N, C1, H, W); Y - (N, C2, H, W)
        X, Y = batch 
        N, C, H, W = X.shape 
        
        pred = self.model(X) 
        loss = self.criterion(pred, Y)

        # outputs = torch.cat(outputs, dim=1)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        with torch.no_grad():
        
            # visualize the reconstructed images
            if batch_idx % 100 == 0:
                X0 = X[0].detach().clone()

                # gt images
                gt_deform_color = X0[:3, :, :].clamp_(min=0., max=1.).detach().cpu().numpy()
                gt_undeform_color = X0[3:6, :, :].clamp_(min=0, max=1.).detach().cpu().numpy()
                gt_diff_color = X0[[6], :, :].detach().cpu().numpy()

                self.logger.experiment.add_image('gt/deform_color', gt_deform_color, self.global_step)
                self.logger.experiment.add_image('gt/undeform_color', gt_undeform_color, self.global_step)
                self.logger.experiment.add_image('gt/diff_color', gt_diff_color, self.global_step)
        
        return {"loss": loss}
    
    def on_validation_epoch_start(self):
        self.validation_stats = {
            "mse": [],
        }

        self.validation_stats["depth"] = dict(NegativeNum=0, FNNum=0, PosNum=0, error_cureve=np.zeros(11))

    @torch.no_grad()
    def compute_metric(self, prediction:torch.Tensor, label:torch.Tensor):
        """ Compute the prediction metric 
        
        Args:
            prediction: (N, C, H, W)
            label: (N, C, H, W)

        Returns:
            dict: dictionary of metrics
                NegativeNum: Number of negative samples
                FNNum: Number of false negative samples
                PosNum: Number of positive samples
                error_curve: Error curve (11)
                    [<0.1, <0.2, ..., <0.9, <1.0, >1.0]
        """
        prediction_flatten = prediction.flatten()
        label_flatten = label.flatten()

        num_pixels = label_flatten.shape[0]
        threshold_ratio = [k * 0.1 for k in range(1, 11)]

        # Here, negative samples are the pixels with value than 1e-6
        # positive samples are the pixels with value greater than 1e-6
        eps = 1e-6

        # compute the negative pixels
        Neg_mask = torch.abs(label_flatten) < eps
        Pos_mask = torch.abs(label_flatten) >= eps

        # compute the false negative
        NegativeNum = torch.sum(Neg_mask).item()
        FNNum = torch.sum(Neg_mask * (torch.abs(prediction_flatten) >= eps))

        # compute the error curve
        gt_pos_items = label_flatten[Pos_mask]
        pred_pos_items = prediction_flatten[Pos_mask]

        PosNum = torch.sum(Pos_mask).item()
        error_curve = []
        if PosNum > 0:
            rel_error = torch.abs(gt_pos_items - pred_pos_items) / torch.abs(gt_pos_items)

            # compute the threshold ratio
            for ratio in threshold_ratio:
                error_curve.append(torch.sum(rel_error < ratio).item())

            error_curve.append(torch.sum(rel_error > 1).item())
            error_curve = np.array(error_curve)
        
        return dict(NegativeNum=NegativeNum, FNNum=FNNum, PosNum=PosNum, error_cureve=error_curve)
    
    def update_metric(self, cumulative_metric_dict:dict, current_step_dict:dict):
        for key, item in current_step_dict.items():
            cumulative_metric_dict[key] += item

    def validation_step(self, batch, batch_idx):
        # X - (N, C1, H, W); Y - (N, C2, H, W)
        X, Y = batch 
        N, C, H, W = X.shape 
        
        pred = self.model(X) 
        mse = self.mse(pred, Y)

        # outputs = torch.cat(outputs, dim=1)
        self.log('val/mse', mse, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.validation_stats["mse"].append(mse.item())

        depth_metric_dict = self.compute_metric(pred[:, 0, :, :], Y[:, 0, :, :])
        self.update_metric(self.validation_stats["depth"], depth_metric_dict)
    
    def on_validation_epoch_end(self):
        avg_mse = np.array(self.validation_stats["mse"])
        avg_mse = np.mean(avg_mse)
        self.log('val/avg_mse', avg_mse, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        if self.trainer.strategy == "ddp":
            # summarize the results from multiple GPUs
            output = [None for _ in range(dist.get_world_size())]

            # gather the results from all GPUs
            dist.all_gather_object(output, self.validation_stats, async_op=True)

            if self.global_rank == 0:
                # sum to the validation stats
                for stats in output[1:]:
                    
                    for key, item in stats:
                        if isinstance(item, dict):
                            self.validation_stats[key] = self.update_metric(self.validation_stats[key], item)
                        else:
                            self.validation_stats[key] += stats[key]
        
        # compute the metric in the rank = 0
        if self.global_rank == 0:
            FNR = self.validation_stats["depth"]["FNNum"] / self.validation_stats["depth"]["NegativeNum"]

            # here FPR is defined as the rel error > 1.0
            FPR = self.validation_stats["depth"]["error_cureve"][-1] / self.validation_stats["depth"]["PosNum"]

            # log
            self.log('val/FNR', FNR, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val/FPR', FPR, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            # draw the curve
            error_curve = self.validation_stats["depth"]["error_cureve"][:-1] / self.validation_stats["depth"]["PosNum"]

            # compute the Area under curve
            AUC = np.sum(error_curve) / 10
            self.log('val/AUC', AUC, on_step=False, on_epoch=True, prog_bar=True, logger=True)


            # draw the plot of the error curve
            plt.plot(np.arange(0.1, 1.1, 0.1), error_curve)
            plt.xlabel("Threshold")
            plt.ylabel("Error Rate")

            self.logger.experiment.add_figure("val/Error Curve", plt.gcf(), self.global_step)
            
    def configure_optimizers(self):
        if self.cfg.optimizer.name == "Adam":
            optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.optimizer.lr)
        elif self.cfg.optimizer.name == "AdamW":
            optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.optimizer.lr)
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.total_steps)
        return {"optimizer": optimizer, 
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "monitor": "train_loss"
                }}


if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('--gpus', type=int, default=1)
    arg.add_argument('--exp_name', type=str, default="DT_Model")
    arg.add_argument('--ckpt_path', type=str, default=None)
    arg.add_argument('--config', type=str, default="configs/dt_vit.yaml")
    arg.add_argument('--eval', action='store_true')
    opt = arg.parse_args()

    # load config
    cfg = get_cfg_defaults()
    cfg.merge_from_file(opt.config)
    # copy config file
    os.makedirs(opt.exp_name, exist_ok=True)
    shutil.copy(opt.config, osp.join(opt.exp_name, 'config.yaml'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((cfg.model.img_size, cfg.model.img_size), antialias=True),
    ])
    
    dataset = FullDataset(transform=transform, output_type=cfg.dataset.output_type)
    print("Dataset total samples: {}".format(len(dataset)))
    full_dataset_length = len(dataset)

    # take only 10 percent of dataset for train and test
    dataset_length = int(cfg.dataset_ratio * full_dataset_length)
    train_size = int(0.8 * dataset_length)
    test_size = dataset_length - train_size
    cfg.total_steps = train_size * cfg.epochs
    
    train_dataset, test_dataset, _ = random_split(dataset, [train_size, test_size, full_dataset_length - dataset_length])
    print(f"Train dataset length: {len(train_dataset)}, Test dataset length: {len(test_dataset)}")
    
    dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=12)

    calibration_model = LightningDTModel(cfg)

    logger = TensorBoardLogger(osp.join(opt.exp_name, 'tb_logs/'), name="lightning_logs")
    
    # create callback to save checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor='val/avg_mse',
        dirpath=osp.join(opt.exp_name, 'checkpoints/'),
        filename='dt_model-{epoch:02d}-{val_mse:.2f}',
        save_top_k=1,
        save_last=True,
        mode='min',
    )

    # log learning rate
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # add callbacks
    strategy = "ddp" if opt.gpus > 1 else "auto"
    callbacks = [checkpoint_callback, lr_monitor]
    trainer = L.Trainer(max_epochs=cfg.epochs, callbacks=callbacks, logger=logger,
                        accelerator="gpu", devices=opt.gpus, strategy=strategy)
    
    if opt.eval:
        trainer.test(model=calibration_model, test_dataloaders=test_dataloader, ckpt_path=opt.ckpt_path)
    else:
        trainer.fit(model=calibration_model, train_dataloaders=dataloader, 
                    ckpt_path=opt.ckpt_path, val_dataloaders=test_dataloader)