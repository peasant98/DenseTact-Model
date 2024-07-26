import os
import os.path as osp
import cv2
import numpy as np
import argparse

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

import models.model_mae as mae 
from util.loss_util import ssim

class SILogLoss(nn.Module):
    def __init__(self):
        super(SILogLoss, self).__init__()

    def forward(self, pred, target):
        eps = 1e-7
        pred_log = torch.log(pred + eps)
        target_log = torch.log(target + eps)
        diff_log = pred_log - target_log

        diff_log_sq = diff_log ** 2
        N = torch.numel(diff_log)
        first_term = diff_log_sq.mean()

        second_term = (diff_log.sum() ** 2) / (N ** 2)

        return first_term - second_term

class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, pred, target):
        pred_dy, pred_dx = self.gradient(pred)
        target_dy, target_dx = self.gradient(target)
        grad_diff_y = torch.abs(pred_dy - target_dy)
        grad_diff_x = torch.abs(pred_dx - target_dx)
        return grad_diff_y.mean() + grad_diff_x.mean()

    def gradient(self, x):
        D_dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        D_dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        return D_dy, D_dx


class LightningDTModel(L.LightningModule):
    def __init__(self, 
                 model_name='base', 
                 num_epochs=100,
                 total_steps=1e6,
                 mask_ratio=0.75,
                 learning_rate=1e-3
        ):
        """ MAE Model for training on the DT dataset """
        super(LightningDTModel, self).__init__()
        
        self.model = mae.mae_vit_base_patch16(img_size=256, in_chans=7)

        self.criterion = nn.L1Loss()
        # self.criterion = nn.MSELoss()
        self.mse = nn.MSELoss()
        
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.total_steps = total_steps
        self.mask_ratio = mask_ratio
        
    def training_step(self, batch, batch_idx):
        # X - (N, C1, H, W); Y - (N, C2, H, W)
        X, _ = batch 
        N, C, H, W = X.shape 
        loss, pred, mask = self.model(X, self.mask_ratio) 

        # outputs = torch.cat(outputs, dim=1)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        with torch.no_grad():
            pred = pred.detach().clone()
            X = X.detach().clone()
            pred = pred.reshape(N, 16, 16, 16, 16, C).permute(0, 5, 1, 3, 2, 4)
            pred = pred.reshape(N, C, H, W)

            # compute PSNR
            mse = self.mse(pred, X)
            psnr = 10 * torch.log10(1 / mse)
            self.log('train/psnr', psnr, on_step=True, on_epoch=True, prog_bar=False, logger=True)

            ssim_loss = ssim(pred, X)
            self.log('train/ssim', ssim_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        
            # visualize the reconstructed images
            if batch_idx % 100 == 0:
                # reshape pred to (N, C, H, W)
                deform_color = pred[0, :3, :, :].clamp_(min=0., max=1.).detach().cpu().numpy()
                undeform_color = pred[0, 3:6, :, :].clamp_(min=0, max=1.).detach().cpu().numpy()
                diff_color = pred[0, [6], :, :].detach().cpu().numpy()
                
                # log images in pytorch lightning
                self.logger.experiment.add_image('reconstruct/deform_color', deform_color, self.global_step)
                self.logger.experiment.add_image('reconstruct/undeform_color', undeform_color, self.global_step)
                self.logger.experiment.add_image('reconstruct/diff_color', diff_color, self.global_step)

                # gt images
                gt_deform_color = X[0, :3, :, :].clamp_(min=0., max=1.).detach().cpu().numpy()
                gt_undeform_color = X[0, 3:6, :, :].clamp_(min=0, max=1.).detach().cpu().numpy()
                gt_diff_color = X[0, [6], :, :].detach().cpu().numpy()

                self.logger.experiment.add_image('gt/deform_color', gt_deform_color, self.global_step)
                self.logger.experiment.add_image('gt/undeform_color', gt_undeform_color, self.global_step)
                self.logger.experiment.add_image('gt/diff_color', gt_diff_color, self.global_step)

                mask = mask.reshape(N, 16, 16)
                mask = mask.repeat_interleave(16, dim=1).repeat_interleave(16, dim=2).unsqueeze(1) # to (N, 1, H, W)
                mask = mask[0].detach().cpu().numpy()
                self.logger.experiment.add_image('gt/mask', mask, self.global_step)

        
        return {"loss": loss, "train_psnr": psnr, "train_ssim": ssim_loss}
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.total_steps)
        return {"optimizer": optimizer, 
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "monitor": "train_loss"
                }}


if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('--dataset_ratio', type=float, default=1.0)
    arg.add_argument('--epochs', type=int, default=100)
    arg.add_argument('--gpus', type=int, default=1)
    arg.add_argument('--batch_size', type=int, default=32)
    arg.add_argument('--num_workers', type=int, default=32)
    arg.add_argument('--mask_ratio', type=float, default=0.75)
    arg.add_argument('--exp_name', type=str, default="exp/DT_Model")
    arg.add_argument('--ckpt_path', type=str, default=None)
    opt = arg.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256), antialias=True),
    ])
    
    dataset = FullDataset(transform=transform, output_type='none')
    print("Dataset total samples: {}".format(len(dataset)))
    full_dataset_length = len(dataset)

    # take only 10 percent of dataset for train and test
    dataset_length = int(opt.dataset_ratio * full_dataset_length)
    train_size = int(0.8 * dataset_length)
    test_size = dataset_length - train_size
    
    train_dataset, test_dataset, _ = random_split(dataset, [train_size, test_size, full_dataset_length - dataset_length])
    print(f"Train dataset length: {len(train_dataset)}, Test dataset length: {len(test_dataset)}")
    
    dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=12)

    calibration_model = LightningDTModel(
        model_name='vit', 
        num_epochs=opt.epochs,
        total_steps=opt.epochs * len(train_dataset),
        mask_ratio=opt.mask_ratio,
        learning_rate=1e-3
    )

    logger = TensorBoardLogger(osp.join(opt.exp_name, 'tb_logs/'), name="lightning_logs")
    
    # create callback to save checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
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