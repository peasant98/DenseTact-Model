import os
import os.path as osp
import cv2
import math
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

from models import pretrain_dict
from util.loss_util import ssim
from util.scheduler_util import LinearWarmupCosineAnnealingLR

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
        
        if model_name in pretrain_dict:
            self.model = pretrain_dict[model_name](input_size=256, in_chans=7)
        else:
            print("Model {} not found in pretrain_dict".format(model_name))
            print("Available models: {}".format(pretrain_dict.keys()))
            exit()

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
            
            pred = self.model.reconstruct_img(pred)

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

                # spatial tokens
                spatial_num = mask.shape[1]
                # assume square image
                num = int(math.sqrt(spatial_num))
                mask_shape = H // num
                mask = mask.reshape(N, num, num)
                mask = mask.repeat_interleave(mask_shape, dim=1).repeat_interleave(mask_shape, dim=2).unsqueeze(1) # to (N, 1, H, W)
                mask = mask[0].detach().cpu().numpy()
                self.logger.experiment.add_image('gt/mask', mask, self.global_step)

        
        return {"loss": loss, "train_psnr": psnr, "train_ssim": ssim_loss}
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.95), weight_decay=0.05)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_steps=10, T_max=self.num_epochs)
        return {"optimizer": optimizer, 
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }}


if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('--dataset_ratio', type=float, default=1.0)
    arg.add_argument('--dataset_dir', type=str, default="./output")
    arg.add_argument('--epochs', type=int, default=150)
    arg.add_argument('--gpus', type=int, default=1)
    arg.add_argument('--model', type=str, default="hiera", help="Model Architectire, choose either hiera or vit")
    arg.add_argument('--batch_size', type=int, default=64)
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
    
    dataset = FullDataset(transform=transform, output_type='none', samples_dir=opt.dataset_dir)
    print("Dataset total samples: {}".format(len(dataset)))
    full_dataset_length = len(dataset)

    # take only 10 percent of dataset for train and test
    dataset_length = int(opt.dataset_ratio * full_dataset_length)
    train_size = int(0.95 * dataset_length)
    test_size = dataset_length - train_size
    
    train_dataset, test_dataset, _ = random_split(dataset, [train_size, test_size, full_dataset_length - dataset_length])
    print(f"Train dataset length: {len(train_dataset)}, Test dataset length: {len(test_dataset)}")
    
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