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

from models.dt_vit import dt_vit_large_patch16 
from util.loss_util import ssim

class LightningDTModel(L.LightningModule):
    def __init__(self, 
                 model_name='base', 
                 num_epochs=100,
                 total_steps=1e6,
                 learning_rate=1e-3
        ):
        """ MAE Model for training on the DT dataset """
        super(LightningDTModel, self).__init__()
        
        self.model = dt_vit_large_patch16(img_size=256, in_chans=7, out_chans=15)

        self.criterion = nn.L1Loss()
        # self.criterion = nn.MSELoss()
        self.mse = nn.MSELoss()
        
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.total_steps = total_steps
        self.validation_stats = {}
        
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
            "mse": []
        }

    def validation_step(self, batch, batch_idx):
        # X - (N, C1, H, W); Y - (N, C2, H, W)
        X, Y = batch 
        N, C, H, W = X.shape 
        
        pred = self.model(X) 
        mse = self.mse(pred, Y)

        # outputs = torch.cat(outputs, dim=1)
        self.log('val/mse', mse, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.validation_stats["mse"].append(mse.item())
    
    def on_validation_epoch_end(self):
        avg_mse = np.array(self.validation_stats["mse"])
        avg_mse = np.mean(avg_mse)
        self.log('val/avg_mse', avg_mse, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    
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
    arg.add_argument('--batch_size', type=int, default=16)
    arg.add_argument('--num_workers', type=int, default=16)
    arg.add_argument('--exp_name', type=str, default="DT_Model")
    arg.add_argument('--ckpt_path', type=str, default=None)
    opt = arg.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256), antialias=True),
    ])
    
    dataset = FullDataset(transform=transform)
    print("Dataset total samples: {}".format(len(dataset)))
    full_dataset_length = len(dataset)

    # take only 10 percent of dataset for train and test
    dataset_length = int(opt.dataset_ratio * full_dataset_length)
    train_size = int(0.8 * dataset_length)
    test_size = dataset_length - train_size
    
    train_dataset, test_dataset, _ = random_split(dataset, [train_size, test_size, full_dataset_length - dataset_length])
    print(f"Train dataset length: {len(train_dataset)}, Test dataset length: {len(test_dataset)}")
    
    dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=12)

    calibration_model = LightningDTModel(
        model_name='vit', 
        num_epochs=opt.epochs,
        total_steps=opt.epochs * len(train_dataset),
        learning_rate=1e-3
    )

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
    callbacks = [checkpoint_callback, lr_monitor]
    trainer = L.Trainer(max_epochs=opt.epochs, callbacks=callbacks, logger=logger,
                        accelerator="gpu", devices=opt.gpus, strategy="ddp")
    
    trainer.fit(model=calibration_model, train_dataloaders=dataloader, 
                 ckpt_path=opt.ckpt_path, val_dataloaders=test_dataloader)