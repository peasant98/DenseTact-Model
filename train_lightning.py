import os

import cv2
import numpy as np

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models

from torchvision import transforms
from densetactnet import DTDenseNet
from dt_vit import VisionTransformer
from multi_head_beast import DTMassiveModel
from process_data import FullDataset

from dt_model import DTBaseModel

from tqdm import tqdm

import time

from resnet152 import DTResnet152
from skip_dt_model import ConvAutoencoder
from dt_skip_model_large import DTSkipModelLarge
from testing import DTSmallModel

import lightning as L

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
    def __init__(self, model_name='base', learning_rate=1e-3):
        super(LightningDTModel, self).__init__()
        
        if model_name == 'massive':
            self.model = DTMassiveModel()
        elif model_name == 'resnet152':
            self.model = DTResnet152()
        elif model_name == 'small':
            self.model = DTSmallModel()
        elif model_name == 'densenet':
            self.model = DTDenseNet()
        elif model_name == 'skip':
            self.model = ConvAutoencoder()
        elif model_name == 'skip_large':
            self.model = DTSkipModelLarge()
        elif model_name == 'vit':
            self.model = VisionTransformer()
        else:
            self.model = DTBaseModel()

        self.criterion = nn.L1Loss()
        # self.criterion = nn.MSELoss()
        self.mse = nn.MSELoss()
        self.learning_rate = learning_rate
        self.file_num = 0
        
    def training_step(self, batch, batch_idx):
        X, y = batch
        # X = torch.nn.functional.interpolate(X, size=(224, 224), mode='bilinear', align_corners=False)
        
        # only train on depth channels of y
        y = y[:, 9:12, :, :]
        outputs = self.model(X)
        
        # visualize displacement
        # d_1 = y[0][0].cpu().detach().numpy()
        # d_2 = y[0][1].cpu().detach().numpy()
        # d_3 = y[0][2].cpu().detach().numpy()
        # # visualize all 3 in one plot
        # fig, axs = plt.subplots(1, 3)
        # axs[0].imshow(d_1, cmap='viridis')
        # axs[1].imshow(d_2, cmap='viridis')
        # axs[2].imshow(d_3, cmap='viridis')
        # plt.show()
        
        # outputs = torch.cat(outputs, dim=1)
        loss = self.criterion(outputs, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        if batch_idx == 0:
            cnorm_1 = outputs[0][0].cpu().detach().numpy()
            # save cnorm
            if cnorm_1.dtype != np.uint8:
                cnorm_1 = cv2.normalize(cnorm_1, None, 0, 255, cv2.NORM_MINMAX)
                cnorm_1 = np.uint8(cnorm_1)
                # save as image with colormap
                depth_colormap = cv2.applyColorMap(cnorm_1, cv2.COLORMAP_JET)
                cv2.imwrite(f'viz_lightning/{self.file_num}_output_base.png', depth_colormap)
                
            self.file_num += 1
        
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer


if __name__ == '__main__':
    train = True
    model_save_folder = 'models/dt_model_1/'
    # make the model save folder if it doesn't exist
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)
        
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
    ])
    
    dataset = FullDataset(transform=transform)
    print(len(dataset), 'length')
    full_dataset_length = 132546
    
    # take only 10 percent of dataset for train and test
    dataset_length = int(0.1 * full_dataset_length)
    train_size = int(0.8 * dataset_length)
    test_size = dataset_length - train_size
    
    train_dataset, test_dataset, _ = random_split(dataset, [train_size, test_size, full_dataset_length - dataset_length])
    print(f"Train dataset length: {len(train_dataset)}, Test dataset length: {len(test_dataset)}")
    
    train_batch_size = 32
    
    dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=12)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=12)
    
    beta = 0.1
    
    calibration_model = LightningDTModel(model_name='skip_large', learning_rate=1e-3)
    trainer = L.Trainer(limit_train_batches=1000, max_epochs=1000)
    # trainer = L.Trainer(limit_train_batches=1000, max_epochs=1000, strategy='ddp_find_unused_parameters_true')
    
    
    trainer.fit(model=calibration_model, train_dataloaders=dataloader)