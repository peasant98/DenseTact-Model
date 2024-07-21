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
from multi_head_beast import DTMassiveModel
from process_data import FullDataset

from dt_model import DTBaseModel

from tqdm import tqdm

import time


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
    dataset_length = int(0.2 * full_dataset_length)
    
    
    train_size = int(0.8 * dataset_length)
    test_size = dataset_length - train_size
    
    train_dataset, test_dataset, _ = random_split(dataset, [train_size, test_size, full_dataset_length - dataset_length])
    print(f"Train dataset length: {len(train_dataset)}, Test dataset length: {len(test_dataset)}")
    
    train_batch_size = 1
    
    dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=12)
    
    beta = 0.1
    
    if not train:
        model_path = f'{model_save_folder}/dt_cnn_model_50.pth'
        # run_inference_and_plot(model_path, test_dataloader, device)
    else:
        # model = DTMassiveModel().to(device)  # Move model to device
        model = DTBaseModel().to(device)  # Move model to device
        # model = DTSkipModel().to(device)  # Move model to device
        
        model.float()
        gradient_loss = GradientLoss()
        criterion = nn.L1Loss()
        mse = nn.MSELoss()
        
        # define optimizer
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        experiment_name = "dt_base_model_l1_loss"
        log_dir = f"./logs/{experiment_name}_{int(time.time())}"
        writer = SummaryWriter(log_dir=log_dir)

        epochs = 500
        total_steps = 0
        running_loss = 0.0
        
        for epoch in range(epochs):
            model.train()
            for i, (X, y) in enumerate(tqdm(dataloader)):
                X = X.to(device)
                y = y.to(device)
                
                deformed = X[0][:3].cpu().detach().numpy().transpose(1, 2, 0)
                undeformed = X[0][3:6].cpu().detach().numpy().transpose(1, 2, 0)
                img_diff = X[0][6:].cpu().detach().numpy().transpose(1, 2, 0)
                
                # plot all three images
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                axs[0].imshow(deformed)
                axs[0].set_title('Deformed Image')
                
                axs[1].imshow(undeformed)
                axs[1].set_title('Undeformed Image')
                
                axs[2].imshow(img_diff)
                axs[2].set_title('Image Difference')
                
                plt.show()
                
                # visualize y 
                cnorm_1 = y[0][0].cpu().detach().numpy()
                
                stress_1 = y[0][4].cpu().detach().numpy()
                
                stress_2 = y[0][7].cpu().detach().numpy()
                
                force_1 = y[0][10].cpu().detach().numpy()
                
                area = y[0][12].cpu().detach().numpy()
                
                # plot all five outputs in one plot
                fig, axs = plt.subplots(1, 5, figsize=(20, 5))
                axs[0].imshow(cnorm_1, cmap='viridis')
                axs[0].set_title('Cnorm 1')
                
                axs[1].imshow(stress_1, cmap='viridis')
                axs[1].set_title('Stress 1')
                
                axs[2].imshow(stress_2, cmap='viridis')
                axs[2].set_title('Stress 2')
                
                axs[3].imshow(force_1, cmap='viridis')
                axs[3].set_title('Force 1')
                
                axs[4].imshow(area, cmap='viridis')
                axs[4].set_title('Area')
                
                plt.show()
                
                # optimizer.zero_grad()
                # outputs = model(X)
                # outputs = torch.cat(outputs, dim=1)
                # loss = criterion(outputs, y)
                # # print("Loss: ", loss)
                
                # loss.backward()
                # optimizer.step()
                # reg_loss = loss.item()
                # running_loss += loss.item()
                # total_steps += 1
                # avg_train_loss = running_loss / (total_steps * train_batch_size)
                # writer.add_scalar('training loss', reg_loss, total_steps)
                
                # if i == 0:
                #     cnorm_1 = outputs[0][0].cpu().detach().numpy()
                #     if cnorm_1.dtype != np.uint8:
                #         cnorm_1 = cv2.normalize(cnorm_1, None, 0, 255, cv2.NORM_MINMAX)
                #         cnorm_1 = np.uint8(cnorm_1)
                #     # save as image with colormap
                #     depth_colormap = cv2.applyColorMap(cnorm_1, cv2.COLORMAP_JET)
                #     cv2.imwrite(f'viz/{epoch}_output_big.png', cnorm_1)
                
                # if i % 10 == 9:
                #     print(f"[{epoch + 1}, {i + 1}] loss: {loss}")
                #     # plot sample output
                #     # cnorm = 
                #     # if i % 80 == 79:
                #     #     # show output
                #     #     plt.imshow(outputs[0][0].cpu().detach().numpy(), cmap='viridis')
                #     #     plt.show()
                    
            # Evaluate the test dataset after each epoch
            model.eval()
            test_loss = 0.0
            mse_loss = 0.0
            mse_deformation_loss = 0.0
            total = 0
            with torch.no_grad():
                for i, (X, y) in enumerate(tqdm(test_dataloader)):
                    X = X.to(device)
                    # depth, normal force, stress data 1, stress data 2, force, area and shear
                    y = y.to(device)
                    
                    outputs = model(X)
                    outputs = torch.cat(outputs, dim=1)
                    loss = criterion(outputs, y)
                    
                    test_mse = mse(outputs, y)
                    test_loss += loss.item()
                    mse_loss += test_mse.item()
                    
                    # get mse on y values that are not zero
                    y_values = y[y != 0]
                    outputs_values = outputs[y != 0]
                    mse_deformation_loss_1_sample = mse(outputs_values, y_values).item()
                    # add to total mse deformation loss if not nan
                    if not np.isnan(mse_deformation_loss_1_sample):
                        mse_deformation_loss += mse_deformation_loss_1_sample
                        total += 1
           
            avg_test_loss = test_loss / test_size
            avg_mse_loss = mse_loss / test_size
            avg_mse_deformation_loss = mse_deformation_loss / total
            print(f"Epoch {epoch + 1}, Test loss: {avg_test_loss:.3f}")
            writer.add_scalar('test loss', avg_test_loss, epoch)
            writer.add_scalar('test mse loss', avg_mse_loss, epoch)
            writer.add_scalar('test mse deformation loss', avg_mse_deformation_loss, epoch)
            model.train()
            
            torch.save(model.state_dict(), f'{model_save_folder}/dt_model{epoch + 1}.pth')
        
        print('Finished Training')
        writer.close()
        # Save the model
        torch.save(model.state_dict(), f'{model_save_folder}/dt_model.pth')
        
    
    
