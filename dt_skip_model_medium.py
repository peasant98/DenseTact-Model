import torch
import torch.nn as nn
from torchvision import models

class DTSkipModelMedium(nn.Module):
    def __init__(self, input_dim_size=7):
        super(DTSkipModelMedium, self).__init__()
        # Load pre-trained ResNet and modify the first layer to accept 7 input channels
        resnet = models.resnet152(pretrained=True)
        
        # Modify the first convolutional layer to accept 7 input channels
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.encoder[0] = nn.Conv2d(input_dim_size, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.decoder_width = 0.5
        self.resnet_num_features = 2048
        self.decoder_features = int(self.resnet_num_features * self.decoder_width)
        
        self.conv_resize = nn.Conv2d(self.resnet_num_features, self.decoder_features, kernel_size=1)
        
        # Define the decoder with skip connections
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1024 + 1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        self.conv1a = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(512 + 1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.conv2a = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(512 + 256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv3a = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(320, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv4a = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.upsample5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upconv5 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Encoder
        x1 = self.encoder[0](x)  # conv1
        x1 = self.encoder[1](x1)  # bn1
        x1 = self.encoder[2](x1)  # relu
        x1 = self.encoder[3](x1)  # maxpool
        x2 = self.encoder[4](x1)  # layer1
        x3 = self.encoder[5](x2)  # layer2
        x4 = self.encoder[6](x3)  # layer3
        x5 = self.encoder[7](x4)  # layer4
        
        
        x = self.conv_resize(x5)
        x = self.upsample1(x)
        x = torch.cat([x, self._interpolate(x4, x)], dim=1)
        x = self.conv1(x)
        x = self.conv1a(x)
        
        x = self.upsample2(x)
        x = torch.cat([x, self._interpolate(x3, x)], dim=1)
        x = self.conv2(x)
        x = self.conv2a(x)
        
        x = self.upsample3(x)
        x = torch.cat([x, self._interpolate(x2, x)], dim=1)
        x = self.conv3(x)
        x = self.conv3a(x)
        
        x = self.upsample4(x)
        x = torch.cat([x, self._interpolate(x1, x)], dim=1)
        x = self.conv4(x)
        x = self.conv4a(x)
        
        x = self.upsample5(x)
        x = self.conv5(x)
        x = self.final_conv(x)
        
        return x

    def _interpolate(self, enc_ftrs, x):
        """Resize the encoder features to match the dimensions of the decoder features."""
        return torch.nn.functional.interpolate(enc_ftrs, size=x.shape[2:], mode='bilinear', align_corners=False)

if __name__ == '__main__':
    model = DTSkipModelMedium(input_dim_size=7)
    # Test the model with a random input tensor of shape (1, 7, 256, 256)
    input_tensor = torch.randn(1, 7, 256, 256)  # Batch size of 1, 7 channels, 256x256 image
    output = model(input_tensor)
    print(output.shape)  # Should be (1, 1, 256, 256) if input image size is 256x256