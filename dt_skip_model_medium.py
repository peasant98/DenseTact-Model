import torch
import torch.nn as nn
from torchvision import models

class DTSkipModelMedium(nn.Module):
    def __init__(self, input_dim_size=7):
        super(DTSkipModelLarge, self).__init__()
        # Load pre-trained ResNet and modify the first layer to accept 7 input channels
        resnet = models.resnet152(pretrained=True)
        
        # Modify the first convolutional layer to accept 7 input channels
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.encoder[0] = nn.Conv2d(input_dim_size, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.decoder_width = 0.5
        
        self.conv_resize = nn.Conv2d()
        
        # Define the decoder with skip connections
        self.upconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1024 + 1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        self.upconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(512 + 512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.upconv5 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # only try to learn normal force
        
        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
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
        
        # Decoder with skip connections
        x = self.upconv1(x5)
        x = torch.cat([x, self._interpolate(x4, x)], dim=1)
        x = self.conv1(x)
        
        x = self.upconv2(x)
        
        x = torch.cat([x, self._interpolate(x3, x)], dim=1)
        x = self.conv2(x)

        
        x = self.upconv3(x)
        x = torch.cat([x, self._interpolate(x2, x)], dim=1)
        x = self.conv3(x)
        
        x = self.upconv4(x)
        x = torch.cat([x, self._interpolate(x1, x)], dim=1)
        x = self.conv4(x)

        
        x = self.upconv5(x)
        x = self.conv5(x)
        
        x = self.final_conv(x)
        
        # optional activation layer 
        # x = torch.sigmoid(x)
        
        return x

    def _interpolate(self, enc_ftrs, x):
        """Resize the encoder features to match the dimensions of the decoder features."""
        return torch.nn.functional.interpolate(enc_ftrs, size=x.shape[2:], mode='bilinear', align_corners=False)

if __name__ == '__main__':
    model = DTSkipModelLarge(input_dim_size=7)
    # Test the model with a random input tensor of shape (1, 7, 256, 256)
    input_tensor = torch.randn(1, 7, 256, 256)  # Batch size of 1, 7 channels, 256x256 image
    output = model(input_tensor)
    print(output.shape)  # Should be (1, 1, 256, 256) if input image size is 256x256


    # # Create a dummy input tensor with the specified size (batch_size, 3, 256, 256)
    # dummy_input = torch.randn(1, 3, 256, 256)

    # # Run inference
    # model.eval()  # Set the model to evaluation mode
    # with torch.no_grad():  # Disable gradient computation for inference
    #     outputs = model(dummy_input)

    # # Print the shapes of the outputs
    # for i, output in enumerate(outputs, 1):
    #     print(f"Output {i} shape: {output.shape}")