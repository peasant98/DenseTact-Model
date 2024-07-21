import torch
import torch.nn as nn
import torchvision.models as models

class DTDenseNet(nn.Module):
    def __init__(self):
        super(DTDenseNet, self).__init__()
        
        # Encoder: Using a pre-trained DenseNet-161 model
        self.encoder = models.densenet161(pretrained=True)
        self.encoder.features.conv0 = nn.Conv2d(7, 96, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Extract the number of features from the DenseNet model
        self.encoder_out_channels = 2208  # For densenet161
        
        # Decoders for each head
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(self.encoder_out_channels, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
        )
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(self.encoder_out_channels, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
        )
        
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(self.encoder_out_channels, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
        )
        
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(self.encoder_out_channels, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
        )
        
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(self.encoder_out_channels, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
        )
        
        

    def forward(self, x):
        # Encoding
        features = self.encoder.features(x)
        
        # Reshape for decoder
        # x = features.view(features.size(0), self.encoder_out_channels, 1, 1)
        
        # Decoding for each head
        out1 = self.decoder1(features)
        out2 = self.decoder2(features)
        out3 = self.decoder3(features)
        out4 = self.decoder4(features)
        out5 = self.decoder5(features)
        
        return out1, out2, out3, out4, out5
    
if __name__ == "__main__":
    # Instantiate the model
    model = DTDenseNet()

    # Create a dummy input tensor with the specified size (batch_size, 7, 256, 256)
    dummy_input = torch.randn(1, 7, 256, 256)

    # Run inference
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation for inference
        outputs = model(dummy_input)
        
    print(torch.cat(outputs, dim=1).shape)

    # Print the shapes of the outputs
    for i, output in enumerate(outputs, 1):
        print(f"Output {i} shape: {output.shape}")
