import torch
import torch.nn as nn
import torchvision.models as models

class DTResnet152(nn.Module):
    def __init__(self):
        super(DTResnet152, self).__init__()
        
        # Encoder: Using a pre-trained ResNet-152 model
        self.encoder = models.resnet152(pretrained=True)
        self.encoder.conv1 = nn.Conv2d(7, 64, kernel_size=6, stride=2, padding=3, bias=False)
        self.encoder.fc = nn.Identity()  # Removing the classification layer
        
        # Decoders for each head
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
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
            nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1),
        )
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
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
            nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1),
        )

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
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
            nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1),
        )

        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
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
            nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1),
        )

        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
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
            nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        # Encoding
        x = self.encoder(x)
        
        # Reshape for decoder (ResNet outputs [batch, 2048])
        x = x.view(x.size(0), 2048, 1, 1)
        
        # Decoding for each head
        out1 = self.decoder1(x)
        out2 = self.decoder2(x)
        out3 = self.decoder3(x)
        out4 = self.decoder4(x)
        out5 = self.decoder5(x)
        
        return out1, out2, out3, out4, out5
    
if __name__ == "__main__":
    # Instantiate the model
    model = DTResnet152()

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
