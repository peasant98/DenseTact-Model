import torch
import torch.nn as nn
import torchvision.models as models

class DTSmallModel(nn.Module):
    def __init__(self):
        super(DTSmallModel, self).__init__()
        
        # Encoder: Using a pre-trained ResNet-34 model
        self.encoder = models.resnet34(pretrained=True)
        self.encoder.conv1 = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder.fc = nn.Identity()  # Removing the classification layer
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        
        # Multi-head output layers
        self.head1 = nn.Conv2d(16, 3, kernel_size=1)
        self.head2 = nn.Conv2d(16, 3, kernel_size=1)
        self.head3 = nn.Conv2d(16, 3, kernel_size=1)
        self.head4 = nn.Conv2d(16, 3, kernel_size=1)
        self.head5 = nn.Conv2d(16, 3, kernel_size=1)

    def forward(self, x):
        # Encoding
        x = self.encoder(x)
        
        # Reshape for decoder (ResNet outputs [batch, 512, H, W])
        x = x.view(x.size(0), 512, 1, 1)
        
        x = self.decoder(x)
        
        # Multi-head outputs
        out1 = self.head1(x)
        out2 = self.head2(x)
        out3 = self.head3(x)
        out4 = self.head4(x)
        out5 = self.head5(x)
        
        return out1, out2, out3, out4, out5
    
if __name__ == "__main__":
    # Instantiate the model
    model = DTSmallModel()

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
