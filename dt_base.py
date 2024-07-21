import torch
import torch.nn as nn
import torchvision.models as models

class DTBaseModel(nn.Module):
    def __init__(self):
        super(DTBaseModel, self).__init__()
        
        # Encoder: Using a pre-trained ResNet-152 model
        self.encoder = models.resnet152(pretrained=True)
        self.encoder.conv1 = nn.Conv2d(7, 64, kernel_size=6, stride=2, padding=1)
        self.encoder.fc = nn.Identity()  # Removing the classification layer
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
        )
        
        # Multi-head output layers
        self.head1 = nn.Conv2d(32, 3, kernel_size=1)
        self.head2 = nn.Conv2d(32, 3, kernel_size=1)
        self.head3 = nn.Conv2d(32, 3, kernel_size=1)
        self.head4 = nn.Conv2d(32, 3, kernel_size=1)
        self.head5 = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, x):
        # Encoding
        x = self.encoder(x)
        
        # Reshape for decoder (ResNet outputs [batch, 2048])
        x = x.view(x.size(0), 2048, 1, 1)
        
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
    model = DTBaseModel()

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
