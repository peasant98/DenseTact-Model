import torch
import torch.nn as nn
import torchvision.models as models

class DTDenseNet(nn.Module):
    def __init__(self, out_chans=[3, 3, 3, 1]):
        super(DTDenseNet, self).__init__()
        
        # Encoder: Using a pre-trained DenseNet-161 model
        self.encoder = models.densenet161(pretrained=True)
        self.encoder.features.conv0 = nn.Conv2d(7, 96, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Extract the number of features from the DenseNet model
        self.encoder_out_channels = 2208  # For densenet161
        
        self.decoders = nn.ModuleList()
        for i, out_chan in enumerate(out_chans):
            decoder = nn.Sequential(
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
                nn.ConvTranspose2d(64, out_chan, kernel_size=4, stride=2, padding=1),
            )

            self.decoders.append(decoder)

    def forward(self, x):
        # Encoding
        features = self.encoder.features(x)
        
        # Reshape for decoder
        # x = features.view(features.size(0), self.encoder_out_channels, 1, 1)
        outputs = []

        # Decoding for each head
        for decoder in self.decoders:
            out = decoder(features)
            outputs.append(out)
        
        return torch.cat(outputs, dim=1)
    
    def load_from_pretrained_model(self, model_path:str):
        """
        Load weights from a pre-trained model
        """
        # print in red color
        print("\033[91m" + " [WARN] Pre-trained model not available for DenseNet. Skipping loading weights." + "\033[0m")

        pass

    def freeze_encoder(self):
        """
        Freeze the encoder weights
        """
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """
        Unfreeze the encoder weights
        """
        for param in self.encoder.parameters():
            param.requires_grad = True
    
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
