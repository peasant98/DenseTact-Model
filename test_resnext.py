import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

class ResnextEncoder(nn.Module):
    def __init__(self, cfg):
        super(ResnextEncoder, self).__init__()
        backbone_name = cfg.model.backbone  # e.g., 'resnext101_32x8d'
        in_chans = cfg.model.in_chans       # Number of input channels
        pretrained = cfg.model.imagenet_pretrained  # Boolean for pretrained weights
        import pdb; pdb.set_trace()

        # Initialize the ResNeXt model
        self.encoder = getattr(models, backbone_name)(pretrained=pretrained)
        last_feat_dim = self.encoder.fc.in_features  # Feature dimension before the fully connected layer

        # Modify the first convolutional layer if input channels are not equal to 3
        if in_chans != 3:
            self.encoder.conv1 = nn.Conv2d(
                in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Remove the fully connected layer and the average pooling layer
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])

        # Extract feature channels from different layers
        self.feature_channels = [
            last_feat_dim,  # Output channels from layer4
            self.encoder[7][-1].conv3.out_channels,  # Output channels from layer3
            self.encoder[6][-1].conv3.out_channels,  # Output channels from layer2
            self.encoder[5][-1].conv3.out_channels,  # Output channels from layer1
            self.encoder[1].num_features,            # Output channels after maxpool
        ]

    def forward(self, x):
        features = []

        # Initial convolution and pooling layers
        x = self.encoder[0](x)  # conv1
        x = self.encoder[1](x)  # bn1
        x = self.encoder[2](x)  # relu
        x = self.encoder[3](x)  # maxpool
        features.append(x)      # Output after maxpool

        # Layer 1
        x = self.encoder[4](x)
        features.append(x)

        # Layer 2
        x = self.encoder[5](x)
        features.append(x)

        # Layer 3
        x = self.encoder[6](x)
        features.append(x)

        # Layer 4
        x = self.encoder[7](x)
        features.append(x)

        # Return features in reverse order (from deepest to shallowest)
        return features[::-1]

if __name__ == '__main__':
    # Configuration class
    class Config:
        class Model:
            backbone = 'resnext101_32x8d'
            imagenet_pretrained = True
            in_chans = 3  # Change this if you have a different number of input channels
        model = Model()
    cfg = Config()

    # Create ResNeXt encoder instance
    encoder = ResnextEncoder(cfg)

    # Test with a dummy input tensor
    x = torch.randn(1, cfg.model.in_chans, 224, 224)
    features = encoder(x)

    # Print the shape of the output features from different layers
    for idx, f in enumerate(features):
        print(f"Feature {len(features) - idx}: Shape {f.shape}")
