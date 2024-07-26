"""Densenet for the Densetact model."""

import torch
import torch.nn as nn
import torchvision.models as models

class DecoderHead(nn.Module):
    """Decoder head module."""
    def __init__(self, output_channels=1):
        super(DecoderHead, self).__init__()
        self.conv1 = nn.Conv2d(64, output_channels, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh(x)
        
        return x    
    
class DecoderBlock(nn.Module):
    """Decoder block module using Upsampling and Convolution."""
    def __init__(self, in_channels, mid_channels, out_channels=None):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = None
        
        if out_channels is not None:
            self.conv2= nn.Sequential(
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, self._interpolate(skip, x)], dim=1)
        x = self.conv1(x)
        
        if self.conv2 is not None:
            x = self.conv2(x)
        
        return x
        
    def _interpolate(self, enc_ftrs, x):
        """Resize the encoder features to match the dimensions of the decoder features."""
        return torch.nn.functional.interpolate(enc_ftrs, size=x.shape[2:], mode='bilinear', align_corners=False)
        

class ResizeConv(nn.Module):
    def __init__(self, in_channels, decoder_channels):
        super(ResizeConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, decoder_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.conv(x)
        return x
    
class FullDecoder(nn.Module):
    def __init__(self, encoder_num_features, decoder_features, output_channels=1, encoder='densenet'):
        super(FullDecoder, self).__init__()
        self.conv_resize = ResizeConv(encoder_num_features, decoder_features)
        if encoder == 'densenet':
            start_features = [2080, 1408, 704, 352, 128]
        else: 
            # default to resnet
            start_features = [2048, 1536, 768, 320, 128]
        self.decoder= nn.ModuleList([
            DecoderBlock(start_features[0], 1024, 1024),
            DecoderBlock(start_features[1], 1024, 512),
            DecoderBlock(start_features[2], 512, 256),
            DecoderBlock(start_features[3], 256, 128),
            DecoderBlock(start_features[4], 64),
        ])
        
        self.head = DecoderHead(output_channels)
        
    def forward(self, x, skip_connections):
        x = self.conv_resize(x)
        
        for i, block in enumerate(self.decoder):
            x = block(x, skip=skip_connections[i])
            
        x = self.head(x)
        
        return x
    
class ResnetEncoder(nn.Module):
    def __init__(self, input_channels=7):
        super(ResnetEncoder, self).__init__()
        self.encoder = models.resnet152(pretrained=True)
        self.encoder.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        self.encoder[0] = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
    def forward(self, x):
        x1 = self.encoder[0](x)  # conv1
        x1 = self.encoder[1](x1)  # bn1
        x1 = self.encoder[2](x1)  # relu
        x1 = self.encoder[3](x1)  # maxpool
        x2 = self.encoder[4](x1)  # layer1
        x3 = self.encoder[5](x2)  # layer2
        x4 = self.encoder[6](x3)  # layer3
        x5 = self.encoder[7](x4)  # layer4
        
        return x1, x2, x3, x4, x5
    
class DensenetEncoder(nn.Module):
    def __init__(self, input_channels=7):
        super(DensenetEncoder, self).__init__()
        self.encoder = models.densenet161(pretrained=True)
        self.encoder.features.conv0 = nn.Conv2d(input_channels, 96, kernel_size=7, stride=2, padding=3, bias=False)
    
    def forward(self, x):
        x1 = self.encoder.features.relu0(self.encoder.features.norm0(self.encoder.features.conv0(x)))
        x1 = self.encoder.features.pool0(x1)
        x2 = self.encoder.features.denseblock1(x1)
        x2 = self.encoder.features.transition1(x2)
        x3 = self.encoder.features.denseblock2(x2)
        x3 = self.encoder.features.transition2(x3)
        x4 = self.encoder.features.denseblock3(x3)
        x4 = self.encoder.features.transition3(x4)
        x5 = self.encoder.features.denseblock4(x4)
        
        return x1, x2, x3, x4, x5

class DTNet(nn.Module):
    def __init__(self, input_channels=7, n_heads=1, head_output_channels=1, encoder='resnet'):
        super(DTNet, self).__init__()
        
        # Encoder: Using a pre-trained DenseNet-161 model
        encoder_features = None
        if encoder == 'densenet':
            self.encoder = DensenetEncoder(input_channels=input_channels)
            encoder_features = 2208
        else:
            # default to resnet152
            self.encoder = ResnetEncoder(input_channels=input_channels)
            encoder_features = 2048
        
        self.decoders = nn.ModuleList()
        # start with 1024 features in decoder
        self.decoder_features = 1024
        for _ in range(n_heads):
            # Decoder: Using a custom decoder
            self.decoders.append(FullDecoder(encoder_features, self.decoder_features, 
                                             output_channels=head_output_channels, encoder=encoder))
        
        
    def forward(self, x):
        # Densenet Encoder
        x1, x2, x3, x4, x5 = self.encoder(x)
        
        # Decoder
        # go through each decoder
        outputs = []
        for decoder in self.decoders:
            outputs.append(decoder(x5, [x4, x3, x2, x1, None]))
            
        # combine the outputs into:
        # (B, head_output_channels x n_heads output, H, W)
        outputs = torch.cat(outputs, dim=1)
        
        return outputs
    
    
if __name__ == "__main__":
    # Instantiate the model
    # sample usage with 1 heads and 1 output channels per head.
    model = DTNet(n_heads=1, head_output_channels=1, encoder='densenet')
    
    # swap encoder (resnet)
    # model = DTNet(n_heads=1, head_output_channels=1, encoder='resnet')
    
    # test input and output shape is desired
    dummy_input = torch.randn(1, 7, 256, 256)

    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_input)
        
    print(outputs.shape)