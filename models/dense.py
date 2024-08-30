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

class DecoderHead(nn.Module):
    """Decoder head module."""
    def __init__(self, in_chans=64, output_channels=1):
        super(DecoderHead, self).__init__()
        self.conv1 = nn.Conv2d(in_chans, output_channels, kernel_size=3, padding=1)
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

        self.apply(self.init_weights)
    
    @torch.no_grad()
    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
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
    def __init__(self, encoder_num_features, decoder_features,
                        decoder_mid_dim, decoder_output_dim,
                        output_channels=1):
        """
        Full Decoder of DenseNet
        encoder_num_features List[int]: number of features from the encoder, from deep to shallow 
        decoder_features int: number of features in the first layer of the decoder
        decoder_mid_dim List[int]: number of features in the middle of the decoder
        decoder_output_dim List[int]: number of features in the output of the decoder
        output_channels int: number of output channels
        """
        super(FullDecoder, self).__init__()
        self.conv_resize = ResizeConv(encoder_num_features[0], decoder_features)
        
        self.decoder = nn.ModuleList()
        for i in range(1, len(encoder_num_features)):
            if i == 1:
                decoder_input_dim = decoder_features + encoder_num_features[i]
            else:
                decoder_input_dim = decoder_output_dim[i - 2] + encoder_num_features[i]
            
            self.decoder.append(DecoderBlock(decoder_input_dim,
                                             decoder_mid_dim[i - 1],
                                             decoder_output_dim[i - 1]))
        
        # no concatenation in the last layer
        # self.decoder.append(DecoderBlock(decoder_output_dim[-2], decoder_mid_dim[-1], decoder_output_dim[-1]))
        # [Deprecated] Use the following line instead of the above line for old weights
        self.decoder.append(DecoderBlock(decoder_output_dim[-2], decoder_mid_dim[-1])) # , decoder_output_dim[-1]))
        
        self.head = DecoderHead(decoder_output_dim[-1], output_channels)
        
    def forward(self, x, skip_connections):
        x = self.conv_resize(x)
        
        for i, block in enumerate(self.decoder):
            x = block(x, skip=skip_connections[i])
            
        x = self.head(x)
        
        return x
    
class ResnetEncoder(nn.Module):
    def __init__(self, cfg):
        super(ResnetEncoder, self).__init__()
        self.encoder = getattr(models, cfg.model.backbone)(pretrained=cfg.model.imagenet_pretrained)
        last_feat_dim = self.encoder.fc.in_features

        self.encoder.conv1 = nn.Conv2d(cfg.model.in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        self.encoder[0] = nn.Conv2d(cfg.model.in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.feature_channels = [ last_feat_dim,
                                self.encoder[7][0].conv1.in_channels,
                                self.encoder[6][0].conv1.in_channels, 
                                self.encoder[5][0].conv1.in_channels,
                                self.encoder[4][0].conv1.in_channels]

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
    
    def get_feature_channels(self):
        """ get feature channels from the encoder, from deep to shallow """
        return self.feature_channels
    
class DensenetEncoder(nn.Module):
    def __init__(self, cfg):
        super(DensenetEncoder, self).__init__()
        self.encoder = getattr(models, cfg.model.backbone)(pretrained=True)
        self.feature_channels = [ self.encoder.classifier.in_features,
                                self.encoder.features.denseblock4.denselayer1.norm1.num_features,
                                self.encoder.features.denseblock3.denselayer1.norm1.num_features, 
                                self.encoder.features.denseblock2.denselayer1.norm1.num_features,
                                self.encoder.features.denseblock1.denselayer1.norm1.num_features]

        self.in_chans = cfg.model.in_chans
        # modify the first layer to accept 7 channels
        out_channels = self.encoder.features.conv0.out_channels
        kernel_size = self.encoder.features.conv0.kernel_size
        stride = self.encoder.features.conv0.stride
        padding = self.encoder.features.conv0.padding
        have_bias = self.encoder.features.conv0.bias is not None
        self.encoder.features.conv0 = nn.Conv2d(cfg.model.in_chans, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=have_bias)

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
    
    def get_feature_channels(self):
        """ get feature channels from the encoder, from deep to shallow """
        return self.feature_channels

class DTNet(nn.Module):
    def __init__(self, cfg):
        super(DTNet, self).__init__()
        
        # Encoder: Using a pre-trained DenseNet-161 model
        encoder_features = None
        if cfg.model.encoder == 'densenet':
            self.encoder = DensenetEncoder(cfg)
            # remove classifier layer
            # [Deprecated] comment the following line for old weights
            del self.encoder.encoder.classifier
        # resnet series
        elif cfg.model.encoder in ['resnet', 'resnext']:
            # default to resnet152
            self.encoder = ResnetEncoder(cfg)
            # remove the classifier layer
            # this has been done in the __init__ function
        else:
            raise NotImplementedError("Encoder not implemented {}".format(cfg.model.encoder))
        
        encoder_features = self.encoder.get_feature_channels()
        
        self.decoders = nn.ModuleList()
        # start with 1024 features in decoder
        out_chans = [cfg.model.out_chans] if isinstance(cfg.model.out_chans, int) else cfg.model.out_chans
        self.decoder_features = 1024
        for head_output_channels in out_chans:
            # Decoder: Using a custom decoder
            self.decoders.append(FullDecoder(encoder_features, self.decoder_features, 
                                            cfg.model.cnn.decoder_mid_dim, cfg.model.cnn.decoder_output_dim,
                                            output_channels=head_output_channels))
        
        
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
