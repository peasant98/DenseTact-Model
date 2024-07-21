import torch
import torch.nn as nn
import timm
from einops import rearrange, repeat

class DTBaseModel(nn.Module):
    def __init__(self):
        super(DTBaseModel, self).__init__()
        
        # Encoder: Using a pre-trained Vision Transformer model from timm
        self.encoder = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.encoder.patch_embed.proj = nn.Conv2d(7, self.encoder.embed_dim, kernel_size=16, stride=16)
        self.encoder.head = nn.Identity()  # Removing the classification head
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.encoder.embed_dim, 1024, kernel_size=4, stride=4, padding=0),
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
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True)
        )
        
        # Multi-head output layers
        self.head1 = nn.Conv2d(128, 3, kernel_size=1)
        self.head2 = nn.Conv2d(128, 3, kernel_size=1)
        self.head3 = nn.Conv2d(128, 3, kernel_size=1)
        self.head4 = nn.Conv2d(128, 3, kernel_size=1)
        self.head5 = nn.Conv2d(128, 3, kernel_size=1)

    def forward(self, x):
        # resize input to 224x224
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear')
        # Encoding
        x = self.encoder.patch_embed(x)
        print(x.shape)
        cls_tokens = self.encoder.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.encoder.pos_drop(x + self.encoder.pos_embed)
        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.encoder.norm(x)
        x = x[:, 1:, :].permute(0, 2, 1).contiguous()
        
        # Reshape for decoder
        h = w = int(x.size(1) ** 0.5)  # Calculate the original height and width
        x = rearrange(x, 'b (h w) e -> b e h w', h=h, w=w)
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
