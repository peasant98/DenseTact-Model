import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=7, patch_size=16, emb_size=768, img_size=256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.img_size = img_size
        self.emb_size = emb_size

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'b e (h) (w) -> b (h w) e')
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, emb_size=768, depth=12, num_heads=12, mlp_dim=3072, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=emb_size, nhead=num_heads, dim_feedforward=mlp_dim, dropout=dropout)
            for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ViT(nn.Module):
    def __init__(self, in_channels=7, patch_size=16, emb_size=768, depth=12, num_heads=12, mlp_dim=3072, img_size=256):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embed = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, emb_size))
        self.transformer = TransformerEncoder(emb_size, depth, num_heads, mlp_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 e -> b 1 e', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed[:, :(n + 1)]
        x = self.transformer(x)
        return x[:, 1:]

class CNNDecoder(nn.Module):
    def __init__(self, emb_size=768, img_size=256, out_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(emb_size, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, out_channels, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.img_size = img_size

    def forward(self, x):
        x = rearrange(x, 'b (h w) e -> b e h w', h=self.img_size // 16)
        x = self.up(F.relu(self.conv1(x)))
        x = self.up(F.relu(self.conv2(x)))
        x = self.up(F.relu(self.conv3(x)))
        x = self.up(self.conv4(x))
        return x

class MultiHeadDecoder(nn.Module):
    def __init__(self, emb_size=768, img_size=256, num_heads=5, out_channels=3):
        super().__init__()
        self.decoders = nn.ModuleList([CNNDecoder(emb_size, img_size, out_channels) for _ in range(num_heads)])

    def forward(self, x):
        return torch.stack([decoder(x) for decoder in self.decoders], dim=1)

class VisionTransformerWithMultiHeadDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ViT()
        self.decoder = MultiHeadDecoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Instantiate and test the model
model = VisionTransformerWithMultiHeadDecoder()
input_tensor = torch.randn(1, 7, 256, 256)
output = model(input_tensor)
print(output.shape)  # Should be [1, 5, 3, 256, 256]
