# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block
import sys
sys.path.append('/home/arm-beast/Desktop/DenseTact-Model')
from models.dpt import DPTHead
from util.pos_embed import get_2d_sincos_pos_embed



class DTViTDPT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone and DPT (2021) head 
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, out_chans=7,
                 embed_dim=1024, depth=24, num_heads=16, features=256, use_bn=False,
                 out_dims=3, out_channels=[256, 512, 1024, 1024], use_clstoken=True,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        self.block_n = [2, 5, 8, 11]


        # --------------------------------------------------------------------------
        # ViT encoder specifics
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # TODO: I do not think we need cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # DPT decoder specifics

        self.depth_head = DPTHead(embed_dim, features, use_bn, output_dim=out_dims, 
                                out_channels=out_channels, use_clstoken=False)

        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.patch_size = patch_size

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.out_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.out_chans, h * p, h * p))
        return imgs


    def get_forward_features(self, x):
        # embed patches
        x = self.patch_embed(x)

        # append cls token 
        cls_token = self.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat((cls_token, x), dim=1)

        # add pos embed w/o cls token
        x = x + self.pos_embed
        
        outputs, total_block_len = [], len(self.blocks)

        blocks_to_take = range(total_block_len - self.block_n, total_block_len) if isinstance(self.block_n, int) else self.block_n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                outputs.append(x)
        
        outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1:] for out in outputs]

        return tuple(zip(outputs, class_tokens))


    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # append cls token 
        cls_token = self.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat((cls_token, x), dim=1)

        # add pos embed w/o cls token
        x = x + self.pos_embed
        
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_decoder(self, x, patch_h, patch_w):
        x = self.depth_head(x, patch_h, patch_w)
        return x

    def forward(self, imgs):
        patch_h, patch_w = imgs.shape[-2] // 16, imgs.shape[-1] // 16
        latent = self.get_forward_features(imgs)

        pred = self.forward_decoder(latent, patch_h, patch_w)  # [N, L, p*p*3]
        return pred
    
    def load_from_pretrained_model(self, model_path:str):
        """
        Load weights from a pre-trained model
        """
        model_ckpt = torch.load(model_path)

        print("Pretrained model epochs: ", model_ckpt["epoch"])

        # load encoder weights
        patch_embed_dict = {'.'.join(k.split('.')[2:]):v for k, v in model_ckpt["state_dict"].items() if 'patch_embed' in k}
        self.patch_embed.load_state_dict(patch_embed_dict)
        
        self.cls_token.data.copy_(model_ckpt["state_dict"]['model.cls_token'])
        self.pos_embed.data.copy_(model_ckpt["state_dict"]["model.pos_embed"])

        for i, blk in enumerate(self.blocks):
            block_dict = {'.'.join(k.split('.')[3:]):v for k, v in model_ckpt["state_dict"].items() if 'model.blocks' in k}
            blk.load_state_dict(block_dict)
        
    def freeze_encoder(self):
        """ Freeze the encoder weights """
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        
        self.cls_token.requires_grad = False

        for blk in self.blocks:
            blk.eval()
            for param in blk.parameters():
                param.requires_grad = False
    
    def unfreeze_encoder(self):
        """ Unfreeze the encoder """
        self.patch_embed.train()
        for param in self.patch_embed.parameters():
            param.requires_grad = True
        
        self.cls_token.requires_grad = True

        for blk in self.blocks:
            blk.train()
            for param in blk.parameters():
                param.requires_grad = True


class DTViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, out_chans=7,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # TODO I do not think we need cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        # LayerNorm is used here;
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * out_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.out_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.out_chans, h * p, h * p))
        return imgs

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # append cls token 
        cls_token = self.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat((cls_token, x), dim=1)

        # add pos embed w/o cls token
        x = x + self.pos_embed
        
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_decoder(self, x):
        # embed tokens
        x = self.decoder_embed(x)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward(self, imgs):
        latent = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent)  # [N, L, p*p*3]
        pred = self.unpatchify(pred)
        # loss = self.forward_loss(imgs, pred)
        return pred
    
    def load_from_pretrained_model(self, model_path:str):
        """
        Load weights from a pre-trained model
        """
        model_ckpt = torch.load(model_path)

        print("Pretrained model epochs: ", model_ckpt["epoch"])

        # load encoder weights
        patch_embed_dict = {'.'.join(k.split('.')[2:]):v for k, v in model_ckpt["state_dict"].items() if 'patch_embed' in k}
        self.patch_embed.load_state_dict(patch_embed_dict)
        
        self.cls_token.data.copy_(model_ckpt["state_dict"]['model.cls_token'])
        self.pos_embed.data.copy_(model_ckpt["state_dict"]["model.pos_embed"])

        for i, blk in enumerate(self.blocks):
            block_dict = {'.'.join(k.split('.')[3:]):v for k, v in model_ckpt["state_dict"].items() if 'model.blocks' in k}
            blk.load_state_dict(block_dict)
        
    def freeze_encoder(self):
        """ Freeze the encoder weights """
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        
        self.cls_token.requires_grad = False

        for blk in self.blocks:
            blk.eval()
            for param in blk.parameters():
                param.requires_grad = False
    
    def unfreeze_encoder(self):
        """ Unfreeze the encoder """
        self.patch_embed.train()
        for param in self.patch_embed.parameters():
            param.requires_grad = True
        
        self.cls_token.requires_grad = True

        for blk in self.blocks:
            blk.train()
            for param in blk.parameters():
                param.requires_grad = True


def dt_vit_base_patch16_dec512d8b(**kwargs):
    model = DTViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def dt_vit_dpt_base_patch16_dec512d8b(**kwargs):
    model = DTViTDPT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def dt_vit_large_patch16_dec512d8b(**kwargs):
    model = DTViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def dt_vit_huge_patch14_dec512d8b(**kwargs):
    model = DTViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
dt_vit_dpt_base_patch16 = dt_vit_dpt_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
dt_vit_base_patch16 = dt_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
dt_vit_large_patch16 = dt_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
dt_vit_huge_patch14 = dt_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks

if __name__ == '__main__':
    import torch

    model = dt_vit_dpt_base_patch16(img_size=256, in_chans=6, out_chans=7)

    # sample inference and move model to CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    x = torch.randn(1, 6, 256, 256).to(device)
    pred = model(x)
    print("Model: ", model)
    print("Pred shape: ", pred.shape)
