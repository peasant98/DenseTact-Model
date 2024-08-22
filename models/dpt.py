import cv2
import torch
import math
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from models.dinov2 import DINOv2
from models.util.blocks import FeatureFusionBlock, _make_scratch

# Hiera Encoder
from models.hiera_layers.hiera import Hiera, HieraBlock
from models.hiera_layers.hiera_utils import conv_nd


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.LeakyReLU(),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class PermuteLayer(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)

class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.conv_block(x)


class DPTHead(nn.Module):
    def __init__(
        self, 
        in_channels, 
        features=256, 
        use_bn=False, 
        output_dim=15,
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False
    ):
        super(DPTHead, self).__init__()
        
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )
    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 16), int(patch_w * 16)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        
        return out

class HieraDPTHead(nn.Module):
    """ DPT Head for Hiera Encoder """
    def __init__(self,
                features,
                in_channels,
                out_channels,
                output_dim,
                use_bn=False, 
                ):
        """
            Args:
                features: int, number of features inside the head
                in_channels: list, number of channels for intermediate features
                out_channels: list, number of output channels
                output_dim: int, output dimension of the model
                use_bn: bool, use batch normalization
        """
        super().__init__()

        n_layers = len(in_channels)
        assert n_layers == len(out_channels), "Number of input and output channels must be the same"

        # head to project features
        self.projects = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(in_chan),
                PermuteLayer((0, 3, 1, 2)),
                nn.Conv2d(
                    in_channels=in_chan,
                    out_channels=features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            ) for in_chan, out_channel in zip(in_channels, out_channels)
        ])

        # self.conv_rn = nn.ModuleList([
        #     nn.Conv2d(out_channel, features, kernel_size=3, stride=1, padding=1, bias=False, groups=1) for out_channel in out_channels
        # ])

        self.refinement_blocks = nn.ModuleList([
            _make_fusion_block(features, use_bn) for _ in out_channels
        ])

        # remove the resConf in the last block, since we don't need it
        del self.refinement_blocks[-1].resConfUnit1

        head_features_1 = features
        head_features_2 = 32
        
        self.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            # maybe LeakyReLU here
            nn.LeakyReLU(),
            # nn.BatchNorm2d(head_features_2),
            nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, out_features):
        out = []
        for i, x in enumerate(out_features):
            x = self.projects[i](x)
            # x = self.conv_rn[i](x)
            out.append(x)

        # fuse the feature hierarchically
        prev = None
        for layer_id in range(len(out) - 1, -1, -1):
            x = out[layer_id]
            inputs = [x] if prev is None else [prev, x]
            size = None if layer_id == 0 else out[layer_id - 1].shape[2:]
            prev = self.refinement_blocks[layer_id](*inputs, size=size)

        out = self.output_conv1(prev)
        out = F.interpolate(out, (256, 256), mode="bilinear", align_corners=True)
        out = self.output_conv2(out)

        return out
    
class VanillaHieraFusionDecoder(nn.Module):
    def __init__(self,   
                in_channels,
                output_dim,
                norm_layer = partial(nn.LayerNorm, 1e-6),
                decoder_embed_dim = 256, 
                decoder_depth = 3,
                mlp_ratio = 4.0,
                q_pool = 2,
                mask_unit_size = (8, 8),
                q_stride = 2,
                patch_stride = (4, 4),
                tokens_spatial_shape = (16, 16),
                stage_ends = [2, 3, 16, 3],
                decoder_num_heads = 4) -> None:
        """ Vanilla Hiera Fusion Decoder """
        super().__init__()

        curr_mu_size = mask_unit_size
        self.multi_scale_fusion_heads = nn.ModuleList()

        mask_unit_spatial_shape_final = [
            i // s ** (q_pool) for i, s in zip(mask_unit_size, q_stride)
        ]

        tokens_spatial_shape_final = [
            i // s ** (q_pool)
            for i, s in zip(tokens_spatial_shape, q_stride)
        ]

        for idx, i in enumerate(stage_ends[:q_pool]):  # resolution constant after q_pool
            kernel = [
                i // s for i, s in zip(curr_mu_size, mask_unit_spatial_shape_final)
            ]
            curr_mu_size = [i // s for i, s in zip(curr_mu_size, q_stride)]
            self.multi_scale_fusion_heads.append(
                conv_nd(len(q_stride))(
                    in_channels[idx],
                    in_channels[-1],
                    kernel_size=kernel,
                    stride=kernel,
                )
            )

        self.multi_scale_fusion_heads.append(nn.Identity())  # final stage, no transform
        self.fusion_norm = norm_layer(in_channels[-1])

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(in_channels[-1], decoder_embed_dim)

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(
                1, math.prod(tokens_spatial_shape_final), decoder_embed_dim
            )
        )

        self.decoder_blocks = nn.ModuleList(
            [
                HieraBlock(
                    dim=decoder_embed_dim,
                    dim_out=decoder_embed_dim,
                    heads=decoder_num_heads,
                    norm_layer=norm_layer,
                    mlp_ratio=mlp_ratio,
                )
                for i in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.pred_stride = patch_stride[-1] * (
            q_stride[-1] ** q_pool
        )  # patch stride of prediction

        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            (self.pred_stride ** min(2, len(q_stride))) * output_dim,
        )  # predictor
        # --------------------------------------------------------------------------

        self.output_dim = output_dim

        # TODO Initialize weights

    def forward(self, intermediates):
        # Multi-scale fusion
        x = 0.0
        for head, interm_x in zip(self.multi_scale_fusion_heads, intermediates):
            # since output is in spatial format, forward conv layer to fuse
            interm_x = interm_x.permute(0, 3, 1, 2)
            x += head(interm_x)

        # change to (B, H, W, C) format
        x = x.permute(0, 2, 3, 1)
        x = self.fusion_norm(x)

        # Embed tokens
        x = self.decoder_embed(x)

        # Combine visible and mask tokens

        # Flatten
        x = x.reshape(x.shape[0], -1, x.shape[-1])

        # Add pos embed
        x = x + self.decoder_pos_embed

        # Apply decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Predictor projection
        x = self.decoder_pred(x)

        # reshape to spatial format
        x = self.reconstruct_img(x)

        return x
    
    def reconstruct_img(self, pred:torch.Tensor):
        # TODO Set Spatial Size
        size = self.pred_stride
        pred = pred.reshape(-1, 256 // size, 256 // size, self.output_dim, size, size)

        pred = pred.permute(0, 1, 4, 2, 5, 3)
        pred = pred.reshape(-1, 256, 256, self.output_dim)

        pred = pred.permute(0, 3, 1, 2)
        return pred
    

class DPTV2Net(nn.Module):
    def __init__(
        self, 
        img_size=256,
        patch_size=16,
        in_chans=3,
        encoder='vitl', 
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        out_dims=15,
        use_bn=False, 
        use_clstoken=False
    ):
        super(DPTV2Net, self).__init__()
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder, img_size=img_size, in_chans=in_chans, patch_size=patch_size)
        self.depth_head = DPTHead(self.pretrained.embed_dim, features, use_bn, output_dim=out_dims, 
                                out_channels=out_channels, use_clstoken=use_clstoken)
    
    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 16, x.shape[-1] // 16
        # extract intermediate features.
        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        # predictions
        depth = self.depth_head(features, patch_h, patch_w)

        return depth
    
    def freeze_encoder(self):
        for param in self.pretrained.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.pretrained.parameters():
            param.requires_grad = True

    def load_from_pretrained_model(self, model_path:str):
        """
        Load weights from a pre-trained model
        """
        model_ckpt = torch.load(model_path)

        # load encoder weights
        self.pretrained.load_state_dict(model_ckpt["model"]['pretrained'])

class HieraDPT(nn.Module):
    def __init__(
        self, 
        cfg
    ):
        super().__init__()

        in_chans = cfg.model.in_chans
        out_dims = cfg.model.out_chans
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        encoder_kwargs = dict(
            input_size=(cfg.model.img_size, cfg.model.img_size),
            embed_dim=cfg.model.hiera.embed_dim, 
            num_heads=cfg.model.hiera.num_heads, 
            stages=cfg.model.hiera.stages, 
            q_pool=cfg.model.hiera.q_pool, 
            patch_stride=cfg.model.hiera.patch_stride,
            mlp_ratio=cfg.model.hiera.mlp_ratio)

        # define the Hiera Encoder
        self.encoder = Hiera(
            in_chans=in_chans,
            norm_layer=norm_layer,
            **encoder_kwargs)
        
        del self.encoder.norm, self.encoder.head

        # define the Hiera Decoder
        encoder_channels = self.encoder.get_out_dim(True)
        encoder_channels = encoder_channels[:self.encoder.q_pool] + encoder_channels[-1:]

        if cfg.model.hiera.decoder == "DPT":
            self.decoder = HieraDPTHead(cfg.model.hiera.decoder_embed_dim, 
                                        encoder_channels, out_channels=cfg.model.hiera.decoder_mapping_channels, 
                                        output_dim=out_dims, use_bn=cfg.model.hiera.use_bn)
       
        elif cfg.model.hiera.decoder == "Vanilla":

            decoder_kwargs = dict(
                decoder_embed_dim=cfg.model.hiera.decoder_embed_dim, 
                decoder_num_heads=cfg.model.hiera.decoder_num_heads, 
                q_pool=cfg.model.hiera.q_pool,
                mlp_ratio=cfg.model.hiera.mlp_ratio,
                mask_unit_size=self.encoder.mask_unit_size,
                patch_stride = self.encoder.patch_stride,
                q_stride = self.encoder.q_stride,
                stage_ends = self.encoder.stage_ends,
                tokens_spatial_shape = self.encoder.tokens_spatial_shape,
                decoder_depth=cfg.model.hiera.decoder_depth)

            self.decoder = VanillaHieraFusionDecoder(encoder_channels, output_dim=out_dims, norm_layer=norm_layer, 
                                                     **decoder_kwargs)
        

    def forward(self, x):
        _, intermediates = self.encoder(x, None, return_intermediates=True)
        intermediates = intermediates[: self.encoder.q_pool] + intermediates[-1:]

        # predictions
        depth = self.decoder(intermediates)

        return depth    

    def freeze_encoder(self):
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        self.encoder.train()
        for param in self.encoder.parameters():
            param.requires_grad = True

    def load_from_pretrained_model(self, model_path:str):
        model_ckpt = torch.load(model_path)

        # load encoder weights
        state_dict = {}
        for k in self.encoder.state_dict().keys():
            if 'model.' + k in model_ckpt["state_dict"]:
                state_dict[k] = model_ckpt["state_dict"]['model.' + k]
        
        res = self.encoder.load_state_dict(state_dict)
        print(res)


if __name__ == "__main__":
    from configs import get_cfg_defaults

    # load config
    cfg = get_cfg_defaults()
    cfg.merge_from_file("configs/VanillaHiera_depth_subset.yaml")

    model = HieraDPT(cfg)
    model.cuda()

    x = torch.randn(1, 7, 256, 256).cuda()
    out = model(x)
