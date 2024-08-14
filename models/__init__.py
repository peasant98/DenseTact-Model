from .dt_vit import dt_vit_base_patch16, dt_vit_large_patch16, dt_vit_huge_patch14
from .dpt import DPTV2Net, HieraDPT
from .dense import DTDenseNet, DTNet

from .model_mae import mae_vit_base_patch16, mae_vit_large_patch16
from .hiera_mae import mae_hiera_base_256, mae_hiera_base_plus_256, mae_hiera_large_256

pretrain_dict = {
    "mae_hiera_base_256": mae_hiera_base_256,
    "mae_hiera_base_plus_256": mae_hiera_base_plus_256,
    "mae_hiera_large_256": mae_hiera_large_256,
    "mae_vit_base_patch16": mae_vit_base_patch16,
    "mae_vit_large_patch16": mae_vit_large_patch16
}

def build_model(cfg):

    # build model
    if cfg.model.name == "ViT":
        model = dt_vit_base_patch16(img_size=cfg.model.img_size, 
                        in_chans=cfg.model.in_chans, out_chans=cfg.model.out_chans)
    elif cfg.model.name == "DPT":
        model = DPTV2Net(img_size=cfg.model.img_size, 
                        in_chans=cfg.model.in_chans, out_dims=cfg.model.out_chans)
    elif cfg.model.name == "DenseNet":
        model = DTDenseNet(out_chans=[cfg.model.out_chans])
    elif cfg.model.name == "DenseNetV2":
        model = DTNet(in_chans=cfg.model.in_chans, out_chans=[cfg.model.out_chans], encoder=cfg.model.encoder)
    elif cfg.model.name == "HieraDPT":
        model = HieraDPT(cfg)
    else:
        raise NotImplementedError("Model not implemented {}".format(cfg.model.name))

    return model

