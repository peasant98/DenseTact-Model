from .dt_vit import dt_vit_base_patch16, dt_vit_large_patch16, dt_vit_huge_patch14
from .dpt import DPTV2Net
from .dense import DTDenseNet

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
    else:
        raise NotImplementedError("Model not implemented {}".format(cfg.model.name))

    return model

