# pytorch hub configuration file
# hubconf.py
dependencies = ["torch", "yacs", "pyyaml"]

import torch
from torch.hub import load_state_dict_from_url
from configs import get_cfg_defaults


from models import build_model, replace_LoRA, MonkeyPatchLoRALinear, HieraDPT


# weights of calibration model
_WEIGHTS = {
    "base": "https://github.com/peasant98/DenseTact-Model/releases/download/models/last.ckpt"
}

LOCAL_CONFIG_PATH = 'configs/dt_ultra.yaml' 

def hiera(pretrained=True, variant="base", map_location="cpu", **kwargs):

    cfg = get_cfg_defaults()
    cfg.merge_from_file(LOCAL_CONFIG_PATH)

    model = build_model(cfg)
    model.replace_LoRA(4, 1)

    if pretrained:
        state = load_state_dict_from_url(
            _WEIGHTS[variant],
            map_location=map_location,
            check_hash=True  # set to False if your filename has no hash
        )

        # Lightning .ckpt vs plain state_dict
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        # Strip common wrappers
        cleaned = {}
        for k, v in state.items():
            k2 = k.replace("model.", "").replace("module.", "")
            cleaned[k2] = v


        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing or unexpected:
             print(f"[hub] missing keys: {missing}\n[hub] unexpected keys: {unexpected}")

    return model
