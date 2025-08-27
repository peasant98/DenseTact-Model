import torch


from pretraining_configs import get_cfg_defaults


import pdb; pdb.set_trace()  # noqa: E402

if __name__ == "__main__":

    # load config
    cfg = get_cfg_defaults()
    cfg.merge_from_file("pretraining_configs/config.yaml")

    # step 1: construct loss
    loss = torch.nn.MSELoss()


    # step 1: construct encoder
