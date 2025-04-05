from yacs.config import CfgNode as CN

# Define the default configuration using YACS CfgNode
cfg = CN()


# Encoder settings
cfg.encoder = CN()
cfg.encoder.type = "vit_base"
cfg.encoder.img_size = 224
cfg.encoder.in_chans = 3
cfg.encoder.patch_size = 16
cfg.encoder.num_register_tokens = 1
cfg.encoder.drop_path_rate = 0.1
cfg.encoder.drop_path_uniform = True
cfg.encoder.init_values = 1e-5  # Layerscale init values

# DINO head settings
cfg.dino_head = CN()
cfg.dino_head._partial_ = True
cfg.dino_head.type = "DINOHead"
cfg.dino_head.out_dim = 65536

# Masking settings
cfg.num_global_masks = 2
cfg.num_local_masks = 8
cfg.global_mask_scale = [0.32, 1.0]
cfg.local_mask_scale = [0.1, 0.32]
cfg.moving_average_decay = 0.994
cfg.allow_mask_overlap = True
cfg.teacher_temp = [0.04, 0.06]
cfg.ibot_separate_head = True
cfg.teacher_warmup_epochs = 30

# Optimizer settings
cfg.optim_cfg = CN()
cfg.optim_cfg._partial_ = True
cfg.optim_cfg.type = "torch.optim.AdamW"
cfg.optim_cfg.lr = 1e-4
cfg.optim_cfg.weight_decay = 0.05

# Learning rate scheduler settings
cfg.lr_scheduler_cfg = CN()
cfg.lr_scheduler_cfg._partial_ = True
cfg.lr_scheduler_cfg.type = "custom_scheduler.WarmupCosineScheduler"
cfg.lr_scheduler_cfg.steps_per_epoch = None  # Marked as ??? in original
cfg.lr_scheduler_cfg.T_max = None  # Marked as ??? in original
cfg.lr_scheduler_cfg.final_lr = 1.0e-6
cfg.lr_scheduler_cfg.start_lr = 1e-5
cfg.lr_scheduler_cfg.warmup_epochs = 30

# Weight decay scheduler settings
cfg.wd_scheduler_cfg = CN()
cfg.wd_scheduler_cfg._partial_ = True
cfg.wd_scheduler_cfg.type = "custom_scheduler.CosineWDSchedule"
cfg.wd_scheduler_cfg.final_weight_decay = 0.4
cfg.wd_scheduler_cfg.ref_weight_decay = 0.04
cfg.wd_scheduler_cfg.T_max = None  # Marked as ??? in original

# Online probes
cfg.online_probes = CN()
cfg.online_probes[0] = CN()
cfg.online_probes[0].type = "probe.OnlineProbeModule"
cfg.online_probes[0].probe_name = "reconstruction"
cfg.online_probes[0].decoder = CN()
cfg.online_probes[0].decoder.type = "probes.DecoderImage"
cfg.online_probes[0].decoder.in_chans = 3
cfg.online_probes[0].decoder.patch_size = 16
cfg.online_probes[0].decoder.input_embed_dim = 768
cfg.online_probes[0].decoder.embed_dim = 192
cfg.online_probes[0].decoder.depth = 4
cfg.online_probes[0].loss_fn = CN()
cfg.online_probes[0].loss_fn.loss = "torch.nn.MSELoss"

# Online probes learning rates and logging frequency
cfg.online_probes_lrs = [1e-4]
cfg.log_freq_reconstruction = 1000

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    return cfg.clone()

