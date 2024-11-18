from yacs.config import CfgNode as CN

# Define the default configuration using YACS CfgNode
cfg = CN()

cfg.epochs = 100
cfg.batch_size = 16
cfg.num_workers = 16
cfg.dataset_ratio = 1.
# seed for every thing
# include dataset split
cfg.seed = 42
cfg.scale = 1.

cfg.gradient_clip_val = None
cfg.gradient_clip_algorithm = None

cfg.loss = CN()
cfg.loss.stress_weight = 1.
cfg.loss.disp_weight = 1.
cfg.loss.depth_weight = 1.

cfg.model = CN()
cfg.model.name = "DPT"
cfg.model.pretrained_model = ""
cfg.model.img_size = 256
cfg.model.patch_size = 16
cfg.model.in_chans = 7
cfg.model.encoder = 'vitl'
cfg.model.out_chans = [1]
cfg.model.loss = "L1"
cfg.model.backbone = ""
cfg.model.imagenet_pretrained = False
cfg.model.LoRA = False
cfg.model.LoRA_rank = 4
cfg.model.LoRA_scale = 1

cfg.model.cnn = CN()
cfg.model.cnn.decoder_mid_dim = [1024, 1024, 512, 256, 64]
cfg.model.cnn.decoder_output_dim = [1024, 512, 256, 128, 64]

# parameters for Hiera model
cfg.model.hiera = CN()
# this is the defaut for encoder
cfg.model.hiera.embed_dim = 96
cfg.model.hiera.num_heads = 1
cfg.model.hiera.stages = [2, 3, 16, 3]
cfg.model.hiera.q_pool = 2
cfg.model.hiera.patch_stride = [4, 4]
cfg.model.hiera.mlp_ratio = 4.0

cfg.model.hiera.decoder = "DPT"
# for DPT Decoder Head
cfg.model.hiera.use_bn = False
cfg.model.hiera.activation = "leaky_relu"
cfg.model.hiera.decoder_embed_dim = 256
cfg.model.hiera.decoder_mapping_channels = [256, 512, 1024]
cfg.model.hiera.decoder_depth_per_stage = 4

# for Vanilla model (decoder = "Vanilla")
cfg.model.hiera.decoder_num_heads = 1
cfg.model.hiera.decoder_depth = 3

cfg.dataset = CN()
cfg.dataset.output_type = []
cfg.dataset.contiguous_on_direction = False

cfg.optimizer = CN()
cfg.optimizer.name = "Adam"
cfg.optimizer.lr = 1e-3
cfg.optimizer.eta_min = 1e-6

cfg.scheduler = CN()
# choose from cosine, linear_cosine
cfg.scheduler.name = "cosine"
# steps for linear warmup 
cfg.scheduler.warmup = 2000
# cycles for trainig (2k + 1)
cfg.scheduler.cycle_k = 0

cfg.metric = CN()
# value under this will be considered as negative
cfg.metric.PN_thresh = 0.0001
# rel error ratio (%) under this will be considered as true positive
cfg.metric.TF_rel_error_rate = 1.
# abs error ratio (m) under this will be considered as true positive
cfg.metric.TF_abs_error_thresh = 0.001
# For the curve
cfg.metric.num_bins = 10


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return cfg.clone()