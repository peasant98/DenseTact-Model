from yacs.config import CfgNode as CN

# Define the default configuration using YACS CfgNode
cfg = CN()

cfg.epochs = 100
cfg.batch_size = 32
cfg.num_workers = 16
cfg.dataset_ratio = 1.
# set this greater than 1.0 to disable finetune of encoder
cfg.finetune_ratio = 0.8

cfg.model = CN()
cfg.model.name = "DPT"
cfg.model.pretrained_model = ""
cfg.model.img_size = 256
cfg.model.patch_size = 16
cfg.model.in_chans=7
cfg.model.encoder='vitl'
cfg.model.out_chans=15
cfg.model.loss = "L1"

cfg.dataset = CN()
cfg.dataset.output_type = "depth"

cfg.optimizer = CN()
cfg.optimizer.name = "Adam"
cfg.optimizer.lr = 1e-3

cfg.metric = CN()
# value under this will be considered as negative
cfg.metric.PN_thresh = 0.000001
# rel error ratio (%) under this will be considered as true positive
cfg.metric.TF_rel_error_rate = 1.
# For the curve
cfg.metric.num_bins = 10


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return cfg.clone()