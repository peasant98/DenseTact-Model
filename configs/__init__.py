from yacs.config import CfgNode as CN

# Define the default configuration using YACS CfgNode
cfg = CN()

cfg.epochs = 100
cfg.batch_size = 32
cfg.num_workers = 32
cfg.dataset_ratio = 1.
cfg.learning_rate = 0.001
cfg.finetune_ratio = 0.8

cfg.model = CN()
cfg.model.name = "DPT"
cfg.model.pretrained_model = "none"
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

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return cfg.clone()