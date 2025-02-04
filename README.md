# DenseTact Model

One DenseTact Model to rule them all.

We present the most complete tactile sensor; predicting displacement, shear, stress, force.

# Train the model 

```bash
# train from sketch
python train_ViT_lightning.py --config configs/your_config.yaml --exp_name /path/to/expname

# resume training
python train_ViT_lightning.py --config configs/your_config.yaml --exp_name /path/to/expname --ckpt_path /path/to/checkpoint

# finetune the encoder
python train_ViT_lightning.py --config configs/your_config.yaml --exp_name /path/to/expname --ckpt_path /path/to/checkpoint --finetune

# evaluation
python train_ViT_lightning.py --config configs/your_config.yaml --exp_name /path/to/expname --ckpt_path /path/to/checkpoint --eval
```

Note:

1. train / val split is controlled by the seed in the configuration file. To reproduce results, keep the seed as default. 

2. 

# TODOS

1. Q Upsampling for Hiera Decoder
