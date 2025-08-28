# DenseTact Model

One DenseTact Model to rule them all.

Use Torch Hub to load our models! It's easy!


```sh
pip install torch torchvision yacs timm
```
```python
>>> import torch
>>> model = torch.hub.load('peasant98/DenseTact-Model', 'hiera', pretrained=True, map_location='cpu', trust_repo=True)
>>> model = model.cuda()
```

We have a demo to run on sample images:

We also provide steps for running the encoder, which can be found in the file!

```sh
python3 test_hub.py
```


We present the most complete optical tactile sensor; predicting displacement, shear, stress, force.

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
