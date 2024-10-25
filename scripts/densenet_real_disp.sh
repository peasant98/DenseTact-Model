
python3 train_ViT_lightning.py --config configs/densenet_real_disp.yaml \
--exp_name exp/densenet_real_disp_AdamW_10_24  \


# python3 train_ViT_lightning.py --ckpt_path 'exp/densenet_real_depth_finetune_10_22/checkpoints/dt_model-epoch=94-AUC=71.73.ckpt' --finetune --config configs/densenet_real_disp.yaml \
# --exp_name exp/densenet_real_disp_AdamW_pretrain --real_world \
