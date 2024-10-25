

python3 train_ViT_lightning.py --ckpt_path 'exp/densenet_sim_depth_10_22/checkpoints/dt_model-epoch=95-AUC=85.14.ckpt' --finetune --config configs/densenet_depth_subset_real_world.yaml \
--exp_name exp/densenet_real_depth_finetune_10_22 --real_world 
