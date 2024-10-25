

python3 train_ViT_lightning.py --config configs/densenet_depth_subset.yaml \
--exp_name exp/densenet_sim_depth_10_22\

# python3 train_ViT_lightning.py --config configs/densenet_depth_subset_real_world.yaml \
# --exp_name exp/densenet_depth_subset_real_world \
# --ckpt_path exp/densenet_depth_subset_scratch_exp/checkpoints/dt_model-epoch=48-AUC=79.16.ckpt \
# --real_world --finetune