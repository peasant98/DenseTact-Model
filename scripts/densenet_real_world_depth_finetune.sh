

python3 train_ViT_lightning.py --ckpt_path 'exp/densenet_depth_sim_depth_all_batch_norm/checkpoints/dt_model-epoch=99-AUC=85.98.ckpt' --finetune --config configs/densenet_depth_subset_real_world.yaml \
--exp_name exp/densenet_real_depth_finetune --real_world 