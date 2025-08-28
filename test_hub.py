import numpy as np

import torch
import hubconf
from torchvision import transforms
import matplotlib.pyplot as plt


from configs import get_cfg_defaults

from process_data import FullDataset, FEAT_CHANNEL

LOCAL_CONFIG_PATH = 'configs/dt_ultra.yaml' 

def denormalize_output(pred, cfg, output_names):
    """Denormalize model predictions following train_ViT_lightning.py"""
    pred_denorm = pred.clone()
    
    for idx, name in enumerate(output_names):
        scale = 1
        if 'disp' in name:
            scale = cfg.scales.disp
        elif 'stress1' in name:
            scale = cfg.scales.stress
        elif 'stress2' in name:
            scale = cfg.scales.stress2
        elif 'depth' in name:
            scale = cfg.scales.depth
        elif 'cnorm' in name:
            scale = cfg.scales.cnorm
        elif 'shear' in name:
            scale = cfg.scales.area_shear
        
        pred_denorm[:, idx, :, :] /= scale
    
    return pred_denorm

def get_output_names(cfg):
    """Get output names following train_ViT_lightning.py logic"""
    output_names = []
    dataset_output_type = cfg.dataset.output_type
    
    if cfg.dataset.contiguous_on_direction:
        # depth always comes first
        if "depth" in dataset_output_type:
            output_names.append("depth")

        # add the other channels
        for d in ["x", "y", "z"]:
            for t in dataset_output_type:
                if t == "depth":
                    continue
                else:
                    # extend the three channels
                    output_names.append(f"{t}_{d}")
    
    else:
        for t in dataset_output_type:
            if t == "depth":
                output_names.append("depth")
            else:
                for d in ["x", "y", "z"]:
                    output_names.append(f"{t}_{d}")
    
    return output_names

if __name__ == '__main__':
    tactile_model = hubconf.hiera()
    # get calibration output from the tactile image.    

    # send model to cuda
    tactile_model = tactile_model.cuda()

    # get item from dataset
    cfg = get_cfg_defaults()
    cfg.merge_from_file(LOCAL_CONFIG_PATH)

    extra_samples_dirs = ['/arm/u/maestro/Desktop/DenseTact-Model/es1t/dataset_local/', 
                          '/arm/u/maestro/Desktop/DenseTact-Model/es2t/es2t/dataset_local/',
                          '/arm/u/maestro/Desktop/DenseTact-Model/es3t/es3t/dataset_local/']
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((cfg.model.img_size, cfg.model.img_size), antialias=True),
    ])

    X_transform = transforms.Compose([
        transforms.Resize((cfg.model.img_size, cfg.model.img_size), antialias=True),
    ])

    dataset_dir = '/arm/u/maestro/Desktop/DenseTact-Model/es4t/es4t/dataset_local/'

    real_world = False

    dataset = FullDataset(cfg, samples_dir=dataset_dir, transform=transform, X_transform=X_transform, 
                          extra_samples_dirs=extra_samples_dirs, is_real_world=real_world)

    X, y = dataset[100]
    orig_deform = X[:3, :, :].clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0)
    orig_undeform = X[ 3:6, :, :].clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(orig_deform)
    axes[0, 0].set_title('Original Deformed')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(orig_undeform)
    axes[0, 1].set_title('Original Undeformed')
    axes[0, 1].axis('off')

    # save figure
    plt.savefig("sample_image_inference.png")
    plt.close()

    # Get output names for denormalization
    output_names = get_output_names(cfg)
    print(f"Output names: {output_names}")
    
    tactile_model.eval()

    # Add batch dimension and send to cuda
    X = X.unsqueeze(0).cuda()  # Shape: (1, 6, 256, 256)
    print(f"Input shape: {X.shape}")

    with torch.no_grad():
        output = tactile_model(X)
        print(f"Raw output shape: {output.shape}")  # Should be (1, 15, 256, 256)
        
        # Denormalize output
        output_denorm = denormalize_output(output, cfg, output_names)
        print(f"Denormalized output shape: {output_denorm.shape}")

    # Visualize all outputs
    output_vis = output_denorm[0].detach().cpu().numpy()  # Remove batch dimension (15, 256, 256)
    
    # Create visualization grid
    num_channels = output_vis.shape[0]
    cols = 5  # Show 5 columns
    rows = (num_channels + cols - 1) // cols  # Calculate rows needed
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4*rows))
    axes = axes.flatten() if rows > 1 else [axes] if rows == 1 else axes
    
    for i in range(num_channels):
        ax = axes[i]
        
        # Apply colormap based on output type
        if 'depth' in output_names[i]:
            im = ax.imshow(output_vis[i], cmap='viridis', vmin=0, vmax=output_vis[i].max())
        elif 'stress' in output_names[i] or 'disp' in output_names[i]:
            # Use diverging colormap for stress/displacement (can be positive/negative)
            vmax = max(abs(output_vis[i].min()), abs(output_vis[i].max()))
            im = ax.imshow(output_vis[i], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(output_vis[i], cmap='viridis')
        
        ax.set_title(f'{output_names[i]}\nRange: [{output_vis[i].min():.3f}, {output_vis[i].max():.3f}]')
        ax.axis('off')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Hide unused subplots
    for i in range(num_channels, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('tactile_output_visualization.png', dpi=150, bbox_inches='tight')
    
    print("Visualization saved as 'tactile_output_visualization.png'")