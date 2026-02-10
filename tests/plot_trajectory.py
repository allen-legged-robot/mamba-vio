import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.utils.kitti_utils import path_accu

# Configuration
seqs = ['05', '07', '10']  # Sequences to plot
save_dir = './plots'
os.makedirs(save_dir, exist_ok=True)

for seq in seqs:
    # Ground truth path (single)
    gt_path = f'logs/train/runs/MAMBA2_RPMG_6/tensorboard/version_0/{seq}_gt_poses.npy'

    # Multiple model predictions: {model_name: path_to_npy}
    model_paths = {
        'Mamba2_RPMG': f'logs/train/runs/MAMBA2_RPMG_6/tensorboard/version_0/{seq}_estimated_poses.npy',
        'Mamba2': f'logs/train/runs/MAMBA2_6/tensorboard/version_0/{seq}_estimated_poses.npy',
        'Transformer_RPMG': f'logs/train/runs/TF_RMGP/tensorboard/version_0/{seq}_estimated_poses.npy',
        'Transformer': f'logs/train/runs/TF/tensorboard/version_0/{seq}_estimated_poses.npy',
        'MLP': f'logs/train/runs/MLP/tensorboard/version_0/{seq}_estimated_poses.npy',
    }

    # Colors, line styles, and markers for different models
    colors = ['b', 'g', 'c', 'm', 'orange', 'purple', 'brown']
    linestyles = ['--', '-.', ':', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']  # circle, square, triangle-up, diamond, triangle-down, pentagon, star

    # Load ground truth
    gt_poses = np.load(gt_path)
    gt_mat = path_accu(gt_poses)
    x_gt = np.asarray([pose[0, 3] for pose in gt_mat])
    y_gt = np.asarray([pose[1, 3] for pose in gt_mat])
    z_gt = np.asarray([pose[2, 3] for pose in gt_mat])

    # Create figure with two subplots (XZ and XY views)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot ground truth on both subplots
    axes[0].plot(x_gt, z_gt, 'r-', linewidth=2, label='Ground Truth')
    axes[1].plot(x_gt, y_gt, 'r-', linewidth=2, label='Ground Truth')

    # Plot each model's prediction
    for i, (model_name, est_path) in enumerate(model_paths.items()):
        if not os.path.exists(est_path):
            print(f"Warning: {est_path} not found, skipping {model_name}")
            continue

        est_poses = np.load(est_path)
        est_mat = path_accu(est_poses)

        x_est = np.asarray([pose[0, 3] for pose in est_mat])
        y_est = np.asarray([pose[1, 3] for pose in est_mat])
        z_est = np.asarray([pose[2, 3] for pose in est_mat])

        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        marker = markers[i % len(markers)]
        # markevery: show marker every N points to avoid clutter
        markevery = max(1, len(x_est) // 20)
        axes[0].plot(x_est, z_est, color=color, linestyle=linestyle, marker=marker,
                     markevery=markevery, markersize=5, linewidth=1.5, label=model_name)
        axes[1].plot(x_est, y_est, color=color, linestyle=linestyle, marker=marker,
                     markevery=markevery, markersize=5, linewidth=1.5, label=model_name)

    # Configure XZ plot (top-down view)
    axes[0].set_xlabel('X (m)', fontsize=12)
    axes[0].set_ylabel('Z (m)', fontsize=12)
    axes[0].set_title(f'Sequence {seq} - XZ Trajectory (Top-Down View)', fontsize=14)
    axes[0].legend(loc='best')
    axes[0].axis('equal')
    axes[0].grid(True, alpha=0.3)

    # Configure XY plot (side view)
    axes[1].set_xlabel('X (m)', fontsize=12)
    axes[1].set_ylabel('Y (m)', fontsize=12)
    axes[1].set_title(f'Sequence {seq} - XY Trajectory (Side View)', fontsize=14)
    axes[1].legend(loc='best')
    axes[1].axis('equal')
    axes[1].grid(True, alpha=0.3)

    # Mark start and end points
    axes[0].plot(x_gt[0], z_gt[0], 'ko', markersize=8, label='Start')
    axes[0].plot(x_gt[-1], z_gt[-1], 'k^', markersize=8, label='End')

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(save_dir, f'{seq}_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_path}")

    # plt.show()
