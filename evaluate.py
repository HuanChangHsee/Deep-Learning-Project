
"""
evaluate.py

Loads trained GCN and MeshGraphNets checkpoints and evaluates both models on
the test set. Generates figures and prints a results table for the report.

Outputs (saved to results/):
    - training_curves.png     : train loss and val NRMSE over epochs
    - rollout_nrmse.png       : rollout NRMSE over timesteps for both models
    - flow_visualization.png  : predicted vs ground truth flow fields
    - results_table.txt       : one-step and rollout NRMSE for both models

Usage:
    python evaluate.py
    python evaluate.py --checkpoint_dir checkpoints --results_dir results
"""

import os
import pickle
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import sys
sys.path.append('/content/Deep-Learning-Project')

from utils.dataset import get_splits, CylinderFlowDataset
from utils.metrics import nrmse
from models.gcn import GCN
from models.mgn import MeshGraphNets


# Argument parsing

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_path', type=str,
                        default='/content/Deep-Learning-Project/data/train_downsampled_labeled.h5')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='/content/Deep-Learning-Project/checkpoints')
    parser.add_argument('--results_dir', type=str,
                        default='/content/Deep-Learning-Project/results')
    return parser.parse_args()


# Load model from checkpoint

def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt['args']

    if ckpt['model'] == 'gcn':
        model = GCN(
            in_dim=7,
            hidden_dim=args['hidden_dim'],
            out_dim=3,
            num_layers=args['num_layers']
        ).to(device)
    else:
        model = MeshGraphNets(
            in_node_dim=7,
            in_edge_dim=3,
            hidden_dim=args['hidden_dim'],
            out_dim=3,
            num_layers=args['num_layers']
        ).to(device)

    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f"Loaded {ckpt['model'].upper()} from epoch {ckpt['epoch']} "
          f"(val_nrmse: {ckpt['val_nrmse']:.6f})")
    return model


# One-step NRMSE on test set

def compute_one_step_nrmse(model, test_ds, device, batch_size=32):
    from torch_geometric.loader import DataLoader
    loader = DataLoader(test_ds, batch_size=batch_size,
                        shuffle=False, num_workers=0)
    total, count = 0.0, 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data)
            total += nrmse(pred, data.y).item()
            count += 1
    return total / count


# Rollout NRMSE for one simulation

def compute_rollout(model, sim_data_list, device):
    """
    Autoregressively roll out one full simulation.
    sim_data_list: list of Data objects for one simulation, ordered t=0..T-1

    Returns:
        nrmse_curve : list of NRMSE values at each timestep
        preds       : list of (N,3) predicted states (normalized)
        targets     : list of (N,3) ground truth states (normalized)
    """
    model.eval()
    nrmse_curve = []
    preds, targets = [], []

    current_state = sim_data_list[0].x[:, :3].clone().to(device)

    with torch.no_grad():
        for data in sim_data_list:
            data = data.to(device)
            # Replace fluid state with rolled-out prediction
            data.x = torch.cat([current_state, data.x[:, 3:]], dim=-1)

            pred = model(data)          # (N, 3)
            target = data.y             # (N, 3)

            err = nrmse(pred, target).item()
            nrmse_curve.append(err)
            preds.append(pred.cpu().numpy())
            targets.append(target.cpu().numpy())

            current_state = pred.detach()

    return nrmse_curve, preds, targets


def compute_mean_rollout_nrmse(model, test_ds, device, num_sims=20):
    """
    Average rollout NRMSE curve over num_sims test simulations.
    Builds per-simulation lists from the flat test dataset index.
    """
    # Group test dataset index by simulation key
    from collections import defaultdict
    sim_dict = defaultdict(list)
    for idx, (key, t) in enumerate(test_ds.index):
        sim_dict[key].append((t, idx))

    sim_keys = list(sim_dict.keys())[:num_sims]
    all_curves = []

    for key in sim_keys:
        pairs = sorted(sim_dict[key], key=lambda x: x[0])
        sim_data_list = [test_ds[idx] for _, idx in pairs]
        curve, _, _ = compute_rollout(model, sim_data_list, device)
        all_curves.append(curve)

    min_len = min(len(c) for c in all_curves)
    curves  = np.array([c[:min_len] for c in all_curves])
    return curves.mean(axis=0), curves.std(axis=0)


# Plot figures

def plot_training_curves(gcn_history, mgn_history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # train loss
    axes[0].plot(gcn_history['train_loss'], label='GCN',          color='steelblue')
    axes[0].plot(mgn_history['train_loss'], label='MeshGraphNets', color='coral')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # val NRMSE
    axes[1].plot(gcn_history['val_nrmse'], label='GCN',          color='steelblue')
    axes[1].plot(mgn_history['val_nrmse'], label='MeshGraphNets', color='coral')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('NRMSE')
    axes[1].set_title('Validation NRMSE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_rollout_nrmse(gcn_mean, gcn_std, mgn_mean, mgn_std, save_path):
    timesteps = np.arange(1, len(gcn_mean) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(timesteps, gcn_mean, label='GCN',          color='steelblue')
    ax.fill_between(timesteps,
                    gcn_mean - gcn_std,
                    gcn_mean + gcn_std,
                    alpha=0.2, color='steelblue')

    ax.plot(timesteps, mgn_mean, label='MeshGraphNets', color='coral')
    ax.fill_between(timesteps,
                    mgn_mean - mgn_std,
                    mgn_mean + mgn_std,
                    alpha=0.2, color='coral')

    ax.set_xlabel('Rollout Timestep')
    ax.set_ylabel('NRMSE')
    ax.set_title('Rollout NRMSE over Time')
    ax.set_xlim(1, 24)  
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_flow_visualization(model_gcn, model_mgn, test_ds, device,
                            stats, save_path, sim_idx=0, timestep=10):
    """
    Side-by-side visualization of ground truth vs GCN vs MGN predictions
    for x-velocity (u) at a specific timestep in a rollout.
    """
    from collections import defaultdict
    sim_dict = defaultdict(list)
    for idx, (key, t) in enumerate(test_ds.index):
        sim_dict[key].append((t, idx))

    key = list(sim_dict.keys())[sim_idx]
    pairs = sorted(sim_dict[key], key=lambda x: x[0])

    # Roll out both models to the target timestep
    def get_pred_at_t(model, t_target):
        sim_data_list = [test_ds[idx] for _, idx in pairs[:t_target + 1]]
        _, preds, targets = compute_rollout(model, sim_data_list, device)
        return preds[t_target], targets[t_target]

    t = min(timestep, len(pairs) - 1)
    gcn_pred, gt = get_pred_at_t(model_gcn, t)
    mgn_pred, _  = get_pred_at_t(model_mgn, t)

    # Get mesh positions for triangulation
    sample_data = test_ds[pairs[0][1]]
    pos = sample_data.pos.numpy()     # (N, 2)

    # Denormalize u channel (index 0) for display
    mean = stats['mean'][0].item()
    std  = stats['std'][0].item()
    gt_u    = gt[:, 0]    * std + mean
    gcn_u   = gcn_pred[:, 0] * std + mean
    mgn_u   = mgn_pred[:, 0] * std + mean

    vmin = gt_u.min()
    vmax = gt_u.max()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = ['Ground Truth', 'GCN Prediction', 'MeshGraphNets Prediction']
    fields = [gt_u, gcn_u, mgn_u]

    for ax, title, field in zip(axes, titles, fields):
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=field,
                        cmap='RdBu_r', vmin=vmin, vmax=vmax,
                        s=1, rasterized=True)
        ax.set_title(f'{title}(t={t+1})')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        plt.colorbar(sc, ax=ax, label='u (m/s)')

    plt.suptitle('X-Velocity Field Comparison', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# Results table

def save_results_table(gcn_one_step, mgn_one_step,
                       gcn_rollout_mean, mgn_rollout_mean,
                       save_path):
    lines = [
        "=" * 50,
        "Results Summary",
        "=" * 50,
        f"{'Model':<20} {'One-Step NRMSE':>15} {'Rollout NRMSE':>15}",
        "-" * 50,
        f"{'GCN (baseline)':<20} {gcn_one_step:>15.6f} {gcn_rollout_mean:>15.6f}",
        f"{'MeshGraphNets':<20} {mgn_one_step:>15.6f} {mgn_rollout_mean:>15.6f}",
        "=" * 50,
    ]
    text = "".join(lines)
    print("" + text)
    with open(save_path, 'w') as f:
        f.write(text)
    print(f"Saved: {save_path}")


# Main

def main():
    args = get_args()
    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load stats and test set
    stats_path = os.path.join(args.checkpoint_dir, 'stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    print("Loading dataset...")
    _, _, test_ds, _ = get_splits(args.h5_path,
                                   train_frac=0.1,
                                   val_frac=0.05,
                                   test_frac=0.05,
                                   stats=stats)

    # Load models
    gcn_path = os.path.join(args.checkpoint_dir, 'gcn_best.pt')
    mgn_path = os.path.join(args.checkpoint_dir, 'mgn_best.pt')

    model_gcn = load_model(gcn_path, device)
    model_mgn = load_model(mgn_path, device)

    # Load training histories
    with open(os.path.join(args.checkpoint_dir, 'gcn_history.pkl'), 'rb') as f:
        gcn_history = pickle.load(f)
    with open(os.path.join(args.checkpoint_dir, 'mgn_history.pkl'), 'rb') as f:
        mgn_history = pickle.load(f)

    # One-step NRMSE
    print("Computing one-step NRMSE on test set...")
    gcn_one_step = compute_one_step_nrmse(model_gcn, test_ds, device)
    mgn_one_step = compute_one_step_nrmse(model_mgn, test_ds, device)
    print(f"GCN one-step NRMSE:          {gcn_one_step:.6f}")
    print(f"MeshGraphNets one-step NRMSE: {mgn_one_step:.6f}")

    # Rollout NRMSE
    print("Computing rollout NRMSE (20 test simulations)...")
    gcn_rollout_mean, gcn_rollout_std = compute_mean_rollout_nrmse(
        model_gcn, test_ds, device, num_sims=20)
    mgn_rollout_mean, mgn_rollout_std = compute_mean_rollout_nrmse(
        model_mgn, test_ds, device, num_sims=20)
    print(f"GCN mean rollout NRMSE:          {gcn_rollout_mean.mean():.6f}")
    print(f"MeshGraphNets mean rollout NRMSE: {mgn_rollout_mean.mean():.6f}")

    # Figures
    print("Generating figures...")

    plot_training_curves(
        gcn_history, mgn_history,
        os.path.join(args.results_dir, 'training_curves.png')
    )

    plot_rollout_nrmse(
        gcn_rollout_mean, gcn_rollout_std,
        mgn_rollout_mean, mgn_rollout_std,
        os.path.join(args.results_dir, 'rollout_nrmse.png')
    )

    plot_flow_visualization(
        model_gcn, model_mgn, test_ds, device, stats,
        os.path.join(args.results_dir, 'flow_visualization.png'),
        sim_idx=0, timestep=10
    )

    # Results table
    save_results_table(
        gcn_one_step, mgn_one_step,
        gcn_rollout_mean.mean(), mgn_rollout_mean.mean(),
        os.path.join(args.results_dir, 'results_table.txt')
    )

    print("All results saved to", args.results_dir)


if __name__ == '__main__':
    main()
