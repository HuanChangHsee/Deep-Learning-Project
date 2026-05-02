
"""
reproduce.py

Reproduces the key results from the report without retraining from scratch.
Loads saved model checkpoints and regenerates all figures and metrics.

Usage:
    python reproduce.py

Requirements:
    - Checkpoints must be present in checkpoints/
        - checkpoints/gcn_best.pt
        - checkpoints/mgn_best.pt
        - checkpoints/stats.pkl
    - Dataset must be present at data/train_downsampled_labeled.h5
      Download from: https://huggingface.co/datasets/ayz2/ldm_pdes
      Then unzip: unzip cylinder_flow_captioned.zip -d data/

Outputs (saved to results/):
    - training_curves.png
    - rollout_nrmse.png
    - flow_visualization.png
    - results_table.txt
"""

import os
import sys
import pickle
import torch
import numpy as np

sys.path.append('/content/Deep-Learning-Project')

from utils.dataset import get_splits
from utils.metrics import nrmse
from models.gcn import GCN
from models.mgn import MeshGraphNets

# Paths

H5_PATH        = 'data/train_downsampled_labeled.h5'
CHECKPOINT_DIR = 'checkpoints'
RESULTS_DIR    = 'results'

# Setup

os.makedirs(RESULTS_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Verify files exist

required = [
    os.path.join(CHECKPOINT_DIR, 'gcn_best.pt'),
    os.path.join(CHECKPOINT_DIR, 'mgn_best.pt'),
    os.path.join(CHECKPOINT_DIR, 'stats.pkl'),
    os.path.join(CHECKPOINT_DIR, 'gcn_history.pkl'),
    os.path.join(CHECKPOINT_DIR, 'mgn_history.pkl'),
    H5_PATH,
]

for path in required:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {path}"
            f"Please follow the setup instructions in README.md"
        )

print("All required files found.")

# Load stats and dataset

print("Loading dataset...")
with open(os.path.join(CHECKPOINT_DIR, 'stats.pkl'), 'rb') as f:
    stats = pickle.load(f)

_, _, test_ds, _ = get_splits(
    H5_PATH,
    train_frac=0.1,
    val_frac=0.05,
    test_frac=0.05,
    stats=stats
)
print(f"Test set: {len(test_ds)} timestep pairs")

# Load models

def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    args = ckpt['args']
    if ckpt['model'] == 'gcn':
        model = GCN(in_dim=7, hidden_dim=args['hidden_dim'],
                    out_dim=3, num_layers=args['num_layers']).to(device)
    else:
        model = MeshGraphNets(in_node_dim=7, in_edge_dim=3,
                              hidden_dim=args['hidden_dim'], out_dim=3,
                              num_layers=args['num_layers']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f"Loaded {ckpt['model'].upper()} "
          f"(epoch {ckpt['epoch']}, val_nrmse: {ckpt['val_nrmse']:.6f})")
    return model

print("Loading models...")
model_gcn = load_model(os.path.join(CHECKPOINT_DIR, 'gcn_best.pt'), device)
model_mgn = load_model(os.path.join(CHECKPOINT_DIR, 'mgn_best.pt'), device)

# One-step NRMSE

from torch_geometric.loader import DataLoader

def compute_one_step_nrmse(model, test_ds, device, batch_size=32):
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

print("Computing one-step NRMSE...")
gcn_one_step = compute_one_step_nrmse(model_gcn, test_ds, device)
mgn_one_step = compute_one_step_nrmse(model_mgn, test_ds, device)
print(f"GCN one-step NRMSE:           {gcn_one_step:.6f}")
print(f"MeshGraphNets one-step NRMSE: {mgn_one_step:.6f}")

# Rollout NRMSE

from collections import defaultdict

def compute_rollout(model, sim_data_list, device):
    model.eval()
    nrmse_curve, preds, targets = [], [], []
    current_state = sim_data_list[0].x[:, :3].clone().to(device)
    with torch.no_grad():
        for data in sim_data_list:
            data = data.to(device)
            data.x = torch.cat([current_state, data.x[:, 3:]], dim=-1)
            pred   = model(data)
            target = data.y
            nrmse_curve.append(nrmse(pred, target).item())
            preds.append(pred.cpu().numpy())
            targets.append(target.cpu().numpy())
            current_state = pred.detach()
    return nrmse_curve, preds, targets

def compute_mean_rollout(model, test_ds, device, num_sims=20):
    sim_dict = defaultdict(list)
    for idx, (key, t) in enumerate(test_ds.index):
        sim_dict[key].append((t, idx))
    all_curves = []
    for key in list(sim_dict.keys())[:num_sims]:
        pairs = sorted(sim_dict[key], key=lambda x: x[0])
        sim_data_list = [test_ds[idx] for _, idx in pairs]
        curve, _, _ = compute_rollout(model, sim_data_list, device)
        all_curves.append(curve)
    min_len = min(len(c) for c in all_curves)
    curves  = np.array([c[:min_len] for c in all_curves])
    return curves.mean(axis=0), curves.std(axis=0)

print("Computing rollout NRMSE (20 test simulations)...")
gcn_rollout_mean, gcn_rollout_std = compute_mean_rollout(model_gcn, test_ds, device)
mgn_rollout_mean, mgn_rollout_std = compute_mean_rollout(model_mgn, test_ds, device)
print(f"GCN mean rollout NRMSE:           {gcn_rollout_mean.mean():.6f}")
print(f"MeshGraphNets mean rollout NRMSE: {mgn_rollout_mean.mean():.6f}")

# Figures

import matplotlib.pyplot as plt

print("Generating figures...")

# Training curves
with open(os.path.join(CHECKPOINT_DIR, 'gcn_history.pkl'), 'rb') as f:
    gcn_history = pickle.load(f)
with open(os.path.join(CHECKPOINT_DIR, 'mgn_history.pkl'), 'rb') as f:
    mgn_history = pickle.load(f)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(gcn_history['train_loss'], label='GCN',          color='steelblue')
axes[0].plot(mgn_history['train_loss'], label='MeshGraphNets', color='coral')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('Training Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[1].plot(gcn_history['val_nrmse'], label='GCN',          color='steelblue')
axes[1].plot(mgn_history['val_nrmse'], label='MeshGraphNets', color='coral')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('NRMSE')
axes[1].set_title('Validation NRMSE')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: training_curves.png")

# Rollout NRMSE
timesteps = np.arange(1, len(gcn_rollout_mean) + 1)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(timesteps, gcn_rollout_mean, label='GCN',          color='steelblue')
ax.fill_between(timesteps, gcn_rollout_mean - gcn_rollout_std,
                gcn_rollout_mean + gcn_rollout_std, alpha=0.2, color='steelblue')
ax.plot(timesteps, mgn_rollout_mean, label='MeshGraphNets', color='coral')
ax.fill_between(timesteps, mgn_rollout_mean - mgn_rollout_std,
                mgn_rollout_mean + mgn_rollout_std, alpha=0.2, color='coral')
ax.set_xlabel('Rollout Timestep')
ax.set_ylabel('NRMSE')
ax.set_title('Rollout NRMSE over Time')
ax.set_xlim(1, 24)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'rollout_nrmse.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: rollout_nrmse.png")

# Flow visualization
sim_dict = defaultdict(list)
for idx, (key, t) in enumerate(test_ds.index):
    sim_dict[key].append((t, idx))
key   = list(sim_dict.keys())[0]
pairs = sorted(sim_dict[key], key=lambda x: x[0])
t_vis = min(10, len(pairs) - 1)

def get_pred_at_t(model, t_target):
    sim_data_list = [test_ds[idx] for _, idx in pairs[:t_target + 1]]
    _, preds, targets = compute_rollout(model, sim_data_list, device)
    return preds[t_target], targets[t_target]

gcn_pred, gt = get_pred_at_t(model_gcn, t_vis)
mgn_pred, _  = get_pred_at_t(model_mgn, t_vis)
pos  = test_ds[pairs[0][1]].pos.numpy()
mean_u = stats['mean'][0].item()
std_u  = stats['std'][0].item()
gt_u   = gt[:, 0]       * std_u + mean_u
gcn_u  = gcn_pred[:, 0] * std_u + mean_u
mgn_u  = mgn_pred[:, 0] * std_u + mean_u
vmin, vmax = gt_u.min(), gt_u.max()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, title, field in zip(axes,
    ['Ground Truth', 'GCN Prediction', 'MeshGraphNets Prediction'],
    [gt_u, gcn_u, mgn_u]):
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=field, cmap='RdBu_r',
                    vmin=vmin, vmax=vmax, s=1, rasterized=True)
    ax.set_title(f'{title} (t={t_vis+1})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax, label='u (m/s)')
plt.suptitle('X-Velocity Field Comparison', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'flow_visualization.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: flow_visualization.png")

# Results table
lines = [
    "=" * 50,
    "Results Summary",
    "" + "=" * 50,
    f"{'Model':<20} {'One-Step NRMSE':>15} {'Rollout NRMSE':>15}",
    "" + "-" * 50,
    f"{'GCN (baseline)':<20} {gcn_one_step:>15.6f} {gcn_rollout_mean.mean():>15.6f}",
    f"{'MeshGraphNets':<20} {mgn_one_step:>15.6f} {mgn_rollout_mean.mean():>15.6f}",
    "" + "=" * 50,
]
text = "".join(lines)
print(text)
with open(os.path.join(RESULTS_DIR, 'results_table.txt'), 'w') as f:
    f.write(text)
print("Saved: results_table.txt")

print(f"All results saved to {RESULTS_DIR}/")