
"""
train.py

Training loop for cylinder flow surrogate models.
Supports both GCN (baseline) and MeshGraphNets (variant).

Usage:
    python train.py --model gcn  --epochs 50 --lr 1e-3
    python train.py --model mgn  --epochs 50 --lr 1e-3

Checkpoints are saved to checkpoints/<model>_best.pt
Training curves are saved to checkpoints/<model>_history.pkl
"""

import os
import pickle
import argparse
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

import sys
sys.path.append('/content/Deep-Learning-Project')

from utils.dataset import get_splits
from utils.metrics import nrmse, one_step_nrmse
from models.gcn import GCN
from models.mgn import MeshGraphNets


# Argument parsing
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      type=str,   default='gcn',
                        choices=['gcn', 'mgn'],
                        help='which model to train')
    parser.add_argument('--h5_path',    type=str,
                        default='/content/Deep-Learning-Project/data/train_downsampled_labeled.h5')
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int,   default=128)
    parser.add_argument('--num_layers', type=int,   default=6)
    parser.add_argument('--batch_size', type=int,   default=4)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    return parser.parse_args()


# Training

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    count = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        pred = model(data)           # (N, 3)
        loss = nn.MSELoss()(pred, data.y)

        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        count += 1

    return total_loss / count

def evaluate(model, loader, device):
    model.eval()
    total_nrmse = 0.0
    count = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data)
            err  = nrmse(pred, data.y)
            total_nrmse += err.item()
            count += 1

    return total_nrmse / count


# Main

def main():
    args = get_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Data
    print("Loading dataset...")
    train_ds, val_ds, test_ds, stats = get_splits(args.h5_path, train_frac=0.1, val_frac=0.05, test_frac=0.05)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2)

    # Save stats alongside checkpoints for use in evaluate.py
    stats_path = os.path.join(args.checkpoint_dir, 'stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    print(f"Stats saved to {stats_path}")

    # Model
    if args.model == 'gcn':
        model = GCN(in_dim=7, hidden_dim=args.hidden_dim,
                    out_dim=3, num_layers=args.num_layers).to(device)
    else:
        model = MeshGraphNets(in_node_dim=7, in_edge_dim=3,
                              hidden_dim=args.hidden_dim, out_dim=3,
                              num_layers=args.num_layers).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model.upper()}  |  Parameters: {total_params:,}")

    # Optimizer + scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    history = {
        'train_loss': [],
        'val_nrmse':  [],
        'best_epoch': 0,
    }
    best_val_nrmse = float('inf')
    checkpoint_path = os.path.join(args.checkpoint_dir, f'{args.model}_best.pt')

    print(f"Training for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_nrmse  = evaluate(model, val_loader, device)

        scheduler.step(val_nrmse)

        history['train_loss'].append(train_loss)
        history['val_nrmse'].append(val_nrmse)

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss: {train_loss:.6f}  "
              f"val_nrmse: {val_nrmse:.6f}")

        # Save best checkpoint
        if val_nrmse < best_val_nrmse:
            best_val_nrmse = val_nrmse
            history['best_epoch'] = epoch
            torch.save({
                'epoch':      epoch,
                'model':      args.model,
                'state_dict': model.state_dict(),
                'val_nrmse':  val_nrmse,
                'args':       vars(args),
            }, checkpoint_path)
            print(f"  --> saved best checkpoint (val_nrmse: {best_val_nrmse:.6f})")

    # Save training history
    history_path = os.path.join(args.checkpoint_dir, f'{args.model}_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)

    print(f"Done. Best epoch: {history['best_epoch']}  "
          f"Best val NRMSE: {best_val_nrmse:.6f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History:    {history_path}")


if __name__ == '__main__':
    main()
