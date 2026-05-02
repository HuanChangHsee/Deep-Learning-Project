"""
utils/metrics.py

Evaluation metrics for cylinder flow surrogate models.

Metrics:
    - NRMSE: Normalized Root Mean Squared Error (one-step)
    - Rollout NRMSE: autoregressive rollout error averaged over timesteps
"""

import torch
import numpy as np


def nrmse(pred, target, eps=1e-8):
    """
    Normalized RMSE for a single prediction step.

    NRMSE = ||pred - target||_2 / (||target||_2 + eps)

    Args:
        pred   : (N, 3) predicted state
        target : (N, 3) ground truth state
    Returns:
        scalar tensor
    """
    diff_norm   = torch.norm(pred - target)
    target_norm = torch.norm(target) + eps
    return diff_norm / target_norm


def rollout_nrmse(model, sample_list, device, stats, max_steps=None):
    """
    Compute average rollout NRMSE over a list of full simulations.

    For each simulation, the model is applied autoregressively:
        state_0 -> pred_1 -> pred_2 -> ... -> pred_T
    and NRMSE is computed at each step against the ground truth.

    Args:
        model       : trained GCN or MeshGraphNets model
        sample_list : list of PyG Data objects, one per timestep transition,
                      belonging to the SAME simulation in order t=0,1,...
                      OR a list of lists (one list per simulation)
        device      : torch device
        stats       : dict with 'mean' and 'std' tensors (N,3) for denorm
        max_steps   : if set, only roll out this many steps

    Returns:
        mean_nrmse  : float, average NRMSE across all timesteps and simulations
        nrmse_curve : np.array of shape (T,), NRMSE at each timestep
                      averaged across simulations
    """
    model.eval()
    mean = stats['mean'].to(device)   # (3,)
    std  = stats['std'].to(device)    # (3,)

    # Handle both single simulation and multiple simulations
    if not isinstance(sample_list[0], list):
        sample_list = [sample_list]

    all_curves = []

    with torch.no_grad():
        for sim_samples in sample_list:
            if max_steps is not None:
                sim_samples = sim_samples[:max_steps]

            T = len(sim_samples)
            nrmse_curve = []

            # Initialize current state from normalized first sample
            current_data = sim_samples[0].to(device)
            current_state = current_data.x[:, :3].clone()  # (N, 3) normalized

            for t, data in enumerate(sim_samples):
                data = data.to(device)

                # Replace node state features with current rolled-out state
                # Keep node type features (last 4 columns) unchanged
                data.x = torch.cat([current_state, data.x[:, 3:]], dim=-1)

                # Predict next state
                pred = model(data)                  # (N, 3) normalized

                # Ground truth of next state (normalized)
                target = data.y                     # (N, 3) normalized

                # Compute NRMSE in normalized space
                err = nrmse(pred, target)
                nrmse_curve.append(err.item())

                # Advance state
                current_state = pred.detach()

            all_curves.append(nrmse_curve)

    # Pad curves to same length and average
    min_len = min(len(c) for c in all_curves)
    curves  = np.array([c[:min_len] for c in all_curves])  # (num_sims, T)
    mean_curve = curves.mean(axis=0)                        # (T,)

    return float(mean_curve.mean()), mean_curve


def one_step_nrmse(model, dataloader, device):
    """
    Compute average one-step NRMSE over a dataloader.
    Each batch is a single (input, target) graph pair.

    Args:
        model      : trained model
        dataloader : PyG DataLoader
        device     : torch device
    Returns:
        float, mean one-step NRMSE
    """
    model.eval()
    total, count = 0.0, 0

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            pred = model(data)
            err  = nrmse(pred, data.y)
            total += err.item()
            count += 1

    return total / count if count > 0 else 0.0