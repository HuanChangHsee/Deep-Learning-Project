# Deep Learning Project - Cylinder Flow Prediction
24-788 Introduction to Deep Learning, Spring 2026
Huan Chang Hsee | Carnegie Mellon University

## Problem
We train two neural surrogate models to predict the temporal evolution of 
cylinder flow on an unstructured mesh, replacing expensive COMSOL 
simulations with fast graph neural network inference.

## Models
- **Baseline**: Graph Convolutional Network (GCN)
- **Variant**: MeshGraphNets (Pfaff et al., ICLR 2021)

## Dataset
Cylinder flow dataset from HuggingFace (ayz2/ldm_pdes).
3 channels (x-velocity, y-velocity, pressure) on an irregular 
triangular mesh. 25 timesteps per sample, 1000 simulations total.

Download:
```bash
wget https://huggingface.co/datasets/ayz2/ldm_pdes/resolve/main/cylinder_flow_captioned.zip
unzip cylinder_flow_captioned.zip -d data/
```

## Setup
```bash
pip install torch torch_geometric h5py numpy matplotlib pickle5
```

## Repository Structure
```
Deep-Learning-Project/
├── models/
│   ├── gcn.py          # GCN baseline
│   └── mgn.py          # MeshGraphNets variant
├── utils/
│   ├── dataset.py      # Data loading and graph construction
│   └── metrics.py      # NRMSE evaluation metrics
├── checkpoints/        # Saved model weights
│   ├── gcn_best.pt
│   ├── mgn_best.pt
│   └── stats.pkl
├── results/            # Generated figures and metrics
├── train.py            # Training script
├── evaluate.py         # Full evaluation and figure generation
└── reproduce.py        # Reproduce all results from checkpoints
```

## Reproducing Results
Checkpoints are included in the repo. No retraining required.

```bash
python reproduce.py
```

Figures and metrics will be saved to `results/`.

## Training From Scratch
```bash
# Train GCN
python train.py --model gcn --epochs 30 --lr 1e-3 --batch_size 128 --num_layers 3 --hidden_dim 64

# Train MeshGraphNets
python train.py --model mgn --epochs 50 --lr 3e-4 --batch_size 128 --num_layers 4 --hidden_dim 128
```

## Evaluation Metric
Normalized RMSE (NRMSE) for both one-step prediction and 
autoregressive rollout over full trajectories:

$$NRMSE = \frac{||\hat{u} - u||_2}{||u||_2}$$

## Implementation Notes
The GCN and MeshGraphNets implementations are written from scratch using 
PyTorch and PyTorch Geometric, based on the original paper descriptions. 
No existing model implementation was directly adapted.

## References
- Pfaff et al., "Learning Mesh-Based Simulation with Graph Networks", ICLR 2021
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io
- Dataset: https://huggingface.co/datasets/ayz2/ldm_pdes
