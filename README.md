# Deep Learning Project - Cylinder Flow
24-788 Introduction to Deep Learning, Spring 2026

## Problem


## Models
- **Baseline**: Graph Convolutional Network (GCN)
- **Variant**: MeshGraphNets

## Dataset
Cylinder flow dataset from HuggingFace (ayz2/ldm_pdes).
3 channels (x-velocity, y-velocity, pressure) on an irregular triangular mesh.
25 timesteps per sample.

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

## Reproducing Results

## Evaluation Metric

## References
