"""
utils/dataset.py
 
Cylinder Flow dataset loader.
Reads the h5 file, builds PyTorch Geometric Data objects, and returns train/val/test splits.
 
H5 structure (per sample key '0', '1', ...):
  cells       : (num_edges, 3)                   triangular mesh connectivity
  mesh_pos    : (num_nodes, 2)                   x, y coordinates of each node
  node_type   : (num_nodes, 1)                   0 = fluid, 4 = inlet, 5 = outlet, 6 = boundaries/wall
  pressure    : (num_timesteps, num_nodes, 1)    mesh point ressure
  u           : (num_timesteps, num_nodes)       x velocity
  v           : (num_timesteps, num_nodes)       y velocity
  metadata    : center                           cylinder center,
                domain_x                         x bounds, 
                domain_y                         y bounds, 
                radius                           cylinder radius, 
                reynolds_number                  reynolds number, 
                t_end                            simulation time, 
                u_inlet                          x velocity inlet, 
                v_inlet                          y velocity inlet
"""
 
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
 
 
# Convert mesh into graph
def _triangles_to_edges(cells):
    """
    Convert (num_triangles, 3) cell array to an undirected edge index
    of shape (2, num_edges).  Duplicate / reverse edges are both kept so
    message passing is bidirectional.
    """
    # Each triangle (i,j,k) yields edges i-j, j-k, i-k in both directions
    edges = set()
    for tri in cells:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        for a, b in [(i, j), (j, k), (i, k)]:
            edges.add((a, b))
            edges.add((b, a))
    edge_index = np.array(list(edges), dtype=np.int64).T    # (2, E)
    return edge_index
 
# Build features for each edge (for MeshGraphNets)
def _build_edge_features(mesh_pos, edge_index):
    """
    For each edge (i -> j) compute [dx, dy, ||d||] where d = pos_j - pos_i.
    Returns tensor of shape (num_edges, 3).
    These are used by MeshGraphNets but not by GCN.
    """
    src, dst = edge_index[0], edge_index[1]
    diff = mesh_pos[dst] - mesh_pos[src]                    # (E, 2)
    distance = np.linalg.norm(diff, axis=1, keepdims=True)  # (E, 1)
    edge_attr = np.concatenate([diff, distance], axis=1)    # (E, 3)
    return torch.tensor(edge_attr, dtype=torch.float32)
 
# Build features for each node type 
def _one_hot_node_type(node_type):
    """
    node_type values: 0 (fluid), 4 (inlet), 5 (outlet), 6 (boundaries/wall).
    Returns one-hot array of shape (num_nodes, 4).
    """
    mapping = {0: 0, 4: 1, 5: 2, 6: 3}
    nt = node_type.flatten().astype(int)
    one_hot = np.zeros((len(nt), 4), dtype=np.float32)
    for idx, val in enumerate(nt):
        one_hot[idx, mapping.get(val, 0)] = 1.0
    return one_hot
 
# Dataset
class CylinderFlowDataset(Dataset):
    """
    Each item is a (input_graph, target_graph) pair for one timestep transition
    t -> t+1 within one simulation sample.
 
    Node features (x):
        [u, v, pressure, node_type_onehot(4)]  ->  7 features total
 
    Edge features (edge_attr):
        [dx, dy, distance]  ->  3 features  (used by MeshGraphNets)
 
    Target (y):
        [u, v, pressure] at t+1  ->  3 values per node
 
    Args:
        h5_path   : path to the .h5 file
        split_keys: list of sample keys (strings) to include, e.g. ['0','1',...]
        normalize : if True, apply z-score normalization using provided stats
        stats     : dict with 'mean' and 'std' tensors of shape (3,)
                    for [u, v, pressure].  If None and normalize=True,
                    stats are computed from this split.
    """
 
    def __init__(self, h5_path: str, split_keys: list,
                 normalize: bool = True, stats: dict = None):
        super().__init__()
        self.h5_path = h5_path
        self.split_keys = split_keys
        self.normalize = normalize
 
        # Build index: list of (sample_key, t) pairs
        self.index = []
        with h5py.File(h5_path, 'r') as f:
            for key in split_keys:
                num_timesteps = f[key]['u'].shape[0]
                # t goes from 0 to T-2 (predict t -> t+1)
                for t in range(num_timesteps - 1):
                    self.index.append((key, t))
 
        # Compute or store normalization statistics
        if normalize:
            if stats is not None:
                self.stats = stats
            else:
                self.stats = self._compute_stats()
        else:
            self.stats = None
 
    def _compute_stats(self) -> dict:
        """Compute mean and std of [u, v, pressure] over the entire split."""
        all_u, all_v, all_p = [], [], []
        with h5py.File(self.h5_path, 'r') as f:
            for key in self.split_keys:
                all_u.append(f[key]['u'][:].flatten())
                all_v.append(f[key]['v'][:].flatten())
                all_p.append(f[key]['pressure'][:].flatten())
        u = np.concatenate(all_u)
        v = np.concatenate(all_v)
        p = np.concatenate(all_p)
        mean = torch.tensor([u.mean(), v.mean(), p.mean()], dtype=torch.float32)
        std  = torch.tensor([u.std(),  v.std(),  p.std()],  dtype=torch.float32)
        std  = torch.clamp(std, min=1e-8)
        return {'mean': mean, 'std': std}
 
    def __len__(self) -> int:
        return len(self.index)
 
    def __getitem__(self, idx: int) -> Data:
        key, t = self.index[idx]
 
        with h5py.File(self.h5_path, 'r') as f:
            sample = f[key]
 
            # Spatial structure (same for all timesteps)
            mesh_pos  = sample['mesh_pos'][:]          # (N, 2)
            cells     = sample['cells'][:]             # (num_tri, 3)
            node_type = sample['node_type'][:]         # (N, 1)
 
            # Fluid state at t and t+1 
            u_t   = sample['u'][t]                     # (N,)
            v_t   = sample['v'][t]                     # (N,)
            p_t   = sample['pressure'][t, :, 0]        # (N,)
 
            u_t1  = sample['u'][t + 1]
            v_t1  = sample['v'][t + 1]
            p_t1  = sample['pressure'][t + 1, :, 0]
 
        # Build edges 
        edge_index = _triangles_to_edges(cells)        # (2, E)
        edge_attr  = _build_edge_features(mesh_pos, edge_index)  # (E, 3)
 
        # Build node features 
        state_t  = np.stack([u_t, v_t, p_t], axis=1).astype(np.float32)   # (N,3)
        state_t1 = np.stack([u_t1, v_t1, p_t1], axis=1).astype(np.float32)
 
        nt_onehot = _one_hot_node_type(node_type)      # (N, 4)
 
        # Normalize
        state_t_tensor  = torch.tensor(state_t)
        state_t1_tensor = torch.tensor(state_t1)
 
        if self.normalize and self.stats is not None:
            mean = self.stats['mean']   # (3,)
            std  = self.stats['std']    # (3,)
            state_t_tensor  = (state_t_tensor  - mean) / std
            state_t1_tensor = (state_t1_tensor - mean) / std
 
        # Concatenate node features: [u,v,p, node_type_onehot]
        node_features = torch.cat([
            state_t_tensor,
            torch.tensor(nt_onehot)
        ], dim=1)  # (N, 7)
 
        # Assemble PyTorch Geometric Data objects
        data = Data(
            x          = node_features,                          # (N, 7)
            edge_index = torch.tensor(edge_index, dtype=torch.long),  # (2, E)
            edge_attr  = edge_attr,                              # (E, 3)
            y          = state_t1_tensor,                        # (N, 3)  target
            pos        = torch.tensor(mesh_pos, dtype=torch.float32),  # (N, 2)
        )
        return data
 
 
# Split samples to train/val/test
def get_splits(h5_path: str,
               train_frac: float = 0.1,
               val_frac:   float = 0.05,
               test_frac:  float = 0.05,
               seed: int = 42, stats=None):
    """
    Returns (train_dataset, val_dataset, test_dataset).
 
    The split is over simulation *samples* (not timesteps),
    so there is no data leakage between splits.
 
    Normalization stats are computed on the training split only and
    shared with val/test.
    """
    with h5py.File(h5_path, 'r') as f:
        all_keys = sorted(f.keys(), key=lambda k: int(k))
 
    rng = np.random.default_rng(seed)
    keys = np.array(all_keys)
    rng.shuffle(keys)
 
    n = len(keys)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    n_test  = int(n * test_frac)

    train_keys = keys[:n_train].tolist()
    val_keys   = keys[n_train : n_train + n_val].tolist()
    test_keys  = keys[n_train + n_val : n_train + n_val + n_test].tolist()
 
    print(f"Split: {len(train_keys)} train / {len(val_keys)} val / "
          f"{len(test_keys)} test samples")
 
    # Compute stats from training data only
    train_ds = CylinderFlowDataset(h5_path, train_keys, normalize=True,
                                   stats=stats)
    if stats is None:
        stats = train_ds.stats
 
    val_ds  = CylinderFlowDataset(h5_path, val_keys,  normalize=True, stats=stats)
    test_ds = CylinderFlowDataset(h5_path, test_keys, normalize=True, stats=stats)
 
    return train_ds, val_ds, test_ds, stats
