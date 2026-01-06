import os
import json
import numpy as np
from scipy.sparse import load_npz, coo_matrix
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

class GraphDataLoader:
    """
    Loads processed graph data and performs train/validation/test splits
    for link prediction tasks.
    """
    def __init__(self, processed_data_dir: str, val_split=0.05, test_split=0.10):
        self.processed_data_dir = processed_data_dir
        self.adj_path = os.path.join(processed_data_dir, 'adj.npz')
        self.node2idx_path = os.path.join(processed_data_dir, 'node2idx.json')
        self.stats_path = os.path.join(processed_data_dir, 'stats.json')

        self.val_split = val_split
        self.test_split = test_split

        self._load_data()
        self._perform_splits()

    def _load_data(self):
        """Loads all necessary processed files."""
        # ... (loading logic remains the same)
        print(f"Loading processed data from {self.processed_data_dir}...")
        self.adj_full = load_npz(self.adj_path).astype(np.float32)
        
        with open(self.node2idx_path, 'r') as f:
            self.node2idx = json.load(f)
        
        with open(self.stats_path, 'r') as f:
            self.stats = json.load(f)

        self.num_nodes = self.stats['num_nodes']
        print("Data loading complete.")

    def _perform_splits(self):
        """Splits edges into train, val, and test sets and performs negative sampling."""
        print("Performing train/val/test splits...")
        adj_coo = self.adj_full.tocoo()
        # Get all edges and remove duplicates for undirected graph
        edges = np.stack([adj_coo.row, adj_coo.col], axis=0)
        edges = to_undirected(torch.from_numpy(edges)).numpy()
        edges = edges[:, edges[0] < edges[1]] # Keep only one direction to avoid duplicates
        num_edges = edges.shape[1]

        # Shuffle edges
        np.random.shuffle(edges.T)

        # Split indices
        num_test = int(num_edges * self.test_split)
        num_val = int(num_edges * self.val_split)
        
        self.test_pos_edges = edges[:, :num_test]
        self.val_pos_edges = edges[:, num_test : num_test + num_val]
        self.train_pos_edges = edges[:, num_test + num_val :]

        # Create training adjacency matrix
        row, col = self.train_pos_edges
        data = np.ones(row.shape[0])
        self.adj_train = coo_matrix((data, (row, col)), shape=(self.num_nodes, self.num_nodes)).toarray()
        self.adj_train = self.adj_train + self.adj_train.T # Make symmetric
        self.adj_train_tensor = torch.from_numpy(self.adj_train).float()

        # Negative sampling
        print("Performing negative sampling for validation and test sets...")
        self.val_neg_edges = self._sample_negative_edges(num_val)
        self.test_neg_edges = self._sample_negative_edges(num_test)
        print("Splits created successfully.")

    def _sample_negative_edges(self, num_samples):
        """Samples non-existent edges."""
        neg_edges = []
        adj_dense = self.adj_full.toarray()
        while len(neg_edges) < num_samples:
            i, j = np.random.randint(0, self.num_nodes, 2)
            if i != j and adj_dense[i, j] == 0:
                neg_edges.append([i, j])
        return np.array(neg_edges).T

    def get_train_graph_tensor(self) -> torch.Tensor:
        """Returns the training adjacency matrix as a dense torch tensor."""
        return self.adj_train_tensor

    def get_val_edges(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns positive and negative edges for validation."""
        return self.val_pos_edges, self.val_neg_edges

    def get_test_edges(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns positive and negative edges for testing."""
        return self.test_pos_edges, self.test_neg_edges

    def get_num_nodes(self) -> int:
        return self.num_nodes
    
    def get_node2idx_mapping(self) -> dict:
        return self.node2idx

