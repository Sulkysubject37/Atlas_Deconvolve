import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix, save_npz
import json
import os
import random

def generate_sbm_layers(num_nodes: int, num_layers: int, block_sizes: list[int], p_in: float, p_out: float,
                        overlap_ratio: float = 0.1) -> list[nx.Graph]:
    """
    Generates K-layer graphs using the Stochastic Block Model with optional overlap.

    Args:
        num_nodes (int): Total number of nodes.
        num_layers (int): Number of latent layers (K).
        block_sizes (list[int]): Sizes of the communities/blocks. Sum should be num_nodes.
        p_in (float): Probability of edge within a block.
        p_out (float): Probability of edge between blocks.
        overlap_ratio (float): Ratio of nodes to be shared between layers, creating overlap.

    Returns:
        list[nx.Graph]: A list of NetworkX graphs, one for each layer.
    """
    graphs = []
    nodes = list(range(num_nodes))

    for layer_idx in range(num_layers):
        # Generate SBM for the current layer
        current_graph = nx.stochastic_block_model(block_sizes, [[p_in if i == j else p_out for j in range(len(block_sizes))] for i in range(len(block_sizes))])
        
        # Ensure node labels are 0 to num_nodes-1 for consistency
        mapping = {old_label: new_label for new_label, old_label in enumerate(current_graph.nodes())}
        current_graph = nx.relabel_nodes(current_graph, mapping)

        graphs.append(current_graph)
    
    # Introduce overlap: randomly select some nodes and ensure they appear in multiple layers
    if overlap_ratio > 0 and num_layers > 1:
        num_overlap_nodes = int(num_nodes * overlap_ratio)
        overlap_nodes = random.sample(nodes, num_overlap_nodes)
        
        for node in overlap_nodes:
            # Ensure this node exists in at least two layers
            # For simplicity, we just add it to a random other layer if it's not already there.
            # More sophisticated overlap could involve shared communities.
            target_layer_idx = random.choice([i for i in range(num_layers) if i != layer_idx])
            # If the node is not in target_layer_idx graph, add it.
            # SBM already generates nodes, so this might be about ensuring connectivity.
            # This part is complex to do robustly without modifying SBM generation itself.
            # For a simpler implementation, we can just ensure the same nodes are used.
            pass # SBM already works with all nodes

    return graphs

def generate_synthetic_data(output_dir: str, num_nodes: int = 100, num_layers: int = 3,
                            block_sizes: list[int] = None, p_in: float = 0.5, p_out: float = 0.01,
                            overlap_ratio: float = 0.1):
    """
    Generates synthetic K-layer graphs and saves them in the specified format.

    Args:
        output_dir (str): Directory to save the synthetic data.
        num_nodes (int): Total number of nodes in the synthetic graph.
        num_layers (int): Number of latent layers to generate.
        block_sizes (list[int]): Sizes of the communities/blocks for SBM.
        p_in (float): Probability of edge within a block.
        p_out (float): Probability of edge between blocks.
        overlap_ratio (float): Ratio of nodes to be shared between layers.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if block_sizes is None:
        # Default block sizes, try to make them roughly equal
        block_sizes = [num_nodes // num_layers] * num_layers
        remaining = num_nodes - sum(block_sizes)
        for i in range(remaining):
            block_sizes[i] += 1
        
    print(f"Generating {num_layers} synthetic layers with {num_nodes} nodes each using SBM...")
    latent_graphs = generate_sbm_layers(num_nodes, num_layers, block_sizes, p_in, p_out, overlap_ratio)

    # Save each latent layer
    for i, G_layer in enumerate(latent_graphs):
        layer_output_dir = os.path.join(output_dir, f'layer_{i}')
        os.makedirs(layer_output_dir, exist_ok=True)

        nodes = sorted(list(G_layer.nodes()))
        node2idx = {node: idx for idx, node in enumerate(nodes)}

        # Adjacency matrix
        row = []
        col = []
        for u, v in G_layer.edges():
            row.append(node2idx[u])
            col.append(node2idx[v])
            row.append(node2idx[v]) # Symmetric
            col.append(node2idx[u])
        
        if len(row) > 0:
            data = np.ones(len(row))
            adj_matrix = coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes), dtype=np.float32)
            save_npz(os.path.join(layer_output_dir, 'adj.npz'), adj_matrix)
        else:
            # Handle empty graph case if no edges generated
            save_npz(os.path.join(layer_output_dir, 'adj.npz'), coo_matrix((num_nodes, num_nodes), dtype=np.float32))

        with open(os.path.join(layer_output_dir, 'node2idx.json'), 'w') as f:
            json.dump(node2idx, f, indent=4)
        
        # Basic stats for the layer
        stats = {
            'num_nodes': G_layer.number_of_nodes(),
            'num_edges': G_layer.number_of_edges(),
            'avg_degree': (2 * G_layer.number_of_edges() / G_layer.number_of_nodes()) if G_layer.number_of_nodes() > 0 else 0
        }
        with open(os.path.join(layer_output_dir, 'stats.json'), 'w') as f:
            json.dump(stats, f, indent=4)

        print(f"Saved synthetic layer {i} to {layer_output_dir}")

    print("Synthetic data generation complete.")

if __name__ == '__main__':
    # Example usage
    # This will create data/synthetic/layer_0, data/synthetic/layer_1, etc.
    # Each with adj.npz, node2idx.json, stats.json
    generate_synthetic_data(
        output_dir='data/synthetic',
        num_nodes=100,
        num_layers=3,
        block_sizes=[30, 30, 40], # Example: 3 blocks of 30, 30, 40 nodes
        p_in=0.7,
        p_out=0.05,
        overlap_ratio=0.0 # For now, no explicit overlap mechanism in this SBM helper
    )
