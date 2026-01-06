import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.sparse import load_npz
import json

def create_network_layout_figure():
    """
    Generates a force-directed layout of the full PPI network, with nodes
    colored by their assigned cluster.
    """
    print("--- Generating Network Layout Figure ---")

    # --- 1. Load the full graph ---
    adj_path = 'data/processed/adj.npz'
    node2idx_path = 'data/processed/node2idx.json'
    print(f"Loading full graph from '{adj_path}'...")
    try:
        adj = load_npz(adj_path)
        with open(node2idx_path, 'r') as f:
            node2idx = json.load(f)
        idx2node = {i: name for name, i in node2idx.items()}
    except FileNotFoundError as e:
        print(f"Error loading graph files: {e}")
        return

    G = nx.from_scipy_sparse_array(adj)
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    # --- 2. Load cluster assignments for coloring ---
    cluster_dir = "experiments/gvae_run/clusters"
    cluster_files = glob.glob(os.path.join(cluster_dir, "cluster_*_proteins.txt"))
    
    if not cluster_files:
        print(f"Warning: No cluster files found in '{cluster_dir}'. Nodes will not be colored.")
        node_to_cluster = {}
    else:
        print(f"Loading cluster assignments from '{cluster_dir}'...")
        node_to_cluster = {}
        for f in cluster_files:
            cluster_id = int(os.path.basename(f).split('_')[1])
            with open(f, 'r') as reader:
                for protein in reader:
                    protein_name = protein.strip().replace('_HUMAN', '') # Use cleaned names
                    node_to_cluster[protein_name] = cluster_id
    
    # Map cluster IDs to nodes in the graph
    colors = [node_to_cluster.get(idx2node[i].replace('_HUMAN', ''), -1) for i in G.nodes()]

    # --- 3. Compute force-directed layout ---
    # This is computationally expensive and may take several minutes.
    # The number of iterations can be adjusted to trade speed for layout quality.
    print("Computing force-directed layout (this may take several minutes)...")
    pos = nx.spring_layout(G, iterations=50, seed=42)
    print("Layout computation complete.")

    # --- 4. Draw the graph ---
    print("Drawing graph...")
    plt.style.use('dark_background') # Use a dark style for a "galaxy" look
    fig, ax = plt.subplots(figsize=(20, 20))

    # Draw edges first, with high transparency
    nx.draw_networkx_edges(G, pos, alpha=0.05, width=0.5, edge_color='gray')

    # Draw nodes, colored by cluster
    # Use a categorical colormap
    cmap = plt.cm.get_cmap('viridis', len(set(colors)))
    nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=cmap, node_size=10, alpha=0.8)

    ax.set_title('Latent Space Organization of the PPI Network', fontsize=24, fontweight='bold')
    plt.axis('off')
    
    # --- 5. Save the figure ---
    save_path = 'docs/images/figure_network_layout.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Saving figure to '{save_path}'...")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='black')
    
    print("--- Figure Generation Finished ---")


if __name__ == '__main__':
    create_network_layout_figure()
