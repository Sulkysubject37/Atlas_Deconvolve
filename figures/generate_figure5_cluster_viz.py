import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import json

def create_cluster_visualization(cluster_id_to_visualize=9, figure_name="Apoptosis"):
    """
    Generates and saves a network visualization of a specific, smaller protein cluster
    with readable node labels.
    """
    # --- 1. Load Data ---
    print("Loading data...")
    try:
        embeddings = torch.load('experiments/gvae_run/final_embeddings.pth').numpy()
        with open('data/processed/node2idx.json', 'r') as f:
            node2idx = json.load(f)
        idx2node = {v: k for k, v in node2idx.items()}
        edgelist_df = pd.read_csv('data/processed/edgelist.tsv', sep='\t')
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # --- 2. Perform K-Means Clustering ---
    print(f"Performing K-Means clustering to identify members of cluster {cluster_id_to_visualize}...")
    kmeans = KMeans(n_clusters=10, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(embeddings)

    # --- 3. Identify Nodes in the Target Cluster ---
    target_cluster_indices = np.where(cluster_labels == cluster_id_to_visualize)[0]
    target_cluster_nodes = [idx2node[i] for i in target_cluster_indices]
    
    if not target_cluster_nodes:
        print(f"No nodes found for cluster {cluster_id_to_visualize}. Exiting.")
        return
    
    print(f"Found {len(target_cluster_nodes)} nodes in Cluster {cluster_id_to_visualize}.")

    # --- 4. Build Full Graph and Extract Subgraph ---
    print("Building full graph and extracting subgraph...")
    full_graph = nx.from_pandas_edgelist(edgelist_df, 'node1_name', 'node2_name')
    subgraph = full_graph.subgraph(target_cluster_nodes)

    # --- 5. Create and Save the Plot ---
    print("Generating plot with node labels...")
    if subgraph.number_of_nodes() == 0:
        print("Subgraph is empty, cannot generate plot.")
        return
        
    plt.style.use('seaborn-v0_8-whitegrid')
    # Increase figure size for better label spacing
    fig, ax = plt.subplots(figsize=(20, 20))

    # Use a spring layout with increased spacing
    pos = nx.spring_layout(subgraph, seed=42, k=1.5 / np.sqrt(subgraph.number_of_nodes()))

    # Draw the graph with improved aesthetics
    nx.draw_networkx_nodes(subgraph, pos, ax=ax, node_size=300, node_color='#FF6347', alpha=0.9, edgecolors='k', linewidths=0.5)
    nx.draw_networkx_edges(subgraph, pos, ax=ax, width=0.8, edge_color='gray', alpha=0.6)
    
    # Draw node labels
    nx.draw_networkx_labels(subgraph, pos, ax=ax, font_size=8, font_color='black', font_weight='bold')

    ax.set_title(f'Discovered Module Visualization (Cluster {cluster_id_to_visualize}: {figure_name})', fontsize=22, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()

    # Save Figure
    save_path = f'docs/images/figure5_cluster{cluster_id_to_visualize}_{figure_name.lower()}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure 5 (revised) saved to {save_path}")

if __name__ == '__main__':
    # Visualizing Cluster 9, which we identified as related to Apoptosis
    create_cluster_visualization(cluster_id_to_visualize=9, figure_name="Apoptosis")
