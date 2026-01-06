import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import json

def create_ego_prediction_visualization(ego_protein_name='BID_HUMAN', num_novel_partners=5):
    """
    Generates a network visualization centered on an "ego" protein, showing both
    its known interactions and the top novel interactions predicted by the model.
    """
    # --- 1. Load All Necessary Data ---
    print("Loading data and embeddings...")
    try:
        embeddings = torch.load('experiments/gvae_run/final_embeddings.pth')
        with open('data/processed/node2idx.json', 'r') as f:
            node2idx = json.load(f)
        idx2node = {v: k for k, v in node2idx.items()}
        edgelist_df = pd.read_csv('data/processed/edgelist.tsv', sep='\t')
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    if ego_protein_name not in node2idx:
        print(f"Ego protein '{ego_protein_name}' not found in the network. Please choose another.")
        return
        
    ego_idx = node2idx[ego_protein_name]

    # --- 2. Build Full Graph & Find Known Neighbors ---
    full_graph = nx.from_pandas_edgelist(edgelist_df, 'node1_name', 'node2_name')
    known_neighbors = list(full_graph.neighbors(ego_protein_name))
    print(f"Found {len(known_neighbors)} known neighbors for {ego_protein_name}.")

    # --- 3. Predict Novel Partners ---
    print(f"Predicting novel partners for {ego_protein_name}...")
    with torch.no_grad():
        ego_embedding = embeddings[ego_idx]
        # Calculate dot product between ego and all other proteins
        all_scores = torch.matmul(embeddings, ego_embedding)
        all_probs = torch.sigmoid(all_scores)

    # Create a DataFrame for easy filtering
    predictions_df = pd.DataFrame({
        'protein_idx': range(len(idx2node)),
        'protein_name': [idx2node[i] for i in range(len(idx2node))],
        'confidence': all_probs.numpy()
    })
    
    # Filter out the ego protein itself and its known neighbors
    existing_partners = known_neighbors + [ego_protein_name]
    predictions_df = predictions_df[~predictions_df['protein_name'].isin(existing_partners)]
    
    # Get the top N novel predictions
    novel_partners_df = predictions_df.nlargest(num_novel_partners, 'confidence')
    novel_partners = novel_partners_df['protein_name'].tolist()
    print(f"Top {num_novel_partners} novel predicted partners: {novel_partners}")

    # --- 4. Construct the Ego-Prediction Graph ---
    node_list = [ego_protein_name] + known_neighbors + novel_partners
    subgraph = full_graph.subgraph(node_list)
    
    # Create a new graph for plotting so we can add the predicted edges
    plot_graph = nx.Graph()
    # Add nodes and define their types for coloring
    plot_graph.add_node(ego_protein_name, type='Ego')
    for node in known_neighbors:
        plot_graph.add_node(node, type='Known Partner')
    for node in novel_partners:
        plot_graph.add_node(node, type='Predicted Partner')
        
    # Add known edges (solid lines)
    for u, v in subgraph.edges():
        plot_graph.add_edge(u, v, type='known')
        
    # Add predicted edges (dashed lines)
    for partner in novel_partners:
        plot_graph.add_edge(ego_protein_name, partner, type='predicted')

    # --- 5. Generate the Plot ---
    print("Generating final plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 12))

    pos = nx.spring_layout(plot_graph, seed=42, k=1.8)
    
    # Define colors and styles
    node_colors = {'Ego': '#9370DB', 'Known Partner': '#6495ED', 'Predicted Partner': '#3CB371'}
    edge_styles = {'known': 'solid', 'predicted': 'dashed'}
    
    # Draw nodes by type
    for node_type, color in node_colors.items():
        nodelist = [node for node, attr in plot_graph.nodes(data=True) if attr['type'] == node_type]
        nx.draw_networkx_nodes(plot_graph, pos, nodelist=nodelist, node_color=color, node_size=2500, label=node_type, ax=ax, edgecolors='black', linewidths=0.5)

    # Draw edges by type
    known_edges = [(u, v) for u, v, attr in plot_graph.edges(data=True) if attr['type'] == 'known']
    predicted_edges = [(u, v) for u, v, attr in plot_graph.edges(data=True) if attr['type'] == 'predicted']
    nx.draw_networkx_edges(plot_graph, pos, edgelist=known_edges, style='solid', width=1.5, ax=ax)
    nx.draw_networkx_edges(plot_graph, pos, edgelist=predicted_edges, style='dashed', width=2.0, edge_color='red', ax=ax)

    # Draw labels
    nx.draw_networkx_labels(plot_graph, pos, font_size=10, font_weight='bold', ax=ax)

    ax.set_title(f"Ego Network for '{ego_protein_name}' with Novel Predicted Interactions", fontsize=20, fontweight='bold')
    ax.legend(scatterpoints=1, loc='upper right')
    plt.tight_layout()

    # Save Figure
    save_path = f'docs/images/figure5_ego_network_{ego_protein_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure 5 (ego-prediction network) saved to {save_path}")


if __name__ == '__main__':
    create_ego_prediction_visualization()
