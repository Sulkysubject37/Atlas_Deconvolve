import pandas as pd
import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix, save_npz
import json
import os
import argparse

def load_hippie_data(file_path: str, confidence_threshold: float = 0.84) -> pd.DataFrame:
    """
    Loads HIPPIE data from a TSV file and filters by confidence.

    Args:
        file_path (str): Path to the HIPPIE TSV file.
        confidence_threshold (float): Minimum confidence score for interactions.

    Returns:
        pd.DataFrame: Filtered DataFrame with 'Protein1', 'Protein2', 'Confidence'.
    """
    print(f"Loading HIPPIE data from {file_path}...")
    # Assuming HIPPIE format:
    # Protein1_ID Protein1_Name Protein2_ID Protein2_Name Interaction_Type Confidence Ref_ID ...
    # Observed format: Protein1_Name, Protein1_ID, Protein2_Name, Protein2_ID, Confidence, Info
    df = pd.read_csv(file_path, sep='\t', header=None,
                     names=['Protein1_Name', 'Protein1_ID', 'Protein2_Name', 'Protein2_ID',
                            'Confidence', 'Info'])

    # Convert 'Confidence' column to numeric, coercing errors to NaN, then drop rows with NaN confidence
    df['Confidence'] = pd.to_numeric(df['Confidence'], errors='coerce')
    df.dropna(subset=['Confidence'], inplace=True)

    # Ensure protein names are strings
    df['Protein1_Name'] = df['Protein1_Name'].astype(str)
    df['Protein2_Name'] = df['Protein2_Name'].astype(str)

    original_rows = len(df)
    df_filtered = df[df['Confidence'] >= confidence_threshold].copy()
    print(f"Filtered {original_rows} interactions down to {len(df_filtered)} with confidence >= {confidence_threshold}.")

    # Use Protein Names for consistent node representation
    df_filtered['Protein1'] = df_filtered['Protein1_Name']
    df_filtered['Protein2'] = df_filtered['Protein2_Name']

    return df_filtered[['Protein1', 'Protein2', 'Confidence']]

def preprocess_ppi_network(df: pd.DataFrame, min_degree: int = 2) -> tuple[nx.Graph, dict, pd.DataFrame, dict]:
    """
    Processes the PPI network: prunes low-degree nodes, takes LCC, and creates mappings.

    Args:
        df (pd.DataFrame): DataFrame with 'Protein1', 'Protein2' (node names), 'Confidence'.
        min_degree (int): Minimum degree for nodes to be kept.

    Returns:
        tuple:
            - nx.Graph: The processed NetworkX graph (LCC, pruned, no isolated nodes).
            - dict: node2idx mapping.
            - pd.DataFrame: edgelist with canonical indices.
            - dict: stats (degree, clustering, triangles).
    """
    print("Building initial graph...")
    G = nx.from_pandas_edgelist(df, 'Protein1', 'Protein2', edge_attr='Confidence')
    print(f"Initial graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    # 1. Prune nodes with degree < min_degree
    nodes_to_remove = [node for node, degree in dict(G.degree()).items() if degree < min_degree]
    G.remove_nodes_from(nodes_to_remove)
    G.remove_edges_from(nx.selfloop_edges(G)) # Remove self-loops that might have been created by filtering
    print(f"Graph after pruning nodes with degree < {min_degree}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    # 2. Take largest connected component
    if not nx.is_connected(G):
        print("Graph is not connected. Extracting largest connected component (LCC)...")
        G_lcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        print(f"LCC: {G_lcc.number_of_nodes()} nodes, {G_lcc.number_of_edges()} edges.")
    else:
        G_lcc = G
        print("Graph is already connected.")

    # Ensure no isolated nodes in LCC (should be handled by LCC extraction and pruning, but good check)
    G_lcc.remove_nodes_from(list(nx.isolates(G_lcc)))
    print(f"Final graph (LCC, pruned, no isolates): {G_lcc.number_of_nodes()} nodes, {G_lcc.number_of_edges()} edges.")

    # Create deterministic node2idx mapping
    nodes = sorted(list(G_lcc.nodes()))
    node2idx = {node: i for i, node in enumerate(nodes)}

    # Create canonical edgelist
    idx2node = {i: node for node, i in node2idx.items()}
    canonical_edgelist = []
    for u, v in G_lcc.edges():
        idx_u, idx_v = node2idx[u], node2idx[v]
        canonical_edgelist.append({'node1_name': u, 'node1_idx': idx_u, 'node2_name': v, 'node2_idx': idx_v})
    canonical_edgelist_df = pd.DataFrame(canonical_edgelist)

    # Calculate stats
    degrees = dict(G_lcc.degree())
    avg_degree = sum(degrees.values()) / G_lcc.number_of_nodes()
    clustering_coeffs = nx.clustering(G_lcc)
    avg_clustering = sum(clustering_coeffs.values()) / G_lcc.number_of_nodes()
    num_triangles = sum(nx.triangles(G_lcc).values()) / 3 # Each triangle counted 3 times
    
    stats = {
        'num_nodes': G_lcc.number_of_nodes(),
        'num_edges': G_lcc.number_of_edges(),
        'avg_degree': avg_degree,
        'avg_clustering_coefficient': avg_clustering,
        'num_triangles': num_triangles,
        'max_degree': max(degrees.values()),
        'min_degree': min(degrees.values()),
    }
    print("Network stats calculated.")

    return G_lcc, node2idx, canonical_edgelist_df, stats

def save_processed_data(output_dir: str, G: nx.Graph, node2idx: dict, edgelist_df: pd.DataFrame, stats: dict):
    """
    Saves the processed graph data to specified files.
    """
    os.makedirs(output_dir, exist_ok=True)

    num_nodes = G.number_of_nodes()
    # Create sparse adjacency matrix
    row = []
    col = []
    for u, v in G.edges():
        idx_u, idx_v = node2idx[u], node2idx[v]
        row.extend([idx_u, idx_v])
        col.extend([idx_v, idx_u]) # Add reverse for symmetric matrix

    data = np.ones(len(row)) # Binary adjacency
    adj_matrix = coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes), dtype=np.float32)
    save_npz(os.path.join(output_dir, 'adj.npz'), adj_matrix)
    print(f"Saved adj.npz to {output_dir}")

    edgelist_df.to_csv(os.path.join(output_dir, 'edgelist.tsv'), sep='\t', index=False)
    print(f"Saved edgelist.tsv to {output_dir}")

    with open(os.path.join(output_dir, 'node2idx.json'), 'w') as f:
        json.dump(node2idx, f, indent=4)
    print(f"Saved node2idx.json to {output_dir}")

    with open(os.path.join(output_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"Saved stats.json to {output_dir}")

def run_preprocessing(raw_data_path: str, processed_output_dir: str,
                      confidence_threshold: float = 0.84, min_degree: int = 2):
    """
    Main function to run the entire PPI network preprocessing pipeline.
    """
    # Load and filter raw data
    df_filtered = load_hippie_data(raw_data_path, confidence_threshold)

    # Process graph
    G_processed, node2idx, edgelist_df, stats = preprocess_ppi_network(df_filtered, min_degree)

    # Save processed data
    save_processed_data(processed_output_dir, G_processed, node2idx, edgelist_df, stats)
    print("Preprocessing complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess PPI network data")
    parser.add_argument('--raw_data_path', type=str, default='data/raw/hippie_current.tsv',
                        help='Path to the raw HIPPIE TSV file')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='Directory to save processed data')
    parser.add_argument('--threshold', type=float, default=0.84,
                        help='Confidence threshold for filtering interactions')
    parser.add_argument('--min_degree', type=int, default=2,
                        help='Minimum degree for pruning nodes')
    
    args = parser.parse_args()

    # Check if raw data exists
    if not os.path.exists(args.raw_data_path):
        print(f"Error: Raw data file not found at '{args.raw_data_path}'.")
        # For CI robustness, we can create a small dummy file if it doesn't exist
        # and we are explicitly asked to (not implemented here but good to keep in mind)
        exit(1)

    run_preprocessing(args.raw_data_path, args.output_dir, args.threshold, args.min_degree)
