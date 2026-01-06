import pandas as pd
import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix, save_npz
import json
import os

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
    # Example usage:
    # This script would typically be run as:
    # python src/data/preprocess.py --raw_data_path data/raw/hippie_current.tsv --output_dir data/processed

    # Placeholder for actual raw data file. User needs to provide this.
    # For testing, you might create a dummy file or expect it to be present.
    # raw_hippie_path = os.path.join('data', 'raw', 'hippie_current.tsv')
    # processed_output_path = os.path.join('data', 'processed')

    # Example: create a dummy file for demonstration if needed
    # dummy_data = {
    #     'Protein1_ID': ['P1', 'P2', 'P3', 'P4', 'P5'],
    #     'Protein1_Name': ['GeneA', 'GeneB', 'GeneC', 'GeneD', 'GeneE'],
    #     'Protein2_ID': ['P2', 'P3', 'P1', 'P5', 'P1'],
    #     'Protein2_Name': ['GeneB', 'GeneC', 'GeneA', 'GeneE', 'GeneA'],
    #     'Interaction_Type': ['pp', 'pp', 'pp', 'pp', 'pp'],
    #     'Confidence': [0.9, 0.7, 0.95, 0.6, 0.85],
    #     'Ref_ID': ['R1', 'R2', 'R3', 'R4', 'R5'],
    #     'Experiment': ['Exp1', 'Exp2', 'Exp3', 'Exp4', 'Exp5'],
    #     'Species': ['Human', 'Human', 'Human', 'Human', 'Human'],
    # }
    # pd.DataFrame(dummy_data).to_csv(raw_hippie_path, sep='\t', index=False, header=False)


    print("Please ensure your raw HIPPIE data (e.g., 'hippie_current.tsv') is placed in the 'data/raw/' directory.")
    print("You can then run `python src/data/preprocess.py` after adapting the `run_preprocessing` call.")
