import torch
import json
import os
from sklearn.cluster import KMeans
import numpy as np
import argparse

def analyze_and_save_clusters(embedding_path: str, node2idx_path: str, output_dir: str, n_clusters: int = 10):
    """
    Loads node embeddings, performs K-Means clustering, and saves the gene lists for each cluster.
    """
    print("--- Starting Cluster Analysis ---")
    
    # 1. Load data
    try:
        embeddings = torch.load(embedding_path).numpy()
        with open(node2idx_path, 'r') as f:
            node2idx = json.load(f)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    print(f"Loaded {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}.")

    # Create idx -> node name mapping
    idx2node = {idx: name for name, idx in node2idx.items()}

    # 2. Perform K-Means clustering
    print(f"Performing K-Means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    print("Clustering complete.")

    # 3. Group proteins by cluster
    clusters = {i: [] for i in range(n_clusters)}
    for node_idx, cluster_id in enumerate(cluster_labels):
        protein_name = idx2node.get(node_idx)
        if protein_name:
            clusters[cluster_id].append(protein_name)

    # 4. Save protein lists to files
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving cluster gene lists to '{output_dir}'...")
    for cluster_id, proteins in clusters.items():
        if not proteins:
            print(f"  - Cluster {cluster_id} is empty. Skipping.")
            continue
            
        file_path = os.path.join(output_dir, f"cluster_{cluster_id}_proteins.txt")
        with open(file_path, 'w') as f:
            for protein in proteins:
                f.write(f"{protein}\n")
        print(f"  - Saved Cluster {cluster_id} with {len(proteins)} proteins.")
        
    print("--- Cluster Analysis Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze node clusters using K-Means")
    parser.add_argument('--embedding_path', type=str, default='experiments/gvae_run/final_embeddings.pth',
                        help='Path to the saved node embeddings')
    parser.add_argument('--node2idx_path', type=str, default='data/processed/node2idx.json',
                        help='Path to the node to index mapping JSON file')
    parser.add_argument('--output_dir', type=str, default='experiments/gvae_run/clusters',
                        help='Directory to save cluster protein lists')
    parser.add_argument('--n_clusters', type=int, default=10,
                        help='Number of clusters for K-Means')
    
    args = parser.parse_args()

    analyze_and_save_clusters(
        embedding_path=args.embedding_path,
        node2idx_path=args.node2idx_path,
        output_dir=args.output_dir,
        n_clusters=args.n_clusters
    )
