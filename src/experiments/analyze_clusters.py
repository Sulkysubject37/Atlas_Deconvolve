import torch
import json
import os
from sklearn.cluster import KMeans
import numpy as np

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
    analyze_and_save_clusters(
        embedding_path="experiments/gvae_run/final_embeddings.pth",
        node2idx_path="data/processed/node2idx.json",
        output_dir="experiments/gvae_run/clusters",
        n_clusters=10
    )
