import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import os

def create_k_sensitivity_plot():
    """
    Performs K-Means clustering for a range of K values and plots the silhouette
    score for each to evaluate the choice of K.
    """
    # --- 1. Load Embeddings ---
    embedding_path = "experiments/gvae_run/final_embeddings.pth"
    print(f"Loading embeddings from '{embedding_path}'...")
    try:
        embeddings = torch.load(embedding_path).numpy()
    except FileNotFoundError:
        print(f"Error: Embeddings file not found at '{embedding_path}'.")
        print("Please ensure the model training and embedding generation have been run.")
        return

    # --- 2. Calculate Silhouette Scores for a Range of K ---
    k_range = range(2, 21) # Test K from 2 to 20
    silhouette_scores = []
    print(f"Calculating silhouette scores for K in {k_range}...")

    # Using a subset of the data can speed up silhouette score calculation significantly
    # as it can be computationally intensive.
    sample_size = min(5000, embeddings.shape[0])
    np.random.seed(42)
    sample_indices = np.random.choice(embeddings.shape[0], sample_size, replace=False)
    embeddings_sample = embeddings[sample_indices]
    
    for k in tqdm(k_range):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Calculate silhouette score on the sample
        score = silhouette_score(embeddings_sample, kmeans.predict(embeddings_sample))
        silhouette_scores.append(score)

    print("Score calculation complete.")

    # --- 3. Create and Save the Plot ---
    print("Generating plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(k_range, silhouette_scores, marker='o', linestyle='-', color='royalblue')
    
    # Highlight the chosen K=10
    chosen_k = 10
    chosen_k_score = silhouette_scores[k_range.index(chosen_k)]
    ax.axvline(x=chosen_k, color='darkorange', linestyle='--', lw=2, label=f'Chosen K = {chosen_k}')
    ax.plot(chosen_k, chosen_k_score, marker='*', color='red', markersize=15, label=f'Score at K=10: {chosen_k_score:.3f}')

    ax.set_xlabel('Number of Clusters (K)', fontsize=14)
    ax.set_ylabel('Silhouette Score', fontsize=14)
    ax.set_title('K-Means Clustering: Sensitivity to K', fontsize=16, fontweight='bold')
    ax.set_xticks(k_range)
    ax.legend(fontsize=12)
    ax.grid(True)

    plt.tight_layout()

    # Save Figure
    save_path = 'docs/images/figure_supplementary_k_sensitivity.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"K-sensitivity plot saved to {save_path}")

if __name__ == '__main__':
    create_k_sensitivity_plot()
