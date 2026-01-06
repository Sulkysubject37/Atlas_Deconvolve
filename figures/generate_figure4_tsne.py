import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def create_tsne_visualization():
    """
    Generates and saves a t-SNE visualization of the protein embeddings,
    colored by their K-Means cluster assignment.
    """
    # --- 1. Load Data ---
    print("Loading protein embeddings...")
    try:
        embeddings = torch.load('experiments/gvae_run/final_embeddings.pth').numpy()
    except FileNotFoundError:
        print("Error: `experiments/gvae_run/final_embeddings.pth` not found.")
        print("Please ensure the model has been trained and embeddings are saved.")
        return

    # --- 2. Perform K-Means Clustering ---
    print("Performing K-Means clustering (K=10)...")
    kmeans = KMeans(n_clusters=10, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(embeddings)

    # --- 3. Perform t-SNE Dimensionality Reduction ---
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    # --- 4. Create DataFrame for Plotting ---
    df = pd.DataFrame({
        'tSNE-1': embeddings_2d[:, 0],
        'tSNE-2': embeddings_2d[:, 1],
        'Cluster': cluster_labels
    })
    
    # --- 5. Create and Save the Plot ---
    print("Generating plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))

    # Use a qualitative palette that is distinct
    palette = sns.color_palette("deep", 10)

    sns.scatterplot(
        data=df,
        x='tSNE-1',
        y='tSNE-2',
        hue='Cluster',
        palette=palette,
        ax=ax,
        s=15,          # Smaller marker size
        alpha=0.7,     # Add some transparency
        edgecolor='w', # White edges for better separation
        linewidth=0.5
    )

    # Improve legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Add titles and labels
    ax.set_title('t-SNE Visualization of Protein Embeddings by Functional Cluster', fontsize=18, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make room for legend

    # Save Figure
    save_path = 'docs/images/figure4_tsne_clusters.png'
    plt.savefig(save_path, dpi=300)
    print(f"Figure 4 saved to {save_path}")

if __name__ == '__main__':
    create_tsne_visualization()
