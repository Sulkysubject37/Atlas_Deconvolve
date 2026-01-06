import os
import glob
import json
import pandas as pd
from gprofiler import GProfiler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_enrichment_dot_plot():
    """
    Performs GO enrichment analysis on protein clusters and generates a dot plot
    of the most significant results.
    """
    # --- 1. Load background genes and find cluster files ---
    print("Loading custom background genes for analysis...")
    background_genes_path = "data/processed/node2idx.json"
    try:
        with open(background_genes_path, 'r') as f:
            node2idx = json.load(f)
        background_genes = [gene.replace('_HUMAN', '') for gene in list(node2idx.keys())]
        print(f"Loaded {len(background_genes)} total proteins for background.")
    except FileNotFoundError:
        print(f"Error: Background genes file not found at '{background_genes_path}'.")
        return

    cluster_dir = "experiments/gvae_run/clusters"
    cluster_files = sorted(glob.glob(os.path.join(cluster_dir, "cluster_*_proteins.txt")))

    if not cluster_files:
        print(f"Error: No cluster files found in '{cluster_dir}'.")
        print("Please run the clustering script first.")
        return

    print(f"Found {len(cluster_files)} cluster files to analyze.")

    # --- 2. Perform GO Enrichment Analysis for each cluster ---
    gp = GProfiler(return_dataframe=True)
    all_enrichment_results = []

    for cluster_file in cluster_files:
        cluster_id_str = os.path.basename(cluster_file).replace('_proteins.txt', '')
        
        with open(cluster_file, 'r') as f:
            protein_list_raw = []
            for line in f:
                parts = [p.strip() for p in line.strip().split(',') if p.strip()]
                protein_list_raw.extend(parts)
            # Strip _HUMAN suffix from query proteins
            protein_list = [gene.replace('_HUMAN', '') for gene in protein_list_raw]

        if not protein_list or len(protein_list) < 5:
            print(f"Skipping {cluster_id_str}: not enough proteins ({len(protein_list)}).")
            continue
            
        print(f"Analyzing {cluster_id_str} with {len(protein_list)} proteins...")

        enrichment_df = gp.profile(organism='hsapiens',
                                   query=protein_list,
                                   sources=['GO:BP'], # Biological Process only
                                   user_threshold=0.05,
                                   no_evidences=True,
                                   background=background_genes,
                                   domain_scope='custom')
        
        if enrichment_df.empty:
            print(f"  - No significant results for {cluster_id_str}.")
            continue

        # Filter and select top terms
        enrichment_df = enrichment_df[enrichment_df['source'] == 'GO:BP']
        top_terms = enrichment_df.nsmallest(5, 'p_value').copy() # Get top 5 by p-value
        top_terms['cluster'] = cluster_id_str
        all_enrichment_results.append(top_terms)
        print(f"  - Found {len(top_terms)} significant terms for {cluster_id_str}.")

    if not all_enrichment_results:
        print("No enrichment results found across any clusters. Cannot generate plot.")
        return

    # --- 3. Prepare data for plotting ---
    full_results_df = pd.concat(all_enrichment_results, ignore_index=True)
    full_results_df['-log10(p-value)'] = -np.log10(full_results_df['p_value'])
    
    # Create a more descriptive label for the y-axis
    full_results_df['y_label'] = full_results_df['cluster'] + ': ' + full_results_df['name']

    # Sort for plotting
    full_results_df = full_results_df.sort_values(by='-log10(p-value)', ascending=False)
    
    # Let's cap the number of terms to plot to keep it readable
    plot_df = full_results_df.head(25)
    plot_df = plot_df.sort_values(by='-log10(p-value)', ascending=True)


    # --- 4. Create and Save the Dot Plot ---
    print("Generating dot plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))
    
    scatter = ax.scatter(
        x=plot_df['-log10(p-value)'],
        y=plot_df['y_label'],
        s=(plot_df['intersection_size'] + 5) * 5,  # Adjusted Scale size for visibility
        c=plot_df['-log10(p-value)'],
        cmap='magma', # Changed colormap for different aesthetic
        alpha=0.7,
        edgecolors="w",
        linewidth=1
    )

    ax.set_xlabel('-log10(p-value)', fontsize=14)
    ax.set_ylabel('Cluster & GO Term', fontsize=14)
    ax.set_title('Top GO Biological Process Enrichment Results per Cluster', fontsize=16, fontweight='bold')
    ax.grid(True)
    
    # Create a colorbar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.ax.set_ylabel('-log10(p-value)', rotation=-90, va="bottom", fontsize=12)

    # Create a legend for the scatter plot sizes, placed below the plot
    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6, num=4)
    size_legend = ax.legend(handles, labels, loc='upper center',
                            title="Number of Genes",
                            bbox_to_anchor=(0.5, -0.15),
                            ncol=len(handles),
                            fontsize=12, title_fontsize=14)
    ax.add_artist(size_legend)

    fig.subplots_adjust(bottom=0.2) # Make space for the legend at the bottom
    plt.tight_layout(rect=[0, 0.1, 1, 1]) # Adjust layout

    # Save Figure
    save_path = 'docs/images/figure_enrichment_dot_plot.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Enrichment dot plot saved to {save_path}")

if __name__ == '__main__':
    create_enrichment_dot_plot()
