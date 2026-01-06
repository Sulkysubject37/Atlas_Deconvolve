import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_architecture_diagram():
    """
    Generates and saves a high-quality, branded block diagram of the 
    "Atlas Deconvolve" architecture for the paper.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Add a main title for the framework
    fig.suptitle("The Atlas Deconvolve Framework", fontsize=20, fontweight='bold', y=0.98)

    # Define styles with a new color palette
    box_style = dict(boxstyle='round,pad=0.6', fc='white', ec='black', lw=1.5)
    arrow_style = dict(arrowstyle='-|>', color='black', lw=2, shrinkA=10, shrinkB=10)
    stage_style = dict(fontsize=16, fontweight='bold', ha='center')
    box_text_style = dict(fontsize=12, ha='center', va='center', wrap=True)
    
    # New color scheme
    encode_bg = '#E3F2FD'  # Light Blue
    cluster_bg = '#E8F5E9'  # Light Green
    embedding_color = '#FFF9C4' # Light Yellow

    # --- STAGE 1: Representation Learning (Encode) ---
    ax.text(3.5, 7.5, 'Stage 1: Representation Learning (Encode)', **stage_style)
    ax.add_patch(patches.Rectangle((0.5, 0.5), 6, 6.5, fc=encode_bg, alpha=0.5, linestyle='--', ec='gray'))

    # Input Data
    ax.text(3.5, 6, 'Input: High-Confidence\nPPI Network Graph `G`', **box_text_style, bbox=box_style)

    # GVAE Model Box
    gvae_box = patches.FancyBboxPatch((2, 3), 3, 2, boxstyle="round,pad=0.3", fc='white', ec='black', lw=1.5)
    ax.add_patch(gvae_box)
    ax.text(3.5, 4.2, "GVAE Model", fontsize=14, fontweight='bold', ha='center')
    ax.text(3.5, 3.6, "Encoder: 2-Layer GATv2\nDecoder: Inner Product", ha='center', fontsize=10)

    # Output Embeddings
    ax.text(3.5, 1.5, 'Output: Learned Protein Embeddings\n(64-dim Latent Space `Z`)', **box_text_style, bbox=dict(boxstyle='round,pad=0.6', fc=embedding_color, ec='black', lw=1.5))
    
    # Arrows for Stage 1
    ax.add_patch(patches.FancyArrowPatch((3.5, 5.2), (3.5, 5.0), **arrow_style)) # Input -> GVAE
    ax.add_patch(patches.FancyArrowPatch((3.5, 3.0), (3.5, 2.7), **arrow_style)) # GVAE -> Embeddings

    # --- STAGE 2: Module Discovery (Cluster) ---
    ax.text(10.5, 7.5, 'Stage 2: Module Discovery (Cluster)', **stage_style)
    ax.add_patch(patches.Rectangle((7.5, 0.5), 6, 6.5, fc=cluster_bg, alpha=0.5, linestyle='--', ec='gray'))
    
    # K-Means Algorithm
    ax.text(10.5, 4.5, 'Clustering Algorithm:\nK-Means', **box_text_style, bbox=box_style)
    
    # Final Output
    ax.text(10.5, 1.5, 'Final Output:\nDiscovered Functional Modules\n(Clusters `C_k`)', **box_text_style, bbox=box_style)
    
    # Arrows for Stage 2
    ax.add_patch(patches.FancyArrowPatch((10.5, 3.7), (10.5, 2.7), **arrow_style)) # K-Means -> Modules

    # --- Connecting Arrow ---
    ax.add_patch(patches.FancyArrowPatch((6.3, 1.5), (8.5, 4.5), connectionstyle="arc3,rad=0.3", **arrow_style))
    ax.text(7.4, 3.0, 'High-fidelity embeddings\n(ROC AUC > 0.99)', ha='center', style='italic', color='black', fontsize=10)


    # Final Touches
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make room for suptitle
    
    # Save Figure
    save_path = 'docs/images/figure1_architecture.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Figure 1 (revised) saved to {save_path}")

if __name__ == '__main__':
    create_architecture_diagram()
