import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_loss_curves_plot():
    """
    Generates and saves a plot of the training loss curves (reconstruction and KL)
    from the saved loss history file.
    """
    # --- 1. Load Data ---
    print("Loading loss history...")
    try:
        loss_df = pd.read_csv('experiments/gvae_run/loss_history.csv')
    except FileNotFoundError:
        print("Error: `experiments/gvae_run/loss_history.csv` not found.")
        print("Please ensure the model training script has been run successfully.")
        return

    # --- 2. Create and Save the Plot ---
    print("Generating plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot Reconstruction Loss
    sns.lineplot(data=loss_df, x='epoch', y='recon_loss', ax=ax1, color='royalblue', lw=2)
    ax1.set_title('Reconstruction Loss per Epoch', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('BCE Loss', fontsize=12)
    ax1.grid(True)
    ax1.set_yscale('log') # Loss often decreases exponentially

    # Plot KL Divergence
    sns.lineplot(data=loss_df, x='epoch', y='kl_loss', ax=ax2, color='darkorange', lw=2)
    ax2.set_title('KL Divergence per Epoch', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('KL Divergence', fontsize=12)
    ax2.grid(True)

    fig.suptitle('GVAE Training Loss Curves', fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save Figure
    save_path = 'docs/images/figure6_loss_curves.png'
    plt.savefig(save_path, dpi=300)
    print(f"Figure 6 saved to {save_path}")

if __name__ == '__main__':
    create_loss_curves_plot()
