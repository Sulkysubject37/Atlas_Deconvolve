import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def create_roc_pr_curves():
    """
    Generates and saves the ROC and Precision-Recall curves based on saved
    test set predictions.
    """
    # --- 1. Load Data ---
    print("Loading GVAE test set predictions...")
    try:
        gvae_data = torch.load('experiments/gvae_run/test_predictions.pt')
        y_true_gvae = gvae_data['y_true'].numpy()
        y_pred_gvae = gvae_data['y_pred'].numpy()
    except FileNotFoundError:
        print("Error: `experiments/gvae_run/test_predictions.pt` not found.")
        print("Please ensure the model training script has been run successfully.")
        return

    print("Loading Adamic-Adar baseline predictions...")
    try:
        baseline_data = torch.load('experiments/gvae_run/adamic_adar_predictions.pt')
        y_true_baseline = baseline_data['labels'].numpy()
        y_pred_baseline = baseline_data['scores'].numpy()
    except FileNotFoundError:
        print("Error: `experiments/gvae_run/adamic_adar_predictions.pt` not found.")
        print("Please ensure the baseline generation script has been run successfully.")
        return

    # --- 2. Calculate Curve Data ---
    print("Calculating ROC and PR curve data for both models...")
    # GVAE Model
    fpr_gvae, tpr_gvae, _ = roc_curve(y_true_gvae, y_pred_gvae)
    roc_auc_gvae = auc(fpr_gvae, tpr_gvae)
    precision_gvae, recall_gvae, _ = precision_recall_curve(y_true_gvae, y_pred_gvae)
    ap_score_gvae = average_precision_score(y_true_gvae, y_pred_gvae)

    # Adamic-Adar Baseline
    fpr_baseline, tpr_baseline, _ = roc_curve(y_true_baseline, y_pred_baseline)
    roc_auc_baseline = auc(fpr_baseline, tpr_baseline)
    precision_baseline, recall_baseline, _ = precision_recall_curve(y_true_baseline, y_pred_baseline)
    ap_score_baseline = average_precision_score(y_true_baseline, y_pred_baseline)

    # --- 3. Create and Save the Plot ---
    print("Generating plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Plot ROC Curve
    ax1.plot(fpr_gvae, tpr_gvae, color='darkorange', lw=2.5, label=f'GVAE (AUC = {roc_auc_gvae:.3f})')
    ax1.plot(fpr_baseline, tpr_baseline, color='cornflowerblue', lw=2.5, label=f'Adamic-Adar (AUC = {roc_auc_baseline:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=14)
    ax1.set_ylabel('True Positive Rate', fontsize=14)
    ax1.set_title('Receiver Operating Characteristic (ROC)', fontsize=16, fontweight='bold')
    ax1.legend(loc="lower right", fontsize=12)
    ax1.grid(True)

    # Plot Precision-Recall Curve
    ax2.plot(recall_gvae, precision_gvae, color='darkorange', lw=2.5, label=f'GVAE (AP = {ap_score_gvae:.3f})')
    ax2.plot(recall_baseline, precision_baseline, color='cornflowerblue', lw=2.5, label=f'Adamic-Adar (AP = {ap_score_baseline:.3f})')
    # Plot no-skill line
    no_skill = len(y_true_gvae[y_true_gvae==1]) / len(y_true_gvae)
    ax2.plot([0, 1], [no_skill, no_skill], color='navy', lw=2, linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', fontsize=14)
    ax2.set_ylabel('Precision', fontsize=14)
    ax2.set_title('Precision-Recall (PR) Curve', fontsize=16, fontweight='bold')
    ax2.legend(loc="upper right", fontsize=12)
    ax2.grid(True)
    
    fig.suptitle('Link Prediction Performance: GVAE vs. Adamic-Adar Baseline', fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save Figure
    save_path = 'docs/images/figure2_roc_pr_curves_with_baseline.png'
    plt.savefig(save_path, dpi=300)
    print(f"Figure 2 (with baseline) saved to {save_path}")

if __name__ == '__main__':
    create_roc_pr_curves()
