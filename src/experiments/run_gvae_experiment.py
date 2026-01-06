import argparse
import yaml
from box import Box
import torch
import random
import numpy as np
import os
import shutil

from src.data.loader import GraphDataLoader
from src.models.gvae import GVAE
from src.training.gvae_trainer import GVAETrainer

def load_config(config_path="src/experiments/configs/gvae_config.yaml"):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Box(config_dict)

import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Run GVAE Experiment")
    parser.add_argument('--config', type=str, default='src/experiments/configs/gvae_config.yaml',
                        help='Path to GVAE configuration YAML file')
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Loaded configuration from {args.config}")

    # --- Setup ---
    torch.manual_seed(cfg.SYSTEM.SEED)
    random.seed(cfg.SYSTEM.SEED)
    np.random.seed(cfg.SYSTEM.SEED)
    device = torch.device(cfg.SYSTEM.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading and Splitting ---
    print("Loading data and performing splits...")
    data_loader = GraphDataLoader(cfg.DATA.PROCESSED_DIR)
    num_nodes = data_loader.get_num_nodes()
    adj_train_tensor = data_loader.get_train_graph_tensor()
    val_pos_edges, val_neg_edges = data_loader.get_val_edges()
    test_pos_edges, test_neg_edges = data_loader.get_test_edges()
    print("Data loading and splitting complete.")

    # --- Model and Trainer Initialization ---
    print("Initializing GVAE model and trainer...")
    model = GVAE(
        input_dim=num_nodes,
        hidden_dim=cfg.MODEL.HIDDEN_DIM,
        latent_dim=cfg.MODEL.LATENT_DIM,
        num_layers=cfg.MODEL.NUM_LAYERS,
        heads=cfg.MODEL.HEADS
    ).to(device)
    trainer = GVAETrainer(model, num_nodes, cfg)

    # --- Training Loop ---
    print("Starting GVAE training...")
    recon_losses, kl_losses = [], []
    for epoch in range(1, cfg.TRAIN.EPOCHS + 1):
        recon_loss, kl_loss = trainer.train_step(adj_train_tensor, epoch)
        recon_losses.append(recon_loss)
        kl_losses.append(kl_loss)

        if epoch % cfg.TRAIN.EVAL_FREQ == 0:
            print(f"--- Eval at Epoch {epoch} on Validation Set ---")
            val_metrics, _, _ = trainer.evaluate(adj_train_tensor, val_pos_edges, val_neg_edges)
            for name, value in val_metrics.items():
                print(f"{name}: {value:.4f}")
            print("------------------------------------------------")

    print("Training finished.")

    # --- Final Evaluation on Test Set ---
    print("\n--- Running Final Evaluation on Test Set ---")
    test_metrics, y_true, y_pred = trainer.evaluate(adj_train_tensor, test_pos_edges, test_neg_edges)
    for name, value in test_metrics.items():
        print(f"Final Test {name}: {value:.4f}")
    print("------------------------------------------")

    # --- Save Artifacts for Figure Generation ---
    output_dir = f"experiments/{cfg.EXPERIMENT_NAME}/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save loss history
    loss_df = pd.DataFrame({'epoch': range(1, cfg.TRAIN.EPOCHS + 1), 'recon_loss': recon_losses, 'kl_loss': kl_losses})
    loss_save_path = os.path.join(output_dir, "loss_history.csv")
    loss_df.to_csv(loss_save_path, index=False)
    print(f"Loss history saved to {loss_save_path}")

    # Save final test predictions for ROC/PR curves
    predictions_save_path = os.path.join(output_dir, "test_predictions.pt")
    torch.save({'y_true': y_true.cpu(), 'y_pred': y_pred.cpu()}, predictions_save_path)
    print(f"Test predictions for figures saved to {predictions_save_path}")

    # --- Save Final Embeddings ---
    print("\nExtracting and saving final node embeddings...")
    model.eval()
    with torch.no_grad():
        x = torch.eye(num_nodes, device=device)
        edge_index_train = adj_train_tensor.to(device).to_sparse()._indices()
        mu, _ = model.encode(x, edge_index_train)
        
        embedding_save_path = os.path.join(output_dir, "final_embeddings.pth")
        torch.save(mu.detach().cpu(), embedding_save_path)
        print(f"Final node embeddings saved to {embedding_save_path}")

if __name__ == "__main__":
    main()
