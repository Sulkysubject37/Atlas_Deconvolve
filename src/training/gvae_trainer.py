import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from src.models.gvae import GVAE
from src.utils.metrics import calculate_roc_pr_auc

class GVAETrainer:
    def __init__(self, model: GVAE, num_nodes: int, cfg: dict):
        self.model = model
        self.num_nodes = num_nodes
        self.cfg = cfg
        self.optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
        self.device = torch.device(cfg.SYSTEM.DEVICE if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _calculate_loss(self, adj_recon_logits: torch.Tensor, adj_train: torch.Tensor,
                        mu: torch.Tensor, log_var: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Use adj_train for loss calculation
        pos_weight = (adj_train.numel() - adj_train.sum()) / adj_train.sum()
        recon_loss = F.binary_cross_entropy_with_logits(adj_recon_logits, adj_train, pos_weight=pos_weight)
        kl_loss = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))
        total_loss = recon_loss + self.cfg.LOSS.KL_WEIGHT * kl_loss
        return total_loss, recon_loss, kl_loss

    def train_step(self, adj_train: torch.Tensor, epoch: int) -> tuple[float, float]:
        self.model.train()
        self.optimizer.zero_grad()

        x = torch.eye(self.num_nodes, device=self.device)
        adj_train = adj_train.to(self.device)

        # The model needs edge_index for message passing, derived from adj_train
        edge_index_train = adj_train.to_sparse()._indices()

        adj_recon_logits, mu, log_var = self.model(x, edge_index_train)

        total_loss, recon_loss, kl_loss = self._calculate_loss(adj_recon_logits, adj_train, mu, log_var)

        total_loss.backward()
        self.optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{self.cfg.TRAIN.EPOCHS}]: total_loss={total_loss.item():.4f}, recon_loss={recon_loss.item():.4f}, kl_loss={kl_loss.item():.4f}")
        
        return recon_loss.item(), kl_loss.item()

    def evaluate(self, adj_train: torch.Tensor, pos_edges: np.ndarray, neg_edges: np.ndarray) -> tuple[dict, torch.Tensor, torch.Tensor]:
        self.model.eval()
        metrics = {}
        with torch.no_grad():
            x = torch.eye(self.num_nodes, device=self.device)
            # Encoder uses the training graph structure for message passing
            edge_index_train = adj_train.to(self.device).to_sparse()._indices()
            mu, _ = self.model.encode(x, edge_index_train)
            
            # Decoder computes probabilities for specific edges
            def decode_edges(edges):
                src, dest = edges
                return torch.sigmoid((mu[src] * mu[dest]).sum(dim=1))

            pos_preds = decode_edges(pos_edges)
            neg_preds = decode_edges(neg_edges)

            y_pred = torch.cat([pos_preds, neg_preds], dim=0)
            y_true = torch.cat([torch.ones_like(pos_preds), torch.zeros_like(neg_preds)], dim=0)

            roc_auc, ap_score = calculate_roc_pr_auc(y_true, y_pred)
            metrics["eval/roc_auc"] = roc_auc
            metrics["eval/ap_score"] = ap_score
        
        return metrics, y_true, y_pred
