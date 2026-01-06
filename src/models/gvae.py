import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class GVAE(nn.Module):
    """
    Graph Variational Autoencoder with a GATv2-based Encoder.
    """
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_layers: int = 2, heads: int = 4):
        """
        Args:
            input_dim (int): Dimension of input node features.
            hidden_dim (int): Dimension of hidden layers in the GATv2 encoder.
            latent_dim (int): Dimension of the latent space for each node.
            num_layers (int): Number of GATv2 layers in the encoder.
            heads (int): Number of attention heads for the GATv2 layers.
        """
        super().__init__()
        self.num_layers = num_layers

        # --- Encoder ---
        self.encoder_convs = nn.ModuleList()
        self.encoder_convs.append(GATv2Conv(input_dim, hidden_dim, heads=heads, concat=True))
        for _ in range(num_layers - 1):
            self.encoder_convs.append(GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, concat=True))

        self.mu_head = GATv2Conv(hidden_dim * heads, latent_dim, heads=1, concat=False)
        self.log_var_head = GATv2Conv(hidden_dim * heads, latent_dim, heads=1, concat=False)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for i, conv in enumerate(self.encoder_convs):
            x = conv(x, edge_index)
            x = F.elu(x)
        mu = self.mu_head(x, edge_index)
        log_var = self.log_var_head(x, edge_index)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * log_var)
            epsilon = torch.randn_like(std)
            return mu + epsilon * std
        else:
            return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        adj = torch.matmul(z, z.t())
        return adj

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x, edge_index)
        z = self.reparameterize(mu, log_var)
        adj_recon = self.decode(z)
        return adj_recon, mu, log_var
