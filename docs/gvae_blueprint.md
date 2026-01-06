# GVAE: Graph Variational Autoencoder for Biological Network Analysis

## 1. Project Goal & Strategy

The primary goal is to identify meaningful sub-components within a single, complex Protein-Protein Interaction (PPI) network. While the original project attempted an end-to-end deconvolution, this has been revised to a more robust, two-step **"Encode-then-Cluster"** strategy:

1.  **Encode**: Use a powerful Graph Variational Autoencoder (GVAE) to learn a rich, biologically coherent latent space for all proteins in the network.
2.  **Cluster**: Use standard clustering algorithms (e.g., K-Means) on the learned node embeddings to partition the proteins into `K` distinct groups. These groups represent the discovered functional modules or "latent layers".

This approach leverages a highly successful and stable generative model to create the embeddings, and then uses a simple, interpretable method to define the final graph layers.

## 2. Core Architecture: Graph Attention VAE (GVAE)

The model is a standard Variational Autoencoder adapted for graph data, using Graph Attention layers for the encoder.

### 2.1. Encoder

The encoder learns a mapping from the input graph to a latent distribution for each node.

-   **Input**: The full graph structure (Adjacency Matrix `A`) and node features `X`. In this implementation, `X` is an identity matrix, meaning the model learns node features based purely on the graph topology.
-   **Architecture**: A multi-layer Graph Attention Network (`GATv2`). This allows the encoder to learn a topology-aware representation for every protein.
-   **Output**: The encoder produces two matrices that define the parameters of a Gaussian distribution for each node's latent representation: a **mean vector (μ)** and a **log-variance vector (log_var)**.

### 2.2. Latent Space & Decoder

-   **Reparameterization**: A latent embedding `Z` for all nodes is sampled from the learned distributions (`z = μ + ε * σ`). This is a standard VAE technique that enables backpropagation through the sampling process.
-   **Decoder**: The decoder is a simple, non-parametric **inner product decoder**. It reconstructs the adjacency matrix `Â` by taking the dot product of the latent node embeddings: `Â = sigmoid(Z @ Z.T)`.

## 3. Training

The model is trained by optimizing a single, combined loss function.

-   **Reconstruction Loss**: A weighted Binary Cross-Entropy (BCE) loss between the reconstructed adjacency matrix `Â` and the original graph `A`. The `pos_weight` parameter is used to handle the high sparsity of PPI networks.
-   **KL Divergence Loss**: A regularization term that forces the learned latent distributions to be close to a standard normal distribution (`N(0, 1)`). This encourages a smooth and well-organized latent space, which is critical for meaningful clustering in the downstream task.
-   **Combined Loss**: `Total Loss = Reconstruction Loss + (KL_WEIGHT * KL Divergence Loss)`

This architecture has proven to be highly effective, achieving an ROC AUC of over 0.99 for link prediction and producing a latent space where nodes cluster into demonstrably distinct and significant biological functions.
