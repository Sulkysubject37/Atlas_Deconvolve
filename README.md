# Atlas Deconvolve: A Graph Variational Autoencoder for Biological Network Analysis

This project implements a Graph Variational Autoencoder (GVAE) with an attention-based encoder (`GATv2`) to learn meaningful latent representations of nodes in a Protein-Protein Interaction (PPI) network.

The model is designed to reconstruct the graph structure and, in doing so, learn powerful node embeddings that can be used for various downstream bioinformatics tasks, such as link prediction, community detection, and functional module discovery.

## Architecture

The core of this project is a **Graph Attention VAE**.

1.  **Encoder**: The encoder uses a series of `GATv2Conv` (Graph Attention) layers to process the entire graph and produce a latent distribution (mean and log-variance) for each node. This allows the model to learn a rich, topology-aware representation for every protein.
2.  **Decoder**: The decoder reconstructs the graph's adjacency matrix by taking the inner product of the latent node embeddings. This is a direct and efficient method for graph generation from a latent space.
3.  **Loss Function**: The model is trained by optimizing a combination of:
    *   **Reconstruction Loss**: A weighted Binary Cross-Entropy loss that ensures the decoded graph is similar to the original.
    *   **KL Divergence Loss**: A regularization term that ensures the learned latent space is smooth and well-structured.

This architecture has proven highly effective for link prediction on the provided dataset, achieving an ROC AUC of over 0.99.

## Cross-Platform Support

This project has been updated to support cross-platform execution on:
*   **NVIDIA GPUs (CUDA)** for Linux/Windows.
*   **Apple Silicon GPUs (MPS)** for macOS.
*   **CPUs** as a fallback.

The system automatically detects the best available hardware accelerator.

## How to Run

### 1. Setup

**Prerequisites:**
*   Python 3.10 or higher (3.10, 3.11, 3.12 are supported and tested).

Create a virtual environment and install the required dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Data Preprocessing

The model expects a processed graph. If you have a raw edgelist (e.g., `data/raw/hippie_current.tsv`), run the preprocessing script:

```bash
python3 -m src.data.preprocess
```
This will generate the necessary `adj.npz`, `node2idx.json`, and `stats.json` files in the `data/processed` directory.

Alternatively, you can generate a synthetic dataset for testing:
```bash
python3 -m src.data.synthetic_generator --output_dir data/processed --num_nodes 100 --num_layers 2
```

### 3. Training the GVAE Model

To train the GVAE model, run the main experiment script:

```bash
python3 -m src.experiments.run_gvae_experiment
```

This will start the training process using the parameters defined in `src/experiments/configs/gvae_config.yaml`. The script will automatically select the best available device ("auto"). It will periodically print evaluation metrics and, upon completion, save the final node embeddings to `experiments/gvae_run/final_embeddings.pth`.

### 4. Interpretation and Downstream Analysis

The primary output of the GVAE is the learned node embeddings (`final_embeddings.pth`). These embeddings can be used for further biological analysis:

*   **Cluster Analysis:**
    ```bash
    python3 -m src.experiments.analyze_clusters --embedding_path experiments/gvae_run/final_embeddings.pth
    ```
*   **Link Prediction:**
    ```bash
    python3 -m src.experiments.predict_fgfr_interactions --skip_api
    ```

## Automated Workflow

This project includes a GitHub Actions workflow (`.github/workflows/main.yml`) that automatically:
1.  Sets up a Python environment (3.10/3.11/3.12).
2.  Installs dependencies.
3.  Generates synthetic data.
4.  Trains the model.
5.  Runs analysis and figure generation scripts.
6.  Uploads generated figures as artifacts.

This ensures robustness across Ubuntu (CUDA-ready environment) and macOS (MPS-ready environment).

## Next Steps

Use the GVAE either as a powerful component (embeddings) for Atlas Deconvolve or extend it into a Mixture-of-VAEs (K decoders + aggregation + diversity/motif losses) — that change will make it fulfill the original project goal.