# Summary of Modifications

This file documents the series of changes made to the Atlas Deconvolve project to improve model performance, with a focus on matching the graph's degree distribution.

### 1. Architectural Upgrade: GCN/GAT to GATv2

-   **`src/models/refinement_net.py`**
    -   Replaced `GATConv` with `GATv2Conv` to use a more expressive graph attention mechanism.
-   **`src/models/critic.py`**
    -   Replaced `GCNConv` with `GATv2Conv` to align the critic's architecture with the generator's and improve its discriminative power.

### 2. Hyperparameter Tuning

-   **`src/experiments/configs/config.yaml`**
    -   **Initial GATv2 Tuning:** Increased `GAT_HEADS` to 4, `REFINEMENT_HIDDEN_DIM` to 128, and `CRITIC_HIDDEN_DIM` to 128. Lowered `LR_G` and `LR_D` to `0.0001`.
    -   **Memory Optimization:** To resolve a CUDA out-of-memory error, hyperparameters were adjusted:
        -   `REFINEMENT_HIDDEN_DIM` and `CRITIC_HIDDEN_DIM` were returned to `64`.
        -   `GAT_HEADS` was reduced from 4 to `2`.
        -   `BATCH_SIZE` was drastically reduced from 32 to `4`.
    -   Changed `EXPERIMENT_NAME` to `"gatv2_run"` to log the new experiment separately.

### 3. Data Splitting for Testing

-   **`process_split.py`** (Temporary script created in the root directory)
    -   Created a 90/10 train/test split from the original `edgelist.tsv`.
    -   Wrote a temporary script to process these new edgelist files using functions imported from `src/data/preprocess.py`.
    -   This script correctly formats the new splits into the `adj.npz`, `node2idx.json` required by the models.

### 4. Advanced Loss Function Implementation

-   **Sparsity Loss Reduction:**
    -   In `src/experiments/configs/config.yaml`, `LAMBDA_SPARSE` was reduced from `0.01` to `0.0001` to prevent it from overpowering other structural losses.
-   **New Degree Distribution Loss:**
    -   **`src/experiments/configs/config.yaml`**:
        -   Introduced a new loss weight, `LAMBDA_DEGREE`, initially set to `10.0`.
        -   Later increased to `50.0` to give it more influence.
    -   **`src/training/losses.py`**:
        -   Initially, a new function `degree_distribution_loss` was added to calculate the Mean Squared Error (MSE) between degree histograms.
        -   This was later upgraded to calculate the **1D Wasserstein Distance** between the degree distributions for a more robust comparison.
    -   **`src/training/trainer.py`**:
        -   Modified the `train_step` to import and compute the new `degree_distribution_loss` and add it to the total generator loss.
        -   Added `G/loss_degree` to the training logs for monitoring.
