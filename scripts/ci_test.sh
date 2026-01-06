#!/bin/bash
set -e

echo "=== Atlas Deconvolve CI Test ==="

# 1. Generate Synthetic Data
echo "Generating synthetic data..."
python3 -m src.data.synthetic_generator --output_dir data/processed --num_nodes 100 --num_layers 2

# 2. Train GVAE Model (minimal epochs for CI)
echo "Training GVAE model..."
# We can override config values via a temporary yaml or just let it run 50 epochs if fast enough.
# The current config has 50 epochs which should be fast for 100 nodes.
python3 -m src.experiments.run_gvae_experiment

# 3. Analyze Clusters
echo "Analyzing clusters..."
python3 -m src.experiments.analyze_clusters --embedding_path experiments/gvae_run/final_embeddings.pth --node2idx_path data/processed/node2idx.json --output_dir experiments/gvae_run/clusters

# 4. Generate Baseline Predictions
echo "Generating Adamic-Adar baseline..."
# Note: For synthetic data, we use the same dir for train and full for simplicity in CI
python3 -m src.experiments.generate_baseline_predictions --train_dir data/processed --full_dir data/processed --output_dir experiments/gvae_run

# 5. Predict Novel Interactions
echo "Predicting FGFR interactions (skip API)..."
python3 -m src.experiments.predict_fgfr_interactions --skip_api --data_dir data/processed --embedding_path experiments/gvae_run/final_embeddings.pth

# 6. Generate Figures (at least one)
echo "Generating figures..."
python3 figures/generate_figure6_loss_curves.py
python3 figures/generate_figure4_tsne.py
python3 figures/generate_figure2_roc_pr.py
python3 figures/generate_figure_enrichment_plot.py --skip_api

echo "=== CI Test Completed Successfully ==="
