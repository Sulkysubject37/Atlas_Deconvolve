import os
import argparse
import yaml
from box import Box
import torch
import numpy as np
import random

from src.data.loader import GraphDataLoader
from src.models.node_features import NodeFeatureGenerator
from src.models.walk_generator import WalkGenerator
from src.models.refinement_net import RefinementNetwork
from src.models.critic import GraphCritic
from src.training.trainer import Trainer

def load_config(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Box(config_dict)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Best Atlas Deconvolve Model")
    parser.add_argument('--config', type=str, default='src/experiments/configs/config.yaml',
                        help='Path to configuration YAML file')
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)
    print(f"Loaded configuration from {args.config}")

    # Determine device
    device = torch.device(cfg.SYSTEM.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    print(f"Loading data from {cfg.DATA.PROCESSED_DIR}...")
    data_loader = GraphDataLoader(cfg.DATA.PROCESSED_DIR)
    graph_data = data_loader.get_graph_data()
    num_nodes = data_loader.get_num_nodes()
    print(f"Graph loaded: {num_nodes} nodes.")

    # Initialize Models
    print("Initializing models...")
    node_feature_generator = NodeFeatureGenerator(
        num_nodes=num_nodes,
        feature_dim=cfg.MODEL.NODE_FEATURE_DIM,
        feature_type=cfg.MODEL.NODE_FEATURE_TYPE
    ).to(device)

    walk_generator = WalkGenerator(
        input_dim=cfg.MODEL.NODE_FEATURE_DIM,
        hidden_dim=cfg.MODEL.WALK_HIDDEN_DIM,
        num_nodes=num_nodes,
        walk_length=cfg.MODEL.WALK_LENGTH
    ).to(device)

    refinement_net = RefinementNetwork(
        input_dim=cfg.MODEL.NODE_FEATURE_DIM,
        hidden_dim=cfg.MODEL.REFINEMENT_HIDDEN_DIM,
        output_dim=cfg.MODEL.REFINEMENT_OUTPUT_DIM,
        num_layers=cfg.MODEL.REFINEMENT_LAYERS,
        num_nodes=num_nodes,
        heads=cfg.MODEL.GAT_HEADS,
        dropout=cfg.MODEL.GAT_DROPOUT
    ).to(device)

    critic = GraphCritic(
        input_dim=cfg.MODEL.NODE_FEATURE_DIM,
        hidden_dim=cfg.MODEL.CRITIC_HIDDEN_DIM,
        num_layers=cfg.MODEL.CRITIC_LAYERS
    ).to(device)

    # Load Best Checkpoints
    checkpoint_dir = os.path.join("experiments", cfg.EXPERIMENT_NAME, "checkpoints")
    print(f"Loading best models from {checkpoint_dir}...")
    
    try:
        node_feature_generator.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_node_features.pth"), map_location=device))
        walk_generator.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_walk_generator.pth"), map_location=device))
        refinement_net.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_refinement_net.pth"), map_location=device))
        critic.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_critic.pth"), map_location=device))
        print("Models loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading checkpoints: {e}")
        return

    # Initialize Trainer (just for the evaluate method)
    trainer = Trainer(
        node_feature_generator=node_feature_generator,
        walk_generator=walk_generator,
        refinement_net=refinement_net,
        critic=critic,
        num_nodes=num_nodes,
        cfg=cfg
    )

    # Run Evaluation
    print("Evaluating best model...")
    metrics = trainer.evaluate(graph_data)
    
    print("\n--- Best Model Results ---")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

if __name__ == "__main__":
    main()
