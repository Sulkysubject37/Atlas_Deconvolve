import os
import argparse
import yaml
from box import Box
import torch
import networkx as nx
from torch_geometric.data import Data
from scipy.sparse import load_npz, coo_matrix

from src.data.loader import GraphDataLoader
from src.models.node_features import NodeFeatureGenerator
from src.models.walk_generator import WalkGenerator
from src.models.refinement_net import RefinementNetwork
from src.models.critic import GraphCritic # Critic might not be directly used for generation, but for consistency

from src.utils.metrics import (
    calculate_roc_pr_auc,
    reconstruction_accuracy,
    degree_distribution_ks_test,
    triangle_count_consistency,
    per_layer_edge_f1 # For synthetic data
)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Box(config_dict)

def load_model_components(cfg, num_nodes, device):
    """Loads all model components from saved checkpoints."""
    model_dir = os.path.join("experiments", cfg.EXPERIMENT_NAME, "checkpoints")

    node_feature_generator = NodeFeatureGenerator(
        num_nodes=num_nodes,
        feature_dim=cfg.MODEL.NODE_FEATURE_DIM,
        feature_type=cfg.MODEL.NODE_FEATURE_TYPE
    ).to(device)
    node_feature_generator.load_state_dict(torch.load(os.path.join(model_dir, "node_features.pth"), map_location=device))
    node_feature_generator.eval()

    walk_generator = WalkGenerator(
        input_dim=cfg.MODEL.NODE_FEATURE_DIM,
        hidden_dim=cfg.MODEL.WALK_HIDDEN_DIM,
        num_nodes=num_nodes,
        walk_length=cfg.MODEL.WALK_LENGTH
    ).to(device)
    walk_generator.load_state_dict(torch.load(os.path.join(model_dir, "walk_generator.pth"), map_location=device))
    walk_generator.eval()

    refinement_net = RefinementNetwork(
        input_dim=cfg.MODEL.NODE_FEATURE_DIM,
        hidden_dim=cfg.MODEL.REFINEMENT_HIDDEN_DIM,
        output_dim=cfg.MODEL.REFINEMENT_OUTPUT_DIM,
        num_layers=cfg.MODEL.REFINEMENT_LAYERS,
        num_nodes=num_nodes
    ).to(device)
    refinement_net.load_state_dict(torch.load(os.path.join(model_dir, "refinement_net.pth"), map_location=device))
    refinement_net.eval()
    
    # Critic is not typically used for generation, but load for completeness if needed for feature extraction
    critic = GraphCritic(
        input_dim=cfg.MODEL.NODE_FEATURE_DIM,
        hidden_dim=cfg.MODEL.CRITIC_HIDDEN_DIM,
        num_layers=cfg.MODEL.CRITIC_LAYERS
    ).to(device)
    critic.load_state_dict(torch.load(os.path.join(model_dir, "critic.pth"), map_location=device))
    critic.eval()

    return node_feature_generator, walk_generator, refinement_net, critic

def generate_graphs_from_model(
    node_feature_generator: NodeFeatureGenerator,
    walk_generator: WalkGenerator,
    refinement_net: RefinementNetwork,
    num_nodes: int,
    cfg: Box,
    device: torch.device,
    data_loader: GraphDataLoader # Added data_loader to get node degrees if needed
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Generates K latent layer adjacency matrices and their aggregated form."""
    
    # Get all node features (potentially depends on NODE_FEATURE_TYPE)
    if cfg.MODEL.NODE_FEATURE_TYPE == 'degree_learned':
        node_features_all = node_feature_generator(degrees=data_loader.get_node_degrees().to(device))
    else:
        node_features_all = node_feature_generator()
    
    latent_adj_matrices = []

    # For evaluation, we might want to sample multiple times and average, or just one deterministic pass
    # For now, one pass
    initial_nodes_for_walks = torch.randint(0, num_nodes, (cfg.TRAIN.BATCH_SIZE,), device=device) # Use batch_size for sampling walks
    
    with torch.no_grad():
        for _ in range(cfg.MODEL.NUM_LATENT_LAYERS):
            # Generate walks for each layer
            walks_batch = walk_generator(initial_nodes_for_walks, node_features_all)

            # Convert walks to edge_index
            edges_list = []
            for walk in walks_batch:
                for i in range(walk.size(0) - 1):
                    edges_list.append([walk[i], walk[i+1]])
                    edges_list.append([walk[i+1], walk[i]]) # Add reverse edge for undirected

            if not edges_list:
                layer_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            else:
                layer_edge_index = torch.tensor(edges_list, dtype=torch.long, device=device).t().contiguous()

            # Refine adjacency for the current layer
            refined_adj = refinement_net(node_features_all, layer_edge_index)
            latent_adj_matrices.append(refined_adj)

        # Probabilistic Layer Aggregation
        aggregated_adj = torch.ones_like(latent_adj_matrices[0])
        for adj_k in latent_adj_matrices:
            aggregated_adj = aggregated_adj * (1 - adj_k)
        aggregated_adj = 1 - aggregated_adj
        
    return latent_adj_matrices, aggregated_adj

def main():
    parser = argparse.ArgumentParser(description="Evaluate Atlas Deconvolve Model")
    parser.add_argument('--config', type=str, default='src/experiments/configs/config.yaml',
                        help='Path to configuration YAML file')
    # Add optional argument for loading true latent layers for synthetic data evaluation
    parser.add_argument('--true_latent_dir', type=str, default=None,
                        help='Path to directory containing true latent layers for synthetic data evaluation (e.g., data/synthetic)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.SYSTEM.DEVICE if torch.cuda.is_available() else "cpu")

    # Load Real Graph Data (or processed data from synthetic run)
    data_loader = GraphDataLoader(cfg.DATA.PROCESSED_DIR)
    real_graph_data = data_loader.get_graph_data().to(device)
    num_nodes = data_loader.get_num_nodes()
    real_adj = real_graph_data.to_dense_adj()[0].to(device)

    # Load Model Components
    node_feature_generator, walk_generator, refinement_net, _ = load_model_components(cfg, num_nodes, device)

    # Generate Graphs from Model
    print("Generating graphs from trained model...")
    latent_fake_adjs, aggregated_fake_adj = generate_graphs_from_model(
        node_feature_generator, walk_generator, refinement_net, num_nodes, cfg, device
    )

    # --- Evaluation ---
    print("\n--- Evaluation Results ---")

    # 1. Quantitative Evaluation on Aggregated Graph
    print("\nAggregated Graph Metrics:")
    
    # Convert aggregated_fake_adj to a NetworkX graph for degree/triangle counts
    # A threshold is needed to binarize for NetworkX conversion
    threshold_for_nx = 0.5
    binary_fake_adj_np = (aggregated_fake_adj.cpu().numpy() > threshold_for_nx).astype(int)
    fake_graph_nx = nx.from_numpy_array(binary_fake_adj_np)
    real_graph_nx = nx.from_numpy_array(real_adj.cpu().numpy()) # Assuming real_adj is already binary from preprocessing

    roc_auc, ap_score = calculate_roc_pr_auc(real_adj, aggregated_fake_adj)
    print(f"  Link Prediction ROC AUC: {roc_auc:.4f}")
    print(f"  Link Prediction AP Score: {ap_score:.4f}")

    rec_acc = reconstruction_accuracy(real_adj, aggregated_fake_adj)
    print(f"  Reconstruction Accuracy (threshold=0.5): {rec_acc:.4f}")

    ks_p_value = degree_distribution_ks_test(real_graph_nx, fake_graph_nx)
    print(f"  Degree Distribution KS-test p-value: {ks_p_value:.4f}")

    tri_diff = triangle_count_consistency(real_graph_nx, fake_graph_nx)
    print(f"  Triangle Count Absolute Difference: {tri_diff}")
    
    # Graphlet similarity - placeholder in metrics.py
    # graphlet_sim = graphlet_similarity(real_graph_nx, fake_graph_nx)
    # print(f"  Graphlet Similarity: {graphlet_sim:.4f}")

    # 2. Layer Quality (if true latent layers are available)
    if args.true_latent_dir:
        print("\nLayer Quality Metrics (Synthetic Data):")
        true_latent_adjs = []
        for i in range(cfg.MODEL.NUM_LATENT_LAYERS):
            layer_path = os.path.join(args.true_latent_dir, f'layer_{i}', 'adj.npz')
            if os.path.exists(layer_path):
                true_latent_adjs.append(torch.tensor(load_npz(layer_path).toarray(), dtype=torch.float32, device=device))
            else:
                print(f"Warning: True latent layer {i} not found at {layer_path}. Skipping layer metrics.")
                true_latent_adjs = [] # Clear list if any missing to avoid partial evaluation
                break
        
        if true_latent_adjs:
            if len(true_latent_adjs) != len(latent_fake_adjs):
                print("Warning: Number of true and fake latent layers do not match. Skipping layer metrics.")
            else:
                for i in range(len(true_latent_adjs)):
                    f1 = per_layer_edge_f1(true_latent_adjs[i], latent_fake_adjs[i])
                    print(f"  Layer {i} Edge F1 Score: {f1:.4f}")
                # ARI and Motif recovery are placeholders in metrics.py

if __name__ == "__main__":
    main()
