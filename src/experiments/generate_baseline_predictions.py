import os
import json
import torch
import networkx as nx
import numpy as np
from tqdm import tqdm
import argparse

from src.data.loader import GraphDataLoader

def generate_adamic_adar_predictions(train_data_dir: str, full_data_dir: str, output_dir: str):
    """
    Calculates Adamic-Adar link prediction scores for the test set edges
    and saves them for later evaluation.
    """
    # --- 1. Load graph data ---
    print("Loading graph data...")
    # Load the training graph without re-splitting it by setting val_split and test_split to 0.
    data_loader = GraphDataLoader(train_data_dir, val_split=0, test_split=0)
    adj_train_small = data_loader.get_train_graph_tensor().numpy()

    # We need to ensure the training graph has the same number of nodes as the full graph,
    # so that test set node indices are valid.
    with open(os.path.join(full_data_dir, 'stats.json'), 'r') as f:
        full_stats = json.load(f)
    num_nodes_full = full_stats['num_nodes']

    # Create a full-sized adjacency matrix and copy the training data into it.
    adj_train_full = np.zeros((num_nodes_full, num_nodes_full))
    num_nodes_small = adj_train_small.shape[0]
    adj_train_full[:num_nodes_small, :num_nodes_small] = adj_train_small
    
    G_train = nx.from_numpy_array(adj_train_full)
    
    print(f"Training graph loaded: {G_train.number_of_nodes()} nodes, {G_train.number_of_edges()} edges.")

    # --- 2. Prepare test edges ---
    # We need to sample negative edges for the test set.
    # The GraphDataLoader can do this for us.
    print("Loading test edges and sampling negatives...")
    # A bit of a hack: we instantiate a loader for the full graph to use its sampling method.
    full_loader = GraphDataLoader(full_data_dir)
    test_pos_edges = full_loader.get_test_edges()[0]
    test_neg_edges = full_loader.get_test_edges()[1]

    # Create an "ebunch" (edge bunch) for the Adamic-Adar function
    test_edges = np.concatenate([test_pos_edges, test_neg_edges], axis=1).T
    test_labels = np.concatenate([np.ones(test_pos_edges.shape[1]), np.zeros(test_neg_edges.shape[1])])

    # --- 3. Calculate Adamic-Adar scores ---
    print(f"Calculating Adamic-Adar scores for {len(test_edges)} test edges...")
    
    # The adamic_adar_index function takes an 'ebunch' of edges to score.
    predictions = nx.adamic_adar_index(G_train, ebunch=test_edges)
    
    # The function returns a generator. We need to unpack it into a list of scores.
    # We also need to be careful about the order. The ebunch order is preserved.
    scores = [p for u, v, p in tqdm(predictions, total=len(test_edges))]

    # --- 4. Save predictions and labels ---
    # Save the scores and corresponding labels to be used in the evaluation script.
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'scores': torch.tensor(scores, dtype=torch.float32),
        'labels': torch.tensor(test_labels, dtype=torch.float32)
    }
    
    save_path = os.path.join(output_dir, 'adamic_adar_predictions.pt')
    torch.save(results, save_path)
    
    print(f"Adamic-Adar predictions saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Adamic-Adar baseline predictions")
    parser.add_argument('--train_dir', type=str, default='data/processed_split/train',
                        help='Directory containing processed training split')
    parser.add_argument('--full_dir', type=str, default='data/processed',
                        help='Directory containing processed full graph data')
    parser.add_argument('--output_dir', type=str, default='experiments/gvae_run',
                        help='Directory to save baseline predictions')
    
    args = parser.parse_args()

    generate_adamic_adar_predictions(args.train_dir, args.full_dir, args.output_dir)
