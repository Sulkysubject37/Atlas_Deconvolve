import pandas as pd
import networkx as nx
import torch
from node2vec import Node2Vec

from src.data.loader import GraphDataLoader

import os

def generate_embeddings():
    """
    Trains a Node2Vec model on the training graph and saves the learned embeddings.
    """
    # --- 1. Load the training graph ---
    # We must train Node2Vec ONLY on the training portion of the graph for a fair comparison.
    print("Loading training graph data...")
    # The refactored data loader gives us the training adjacency matrix
    data_loader = GraphDataLoader('data/processed')
    adj_train = data_loader.get_train_graph_tensor().numpy()
    idx2node = {v: k for k, v in data_loader.get_node2idx_mapping().items()}
    
    # Create a networkx graph from the training adjacency matrix
    G_train = nx.from_numpy_array(adj_train)
    print(f"Training graph loaded: {G_train.number_of_nodes()} nodes, {G_train.number_of_edges()} edges.")

    # --- 2. Configure and run Node2Vec ---
    # The dimensions should match our GVAE model for a fair comparison
    # Use a single worker to avoid a persistent multiprocessing bug in gensim on macOS.
    num_workers = 1
    print(f"Initializing Node2Vec model with {num_workers} worker (single-threaded)...")
    node2vec = Node2Vec(G_train, dimensions=64, walk_length=30, num_walks=200, workers=num_workers, quiet=True)

    print("Training Node2Vec model (this may take several minutes)...")
    # Any keywords arguments will be passed to gensim's Word2Vec
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    
    print("Node2Vec training complete.")

    # --- 3. Save the embeddings ---
    # The model's wv property holds the learned embeddings
    # We will save them in a dictionary mapping the protein name to the embedding vector
    embeddings_dict = {idx2node[int(node_idx)]: model.wv[node_idx] for node_idx in model.wv.index_to_key}
    
    # Convert to a single tensor for consistency, plus the mapping
    # Sort by node index to ensure order
    node_indices = sorted(embeddings_dict.keys(), key=lambda name: data_loader.node2idx[name])
    embedding_tensor = torch.tensor([embeddings_dict[name] for name in node_indices])

    save_path = 'experiments/gvae_run/node2vec_embeddings.pth'
    torch.save(embedding_tensor, save_path)
    print(f"Node2Vec embeddings saved to {save_path}")

if __name__ == '__main__':
    generate_embeddings()
