import torch
from torch_geometric.data import Data

def validate_graph_data(data: Data, num_nodes: int):
    """
    Validates a PyG Data object according to the Critical PyG Rules.

    Args:
        data (torch_geometric.data.Data): The graph data object to validate.
        num_nodes (int): The expected number of nodes in the graph.
    """
    if not isinstance(data, Data):
        raise TypeError(f"Expected data to be an instance of torch_geometric.data.Data, but got {type(data)}")

    # Rule 1 & 5: Explicit num_nodes and validate fake graph shapes
    if data.num_nodes is None:
        raise ValueError("data.num_nodes must be explicitly set.")
    if data.num_nodes != num_nodes:
        raise ValueError(f"Mismatched num_nodes. Expected {num_nodes}, got {data.num_nodes}")

    if data.x is not None:
        if data.x.size(0) != num_nodes:
            raise ValueError(f"Node feature tensor 'x' has {data.x.size(0)} rows, but expected {num_nodes}.")
        # Rule 6: Do not use N-dimensional features, stick to dimension 32 (as a recommendation)
        # This can be a warning or a stricter check based on configuration
        if data.x.dim() > 2 or (data.x.dim() == 2 and data.x.size(1) != 32):
             # For now, a warning, but could be an error if strictly enforced
            print(f"Warning: Node features 'x' has dim {data.x.dim()} or feature_dim {data.x.size(1)}, "
                  "recommended feature_dim is 32.")

    if data.edge_index is not None:
        if data.edge_index.dim() != 2 or data.edge_index.size(0) != 2:
            raise ValueError(f"Edge index tensor 'edge_index' has invalid shape {data.edge_index.shape}. "
                             "Expected (2, num_edges).")

        # Rule 5: Validate edge_index bounds
        if data.edge_index.max() >= num_nodes:
            raise ValueError(f"Edge index contains node ID {data.edge_index.max()} which is out of bounds for {num_nodes} nodes.")
        if data.edge_index.min() < 0:
            raise ValueError(f"Edge index contains negative node ID {data.edge_index.min()}.")

    # Rule 2: Clamp generator outputs - This is a usage rule, not directly a validation on a Data object,
    # but the output of generators producing node IDs should be clamped before forming edge_index.

    # Rule 3 & 4: Coalesce and add self-loops with explicit num_nodes
    # These are operations that should be performed correctly, not directly validated on the final Data object,
    # but their correct application would lead to a valid edge_index (checked above).

    print(f"Graph data validated successfully for {num_nodes} nodes.")

def clamp_node_id(node_id: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Clamps a node ID tensor to be within the valid range [0, num_nodes-1].
    (Corresponds to PyG Rule 2)

    Args:
        node_id (torch.Tensor): The tensor containing node IDs.
        num_nodes (int): The total number of nodes.

    Returns:
        torch.Tensor: The clamped node ID tensor.
    """
    return node_id.clamp(0, num_nodes - 1).long()

# Placeholder for other potential utility functions related to validation.
