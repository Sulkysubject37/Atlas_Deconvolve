
import torch
import pandas as pd
import numpy as np
from gprofiler import GProfiler

from src.data.loader import GraphDataLoader

def main():
    # --- 1. Fetch FGFR signaling pathway proteins ---
    print("Fetching FGFR signaling pathway proteins from g:Profiler...")
    gp = GProfiler(return_dataframe=True)
    fgfr_pathway_id = "GO:0008543" # GO term for FGFR signaling pathway
    fgfr_genes_df = gp.convert(organism='hsapiens',
                             query=fgfr_pathway_id,
                             target_namespace='UNIPROTSWISSPROT') # The exact target doesn't matter, we just need the 'name' column

    if fgfr_genes_df is None or fgfr_genes_df.empty:
        print(f"Could not retrieve genes for pathway {fgfr_pathway_id}.")
        # Fallback to a known list if the API fails, now using the correct _HUMAN format
        fgfr_proteins = {"FGFR1_HUMAN", "FGFR2_HUMAN", "FGFR3_HUMAN", "FGFR4_HUMAN", "FGF1_HUMAN", "FGF2_HUMAN", "GRB2_HUMAN", "SOS1_HUMAN", "SHC1_HUMAN", "GAB1_HUMAN", "PTPN11_HUMAN", "PIK3R1_HUMAN", "PLCG1_HUMAN"}
        print("Using a small fallback list of core FGFR proteins.")
    else:
        # CONSTRUCT THE ID MANUALLY: Take gene symbol from 'name' and append '_HUMAN'
        gene_symbols = set(fgfr_genes_df['name'])
        fgfr_proteins = {f"{symbol}_HUMAN" for symbol in gene_symbols}
        print(f"Constructed {len(fgfr_proteins)} UniProt-style IDs for proteins in the pathway.")


    # --- 2. Load our network data and model ---
    print("Loading local network data and model embeddings...")
    data_loader = GraphDataLoader('data/processed')
    node2idx = data_loader.get_node2idx_mapping()
    idx2node = {v: k for k, v in node2idx.items()}
    all_network_proteins = set(node2idx.keys())

    # Find which FGFR proteins are in our network
    fgfr_in_network = fgfr_proteins.intersection(all_network_proteins)
    if not fgfr_in_network:
        print("None of the FGFR pathway proteins were found in the network. Cannot proceed.")
        return
    print(f"{len(fgfr_in_network)} FGFR proteins are present in our network.")

    # Load existing edges to filter them out later
    edgelist_df = pd.read_csv('data/processed/edgelist.tsv', sep='\t')
    # Clean the dataframe to prevent type errors
    edgelist_df.dropna(subset=['node1_name', 'node2_name'], inplace=True)
    edgelist_df['node1_name'] = edgelist_df['node1_name'].astype(str)
    edgelist_df['node2_name'] = edgelist_df['node2_name'].astype(str)
    existing_edges = set(map(tuple, edgelist_df.apply(lambda row: sorted((row['node1_name'], row['node2_name'])), axis=1).values))

    # Load embeddings
    embeddings = torch.load('experiments/gvae_run/final_embeddings.pth')
    
    # --- 3. Use decoder to predict all interaction scores ---
    print("Decoding interaction scores from embeddings...")
    # The GVAE decode method reconstructs the adjacency matrix from the latent embeddings.
    # We pass the embeddings through the decoder and apply a sigmoid to get probabilities.
    # Note: In a real GVAE, we'd instantiate the model first. Here, we approximate by
    # assuming a simple dot-product decoder on the final embeddings for simplicity.
    # A_hat = z * z.T
    pred_adj = torch.sigmoid(torch.matmul(embeddings, embeddings.t()))

    # --- 4. Identify novel high-confidence interactions ---
    print("Identifying novel high-confidence interactions...")
    fgfr_indices = [node2idx[p] for p in fgfr_in_network]
    network_indices = list(range(len(all_network_proteins)))
    
    novel_interactions = []

    for fgfr_protein in fgfr_in_network:
        fgfr_idx = node2idx[fgfr_protein]
        
        # Get scores for this FGFR protein against all others
        scores = pred_adj[fgfr_idx]
        
        # Find top partners
        top_scores, top_indices = torch.topk(scores, k=50)

        for score, partner_idx in zip(top_scores, top_indices):
            if fgfr_idx == partner_idx:
                continue

            partner_protein = idx2node[partner_idx.item()]
            
            # Prediction is novel if partner is not in FGFR pathway and edge doesn't already exist
            is_novel_partner = partner_protein not in fgfr_in_network
            edge_tuple = tuple(sorted((fgfr_protein, partner_protein)))

            if is_novel_partner and edge_tuple not in existing_edges:
                novel_interactions.append({
                    "fgfr_protein": fgfr_protein,
                    "predicted_partner": partner_protein,
                    "confidence": score.item()
                })

    # --- 5. Rank and display results ---
    print("Top 10 Novel Predicted Interactions with the FGFR Pathway:")
    
    # Sort by confidence and get unique interactions
    seen_edges = set()
    unique_novel_interactions = []
    for interaction in sorted(novel_interactions, key=lambda x: x['confidence'], reverse=True):
        edge = tuple(sorted((interaction['fgfr_protein'], interaction['predicted_partner'])))
        if edge not in seen_edges:
            unique_novel_interactions.append(interaction)
            seen_edges.add(edge)
    
    if not unique_novel_interactions:
        print("No novel interactions with confidence > threshold found.")
    else:
        results_df = pd.DataFrame(unique_novel_interactions).head(10)
        print(results_df.to_string(index=False))


if __name__ == '__main__':
    main()
