import torch
import pandas as pd
import numpy as np
from gprofiler import GProfiler
import argparse
import os

from src.data.loader import GraphDataLoader

def main():
    parser = argparse.ArgumentParser(description="Predict novel interactions for FGFR pathway")
    parser.add_argument('--skip_api', action='store_true', help='Skip g:Profiler API fetch and use fallback list')
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Directory containing processed graph data')
    parser.add_argument('--embedding_path', type=str, default='experiments/gvae_run/final_embeddings.pth', help='Path to saved node embeddings')
    parser.add_argument('--output_limit', type=int, default=10, help='Number of top novel interactions to display')
    
    args = parser.parse_args()

    # --- 1. Fetch FGFR signaling pathway proteins ---
    fgfr_proteins = {"FGFR1_HUMAN", "FGFR2_HUMAN", "FGFR3_HUMAN", "FGFR4_HUMAN", "FGF1_HUMAN", "FGF2_HUMAN", "GRB2_HUMAN", "SOS1_HUMAN", "SHC1_HUMAN", "GAB1_HUMAN", "PTPN11_HUMAN", "PIK3R1_HUMAN", "PLCG1_HUMAN"}

    if not args.skip_api:
        print("Fetching FGFR signaling pathway proteins from g:Profiler...")
        try:
            gp = GProfiler(return_dataframe=True)
            fgfr_pathway_id = "GO:0008543" # GO term for FGFR signaling pathway
            fgfr_genes_df = gp.convert(organism='hsapiens',
                                     query=fgfr_pathway_id,
                                     target_namespace='UNIPROTSWISSPROT')

            if fgfr_genes_df is not None and not fgfr_genes_df.empty:
                gene_symbols = set(fgfr_genes_df['name'])
                fgfr_proteins = {f"{symbol}_HUMAN" for symbol in gene_symbols}
                print(f"Constructed {len(fgfr_proteins)} UniProt-style IDs for proteins in the pathway.")
            else:
                print("g:Profiler returned no results. Using fallback list.")
        except Exception as e:
            print(f"Error fetching from g:Profiler: {e}. Using fallback list.")
    else:
        print("Skipping API fetch. Using core FGFR proteins fallback list.")


    # --- 2. Load our network data and model ---
    print("Loading local network data and model embeddings...")
    data_loader = GraphDataLoader(args.data_dir)
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
    edgelist_path = os.path.join(args.data_dir, 'edgelist.tsv')
    edgelist_df = pd.read_csv(edgelist_path, sep='\t')
    # Clean the dataframe to prevent type errors
    edgelist_df.dropna(subset=['node1_name', 'node2_name'], inplace=True)
    edgelist_df['node1_name'] = edgelist_df['node1_name'].astype(str)
    edgelist_df['node2_name'] = edgelist_df['node2_name'].astype(str)
    existing_edges = set(map(tuple, edgelist_df.apply(lambda row: sorted((row['node1_name'], row['node2_name'])), axis=1).values))

    # Load embeddings
    if not os.path.exists(args.embedding_path):
        print(f"Error: Embedding file not found at '{args.embedding_path}'.")
        return
    embeddings = torch.load(args.embedding_path)
    
    # --- 3. Use decoder to predict all interaction scores ---
    print("Decoding interaction scores from embeddings...")
    pred_adj = torch.sigmoid(torch.matmul(embeddings, embeddings.t()))

    # --- 4. Identify novel high-confidence interactions ---
    print("Identifying novel high-confidence interactions...")
    novel_interactions = []

    for fgfr_protein in fgfr_in_network:
        fgfr_idx = node2idx[fgfr_protein]
        
        # Get scores for this FGFR protein against all others
        scores = pred_adj[fgfr_idx]
        
        # Find top partners
        top_scores, top_indices = torch.topk(scores, k=min(50, len(all_network_proteins)))

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
    print(f"Top {args.output_limit} Novel Predicted Interactions with the FGFR Pathway:")
    
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
        results_df = pd.DataFrame(unique_novel_interactions).head(args.output_limit)
        print(results_df.to_string(index=False))


if __name__ == '__main__':
    main()
