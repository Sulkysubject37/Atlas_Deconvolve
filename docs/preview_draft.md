Target Journal - PLOS Computational Biology 
Preview Version - 0.2.1 (Revised Draft)


# Title

**Atlas Deconvolve: A GVAE-based "Encode-then-Cluster" Strategy for Latent Module Discovery in Protein–Protein Interaction Networks**

---

# Abstract (Structured – High-Level Draft)

**Background:** Protein–protein interaction (PPI) networks are constructed by aggregating evidence across multiple biological contexts, resulting in a single static graph that hides functional, temporal, and condition-specific interaction modes.
**Methods:** We introduce a robust two-stage "Encode-then-Cluster" framework. In the "Encode" stage, a Graph Variational Autoencoder (GVAE) with an attention-based (GATv2) encoder is trained to learn high-fidelity latent representations of all proteins in the network. In the "Cluster" stage, the K-Means algorithm is applied to these embeddings to partition proteins into distinct functional modules.
**Results:** Using high-confidence human PPI data, our GVAE achieves outstanding link prediction performance (ROC AUC > 0.99), validating the quality of the learned embeddings. The subsequent clustering reveals highly interpretable subgraphs enriched in known protein complexes and pathways, including the Ribosome and Spliceosome.
**Conclusions:** This framework provides a transparent, powerful, and effective method for uncovering hidden biological modules from a single, static PPI network, emphasizing the critical role of high-quality representation learning.

**Author Summary:** Written for a broad scientific audience, this section will summarize the problem of aggregated PPIs and explain the importance of revealing latent structure using a two-step deep learning approach: first learning a "location" for each protein, then grouping them.

---

# Introduction (Polished Draft)

Protein–protein interactions underpin nearly every cellular process. However, PPI databases integrate evidence from diverse experimental systems such as tissues, stimuli, developmental stages, or perturbations. As a result, the commonly used “static interactome” is an artificial amalgamation of many underlying biological states. This conflation limits our ability to identify functional modules, reconstruct pathways, or detect disease-relevant rewiring.

Existing computational methods model PPIs as single networks or rely on external omics data (e.g., expression, perturbation profiles) to derive cell-type–specific or condition-specific interactomes. Yet no method can infer latent interaction modes directly from a unimodal PPI network.

To address this gap, we propose the "Encode-then-Cluster" framework, a robust two-stage strategy that first leverages a deep graph learning model to learn high-quality representations of proteins and then uses these representations to discover functional modules via clustering. This approach provides a powerful and interpretable path to deconvolving static PPI networks.

---

# 5. Literature Review 

*(This section is well-posed and effectively establishes the gap in the field, so it remains unchanged.)*

This section justifies the need for the model and positions your contribution.

5.1 Multi-layer and context-specific interactomes
	•	Tissue/cell-type specific PPIs
	•	Dynamic interactomes
	•	Perturbation/static integration methods
Limitation: require multi-omics input; cannot work from a single PPI graph.

5.2 Graph generative models in biology
	•	NetGAN, GraphVAE, VGAE, ARGA, GraphRNN
Limitation: produce single-graph samples; no latent layer decomposition.

5.3 Latent variable models for biological networks
	•	MMSBM & community models
	•	Graph embedding & node2vec limitations
Limitation: not generative w.r.t. adjacency; not multi-layer.

5.4 Need for unimodal, unsupervised decomposition
	•	No method can reconstruct multiple hidden subnetworks from a single PPI input.
	•	Biological processes such as complexes vs signaling vs ubiquitination leave different topological signatures, which generative latent layers could capture.

5.5 Summary

This literature review culminates in a clear gap: There is no existing method that, given only a single PPI network, can infer latent interaction layers that correspond to hidden biological states or modes. Atlas Deconvolve fills this gap.

---

# 6. Methods (moved before Results, as requested)

This section details our two-stage "Encode-then-Cluster" methodology.

### 6.1 Data Preparation
	•	HIPPIE dataset
	•	Filtering criteria (confidence score ≥ 0.75)
	•	Largest connected component
	•	Final stats: **10,957 nodes**, **80,486 edges**
	•	Edge split: 85% training, 5% validation, 10% testing.

### 6.2 Problem Definition

The objective is to partition the set of proteins `V` into `K` disjoint modules `C = {C_1, ..., C_K}`. We approach this in two stages:
1.  **Encode:** Learn a function `f: V -> Z` that maps each protein `v_i` to a low-dimensional latent embedding `z_i` that captures the network's topology.
2.  **Cluster:** Apply an unsupervised clustering algorithm to the embedding set `Z` to derive the partition `C`.

### 6.3 Model Architecture

#### 6.3.1 Stage 1: The "Encode" GVAE Model
We use a Graph Variational Autoencoder (GVAE) for representation learning.

*   **Encoder:** Employs two GATv2Conv layers for attention-based message passing. For each node `i`, it outputs parameters for a latent Gaussian distribution: `z_i ~ N(μ_i, σ_i^2)`.
*   **Decoder:** Reconstructs edge probabilities via a simple inner product: `P(A_ij=1 | z_i, z_j) = sigmoid(z_i^T z_j)`.

#### 6.3.2 Stage 2: The "Cluster" Step
*   **K-Means Clustering:** After training the GVAE, the final mean embeddings (`μ`) are extracted. Standard K-Means clustering is then applied to partition the proteins into `K` groups based on their positions in the latent space. We used `K=10`.

### 6.4 Loss Function
The GVAE is trained on a standard VAE loss (the Evidence Lower Bound, or ELBO), composed of two terms:
1.  **Reconstruction Loss:** Weighted Binary Cross-Entropy, to measure how accurately the decoder reconstructs the input graph.
2.  **KL Divergence:** A regularizer that encourages the learned latent distributions to be close to a standard normal distribution.

### 6.5 Optimization
	•	Adam optimizer
	•	Learning rate schedule with warm-up
	•	KL annealing
	•	Early stopping based on validation link prediction performance.

### 6.6 Evaluation Protocol
*   **Link Prediction:** To validate embedding quality, we measure **ROC AUC** and **Average Precision (AP)** on a held-out test set of edges.
*   **Topological Analysis:** We compare the degree distribution and triangle counts of the original graph to those of a graph reconstructed from the model's predictions to assess topological realism.
*   **Biological Enrichment:** To validate the biological meaning of the discovered clusters, we perform Gene Ontology (GO) enrichment analysis for each cluster.

---

# 7. Results

### 7.1 Link Prediction Performance Validates Embedding Quality
To verify that our GVAE encoder learned a high-fidelity representation of the network, we evaluated its performance on the link prediction task. The model achieved a **ROC AUC of 0.992** and an **Average Precision of 0.989** on the 10% held-out test set. This outstanding performance confirms the embeddings accurately capture the network's topological structure, making them a strong basis for downstream clustering.

*Figures Planned: ROC curve, PR curve.*

### 7.2 Topological Realism
The degree distribution and clustering coefficients of the graph reconstructed from the model's predictions closely matched those of the original input graph, indicating that the model successfully learned the fundamental structural properties of the PPI network.

*Figures Planned: Degree histogram overlay, Triangle count comparison.*

### 7.3 Biological Interpretability of Discovered Clusters
Applying K-Means to the GVAE embeddings partitioned the network into `K=10` distinct modules. GO enrichment analysis on these clusters revealed a strong separation of biological functions. Several clusters mapped directly to well-known cellular machinery:

*   **Cluster 3 (Ribosome):** Highly enriched for translation and ribonucleoprotein complex biogenesis (p < 1e-100).
*   **Cluster 5 (Spliceosome):** Enriched for mRNA splicing via spliceosome (p < 1e-50).
*   **Cluster 0 (Transcription):** Enriched for regulation of gene expression and nuclear proteins.
*   **Cluster 8 (Mitochondrion):** Enriched for aerobic respiration and mitochondrial components.
*   **Cluster 9 (Apoptosis):** Enriched for regulation of the apoptotic process.

These results demonstrate that the geometric structure of the latent space learned by the GVAE corresponds directly to the functional organization of the cell.

*Figures Planned: Latent Layer (Cluster) Visualizations, Enriched Pathways Table per Cluster.*

---

# 8. Discussion

### 8.1 Biological Interpretation
The "Encode-then-Cluster" framework successfully deconvolved the static PPI network into modules with clear, distinct biological identities. Our results effectively rediscovered fundamental biological knowledge from network topology alone, validating the approach. The discovered modules can now be used to generate new hypotheses; for instance, uncharacterized proteins within the "Spliceosome" cluster are strong candidates for further investigation as potential splicing factors.

### 8.2 Computational Perspective
The strength of this two-stage approach lies in its modularity and transparency. The near-perfect link prediction result provides a quantitative checkpoint that assures the quality of the embeddings before the more exploratory clustering step is even attempted. This avoids the pitfalls of complex, end-to-end models where poor results can be difficult to diagnose.

### 8.3 Limitations
The choice of `K` (the number of clusters) is a primary limitation and was set manually. Future work could employ statistical methods to guide the selection of `K`. Additionally, K-Means forces each protein into a single cluster, whereas many proteins are pleiotropic. A soft clustering approach could provide a more realistic model of protein function.

### 8.4 Future Directions
This framework can be readily extended. The GVAE could be conditioned on external data (like gene expression) to learn context-specific embeddings. Furthermore, replacing K-Means with more sophisticated, overlapping clustering algorithms could yield richer insights into the multifunctional nature of proteins.

---

# Conclusion (Polished Draft)

Our GVAE-based "Encode-then-Cluster" framework is a novel, robust, and interpretable method for decomposing a single PPI network into biologically meaningful functional modules. This work provides a foundational step toward interpretable interactome modeling and demonstrates that high-quality representation learning is a key prerequisite for biological discovery in network data.

---

# Figures (Planned)
	1.	Architecture Diagram for the "Encode-then-Cluster" Framework
	2.	Link Prediction ROC/PR Curves
	3.	Topology Comparison (degree distribution, clustering coefficient)
	4.	t-SNE/UMAP visualization of the learned embeddings, colored by cluster
	5.	Example Cluster Visualizations (e.g., the Ribosome cluster)
	6.	GVAE Loss Curves (Reconstruction & KL)

---

# Tables (Planned)
	1.	Dataset Summary
	2.	Link Prediction Results (AUC, AP) vs. Baselines
	3.	Graph Motif Preservation Metrics
	4.	Top 5 Enriched GO Terms per Discovered Cluster

---

# Supporting Information (Planned)
	•	Hyperparameter settings
	•	Ablation studies (e.g., effect of embedding dimensionality)
	•	Full enrichment tables for all clusters
	•	Additional cluster visualizations

