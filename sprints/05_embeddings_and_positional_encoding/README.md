# Sprint 5: Embeddings & Positional Encoding

## Goals

- Understand and implement word embeddings using `nn.Embedding`.
- Explore different approaches to positional encoding (sinusoidal, learned).
- Gain foundational knowledge for representing sequences in PyTorch, crucial for subsequent Transformer components.
- Understand the concept of embedding visualization (though implementation might be deferred).

## Tasks

- [x] **`nn.Embedding` Basics:**
  - [x] Understand the purpose of embeddings (mapping discrete tokens to dense vectors).
  - [x] Implement `nn.Embedding`, understand `num_embeddings` and `embedding_dim`.
  - [x] Demonstrate looking up embeddings for input indices.
  - [x] _Results:_ `results/01_nn_embedding_basics.py`
  - [x] _Notes:_ `notes/01_nn_embedding_notes.md`
- [x] **Custom Embedding Layers:**
  - [x] Implement a simple embedding layer from scratch (optional, for understanding).
  - [x] Discuss initialization strategies for embeddings. (Covered conceptually)
  - [x] _Results:_ (Optional) `results/02_custom_embedding.py`
  - [x] _Notes:_ `notes/02_custom_embedding_notes.md`
- [x] **Sinusoidal Positional Encoding:**
  - [x] Understand why positional information is needed for sequence models like Transformers.
  - [x] Implement the sinusoidal positional encoding formula.
  - [x] Visualize the positional encoding patterns.
  - [x] Integrate positional encodings with token embeddings.
  - [x] _Results:_ `results/03_positional_encoding.py`
  - [x] _Notes:_ `notes/03_positional_encoding_notes.md`
- [x] **Learned Positional Embeddings:**
  - [x] Understand the concept of learning positional information using another `nn.Embedding` layer.
  - [x] Implement learned positional embeddings.
  - [x] Compare/contrast with sinusoidal encoding.
  - [x] _Results:_ `results/learned_pe_example.py`
  - [x] _Notes:_ `notes/04_learned_positional_embeddings.md`
- [x] **Embedding Visualization (Conceptual):**
  - [x] Understand techniques like t-SNE or PCA for visualizing high-dimensional embeddings.
  - [x] Discuss what insights visualization can provide.
  - [x] (Implementation Optional/Deferred) -> Implemented!
  - [x] _Notes:_ `notes/05_embedding_visualization_notes.md`, [`PCA`](notes/05a_pca_explained_novice.md), [`t-SNE`](notes/05b_tsne_explained.md), [`UMAP`](notes/05c_umap_explained_novice.md)
  - [x] _Results:_ [`PCA`](results/pca_example.py), [`t-SNE`](results/tsne_example.py), [`UMAP`](results/umap_example.py)

## Key Learnings & Insights

- `nn.Embedding` provides a trainable lookup table for converting token IDs into dense vectors.
- Sinusoidal Positional Encoding offers a fixed, non-learned way to inject sequence order information by adding unique vectors based on position and dimension, using sine and cosine functions of varying frequencies (the "Label Factory" analogy).
- Learned Positional Embeddings provide an alternative where the model learns optimal position representations using another `nn.Embedding` layer, offering flexibility but limited by `max_seq_len`.
- Dimensionality reduction (PCA, t-SNE, UMAP) helps visualize high-dimensional embeddings, revealing structure like clusters, with different methods prioritizing global vs. local structure.
- Environment dependency management (`pyproject.toml`, `uv pip sync`) can be complex, sometimes requiring explicit listing of transitive dependencies.

_(To be filled in further as the sprint progresses)_ -> Sprint Complete!

## Links to Notes and Results

- **Notes:**
  - [Embedding Basics](notes/01_nn_embedding_notes.md)
  - [Custom Embeddings](notes/02_custom_embedding_notes.md)
  - [Positional Encoding](notes/03_positional_encoding_notes.md)
  - [Learned Positional Embeddings](notes/04_learned_positional_embeddings.md)
  - [Embedding Visualization](notes/05_embedding_visualization_notes.md)
  - [PCA Explained](notes/05a_pca_explained_novice.md)
  - [t-SNE Explained](notes/05b_tsne_explained.md)
  - [UMAP Explained](notes/05c_umap_explained_novice.md)
- **Results:**
  - [nn.Embedding Basics](results/01_nn_embedding_basics.py)
  - [Custom Embedding (Optional)](results/02_custom_embedding.py)
  - [Sinusoidal Positional Encoding](results/03_positional_encoding.py)
  - [Learned Positional Embedding Example](results/learned_pe_example.py)
  - [PCA Example](results/pca_example.py)
  - [t-SNE Example](results/tsne_example.py)
  - [UMAP Example](results/umap_example.py)
