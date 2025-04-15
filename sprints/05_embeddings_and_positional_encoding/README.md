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
- [ ] **Sinusoidal Positional Encoding:**
  - [ ] Understand why positional information is needed for sequence models like Transformers.
  - [ ] Implement the sinusoidal positional encoding formula.
  - [ ] Visualize the positional encoding patterns.
  - [ ] Integrate positional encodings with token embeddings.
  - [ ] _Results:_ `results/03_sinusoidal_positional_encoding.py`
  - [ ] _Notes:_ `notes/03_positional_encoding_notes.md`
- [ ] **Learned Positional Embeddings:**
  - [ ] Understand the concept of learning positional information using another `nn.Embedding` layer.
  - [ ] Implement learned positional embeddings.
  - [ ] Compare/contrast with sinusoidal encoding.
  - [ ] _Results:_ `results/04_learned_positional_embedding.py`
  - [ ] _Notes:_ (Can be added to `notes/03_positional_encoding_notes.md`)
- [ ] **Embedding Visualization (Conceptual):**
  - [ ] Understand techniques like t-SNE or PCA for visualizing high-dimensional embeddings.
  - [ ] Discuss what insights visualization can provide.
  - [ ] (Implementation Optional/Deferred)
  - [ ] _Notes:_ `notes/04_embedding_visualization_notes.md`

## Key Learnings & Insights

_(To be filled in as the sprint progresses)_

## Links to Notes and Results

- **Notes:**
  - [Embedding Basics](notes/01_nn_embedding_notes.md)
  - [Custom Embeddings](notes/02_custom_embedding_notes.md)
  - [Positional Encoding](notes/03_positional_encoding_notes.md)
  - [Embedding Visualization](notes/04_embedding_visualization_notes.md)
- **Results:**
  - [nn.Embedding Basics](results/01_nn_embedding_basics.py)
  - [Custom Embedding (Optional)](results/02_custom_embedding.py)
  - [Sinusoidal Positional Encoding](results/03_sinusoidal_positional_encoding.py)
  - [Learned Positional Embedding](results/04_learned_positional_embedding.py)
