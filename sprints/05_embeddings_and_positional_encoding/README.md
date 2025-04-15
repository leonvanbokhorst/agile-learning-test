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
- [ ] **Embedding Visualization (Conceptual):**
  - [ ] Understand techniques like t-SNE or PCA for visualizing high-dimensional embeddings.
  - [ ] Discuss what insights visualization can provide.
  - [ ] (Implementation Optional/Deferred)
  - [ ] _Notes:_ `notes/05_embedding_visualization_notes.md`

## Key Learnings & Insights

- `nn.Embedding` provides a trainable lookup table for converting token IDs into dense vectors.
- Sinusoidal Positional Encoding offers a fixed, non-learned way to inject sequence order information by adding unique vectors based on position and dimension, using sine and cosine functions of varying frequencies (the "Label Factory" analogy).
- Learned Positional Embeddings provide an alternative where the model learns optimal position representations using another `nn.Embedding` layer, offering flexibility but limited by `max_seq_len`.

_(To be filled in further as the sprint progresses)_

## Links to Notes and Results

- **Notes:**
  - [Embedding Basics](notes/01_nn_embedding_notes.md)
  - [Custom Embeddings](notes/02_custom_embedding_notes.md)
  - [Positional Encoding](notes/03_positional_encoding_notes.md)
  - [Learned Positional Embeddings](notes/04_learned_positional_embeddings.md)
  - [Embedding Visualization](notes/05_embedding_visualization_notes.md)
- **Results:**
  - [nn.Embedding Basics](results/01_nn_embedding_basics.py)
  - [Custom Embedding (Optional)](results/02_custom_embedding.py)
  - [Sinusoidal Positional Encoding](results/03_positional_encoding.py)
  - [Learned Positional Embedding Example](results/learned_pe_example.py)
