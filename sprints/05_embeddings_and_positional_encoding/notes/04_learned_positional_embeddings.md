# Learned Positional Embeddings

Learned Positional Embeddings are a mechanism used in Transformer models (and other sequence models) to provide the model with information about the position of tokens within a sequence. Unlike fixed methods like [sinusoidal positional encoding](03a_positional_encoding.md), these embeddings are learned during the training process.

## How it Works

1.  **Embedding Layer for Positions:** A standard embedding layer is created, similar to the one used for word embeddings. However, this layer's vocabulary size corresponds to the maximum expected sequence length (e.g., 512 positions).
2.  **Lookup:** For each token in the input sequence, its position index (0 for the first token, 1 for the second, etc.) is used to look up the corresponding position embedding vector from this layer.
3.  **Combination:** This position embedding vector is then typically added to the token's word embedding vector. The resulting combined vector represents both the token's identity and its position.
4.  **Learning:** The weights (vectors) within the position embedding layer are initialized (often randomly or using a simple pattern) and then updated via backpropagation during model training, just like all other model parameters (including the word embeddings). The model learns the optimal vector representations for each position based on the task it's being trained on.

## Analogy: Learning City Routes

- **Fixed Positional Encoding (Sinusoidal):** Using a strict grid map (like Manhattan). It works predictably for any location (sequence length) but might not represent the _best_ or most intuitive routes (positional relationships) for your specific needs.
- **Learned Positional Embeddings:** Exploring the city yourself. You learn landmarks, shortcuts, and traffic patterns relevant to _your_ common destinations (the task). Your internal map is optimized but might only cover the areas you've explored (maximum sequence length during training).

## Advantages

- **Flexibility & Optimality:** The model can learn positional representations potentially better suited to the specific language nuances and task requirements than a fixed formula.
- **Conceptual Simplicity:** Uses a standard, well-understood neural network component (embedding layer).

## Disadvantages

- **Limited Sequence Length:** Only works effectively up to the maximum sequence length defined and learned during training. Extrapolation to longer sequences is problematic.
- **More Parameters:** Increases the total number of trainable parameters in the model compared to fixed methods.

## Summary

Learned Positional Embeddings treat positions like vocabulary items, allowing the model to learn dedicated vector representations for each position index up to a defined limit. This provides flexibility but restricts the model to the sequence lengths seen during training.
