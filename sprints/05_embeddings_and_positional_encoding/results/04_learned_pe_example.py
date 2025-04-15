import torch
import torch.nn as nn

# --- Configuration ---
vocab_size = 100  # Size of our dummy vocabulary
embed_dim = 16  # Dimension for both word and position embeddings
max_seq_len = 50  # Maximum sequence length the model can handle
batch_size = 4  # Number of sequences in a batch
seq_len = 10  # Length of our example sequences (must be <= max_seq_len)

# --- Embedding Layers ---

# 1. Word Embedding Layer
# Maps each word index (0 to vocab_size-1) to a vector of size embed_dim
word_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
print(f"Word Embedding Layer: {word_embedding}\n")

# 2. Learned Positional Embedding Layer
# Maps each position index (0 to max_seq_len-1) to a vector of size embed_dim
# Note: The number of embeddings is max_seq_len
position_embedding = nn.Embedding(num_embeddings=max_seq_len, embedding_dim=embed_dim)
print(f"Position Embedding Layer: {position_embedding}\n")


# --- Example Input Data ---

# Create a dummy batch of input sequences (indices representing words)
# Shape: (batch_size, seq_len)
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
print(f"Input Word Indices (Shape: {input_ids.shape}):\n{input_ids}\n")

# Generate position indices for the sequence
# Shape: (seq_len,) -> then expanded to (batch_size, seq_len)
position_ids = torch.arange(seq_len, dtype=torch.long)
# We need position IDs for each sequence in the batch, so we expand:
position_ids = position_ids.unsqueeze(0).expand_as(
    input_ids
)  # Shape becomes (batch_size, seq_len)
# Alternative expansion: position_ids = position_ids.repeat(batch_size, 1)
print(f"Position Indices (Shape: {position_ids.shape}):\n{position_ids}\n")


# --- Get Embeddings ---

# 1. Get word embeddings
word_embeds = word_embedding(input_ids)
print(
    f"Word Embeddings (Shape: {word_embeds.shape})\n"
)  # Shape: (batch_size, seq_len, embed_dim)

# 2. Get positional embeddings
position_embeds = position_embedding(position_ids)
print(
    f"Positional Embeddings (Shape: {position_embeds.shape})\n"
)  # Shape: (batch_size, seq_len, embed_dim)


# --- Combine Embeddings ---

# Add word embeddings and positional embeddings element-wise
final_embeddings = word_embeds + position_embeds
print(f"Final Combined Embeddings (Shape: {final_embeddings.shape})\n")


# --- Important Notes ---
# - These embedding layers (word_embedding, position_embedding) would be part of a larger model.
# - Their weights are initialized (often randomly) and then *learned* during the model training process
#   via backpropagation based on the task's loss function.
# - The `max_seq_len` limits the length of sequences the model can directly process with these learned embeddings.
