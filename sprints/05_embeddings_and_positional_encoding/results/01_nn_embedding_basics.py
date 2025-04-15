import torch
import torch.nn as nn

# --- What are Embeddings? ---
# Embeddings map discrete items (like words, represented by integer indices)
# to continuous, dense vectors. These vectors can capture semantic relationships
# and are learnable parameters within a neural network.

# --- nn.Embedding ---
# A simple lookup table that stores embeddings of a fixed dictionary and size.

# Vocabulary size (e.g., how many unique words we have)
num_embeddings = 10
# Dimension of the embedding vector for each word
embedding_dim = 3

# Create the embedding layer
# It initializes a weight matrix of size (num_embeddings, embedding_dim)
embedding_layer = nn.Embedding(
    num_embeddings=num_embeddings, embedding_dim=embedding_dim
)

print(f"Embedding layer weight shape: {embedding_layer.weight.shape}")
print(f"Initial embedding weights:\n{embedding_layer.weight}\n")

# --- Looking up Embeddings ---
# We need input indices (must be LongTensor) to look up embeddings.
# Let's look up embeddings for indices 1, 5, and 0.
# Input shape: (sequence_length) or (batch_size, sequence_length)
input_indices_1d = torch.LongTensor([1, 5, 0, 5])  # A single sequence
print(f"Input indices (1D): {input_indices_1d.shape} -> {input_indices_1d}")

# The output will have an extra dimension: the embedding_dim
output_embeddings_1d = embedding_layer(input_indices_1d)
print(f"Output embeddings (1D): {output_embeddings_1d.shape}\n{output_embeddings_1d}\n")

# Let's try a batch of sequences
# Input shape: (batch_size=2, sequence_length=4)
input_indices_2d = torch.LongTensor([[1, 2, 4, 5], [4, 3, 0, 9]])
print(f"Input indices (2D): {input_indices_2d.shape}\n{input_indices_2d}")

output_embeddings_2d = embedding_layer(input_indices_2d)
# Output shape: (batch_size=2, sequence_length=4, embedding_dim=3)
print(f"Output embeddings (2D): {output_embeddings_2d.shape}\n{output_embeddings_2d}\n")

# --- Key Parameters ---
# num_embeddings: Size of the dictionary (vocabulary size). Indices must be < num_embeddings.
# embedding_dim: The size of each embedding vector.

# --- Example: Verifying Lookup ---
# The embedding for index 5 should be the 5th row of the weight matrix.
index_to_check = 5
lookup_result = embedding_layer(torch.LongTensor([index_to_check]))
weight_row = embedding_layer.weight[index_to_check]

print(f"Lookup result for index {index_to_check}:\n{lookup_result}")
print(f"Weight matrix row {index_to_check}:\n{weight_row}")
print(f"Are they the same? {torch.equal(lookup_result.squeeze(), weight_row)}")

# Note: The weights are initialized randomly (often from a normal distribution)
# and are typically learned during model training.
