import torch
import torch.nn as nn

# --- Why Build a Custom Embedding? ---
# Understanding nn.Embedding is easier when you see it's essentially just:
# 1. A learnable weight matrix (the embedding table).
# 2. An indexing operation in the forward pass.


class CustomEmbedding(nn.Module):
    """A simplified custom embedding layer mimicking nn.Embedding."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        """Initializes the embedding layer.

        Args:
            num_embeddings: The size of the vocabulary (how many items).
            embedding_dim: The dimension of the embedding vector for each item.
        """
        super().__init__()  # IMPORTANT: Always call parent constructor

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # --- The Core: The Learnable Weight Matrix ---
        # Create the embedding matrix as a learnable parameter.
        # nn.Parameter tells PyTorch to track gradients for this tensor.
        # Initialize weights randomly (e.g., from a standard normal distribution)
        # Shape: (num_embeddings, embedding_dim)
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        # Note: nn.Embedding has more sophisticated default initialization,
        # but random normal is fine for demonstration.

    def forward(self, input_indices: torch.Tensor) -> torch.Tensor:
        """Performs the embedding lookup.

        Args:
            input_indices: A tensor of Long type containing indices to look up.
                           Shape: (batch_size, seq_len) or (seq_len).

        Returns:
            The corresponding embeddings. Shape: (batch_size, seq_len, embedding_dim)
            or (seq_len, embedding_dim).
        """
        # --- The Core: The Indexing Operation ---
        # We simply use the input indices to select rows from our weight matrix.
        # PyTorch handles the indexing efficiently even for batches.
        # Input indices MUST be LongTensor for indexing.
        if not torch.is_tensor(input_indices) or input_indices.dtype != torch.long:
            raise TypeError("Input indices must be a LongTensor.")

        # Check if indices are within valid range
        if input_indices.max() >= self.num_embeddings or input_indices.min() < 0:
            raise IndexError(
                f"Index out of range (expected 0 <= index < {self.num_embeddings}, but got min {input_indices.min()} max {input_indices.max()})"
            )

        # The actual lookup!
        embedded_vectors = self.weight[input_indices]

        return embedded_vectors

    def extra_repr(self) -> str:
        """Provides a nice string representation when printing the module."""
        return (
            f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"
        )


# --- Testing the Custom Embedding Layer ---
if __name__ == "__main__":
    vocab_size = 10
    embed_dim = 4

    # Instantiate our custom layer
    custom_embed = CustomEmbedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
    print(f"Custom Embedding Layer:\n{custom_embed}\n")
    print(f"Custom Embedding Weight Shape: {custom_embed.weight.shape}\n")

    # Instantiate the built-in nn.Embedding for comparison
    # Copy weights from custom to built-in for a fair comparison
    pytorch_embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
    # Use .data to assign weights without involving autograd tracking here
    pytorch_embed.weight.data.copy_(custom_embed.weight.data)
    print(f"PyTorch nn.Embedding Layer:\n{pytorch_embed}\n")
    print(f"PyTorch Embedding Weight Shape: {pytorch_embed.weight.shape}\n")

    # Create some input indices
    input_ids = torch.LongTensor([[0, 3, 5, 3], [9, 1, 0, 2]])
    print(f"Input Indices:\n{input_ids}\n")

    # Get embeddings from both layers
    custom_output = custom_embed(input_ids)
    pytorch_output = pytorch_embed(input_ids)

    print(f"Custom Output Shape: {custom_output.shape}")
    print(f"PyTorch Output Shape: {pytorch_output.shape}\n")

    # Verify the outputs are identical
    print(f"Outputs are identical: {torch.equal(custom_output, pytorch_output)}")

    # Demonstrate learnability (check weights require grad)
    print(f"Custom weights require grad: {custom_embed.weight.requires_grad}")
    print(f"PyTorch weights require grad: {pytorch_embed.weight.requires_grad}")

    # Example of index out of bounds error
    try:
        invalid_ids = torch.LongTensor([0, 10])  # 10 is out of bounds for vocab_size=10
        custom_embed(invalid_ids)
    except IndexError as e:
        print(f"\nCaught expected error: {e}")

    # Example of wrong dtype error
    try:
        invalid_dtype_ids = torch.Tensor([0, 1])  # FloatTensor instead of LongTensor
        custom_embed(invalid_dtype_ids)
    except TypeError as e:
        print(f"Caught expected error: {e}")
