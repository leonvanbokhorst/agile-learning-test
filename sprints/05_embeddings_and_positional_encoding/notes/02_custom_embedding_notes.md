# Notes: Custom Embedding Layer

**Corresponding Results:** [`results/02_custom_embedding.py`](../results/02_custom_embedding.py)

## Motivation

Implementing a simplified version of `nn.Embedding` from scratch helps solidify understanding by revealing the core components:

1.  **A Learnable Weight Matrix:** This is the actual table storing the embedding vectors.
2.  **An Indexing Operation:** The `forward` method simply looks up rows in this matrix based on input indices.

## Implementation (`CustomEmbedding` class)

```python
import torch
import torch.nn as nn

class CustomEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Core Part 1: The learnable weight matrix
        # nn.Parameter registers this tensor with PyTorch for gradient tracking.
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, input_indices: torch.Tensor) -> torch.Tensor:
        # Input validation (good practice)
        if not torch.is_tensor(input_indices) or input_indices.dtype != torch.long:
            raise TypeError("Input indices must be a LongTensor.")
        if input_indices.max() >= self.num_embeddings or input_indices.min() < 0:
            raise IndexError("Index out of range")

        # Core Part 2: The indexing operation
        # This efficiently selects the rows corresponding to the indices.
        embedded_vectors = self.weight[input_indices]
        return embedded_vectors
```

## Key Takeaways

- **`nn.Parameter`:** This wrapper is essential. It tells PyTorch that the `self.weight` tensor is a parameter of the `nn.Module` and should have gradients computed and be updated by the optimizer during training.
- **Tensor Indexing:** The `forward` method leverages PyTorch's powerful tensor indexing. Providing a tensor of indices (`input_indices`) to select from `self.weight` automatically handles batching and returns the corresponding rows (embedding vectors) in the correct output shape.
- **Simplicity:** Under the hood, the basic embedding lookup is just this matrix indexing operation. The built-in `nn.Embedding` adds more features (like padding index, initialization options, norm constraints), but the core principle is the same.
- **Learnability:** Because `self.weight` is an `nn.Parameter`, the custom layer is learnable, just like the built-in version.

This exercise demonstrates that `nn.Embedding` isn't magic; it's a well-defined module encapsulating a learnable lookup table accessed via tensor indexing.
