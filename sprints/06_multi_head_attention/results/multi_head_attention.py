import torch
import torch.nn as nn
import math

# Assuming scaled_dot_product_attention.py is in the same directory or accessible
# If running as a script, relative import works. If used as part of a larger package,
# adjust the import accordingly.

from scaled_dot_product_attention import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    """Implements the Multi-Head Attention mechanism.

    Args:
        d_model (int): The dimensionality of the input and output features.
        num_heads (int): The number of attention heads.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension of keys/queries per head

        # Linear layers for Q, K, V projections (input projections)
        self.W_q = nn.Linear(
            d_model, d_model, bias=False
        )  # Bias often omitted in transformers
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Final linear layer for output projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Splits the last dimension into (num_heads, d_k).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: Tensor of shape (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.size()
        # Reshape to (batch_size, seq_len, num_heads, d_k)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        # Transpose to (batch_size, num_heads, seq_len, d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Combines heads back from (batch_size, num_heads, seq_len, d_k) to (batch_size, seq_len, d_model)."""
        batch_size, _, seq_len, _ = x.size()
        # Transpose back to (batch_size, seq_len, num_heads, d_k)
        x = x.transpose(1, 2).contiguous()  # Ensure contiguous memory for view
        # Reshape to (batch_size, seq_len, d_model)
        return x.view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for Multi-Head Attention.

        Args:
            query (torch.Tensor): Query tensor; shape (batch_size, seq_len_q, d_model).
            key (torch.Tensor): Key tensor; shape (batch_size, seq_len_k, d_model).
            value (torch.Tensor): Value tensor; shape (batch_size, seq_len_k, d_model).
            mask (torch.Tensor | None, optional): Mask tensor.
                If specified, it should indicate positions to mask out (False) or keep (True).
                It needs appropriate broadcasting dimensions for attention, often
                (batch_size, 1, seq_len_q, seq_len_k) or (batch_size, 1, 1, seq_len_k).
                Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - output (torch.Tensor): The final attention output; shape (batch_size, seq_len_q, d_model).
                - attention_weights (torch.Tensor): The attention weights from the scaled dot-product attention;
                                                    shape (batch_size, num_heads, seq_len_q, seq_len_k).
        """
        batch_size = query.size(0)

        # 1. Apply linear projections
        q_proj = self.W_q(query)
        k_proj = self.W_k(key)
        v_proj = self.W_v(value)

        # 2. Split into multiple heads
        # q_split shape: (batch_size, num_heads, seq_len_q, d_k)
        # k_split shape: (batch_size, num_heads, seq_len_k, d_k)
        # v_split shape: (batch_size, num_heads, seq_len_k, d_k) - Note d_k here, assuming d_v = d_k
        q_split = self.split_heads(q_proj)
        k_split = self.split_heads(k_proj)
        v_split = self.split_heads(v_proj)

        # 3. Apply scaled dot-product attention for all heads in parallel
        # context shape: (batch_size, num_heads, seq_len_q, d_k)
        # attention_weights shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        context, attention_weights = scaled_dot_product_attention(
            q_split, k_split, v_split, mask=mask
        )

        # 4. Combine heads
        # context shape: (batch_size, seq_len_q, d_model)
        context = self.combine_heads(context)

        # 5. Apply final linear projection
        output = self.W_o(context)

        return output, attention_weights


# Example Usage (to be added later)
if __name__ == "__main__":
    print("--- Testing MultiHeadAttention Module ---")

    # Define dimensions
    batch_size = 2
    seq_len_q = 5  # Query sequence length
    seq_len_k = 7  # Key/Value sequence length (can be different)
    d_model = 128  # Model dimension (must be divisible by num_heads)
    num_heads = 8  # Number of attention heads

    # Create random tensors (representing embeddings + positional encoding)
    # Note: Q, K, V usually come from the same sequence in self-attention,
    # but can come from different sequences in encoder-decoder attention.
    # Here, we use different lengths for generality.
    query = torch.randn(batch_size, seq_len_q, d_model)
    key = torch.randn(batch_size, seq_len_k, d_model)
    value = torch.randn(batch_size, seq_len_k, d_model)  # Often same as key

    print(f"\nInput shapes:")
    print(f"  Query: {query.shape}")
    print(f"  Key:   {key.shape}")
    print(f"  Value: {value.shape}")
    print(f"  d_model: {d_model}, num_heads: {num_heads}")

    # Instantiate the module
    multi_head_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    print("\nMultiHeadAttention module instantiated.")

    # --- Test Case 1: No Mask ---
    print("\nTest Case 1: No Mask")
    output_no_mask, attn_weights_no_mask = multi_head_attn(query, key, value)

    print(f"  Output shape: {output_no_mask.shape}")
    print(f"  Attention weights shape: {attn_weights_no_mask.shape}")

    # --- Test Case 2: With Padding Mask ---
    print("\nTest Case 2: With Padding Mask")
    # Create a simple padding mask for the key/value sequence.
    # Example: first batch item has 5 valid keys, second has 6 valid keys.
    key_padding_mask = torch.tensor(
        [[True] * 5 + [False] * 2, [True] * 6 + [False] * 1], dtype=torch.bool
    )
    # Shape: (batch_size, seq_len_k)

    # Attention mask needs to be broadcastable to (batch_size, num_heads, seq_len_q, seq_len_k)
    # We need to add dimensions for heads and query sequence length.
    # Shape becomes: (batch_size, 1, 1, seq_len_k)
    attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)

    print(f"  Key Padding Mask shape: {key_padding_mask.shape}")
    print(f"  Attention Mask shape for calculation: {attn_mask.shape}")
    print(f"  Example Key Padding Mask (Batch 0): {key_padding_mask[0]}")

    output_with_mask, attn_weights_with_mask = multi_head_attn(
        query, key, value, mask=attn_mask
    )

    print(f"  Output shape: {output_with_mask.shape}")
    print(f"  Attention weights shape: {attn_weights_with_mask.shape}")

    # Verify output shapes match expectations
    expected_output_shape = (batch_size, seq_len_q, d_model)
    expected_attn_shape = (batch_size, num_heads, seq_len_q, seq_len_k)

    assert (
        output_no_mask.shape == expected_output_shape
    ), f"No mask: Expected output shape {expected_output_shape}, got {output_no_mask.shape}"
    assert (
        attn_weights_no_mask.shape == expected_attn_shape
    ), f"No mask: Expected attention shape {expected_attn_shape}, got {attn_weights_no_mask.shape}"
    assert (
        output_with_mask.shape == expected_output_shape
    ), f"With mask: Expected output shape {expected_output_shape}, got {output_with_mask.shape}"
    assert (
        attn_weights_with_mask.shape == expected_attn_shape
    ), f"With mask: Expected attention shape {expected_attn_shape}, got {attn_weights_with_mask.shape}"

    # Optional: Check if masked weights are zero (can be tricky due to broadcasting)
    # Example check on the first head, first batch item, first query:
    weights_b0_h0_q0 = attn_weights_with_mask[0, 0, 0, :]
    masked_indices_b0 = ~key_padding_mask[0]  # Indices where mask is False
    # print(f"  Weights for masked positions (Batch 0, Head 0, Query 0): {weights_b0_h0_q0[masked_indices_b0]}")
    # assert torch.allclose(weights_b0_h0_q0[masked_indices_b0], torch.tensor(0.0)), "Masked weights are not zero!"

    print(
        "\nAll tests passed! MultiHeadAttention module seems to be working correctly. 🎉"
    )

    # --- Test Case 3: With Look-Ahead Mask (Self-Attention Scenario) ---
    print("\nTest Case 3: With Look-Ahead Mask")
    # For look-ahead, query, key, and value sequences are usually the same length
    seq_len = 6
    query_self = torch.randn(batch_size, seq_len, d_model)
    # In self-attention, Q, K, V typically come from the same input
    key_self = query_self
    value_self = query_self

    print(f"  Self-Attention Input shape: {query_self.shape}")

    # Create the look-ahead mask
    # Shape: (seq_len, seq_len)
    look_ahead_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

    # Broadcastable shape for attention: (batch_size, num_heads, seq_len_q, seq_len_k)
    # We need to add batch and head dimensions.
    # Shape: (1, 1, seq_len, seq_len)
    attn_mask_look_ahead = look_ahead_mask.unsqueeze(0).unsqueeze(1)

    print(f"  Look-Ahead Mask shape (base): {look_ahead_mask.shape}")
    print(f"  Look-Ahead Mask shape (broadcastable): {attn_mask_look_ahead.shape}")
    print("  Example Look-Ahead Mask (Base):")
    print(look_ahead_mask)

    output_look_ahead, attn_weights_look_ahead = multi_head_attn(
        query_self, key_self, value_self, mask=attn_mask_look_ahead
    )

    print(f"  Output shape: {output_look_ahead.shape}")
    print(f"  Attention weights shape: {attn_weights_look_ahead.shape}")

    # Verify output shapes match expectations
    expected_output_shape_self = (batch_size, seq_len, d_model)
    expected_attn_shape_self = (batch_size, num_heads, seq_len, seq_len)

    assert (
        output_look_ahead.shape == expected_output_shape_self
    ), f"Look-ahead: Expected output shape {expected_output_shape_self}, got {output_look_ahead.shape}"
    assert (
        attn_weights_look_ahead.shape == expected_attn_shape_self
    ), f"Look-ahead: Expected attention shape {expected_attn_shape_self}, got {attn_weights_look_ahead.shape}"

    # Check if weights above the diagonal are zero for the first head/batch item
    weights_b0_h0 = attn_weights_look_ahead[0, 0, :, :]
    upper_tri_mask = ~look_ahead_mask  # False on and below diagonal, True above
    print(
        f"  Weights for masked future positions (Batch 0, Head 0, sum): {weights_b0_h0[upper_tri_mask].sum():.4f}"
    )
    assert torch.allclose(
        weights_b0_h0[upper_tri_mask], torch.tensor(0.0)
    ), "Look-ahead mask failed: Future positions have non-zero weights!"

    print(
        "\nAll tests passed! MultiHeadAttention module seems to be working correctly. 🎉"
    )
