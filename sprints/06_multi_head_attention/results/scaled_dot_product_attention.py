import torch
import torch.nn.functional as F
import math


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the Scaled Dot-Product Attention.

    Args:
        query (torch.Tensor): Query tensor; shape (..., seq_len_q, d_k).
        key (torch.Tensor): Key tensor; shape (..., seq_len_k, d_k).
        value (torch.Tensor): Value tensor; shape (..., seq_len_k, d_v).
        mask (torch.Tensor | None, optional): Mask tensor.
            If specified, it must be broadcastable to (..., seq_len_q, seq_len_k).
            Masked positions should be indicated by True (for positions to keep) or False (for positions to mask out).
            Defaults to None.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - output (torch.Tensor): The attention output; shape (..., seq_len_q, d_v).
            - attention_weights (torch.Tensor): The attention weights; shape (..., seq_len_q, seq_len_k).
    """
    # Ensure d_k is the last dimension for query and key
    d_k = query.size(-1)

    # 1. Calculate similarity scores: QK^T
    # We need to transpose the last two dimensions of the key tensor
    # (..., seq_len_k, d_k) -> (..., d_k, seq_len_k)
    # Resulting shape: (..., seq_len_q, seq_len_k)
    attention_scores = torch.matmul(query, key.transpose(-2, -1))

    # 2. Scale the scores
    # Scale by the square root of the dimension of the key vectors
    scale_factor = math.sqrt(d_k)
    scaled_attention_scores = attention_scores / scale_factor

    # 3. Apply the mask (if provided)
    # The mask should have False values for positions to be masked (set to -inf)
    if mask is not None:
        # Ensure mask is broadcastable
        # Add dimensions if necessary for broadcasting (e.g., batch or head dims)
        while mask.dim() < scaled_attention_scores.dim():
            mask = mask.unsqueeze(0)
        # In PyTorch's masked_fill, True values are filled.
        # We want to fill positions where mask is False.
        scaled_attention_scores = scaled_attention_scores.masked_fill(
            mask == 0, float("-inf")
        )

    # 4. Apply softmax to get attention weights
    # Apply softmax along the last dimension (seq_len_k)
    # Shape: (..., seq_len_q, seq_len_k)
    attention_weights = F.softmax(scaled_attention_scores, dim=-1)

    # Handle potential NaNs after softmax if all scores in a row were -inf
    # This can happen if a query attends to only masked keys.
    attention_weights = torch.nan_to_num(attention_weights)  # Replace NaN with 0

    # 5. Multiply weights by values to get the output
    # Shape: (..., seq_len_q, seq_len_k) @ (..., seq_len_k, d_v) -> (..., seq_len_q, d_v)
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


# Example Usage (to be added later)
if __name__ == "__main__":
    print("--- Testing Scaled Dot-Product Attention ---")

    # Define dimensions
    batch_size = 2
    seq_len_q = 4  # Query sequence length
    seq_len_k = 6  # Key/Value sequence length
    d_k = 8  # Dimension of keys/queries
    d_v = 10  # Dimension of values

    # Create random tensors
    query = torch.randn(batch_size, seq_len_q, d_k)
    key = torch.randn(batch_size, seq_len_k, d_k)
    value = torch.randn(batch_size, seq_len_k, d_v)

    print(f"\nInput shapes:")
    print(f"  Query: {query.shape}")
    print(f"  Key:   {key.shape}")
    print(f"  Value: {value.shape}")

    # Test Case 1: No mask
    print("\nTest Case 1: No Mask")
    output_no_mask, attn_weights_no_mask = scaled_dot_product_attention(
        query, key, value
    )
    print(f"  Output shape: {output_no_mask.shape}")
    print(f"  Attention weights shape: {attn_weights_no_mask.shape}")
    # Check if attention weights sum to 1 (approximately) along the key dimension
    print(
        f"  Attention weights sum check (first batch, first query): {attn_weights_no_mask[0, 0, :].sum():.4f}"
    )

    # Test Case 2: With a mask
    print("\nTest Case 2: With Mask")
    # Create a mask: mask out the last two key/value positions for all queries
    # Shape: (batch_size, seq_len_q, seq_len_k)
    # We want True for positions to *keep*
    mask = torch.ones(batch_size, seq_len_q, seq_len_k, dtype=torch.bool)
    mask[:, :, -2:] = False  # Mask out last two columns (keys)

    print(f"  Mask shape: {mask.shape}")
    print("  Example Mask (first batch, first query):")
    print(mask[0, 0])
    print("")  # Add a blank line for spacing

    output_with_mask, attn_weights_with_mask = scaled_dot_product_attention(
        query, key, value, mask=mask
    )
    print(f"  Output shape: {output_with_mask.shape}")
    print(f"  Attention weights shape: {attn_weights_with_mask.shape}")
    print(
        f"  Attention weights sum check (first batch, first query): {attn_weights_with_mask[0, 0, :].sum():.4f}"
    )

    # Check if masked positions have near-zero weights
    print("  Weights for masked positions (should be ~0):")
    print(f"    Last key weight: {attn_weights_with_mask[0, 0, -1]:.4f}")
    print(f"    Second-to-last key weight: {attn_weights_with_mask[0, 0, -2]:.4f}")

    # Verify that the output shapes match expectations
    expected_output_shape = (batch_size, seq_len_q, d_v)
    expected_attn_shape = (batch_size, seq_len_q, seq_len_k)

    assert (
        output_no_mask.shape == expected_output_shape
    ), f"Expected output shape {expected_output_shape}, got {output_no_mask.shape}"
    assert (
        attn_weights_no_mask.shape == expected_attn_shape
    ), f"Expected attention shape {expected_attn_shape}, got {attn_weights_no_mask.shape}"
    assert (
        output_with_mask.shape == expected_output_shape
    ), f"Expected output shape {expected_output_shape}, got {output_with_mask.shape}"
    assert (
        attn_weights_with_mask.shape == expected_attn_shape
    ), f"Expected attention shape {expected_attn_shape}, got {attn_weights_with_mask.shape}"

    print(
        "\nAll tests passed! Scaled Dot-Product Attention seems to be working correctly. 🎉"
    )