import torch
import torch.nn as nn
from typing import Union, Sequence


def main():
    """Demonstrates the usage of torch.nn.LayerNorm."""
    # Define typical Transformer input dimensions
    batch_size: int = 4
    seq_len: int = 10
    d_model: int = 64  # Embedding dimension / Feature size

    # Create a random input tensor simulating batch_size sequences of seq_len tokens,
    # each represented by a d_model dimensional vector.
    # Shape: (batch_size, seq_len, d_model)
    input_tensor = torch.randn(batch_size, seq_len, d_model)

    print(f"Input Tensor Shape: {input_tensor.shape}")

    # --- Layer Normalization ---

    # Initialize LayerNorm.
    # `normalized_shape` specifies the dimension(s) along which the mean and std dev
    # will be computed. For Transformers, we typically normalize across the feature
    # dimension (d_model), which is the last dimension in our (batch, seq, feature) setup.
    # LayerNorm has learnable parameters `weight` (gamma) and `bias` (beta) by default.
    layer_norm = nn.LayerNorm(
        normalized_shape=d_model
    )  # Equivalent to nn.LayerNorm((d_model,))

    # Print the LayerNorm module to see its configuration
    print(f"\nInitialized LayerNorm: {layer_norm}")
    print(
        f"Learnable gamma (weight) shape: {layer_norm.weight.shape}"
    )  # Shape: (d_model,)
    print(f"Learnable beta (bias) shape: {layer_norm.bias.shape}")  # Shape: (d_model,)

    # Apply Layer Normalization to the input tensor
    output_tensor = layer_norm(input_tensor)

    print(f"\nOutput Tensor Shape: {output_tensor.shape}")

    # --- Verification ---
    # Let's verify the normalization for one specific sample and position in the sequence.
    # We expect the features (along the d_model dimension) for this specific token
    # to have a mean close to 0 and a standard deviation close to 1 *after* normalization
    # *before* the learnable gamma/beta are applied. Since gamma=1 and beta=0 initially,
    # the output should reflect this normalization.

    sample_idx = 0
    position_idx = 0
    normalized_features = output_tensor[sample_idx, position_idx, :]

    mean_after_norm = normalized_features.mean()
    std_dev_after_norm = normalized_features.std(
        unbiased=False
    )  # Use population std dev

    print(f"\n--- Verification for sample {sample_idx}, position {position_idx} ---")
    print(f"Mean across features: {mean_after_norm.item():.6f} (Expected: ~0)")
    print(f"Std Dev across features: {std_dev_after_norm.item():.6f} (Expected: ~1)")

    # Check a different sample/position to be sure
    sample_idx = 2
    position_idx = 5
    normalized_features_other = output_tensor[sample_idx, position_idx, :]
    mean_other = normalized_features_other.mean()
    std_dev_other = normalized_features_other.std(unbiased=False)
    print(f"\n--- Verification for sample {sample_idx}, position {position_idx} ---")
    print(f"Mean across features: {mean_other.item():.6f} (Expected: ~0)")
    print(f"Std Dev across features: {std_dev_other.item():.6f} (Expected: ~1)")

    # --- Note on elementwise_affine --- #
    # If elementwise_affine=False, gamma and beta are not created/learned,
    # and the output is purely the normalized values.
    layer_norm_no_affine = nn.LayerNorm(
        normalized_shape=d_model, elementwise_affine=False
    )
    output_no_affine = layer_norm_no_affine(input_tensor)
    print(f"\n--- No Affine Transformation --- ")
    print(f"LayerNorm without affine parameters: {layer_norm_no_affine}")
    # print(layer_norm_no_affine.weight) # This would raise an AttributeError


if __name__ == "__main__":
    main()
