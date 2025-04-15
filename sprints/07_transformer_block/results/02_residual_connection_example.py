import torch
import torch.nn as nn


class DummySublayer(nn.Module):
    """A placeholder for a Transformer sub-layer like Multi-Head Attention or FFN."""

    def __init__(self, d_model: int):
        super().__init__()
        # Simulate some computation, e.g., a simple linear transformation
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the transformation
        return self.linear(x)


class AddNormWrapper(nn.Module):
    """Implements the Add & Norm pattern around a given sub-layer."""

    def __init__(self, normalized_shape: int, dropout_prob: float, sublayer: nn.Module):
        """
        Args:
            normalized_shape: The shape/dimension(s) to normalize over (typically d_model).
            dropout_prob: The dropout probability.
            sublayer: The sub-layer module (e.g., MultiHeadAttention, FeedForward) to wrap.
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape)
        self.dropout = nn.Dropout(dropout_prob)
        self.sublayer = sublayer

    def forward(
        self, x: torch.Tensor, *sublayer_args, **sublayer_kwargs
    ) -> torch.Tensor:
        """
        Applies the Add & Norm pattern.

        Args:
            x: The input tensor to the wrapper (and the residual connection).
            *sublayer_args: Additional positional arguments for the sublayer's forward method.
            **sublayer_kwargs: Additional keyword arguments for the sublayer's forward method.

        Returns:
            The output tensor after applying Sublayer -> Dropout -> Add -> Norm.
        """
        # 1. Apply the sublayer (e.g., Attention, FFN)
        # Pass any additional arguments needed by the specific sublayer
        sublayer_output = self.sublayer(x, *sublayer_args, **sublayer_kwargs)

        # 2. Apply dropout to the output of the sublayer
        dropped_output = self.dropout(sublayer_output)

        # 3. Add the original input (residual connection)
        added_output = x + dropped_output

        return self.layer_norm(added_output)


def main():
    """Demonstrates the Add & Norm wrapper."""
    batch_size = 4
    seq_len = 10
    d_model = 64
    dropout_prob = 0.1  # Typical dropout rate

    # Input tensor
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")

    # Create a dummy sub-layer
    dummy_sublayer = DummySublayer(d_model)

    # Wrap the dummy sub-layer with Add & Norm
    add_norm_wrapper = AddNormWrapper(
        normalized_shape=d_model, dropout_prob=dropout_prob, sublayer=dummy_sublayer
    )
    print(f"\nAdd & Norm Wrapper: {add_norm_wrapper}")

    # Apply the wrapper
    # Note: If the sublayer needed more args (like masks for attention),
    # they would be passed here: output = add_norm_wrapper(x, mask=some_mask)
    output = add_norm_wrapper(x)

    print(f"\nOutput shape: {output.shape}")

    # --- Verification ---
    # Check that shapes are maintained and output stats look reasonable (normalized)
    sample_idx = 0
    position_idx = 0
    output_features = output[sample_idx, position_idx, :]
    mean_after_norm = output_features.mean()
    std_dev_after_norm = output_features.std(unbiased=False)

    print(f"\n--- Verification for sample {sample_idx}, position {position_idx} ---")
    print(f"Output Mean across features: {mean_after_norm.item():.6f} (Should be ~0)")
    print(
        f"Output Std Dev across features: {std_dev_after_norm.item():.6f} (Should be ~1)"
    )


if __name__ == "__main__":
    main()
