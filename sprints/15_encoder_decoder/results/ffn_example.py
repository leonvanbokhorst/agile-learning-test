import torch
import torch.nn as nn


class PositionWiseFeedForward(nn.Module):
    """Implements the Position-wise Feed-Forward Network (FFN) sub-layer."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout_prob: float = 0.1,
        activation: nn.Module = nn.GELU(),
    ):
        """
        Args:
            d_model: The input and output dimension of the model.
            d_ff: The inner dimension of the feed-forward network.
                  Typically d_ff = 4 * d_model.
            dropout_prob: The dropout probability.
            activation: The activation function module (e.g., nn.ReLU(), nn.GELU()).
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = activation
        # Dropout is often applied *after* the activation or after the second linear layer.
        # Applying after activation is common.
        self.dropout = nn.Dropout(dropout_prob)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes the input through the FFN.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        # x: (batch_size, seq_len, d_model)
        x = self.linear1(x)  # -> (batch_size, seq_len, d_ff)
        x = self.activation(x)
        x = self.dropout(x)  # Apply dropout
        x = self.linear2(x)  # -> (batch_size, seq_len, d_model)
        return x


def main():
    """Demonstrates the PositionWiseFeedForward module."""
    batch_size = 4
    seq_len = 10
    d_model = 64
    d_ff = d_model * 4  # Standard expansion factor
    dropout_prob = 0.1

    # Input tensor
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")

    # Create the FFN module
    ffn = PositionWiseFeedForward(d_model, d_ff, dropout_prob)
    print(f"\nFFN Module: {ffn}")

    # Apply the FFN
    output = ffn(x)
    print(f"\nOutput shape: {output.shape}")

    # Verification: Check that the output shape is correct
    assert output.shape == (
        batch_size,
        seq_len,
        d_model,
    ), f"Expected output shape {(batch_size, seq_len, d_model)}, but got {output.shape}"
    print("Output shape verified successfully.")

    # --- Example with ReLU activation ---
    ffn_relu = PositionWiseFeedForward(
        d_model, d_ff, dropout_prob, activation=nn.ReLU()
    )
    print(f"\nFFN Module (ReLU): {ffn_relu}")
    output_relu = ffn_relu(x)
    assert output_relu.shape == (batch_size, seq_len, d_model)
    print("ReLU FFN output shape verified successfully.")


if __name__ == "__main__":
    main()
