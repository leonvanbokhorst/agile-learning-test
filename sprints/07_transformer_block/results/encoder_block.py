import torch
import torch.nn as nn

# Assume MultiHeadAttention is defined in the specified path relative to the project root
# We need to import it correctly based on the project structure.
# If running this script directly, relative imports might fail unless using `python -m ...`
# --- Updated Import --- #
# Import from the current directory since files were copied
from multi_head_attention import MultiHeadAttention
from ffn_example import PositionWiseFeedForward

# --- End Updated Import --- #

# Commented out old imports
# try:
#     # Try relative import first (works when run as part of a package)
#     from ....sprints.sprint_06_multi_head_attention.results.multi_head_attention import (
#         MultiHeadAttention,
#     )
# except ImportError:
#     # Fallback for potentially running the script directly or different structure
#     # This might require adjusting based on actual execution context or adding
#     # the project root to PYTHONPATH.
#     # A more robust solution involves proper packaging or consistent execution.
#     print(
#         "Warning: Relative import failed. Attempting fallback import for MultiHeadAttention."
#         " Ensure the path is correct or run using `python -m ...`"
#     )
#     # This assumes a certain structure or that the path is added elsewhere
#     # Adjust the path as necessary for your specific setup.
#     from sprints.sprint_06_multi_head_attention.results.multi_head_attention import (
#         MultiHeadAttention,
#     )
#
#
# from sprints.sprint_07_transformer_block.results.ffn_example import (
#     PositionWiseFeedForward,
# )


class EncoderBlock(nn.Module):
    """Implements a single Transformer Encoder Block."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_prob: float = 0.1,
        # attention_dropout_prob: float = 0.1, # Removed - not used directly here
        activation: nn.Module = nn.GELU(),
    ):
        """
        Args:
            d_model: Dimension of the model (embeddings and outputs).
            num_heads: Number of attention heads.
            d_ff: Inner dimension of the Feed-Forward Network.
            dropout_prob: Dropout probability for FFN output and Add&Norm steps.
            # attention_dropout_prob: (Removed) Dropout for attention handled within MHA or after.
            activation: Activation function for the FFN.
        """
        super().__init__()

        # --- Self-Attention Sub-layer --- #
        # Initialize MultiHeadAttention without dropout_prob arg
        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            # dropout_prob=attention_dropout_prob # Removed this line
        )
        # Dropout after attention application, before residual connection
        self.dropout1 = nn.Dropout(dropout_prob)  # This dropout remains
        self.norm1 = nn.LayerNorm(d_model)

        # --- Feed-Forward Sub-layer --- #
        self.feed_forward = PositionWiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout_prob=dropout_prob,  # Pass general dropout to FFN internal dropout
            activation=activation,
        )
        # Dropout after FFN application, before residual connection
        self.dropout2 = nn.Dropout(dropout_prob)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass through the Encoder Block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            mask: Optional boolean mask for self-attention. Typically a padding mask.
                  Shape should be broadcastable to (batch_size, num_heads, seq_len, seq_len).
                  Usually (batch_size, 1, 1, seq_len) for padding masks.
                  `False` indicates positions to attend to, `True` indicates masked positions.

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        # 1. Multi-Head Self-Attention + Add & Norm
        # Apply self-attention (Query=Key=Value=x)
        # Assuming self_attention returns (output, attention_weights)
        attn_output_tuple = self.self_attention(query=x, key=x, value=x, mask=mask)
        attn_output = attn_output_tuple[0]  # Take only the attention output tensor

        # Residual connection: Add input x to attention output (after dropout)
        x = self.norm1(x + self.dropout1(attn_output))

        # 2. Position-wise Feed-Forward + Add & Norm
        # Apply feed-forward network
        ffn_output = self.feed_forward(x)
        # Residual connection: Add input (from previous step) to FFN output (after dropout)
        x = self.norm2(x + self.dropout2(ffn_output))

        return x


def main():
    """Demonstrates the EncoderBlock module."""
    batch_size = 4
    seq_len = 10
    d_model = 64
    num_heads = 8
    d_ff = d_model * 4
    dropout_prob = 0.1
    num_blocks = 2  # Number of blocks to stack

    # Input tensor
    x = torch.randn(batch_size, seq_len, d_model)
    # Dummy padding mask (masking the last 2 positions for the first 2 samples)
    # Shape: (batch_size, 1, 1, seq_len) -> broadcastable for attention
    mask = torch.zeros(batch_size, 1, 1, seq_len, dtype=torch.bool)
    mask[0:2, :, :, -2:] = True  # Mask last 2 tokens for first 2 sequences
    print(f"Input shape: {x.shape}")
    print(f"Mask shape: {mask.shape}")
    # print(f"Mask example:\n{mask[0]}") # Can be large

    # --- Single Block Test --- #
    print("\n--- Testing Single Encoder Block ---")
    encoder_block_single = EncoderBlock(d_model, num_heads, d_ff, dropout_prob)
    print(f"Encoder Block: {encoder_block_single}")
    output_single = encoder_block_single(x, mask=mask)
    print(f"Single Block Output shape: {output_single.shape}")
    assert output_single.shape == (
        batch_size,
        seq_len,
        d_model,
    ), f"Expected output shape {(batch_size, seq_len, d_model)}, but got {output_single.shape}"
    print("Single Block Output shape verified successfully.")

    # --- Stacked Blocks Test --- #
    print(f"\n--- Testing Stacked Encoder Blocks (N={num_blocks}) ---")
    # Create a list or ModuleList of encoder blocks
    encoder_blocks = nn.ModuleList(
        [
            EncoderBlock(d_model, num_heads, d_ff, dropout_prob)
            for _ in range(num_blocks)
        ]
    )
    print(f"Stacked Encoder Blocks: {encoder_blocks}")

    # Pass input through the stack
    stacked_output = x  # Start with the initial input
    for i, block in enumerate(encoder_blocks):
        print(f"Passing through block {i+1}...")
        stacked_output = block(stacked_output, mask=mask)
        print(f"  Output shape after block {i+1}: {stacked_output.shape}")

    print(f"\nFinal Stacked Output shape: {stacked_output.shape}")

    # Verification: Check that the final output shape is correct
    assert stacked_output.shape == (
        batch_size,
        seq_len,
        d_model,
    ), f"Expected final output shape {(batch_size, seq_len, d_model)}, but got {stacked_output.shape}"
    print("Stacked Blocks Output shape verified successfully.")

    # Check output stats for one position (should be roughly normalized due to final LayerNorm)
    sample_idx = 0
    position_idx = 0
    final_output_features = stacked_output[sample_idx, position_idx, :]
    mean_after_stack = final_output_features.mean()
    std_dev_after_stack = final_output_features.std(unbiased=False)
    print(
        f"\n--- Final Output Stats Verification (sample {sample_idx}, position {position_idx}) ---"
    )
    print(f"Output Mean across features: {mean_after_stack.item():.6f} (Should be ~0)")
    print(
        f"Output Std Dev across features: {std_dev_after_stack.item():.6f} (Should be ~1)"
    )


if __name__ == "__main__":
    main()
