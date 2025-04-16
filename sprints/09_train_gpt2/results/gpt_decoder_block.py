import torch
import torch.nn as nn

# Assuming these are in the same directory or accessible via path
from multi_head_attention import MultiHeadAttention
from ffn_example import (
    PositionWiseFeedForward,
)  # Renamed from original example maybe? Check filename.


class GPTDecoderBlock(nn.Module):
    """Implements a single GPT-style (decoder-only) Transformer Block.

    This block consists of:
    1. Masked Multi-Head Self-Attention
    2. Add & Norm
    3. Position-wise Feed-Forward Network
    4. Add & Norm
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_prob: float = 0.1,
        activation: nn.Module = nn.GELU(),
    ):
        """
        Args:
            d_model: Dimension of the model (embedding dimension).
            num_heads: Number of attention heads.
            d_ff: Inner dimension of the Feed-Forward Network. Typically 4 * d_model.
            dropout_prob: Dropout probability applied to attention and FFN outputs.
            activation: Activation function for the FFN (e.g., nn.ReLU(), nn.GELU()).
        """
        super().__init__()

        # --- Masked Self-Attention Sub-layer --- #
        # Performs attention on the input sequence, masked to prevent attending to future positions.
        self.masked_self_attention = MultiHeadAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.norm1 = nn.LayerNorm(d_model)  # Layer normalization after attention

        # --- Feed-Forward Sub-layer --- #
        # Processes each position independently after attention.
        self.feed_forward = PositionWiseFeedForward(
            d_model, d_ff, dropout_prob, activation
        )
        self.dropout2 = nn.Dropout(dropout_prob)  # Renamed from dropout3
        self.norm2 = nn.LayerNorm(
            d_model
        )  # Renamed from norm3 - Layer normalization after FFN

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,  # Combined mask (look-ahead + padding)
    ) -> torch.Tensor:
        """
        Forward pass through the GPT Decoder Block.

        Args:
            x: Input sequence tensor.
               Shape: (batch_size, seq_len, d_model).
            mask: Boolean attention mask. Combines look-ahead and padding masks.
                  Shape: (batch_size, 1, seq_len, seq_len) or similar broadcastable shape.
                  `True` indicates positions that should *not* be attended to.

        Returns:
            Output tensor of the same shape as input: (batch_size, seq_len, d_model).
        """
        # 1. Masked Multi-Head Self-Attention + Add & Norm
        # The sequence attends to itself, respecting the mask.
        attn_output, _ = self.masked_self_attention(query=x, key=x, value=x, mask=mask)
        # Residual connection followed by layer normalization
        x = self.norm1(x + self.dropout1(attn_output))

        # 2. Position-wise Feed-Forward + Add & Norm
        # Apply the feed-forward network to the output of the attention layer.
        ffn_output = self.feed_forward(x)
        # Residual connection followed by layer normalization
        x = self.norm2(x + self.dropout2(ffn_output))

        return x


# Optional: Add a main block for testing this specific module
if __name__ == "__main__":
    batch_size = 4
    seq_len = 10
    d_model = 64
    num_heads = 8
    d_ff = d_model * 4
    dropout_prob = 0.1

    # Input tensor
    input_tensor = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {input_tensor.shape}")

    # Mask (combined look-ahead and padding - simplified for example)
    # Create a look-ahead mask
    look_ahead_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    # Assume no padding for simplicity here, expand mask dims for MHA
    attn_mask = look_ahead_mask.unsqueeze(0).unsqueeze(
        0
    )  # Shape: (1, 1, seq_len, seq_len)
    # Replicate mask for batch size if needed by MHA implementation (check yours)
    # attn_mask = attn_mask.expand(batch_size, -1, -1, -1)
    print(f"Attention mask shape: {attn_mask.shape}")

    # Create the GPTDecoderBlock instance
    gpt_block = GPTDecoderBlock(d_model, num_heads, d_ff, dropout_prob)
    print(f"\nGPT Decoder Block:\n{gpt_block}")

    # Perform forward pass
    output_tensor = gpt_block(input_tensor, mask=attn_mask)
    print(f"\nOutput shape: {output_tensor.shape}")

    # Verification
    assert (
        output_tensor.shape == input_tensor.shape
    ), f"Output shape {output_tensor.shape} does not match input shape {input_tensor.shape}"
    print("Output shape verified successfully.")

    # Check output stats (should be roughly normalized)
    sample_output = output_tensor[0, 0, :]  # First batch, first position
    mean_out = sample_output.mean().item()
    std_out = sample_output.std().item()
    print(f"\nSample output stats (position 0): Mean={mean_out:.4f}, Std={std_out:.4f}")
    print("(Expect Mean ~0, Std ~1 due to final LayerNorm)")
