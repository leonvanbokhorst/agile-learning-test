import torch
import torch.nn as nn

# Import necessary components (assuming they are in the same directory)
from multi_head_attention import MultiHeadAttention
from ffn_example import PositionWiseFeedForward


class DecoderBlock(nn.Module):
    """Implements a single Transformer Decoder Block."""

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
            d_model: Dimension of the model.
            num_heads: Number of attention heads.
            d_ff: Inner dimension of the Feed-Forward Network.
            dropout_prob: Dropout probability.
            activation: Activation function for the FFN.
        """
        super().__init__()

        # --- Masked Self-Attention Sub-layer --- #
        self.masked_self_attention = MultiHeadAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.norm1 = nn.LayerNorm(d_model)

        # --- Cross-Attention (Encoder-Decoder) Sub-layer --- #
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.norm2 = nn.LayerNorm(d_model)

        # --- Feed-Forward Sub-layer --- #
        self.feed_forward = PositionWiseFeedForward(
            d_model, d_ff, dropout_prob, activation
        )
        self.dropout3 = nn.Dropout(dropout_prob)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        target: torch.Tensor,
        encoder_output: torch.Tensor,
        target_mask: torch.Tensor | None = None,  # Look-ahead mask for target sequence
        encoder_mask: torch.Tensor | None = None,  # Padding mask for encoder sequence
    ) -> torch.Tensor:
        """
        Forward pass through the Decoder Block.

        Args:
            target: Target sequence tensor (input to the decoder).
                    Shape: (batch_size, target_seq_len, d_model).
            encoder_output: Output tensor from the final encoder layer.
                            Shape: (batch_size, source_seq_len, d_model).
            target_mask: Boolean mask for the target sequence (masked self-attention).
                         Prevents attention to future tokens (look-ahead mask).
                         Shape: (batch_size, 1, target_seq_len, target_seq_len).
                         `True` indicates masked positions.
            encoder_mask: Boolean mask for the encoder output sequence (cross-attention).
                          Prevents attention to padding tokens in the source sequence.
                          Shape: (batch_size, 1, 1, source_seq_len).
                          `True` indicates masked positions.

        Returns:
            Output tensor of shape (batch_size, target_seq_len, d_model).
        """
        # 1. Masked Multi-Head Self-Attention + Add & Norm
        # target attends to itself (with look-ahead mask)
        masked_attn_out_tuple = self.masked_self_attention(
            query=target, key=target, value=target, mask=target_mask
        )
        masked_attn_out = masked_attn_out_tuple[0]  # Get tensor output
        # Residual connection
        norm_masked_attn_out = self.norm1(target + self.dropout1(masked_attn_out))

        # 2. Multi-Head Cross-Attention + Add & Norm
        # Query = output from previous step
        # Key/Value = encoder_output (with padding mask)
        cross_attn_out_tuple = self.cross_attention(
            query=norm_masked_attn_out,
            key=encoder_output,
            value=encoder_output,
            mask=encoder_mask,
        )
        cross_attn_out = cross_attn_out_tuple[0]  # Get tensor output
        # Residual connection
        norm_cross_attn_out = self.norm2(
            norm_masked_attn_out + self.dropout2(cross_attn_out)
        )

        # 3. Position-wise Feed-Forward + Add & Norm
        ffn_output = self.feed_forward(norm_cross_attn_out)
        # Residual connection
        decoder_output = self.norm3(norm_cross_attn_out + self.dropout3(ffn_output))

        return decoder_output


def main():
    """Demonstrates the DecoderBlock module."""
    batch_size = 4
    target_seq_len = 12
    source_seq_len = 10
    d_model = 64
    num_heads = 8
    d_ff = d_model * 4
    dropout_prob = 0.1

    # Inputs
    target_input = torch.randn(batch_size, target_seq_len, d_model)
    encoder_output = torch.randn(batch_size, source_seq_len, d_model)
    print(f"Target Input shape: {target_input.shape}")
    print(f"Encoder Output shape: {encoder_output.shape}")

    # Masks
    # Look-ahead mask for target sequence (upper triangle mask)
    look_ahead_mask = torch.triu(
        torch.ones(target_seq_len, target_seq_len), diagonal=1
    ).bool()
    target_mask = look_ahead_mask.unsqueeze(0).unsqueeze(
        0
    )  # Shape: (1, 1, target_seq_len, target_seq_len)
    print(f"Target Mask shape: {target_mask.shape}")
    # print(f"Target Mask example:\n{target_mask[0, 0]}") # Might be large

    # Padding mask for encoder sequence (e.g., last 2 source tokens padded)
    encoder_mask = torch.zeros(batch_size, 1, 1, source_seq_len, dtype=torch.bool)
    encoder_mask[:, :, :, -2:] = True
    print(f"Encoder Mask shape: {encoder_mask.shape}")
    print(f"Encoder Mask example:\n{encoder_mask[0]}")

    # Create the Decoder Block
    decoder_block = DecoderBlock(d_model, num_heads, d_ff, dropout_prob)
    print(f"\nDecoder Block: {decoder_block}")

    # Apply the Decoder Block
    output = decoder_block(target_input, encoder_output, target_mask, encoder_mask)
    print(f"\nOutput shape: {output.shape}")

    # Verification: Check that the output shape is correct
    assert output.shape == (
        batch_size,
        target_seq_len,
        d_model,
    ), f"Expected output shape {(batch_size, target_seq_len, d_model)}, but got {output.shape}"
    print("Output shape verified successfully.")

    # Check output stats for one position (should be roughly normalized)
    sample_idx = 0
    position_idx = 0
    output_features = output[sample_idx, position_idx, :]
    mean_after_block = output_features.mean()
    std_dev_after_block = output_features.std(unbiased=False)
    print(
        f"\n--- Output Stats Verification (sample {sample_idx}, position {position_idx}) ---"
    )
    print(f"Output Mean across features: {mean_after_block.item():.6f} (Should be ~0)")
    print(
        f"Output Std Dev across features: {std_dev_after_block.item():.6f} (Should be ~1)"
    )


if __name__ == "__main__":
    main()
