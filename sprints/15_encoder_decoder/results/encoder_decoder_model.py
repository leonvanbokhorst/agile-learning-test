import torch
import torch.nn as nn
from typing import Optional
import math  # Added for PositionalEncoding

# Assuming components are available in the same directory or PYTHONPATH is set
from encoder_block import EncoderBlock
from decoder_block import DecoderBlock  # Import DecoderBlock

# We'll need positional encoding from Sprint 5 - let's assume it's copied or importable
# from positional_encoding import PositionalEncoding # Placeholder import


# --- Replace Placeholder PE with Sinusoidal PE from Sprint 5 --- #
class PositionalEncoding(nn.Module):
    """
    Injects positional information into the input embeddings.

    Uses sinusoidal functions of different frequencies, as described in
    "Attention Is All You Need". The positional encodings have the same
    dimension as the embeddings so that the two can be summed.

    Args:
        d_model (int): The dimension of the embeddings (required).
        dropout_prob (float): Dropout probability (default=0.1).
        max_len (int): The maximum sequence length (default=5000).
    """

    def __init__(self, d_model: int, dropout_prob: float = 0.1, max_len: int = 5000):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be a positive integer.")
        if not 0.0 <= dropout_prob <= 1.0:
            raise ValueError("dropout_prob must be between 0 and 1.")
        if max_len <= 0:
            raise ValueError("max_len must be a positive integer.")

        self.dropout = nn.Dropout(p=dropout_prob)

        position = torch.arange(max_len).unsqueeze(1)  # Shape: [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # Shape: [d_model / 2]
        pe = torch.zeros(max_len, d_model)  # Shape: [max_len, d_model]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register 'pe' as a buffer.
        self.register_buffer("pe", pe)  # Shape: [max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with embeddings.
                               Assumes batch_first=True: [batch_size, seq_len, d_model].
                               We will transpose it for PE calculation.

        Returns:
            torch.Tensor: Tensor with positional information added.
                          Shape: [batch_size, seq_len, d_model].
        """
        # Transpose for PE: [batch_size, seq_len, d_model] -> [seq_len, batch_size, d_model]
        x = x.transpose(0, 1)

        # Select the required PE slice and add
        # self.pe shape: [max_len, d_model]
        # Slice shape: [seq_len, d_model]
        # Unsqueeze for broadcasting: [seq_len, 1, d_model]
        pos_encoding_slice = self.pe[: x.size(0), :].unsqueeze(1)

        # Add positional encoding to the input tensor x.
        x = x.float() + pos_encoding_slice

        # Transpose back: [seq_len, batch_size, d_model] -> [batch_size, seq_len, d_model]
        x = x.transpose(0, 1)

        # Apply dropout
        return self.dropout(x)


# --- End Sinusoidal PE --- #


class Encoder(nn.Module):
    """Stacks multiple EncoderBlocks."""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        input_vocab_size: int,
        max_seq_len: int,
        dropout_prob: float = 0.1,
        activation: nn.Module = nn.GELU(),
    ):
        """
        Args:
            num_layers: Number of EncoderBlocks to stack.
            d_model: Dimension of embeddings and hidden states.
            num_heads: Number of attention heads in each EncoderBlock.
            d_ff: Inner dimension of the Feed-Forward Networks.
            input_vocab_size: Size of the source vocabulary.
            max_seq_len: Maximum sequence length (for positional encoding).
            dropout_prob: Dropout probability.
            activation: Activation function for FFNs.
        """
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(input_vocab_size, d_model)
        # Use Sinusoidal PE
        self.pos_encoding = PositionalEncoding(d_model, dropout_prob, max_seq_len)
        # We apply dropout within PE now, so remove the separate one after PE
        # self.dropout = nn.Dropout(dropout_prob)
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout_prob=dropout_prob,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )
        # Optional final LayerNorm
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for the entire Encoder stack.

        Args:
            x: Input tensor of token IDs. Shape: (batch_size, seq_len).
            mask: Padding mask for the input sequence.
                  Shape: (batch_size, 1, 1, seq_len). `True` indicates masked positions.

        Returns:
            Output tensor from the final EncoderBlock. Shape: (batch_size, seq_len, d_model).
        """
        batch_size, seq_len = x.shape

        # 1. Embeddings and Positional Encoding
        token_embeddings = self.embedding(x)  # (batch_size, seq_len, d_model)
        # Scale embeddings (common practice)
        token_embeddings *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        # Add positional encodings (dropout is handled inside PE)
        out = self.pos_encoding(token_embeddings)
        # Removed separate dropout: out = self.dropout(embeddings_with_pe)

        # 2. Pass through Encoder Blocks
        for layer in self.layers:
            out = layer(out, mask=mask)

        # 3. Optional Final LayerNorm
        if self.final_norm is not None:
            out = self.final_norm(out)

        return out


class Decoder(nn.Module):
    """Stacks multiple DecoderBlocks."""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        target_vocab_size: int,
        max_seq_len: int,
        dropout_prob: float = 0.1,
        activation: nn.Module = nn.GELU(),
    ):
        """
        Args:
            num_layers: Number of DecoderBlocks to stack.
            d_model: Dimension of embeddings and hidden states.
            num_heads: Number of attention heads in each DecoderBlock.
            d_ff: Inner dimension of the Feed-Forward Networks.
            target_vocab_size: Size of the target vocabulary.
            max_seq_len: Maximum sequence length (for positional encoding).
            dropout_prob: Dropout probability.
            activation: Activation function for FFNs.
        """
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(target_vocab_size, d_model)
        # Use Sinusoidal PE
        self.pos_encoding = PositionalEncoding(d_model, dropout_prob, max_seq_len)
        # Removed separate dropout
        # self.dropout = nn.Dropout(dropout_prob)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout_prob=dropout_prob,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )
        # Optional final LayerNorm
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        target: torch.Tensor,
        encoder_output: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the entire Decoder stack.

        Args:
            target: Target token IDs. Shape: (batch_size, target_seq_len).
            encoder_output: Output from the Encoder. Shape: (batch_size, source_seq_len, d_model).
            target_mask: Look-ahead mask for the target sequence.
                         Shape: (batch_size/1, 1, target_seq_len, target_seq_len). `True` = masked.
            encoder_mask: Padding mask for the source sequence (used in cross-attention).
                          Shape: (batch_size, 1, 1, source_seq_len). `True` = masked.

        Returns:
            Output tensor from the final DecoderBlock. Shape: (batch_size, target_seq_len, d_model).
        """
        batch_size, target_seq_len = target.shape

        # 1. Embeddings and Positional Encoding for Target Sequence
        target_embeddings = self.embedding(target)
        target_embeddings *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        # Add positional encodings (dropout is handled inside PE)
        out = self.pos_encoding(target_embeddings)
        # Removed separate dropout: out = self.dropout(embeddings_with_pe)

        # 2. Pass through Decoder Blocks
        for layer in self.layers:
            out = layer(
                target=out,
                encoder_output=encoder_output,
                target_mask=target_mask,
                encoder_mask=encoder_mask,
            )

        # 3. Optional Final LayerNorm
        if self.final_norm is not None:
            out = self.final_norm(out)

        return out


class EncoderDecoder(nn.Module):
    """The complete Encoder-Decoder Transformer model."""

    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        input_vocab_size: int,
        target_vocab_size: int,
        max_seq_len: int,
        dropout_prob: float = 0.1,
        activation: nn.Module = nn.GELU(),
        padding_idx: int = 0,  # Assuming padding token ID is 0
    ):
        """
        Args:
            num_encoder_layers: Number of layers in the encoder.
            num_decoder_layers: Number of layers in the decoder.
            d_model: Model dimension.
            num_heads: Number of attention heads.
            d_ff: Feed-forward inner dimension.
            input_vocab_size: Source vocabulary size.
            target_vocab_size: Target vocabulary size.
            max_seq_len: Maximum sequence length for positional encoding.
            dropout_prob: Dropout probability.
            activation: FFN activation function.
            padding_idx: Index of the padding token in the vocabulary.
        """
        super().__init__()
        self.padding_idx = padding_idx

        self.encoder = Encoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            input_vocab_size=input_vocab_size,
            max_seq_len=max_seq_len,
            dropout_prob=dropout_prob,
            activation=activation,
        )

        self.decoder = Decoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            target_vocab_size=target_vocab_size,
            max_seq_len=max_seq_len,
            dropout_prob=dropout_prob,
            activation=activation,
        )

        # Final linear layer to project decoder output to target vocabulary size
        self.final_linear = nn.Linear(d_model, target_vocab_size)

        # Optional: Weight tying (share weights between target embedding and final linear layer)
        # self.final_linear.weight = self.decoder.embedding.weight

        # Initialize parameters (optional but good practice)
        self._initialize_weights()

    def _initialize_weights(self):
        # Simple Xavier initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _create_padding_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """Creates a padding mask from token IDs."""
        # tokens shape: (batch_size, seq_len)
        mask = tokens == self.padding_idx
        # Reshape for multi-head attention: (batch_size, 1, 1, seq_len)
        return mask.unsqueeze(1).unsqueeze(2)

    def _create_look_ahead_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Creates a look-ahead mask for target sequence."""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
        # Reshape for multi-head attention: (1, 1, size, size)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self, src_tokens: torch.Tensor, target_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the Encoder-Decoder model.

        Args:
            src_tokens: Source sequence token IDs. Shape: (batch_size, src_seq_len).
            target_tokens: Target sequence token IDs (shifted right during training).
                           Shape: (batch_size, target_seq_len).

        Returns:
            Output logits over the target vocabulary.
            Shape: (batch_size, target_seq_len, target_vocab_size).
        """
        # 1. Create Masks
        src_padding_mask = self._create_padding_mask(src_tokens)
        target_padding_mask = self._create_padding_mask(target_tokens)
        target_look_ahead_mask = self._create_look_ahead_mask(
            target_tokens.size(1), target_tokens.device
        )
        # Combine target padding and look-ahead masks
        combined_target_mask = torch.logical_or(
            target_padding_mask, target_look_ahead_mask
        )

        # 2. Pass through Encoder
        encoder_output = self.encoder(src_tokens, mask=src_padding_mask)

        # 3. Pass through Decoder
        decoder_output = self.decoder(
            target=target_tokens,
            encoder_output=encoder_output,
            target_mask=combined_target_mask,
            encoder_mask=src_padding_mask,  # Use source padding mask for cross-attention
        )

        # 4. Final Linear Projection
        output_logits = self.final_linear(decoder_output)

        return output_logits


# --- Update Example Usage for Full Model --- #
if __name__ == "__main__":
    # Hyperparameters
    NUM_ENCODER_LAYERS = 2
    NUM_DECODER_LAYERS = 2
    D_MODEL = 64
    NUM_HEADS = 8
    D_FF = D_MODEL * 4
    INPUT_VOCAB_SIZE = 1000
    TARGET_VOCAB_SIZE = 1200
    MAX_SEQ_LEN = 50
    DROPOUT_PROB = 0.1
    PADDING_IDX = 0  # Assume padding token ID is 0

    BATCH_SIZE = 4
    SOURCE_SEQ_LEN = 30
    TARGET_SEQ_LEN = 35

    # --- Create Dummy Data --- #
    # Ensure padding_idx is used in the data
    src_tokens = torch.randint(1, INPUT_VOCAB_SIZE, (BATCH_SIZE, SOURCE_SEQ_LEN))
    src_tokens[0, -5:] = PADDING_IDX  # Add padding to first sample
    src_tokens[1, -10:] = PADDING_IDX  # Add padding to second sample

    target_tokens = torch.randint(1, TARGET_VOCAB_SIZE, (BATCH_SIZE, TARGET_SEQ_LEN))
    target_tokens[2, -3:] = PADDING_IDX  # Add padding to third sample
    # Note: During training, target_tokens are typically shifted right
    # (e.g., start with <BOS>, end before <EOS>)

    print(f"Source Tokens shape: {src_tokens.shape}")
    print(f"Target Tokens shape: {target_tokens.shape}")

    # --- Instantiate the Full Model --- #
    model = EncoderDecoder(
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        input_vocab_size=INPUT_VOCAB_SIZE,
        target_vocab_size=TARGET_VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        dropout_prob=DROPOUT_PROB,
        padding_idx=PADDING_IDX,
    )
    print("\nEncoder-Decoder Model Architecture:")
    # print(model) # Very verbose!
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {num_params:,}")

    # --- Perform Forward Pass --- #
    output_logits = model(src_tokens, target_tokens)

    print(f"\nOutput Logits shape: {output_logits.shape}")

    # --- Verification --- #
    expected_shape = (BATCH_SIZE, TARGET_SEQ_LEN, TARGET_VOCAB_SIZE)
    assert (
        output_logits.shape == expected_shape
    ), f"Expected output shape {expected_shape}, but got {output_logits.shape}"
    print("Encoder-Decoder Output shape verified successfully.")

    # --- Optional: Check Mask Creation --- #
    print("\n--- Mask Checks (Optional) ---")
    src_pad_mask = model._create_padding_mask(src_tokens)
    target_pad_mask = model._create_padding_mask(target_tokens)
    target_look_ahead = model._create_look_ahead_mask(TARGET_SEQ_LEN, src_tokens.device)
    combined_target = torch.logical_or(target_pad_mask, target_look_ahead)

    print(f"Source Padding Mask shape: {src_pad_mask.shape}")
    print(f"Target Padding Mask shape: {target_pad_mask.shape}")
    print(f"Target Look-Ahead Mask shape: {target_look_ahead.shape}")
    print(f"Combined Target Mask shape: {combined_target.shape}")
    # Example: Check src mask for first sample (padded last 5)
    print(f"Source Mask (Sample 0, last 10 tokens): {src_pad_mask[0, 0, 0, -10:]}")
    # Example: Check target look-ahead mask (first 5x5 corner)
    print(f"Target Look-Ahead (First 5x5):\n{target_look_ahead[0, 0, :5, :5]}")
