import torch
import torch.nn as nn
import math


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

        # Create position indices (0, 1, ..., max_len-1)
        position = torch.arange(max_len).unsqueeze(1)  # Shape: [max_len, 1]

        # Calculate the division term for the frequencies
        # div_term = 1 / (10000^(2i / d_model))
        # Use exp(log()) trick for numerical stability:
        # div_term = exp( (0, 2, ..., d_model-2) * -(log(10000) / d_model) )
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # Shape: [d_model / 2]

        # Initialize positional encoding matrix
        pe = torch.zeros(max_len, d_model)  # Shape: [max_len, d_model]

        # Calculate sine for even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # Calculate cosine for odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension (making it [max_len, 1, d_model]) so it can be easily
        # added to the input tensor using broadcasting, assuming input shape
        # [seq_len, batch_size, d_model].
        # We unsqueeze(0) previously, let's correct this to unsqueeze(1)
        # Original paper adds PE, common implementations often use it like this.
        # Let's keep it [max_len, d_model] and add unsqueeze(1)
        # pe = pe.unsqueeze(0) # Shape: [1, max_len, d_model] - For batch_first=True
        # Let's stick to the original paper format [max_len, d_model] first
        # and adjust in forward if needed.

        # Register 'pe' as a buffer, not a parameter. Buffers are part of the model's
        # state but are not updated by the optimizer during training.
        self.register_buffer("pe", pe)  # Shape: [max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with embeddings.
                               Expected shape: [seq_len, batch_size, d_model].

        Returns:
            torch.Tensor: Tensor with positional information added.
                          Shape: [seq_len, batch_size, d_model].
        """
        # x shape: [seq_len, batch_size, d_model]
        # self.pe shape: [max_len, d_model]
        # We need the positional encoding for the first 'seq_len' positions.
        # self.pe[:x.size(0), :] gives shape [seq_len, d_model]
        # We need to add this to x. PyTorch broadcasting handles adding
        # [seq_len, d_model] to [seq_len, batch_size, d_model] by expanding
        # the PE tensor along the batch dimension implicitly.
        # However, it's often clearer to explicitly match dimensions if possible.
        # Let's stick to the implicit broadcasting for now as it's common.

        # Add positional encoding to the input tensor x.
        # self.pe is [max_len, d_model]. We select the part needed for the input sequence length x.size(0)
        # The shape required is [seq_len, 1, d_model] to broadcast correctly with [seq_len, batch_size, d_model]
        # OR keep PE as [max_len, d_model] and select slice [seq_len, d_model], then add.
        # Let's try the slice and add approach first.
        # pos_encoding_slice = self.pe[:x.size(0), :] # Shape: [seq_len, d_model]
        # Need to unsqueeze the batch dimension for broadcasting:
        pos_encoding_slice = self.pe[: x.size(0), :].unsqueeze(
            1
        )  # Shape: [seq_len, 1, d_model]

        # Ensure x is float for addition (usually is after embedding)
        x = x.float() + pos_encoding_slice

        # Apply dropout
        return self.dropout(x)


# Example Usage (Optional - can be run if this file is executed directly)
if __name__ == "__main__":
    # Parameters
    d_model = 512  # Dimension of embeddings
    vocab_size = 1000  # Example vocabulary size
    seq_len = 35  # Example sequence length
    batch_size = 4  # Example batch size
    max_len_pe = 100  # Max length for positional encoding

    # Create embedding layer
    embedding = nn.Embedding(vocab_size, d_model)

    # Create positional encoding layer
    pos_encoder = PositionalEncoding(d_model, max_len=max_len_pe)

    # Example input (batch of sequences of token IDs)
    # Shape: [seq_len, batch_size]
    src = torch.randint(0, vocab_size, (seq_len, batch_size))

    # 1. Get embeddings
    # Shape: [seq_len, batch_size, d_model]
    embedded_src = embedding(src) * math.sqrt(
        d_model
    )  # Scale embeddings (common practice)

    # 2. Add positional encoding
    # Shape: [seq_len, batch_size, d_model]
    positioned_src = pos_encoder(embedded_src)

    print("Input shape (token IDs):", src.shape)
    print("Shape after embedding:", embedded_src.shape)
    print("Shape after positional encoding:", positioned_src.shape)

    # Verify that PE values are added
    print("\nPositional Encoding values for position 0:")
    print(pos_encoder.pe[0, :8])  # Print first 8 dims for pos 0

    print("\nPositional Encoding values for position 1:")
    print(pos_encoder.pe[1, :8])  # Print first 8 dims for pos 1

    print("\nInput embedding for first token, first batch element (first 8 dims):")
    print(embedded_src[0, 0, :8])

    print("\nOutput after PE for first token, first batch element (first 8 dims):")
    print(positioned_src[0, 0, :8])

    # Check if the difference matches the PE
    diff = positioned_src[0, 0, :8] - embedded_src[0, 0, :8]
    print("\nDifference (should approx match PE[0] before dropout):")
    print(diff)

    # Test with dropout disabled for exact match verification
    pos_encoder_no_dropout = PositionalEncoding(
        d_model, dropout_prob=0.0, max_len=max_len_pe
    )
    positioned_src_no_dropout = pos_encoder_no_dropout(embedded_src)
    diff_no_dropout = positioned_src_no_dropout[0, 0, :8] - embedded_src[0, 0, :8]
    print("\nDifference without dropout (should exactly match PE[0]):")
    print(diff_no_dropout)
    print("PE[0] for comparison:")
    print(pos_encoder_no_dropout.pe[0, :8])

    # Test edge cases for __init__
    try:
        PositionalEncoding(d_model=-1)
    except ValueError as e:
        print(f"\nCaught expected error: {e}")
    try:
        PositionalEncoding(d_model=d_model, dropout_prob=1.1)
    except ValueError as e:
        print(f"Caught expected error: {e}")
    try:
        PositionalEncoding(d_model=d_model, max_len=0)
    except ValueError as e:
        print(f"Caught expected error: {e}")
