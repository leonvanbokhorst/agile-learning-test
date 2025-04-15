import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings.

    Uses the sinusoidal positional encoding formula from "Attention Is All You Need".
    The encodings are fixed (not learned) but registered as a buffer.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout_prob: float = 0.1):
        """
        Args:
            d_model: The dimension of the embeddings.
            max_len: The maximum sequence length that this module will support.
            dropout_prob: Dropout probability.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_prob)

        # Create a long enough positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)  # Shape: (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )  # Shape: (d_model / 2)

        # Initialize the positional encoding matrix (pe)
        pe = torch.zeros(max_len, 1, d_model)  # Shape: (max_len, 1, d_model)

        # Apply sin to even indices in the tensor; 2i
        pe[:, 0, 0::2] = torch.sin(position * div_term)

        # Apply cos to odd indices in the tensor; 2i+1
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        # Register pe as a buffer. Buffers are part of the model's state_dict,
        # but are not considered parameters to be optimized during training.
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.

        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model).
               Note: Assumes Batch First is FALSE during init, but we'll adapt.
               Let's adjust for Batch First = TRUE input: (batch_size, seq_len, d_model)

        Returns:
            Output tensor with positional encoding added.
            Shape: (batch_size, seq_len, d_model).
        """
        # x shape: (batch_size, seq_len, d_model)
        # self.pe shape: (max_len, 1, d_model)
        # Need self.pe[:x.size(1)] -> slice to seq_len, transpose? No, just slice.
        # self.pe[:x.size(1), :] shape: (seq_len, 1, d_model)

        # Add positional encoding to the input embeddings
        # We need to slice self.pe up to the sequence length of x
        # The buffer `pe` has shape (max_len, 1, d_model), but input x is (B, T, D)
        # So, we slice pe and implicitly broadcast over the batch dimension.
        x = x + self.pe[: x.size(1)].transpose(
            0, 1
        )  # Transpose (T, 1, D) -> (1, T, D) for broadcasting
        # Alternative: Adjust PE shape during init to (1, max_len, d_model) ? Yes, cleaner.

        # Let's redefine pe shape for easier broadcasting with (B, T, D) input
        # We'll adjust the __init__ slightly.

        return self.dropout(x)


# --- Let's refine the __init__ for batch-first compatibility --- #


class PositionalEncodingBatchFirst(nn.Module):
    """Injects positional information into the input embeddings (Batch First version).

    Uses the sinusoidal positional encoding formula from "Attention Is All You Need".
    Optimized for input shape (batch_size, seq_len, d_model).
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout_prob: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_prob)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        # Create PE matrix with shape (1, max_len, d_model) for easy broadcasting
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Output tensor with positional encoding added.
            Shape: (batch_size, seq_len, d_model).
        """
        # x shape: (batch_size, seq_len, d_model)
        # self.pe shape: (1, max_len, d_model)
        # Slice pe up to the input sequence length and add
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# --- Basic Test --- #
if __name__ == "__main__":
    d_model_test = 64
    max_len_test = 100
    dropout_prob_test = 0.1
    batch_size_test = 4
    seq_len_test = 20  # Must be <= max_len_test

    # Use the Batch First version
    pos_encoder = PositionalEncodingBatchFirst(
        d_model_test, max_len_test, dropout_prob_test
    )
    print(f"Positional Encoder (Batch First):\n{pos_encoder}")
    print(f"Stored PE buffer shape: {pos_encoder.pe.shape}")

    # Create dummy input (batch first)
    dummy_input = torch.zeros(batch_size_test, seq_len_test, d_model_test)
    print(f"\nInput shape: {dummy_input.shape}")

    # Apply positional encoding
    output = pos_encoder(dummy_input)
    print(f"Output shape: {output.shape}")

    # Verification
    assert (
        output.shape == dummy_input.shape
    ), f"Output shape {output.shape} does not match input shape {dummy_input.shape}"
    print("Output shape verified successfully.")

    # Check if encoding was actually added (output should not be all zeros)
    assert not torch.all(
        output == 0
    ), "Output is all zeros, PE was not added correctly."
    print("Positional encoding values verified (output is not zero).")

    # Optional: Visualize the encoding for a small d_model
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        pe_to_plot = pos_encoder.pe[0, :, :].squeeze().numpy()
        plt.figure(figsize=(10, 5))
        plt.pcolormesh(pe_to_plot, cmap="viridis")
        plt.xlabel("Embedding Dimension")
        plt.xlim((0, d_model_test))
        plt.ylabel("Position")
        plt.colorbar()
        plt.title("Sinusoidal Positional Encoding")
        print("\nPlotting positional encoding matrix...")
        # plt.show() # Can uncomment this if running interactively
        print(
            "(Plot generation attempted. If running non-interactively, it might not display)"
        )
    except ImportError:
        print("\nmatplotlib not found, skipping visualization.")
