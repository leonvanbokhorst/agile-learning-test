from dataclasses import dataclass, field


@dataclass
class GPTConfig:
    """Configuration settings for the GPT model."""

    # --- Architectural Parameters --- #
    vocab_size: int = 50257  # Vocabulary size (e.g., GPT-2 base)
    d_model: int = 768  # Embedding dimension (e.g., GPT-2 base)
    n_layers: int = 12  # Number of transformer blocks (e.g., GPT-2 base)
    n_heads: int = 12  # Number of attention heads (e.g., GPT-2 base)
    max_seq_len: int = 1024  # Maximum sequence length (context window)

    # --- Feed-Forward Network --- #
    # If d_ff is not specified, calculate based on d_model
    d_ff: int = field(init=False)  # Dimension of the FFN's inner layer

    # --- Regularization --- #
    dropout_prob: float = 0.1  # Dropout probability

    # --- Optional --- #
    # activation_function: str = "gelu" # Example: Could add activation choice here
    # bias: bool = True # Example: Bias in linear layers

    def __post_init__(self):
        """Calculate derived parameters after initialization."""
        # Often, d_ff is a multiple of d_model
        self.d_ff = self.d_model * 4

    # Example usage:
    # config = GPTConfig()
    # model = GPT(config)


# --- Basic Test --- #
if __name__ == "__main__":
    config_small = GPTConfig(
        vocab_size=1000,
        d_model=64,
        n_layers=2,
        n_heads=4,
        max_seq_len=128,
        dropout_prob=0.1,
    )
    print("--- Small Config Example ---")
    print(config_small)
    # d_ff should be calculated:
    print(f"Calculated d_ff: {config_small.d_ff} (Expected: {64*4})")
    assert config_small.d_ff == 64 * 4

    config_defaults = GPTConfig()
    print("\n--- Default Config Example (GPT-2 Base like) ---")
    print(config_defaults)
    print(f"Calculated d_ff: {config_defaults.d_ff} (Expected: {768*4})")
    assert config_defaults.d_ff == 768 * 4

    print("\nConfig tests passed.")
