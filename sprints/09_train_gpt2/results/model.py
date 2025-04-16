import torch
import torch.nn as nn
import torch.nn.functional as F  # Often useful for activation functions or loss
import math  # Added for potential scaling

# Import our building blocks and config
from gpt_decoder_block import GPTDecoderBlock
from positional_encoding import PositionalEncodingBatchFirst  # Import the new PE
from config import GPTConfig  # Import the config class

# We'll need positional encoding too - let's assume it's in a file named positional_encoding.py
# from .positional_encoding import PositionalEncoding # TODO: Import later


class GPT(nn.Module):
    """The full GPT-style Transformer model (decoder-only)."""

    def __init__(self, config: GPTConfig):
        """
        Initializes the GPT model based on the provided configuration.

        Args:
            config: A GPTConfig object containing model hyperparameters.
        """
        super().__init__()
        # Store config for potential reference later (e.g., in helper methods)
        self.config = config

        # --- 1. Input Embeddings --- #
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_encoding = PositionalEncodingBatchFirst(
            config.d_model, max_len=config.max_seq_len, dropout_prob=config.dropout_prob
        )

        # --- 2. Stack of Decoder Blocks --- #
        self.decoder_blocks = nn.ModuleList(
            [
                GPTDecoderBlock(
                    d_model=config.d_model,
                    num_heads=config.n_heads,
                    d_ff=config.d_ff,  # Use calculated d_ff from config
                    dropout_prob=config.dropout_prob,
                    # Assuming GELU activation is default in GPTDecoderBlock
                )
                for _ in range(config.n_layers)
            ]
        )

        # --- 3. Output Layer --- #
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)

        # Optional: Weight tying (good practice for language models)
        # Improves performance and reduces parameter count
        self.token_embedding.weight = self.output_projection.weight

        # Initialize weights (important for stability and performance)
        self.apply(self._init_weights)

        print(
            f"Initialized GPT model with {config.n_layers} layers. Weight tying enabled."
        )
        print(f"Number of parameters: {self.get_num_params()/1e6:.2f}M")

    def _init_weights(self, module):
        """Applies initial weights to certain module types."""
        if isinstance(module, nn.Linear):
            # Initialize linear layers with normal distribution (std=0.02)
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Initialize embedding layers with normal distribution
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # Note: LayerNorm weights are usually left at default (1s and 0s)

    def get_num_params(self, non_embedding=True):
        """Helper to count model parameters.
        By default, excludes positional embeddings if they are fixed (not nn.Parameter).
        If weight tying is used, embeddings are counted with the output layer.
        """
        # Sum the number of elements for all parameters that require gradients
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Adjust if excluding certain embeddings (our PE is buffer, so excluded by default)
        # If we had learned PE (nn.Parameter), might exclude it here if non_embedding=True
        return n_params

    def forward(
        self, idx: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass of the GPT model.

        Args:
            idx: Input tensor of token indices.
                 Shape: (batch_size, seq_len).
            mask: Attention mask (combining look-ahead and padding).
                  Shape: (batch_size, 1, seq_len, seq_len) or similar.
                  `True` indicates positions to be masked.

        Returns:
            Logits tensor over the vocabulary.
            Shape: (batch_size, seq_len, vocab_size).
        """
        batch_size, seq_len = idx.shape

        # --- 1. Get Embeddings and add Positional Encoding --- #
        token_embed = self.token_embedding(idx)  # (B, T, D)
        # Scaling embedding as per original GPT-2 practice
        token_embed = token_embed * math.sqrt(self.config.d_model)
        x = self.positional_encoding(
            token_embed
        )  # PE adds encoding and applies dropout

        # --- 2. Pass through Decoder Stack --- #
        # Generate the attention mask if not provided (standard causal mask)
        if mask is None:
            # Create a causal mask: (1, 1, seq_len, seq_len)
            # True values indicate positions to be masked.
            mask = torch.triu(
                torch.ones(1, 1, seq_len, seq_len, device=idx.device), diagonal=1
            ).bool()

        for block in self.decoder_blocks:
            x = block(x, mask=mask)

        # --- 3. Final Projection --- #
        x = self.final_norm(x)
        logits = self.output_projection(x)

        return logits


# --- Basic Test --- #
if __name__ == "__main__":
    # Use the config object
    test_config = GPTConfig(
        vocab_size=1000,
        d_model=64,
        n_layers=2,
        n_heads=4,
        max_seq_len=32,
        dropout_prob=0.1,
    )
    print(f"--- Testing Model with Config ---")
    print(test_config)

    # Create model instance using the config
    model = GPT(test_config)
    # print(f"\nGPT Model Architecture:\n{model}") # Can be verbose

    # Create dummy input data
    batch_size_test = 4
    # Use seq_len <= max_seq_len from config
    seq_len_test = min(10, test_config.max_seq_len)
    dummy_input_idx = torch.randint(
        0, test_config.vocab_size, (batch_size_test, seq_len_test)
    )
    print(f"\nInput indices shape: {dummy_input_idx.shape}")

    # Mask is now optional in forward, let's test both ways
    # Test 1: No mask provided (should create causal mask internally)
    print("\n--- Testing forward pass (no mask provided) ---")
    with torch.no_grad():
        logits_output_no_mask = model(dummy_input_idx)
    print(f"Output logits shape (no mask): {logits_output_no_mask.shape}")
    expected_shape = (batch_size_test, seq_len_test, test_config.vocab_size)
    assert logits_output_no_mask.shape == expected_shape, "Shape mismatch (no mask)"
    print("Output shape (no mask) verified.")

    # Test 2: Providing an explicit mask
    print("\n--- Testing forward pass (explicit mask provided) ---")
    look_ahead_mask = torch.triu(
        torch.ones(seq_len_test, seq_len_test), diagonal=1
    ).bool()
    dummy_mask = look_ahead_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
    print(f"Explicit mask shape: {dummy_mask.shape}")
    with torch.no_grad():
        logits_output_mask = model(dummy_input_idx, mask=dummy_mask)
    print(f"Output logits shape (mask provided): {logits_output_mask.shape}")
    assert logits_output_mask.shape == expected_shape, "Shape mismatch (mask provided)"
    print("Output shape (mask provided) verified.")

    # Check parameter count
    num_params = model.get_num_params()
    print(f"\nTotal number of parameters: {num_params} (~{num_params/1e6:.2f}M)")
    print("\nModel basic tests passed.")
