import torch
import os
import json
from typing import Any, Dict

# Assuming model and config are in the same directory or accessible
from .model import GPT
from .config import GPTConfig


def save_checkpoint(model: GPT, config: GPTConfig, filename: str):
    """Saves model checkpoint.

    Includes model state dict and the configuration object.

    Args:
        model: The GPT model instance.
        config: The GPTConfig object associated with the model.
        filename: Path to save the checkpoint file (e.g., 'model_checkpoint.pt').
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Create checkpoint dictionary
    checkpoint = {
        "config": config.__dict__,  # Store config attributes as dict
        "model_state_dict": model.state_dict(),
        # Future additions: optimizer_state_dict, epoch, loss, etc.
    }

    # Save the checkpoint
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved successfully to {filename}")


def load_checkpoint(filename: str, device: str | torch.device = "cpu") -> GPT | None:
    """Loads model checkpoint.

    Reconstructs the model based on the saved configuration.

    Args:
        filename: Path to the checkpoint file.
        device: The device to load the model onto ('cpu', 'cuda', etc.).

    Returns:
        The loaded GPT model instance, or None if the file doesn't exist.
    """
    if not os.path.exists(filename):
        print(f"Checkpoint file not found: {filename}")
        return None

    # Load the checkpoint dictionary
    checkpoint = torch.load(filename, map_location=device)
    print(f"Checkpoint loaded from {filename}")

    # Recreate config from the saved dictionary
    # Need to handle potential discrepancies if GPTConfig changes over time,
    # but for now, assume direct mapping.
    config_dict = checkpoint.get("config", None)
    if config_dict is None:
        # Handle older checkpoints or different formats if necessary
        raise ValueError("Checkpoint does not contain configuration information.")

    # Filter config_dict to only include keys expected by GPTConfig
    # This adds some robustness if extra keys were saved.
    valid_keys = GPTConfig.__dataclass_fields__.keys()
    filtered_config_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
    config = GPTConfig(**filtered_config_dict)

    # Recreate the model instance with the loaded config
    model = GPT(config)

    # Load the model state dict
    model.load_state_dict(checkpoint["model_state_dict"])

    # Move model to the specified device
    model.to(device)

    print(f"Model reconstructed and state dict loaded successfully.")
    print(f"Model is on device: {next(model.parameters()).device}")

    # Make sure model is in eval mode after loading by default
    model.eval()

    return model


# --- Basic Test --- #
if __name__ == "__main__":
    print("--- Testing Checkpoint Utils ---")

    # 1. Create a dummy model and config
    test_config = GPTConfig(
        vocab_size=50, d_model=16, n_layers=1, n_heads=2, max_seq_len=32
    )
    test_model = GPT(test_config)
    test_model.eval()  # Start in eval mode for consistency

    print("\nCreated dummy model and config.")

    # 2. Define checkpoint path
    checkpoint_dir = "./checkpoints_test"
    checkpoint_path = os.path.join(checkpoint_dir, "test_checkpoint.pt")

    # 3. Save the checkpoint
    print(f"\nAttempting to save checkpoint to {checkpoint_path}...")
    save_checkpoint(test_model, test_config, checkpoint_path)

    # Verify file exists
    assert os.path.exists(
        checkpoint_path
    ), f"Checkpoint file was not created at {checkpoint_path}"
    print("Checkpoint file existence verified.")

    # 4. Load the checkpoint
    print(f"\nAttempting to load checkpoint from {checkpoint_path}...")
    loaded_model = load_checkpoint(checkpoint_path, device="cpu")

    # Verify model loaded
    assert loaded_model is not None, "Failed to load the model."
    print("Model loaded successfully.")

    # 5. Basic comparison (check parameter shapes and a few values)
    print("\nComparing original and loaded model parameters...")
    original_params = dict(test_model.named_parameters())
    loaded_params = dict(loaded_model.named_parameters())

    assert original_params.keys() == loaded_params.keys(), "Parameter names differ!"

    all_match = True
    for name, orig_param in original_params.items():
        loaded_param = loaded_params[name]
        if not torch.equal(orig_param.data, loaded_param.data):
            print(f"Mismatch found in parameter: {name}")
            all_match = False
            # break # Stop at first mismatch

    if all_match:
        print("All parameters match successfully!")
    else:
        print("Parameter mismatch detected.")

    # 6. Test forward pass of loaded model
    print("\nTesting forward pass of loaded model...")
    dummy_input_idx = torch.randint(
        0, test_config.vocab_size, (2, 10)
    )  # Batch=2, SeqLen=10
    try:
        with torch.no_grad():
            original_output = test_model(dummy_input_idx)
            loaded_output = loaded_model(dummy_input_idx)

        assert torch.allclose(
            original_output, loaded_output, atol=1e-6
        ), "Outputs of original and loaded models do not match."
        print("Forward pass output matches successfully.")
    except Exception as e:
        print(f"Error during loaded model forward pass test: {e}")

    # 7. Clean up the test checkpoint
    try:
        os.remove(checkpoint_path)
        os.rmdir(checkpoint_dir)  # Only removes if empty
        print(f"\nCleaned up test checkpoint file and directory.")
    except OSError as e:
        print(f"\nError cleaning up test files: {e}")

    print("\nCheckpoint utils tests completed.")
