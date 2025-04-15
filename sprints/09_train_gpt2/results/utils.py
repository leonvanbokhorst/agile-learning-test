import torch
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict

# Assuming model and config are in the same directory or accessible
from model import GPT
from config import GPTConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def save_checkpoint(
    state: dict,
    is_best: bool,
    directory: str | Path,
    filename: str = "checkpoint.pth.tar",
):
    """Saves model and training state checkpoint.

    Args:
        state (dict): Contains model's state_dict and other necessary training states
                      (e.g., epoch, optimizer_state_dict, best_val_loss, config).
        is_best (bool): If True, copies the checkpoint to a 'model_best.pth.tar' file.
        directory (str | Path): Directory to save the checkpoint.
        filename (str): Base filename for the checkpoint.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    filepath = directory / filename

    # Save the checkpoint
    torch.save(state, filepath)
    logging.info(f"Checkpoint saved to {filepath}")

    # If this is the best model so far, create a copy named 'model_best.pth.tar'
    if is_best:
        best_filepath = directory / "model_best.pth.tar"
        shutil.copyfile(filepath, best_filepath)
        logging.info(f"Best model checkpoint copied to {best_filepath}")


def load_checkpoint(
    filepath: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    device: torch.device = None,
) -> dict | None:
    """Loads model and training state from a checkpoint file.

    Args:
        filepath (str | Path): Path to the checkpoint file.
        model (torch.nn.Module): Model instance to load the state into.
        optimizer (torch.optim.Optimizer, optional): Optimizer instance to load the state into. Defaults to None.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler instance to load state into. Defaults to None.
        device (torch.device, optional): Device to map the loaded tensors to. If None, uses default device.

    Returns:
        dict | None: The loaded state dictionary if successful, otherwise None.
                     Contains keys like 'epoch', 'global_step', 'best_val_loss', 'config', etc.
    """
    filepath = Path(filepath)
    if not filepath.is_file():
        logging.error(f"Checkpoint file not found at {filepath}")
        return None

    logging.info(f"Loading checkpoint from {filepath}...")
    try:
        # Determine map_location based on the provided device
        if device:
            map_location = device
        else:
            # Default to CPU if no device is specified, can be moved later
            map_location = torch.device("cpu")

        checkpoint = torch.load(filepath, map_location=map_location)

        # Load model state
        # Add strict=False if you want to allow loading partial weights or different architectures
        model.load_state_dict(checkpoint["model_state_dict"])
        logging.info("Model state loaded successfully.")

        # Load optimizer state if optimizer is provided and state exists
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logging.info("Optimizer state loaded successfully.")
        elif optimizer:
            logging.warning("Optimizer state not found in checkpoint, skipping.")

        # Load scheduler state if scheduler is provided and state exists
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logging.info("Scheduler state loaded successfully.")
        elif scheduler:
            logging.warning("Scheduler state not found in checkpoint, skipping.")

        # It's generally recommended to move the optimizer state to the correct device *after* loading
        # Although AdamW state usually isn't huge, this is good practice.
        if optimizer and device:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            logging.info(f"Optimizer state tensors moved to {device}.")

        logging.info(f"Checkpoint loaded successfully from {filepath}.")
        # Return the rest of the state (epoch, step, loss, config, etc.)
        return checkpoint

    except Exception as e:
        logging.error(f"Failed to load checkpoint from {filepath}: {e}")
        return None


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
