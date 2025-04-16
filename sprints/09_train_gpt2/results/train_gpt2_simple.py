import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import logging
from pathlib import Path
from typing import Tuple

# Import our existing components
from model import GPT, GPTConfig
from dataset import TextDataset
from utils import get_device, setup_logging


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Calculate loss for a single batch."""
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    B, T, V = logits.shape
    loss = nn.CrossEntropyLoss()(logits.view(B * T, V), target_batch.view(B * T))
    return loss


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    eval_iter: int,
) -> Tuple[float, float]:
    """Evaluate model on both training and validation sets."""
    model.eval()
    train_loss, val_loss = 0.0, 0.0

    # Evaluate on training set
    for i, (inputs, targets) in enumerate(train_loader):
        if i >= eval_iter:
            break
        train_loss += calc_loss_batch(inputs, targets, model, device).item()

    # Evaluate on validation set
    for i, (inputs, targets) in enumerate(val_loader):
        if i >= eval_iter:
            break
        val_loss += calc_loss_batch(inputs, targets, model, device).item()

    model.train()
    return train_loss / eval_iter, val_loss / eval_iter


def generate_and_print_sample(
    model: nn.Module, tokenizer: object, device: torch.device, start_context: str
) -> None:
    """Generate and print a sample from the model."""
    model.eval()
    with torch.no_grad():
        # Convert start_context to tensor
        input_ids = tokenizer.encode(start_context)
        input_tensor = torch.tensor(
            input_ids, dtype=torch.long, device=device
        ).unsqueeze(0)

        # Generate
        output = model.generate(input_tensor, max_new_tokens=50)

        # Decode and print
        generated_text = tokenizer.decode(output[0].tolist())
        print(f"\nGenerated text: {generated_text}")

    model.train()


def train_model_simple(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer: object,
) -> Tuple[list, list, list]:
    """Simple training loop implementation."""
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )

        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen


def main():
    # Setup logging here instead
    setup_logging()
    logging.info("Starting simple training loop experiment")

    # Setup device
    device = get_device()
    logging.info(f"Using device: {device}")

    # Setup data
    data_path = Path("./sprints/09_train_gpt2/data")
    train_file = data_path / "train.txt"
    val_file = data_path / "val.txt"

    # Hyperparameters
    seq_length = 64
    batch_size = 32
    num_epochs = 5
    eval_freq = 100  # Evaluate every 100 steps
    eval_iter = 10  # Use 10 batches for evaluation
    learning_rate = 6e-4

    # Create datasets and dataloaders
    train_dataset = TextDataset(train_file, seq_length)
    val_dataset = TextDataset(val_file, seq_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Setup model
    config = GPTConfig(vocab_size=train_dataset.vocab_size, max_seq_len=seq_length)
    model = GPT(config)
    model.to(device)

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Start context for generation
    start_context = "Once upon a time"

    # Train
    logging.info("Starting training...")
    start_time = time.time()

    train_losses, val_losses, track_tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs,
        eval_freq,
        eval_iter,
        start_context,
        train_dataset.tokenizer,
    )

    training_time = time.time() - start_time
    logging.info(f"Training completed in {training_time:.2f} seconds")

    # Print final results
    logging.info(f"Final training loss: {train_losses[-1]:.4f}")
    logging.info(f"Final validation loss: {val_losses[-1]:.4f}")


if __name__ == "__main__":
    main()
