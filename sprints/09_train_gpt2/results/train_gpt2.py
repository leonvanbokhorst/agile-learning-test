import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import argparse
import logging
import os
import time
from pathlib import Path
import math  # Needed for lr scheduler

# Assuming Sprint 8 results (model, config, utils, tokenizer) are copied
# and accessible via relative imports
from dataset import TextDataset
from model import GPT, GPTConfig  # Assuming model.py exists here
from utils import save_checkpoint, load_checkpoint  # Assuming utils.py exists here

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)


def get_device():
    """Gets the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Note: MPS support might still be experimental for some operations
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_lr_scheduler(optimizer, warmup_iters, lr_decay_iters, min_lr, start_lr):
    """Creates a learning rate scheduler with linear warmup and cosine decay."""

    def lr_lambda(current_iter):
        if current_iter < warmup_iters:
            # Linear warmup
            return float(current_iter) / float(max(1, warmup_iters))
        if current_iter > lr_decay_iters:
            # Constant minimum learning rate after decay
            # Scale min_lr by start_lr because LambdaLR multiplies the base LR
            return min_lr / start_lr
        # Cosine decay phase
        decay_ratio = (current_iter - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0.0 <= decay_ratio <= 1.0
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        # Linearly interpolate between start_lr and min_lr
        return (min_lr + coeff * (start_lr - min_lr)) / start_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()  # Decorator to disable gradient calculations
def evaluate(
    model: GPT, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float]:
    """Evaluates the model on the given data loader.

    Args:
        model: The model to evaluate.
        loader: DataLoader for the evaluation data (e.g., validation set).
        criterion: The loss function.
        device: The device to run evaluation on.

    Returns:
        A tuple containing (average loss, perplexity).
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_tokens = 0

    start_time = time.time()
    logging.info(f"Starting evaluation on {len(loader)} batches...")

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        logits, loss = model(inputs, targets=targets)  # Assuming model returns loss

        # Accumulate loss - multiply by target sequence length for weighting
        # CrossEntropyLoss typically averages over the batch *and* sequence length
        # To get total loss before averaging for perplexity, we need to account for tokens
        # B = batch size, T = sequence length
        # loss is usually (1/B * 1/T) * sum_over_all_tokens(single_token_loss)
        # We want sum_over_all_tokens(single_token_loss) / total_num_tokens
        # loss.item() * targets.size(0) gives sum_loss_in_batch / T
        # This isn't quite right for perplexity. Let's use total loss / total tokens.
        # We need the sum of losses for *each token* in the batch.
        # The 'loss' returned by the model *should* be the average loss per token already.
        # Let's assume `loss` is the mean loss for the batch.
        total_loss += loss.item() * inputs.size(
            0
        )  # Accumulate sum of losses (scaled by batch size)
        # total_tokens += targets.numel() # Count total number of target tokens

    avg_loss = total_loss / len(loader.dataset)  # Average loss per sequence/example
    # A different approach for perplexity: Average loss per batch item is often used directly
    avg_batch_loss = total_loss / len(loader)  # Average the per-batch mean losses

    # Calculate perplexity: exp(average loss)
    # Use the average loss across all batches for perplexity calculation
    try:
        perplexity = math.exp(avg_batch_loss)
    except OverflowError:
        perplexity = float("inf")  # Handle potential overflow if loss is very high

    model.train()  # Set the model back to training mode

    eval_time = time.time() - start_time
    logging.info(
        f"Evaluation finished in {eval_time:.2f}s. Avg Loss: {avg_batch_loss:.4f}, Perplexity: {perplexity:.2f}"
    )

    return avg_batch_loss, perplexity


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train a GPT-2 style model on text data."
    )

    # Data Args
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./sprints/09_train_gpt2/data",
        help="Directory containing train.txt and val.txt",
    )
    parser.add_argument(
        "--seq_length", type=int, default=256, help="Sequence length for training."
    )

    # Model Args (using defaults from GPTConfig)
    # These could be overridden if needed, but let's keep it simple for now
    # and rely on modifying a default GPTConfig object if necessary.
    # parser.add_argument("--embed_dim", type=int, default=768)
    # parser.add_argument("--num_heads", type=int, default=12)
    # parser.add_argument("--num_layers", type=int, default=12)
    # parser.add_argument("--dropout", type=float, default=0.1)

    # Training Args
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=5, help="Number of training epochs."
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=None,
        help="Maximum training iterations (overrides epochs if set).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=6e-4,
        help="Maximum optimizer learning rate (after warmup).",
    )
    parser.add_argument(
        "--min_lr", type=float, default=6e-5, help="Minimum learning rate after decay."
    )
    parser.add_argument(
        "--warmup_iters", type=int, default=100, help="Number of warmup iterations."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.1, help="Optimizer weight decay."
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient clipping value (0 to disable).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, mps, cpu). Auto-detects if None.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of DataLoader workers."
    )

    # Checkpointing & Logging Args
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--log_dir", type=str, default="./runs", help="Directory for TensorBoard logs."
    )
    parser.add_argument(
        "--save_interval", type=int, default=1, help="Save checkpoint every N epochs."
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=1,
        help="Evaluate on validation set every N epochs.",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    logging.info(f"Starting training with arguments: {args}")

    # --- Setup Device ---
    device = torch.device(args.device) if args.device else get_device()
    logging.info(f"Using device: {device}")

    # --- Setup Directories ---
    data_path = Path(args.data_dir)
    train_file = data_path / "train.txt"
    val_file = data_path / "val.txt"
    checkpoint_path = Path(args.checkpoint_dir)
    log_path = Path(args.log_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)

    # --- Setup DataLoaders ---
    logging.info("Setting up datasets and dataloaders...")
    train_dataset = TextDataset(train_file, args.seq_length)
    val_dataset = TextDataset(val_file, args.seq_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(
            True if device.type == "cuda" else False
        ),  # Pin memory for faster GPU transfer
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # No need to shuffle validation data
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )
    logging.info(
        f"Training data: {len(train_dataset)} sequences, {len(train_loader)} batches"
    )
    logging.info(
        f"Validation data: {len(val_dataset)} sequences, {len(val_loader)} batches"
    )

    # Determine total training iterations for scheduler
    # Use max_iters if provided, otherwise calculate from epochs
    if args.max_iters:
        total_train_iters = args.max_iters
        args.num_epochs = -1  # Indicate iteration-based training
        logging.info(f"Training for a maximum of {total_train_iters} iterations.")
    else:
        iters_per_epoch = len(train_loader)
        total_train_iters = args.num_epochs * iters_per_epoch
        logging.info(
            f"Training for {args.num_epochs} epochs ({total_train_iters} iterations)."
        )

    lr_decay_iters = total_train_iters  # Decay over the whole training duration

    # --- Setup Model ---
    # Use vocab size from tokenizer via dataset
    config = GPTConfig(vocab_size=train_dataset.vocab_size, max_seq_len=args.seq_length)
    # TODO: Potentially load config from checkpoint if resuming?
    # TODO: Allow overriding config defaults via command line?

    logging.info(f"Initializing model with config: {config}")
    model = GPT(config)
    model.to(device)
    logging.info(f"Model initialized with {model.get_num_params():,} parameters.")

    # --- Setup Optimizer ---
    # Use AdamW optimizer, common for Transformers
    # TODO: Consider more advanced optimizer configurations (e.g., separate LR for biases/embeddings)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    logging.info(
        f"Optimizer: {type(optimizer).__name__} with max_lr={args.learning_rate}, weight_decay={args.weight_decay}"
    )

    # --- Setup Loss Function ---
    criterion = (
        nn.CrossEntropyLoss()
    )  # Standard loss for classification/language modeling
    logging.info(f"Loss Function: {type(criterion).__name__}")

    # --- Setup LR Scheduler ---
    scheduler = get_lr_scheduler(
        optimizer,
        warmup_iters=args.warmup_iters,
        lr_decay_iters=lr_decay_iters,  # Decay over total iterations
        min_lr=args.min_lr,
        start_lr=args.learning_rate,  # Use max LR as the target after warmup
    )
    logging.info(
        f"LR Scheduler: LambdaLR with Linear Warmup ({args.warmup_iters} iters) and Cosine Decay (to {args.min_lr})"
    )

    # --- Setup TensorBoard ---
    logging.info(f"Initializing TensorBoard SummaryWriter in {log_path}...")
    # Add a timestamped subdirectory for this specific run
    run_name = f"gpt2_seq{args.seq_length}_batch{args.batch_size}_lr{args.learning_rate}_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=log_path / run_name)
    logging.info(f"TensorBoard run name: {run_name}")

    # --- Resume from Checkpoint (Optional) ---
    start_epoch = 0
    best_val_loss = float("inf")
    global_step = 0  # Initialize global step

    if args.resume_from:
        logging.info(f"Resuming training from checkpoint: {args.resume_from}")
        # Pass model, optimizer, and scheduler to load their states
        checkpoint_data = load_checkpoint(
            args.resume_from, model, optimizer, scheduler, device
        )
        if checkpoint_data:
            # Check for potential key errors for backward compatibility
            start_epoch = checkpoint_data.get("epoch", -1) + 1  # Start from next epoch
            global_step = checkpoint_data.get("global_step", 0)  # Resume step count
            best_val_loss = checkpoint_data.get("best_val_loss", float("inf"))
            # We might want to verify the config matches, but skip for now
            # loaded_config = GPTConfig(**checkpoint_data['config'])
            logging.info(
                f"Resumed from Epoch {start_epoch}, Global Step {global_step}, Best Val Loss {best_val_loss:.4f}"
            )
        else:
            logging.error(
                f"Failed to load checkpoint {args.resume_from}. Starting from scratch."
            )
            # Reset variables just in case
            start_epoch = 0
            global_step = 0
            best_val_loss = float("inf")

    # --- Training Loop ---
    logging.info(f"Starting training...")

    # Removed `current_iter` as global_step replaces it for tracking progress
    training_complete = False
    for epoch in (
        range(start_epoch, args.num_epochs) if args.num_epochs > 0 else iter(int, 1)
    ):
        epoch_num = epoch + 1
        logging.info(
            f"--- Epoch {epoch_num}/{args.num_epochs if args.num_epochs > 0 else 'N/A'} ---"
        )

        model.train()  # Set model to training mode
        epoch_start_time = time.time()
        batch_iter_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):

            # Check if max_iters reached
            if args.max_iters and global_step >= args.max_iters:
                training_complete = True
                break

            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            # TODO: Potentially add autocast for mixed precision
            # Model expects only input indices (idx)
            logits = model(inputs)

            # Calculate loss using criterion
            # CrossEntropyLoss expects logits: (Batch, C, ...) and targets: (Batch, ...)
            # For NLP: Logits=(B, T, V), Targets=(B, T)
            # We need to reshape logits to (B*T, V) and targets to (B*T)
            B, T, V = logits.shape
            loss = criterion(logits.view(B * T, V), targets.view(B * T))

            # Backward pass and optimization
            optimizer.zero_grad(set_to_none=True)  # More efficient zeroing
            loss.backward()

            # Gradient Clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            # Update LR scheduler (per iteration)
            scheduler.step()

            # Update global step after scheduler step
            global_step += 1

            # Logging (use global_step)
            if global_step % 10 == 0:  # Log every 10 iterations
                batch_process_time = time.time() - batch_iter_time
                current_lr = scheduler.get_last_lr()[0]
                logging.info(
                    f"Iter {global_step}/{total_train_iters} | Loss: {loss.item():.4f} | LR: {current_lr:.6f} | Batch Time: {batch_process_time*1000:.2f}ms"
                )
                # Log to TensorBoard (use global_step)
                if writer:
                    writer.add_scalar("Loss/train", loss.item(), global_step)
                    writer.add_scalar("LearningRate", current_lr, global_step)
                batch_iter_time = time.time()

        # --- End of Epoch ---
        epoch_duration = time.time() - epoch_start_time
        logging.info(f"Epoch {epoch_num} finished in {epoch_duration:.2f} seconds.")

        if training_complete:
            logging.info(f"Reached max_iters ({args.max_iters}). Stopping training.")
            break

        # --- Evaluation and Checkpointing (End of Epoch based) ---
        # We only run evaluation and checkpointing if training by epochs
        if args.num_epochs > 0:
            if epoch_num % args.eval_interval == 0:
                logging.info(f"Running evaluation for epoch {epoch_num}...")
                val_loss, perplexity = evaluate(model, val_loader, criterion, device)
                logging.info(
                    f"Epoch {epoch_num} Validation Loss: {val_loss:.4f}, Perplexity: {perplexity:.2f}"
                )

                if writer:
                    writer.add_scalar("Loss/validation", val_loss, global_step)
                    writer.add_scalar("Perplexity/validation", perplexity, global_step)

                # Save checkpoint if it's the best one so far
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    logging.info(
                        f"New best validation loss: {best_val_loss:.4f} (at epoch {epoch_num}, step {global_step})"
                    )
                    # Save the best model checkpoint
                    state_dict = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": (
                            scheduler.state_dict() if scheduler else None
                        ),
                        "best_val_loss": best_val_loss,
                        "config": config.__dict__,  # Save model config as dict
                    }
                    save_checkpoint(
                        state_dict,
                        is_best=True,
                        directory=checkpoint_path,
                        filename="model_best.pth.tar",
                    )

            # Save periodic checkpoints (even if not the best)
            if epoch_num % args.save_interval == 0:
                logging.info(f"Saving periodic checkpoint for epoch {epoch_num}...")
                state_dict = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": (
                        scheduler.state_dict() if scheduler else None
                    ),
                    "best_val_loss": best_val_loss,  # Still save the current best loss
                    "config": config.__dict__,
                }
                save_checkpoint(
                    state_dict,
                    is_best=False,
                    directory=checkpoint_path,
                    filename=f"checkpoint_epoch_{epoch_num}.pth.tar",
                )

    logging.info("--- Training Finished ---")

    # TODO: Consider loading the best model before final evaluation?
    # if Path(checkpoint_path / "model_best.pth.tar").exists():
    #     logging.info("Loading best model for final evaluation...")
    #     load_checkpoint(checkpoint_path / "model_best.pth.tar", model, device=device)

    # Perform final evaluation
    logging.info("Performing final evaluation on validation set...")
    final_val_loss, final_perplexity = evaluate(model, val_loader, criterion, device)
    logging.info(
        f"Final Validation Loss: {final_val_loss:.4f}, Final Perplexity: {final_perplexity:.2f}"
    )

    # Close TensorBoard writer
    if writer:
        writer.close()

        # Evaluation (run periodically based on epochs if not iter-based)
        if args.num_epochs > 0 and epoch_num % args.eval_interval == 0:
            logging.info(f"Running evaluation for epoch {epoch_num}...")
            val_loss, perplexity = evaluate(model, val_loader, criterion, device)
            logging.info(
                f"Epoch {epoch_num} Validation Loss: {val_loss:.4f}, Perplexity: {perplexity:.2f}"
            )

            if writer:
                writer.add_scalar(
                    "Loss/validation", val_loss, global_step
                )  # Log validation loss
                writer.add_scalar(
                    "Perplexity/validation", perplexity, global_step
                )  # Log perplexity

            # Checkpointing based on eval
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                logging.info(
                    f"New best validation loss: {best_val_loss:.4f} (at epoch {epoch_num})"
                )
                # TODO: Implement checkpoint saving for best model
                # save_checkpoint({ ... state ... }, is_best=True, ...)
            pass  # Placeholder

        # Checkpointing (run periodically based on epochs if not iter-based)
        if args.num_epochs > 0 and epoch_num % args.save_interval == 0:
            # TODO: Implement checkpointing
            logging.info(f"Saving checkpoint for epoch {epoch_num}... (Placeholder)")
            # save_checkpoint(...)
            pass  # Placeholder

    logging.info("--- Training Finished ---")

    # Perform final evaluation?
    logging.info("Performing final evaluation...")
    final_val_loss, final_perplexity = evaluate(model, val_loader, criterion, device)
    logging.info(
        f"Final Validation Loss: {final_val_loss:.4f}, Final Perplexity: {final_perplexity:.2f}"
    )
    # TODO: Maybe save final model separately?

    # Close TensorBoard writer
    if writer:
        writer.close()
        logging.info("TensorBoard writer closed.")


if __name__ == "__main__":
    # Ensure the script can find modules in the current sprint directory
    # when run with `python -m sprints.09...`
    # This might not be strictly necessary depending on exact structure and PYTHONPATH,
    # but can help avoid import issues.
    # print(f"Current working directory: {os.getcwd()}")
    # sprint_dir = Path(__file__).parent.parent
    # if str(sprint_dir) not in sys.path:
    #     sys.path.insert(0, str(sprint_dir.parent)) # Add project root potentially
    # print(sys.path)

    main()
