import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
import argparse
import logging
import os
import time
import math
from tqdm import tqdm
from typing import Tuple

# Import our custom dataset
from dataset import TextDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_device() -> torch.device:
    """Gets the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        logging.info("CUDA detected, using GPU.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        logging.info("MPS detected, using Apple Silicon GPU.")
        return torch.device("mps")
    else:
        logging.info("No GPU detected, using CPU.")
        return torch.device("cpu")


@torch.no_grad()
def evaluate(
    model: AutoModelForCausalLM, loader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    """Evaluates the model on the validation set.

    Args:
        model: The model to evaluate.
        loader: The DataLoader for the validation set.
        device: The device to run evaluation on.

    Returns:
        A tuple containing (average validation loss, validation perplexity).
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    num_batches = 0
    logging.info("Starting evaluation...")
    pbar = tqdm(loader, desc="Evaluating", leave=False)
    for batch in pbar:
        xb, yb = batch
        xb, yb = xb.to(device), yb.to(device)

        outputs = model(input_ids=xb, labels=yb)
        loss = outputs.loss
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)
    logging.info(
        f"Evaluation complete. Avg Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}"
    )
    model.train()  # Set model back to training mode
    return avg_loss, perplexity


def main(args):
    """Main function to set up and run the fine-tuning process."""
    start_time = time.time()
    logging.info("Starting generative fine-tuning script.")

    # --- 1. Device Setup ---
    device = get_device()

    # --- 2. Load Tokenizer ---
    logging.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        logging.info("Tokenizer does not have a pad token, setting it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    logging.info(f"Tokenizer loaded. Pad token ID: {tokenizer.pad_token_id}")

    # --- 3. Load Model ---
    logging.info(f"Loading pre-trained model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(device)
    logging.info(f"Model loaded and moved to {device}.")

    # --- 4. Prepare Datasets & DataLoaders ---
    logging.info("Setting up datasets and dataloaders...")
    train_dataset = TextDataset(
        data_file=args.train_data_path, block_size=args.block_size
    )
    val_dataset = TextDataset(data_file=args.val_data_path, block_size=args.block_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    logging.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # --- 5. Optimizer & Scheduler Setup ---
    logging.info("Setting up optimizer and learning rate scheduler...")
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    num_training_steps = args.epochs * len(train_loader)
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    logging.info(
        f"Optimizer: AdamW, LR: {args.learning_rate}, Weight Decay: {args.weight_decay}"
    )
    logging.info(
        f"Scheduler: {args.lr_scheduler_type}, Total steps: {num_training_steps}, Warmup steps: {num_warmup_steps}"
    )

    # --- 6. Training Loop ---
    logging.info("Starting training loop...")
    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
        model.train()  # Set model to training mode
        epoch_loss = 0.0
        num_batches_epoch = 0
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True
        )

        for batch in progress_bar:
            optimizer.zero_grad()

            # Move batch to device
            xb, yb = batch
            xb, yb = xb.to(device), yb.to(device)

            # Forward pass
            # Model outputs loss when labels are provided
            outputs = model(input_ids=xb, labels=yb)
            loss = outputs.loss

            # Backward pass
            loss.backward()

            # Optional: Gradient Clipping (uncomment if needed)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # Optimizer and scheduler step
            optimizer.step()
            lr_scheduler.step()

            # Logging
            epoch_loss += loss.item()
            num_batches_epoch += 1
            global_step += 1
            progress_bar.set_postfix(
                {"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
            )

            # Optional: Log loss more frequently (e.g., every N steps)
            # if global_step % args.log_interval == 0:
            #     logging.info(f"Step: {global_step}, Loss: {loss.item():.4f}")

        avg_epoch_loss = epoch_loss / num_batches_epoch
        logging.info(
            f"Epoch {epoch+1} finished. Average Training Loss: {avg_epoch_loss:.4f}"
        )

        # --- 7. Evaluation & Checkpointing (End of Epoch) ---
        if val_loader:
            val_loss, val_perplexity = evaluate(model, val_loader, device)
            logging.info(
                f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}, Perplexity: {val_perplexity:.4f}"
            )

            # Save checkpoint if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logging.info(
                    f"New best validation loss: {best_val_loss:.4f}. Saving model..."
                )
                # Save the model using Hugging Face standard method (saves config too)
                model.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)  # Save tokenizer too
                # Optionally save just the state dict
                # torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model_state_dict.pt"))
            else:
                logging.info("Validation loss did not improve.")
        else:
            # If no validation set, maybe save periodically based on epoch/steps
            logging.warning(
                "No validation loader provided, skipping evaluation and best model saving."
            )
            # Example: Save every epoch if no validation
            # model_save_path = os.path.join(args.output_dir, f"epoch_{epoch+1}_model")
            # model.save_pretrained(model_save_path)
            # tokenizer.save_pretrained(model_save_path)

    end_time = time.time()
    logging.info(f"Training complete in {end_time - start_time:.2f} seconds.")
    logging.info(f"Best validation loss achieved: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune a GPT-2 model for generative tasks."
    )

    # Model & Tokenizer Args
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt2",
        help="Name of the pre-trained model to use (e.g., gpt2, gpt2-medium).",
    )

    # Data Args
    parser.add_argument(
        "--train-data-path",
        type=str,
        default="data/train.bin",
        help="Path to the training data (.bin file).",
    )
    parser.add_argument(
        "--val-data-path",
        type=str,
        default="data/val.bin",
        help="Path to the validation data (.bin file).",
    )
    # Use a smaller block size for fine-tuning if memory is limited
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Context length (sequence length) for the model. Max 1024 for GPT-2.",
    )

    # Training Hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs."
    )
    # Adjust batch size based on GPU memory
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-5,
        help="Initial learning rate (often smaller for fine-tuning).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW optimizer.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.06,
        help="Ratio of training steps for linear warmup (common default for fine-tuning).",
    )
    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        default="cosine",
        help="Learning rate scheduler type (e.g., linear, cosine).",
    )

    # parser.add_argument('--max-grad-norm', type=float, default=1.0, help='Gradient clipping value.') # Optional

    # Runtime Args
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of worker processes for DataLoader (0 means main process).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/finetuned_model",
        help="Directory to save best model checkpoint.",
    )
    # parser.add_argument('--log-interval', type=int, default=50, help='Log training loss every N steps.') # Optional

    args = parser.parse_args()

    # Adjust paths relative to the sprint directory
    base_dir = os.path.join("sprints", "12_finetune_gpt2_generative", "results")
    args.train_data_path = os.path.join(base_dir, args.train_data_path)
    args.val_data_path = os.path.join(base_dir, args.val_data_path)
    args.output_dir = os.path.join(base_dir, args.output_dir)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Checkpoints will be saved to: {args.output_dir}")

    main(args)
