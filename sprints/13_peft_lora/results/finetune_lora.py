import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from peft import LoraConfig, get_peft_model, TaskType  # PEFT imports
import argparse
import logging
import os
import time
import math
from tqdm import tqdm
from typing import Tuple

# Import our custom dataset specific to this sprint
from dataset import TextDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_device() -> torch.device:
    """Gets the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        logging.info("[LoRA Finetune] CUDA detected, using GPU.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Note: MPS support for PEFT/certain ops might vary.
        logging.info("[LoRA Finetune] MPS detected, using Apple Silicon GPU.")
        return torch.device("mps")
    else:
        logging.info("[LoRA Finetune] No GPU detected, using CPU.")
        return torch.device("cpu")


def print_trainable_parameters(model):
    """Prints the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    pct_trainable = 100 * trainable_params / all_param
    logging.info(
        f"Trainable params: {trainable_params:,} | All params: {all_param:,} | "
        f"Trainable %: {pct_trainable:.4f}"
    )


@torch.no_grad()
def evaluate(
    model, loader: DataLoader, device: torch.device  # Model type is now PeftModel
) -> Tuple[float, float]:
    """Evaluates the PEFT model on the validation set.

    Args:
        model: The PEFT model to evaluate.
        loader: The DataLoader for the validation set.
        device: The device to run evaluation on.

    Returns:
        A tuple containing (average validation loss, validation perplexity).
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    num_batches = 0
    logging.info("[LoRA Finetune] Starting evaluation...")
    pbar = tqdm(loader, desc="Evaluating (LoRA)", leave=False)
    for batch in pbar:
        xb, yb = batch
        xb, yb = xb.to(device), yb.to(device)

        # Pass input_ids and labels directly to the PEFT model
        outputs = model(input_ids=xb, labels=yb)
        loss = outputs.loss
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"eval_loss": loss.item()})

    # Handle case where loader is empty
    if num_batches == 0:
        logging.warning(
            "[LoRA Finetune] Evaluation loader was empty. Returning inf loss/perplexity."
        )
        return float("inf"), float("inf")

    avg_loss = total_loss / num_batches
    try:
        perplexity = math.exp(avg_loss)
    except OverflowError:
        perplexity = float("inf")
        logging.warning(
            f"[LoRA Finetune] Perplexity calculation overflowed (Avg Loss: {avg_loss}). Reporting inf."
        )

    logging.info(
        f"[LoRA Finetune] Evaluation complete. Avg Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}"
    )
    model.train()  # Set model back to training mode
    return avg_loss, perplexity


def main(args):
    """Main function to set up and run the LoRA fine-tuning process."""
    start_time = time.time()
    logging.info("[LoRA Finetune] Starting LoRA fine-tuning script for Sprint 13.")

    # --- 1. Device Setup ---
    device = get_device()

    # --- 2. Load Tokenizer (Same as before) ---
    logging.info(f"[LoRA Finetune] Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        logging.warning(
            "[LoRA Finetune] Tokenizer does not have a pad token, setting it to eos_token."
        )
        tokenizer.pad_token = tokenizer.eos_token
    logging.info(
        f"[LoRA Finetune] Tokenizer loaded. Vocab size: {tokenizer.vocab_size}, Pad token ID: {tokenizer.pad_token_id}"
    )

    # --- 3. Load Base Model (Not PEFT yet) ---
    logging.info(f"[LoRA Finetune] Loading base pre-trained model: {args.model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    logging.info(f"[LoRA Finetune] Base model loaded.")

    # --- 4. Configure LoRA ---
    logging.info("[LoRA Finetune] Configuring LoRA...")
    # Common target modules for GPT-2: "c_attn" (query, key, value projections),
    # "c_proj" (output projection in attention), "c_fc" (first layer in FFN),
    # "c_proj" after FFN. Let's start with just attention projections.
    lora_config = LoraConfig(
        r=args.lora_r,  # Rank of the update matrices
        lora_alpha=args.lora_alpha,  # Alpha scaling factor (often 2*r)
        target_modules=args.lora_target_modules,  # Modules to apply LoRA to
        lora_dropout=args.lora_dropout,  # Dropout probability for LoRA layers
        bias="none",  # Whether to train bias parameters ("none", "all", or "lora_only")
        task_type=TaskType.CAUSAL_LM,  # Task type for PEFT
    )
    logging.info(f"[LoRA Finetune] LoRA Config: {lora_config}")

    # --- 5. Apply PEFT to the Model ---
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    logging.info(f"[LoRA Finetune] PEFT model created and moved to {device}.")
    print_trainable_parameters(model)  # Crucial step to see the parameter reduction!

    # --- 6. Prepare Datasets & DataLoaders ---
    logging.info("[LoRA Finetune] Setting up datasets and dataloaders...")
    # Use the TextDataset from this sprint's dataset.py
    train_dataset = TextDataset(
        data_file=args.train_data_path, block_size=args.block_size
    )
    val_dataset = TextDataset(data_file=args.val_data_path, block_size=args.block_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),  # Pin memory if using CUDA
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    logging.info(
        f"[LoRA Finetune] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
    )

    # --- 7. Optimizer & Scheduler Setup ---
    # We only optimize the PEFT parameters!
    logging.info("[LoRA Finetune] Setting up optimizer and learning rate scheduler...")
    optimizer = optim.AdamW(
        model.parameters(),  # Optimizer now targets only trainable PEFT params
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Calculate total training steps
    # Ensure train_loader is not empty before calculating steps
    if len(train_loader) == 0:
        logging.error(
            "[LoRA Finetune] Training data loader has length 0. Cannot train."
        )
        return  # Exit if no training data

    num_training_steps = args.epochs * len(train_loader)
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    logging.info(
        f"[LoRA Finetune] Optimizer: AdamW (targeting PEFT params), LR: {args.learning_rate}, Weight Decay: {args.weight_decay}"
    )
    logging.info(
        f"[LoRA Finetune] Scheduler: {args.lr_scheduler_type}, Total steps: {num_training_steps}, Warmup steps: {num_warmup_steps}"
    )

    # --- 8. Training Loop ---
    logging.info("[LoRA Finetune] Starting training loop...")
    best_val_loss = float("inf")
    global_step = 0

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"[LoRA Finetune] Checkpoints will be saved to: {args.output_dir}")

    for epoch in range(args.epochs):
        model.train()  # Ensure model is in training mode
        epoch_loss = 0.0
        num_batches_epoch = 0
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{args.epochs} (LoRA)", leave=True
        )

        for batch in progress_bar:
            optimizer.zero_grad()

            # Move batch to device
            xb, yb = batch
            xb, yb = xb.to(device), yb.to(device)

            # Forward pass through the PEFT model
            outputs = model(input_ids=xb, labels=yb)
            loss = outputs.loss

            # Backward pass (calculates gradients only for LoRA params)
            loss.backward()

            # Optional: Gradient Clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # Optimizer and scheduler step
            optimizer.step()
            lr_scheduler.step()

            # Logging
            epoch_loss += loss.item()
            num_batches_epoch += 1
            global_step += 1
            current_lr = (
                lr_scheduler.get_last_lr()[0] if lr_scheduler else args.learning_rate
            )
            progress_bar.set_postfix({"loss": loss.item(), "lr": current_lr})

            # Optional: Log loss more frequently
            # if global_step % args.log_interval == 0:
            #     logging.info(f"[LoRA Finetune] Step: {global_step}, Loss: {loss.item():.4f}, LR: {current_lr}")

        # Avoid division by zero if epoch had no batches
        if num_batches_epoch > 0:
            avg_epoch_loss = epoch_loss / num_batches_epoch
            logging.info(
                f"[LoRA Finetune] Epoch {epoch+1} finished. Average Training Loss: {avg_epoch_loss:.4f}"
            )
        else:
            logging.warning(f"[LoRA Finetune] Epoch {epoch+1} had no batches.")
            continue  # Skip evaluation if no training happened

        # --- 9. Evaluation & Checkpointing (End of Epoch) ---
        if val_loader and len(val_loader) > 0:
            val_loss, val_perplexity = evaluate(model, val_loader, device)
            logging.info(
                f"[LoRA Finetune] Epoch {epoch+1} Validation Loss: {val_loss:.4f}, Perplexity: {val_perplexity:.4f}"
            )

            # Save PEFT checkpoint if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logging.info(
                    f"[LoRA Finetune] New best validation loss: {best_val_loss:.4f}. Saving PEFT adapter model..."
                )
                # Save the PEFT adapter weights and config
                model.save_pretrained(args.output_dir)
                # Optionally save tokenizer too, although it hasn't changed
                # tokenizer.save_pretrained(args.output_dir)
                logging.info(
                    f"[LoRA Finetune] PEFT adapter model saved to {args.output_dir}"
                )
            else:
                logging.info("[LoRA Finetune] Validation loss did not improve.")
        else:
            logging.warning(
                "[LoRA Finetune] No validation loader provided or it's empty, skipping evaluation and best model saving."
            )
            # If needed, save checkpoint unconditionally every epoch
            # model.save_pretrained(os.path.join(args.output_dir, f"epoch_{epoch+1}"))

    end_time = time.time()
    logging.info(
        f"[LoRA Finetune] Training complete in {end_time - start_time:.2f} seconds."
    )
    logging.info(f"[LoRA Finetune] Best validation loss achieved: {best_val_loss:.4f}")
    final_perplexity = (
        math.exp(best_val_loss) if best_val_loss != float("inf") else float("inf")
    )
    logging.info(
        f"[LoRA Finetune] Corresponding best validation perplexity: {final_perplexity:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune a GPT-2 model for generative tasks using LoRA (Sprint 13)."
    )

    # Model & Tokenizer Args
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt2",
        help="Name of the base pre-trained model (e.g., gpt2, gpt2-medium).",
    )

    # Data Args
    parser.add_argument(
        "--train-data-path",
        type=str,
        default="sprints/13_peft_lora/results/data/train.bin",  # Point to local data subdir
        help="Path to the training data (.bin file, relative to script location).",
    )
    parser.add_argument(
        "--val-data-path",
        type=str,
        default="sprints/13_peft_lora/results/data/val.bin",  # Point to local data subdir
        help="Path to the validation data (.bin file, relative to script location).",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Context length for the model.",
    )

    # Training Hyperparameters (can be adjusted)
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,  # May need adjustment based on GPU memory with PEFT
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,  # LoRA often uses a slightly higher LR than full FT
        help="Initial learning rate for LoRA parameters.",
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
        help="Ratio of training steps for linear warmup.",
    )
    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        default="cosine",
        help="Learning rate scheduler type.",
    )

    # LoRA Specific Args
    parser.add_argument("--lora-r", type=int, default=8, help="Rank r for LoRA.")
    parser.add_argument(
        "--lora-alpha", type=int, default=16, help="Alpha scaling for LoRA."
    )
    parser.add_argument(
        "--lora-dropout", type=float, default=0.05, help="Dropout for LoRA layers."
    )
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["c_attn"],  # Target attention projections by default for GPT-2
        help="List of module names or regex patterns to apply LoRA to.",
    )

    # Runtime Args
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of worker processes for DataLoader.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sprints/13_peft_lora/results/checkpoints/lora_finetuned_model",
        help="Directory to save LoRA adapter checkpoints.",
    )
    # parser.add_argument('--log-interval', type=int, default=50, help='Log training loss every N steps.')
    # parser.add_argument('--max-grad-norm', type=float, default=1.0, help='Gradient clipping value.')

    args = parser.parse_args()

    # Note: Unlike Sprint 12, we don't prepend base_dir to paths here, as defaults now include sprint path

    main(args)
