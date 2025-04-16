"""
Implements the fine-tuning loop for GPT-2 sequence classification.
Sprint 11 - Task 4
"""

# pip install torch transformers datasets tqdm scikit-learn
# Or: uv pip install torch transformers datasets tqdm scikit-learn

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoConfig
from datasets import load_from_disk
from tqdm.auto import tqdm  # Progress bars
import time
import os

# --- Configuration ---
SAVED_DATASET_DIR = (
    "sprints/11_finetune_gpt2_classification/results/tokenized_data_split"
)
MODEL_CHECKPOINT = "gpt2"
NUM_LABELS = 2
BATCH_SIZE = 4  # Reduced batch size for potential memory constraints, adjust as needed
NUM_EPOCHS = 1  # Start with 1 epoch for testing the loop
LEARNING_RATE = 5e-5  # Common starting point for fine-tuning transformers
OUTPUT_MODEL_DIR = "sprints/11_finetune_gpt2_classification/results/finetuned_model"
EVAL_STEPS = 250  # How often to run validation within an epoch


# --- Helper Functions ---
def get_device():
    """Gets the appropriate device (CUDA or CPU)."""
    if torch.cuda.is_available():
        print("Using CUDA (GPU)")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon GPU)")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")


def load_data(dataset_dir, batch_size):
    """Loads the split dataset and creates DataLoaders."""
    print(f"Loading tokenized and split dataset from {dataset_dir}...")
    try:
        split_dataset = load_from_disk(dataset_dir)
        split_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )
        print("Dataset loaded and format set.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

    print(f"Creating DataLoaders with batch_size={batch_size}...")
    train_dataloader = DataLoader(
        split_dataset["train"], shuffle=True, batch_size=batch_size
    )
    val_dataloader = DataLoader(
        split_dataset["validation"], shuffle=False, batch_size=batch_size
    )
    # We might not use test_dataloader in the training script itself
    # test_dataloader = DataLoader(split_dataset["test"], shuffle=False, batch_size=batch_size)
    print("DataLoaders created.")
    return train_dataloader, val_dataloader


def load_model(checkpoint, num_labels, device):
    """Loads the pre-trained model with a classification head."""
    print(f"Loading pre-trained model: {checkpoint}...")
    try:
        config = AutoConfig.from_pretrained(checkpoint)
        config.num_labels = num_labels
        # Important for GPT-2: Ensure pad token ID is set in config if tokenizer uses it
        # (We set tokenizer.pad_token = tokenizer.eos_token before)
        config.pad_token_id = config.eos_token_id

        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, config=config
        )
        model.to(device)
        print("Model loaded and moved to device.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    return model


def evaluate_model(model, dataloader, device):
    """Evaluates the model on the given dataloader."""
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    progress_bar = tqdm(
        dataloader, desc="Evaluating", leave=False
    )  # leave=False for intermediate evaluations

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            progress_bar.set_postfix({"eval_loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    model.train()  # Set model back to training mode
    return avg_loss, accuracy


# --- Main Training Logic ---
def main():
    device = get_device()

    try:
        train_dataloader, val_dataloader = load_data(SAVED_DATASET_DIR, BATCH_SIZE)
        model = load_model(MODEL_CHECKPOINT, NUM_LABELS, device)
    except Exception as e:
        print(f"Failed to load data or model. Exiting. Error: {e}")
        return

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = NUM_EPOCHS * len(train_dataloader)

    print("\n--- Starting Training --- ")
    print(f"Total Epochs: {NUM_EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Steps per Epoch: {len(train_dataloader)}")
    print(f"Total Training Steps: {num_training_steps}")
    print(f"Evaluation every {EVAL_STEPS} steps.")
    start_time = time.time()

    global_step = 0
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        epoch_start_time = time.time()

        # --- Training Phase ---
        model.train()  # Make sure model is in training mode at start of epoch
        total_train_loss_epoch = 0
        train_progress_bar = tqdm(
            train_dataloader, desc=f"Epoch {epoch+1} Training", leave=True
        )

        for batch in train_progress_bar:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_train_loss_epoch += loss.item()
            global_step += 1
            train_progress_bar.set_postfix(
                {"train_loss": loss.item(), "step": global_step}
            )

            # --- Intermediate Validation ---
            if global_step % EVAL_STEPS == 0:
                print(f"\n-- Running intermediate evaluation at Step {global_step} --")
                avg_val_loss, accuracy = evaluate_model(model, val_dataloader, device)
                print(f"Step {global_step} Validation Loss: {avg_val_loss:.4f}")
                print(f"Step {global_step} Validation Accuracy: {accuracy:.4f}")
                # Make sure model is back in training mode after evaluation
                model.train()

        avg_train_loss_epoch = total_train_loss_epoch / len(train_dataloader)
        epoch_end_time = time.time()
        print(f"End of Epoch {epoch + 1}")
        print(f"  Average Training Loss: {avg_train_loss_epoch:.4f}")
        print(f"  Epoch Duration: {epoch_end_time - epoch_start_time:.2f} seconds")

        # --- Final Validation for the Epoch ---
        print("-- Running final validation for Epoch {epoch + 1} --")
        avg_val_loss, accuracy = evaluate_model(model, val_dataloader, device)
        print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f}")
        print(f"Epoch {epoch + 1} Validation Accuracy: {accuracy:.4f}")

    # --- End of Training ---
    total_end_time = time.time()
    print(f"\n--- Training Finished --- ")
    print(f"Total Training Time: {total_end_time - start_time:.2f} seconds")

    # --- Saving the Model (Optional but Recommended) ---
    save_model = input("\nDo you want to save the fine-tuned model? (y/n): ").lower()
    if save_model == "y":
        print(f"Saving model to {OUTPUT_MODEL_DIR}...")
        os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
        try:
            model.save_pretrained(OUTPUT_MODEL_DIR)
            # If you want to save the tokenizer too (good practice)
            # tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
            print("Model saved successfully.")
        except Exception as e:
            print(f"Error saving model: {e}")

    print("\nScript finished.")


if __name__ == "__main__":
    main()
