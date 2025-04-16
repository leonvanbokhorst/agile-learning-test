"""
Evaluates the fine-tuned model on the test set.
Sprint 11 - Task 5
"""

# pip install torch transformers datasets scikit-learn tqdm
# Or: uv pip install torch transformers datasets scikit-learn tqdm

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, classification_report
from tqdm.auto import tqdm
import time

# --- Configuration ---
SAVED_DATASET_DIR = (
    "sprints/11_finetune_gpt2_classification/results/tokenized_data_split"
)
FINETUNED_MODEL_DIR = "sprints/11_finetune_gpt2_classification/results/finetuned_model"
BATCH_SIZE = 16  # Can potentially be larger than training batch size for evaluation


# --- Helper Functions ---
def get_device():
    """Gets the appropriate device (CUDA/MPS/CPU)."""
    if torch.cuda.is_available():
        print("Using CUDA (GPU)")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon GPU)")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")


def load_test_data(dataset_dir, batch_size):
    """Loads the test split and creates a DataLoader."""
    print(f"Loading dataset from {dataset_dir}...")
    try:
        split_dataset = load_from_disk(dataset_dir)
        if "test" not in split_dataset:
            raise ValueError("'test' split not found in the saved dataset.")

        split_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )
        test_dataset = split_dataset["test"]
        print("Test dataset loaded and format set.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

    print(f"Creating Test DataLoader with batch_size={batch_size}...")
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    print("Test DataLoader created.")
    return test_dataloader


def load_finetuned_model(model_dir, device):
    """Loads the fine-tuned model from the specified directory."""
    print(f"Loading fine-tuned model from: {model_dir}...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.to(device)
        print("Model loaded successfully and moved to device.")
    except Exception as e:
        print(f"Error loading model from {model_dir}: {e}")
        print("Ensure the path is correct and the model was saved properly.")
        raise
    return model


# --- Evaluation Logic ---
def evaluate_on_test_set(model, test_dataloader, device):
    """Runs evaluation on the test set and returns predictions and labels."""
    model.eval()
    all_predictions = []
    all_labels = []
    progress_bar = tqdm(test_dataloader, desc="Evaluating on Test Set")
    start_time = time.time()

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    end_time = time.time()
    print(f"Evaluation completed in {end_time - start_time:.2f} seconds.")
    return all_predictions, all_labels


# --- Main Execution ---
def main():
    device = get_device()

    try:
        test_dataloader = load_test_data(SAVED_DATASET_DIR, BATCH_SIZE)
        model = load_finetuned_model(FINETUNED_MODEL_DIR, device)
    except Exception as e:
        print(f"Failed to load data or model. Exiting. Error: {e}")
        return

    print("\n--- Starting Evaluation on Test Set --- ")
    try:
        predictions, true_labels = evaluate_on_test_set(model, test_dataloader, device)

        print("\n--- Evaluation Results --- ")
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(
            true_labels, predictions, target_names=["Fake", "Real"]
        )  # Assumes 0=Fake, 1=Real

        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)

    except Exception as e:
        print(f"An error occurred during evaluation: {e}")

    print("\nScript finished.")


if __name__ == "__main__":
    main()
