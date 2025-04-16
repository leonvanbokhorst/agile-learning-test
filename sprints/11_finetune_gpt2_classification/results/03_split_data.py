"""
Splits the tokenized dataset into train, validation, and test sets.
Sprint 11 - Part of Task 2 (Data Preparation)
"""

# pip install transformers datasets pyarrow

import os
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import pyarrow  # Often needed for saving/loading datasets
import datasets  # Need this for ClassLabel

# Configuration (should match 02_tokenize_data.py)
MODEL_CHECKPOINT = "gpt2"
DATASET_NAME = "Pulk17/Fake-News-Detection-dataset"
TEXT_COLUMN = "text"
MAX_LENGTH = 512
OUTPUT_DIR = "sprints/11_finetune_gpt2_classification/results/tokenized_data_split"


# --- Reusing Tokenization Logic (from 02_tokenize_data.py) ---
def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples[TEXT_COLUMN],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )


def load_and_tokenize_data(tokenizer):
    print(f"Loading dataset: {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME)
    print("Dataset loaded.")

    if "train" not in dataset:
        raise ValueError(f"'train' split not found in dataset {DATASET_NAME}.")

    train_dataset = dataset["train"]
    original_columns = train_dataset.column_names
    print(f"Original columns: {original_columns}")

    print(f"Tokenizing dataset (using {TEXT_COLUMN}, max_length={MAX_LENGTH})...")
    tokenized_dataset = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        desc="Running tokenizer on dataset",
    )
    print("Tokenization complete.")

    columns_to_remove = [
        col
        for col in original_columns
        if col not in ["label", "input_ids", "attention_mask"]
    ]
    if columns_to_remove:
        print(f"Removing columns: {columns_to_remove}")
        tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)
        print("Columns removed.")

    # Cast the label column to ClassLabel before splitting
    print("Casting label column to ClassLabel...")
    # Assuming 0 = Fake, 1 = Real from previous exploration
    try:
        tokenized_dataset = tokenized_dataset.cast_column(
            "label", datasets.ClassLabel(num_classes=2, names=["Fake", "Real"])
        )
        print(
            f"Label column features after cast: {tokenized_dataset.features['label']}"
        )
    except Exception as e:
        print(
            f"Warning: Could not cast label column: {e}. Stratification might fail or be inaccurate."
        )

    print(f"Tokenized dataset columns: {tokenized_dataset.column_names}")
    return tokenized_dataset


# --- End of Reused Logic ---


def main():
    """Loads, tokenizes, splits, and saves the dataset."""
    print(f"Loading tokenizer for checkpoint: {MODEL_CHECKPOINT}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Set tokenizer pad_token to eos_token.")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    try:
        # Load and tokenize the data
        tokenized_data = load_and_tokenize_data(tokenizer)

        # Split: 80% train, 20% temp
        print("\nSplitting data (80% train, 20% temporary)...")
        train_testvalid = tokenized_data.train_test_split(
            test_size=0.2, seed=42, stratify_by_column="label"
        )

        # Split temp: 50% validation, 50% test (10% of original each)
        print("Splitting temporary data (50% validation, 50% test)...")
        test_valid = train_testvalid["test"].train_test_split(
            test_size=0.5, seed=42, stratify_by_column="label"
        )

        # Combine into a final DatasetDict
        split_dataset = DatasetDict(
            {
                "train": train_testvalid["train"],
                "validation": test_valid["train"],  # Note: first part of split is train
                "test": test_valid["test"],
            }
        )

        print("\nDataset splits created:")
        print(f"  Train examples:      {len(split_dataset['train'])}")
        print(f"  Validation examples: {len(split_dataset['validation'])}")
        print(f"  Test examples:       {len(split_dataset['test'])}")

        # Save the splits to disk
        print(f"\nSaving dataset splits to {OUTPUT_DIR}...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        split_dataset.save_to_disk(OUTPUT_DIR)
        print("Dataset splits saved successfully.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")

    print("\nScript finished.")


if __name__ == "__main__":
    main()
