"""
Tokenizes the Fake News dataset using the GPT-2 tokenizer.
Sprint 11 - Task 2
"""

# pip install transformers datasets

import datasets
from datasets import load_dataset
from transformers import AutoTokenizer

# Configuration
MODEL_CHECKPOINT = "gpt2"
DATASET_NAME = "Pulk17/Fake-News-Detection-dataset"
TEXT_COLUMN = "text"  # Column containing the news article text
MAX_LENGTH = 512  # Max sequence length for GPT-2 (can be up to 1024, but 512 is often sufficient and faster)


def tokenize_function(examples, tokenizer):
    """Applies GPT-2 tokenizer to a batch of examples."""
    # The tokenizer handles padding and truncation.
    return tokenizer(
        examples[TEXT_COLUMN],
        padding="max_length",  # Pad to max_length
        truncation=True,  # Truncate sequences longer than max_length
        max_length=MAX_LENGTH,
    )


def main():
    """Loads dataset, tokenizes it, and prints info."""
    print(f"Loading tokenizer for checkpoint: {MODEL_CHECKPOINT}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
        # GPT-2 doesn't have a default pad token, so we set it to the EOS token.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Set tokenizer pad_token to eos_token.")

    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Ensure 'transformers' is installed and the checkpoint name is correct.")
        return

    print(f"\nLoading dataset: {DATASET_NAME}...")
    try:
        dataset = load_dataset(DATASET_NAME)
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # We only have a 'train' split, let's work with that for now
    if "train" not in dataset:
        print(f"Error: 'train' split not found in dataset {DATASET_NAME}.")
        return

    train_dataset = dataset["train"]
    print(f"\nOriginal columns: {train_dataset.column_names}")

    print(
        f"\nTokenizing dataset (using {TEXT_COLUMN} column, max_length={MAX_LENGTH})..."
    )
    # Use batched=True for faster processing
    # Pass the tokenizer to the function using functools.partial or a lambda
    tokenized_dataset = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer), batched=True
    )
    print("Tokenization complete.")

    # Remove columns we likely won't need for training to keep things clean
    # This also helps when using Hugging Face Trainer later
    columns_to_remove = [
        col
        for col in train_dataset.column_names
        if col not in ["label", "input_ids", "attention_mask"]
    ]
    if columns_to_remove:
        print(f"\nRemoving original text columns: {columns_to_remove}")
        try:
            tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)
            print("Columns removed.")
        except Exception as e:
            print(f"Warning: Could not remove columns: {e}")

    print(f"\nTokenized dataset columns: {tokenized_dataset.column_names}")

    print("\nShowing first example details:")
    if len(tokenized_dataset) > 0:
        example = tokenized_dataset[0]
        print(f"  Label: {example['label']}")
        print(f"  Input IDs (first 50): {example['input_ids'][:50]}...")
        print(f"  Attention Mask (first 50): {example['attention_mask'][:50]}...")
        print(f"  Length of Input IDs: {len(example['input_ids'])}")
    else:
        print("Dataset is empty.")

    print(
        "\nScript finished. Next steps typically involve splitting data and setting up DataLoaders."
    )


if __name__ == "__main__":
    main()
