"""
Creates PyTorch DataLoaders for the tokenized and split dataset.
Sprint 11 - Final part of Task 2 (Data Preparation)
"""

# pip install torch datasets

import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk

# Configuration
SAVED_DATASET_DIR = (
    "sprints/11_finetune_gpt2_classification/results/tokenized_data_split"
)
BATCH_SIZE = 16  # Adjust based on GPU memory


def main():
    """Loads the split dataset and creates DataLoaders."""
    print(f"Loading tokenized and split dataset from {SAVED_DATASET_DIR}...")
    try:
        split_dataset = load_from_disk(SAVED_DATASET_DIR)
        print("Dataset loaded successfully.")
        print(split_dataset)
    except FileNotFoundError:
        print(f"Error: Saved dataset not found at {SAVED_DATASET_DIR}.")
        print("Please run the previous script (03_split_data.py) first.")
        return
    except Exception as e:
        print(f"Error loading dataset from disk: {e}")
        return

    # Ensure datasets output PyTorch tensors
    print("\nSetting dataset format to PyTorch tensors...")
    try:
        split_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )
        print("Format set.")
    except Exception as e:
        print(f"Error setting dataset format: {e}")
        return

    # Create DataLoaders
    print(f"\nCreating DataLoaders with batch_size={BATCH_SIZE}...")
    try:
        train_dataloader = DataLoader(
            split_dataset["train"],
            shuffle=True,  # Shuffle training data
            batch_size=BATCH_SIZE,
        )

        val_dataloader = DataLoader(
            split_dataset["validation"],
            shuffle=False,  # No need to shuffle validation data
            batch_size=BATCH_SIZE,
        )

        test_dataloader = DataLoader(
            split_dataset["test"],
            shuffle=False,  # No need to shuffle test data
            batch_size=BATCH_SIZE,
        )
        print("DataLoaders created.")
    except Exception as e:
        print(f"Error creating DataLoaders: {e}")
        return

    # Test one batch from the train_dataloader
    print("\nTesting one batch from train_dataloader...")
    try:
        batch = next(iter(train_dataloader))
        print("Batch successfully retrieved.")
        print("Keys in batch:", batch.keys())
        print("Shapes of tensors in batch:")
        for key, value in batch.items():
            print(f"  {key}: {value.shape}")
        # Example: Check label tensor type and values
        print(f"  Label tensor dtype: {batch['label'].dtype}")
        print(f"  Example labels in batch: {batch['label'][:5]}...")
    except StopIteration:
        print("Error: Train dataloader seems empty.")
    except Exception as e:
        print(f"Error retrieving or inspecting batch: {e}")

    print("\nScript finished. DataLoaders are ready for model training/evaluation.")


if __name__ == "__main__":
    main()
