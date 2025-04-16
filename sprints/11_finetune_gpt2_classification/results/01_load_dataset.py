"""
Loads and explores the Onion vs. Real News headlines dataset.
Sprint 11 - Task 1
"""

# pip install datasets pyarrow

# import pyarrow as pa

from datasets import load_dataset


def main():
    """Loads and prints basic info about the dataset."""
    print("Loading dataset...")
    # Let's try the Pulk17 dataset
    try:
        # dataset = load_dataset("onion") # Changed dataset name - didn't work
        dataset = load_dataset(
            "Pulk17/Fake-News-Detection-dataset"
        )  # Trying Pulk17 dataset
        print("\nDataset loaded successfully!")
    except Exception as e:
        print(f"\nError loading dataset: {e}")
        print(
            "Please ensure you have internet connectivity and the 'datasets' library installed."
        )
        print(
            "Try: pip install datasets pyarrow"
        )  # Added pyarrow as it's often a dependency
        return

    print("\nDataset structure:")
    print(dataset)

    # Usually there's a 'train' split, let's check
    if "train" in dataset:
        train_split = dataset["train"]
        print("\nFeatures in the 'train' split:")
        print(train_split.features)

        print(f"\nNumber of examples in 'train' split: {len(train_split)}")

        print("\nFirst 5 examples:")
        for i in range(min(5, len(train_split))):
            # Adjusted to handle potential key errors if features change
            example = train_split[i]
            text = example.get("text", "N/A")  # Check feature names after loading
            label = example.get(
                "label", "N/A"
            )  # Check feature names and label mapping after loading
            print(f"  Example {i+1}:")
            print(f"    Text: {text[:100]}...")  # Truncate long headlines
            # We might need to adjust the label interpretation based on this dataset's specifics
            print(
                f"    Label: {label} "
            )  # Temporarily removing interpretation until we see the data
    else:
        print("\nCould not find a 'train' split in the dataset.")
        print("Available splits:", list(dataset.keys()))


if __name__ == "__main__":
    main()
