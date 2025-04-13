import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

print("--- Simple Dataset Example ---")


# Let's create a simple dataset for demonstration
class SimpleDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        """
        Initialize the dataset.

        Args:
            data (np.ndarray): Input data
            targets (np.ndarray): Target labels
            transform (callable, optional): Optional transform to be applied to data
        """
        self.data = torch.from_numpy(data).float()
        self.targets = torch.from_numpy(targets).float()
        self.transform = transform

    def __len__(self):
        """Return the total number of samples."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single sample and its target.

        Args:
            idx (int): Index of the sample to return

        Returns:
            tuple: (sample, target)
        """
        sample = self.data[idx]
        target = self.targets[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, target


# Create some dummy data
# Let's say we have 100 samples with 5 features each
n_samples = 100
n_features = 5

# Generate random data and targets
data = np.random.randn(n_samples, n_features)
targets = np.random.randn(n_samples, 1)  # Single target per sample

print(f"\nData shape: {data.shape}")
print(f"Targets shape: {targets.shape}")

# Create dataset instance
dataset = SimpleDataset(data, targets)

# Test dataset functionality
print("\nTesting dataset:")
print(f"Dataset length: {len(dataset)}")
sample, target = dataset[0]
print(f"First sample shape: {sample.shape}")
print(f"First target shape: {target.shape}")

# Create a DataLoader
dataloader = DataLoader(
    dataset, batch_size=32, shuffle=True, num_workers=0  # Set to 0 for this example
)

# Test DataLoader
print("\nTesting DataLoader:")
for batch_idx, (batch_data, batch_targets) in enumerate(dataloader):
    print(f"Batch {batch_idx + 1}:")
    print(f"  Data shape: {batch_data.shape}")
    print(f"  Targets shape: {batch_targets.shape}")
    if batch_idx == 2:  # Just show first 3 batches
        break


# Example of a simple transform
def normalize_transform(x):
    """Normalize the data to have zero mean and unit variance."""
    mean = x.mean()
    std = x.std()
    return (x - mean) / std


# Create dataset with transform
dataset_with_transform = SimpleDataset(data, targets, transform=normalize_transform)

# Test transformed dataset
print("\nTesting dataset with transform:")
sample, target = dataset_with_transform[0]
print(f"First sample (transformed): {sample}")
print(f"Mean: {sample.mean():.4f}")
print(f"Std: {sample.std():.4f}")

print("\nExample complete!")
