import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from typing import Any  # Use Any for flexibility in transform __call__ type hints


# --- Custom Transform Definition ---
class AddSquareFeatures:
    """A custom transform that concatenates the square of features.

    Input: Tensor of shape (N,)
    Output: Tensor of shape (2*N,)
    """

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        """Applies the transformation.

        Args:
            sample (torch.Tensor): The input tensor (features).

        Returns:
            torch.Tensor: The original features concatenated with their squares.
        """
        if not isinstance(sample, torch.Tensor):
            raise TypeError(f"Input sample must be a torch.Tensor, got {type(sample)}")

        squared_features = sample**2
        # Concatenate along the feature dimension (assuming sample is 1D or features are last dim)
        # For a 1D tensor (N,), dim=0 works.
        # For a 2D tensor (Batch, N), dim=1 would work.
        # Since this likely operates on single samples from __getitem__, dim=0 is appropriate.
        return torch.cat((sample, squared_features), dim=0)


# --- Simple Dataset (similar to before) ---
class SimpleDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    A simple dataset generating random data and targets.
    Applies transforms if provided.
    """

    def __init__(
        self,
        num_samples: int = 100,
        num_features: int = 3,  # Using fewer features for clarity
        seed: int = 42,
        transform: Any | None = None,  # Accept any callable transform
        target_transform: Any | None = None,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        # Generate data in a range where squaring makes a noticeable difference
        self.data = (
            torch.rand(num_samples, num_features) * 4 - 2
        )  # Data between -2 and 2
        self.targets = torch.randn(num_samples, 1)
        self.num_samples = num_samples
        self.transform = transform
        self.target_transform = target_transform
        print(f"Initialized SimpleDataset with {num_samples} samples.")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= idx < self.num_samples:
            raise IndexError("Index out of bounds")
        sample = self.data[idx]
        target = self.targets[idx]

        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            target = self.target_transform(target)

        return sample, target


# --- Demonstrate the Transform ---

# Create an instance of the custom transform
square_transform = AddSquareFeatures()

# Create a sample tensor
original_features = torch.tensor([1.0, -2.0, 3.0])
print(f"Original features: {original_features}")
print(f"Original shape: {original_features.shape}")

# Apply the transform
transformed_features = square_transform(original_features)
print(f"\nTransformed features (with squares): {transformed_features}")
print(f"Transformed shape: {transformed_features.shape}")

# --- Use Transform with Dataset and DataLoader ---
num_features_original = 3

# 1. Dataset without transform
dataset_no_transform = SimpleDataset(
    num_samples=5, num_features=num_features_original, seed=123
)
sample_no_transform, target_no_transform = dataset_no_transform[0]
print(f"\nSample 0 (no transform) shape: {sample_no_transform.shape}")
print(f"Sample 0 (no transform) data: {sample_no_transform}")

# 2. Dataset WITH the custom transform
dataset_with_transform = SimpleDataset(
    num_samples=5,
    num_features=num_features_original,
    seed=123,
    transform=square_transform,
)
sample_with_transform, target_with_transform = dataset_with_transform[0]
print(
    f"\nSample 0 (with AddSquareFeatures) shape: {sample_with_transform.shape}"
)  # Should be double the features
print(f"Sample 0 (with AddSquareFeatures) data: {sample_with_transform}")

# Verify the second half is the square of the first half
assert torch.allclose(
    sample_with_transform[num_features_original:],
    sample_with_transform[:num_features_original] ** 2,
)
print("Verified: Second half of transformed data is the square of the first half.")

# 3. Combine with other transforms using Compose
# IMPORTANT: Normalization should usually happen AFTER adding features if the
# added features change the scale/distribution significantly.
# However, a simple Normalize expects a certain number of features/channels.
# Let's define a simple normalization based on the *transformed* data stats (which is cheating a bit,
# normally you calculate stats on the training set BEFORE adding features that change dimensionality,
# or normalize per feature group).
# For simplicity here, let's just show Compose without Normalize.

composed_transform = transforms.Compose(
    [
        AddSquareFeatures(),
        # We could add another simple transform, e.g., converting to float64 just to show Compose
        lambda x: x.to(torch.float64),  # Example of a lambda transform
    ]
)

dataset_composed = SimpleDataset(
    num_samples=5,
    num_features=num_features_original,
    seed=123,
    transform=composed_transform,
)
sample_composed, target_composed = dataset_composed[0]
print(
    f"\nSample 0 (composed: AddSquareFeatures + ToFloat64) shape: {sample_composed.shape}"
)
print(f"Sample 0 (composed: AddSquareFeatures + ToFloat64) data: {sample_composed}")
print(
    f"Sample 0 (composed: AddSquareFeatures + ToFloat64) dtype: {sample_composed.dtype}"
)

# --- Using DataLoader ---
# DataLoader will now yield batches where the data tensor has doubled feature size
dl = DataLoader(dataset_with_transform, batch_size=2)
batch_data, batch_targets = next(iter(dl))
print(
    f"\nDataLoader batch data shape: {batch_data.shape}"
)  # [batch_size, 2 * num_features_original]
print(f"DataLoader batch targets shape: {batch_targets.shape}")

print("\nCustom transform demonstration complete.")
