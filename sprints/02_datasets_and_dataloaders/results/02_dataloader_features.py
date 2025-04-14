import torch
from torch.utils.data import Dataset, DataLoader
import time


# Reuse SimpleDataset from the previous example
class SimpleDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    A simple dataset generating random data and targets.
    """

    def __init__(
        self, num_samples: int = 100, num_features: int = 5, seed: int = 42
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.data = torch.randn(num_samples, num_features)
        self.targets = torch.randn(num_samples, 1)
        self.num_samples = num_samples
        print(f"Initialized SimpleDataset with {num_samples} samples.")

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the sample and target at the given index."""
        if not 0 <= idx < self.num_samples:
            raise IndexError("Index out of bounds")
        # Remove the small delay
        # time.sleep(0.001) # <-- Keep this commented out or remove
        return self.data[idx], self.targets[idx]


# Function to time the iteration over a DataLoader
def time_dataloader_iteration(loader: DataLoader, num_epochs_to_time: int) -> float:
    start_time = time.time()
    end_time = time.time()
    return end_time - start_time


# --- Main execution block ---
if __name__ == "__main__":
    # --- Configuration ---
    dataset_size = 200  # Let's use a slightly larger dataset
    batch_size = 32
    num_epochs = 2
    num_workers = 2  # Use 2 worker processes for parallel loading
    seed = 42

    # --- Create Dataset ---
    dataset = SimpleDataset(num_samples=dataset_size, seed=seed)

    print("\n--- DataLoader without Shuffling ---")
    dl_no_shuffle = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )  # Explicitly set workers=0 here for clarity

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1} (No Shuffle):")
        epoch_first_idx = -1
        for i, (batch_data, batch_targets) in enumerate(dl_no_shuffle):
            if i == 0:
                first_sample_in_batch = batch_data[0]
                original_idx = (
                    (dataset.data == first_sample_in_batch.unsqueeze(0))
                    .all(dim=1)
                    .nonzero(as_tuple=True)[0]
                    .item()
                )
                print(
                    f"  Batch {i}: First sample corresponds to original index {original_idx}"
                )
                assert (
                    original_idx == 0
                ), f"Epoch {epoch+1} (No Shuffle): First batch didn't start with index 0, but with {original_idx}!"
                epoch_first_idx = original_idx
            if i < 3:
                print(
                    f"  Batch {i}: Data shape {batch_data.shape}, Targets shape {batch_targets.shape}"
                )
        print(
            f"Epoch {epoch+1} completed. First batch started with original index {epoch_first_idx}."
        )

    print("\n--- DataLoader with Shuffling ---")
    # Use num_workers=0 for the shuffle demonstration part as well, unless specifically testing worker interaction with shuffling
    dl_shuffle = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )

    epoch1_first_idx = -1
    epoch2_first_idx = -1

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1} (Shuffle=True):")
        for i, (batch_data, batch_targets) in enumerate(dl_shuffle):
            if i == 0:
                first_sample_in_batch = batch_data[0]
                original_idx = (
                    (dataset.data == first_sample_in_batch.unsqueeze(0))
                    .all(dim=1)
                    .nonzero(as_tuple=True)[0]
                    .item()
                )
                print(
                    f"  Batch {i}: First sample corresponds to original index {original_idx}"
                )
                if epoch == 0:
                    epoch1_first_idx = original_idx
                else:
                    epoch2_first_idx = original_idx
            if i < 3:
                print(
                    f"  Batch {i}: Data shape {batch_data.shape}, Targets shape {batch_targets.shape}"
                )

    print(
        f"\nShuffle Check: Epoch 1 started with index {epoch1_first_idx}, Epoch 2 started with index {epoch2_first_idx}"
    )
    assert (
        epoch1_first_idx != epoch2_first_idx
    ), "Shuffling failed! First index was the same across epochs."
    print("Shuffle successful: First batch index differed between epochs.")

    print("\n--- DataLoader with num_workers timing ---")
    # Now, specifically time the difference with and without workers
    print("Timing iteration with 0 workers...")
    dl_0_workers = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    time_0_workers = time_dataloader_iteration(dl_0_workers, num_epochs)
    print(f"Time with 0 workers: {time_0_workers:.4f} seconds")

    print(f"\nTiming iteration with {num_workers} workers...")
    dl_n_workers = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    time_n_workers = time_dataloader_iteration(dl_n_workers, num_epochs)
    print(f"Time with {num_workers} workers: {time_n_workers:.4f} seconds")

    print(
        f"\nNote: The time difference for num_workers might be small or even slower for this simple dataset."
    )
    print(
        "The benefit is more pronounced with complex data loading/preprocessing steps."
    )
    print(
        "Also, the first iteration with workers might be slower due to process startup overhead."
    )
