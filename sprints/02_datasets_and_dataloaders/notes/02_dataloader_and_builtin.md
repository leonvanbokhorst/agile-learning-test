# Notes: DataLoader Features & Built-in Datasets

This covers learnings from `results/02_dataloader_features.py` and `results/03_builtin_datasets.py`.

## DataLoader Features (`02_dataloader_features.py`)

### Shuffling (`shuffle=True`)

- **Purpose:** Randomizes the order in which data samples are yielded by the `DataLoader` in each epoch.
- **Why:** Crucial during model training to prevent the model from learning spurious patterns based on the data order. Ensures batches are different across epochs.
- **Implementation:** Set `shuffle=True` when creating the `DataLoader`.
- **Reproducibility:** For reproducible shuffling (e.g., for debugging), you can pass a seeded `torch.Generator` to the `generator` argument.
  ```python
  # Example with seeded generator
  seed = 42
  g = torch.Generator().manual_seed(seed)
  dl_shuffle = DataLoader(dataset, batch_size=32, shuffle=True, generator=g)
  ```

### Parallel Loading (`num_workers > 0`)

- **Purpose:** Uses multiple subprocesses to load data in parallel with the main training process.
- **Why:** Can significantly speed up training _if_ data loading and preprocessing (`__getitem__` in the `Dataset`) is a bottleneck (e.g., reading from disk, complex augmentations).
- **Implementation:** Set `num_workers` to the desired number of worker processes (e.g., `num_workers=4`).
- **Caveats:**
  - **Overhead:** Creating and managing worker processes has overhead. If `__getitem__` is very fast, using workers can actually _slow down_ loading (as observed in our simple example).
  - **`if __name__ == '__main__':` Guard:** On Windows and macOS (using the 'spawn' start method), the main script logic _must_ be protected by an `if __name__ == '__main__':` block to prevent infinite process creation when the script is re-imported by workers.
  - **Recommendation:** Start with `num_workers=0` (default). Profile your training loop. If data loading is slow, cautiously increase `num_workers` and measure the impact.

## Built-in Datasets (`03_builtin_datasets.py`)

PyTorch libraries like `torchvision`, `torchtext`, and `torchaudio` provide easy access to many standard datasets.

### Loading (Example: `torchvision.datasets.MNIST`)

- **Usage:** Import the dataset class (e.g., `from torchvision.datasets import MNIST`).
- **Parameters:**
  - `root`: The directory path where the dataset will be stored/cached.
  - `train`: Boolean (`True` for training set, `False` for test set). Defines standard dataset splits.
  - `download`: Boolean (`True` to automatically download if not found in `root`).
  - `transform`: Accepts a callable (often `torchvision.transforms.Compose`) to apply transformations to the data _as it's loaded_.
  - `target_transform`: Similar to `transform`, but applied to the labels/targets.

### Common Transformations (`torchvision.transforms`)

- Built-in datasets (especially image ones) often return data in formats like PIL Images.
- **`transforms.ToTensor()`:** Converts PIL Image or NumPy array (H x W x C) to a FloatTensor (C x H x W) and scales pixel values from `[0, 255]` to `[0.0, 1.0]`. Usually the first transform for image data.
- **`transforms.Normalize(mean, std)`:** Normalizes a tensor image using the provided mean and standard deviation for each channel. Often applied after `ToTensor`.
  ```python
  # Example for MNIST (single channel)
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,)) # MNIST mean and std
  ])
  ```
- **`transforms.Compose([...])`:** Chains multiple transforms together in sequence.

### Workflow

1.  Define necessary transformations (`transforms.Compose`).
2.  Instantiate the dataset (`torchvision.datasets.MNIST(...)`), passing `root`, `train`, `download`, and `transform`.
3.  Wrap the dataset instance in a `DataLoader` for batching, shuffling, etc.
4.  Iterate through the `DataLoader` during training/evaluation.
