# Sprint 2: Datasets & DataLoaders

**Parent Epic:** Master PyTorch Fundamentals and Construct a Foundational Transformer LLM (See `../../README.md`)

## Sprint Goal

Learn how to efficiently load, preprocess, and manage data using PyTorch's `Dataset` and `DataLoader` classes. This will form the foundation for all future training tasks.

## Tasks / Learning Objectives

1. **Understanding PyTorch Dataset:**

   - [x] Study the `Dataset` class interface ([notes/01_dataset_basics.md](notes/01_dataset_basics.md))
   - [x] Implement a custom dataset for a simple task ([results/01_simple_dataset.py](results/01_simple_dataset.py))
   - [x] Understand `__len__` and `__getitem__` methods
   - [x] Handle basic data loading and preprocessing

2. **Working with DataLoaders:**

   - [x] Configure batch size and shuffling ([results/02_dataloader_features.py](results/02_dataloader_features.py), [notes/02_dataloader_and_builtin.md](notes/02_dataloader_and_builtin.md))
   - [x] Implement parallel data loading with workers ([results/02_dataloader_features.py](results/02_dataloader_features.py), [notes/02_dataloader_and_builtin.md](notes/02_dataloader_and_builtin.md))
   - [x] Handle different data types (images explored via MNIST)
   - [ ] Understand memory management (Basic understanding, not deep dive)

3. **Data Transformation Pipeline:**

   - [x] Implement data augmentation ([notes/03_data_augmentation_guide.md](notes/03_data_augmentation_guide.md), [results/03_load_mnist_with_augmentation.py](results/03_load_mnist_with_augmentation.py))
   - [x] Create custom transforms
   - [x] Use `torchvision.transforms` (ToTensor, Normalize, Compose, Augmentations) ([results/03_load_mnist_with_augmentation.py](results/03_load_mnist_with_augmentation.py))
   - [x] Handle normalization and basic preprocessing ([results/01_simple_dataset.py](results/01_simple_dataset.py), [results/03_load_mnist_with_augmentation.py](results/03_load_mnist_with_augmentation.py))

4. **Built-in Datasets:**
   - [x] Explore `torchvision.datasets` (MNIST) ([results/03_builtin_datasets.py](results/03_builtin_datasets.py))
   - [ ] ~~Work with `torchtext.datasets`~~ (Skipped due to lib deprecation/compat issues)
   - [x] Understand dataset splits (train/val/test) ([results/03_builtin_datasets.py](results/03_builtin_datasets.py))
   - [x] Handle dataset downloading and caching ([results/03_builtin_datasets.py](results/03_builtin_datasets.py))

## Definition of Done / Key Questions Answered

- [x] Can create custom datasets for different data types (Demonstrated with `SimpleDataset`)
- [x] Understand how to efficiently load and preprocess data (Core `DataLoader` usage, shuffling, worker concepts, basic & custom transforms)
- [x] Can implement data augmentation and transformation pipelines (Basic pipelines with `Compose`, `ToTensor`, `Normalize`, augmentations like `RandomRotation`, custom transforms implemented)
- [x] Know how to use built-in datasets and create custom ones (MNIST example, `SimpleDataset` example, custom transform example)
- [x] Key Question: How does PyTorch handle data loading and preprocessing efficiently? (Answered via `Dataset`, `DataLoader`, transforms, workers)

## Expected Outcomes

1. **Code Examples:**

   - [x] Custom dataset implementation ([results/01_simple_dataset.py](results/01_simple_dataset.py))
   - [x] DataLoader configuration ([results/02_dataloader_features.py](results/02_dataloader_features.py))
   - [x] Transformation pipeline (Incl. Augmentation) ([results/01_simple_dataset.py](results/01_simple_dataset.py), [results/03_load_mnist_with_augmentation.py](results/03_load_mnist_with_augmentation.py), [results/04_custom_transform.py](results/04_custom_transform.py))
   - [x] Working with built-in datasets ([results/03_load_mnist_with_augmentation.py](results/03_load_mnist_with_augmentation.py))

2. **Documentation:**
   - [x] Notes on dataset creation ([notes/01_dataset_basics.md](notes/01_dataset_basics.md))
   - [x] Best practices for data loading ([notes/02_dataloader_and_builtin.md](notes/02_dataloader_and_builtin.md) - covers core concepts)
   - [x] Notes on DataLoader features and Built-in datasets ([notes/02_dataloader_and_builtin.md](notes/02_dataloader_and_builtin.md))
   - [x] Notes on Data Augmentation ([notes/03_data_augmentation_guide.md](notes/03_data_augmentation_guide.md))
   - [ ] Memory management considerations
   - [x] Performance optimization tips ([notes/02_dataloader_and_builtin.md](notes/02_dataloader_and_builtin.md) - worker caveats)

## Prerequisites

- Completion of Sprint 1 (Setup & Basics)
- Understanding of PyTorch tensors and basic operations
- Familiarity with Python classes and inheritance

## Next Steps

After completing this sprint, we'll move on to:

- Building more complex neural networks
- Implementing transformer components
- Setting up training pipelines
