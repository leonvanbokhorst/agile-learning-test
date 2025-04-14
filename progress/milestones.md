# Project Milestones

## Sprint 1: Environment Setup & PyTorch Fundamentals

### Completed

- [x] Environment setup with `pyproject.toml` and `uv`
- [x] Basic tensor operations and manipulations
- [x] Understanding of autograd and gradient computation
- [x] Implementation of a simple neural network with one hidden layer
- [x] Documentation of neural network concepts and architecture
- [x] Comprehensive progress tracking and documentation

### Current Focus

- Deepening understanding of neural network components
- Exploring more complex network architectures
- Preparing for transformer architecture implementation

### Next Steps

- Moving towards transformer architecture components
  - Implementing attention mechanisms
  - Understanding positional encoding
  - Building multi-head attention
- Building towards the final LLM implementation
  - Implementing transformer blocks
  - Creating the full model architecture
  - Setting up training pipeline

### Documentation

- Created detailed notes in `sprints/01_setup_and_basics/notes/`
- Updated progress in README files
- Maintained skills and competencies log
- Documented neural network concepts in `02d_hidden_layer_network.md`

## Sprint 2: Datasets & DataLoaders

### Completed

- [x] Implementation of custom `Dataset` (`__len__`, `__getitem__`)
- [x] Understanding and configuration of `DataLoader` (batching, shuffling)
- [x] Understanding parallel loading (`num_workers`) and associated caveats (`if __name__ == '__main__':`)
- [x] Loading and using built-in datasets (`torchvision.datasets.MNIST`)
- [x] Applying basic data transformations (`Compose`, `ToTensor`, `Normalize`)
- [x] Documentation of Dataset and DataLoader concepts and examples
- [x] Implemented custom transform class ([results/04_custom_transform.py](results/04_custom_transform.py))
- [x] Implemented basic data augmentation (`torchvision.transforms` like `RandomRotation`)

### Key Insights

- Mastered the `Dataset` and `DataLoader` workflow, the core PyTorch mechanism for feeding data to models efficiently.
- Gained practical experience with essential `DataLoader` features like batching, shuffling, and parallel loading (`num_workers`), including platform-specific considerations (`if __name__ == '__main__':`).
- Successfully used `torchvision` to load standard datasets (MNIST) and apply crucial transformations (`ToTensor`, `Normalize`), including basic data augmentation (`RandomRotation`).
- Learned to define custom datasets and transforms, providing flexibility for non-standard data.
- Recognized the importance of applying appropriate transformations (especially normalization) and the distinction between training-time augmentation and validation/test-time preprocessing.

### Skipped/Deferred

- Deeper dive into DataLoader memory management.
- `torchtext` dataset handling (due to library deprecation/compatibility).

### Current Focus

- Consolidating understanding of data handling pipeline.
- Preparing for model building and training loop implementation in next sprint.

### Next Steps

- **Sprint 3: Models & Training Loops**
  - Defining `nn.Module` based models.
  - Implementing loss functions (`nn.CrossEntropyLoss`, etc.).
  - Understanding optimizers (`torch.optim.Adam`, `SGD`).
  - Building a complete training loop (forward pass, loss calc, backward pass, optimizer step).
  - Implementing basic evaluation metrics.

### Documentation

- Created notes in [sprints/02_datasets_and_dataloaders/notes/](sprints/02_datasets_and_dataloaders/notes/)
- Updated Sprint 2 `README.md`
