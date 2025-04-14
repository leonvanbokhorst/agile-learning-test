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

### Documentation

- Created notes in [sprints/02_datasets_and_dataloaders/notes/](sprints/02_datasets_and_dataloaders/notes/)
- Updated Sprint 2 `README.md`

## Sprint 3: Models & Training Loops

### Completed

- [x] Defined a basic `nn.Module` (Linear Regression example).
- [x] Understood `__init__` and `forward` methods.
- [x] Implemented common loss functions (`nn.MSELoss`).
- [x] Implemented optimizers (`torch.optim.Adam`).
- [x] Built a complete training loop (forward, loss, backward, step, zero_grad).
- [x] Integrated `tqdm` for progress visualization.
- [x] Implemented basic evaluation loop (`model.eval()`, `torch.no_grad()`).
- [x] Calculated validation loss to monitor generalization.

### Key Insights

- Mastered the fundamental PyTorch training pipeline.
- Understood the roles and interactions of models, loss functions, and optimizers.
- Recognized the importance of `model.train()` vs `model.eval()` modes.
- Appreciated the efficiency gains from `torch.no_grad()` during evaluation.
- Gained practical experience implementing and monitoring a training/validation cycle.

### Next Steps

- **Sprint 4: Advanced Training Techniques & MNIST Classification** (Tentative)
  - Building a more complex CNN for image classification.
  - Implementing techniques like learning rate scheduling, early stopping.
  - Using TensorBoard for visualization.
  - Training a model on the MNIST dataset.

### Documentation

- Created notes and results in [sprints/03_models_and_training_loops/](sprints/03_models_and_training_loops/)
- Filled out Sprint 3 `README.md` retrospective.

## Sprint 4: Advanced Training Techniques & MNIST Classification

### Completed

- [x] Defined a basic CNN architecture (`nn.Conv2d`, `nn.MaxPool2d`, etc.).
- [x] Implemented TensorBoard integration for logging metrics (`SummaryWriter`, `add_scalar`).
- [x] Implemented Learning Rate Scheduling (`CosineAnnealingLR`, `scheduler.step()`).
- [x] Implemented Early Stopping logic (monitoring validation loss, patience, saving best model).
- [x] Combined all components into a full training/validation loop for MNIST.
- [x] Successfully trained the CNN on MNIST, observing the effects of LR scheduling and early stopping.
- [x] Correctly handled multiprocessing issues with `DataLoader` (`if __name__ == '__main__':`).
- [x] Practiced running scripts as modules (`python -m ...`) for relative imports.

### Key Insights

- Understood the practical application and benefits of TensorBoard, LR scheduling, and early stopping for managing the training process.
- Gained experience debugging common PyTorch issues like `DataLoader` multiprocessing errors and import path problems.
- Reinforced understanding of the complete PyTorch workflow from data loading to model definition, training, and basic evaluation.
- Recognized that MNIST served as a practical exercise for learning these techniques, which are transferable to more complex tasks like sequence modeling.

### Next Steps

- **Sprint 5: Embeddings & Positional Encoding** (Tentative)
  - Understanding `nn.Embedding` for representing discrete tokens (like words).
  - Implementing different types of positional encoding to inject sequence order information.
  - Moving towards building the foundational components of sequence-to-sequence models and Transformers.

### Documentation

- Created notes and results in [sprints/04_advanced_training_mnist/](./sprints/04_advanced_training_mnist/)
- Filled out Sprint 4 `README.md`.
