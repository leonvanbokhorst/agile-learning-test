# Skills & Competencies Log

## Baseline Assessment (Start of Project)

- **General Programming (Python):** Proficient (Assumed based on user input)
  - _Note: Demonstrated understanding of modern Python packaging (`pyproject.toml`, `src`-layout)._
- **Deep Learning Concepts:** Familiar with Practical Applications (User reports experience with training, fine-tuning (LoRA), and quantization (gguf) of DL models, especially LLMs)
  - _Goal: Deepen foundational understanding required for building models from scratch._
- **PyTorch Framework:** Familiar (User reports experience with training/fine-tuning existing models)
  - _Goal: Gain proficiency in *implementing* core components (Tensors, Autograd, nn.Module internals, DataLoader customization) from scratch, not just using high-level APIs._
- **Transformer Architecture:** Novice (Conceptual understanding likely present, but practical implementation from scratch is the goal)
  - _Goal: Understand and implement key components (Attention, Positional Encoding, etc.) in PyTorch._
- **LLM (GPT-like) Implementation:** Novice (in terms of building from scratch)
  - _User has conceptual knowledge and practical experience fine-tuning, quantizing, and deploying LLMs._
  - _Goal: Build a functional model from foundational PyTorch components._
- **Hugging Face Ecosystem:** Familiar / Proficient (User reports using it for datasets and models)
- **Virtual Environments (venv/conda):** Proficient (User reports daily usage)
  - _Note: Successfully used `uv` for environment creation and package installation._
- **Git / Version Control (Git/GitHub):** Proficient (User reports essential daily usage)

## Sprint 1 Progress (Setup & Basics)

- **Environment Setup:** Completed.

  - Configured `pyproject.toml` for project dependencies and metadata.
  - Utilized `uv` to install dependencies (including `torch` and dev tools) into the virtual environment.
  - Implemented standard `src`-layout to resolve packaging ambiguity.

- **Tensor Basics:** Completed exercises in `results/01_tensor_basics.py` covering:

  - Creating tensors (various methods)
  - Basic tensor operations (+, -, \*, /)
  - Indexing and slicing
  - Reshaping (`view`, `reshape`)
  - Permuting dimensions (`permute`)
  - Moving tensors between CPU/GPU (`.to()`, `.cuda()`, `.cpu()`)

- **Autograd & Gradients:** Completed exercises in `results/02b_autograd_scalar_example.py` and `results/02c_neural_network_gradients.py`

  - Understanding computation graphs and gradient tracking
  - Using `requires_grad=True` for automatic differentiation
  - Computing gradients with `.backward()`
  - Managing gradient accumulation with `.zero_grad()`
  - Understanding gradient flow in neural networks
  - Implementing gradient descent with learning rates

- **Neural Network Basics:** Completed exercises in `results/02d_hidden_layer_network.py`

  - Implementing a simple network with one hidden layer
  - Understanding parameter count and connections
  - Working with activation functions (ReLU)
  - Understanding signal flow through layers
  - Documenting network architecture and concepts
  - Creating comprehensive documentation in `notes/02d_hidden_layer_network.md`

- **Documentation & Progress Tracking:**
  - Maintaining detailed notes in the `notes/` directory
  - Updating sprint progress in README files
  - Tracking milestones and competencies
  - Creating clear, educational examples with comments

## Sprint 2 Progress (Datasets & DataLoaders)

- **Custom Datasets:** Completed exercises in [results/01_simple_dataset.py](results/01_simple_dataset.py) covering:

  - Implementing `torch.utils.data.Dataset` interface (`__len__`, `__getitem__`).
  - Handling data generation within the dataset.
  - Basic type hinting for datasets (`Dataset[tuple[torch.Tensor, torch.Tensor]]`).
  - Applying simple transforms within `__init__` (though `transform` argument is preferred).

- **DataLoaders:** Completed exercises in [results/01_simple_dataset.py](results/01_simple_dataset.py) and [results/02_dataloader_features.py](results/02_dataloader_features.py) covering:

  - Wrapping a `Dataset` with `torch.utils.data.DataLoader`.
  - Configuring `batch_size`.
  - Understanding and implementing `shuffle=True` for training data randomization.
  - Understanding `num_workers` for parallel data loading.
    - Recognizing performance implications (overhead vs. `__getitem__` complexity).
    - Implementing the `if __name__ == '__main__':` guard for multiprocessing compatibility (Windows/macOS).

- **Built-in Datasets:** Completed exercises in [results/03_builtin_datasets.py](results/03_builtin_datasets.py) using `torchvision`:

  - Loading standard datasets (e.g., `torchvision.datasets.MNIST`).
  - Understanding `root`, `train`, `download` parameters.
  - Automatic dataset downloading and caching.
  - Understanding standard dataset splits (train/test).

- **Data Transformations:** Used basic transforms in [results/01_simple_dataset.py](results/01_simple_dataset.py) and [results/03_builtin_datasets.py](results/03_builtin_datasets.py):

  - Using `torchvision.transforms.Compose` to chain transforms.
  - Using `torchvision.transforms.ToTensor()` to convert image data (PIL/NumPy) to tensors and scale.
  - Using `torchvision.transforms.Normalize()` for data normalization (with dataset-specific means/stds).
  - Applying transforms via the `transform` argument in `Dataset` constructors.
  - Implementing custom transform classes (`__call__` method) ([results/04_custom_transform.py](results/04_custom_transform.py)).
  - Implementing basic data augmentation using `torchvision.transforms` (e.g., `RandomRotation`) ([results/03_load_mnist_with_augmentation.py](results/03_load_mnist_with_augmentation.py), [notes/03_data_augmentation_guide.md](notes/03_data_augmentation_guide.md)).

- **Documentation:**

  - Created notes on Dataset basics ([notes/01_dataset_basics.md](notes/01_dataset_basics.md)).
  - Created notes on DataLoader features and built-in datasets ([notes/02_dataloader_and_builtin.md](notes/02_dataloader_and_builtin.md)).
  - Created notes on Data Augmentation ([notes/03_data_augmentation_guide.md](notes/03_data_augmentation_guide.md)).
  - Updated sprint `README.md` checklists.

- **Key Insights:**

  - Mastered the `Dataset` and `DataLoader` workflow, the core PyTorch mechanism for feeding data to models efficiently.
  - Gained practical experience with essential `DataLoader` features like batching, shuffling, and parallel loading (`num_workers`), including platform-specific considerations (`if __name__ == '__main__':`).
  - Successfully used `torchvision` to load standard datasets (MNIST) and apply crucial transformations (`ToTensor`, `Normalize`), including basic data augmentation (`RandomRotation`).
  - Learned to define custom datasets and transforms, providing flexibility for non-standard data.
  - Recognized the importance of applying appropriate transformations (especially normalization) and the distinction between training-time augmentation and validation/test-time preprocessing.

- **Skipped/Deferred Topics:**
  - Advanced memory management techniques for DataLoaders.
  - `torchtext` usage (due to deprecation and compatibility issues with current PyTorch version).

## Sprint 3 Progress (Models & Training Loops)

- **`nn.Module` Basics:** Completed work in [`sprints/03_models_and_training_loops/results/01_define_nn_module.py`](../sprints/03_models_and_training_loops/results/01_define_nn_module.py) and notes in [`sprints/03_models_and_training_loops/notes/01_define_nn_module_notes.md`](../sprints/03_models_and_training_loops/notes/01_define_nn_module_notes.md) covering:
  - Defining custom models by inheriting from `nn.Module`.
  - Understanding the role of `__init__` (layer definition, calling `super().__init__()`).
  - Understanding the role of `forward` (defining data flow).
  - Instantiating and using standard layers (`nn.Linear`, `nn.Flatten`, `nn.ReLU`).
  - Testing model structure and forward pass with dummy data.
- **Activation Functions & Non-Linearity:** Documented concepts in [`sprints/03_models_and_training_loops/notes/02_activation_functions_notes.md`](../sprints/03_models_and_training_loops/notes/02_activation_functions_notes.md) covering:

  - The necessity of non-linearity to learn complex patterns.
  - The limitation of stacking only linear layers.
  - The role of activation functions (ReLU, Sigmoid, Tanh) in introducing non-linearity.
  - The placement of activation functions (typically after linear layers).

- **Loss Functions:** Completed work in [`sprints/03_models_and_training_loops/results/02_loss_functions.py`](...) and notes in [`sprints/03_models_and_training_loops/notes/02_loss_functions_training_loop_notes.md`](...) covering:

  - Understanding the purpose of loss functions (measuring error).
  - Using common loss functions (`nn.CrossEntropyLoss`, `nn.MSELoss`).
  - Understanding the input/output shapes expected by loss functions.
  - Cross-Entropy specifics (combines LogSoftmax and NLLLoss).

- **Optimizers:** Completed work in [`sprints/03_models_and_training_loops/results/03_optimizers.py`](...) and notes in [`sprints/03_models_and_training_loops/notes/03_optimizers_notes.md`](...) covering:
  - Understanding the role of optimizers in updating weights based on gradients.
  - Instantiating common optimizers (`torch.optim.Adam`, `torch.optim.SGD`).
  - Understanding key hyperparameters (`lr`, `momentum`, `betas`).
  - Recognizing Adam/AdamW as common default choices.

_(Update this section as sprints are completed or significant learning occurs. Add specific skills or concepts learned under relevant headings.)_
