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

_(Update this section as sprints are completed or significant learning occurs. Add specific skills or concepts learned under relevant headings.)_
