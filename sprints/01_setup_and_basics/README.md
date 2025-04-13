# Sprint 1: Environment Setup & PyTorch Fundamentals

**Parent Epic:** Master PyTorch Fundamentals and Construct a Foundational Transformer LLM (See `../../README.md`)

## Sprint Goal

Set up the development environment with necessary libraries (PyTorch, etc.) and gain hands-on experience with core PyTorch concepts: Tensors, Autograd, and basic Neural Network modules.

## Tasks / Learning Objectives

1.  [x] **Environment Setup:**
    - [x] Create a virtual environment (e.g., `venv` or `conda`). (Done via `uv`)
    - [x] Install PyTorch (CPU or GPU version as appropriate). (Done via `pyproject.toml` + `uv`)
    - [x] Install other useful libraries (e.g., `numpy`, `matplotlib`, `jupyter` if desired for experimentation). (Done via `pyproject.toml` + `uv` for dev tools)
    - [x] Verify installation by running simple PyTorch commands. (Implicitly done by successful `uv install`)
2.  [x] **PyTorch Tensors:**
    - [x] Create tensors of different shapes and types.
    - [x] Perform basic tensor operations (addition, multiplication, indexing, slicing).
    - [x] Understand the difference between CPU and GPU tensors and how to move them.
    - [x] Practice reshaping and manipulating tensor dimensions.
3.  [x] **Autograd Introduction:**
    - [x] Create tensors with `requires_grad=True`. <-- _Completed 2024-04-13 16:25_
    - [x] Perform operations and understand how the computation graph is built.
    - [x] Use `.backward()` to compute gradients.
    - [x] Understand the concept of gradient accumulation and `.zero_grad()`.
4.  [x] **Basic `nn.Module`:**
    - [x] Understand the structure of a simple `nn.Module`.
    - [x] Define a basic linear layer (`nn.Linear`).
    - [x] Pass a tensor through the layer.
    - [x] Inspect the layer's parameters.

## Definition of Done / Key Questions Answered

- [x] Development environment is functional and PyTorch is installed.
- [x] Can confidently create and manipulate PyTorch tensors.
- [x] Understand how to compute gradients using Autograd for simple operations.
- [x] Can define and use a basic `nn.Linear` layer within an `nn.Module`.
- [x] Key Question: What are the fundamental building blocks (tensors, autograd) provided by PyTorch for deep learning?

## Findings & Notes

### Environment Setup

- Project setup using `pyproject.toml` and `uv` was successful
- Encountered and resolved packaging ambiguity by implementing `src`-layout
- Development environment is reproducible and well-documented

### Tensor Operations

- Mastered tensor creation and manipulation techniques
- Understood the importance of tensor shapes and dimensions
- Gained practical experience with CPU/GPU tensor movement
- Learned efficient tensor reshaping and permutation methods

### Autograd System

- Completed understanding of computation graphs and their role in backpropagation
- Learned how gradients flow through the network
- Understood the importance of `.zero_grad()` for proper gradient accumulation
- Gained practical experience with gradient computation and weight updates

### Neural Network Basics

- Created and analyzed a network with one hidden layer
- Documented parameter counting and network architecture
- Understood the role of activation functions (ReLU)
- Learned about signal flow and weight connections
- Created detailed documentation in `notes/02d_hidden_layer_network.md`

### Key Learnings

1. **Computation Graphs**: PyTorch's autograd system builds a dynamic computation graph that tracks all operations
2. **Parameter Management**: Even simple networks can have many interconnected parameters
3. **Gradient Flow**: Understanding how gradients propagate through the network is crucial
4. **Activation Functions**: ReLU's role in introducing non-linearity and its impact on learning

### Next Steps

- Move on to more complex network architectures
- Explore different activation functions
- Implement custom loss functions
- Begin work on transformer components
