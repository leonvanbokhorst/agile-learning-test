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
    - [x] Create tensors with `requires_grad=True`.
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

- Project setup using `pyproject.toml` and `uv` was successful.
- Encountered and resolved packaging ambiguity by implementing `src`-layout.

### Tensor Basics

- Completed exercises in `results/01_tensor_basics.py`
- Gained proficiency in tensor creation, manipulation, and GPU operations

### Autograd & Gradients

- Completed exercises in `results/02b_autograd_scalar_example.py` and `results/02c_neural_network_gradients.py`
- Demonstrated understanding of computation graphs and gradient flow
- Implemented gradient descent with learning rates

### Neural Networks

- Completed exercise in `results/02d_hidden_layer_network.py`
- Documented network architecture and concepts in `notes/02d_hidden_layer_network.md`
- Gained understanding of parameter connections and signal flow

### Documentation

- Maintained detailed notes in the `notes/` directory
- Updated progress tracking in README files
- Created educational examples with clear comments

## Next Steps

- Moving towards transformer architecture components
- Implementing attention mechanisms
- Building towards the final LLM implementation
