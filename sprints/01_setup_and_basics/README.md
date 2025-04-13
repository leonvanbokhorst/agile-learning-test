# Sprint 1: Environment Setup & PyTorch Fundamentals

**Parent Epic:** Master PyTorch Fundamentals and Construct a Foundational Transformer LLM (See `../../README.md`)

## Sprint Goal

Set up the development environment with necessary libraries (PyTorch, etc.) and gain hands-on experience with core PyTorch concepts: Tensors, Autograd, and basic Neural Network modules.

## Tasks / Learning Objectives

1.  [ ] **Environment Setup:**
    - [ ] Create a virtual environment (e.g., `venv` or `conda`).
    - [ ] Install PyTorch (CPU or GPU version as appropriate).
    - [ ] Install other useful libraries (e.g., `numpy`, `matplotlib`, `jupyter` if desired for experimentation).
    - [ ] Verify installation by running simple PyTorch commands.
2.  [ ] **PyTorch Tensors:**
    - [ ] Create tensors of different shapes and types.
    - [ ] Perform basic tensor operations (addition, multiplication, indexing, slicing).
    - [ ] Understand the difference between CPU and GPU tensors and how to move them.
    - [ ] Practice reshaping and manipulating tensor dimensions.
3.  [ ] **Autograd Introduction:**
    - [ ] Create tensors with `requires_grad=True`.
    - [ ] Perform operations and understand how the computation graph is built.
    - [ ] Use `.backward()` to compute gradients.
    - [ ] Understand the concept of gradient accumulation and `.zero_grad()`.
4.  [ ] **Basic `nn.Module`:**
    - [ ] Understand the structure of a simple `nn.Module`.
    - [ ] Define a basic linear layer (`nn.Linear`).
    - [ ] Pass a tensor through the layer.
    - [ ] Inspect the layer's parameters.

## Definition of Done / Key Questions Answered

- [ ] Development environment is functional and PyTorch is installed.
- [ ] Can confidently create and manipulate PyTorch tensors.
- [ ] Understand how to compute gradients using Autograd for simple operations.
- [ ] Can define and use a basic `nn.Linear` layer within an `nn.Module`.
- [ ] Key Question: What are the fundamental building blocks (tensors, autograd) provided by PyTorch for deep learning?

## Findings & Notes

_(Record findings, challenges, code snippets, and reflections in `notes/` directory or here)_
