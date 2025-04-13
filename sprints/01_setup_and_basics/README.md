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
3.  [ ] **Autograd Introduction:**
    - [ ] Create tensors with `requires_grad=True`. <-- _Started 2025-04-13 14:54_
    - [ ] Perform operations and understand how the computation graph is built.
    - [ ] Use `.backward()` to compute gradients.
    - [ ] Understand the concept of gradient accumulation and `.zero_grad()`.
4.  [ ] **Basic `nn.Module`:**
    - [ ] Understand the structure of a simple `nn.Module`.
    - [ ] Define a basic linear layer (`nn.Linear`).
    - [ ] Pass a tensor through the layer.
    - [ ] Inspect the layer's parameters.

## Definition of Done / Key Questions Answered

- [x] Development environment is functional and PyTorch is installed.
- [x] Can confidently create and manipulate PyTorch tensors.
- [ ] Understand how to compute gradients using Autograd for simple operations.
- [ ] Can define and use a basic `nn.Linear` layer within an `nn.Module`.
- [ ] Key Question: What are the fundamental building blocks (tensors, autograd) provided by PyTorch for deep learning?

## Findings & Notes

_(Record findings, challenges, code snippets, and reflections in `notes/` directory or here)_

- Project setup using `pyproject.toml` and `uv` was successful.
- Encountered and resolved packaging ambiguity by implementing `src`-layout.
