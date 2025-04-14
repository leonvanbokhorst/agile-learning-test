# Sprint 3: Models & Training Loops

Focus: Defining neural network models using `nn.Module`, understanding loss functions and optimizers, and building a complete training loop.

## Tasks

- [x] **Define a basic `nn.Module`:**
  - Create a simple linear model or a basic CNN using `torch.nn.Module`.
  - Understand the `__init__` and `forward` methods.
  - [Define a `nn.Module`](notes/01_define_nn_module_notes.md), [Module Composition](notes/03_module_composition_notes.md)
  - [Results](results/01_define_nn_module.py)
- [x] **Implement Loss Functions:**
  - Use common loss functions like `nn.CrossEntropyLoss` for classification or `nn.MSELoss` for regression.
  - Understand how loss functions measure model error.
  - [Loss Functions](notes/02_loss_functions_training_loop_notes.md), [Cross-Entropy Explained](notes/02a_cross_entropy_explained.md)
  - [Results](results/02_loss_functions.py)
- [x] **Implement Optimizers:**
  - Use optimizers like `torch.optim.Adam` or `torch.optim.SGD`.
  - Understand the role of optimizers in updating model parameters.
  - [Optimizers](notes/03_optimizers_notes.md)
  - [Results](results/03_optimizers.py)
- [x] **Build a Training Loop:**
  - Combine the model, loss function, and optimizer into a complete training loop.
  - Implement the forward pass, loss calculation, backward pass (`loss.backward()`), and optimizer step (`optimizer.step()`).
  - Remember to zero gradients (`optimizer.zero_grad()`).
  - Iterate over the `DataLoader`.
  - [Training Loop](notes/04_training_loop_notes.md)
  - [Results](results/04_training_loop.py)
- [x] **Implement Basic Evaluation:**
  - Add a basic evaluation step within or after the training loop.
  - Calculate accuracy or another relevant metric on a validation set.
  - Understand the importance of `model.eval()` and `with torch.no_grad():`.
  - [Basic Evaluation](notes/05_basic_evaluation_notes.md)
  - [Results](results/05_basic_evaluation.py)


## Retrospective

_(To be filled out at the end of the sprint)_

- **What went well?**
  - Building the core components (`nn.Module`, loss, optimizer) felt quite logical and modular.
  - Putting them together into the training loop structure clicked nicely.
  - Understanding the cycle (forward, loss, backward, step, zero_grad) seems solid.
  - Adding the evaluation loop with `model.eval()` and `torch.no_grad()` went smoothly.
  - Getting `tqdm` working provided helpful progress feedback.
- **What could be improved?**
  - The initial `tqdm` setup with nested loops needed fine-tuning.
  - Remembering to switch between `model.train()` and `model.eval()` is crucial.
  - Could explore different learning rates/optimizers later to see their impact.
- **What did I learn?**
  - The fundamental recipe for training a neural network in PyTorch.
  - The specific roles and interactions of the model, loss function, and optimizer.
  - How to implement backpropagation and weight updates (`loss.backward()`, `optimizer.step()`).
  - The critical difference between training and evaluation modes (`.train()`, `.eval()`) and the importance of `torch.no_grad()`.
  - How to set up a basic validation loop to monitor generalization.
  - Practical application of `tqdm` for visualizing training progress.
