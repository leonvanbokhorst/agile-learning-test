# Sprint 3: Models & Training Loops

Focus: Defining neural network models using `nn.Module`, understanding loss functions and optimizers, and building a complete training loop.

## Tasks

- [x] **Define a basic `nn.Module`:**
  - Create a simple linear model or a basic CNN using `torch.nn.Module`.
  - Understand the `__init__` and `forward` methods.
  - [Notes](notes/01_define_nn_module_notes.md), [Module Composition](notes/03_module_composition_notes.md)
  - [Results](results/01_define_nn_module.py)
- [x] **Implement Loss Functions:**
  - Use common loss functions like `nn.CrossEntropyLoss` for classification or `nn.MSELoss` for regression.
  - Understand how loss functions measure model error.
  - [Notes](notes/02_loss_functions_training_loop_notes.md), [Cross-Entropy Explained](notes/02a_cross_entropy_explained.md)
  - [Results](results/02_loss_functions.py)
- [x] **Implement Optimizers:**
  - Use optimizers like `torch.optim.Adam` or `torch.optim.SGD`.
  - Understand the role of optimizers in updating model parameters.
  - [Notes](notes/03_optimizers_notes.md)
  - [Results](results/03_optimizers.py)
- [ ] **Build a Training Loop:**
  - Combine the model, loss function, and optimizer into a complete training loop.
  - Implement the forward pass, loss calculation, backward pass (`loss.backward()`), and optimizer step (`optimizer.step()`).
  - Remember to zero gradients (`optimizer.zero_grad()`).
  - Iterate over the `DataLoader`.
  - [Notes]()
  - [Results]()
- [ ] **Implement Basic Evaluation:**
  - Add a basic evaluation step within or after the training loop.
  - Calculate accuracy or another relevant metric on a validation set.
  - Understand the importance of `model.eval()` and `with torch.no_grad():`.
  - [Notes]()
  - [Results]()

## Notes & Results

- Notes: [notes/](notes/)
- Results: [results/](results/)

## Retrospective

_(To be filled out at the end of the sprint)_

- **What went well?**
- **What could be improved?**
- **What did I learn?**
