# Optimizer Notes

## What are Optimizers?

- Algorithms that decide _how_ to update the neural network's weights (parameters) based on the gradients calculated during backpropagation.
- Think of them as the navigation strategy for descending the "loss landscape" mountain. They determine the step size and direction.

## Why are they needed?

- The goal is to minimize the loss function. Gradients tell us the _direction_ of steepest ascent, so we go the opposite way.
- Optimizers make this descent efficient and effective. They figure out the best way to use the gradient information to reach a low loss value quickly and reliably.
- Needed because simply moving in the exact opposite direction of the gradient (Batch Gradient Descent) can be slow, get stuck, or computationally infeasible for large datasets. Mini-batch approaches (like SGD and Adam) are standard practice.

## Common Optimizers

### 1. SGD (Stochastic Gradient Descent)

- Often refers to _Mini-Batch_ Gradient Descent in practice.
- Updates weights using gradients calculated from a small, random batch of data.
- Pros: Computationally cheaper per step than Batch GD, the noise can help escape some local minima/saddle points.
- Cons: Can be noisy/oscillate, might be slow to converge on the optimal path without additions like momentum. Sensitive to learning rate choice.
- Key Hyperparameters: `lr` (learning rate), `momentum` (helps smooth out updates and accelerate in the consistent direction).

### 2. Adam (Adaptive Moment Estimation)

- Combines ideas from Momentum (using a moving average of past gradients) and RMSprop (using adaptive learning rates per parameter).
- It computes _adaptive_ learning rates for each parameter based on estimates of the first moment (mean) and second moment (uncentered variance) of the gradients.
- Pros: Generally works well across a wide range of problems with default hyperparameters. Often converges faster than SGD. Adapts learning rates, reducing sensitivity to the initial learning rate choice (though it still matters!).
- Cons: Can sometimes converge to poorer minima than SGD+Momentum on some tasks (though this is debated and often solvable with tuning). Uses more memory to store the moment estimates.
- Key Hyperparameters: `lr` (learning rate, often smaller than for SGD, e.g., 0.001), `betas` (decay rates for the moment estimates, default `(0.9, 0.999)` is usually fine), `eps` (small term to prevent division by zero, default `1e-8` is usually fine).

* **Note:** Adam (and its variant AdamW, which handles weight decay differently and is often preferred) is frequently the **default go-to optimizer** in modern deep learning due to its strong general performance and relatively fast convergence with less tuning compared to SGD.
* Variants like **8-bit Adam** exist to significantly reduce the memory footprint of the optimizer's state, which is crucial for training very large models.

### Other Optimizers (Optional)

- Adagrad
- RMSprop

## How are they used in PyTorch?

- Import: `import torch.optim as optim`
- Instantiation: `optimizer = optim.OptimizerName(model.parameters(), lr=...)`
- Key steps in training loop:
  1.  `optimizer.zero_grad()`: Clear old gradients.
  2.  `loss.backward()`: Calculate new gradients.
  3.  `optimizer.step()`: Update model weights.
