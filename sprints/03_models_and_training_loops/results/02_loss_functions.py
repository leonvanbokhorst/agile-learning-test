import torch
import torch.nn as nn

# This file demonstrates the concept of using a loss function.
# In practice, loss functions are used within a training loop.

# Example: CrossEntropyLoss for a classification task

# Assume model output (logits) for a batch of 2 samples, 3 classes: Apple, Banana, Orange
# Model is somewhat confident sample 0 is class 1 (Banana), sample 1 is class 0 (Apple)
logits = torch.tensor([[0.1, 1.5, -0.5], [2.0, -0.2, 0.3]])  # Sample 0 is Banana, Sample 1 is Apple

# True labels
targets = torch.tensor([1, 0])  # Sample 0 is Banana, Sample 1 is Apple, Sample 2 is Orange

# Instantiate the loss function
loss_fn = nn.CrossEntropyLoss()

# Calculate the loss
loss = loss_fn(logits, targets)

print(f"Example Logits:\n{logits}")  # logits are the raw output of the model
print(f"Example Targets: {targets}")  # targets are the correct labels for Apple and Banana
print(f"Calculated CrossEntropyLoss: {loss.item():.4f}")  # for a batch of 2 samples
print(f"Loss for each sample: {loss_fn(logits[0], targets[0]).item():.4f}, {loss_fn(logits[1], targets[1]).item():.4f}")

# Note: This loss value would typically be used in loss.backward()
# within a training loop to compute gradients.
