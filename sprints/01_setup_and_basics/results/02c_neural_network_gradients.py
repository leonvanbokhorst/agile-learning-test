import torch
import torch.nn as nn

print("--- Neural Network Gradients Example ---")

# Let's create a simple neural network layer
# This is like a tiny brain cell that takes 3 inputs and produces 1 output
layer = nn.Linear(in_features=3, out_features=1)

# Create some input data (like 3 features for one sample)
# Let's say these are measurements: [height, weight, age]
x = torch.tensor([1.75, 70.0, 25.0], dtype=torch.float32)
print(f"\nInput features: {x}")

# Forward pass (like asking the brain cell to think)
output = layer(x)
print(f"Raw output before activation: {output}")

# Let's use a sigmoid activation to get a probability between 0 and 1
# This is like the brain cell's "confidence" in its answer
sigmoid = nn.Sigmoid()
probability = sigmoid(output)
print(f"Probability after sigmoid: {probability}")

# Now, let's pretend this is a prediction for "is this person tall?"
# And let's say the correct answer is "yes" (1.0)
target = torch.tensor([1.0])

# Calculate how wrong we were (the loss)
# This is like how surprised the brain cell is by the answer
criterion = nn.BCELoss()
loss = criterion(probability, target)
print(f"\nCurrent loss: {loss.item()}")

# Now for the magic! Let's see how each weight affects the loss
print("\nGradients before backward pass:")
for name, param in layer.named_parameters():
    print(f"{name} gradient: {param.grad}")

# Calculate gradients (how to adjust the weights to reduce loss)
loss.backward()

print("\nGradients after backward pass:")
for name, param in layer.named_parameters():
    print(f"{name} gradient: {param.grad}")

# The gradients tell us:
# 1. Which direction to move each weight to reduce the loss
# 2. How much to move each weight (the steeper the gradient, the bigger the change needed)

# Let's update the weights a tiny bit in the direction that reduces loss
learning_rate = 0.0001
with torch.no_grad():  # We don't want to track this operation
    for param in layer.parameters():
        param -= learning_rate * param.grad

print("\nAfter one tiny learning step:")
new_output = layer(x)
new_probability = sigmoid(new_output)
new_loss = criterion(new_probability, target)
print(f"New probability: {new_probability}")
print(f"New loss: {new_loss.item()}")

# Let's do this again 10 times
print("\nTraining for 100 more steps:")
for step in range(100):
    # Forward pass
    output = layer(x)
    probability = sigmoid(output)
    loss = criterion(probability, target)

    # Zero the gradients (important!)
    layer.zero_grad() # before we do the backward pass, we need to zero the gradients

    # Backward pass
    loss.backward()

    # Update weights
    with torch.no_grad():
        for param in layer.parameters():
            param -= learning_rate * param.grad

    # Print the loss every 10 steps
    if (step + 1) % 10 == 0:
        print(f"Step {step + 1}: Loss = {loss.item():.4f}")

print("\nExample complete!")
