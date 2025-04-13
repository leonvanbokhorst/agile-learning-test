import torch

# --- Autograd Introduction ---
# Let's explore how PyTorch automatically calculates gradients!

# TODO 1: Create a tensor 'x' with a single value (e.g., 3.0).
# Crucially, set requires_grad=True to tell PyTorch we want to track operations involving it.
# Print the tensor and its requires_grad attribute.
x = torch.tensor([3.0, 5.0], dtype=torch.float32, requires_grad=True)
print(x)
print(x.requires_grad)

# TODO 2: Define a new tensor 'y' by performing some operations on 'x'.
# For example, y = x * x + 2 * x + 1.
# Print y and its grad_fn attribute (this shows the function that created it in the graph).
y = x*.1
print(y)
print(y.grad_fn)
y = y[1] - 1
print(y)
print(y.grad_fn)

# TODO 3: Calculate the gradients. Call the .backward() method on 'y'.
# This computes dy/dx (the gradient of y with respect to x).
y.backward()

# TODO 4: Inspect the gradient. The computed gradient is stored in the .grad attribute of the original tensor 'x'.
# Print x.grad. What value do you expect based on the formula for y? (Derivative of x^2 + 2x + 1 is 2x + 2)
print(x.grad)

# --- Gradient Accumulation ---
# TODO 5: Let's see what happens if we run backward again on a different operation.
# Create a new tensor 'z' = x * 3
# Print z
z = None  # Replace None with your calculation
# print(...)

# TODO 6: Run backward on 'z'.
# z.backward() # Uncomment when ready

# TODO 7: Print x.grad again. What happened? Why is it important to zero out gradients?
# print(...)

# TODO 8: Zero out the gradients on 'x' using x.grad.zero_()
# x.grad.zero_() # Uncomment when ready

# TODO 9: Now, recalculate the gradient for 'z' only by calling z.backward() again.
# z.backward() # Uncomment when ready

# TODO 10: Print x.grad one last time. Does it now reflect only the gradient from z?
# print(...)


print("\nAutograd exercises setup complete. Fill in the TODOs!")
