import torch

# --- Super Simple Autograd Example (Scalar Output) ---
# This example demonstrates the basic usage of autograd when the
# final output, on which .backward() is called, is a scalar (single value).

print("--- Scalar Autograd Example ---")

# 1. Create our input tensor 'a'. It's just one number!
# We MUST set requires_grad=True to tell PyTorch: "Track this guy!"
# This starts building the computation graph.
a = torch.tensor(3.0, requires_grad=True)
print(f"Input tensor a: {a} (requires_grad={a.requires_grad})")

# 2. Perform an operation: Let's calculate b = a squared.
# Because 'a' requires grad, PyTorch tracks this operation and adds it
# to the computation graph.
b = a * a
# 'b' now implicitly knows it was created by squaring 'a'.
# It has a 'grad_fn' pointing to the squaring operation's backward function.
# This 'grad_fn' is a node in our computation graph map.
print(f"Calculated tensor b: {b}")
print(f"Gradient function for b: {b.grad_fn}")  # Shows the operation history map node

# 3. THE KEY STEP: Call backward() on the scalar output 'b'.
# Since 'b' contains only ONE number (it's a scalar!),
# PyTorch knows exactly where to start the backward pass from.
# It asks: "How does 'b' change when 'a' changes?" (db/da)
# It navigates the computation graph backwards from 'b' to 'a'.
print("\nCalling b.backward()...")
b.backward()

# 4. Inspect the gradient stored in 'a'.
# The derivative of b = a^2 with respect to 'a' is 2*a.
# Since a was 3.0, we expect the gradient db/da to be 2 * 3.0 = 6.0.
# The result is stored in the .grad attribute of the original tensor 'a'.
# Autograd automatically computed this by traversing the graph.
print(f"Gradient stored in a.grad (db/da): {a.grad}")

# --- Why this worked directly ---
# We called .backward() on 'b', which was a SCALAR tensor (tensor(9.)).
# PyTorch didn't need any extra instructions (like gradient=...) because
# there was only one value to start the gradient calculation map traversal from.
print("\nExample complete.")
