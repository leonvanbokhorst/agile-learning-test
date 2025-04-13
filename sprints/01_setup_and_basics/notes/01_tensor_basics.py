import torch
import numpy as np
# Let's explore PyTorch Tensors!

# 1. Creating Tensors
# --------------------

# Create a tensor from a Python list
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(f"Tensor from list:\n{x_data}\n")

# Create a tensor of zeros
shape = (
    2,
    3,
)
zeros_tensor = torch.zeros(shape)
print(f"Zeros tensor (shape {shape}):\n{zeros_tensor}\n")

# Create a tensor of ones
ones_tensor = torch.ones(shape)
print(f"Ones tensor (shape {shape}):\n{ones_tensor}\n")

# Create a tensor with random values (uniform distribution between 0 and 1)
rand_tensor = torch.rand(shape)
print(f"Random tensor (shape {shape}):\n{rand_tensor}\n")

# Create a tensor with random values (standard normal distribution)
randn_tensor = torch.randn(shape)
print(f"Random Normal tensor (shape {shape}):\n{randn_tensor}\n")

# TODO: Try creating tensors with different data types (dtype), e.g., torch.float32, torch.int64
floaty_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
print(f"Floaty tensor:\n{floaty_tensor}\n")

# TODO: Create a tensor from a NumPy array (requires numpy dependency)
numpy_array = np.array([1, 2, 3])
tensor_from_numpy = torch.from_numpy(numpy_array)
print(f"Tensor from NumPy array:\n{tensor_from_numpy}\n")

# 2. Tensor Attributes
# ---------------------
print("Exploring attributes of rand_tensor:")
print(f"  Shape: {rand_tensor.shape}")
print(f"  Datatype: {rand_tensor.dtype}")
print(f"  Device tensor is stored on: {rand_tensor.device}\n")

# TODO: Move a tensor to GPU if available (torch.cuda.is_available())
if torch.cuda.is_available():
    print("Moving tensor to GPU...")
    rand_tensor = rand_tensor.to("cuda")
    print(f"  New device: {rand_tensor.device}\n")

# 3. Tensor Operations
# ---------------------
# TODO: Indexing and slicing (like NumPy)
print("Indexing and slicing of rand_tensor:")
print(f"  First element: {rand_tensor[0]}")
print(f"  Last element: {rand_tensor[-1]}")
print(f"  Slicing first two elements: {rand_tensor[:2]}")
print(f"  Slicing last two elements: {rand_tensor[-2:]}")


# TODO: Reshaping tensors (view, reshape)
print("Reshaping rand_tensor:")
reshaped_tensor = rand_tensor.view(2, 3)
print(f"  Original shape: {rand_tensor.shape}")
print(f"  Reshaped to: {reshaped_tensor.shape}")


# TODO: Basic math operations (add, subtract, multiply, divide, matrix multiplication)
print("Basic math operations on rand_tensor:")
print(f"  Add 1 to each element: {rand_tensor + 1}")
print(f"  Subtract 2 from each element: {rand_tensor - 2}")
print(f"  Multiply by 3: {rand_tensor * 3}")
print(f"  Divide by 2: {rand_tensor / 2}")

# TODO: Aggregations (sum, mean, max, min)
print("Aggregations on rand_tensor:")
print(f"  Sum of all elements: {rand_tensor.sum()}")
print(f"  Mean of all elements: {rand_tensor.mean()}")
print(f"  Maximum element: {rand_tensor.max()}")
print(f"  Minimum element: {rand_tensor.min()}")


print("End of initial tensor exploration.")
