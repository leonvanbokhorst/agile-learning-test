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

# --- CPU / GPU ---
print(f"Is CUDA available? {torch.cuda.is_available()}")

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a tensor on the default device (usually CPU)
tensor_cpu = torch.randn(2, 3)
print(f"\nTensor on CPU: {tensor_cpu}")
print(f"Device of tensor_cpu: {tensor_cpu.device}")

# Move the tensor to the target device (GPU if available)
tensor_gpu = tensor_cpu.to(device)
print(f"\nTensor moved to {device}: {tensor_gpu}")
print(
    f"Device of tensor_gpu: {tensor_gpu.device}"
)  # Note: This might still say cpu if no CUDA GPU is available

# You can also create tensors directly on the device
tensor_direct = torch.randn(3, 2, device=device)
print(f"\nTensor created directly on {device}: {tensor_direct}")
print(f"Device of tensor_direct: {tensor_direct.device}")

# Trying to operate on tensors on different devices will fail!
# Uncommenting the next line will likely cause a runtime error if CUDA is available
# result = tensor_cpu + tensor_gpu

# Operations work if tensors are on the same device
tensor_gpu_2 = torch.randn(2, 3, device=device)
result_gpu = tensor_gpu + tensor_gpu_2
print(f"\nResult of adding two tensors on {device}: {result_gpu}")
print(f"Device of result_gpu: {result_gpu.device}")

# Move result back to CPU (e.g., for use with NumPy or printing)
result_cpu = result_gpu.cpu()
print(f"\nResult moved back to CPU: {result_cpu}")
print(f"Device of result_cpu: {result_cpu.device}")

# Note: Moving data between CPU and GPU has a cost! Avoid unnecessary transfers.
# MPS: Metal Performance Shaders

if torch.backends.mps.is_available():
    print("MPS is available!")
else:
    print("MPS is not available.")

# Create a tensor on MPS if available
if torch.backends.mps.is_available():
    tensor_mps = torch.randn(2, 3, device="mps")
    print(f"\nTensor created on MPS: {tensor_mps}")
    print(f"Device of tensor_mps: {tensor_mps.device}")
else:
    print("MPS is not available.")

# --- Reshaping and Manipulating Dimensions ---

print("\n--- Reshaping and Manipulating Dimensions ---")

# Create a sample tensor
original_tensor = torch.randn(2, 1, 3, 1)  # Shape: (2, 1, 3, 1) - 6 elements
print(f"\nOriginal Tensor (shape {original_tensor.shape}):\n{original_tensor}")

# TODO 1: Use view() or reshape() to change the shape to (2, 3)
# Note: The total number of elements must remain the same (2*3 = 6)
reshaped_tensor = original_tensor.view(2, 3)
print(f"\nReshaped to (2, 3) (shape {reshaped_tensor.shape}):\n{reshaped_tensor}")

# TODO 2: Use permute() to swap the first and third dimensions (0 and 2)
# Original: (2, 1, 3, 1) -> Permuted: (3, 1, 2, 1)
permuted_tensor = original_tensor.permute(2, 1, 0, 3)
print(
    f"\nPermuted (dims 0 and 2 swapped) (shape {permuted_tensor.shape}):\n{permuted_tensor}"
)

# TODO 3: Use transpose() to swap the first and third dimensions (0 and 2)
# This is similar to permute(2, 1, 0, 3) in this specific case for 4D
transposed_tensor = original_tensor.transpose(0, 2)
print(
    f"\nTransposed (dims 0 and 2 swapped) (shape {transposed_tensor.shape}):\n{transposed_tensor}"
)

# TODO 4: Use squeeze() to remove all dimensions of size 1
# Original: (2, 1, 3, 1) -> Squeezed: (2, 3)
squeezed_tensor = original_tensor.squeeze()
print(
    f"\nSqueezed (removed dims of size 1) (shape {squeezed_tensor.shape}):\n{squeezed_tensor}"
)

# TODO 5: Use unsqueeze() to add a dimension of size 1 at the beginning (dim=0)
# Start with squeezed_tensor: (2, 3) -> Unsqueezed: (1, 2, 3)
unsqueezed_tensor = squeezed_tensor.unsqueeze(0)
print(
    f"\nUnsqueezed (added dim at pos 0) (shape {unsqueezed_tensor.shape}):\n{unsqueezed_tensor}"
)
