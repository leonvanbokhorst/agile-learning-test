# Notes: Defining a Basic `nn.Module`

**File:** [`results/01_define_nn_module.py`](../results/01_define_nn_module.py)

**Objective:** Create a basic neural network model using PyTorch's `nn.Module` class, suitable for tasks like MNIST classification.

## Key Concepts

1.  **`nn.Module`:** The fundamental building block for all neural network models in PyTorch. Any custom model must inherit from this class.

2.  **`__init__(self)`:**

    - The constructor where you define the layers and other components of the model.
    - Crucially, you **must** call `super().__init__()` at the beginning.
    - Layers (like `nn.Linear`, `nn.Conv2d`, `nn.ReLU`, `nn.Flatten`) are typically assigned as attributes (`self.layer_name = nn.LayerType(...)`).

3.  **`forward(self, x)`:**
    - Defines the computation performed at every call. It dictates how the input tensor `x` flows through the layers defined in `__init__`.
    - Takes the input tensor(s) as argument(s) and returns the output tensor(s).
    - You use the layers defined in `__init__` here (e.g., `x = self.linear1(x)`).

## `SimpleLinearModel` Implementation

- **Inheritance:** `class SimpleLinearModel(nn.Module):`
- **Initialization (`__init__`):**
  - `nn.Flatten()`: Reshapes the input image (e.g., `[batch_size, 1, 28, 28]`) into a flat vector (e.g., `[batch_size, 784]`).
  - `nn.Linear(input_size, hidden_size)`: First fully connected layer.
  - `nn.ReLU()`: Rectified Linear Unit activation function introduces non-linearity.
  - `nn.Linear(hidden_size, output_size)`: Second fully connected layer producing the final logits (raw scores) for each class.
- **Forward Pass (`forward`):** Defines the sequence: `Flatten -> Linear1 -> ReLU -> Linear2`.
- **Final Activation:** The final activation function (like `nn.Softmax` for multi-class classification) is often _omitted_ from the `forward` method. This is because common loss functions like `nn.CrossEntropyLoss` integrate the Softmax calculation internally for better numerical stability and efficiency.

## Testing (`if __name__ == "__main__":`)\*\*

- This block allows the script to be run directly for testing.
- It instantiates the model.
- Creates dummy input data (`torch.randn`) with the expected shape.
- Performs a forward pass (`output = model(dummy_input)`).
- Prints the model architecture and checks if the output shape is as expected (`[batch_size, num_classes]`). This is a quick sanity check.

## Summary

We successfully defined a simple, two-layer linear model using `nn.Module`, understanding the core `__init__` and `forward` methods, and how to structure and test a basic PyTorch model.
