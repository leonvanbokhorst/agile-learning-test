import torch
import torch.nn as nn
import torch.nn.functional as F  # Often used for activation functions, though nn.ReLU() layer is also common

# Define device at top level so it can be imported
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class SimpleCNN(nn.Module):
    """
    A basic Convolutional Neural Network for MNIST-like images.
    Architecture: Conv -> ReLU -> Pool -> Flatten -> Linear
    """

    def __init__(self, num_classes: int = 10):
        """
        Initializes the layers of the CNN.

        Args:
            num_classes (int): The number of output classes (e.g., 10 for MNIST digits).
        """
        super().__init__()  # Essential step! Initialize the parent nn.Module

        # Convolutional Layer 1
        # Input: (N, 1, 28, 28) - N batches, 1 channel (grayscale), 28x28 pixels
        # Output: (N, 16, 28, 28)
        # Kernel: 3x3, Stride: 1, Padding: 1 (preserves dimension: 28 - 3 + 2*1 / 1 + 1 = 28)
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        # Activation Function (can be a layer or used functionally in forward)
        self.relu1 = nn.ReLU()
        # Max Pooling Layer 1
        # Input: (N, 16, 28, 28)
        # Output: (N, 16, 14, 14)
        # Kernel: 2x2, Stride: 2 (halves dimensions: 28 / 2 = 14)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten Layer
        # Input: (N, 16, 14, 14)
        # Output: (N, 16 * 14 * 14) = (N, 3136)
        self.flatten = nn.Flatten()

        # Fully Connected (Linear) Layer
        # Input: (N, 3136)
        # Output: (N, num_classes)
        self.fc1 = nn.Linear(in_features=16 * 14 * 14, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor (batch of images).
                               Shape typically (N, 1, H, W).

        Returns:
            torch.Tensor: The output tensor (raw scores/logits for each class).
                          Shape (N, num_classes).
        """
        # Apply Conv1 -> ReLU -> Pool1
        x = self.conv1(x)
        x = self.relu1(x)  # Or F.relu(x)
        x = self.pool1(x)

        # Flatten the output for the linear layer
        x = self.flatten(x)

        # Apply the final fully connected layer
        x = self.fc1(x)

        return x


# --- Test the Model Definition ---
if __name__ == "__main__":
    # Move the print statement inside the main block
    print(f"Using device: {device}")
    print("--- Testing SimpleCNN Definition ---")

    # Create a dummy input tensor representing a batch of 4 grayscale images (28x28)
    # Shape: (batch_size, channels, height, width)
    dummy_input = torch.randn(4, 1, 28, 28).to(device)
    print(f"Dummy input shape: {dummy_input.shape}")

    # Instantiate the model
    model = SimpleCNN(num_classes=10).to(device)  # Move model to the chosen device
    print(f"Model instantiated: {model}")

    # --- Perform a forward pass ---
    print("\n--- Performing Forward Pass ---")
    try:
        # Set model to evaluation mode (good practice, affects dropout/batchnorm if used)
        model.eval()
        # Disable gradient calculation for inference
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        # Check if output shape is as expected (batch_size, num_classes)
        assert output.shape == (4, 10)
        print("Forward pass successful! Output shape is correct.")

        # --- Print Model Summary (Number of Parameters) ---
        print("\n--- Model Summary ---")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params:,}")  # Add comma separators

    except Exception as e:
        print(f"An error occurred during testing: {e}")
        import traceback

        traceback.print_exc()

    print("\n--- Test Complete ---")
