import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleLinearModel(nn.Module):
    """
    A basic linear model for MNIST (28x28 images).
    Flattens the input image and passes it through two linear layers.
    """

    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        """
        Initializes the layers of the model.

        Args:
            input_size (int): The size of the input features (28*28 for MNIST).
            hidden_size (int): The size of the hidden layer.
            output_size (int): The number of output classes (10 for MNIST digits).
        """
        super().__init__()  # Gotta call the parent constructor! It's the law.
        self.flatten = nn.Flatten()  # Squishes the 28x28 image into a 784 vector
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = (
            nn.ReLU()
        )  # Gotta have some non-linearity, keeps things interesting!
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor (batch_size, 1, 28, 28 for MNIST).

        Returns:
            torch.Tensor: The output tensor (batch_size, output_size).
        """
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        # Usually, a final activation (like Softmax) is applied *outside* the model,
        # often combined with the loss function (like nn.CrossEntropyLoss) for stability.
        return x


if __name__ == "__main__":
    # Create a dummy input tensor (like one batch of MNIST images)
    # Batch size = 4, Channels = 1, Height = 28, Width = 28
    dummy_input = torch.randn(4, 1, 28, 28)

    # Instantiate the model
    model = SimpleLinearModel()
    print("Model Architecture:")
    print(model)

    # Pass the dummy input through the model
    try:
        output = model(dummy_input)
        print("\nSuccessfully passed dummy input through the model.")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        # The output shape should be (batch_size, num_classes), e.g., (4, 10)
        assert output.shape == (4, 10)
    except Exception as e:
        print(f"\nError during forward pass: {e}")
