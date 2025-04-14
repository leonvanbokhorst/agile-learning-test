import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset  # Assuming we'll use these
from tqdm.auto import tqdm  # Import tqdm


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
):
    """
    A basic PyTorch training loop.
    """
    model.train()  # Set the model to training mode
    print("Starting training...")

    # use tqdm to show progress bar
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0.0  # Keep track of loss for this epoch
        num_batches = len(dataloader)

        # --- Inner loop for batches ---
        # Wrap dataloader with tqdm for a progress bar
        batch_iterator = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{num_epochs} Batch",
            leave=False,
            unit="batch",
        )
        for inputs, targets in batch_iterator:
            # 1. Forward pass
            outputs = model(inputs)

            # 2. Calculate loss
            loss = loss_fn(outputs, targets)

            # 3. Zero gradients
            optimizer.zero_grad()

            # 4. Backward pass (calculate gradients)
            loss.backward()

            # 5. Update weights
            optimizer.step()

            epoch_loss += loss.item()  # Accumulate loss (.item() gets the scalar value)

        avg_epoch_loss = epoch_loss / num_batches
        #print(f"Epoch {epoch+1} finished. Average Loss: {avg_epoch_loss:.4f}")
        # --- End of inner loop ---

        # Placeholder for now - REMOVED
        # pass

    print("Training finished.")


# Example Usage (with dummy data/model for now)
if __name__ == "__main__":
    # 1. Dummy Data and DataLoader
    # Let's create some simple linear data: y = 2x + 1 + noise
    X_train = torch.randn(100, 1) * 10
    y_train = 2 * X_train + 1 + torch.randn(100, 1) * 2
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 2. Dummy Model (Simple Linear Regression)
    class LinearRegression(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 1)

        def forward(self, x):
            return self.linear(x)

    model = LinearRegression()

    # 3. Loss Function
    loss_fn = nn.MSELoss()

    # 4. Optimizer (Using Adam as suggested in notes)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 5. Train the model
    train_model(model, dataloader, loss_fn, optimizer, num_epochs=1000)

    # 6. Inspect the learned parameters
    print("\nLearned parameters after training:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.data.numpy()}")
