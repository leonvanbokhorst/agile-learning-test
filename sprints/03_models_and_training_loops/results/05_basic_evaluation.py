import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    val_dataloader: DataLoader | None = None,  # Add val_dataloader optional argument
):
    """
    A PyTorch training loop with optional validation.
    """
    print("Starting training...")

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode for this epoch
        epoch_loss = 0.0
        num_batches = len(dataloader)

        batch_iterator = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{num_epochs} Training",
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

            # 4. Backward pass
            loss.backward()

            # 5. Update weights
            optimizer.step()

            epoch_loss += loss.item()
            batch_iterator.set_postfix(loss=loss.item())  # Show loss in progress bar

        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} finished. Average Training Loss: {avg_epoch_loss:.4f}")

        # --- Optional Evaluation Step ---
        if val_dataloader:
            evaluate_model(model, val_dataloader, loss_fn, epoch)

    print("Training finished.")


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    epoch: int | None = None,  # Optional epoch number for context
):
    """
    Evaluates the model on the given dataloader.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():  # Disable gradient calculations for evaluation
        batch_iterator = tqdm(
            dataloader,
            desc=(
                f"Epoch {epoch+1}/{epoch+1} Evaluation"
                if epoch is not None
                else "Evaluation"
            ),
            leave=False,
            unit="batch",
        )
        for inputs, targets in batch_iterator:
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            batch_iterator.set_postfix(loss=loss.item())

    print(f"Evaluation finished. Average Validation Loss: {total_loss / num_batches:.4f}")
    return total_loss / num_batches


# Example Usage
if __name__ == "__main__":
    # 1. Dummy Data (Train and Validation)
    # Training data
    X_train = torch.randn(100, 1) * 10
    y_train = 2 * X_train + 1 + torch.randn(100, 1) * 2
    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Validation data (let's create a separate small set)
    X_val = torch.randn(50, 1) * 10  # Smaller validation set
    y_val = 2 * X_val + 1 + torch.randn(50, 1) * 2  # Same underlying function
    val_dataset = TensorDataset(X_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=16)  # No shuffle needed usually

    # 2. Dummy Model
    class LinearRegression(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 1)

        def forward(self, x):
            return self.linear(x)

    model = LinearRegression()

    # 3. Loss Function
    loss_fn = nn.MSELoss()

    # 4. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 5. Train the model (passing the validation dataloader)
    train_model(
        model,
        train_dataloader,
        loss_fn,
        optimizer,
        num_epochs=100,
        val_dataloader=val_dataloader,
    )  # Reduced epochs for quicker demo

    # 6. Final Evaluation (Optional, could just rely on per-epoch evaluation)
    # print("Final evaluation on validation set:")
    # evaluate_model(model, val_dataloader, loss_fn)

    # 7. Inspect the learned parameters
    print("Learned parameters after training:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.data.numpy()}")
