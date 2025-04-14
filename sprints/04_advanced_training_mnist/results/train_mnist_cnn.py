import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import time
import copy  # For deep copying model state in early stopping
import numpy as np  # For infinity in early stopping

# Import our SimpleCNN model
# Use relative import since define_cnn.py is in the same directory
from define_cnn import (
    SimpleCNN,
    device,
)  # Use device from the model file


# --- Main Execution Guard ---
if __name__ == "__main__":
    print(f"Using device: {device}")

    # --- Configuration / Hyperparameters ---
    LEARNING_RATE = 0.001
    BATCH_SIZE = 64
    NUM_EPOCHS = 50  # Set a higher number, early stopping will likely trigger
    NUM_CLASSES = 10

    # TensorBoard configuration
    TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
    TB_LOG_DIR = f"runs/mnist_simple_cnn_{TIMESTAMP}"

    # Early Stopping configuration
    EARLY_STOPPING_PATIENCE = 7
    EARLY_STOPPING_MIN_DELTA = (
        0.001  # Minimum change in validation loss to count as improvement
    )

    # LR Scheduler Configuration
    LR_SCHEDULER_T_MAX = NUM_EPOCHS  # For CosineAnnealingLR, decay over all epochs
    LR_SCHEDULER_ETA_MIN = 1e-6  # Minimum learning rate

    # --- Data Loading & Preprocessing ---
    print("\n--- Loading Data ---")

    # MNIST statistics (mean, std) - commonly used
    # Calculated across the training set
    MNIST_MEAN = (0.1307,)
    MNIST_STD = (0.3081,)

    # Transformations
    # 1. Convert image to PyTorch Tensor
    # 2. Normalize using MNIST mean and std
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)]
    )

    # Download and load the training data
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    # Download and load the test data (we'll use this for validation during training)
    # Ideally, you'd have a separate validation set, but for simplicity with MNIST,
    # we often use the test set for validation during development.
    # For a final evaluation, a truly held-out test set is best.
    val_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Create DataLoaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,  # Adjust based on your system
        pin_memory=True,  # Helps speed up CPU-to-GPU transfer if using CUDA
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # No need to shuffle validation data
        num_workers=4,
        pin_memory=True,
    )

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches per epoch: {len(val_loader)}")

    # --- Model, Loss, Optimizer, Scheduler, TensorBoard Writer ---
    print("\n--- Initializing Components ---")

    # Model
    model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
    print(f"Model: {model.__class__.__name__}")

    # Loss function
    criterion = nn.CrossEntropyLoss()
    print(f"Loss function: {criterion.__class__.__name__}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"Optimizer: {optimizer.__class__.__name__}")

    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=LR_SCHEDULER_T_MAX, eta_min=LR_SCHEDULER_ETA_MIN
    )
    print(f"LR Scheduler: {scheduler.__class__.__name__}")

    # TensorBoard Summary Writer
    writer = SummaryWriter(log_dir=TB_LOG_DIR)
    print(f"TensorBoard logs directory: {TB_LOG_DIR}")

    # --- Early Stopping Initialization ---
    best_val_loss = np.inf
    patience_counter = 0
    best_model_state = None

    # --- Training Loop ---
    print("\n--- Starting Training --- ")
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        # --- Training Phase ---
        model.train()  # Set model to training mode
        running_train_loss = 0.0
        # Add tqdm here later for progress bar if desired

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

            # Log training loss per batch (optional, can be noisy)
            # batch_idx = epoch * len(train_loader) + i
            # writer.add_scalar('Loss/train_batch', loss.item(), batch_idx)

        avg_train_loss = running_train_loss / len(train_loader)
        writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
        print(f"  Avg. Training Loss: {avg_train_loss:.4f}")

        # --- Validation Phase ---
        model.eval()  # Set model to evaluation mode
        running_val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():  # No need to track gradients during validation
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                # Calculate accuracy
                _, predicted_labels = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted_labels == labels).sum().item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_accuracy = 100.0 * correct_predictions / total_predictions

        writer.add_scalar("Loss/validation_epoch", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/validation_epoch", val_accuracy, epoch)
        # Log learning rate
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

        print(f"  Avg. Validation Loss: {avg_val_loss:.4f}")
        print(f"  Validation Accuracy: {val_accuracy:.2f}%")

        # --- LR Scheduler Step ---
        scheduler.step()

        # --- Early Stopping Check ---
        if avg_val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the best model state
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"  Validation loss improved. Saving model state.")
        else:
            patience_counter += 1
            print(
                f"  Validation loss did not improve. Patience counter: {patience_counter}/{EARLY_STOPPING_PATIENCE}"
            )

        epoch_duration = time.time() - epoch_start_time
        print(f"  Epoch Duration: {epoch_duration:.2f}s")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break  # Exit the training loop

    # --- Post-Training ---
    total_training_time = time.time() - start_time
    print(f"\n--- Training Finished --- ")
    print(f"Total Training Time: {total_training_time:.2f}s")

    # Load the best model weights found during training
    if best_model_state is not None:
        print("Loading best model weights from early stopping.")
        model.load_state_dict(best_model_state)
    else:
        print("Warning: Early stopping did not save a best model state.")

    # Close TensorBoard writer
    writer.close()
    print("TensorBoard writer closed.")

    # Optional: Final evaluation on the test set with the best model
    # (Currently, we used the test set for validation, so this would be redundant
    # unless we had a separate validation set originally)
    # print("\n--- Final Evaluation on Test Set ---")
    # model.eval()
    # ... (similar logic to validation loop) ...

    print("\nScript finished.")
