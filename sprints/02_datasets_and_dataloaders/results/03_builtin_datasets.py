import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# --- Configuration ---
batch_size = 64
data_root = "./data"  # Directory to download/cache the dataset

print(f"PyTorch Version: {torch.__version__}")
print(f"Torchvision Version: {torchvision.__version__}")

# --- Transformations ---
# Built-in datasets often return PIL Images. We need to convert them to tensors.
# We also typically normalize image data.
# transforms.Compose chains multiple transforms together.
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # Converts PIL Image (H x W x C) or numpy.ndarray (H x W x C) to torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        transforms.Normalize((0.1307,), (0.3081,)),  # Normalize with MNIST mean and std
    ]
)
print("\nDefined transformations (ToTensor, Normalize).", flush=True)

# --- Load MNIST Dataset ---
print(f"\nAttempting to load MNIST dataset from {os.path.abspath(data_root)}...")
print("This may download the dataset if it's not already cached.")

# train=True loads the training set (60,000 images)
# train=False loads the test set (10,000 images)
# download=True will download the dataset if not found in data_root
train_dataset = torchvision.datasets.MNIST(
    root=data_root,
    train=True,
    download=True,
    transform=transform,  # Apply the transformations
)

test_dataset = torchvision.datasets.MNIST(
    root=data_root, train=False, download=True, transform=transform
)

print("\nMNIST dataset loaded successfully.")
print(f"  Training dataset size: {len(train_dataset)}")
print(f"  Test dataset size: {len(test_dataset)}")

# --- Create DataLoaders ---
# Use num_workers=0 for simplicity, especially on Windows/macOS
# Shuffle the training data, but not the test data
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
)

print("\nDataLoaders created.")
print(f"  Number of training batches: {len(train_loader)}")
print(f"  Number of test batches: {len(test_loader)}")

# --- Inspect a Batch ---
print("\nInspecting the first batch from the training loader...")
# Get one batch of training images
dataiter = iter(train_loader)
try:
    images, labels = next(dataiter)

    print(
        f"  Batch images shape: {images.shape}"
    )  # [batch_size, channels, height, width]
    print(f"  Batch labels shape: {labels.shape}")  # [batch_size]
    print(f"  Example labels in batch: {labels[:10]}...")  # Print first 10 labels

    # --- Visualize Some Images ---
    print("\nAttempting to visualize some images (requires matplotlib)...")
    try:
        # We need to un-normalize and reshape to plot correctly
        # Create a reverse transform (approximate)
        # Note: Visualizing normalized images directly is also possible but might look weird
        inv_normalize = transforms.Normalize(mean=[-0.1307 / 0.3081], std=[1 / 0.3081])

        fig = plt.figure(figsize=(10, 4))
        for idx in range(min(images.shape[0], 10)):  # Show up to 10 images
            ax = fig.add_subplot(2, 5, idx + 1, xticks=[], yticks=[])
            img = inv_normalize(images[idx])  # Un-normalize
            # matplotlib expects (H, W, C) or (H, W)
            # Our image is (C, H, W) with C=1, so we squeeze it
            plt.imshow(img.squeeze(), cmap="gray")
            ax.set_title(f"Label: {labels[idx].item()}")
        plt.suptitle("Sample Images from First Training Batch (Un-normalized)")
        # Try saving the plot instead of showing it interactively, which might cause issues
        plot_filename = "sprints/02_datasets_and_dataloaders/results/mnist_sample.png"
        plt.savefig(plot_filename)
        print(f"Saved sample images plot to {plot_filename}")
        # plt.show() # Avoid plt.show() in non-interactive environments
        plt.close(fig)  # Close the figure to free memory

    except ImportError:
        print("  matplotlib not found. Skipping visualization.")
    except Exception as e:
        print(f"  Error during visualization: {e}")

except StopIteration:
    print("  Could not retrieve a batch from the DataLoader.")
except Exception as e:
    print(f"  An error occurred while inspecting the batch: {e}")

print("\nScript finished.")
