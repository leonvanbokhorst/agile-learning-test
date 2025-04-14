"""
Demonstrates loading the MNIST dataset with basic data augmentation
using torchvision.transforms.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def main():
    """Loads MNIST with augmentation and prints the shape of one batch."""
    print("Applying data augmentation to MNIST...")

    # Define transformations
    # MNIST dataset standard mean and standard deviation
    mnist_mean = (0.1307,)
    mnist_std = (0.3081,)

    # Chain transforms together:
    # 1. Randomly rotate the image by +/- 15 degrees
    # 2. Convert the PIL Image (range [0, 255]) to a FloatTensor (range [0.0, 1.0])
    # 3. Normalize the tensor image
    transform_pipeline = transforms.Compose(
        [
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mnist_mean, mnist_std),
        ]
    )

    print(f"Transform pipeline defined: {transform_pipeline}")

    # Load the MNIST training dataset
    # Apply the defined transformations
    print("Loading MNIST training dataset (will download if needed)...")
    train_dataset = torchvision.datasets.MNIST(
        root="./data",  # Directory to save/load data
        train=True,  # Load the training split
        download=True,  # Download if not present
        transform=transform_pipeline,  # Apply the augmentations
    )
    print(f"Training dataset loaded. Number of samples: {len(train_dataset)}")

    # Create a DataLoader
    batch_size = 64
    data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle data in each epoch
        num_workers=2,  # Use multiple processes to load data (adjust as needed)
    )
    print(f"DataLoader created with batch size {batch_size}.")

    # Get one batch of data to inspect
    print("Fetching one batch...")
    images, labels = next(iter(data_loader))

    # Print shapes
    # Images shape: [batch_size, channels, height, width]
    # Labels shape: [batch_size]
    print(f"\nShape of one batch of images: {images.shape}")
    print(f"Shape of one batch of labels: {labels.shape}")

    # You could optionally visualize some augmented images here
    # import matplotlib.pyplot as plt
    # plt.imshow(images[0].squeeze(), cmap='gray') # Squeeze removes the channel dim
    # plt.title(f'Label: {labels[0]}')
    # plt.show()


if __name__ == "__main__":
    main()
