# Sprint 4: Advanced Training Techniques & MNIST Classification

## Goals

- Build a Convolutional Neural Network (CNN) for image classification.
- Implement and understand advanced training techniques like learning rate scheduling and early stopping.
- Learn to use TensorBoard for visualizing the training process.
- Apply these concepts to train a model on the MNIST dataset.

## Tasks

- [x] **CNN Architecture:**
  - [x] Define a basic CNN architecture using `nn.Conv2d`, `nn.ReLU`, `nn.MaxPool2d`, `nn.Flatten`, `nn.Linear`.
  - [x] Understand the flow of data (dimensions) through convolutional and pooling layers.
  - [x] Implement a `forward` method for the CNN.
  - [x] _Results:_ `results/define_cnn.py` (Also see `results/define_resnet_mnist.py` for a more complex example)
  - [x] _Notes:_ `notes/01_cnn_architecture.md`
- [x] **TensorBoard Integration:**
  - [x] Set up `torch.utils.tensorboard.SummaryWriter`.
  - [x] Log training and validation loss curves.
  - [x] Log learning rate changes.
  - [ ] (Optional) Log model graph or sample images. (Skipped for now)
  - [x] _Results:_ Integrated into `results/train_mnist_cnn.py`
  - [x] _Notes:_ `notes/02_tensorboard_basics.md`
- [x] **Learning Rate Scheduling:**
  - [x] Understand the concept and benefits of learning rate scheduling.
  - [x] Implement a simple scheduler (e.g., `torch.optim.lr_scheduler.CosineAnnealingLR`).
  - [x] Integrate the scheduler into the training loop (`scheduler.step()`).
  - [x] _Results:_ Integrated into `results/train_mnist_cnn.py`
  - [x] _Notes:_ `notes/03_learning_rate_scheduling.md`
- [x] **Early Stopping:**
  - [x] Understand the concept of early stopping to prevent overfitting.
  - [x] Implement a basic early stopping mechanism (monitoring validation loss, patience counter, saving best model).
  - [x] Integrate early stopping logic into the training loop.
  - [x] _Results:_ Integrated into `results/train_mnist_cnn.py`
  - [x] _Notes:_ `notes/04_early_stopping.md`
- [x] **MNIST Training Loop:**
  - [x] Prepare MNIST `Dataset` and `DataLoader` (including necessary transforms like `Normalize`).
  - [x] Combine the CNN, loss function (`nn.CrossEntropyLoss`), optimizer, TensorBoard, LR scheduling, and early stopping into a complete training and validation loop.
  - [x] Train the model on MNIST and monitor performance.
  - [x] Calculate final test accuracy. (Validation accuracy calculated)
  - [x] _Results:_ `results/train_mnist_cnn.py`
  - [ ] _Notes:_ `notes/05_mnist_training_summary.md` (We didn't explicitly create this summary note, but the script itself serves as documentation)

## Key Learnings & Insights

- Successfully defined and understood a basic Convolutional Neural Network (CNN) architecture (`nn.Conv2d`, `nn.ReLU`, `nn.MaxPool2d`, etc.) and its data flow.
- Gained practical experience integrating essential training enhancement techniques into a PyTorch loop:
  - **TensorBoard:** Visualized training/validation loss, accuracy, and learning rate, providing valuable insights into the training dynamics.
  - **Learning Rate Scheduling (`CosineAnnealingLR`):** Implemented automatic adjustment of the learning rate for potentially better convergence.
  - **Early Stopping:** Implemented logic to monitor validation loss and stop training automatically when generalization stopped improving, saving time and preventing overfitting.
- Successfully trained a basic CNN on the MNIST dataset, applying all the above techniques.
- **Context:** The primary goal of this sprint was to learn these advanced training tools and the CNN architecture using MNIST as a concrete, manageable example. While we achieved good performance on MNIST, the main takeaway is the understanding and application of these techniques, which are foundational for training more complex models (like Transformers) later.

_(Add more specific insights as needed)_

## Links to Notes and Results

- **Notes:**
  - [CNN Architecture](notes/01_cnn_architecture.md)
  - [TensorBoard Basics](notes/02_tensorboard_basics.md)
  - [Learning Rate Scheduling](notes/03_learning_rate_scheduling.md)
  - [Early Stopping](notes/04_early_stopping.md)
  - [MNIST Training Summary](notes/05_mnist_training_summary.md)
- **Results:**
  - [Define CNN](results/define_cnn.py)
  - [Define ResNet for MNIST](results/define_resnet_mnist.py) (Optional complex example)
  - [Train MNIST CNN](results/train_mnist_cnn.py) (Integrates TensorBoard, LR Scheduling, Early Stopping)
