# Loss Functions & The Training Loop

## What are Loss Functions?

A **loss function** (or cost function) measures how far off a model's prediction is from the actual target value (the ground truth). Think of it as quantifying the "error" or "badness" of the model's guess.

- It outputs a single scalar value (the **loss**).
- **Lower loss** = Better model performance.
- **Higher loss** = Worse model performance.
- The primary goal during training is to **minimize** this loss value.

Optimization algorithms (like Gradient Descent) use the loss to figure out how to adjust the model's internal parameters (weights and biases) to make better predictions in the future.

### Common Example: `torch.nn.CrossEntropyLoss`

- **Purpose:** Ideal for multi-class classification tasks (like MNIST digit recognition, where you have 10 classes).
- **Inputs:**
  1.  **Model Predictions (Logits):** Raw, unnormalized scores from the model's final layer. For a batch of N samples and C classes, this is typically a tensor of shape `(N, C)`. Example: `(64, 10)` for MNIST.
  2.  **True Labels (Targets):** The correct class index for each sample. For a batch of N samples, this is typically a tensor of shape `(N,)` containing integers representing the class index (e.g., 0-9 for MNIST).
- **What it Does Internally:**
  1.  **Applies `LogSoftmax`:** It implicitly applies `LogSoftmax` to the model's logits. Softmax converts logits into probabilities that sum to 1, representing the model's confidence for each class. `LogSoftmax` takes the logarithm for numerical stability and compatibility with the next step.
  2.  **Calculates `NLLLoss` (Negative Log Likelihood Loss):** It then calculates the loss based on the log-probabilities and the true target labels. It heavily penalizes the model if it assigns a low probability to the correct class.

### Other Common Loss Functions

While `CrossEntropyLoss` is prevalent for multi-class classification, other loss functions are essential for different tasks:

1.  **`nn.MSELoss` (Mean Squared Error Loss):**

    - **What it is:** Calculates the average squared difference between predicted values $(\hat{y}_i)$ and actual target values $(y_i)$. Formula: $L = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$
    - **When to use it:** The standard choice for **regression tasks** (predicting continuous values like price, temperature).
    - **Why it's used:** Penalizes larger errors more significantly due to squaring. Mathematically convenient and differentiable.

2.  **`nn.L1Loss` (Mean Absolute Error - MAE):**

    - **What it is:** Calculates the average absolute difference between predicted and target values. Formula: $L = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$
    - **When to use it:** Also for **regression tasks**. Useful when the dataset might contain significant outliers, as it's less sensitive to them than MSE.
    - **Why it's used:** Provides a more robust measure of error in the presence of outliers.

3.  **`nn.BCEWithLogitsLoss` (Binary Cross Entropy with Logits Loss):**
    - **What it is:** Specifically designed for **binary classification** (two classes, 0 or 1). Combines a `Sigmoid` activation and `BCELoss` for numerical stability.
    - **When to use it:** When the model outputs a single raw logit per sample, and the target is either 0 or 1.
    - **Why it's used:** Standard and effective loss for binary classification, directly evaluating the predicted probability for the positive class.

**Why Different Losses?**

The choice depends critically on the **problem type** (classification/regression, binary/multi-class) and the **nature of the model's output**. Using the correct loss function ensures the model optimizes for the right objective.

## The Training Loop: The Heartbeat of Training

The training loop is the core process where the model learns. Here's a typical structure in PyTorch:

```python
# Assume: model, train_loader, optimizer, loss_fn are defined
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Example device
# model.to(device)
# loss_fn = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train() # Set the model to training mode

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # 1. Move data to the target device (e.g., GPU)
        data, target = data.to(device), target.to(device)

        # 2. Zero the gradients
        # Reset gradients from the previous iteration before calculating new ones.
        optimizer.zero_grad()

        # 3. Forward Pass: Get model predictions
        # Pass the input data through the model.
        outputs = model(data) # These are typically logits

        # 4. Calculate the Loss
        # Compare model outputs with the true labels.
        loss = loss_fn(outputs, target)

        # 5. Backward Pass: Compute Gradients
        # Calculate the gradient of the loss with respect to model parameters.
        # This is where PyTorch's autograd magic happens!
        loss.backward()

        # 6. Optimizer Step: Update Parameters
        # Adjust the model's parameters based on the computed gradients
        # to minimize the loss.
        optimizer.step()

        # Optional: Logging progress
        if batch_idx % log_interval == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}')

```

**Key Steps:**

1.  **Zero Gradients (`optimizer.zero_grad()`):** Clear old gradients before computing new ones.
2.  **Forward Pass (`model(data)`):** Get the model's predictions.
3.  **Calculate Loss (`loss_fn(...)`):** Measure the error.
4.  **Backward Pass (`loss.backward()`):** Calculate gradients (how to change parameters).
5.  **Optimizer Step (`optimizer.step()`):** Update the parameters.

This cycle repeats for multiple batches and epochs, gradually improving the model.
