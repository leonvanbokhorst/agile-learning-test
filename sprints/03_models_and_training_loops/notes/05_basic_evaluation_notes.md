# Notes: Basic Model Evaluation

Evaluating a model is crucial to understand how well it generalizes to unseen data and to prevent overfitting (where the model performs well on training data but poorly on new data).

## Key Concepts

1.  **Validation Set:**

    - A separate dataset (not used for training) is needed to get an unbiased estimate of the model's performance on unseen data.
    - We created `X_val`, `y_val`, and `val_dataloader` for this purpose.

2.  **`model.eval()`:**

    - Sets the model to evaluation mode.
    - This is important because some layers behave differently during training and evaluation (e.g., Dropout layers are disabled, Batch Normalization layers use running statistics instead of batch statistics).
    - Forgetting this can lead to inconsistent or incorrect evaluation results.

3.  **`with torch.no_grad():`:**

    - A context manager that disables gradient calculations within its block.
    - During evaluation, we don't need to compute gradients (since we're not updating the model weights).
    - Disabling gradients reduces memory consumption and speeds up computation.

4.  **Evaluation Loop:**

    - Similar to the training loop, but without the backward pass and optimizer step.
    - Iterate through the validation dataloader.
    - Perform the forward pass (`outputs = model(inputs)`).
    - Calculate the loss or other relevant metrics (e.g., accuracy for classification).
    - Aggregate the metrics over the entire validation set.

5.  **Metrics:**
    - For our linear regression example, we calculated the average **validation loss** (MSE).
    - For classification tasks, **accuracy** (percentage of correctly classified samples) is a common metric. Other metrics like precision, recall, and F1-score are also used depending on the problem.

## Implementation (`05_basic_evaluation.py`)

- Created an `evaluate_model` function incorporating `model.eval()` and `with torch.no_grad():`.
- Modified `train_model` to optionally accept a `val_dataloader` and call `evaluate_model` after each training epoch. This allows monitoring performance during training.
