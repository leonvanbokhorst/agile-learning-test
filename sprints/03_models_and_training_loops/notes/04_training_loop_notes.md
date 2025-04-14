# Training Loop Notes

## Purpose

The training loop is the core engine that iteratively trains a neural network. It feeds data to the model, calculates how wrong the model's predictions are (loss), computes gradients to figure out how to improve, and updates the model's parameters using an optimizer.

## Key Components

1.  **Model (`nn.Module`)**: The neural network architecture to be trained.
2.  **DataLoader**: Provides batches of training data (inputs and targets).
3.  **Loss Function (`nn.Module`)**: Measures the discrepancy between model outputs and actual targets.
4.  **Optimizer (`torch.optim.Optimizer`)**: Implements the algorithm to update model parameters based on gradients (e.g., Adam, SGD).
5.  **Epochs**: The number of times the entire dataset is passed through the training loop.

## Core Steps (Inside the Epoch Loop)

For each batch provided by the `DataLoader`:

1.  **Set Model to Train Mode**: `model.train()` - Ensures layers like Dropout and BatchNorm behave correctly during training. (Done once before the epoch loop usually).
2.  **Forward Pass**: Feed the input data from the current batch through the model to get predictions.
    ```python
    outputs = model(inputs)
    ```
3.  **Calculate Loss**: Compare the model's `outputs` with the `targets` using the chosen `loss_fn`.
    ```python
    loss = loss_fn(outputs, targets)
    ```
4.  **Zero Gradients**: Clear gradients from the previous batch/step. If you don't do this, gradients will accumulate, which is usually not desired.
    ```python
    optimizer.zero_grad()
    ```
5.  **Backward Pass (Backpropagation)**: Calculate the gradients of the loss with respect to all model parameters (`requires_grad=True`).
    ```python
    loss.backward()
    ```
6.  **Optimizer Step**: Update the model's parameters based on the computed gradients and the optimizer's internal logic (e.g., learning rate, momentum).
    ```python
    optimizer.step()
    ```

## Important Considerations

- **`model.train()` vs `model.eval()`**: Crucial to switch modes. `eval()` is used during validation/testing to disable dropout and use running statistics for batch normalization.
- **`torch.no_grad()`**: Used during validation/testing to disable gradient calculation, saving memory and computation when gradients aren't needed.
- **Device Management (`.to(device)`)**: In real scenarios, ensure the model and all data tensors are on the same device (CPU or GPU).
- **Tracking Metrics**: Accumulate loss per epoch (using `loss.item()`) and potentially other metrics like accuracy.
- **Sufficient Training Time**: Sometimes, a high final loss doesn't mean the loop is wrong, just that it hasn't run long enough! Increasing the number of epochs can significantly improve results, allowing the optimizer enough steps to converge. For example, our initial linear regression test showed a high loss after 10 epochs, but achieved a loss close to the theoretical minimum (noise variance) after 100 epochs.
