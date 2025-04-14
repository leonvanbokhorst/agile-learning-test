# Notes: Early Stopping - Knowing When to Quit!

Remember studying for that test? Early stopping is like:

1.  **Take Practice Tests:** After each study session (epoch), check your score on a practice test (validation set). Use a specific score (metric) like validation loss or accuracy.
2.  **Track Your Best:** Remember your highest practice test score so far. If you get a new high score, save your notes (model weights) from that session.
3.  **Patience is Key:** If your score _doesn't_ improve for a few sessions (e.g., 5 times - this is your **patience**!), maybe you've peaked.
4.  **Stop Studying:** If you hit your patience limit, stop studying (stop training).
5.  **Use Best Notes:** Go back to the notes (model weights) you saved when you got your best practice score.

It stops you from "over-studying" (overfitting) and saves time!

## Why Use Early Stopping?

As we train models, performance on the training data usually keeps improving. However, performance on unseen validation data often improves for a while, then peaks, and may even start to get _worse_ if training continues too long. This worsening phase is **overfitting** – the model is learning noise or specifics from the training data that don't generalize.

Early stopping helps to:

1.  **Prevent Overfitting:** By stopping training around the point where validation performance is best, we get a model that generalizes better.
2.  **Reduce Training Time:** Avoids wasting time and computation on epochs that aren't actually improving the model's generalization ability.

## How It Works (The Logic)

As you know from experience, the core components are:

1.  **Monitor a Metric:** Choose a metric on the validation set to track. Common choices:
    - **Validation Loss:** Aim to _minimize_ this. Lower is better.
    - **Validation Accuracy:** Aim to _maximize_ this. Higher is better.
2.  **Patience:** Define how many epochs you're willing to wait _without improvement_ in the monitored metric before stopping. Your example of 5 evaluations is a typical patience value.
3.  **Track Best Score:** Keep track of the best metric value seen so far.
4.  **Save Best Model:** Store the model's `state_dict` whenever the monitored metric improves past the current best.
5.  **Counter:** If the metric does _not_ improve in an epoch, increment a counter.
6.  **Check & Stop:** If the counter reaches the patience limit, stop the training loop.
7.  **Load Best:** After stopping (either by early stopping or finishing all epochs), load the saved weights corresponding to the best validation score.

### Pseudo-Code Example

```python
import torch
import numpy as np # For infinity

# --- Configuration ---
patience = 5 # Number of epochs to wait for improvement
min_delta = 0.001 # Minimum change to qualify as improvement (optional, helps avoid stopping on tiny fluctuations)
mode = 'min' # Are we minimizing (loss) or maximizing (accuracy)? ('min' or 'max')
best_metric = np.inf if mode == 'min' else -np.inf # Initialize best score
patience_counter = 0
best_model_weights = None # To store the state_dict

# --- Inside Training Loop (after validation epoch) ---

# Assume current_validation_metric holds the calculated loss or accuracy for this epoch
current_validation_metric = 0.5 # Example: validation loss

improved = False
if mode == 'min':
    # Check if loss improved by at least min_delta
    if current_validation_metric < best_metric - min_delta:
        improved = True
elif mode == 'max':
    # Check if accuracy improved by at least min_delta
    if current_validation_metric > best_metric + min_delta:
        improved = True

if improved:
    print(f"Validation metric improved ({best_metric:.6f} --> {current_validation_metric:.6f}). Saving model...")
    best_metric = current_validation_metric
    # Save the model state (important! deepcopy avoids issues)
    import copy
    best_model_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0 # Reset patience counter
else:
    patience_counter += 1
    print(f"Validation metric did not improve. Patience counter: {patience_counter}/{patience}")

# Check if training should stop
if patience_counter >= patience:
    print(f"Early stopping triggered after {patience} epochs without improvement.")
    # break the training loop
    # training_should_stop = True

# --- After Training Loop ---

if best_model_weights is not None:
    print("Loading best model weights achieved during training.")
    model.load_state_dict(best_model_weights)
else:
    print("Warning: No best model weights were saved (maybe training was too short or metric never improved?).")

```

**Note:** While you can implement this logic yourself (as shown above), many training frameworks and libraries (like PyTorch Lightning, fastai, Keras, Hugging Face Trainer) have built-in Early Stopping callbacks that handle this logic for you, which is often more convenient.
