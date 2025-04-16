# Sprint 12: Training Loop Logic Notes

## 1. Overview

The core training and evaluation logic resides within the `main` function and the `evaluate` helper function in [`results/finetune_generative.py`](../results/finetune_generative.py).

## 2. Training Loop (`main` function)

- **Outer Loop:** Iterates for the number of epochs specified by `--epochs`.
- **Progress Bar:** Uses `tqdm` to display a progress bar for each epoch, showing loss and learning rate.
- **Mode Setting:** Sets the model to training mode (`model.train()`) at the beginning of each epoch. This ensures layers like Dropout behave correctly during training.
- **Inner Loop (Batch Iteration):** Iterates through batches provided by the `train_loader`.
- **Gradient Zeroing:** Clears previous gradients (`optimizer.zero_grad()`) before processing each batch.
- **Device Transfer:** Moves the input batch (`xb`, `yb`) to the designated `device` (GPU or CPU).
- **Forward Pass & Loss Calculation:**
  - Feeds the input IDs (`xb`) and corresponding labels (`yb`) directly to the model: `model(input_ids=xb, labels=yb)`.
  - Hugging Face Causal LM models automatically calculate the cross-entropy loss for language modeling when `labels` are provided.
  - The calculated loss is extracted from the model's output (`outputs.loss`).
- **Backward Pass:** Calculates gradients based on the loss (`loss.backward()`).
- **Optimizer Step:** Updates the model's weights based on the calculated gradients (`optimizer.step()`).
- **Scheduler Step:** Updates the learning rate according to the defined schedule (`lr_scheduler.step()`).
- **Loss Logging:** Accumulates the batch loss and updates the `tqdm` progress bar postfix.

## 3. Evaluation Function (`evaluate`)

- **Decorator:** Uses `@torch.no_grad()` decorator to disable gradient calculation during evaluation, saving memory and computation.
- **Mode Setting:** Sets the model to evaluation mode (`model.eval()`) at the beginning. This disables Dropout and ensures layers like BatchNorm use running statistics.
- **Looping:** Iterates through batches from the validation `DataLoader` (`val_loader`).
- **Device Transfer:** Moves the validation batch to the `device`.
- **Forward Pass & Loss:** Performs a forward pass similar to training, providing `labels` to get the loss.
- **Loss Accumulation:** Sums the loss across all validation batches.
- **Metrics Calculation:**
  - Calculates the average validation loss.
  - Calculates perplexity using `math.exp(avg_loss)`. Perplexity is a common metric for language models, with lower values indicating better performance.
- **Mode Reset:** Sets the model back to training mode (`model.train()`) before returning.

## 4. Checkpointing (within `main` function, end of epoch)

- **Trigger:** Occurs after the evaluation at the end of each epoch.
- **Condition:** Checks if the current `val_loss` is lower than the `best_val_loss` seen so far.
- **Saving:** If the validation loss has improved:
  - Updates `best_val_loss`.
  - Logs the improvement.
  - Saves the model's weights and configuration using `model.save_pretrained(args.output_dir)`.
  - Saves the tokenizer's state using `tokenizer.save_pretrained(args.output_dir)`. This ensures the correct tokenizer is bundled with the fine-tuned model weights.
- **No Validation Set:** If no validation loader is available, a warning is logged, and checkpointing based on validation loss is skipped (though example code shows how one might save every epoch instead).
