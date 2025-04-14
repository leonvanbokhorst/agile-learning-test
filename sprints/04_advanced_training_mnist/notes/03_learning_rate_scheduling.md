# Notes: Learning Rate Scheduling - Your Training GPS

Imagine finding the lowest valley (minimum loss) in a huge mountain range.

- **Starting:** You want a **fast car (high learning rate)** to cover ground quickly and get near the valley.
- **Getting Closer:** As you approach the valley, you need to **slow down (lower learning rate)** to actually find the _exact_ lowest spot without zooming past it or crashing around.

**Learning Rate Scheduling is your GPS that automatically slows the car down!**

1.  **Pick a Plan:** Choose a way to slow down (e.g., "cut speed every 10 miles," "slow down smoothly like a cosine wave").
2.  **Tell PyTorch:** Import `torch.optim.lr_scheduler`, create your optimizer (`optim.Adam`, etc.), then create a scheduler linked to it: `scheduler = scheduler_type(optimizer, ...)`.
3.  **Update the GPS:** At the end of each epoch (usually), tell the scheduler to update the optimizer's speed: `scheduler.step()`.

That's it! The optimizer's learning rate will now change automatically during training according to your plan.

## Why Change the Learning Rate?

As we discussed, a fixed learning rate is often a compromise:

- **Too High:** Fast initial progress, but might bounce around the minimum loss point erratically or overshoot it completely later in training. Training might become unstable.
- **Too Low:** Stable convergence near the minimum, but can take _forever_ to get there and might get stuck in small, suboptimal dips (local minima).

Learning rate scheduling aims to get the best of both worlds: start faster, then refine carefully.

## How to Use Schedulers in PyTorch

It's a two-step process after you've defined your model and optimizer:

1.  **Instantiate a Scheduler:** Choose a scheduler from `torch.optim.lr_scheduler` and link it to your optimizer.
2.  **Call `scheduler.step()`:** Update the learning rate based on the schedule. This is typically done once per epoch, _after_ `optimizer.step()`. (Some schedulers might be updated per batch, but per-epoch is most common).

```python
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter # To log the LR!

# Assume model and optimizer are already defined
# model = YourModel()
# optimizer = optim.Adam(model.parameters(), lr=0.01) # Initial LR

# --- 1. Instantiate Scheduler ---

# Example 1: StepLR - Decays LR by 'gamma' every 'step_size' epochs
# scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# (Reduces LR to 10% every 10 epochs)

# Example 2: CosineAnnealingLR - Smoothly decays LR following a cosine curve
# T_max is the number of epochs until the LR reaches its minimum (often eta_min)
# eta_min is the minimum learning rate (defaults to 0)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.0001)

#   *What does "Annealing" mean here?* It's inspired by metallurgy!
#   Imagine heating metal and letting it cool very slowly. This process, called
#   annealing, makes the metal stronger and less brittle by allowing atoms to settle
#   into a stable state. In LR scheduling, "annealing" refers to the process of
#   *gradually reducing the learning rate* (like slow cooling). This helps the
#   model's weights "settle down" gently into a good, stable minimum loss value
#   without jumping around wildly or getting stuck.

# Example 3: ReduceLROnPlateau - Reduces LR when a metric stops improving
# Requires you to pass the metric value to scheduler.step(metric_value)
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
# (If 'metric_value' doesn't decrease ('min') for 5 epochs ('patience'), reduce LR by factor 0.1)

# --- Training Loop ---
num_epochs = 100
# writer = SummaryWriter(...) # Optional: For logging

for epoch in range(num_epochs):
    # --- Your training steps ---
    # model.train()
    # for batch in dataloader:
    #     optimizer.zero_grad()
    #     outputs = model(batch.data)
    #     loss = criterion(outputs, batch.labels)
    #     loss.backward()
    #     optimizer.step() # Optimizer step FIRST
    #     ...
    # --- End of training steps ---

    # --- Your validation steps ---
    # model.eval()
    # validation_loss = ...
    # --- End of validation steps ---

    # --- 2. Scheduler Step ---
    # For most schedulers:
    scheduler.step()

    # If using ReduceLROnPlateau:
    # scheduler.step(validation_loss) # Pass the monitored metric

    # --- Optional: Log LR to TensorBoard ---
    current_lr = optimizer.param_groups[0]['lr'] # Get the current LR
    # writer.add_scalar('LearningRate', current_lr, epoch)
    print(f"Epoch {epoch+1}, Current LR: {current_lr}")

# writer.close()
```

## Common Scheduler Types:

- **`StepLR`:** Simple, predictable drops. Good if you know roughly when you want to slow down.
- **`CosineAnnealingLR`:** Smooth, gradual decay. Often works very well without much tuning. The `T_max` parameter controls how quickly it decays. (The "annealing" part refers to the gradual reduction, like slow cooling, to help the model settle into a good minimum).
- **`ReduceLROnPlateau`:** Adaptive. Waits for progress to stall before reducing LR. Needs a metric (like validation loss) to monitor.
- **Others:** `ExponentialLR`, `MultiStepLR`, `LambdaLR` (for custom functions), etc.

**Which one to choose?** `CosineAnnealingLR` is a strong default choice nowadays. `ReduceLROnPlateau` is good if you prefer an adaptive approach. `StepLR` is simple to understand and implement.

Experimenting is key! Now you have the tools to guide your training convergence more effectively.

### Connection to Hugging Face `Trainer`

If you've used the Hugging Face `Trainer` before, you might have specified `lr_scheduler_type='cosine'` in `TrainingArguments`. This is essentially the same core strategy as PyTorch's `CosineAnnealingLR`!

- **Core Idea:** Both smoothly decrease the learning rate following a cosine curve.
- **Abstraction:** In `Trainer`, you just provide the name `'cosine'`, and the library often figures out the decay period (`T_max` equivalent) based on your total training steps or epochs.
- **Warmup:** The Hugging Face `'cosine'` scheduler typically includes a linear "warmup" phase at the beginning (controlled by `warmup_steps` or `warmup_ratio`), where the learning rate increases from 0 to the initial rate before starting the cosine decay. PyTorch's base `CosineAnnealingLR` doesn't include this by default, but other schedulers or combinations can achieve it.

So, `lr_scheduler_type='cosine'` in `Trainer` is like getting `CosineAnnealingLR` with automatic configuration and often warmup included.
