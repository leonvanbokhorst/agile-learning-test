# Notes: TensorBoard Basics - Visualizing Training

TensorBoard is like a fancy dashboard for your PyTorch training. It shows you graphs and stuff so you don't have to squint at terminal output!

1.  **Import:** `from torch.utils.tensorboard import SummaryWriter`
2.  **Create Writer:** `writer = SummaryWriter('runs/my_cool_experiment')` (Creates a folder `runs/my_cool_experiment` to store logs).
3.  **Log Stuff:** Inside your training loop:
    - `writer.add_scalar('Loss/train', current_loss, global_step=epoch)`
    - `writer.add_scalar('Accuracy/val', current_accuracy, global_step=epoch)`
    - `global_step` is just a counter (like epoch number or batch number) for the x-axis of your graph.
4.  **Close Writer:** `writer.close()` (When training finishes).
5.  **View Dashboard:** Open your terminal _in the project root directory_ (where the `runs` folder is) and run: `tensorboard --logdir=runs`
6.  **Open Browser:** Go to the web address it gives you (usually `http://localhost:6006/`).

Now you have pretty graphs instead of boring numbers! Easier to see if things are going well or terribly wrong. 📉📈

## Key Concepts & Usage

TensorBoard provides visualization tools for understanding, debugging, and optimizing your deep learning models.

### 1. Setting up the `SummaryWriter`

The central component in PyTorch for using TensorBoard is the `SummaryWriter`.

```python
from torch.utils.tensorboard import SummaryWriter
import time

# Best practice: Create a unique log directory for each run.
# Using a timestamp is common.
timestamp = time.strftime("%Y%m%d-%H%M%S")
log_dir = f'runs/mnist_cnn_{timestamp}'

# Instantiate the writer
writer = SummaryWriter(log_dir=log_dir)

print(f"TensorBoard logs will be saved to: {log_dir}")
# Example: writer = SummaryWriter('runs/simple_cnn_experiment_1')
# This will create a directory structure ./runs/simple_cnn_experiment_1/
# where TensorBoard event files will be written.
```

- **`log_dir`:** Specifies the directory where log files will be saved. It's crucial to use separate directories for different experiments or runs if you want to compare them easily in the TensorBoard UI.
- Using timestamps or experiment names in the `log_dir` is highly recommended.

### 2. Logging Scalars

The most common use case is logging scalar values like loss and accuracy during training and validation.

```python
# Inside your training loop (e.g., at the end of each epoch)
training_loss = 0.5 # Replace with your actual calculated loss
validation_accuracy = 0.9 # Replace with your actual calculated accuracy
current_epoch = 5 # Replace with your epoch counter

# Log training loss
# The tag 'Loss/train' organizes scalars under 'Loss', labeled 'train'
writer.add_scalar('Loss/train', training_loss, global_step=current_epoch)

# Log validation accuracy
# The tag 'Accuracy/val' organizes scalars under 'Accuracy', labeled 'val'
writer.add_scalar('Accuracy/val', validation_accuracy, global_step=current_epoch)

# You can also log learning rate
# optimizer = torch.optim.Adam(...)
# current_lr = optimizer.param_groups[0]['lr']
# writer.add_scalar('LearningRate', current_lr, global_step=current_epoch)
```

- **`tag`:** A string name for the scalar (e.g., `'Loss/train'`). Using slashes (`/`) helps organize plots in the TensorBoard UI.
- **`scalar_value`:** The actual Python number (float or int) you want to log.
- **`global_step`:** The value for the x-axis on the plot. This is typically the epoch number or the total number of batches processed.

### 3. Other Logging Capabilities (Briefly)

TensorBoard can log more than just scalars:

- **Images:** `writer.add_image()` (Visualize input images, feature maps, filters)
- **Histograms:** `writer.add_histogram()` (Visualize distributions of weights or gradients)
- **Model Graph:** `writer.add_graph()` (Visualize the model architecture - requires a sample input)
- **Embeddings:** Visualize high-dimensional embeddings (e.g., from `nn.Embedding`).
- **Text:** `writer.add_text()`

### 4. Closing the Writer

It's important to close the writer when you're done logging (e.g., after the training loop finishes) to ensure all data is written to disk.

```python
# After training completes
writer.close()
```

### 5. Launching TensorBoard

Once you have generated some log files:

1.  Open your terminal.
2.  Navigate (`cd`) to the directory that _contains_ your `runs` folder (usually your project root).
3.  Run the command: `tensorboard --logdir=runs`
4.  TensorBoard will start a local web server and print the address (usually `http://localhost:6006/`).
5.  Open that address in your web browser.

You should see the TensorBoard dashboard where you can explore the logged data.
